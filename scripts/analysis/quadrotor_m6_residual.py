"""v2.6.0 M6 residual diagnostics (D1-D4) — read-only characterization of the residual timeout + collision
on best.pt (step 46500), full pool n2000/seed23456 (= M6/M3). Two instrumented read-only re-rolls of the
filtered learned policy (no state/config/git change):
  Roll A (max_steps=200, canonical): D2 collision-feasibility split, D3 IC stratification, D4 torque headroom.
  Roll B (max_steps=400): D1 timeout conversion + original-timeout ||p-g|| trajectory classification.
Every printed number is disk-derived; the script writes a JSON dump. NOT SOTA-eligible (D1 changes the
deploy max_steps axis) — flag Researcher.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
TAU_MAX = 0.2
KPREC = 5                       # precursor window (steps before the event)


def load(run_dir, dev):
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    return ck, cfg, system, vnet, make_h_fn(vnet, system), policy


def roll_chunk(system, h_fn, policy, cfg, bs, dev, max_steps):
    """Filtered learned-policy rollout of one scene-batch; returns per-step tensors for diagnostics."""
    params = _hardnet_params(cfg); bounds = system.u_bounds; dt = float(cfg["env"]["dt"])
    x = initial_states_from_batch(bs).to(dev, torch.float32); B = x.shape[0]
    states = [x]
    empt = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
    sviol = torch.zeros_like(empt)
    usafe = torch.zeros(max_steps, B, system.action_dim, device=dev)
    boxclip_tau = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(max_steps):
            un = policy(system.observation(x, bs))
            h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
            h, lf, lg = h.detach(), lf.detach(), lg.detach()
            alpha = _base_alpha(h, params); row = -lf - alpha * h
            # raw (pre-box-clamp) CBF correction to detect torque-box binding on the SAFETY action
            lhs = torch.sum(lg * un, dim=1); viol = torch.relu(lhs - row)
            denom = torch.sum(lg * lg, dim=1) + params.epsilon ** 2 + params.lg_reg_eps
            raw = un - lg * (viol / denom).unsqueeze(1)                     # unclamped demanded action
            boxclip_tau[t] = raw[:, 1].abs() > TAU_MAX + 1e-6              # box clips torque channel
            proj = _base_projection(un, lg, row, bounds, params)
            sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            u, empty = _box_aware_projection(un, proj, lg, row, bounds)
            empt[t] = empty; sviol[t] = sing & (row < 0.0); usafe[t] = u
            x = rk4_step(system, x, u, dt); states.append(x)
    S = torch.stack(states, 0)                                             # [T+1, B, 6]
    masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
    return S, res, empt, sviol, usafe, boxclip_tau, masks.collided


def inward_speed(x0, bs):
    """v_0 projected toward the NEAREST active obstacle (positive = approaching)."""
    p = x0[:, :2]; v = x0[:, 3:5]
    centers = bs.obstacle_centers.to(x0.dtype); radii = bs.obstacle_radii.to(x0.dtype)
    active = bs.obstacle_active.to(torch.bool)
    d = torch.linalg.norm(p.unsqueeze(1) - centers, dim=-1) - radii
    d = d.masked_fill(~active, float("inf"))
    j = torch.argmin(d, dim=1)
    ctr = centers[torch.arange(centers.shape[0]), j]
    to_obs = ctr - p; n = torch.linalg.norm(to_obs, dim=1, keepdim=True).clamp_min(1e-9)
    return torch.sum(v * (to_obs / n), dim=1)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, system, vnet, h_fn, policy = load(run_dir, dev)
    step = int(ck.get("step", -1)); max_steps = int(cfg["eval"]["max_steps"]); chunk = 250
    scenes = load_pool(POOL).scenes; N = len(scenes)
    goal_np = np.stack([np.asarray(s.goal) for s in scenes])

    # ---- Roll A (canonical 200) : per-episode diagnostic records ----
    rec = []                       # per-episode dict
    t0 = time.time()
    for s0 in range(0, N, chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        S, res, empt, sviol, usafe, boxclip_tau, collided = roll_chunk(system, h_fn, policy, cfg, bs, dev, max_steps)
        x0 = S[0]; th0 = x0[:, 2].abs().cpu().numpy(); om0 = x0[:, 5].abs().cpu().numpy()
        inw = inward_speed(x0, bs).cpu().numpy()
        feas = ~(empt | sviol)                                             # [T,B] a safe action existed
        satu_tau = usafe[:, :, 1].abs() >= (TAU_MAX - 1e-4)                # torque saturated at +/-0.2
        B = x0.shape[0]
        for i in range(B):
            o = res.outcome[i]; es = int(res.event_step[i])               # collision/goal/... step; -1 if timeout
            d = dict(idx=s0 + i, outcome=o, event_step=es,
                     th0=float(th0[i]), om0=float(om0[i]), inw=float(inw[i]))
            if o == "collision":
                tc = es if es >= 0 else int(torch.argmax(collided[:, i].to(torch.int8)))
                d["feas_at_collision"] = bool(feas[tc, i])
                d["feas_at_prec"] = [bool(feas[max(0, tc - k), i]) for k in range(1, KPREC + 1)]
            # precursor window for D4 (last KPREC steps ending at event, or at max_steps for timeout)
            end = (es if es >= 0 else max_steps)
            w0 = max(0, end - KPREC); w1 = max(w0 + 1, end)
            d["prec_sat_tau"] = float(satu_tau[w0:w1, i].float().mean())
            d["prec_boxclip_tau"] = float(boxclip_tau[w0:w1, i].float().mean())
            rec.append(d)
    wallA = time.time() - t0
    outc = np.array([r["outcome"] for r in rec])
    rate = lambda k: float((outc == k).mean())
    n_coll = int((outc == "collision").sum()); n_to = int((outc == "timeout").sum())

    # ---- D2 collision feasibility split ----
    coll = [r for r in rec if r["outcome"] == "collision"]
    feas_at_coll = np.array([r["feas_at_collision"] for r in coll])
    # feasible at ANY precursor within K (a safe action existed on the approach)
    feas_prec_any = np.array([any(r["feas_at_prec"]) for r in coll])
    D2 = dict(n_collision=n_coll,
              frac_feasible_at_collision=float(feas_at_coll.mean()),
              frac_infeasible_at_collision=float((~feas_at_coll).mean()),
              frac_feasible_any_prec_K5=float(feas_prec_any.mean()))

    # ---- D3 IC stratification ----
    th = np.array([r["th0"] for r in rec]); om = np.array([r["om0"] for r in rec])
    inw = np.array([r["inw"] for r in rec])
    def strat(mask_name, bins, key):
        arr = {"th": th, "om": om, "inw": inw}[key]
        rows = []
        for lo, hi, lab in bins:
            m = (arr >= lo) & (arr < hi)
            n = int(m.sum())
            rows.append(dict(band=lab, n=n,
                             coll=float((outc[m] == "collision").mean()) if n else 0.0,
                             timeout=float((outc[m] == "timeout").mean()) if n else 0.0,
                             reach=float((outc[m] == "goal").mean()) if n else 0.0))
        return rows
    pi = np.pi
    D3 = dict(
        by_theta0=strat("theta", [(0, pi / 6, "[0,pi/6)"), (pi / 6, pi / 2, "[pi/6,pi/2)"),
                                  (pi / 2, pi + 1e-6, "[pi/2,pi]")], "th"),
        by_inward_speed=strat("inw", [(-1e9, 0.0, "receding(<0)"), (0.0, 0.75, "[0,0.75)"),
                                      (0.75, 1e9, ">=0.75")], "inw"),
        by_omega0=strat("omega", [(0, 0.5, "[0,0.5)"), (0.5, 1.0, "[0.5,1.0)"), (1.0, 1e9, ">=1.0")], "om"),
    )

    # ---- D4 torque headroom at precursors (over failing episodes: collision + timeout) ----
    fail = [r for r in rec if r["outcome"] in ("collision", "timeout")]
    D4 = dict(n_fail=len(fail),
              mean_prec_sat_tau_frac=float(np.mean([r["prec_sat_tau"] for r in fail])),
              mean_prec_boxclip_tau_frac=float(np.mean([r["prec_boxclip_tau"] for r in fail])),
              coll_prec_sat_tau=float(np.mean([r["prec_sat_tau"] for r in coll])) if coll else 0.0,
              coll_prec_boxclip_tau=float(np.mean([r["prec_boxclip_tau"] for r in coll])) if coll else 0.0,
              to_prec_sat_tau=float(np.mean([r["prec_sat_tau"] for r in rec if r["outcome"] == "timeout"])) if n_to else 0.0,
              to_prec_boxclip_tau=float(np.mean([r["prec_boxclip_tau"] for r in rec if r["outcome"] == "timeout"])) if n_to else 0.0)

    # ---- Roll B (400) : D1 timeout conversion ----
    t0 = time.time(); MS = 400
    o200 = []; o400 = []; dist_at200 = []; dist_min400 = []; dist_at400 = []
    slope_180_200 = []; to200_mask = []
    for s0 in range(0, N, chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        S, res400, *_ = roll_chunk(system, h_fn, policy, cfg, bs, dev, MS)
        # outcome at 200 = resolve on the truncated first 201 states
        S200 = S[:201]; m200 = step_outcomes(S200, bs, system, cfg); r200 = resolve_outcome(m200)
        g = torch.as_tensor(goal_np[s0:s0 + S.shape[1]], device=dev, dtype=torch.float32)
        p = system.position(S)                                             # [MS+1, B, 2]
        dist = torch.linalg.norm(p - g.unsqueeze(0), dim=-1)               # [MS+1, B]
        B = S.shape[1]
        for i in range(B):
            o2 = r200.outcome[i]; o4 = res400.outcome[i]
            o200.append(o2); o400.append(o4)
            to200_mask.append(o2 == "timeout")
            dist_at200.append(float(dist[200, i])); dist_at400.append(float(dist[MS, i]))
            dist_min400.append(float(dist[:, i].min()))
            slope_180_200.append(float(dist[200, i] - dist[180, i]))       # <0 = still descending
    wallB = time.time() - t0
    o200 = np.array(o200); o400 = np.array(o400); to200 = np.array(to200_mask)
    d200 = np.array(dist_at200); d400 = np.array(dist_at400); dmin = np.array(dist_min400); sl = np.array(slope_180_200)
    # of the ORIGINAL 200-step timeouts, what did 400 steps do?
    conv = o400[to200]
    n_to200 = int(to200.sum())
    goal_radius = float(cfg["env"]["goal_radius"])
    # descending (en-route) vs loiter (asymptotic) among original timeouts, by [180,200] slope
    desc = sl[to200] < -0.02                                              # still closing > 0.02 over 20 steps
    D1 = dict(
        outcome_shift_200_to_400=dict(
            reach_200=float((o200 == "goal").mean()), reach_400=float((o400 == "goal").mean()),
            timeout_200=float((o200 == "timeout").mean()), timeout_400=float((o400 == "timeout").mean()),
            collision_200=float((o200 == "collision").mean()), collision_400=float((o400 == "collision").mean())),
        n_timeout_200=n_to200,
        of_timeout200_at400=dict(
            goal=float((conv == "goal").mean()), timeout=float((conv == "timeout").mean()),
            collision=float((conv == "collision").mean()), oob=float((conv == "oob").mean()),
            stuck=float((conv == "stuck").mean())),
        timeout200_traj=dict(
            median_min_dist=float(np.median(dmin[to200])), median_dist_at200=float(np.median(d200[to200])),
            median_dist_at400=float(np.median(d400[to200])),
            frac_descending_at200=float(desc.mean()), frac_loiter_at200=float((~desc).mean()),
            frac_min_dist_within_goalradius=float((dmin[to200] <= goal_radius).mean())),
        wallB_s=round(wallB, 1))

    report = dict(run=run_dir.name, step=step, n=N, max_steps=max_steps,
                  outcomes=dict(reach=rate("goal"), collision=rate("collision"), oob=rate("oob"),
                                stuck=rate("stuck"), timeout=rate("timeout")),
                  wallA_s=round(wallA, 1), D1=D1, D2=D2, D3=D3, D4=D4)
    json.dump(report, open(SP / "quadrotor_m6_residual.json", "w"), indent=2)

    # ---- print ----
    print(f"M6 residual diagnostics @ {run_dir.name}/best.pt step {step}  pool n={N}")
    print(f"  Roll A outcomes (max_steps={max_steps}): reach={rate('goal'):.4f} coll={rate('collision'):.4f} "
          f"timeout={rate('timeout'):.4f} oob={rate('oob'):.4f} stuck={rate('stuck'):.4f}  ({wallA:.0f}s)")
    print(f"\nD1 timeout conversion (Roll B max_steps=400, {wallB:.0f}s):")
    s = D1["outcome_shift_200_to_400"]
    print(f"  outcome 200->400: reach {s['reach_200']:.4f}->{s['reach_400']:.4f}  timeout {s['timeout_200']:.4f}->{s['timeout_400']:.4f}  "
          f"coll {s['collision_200']:.4f}->{s['collision_400']:.4f}")
    c = D1["of_timeout200_at400"]
    print(f"  of the {n_to200} original 200-timeouts, at 400: goal={c['goal']:.3f} timeout={c['timeout']:.3f} "
          f"coll={c['collision']:.3f} oob={c['oob']:.3f} stuck={c['stuck']:.3f}")
    tr = D1["timeout200_traj"]
    print(f"  200-timeout ||p-g||: median min={tr['median_min_dist']:.3f} @200={tr['median_dist_at200']:.3f} "
          f"@400={tr['median_dist_at400']:.3f} | descending@200={tr['frac_descending_at200']:.3f} "
          f"loiter@200={tr['frac_loiter_at200']:.3f} | min-dist-within-goalR={tr['frac_min_dist_within_goalradius']:.3f}")
    print(f"\nD2 collision feasibility split (n_coll={D2['n_collision']}):")
    print(f"  FEASIBLE at collision (safe action existed -> policy/torque-bound) = {D2['frac_feasible_at_collision']:.3f}")
    print(f"  INFEASIBLE at collision (none -> unavoidable given IC)            = {D2['frac_infeasible_at_collision']:.3f}")
    print(f"  feasible at ANY of K=1..5 precursors                             = {D2['frac_feasible_any_prec_K5']:.3f}")
    print(f"\nD3 IC stratification (collision / timeout / reach rate per band):")
    for key, rows in D3.items():
        print(f"  {key}:")
        for r in rows:
            print(f"    {r['band']:>16}  n={r['n']:>4}  coll={r['coll']:.3f} timeout={r['timeout']:.3f} reach={r['reach']:.3f}")
    print(f"\nD4 torque headroom at precursors (last {KPREC} steps before event):")
    print(f"  over all failing (n={D4['n_fail']}): |tau| saturated frac={D4['mean_prec_sat_tau_frac']:.3f}  "
          f"torque-box-clip frac={D4['mean_prec_boxclip_tau_frac']:.3f}")
    print(f"  collisions: sat={D4['coll_prec_sat_tau']:.3f} boxclip={D4['coll_prec_boxclip_tau']:.3f} | "
          f"timeouts: sat={D4['to_prec_sat_tau']:.3f} boxclip={D4['to_prec_boxclip_tau']:.3f}")
    print(f"\nsaved {SP / 'quadrotor_m6_residual.json'}")


if __name__ == "__main__":
    main()
