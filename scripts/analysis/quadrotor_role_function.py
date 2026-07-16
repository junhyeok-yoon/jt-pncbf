"""v2.6.1 role-function diagnostic (R1-R3) — do the two networks perform their DESIGNED roles on the seed-42
M6 best.pt? Not "is cps good" but: does V̂ certify correctly, does π navigate on its own, and is the filter a
rare corrector (healthy labor division) vs a driver propping up a deferring policy? Read-only; value+policy
FROZEN; full pool n2000 seed23456. Two read-only re-rolls (filtered + nominal-only) + a near-B_0 sampled set.
Every number disk-derived; writes a JSON dump.
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
from src.common.quadrotor_barrier import make_barrier_fn, sample_near_B0_states, thrust_axis
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.evaluate import first_physical_event_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
ACTIVE_THR = 1.0e-3
BENIGN_SURF = 1.0    # nearest-obstacle surface distance > 1.0 = benign region
KPREC = 5


def nearest_obstacle(x, sc):
    """(surface_dist [k], away_unit [k,2]) to the nearest ACTIVE obstacle, for a SINGLE scene sc with
    obstacle_centers [K,2] / radii [K] / active [K] shared across the k states x [k,6]."""
    p = x[:, :2]                                                        # [k,2]
    c = sc.obstacle_centers.to(x.dtype); r = sc.obstacle_radii.to(x.dtype); a = sc.obstacle_active.to(torch.bool)
    d = torch.linalg.norm(p.unsqueeze(1) - c.unsqueeze(0), dim=-1) - r.unsqueeze(0)   # [k,K]
    d = d.masked_fill(~a.unsqueeze(0), float("inf"))
    j = torch.argmin(d, dim=1)                                         # [k]
    ar = torch.arange(p.shape[0], device=p.device)
    ctr = c[j]; surf = d[ar, j]                                        # [k,2], [k]
    away = p - ctr; away = away / torch.linalg.norm(away, dim=1, keepdim=True).clamp_min(1e-9)
    return surf, away


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.1__*seed42"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]; step = int(ck.get("step", -1))
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system)
    c_gain = float(cfg["env"]["quadrotor_planar"]["c_gain"]); h_scale = float(cfg["env"].get("h_scale", 1.0))
    hstar_fn = make_barrier_fn(c_gain, h_scale, position_only=False)
    params = _hardnet_params(cfg); bounds = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"]); chunk = 250
    m = float(system.mass); g = float(system.gravity)
    scenes = load_pool(POOL).scenes; N = len(scenes)
    pol = lambda x, sc: policy(system.observation(x, sc))

    # accumulators
    dun_all = []            # per-step ||u_safe-u_nom|| (subsample)
    unorm_all = []          # per-step ||u_safe|| (subsample)
    dun_benign = []; dun_near = []
    filt_out = []; nom_out = []          # per-episode outcomes (aligned by index)
    v0 = []                              # V̂ at IC per episode
    coll_tags = []                       # per collision: dict of role tags
    to_tags = []                         # per timeout: dict
    fastapp_away = []                    # per pre-collision step: away-accel sign (nominal)

    t0 = time.time()
    for s0 in range(0, N, chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        # V̂ at IC
        with torch.no_grad():
            v0c = h_fn(x, bs).reshape(-1).cpu().numpy()
        # ---- FILTERED roll (capture per step) ----
        states = [x.clone()]; un_s = []; us_s = []; empt = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        xx = x.clone()
        with torch.no_grad():
            for t in range(max_steps):
                un = pol(xx, bs)
                h, lf, lg = _cbf_terms(system, h_fn, xx, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params); row = -lf - alpha * h
                proj = _base_projection(un, lg, row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg, row, bounds)
                empt[t] = empty; un_s.append(un); us_s.append(u)
                xx = rk4_step(system, xx, u, dt); states.append(xx.clone())
        S = torch.stack(states, 0); UN = torch.stack(un_s, 0); US = torch.stack(us_s, 0)
        masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        collided = masks.collided
        an = first_physical_event_step(masks); an = torch.where(an >= 0, an, torch.full_like(an, max_steps))
        act = torch.arange(max_steps, device=dev).unsqueeze(1) < an.unsqueeze(0)
        dun = torch.linalg.norm(US - UN, dim=2)             # [T,B]
        usn = torch.linalg.norm(US, dim=2)
        # region tag per step by nearest-obstacle surface dist (use state at each step)
        for i in range(B):
            am = act[:, i]
            if int(am.sum()) == 0:
                continue
            dsub = dun[am, i].cpu().numpy(); usub = usn[am, i].cpu().numpy()
            dun_all.append(dsub[::5]); unorm_all.append(usub[::5])
        # region-stratified ||du|| (benign vs near) — sample states along active steps
        with torch.no_grad():
            for i in range(B):
                am = act[:, i]
                idx = torch.nonzero(am).flatten()
                if idx.numel() == 0:
                    continue
                xi = S[idx, i]                       # [k,6]
                surf, _ = nearest_obstacle(xi, _single_scene(bs, i, dev))
                du_i = dun[idx, i]
                benign = surf > BENIGN_SURF
                if benign.any():
                    dun_benign.append(float(du_i[benign].mean()))
                if (~benign).any():
                    dun_near.append(float(du_i[~benign].mean()))
        # per-episode outcomes + V0
        for i in range(B):
            filt_out.append(res.outcome[i]); v0.append(float(v0c[i]))
        # collision attribution + fast-approach away-analysis
        for i in range(B):
            if res.outcome[i] != "collision":
                continue
            tc = int(torch.argmax(collided[:, i].to(torch.int8)))
            w0 = max(0, tc - KPREC)
            # role tags
            with torch.no_grad():
                xw = S[w0:tc + 1, i]                                    # pre-collision states
                sc_i = _single_scene(bs, i, dev)
                vhat_w = h_fn(xw, sc_i).reshape(-1)
                hstar_w = hstar_fn(xw, sc_i).reshape(-1)
                # mis-certified: V̂<=0 anywhere in the window while h_star>0 (reachable unsafe uncertified)
                mis_cert = bool(((vhat_w <= 0.0) & (hstar_w > 0.0)).any().item())
                # filter infeasible in box at any pre-collision step
                infeas = bool(empt[w0:tc, i].any().item()) if tc > w0 else bool(empt[tc, i].item())
                # nominal toward obstacle: net accel from u_nom projected on away-dir (avg over window)
                un_w = UN[w0:min(tc + 1, max_steps), i]                 # [k,2]
                xw2 = S[w0:min(tc + 1, max_steps), i]
                Re = thrust_axis(xw2)                                   # [k,2]
                a_nom = (un_w[:, :1] / m) * Re - torch.tensor([0.0, g], device=dev)  # accel
                _, away = nearest_obstacle(xw2, sc_i)
                away_comp = torch.sum(a_nom * away, dim=1)             # >0 = accel away
                toward = bool((away_comp.mean() < 0.0).item())
                fastapp_away.extend(away_comp.cpu().numpy().tolist())
            coll_tags.append({"mis_cert": mis_cert, "toward_obstacle": toward, "filter_infeasible": infeas,
                              "t_collision": tc})
        # timeout attribution
        for i in range(B):
            if res.outcome[i] != "timeout":
                continue
            with torch.no_grad():
                xe = S[max_steps - KPREC:max_steps, i]                  # last states
                sc_i = _single_scene(bs, i, dev)
                goal = bs.goal[i] if bs.goal.ndim > 1 else bs.goal
                pe = xe[:, :2]; dist = torch.linalg.norm(pe - goal.to(dev), dim=1)
                near_goal = bool((dist.min() < 0.5).item())
                # pi goal-directed near goal: nominal accel component toward goal
                un_e = UN[max_steps - KPREC:max_steps, i]
                Re = thrust_axis(xe)
                a_nom = (un_e[:, :1] / m) * Re - torch.tensor([0.0, g], device=dev)
                to_goal = (goal.to(dev) - pe); to_goal = to_goal / torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-9)
                goal_comp = torch.sum(a_nom * to_goal, dim=1).mean()
                pi_goaldir = bool((goal_comp > 0.0).item())
                # V̂ blocks: V̂>0 (unsafe-flagged) on the near-goal states -> filter deflects
                vhat_e = h_fn(xe, sc_i).reshape(-1)
                vhat_blocks = bool((vhat_e > 0.0).any().item())
            to_tags.append({"near_goal": near_goal, "pi_goal_directed": pi_goaldir, "vhat_blocks": vhat_blocks})
        # ---- NOMINAL-ONLY roll (filter OFF) ----
        xx = x.clone(); nstates = [xx.clone()]
        with torch.no_grad():
            for t in range(max_steps):
                un = pol(xx, bs)
                u = torch.clamp(un, min=bounds[:, 0], max=bounds[:, 1])
                xx = rk4_step(system, xx, u, dt); nstates.append(xx.clone())
        NS = torch.stack(nstates, 0); nmask = step_outcomes(NS, bs, system, cfg); nres = resolve_outcome(nmask)
        for i in range(B):
            nom_out.append(nres.outcome[i])
    wall = time.time() - t0

    # ---------- R2 near-B0 direction / boundary / soundness ----------
    gen = torch.Generator(device="cpu")
    xb, scb = sample_near_B0_states(system, cfg, 4000, torch.device("cpu"), torch.float32, generator=None) \
        if False else sample_near_B0_states(system, cfg, 4000, dev, torch.float32)
    with torch.no_grad():
        vb = h_fn(xb, scb).reshape(-1)
        hb = hstar_fn(xb, scb).reshape(-1)
    # L_g of each in control space
    zero_u = torch.zeros(xb.shape[0], system.action_dim, device=dev)
    _, _, lgV = _cbf_terms(system, h_fn, xb, scb, zero_u, create_graph=False)
    _, _, lgH = _cbf_terms(system, hstar_fn, xb, scb, zero_u, create_graph=False)
    lgV = lgV.detach(); lgH = lgH.detach()
    cos = torch.sum(lgV * lgH, dim=1) / (torch.linalg.norm(lgV, dim=1) * torch.linalg.norm(lgH, dim=1)).clamp_min(1e-9)
    cos_np = cos.cpu().numpy()
    band = 0.1
    boundary_mismatch = float(((vb.sign() != hb.sign()) & (vb.abs() > band) & (hb.abs() > band)).float().mean())
    cert_safe = vb <= 0.0
    cert_truly_safe = float((hb[cert_safe] <= 0.0).float().mean()) if cert_safe.any() else float("nan")

    # ---------- assemble ----------
    fo = np.array(filt_out); no = np.array(nom_out); v0a = np.array(v0)
    r = lambda arr, k: float((arr == k).mean())
    dun_cat = np.concatenate(dun_all); un_cat = np.concatenate(unorm_all)
    active_frac = float((dun_cat > ACTIVE_THR).mean())
    effort_share = float(dun_cat.sum() / max(un_cat.sum(), 1e-9))
    # forward invariance: episodes with V0<=0 that collide
    safe0 = v0a <= 0.0
    fi_collide = float((fo[safe0] == "collision").mean()) if safe0.any() else float("nan")
    # compensation: nominal-fail episodes rescued by filter
    nom_fail = (no != "goal")
    rescued = float((fo[nom_fail] == "goal").mean()) if nom_fail.any() else float("nan")

    def taghist(tags, keys):
        return {k: (float(np.mean([t[k] for t in tags])) if tags else None) for k in keys}

    out = {
        "run": run_dir.name, "step": step, "n": N, "wall_s": round(wall, 1),
        "R1": {
            "nominal_only": {"reach": r(no, "goal"), "collision": r(no, "collision"),
                             "timeout": r(no, "timeout"), "oob": r(no, "oob"), "stuck": r(no, "stuck")},
            "filtered": {"reach": r(fo, "goal"), "collision": r(fo, "collision"), "timeout": r(fo, "timeout")},
            "filter_active_frac": active_frac,
            "filter_effort_share": effort_share,
            "dun_median": float(np.median(dun_cat)), "dun_p90": float(np.percentile(dun_cat, 90)),
            "dun_benign_mean": float(np.mean(dun_benign)) if dun_benign else None,
            "dun_near_mean": float(np.mean(dun_near)) if dun_near else None,
            "fastapp_toward_frac": float(np.mean([t["toward_obstacle"] for t in coll_tags])) if coll_tags else None,
            "fastapp_away_accel_median": float(np.median(fastapp_away)) if fastapp_away else None,
        },
        "R2": {
            "forward_invariance_v0safe_collide_frac": fi_collide,
            "n_v0_safe": int(safe0.sum()),
            "direction_cos_lgV_lgHstar_median": float(np.median(cos_np)),
            "direction_cos_aligned_frac": float((cos_np > 0).mean()),
            "boundary_sign_mismatch_frac": boundary_mismatch,
            "certified_safe_truly_safe_frac": cert_truly_safe,
        },
        "R3": {
            "n_collision": len(coll_tags), "n_timeout": len(to_tags),
            "collision_attribution": taghist(coll_tags, ["mis_cert", "toward_obstacle", "filter_infeasible"]),
            "timeout_attribution": taghist(to_tags, ["near_goal", "pi_goal_directed", "vhat_blocks"]),
            "compensation_nomfail_rescued_frac": rescued, "n_nom_fail": int(nom_fail.sum()),
        },
    }
    json.dump(out, open(SP / "quadrotor_role_function.json", "w"), indent=2)
    import pprint
    pprint.pprint(out)
    print("saved", SP / "quadrotor_role_function.json")


def _single_scene(bs, i, dev):
    from types import SimpleNamespace
    return SimpleNamespace(
        obstacle_centers=bs.obstacle_centers[i], obstacle_radii=bs.obstacle_radii[i],
        obstacle_active=bs.obstacle_active[i],
        goal=bs.goal[i] if bs.goal.ndim > 1 else bs.goal)


if __name__ == "__main__":
    main()
