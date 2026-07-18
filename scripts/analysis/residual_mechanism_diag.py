"""v2.7.0 residual-mechanism discrimination (DIAGNOSTIC, no training, read-only).

Discriminates the mechanism behind the M5 residual collisions: recovery TIME/CREDIT (BPTT window too short)
vs LABEL self-confirmation (pessimistic certificate). Re-rolls the M5 best.pt on the frozen full pool at
batch_size=2000 (identical to run_full's evaluate(), src/eval/evaluate.py:139 batch=len(scenes)), captures
per-step states + V_hat (deployed_h) + h (signed_h geometric) + HardNet infeasibility, and VERIFIES the
collision episode-ID set matches the M5 eval_episodes.csv before any analysis (mismatch reported, affected
measurement stopped). D-i V_hat descent anatomy · D-ii recovery duration vs bptt_T · D-iii false-doom overlap.
Facts only; the verdict is not drawn here.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

import sys

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
# v2.7.0 iter-2: run dir from argv (default = the obs-fix iteration-1 run). Output is run-id-scoped.
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/v2.7.0__20260717-025050__seed42"
RID = RUN.name
FIGDIR = RUN / "figures" / "residual_mechanism"
TILT = np.pi / 2
TOL = 1e-3


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def eval_collision_ids():
    ids = []
    with open(RUN / "eval_episodes.csv") as f:
        for row in csv.DictReader(f):
            if row["mode"] == "final" and row["outcome"] == "collision":
                ids.append(int(row["episode_idx"]))
    return sorted(ids)


def roll(dev):
    """Single-batch (2000) roll matching run_full's evaluate(); capture states, V_hat, empty, outcome."""
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    ms = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"]); h_scale = float(cfg["env"]["h_scale"])
    scenes = load_pool(POOL).scenes; N = len(scenes)

    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)            # ALL 2000 in one batch
    x = initial_states_from_batch(bs).float()
    states = [x.clone()]; vhat = []; emp = []
    with torch.no_grad():
        for t in range(ms):
            obs = system.observation(x, bs)
            vhat.append(h_fn(x, bs).reshape(-1).clone())                  # V_hat(x_t) = deployed_h
            un = policy(obs)
            h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
            alpha = _base_alpha(h, params); row = -lf - alpha * h
            proj = _base_projection(un, lg, row, bounds, params)
            u, empty = _box_aware_projection(un, proj, lg, row, bounds)
            emp.append(empty.clone())
            x = rk4_step(system, x, u, dt); states.append(x.clone())
    S = torch.stack(states, 0)                                            # (T+1, N, 6)
    res = resolve_outcome(step_outcomes(S, bs, system, cfg))
    Snp = S.permute(1, 0, 2).cpu().numpy()                                # (N, T+1, 6)
    Vh = torch.stack(vhat, 0).permute(1, 0).cpu().numpy()                 # (N, T)
    E = torch.stack(emp, 0).permute(1, 0).cpu().numpy()                   # (N, T) bool infeasible
    outcome = np.array([1 if res.outcome[i] == "collision" else (2 if res.outcome[i] == "goal" else 0)
                        for i in range(N)], np.int8)
    # geometric signed_h per step + collision (impact) step = first t with signed_h < 0
    Sh = np.zeros((N, ms + 1), np.float32)
    with torch.no_grad():
        for t in range(ms + 1):
            Sh[:, t] = signed_h(S[t, :, :2], bs, h_scale).reshape(-1).cpu().numpy()
    return dict(states=Snp, vhat=Vh, empty=E, outcome=outcome, signed_h=Sh, scenes=scenes, ms=ms, dt=dt,
                bptt_T=int(cfg["training"]["jt"]["bptt_T"]),
                # handles for the authority / offline probes (P-A4/P-B/P-C/P-D) — additive, existing keys unchanged
                system=system, h_fn=h_fn, params=params, bounds=bounds, cfg=cfg, vnet=vnet, dev=dev, policy=policy,
                h_scale=h_scale)


def impact_step(sh_row, ms):
    below = np.where(sh_row < 0)[0]
    return int(below[0]) if below.size else ms


def classify_vhat(v, imp):
    """(a) crossed-into-safe / (b) descended-never-safe / (c) never-descended / (other)."""
    seg = v[:imp + 1] if imp + 1 <= len(v) else v
    if seg.size < 2:
        return "other"
    v0 = seg[0]; vmin = seg.min(); targmin = int(seg.argmin())
    rose_after = (seg[targmin:].max() > vmin + TOL)
    if vmin >= v0 - TOL:
        return "c_never_descended"
    if vmin < 0 and rose_after:
        return "a_crossed_into_safe"
    if vmin >= 0:
        return "b_descended_never_safe"
    return "other"                                                        # descended <0 but never rose


INTERP_MAP = (
    "[P-B interpretation map — printed verbatim, no conclusion]\n"
    "  continuous-infeasible AND discrete-improvable -> supports a discrete/multi-step filter row axis\n"
    "  both dead (continuous-infeasible AND NOT discrete-improvable) -> supports upstream policy-side avoidance\n"
    "  mixed -> report mixed")


def _grid_controls(system, dev):
    """G = box corners + hover + max-thrust pair (|G|<=16). Shared with the filter fallback (no fork)."""
    from src.common.kstep_fallback import grid_controls
    return grid_controls(system, dev, torch.float32)


def _mc_md_lg(R, ep_idx, t_idx):
    """Continuous margin m_c, discrete margin m_d, ||L_g V_hat|| for (episode, step) pairs. Batched."""
    system, h_fn, params, dev = R["system"], R["h_fn"], R["params"], R["dev"]
    scenes = R["scenes"]; dt = R["dt"]
    sub = batch_scenes([scenes[i] for i in ep_idx], device=dev, dtype=torch.float32)
    x = torch.tensor(np.stack([R["states"][i, t] for i, t in zip(ep_idx, t_idx)]), dtype=torch.float32, device=dev)
    un = torch.zeros(x.shape[0], 2, device=dev, dtype=torch.float32)
    h, lf, lg = _cbf_terms(system, h_fn, x, sub, un, create_graph=False)   # h=V_hat, lf=L_fV, lg=L_gV [n,2]
    alpha = _base_alpha(h, params)
    b = system.u_bounds.to(dev, torch.float32)
    umin = b[:, 0].unsqueeze(0); umax = b[:, 1].unsqueeze(0)               # [1,2]
    min_lgu = torch.where(lg > 0, lg * umin, lg * umax).sum(dim=1)         # min_{u in U} L_gV u
    m_c = (lf + min_lgu + alpha * h).detach().cpu().numpy()                # >0 == deployed row infeasible
    lg_norm = torch.linalg.norm(lg.detach(), dim=1).cpu().numpy()
    # discrete margin over the control grid
    G = _grid_controls(system, dev); k = G.shape[0]
    with torch.no_grad():
        v_x = h_fn(x, sub).reshape(-1)                                     # [n]
        xe = x.unsqueeze(1).expand(-1, k, -1).reshape(-1, x.shape[1])      # [n*k,6]
        ue = G.unsqueeze(0).expand(x.shape[0], -1, -1).reshape(-1, 2)      # [n*k,2]
        sube = batch_scenes([scenes[i] for i in ep_idx for _ in range(k)], device=dev, dtype=torch.float32)
        xn = rk4_step(system, xe, ue, dt)
        vn = h_fn(xn, sube).reshape(x.shape[0], k)
        m_d = (vn - v_x.unsqueeze(1)).min(dim=1).values.detach().cpu().numpy()   # <0 == discrete-improvable
    return m_c, m_d, lg_norm, h.detach().cpu().numpy()


def _md_kstep(R, ep_idx, t_idx, k):
    """P-B' k-step discrete margin: min over TWO-PHASE piecewise-constant candidate sequences (G per phase,
    phase lengths ceil(k/2) + floor(k/2)) of [V_hat(x_k) - V_hat(x0)]. <0 == k-step improvable. Removes P-B's
    1-step-underestimation caveat by giving torque time to act over O(k*dt)."""
    system, h_fn, dev = R["system"], R["h_fn"], R["dev"]; scenes = R["scenes"]; dt = R["dt"]
    G = _grid_controls(system, dev); ng = G.shape[0]; k1 = (k + 1) // 2; k2 = k - k1
    sub = batch_scenes([scenes[i] for i in ep_idx], device=dev, dtype=torch.float32)
    x0 = torch.tensor(np.stack([R["states"][i, t] for i, t in zip(ep_idx, t_idx)]), dtype=torch.float32, device=dev)
    n = x0.shape[0]
    with torch.no_grad():
        v0 = h_fn(x0, sub).reshape(-1)
        best = torch.full((n,), float("inf"), device=dev)
        for a in range(ng):
            for b in range(ng):
                x = x0.clone()
                u1 = G[a].unsqueeze(0).expand(n, -1); u2 = G[b].unsqueeze(0).expand(n, -1)
                for _ in range(k1):
                    x = rk4_step(system, x, u1, dt)
                for _ in range(k2):
                    x = rk4_step(system, x, u2, dt)
                best = torch.minimum(best, h_fn(x, sub).reshape(-1) - v0)
    return best.cpu().numpy()


def probe_pbprime(R, pb_recs, TILT):
    """P-B' — k-step (k in {5,10}) authority 2x2 tables on the t0 residual states, beside 1-step m_d / m_c."""
    t0 = [r for r in pb_recs if r["state"] == "t0"]
    ep = [r["episode_idx"] for r in t0]; ts = [r["t_step"] for r in t0]
    out = {"n": len(t0)}
    for k in (5, 10):
        mdk = _md_kstep(R, ep, ts, k)
        tab = {(ci, di): 0 for ci in (0, 1) for di in (0, 1)}
        tabt = {(ci, di): 0 for ci in (0, 1) for di in (0, 1)}
        for r, m in zip(t0, mdk):
            ci = r["cont_infeasible"]; di = int(m < 0)
            tab[(ci, di)] += 1
            if r["tilted"] == 1:
                tabt[(ci, di)] += 1
        out[f"k{k}"] = {"all": _fmt_tbl(tab, len(t0)),
                        "tilted": _fmt_tbl(tabt, sum(1 for r in t0 if r["tilted"] == 1)),
                        "frac_kstep_improvable": round(float((mdk < 0).mean()), 3),
                        "cont_infeas_and_kstep_improvable": tab[(1, 1)],
                        "cont_infeas_and_both_dead": tab[(1, 0)]}
    return out


def probe_pb(R, aset, th0, TILT):
    """P-B — discrete-vs-continuous authority at failure states (t0 and first-infeasibility if later)."""
    E, states = R["empty"], R["states"]; ms = R["ms"]
    rows = []
    for gi in aset:
        gi = int(gi)
        infs = np.where(E[gi])[0]
        tfi = int(infs[0]) if infs.size else None
        pts = [("t0", 0)]
        if tfi is not None and tfi > 0:
            pts.append(("first_infeas", tfi))
        for tag, t in pts:
            rows.append((gi, tag, t))
    ep_idx = [r[0] for r in rows]; t_idx = [r[2] for r in rows]
    m_c, m_d, lgn, vh = _mc_md_lg(R, ep_idx, t_idx)
    system = R["system"]; dev = R["dev"]; scenes = R["scenes"]
    # per-state kinematics
    xt = torch.tensor(np.stack([states[i, t] for i, t in zip(ep_idx, t_idx)]), dtype=torch.float32, device=dev)
    spd = system.speed(xt).cpu().numpy()
    recs = []
    for j, (gi, tag, t) in enumerate(rows):
        thg = abs(float(wrap(states[gi, 0, 2])))
        recs.append(dict(episode_idx=gi, state=tag, t_step=t, theta0=round(thg, 4), tilted=int(thg >= TILT),
                         speed=round(float(spd[j]), 4), V_hat=round(float(vh[j]), 4),
                         lg_norm=round(float(lgn[j]), 5), m_c=round(float(m_c[j]), 5), m_d=round(float(m_d[j]), 5),
                         cont_infeasible=int(m_c[j] > 0), disc_improvable=int(m_d[j] < 0),
                         empty_flag=int(E[gi, min(t, ms - 1)])))
    # 2x2 tables at t0
    def table(mask):
        sub = [r for r in recs if r["state"] == "t0" and mask(r)]
        t = {(ci, di): 0 for ci in (0, 1) for di in (0, 1)}
        for r in sub:
            t[(r["cont_infeasible"], r["disc_improvable"])] += 1
        return t, len(sub)
    tbl_all, n_all = table(lambda r: True)
    tbl_tilt, n_tilt = table(lambda r: r["tilted"] == 1)
    # cross-check m_c>0 vs deployed empty@t0
    t0recs = [r for r in recs if r["state"] == "t0"]
    agree = sum(1 for r in t0recs if r["cont_infeasible"] == r["empty_flag"])
    return recs, dict(table_all=_fmt_tbl(tbl_all, n_all), table_tilted=_fmt_tbl(tbl_tilt, n_tilt),
                      n_states_total=len(recs), n_first_infeas=sum(1 for r in recs if r["state"] == "first_infeas"),
                      mc_vs_empty_t0_agreement=f"{agree}/{len(t0recs)}")


def _fmt_tbl(t, n):
    return {"n": n, "cont_infeas & disc_improvable": t[(1, 1)], "cont_infeas & NOT disc_improvable": t[(1, 0)],
            "feasible & disc_improvable": t[(0, 1)], "feasible & NOT disc_improvable": t[(0, 0)]}


def probe_pc(R, aset, th0, TILT):
    """P-C — entry anatomy of the residual collisions: t0->first-infeasibility timing/speed/h/approach-speed."""
    E, states, Sh, ms, dt = R["empty"], R["states"], R["signed_h"], R["ms"], R["dt"]
    system, dev = R["system"], R["dev"]
    # pool median speed over all rolled episode states (reference threshold for "fast approach")
    allx = torch.tensor(states.reshape(-1, states.shape[-1]), dtype=torch.float32, device=dev)
    pool_med_speed = float(np.median(system.speed(allx).cpu().numpy()))
    born, fast, other = [], [], []
    recs = []
    for gi in aset:
        gi = int(gi)
        infs = np.where(E[gi])[0]; tfi = int(infs[0]) if infs.size else None
        x_entry = states[gi, tfi] if tfi is not None else states[gi, 0]
        xt = torch.tensor(x_entry[None], dtype=torch.float32, device=dev)
        v_entry = float(system.speed(xt).cpu().numpy()[0])
        h_entry = float(Sh[gi, tfi if tfi is not None else 0])
        # approach speed toward nearest active obstacle at entry
        sc = R["scenes"][gi]
        cen = np.asarray(sc.obstacle_centers); act = np.asarray(sc.obstacle_active).astype(bool)
        p = x_entry[:2]; vel = x_entry[3:5]
        appr = None
        if act.any():
            d = np.linalg.norm(cen[act] - p, axis=1); nn = np.where(act)[0][int(d.argmin())]
            u = cen[nn] - p; un = u / (np.linalg.norm(u) + 1e-9); appr = float(np.dot(vel, un))
        born_flag = bool(E[gi, 0])
        cls = "born_inside" if born_flag else ("fast_approach" if (v_entry > pool_med_speed) else "other")
        (born if cls == "born_inside" else fast if cls == "fast_approach" else other).append(gi)
        recs.append(dict(episode_idx=gi, t0_infeasible=int(born_flag), tfi=(tfi if tfi is not None else -1),
                         t_to_infeas_s=round((tfi * dt) if tfi is not None else -1, 4),
                         speed_entry=round(v_entry, 4), h_entry=round(h_entry, 4),
                         approach_speed=round(appr, 4) if appr is not None else None,
                         tilted=int(th0[gi] >= TILT), entry_class=cls))
    def summ(ids):
        if not ids:
            return {"n": 0}
        vv = [r["speed_entry"] for r in recs if r["episode_idx"] in set(ids)]
        return {"n": len(ids), "speed_entry_p50": round(float(np.median(vv)), 3)}
    return recs, dict(pool_median_speed=round(pool_med_speed, 4), n_total=len(aset),
                      born_inside=summ(born), fast_approach=summ(fast), other=summ(other),
                      counts=f"born_inside={len(born)} fast_approach={len(fast)} other={len(other)} (sum={len(born)+len(fast)+len(other)}/{len(aset)})")


def probe_d2(R):
    """P-A4 — D2 theta-decodability spot check: fit a small MLP obs->(sin,cos theta) on rolled states."""
    import time
    system, dev, cfg = R["system"], R["dev"], R["cfg"]
    scenes = R["scenes"]; states = R["states"]; ms = R["ms"]
    rng = np.random.default_rng(20260718)
    N = states.shape[0]
    # sample states across episodes/time; keep t index for the failure-cell (t<0.5s) split
    n_s = 12000
    ei = rng.integers(0, N, n_s); ti = rng.integers(0, states.shape[1], n_s)
    x = torch.tensor(np.stack([states[e, t] for e, t in zip(ei, ti)]), dtype=torch.float32, device=dev)
    sub = batch_scenes([scenes[int(e)] for e in ei], device=dev, dtype=torch.float32)
    with torch.no_grad():
        obs = system.observation(x, sub)
    th = x[:, 2]; targ = torch.stack([torch.sin(th), torch.cos(th)], dim=1)
    ntr = 10000
    net = torch.nn.Sequential(torch.nn.Linear(obs.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(1500):
        opt.zero_grad(); p = net(obs[:ntr]); loss = ((p - targ[:ntr]) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        pv = net(obs[ntr:]); pth = torch.atan2(pv[:, 0], pv[:, 1])
        err = np.abs(wrap(pth.cpu().numpy() - th[ntr:].cpu().numpy())) * 180 / np.pi
    th0_te = np.abs(wrap(states[ei[ntr:], 0, 2])); t_te = ti[ntr:]
    cell = (np.abs(wrap(th[ntr:].cpu().numpy())) >= (np.pi / 2)) & (t_te * R["dt"] < 0.5)
    return dict(overall_deg=round(float(err.mean()), 3),
                failure_cell_deg=round(float(err[cell].mean()), 3) if cell.any() else None,
                failure_cell_n=int(cell.sum()), baseline="iter-1 M6 failure-cell 2.28 deg (obs unchanged)")


def probe_pd(R):
    """P-D — offline collection with the FINAL trained policy (continuing vs legacy), distinct seed. GPU."""
    import time
    from src.common.filter_hardnet import _base_alpha, _base_projection, _box_aware_projection, _cbf_terms
    from src.common.quadrotor_barrier import value_target_barrier
    from src.envs.scene_init import sample_train_scene
    from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
    from src.frameworks.oc_pncbf.collection import OCReplayBuffer
    system, dev, cfg = R["system"], R["dev"], R["cfg"]
    h_fn, params, bounds, policy = R["h_fn"], R["params"], R["bounds"], R["policy"]
    dt = R["dt"]; horizon = int(cfg["training"]["oc_pncbf"]["horizon"]); B = int(cfg["collection"]["jt"]["episodes_per_collect"])
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")

    def jt_step(x, bs):
        un = policy(system.observation(x, bs))
        h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
        alpha = _base_alpha(h, params); row = -lf - alpha * h
        proj = _base_projection(un, lg, row, bounds, params)
        u, _ = _box_aware_projection(un, proj, lg, row, bounds)
        return u.detach()

    def h_batch(states_g, bscene):
        return value_target_barrier(system, states_g, bscene, cfg)

    NR = 25
    # continuing
    rng = np.random.default_rng(2024)
    buf = OCReplayBuffer(capacity=5_000_000)
    stt = ContinuingState.create(system, sampler, rng, B, cfg, dev, torch.float32, system_name="quadrotor_planar")
    tot = dict(r1=0, r2=0, r3=0, refill=0, seg=0, bnd=0, steps=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(NR):
        s = advance_round(stt, round_length=horizon, step_fn=jt_step, h_batch_fn=h_batch, scene_sampler=sampler,
                          rng=rng, config=cfg, buffer=buf, dt=dt, system_name="quadrotor_planar")
        tot["r1"] += s.r1_count; tot["r2"] += s.r2_count; tot["r3"] += s.r3_count; tot["refill"] += s.refill_count
        tot["seg"] += s.segments; tot["bnd"] += s.boundary_segments; tot["steps"] += s.steps
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cont_t = time.perf_counter() - t0
    seg_lens = np.array([tr.states.shape[0] for tr in buf._trajectories])
    den = max(tot["r1"] + tot["r2"] + tot["r3"], 1)
    # legacy: fixed-horizon rollouts with the same policy, identical total steps
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch
    rng2 = np.random.default_rng(7788)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tl = time.perf_counter()
    leg_eps = 0
    for i in range(NR):
        scs = [sampler(rng2) for _ in range(B)]
        bs = batch_scenes(scs, device=dev, dtype=torch.float32)
        x = system.wrap_state(initial_states_from_batch(bs).float())
        with torch.no_grad():
            for _ in range(horizon):
                x = rk4_step(system, x, jt_step(x, bs), dt)
        leg_eps += B
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    leg_t = time.perf_counter() - tl
    return dict(B=B, rounds=NR, horizon=horizon, note="OFFLINE measurement (final policy, seed 2024/7788, distinct from pool seed 23456)",
                R1_R2_R3=f"{tot['r1']}:{tot['r2']}:{tot['r3']}", mix=[round(tot['r1']/den,3), round(tot['r2']/den,3), round(tot['r3']/den,3)],
                refills=tot["refill"], segments=tot["seg"], boundary_segments=tot["bnd"], steps=tot["steps"],
                eff_ep_len_steps_per_refill=round(tot["steps"] / max(tot["refill"], 1), 1),
                seg_len_p50=int(np.percentile(seg_lens, 50)), seg_len_p90=int(np.percentile(seg_lens, 90)), seg_len_max=int(seg_lens.max()),
                continuing_episodes_per_hr=round(tot["refill"] / (cont_t / 3600), 0), continuing_sec=round(cont_t, 2),
                legacy_episodes_per_hr=round(leg_eps / (leg_t / 3600), 0), legacy_sec=round(leg_t, 2),
                continuing_steps_per_hr=round(tot["steps"] / (cont_t / 3600), 0), legacy_steps_per_hr=round(B * horizon * NR / (leg_t / 3600), 0))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    R = roll(dev)
    states, Vh, E, outcome, Sh, ms = R["states"], R["vhat"], R["empty"], R["outcome"], R["signed_h"], R["ms"]
    N = len(outcome); th0 = np.abs(wrap(states[:, 0, 2]))
    coll = np.where(outcome == 1)[0]

    # ---- Data provenance: verify collision set vs eval_episodes ----
    eval_ids = set(eval_collision_ids())
    roll_ids = set(int(i) for i in coll)
    match = (eval_ids == roll_ids)
    only_roll = sorted(roll_ids - eval_ids); only_eval = sorted(eval_ids - roll_ids)
    agreed = sorted(roll_ids & eval_ids)
    print(f"[provenance] eval_coll={len(eval_ids)} roll_coll={len(roll_ids)} EXACT_MATCH={match} "
          f"agreed={len(agreed)} only_roll={only_roll} only_eval={only_eval}", flush=True)

    # analysis set = the AGREED collisions (both paths call collision) — robust; boundary IDs reported, stopped
    aset = np.array(agreed)
    tilt = aset[th0[aset] >= TILT]; ntilt = aset[th0[aset] < TILT]
    print(f"[sets] agreed collisions={len(aset)} tilted(|th0|>pi/2)={len(tilt)} non-tilted={len(ntilt)}", flush=True)

    # ---- D-i: V_hat descent anatomy ----
    def d_i(idxs):
        classes = {"a_crossed_into_safe": [], "b_descended_never_safe": [], "c_never_descended": [], "other": []}
        for gi in idxs:
            imp = impact_step(Sh[gi], ms)
            cls = classify_vhat(Vh[gi], imp)
            classes[cls].append(int(gi))
        return classes

    ci_t = d_i(tilt); ci_n = d_i(ntilt)
    def infrac(idxs, at_impact=False):
        vals = []
        for gi in idxs:
            imp = impact_step(Sh[gi], ms)
            if at_impact:
                vals.append(float(E[gi, min(imp, ms - 1)]))
            else:
                vals.append(float(E[gi, :max(1, imp)].mean()))
        return round(float(np.mean(vals)), 4) if vals else None

    d_i_out = {"tilted": {k: len(v) for k, v in ci_t.items()}, "non_tilted": {k: len(v) for k, v in ci_n.items()},
               "infeas_frac_along_traj": {"tilted": infrac(tilt), "non_tilted": infrac(ntilt)},
               "infeas_frac_at_impact": {"tilted": infrac(tilt, True), "non_tilted": infrac(ntilt, True)}}
    print("[D-i]", json.dumps(d_i_out), flush=True)

    # 3 representative V_hat-vs-t traces per class (tilted), PNG
    for cls, ids in ci_t.items():
        if not ids:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for gi in ids[:3]:
            imp = impact_step(Sh[gi], ms)
            ax.plot(np.arange(imp + 1) * R["dt"], Vh[gi, :imp + 1] if imp < Vh.shape[1] else Vh[gi],
                    label=f"ep {gi}")
        ax.axhline(0, color="k", lw=0.5, ls="--"); ax.set_xlabel("t (s)"); ax.set_ylabel("V_hat")
        ax.set_title(f"tilted {cls} (n={len(ids)})"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGDIR / f"vhat_{cls}.png", dpi=90); plt.close(fig)

    # ---- D-ii: recovery duration vs BPTT window (high-tilt SUCCESSFUL reaches) ----
    goal = np.where(outcome == 2)[0]; hi_goal = goal[th0[goal] >= TILT]
    tau_rec = []; reach_no_right = 0
    for gi in hi_goal:
        thr = np.abs(wrap(states[gi, :, 2]))
        upright = thr < (np.pi / 6)
        # first t where upright holds for 5 consecutive steps
        rec = None
        run = 0
        for t in range(len(upright)):
            run = run + 1 if upright[t] else 0
            if run >= 5:
                rec = t - 4; break
        if rec is None:
            reach_no_right += 1
        else:
            tau_rec.append(rec)
    tau_rec = np.array(tau_rec)
    d_ii_out = {"n_hi_tilt_reach": int(len(hi_goal)), "reach_without_righting": int(reach_no_right),
                "n_righted": int(len(tau_rec)),
                "tau_rec_steps": {"p50": float(np.percentile(tau_rec, 50)) if tau_rec.size else None,
                                  "p90": float(np.percentile(tau_rec, 90)) if tau_rec.size else None,
                                  "max": int(tau_rec.max()) if tau_rec.size else None},
                "bptt_T": R["bptt_T"]}
    print("[D-ii]", json.dumps(d_ii_out), flush=True)

    # ---- D-iii: false-doom overlap ----
    inf0 = E[:, 0]                                                        # HardNet infeasible at t=0
    tilt_inf0 = int(inf0[tilt].sum()); all_inf0 = int(inf0[aset].sum())
    # cross-tab D-i class x infeasible@t0 for tilted
    xtab = {}
    for cls, ids in ci_t.items():
        if not ids:
            continue
        ids = np.array(ids)
        xtab[cls] = {"n": len(ids), "infeasible@t0": int(inf0[ids].sum()), "feasible@t0": int((~inf0[ids]).sum())}
    d_iii_out = {"tilted_infeasible@t0": tilt_inf0, "tilted_n": len(tilt),
                 "tilted_infeasible@t0_frac": round(tilt_inf0 / max(1, len(tilt)), 4),
                 "agreed_infeasible@t0": all_inf0, "agreed_n": len(aset),
                 "crosstab_class_x_inf0_tilted": xtab}
    print("[D-iii]", json.dumps(d_iii_out), flush=True)

    # baseline comparison (iteration-1 obs-fix run): monotone-descent 62/67 tilted; tilted strata 76/94;
    # late-path V_hat rise presence = class (a) count (crossed-into-safe → rose).
    n_coll_roll = int((outcome == 1).sum())
    n_tilt_roll = int((np.abs(wrap(states[coll, 0, 2])) >= TILT).sum())
    baseline = dict(monotone_descent_tilted=f"{len(ci_t['other'])}/{len(tilt)} (iter1 62/67)",
                    tilted_band_strata=f"{n_tilt_roll}/{n_coll_roll} residual (iter1 76/94, agreed-set-based)",
                    late_path_vhat_rise_class_a=f"tilted a={len(ci_t['a_crossed_into_safe'])} non-tilt a={len(ci_n['a_crossed_into_safe'])} (iter1 0)",
                    Diii_tilted_inf0_frac=f"{d_iii_out['tilted_infeasible@t0_frac']} (iter1 0.866)")
    print("[BASELINE-CMP]", json.dumps(baseline), flush=True)
    out = dict(run=RID, provenance=dict(eval_coll=len(eval_ids), roll_coll=len(roll_ids), exact_match=match,
                                        agreed=len(agreed), only_roll=only_roll, only_eval=only_eval),
               n_coll_roll=n_coll_roll, n_tilt_coll_roll=n_tilt_roll,
               n_tilted=int(len(tilt)), n_non_tilted=int(len(ntilt)),
               D_i=d_i_out, D_ii=d_ii_out, D_iii=d_iii_out, baseline_cmp=baseline,
               classes_tilted_ids=ci_t)
    # ================= iter-5 deferred + authority probes (P-A4, P-B, P-C, P-D) =================
    d2 = probe_d2(R)
    print("[P-A4 D2]", json.dumps(d2), flush=True)
    pb_recs, pb = probe_pb(R, aset, th0, TILT)
    print("[P-B authority 2x2 ALL]", json.dumps(pb["table_all"]), flush=True)
    print("[P-B authority 2x2 TILTED]", json.dumps(pb["table_tilted"]), flush=True)
    print("[P-B mc-vs-empty@t0 agreement]", pb["mc_vs_empty_t0_agreement"], flush=True)
    pbp = probe_pbprime(R, pb_recs, TILT)
    print("[P-B' k-step k5]", json.dumps(pbp["k5"]["all"]), "frac_improvable", pbp["k5"]["frac_kstep_improvable"], flush=True)
    print("[P-B' k-step k10]", json.dumps(pbp["k10"]["all"]), "frac_improvable", pbp["k10"]["frac_kstep_improvable"], flush=True)
    pc_recs, pc = probe_pc(R, aset, th0, TILT)
    print("[P-C entry anatomy]", json.dumps(pc["counts"]), json.dumps({k: pc[k] for k in ("born_inside", "fast_approach", "other", "pool_median_speed")}), flush=True)
    pd = probe_pd(R)
    print("[P-D offline collection]", json.dumps(pd), flush=True)
    print(INTERP_MAP, flush=True)

    # per-state CSV (P-B + P-C) under the run's eval dir (figures/residual_mechanism)
    csv_path = FIGDIR / f"authority_states_{RID}.csv"
    keys = ["episode_idx", "state", "t_step", "theta0", "tilted", "speed", "V_hat", "lg_norm", "m_c", "m_d",
            "cont_infeasible", "disc_improvable", "empty_flag"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in pb_recs:
            w.writerow(r)
    pc_csv = FIGDIR / f"entry_anatomy_{RID}.csv"
    pck = ["episode_idx", "t0_infeasible", "tfi", "t_to_infeas_s", "speed_entry", "h_entry", "approach_speed", "tilted", "entry_class"]
    with open(pc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pck); w.writeheader()
        for r in pc_recs:
            w.writerow(r)

    out["P_A4_d2"] = d2
    out["P_B_authority"] = pb
    out["P_Bprime_kstep"] = pbp
    out["P_C_entry"] = pc
    out["P_D_offline"] = pd
    out["interp_map"] = INTERP_MAP
    json.dump(out, open(SP / f"residual_mechanism_diag_{RID}.json", "w"), indent=2)
    print("WROTE", SP / f"residual_mechanism_diag_{RID}.json", "| figs", FIGDIR, "| csv", csv_path, pc_csv, flush=True)


if __name__ == "__main__":
    main()
