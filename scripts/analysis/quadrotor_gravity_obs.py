"""v2.6.2 §gravity-observability — is the tilted-band failure an aliasing artifact of the gravity-blind obs?

Hypothesis: obs rotates everything into the body frame and DROPS theta -> it is SO(2)-invariant, which is
symmetry-correct only if the dynamics are SO(2)-equivariant. Gravity (fixed world (0,-g)) breaks that, so a
memoryless obs-conditioned pi / V_hat cannot tell upright from inverted and cannot reliably pick the recovery
rotation. Read-only; eval re-rolls the brake-envelope best.pt on the full pool; NO training/config/git.

D1 thrust-misfire discrimination · D2 theta decodability probe · D3 aliasing + dynamics divergence ·
D4 value-inheritance profile at t=0. Facts only.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/v2.6.2__20260716-182949__seed42"
NMAX = 12
BANDS = [(0.0, np.pi / 6, "[0,pi/6)"), (np.pi / 6, np.pi / 2, "[pi/6,pi/2)"), (np.pi / 2, np.pi + 1e-6, "[pi/2,pi]")]


def band_of(abs_theta):
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= abs_theta < hi:
            return i
    return 2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_build(dev):
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    return cfg, system, vnet, policy, h_fn, params, bounds


def geometry(scenes):
    N = len(scenes)
    C = np.zeros((N, NMAX, 2)); R = np.zeros((N, NMAX)); AC = np.zeros((N, NMAX), bool); G = np.zeros((N, 2))
    for i, sc in enumerate(scenes):
        C[i] = np.asarray(sc.obstacle_centers, np.float64)
        R[i] = np.asarray(sc.obstacle_radii, np.float64)
        AC[i] = np.asarray(sc.obstacle_active, bool)
        G[i] = np.asarray(sc.goal, np.float64)
    return C, R, AC, G


def roll(dev, scenes, system, policy, h_fn, params, bounds):
    """Instrumented full-pool roll. Cache states/actions/empty/outcome to npz."""
    cache = SP / "gravobs_roll.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"[cache] states{z['states'].shape} outcome n_coll={int((z['outcome']==1).sum())}", flush=True)
        return z["states"], z["actions"], z["empty"], z["outcome"]
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]; ms = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"]); N = len(scenes)
    all_S = np.zeros((N, ms + 1, 6), np.float32); all_U = np.zeros((N, ms, 2), np.float32)
    all_E = np.zeros((N, ms), bool); all_O = np.zeros(N, np.int8)          # 0 other,1 coll,2 goal
    for s0 in range(0, N, 250):
        bs = batch_scenes(scenes[s0:s0 + 250], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        states = [x.clone()]; acts = []; emps = []
        with torch.no_grad():
            for t in range(ms):
                un = policy(system.observation(x, bs))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                alpha = _base_alpha(h.detach(), params); row = -lf.detach() - alpha * h.detach()
                proj = _base_projection(un, lg.detach(), row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg.detach(), row, bounds)
                acts.append(u.clone()); emps.append(empty.clone())
                x = rk4_step(system, x, u, dt); states.append(x.clone())
        S = torch.stack(states, 0); res = resolve_outcome(step_outcomes(S, bs, system, cfg))
        all_S[s0:s0 + B] = S.permute(1, 0, 2).cpu().numpy()
        all_U[s0:s0 + B] = torch.stack(acts, 0).permute(1, 0, 2).cpu().numpy()
        all_E[s0:s0 + B] = torch.stack(emps, 0).permute(1, 0).cpu().numpy()
        for i in range(B):
            all_O[s0 + i] = 1 if res.outcome[i] == "collision" else (2 if res.outcome[i] == "goal" else 0)
        print(f"  roll batch {s0}: coll={int((all_O[:s0+B]==1).sum())} goal={int((all_O[:s0+B]==2).sum())}", flush=True)
    np.savez(cache, states=all_S, actions=all_U, empty=all_E, outcome=all_O)
    return all_S, all_U, all_E, all_O


def min_surf(p, C, R, AC):
    """(min surface distance, argmin obstacle index) over active obstacles for one position p."""
    d = np.linalg.norm(C - p[None, :], axis=1) - R
    d = np.where(AC, d, np.inf)
    k = int(np.argmin(d))
    return float(d[k]), k


def collision_step(Straj, C, R, AC, ms):
    for t in range(Straj.shape[0]):
        dmin, _ = min_surf(Straj[t, :2], C, R, AC)
        if dmin < 0:
            return t
    return -1


def pct(a, ps=(10, 50, 90)):
    a = np.asarray(a, float)
    return [round(float(np.percentile(a, p)), 4) for p in ps] if a.size else [None] * len(ps)


# ============================== D1 — thrust-misfire discrimination ==============================
def d1_misfire(states, actions, outcome, C, R, AC, cfg):
    m = float(cfg["env"]["quadrotor_planar"]["mass"]); f_max = float(cfg["env"]["bounds"]["quadrotor_planar"]["f_max"])
    ms = actions.shape[1]
    coll = np.where(outcome == 1)[0]; goal = np.where(outcome == 2)[0]
    # collision misfire steps, per band
    band_coll = [[], [], []]           # misfire flags per step in each band (collisions, last 10 pre-collision)
    per_coll_frac = []
    for gi in coll:
        tc = collision_step(states[gi], C[gi], R[gi], AC[gi], ms)
        if tc <= 0:
            continue
        lo = max(0, tc - 10); flags = []
        for t in range(lo, tc):
            x = states[gi, t]; f = float(actions[gi, t, 0]); th = float(x[2])
            _, k = min_surf(x[:2], C[gi], R[gi], AC[gi])
            n = (x[:2] - C[gi, k]); n = n / (np.linalg.norm(n) + 1e-9)
            re = np.array([-np.sin(th), np.cos(th)])
            toward = float(re @ (-n))                       # thrust axis component toward obstacle
            mis = (f > 0.25 * f_max) and (toward > 0.5)
            band_coll[band_of(abs(wrap(th)))].append(1.0 if mis else 0.0)
            flags.append(1.0 if mis else 0.0)
        if flags:
            per_coll_frac.append(np.mean(flags))
    # matched near-obstacle passages in reach episodes (min surf < 0.5), per band
    band_pass = [[], [], []]
    rng = np.random.default_rng(0)
    for gi in goal[rng.permutation(len(goal))[:400]]:       # sample 400 reach episodes
        for t in range(0, ms, 2):
            x = states[gi, t]; d, k = min_surf(x[:2], C[gi], R[gi], AC[gi])
            if d >= 0.5 or d < 0:
                continue
            f = float(actions[gi, t, 0]); th = float(x[2])
            n = (x[:2] - C[gi, k]); n = n / (np.linalg.norm(n) + 1e-9)
            re = np.array([-np.sin(th), np.cos(th)]); toward = float(re @ (-n))
            mis = (f > 0.25 * f_max) and (toward > 0.5)
            band_pass[band_of(abs(wrap(th)))].append(1.0 if mis else 0.0)
    out = dict(n_coll=len(coll), per_coll_misfire_frac=pct(per_coll_frac),
               coll_misfire_by_band={BANDS[i][2]: (round(float(np.mean(band_coll[i])), 4) if band_coll[i] else None,
                                                    len(band_coll[i])) for i in range(3)},
               pass_misfire_by_band={BANDS[i][2]: (round(float(np.mean(band_pass[i])), 4) if band_pass[i] else None,
                                                   len(band_pass[i])) for i in range(3)})
    return out


# ============================== D2 — theta decodability probe ==============================
def d2_theta_probe(dev, scenes, system, states, cfg):
    dt = float(cfg["env"]["dt"]); N = len(scenes)
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    ts = list(range(0, states.shape[1] - 1, 5))            # subsample timesteps
    obs_list, th_list, v_list, ep_list, t_list = [], [], [], [], []
    with torch.no_grad():
        for t in ts:
            x = torch.tensor(states[:, t, :], device=dev, dtype=torch.float32)
            o = system.observation(x, bs).cpu().numpy()
            obs_list.append(o); th_list.append(states[:, t, 2])
            v_list.append(np.linalg.norm(states[:, t, 3:5], axis=1))
            ep_list.append(np.arange(N)); t_list.append(np.full(N, t * dt))
    O = np.concatenate(obs_list); TH = np.concatenate(th_list)
    V = np.concatenate(v_list); EP = np.concatenate(ep_list); TT = np.concatenate(t_list)
    Y = np.stack([np.sin(TH), np.cos(TH)], 1)
    rng = np.random.default_rng(1); te_ep = set(rng.permutation(N)[:N // 5])
    te = np.array([e in te_ep for e in EP]); tr = ~te
    # ridge
    Xtr, Xte = O[tr], O[te]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd
    A = Xtr_s.T @ Xtr_s + 1.0 * np.eye(Xtr_s.shape[1])
    W = np.linalg.solve(A, Xtr_s.T @ Y[tr])
    pred_r = Xte_s @ W
    # MLP
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(O.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.tensor(Xtr_s, device=dev, dtype=torch.float32); Yt = torch.tensor(Y[tr], device=dev, dtype=torch.float32)
    for _ in range(300):
        idx = torch.randint(0, Xt.shape[0], (4096,), device=dev)
        opt.zero_grad(); loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_m = net(torch.tensor(Xte_s, device=dev, dtype=torch.float32)).cpu().numpy()

    def ang_err(pred):
        p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
        cos = np.clip((p * Y[te]).sum(1), -1, 1)
        return np.degrees(np.arccos(cos))
    er_r, er_m = ang_err(pred_r), ang_err(pred_m)
    absth = np.abs(wrap(TH[te])); vte = V[te]; tte = TT[te]
    vq = np.quantile(vte, [1 / 3, 2 / 3])
    out = dict(overall_deg=dict(ridge=round(float(er_r.mean()), 2), mlp=round(float(er_m.mean()), 2)),
               by_theta_band={}, by_v_tercile={}, early_vs_late={})
    for i, (lo, hi, name) in enumerate(BANDS):
        msk = (absth >= lo) & (absth < hi)
        out["by_theta_band"][name] = dict(mlp_deg=round(float(er_m[msk].mean()), 2) if msk.any() else None, n=int(msk.sum()))
    for lbl, msk in [("v_low", vte < vq[0]), ("v_mid", (vte >= vq[0]) & (vte < vq[1])), ("v_high", vte >= vq[1])]:
        out["by_v_tercile"][lbl] = round(float(er_m[msk].mean()), 2) if msk.any() else None
    for lbl, msk in [("t<0.5s", tte < 0.5), ("t>=0.5s", tte >= 0.5)]:
        out["early_vs_late"][lbl] = round(float(er_m[msk].mean()), 2) if msk.any() else None
    # high-tilt AND early cell (the predicted failure cell)
    cell = (absth >= np.pi / 2) & (tte < 0.5)
    out["failure_cell_hitilt_early_deg"] = round(float(er_m[cell].mean()), 2) if cell.any() else None
    out["failure_cell_n"] = int(cell.sum())
    return out


# ============================== D3 — aliasing existence + dynamics divergence ==============================
def _rot(delta):
    c, s = np.cos(delta), np.sin(delta)
    return np.array([[c, -s], [s, c]])


def d3_aliasing(dev, scenes, system, policy, h_fn, states, outcome, C, R, AC, cfg):
    import dataclasses
    dt = float(cfg["env"]["dt"]); ms = states.shape[1] - 1
    coll = np.where(outcome == 1)[0]; goal = np.where(outcome == 2)[0]
    rng = np.random.default_rng(2)
    # collision-precursor states (last 3 steps before collision) and benign near-obstacle (reach, surf<0.6)
    prec, beni = [], []
    for gi in coll:
        tc = collision_step(states[gi], C[gi], R[gi], AC[gi], ms)
        if tc <= 0:
            continue
        for t in range(max(0, tc - 3), tc):
            prec.append((gi, t))
    for gi in goal[rng.permutation(len(goal))[:300]]:
        for t in range(0, ms, 3):
            d, _ = min_surf(states[gi, t, :2], C[gi], R[gi], AC[gi])
            if 0 <= d < 0.6:
                beni.append((gi, t))
    rng.shuffle(prec); rng.shuffle(beni)
    prec = prec[:600]; beni = beni[:600]
    deltas = [np.pi, np.pi / 2, -np.pi / 2, 2 * np.pi / 3, -2 * np.pi / 3]

    def divergence(pairs):
        obs_gap, pi_gap, h0_gap, dh1, dcl1 = [], [], [], [], []
        for (gi, t) in pairs:
            sc = scenes[gi]; x = states[gi, t].astype(np.float64); p = x[:2]
            delta = float(rng.choice(deltas))
            Rd = _rot(delta)
            c_new = np.asarray(sc.obstacle_centers, np.float64).copy()
            for k in range(NMAX):
                if AC[gi, k]:
                    c_new[k] = p + Rd @ (c_new[k] - p)
            g_new = p + Rd @ (np.asarray(sc.goal, np.float64) - p)
            v_new = Rd @ x[3:5]
            sc2 = dataclasses.replace(sc, obstacle_centers=c_new, goal=g_new,
                                      initial_attitude=float(wrap(x[2] + delta)),
                                      initial_velocity=v_new)
            x2 = x.copy(); x2[2] = wrap(x[2] + delta); x2[3:5] = v_new
            bs = batch_scenes([sc, sc2], device=dev, dtype=torch.float32)
            X = torch.tensor(np.stack([x, x2]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                o = system.observation(X, bs); u = policy(o)
                h0 = h_fn(X, bs).reshape(-1)
                Xn = rk4_step(system, X, u, dt)
                hn = h_fn(Xn, bs).reshape(-1)
            o = o.cpu().numpy(); u = u.cpu().numpy(); Xn = Xn.cpu().numpy(); hn = hn.cpu().numpy(); h0 = h0.cpu().numpy()
            obs_gap.append(float(np.max(np.abs(o[0] - o[1]))))
            pi_gap.append(float(np.max(np.abs(u[0] - u[1]))))
            h0_gap.append(float(abs(h0[0] - h0[1])))
            dh1.append(float(abs(hn[0] - hn[1])))
            # closure toward binding obstacle in each member's own scene, at next state
            def closure(xn, cc, rr, ac):
                d, k = min_surf(xn[:2], cc, rr, ac)
                n = xn[:2] - cc[k]; n = n / (np.linalg.norm(n) + 1e-9)
                return float(-(xn[3:5] @ n))
            cl0 = closure(Xn[0], C[gi], R[gi], AC[gi])
            cl1 = closure(Xn[1], c_new, R[gi], AC[gi])
            dcl1.append(abs(cl0 - cl1))
        return dict(obs_gap_max=round(max(obs_gap), 6), pi_gap_max=round(max(pi_gap), 6),
                    h0_gap_max=round(max(h0_gap), 6),
                    dh1=pct(dh1), dclosure1=pct(dcl1), n=len(pairs))
    return dict(precursor=divergence(prec), benign=divergence(beni))


# ============================== D4 — value-inheritance profile at t=0 ==============================
def d4_t0_profile(dev, scenes, system, h_fn, states, empty, outcome):
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x0 = torch.tensor(states[:, 0, :], device=dev, dtype=torch.float32)
    with torch.no_grad():
        h0 = h_fn(x0, bs).reshape(-1).cpu().numpy()
    infeas0 = empty[:, 0]                                   # HardNet empty (infeasible) at t=0
    th0 = np.abs(wrap(states[:, 0, 2]))
    prof = {}
    for i, (lo, hi, name) in enumerate(BANDS):
        band = (th0 >= lo) & (th0 < hi)
        reach = band & (outcome == 2); coll = band & (outcome == 1)
        prof[name] = dict(
            n=int(band.sum()),
            infeasible0_frac=round(float(infeas0[band].mean()), 4) if band.any() else None,
            Vhat0_reach=round(float(h0[reach].mean()), 4) if reach.any() else None,
            Vhat0_coll=round(float(h0[coll].mean()), 4) if coll.any() else None,
            reach_rate=round(float(reach.sum() / band.sum()), 4) if band.any() else None,
            # the signature: START infeasible yet eventually REACH
            infeasible0_and_reach=int((band & infeas0 & (outcome == 2)).sum()),
            infeasible0_count=int((band & infeas0).sum()),
        )
        n_if = prof[name]["infeasible0_count"]
        prof[name]["frac_infeasible0_that_reach"] = round(prof[name]["infeasible0_and_reach"] / n_if, 4) if n_if else None
    return prof


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, system, vnet, policy, h_fn, params, bounds = load_build(dev)
    scenes = load_pool(POOL).scenes
    C, R, AC, G = geometry(scenes)
    states, actions, empty, outcome = roll(dev, scenes, system, policy, h_fn, params, bounds)
    print(f"outcome: coll={int((outcome==1).sum())} goal={int((outcome==2).sum())} other={int((outcome==0).sum())}", flush=True)

    d1 = d1_misfire(states, actions, outcome, C, R, AC, cfg)
    print("\n[D1] misfire:", json.dumps(d1), flush=True)
    d2 = d2_theta_probe(dev, scenes, system, states, cfg)
    print("\n[D2] theta-probe:", json.dumps(d2), flush=True)
    d3 = d3_aliasing(dev, scenes, system, policy, h_fn, states, outcome, C, R, AC, cfg)
    print("\n[D3] aliasing:", json.dumps(d3), flush=True)
    d4 = d4_t0_profile(dev, scenes, system, h_fn, states, empty, outcome)
    print("\n[D4] t0-profile:", json.dumps(d4), flush=True)

    json.dump(dict(D1=d1, D2=d2, D3=d3, D4=d4), open(SP / "quadrotor_gravity_obs.json", "w"), indent=2)
    print("\nWROTE", SP / "quadrotor_gravity_obs.json", flush=True)


if __name__ == "__main__":
    main()
