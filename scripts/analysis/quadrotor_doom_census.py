"""v2.6.2 §doom census — A2 (ballistic closed form) flag-rate census on the RELAXED 3g system.

Read-only / eval-only / NO pool change / NO training. Re-rolls the brake-envelope best.pt ONCE to classify
born-doomed (collision & (infeasible@0 OR brake-envelope-deficit>0@0)) and reach-outcome ICs, saves the
indices + per-scene ICs, then applies the sound ballistic doom certificate (src.common.quadrotor_ballistic_
doom) to three sets: (i) all 2000 committed-pool ICs, (ii) the D1 born-doomed 72 (context only), (iii) the
~1815 reach-outcome ICs. Gate: reach-outcome flag rate MUST be 0 (a reach outcome is a physical avoidance
witness; any flag = a derivation bug -> STOP, no silent fix). Reports the three rates + flagged-IC strata.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_ballistic_doom import accel_bound, is_doomed_ballistic, radius_bucket_down, min_approach
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/runs/v2.6.2/set__20260716-182949__seed42/v2.6.2__20260716-182949__seed42"
TAU_BRAKE = 0.6


def scene_ic(sc):
    return (np.asarray(sc.start, np.float64), np.asarray(sc.initial_velocity, np.float64),
            float(sc.initial_attitude or 0.0), float(sc.initial_omega or 0.0),
            np.asarray(sc.obstacle_centers, np.float64), np.asarray(sc.obstacle_radii, np.float64),
            np.asarray(sc.obstacle_active, bool))


def reroll_indices():
    """Return (scenes, born_idx, reach_idx) re-rolling brake best.pt (cached to scratchpad npz)."""
    cache = SP / "doom_census_rollout.npz"
    scenes = load_pool(POOL).scenes
    if cache.exists():
        z = np.load(cache)
        print(f"[cache] born={len(z['born'])} reach={len(z['reach'])}", flush=True)
        return scenes, list(z["born"]), list(z["reach"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    ms = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"]); N = len(scenes)
    born, reach = [], []
    for s0 in range(0, N, 250):
        bs = batch_scenes(scenes[s0:s0 + 250], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        oc, orad, oact = scene_obstacle_tensors(bs, dev, torch.float32)
        p0 = x[:, :2]; v0 = x[:, 3:5]; rel0 = p0.unsqueeze(1) - oc; d0c = torch.linalg.norm(rel0, dim=2)
        nrm0 = rel0 / d0c.unsqueeze(2).clamp_min(1e-9)
        inw0 = torch.relu(-torch.sum(v0.unsqueeze(1) * nrm0, dim=2))
        deficit0 = (torch.relu(inw0 * TAU_BRAKE - (d0c - orad)) * oact.bool().float()).max(1).values
        empt0 = torch.zeros(B, dtype=torch.bool, device=dev)
        states = [x.clone()]
        with torch.no_grad():
            for t in range(ms):
                un = policy(system.observation(x, bs))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                alpha = _base_alpha(h.detach(), params); row = -lf.detach() - alpha * h.detach()
                proj = _base_projection(un, lg.detach(), row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg.detach(), row, bounds)
                if t == 0:
                    empt0 = empty
                x = rk4_step(system, x, u, dt); states.append(x.clone())
        S = torch.stack(states, 0); res = resolve_outcome(step_outcomes(S, bs, system, cfg))
        for i in range(B):
            gi = s0 + i
            if res.outcome[i] == "collision" and (bool(empt0[i]) or float(deficit0[i]) > 0):
                born.append(gi)
            elif res.outcome[i] == "goal":
                reach.append(gi)
        print(f"  batch {s0}: born={len(born)} reach={len(reach)}", flush=True)
    np.savez(cache, born=np.array(born), reach=np.array(reach))
    return scenes, born, reach


def strata(scenes, idxs, A):
    """For flagged ICs: |theta0|, inward-v0 (max over obstacles), min surface distance d_k."""
    rows = []
    for gi in idxs:
        p0, v0, th, om, c, r, act = scene_ic(scenes[gi])
        rel = p0[None, :] - c; dist = np.linalg.norm(rel, axis=1)
        n = rel / np.clip(dist, 1e-9, None)[:, None]
        inw = np.maximum(0.0, -(v0[None, :] * n).sum(1)) * act
        dk = (dist - r)
        rows.append(dict(gi=int(gi), abs_theta0=abs(th), inward_v0=float(inw.max()),
                         min_dk=float(dk[act].min()) if act.any() else float("nan")))
    return rows


def main():
    cfg = json.loads((RUN / "config.json").read_text()) if (RUN / "config.json").exists() else None
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    A = accel_bound(cfg)
    print(f"relaxed accel bound A = {A:.4f} (= {A/9.81:.3f} g)", flush=True)

    scenes, born, reach = reroll_indices()
    N = len(scenes)

    def flags(idxs):
        return [gi for gi in idxs if is_doomed_ballistic(*scene_ic(scenes[gi])[:2],
                *scene_ic(scenes[gi])[4:7], A=A)]

    all_idx = list(range(N))
    f_all = flags(all_idx)
    f_born = flags(born)
    f_reach = flags(reach)
    print(f"\nA2 BALLISTIC flag rates (relaxed 3g, radius rounded DOWN):", flush=True)
    print(f"  (i)   all pool ICs : {len(f_all)}/{N} = {len(f_all)/N:.4f}", flush=True)
    print(f"  (ii)  born-doomed  : {len(f_born)}/{len(born)} = {len(f_born)/max(1,len(born)):.4f} (context only)", flush=True)
    print(f"  (iii) reach ICs    : {len(f_reach)}/{len(reach)} = {len(f_reach)/max(1,len(reach)):.4f}  [GATE: must be 0]", flush=True)

    gate_ok = (len(f_reach) == 0)
    strat_all = strata(scenes, f_all, A)
    out = dict(A=A, n_pool=N, n_born=len(born), n_reach=len(reach),
               flag_all=len(f_all), flag_born=len(f_born), flag_reach=len(f_reach),
               reach_gate_pass=gate_ok, flagged_all_idx=[int(i) for i in f_all],
               strata_flagged_all=strat_all)
    json.dump(out, open(SP / "quadrotor_doom_census_A.json", "w"), indent=2)

    if not gate_ok:
        print("\n*** GATE FAIL: a reach-outcome IC is flagged -> derivation bug. STOP (no silent fix). ***", flush=True)
        for gi in f_reach[:10]:
            p0, v0, th, om, c, r, act = scene_ic(scenes[gi])
            print(f"    reach-flag gi={gi} p0={p0} v0={v0} theta0={th:.3f}", flush=True)
    else:
        print("\nGATE PASS: zero reach-outcome ICs flagged.", flush=True)
        if f_all:
            print("flagged-IC strata (all-pool):", flush=True)
            print(f"  |theta0|  : {np.percentile([r['abs_theta0'] for r in strat_all],[10,50,90])}", flush=True)
            print(f"  inward_v0 : {np.percentile([r['inward_v0'] for r in strat_all],[10,50,90])}", flush=True)
            print(f"  min_dk    : {np.percentile([r['min_dk'] for r in strat_all],[10,50,90])}", flush=True)
        else:
            print("NO ICs flagged in the whole pool -> provable-doom set (ballistic) is EMPTY.", flush=True)
    print("WROTE", SP / "quadrotor_doom_census_A.json", flush=True)


if __name__ == "__main__":
    main()
