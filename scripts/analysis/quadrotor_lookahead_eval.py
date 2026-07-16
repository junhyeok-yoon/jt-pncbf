"""v2.6.1 EVAL-ONLY — predictive-filter (lookahead) sweep on the seed-42 M6 best.pt. NO retraining, NO
config commit: the lookahead is a DEPLOY-TIME change applied via a runtime params override. Value + policy
FROZEN. Replicates the dual_arm filtered rollout (m3_eval path, reproduces eval_metrics) and adds the
VERBATIM committed lookahead alpha-inflation (filter_hardnet.__call__:92-93 + _lookahead_peak_h), driven by
the deployed best.pt policy as policy_fn. Reports per cell: collision/timeout/cps_v2, empty-infeasibility
(overall + by obstacle count), mean ||u_safe-u_nom||. Baseline (lookahead OFF) anchors the comparison.
"""
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _lookahead_peak_h, _SINGULAR_LG_THRESHOLD)
from src.common.observation import scene_goal_tensor
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import cps_v2
from src.eval.evaluate import first_physical_event_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
GRID_N = [3, 5, 8]
GRID_BETA = [0.5, 1.0, 2.0]
DELTA = 0.1                          # filter.lookahead.delta config/default
DENS_BINS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 8, "6-8"), (9, 99, "9+")]


def roll(system, h_fn, policy, cfg, scenes, dev, params, la_on, la_params=None, chunk=250):
    """dual_arm-style filtered roll of the deployed policy; optional lookahead alpha-inflation (verbatim
    __call__:92-93 using _lookahead_peak_h with the deployed policy as policy_fn). Returns per-episode
    (outcome, empty_rate, dun_mean, n_obstacles)."""
    bounds = system.u_bounds; max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    pol = lambda x, sc: policy(system.observation(x, sc))
    ep = []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]
        n_obs = bs.obstacle_active.to(torch.int64).sum(dim=1).cpu().numpy()
        empt = torch.zeros(max_steps, B, dtype=torch.bool, device=dev)
        dun = torch.zeros(max_steps, B, device=dev)
        states = [x]
        with torch.no_grad():
            for t in range(max_steps):
                un = pol(x, bs)
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params)
                if la_on:
                    h_peak = _lookahead_peak_h(system=system, h_fn=h_fn, policy_fn=pol, x=x, scene=bs,
                                               params=la_params)
                    gap = torch.relu((h_peak - h) / la_params.lookahead_delta)
                    alpha = alpha * (1.0 + la_params.lookahead_beta * gap)
                row = -lf - alpha * h
                proj = _base_projection(un, lg, row, bounds, params)
                u, empty = _box_aware_projection(un, proj, lg, row, bounds)
                empt[t] = empty
                dun[t] = torch.linalg.norm(u - un, dim=1)
                x = rk4_step(system, x, u, dt); states.append(x)
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        an = first_physical_event_step(masks)
        an = torch.where(an >= 0, an, torch.full_like(an, max_steps))
        act = torch.arange(max_steps, device=dev).unsqueeze(1) < an.unsqueeze(0)
        for i in range(B):
            am = act[:, i]; na = int(am.sum())
            er = float((empt[:, i] & am).sum()) / na if na else 0.0
            dm = float(dun[:, i][am].mean()) if na else 0.0
            ep.append((res.outcome[i], er, dm, int(n_obs[i])))
    return ep


def summarize(ep):
    o = np.array([e[0] for e in ep]); em = np.array([e[1] for e in ep])
    dm = np.array([e[2] for e in ep]); nb = np.array([e[3] for e in ep])
    rate = lambda k: float((o == k).mean())
    out = dict(n=len(ep), reach=rate("goal"), collision=rate("collision"), oob=rate("oob"),
               stuck=rate("stuck"), timeout=rate("timeout"), cps_v2=cps_v2(o, em),
               empty_inf=float(em.mean()), dun_mean=float(dm.mean()))
    by = {}
    for lo, hi, lab in DENS_BINS:
        m = (nb >= lo) & (nb <= hi); n = int(m.sum())
        by[lab] = dict(n=n, empty_inf=float(em[m].mean()) if n else 0.0,
                       collision=float((o[m] == "collision").mean()) if n else 0.0)
    out["by_density"] = by
    return out


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
    scenes = load_pool(POOL).scenes
    base_params = _hardnet_params(cfg)

    # confirm the deploy config values + policy_fn signature compatibility
    la_cfg = cfg["filter"].get("lookahead", {})
    print(f"lookahead config: enabled={la_cfg.get('enabled')} N={la_cfg.get('N')} beta={la_cfg.get('beta')} "
          f"delta={la_cfg.get('delta')} | hardnet.epsilon={cfg['filter']['hardnet']['epsilon']} "
          f"box_aware={cfg['filter']['hardnet']['box_aware']}", flush=True)

    results = {}
    t0 = time.time()
    base = summarize(roll(system, h_fn, policy, cfg, scenes, dev, base_params, la_on=False))
    print(f"BASELINE (lookahead OFF) cps={base['cps_v2']:.4f} coll={base['collision']:.4f} "
          f"timeout={base['timeout']:.4f} empty_inf={base['empty_inf']:.4f} dun={base['dun_mean']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["baseline"] = base

    for n in GRID_N:
        for b in GRID_BETA:
            la_cfg_over = copy.deepcopy(cfg)
            la_cfg_over["filter"]["lookahead"] = {"enabled": True, "N": n, "beta": b, "delta": DELTA}
            la_params = _hardnet_params(la_cfg_over)
            tc = time.time()
            r = summarize(roll(system, h_fn, policy, cfg, scenes, dev, base_params, la_on=True, la_params=la_params))
            key = f"N{n}_beta{b}"
            results[key] = r
            print(f"  {key}: cps={r['cps_v2']:.4f} coll={r['collision']:.4f} timeout={r['timeout']:.4f} "
                  f"empty_inf={r['empty_inf']:.4f} dun={r['dun_mean']:.4f} "
                  f"(dcps={r['cps_v2']-base['cps_v2']:+.4f} dcoll={r['collision']-base['collision']:+.4f} "
                  f"dinf={r['empty_inf']-base['empty_inf']:+.4f}) ({time.time()-tc:.0f}s)", flush=True)

    out = dict(run=run_dir.name, step=step, delta=DELTA, grid_N=GRID_N, grid_beta=GRID_BETA, results=results)
    json.dump(out, open(SP / "quadrotor_lookahead_eval.json", "w"), indent=2)
    print(f"saved {SP / 'quadrotor_lookahead_eval.json'} (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
