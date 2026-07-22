"""v2.6.2 amendment 3 Task B — validate the attitude-aware recoverability criterion against the D1 sets.
Re-roll the brake-envelope best.pt on the full pool (born-doomed = infeasible@0 OR envelope-deficit>0 at
t=0, matching the residual-anatomy classification), collect born-doomed collision ICs + reach-outcome ICs,
and for each margin in {0.0,0.1,0.2} report: (1) born-doomed flag rate (TP, need >=0.80), (2) reach flag
rate (FP, need <=0.02), (3) fresh-sampling rejection fraction. Select the SMALLEST margin passing both.
Read-only; eval-only; the criterion is src/common/quadrotor_recoverability.is_recoverable.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.observation import scene_obstacle_tensors
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_recoverability import is_recoverable, plant_params
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_train_scene
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
RUN = REPO / "data/runs/v2.6.2/set__20260716-182949__seed42/v2.6.2__20260716-182949__seed42"
TAU_BRAKE = 0.6
MARGINS = [0.0, 0.1, 0.2]


def scene_ic(sc, plant):
    """(p0, v0, theta0, omega0, centers, radii, active) numpy for one pool scene."""
    return (np.asarray(sc.start, np.float64), np.asarray(sc.initial_velocity, np.float64),
            float(sc.initial_attitude or 0.0), float(sc.initial_omega or 0.0),
            np.asarray(sc.obstacle_centers, np.float64), np.asarray(sc.obstacle_radii, np.float64),
            np.asarray(sc.obstacle_active, bool))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]; plant = plant_params(cfg)
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    ms = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    scenes = load_pool(POOL).scenes; N = len(scenes)

    # re-roll: classify born-doomed collisions + reach episodes (indices)
    born_idx, reach_idx = [], []
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
                born_idx.append(gi)
            elif res.outcome[i] == "goal":
                reach_idx.append(gi)
    reach_sample = reach_idx[:400] if len(reach_idx) >= 300 else reach_idx
    print(f"born-doomed={len(born_idx)} reach(sample)={len(reach_sample)}/{len(reach_idx)}", flush=True)

    # fresh-sampling rejection fraction (train scenes, seeded)
    rng = np.random.default_rng(20260716)
    fresh = [sample_train_scene(rng, cfg, "quadrotor_planar") for _ in range(1000)]

    rows = []
    for m in MARGINS:
        tp = np.mean([not is_recoverable(*scene_ic(scenes[i], plant), plant=plant, margin=m) for i in born_idx])
        fp = np.mean([not is_recoverable(*scene_ic(scenes[i], plant), plant=plant, margin=m) for i in reach_sample])
        # fresh rejection: fraction of the (already position-filtered) fresh scenes the recov test rejects
        fr = np.mean([not is_recoverable(np.asarray(s.start, np.float64), np.asarray(s.initial_velocity, np.float64),
                                         float(s.initial_attitude or 0.0), float(s.initial_omega or 0.0),
                                         np.asarray(s.obstacle_centers, np.float64), np.asarray(s.obstacle_radii, np.float64),
                                         np.asarray(s.obstacle_active, bool), plant, m) for s in fresh])
        passes = (tp >= 0.80) and (fp <= 0.02)
        rows.append(dict(margin=m, tp_born_flag=float(tp), fp_reach_flag=float(fp), fresh_reject=float(fr), passes=bool(passes)))
        print(f"  margin={m}: TP(born flagged)={tp:.3f} FP(reach flagged)={fp:.3f} fresh_reject={fr:.3f} passes={passes}", flush=True)

    passing = [r["margin"] for r in rows if r["passes"]]
    selected = min(passing) if passing else None
    print(f"SELECTED margin = {selected}" if selected is not None else "NO margin passes both gates", flush=True)
    json.dump(dict(n_born=len(born_idx), n_reach=len(reach_idx), rows=rows, selected=selected),
              open(SP / "quadrotor_recov_validate.json", "w"), indent=2)


if __name__ == "__main__":
    main()
