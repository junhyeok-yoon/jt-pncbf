"""v2.8.0 rate figures — re-roll the 4 selected episodes across all 8 rate x projection cells, logging
per-step states + commanded per-rotor thrust + empty-branch flag + V_hat. The rate cells persisted only
aggregates, so per-step data must be re-rolled; only the 4 selected scenes are rolled (deterministic per
scene, so this reproduces their trajectories in the full-pool cells exactly). No aggregate is recomputed.
Saves data/runs/v2.8.0/rate_figs/roll_<arm>_<proj>.npz and scene_info.npz."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.outcomes import step_outcomes
from src.envs.quadrotor_3d import _quat_to_R

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/rate_figs"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = {"A": (0.05, 0.05), "B": (0.01, 0.05), "C": (0.01, 0.01), "D": (0.002, 0.002)}
# selected episodes (global pool indices), fixed across all cells
EP = {"clean1": 1982, "clean2": 1867, "recovery": 1871, "floor": 3}
IDX = list(EP.values())

scenes_all = load_pool(POOL).scenes
sel = [scenes_all[i] for i in IDX]

# scene info for the trajectory figure (obstacles, goal, start) — same across cells
bscene0 = batch_scenes(sel, device=DEV, dtype=torch.float32)
x0 = initial_states_from_batch(bscene0)
np.savez(OUT / "scene_info.npz",
         idx=np.array(IDX), roles=np.array(list(EP.keys())),
         obstacle_centers=np.array([np.asarray(s.obstacle_centers) for s in sel]),
         obstacle_radii=np.array([np.asarray(s.obstacle_radii) for s in sel]),
         obstacle_active=np.array([np.asarray(s.obstacle_active) for s in sel]),
         goal=np.array([np.asarray(s.goal) for s in sel]),
         start=x0[:, :3].cpu().numpy())
print("selected episodes:", EP)

for arm, (dt_sim, dt_ctrl) in ARMS.items():
    max_steps = int(round(10.0 / dt_sim)); stuck_w = int(round(3.0 / dt_sim)); kfb = int(round(0.15 / dt_sim))
    substeps = int(round(dt_ctrl / dt_sim))
    for proj in ("enumerate", "dual_solve"):
        if (OUT / f"roll_{arm}_{proj}.npz").exists():
            print(f"skip {arm}/{proj} (exists)", flush=True); continue
        ck = torch.load(str(CK), map_location="cpu", weights_only=False)
        filt = copy.deepcopy(ck["config"]["filter"])
        filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": kfb}; filt["projection"] = proj
        over = {"env": {"dt": dt_sim, "stuck_window_steps": stuck_w, "stuck_radius": 0.10,
                        "band_collision_limit": 4.0, "goal_angrate_radius": 0.48},
                "eval": {"max_steps": max_steps, "dt_ctrl": dt_ctrl}, "filter": filt}
        fw, cfg, _ = load_fw(str(CK), config_overrides=over)
        for n in ("value_net", "policy_net"):
            m = getattr(fw, n, None); m.to(DEV) if m is not None else None
        system, h_fn = fw.system, fw._filter.h_fn
        bscene = batch_scenes(sel, device=DEV, dtype=torch.float32)
        x = initial_states_from_batch(bscene)
        states = [x]; us = []; emp = []; Vs = [h_fn(x, bscene).reshape(-1).detach()]
        held_u = held_e = None
        with torch.no_grad():
            for i in range(max_steps):
                if i % substeps == 0:
                    u_nom = fw.policy(x, bscene)
                    u_safe, _ = fw.filter(x, u_nom, bscene)
                    le = getattr(fw._filter, "last_empty", None)
                    held_e = (le.to(DEV).bool() if le is not None else torch.zeros(len(sel), dtype=torch.bool, device=DEV))
                    held_u = u_safe.detach()
                us.append(held_u); emp.append(held_e)
                x = system.wrap_state(rk4_step(system, x, held_u, dt_sim)); states.append(x)
                Vs.append(h_fn(x, bscene).reshape(-1).detach())
        S = torch.stack(states, 0)                          # [T+1,4,13]
        U = torch.stack(us, 0)                              # [T,4,4]
        EM = torch.stack(emp, 0)                            # [T,4]
        V = torch.stack(Vs, 0)                              # [T+1,4]
        R = _quat_to_R(S[..., 3:7])
        tilt = torch.rad2deg(torch.arccos(R[..., 2, 2].clamp(-1, 1)))   # [T+1,4]
        masks = step_outcomes(S, bscene, system, cfg)
        bl = masks.collided_band_lower.cpu().numpy(); ob = masks.collided_obstacle.cpu().numpy()
        bu = masks.collided_band_upper.cpu().numpy()
        np.savez(OUT / f"roll_{arm}_{proj}.npz",
                 dt_sim=dt_sim, dt_ctrl=dt_ctrl, substeps=substeps,
                 pos=S[..., :3].cpu().numpy(), u=U.cpu().numpy(), empty=EM.cpu().numpy(),
                 V=V.cpu().numpy(), tilt=tilt.cpu().numpy(),
                 band_lower=bl, obstacle=ob, band_upper=bu)
        print(f"rolled {arm}/{proj} (dt_sim {dt_sim}, {S.shape[0]-1} steps)", flush=True)
print("REROLL DONE")
