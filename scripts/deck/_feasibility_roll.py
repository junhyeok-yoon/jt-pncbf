"""v2.7.7 M21 helper (Amdt 10) — roll the AFTER system (09c33bf4) closed-loop on the full canonical pool (n=2000)
and record per-episode band outcomes + IC states, for the physics-feasibility stratification. Eval-only; records
min p_z / max p_z (band floor/ceiling violation), min lateral clearance (cylinder collision), and final/nearest
goal distance (reach), from the trajectory. Saved to scratch npz. Uses the committed filter config (empty_fallback
none) — the band-violation SET is physics-driven; any gap vs the recorded kstep-k5 aggregate is reported."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step

CK = Path("data/previous_runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt")
POOL = Path("data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
OUT = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad/feas_roll.npz")
T, DT, GOAL_R = 200, 0.05, 0.15

fw, cfg, _ = _load_framework(CK, config_overrides={"env": {"dt": DT}, "eval": {"max_steps": T}})
sys_ = fw.system; dev = next(fw.value_net.parameters()).device
scenes = load_pool(POOL).scenes
goals = np.array([np.asarray(s.goal, np.float64).reshape(-1)[:3] for s in scenes])
bs = batch_scenes(list(scenes), device=dev, dtype=torch.float32)
x = sys_.wrap_state(initial_states_from_batch(bs).float()).to(dev)
B = x.shape[0]
min_pz = np.full(B, np.inf); max_pz = np.full(B, -np.inf); min_goal = np.full(B, np.inf)
X0 = x.detach().cpu().numpy().copy()
# per-cylinder min clearance
C = [np.asarray(s.obstacle_centers, np.float64)[:, :2] for s in scenes]
R = [np.asarray(s.obstacle_radii, np.float64) for s in scenes]
A = [np.asarray(s.obstacle_active, bool) for s in scenes]
min_clear = np.full(B, np.inf)
if hasattr(fw, "reset_deficit_state"):
    fw.reset_deficit_state()
with torch.no_grad():
    for t in range(T):
        xn = x.detach().cpu().numpy()
        pz = xn[:, 2]; min_pz = np.minimum(min_pz, pz); max_pz = np.maximum(max_pz, pz)
        gd = np.linalg.norm(xn[:, :3] - goals, axis=1); min_goal = np.minimum(min_goal, gd)
        for i in range(B):
            if A[i].any():
                d = np.linalg.norm(xn[i, :2][None, :] - C[i][A[i]], axis=1) - R[i][A[i]]
                min_clear[i] = min(min_clear[i], float(d.min()))
        un = fw.policy(x, bs); out = fw.filter(x, un, bs)
        u = out[0] if isinstance(out, (tuple, list)) else out
        x = rk4_step(sys_, x, u, DT)
np.savez_compressed(OUT, X0=X0, min_pz=min_pz, max_pz=max_pz, min_clear=min_clear, min_goal=min_goal, goals=goals)
band_viol = (min_pz <= -4.0) | (max_pz >= 4.0)
coll = min_clear <= 0.0
reached = (min_goal <= GOAL_R) & ~band_viol & ~coll
print(f"rolled n={B}; band-floor/ceiling violations {int(band_viol.sum())}, cylinder collisions {int(coll.sum())}, "
      f"band-collision (either) {int((band_viol|coll).sum())} ({(band_viol|coll).mean():.4f}), reach {int(reached.sum())} ({reached.mean():.4f})")
print(f"  recorded band row: collision 0.0425 (85), reach 0.9375 (1875)")
