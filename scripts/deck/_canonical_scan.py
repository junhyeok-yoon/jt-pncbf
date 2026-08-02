"""v2.7.7 Amendment 4 — shared batched canonical-pool scan for M10/M15/M16 selection.

Rolls the comparator (244f4f83, without vertical branch) and ours (09c33bf4, with it) closed-loop on the first N
canonical scenes IN ONE BATCH per arm, records full state trajectories, and computes per-episode diagnostics used
by every downstream selector. Eval-only on the canonical pool. Cached to the scratchpad (regenerated headlessly if
absent) so the deck-assets directory holds only final assets.

Operational definitions (recorded in the manifest):
  collided    : min over time of (nearest active-cylinder surface distance) <= 0
  band_exit   : min p_z <= -4  OR  max p_z >= +4
  reached     : final ||p - goal|| <= 0.6 m  (lenient reach proxy; eval uses goal_radius 0.15 + speed gate)
  success     : (not collided) AND (not band_exit) AND reached     [the SAFE-and-arrived episodes]
  min_clear   : min over time of nearest active-cylinder surface distance (lateral)
  n_close     : # active cylinders whose min-over-time surface distance < 0.6 m (close approaches)
  tilt0_deg   : initial tilt = angle between body-up R(q0)e3 and world-up, degrees
  vz0         : initial vertical velocity
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.maneuver_value import build_safety_h_fn

CK = {"comparator": Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"),
      "ours": Path("data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt")}
POOL = Path("data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")   # CANONICAL 0ef3751b
T, DT, LIMIT, REACH, CLOSE = 200, 0.05, 4.0, 0.6, 0.6
CACHE = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad/canonical_scan.npz")


def _quat_to_R_np(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _roll_batch(ck, scenes):
    """Batched closed-loop rollout (the same batched forward the eval harness uses). Records full states plus V-hat
    (safety value) and total thrust per step. NOTE: once an episode enters a divergent falling regime (comparator
    exiting the floor), the exact fall depth is numerically sensitive to batched-vs-single BLAS; the qualitative
    outcome (exits floor / holds) is robust. Selection AND rendering both read this same batched rollout, so the
    reported numbers are internally consistent."""
    fw, cfg, _ = _load_framework(ck, config_overrides={"env": {"dt": DT}, "eval": {"max_steps": T}})
    sys_ = fw.system; dev = next(fw.value_net.parameters()).device
    h_fn = build_safety_h_fn(sys_, cfg, fw.value_net)
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x = sys_.wrap_state(initial_states_from_batch(bs).float()).to(dev)
    B = x.shape[0]
    X = np.empty((T, B, 13), np.float32); V = np.empty((T, B), np.float32); U = np.empty((T, B), np.float32)
    if hasattr(fw, "reset_deficit_state"):
        fw.reset_deficit_state()
    with torch.no_grad():
        for tstep in range(T):
            X[tstep] = x.detach().cpu().numpy()
            V[tstep] = h_fn(x, bs).reshape(-1).detach().cpu().numpy()
            un = fw.policy(x, bs); out = fw.filter(x, un, bs)
            u = out[0] if isinstance(out, (tuple, list)) else out
            U[tstep] = u.sum(dim=1).detach().cpu().numpy()
            x = rk4_step(sys_, x, u, DT)
    return X, V, U                                           # [T,B,13], [T,B], [T,B]


def _diagnostics(X, scenes):
    B = X.shape[1]
    diag = {k: np.zeros(B) for k in ("min_pz", "max_pz", "min_clear", "n_close", "n_close_1p5", "tilt0_deg", "vz0",
                                     "collided", "band_exit", "reached", "success", "final_goal_dist",
                                     "reach_step", "violation_step")}
    for i, sc in enumerate(scenes):
        Xi = X[:, i, :]
        C = np.asarray(sc.obstacle_centers, np.float64)[:, :2]; R = np.asarray(sc.obstacle_radii, np.float64)
        A = np.asarray(sc.obstacle_active, bool); goal = np.asarray(sc.goal, np.float64).reshape(-1)[:3]
        pxy = Xi[:, :2]
        if A.any():
            d = np.linalg.norm(pxy[:, None, :] - C[None, A, :], axis=2) - R[None, A]   # [T, Kactive] surface dist
            per_cyl_min = d.min(axis=0)                       # nearest approach per cylinder
            diag["min_clear"][i] = d.min()
            diag["n_close"][i] = int((per_cyl_min < CLOSE).sum())
            diag["n_close_1p5"][i] = int((per_cyl_min < 1.5).sum())
        else:
            diag["min_clear"][i] = np.inf
        diag["min_pz"][i] = Xi[:, 2].min(); diag["max_pz"][i] = Xi[:, 2].max()
        R0 = _quat_to_R_np(Xi[0, 3:7]); diag["tilt0_deg"][i] = np.degrees(np.arccos(np.clip(R0[2, 2], -1, 1)))
        diag["vz0"][i] = Xi[0, 9]
        gd = np.linalg.norm(Xi[:, :3] - goal[None, :], axis=1)         # distance to goal over time
        diag["final_goal_dist"][i] = float(gd[-1])
        rsteps = np.nonzero(gd <= REACH)[0]; diag["reach_step"][i] = int(rsteps[0]) if len(rsteps) else -1
        vsteps = np.nonzero(np.abs(Xi[:, 2]) >= LIMIT)[0]; diag["violation_step"][i] = int(vsteps[0]) if len(vsteps) else -1
        col = diag["min_clear"][i] <= 0.0
        be = (diag["min_pz"][i] <= -LIMIT) or (diag["max_pz"][i] >= LIMIT)
        rc = diag["reach_step"][i] >= 0                                # reached if it ever entered the goal region
        diag["collided"][i] = float(col); diag["band_exit"][i] = float(be); diag["reached"][i] = float(rc)
        diag["success"][i] = float((not col) and (not be) and rc)
    return diag


def load_or_build_scan(N=200, rebuild=False):
    keys = ("Xc", "Vc", "Uc", "Xo", "Vo", "Uo")
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        if all(k in z for k in keys):
            out = {k: z[k] for k in keys}
            out.update({"diag_c": z["diag_c"].item(), "diag_o": z["diag_o"].item(), "N": int(z["N"])})
            return out
    scenes = load_pool(POOL).scenes[:N]
    Xc, Vc, Uc = _roll_batch(CK["comparator"], scenes); Xo, Vo, Uo = _roll_batch(CK["ours"], scenes)
    diag_c = _diagnostics(Xc, scenes); diag_o = _diagnostics(Xo, scenes)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, Xc=Xc, Vc=Vc, Uc=Uc, Xo=Xo, Vo=Vo, Uo=Uo, diag_c=diag_c, diag_o=diag_o, N=N)
    return {"Xc": Xc, "Vc": Vc, "Uc": Uc, "Xo": Xo, "Vo": Vo, "Uo": Uo,
            "diag_c": diag_c, "diag_o": diag_o, "N": N}


def scenes_of(N):
    return load_pool(POOL).scenes[:N]


if __name__ == "__main__":
    import sys as _s
    N = int(_s.argv[1]) if len(_s.argv) > 1 else 8
    s = load_or_build_scan(N, rebuild=True)
    do, dc = s["diag_o"], s["diag_c"]
    print(f"scan N={N}: ours success {int(do['success'].sum())}/{N}, comparator success {int(dc['success'].sum())}/{N}")
    print(f"  ours band_exit {int(do['band_exit'].sum())}, collided {int(do['collided'].sum())}, reached {int(do['reached'].sum())}")
    print(f"  comparator band_exit {int(dc['band_exit'].sum())}, collided {int(dc['collided'].sum())}")
