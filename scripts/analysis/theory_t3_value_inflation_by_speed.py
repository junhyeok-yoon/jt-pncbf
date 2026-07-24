"""v2.7.4 theory T3 — value inflation decomposed by speed (discriminating test for conj:inflation).

Registered prediction (BEFORE data): evaluating each checkpoint's V_hat on a FIXED state set, the rise from
early to converged checkpoint GROWS with the speed at which the set is evaluated, and is SMALLEST at v = 0.
Falsified if inflation at v = 0 is comparable to inflation at |v_xy| = 1.5 — which would show the loop is not
carried by the velocity term.

Fixed state set (built ONCE, held for every checkpoint): scene index 0 of the frozen d2r in-loop pool; vehicle
position on a grid over the scene region at z = 0; hover attitude (identity quaternion); omega = 0. Four
speeds |v_xy| in {0, 0.5, 1.0, 1.5}, direction toward the NEAREST ACTIVE cylinder axis (worst case for the
radial term, so the effect is not diluted by direction averaging). Reports the MEAN of V_hat over the grid and
separately the mean over grid points OUTSIDE all cylinders (mean, not min — the existing contour evidence
rests on a single order statistic, which is what this measurement exists to replace). The M3 value checkpoint
(the JT run's initialization) is evaluated on the same set so JT-stage inflation is separable from what the
initialization already carried. Eval-only.
"""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np
import torch

from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes

JT_RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
M3_RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-083533__seed42")
POOL = EVAL_POOLS_DIR / "eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)
SPEEDS = [0.0, 0.5, 1.0, 1.5]
G = 41                                                   # grid resolution per axis

scene0 = load_pool(POOL).scenes[0]
cen = np.asarray(scene0.obstacle_centers, float)         # [K,3]
rad = np.asarray(scene0.obstacle_radii, float)
act = np.asarray(scene0.obstacle_active, bool)
act_idx = np.nonzero(act)[0]

# probe a config for world_lim (same for every checkpoint)
_, cfg0, _ = _load_framework(JT_RUN / "checkpoints/best.pt")
W = float(cfg0["env"]["world_lim"])

gx = np.linspace(-W, W, G); gy = np.linspace(-W, W, G)
GX, GY = np.meshgrid(gx, gy, indexing="ij")
P = np.stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)], axis=1)     # z = 0
M = P.shape[0]

# nearest ACTIVE cylinder axis (xy) per grid point -> unit direction TOWARD it (worst case radial)
d_xy = np.linalg.norm(P[:, None, :2] - cen[None, act_idx, :2], axis=2)      # [M, Kact]
near = act_idx[np.argmin(d_xy, axis=1)]
vec = cen[near, :2] - P[:, :2]
nrm = np.linalg.norm(vec, axis=1, keepdims=True)
dir_xy = np.where(nrm > 1e-9, vec / np.maximum(nrm, 1e-9), np.array([[1.0, 0.0]]))
# grid points OUTSIDE all active cylinders
inside_any = (d_xy <= rad[None, act_idx]).any(axis=1)
outside = ~inside_any

def states_at(speed):
    x = np.zeros((M, 13), float)
    x[:, 0:3] = P
    x[:, 3] = 1.0                                        # identity quaternion (hover attitude), (w,x,y,z)
    x[:, 7:9] = dir_xy * speed
    return torch.tensor(x, dtype=torch.float32)

def eval_ckpt(path):
    fw, cfg, _ = _load_framework(Path(path))
    h_fn = make_h_fn(fw.value_net, fw.system)
    bs = batch_scenes([scene0] * M, device=torch.device("cpu"), dtype=torch.float32)
    out = {}
    for s in SPEEDS:
        x = fw.system.wrap_state(states_at(s))
        with torch.no_grad():
            h = h_fn(x, bs).detach().numpy().reshape(-1)
        out[f"{s}"] = {"mean_all": round(float(h.mean()), 6),
                       "mean_outside_cylinders": round(float(h[outside].mean()), 6)}
    return out

ckpts = sorted(JT_RUN.glob("checkpoints/step_*.pt"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
per = []
for cp in ckpts:
    step = int(re.search(r"step_(\d+)", cp.name).group(1))
    per.append({"step": step, "V": eval_ckpt(cp)})
    print(f"JT step {step:6d} " + " ".join(f"v{s}:{per[-1]['V'][str(s)]['mean_all']:+.4f}" for s in SPEEDS), flush=True)

m3 = {"best": eval_ckpt(M3_RUN / "checkpoints/best.pt")}
print("M3 value best.pt " + " ".join(f"v{s}:{m3['best'][str(s)]['mean_all']:+.4f}" for s in SPEEDS), flush=True)

first, last = per[0], per[-1]
delta = {}
for s in SPEEDS:
    k = str(s)
    for key in ("mean_all", "mean_outside_cylinders"):
        delta.setdefault(key, {})[k] = {
            "early_step": first["step"], "early": first["V"][k][key],
            "final_step": last["step"], "final": last["V"][k][key],
            "delta_early_to_final": round(last["V"][k][key] - first["V"][k][key], 6),
            "delta_from_M3_init": round(last["V"][k][key] - m3["best"][k][key], 6),
        }
for key in ("mean_all", "mean_outside_cylinders"):
    d0 = delta[key]["0.0"]["delta_early_to_final"]
    d0m = delta[key]["0.0"]["delta_from_M3_init"]
    for s in SPEEDS:
        k = str(s)
        delta[key][k]["ratio_to_v0_delta"] = (round(delta[key][k]["delta_early_to_final"] / d0, 4) if abs(d0) > 1e-12 else None)
        delta[key][k]["ratio_to_v0_delta_from_M3"] = (round(delta[key][k]["delta_from_M3_init"] / d0m, 4) if abs(d0m) > 1e-12 else None)

out = {"fixed_state_set": {"scene_index": 0, "pool": POOL.name, "grid": f"{G}x{G} over [-{W},{W}]^2 at z=0",
                          "n_grid_points": M, "n_outside_cylinders": int(outside.sum()),
                          "attitude": "identity quaternion (hover)", "omega": 0,
                          "velocity_direction": "toward nearest ACTIVE cylinder axis (worst-case radial)"},
       "speeds": SPEEDS, "jt_per_checkpoint": per, "m3_value_init": m3, "inflation": delta}
(OUT / "t3_value_inflation.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(delta, indent=2))
