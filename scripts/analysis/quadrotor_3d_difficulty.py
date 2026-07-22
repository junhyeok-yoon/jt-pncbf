"""v2.7.3 M1 — quadrotor_3d scene-difficulty calibration (promoted from scratchpad/difficulty3d.py).

Measures, over n scenes drawn at a given seed with a scaled cylinder count/radius:
  (i)  segment-intersect frac>=1  — fraction of scenes with >=1 active cylinder intersecting the start->goal
       xy-segment (PURELY GEOMETRIC; plant-independent).
  (ii) nominal-only collision      — cascaded hover-PD (01_env §3.4), no filter, rolled to eval max_steps.

Two plants for (ii):
  per_rotor  — the v2.7.3 corrected plant (make_system): u=(f1..f4), box [0,4.905]^4.
  old_wrench — the superseded v2.7.2 wrench box [0,19.62]x[-1,1]^3, reconstructed for the 0-old HARNESS check
               only (cascaded PD -> wrench -> old-box clamp; rolled through the SAME dynamics via the mixer
               inverse, so no separate plant is forked).

Writes one JSON per item under the given run dir (measurement persistence, 06_workflow §3.2).
Count/radius scale: n_min,n_max -> round(count_scale*.), r_min,r_max -> radius_scale*. ; scale 1.0 = identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init_eval import sample_eval_scene
from src.frameworks.jt_pncbf.train import load_effective_config, make_system

ap = argparse.ArgumentParser()
ap.add_argument("--count-scale", type=float, required=True)
ap.add_argument("--radius-scale", type=float, default=1.0)
ap.add_argument("--plant", choices=["per_rotor", "old_wrench"], required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=12345)            # IN-LOOP selection seed (never 23456 here)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", required=True)
a = ap.parse_args()


def _seg_point_dist(p0, p1, c):
    ab = p1 - p0
    t = np.clip(np.dot(c - p0, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p0 + t * ab - c))


cfg = load_effective_config()
cfg["run"]["system"] = "quadrotor_3d"
# scaled obstacle override (in-memory, per-system namespace -> merged at the single read site)
base_obs = cfg["obstacle"]
ov = {
    "n_min": int(round(a.count_scale * int(base_obs["n_min"]))),
    "n_max": int(round(a.count_scale * int(base_obs["n_max"]))),
    "r_min": a.radius_scale * float(base_obs["r_min"]),
    "r_max": a.radius_scale * float(base_obs["r_max"]),
}
cfg.setdefault("obstacle", base_obs).setdefault("per_system", {})["quadrotor_3d"] = ov

system = make_system(cfg)
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
rng = np.random.default_rng(a.seed)
scenes = [sample_eval_scene(rng, cfg, "quadrotor_3d") for _ in range(a.n)]

# (i) geometric segment-intersect frac>=1
counts = []
for sc in scenes:
    p0 = np.asarray(sc.start, float)[:2]; p1 = np.asarray(sc.goal, float)[:2]
    ce = np.asarray(sc.obstacle_centers, float); r = np.asarray(sc.obstacle_radii, float)
    act = np.asarray(sc.obstacle_active, bool)
    counts.append(int(sum(1 for j in np.nonzero(act)[0] if _seg_point_dist(p0, p1, ce[j]) < r[j])))
counts = np.array(counts)
frac_ge1 = float((counts >= 1).mean())

# (ii) nominal-only collision
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal = bs.goal
centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=torch.float32)
radii = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=torch.float32)
active = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
collided = torch.zeros(x.shape[0], dtype=torch.bool)

OLD_BOX = torch.tensor([[0.0, 19.62], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)


def _nominal(xx):
    if a.plant == "per_rotor":
        return system.lqr_action(xx, goal)
    # old_wrench: cascaded-PD wrench, clamp to the old wrench box, then map to motor-equiv so the SAME
    # (mixer) dynamics reproduce the wrench exactly (M @ M^{-1} w = w). No per-rotor clamp.
    from src.envs.quadrotor_3d import _quat_to_R, _rot
    p = xx[:, :3]; q = xx[:, 3:7]; v = xx[:, 7:10]; omega = xx[:, 10:13]
    a_des = -system.kp_pos * (p - goal) - system.kd_pos * v
    e_up = torch.zeros_like(a_des); e_up[:, 2] = system.gravity
    f_des = system.mass * (a_des + e_up)
    R = _quat_to_R(q); b3 = R[:, :, 2]
    f_thr = (f_des * b3).sum(dim=1).clamp(min=0.0)
    b3_des = f_des / torch.clamp(torch.linalg.norm(f_des, dim=1, keepdim=True), min=1e-9)
    e_att_body = _rot(R.transpose(1, 2), torch.cross(b3, b3_des, dim=1))
    J = system.inertia.to(xx.device, xx.dtype)
    tau = J * (system.kp_att * e_att_body - system.kd_att * omega)
    wrench = torch.cat([f_thr.unsqueeze(-1), tau], dim=1)
    b = OLD_BOX.to(xx.dtype)
    wrench = torch.clamp(wrench, min=b[:, 0], max=b[:, 1])
    return wrench @ system.mixer_inv.to(xx.device, xx.dtype).t()


with torch.no_grad():
    for _ in range(max_steps):
        u = _nominal(x)
        x = rk4_step(system, x, u, dt)
        d = torch.linalg.norm(centers - x[:, :2].unsqueeze(1), dim=-1) - radii
        collided |= (d.masked_fill(~active, torch.inf).min(dim=1).values < 0.0)
collision = float(collided.float().mean())

res = {
    "label": a.label, "plant": a.plant, "count_scale": a.count_scale, "radius_scale": a.radius_scale,
    "n": a.n, "seed": a.seed, "obstacle_override": ov,
    "segment_intersect_frac_ge1": round(frac_ge1, 4),
    "nominal_only_collision": round(collision, 4),
    "intersect_hist": {str(k): int((counts == k).sum()) for k in range(int(counts.max()) + 1)},
}
run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f"{a.label}.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
