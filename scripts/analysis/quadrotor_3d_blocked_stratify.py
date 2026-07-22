"""v2.7.3 M1-close — navigational-vs-IC-driven difficulty test (replaces frac>=1 as the selection criterion).

Item (a) only (count x1.5), n=500 at in-loop seed 12345, nominal-only (per-rotor cascaded-PD) rolled to eval
max_steps. Per episode records `blocked` (does any ACTIVE cylinder disk meet the xy-projection of the
start->goal segment) and `collision`. Reports collision|blocked, collision|not_blocked with percentile
bootstrap 95% CIs, the blocked share of all collisions, and n per stratum.

PRE-REGISTERED PASS (fixed before running): collision|blocked >= 1.5 * collision|not_blocked AND the two 95%
CIs are separated. FAIL => difficulty is IC-driven, geometry is the wrong lever: HALT, build no pools.
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
ap.add_argument("--count-scale", type=float, default=1.5)
ap.add_argument("--radius-scale", type=float, default=1.0)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--n-resample", type=int, default=1000)
ap.add_argument("--boot-seed", type=int, default=20260508)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", default="a_blocked_stratify")
a = ap.parse_args()


def _seg_point_dist(p0, p1, c):
    ab = p1 - p0
    t = np.clip(np.dot(c - p0, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p0 + t * ab - c))


cfg = load_effective_config(); cfg["run"]["system"] = "quadrotor_3d"
bo = cfg["obstacle"]
ov = {"n_min": int(round(a.count_scale * int(bo["n_min"]))), "n_max": int(round(a.count_scale * int(bo["n_max"]))),
      "r_min": a.radius_scale * float(bo["r_min"]), "r_max": a.radius_scale * float(bo["r_max"])}
cfg["obstacle"].setdefault("per_system", {})["quadrotor_3d"] = ov

system = make_system(cfg)
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
rng = np.random.default_rng(a.seed)
scenes = [sample_eval_scene(rng, cfg, "quadrotor_3d") for _ in range(a.n)]

blocked = np.array([
    any(_seg_point_dist(np.asarray(s.start, float)[:2], np.asarray(s.goal, float)[:2],
                        np.asarray(s.obstacle_centers, float)[j]) < np.asarray(s.obstacle_radii, float)[j]
        for j in np.nonzero(np.asarray(s.obstacle_active, bool))[0])
    for s in scenes
], dtype=bool)

bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())
goal = bs.goal
centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=torch.float32)
radii = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=torch.float32)
active = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
collided = torch.zeros(x.shape[0], dtype=torch.bool)
with torch.no_grad():
    for _ in range(max_steps):
        u = system.lqr_action(x, goal)
        x = rk4_step(system, x, u, dt)
        d = torch.linalg.norm(centers - x[:, :2].unsqueeze(1), dim=-1) - radii
        collided |= (d.masked_fill(~active, torch.inf).min(dim=1).values < 0.0)
coll = collided.numpy().astype(float)


def _boot_ci(flags):
    if flags.size == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(a.boot_seed)
    idx = r.integers(0, flags.size, size=(a.n_resample, flags.size))
    means = flags[idx].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


cb, cnb = coll[blocked], coll[~blocked]
rate_b, rate_nb = float(cb.mean()), float(cnb.mean())
ci_b, ci_nb = _boot_ci(cb), _boot_ci(cnb)
ratio = rate_b / rate_nb if rate_nb > 0 else float("inf")
ci_sep = ci_b[0] > ci_nb[1]
res = {
    "label": a.label, "count_scale": a.count_scale, "radius_scale": a.radius_scale, "n": a.n, "seed": a.seed,
    "n_blocked": int(blocked.sum()), "n_not_blocked": int((~blocked).sum()),
    "collision_given_blocked": round(rate_b, 4), "ci_blocked": ci_b,
    "collision_given_not_blocked": round(rate_nb, 4), "ci_not_blocked": ci_nb,
    "ratio_blocked_over_notblocked": round(ratio, 4),
    "blocked_share_of_all_collisions": round(float(cb.sum() / max(coll.sum(), 1e-9)), 4),
    "total_collision": round(float(coll.mean()), 4),
    "PASS_ge_1.5x": bool(ratio >= 1.5), "PASS_ci_separated": bool(ci_sep),
    "VERDICT": "PASS" if (ratio >= 1.5 and ci_sep) else "FAIL",
}
run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f"{a.label}.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
