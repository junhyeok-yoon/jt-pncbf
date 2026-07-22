"""v2.7.3 M4 power reads (BEFORE the JT run): on the full d2 pool with nominal + HardNet(M3 V_hat),
(a) empty-branch rate — the denominator M7's fallback trial acts on (v2.7.2 context 1.51%);
(b) collision stratified by the Stage-A `blocked` flag — where the residual sits.
Eval-only, per-rotor plant, empty_fallback none. Persists a JSON under the run's own dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import HardNetFilter
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import DEFAULT_OUTPUT_DIR, load_pool, pool_stem, pool_variant
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", default="m4_power_reads")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
system = make_system(cfg)
vnet = ValueNetEnsemble(system.obs_dim, cfg); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
for p in vnet.parameters():
    p.requires_grad_(False)
h_fn = make_h_fn(vnet, system)
filt = HardNetFilter(system, h_fn, cfg)

stem = pool_stem("full", "quadrotor_3d", int(cfg["eval"]["full"]["n"]),
                 int(cfg["eval"]["full"]["seed"]), "random", pool_variant(cfg, "quadrotor_3d"))
pool = load_pool(DEFAULT_OUTPUT_DIR / f"{stem}.pkl")
scenes = pool.scenes


def _segd(p0, p1, c):
    ab = p1 - p0; t = np.clip(np.dot(c - p0, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p0 + t * ab - c))


blocked = np.array([
    any(_segd(np.asarray(s.start, float)[:2], np.asarray(s.goal, float)[:2],
              np.asarray(s.obstacle_centers, float)[j]) < np.asarray(s.obstacle_radii, float)[j]
        for j in np.nonzero(np.asarray(s.obstacle_active, bool))[0]) for s in scenes], dtype=bool)

bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float()); goal = bs.goal
centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=torch.float32)
radii = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=torch.float32)
active_obs = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
gr = float(cfg["env"]["goal_radius"]); gsr = float(cfg["env"]["goal_speed_radius"])

collided = torch.zeros(x.shape[0], dtype=torch.bool)
had_empty = torch.zeros(x.shape[0], dtype=torch.bool)
active = torch.ones(x.shape[0], dtype=torch.bool)     # episode not yet resolved
empty_active_steps = 0; active_steps_total = 0
with torch.no_grad():
    for _ in range(max_steps):
        un = system.lqr_action(x, goal)
        u, _ = filt(x, bs, un)
        em = filt.last_empty.bool()
        empty_active_steps += int((em & active).sum()); active_steps_total += int(active.sum())
        had_empty |= (em & active)
        x = rk4_step(system, x, u, dt)
        d = torch.linalg.norm(centers - x[:, :2].unsqueeze(1), dim=-1) - radii
        pen = (d.masked_fill(~active_obs, torch.inf).min(dim=1).values < 0.0)
        collided |= (pen & active)
        reached = (torch.linalg.norm(x[:, :3] - goal, dim=1) < gr) & (torch.linalg.norm(x[:, 7:10], dim=1) < gsr)
        active &= ~(pen | reached)

coll = collided.numpy().astype(float)
cb, cnb = coll[blocked], coll[~blocked]
res = {
    "label": a.label, "ckpt": a.ckpt, "pool": stem, "n": len(scenes),
    "empty_branch_rate": round(empty_active_steps / max(active_steps_total, 1), 6),
    "episode_empty_share": round(float(had_empty.float().mean()), 4),
    "total_collision": round(float(coll.mean()), 4),
    "collision_given_blocked": round(float(cb.mean()), 4), "n_blocked": int(blocked.sum()),
    "collision_given_not_blocked": round(float(cnb.mean()), 4), "n_not_blocked": int((~blocked).sum()),
    "blocked_share_of_collisions": round(float(cb.sum() / max(coll.sum(), 1e-9)), 4),
}
Path(a.run_dir).mkdir(parents=True, exist_ok=True)
(Path(a.run_dir) / f"{a.label}.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
