"""v2.7.2 M6 (PROTOCOL FOLLOW-UP) — three orthogonal trajectory projections (xy / xz / yz) for quadrotor_3d.
Infinite vertical cylinders: circles in xy, vertical bands in xz/yz. Rolls JT policy+filter on a sample of
the frozen eval pool. Output PNG for 04_eval §3."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool, DEFAULT_OUTPUT_DIR
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--out", required=True)
a = ap.parse_args()

fw, cfg, _ = load_framework_from_checkpoint(Path(a.ckpt))
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
pool = load_pool(DEFAULT_OUTPUT_DIR / "eval_full_quadrotor-3d_n2000_seed23456.pkl")
scenes = pool.scenes[: a.n]
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
x = system.wrap_state(initial_states_from_batch(bs).float())

traj = [x[:, :3].clone()]
with torch.no_grad():
    for _ in range(max_steps):
        un = fw.policy(x, bs); u, _ = fw.filter(x, un, bs); x = rk4_step(system, x, u, dt)
        traj.append(x[:, :3].clone())
T = torch.stack(traj, 0).numpy()                             # [T+1, n, 3]
goals = bs.goal.numpy()
W = float(cfg["env"]["world_lim"])
cmap = plt.cm.viridis(np.linspace(0, 1, a.n))

planes = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=130)
for ax, (name, i, j) in zip(axes, planes):
    # obstacles: union over the sampled scenes (circles in xy; vertical bands in xz/yz)
    for sc in scenes:
        c = np.asarray(sc.obstacle_centers); r = np.asarray(sc.obstacle_radii); act = np.asarray(sc.obstacle_active)
        for k in np.nonzero(act)[0]:
            if name == "xy":
                ax.add_patch(plt.Circle((c[k, 0], c[k, 1]), r[k], color="0.6", alpha=0.18, lw=0))
            else:                                            # infinite cylinder -> vertical band at its i-axis center
                ci = c[k, i]
                ax.axvspan(ci - r[k], ci + r[k], color="0.6", alpha=0.05, lw=0)
    for e in range(a.n):
        ax.plot(T[:, e, i], T[:, e, j], color=cmap[e], lw=1.1, alpha=0.9)
        ax.scatter(T[0, e, i], T[0, e, j], color=cmap[e], marker="o", s=22, zorder=5)
        ax.scatter(goals[e, i], goals[e, j], color=cmap[e], marker="*", s=90, edgecolor="k", lw=0.4, zorder=6)
    ax.set_xlim(-W - 0.5, W + 0.5); ax.set_ylim(-W - 0.5, W + 0.5)
    ax.set_xlabel(name[0]); ax.set_ylabel(name[1]); ax.set_title(f"{name} projection"); ax.set_aspect("equal")
    ax.grid(alpha=0.25)
fig.suptitle(f"quadrotor_3d JT best.pt trajectories (n={a.n}) — o=start, *=goal, bands=infinite cylinders",
             fontsize=12)
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, bbox_inches="tight")
print("saved", a.out)
