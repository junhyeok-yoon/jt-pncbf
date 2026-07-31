"""v2.7.7 M20 (Amdt 8) — shared 2-D trajectory-grid renderer for the per-system SOTA grids (planar, DI,
unicycle). Rolls a secured SOTA checkpoint closed-loop on its recorded eval pool over the first N scenes (NO
outcome filter) and renders a 2-D scene grid: cylinder discs, goal, start, trajectory trail, honest per-cell reach
mark. Position is x[0], x[1] for every 2-D system (DI [px,py,vx,vy]; unicycle [x,y,theta,v]; planar
[px,py,theta,vx,vy,omega]). Eval-only (loads secured ckpt + recorded pool; no src edit)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from scripts.deck.deck_style import save, C_DESIGN, C_VERIFY, C_FLOOR, C_CYL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def render_grid(ck, pool, title, subtitle, out_name, nscene=6, reach_r=0.20):
    fw, cfg, _ = _load_framework(Path(ck))
    sys_ = fw.system; dev = next(fw.value_net.parameters()).device
    DT = float(cfg["env"].get("dt", 0.05)); T = int(cfg.get("eval", {}).get("max_steps", 200))
    scenes = load_pool(Path(pool)).scenes[:nscene]
    bs = batch_scenes(list(scenes), device=dev, dtype=torch.float32)
    x = sys_.wrap_state(initial_states_from_batch(bs).float()).to(dev)
    X = np.empty((T, nscene, x.shape[1]), np.float32)
    if hasattr(fw, "reset_deficit_state"):
        fw.reset_deficit_state()
    with torch.no_grad():
        for k in range(T):
            X[k] = x.detach().cpu().numpy()
            un = fw.policy(x, bs); out = fw.filter(x, un, bs)
            u = out[0] if isinstance(out, (tuple, list)) else out
            x = rk4_step(sys_, x, u, DT)

    def geom(sc):
        c = np.asarray(sc.obstacle_centers, float)[:, :2]; r = np.asarray(sc.obstacle_radii, float)
        a = np.asarray(sc.obstacle_active, bool); g = np.asarray(sc.goal, float).reshape(-1)[:2]
        return c[a], r[a], g

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.22)
    n_reach = 0
    for cell, sc in enumerate(scenes):
        ax = fig.add_subplot(gs[cell // 3, cell % 3])
        c_xy, radii, goal = geom(sc); P = X[:, cell, :2]
        reached = float(np.linalg.norm(P - goal[None, :], axis=1).min()) <= reach_r; n_reach += int(reached)
        for (cx, cy), rr in zip(c_xy, radii):
            ax.add_patch(mpatches.Circle((cx, cy), rr, facecolor=C_CYL, alpha=0.35, edgecolor="black", lw=0.8))
        ax.plot(P[:, 0], P[:, 1], color=C_DESIGN, lw=1.9)
        ax.plot([P[0, 0]], [P[0, 1]], "o", color=C_DESIGN, ms=7, mec="black")
        ax.plot([goal[0]], [goal[1]], "*", ms=17, color=C_VERIFY, mec="black")
        ax.scatter([P[-1, 0]], [P[-1, 1]], s=60, marker=("*" if reached else "X"),
                   color=(C_VERIFY if reached else C_FLOOR), edgecolor="black", zorder=6)
        lim = max(4.0, float(np.abs(np.concatenate([P.ravel(), c_xy.ravel(), goal])).max()) + 0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=10); ax.set_ylabel("y (m)", fontsize=10); ax.tick_params(labelsize=9)
        ax.set_title(f"scene {cell}   {'reaches ✓' if reached else 'does not reach ✗'}", fontsize=11,
                     fontweight="bold", color=("#2e6b2e" if reached else "#a11"))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.045, subtitle, ha="center", fontsize=9.5, style="italic", color="#555555")
    p = save(fig, out_name)
    return p, n_reach, nscene, DT, T
