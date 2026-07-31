"""v2.7.7 M26 (Amdt 13) — shared filter-active rollout + colored-trajectory-grid renderer. Rolls a SOTA checkpoint
closed-loop, recording per step whether the deployed HardNet filter MODIFIED the nominal action beyond a tolerance
(filter-active = any control component changed by more than 1% of its actuator range). Selects the top-6 episodes
by (a) number of active steps, then (b) smallest obstacle clearance, excluding any episode with zero active steps.
Renders the trajectory RED where the filter is active, BLUE where the nominal policy passes through unmodified, with
a legend and an honest per-cell reach mark. 3-D scene style for quadrotor_3d, 2-D for planar/DI/unicycle. The active
mask is computed eval-only from u_nom = policy(x) and u_safe = filter(x, u_nom) — exposed for every system; no src
edits."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from scripts.deck.deck_style import save, C_DESIGN, C_VERIFY, C_FLOOR, C_CYL
import scripts.deck.deck_scene3d as S3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

TOL_REL = 0.01
C_ACT = "#d62728"   # filter active (red); nominal = blue (C_DESIGN)


def roll_active(ck, pool, N):
    fw, cfg, _ = _load_framework(Path(ck))
    sys_ = fw.system; dev = next(fw.value_net.parameters()).device
    dt = float(cfg["env"].get("dt", 0.05)); T = int(cfg.get("eval", {}).get("max_steps", 200))
    goal_r = float(cfg["env"].get("goal_radius", 0.15))
    scenes = load_pool(Path(pool)).scenes[:N]
    bs = batch_scenes(list(scenes), device=dev, dtype=torch.float32)
    x = sys_.wrap_state(initial_states_from_batch(bs).float()).to(dev)
    ub = sys_.u_bounds.to(dev, torch.float32); tol = TOL_REL * (ub[:, 1] - ub[:, 0])
    B = x.shape[0]; dim = x.shape[1]
    X = np.empty((T + 1, B, dim), np.float32); X[0] = x.detach().cpu().numpy()
    act = np.zeros((T, B), bool)
    if hasattr(fw, "reset_deficit_state"):
        fw.reset_deficit_state()
    with torch.no_grad():
        for t in range(T):
            un = fw.policy(x, bs); out = fw.filter(x, un, bs)
            u = out[0] if isinstance(out, (tuple, list)) else out
            act[t] = ((u - un).abs() > tol).any(dim=1).cpu().numpy()
            x = rk4_step(sys_, x, u, dt); X[t + 1] = x.detach().cpu().numpy()
    return X, act, scenes, dt, goal_r


def select_top6(X, act, scenes, goal_r, reach_r=0.20):
    """Amdt-14 selection: REACH episodes only (exclude collision / band violation / stuck / oob / timeout), then
    require >=1 filter-active step, then rank by (#active steps desc, smallest clearance asc); take top 6.
    Returns (order, num_active, min_clear, reached, info)."""
    B = X.shape[1]; num_active = act.sum(0); min_clear = np.full(B, np.inf); reached = np.zeros(B, bool)
    collided = np.zeros(B, bool)
    pos_dim = 3 if X.shape[2] == 13 else 2                         # quadrotor_3d has p_z; 2-D systems don't
    for i, sc in enumerate(scenes):
        c = np.asarray(sc.obstacle_centers, float)[:, :2]; r = np.asarray(sc.obstacle_radii, float)
        a = np.asarray(sc.obstacle_active, bool); goal = np.asarray(sc.goal, float).reshape(-1)
        P = X[:, i, :2]
        if a.any():
            d = np.linalg.norm(P[:, None, :] - c[None, a, :], axis=2) - r[None, a]
            min_clear[i] = float(d.min())
        gd = np.linalg.norm(X[:, i, :pos_dim] - goal[:pos_dim][None, :], axis=1)
        reached[i] = bool(gd.min() <= reach_r)
        band = pos_dim == 3 and (float(X[:, i, 2].min()) <= -4.0 or float(X[:, i, 2].max()) >= 4.0)
        collided[i] = bool(min_clear[i] <= 0.0 or band)
    reach_ep = reached & ~collided                                # HARD requirement: end in reach, no failure
    cand = [i for i in range(B) if reach_ep[i] and num_active[i] > 0]
    order = sorted(cand, key=lambda i: (-int(num_active[i]), min_clear[i]))[:6]
    info = {"n_reach_active_candidates": len(cand), "padded_from_failures": False,
            "chosen_indices": [int(i) for i in order],
            "chosen_active_steps": [int(num_active[i]) for i in order],
            "chosen_min_clearance": [round(float(min_clear[i]), 3) for i in order]}
    return order, num_active, min_clear, reached, info


def _segments_colors(P, act_i):
    segs = np.stack([P[:-1], P[1:]], axis=1)                       # [T, 2, d]
    cols = np.where(act_i[:len(segs)][:, None], C_ACT, C_DESIGN)
    return segs, cols.ravel().tolist()


def render_grid_3d(order, X, act, scenes, sysname, out_name):
    fig = plt.figure(figsize=(16, 10.4)); gs = GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.12)
    for cell, i in enumerate(order):
        ax = fig.add_subplot(gs[cell // 3, cell % 3], projection="3d")
        sc = scenes[i]; centers_xy, radii, goal = S3.scene_geometry(sc); P = X[:, i, :3]
        xlim, ylim, zlim = S3.scene_extent(centers_xy, radii, goal, P); zlim = (max(zlim[0], -6.5), max(zlim[1], 4.5))
        S3.draw_static_3d(ax, centers_xy, radii, goal, xlim, ylim, zlim, title=""); S3.scale_bar_3d(ax, xlim, ylim, zlim)
        segs, cols = _segments_colors(P, act[:, i])
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=1.9))
        ax.tick_params(labelsize=7)
        try:
            ax.set_box_aspect(None, zoom=1.2)
        except TypeError:
            pass
    _legend_suptitle(fig, sysname)
    return save(fig, out_name)


def render_grid_2d(order, X, act, scenes, sysname, out_name):
    fig = plt.figure(figsize=(15, 9.8)); gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.22)
    for cell, i in enumerate(order):
        ax = fig.add_subplot(gs[cell // 3, cell % 3]); sc = scenes[i]
        c = np.asarray(sc.obstacle_centers, float)[:, :2]; r = np.asarray(sc.obstacle_radii, float)
        a = np.asarray(sc.obstacle_active, bool); goal = np.asarray(sc.goal, float).reshape(-1)[:2]; P = X[:, i, :2]
        for (cx, cy), rr in zip(c[a], r[a]):
            ax.add_patch(mpatches.Circle((cx, cy), rr, facecolor=C_CYL, alpha=0.35, edgecolor="black", lw=0.8))
        segs, cols = _segments_colors(P, act[:, i])
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.0))
        ax.plot([P[0, 0]], [P[0, 1]], "o", color="black", ms=6)
        ax.plot([goal[0]], [goal[1]], "*", ms=16, color=C_VERIFY, mec="black")
        lim = max(4.0, float(np.abs(np.concatenate([P.ravel(), c[a].ravel() if a.any() else [], goal])).max()) + 0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=10); ax.set_ylabel("y (m)", fontsize=10); ax.tick_params(labelsize=9)
    _legend_suptitle(fig, sysname)
    return save(fig, out_name)


def _legend_suptitle(fig, sysname):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_ACT, lw=3, label="filter active"),
         Line2D([0], [0], color=C_DESIGN, lw=3, label="nominal policy")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=12, frameon=True)
    fig.suptitle(sysname, fontsize=15, fontweight="bold")
