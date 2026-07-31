"""v2.7.7 (Amdt 7) — after-only reach still (renamed from fig_fall_trajectory). The final system (after the
vertical extension, 09c33bf4) on a canonical scene: descends near the band floor, HOLDS inside the band, and
reaches the goal. Full 3-D scene (cylinders, band planes, goal, 1 m scale bar) + oriented glyph at reach + trail,
slide-readable xy/yz projections, and a p_z(t) strip. Absolute capability of the final system (no before arm).
Eval-only; reads the shared canonical scan; scene idx in the manifest."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import save, C_DESIGN, C_FLOOR, C_CEIL, C_VERIFY, AFTER
import scripts.deck.deck_scene3d as S3
from scripts.deck._canonical_scan import load_or_build_scan, scenes_of
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DT, LIMIT, N, HOLD = 0.05, 4.0, 400, 20

# same canonical scene as the band-hold clip: after descends near the floor, holds, and reaches.
s = load_or_build_scan(N); do, dc = s["diag_o"], s["diag_c"]; idx = np.arange(s["N"])
after_ok = (do["reached"] == 1) & (do["band_exit"] == 0) & (do["collided"] == 0)
before_floor = (dc["band_exit"] == 1) & (dc["collided"] == 0) & (dc["min_pz"] >= -10.0)
gap = do["min_pz"] - dc["min_pz"]
sel_idx = int(idx[after_ok & before_floor][np.argmax(gap[idx[after_ok & before_floor]])])
scene = scenes_of(N)[sel_idx]
X = s["Xo"][:, sel_idx, :]
reach = int(do["reach_step"][sel_idx]); END = min(len(X), reach + HOLD)
centers_xy, radii, goal = S3.scene_geometry(scene)
xlim, ylim, zlim = S3.scene_extent(centers_xy, radii, goal, X[:END, :3]); zlim = (max(zlim[0], -6.5), max(zlim[1], 4.5))

fig = plt.figure(figsize=(12.5, 8.2))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 1, 0.62], hspace=0.42, wspace=0.24)
ax3 = fig.add_subplot(gs[0:2, 0], projection="3d")
axxy = fig.add_subplot(gs[0, 1]); axyz = fig.add_subplot(gs[1, 1]); axpz = fig.add_subplot(gs[2, :])
S3.draw_static_3d(ax3, centers_xy, radii, goal, xlim, ylim, zlim, title="")
S3.scale_bar_3d(ax3, xlim, ylim, zlim)
try:
    ax3.set_box_aspect(None, zoom=1.25)
except TypeError:
    pass
S3.draw_static_xy(axxy, centers_xy, radii, goal, xlim, ylim)
S3.draw_static_yz(axyz, goal, ylim, zlim)
S3.trail_3d(ax3, X[:END, :3], C_DESIGN, alpha=0.6); S3.glyph_3d(ax3, X[reach, :3], S3.quat_to_R(X[reach, 3:7]), C_DESIGN)
S3.trail_proj(axxy, X[:END, :3], C_DESIGN, "xy"); S3.glyph_proj(axxy, X[reach, :3], C_DESIGN, "xy")
S3.trail_proj(axyz, X[:END, :3], C_DESIGN, "yz"); S3.glyph_proj(axyz, X[reach, :3], C_DESIGN, "yz")
t = np.arange(len(X)) * DT
axpz.plot(t[:END], X[:END, 2], color=C_DESIGN, lw=2.2)
axpz.axhspan(zlim[0], -LIMIT, color=C_FLOOR, alpha=0.12); axpz.axhline(-LIMIT, color=C_FLOOR, ls="--", lw=1.4, label="band floor z=-4")
axpz.axhline(LIMIT, color=C_CEIL, ls="--", lw=1.2, label="band ceiling z=+4")
axpz.plot([t[reach]], [X[reach, 2]], "*", ms=16, color=C_VERIFY, mec="black", zorder=6)
axpz.set_xlabel("time (s)", fontsize=10); axpz.set_ylabel("altitude p_z (m)", fontsize=10)
axpz.set_xlim(0, END * DT); axpz.set_ylim(zlim[0], LIMIT + 0.5); axpz.tick_params(labelsize=9); axpz.legend(fontsize=9, loc="lower right", ncol=2)
fig.suptitle(f"{AFTER}: descends near the floor, holds inside the band, and reaches the goal", fontsize=13, fontweight="bold")
p = save(fig, "fig_reach_trajectory.png")
print(f"M14→reach -> {p.name}; canonical scene idx {sel_idx}; min p_z {do['min_pz'][sel_idx]:.2f} m (holds band), reach t={reach*DT:.2f}s")
