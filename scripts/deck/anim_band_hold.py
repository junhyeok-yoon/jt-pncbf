"""v2.7.7 (Amdt 7) — band-hold clip, AFTER-only (renamed from anim_band_rescue). The final system (09c33bf4) on a
canonical scene: descends near the band floor, HOLDS inside the band, threads the cylinders, and reaches the goal.
Single-panel 3-D scene (cylinders, band planes, goal, 1 m scale bar) + oriented glyph + fading trail, slide-readable
xy/yz projections, and V̂ + total-thrust strips. Clip ends at the reach step. Absolute capability of the final
system (no before arm). mp4 (H.264) + GIF. Eval-only; reads the shared canonical scan."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import OUT, DPI, C_DESIGN, HAVE_MP4, AFTER
import scripts.deck.deck_scene3d as S3
from scripts.deck._canonical_scan import load_or_build_scan, scenes_of
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

DT, LIMIT, FPS, N, HOLD = 0.05, 4.0, 12, 400, 20

s = load_or_build_scan(N); do, dc = s["diag_o"], s["diag_c"]; idx = np.arange(s["N"])
after_ok = (do["reached"] == 1) & (do["band_exit"] == 0) & (do["collided"] == 0)
before_floor = (dc["band_exit"] == 1) & (dc["collided"] == 0) & (dc["min_pz"] >= -10.0)
gap = do["min_pz"] - dc["min_pz"]
cc = idx[after_ok & before_floor]
sel_idx = int(cc[np.argmax(gap[cc])])
scene = scenes_of(N)[sel_idx]
X = s["Xo"][:, sel_idx, :]; V = s["Vo"][:, sel_idx]; U = s["Uo"][:, sel_idx]
reach = int(do["reach_step"][sel_idx]); END = min(len(X), reach + HOLD)
STRIDE = int(np.ceil(END / 108)); frames = list(range(0, END, STRIDE))
centers_xy, radii, goal = S3.scene_geometry(scene)
xlim, ylim, zlim = S3.scene_extent(centers_xy, radii, goal, X[:END, :3]); zlim = (max(zlim[0], -6.5), max(zlim[1], 4.5))

fig = plt.figure(figsize=(12.5, 8.6))
gs = GridSpec(4, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 1, 0.5, 0.5], hspace=0.5, wspace=0.24)
ax3 = fig.add_subplot(gs[0:2, 0], projection="3d")
axxy = fig.add_subplot(gs[0, 1]); axyz = fig.add_subplot(gs[1, 1])
axV = fig.add_subplot(gs[2, :]); axU = fig.add_subplot(gs[3, :])
S3.draw_static_3d(ax3, centers_xy, radii, goal, xlim, ylim, zlim, title="")
S3.scale_bar_3d(ax3, xlim, ylim, zlim)
try:
    ax3.set_box_aspect(None, zoom=1.25)
except TypeError:
    pass
S3.draw_static_xy(axxy, centers_xy, radii, goal, xlim, ylim)
S3.draw_static_yz(axyz, goal, ylim, zlim)
ax3.plot(X[:END, 0], X[:END, 1], X[:END, 2], color=C_DESIGN, lw=1.0, alpha=0.22, zorder=6)
axxy.plot(X[:END, 0], X[:END, 1], color=C_DESIGN, lw=1.0, alpha=0.25, zorder=6)
axyz.plot(X[:END, 1], X[:END, 2], color=C_DESIGN, lw=1.0, alpha=0.25, zorder=6)
t = np.arange(len(X)) * DT
axV.plot(t[:END], V[:END], color=C_DESIGN, lw=2.0); axV.axhline(0, color="black", lw=0.8, ls=":")
axV.set_ylabel("V̂ (safety)", fontsize=10); axV.set_xticklabels([]); axV.tick_params(labelsize=9); axV.set_xlim(0, END * DT)
axU.plot(t[:END], U[:END], color=C_DESIGN, lw=2.0); axU.set_ylabel("total thrust (N)", fontsize=10)
axU.set_xlabel("time (s)", fontsize=10); axU.tick_params(labelsize=9); axU.set_xlim(0, END * DT)
curV = axV.axvline(0, color="black", lw=1.0); curU = axU.axvline(0, color="black", lw=1.0)
fig.suptitle(f"{AFTER}: holds the band and reaches the goal", fontsize=13, fontweight="bold")

dyn = {"3d": [], "xy": [], "yz": []}


def update(k_):
    for grp in ("3d", "xy", "yz"):
        S3.remove(dyn[grp]); dyn[grp] = []
    p = X[k_, :3]; R = S3.quat_to_R(X[k_, 3:7])
    dyn["3d"] = S3.trail_3d(ax3, X[:k_ + 1, :3], C_DESIGN) + S3.glyph_3d(ax3, p, R, C_DESIGN)
    dyn["xy"] = S3.trail_proj(axxy, X[:k_ + 1, :3], C_DESIGN, "xy") + S3.glyph_proj(axxy, p, C_DESIGN, "xy")
    dyn["yz"] = S3.trail_proj(axyz, X[:k_ + 1, :3], C_DESIGN, "yz") + S3.glyph_proj(axyz, p, C_DESIGN, "yz")
    curV.set_xdata([t[k_], t[k_]]); curU.set_xdata([t[k_], t[k_]])
    return []


anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "anim_band_hold"
anim.save(str(stem) + ".gif", writer=PillowWriter(fps=FPS), dpi=110); status = "mp4 + GIF"
if HAVE_MP4:
    try:
        anim.save(str(stem) + ".mp4", writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400), dpi=DPI)
    except Exception as e:
        status = f"GIF only (mp4 failed: {e})"
plt.close(fig)
print(f"M10→hold {status} -> {stem.name} ; canonical scene idx {sel_idx}; min p_z {do['min_pz'][sel_idx]:.2f} m (holds), "
      f"reach t={reach*DT:.2f}s; clip {END*DT:.1f}s")
