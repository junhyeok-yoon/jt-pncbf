"""v2.7.7 M15 (Amdt 4/5) — hard-avoidance 3-D scene animation, AFTER vertical extension only (ours reaches).
From the shared canonical scan, difficulty rank among ours-REACH episodes with min clearance >= 0.6 m (so the
fixed 1 m glyph never visually overlaps a cylinder; point-mass contact semantics stay manifest-only): most
approaches within 1.5 m, then smallest min clearance. Rank-1. Comparator side-by-side only if it fails the same
IC; here it also reaches, so single panel. Full 3-D scene (cylinders, band planes, goal, 1 m scale bar) + oriented
glyph + fading trail, with slide-readable xy/yz projections (right column) and a clearance-vs-time strip. Clip ends
at the reach step. mp4+GIF. Deterministic selection; scene idx + criteria in the manifest. Eval-only (canonical)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import OUT, DPI, C_DESIGN, C_FLOOR, HAVE_MP4, AFTER
import scripts.deck.deck_scene3d as S3
from scripts.deck._canonical_scan import load_or_build_scan, scenes_of
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

T, DT, FPS, N, HOLD = 200, 0.05, 12, 400, 20      # HOLD = frames held at the goal after reach

s = load_or_build_scan(N); do = s["diag_o"]; idx = np.arange(s["N"])
cand = idx[(do["reached"] == 1) & (do["band_exit"] == 0) & (do["collided"] == 0) & (do["min_clear"] >= 0.6)]
sel_idx = int(sorted(cand, key=lambda i: (-do["n_close_1p5"][i], do["min_clear"][i]))[0])
scene = scenes_of(N)[sel_idx]
X = s["Xo"][:, sel_idx, :]
reach = int(do["reach_step"][sel_idx]); END = min(T, reach + HOLD)
centers_xy, radii, goal = S3.scene_geometry(scene)
xlim, ylim, zlim = S3.scene_extent(centers_xy, radii, goal, X[:END, :3])
STRIDE = max(1, END // 90); frames = list(range(0, END, STRIDE))


def clearance(Xt):
    if len(radii) == 0:
        return np.full(len(Xt), np.inf)
    d = np.linalg.norm(Xt[:, None, :2] - centers_xy[None, :, :], axis=2) - radii[None, :]
    return d.min(axis=1)


fig = plt.figure(figsize=(12.5, 8.2))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 1, 0.62], hspace=0.42, wspace=0.24)
ax3 = fig.add_subplot(gs[0:2, 0], projection="3d")
axxy = fig.add_subplot(gs[0, 1]); axyz = fig.add_subplot(gs[1, 1]); axC = fig.add_subplot(gs[2, :])
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
t = np.arange(T) * DT
axC.plot(t[:END], clearance(X[:END]), color=C_DESIGN, lw=2.0)
axC.axhline(0.0, color=C_FLOOR, ls="--", lw=1.4, label="0 = cylinder contact")
axC.axhline(0.6, color="gray", ls=":", lw=1.0, label="0.6 m")
axC.set_ylabel("lateral clearance (m)", fontsize=10); axC.set_xlabel("time (s)", fontsize=10)
axC.set_xlim(0, END * DT); axC.tick_params(labelsize=9); axC.legend(fontsize=9, loc="upper right", ncol=2)
curC = axC.axvline(0, color="black", lw=1.0)
fig.suptitle(f"After vertical extension — hardest reach ({int(do['n_close_1p5'][sel_idx])} approaches < 1.5 m, "
             f"min clearance {do['min_clear'][sel_idx]:.2f} m)", fontsize=12.5, fontweight="bold")

dyn = {"3d": [], "xy": [], "yz": []}


def update(k_):
    for grp in ("3d", "xy", "yz"):
        S3.remove(dyn[grp]); dyn[grp] = []
    p = X[k_, :3]; R = S3.quat_to_R(X[k_, 3:7])
    dyn["3d"] = S3.trail_3d(ax3, X[:k_ + 1, :3], C_DESIGN) + S3.glyph_3d(ax3, p, R, C_DESIGN)
    dyn["xy"] = S3.trail_proj(axxy, X[:k_ + 1, :3], C_DESIGN, "xy") + S3.glyph_proj(axxy, p, C_DESIGN, "xy")
    dyn["yz"] = S3.trail_proj(axyz, X[:k_ + 1, :3], C_DESIGN, "yz") + S3.glyph_proj(axyz, p, C_DESIGN, "yz")
    curC.set_xdata([t[k_], t[k_]])
    return []


anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "anim_hard_avoidance"
anim.save(str(stem) + ".gif", writer=PillowWriter(fps=FPS), dpi=110); status = "mp4 + GIF"
if HAVE_MP4:
    try:
        anim.save(str(stem) + ".mp4", writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400), dpi=DPI)
    except Exception as e:
        status = f"GIF only (mp4 failed: {e})"
plt.close(fig)
print(f"M15 {status} -> {stem.name} ; canonical scene idx {sel_idx}; single panel (after; comparator also reaches)")
print(f"  approaches<1.5m {int(do['n_close_1p5'][sel_idx])}, min clearance {do['min_clear'][sel_idx]:.2f} m, "
      f"reach step {reach} (t={reach*DT:.2f}s); clip {END*DT:.1f}s")
