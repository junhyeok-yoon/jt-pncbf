"""v2.7.7 M16 (Amdt 4/5) — attitude-recovery 3-D scene animation, AFTER vertical extension. From the shared
canonical scan, candidates = ours-REACH episodes with initial tilt >=150 deg AND v_z0 < -0.5 (near-inverted,
descending); relax to >=140 deg only if empty; pick the smallest min p_z (nearest floor miss). Full 3-D scene
(cylinders, band planes, goal, 1 m scale bar) + oriented glyph (tilt visibly recovering via R(q)) + fading trail,
slide-readable xy/yz projections (right column), plus a theta(t) strip with a dashed 60 deg 'holding-cone limit'
line. Clip ends at the reach step. mp4+GIF. Deterministic selection; scene idx + criteria in the manifest."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import OUT, DPI, C_DESIGN, C_VERIFY, HAVE_MP4
import scripts.deck.deck_scene3d as S3
from scripts.deck._canonical_scan import load_or_build_scan, scenes_of
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

T, DT, FPS, N, HOLD = 200, 0.05, 12, 400, 20

s = load_or_build_scan(N); do = s["diag_o"]; idx = np.arange(s["N"])
base = (do["reached"] == 1) & (do["band_exit"] == 0) & (do["collided"] == 0) & (do["vz0"] < -0.5)
cand = idx[base & (do["tilt0_deg"] >= 150.0)]
tilt_gate = 150
if len(cand) == 0:                                          # relax to >=140 only if empty
    cand = idx[base & (do["tilt0_deg"] >= 140.0)]; tilt_gate = 140
assert len(cand) > 0, "no ours-reach candidate with tilt>=140 and vz0<-0.5"
sel_idx = int(cand[np.argmin(do["min_pz"][cand])])
scene = scenes_of(N)[sel_idx]
X = s["Xo"][:, sel_idx, :]
reach = int(do["reach_step"][sel_idx]); END = min(T, reach + HOLD)
centers_xy, radii, goal = S3.scene_geometry(scene)
xlim, ylim, zlim = S3.scene_extent(centers_xy, radii, goal, X[:END, :3])


def tilt_series(Xt):
    return np.array([np.degrees(np.arccos(np.clip(S3.quat_to_R(Xt[k, 3:7])[2, 2], -1, 1))) for k in range(len(Xt))])


theta = tilt_series(X); t = np.arange(T) * DT
STRIDE = max(1, END // 90); frames = list(range(0, END, STRIDE))

fig = plt.figure(figsize=(12.5, 8.2))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 1, 0.62], hspace=0.42, wspace=0.24)
ax3 = fig.add_subplot(gs[0:2, 0], projection="3d")
axxy = fig.add_subplot(gs[0, 1]); axyz = fig.add_subplot(gs[1, 1]); axT = fig.add_subplot(gs[2, :])
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
axT.plot(t[:END], theta[:END], color=C_DESIGN, lw=2.0)
axT.axhline(60.0, color=C_VERIFY, ls="--", lw=1.4, label="holding-cone limit (60°)")
axT.set_ylabel("tilt θ (deg)", fontsize=10); axT.set_xlabel("time (s)", fontsize=10)
axT.set_xlim(0, END * DT); axT.set_ylim(0, 180); axT.tick_params(labelsize=9); axT.legend(fontsize=9, loc="upper right")
curT = axT.axvline(0, color="black", lw=1.0)
fig.suptitle(f"After vertical extension — attitude recovery: start tilt {theta[0]:.0f}°, descending, reaches goal\n"
             f"(nearest floor miss {do['min_pz'][sel_idx]:.2f} m)", fontsize=12, fontweight="bold")

dyn = {"3d": [], "xy": [], "yz": []}


def update(k_):
    for grp in ("3d", "xy", "yz"):
        S3.remove(dyn[grp]); dyn[grp] = []
    p = X[k_, :3]; R = S3.quat_to_R(X[k_, 3:7])
    dyn["3d"] = S3.trail_3d(ax3, X[:k_ + 1, :3], C_DESIGN) + S3.glyph_3d(ax3, p, R, C_DESIGN)
    dyn["xy"] = S3.trail_proj(axxy, X[:k_ + 1, :3], C_DESIGN, "xy") + S3.glyph_proj(axxy, p, C_DESIGN, "xy")
    dyn["yz"] = S3.trail_proj(axyz, X[:k_ + 1, :3], C_DESIGN, "yz") + S3.glyph_proj(axyz, p, C_DESIGN, "yz")
    curT.set_xdata([t[k_], t[k_]])
    return []


anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "anim_attitude_recovery"
anim.save(str(stem) + ".gif", writer=PillowWriter(fps=FPS), dpi=110); status = "mp4 + GIF"
if HAVE_MP4:
    try:
        anim.save(str(stem) + ".mp4", writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400), dpi=DPI)
    except Exception as e:
        status = f"GIF only (mp4 failed: {e})"
plt.close(fig)
print(f"M16 {status} -> {stem.name} ; canonical scene idx {sel_idx}; tilt gate {tilt_gate}°; candidates {len(cand)}")
print(f"  start tilt {theta[0]:.1f}°, vz0 {do['vz0'][sel_idx]:.2f}, final tilt {theta[END-1]:.1f}°, "
      f"min p_z {do['min_pz'][sel_idx]:.2f} m, reach step {reach} (t={reach*DT:.2f}s)")
