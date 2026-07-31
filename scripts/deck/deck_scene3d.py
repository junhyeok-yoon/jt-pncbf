"""v2.7.7 Amendment 3 — shared 3-D scene renderer for deck animations / trajectory stills.

Draws the ACTUAL environment the vehicle flies in (not abstract traces):
  - infinite vertical cylinders as translucent gray surfaces (scene.obstacle_centers c_xy, radii r),
  - band floor / ceiling as translucent red / purple planes at z = -+z_lim (= 4.0),
  - goal marker, arena bounds,
  - an oriented quadrotor glyph built from R(q) (X-frame arms + rotor discs) so tilt/recovery are visible,
  - a fading trail, and matching xy-top / yz-side projections with obstacle cross-sections + projected trail.

State x = [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]; q = [w,x,y,z], R(q) body->world (src/envs/quadrotor_3d.py).
GLYPH_SCALE inflates the 0.17 m physical arm to world scale for visibility ONLY — it changes no measured number.
All on-frame text is meeting-clean (no sha / pool / device / jargon). Static scene is drawn ONCE; per frame only the
glyph + trail artists are removed and redrawn (fast enough for a one-shot headless render)."""
from __future__ import annotations
import numpy as np
from scripts.deck.deck_style import C_DESIGN, C_LEARNED, C_VERIFY, C_CYL, C_FLOOR, C_CEIL

Z_LIM = 4.0
GLYPH_ARM = 0.5                        # constant EVERYWHERE (Amdt 5): arm 0.5 m -> X diagonal 1.0 m world.
                                       # No per-asset overrides; a 1 m scale bar accompanies each 3-D panel.
                                       # Vehicle is a point mass in the dynamics; the glyph is a fixed marker.


def quat_to_R(q):
    """q = [w,x,y,z] (unit) -> 3x3 body->world, matching src/envs/quadrotor_3d._quat_to_R."""
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def scene_geometry(scene):
    """Return (centers_xy [K,2], radii [K], goal [3]) for ACTIVE obstacles only."""
    c = np.asarray(scene.obstacle_centers, dtype=float)
    r = np.asarray(scene.obstacle_radii, dtype=float)
    a = np.asarray(scene.obstacle_active, dtype=bool)
    c_xy = c[:, :2][a]
    return c_xy, r[a], np.asarray(scene.goal, dtype=float)


def scene_extent(centers_xy, radii, goal, traj, pad=1.0):
    """World bounds enclosing obstacles, goal, and the flown trajectory (+ pad). z clamped to include the band."""
    xs = [goal[0]] + list(traj[:, 0])
    ys = [goal[1]] + list(traj[:, 1])
    for (cx, cy), rr in zip(centers_xy, radii):
        xs += [cx - rr, cx + rr]; ys += [cy - rr, cy + rr]
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)
    zpad = 1.0
    zlim = (min(-Z_LIM - 0.5, traj[:, 2].min() - zpad), max(Z_LIM + 0.5, traj[:, 2].max() + zpad))
    return xlim, ylim, zlim


# ----------------------------- static scene (drawn once) -----------------------------
def draw_static_3d(ax, centers_xy, radii, goal, xlim, ylim, zlim, title=""):
    zt = np.linspace(zlim[0], zlim[1], 2)
    th = np.linspace(0, 2 * np.pi, 40)
    for (cx, cy), rr in zip(centers_xy, radii):                 # cylinders: translucent gray surfaces
        T, Z = np.meshgrid(th, zt)
        ax.plot_surface(cx + rr * np.cos(T), cy + rr * np.sin(T), Z,
                        color=C_CYL, alpha=0.28, linewidth=0, shade=True, zorder=1)
    xx, yy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, -Z_LIM), color=C_FLOOR, alpha=0.16, linewidth=0, zorder=0)
    ax.plot_surface(xx, yy, np.full_like(xx, Z_LIM), color=C_CEIL, alpha=0.14, linewidth=0, zorder=0)
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=180, color=C_VERIFY, edgecolor="black", zorder=6, label="goal")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.view_init(elev=18, azim=-60)


def draw_static_xy(ax, centers_xy, radii, goal, xlim, ylim):
    for (cx, cy), rr in zip(centers_xy, radii):
        ax.add_patch(plt_circle(cx, cy, rr))
    ax.plot([goal[0]], [goal[1]], "*", ms=15, color=C_VERIFY, mec="black")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=10); ax.set_ylabel("y (m)", fontsize=10)
    ax.set_title("top view (xy)", fontsize=10); ax.tick_params(labelsize=9)


def draw_static_yz(ax, goal, ylim, zlim):
    ax.axhspan(-Z_LIM, zlim[0], color=C_FLOOR, alpha=0.16)      # below floor = out
    ax.axhspan(Z_LIM, zlim[1], color=C_CEIL, alpha=0.14)        # above ceiling = out
    ax.axhline(-Z_LIM, color=C_FLOOR, lw=1.6, ls="--")
    ax.axhline(Z_LIM, color=C_CEIL, lw=1.6, ls="--")
    ax.plot([goal[1]], [goal[2]], "*", ms=15, color=C_VERIFY, mec="black")
    ax.set_xlim(*ylim); ax.set_ylim(*zlim)
    ax.set_xlabel("y (m)", fontsize=10); ax.set_ylabel("z (m)", fontsize=10)
    ax.set_title("side view (yz) — band", fontsize=10); ax.tick_params(labelsize=9)


def scale_bar_3d(ax, xlim, ylim, zlim, length=1.0):
    """1 m reference bar in the lower-front corner of a 3-D panel + '1 m' label (Amdt 5)."""
    x0 = xlim[0] + 0.10 * (xlim[1] - xlim[0]); y0 = ylim[0]; z0 = zlim[0] + 0.04 * (zlim[1] - zlim[0])
    ax.plot([x0, x0 + length], [y0, y0], [z0, z0], color="black", lw=3, zorder=12)
    ax.text(x0 + length / 2, y0, z0 + 0.03 * (zlim[1] - zlim[0]), f"{length:.0f} m",
            color="black", fontsize=9, ha="center", va="bottom", zorder=12)


def plt_circle(cx, cy, rr):
    import matplotlib.patches as mpatches
    return mpatches.Circle((cx, cy), rr, facecolor=C_CYL, alpha=0.35, edgecolor="black", lw=0.8)


# ----------------------------- dynamic glyph + trail (per frame) -----------------------------
def _arm_dirs():
    d = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], dtype=float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def glyph_3d(ax, p, R, color):
    """X-frame arms + rotor discs, oriented by R (body->world). Returns list of added artists.
    Fixed arm length GLYPH_ARM everywhere (Amdt 5): X diagonal = 1.0 m world; no per-asset overrides."""
    arts = []
    L = GLYPH_ARM
    ends = [p + R @ (d * L) for d in _arm_dirs()]
    for e in ends:
        arts += ax.plot([p[0], e[0]], [p[1], e[1]], [p[2], e[2]], color=color, lw=2.2, zorder=8)
    bx, by = R[:, 0], R[:, 1]
    tt = np.linspace(0, 2 * np.pi, 20)
    rr = 0.45 * L
    for e in ends:                                             # rotor discs in the body-xy plane
        circ = e[None, :] + rr * (np.cos(tt)[:, None] * bx[None, :] + np.sin(tt)[:, None] * by[None, :])
        arts += ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color=color, lw=1.4, zorder=8)
    up = p + R[:, 2] * (1.4 * L)                               # body-up (thrust axis) marker
    arts += ax.plot([p[0], up[0]], [p[1], up[1]], [p[2], up[2]], color=color, lw=1.0, ls=":", zorder=8)
    return arts


def trail_3d(ax, traj_upto, color, alpha=0.55):
    if len(traj_upto) < 2:
        return []
    return ax.plot(traj_upto[:, 0], traj_upto[:, 1], traj_upto[:, 2], color=color, lw=1.4, alpha=alpha, zorder=7)


def glyph_proj(ax, p, color, plane):
    i, j = (0, 1) if plane == "xy" else (1, 2)
    return ax.plot([p[i]], [p[j]], "o", ms=7, color=color, mec="black", zorder=8)


def trail_proj(ax, traj_upto, color, plane, alpha=0.6):
    if len(traj_upto) < 2:
        return []
    i, j = (0, 1) if plane == "xy" else (1, 2)
    return ax.plot(traj_upto[:, i], traj_upto[:, j], color=color, lw=1.4, alpha=alpha, zorder=7)


def remove(arts):
    for a in arts:
        try:
            a.remove()
        except Exception:
            pass
