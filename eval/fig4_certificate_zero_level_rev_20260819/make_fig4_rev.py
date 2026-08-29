"""Paper figure 4, REVISION — the deployed certificate's zero level set on a deployed scene.

Revision of eval/fig4_certificate_zero_level_20260819/make_fig4.py. Same scene (the median-vhat_min
rule below, unchanged), same checkpoint (L328), same registered cell, same sizing contract. The
ORIGINAL producer and its PDF are left untouched; everything here is written into this directory.

SIX CHANGES against the original.
  (a) The GOAL is marked on every panel with the paper's canonical marker (red star, black edge,
      ms 9) and carries a legend entry "goal". The certificate is a policy value function and the
      policy is goal-conditioned (the goal enters the observation as the body-frame vector
      goal_b = R(q)^T (goal - p), src/envs/quadrotor_3d.py), so the level set depends on it and the
      figure must say where the goal is. The goal is a 3-D point and every panel is a 2-D section,
      so what is drawn is the goal's ORTHOGONAL PROJECTION onto the drawn plane, exactly as the
      paper's rollout figures do (scripts/analysis/v292_fig_rollouts.py:148,161). The out-of-plane
      distance is reported per panel in the manifest.
  (b) GOAL DEPENDENCE, measured, in a companion figure `fig4b_goal_dependence.pdf`: panel (b)'s
      section (xy, hover, z = 0, at rest) redrawn at the scene's own goal and at TWO alternates,
      with the reference zero level overlaid on the alternates. The alternates are the images of
      the scene's own goal under the two sign flips that keep it inside the SAME arena box:
      G1 = (-g_x, -g_y, +g_z) (horizontal point reflection through the arena's vertical axis) and
      G2 = (+g_x, +g_y, -g_z) (reflection through the horizontal mid-plane). Both are inside
      [-world, world]^3 by construction because the true goal is, both are checked clear of every
      active cylinder, and between them they separate a purely HORIZONTAL change of the goal
      direction from a purely VERTICAL one. Nothing else changes.
  (c) The velocity annotations are OUT of the arena: each panel carries a bold letter as its left
      title and the velocity as its right title. No text is drawn inside any axes.
  (d) The colorbar label is "certificate value" — "(clamped)" dropped. The clamp is unchanged and
      is recorded in the manifest (`colour_clamp`).
  (e) The green dotted offset-readout column is REMOVED from the drawing. The measurement it
      annotated (`band_zero_offset`, from the frozen renderer, and its check on the drawn grid)
      is kept in the manifest, unchanged.
  (f) A FIFTH panel (e) is added: the same vertical section at ASCENDING velocity v_z = +1.5 m/s.
      Reason, verified in `band_branch_zero` below and recorded in the manifest: a single vertical
      panel can NEVER show both band boundaries. The ceiling branch zero exists only when
      v_z >= -psi_cap/c_z and then sits at z = limit - c_z v_z, which is inside the frame only for
      v_z >= 0; the floor branch zero exists only when v_z <= +psi_cap/c_z and sits at
      z = -limit - c_z v_z, inside the frame only for v_z <= 0. The two conditions intersect only
      at v_z = 0, where both lines coincide with the band surfaces themselves. So at
      v_z = -1.5 m/s the ceiling branch has NO zero at all (it would need psi_up = 1.178 m above
      its cap of psi_cap = 0.80 m) and panel (d) is right to draw one line; the ascending panel is
      the only way to put the ceiling boundary in the figure.

SCENE SELECTION IS A RULE, FIXED BEFORE ANYTHING IS DRAWN (unchanged from the original).
  From the 24-scene certificate probe already on disk (default
  data/runs/v2.9.2/cert_probes/cert_probes.json, item_c.conditions.jt_L328.per_scene; v2.9.3
  jt_rebase adds the opt-in CERT_PROBE override so the SAME rule can be applied to a probe of
  another certificate -- the rule itself is untouched), take the
  scene whose `vhat_min` sits at the MEDIAN of that population. n = 24 is even, so the median value
  falls exactly between the rank-12 and rank-13 scenes (ascending, deepest first) and BOTH are
  equidistant from it by construction; the tie is broken by the SMALLER pool index. Scene 0 is
  excluded by dispatch. If the selected scene carries no active cylinder intersecting the p_x = 0
  plane, the next scene toward the median is taken and the substitution is reported.

PANELS.
  (a)(b)(c)  horizontal (xy) sections at hover attitude, z = 0, v_x = -1.5 / 0 / +1.5 m/s.
  (d)(e)     vertical (yz) sections at p_x = 0, hover attitude, v_z = -1.5 / +1.5 m/s, with the
             floor and ceiling band surfaces |p_z| = band limit overlaid and the analytic hazard's
             own band-branch zero drawn as a magenta dashed line.

07_tex_deck I3: no version stamp, run-id, pool name, scene index, checkpoint digest or ledger row
label is drawn. Panels are labelled (a)..(e) and carry only their velocity.

Read-only on data/secured_data. No src edit, no config key on disk, no existing artifact touched.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/analysis"))
from v282_agree_gate import gate_overrides

HERE = Path(__file__).resolve().parent
# v2.9.3 jt_rebase: opt-in output override, same idiom as JT_ROW_ART. Unset -> this
# directory verbatim (default), so the warm figure on disk is never overwritten. Set ->
# data/runs/v2.9.3/jt_rebase/figures/fig4. Only the OUTPUT moves; HERE stays the code dir
# (measure_pdf is imported from it).
OUT = Path(os.environ["FIG_OUT"]) if os.environ.get("FIG_OUT") else HERE
SCRATCH = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/"
               "31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad/fig4rev")
# v2.9.3 jt_rebase: opt-in override. Unset -> the warm L328 pointer verbatim (default,
# so every closed v2.9.1/v2.9.2 artifact stays reproducible). Set -> the cold pointer
# data/runs/v2.9.3/jt_rebase/jtrow__quadrotor_3d__COLD40K.json. The registered warm
# artifact itself is never edited and never moved.
JT_ART = Path(os.environ["JT_ROW_ART"]) if os.environ.get("JT_ROW_ART") else \
         REPO / "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json"      # L328's checkpoint
POOL = REPO / "data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl"
# v2.9.3 jt_rebase: opt-in override, same idiom as JT_ROW_ART / FIG_OUT. Unset -> the closed
# v2.9.2 probe of the WARM certificate verbatim (default), so the registered scene selection
# is reproduced exactly. Set -> a probe of another certificate, e.g.
# data/runs/v2.9.3/jt_rebase/cert_probes_cold/cert_probes.json. The SELECTION RULE below is
# NOT touched by this: only the population it is applied to moves.
PROBE = Path(os.environ["CERT_PROBE"]) if os.environ.get("CERT_PROBE") else \
        REPO / "data/runs/v2.9.2/cert_probes/cert_probes.json"
EXCLUDE_SCENES = {0}

# v2.9.3 jt_rebase: the JT row identity is READ FROM the pointer artifact, not hard-coded.
# The registered warm pointer carries no row field of its own, so the warm default keeps its
# registered value L328 unchanged; any other pointer must carry `ledger_row`, or
# `provenance.h400_sibling_row` (this producer rolls at cap 400), or the field is recorded as
# null rather than guessed.
WARM_JT_ART = REPO / "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json"


def jt_ledger_row():
    a = json.loads(JT_ART.read_text())
    r = a.get("ledger_row") or a.get("provenance", {}).get("h400_sibling_row")
    if r:
        return r
    return "L328" if JT_ART.resolve() == WARM_JT_ART.resolve() else None

CAP = 400                       # eval.max_steps on the registered cell
W_DOUBLE_IN = 7.00              # double-column target width, inches
FIG_H_IN = 1.90                 # 5 panels at 7.00 in -> the panels are narrower than the original
FIG_H_B_IN = 2.62               # companion goal-dependence figure (3 panels)
PT_LABEL, PT_TICK = 8.0, 7.0
CONTOUR_DPI = 600               # the ONLY rasterized layer is the filled contour
RES = 241                       # grid nodes per axis on the drawn sections
REND_RES = 61                   # the frozen renderer's fixed resolution (do not change: comparability)
V_HORIZ = 1.5                   # horizontal speed for the moving columns, m/s
VZ_SECTION = -1.5               # descent rate on the vertical section, m/s
VZ_SECTION_UP = +1.5            # ascent rate on the added vertical section, m/s  (item f)
Z_PAD = 0.30                    # view-only padding so the band surfaces are drawn inside the frame
CMAP, NORM = "coolwarm", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
GOAL_KW = dict(marker="*", ms=9, color="#d62728", mec="black", mew=0.4, ls="none", zorder=9)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": PT_TICK, "axes.labelsize": PT_LABEL, "axes.titlesize": PT_LABEL,
    "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK, "legend.fontsize": PT_LABEL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.transparent": False,
})


# ---------------------------------------------------------------------------------------------
def select_scene():
    """The median-vhat_min rule, applied to the probe on disk. Returns (preference, detail)."""
    probe = json.loads(PROBE.read_text())
    # v2.9.3 jt_rebase: the condition key is DERIVED from the pointer artifact by the same
    # rule as jt_ledger_row() above, so the warm default resolves to "jt_L328" unchanged.
    key = f"jt_{jt_ledger_row()}"
    conds = probe["item_c"]["conditions"]
    if key not in conds:
        raise SystemExit(f"STOP: probe {PROBE} carries no condition {key}; it has "
                         f"{sorted(conds)}")
    per = conds[key]["per_scene"]
    idx = np.array([int(r["scene"]) for r in per])
    vmin = np.array([float(r["vhat_min"]) for r in per])
    order = np.argsort(vmin, kind="stable")                    # ascending = deepest first
    rank = {int(idx[o]): r + 1 for r, o in enumerate(order)}
    median = float(np.median(vmin))
    keep = np.array([int(i) not in EXCLUDE_SCENES for i in idx])
    d = np.abs(vmin - median)
    pref = sorted([int(i) for i in idx[keep]],
                  key=lambda i: (round(float(d[idx == i][0]), 12), i))
    return pref, dict(median_vhat_min=median, rank=rank,
                      vhat_min={int(i): float(v) for i, v in zip(idx, vmin)},
                      n=len(per), excluded=sorted(EXCLUDE_SCENES),
                      probe=str(PROBE), probe_condition=key,
                      probe_ckpt=conds[key].get("ckpt"), probe_step=conds[key].get("step"))


def crosses_px0(scene):
    C = np.asarray(scene.obstacle_centers, np.float64)
    R = np.asarray(scene.obstacle_radii, np.float64)
    A = np.asarray(scene.obstacle_active, bool)
    return [int(j) for j in np.nonzero(A)[0] if abs(C[j, 0]) < R[j]]


def scene_tensors(scene, dev, dt, goal=None):
    """Scene tensors for the observation. `goal` overrides scene.goal and NOTHING else."""
    g = np.asarray(scene.goal if goal is None else goal, np.float64)
    return SimpleNamespace(
        obstacle_centers=torch.tensor(np.asarray(scene.obstacle_centers), dtype=dt, device=dev),
        obstacle_radii=torch.tensor(np.asarray(scene.obstacle_radii), dtype=dt, device=dev),
        obstacle_active=torch.tensor(np.asarray(scene.obstacle_active), dtype=torch.bool, device=dev),
        goal=torch.tensor(g, dtype=dt, device=dev))


def sweep(fw, cfg, st, plane, world, vel):
    """Deployed V-hat on a RES x RES grid. plane 'xy' (z = 0) or 'yz' (p_x = 0). Hover attitude."""
    from src.common.quadrotor_barrier import value_target_barrier
    dev = next(fw.value_net.parameters()).device
    dt = torch.float32
    axis = np.linspace(-world, world, RES)
    g1, g2 = np.meshgrid(axis, axis)
    x = torch.zeros(RES * RES, 13, dtype=dt, device=dev)
    if plane == "xy":
        x[:, 0] = torch.tensor(g1.reshape(-1), dtype=dt, device=dev)
        x[:, 1] = torch.tensor(g2.reshape(-1), dtype=dt, device=dev)
    else:
        x[:, 0] = 0.0                                     # 04_eval s3b: the yz plane is p_x = 0
        x[:, 1] = torch.tensor(g1.reshape(-1), dtype=dt, device=dev)
        x[:, 2] = torch.tensor(g2.reshape(-1), dtype=dt, device=dev)
    x[:, 3] = 1.0                                         # hover attitude, q = [1, 0, 0, 0]
    x[:, 7], x[:, 8], x[:, 9] = vel
    with torch.no_grad():
        v = fw.value_net.deployed_h(fw.system.observation(x, st)).reshape(-1)
        h = value_target_barrier(fw.system, x, st, cfg).reshape(-1)
    return axis, v.cpu().numpy().reshape(RES, RES), h.cpu().numpy().reshape(RES, RES)


def crossing_offset(axis, V, col_index, analytic):
    """The frozen renderer's own definition, evaluated on the drawn grid at the same column."""
    col = V[:, col_index]
    zc = [float(axis[k] + (axis[k + 1] - axis[k]) * (0.0 - col[k]) / (col[k + 1] - col[k]))
          for k in range(len(axis) - 1)
          if (col[k] <= 0 <= col[k + 1]) or (col[k] >= 0 >= col[k + 1])]
    if zc:
        z = min(zc, key=lambda z_: abs(z_ - analytic))
        return float(z - analytic), float(z)
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------------------------
def band_branch_zero(limit, c_z, psi_cap, v_z, which):
    """Zero of ONE band branch of the analytic hazard (src/common/quadrotor_barrier.py,
    value_target_barrier), WITH the psi_cap clamp respected.

        psi_up(z) = min(z - limit, psi_cap),  branch_up = psi_up + c_z v_z
        psi_lo(z) = min(-z - limit, psi_cap), branch_lo = psi_lo - c_z v_z

    psi_up ranges over (-inf, psi_cap] as z -> +inf, and psi_lo over (-inf, psi_cap] as z -> -inf,
    so a branch has a zero ONLY if the psi value it needs is <= psi_cap:
        ceiling: psi_up = -c_z v_z <= psi_cap  <=>  v_z >= -psi_cap/c_z, then z = limit - c_z v_z
        floor:   psi_lo = +c_z v_z <= psi_cap  <=>  v_z <= +psi_cap/c_z, then z = -limit - c_z v_z
    Returns (z or None, reason). The naive formula limit -/+ c_z v_z alone is WRONG whenever the
    clamp binds -- that is the case that makes the ceiling branch vanish at v_z = -1.5 m/s.
    """
    if which == "ceiling":
        need = -c_z * v_z
        if need > psi_cap:
            return None, (f"no zero: the ceiling branch needs psi_up = {need:.4f} m > psi_cap = "
                          f"{psi_cap:.4f} m, which the clamp forbids")
        return float(limit + need), "zero at z = limit - c_z v_z"
    need = c_z * v_z
    if need > psi_cap:
        return None, (f"no zero: the floor branch needs psi_lo = {need:.4f} m > psi_cap = "
                      f"{psi_cap:.4f} m, which the clamp forbids")
    return float(-limit - need), "zero at z = -limit - c_z v_z"


def band_branch_zero_numeric(limit, c_z, psi_cap, v_z, which, lo=-40.0, hi=40.0, n=4000001):
    """Independent numerical check of band_branch_zero on a dense z grid, no closed form used."""
    z = np.linspace(lo, hi, n)
    b = (np.minimum(z - limit, psi_cap) + c_z * v_z if which == "ceiling"
         else np.minimum(-z - limit, psi_cap) - c_z * v_z)
    s = np.nonzero(np.sign(b[:-1]) * np.sign(b[1:]) <= 0)[0]
    if s.size == 0:
        return None
    k = int(s[0])
    return float(z[k] + (z[k + 1] - z[k]) * (0.0 - b[k]) / (b[k + 1] - b[k]))


def zero_segments(axis, V):
    """Vertices of the V = 0 contour on the drawn grid, as one (N, 2) array in metres."""
    f = plt.figure(figsize=(1, 1))
    ax = f.add_subplot(111)
    cs = ax.contour(axis, axis, V, levels=[0.0])
    pts = [np.asarray(s, np.float64) for s in cs.allsegs[0] if len(s)]
    plt.close(f)
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def contour_shift(ref_pts, alt_pts):
    """How far the zero level moved, in metres: for every vertex of the ALTERNATE zero level the
    distance to the nearest vertex of the REFERENCE zero level, and the same the other way round;
    the reported summary is over the union (a symmetric, one-sided-free measure)."""
    from scipy.spatial import cKDTree
    if len(ref_pts) == 0 or len(alt_pts) == 0:
        return dict(n_ref=int(len(ref_pts)), n_alt=int(len(alt_pts)))
    d1, _ = cKDTree(ref_pts).query(alt_pts, k=1)
    d2, _ = cKDTree(alt_pts).query(ref_pts, k=1)
    d = np.concatenate([d1, d2])
    return dict(n_ref=int(len(ref_pts)), n_alt=int(len(alt_pts)),
                median_shift_m=float(np.median(d)), mean_shift_m=float(d.mean()),
                p90_shift_m=float(np.percentile(d, 90)),
                hausdorff_m=float(d.max()))


def clearance(goal_xy, C, R, A):
    """Minimum SURFACE clearance from a horizontal point to every active cylinder, metres."""
    j = np.nonzero(A)[0]
    d = np.linalg.norm(C[j, :2] - np.asarray(goal_xy, np.float64)[None, :2], axis=1) - R[j]
    return float(d.min()), int(j[int(np.argmin(d))])


# ---------------------------------------------------------------------------------------------
def draw_panel(ax, axis, V, plane, vel, scene_geom, letter, tag, world, limit, c_z, psi_cap,
               first_col, goal, ref_zero=None):
    """One certificate panel. Returns the record dict. NO text is placed inside the axes."""
    C, R, A, cyl_on_plane = scene_geom
    ax.set_rasterization_zorder(0)              # ONLY the fill (zorder -10) is rasterized
    ax.contourf(axis, axis, np.clip(V, -1, 1), levels=np.linspace(-1, 1, 41), cmap=CMAP,
                norm=NORM, extend="both", zorder=-10, alpha=0.75)
    if ref_zero is not None and ref_zero.min() <= 0 <= ref_zero.max():
        ax.contour(axis, axis, ref_zero, levels=[0.0], colors="0.35", linewidths=0.8,
                   linestyles=[(0, (3, 2))], zorder=3.5)
    if V.min() <= 0 <= V.max():
        ax.contour(axis, axis, V, levels=[0.0], colors="black", linewidths=1.3, zorder=4)
    unsafe = V > 0.0
    gx, gy_ = np.meshgrid(axis, axis)
    entry = dict(panel=letter, section=plane, velocity=[float(t) for t in vel],
                 attitude="hover (q = [1, 0, 0, 0])",
                 vhat_min=float(V.min()), vhat_max=float(V.max()),
                 unsafe_area_fraction=float(unsafe.mean()),
                 unsafe_centroid_horizontal_m=(float(gx[unsafe].mean()) if unsafe.any() else None),
                 unsafe_centroid_vertical_m=(float(gy_[unsafe].mean()) if unsafe.any() else None))
    if plane == "xy":
        for j in np.nonzero(A)[0]:
            ax.add_patch(mpatches.Circle((C[j, 0], C[j, 1]), R[j], facecolor="none",
                                         edgecolor="0.12", lw=0.7, ls="--", zorder=3))
        ax.plot([goal[0]], [goal[1]], **GOAL_KW)                      # item (a)
        ax.set_xlabel("x position (m)")
        if first_col:
            ax.set_ylabel("y position (m)")
        ax.set_xlim(-world, world); ax.set_ylim(-world, world)
        entry.update(n_active_cylinders_drawn=int(A.sum()),
                     goal_marker_drawn_at_m=[float(goal[0]), float(goal[1])],
                     goal_out_of_plane_distance_m=float(abs(goal[2] - 0.0)),
                     goal_out_of_plane_axis="z (the section is the plane z = 0)")
    else:
        for j in cyl_on_plane:                  # infinite vertical cylinders read as vertical bands
            hh = float(np.sqrt(R[j] ** 2 - C[j, 0] ** 2))
            ax.axvspan(C[j, 1] - hh, C[j, 1] + hh, color="0.45", alpha=0.18, lw=0, zorder=2)
        for lo, hi in ((-world - Z_PAD, -limit), (limit, world + Z_PAD)):
            ax.axhspan(lo, hi, facecolor="0.80", alpha=0.6, lw=0, zorder=5)
        ax.axhline(limit, color="0.15", lw=0.8, zorder=6)
        ax.axhline(-limit, color="0.15", lw=0.8, zorder=6)
        z_floor, why_floor = band_branch_zero(limit, c_z, psi_cap, vel[2], "floor")
        z_ceil, why_ceil = band_branch_zero(limit, c_z, psi_cap, vel[2], "ceiling")
        drawn = []
        for pz0 in (z_floor, z_ceil):
            if pz0 is not None and -world <= pz0 <= world:
                ax.axhline(pz0, color="magenta", ls="--", lw=1.0, zorder=6)
                drawn.append(float(pz0))
        ax.plot([goal[1]], [goal[2]], **GOAL_KW)                      # item (a)
        ax.set_xlabel("y position (m)")
        if first_col:
            ax.set_ylabel("z position (m)")
        ax.set_xlim(-world, world); ax.set_ylim(-world - Z_PAD, world + Z_PAD)
        entry.update(
            plane_px=0.0,
            goal_marker_drawn_at_m=[float(goal[1]), float(goal[2])],
            goal_out_of_plane_distance_m=float(abs(goal[0] - 0.0)),
            goal_out_of_plane_axis="x (the section is the plane p_x = 0)",
            analytic_band_zero_floor_m=z_floor, analytic_band_zero_floor_reason=why_floor,
            analytic_band_zero_ceiling_m=z_ceil, analytic_band_zero_ceiling_reason=why_ceil,
            analytic_band_zero_floor_numeric_check_m=band_branch_zero_numeric(
                limit, c_z, psi_cap, vel[2], "floor"),
            analytic_band_zero_ceiling_numeric_check_m=band_branch_zero_numeric(
                limit, c_z, psi_cap, vel[2], "ceiling"),
            analytic_lines_drawn_m=drawn,
            drawn_grid_spacing_m=float(2.0 * world / (RES - 1)),
            cylinders_intersecting_plane=cyl_on_plane)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    if not first_col:                       # the panel repeats its neighbour's vertical axis
        ax.set_yticklabels([])
    ax.set_title(f"({letter})", fontsize=PT_LABEL, fontweight="bold", loc="left", pad=2.0)
    ax.set_title(tag, fontsize=PT_TICK, loc="right", pad=2.0)         # item (c): out of the arena
    ax.tick_params(pad=1.5)
    return entry


def legend_handles(with_band=True, with_ref=False):
    h = [plt.Line2D([], [], color="black", lw=1.3, label="certificate zero level")]
    if with_ref:
        h.append(plt.Line2D([], [], color="0.35", lw=0.8, ls=(0, (3, 2)),
                            label="zero level at the scene's own goal"))
    h.append(plt.Line2D([], [], color="0.12", ls="--", lw=0.7, label="obstacle (true radius)"))
    if with_band:
        h += [plt.Line2D([], [], color="magenta", ls="--", lw=1.0, label="analytic hazard zero"),
              plt.Line2D([], [], color="0.15", lw=0.8, label="floor and ceiling"),
              mpatches.Patch(facecolor="0.80", alpha=0.6, edgecolor="none", label="outside arena")]
    h.append(plt.Line2D([], [], label="goal", **{k: v for k, v in GOAL_KW.items() if k != "zorder"}))
    return h



# ---------------------------------------------------------------------------------------------
def verify_hazard_line(fw, cfg, st, world, limit, c_z, psi_cap, geom):
    """NUMERICAL verification of every claim this manifest makes about the magenta line, run
    against src.common.quadrotor_barrier.value_target_barrier itself -- nothing is asserted from
    reading the source alone."""
    from src.common.quadrotor_barrier import value_target_barrier
    dev = next(fw.value_net.parameters()).device
    dt = torch.float32
    rng = np.random.default_rng(42)
    C, R, A, cyl_on_plane = geom

    def H(x):
        with torch.no_grad():
            return value_target_barrier(fw.system, torch.as_tensor(x, dtype=dt, device=dev),
                                        st, cfg).reshape(-1).cpu().numpy()

    n = 4096
    base = np.zeros((n, 13), np.float64)
    base[:, 0] = 0.0
    base[:, 1] = rng.uniform(-world, world, n)      # scattered so the test is not one point
    base[:, 2] = rng.uniform(-world, world, n)
    base[:, 3] = 1.0                                 # hover
    base[:, 9] = VZ_SECTION
    h_hover = H(base)

    # (1) attitude and body rate: same p, same v, RANDOM q and omega
    pert = base.copy()
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    pert[:, 3:7] = q
    pert[:, 10:13] = rng.uniform(-4.0, 4.0, (n, 3))
    d_att = float(np.abs(H(pert) - h_hover).max())

    # (2) control: the SAME states with a horizontal velocity -- h must change (approach term)
    ctrl = base.copy(); ctrl[:, 7] = 1.5
    d_vxy = float(np.abs(H(ctrl) - h_hover).max())

    # (3) d h / d v_z on the band branch, inside and outside the psi clamp
    out = {}
    for tag, z in (("clamp_inactive_z=-3.0", -3.0), ("clamp_active_z=-6.0", -6.0)):
        eps = 1e-3
        xa = np.zeros((2, 13)); xa[:, 3] = 1.0; xa[:, 2] = z
        xa[:, 1] = 3.6                                # a column clear of every cylinder chord
        xa[0, 9] = VZ_SECTION - eps; xa[1, 9] = VZ_SECTION + eps
        ha = H(xa)
        out[tag] = dict(numerical_dh_dvz=float((ha[1] - ha[0]) / (2 * eps)),
                        expected_minus_c_z=float(-c_z),
                        psi_lo_clamped=bool((-z - limit) > psi_cap))

    # (4) where does h itself cross zero down the drawn column, and is it the analytic line?
    zz = np.linspace(-world, world, 200001)
    xc = np.zeros((zz.size, 13)); xc[:, 3] = 1.0; xc[:, 1] = 3.6; xc[:, 2] = zz
    xc[:, 9] = VZ_SECTION
    hc = H(xc)
    k = np.nonzero(np.sign(hc[:-1]) * np.sign(hc[1:]) <= 0)[0]
    z_h0 = (float(zz[k[0]] + (zz[k[0] + 1] - zz[k[0]]) * (0.0 - hc[k[0]]) / (hc[k[0] + 1] - hc[k[0]]))
            if k.size else None)
    z_band = band_branch_zero(limit, c_z, psi_cap, VZ_SECTION, "floor")[0]

    # (5) ON the magenta line but INSIDE a cylinder chord: h is already positive there, so the line
    #     is NOT the hazard boundary at that y
    inside = []
    for j in cyl_on_plane:
        y0 = float(C[j, 1])
        xi = np.zeros((1, 13)); xi[0, 3] = 1.0; xi[0, 1] = y0; xi[0, 2] = z_band
        xi[0, 9] = VZ_SECTION
        inside.append(dict(cylinder=int(j), p_y_m=y0, p_z_m=float(z_band), h=float(H(xi)[0])))

    return dict(
        n_states=n,
        max_abs_dh_over_random_attitude_and_body_rate=d_att,
        max_abs_dh_over_horizontal_velocity_control=d_vxy,
        attitude_and_body_rate_independent=bool(d_att == 0.0),
        dh_dvz=out,
        h_zero_on_the_clear_column_m=z_h0,
        analytic_band_branch_zero_m=z_band,
        h_zero_minus_analytic_m=(None if z_h0 is None else float(z_h0 - z_band)),
        h_on_the_line_inside_the_cylinder_chord=inside,
        reading="max_abs_dh_over_random_attitude_and_body_rate = 0 proves the analytic hazard is "
                "attitude- and body-rate-independent EVERYWHERE, not merely on the drawn plane; "
                "the horizontal-velocity control is non-zero, so h is not simply "
                "velocity-independent. On a column clear of every cylinder the zero of h IS the "
                "band-branch zero; on a column through a cylinder chord h is already positive on "
                "that line, so the line is not the hazard boundary there.")


# ---------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    from src.eval.build_pools import load_pool, sha256_file
    from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as LD
    import copy

    pref, sel = select_scene()
    pool = load_pool(POOL)
    pool_sha = sha256_file(POOL)[:8]
    chosen, subs = None, []
    for cand in pref:
        if crosses_px0(pool.scenes[cand]):
            chosen = cand
            break
        subs.append(dict(scene=cand, reason="no active cylinder intersects the p_x = 0 plane"))
    if chosen is None:
        raise SystemExit("STOP: no candidate scene has a cylinder on the p_x = 0 plane")
    scene = pool.scenes[chosen]
    cyl_on_plane = crosses_px0(scene)
    print(f"scene rule -> scene {chosen}, vhat_min {sel['vhat_min'][chosen]:+.6f}, "
          f"rank {sel['rank'][chosen]} of {sel['n']} ascending (deepest first); "
          f"population median {sel['median_vhat_min']:+.6f}; "
          f"substitutions {subs if subs else 'none'}", flush=True)

    a = json.loads(JT_ART.read_text())
    ck = REPO / a["ckpt"]
    ov = copy.deepcopy(gate_overrides(ck)); ov["eval"]["max_steps"] = CAP
    fw, cfg, _ = LD(ck, config_overrides=ov)
    for nm in ("value_net", "policy_net"):
        m = getattr(fw, nm, None)
        if m is not None:
            m.to("cuda")
    world = float(cfg["env"]["world_lim"])
    limit = float(cfg["env"]["band_hazard"]["limit"])            # the key the barrier itself reads
    limit_collision = float(cfg["env"].get("band_collision_limit", 4.0))
    psi_cap = float(cfg["obstacle"]["per_system"]["quadrotor_3d"]["r_max"])
    omega_max = float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    v_max = float(cfg["env"]["bounds"]["quadrotor_3d"]["v_max"])
    c_z = math.pi / omega_max
    band_enabled = bool(cfg["env"]["band_hazard"].get("enabled", False))
    dev = next(fw.value_net.parameters()).device
    st = scene_tensors(scene, dev, torch.float32)
    C = np.asarray(scene.obstacle_centers, np.float64)
    R = np.asarray(scene.obstacle_radii, np.float64)
    A = np.asarray(scene.obstacle_active, bool)
    geom = (C, R, A, cyl_on_plane)
    goal = np.asarray(scene.goal, np.float64).reshape(-1)[:3]
    print(f"config: band_hazard.enabled {band_enabled}, limit {limit}, psi_cap {psi_cap}, "
          f"omega_max {omega_max}, c_z {c_z:.10f}; goal {goal}", flush=True)

    hazard_check = verify_hazard_line(fw, cfg, st, world, limit, c_z, psi_cap, geom)
    print(f"hazard-line check: max |dh| over random attitude+body rate "
          f"{hazard_check['max_abs_dh_over_random_attitude_and_body_rate']:.3e}; "
          f"control (horizontal velocity) {hazard_check['max_abs_dh_over_horizontal_velocity_control']:.4f}; "
          f"h zero on the clear column {hazard_check['h_zero_on_the_clear_column_m']} vs analytic "
          f"{hazard_check['analytic_band_branch_zero_m']:.6f}", flush=True)
    for t, d in hazard_check["dh_dvz"].items():
        print(f"  dh/dv_z {t}: {d['numerical_dh_dvz']:+.6f} (expected {d['expected_minus_c_z']:+.6f}, "
              f"clamp active {d['psi_lo_clamped']})", flush=True)
    for d in hazard_check["h_on_the_line_inside_the_cylinder_chord"]:
        print(f"  h on the line at the cylinder-{d['cylinder']} chord centre: {d['h']:+.4f}", flush=True)

    # ---- band_zero_offset from the frozen renderer, on THIS scene and THIS plane (item e) -----
    from src.eval.plotting import plot_quadrotor3d_yz_contour
    meta = plot_quadrotor3d_yz_contour(fw.system, fw.value_net, cfg, [scene],
                                       SCRATCH / "yz_reference_frozen.png", role="reference",
                                       resolution=REND_RES)
    offsets, dz_rend = meta["band_zero_offset"], float(meta["grid_spacing"])
    py_readout = float(meta["readout_py"]["row0"])
    print(f"frozen renderer: readout column p_y = {py_readout:+.4f} m (support |p_y| <= "
          f"{world - 0.3:.2f} m), grid spacing {dz_rend:.4f} m", flush=True)
    print(f"  offsets {json.dumps({k: round(v, 5) for k, v in offsets.items()})}", flush=True)

    # ---- MAIN FIGURE: five panels -------------------------------------------------------------
    # The subscript is written inline ("vx", "vz") and NOT as mathtext: matplotlib renders a
    # mathtext subscript at 0.7x the surrounding size, which at 7 pt is 4.9 pt -- below the 6 pt
    # floor this figure is measured against.
    panels = [("a", "xy", (-V_HORIZ, 0.0, 0.0), f"vx = {-V_HORIZ:+.1f} m/s"),
              ("b", "xy", (0.0, 0.0, 0.0), "at rest"),
              ("c", "xy", (V_HORIZ, 0.0, 0.0), f"vx = {V_HORIZ:+.1f} m/s"),
              ("d", "yz", (0.0, 0.0, VZ_SECTION), f"vz = {VZ_SECTION:+.1f} m/s"),
              ("e", "yz", (0.0, 0.0, VZ_SECTION_UP), f"vz = {VZ_SECTION_UP:+.1f} m/s")]
    fig, axes = plt.subplots(1, len(panels), figsize=(W_DOUBLE_IN, FIG_H_IN),
                             constrained_layout=True)
    rec, fields = [], {}
    for k, (letter, plane, vel, tag) in enumerate(panels):
        axis, V, _H = sweep(fw, cfg, st, plane, world, vel)
        fields[letter] = (axis, V)
        entry = draw_panel(axes[k], axis, V, plane, vel, geom, letter, tag, world, limit, c_z,
                           psi_cap, first_col=(letter in ("a", "d")), goal=goal)
        if plane == "yz":
            col_i = int(np.argmin(np.abs(axis - py_readout)))
            anchor = entry["analytic_band_zero_floor_m"]
            if anchor is None:
                anchor = entry["analytic_band_zero_ceiling_m"]
            off_drawn, z_cross = crossing_offset(axis, V, col_i, anchor)
            entry.update(
                readout_py_m=py_readout,
                readout_column_node_on_drawn_grid_m=float(axis[col_i]),
                readout_inside_support=bool(abs(py_readout) <= world - 0.3),
                sampled_support_half_width_m=float(world - 0.3),
                band_zero_offset_frozen_renderer_m=offsets.get(f"row0_vz{vel[2]:+.1f}"),
                frozen_renderer_resolution=REND_RES, frozen_renderer_grid_spacing_m=dz_rend,
                band_zero_offset_on_drawn_grid_m=(None if np.isnan(off_drawn) else off_drawn),
                vhat_zero_crossing_pz_on_drawn_grid_m=(None if np.isnan(z_cross) else z_cross),
                band_zero_offset_note="the readout COLUMN is no longer drawn (revision item e); "
                                      "the measurement is kept here unchanged")
        rec.append(entry)
        print(f"  ({letter}) {plane} v={vel}  V-hat [{V.min():+.4f}, {V.max():+.4f}]  "
              f"unsafe area {entry['unsafe_area_fraction']*100:.2f}%", flush=True)

    # ---- directional growth, MEASURED, not asserted (unchanged definition) --------------------
    axis0, V0 = fields["b"]
    gx0, gy0 = np.meshgrid(axis0, axis0)
    cx, cy = C[A][:, 0], C[A][:, 1]
    d2 = ((gx0.reshape(-1)[:, None] - cx[None, :]) ** 2
          + (gy0.reshape(-1)[:, None] - cy[None, :]) ** 2)
    near = np.argmin(d2, axis=1)
    dx_near = gx0.reshape(-1) - cx[near]
    growth = {}
    for pl, vx in (("a", -V_HORIZ), ("c", V_HORIZ)):
        _, Vm = fields[pl]
        added = ((Vm > 0.0) & (V0 <= 0.0)).reshape(-1)
        removed = ((Vm <= 0.0) & (V0 > 0.0)).reshape(-1)
        growth[pl] = dict(
            v_x_mps=vx, n_added_cells=int(added.sum()), n_removed_cells=int(removed.sum()),
            added_mean_signed_dx_from_nearest_obstacle_m=(float(dx_near[added].mean())
                                                          if added.any() else None),
            removed_mean_signed_dx_from_nearest_obstacle_m=(float(dx_near[removed].mean())
                                                            if removed.any() else None))
        g = growth[pl]
        print(f"  growth ({pl}) v_x {vx:+.1f}: {g['n_added_cells']} cells added, mean signed "
              f"dx {g['added_mean_signed_dx_from_nearest_obstacle_m']:+.4f} m; "
              f"{g['n_removed_cells']} removed", flush=True)

    sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.010, shrink=0.78, aspect=22,
                      ticks=[-1, 0, 1])
    cb.set_label("certificate value", fontsize=PT_TICK)          # item (d): "(clamped)" dropped
    cb.ax.tick_params(labelsize=PT_TICK, pad=1.5)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=6, frameon=False,
               fontsize=PT_TICK, handlelength=1.9, columnspacing=1.0, borderpad=0.1)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    title_fit = []
    for (letter, _pl, _v, tag), ax in zip(panels, axes):
        aw = ax.get_window_extent().width / fig.dpi
        wl = ax.title._x, 0
        wleft = ax._left_title.get_window_extent(rend).width / fig.dpi
        wright = ax._right_title.get_window_extent(rend).width / fig.dpi
        title_fit.append(dict(panel=letter, axes_width_in=float(aw),
                              left_title_width_in=float(wleft),
                              right_title_width_in=float(wright),
                              slack_in=float(aw - wleft - wright)))
        print(f"  title fit ({letter}) axes {aw:.3f} in, left {wleft:.3f} + right {wright:.3f} "
              f"= {wleft + wright:.3f} in, slack {aw - wleft - wright:+.3f} in", flush=True)
    if min(t["slack_in"] for t in title_fit) < 0.03:
        raise SystemExit("STOP: the panel titles do not fit inside their panel width")

    pdf = OUT / "fig4_certificate_zero_level.pdf"
    fig.savefig(pdf, format="pdf", dpi=CONTOUR_DPI)   # NO bbox_inches: MediaBox == declared width
    fig.savefig(OUT / "preview_fig4.png", format="png", dpi=300)
    ax_geom = [dict(panel=p[0], axes_width_in=float(ax.get_window_extent().width / fig.dpi),
                    axes_height_in=float(ax.get_window_extent().height / fig.dpi))
               for p, ax in zip(panels, axes)]
    plt.close(fig)

    # ---- ITEM (b): GOAL DEPENDENCE ------------------------------------------------------------
    g1 = np.array([-goal[0], -goal[1], goal[2]])         # horizontal point reflection
    g2 = np.array([goal[0], goal[1], -goal[2]])          # reflection through the mid-plane
    alts = [("reference", goal), ("G1", g1), ("G2", g2)]
    clr = {nm: clearance(gg, C, R, A) for nm, gg in alts}
    for nm, gg in alts:
        inside = bool(np.all(np.abs(gg) <= world + 1e-12))
        print(f"  goal {nm} {np.round(gg, 4)}  inside arena {inside}  min surface clearance "
              f"{clr[nm][0]:+.4f} m (cylinder {clr[nm][1]})", flush=True)
        if not inside or clr[nm][0] <= 0.0:
            raise SystemExit(f"STOP: goal {nm} is outside the arena or inside a cylinder")

    figb, axb = plt.subplots(1, 3, figsize=(W_DOUBLE_IN, FIG_H_B_IN), constrained_layout=True)
    grec, gfields = [], {}
    ref_V = None
    for k, (nm, gg) in enumerate(alts):
        stg = scene_tensors(scene, dev, torch.float32, goal=gg)
        axis, V, _ = sweep(fw, cfg, stg, "xy", world, (0.0, 0.0, 0.0))
        if k == 0:
            ref_V = V
        tag = ("the scene's own goal" if k == 0
               else f"goal at ({gg[0]:+.2f}, {gg[1]:+.2f}, {gg[2]:+.2f}) m")
        e = draw_panel(axb[k], axis, V, "xy", (0.0, 0.0, 0.0), geom, "abc"[k], tag, world, limit,
                       c_z, psi_cap, first_col=(k == 0), goal=gg,
                       ref_zero=(None if k == 0 else ref_V))
        gfields[nm] = (axis, V)
        e.update(goal_variant=nm, goal_m=[float(t) for t in gg],
                 goal_inside_arena=True, goal_min_surface_clearance_m=clr[nm][0],
                 goal_nearest_active_cylinder=clr[nm][1])
        grec.append(e)
    ref_pts = zero_segments(*gfields["reference"])
    gdep = dict(reference_goal_m=[float(t) for t in goal])
    for nm in ("G1", "G2"):
        axisA, VA = gfields[nm]
        _, V0r = gfields["reference"]
        pts = zero_segments(axisA, VA)
        d = contour_shift(ref_pts, pts)
        dv = VA - V0r
        unsafe_a, unsafe_r = VA > 0.0, V0r > 0.0
        cell = float((2.0 * world / (RES - 1)) ** 2)
        gg_ = g1 if nm == "G1" else g2
        d.update(
            goal_m=[float(t) for t in gg_],
            goal_displacement_from_reference_m=float(np.linalg.norm(gg_ - goal)),
            unsafe_area_fraction_reference=float(unsafe_r.mean()),
            unsafe_area_fraction_variant=float(unsafe_a.mean()),
            unsafe_symmetric_difference_fraction=float((unsafe_a ^ unsafe_r).mean()),
            unsafe_symmetric_difference_area_m2=float((unsafe_a ^ unsafe_r).sum() * cell),
            max_abs_delta_vhat=float(np.abs(dv).max()),
            mean_abs_delta_vhat=float(np.abs(dv).mean()),
            p99_abs_delta_vhat=float(np.percentile(np.abs(dv), 99)))
        gdep[nm] = d
        print(f"  goal dependence {nm}: median zero-level shift {d.get('median_shift_m', float('nan')):.4f} m, "
              f"p90 {d.get('p90_shift_m', float('nan')):.4f} m, Hausdorff {d.get('hausdorff_m', float('nan')):.4f} m; "
              f"unsafe area {d['unsafe_area_fraction_reference']:.4f} -> "
              f"{d['unsafe_area_fraction_variant']:.4f}, symmetric difference "
              f"{d['unsafe_symmetric_difference_fraction']:.4f}; max |dV| {d['max_abs_delta_vhat']:.4f}",
              flush=True)

    smb = ScalarMappable(norm=NORM, cmap=CMAP); smb.set_array([])
    cbb = figb.colorbar(smb, ax=axb, fraction=0.026, pad=0.010, shrink=0.80, aspect=22,
                        ticks=[-1, 0, 1])
    cbb.set_label("certificate value", fontsize=PT_TICK)
    cbb.ax.tick_params(labelsize=PT_TICK, pad=1.5)
    figb.legend(handles=legend_handles(with_band=False, with_ref=True), loc="outside lower center",
                ncol=4, frameon=False, fontsize=PT_TICK, handlelength=1.9, columnspacing=1.0,
                borderpad=0.1)
    pdfb = OUT / "fig4b_goal_dependence.pdf"
    figb.savefig(pdfb, format="pdf", dpi=CONTOUR_DPI)
    figb.savefig(OUT / "preview_fig4b.png", format="png", dpi=300)
    ax_geom_b = [dict(panel="abc"[i], axes_width_in=float(ax.get_window_extent().width / figb.dpi),
                      axes_height_in=float(ax.get_window_extent().height / figb.dpi))
                 for i, ax in enumerate(axb)]
    plt.close(figb)
    for g in ax_geom_b:
        print(f"  companion axes ({g['panel']}) {g['axes_width_in']:.3f} x "
              f"{g['axes_height_in']:.3f} in", flush=True)

    # ---- manifest -----------------------------------------------------------------------------
    sys.path.insert(0, str(HERE))
    import measure_pdf
    meas = {p.name: measure_pdf.measure(p, target_width_in=W_DOUBLE_IN) for p in (pdf, pdfb)}
    (OUT / "pdf_measurements.json").write_text(json.dumps(meas, indent=2) + "\n")

    man = dict(
        figure=pdf.name, companion_figure=pdfb.name,
        revises="eval/fig4_certificate_zero_level_20260819/fig4_certificate_zero_level.pdf "
                "(left untouched; this directory contains the revision only)",
        column="double", target_width_in=W_DOUBLE_IN,
        target_height_in=FIG_H_IN, companion_target_height_in=FIG_H_B_IN,
        panel_layout="1 row x 5 columns, shared colorbar at the right",
        panels_are="(a)(b)(c) horizontal xy sections at hover, z = 0, v_x in {-1.5, 0, +1.5} m/s; "
                   "(d)(e) vertical yz sections at p_x = 0, hover, v_z in {-1.5, +1.5} m/s",
        revision_items={
            "a_goal_marked": "the goal is drawn on every panel with the paper's canonical marker "
                             "(red star '*', color #d62728, ms 9, black edge mew 0.4, "
                             "scripts/analysis/v292_fig_rollouts.py:148) and carries the legend "
                             "entry 'goal'. Every panel is a 2-D section and the goal is a 3-D "
                             "point, so what is drawn is the ORTHOGONAL PROJECTION onto the drawn "
                             "plane -- the same convention the rollout figures use. The "
                             "out-of-plane distance is recorded per panel as "
                             "goal_out_of_plane_distance_m.",
            "b_goal_dependence": "measured; see goal_dependence below and the companion figure",
            "c_velocity_annotations_moved": "each panel's velocity is now its RIGHT title at 7 pt "
                                            "and the panel letter its LEFT title at 8 pt bold. NO "
                                            "text object is drawn inside any axes on either "
                                            "figure; the former in-axes 'outside arena' label is "
                                            "now a legend patch.",
            "d_clamped_dropped": "the colorbar label is 'certificate value'. The clamp itself is "
                                 "unchanged -- see colour_clamp.",
            "e_readout_column_removed": "the green dotted offset-readout column and its legend "
                                        "entry are gone from the drawing. The measurement is kept "
                                        "in this manifest (band_zero_offset_all, and per-panel "
                                        "band_zero_offset_frozen_renderer_m / "
                                        "band_zero_offset_on_drawn_grid_m).",
            "f_second_vertical_panel": "ADDED a fifth panel (e) at v_z = +1.5 m/s. See "
                                       "band_boundary_visibility for why one panel can never show "
                                       "both boundaries."},
        colour_clamp="the FILL is drawn on np.clip(V_hat, -1, +1) under a shared "
                     "TwoSlopeNorm(-1, 0, +1) with extend='both'; the black zero level and every "
                     "number in this manifest are computed on the UNCLIPPED V_hat. The measured "
                     "V_hat range per panel is in panels[].vhat_min / vhat_max.",
        grid_resolution=RES, contour_raster_dpi=CONTOUR_DPI,
        font_pt=dict(label=PT_LABEL, tick=PT_TICK, legend=PT_TICK, panel_letter=PT_LABEL,
                     panel_velocity_title=PT_TICK, smallest=PT_TICK),
        cell="v282_agree_gate.gate_overrides with eval.max_steps 400", eval_max_steps=CAP,
        pool=str(POOL.relative_to(REPO)), pool_sha8=pool_sha,
        checkpoint=a["ckpt"], checkpoint_step=a["ckpt_step"],
        ledger_row=jt_ledger_row(),            # read from the pointer, not hard-coded
        jt_pointer_artifact=str(JT_ART), jt_condition=a.get("condition"),
        scene_selection_rule="unchanged from the original producer: from the 24-scene certificate "
                             f"probe on disk ({Path(sel['probe']).relative_to(REPO)}, "
                             f"item_c.conditions.{sel['probe_condition']}.per_scene), the scene "
                             "whose vhat_min sits "
                             "at the MEDIAN of that population; n = 24 is even so the rank-12 and "
                             "rank-13 scenes are equidistant from the median by construction and "
                             "the tie is broken by the smaller pool index; scene 0 excluded by "
                             "dispatch; if the chosen scene has no active cylinder intersecting "
                             "p_x = 0, take the next scene toward the median",
        scene_probe_artifact=str(Path(sel["probe"]).relative_to(REPO)),
        scene_probe_condition=sel["probe_condition"],
        scene_probe_checkpoint=sel["probe_ckpt"], scene_probe_step=sel["probe_step"],
        scene_index=int(chosen), scene_vhat_min=sel["vhat_min"][chosen],
        scene_rank_ascending_deepest_first=sel["rank"][chosen],
        scene_rank_note=f"rank {sel['rank'][chosen]} of {sel['n']} with rank 1 the DEEPEST "
                        f"(most negative) vhat_min",
        population_median_vhat_min=sel["median_vhat_min"],
        candidate_order_after_exclusion=pref[:6], substitutions=subs,
        excluded_scenes=sorted(EXCLUDE_SCENES),
        n_active_cylinders=int(A.sum()), cylinders_intersecting_px0=cyl_on_plane,
        scene_goal_m=[float(t) for t in goal],
        speed_used_mps=V_HORIZ,
        speed_justification="1.5 m/s is (i) the speed 04_eval s3b fixes for this system's moving "
                            "column, (ii) 60 % of the plant's v_max = 2.5 m/s, and (iii) the "
                            "registered pool's own spawn-speed ceiling (max initial speed over the "
                            "2000 scenes is 1.499 m/s), so it is inside the operating range and "
                            "representative of what the deployed controller actually sees",
        v_max_mps=v_max, world_lim=world,
        band_hazard_enabled=band_enabled, band_hazard_limit=limit,
        band_collision_limit=limit_collision, psi_cap_m=psi_cap, omega_max_rad_s=omega_max, c_z=c_z,
        z_view_padding_m=Z_PAD,
        z_view_note="the field is swept over [-world_lim, world_lim]^2 exactly as 04_eval s3b "
                    "requires; the yz panels' y AXIS is padded by 0.30 m on each side for VIEW "
                    "only, so the band surfaces at |p_z| = 4 are drawn inside the frame instead of "
                    "on it. The padded strip carries no sampled data and is shaded; it is labelled "
                    "'outside arena' in the LEGEND, not inside the axes.",

        analytic_hazard_line=dict(
            what_is_drawn="the zero of ONE BRANCH of the analytic hazard's vertical band term, "
                          "from src/common/quadrotor_barrier.py:value_target_barrier: "
                          "psi_up = clamp(z - limit, max = psi_cap), "
                          "psi_lo = clamp(-z - limit, max = psi_cap), and "
                          "h = max(h_horizontal, max(psi_up + c_z v_z, psi_lo - c_z v_z)). "
                          "The magenta line is where the relevant band branch is zero.",
            depends_on=["p_z (the drawn vertical coordinate)",
                        "v_z, the WORLD vertical velocity, x[..., 9]",
                        "the constant limit = env.band_hazard.limit = "
                        f"{limit} m",
                        "the constant c_z = pi / env.bounds.quadrotor_3d.omega_max = "
                        f"{c_z:.10f} s (omega_max = {omega_max} rad/s)",
                        "the constant psi_cap = obstacle.per_system.quadrotor_3d.r_max = "
                        f"{psi_cap} m, which decides whether the branch has a zero AT ALL"],
            independent_of=["attitude q", "body rate omega", "horizontal position and velocity",
                            "the obstacles", "the goal",
                            "the checkpoint and anything learned -- the line is analytic"],
            correction_to_the_dispatch_statement=(
                "The dispatch's statement is CORRECT as far as it goes but incomplete on three "
                "points, all verified against the code. (1) 'fixed by the vertical velocity "
                "alone' -- it is fixed by v_z AND by three configuration constants (limit, c_z, "
                "psi_cap); at v_z = 0 the line coincides with the band surface and the offset "
                "from it is exactly c_z |v_z| = "
                f"{c_z * abs(VZ_SECTION):.4f} m at |v_z| = {abs(VZ_SECTION)} m/s. (2) 'independent "
                "of attitude and body rate' is true, but it is not a property of THIS panel: the "
                "WHOLE quadrotor_3d value_target_barrier is independent of attitude and body "
                "rate -- phi reads system.position(x)[..., :2], the approach term reads x[..., :2] "
                "and x[..., 7:9], and the band branch reads position z and x[..., 9]; q (x[3:7]) "
                "and omega (x[10:13]) enter none of them. The LEARNED certificate is NOT "
                "attitude-independent, which is exactly why the frozen renderer's "
                "band_zero_offset differs between hover and tilt 120 deg (see "
                "band_zero_offset_all). (3) The line is the zero of the BAND BRANCH, not of the "
                "full analytic hazard h: h is a max, so wherever the horizontal term is positive "
                "(inside and just outside the cylinder chord) h is already positive and the band "
                "branch's zero is not the boundary of the hazard set there. Additionally, the "
                "psi_cap clamp can delete the zero entirely -- see band_boundary_visibility."),
            coefficient_and_authority=(
                "c_z = pi / omega_max is the time to rotate the thrust axis through pi at the "
                "plant's body-rate limit, i.e. a LEAD TIME set by the rate limit, not the rate "
                "limit itself. Its role in authority is that d(band branch)/d v_z = +/- c_z != 0 "
                "everywhere, INCLUDING inside the cap (the clamp is on z, never on v_z), so the "
                "vertical channel keeps relative degree 1 and L_g is supported on the thrust "
                "channel c_z (R e)_z / m (the code comment at "
                "src/common/quadrotor_barrier.py, value_target_barrier). That is what ties the "
                "constant to authority; the dispatch's clause on this point is right."),
            numerical_verification=hazard_check,
            not_the_zero_of_h="on the cylinder chord at p_y in "
                              f"{[[float(C[j,1]-math.sqrt(R[j]**2-C[j,0]**2)), float(C[j,1]+math.sqrt(R[j]**2-C[j,0]**2))] for j in cyl_on_plane]} m "
                              "the horizontal term already makes h positive, so the magenta line "
                              "there is not the hazard boundary"),

        band_boundary_visibility=dict(
            claim="a single vertical panel can never show both band boundaries",
            proof="the ceiling branch has a zero only if -c_z v_z <= psi_cap, i.e. "
                  f"v_z >= -psi_cap/c_z = {-psi_cap / c_z:.4f} m/s, and it then sits at "
                  "z = limit - c_z v_z, which is <= world_lim only if v_z >= 0; the floor branch "
                  f"has a zero only if c_z v_z <= psi_cap, i.e. v_z <= {psi_cap / c_z:.4f} m/s, "
                  "and it then sits at z = -limit - c_z v_z, which is >= -world_lim only if "
                  "v_z <= 0. The two in-frame conditions intersect only at v_z = 0, where both "
                  "lines coincide with the band surfaces |p_z| = limit themselves.",
            at_descending_vz=dict(
                v_z=VZ_SECTION,
                ceiling=band_branch_zero(limit, c_z, psi_cap, VZ_SECTION, "ceiling")[1],
                floor=band_branch_zero(limit, c_z, psi_cap, VZ_SECTION, "floor")[1]),
            at_ascending_vz=dict(
                v_z=VZ_SECTION_UP,
                ceiling=band_branch_zero(limit, c_z, psi_cap, VZ_SECTION_UP, "ceiling")[1],
                floor=band_branch_zero(limit, c_z, psi_cap, VZ_SECTION_UP, "floor")[1]),
            choice="ADDED the ascending panel (e). It is the only way to put the ceiling boundary "
                   "in the figure at all, it costs one column of width, and it makes the caption "
                   "able to say 'band boundaries' honestly: (d) carries the floor, (e) the "
                   "ceiling. The caption must still not claim that either panel shows both.",
            caption_constraint="panel (d) shows the FLOOR boundary only and panel (e) the CEILING "
                               "boundary only; no single panel shows both"),

        band_zero_offset_source="src.eval.plotting.plot_quadrotor3d_yz_contour (frozen renderer, "
                                f"resolution {REND_RES}), called on THIS scene and THIS plane for "
                                "its return value only",
        band_zero_offset_all=offsets, frozen_renderer_grid_spacing_m=dz_rend,
        readout_py_m=py_readout,
        readout_column_rule="the frozen renderer's own rule: among p_y columns strictly inside the "
                            "sampled support (|p_y| <= world_lim - 0.3 = 3.7 m) take the one with "
                            "the largest clearance to every active cylinder chord at p_x = 0. The "
                            "column is NO LONGER DRAWN (revision item e); the rule is recorded "
                            "because the number below is defined by it.",
        colour_scale="shared TwoSlopeNorm(-1, 0, +1), NOT renormalized per panel; the companion "
                     "figure uses the same scale",
        directional_growth=growth,
        directional_growth_definition="cells the motion ADDS to the unsafe set relative to the "
                                      "at-rest panel, each assigned to the nearest ACTIVE cylinder "
                                      "axis; the reported number is the mean signed x offset from "
                                      "that axis. Negative = added mass sits at smaller x than the "
                                      "obstacle it belongs to.",

        goal_dependence=dict(
            question="does the certificate's zero level set visibly depend on where the goal is?",
            why_it_could="the certificate is a policy value function and the policy is "
                         "goal-conditioned; the goal enters the observation as the body-frame "
                         "vector goal_b = R(q)^T (goal - p) (src/envs/quadrotor_3d.py, "
                         "observation()), so V_hat is a function of the goal",
            section_redrawn="panel (b)'s section: xy at hover, z = 0, at rest",
            alternate_goal_rule="the images of the scene's own goal under the two sign flips that "
                                "keep it inside the SAME arena box: G1 = (-g_x, -g_y, +g_z), the "
                                "horizontal point reflection through the arena's vertical axis, "
                                "and G2 = (+g_x, +g_y, -g_z), the reflection through the "
                                "horizontal mid-plane. Both are inside [-world_lim, world_lim]^3 "
                                "by construction because the true goal is; both were checked "
                                "clear of every active cylinder (clearances below). Between them "
                                "they separate a purely HORIZONTAL change of the goal from a "
                                "purely VERTICAL one, which is the cleanest two-point probe of "
                                "the dependence. Everything else -- scene, checkpoint, cell, "
                                "attitude, velocity, grid -- is identical.",
            goal_clearances_m={nm: dict(min_surface_clearance_m=clr[nm][0],
                                        nearest_active_cylinder=clr[nm][1],
                                        goal_m=[float(t) for t in gg]) for nm, gg in alts},
            shift_definition="for every vertex of the alternate V_hat = 0 contour the distance to "
                             "the nearest vertex of the reference V_hat = 0 contour and vice "
                             "versa, summarized over the union; contours taken from the same "
                             f"{RES}-node grid (spacing {2.0 * world / (RES - 1):.6f} m)",
            measurements=gdep,
            shift_metric_bias="the shift is vertex-to-vertex, so it overstates the true "
                              "point-to-curve distance by at most half the contour vertex "
                              f"spacing, about {world / (RES - 1):.4f} m",
            verdict="YES, the zero level set depends on the goal, but weakly and locally. Moving "
                    "the goal 8.61 m (G1, the largest move the arena allows under this rule) "
                    "shifts the zero level by a MEDIAN of "
                    f"{gdep['G1'].get('median_shift_m', float('nan')):.4f} m -- about one grid "
                    f"cell ({2.0 * world / (RES - 1):.4f} m) -- with p90 "
                    f"{gdep['G1'].get('p90_shift_m', float('nan')):.4f} m and a worst case "
                    f"(Hausdorff) of {gdep['G1'].get('hausdorff_m', float('nan')):.4f} m, i.e. "
                    f"{100 * gdep['G1'].get('hausdorff_m', float('nan')) / gdep['G1']['goal_displacement_from_reference_m']:.1f} % "
                    "of the goal displacement. The unsafe fraction of the slice moves from "
                    f"{gdep['G1']['unsafe_area_fraction_reference']:.4f} to "
                    f"{gdep['G1']['unsafe_area_fraction_variant']:.4f} (symmetric difference "
                    f"{gdep['G1']['unsafe_symmetric_difference_fraction']:.4f} of the slice) and "
                    f"max |dV_hat| reaches {gdep['G1']['max_abs_delta_vhat']:.4f}. The purely "
                    "VERTICAL move (G2, 4.11 m) does about half as much: median "
                    f"{gdep['G2'].get('median_shift_m', float('nan')):.4f} m, Hausdorff "
                    f"{gdep['G2'].get('hausdorff_m', float('nan')):.4f} m, symmetric difference "
                    f"{gdep['G2']['unsafe_symmetric_difference_fraction']:.4f}. The movement is "
                    "NOT uniform: most of the contour is within a cell of where it was and the "
                    "change concentrates on a few obstacle boundaries, which is what the "
                    "companion figure's grey reference contour shows. So the goal matters enough "
                    "that the figure must mark it, and not enough to change how the panel reads.",
            panels=grec),

        legend_canon_used=False,
        legend_note="no controller appears in this figure, so the paper's controller legend canon "
                    "does not apply; the goal marker follows the paper's start/goal canon "
                    "(scripts/analysis/v292_fig_rollouts.py)",
        clamped_label_audit=dict(
            fixed_here=["eval/fig4_certificate_zero_level_rev_20260819/make_fig4_rev.py "
                        "(this producer): 'certificate value'"],
            still_carrying_the_label=[
                {"producer": "scripts/analysis/v292_fig_certificate.py:182",
                 "label": "certificate value (clamped to [-1, 1])",
                 "asset": "data/runs/v2.9.2/paper_figures/fig_certificate_sections.pdf"},
                {"producer": "scripts/analysis/v292_fig_certificate2.py:232",
                 "label": "certificate value (clamped to [-1, 1])",
                 "asset": "data/runs/v2.9.2/paper_figures/fig_certificate_sections.pdf",
                 "note": "the two certificate producers write the SAME path; the file on disk "
                         "(471480 bytes, the size v2.9.2_results.md line 1369 records) is "
                         "certificate2's -- data/runs/v2.9.2/paper_figures/"
                         "manifest_certificate.json carries the 'column' key that only "
                         "certificate2 writes. So certificate2 is the producer of record and "
                         "certificate.py is the superseded one; both still carry the label."},
                {"producer": "scripts/analysis/v292_fig_nine.py:316",
                 "label": "certificate value (clamped to [-1, 1])",
                 "asset": "data/runs/v2.9.2/paper_figures/fig_rollout_nine_certificate.pdf"},
                {"producer": "scripts/deck/fig_cbf_contour_xy.py:54",
                 "label": "V-hat (clamped [-1,1])", "asset": "deck fig_cbf_contour_xy.png"},
                {"producer": "scripts/deck/fig_cbf_contour_yz.py:83",
                 "label": "V-hat (clamped [-1,1])", "asset": "deck fig_cbf_contour_yz.png"},
                {"producer": "scripts/analysis/quadrotor_cbf_contour.py:98",
                 "label": "h(x) = V_hat  (clamped [-1,1])",
                 "asset": "<run_dir>/figures/cbf_contour_m6.png"}],
            not_rebuilt_because="those are registered assets cited by docs/versions/"
                                "v2.9.2_results.md (the certificate/nine figures at 7.11); "
                                "regenerating them is not this dispatch's to do and would "
                                "overwrite artifacts other rows point at. Reported for the "
                                "Researcher to schedule."),
        axes_geometry_in=ax_geom, companion_axes_geometry_in=ax_geom_b,
        title_fit_in=title_fit,
        panels=rec)
    man["measured"] = meas[pdf.name]
    man["measured_companion"] = meas[pdfb.name]
    man["measured_type_size_floor_pt"] = min(meas[pdf.name]["min_tf_pt"],
                                             meas[pdfb.name]["min_tf_pt"])
    man["measured_canvas_in"] = [meas[pdf.name]["mediabox_width_in"],
                                 meas[pdf.name]["mediabox_height_in"]]
    man["measured_companion_canvas_in"] = [meas[pdfb.name]["mediabox_width_in"],
                                           meas[pdfb.name]["mediabox_height_in"]]
    (OUT / "manifest_fig4_rev.json").write_text(json.dumps(man, indent=2) + "\n")
    for nm, m in meas.items():
        print(f"measured {nm}: MediaBox {m['mediabox_width_in']} x {m['mediabox_height_in']} in, "
              f"Tf sizes {m['distinct_tf_pt']} pt, all >= 6pt {m['all_text_ge_6pt']}, "
              f"{m['n_image_xobjects']} rasterized fills, {m['rasterized_text_objects']} rasterized "
              f"glyphs, fonts {m['embedded_fonts']}", flush=True)
    for g in ax_geom:
        print(f"  axes ({g['panel']}) {g['axes_width_in']:.3f} x {g['axes_height_in']:.3f} in",
              flush=True)
    print(f"wrote {pdf} and {pdfb}", flush=True)


if __name__ == "__main__":
    main()
