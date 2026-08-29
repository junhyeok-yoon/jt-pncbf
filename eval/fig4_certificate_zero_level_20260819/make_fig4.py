"""Paper figure 4 — the deployed certificate's zero level set on a deployed scene.

Checkpoint L328 (jointly trained pair) on quadrotor_3d, scored/loaded on the registered cell
(`v282_agree_gate.gate_overrides`, eval.max_steps 400), the system's registered pool.

SCENE SELECTION IS A RULE, FIXED BEFORE ANYTHING IS DRAWN.
  From the 24-scene certificate probe already on disk
  (data/runs/v2.9.2/cert_probes/cert_probes.json, item_c.conditions.jt_L328.per_scene), take the
  scene whose `vhat_min` sits at the MEDIAN of that population. n = 24 is even, so the median value
  falls exactly between the rank-12 and rank-13 scenes (ascending, deepest first) and BOTH are
  equidistant from it by construction; the tie is broken by the SMALLER pool index. Scene 0 is
  excluded by dispatch (it is among the shallowest of the 24 and is not representative).
  If the selected scene carries no active cylinder intersecting the p_x = 0 plane that item 1b
  needs, the next scene toward the median is taken and the substitution is reported.

PANELS.
  (a)(b)(c)  horizontal (xy) sections at hover attitude, z = 0, at three horizontal velocities
             along the fixed x axis: v_x = -V, 0, +V.  V is stated in the manifest.
  (d)        vertical (yz) section at p_x = 0 -- the plane 04_eval s3b fixes -- sweeping
             (p_y, p_z) over [-world_lim, world_lim]^2, hover attitude, v_z = -1.5 m/s, with the
             floor and ceiling band boundaries |p_z| = band_collision_limit overlaid and the
             analytic hazard's own zero drawn as a distinct dashed line.

band_zero_offset is taken from the FROZEN renderer src.eval.plotting.plot_quadrotor3d_yz_contour,
called on THE SAME SCENE and THE SAME PLANE that panel (d) draws, purely for its returned dict, so
the reported number is the renderer's and not a second implementation. The renderer's own readout
column rule already forces the column strictly inside the sampled support (|p_y| <= world - 0.3).
The same crossing is ALSO evaluated on the drawn 241-node grid at that same column and reported
beside it, so the reader can see the number is not a resolution artifact.

07_tex_deck I3: no version stamp, run-id, pool name, scene index, checkpoint digest or ledger row
label is drawn. Panels are labelled (a)..(d) and nothing else.

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
SCRATCH = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/"
               "31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad/fig4")
JT_ART = REPO / "data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json"      # L328's checkpoint
POOL = REPO / "data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl"
PROBE = REPO / "data/runs/v2.9.2/cert_probes/cert_probes.json"
EXCLUDE_SCENES = {0}

CAP = 400                       # eval.max_steps on the registered cell
W_DOUBLE_IN = 7.00              # double-column target width, inches
FIG_H_IN = 2.38
PT_LABEL, PT_TICK = 8.0, 7.0
CONTOUR_DPI = 600               # the ONLY rasterized layer is the filled contour
RES = 241                       # grid nodes per axis on the drawn sections
REND_RES = 61                   # the frozen renderer's fixed resolution (do not change: comparability)
V_HORIZ = 1.5                   # horizontal speed for the moving columns, m/s
VZ_SECTION = -1.5               # descent rate on the vertical section, m/s
Z_PAD = 0.30                    # view-only padding so the band boundaries are drawn inside the frame
CMAP, NORM = "coolwarm", TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": PT_TICK, "axes.labelsize": PT_LABEL, "axes.titlesize": PT_LABEL,
    "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK, "legend.fontsize": PT_LABEL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.transparent": False,
})


# ---------------------------------------------------------------------------------------------
def select_scene():
    """The median-vhat_min rule, applied to the probe on disk. Returns (index, detail)."""
    probe = json.loads(PROBE.read_text())
    per = probe["item_c"]["conditions"]["jt_L328"]["per_scene"]
    idx = np.array([int(r["scene"]) for r in per])
    vmin = np.array([float(r["vhat_min"]) for r in per])
    order = np.argsort(vmin, kind="stable")                    # ascending = deepest first
    rank = {int(idx[o]): r + 1 for r, o in enumerate(order)}
    median = float(np.median(vmin))
    keep = np.array([int(i) not in EXCLUDE_SCENES for i in idx])
    d = np.abs(vmin - median)
    # order of preference: nearest to the median, then smaller pool index
    pref = sorted([int(i) for i in idx[keep]],
                  key=lambda i: (round(float(d[idx == i][0]), 12), i))
    return pref, dict(median_vhat_min=median, rank=rank,
                      vhat_min={int(i): float(v) for i, v in zip(idx, vmin)},
                      n=len(per), excluded=sorted(EXCLUDE_SCENES))


def crosses_px0(scene):
    C = np.asarray(scene.obstacle_centers, np.float64)
    R = np.asarray(scene.obstacle_radii, np.float64)
    A = np.asarray(scene.obstacle_active, bool)
    return [int(j) for j in np.nonzero(A)[0] if abs(C[j, 0]) < R[j]]


def scene_tensors(scene, dev, dt):
    return SimpleNamespace(
        obstacle_centers=torch.tensor(np.asarray(scene.obstacle_centers), dtype=dt, device=dev),
        obstacle_radii=torch.tensor(np.asarray(scene.obstacle_radii), dtype=dt, device=dev),
        obstacle_active=torch.tensor(np.asarray(scene.obstacle_active), dtype=torch.bool, device=dev),
        goal=torch.tensor(np.asarray(scene.goal), dtype=dt, device=dev))


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
def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    from src.eval.build_pools import load_pool, sha256_file
    from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as LD
    import copy

    pref, sel = select_scene()
    pool = load_pool(POOL)
    pool_sha = sha256_file(POOL)[:8]
    chosen, subs = None, []
    for cand in pref:
        hit = crosses_px0(pool.scenes[cand])
        if hit:
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
    limit = float(cfg["env"].get("band_collision_limit", 4.0))
    v_max = float(cfg["env"]["bounds"]["quadrotor_3d"]["v_max"])
    c_z = math.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    dev = next(fw.value_net.parameters()).device
    st = scene_tensors(scene, dev, torch.float32)
    C = np.asarray(scene.obstacle_centers, np.float64)
    R = np.asarray(scene.obstacle_radii, np.float64)
    A = np.asarray(scene.obstacle_active, bool)

    # ---- band_zero_offset from the frozen renderer, on THIS scene and THIS plane --------------
    from src.eval.plotting import plot_quadrotor3d_yz_contour
    meta = plot_quadrotor3d_yz_contour(fw.system, fw.value_net, cfg, [scene],
                                       SCRATCH / "yz_reference_frozen.png", role="reference",
                                       resolution=REND_RES)
    offsets, dz_rend = meta["band_zero_offset"], float(meta["grid_spacing"])
    py_readout = float(meta["readout_py"]["row0"])
    print(f"frozen renderer: readout column p_y = {py_readout:+.4f} m (support |p_y| <= "
          f"{world - 0.3:.2f} m), grid spacing {dz_rend:.4f} m", flush=True)
    print(f"  offsets {json.dumps({k: round(v, 5) for k, v in offsets.items()})}", flush=True)

    # ---- the four panels ---------------------------------------------------------------------
    panels = [("xy", (-V_HORIZ, 0.0, 0.0)), ("xy", (0.0, 0.0, 0.0)), ("xy", (V_HORIZ, 0.0, 0.0)),
              ("yz", (0.0, 0.0, VZ_SECTION))]
    fig, axes = plt.subplots(1, 4, figsize=(W_DOUBLE_IN, FIG_H_IN), constrained_layout=True)
    rec = []
    fields = {}
    for k, (plane, vel) in enumerate(panels):
        ax = axes[k]
        axis, V, H = sweep(fw, cfg, st, plane, world, vel)
        ax.set_rasterization_zorder(0)          # ONLY the fill (zorder -10) is rasterized
        ax.contourf(axis, axis, np.clip(V, -1, 1), levels=np.linspace(-1, 1, 41), cmap=CMAP,
                    norm=NORM, extend="both", zorder=-10, alpha=0.75)
        if V.min() <= 0 <= V.max():
            ax.contour(axis, axis, V, levels=[0.0], colors="black", linewidths=1.3, zorder=4)
        fields["abcd"[k]] = (axis, V)
        unsafe = V > 0.0
        gx, gy_ = np.meshgrid(axis, axis)
        entry = dict(panel="abcd"[k], section=plane, velocity=[float(t) for t in vel],
                     attitude="hover (q = [1, 0, 0, 0])",
                     vhat_min=float(V.min()), vhat_max=float(V.max()),
                     unsafe_area_fraction=float(unsafe.mean()),
                     unsafe_centroid_horizontal_m=(float(gx[unsafe].mean()) if unsafe.any() else None),
                     unsafe_centroid_vertical_m=(float(gy_[unsafe].mean()) if unsafe.any() else None))
        if plane == "xy":
            for j in np.nonzero(A)[0]:
                ax.add_patch(mpatches.Circle((C[j, 0], C[j, 1]), R[j], facecolor="none",
                                             edgecolor="0.12", lw=0.7, ls="--", zorder=3))
            ax.set_xlabel("x position (m)")
            if k == 0:
                ax.set_ylabel("y position (m)")
            ax.set_xlim(-world, world); ax.set_ylim(-world, world)
            tag = "at rest" if vel[0] == 0.0 else f"v = {vel[0]:+.1f} m/s along x"
            entry["n_active_cylinders_drawn"] = int(A.sum())
        else:
            for j in cyl_on_plane:              # infinite vertical cylinders read as vertical bands
                hh = float(np.sqrt(R[j] ** 2 - C[j, 0] ** 2))
                ax.axvspan(C[j, 1] - hh, C[j, 1] + hh, color="0.45", alpha=0.18, lw=0, zorder=2)
            for lo, hi in ((-world - Z_PAD, -limit), (limit, world + Z_PAD)):
                ax.axhspan(lo, hi, facecolor="0.80", alpha=0.6, lw=0, zorder=5)
            ax.axhline(limit, color="0.15", lw=0.8, zorder=6)
            ax.axhline(-limit, color="0.15", lw=0.8, zorder=6)
            analytic_floor = -limit - c_z * vel[2]
            analytic_ceil = limit - c_z * vel[2]
            for pz0 in (analytic_floor, analytic_ceil):
                if -world <= pz0 <= world:
                    ax.axhline(pz0, color="magenta", ls="--", lw=1.0, zorder=6)
            ax.axvline(py_readout, color="#00a000", ls=":", lw=0.9, zorder=6)
            ax.set_xlabel("y position (m)"); ax.set_ylabel("z position (m)")
            ax.set_xlim(-world, world); ax.set_ylim(-world - Z_PAD, world + Z_PAD)
            ax.text(0.020, 0.978, "outside arena", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="0.15", zorder=7)
            tag = f"v = {vel[2]:+.1f} m/s along z"
            col_i = int(np.argmin(np.abs(axis - py_readout)))
            off_drawn, z_cross = crossing_offset(axis, V, col_i, analytic_floor)
            entry.update(
                plane_px=0.0, readout_py_m=py_readout,
                readout_column_node_on_drawn_grid_m=float(axis[col_i]),
                readout_inside_support=bool(abs(py_readout) <= world - 0.3),
                sampled_support_half_width_m=float(world - 0.3),
                analytic_hstar_zero_floor_m=float(analytic_floor),
                analytic_hstar_zero_ceiling_m=float(analytic_ceil),
                band_zero_offset_frozen_renderer_m=offsets.get(f"row0_vz{vel[2]:+.1f}"),
                frozen_renderer_resolution=REND_RES, frozen_renderer_grid_spacing_m=dz_rend,
                band_zero_offset_on_drawn_grid_m=(None if np.isnan(off_drawn) else off_drawn),
                vhat_zero_crossing_pz_on_drawn_grid_m=(None if np.isnan(z_cross) else z_cross),
                drawn_grid_spacing_m=float(2.0 * world / (RES - 1)),
                cylinders_intersecting_plane=cyl_on_plane)
        ax.set_aspect("equal")
        ax.set_title(f"({'abcd'[k]})", fontsize=PT_LABEL, fontweight="bold", loc="left", pad=2.0)
        ax.text(0.5, 0.022, tag, transform=ax.transAxes, va="bottom", ha="center",
                fontsize=6.5, zorder=8,
                bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none", alpha=0.82))
        ax.tick_params(pad=1.5)
        rec.append(entry)
        print(f"  ({entry['panel']}) {plane} v={vel}  V-hat [{V.min():+.4f}, {V.max():+.4f}]  "
              f"unsafe area {entry['unsafe_area_fraction']*100:.2f}%  "
              f"unsafe centroid ({entry['unsafe_centroid_horizontal_m']:+.4f}, "
              f"{entry['unsafe_centroid_vertical_m']:+.4f}) m", flush=True)

    # ---- directional growth, MEASURED, not asserted ------------------------------------------
    # For each moving xy panel take the cells the motion ADDS to the unsafe set (unsafe when
    # moving, safe at rest), assign each to the nearest ACTIVE cylinder axis, and average the
    # signed x offset from that axis. Negative = the added mass sits UPSTREAM of the obstacles
    # (the direction the vehicle is coming from); positive = downstream.
    axis0, V0 = fields["b"]
    gx0, _gy0 = np.meshgrid(axis0, axis0)
    cx = C[A][:, 0]; cy = C[A][:, 1]
    _gy0f = _gy0.reshape(-1)
    d2 = (gx0.reshape(-1)[:, None] - cx[None, :]) ** 2 + (_gy0f[:, None] - cy[None, :]) ** 2
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
              f"dx from nearest obstacle {g['added_mean_signed_dx_from_nearest_obstacle_m']:+.4f} m; "
              f"{g['n_removed_cells']} removed", flush=True)

    sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.020, pad=0.010, shrink=0.80, aspect=22,
                      ticks=[-1, 0, 1])
    cb.set_label("certificate value (clamped)", fontsize=PT_TICK)
    cb.ax.tick_params(labelsize=PT_TICK, pad=1.5)
    handles = [plt.Line2D([], [], color="black", lw=1.3, label="certificate zero level"),
               plt.Line2D([], [], color="0.12", ls="--", lw=0.7, label="obstacle (true radius)"),
               plt.Line2D([], [], color="magenta", ls="--", lw=1.0, label="analytic hazard zero"),
               plt.Line2D([], [], color="0.15", lw=0.8, label="floor and ceiling"),
               plt.Line2D([], [], color="#00a000", ls=":", lw=0.9, label="offset readout column")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               fontsize=PT_TICK, handlelength=1.9, columnspacing=1.1, borderpad=0.1)

    pdf = HERE / "fig4_certificate_zero_level.pdf"
    fig.savefig(pdf, format="pdf", dpi=CONTOUR_DPI)     # NO bbox_inches: MediaBox == declared width
    fig.savefig(HERE / "preview_fig4.png", format="png", dpi=300)
    plt.close(fig)

    man = dict(
        figure=pdf.name, column="double", target_width_in=W_DOUBLE_IN,
        target_height_in=FIG_H_IN, panel_layout="1 row x 4 columns, shared colorbar at the right",
        panels_are="(a)(b)(c) horizontal xy sections at hover, z = 0, v_x in {-1.5, 0, +1.5} m/s; "
                   "(d) vertical yz section at p_x = 0, hover, v_z = -1.5 m/s",
        grid_resolution=RES, contour_raster_dpi=CONTOUR_DPI,
        font_pt=dict(label=PT_LABEL, tick=PT_TICK, legend=PT_TICK, panel_letter=PT_LABEL,
                     panel_tag=6.5, smallest=6.5),
        cell="v282_agree_gate.gate_overrides with eval.max_steps 400", eval_max_steps=CAP,
        pool=str(POOL.relative_to(REPO)), pool_sha8=pool_sha,
        checkpoint=a["ckpt"], checkpoint_step=a["ckpt_step"], ledger_row="L328",
        scene_selection_rule="from the 24-scene certificate probe on disk "
                             "(data/runs/v2.9.2/cert_probes/cert_probes.json, "
                             "item_c.conditions.jt_L328.per_scene), the scene whose vhat_min sits "
                             "at the MEDIAN of that population; n = 24 is even so the rank-12 and "
                             "rank-13 scenes are equidistant from the median by construction and "
                             "the tie is broken by the smaller pool index; scene 0 excluded by "
                             "dispatch; if the chosen scene has no active cylinder intersecting "
                             "p_x = 0, take the next scene toward the median",
        scene_index=int(chosen),
        scene_vhat_min=sel["vhat_min"][chosen],
        scene_rank_ascending_deepest_first=sel["rank"][chosen],
        scene_rank_note=f"rank {sel['rank'][chosen]} of {sel['n']} with rank 1 the DEEPEST "
                        f"(most negative) vhat_min",
        population_median_vhat_min=sel["median_vhat_min"],
        candidate_order_after_exclusion=pref[:6], substitutions=subs,
        excluded_scenes=sorted(EXCLUDE_SCENES),
        n_active_cylinders=int(A.sum()), cylinders_intersecting_px0=cyl_on_plane,
        speed_used_mps=V_HORIZ,
        speed_justification="1.5 m/s is (i) the speed 04_eval s3b fixes for this system's moving "
                            "column, (ii) 60 % of the plant's v_max = 2.5 m/s, and (iii) the "
                            "registered pool's own spawn-speed ceiling (max initial speed over the "
                            "2000 scenes is 1.499 m/s), so it is inside the operating range and "
                            "representative of what the deployed controller actually sees",
        v_max_mps=v_max, world_lim=world, band_collision_limit=limit, c_z=c_z,
        z_view_padding_m=Z_PAD,
        z_view_note="the field is swept over [-world_lim, world_lim]^2 exactly as 04_eval s3b "
                    "requires; the yz panel's y AXIS is padded by 0.45 m for VIEW only, so the "
                    "band boundaries at |p_z| = 4 are drawn inside the frame instead of on it. "
                    "The padded strip carries no sampled data and is shaded as outside the arena.",
        band_zero_offset_source="src.eval.plotting.plot_quadrotor3d_yz_contour (frozen renderer, "
                                f"resolution {REND_RES}), called on THIS scene and THIS plane for "
                                "its return value only",
        band_zero_offset_all=offsets, frozen_renderer_grid_spacing_m=dz_rend,
        readout_py_m=py_readout,
        readout_column_rule="the frozen renderer's own rule: among p_y columns strictly inside the "
                            "sampled support (|p_y| <= world_lim - 0.3 = 3.7 m) take the one with "
                            "the largest clearance to every active cylinder chord at p_x = 0",
        colour_scale="shared TwoSlopeNorm(-1, 0, +1), NOT renormalized per panel",
        directional_growth=growth,
        directional_growth_definition="cells the motion ADDS to the unsafe set relative to the "
                                      "at-rest panel, each assigned to the nearest ACTIVE cylinder "
                                      "axis; the reported number is the mean signed x offset from "
                                      "that axis. Negative = added mass sits at smaller x than the "
                                      "obstacle it belongs to.",
        legend_canon_used=False,
        legend_note="no controller appears in this figure, so the paper's controller legend canon "
                    "does not apply here",
        panels=rec)
    sys.path.insert(0, str(HERE))
    import measure_pdf
    man["measured"] = measure_pdf.measure(pdf, target_width_in=W_DOUBLE_IN)
    man["measured_type_size_floor_pt"] = man["measured"]["min_tf_pt"]
    man["measured_canvas_in"] = [man["measured"]["mediabox_width_in"],
                                 man["measured"]["mediabox_height_in"]]
    (HERE / "pdf_measurements.json").write_text(
        json.dumps({pdf.name: man["measured"]}, indent=2) + "\n")
    (HERE / "manifest_fig4.json").write_text(json.dumps(man, indent=2) + "\n")
    m = man["measured"]
    print(f"measured: MediaBox {m['mediabox_width_in']} x {m['mediabox_height_in']} in, "
          f"Tf sizes {m['distinct_tf_pt']} pt, all >= 6pt {m['all_text_ge_6pt']}, "
          f"{m['n_image_xobjects']} rasterized fills, fonts {m['embedded_fonts']}", flush=True)
    print(f"wrote {pdf}", flush=True)


if __name__ == "__main__":
    main()
