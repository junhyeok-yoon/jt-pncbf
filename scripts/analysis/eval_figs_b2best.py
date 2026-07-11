"""Analysis-only figure module for the B-2 best system (pi_theta + analytic V_M fast).

Renders figures C / C2 / D by CALLING src.eval.plotting helpers + style constants (production plotting.py
UNTOUCHED). NOT protocol-canonical routing (the canonical contour plotter can't take an h_fn — see
docs/versions/v2.5.0/eval_figs_b2best.md DIFF-C); this is the Option-2 diagnostic render.

  (C)  cbf_contour.png       — §3b layout (2 pool scenes x 3 DI velocities), h = deployed maneuver barrier.
  (C2) contour_slices.png    — 1 scene, 2 rows (V_M fast / v2.3.0 learned V_hat) x 3 velocity slices.
  (D)  stuck_panels_{1..3}   — 21 stall episodes, LQR-only + maneuver-filtered same axes, onset X marker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src._version import __version__
from src.eval import plotting as P
from src.common.control_net import ControlNet
from src.common.filter_hardnet import (_SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection,
                                        _box_aware_projection, _cbf_terms, _hardnet_params)
from src.common.maneuver_value import make_maneuver_h_fn, set_fast_path
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/v2.5.0__20260709-204711__seed42/checkpoints/step_007500.pt"
VHAT_CKPT = REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt"
POOL2K = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"
POOL500 = REPO / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl"
LAT_JS = [1, 2, 3, 4, 5, 6, 7, 8]
GAMMA_M = 0.02
DT = torch.float32


def _load(dev):
    set_fast_path(True)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    cfg = ck["config"]; cfg.setdefault("safety_channel", {})["type"] = "maneuver"; cfg["env"]["stuck_window_steps"] = 60
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    pol = ControlNet(system.obs_dim, system, cfg).to(device=dev, dtype=DT); pol.load_state_dict(ck["pi_state"]); pol.eval()
    return cfg, system, pol


def _eval_h_grid(h_fn, sc, velocity, res, world_lim, system, dev, chunk=4096):
    xs = torch.linspace(-world_lim, world_lim, res, device=dev, dtype=DT)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    vel = torch.as_tensor(velocity, dtype=DT, device=dev)
    states = P._contour_states(system, positions, vel)                      # [N,4]
    N = states.shape[0]; out = torch.empty(N, dtype=DT, device=dev)
    with torch.no_grad():
        for s0 in range(0, N, chunk):
            bs = batch_scenes([sc] * (min(chunk, N - s0)), device=dev, dtype=DT)
            out[s0:s0 + chunk] = torch.clamp(h_fn(states[s0:s0 + chunk], bs), -1.0, 1.0)
    return P._to_numpy(xs), P._to_numpy(out.reshape(res, res))


def _contour_panel(ax, xnp, hgrid, sc, title):
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cf = ax.contourf(xnp, xnp, hgrid, levels=P.CONTOUR_LEVELS, cmap=P.CONTOUR_CMAP, norm=norm, extend="both")
    if float(hgrid.min()) <= 0.0 <= float(hgrid.max()):
        ax.contour(xnp, xnp, hgrid, levels=[0.0], colors=P.ARENA_COLOR, linewidths=P.CONTOUR_ZERO_LINE_WIDTH)
    P._draw_obstacle_outlines(ax, sc)
    ax.plot(sc.goal[0], sc.goal[1], marker=P.GOAL_MARKER, color=P.GOAL_COLOR, markersize=P.GOAL_MARKER_SIZE, linestyle="None")
    ax.set_title(title, fontsize=P.PANEL_TITLE_FONT_SIZE)
    ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(P.ARENA_COLOR)
    return cf


def figure_C(cfg, system, dev, outpath, res=161):
    """§3b contour, h = deployed maneuver barrier (V_M+gamma_m). Rows = pool scenes 0,1; cols = DI velocities."""
    world_lim = float(cfg["env"]["world_lim"])
    scenes = load_pool(POOL2K).scenes[:2]
    h_fn = make_maneuver_h_fn(system, cfg, lateral_js=LAT_JS, gamma_m=GAMMA_M, dt_override=0.05)
    vels = [("(-1.5, 0.5)", (-1.5, 0.5)), ("(0, 0)", (0.0, 0.0)), ("(1.5, -0.5)", (1.5, -0.5))]
    plt.rcParams["font.family"] = P.FONT_FAMILY
    fig, axes = plt.subplots(2, 3, figsize=P.CONTOUR_FIG_SIZE, dpi=P.FIG_DPI)
    fig.suptitle(f"{__version__} · Final eval CBF contour · B-2 V_M fast (deployed h=V_M+gamma_m) · double_integrator",
                 fontsize=P.TITLE_FONT_SIZE)
    cf = None
    for r, sc in enumerate(scenes):
        for c, (vl, vv) in enumerate(vels):
            xnp, hgrid = _eval_h_grid(h_fn, sc, vv, res, world_lim, system, dev)
            cf = _contour_panel(axes[r, c], xnp, hgrid, sc, f"scene {r:02d} · v={vl}")
    fig.subplots_adjust(left=0.05, right=0.86, bottom=0.06, top=0.9, wspace=0.12, hspace=0.16)
    cax = fig.add_axes([0.885, 0.15, 0.018, 0.7]); cb = fig.colorbar(cf, cax=cax); cb.set_label("h = V_M + gamma_m (clip [-1,1])", fontsize=P.CONTROL_LABEL_FONT_SIZE)
    outpath.parent.mkdir(parents=True, exist_ok=True); fig.savefig(outpath); plt.close(fig)
    return {"scene_idx": [0, 1], "velocities": [v[0] for v in vels], "res": res}


def figure_C2(cfg, system, dev, outpath, res=161):
    """1 scene (idx 0), row1 V_M fast vs row2 v2.3.0 learned V_hat; cols = velocity slices; per-row colorbar."""
    world_lim = float(cfg["env"]["world_lim"])
    sc = load_pool(POOL2K).scenes[0]
    h_vm = make_maneuver_h_fn(system, cfg, lateral_js=LAT_JS, gamma_m=GAMMA_M, dt_override=0.05)
    vck = torch.load(VHAT_CKPT, map_location=dev, weights_only=False)
    vcfg = vck["config"]; vn = ValueNetEnsemble(system.obs_dim, vcfg).to(device=dev, dtype=DT)
    vn.load_state_dict(vck["v_s_state"]); vn.eval()
    h_vhat = make_h_fn(vn, system, use_target=False)               # deploy convention: mean-ensemble + read-out clip
    rows = [("V_M (fast, +gamma_m)", h_vm), ("learned V̂ (v2.3.0 deployed)", h_vhat)]
    vels = [("(0, 0)", (0.0, 0.0)), ("(1.25, 0)", (1.25, 0.0)), ("(2.5, 0)", (2.5, 0.0))]
    plt.rcParams["font.family"] = P.FONT_FAMILY
    fig, axes = plt.subplots(2, 3, figsize=P.CONTOUR_FIG_SIZE, dpi=P.FIG_DPI)
    fig.suptitle(f"{__version__} · CBF slices (diagnostic) · scene 00 · V_M vs learned V̂ · double_integrator",
                 fontsize=P.TITLE_FONT_SIZE)
    for r, (rl, h_fn) in enumerate(rows):
        cf = None
        for c, (vl, vv) in enumerate(vels):
            xnp, hgrid = _eval_h_grid(h_fn, sc, vv, res, world_lim, system, dev)
            cf = _contour_panel(axes[r, c], xnp, hgrid, sc, f"{rl.split()[0]} · v={vl}")
        pos = axes[r, 2].get_position()
        cax = fig.add_axes([0.90, pos.y0, 0.014, pos.height]); cb = fig.colorbar(cf, cax=cax)
        cb.set_label("h (clip [-1,1])", fontsize=P.CONTROL_LABEL_FONT_SIZE - 1)
        axes[r, 0].set_ylabel(rl, fontsize=P.PANEL_TITLE_FONT_SIZE)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.06, top=0.9, wspace=0.12, hspace=0.18)
    outpath.parent.mkdir(parents=True, exist_ok=True); fig.savefig(outpath); plt.close(fig)
    return {"scene_idx": 0, "row1": "V_M fast", "row2": "v2.3.0 V_hat deployed",
            "velocities": [v[0] for v in vels], "res": res,
            "note": "row-2 (learned V_hat) is eval-only diagnostic per 04_eval 7.1"}


def _roll_filtered_and_lqr(scenes, cfg, system, pol, dev, max_plant=200):
    """Maneuver-filtered rollout (states + per-step intervention) and LQR-only rollout for `scenes`."""
    bs = batch_scenes(scenes, device=dev, dtype=DT); x = initial_states_from_batch(bs).to(DT); B = x.shape[0]
    params = _hardnet_params(cfg); bounds = system.u_bounds
    h_fn = make_maneuver_h_fn(system, cfg, lateral_js=LAT_JS, gamma_m=GAMMA_M, dt_override=0.05)
    UN = torch.zeros(max_plant, B, 2, device=dev); US = torch.zeros_like(UN); ST = [x]
    with torch.no_grad():
        for t in range(max_plant):
            un = pol(system.observation(x, bs))
            h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False); h, lf, lg = h.detach(), lf.detach(), lg.detach()
            alpha = _base_alpha(h, params); row = -lf - alpha * h; proj = _base_projection(un, lg, row, bounds, params)
            us, _ = _box_aware_projection(un, proj, lg, row, bounds)
            UN[t] = un; US[t] = us; x = rk4_step(system, x, us, 0.05); ST.append(x)
    Sf = torch.stack(ST, 0)
    xl = initial_states_from_batch(bs).to(DT); goal = torch.as_tensor(bs.goal, dtype=DT, device=dev)
    if goal.ndim == 1:
        goal = goal.unsqueeze(0).expand(B, -1)
    LST = [xl]
    with torch.no_grad():
        for _ in range(max_plant):
            xl = rk4_step(system, xl, system.lqr_action(xl, goal), 0.05); LST.append(xl)
    Sl = torch.stack(LST, 0)
    interv = (torch.linalg.norm(US - UN, dim=2) > 1e-3)
    return Sf.cpu(), Sl.cpu(), interv.cpu()


def figure_D(cfg, system, pol, dev, outdir):
    """21 stall episodes: LQR-only (dotted) + maneuver-filtered (same axes) + onset X + dual cluster labels."""
    d = torch.load(REPO / "docs/versions/v2.5.0/reach_witness/stall_onsets.pt", weights_only=False)
    idx = d["scene_idx"]; onset = d["onset"]; cl_s = d["cluster_start"]; cl_o = d["cluster_onset"]
    scenes = load_pool(POOL500).scenes; sub = [scenes[i] for i in idx]
    Sf, Sl, interv = _roll_filtered_and_lqr(sub, cfg, system, pol, dev)
    world_lim = float(cfg["env"]["world_lim"]); plt.rcParams["font.family"] = P.FONT_FAMILY
    paths = []
    for fig_i, s0 in enumerate(range(0, len(sub), 8), start=1):
        grp = list(range(s0, min(s0 + 8, len(sub))))
        fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.4), dpi=P.FIG_DPI)
        fig.suptitle(f"{__version__} · B-2 stall episodes {s0}-{grp[-1]} · LQR-only (dotted) + V_M-filtered · onset=x",
                     fontsize=P.TITLE_FONT_SIZE - 1)
        for ax, j in zip(axes.ravel(), grp):
            sc = sub[j]
            P._draw_arena(ax, world_lim); P._draw_obstacles(ax, sc)
            P._draw_baseline(ax, Sl[:, j, :2].numpy())
            P._draw_filtered_trajectory(ax, Sf[:, j, :2].numpy(), interv[:, j].numpy())
            ax.plot(sc.start[0], sc.start[1], marker=P.START_MARKER, color=P.START_COLOR, markersize=P.START_MARKER_SIZE, linestyle="None")
            ax.plot(sc.goal[0], sc.goal[1], marker=P.GOAL_MARKER, color=P.GOAL_COLOR, markersize=P.GOAL_MARKER_SIZE, linestyle="None")
            ax.scatter(float(onset[j, 0]), float(onset[j, 1]), marker="x", color=(0.1, 0.1, 0.8), s=70, linewidths=1.8, zorder=5)
            ax.set_title(f"#{idx[j]} · start:{cl_s[j]} · onset:{cl_o[j]}", fontsize=P.PANEL_TITLE_FONT_SIZE - 1)
            ax.set_aspect("equal", adjustable="box"); ax.set_xlim(-world_lim, world_lim); ax.set_ylim(-world_lim, world_lim)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color(P.ARENA_COLOR)
        for ax in axes.ravel()[len(grp):]:
            ax.axis("off")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        op = outdir / f"stuck_panels_{fig_i}.png"; fig.savefig(op); plt.close(fig); paths.append(str(op))
    return {"n_episodes": len(sub), "n_figures": len(paths), "paths": paths}


def main():
    import sys
    outdir = Path(sys.argv[1]); (outdir / "figures").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); cfg, system, pol = _load(dev)
    inv = {}
    inv["C"] = figure_C(cfg, system, dev, outdir / "figures/cbf_contour.png")
    print("C done", flush=True)
    inv["C2"] = figure_C2(cfg, system, dev, outdir / "figures/contour_slices.png")
    print("C2 done", flush=True)
    inv["D"] = figure_D(cfg, system, pol, dev, outdir / "figures")
    print("D done", flush=True)
    (outdir / "figures/CD_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    print(json.dumps(inv), flush=True)


if __name__ == "__main__":
    main()
