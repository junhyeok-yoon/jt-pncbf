"""v2.8.4 charter-"v4" Round 3 — the trajectory grids, reusing exactly what the v3 diagnosis established.

`docs/versions/v2.8.4/mppi_diag.md` §2.1 settled two things about plotting an MPPI arm, and neither is
re-derived here:

  1. The shipped in-loop grid `src.eval.plotting.plot_trajectory_control_grid` — the same call
     `src.eval.run_full.write_trajectory_figures` makes for every training run — accepts an MPPI episode
     with NO source edit. What refuses MPPI is the shipped SELECTOR, `_selected_intervention_episodes`,
     which keeps episodes whose CBF intervention mask fired; MPPI enters with an identity filter, so that
     mask is identically false and the selector returns nothing. The selector is not the plotter, so a
     stratified selection is passed straight into the shipped plotter. One cosmetic artefact of that
     reuse: the shipped suptitle template reads "selected N/16 from M interventions", and for this
     filter-free arm "interventions" means "stratified episodes" — the wording is the shipped template's.
  2. The xz/yz projections plus the distance / |v| / |omega| / tilt time series have NO shipped grid
     (`docs/protocol/04_eval.md` §3.5 leaves the projection figure unspecified), so they are emitted here
     as a clearly-labelled REPRODUCTION whose suptitle begins `REPRODUCTION (not a shipped grid)`.

Stratification, per the dispatch: 8 timeout episodes nearest the goal at t_end, 8 farthest, 4 collisions,
4 lowest final distance overall — 24 panel slots against the shipped 8-panels-per-figure convention, so
three shipped-format figures are emitted. Strata may overlap; overlap is NOT removed and the membership is
recorded. Every radius drawn is read from the effective config; none is typed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src._version import __version__
from src.common.system import System
from src.eval.plotting import (
    EpisodeControlSpec,
    FIG_DPI,
    GOAL_COLOR,
    PANELS_PER_FIGURE,
    START_COLOR,
    plot_trajectory_control_grid,
)


def stratify(outcome: np.ndarray, d_end: np.ndarray) -> dict[str, dict[str, Any]]:
    """The dispatch's four strata, by the v3 diagnosis's own ordering (stable sort on d_end)."""
    timeout_idx = np.nonzero(outcome == "timeout")[0]
    order = timeout_idx[np.argsort(d_end[timeout_idx], kind="stable")]
    collision_idx = np.nonzero(outcome == "collision")[0]
    strata = {
        "S1_timeout_nearest_at_t_end": {
            "requested": 8, "available": int(timeout_idx.size),
            "indices": order[:8].tolist(),
        },
        "S2_timeout_farthest_at_t_end": {
            "requested": 8, "available": int(timeout_idx.size),
            "indices": order[::-1][:8].tolist(),
        },
        "S3_collision": {
            "requested": 4, "available": int(collision_idx.size),
            "indices": collision_idx[
                np.argsort(d_end[collision_idx], kind="stable")
            ][:4].tolist(),
        },
        "S4_lowest_final_distance_overall": {
            "requested": 4, "available": int(d_end.size),
            "indices": np.argsort(d_end, kind="stable")[:4].tolist(),
        },
    }
    for block in strata.values():
        block["emitted"] = len(block["indices"])
        block["shortfall"] = max(0, int(block["requested"]) - int(block["emitted"]))
    return strata


def emit_grids(
    *,
    system: System,
    config: Mapping[str, Any],
    trajectories: list[Any],
    episode_rows: list[Mapping[str, Any]],
    tilt_deg: np.ndarray,
    label: str,
    fig_dir: Path,
) -> dict[str, Any]:
    """Emit grids A/B/C with the shipped plotter and the three xz/yz + time-series REPRODUCTIONS.

    All arithmetic here is post-rollout and draws no random numbers; it is nevertheless wrapped in
    `torch.random.fork_rng` so instrumentation can never shift a stream a later cell draws from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    goal_radius = float(config["env"]["goal_radius"])
    goal_speed_radius = float(config["env"]["goal_speed_radius"])
    goal_angrate_radius = float(config["env"]["goal_angrate_radius"])
    band_z = float(config["env"].get("band_collision_limit", 0.0))
    world_lim = float(config["env"]["world_lim"])
    dt = float(config["env"]["dt"])

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        with torch.no_grad():
            outcome = np.array([str(r["outcome"]) for r in episode_rows], dtype=object)
            n_steps = np.array([int(r["n_steps"]) for r in episode_rows])
            d_end = np.zeros(len(episode_rows), dtype=np.float64)
            d_min = np.zeros(len(episode_rows), dtype=np.float64)
            for i, tr in enumerate(trajectories):
                states = tr.filtered.states[: n_steps[i] + 1, 0, :]
                goal = torch.as_tensor(tr.scene.goal, dtype=states.dtype, device=states.device)
                dist = torch.linalg.norm(system.position(states) - goal, dim=-1)
                d_end[i] = float(dist[-1].item())
                d_min[i] = float(dist.min().item())

            strata = stratify(outcome, d_end)
            groups = [
                ("A", "8 timeout episodes NEAREST the goal at t_end",
                 strata["S1_timeout_nearest_at_t_end"]["indices"]),
                ("B", "8 timeout episodes FARTHEST from the goal at t_end",
                 strata["S2_timeout_farthest_at_t_end"]["indices"]),
                ("C", "4 collision episodes + 4 lowest-final-distance episodes overall",
                 strata["S3_collision"]["indices"]
                 + strata["S4_lowest_final_distance_overall"]["indices"]),
            ]

            def spec(i: int) -> EpisodeControlSpec:
                tr = trajectories[i]
                return EpisodeControlSpec(
                    scene=tr.scene, pool_index=i, outcome=tr.filtered_outcome,
                    event_step=tr.filtered_event_step,
                    filtered_states=tr.filtered.states[:, 0, :],
                    intervention_mask=tr.filtered.intervention_mask[:, 0],
                    u_nom=tr.filtered.u_nom[:, 0, :], u_safe=tr.filtered.u_safe[:, 0, :],
                    nominal_states=None,
                )

            shipped: list[dict[str, Any]] = []
            for letter, note, idxs in groups:
                out_path = fig_dir / f"trajectory_grid_{letter}.png"
                plot_trajectory_control_grid(
                    episodes=[spec(i) for i in idxs[:PANELS_PER_FIGURE]],
                    output_path=out_path, config=config,
                    role=f"MPPI goal-attraction redesign Round 3 · {label} · {note}",
                    system_name=str(system.name), letter=letter,
                    u_bounds=system.u_bounds, total_selected=len(idxs), shortfall=0,
                )
                shipped.append({
                    "letter": letter, "path": str(out_path), "note": note,
                    "episode_indices": idxs[:PANELS_PER_FIGURE],
                    "producer": "src.eval.plotting.plot_trajectory_control_grid — the SHIPPED in-loop "
                                "grid, the same call src.eval.run_full.write_trajectory_figures makes "
                                "for every training run; no source edit was needed",
                })
                print(f"wrote {out_path}", flush=True)

            repro: list[dict[str, Any]] = []
            for letter, note, idxs in groups:
                idxs = idxs[:PANELS_PER_FIGURE]
                fig, axes = plt.subplots(len(idxs), 4, figsize=(17.0, 2.7 * len(idxs)), dpi=FIG_DPI,
                                         squeeze=False)
                fig.suptitle(
                    f"REPRODUCTION (not a shipped grid) · {__version__} · MPPI goal-attraction redesign "
                    f"Round 3 · {label} · Figure {letter}-xzyz · {note}", fontsize=13)
                for r, i in enumerate(idxs):
                    tr = trajectories[i]
                    st = tr.filtered.states[: n_steps[i] + 1, 0, :].detach().cpu().numpy()
                    g = np.asarray(tr.scene.goal, dtype=np.float64).reshape(-1)
                    p = st[:, :3].astype(np.float64)
                    sc = tr.scene
                    for c, (a, b, an, bn) in enumerate(((0, 2, "x", "z"), (1, 2, "y", "z"))):
                        ax = axes[r][c]
                        for ctr, rad, act in zip(sc.obstacle_centers, sc.obstacle_radii,
                                                 sc.obstacle_active):
                            if not act:
                                continue
                            ax.axvspan(float(ctr[a]) - float(rad), float(ctr[a]) + float(rad),
                                       color=(0.65, 0.65, 0.65, 0.45), lw=0)
                        if band_z > 0.0:
                            ax.axhline(-band_z, color="black", lw=1.0, ls="--", alpha=0.8)
                            ax.axhline(band_z, color="black", lw=1.0, ls="--", alpha=0.8)
                        ax.plot(p[:, a], p[:, b], color="black", lw=1.4)
                        ax.plot(p[0, a], p[0, b], marker="o", color=START_COLOR, ms=6, ls="None")
                        ax.plot(g[a], g[b], marker="*", color=GOAL_COLOR, ms=11, ls="None")
                        if outcome[i] in ("collision", "oob"):
                            ax.scatter(p[-1, a], p[-1, b], marker="x", color=GOAL_COLOR, s=60)
                        ax.set_xlim(-world_lim, world_lim)
                        ax.set_ylim(-min(band_z or world_lim, world_lim) * 1.15,
                                    min(band_z or world_lim, world_lim) * 1.15)
                        ax.set_xlabel(an, fontsize=8)
                        ax.set_ylabel(bn, fontsize=8)
                        ax.tick_params(labelsize=7)
                        ax.grid(True, color=(0.75, 0.75, 0.75), alpha=0.25, lw=0.4)
                        ax.set_title(f"{i:03d} · {outcome[i]} · {an}{bn}", fontsize=9)
                    ax = axes[r][2]
                    t = np.arange(st.shape[0]) * dt
                    dist = np.linalg.norm(p - g, axis=-1)
                    ax.plot(t, dist, color=(0.0, 0.31, 0.72), lw=1.4, label="d(p,g)")
                    ax.axhline(goal_radius, color=(0.0, 0.31, 0.72), ls="--", lw=1.0)
                    ax.set_yscale("log")
                    ax.set_xlabel("t [s]", fontsize=8)
                    ax.set_ylabel("distance [m]", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, color=(0.75, 0.75, 0.75), alpha=0.25, lw=0.4)
                    ax.set_title(f"d_min {d_min[i]:.2f}  d_end {d_end[i]:.2f}", fontsize=9)
                    ax2 = axes[r][3]
                    v = np.linalg.norm(st[:, 7:10].astype(np.float64), axis=-1)
                    w = np.linalg.norm(st[:, 10:13].astype(np.float64), axis=-1)
                    ax2.plot(t, v, color=(0.84, 0.19, 0.15), lw=1.3, label="|v|")
                    ax2.plot(t, w, color=(0.0, 0.55, 0.30), lw=1.3, label="|omega|")
                    ax2.axhline(goal_speed_radius, color=(0.84, 0.19, 0.15), ls="--", lw=1.0)
                    ax2.axhline(goal_angrate_radius, color=(0.0, 0.55, 0.30), ls=":", lw=1.0)
                    axt = ax2.twinx()
                    up = system.thrust_axis(
                        tr.filtered.states[: n_steps[i] + 1, 0, :]).detach().cpu().numpy()
                    tilt_t = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))
                    axt.plot(t, tilt_t, color=(0.5, 0.5, 0.5), lw=1.0, alpha=0.8)
                    axt.set_ylabel("tilt [deg]", fontsize=8)
                    axt.set_ylim(0, 180)
                    axt.tick_params(labelsize=7)
                    ax2.set_xlabel("t [s]", fontsize=8)
                    ax2.set_ylabel("|v| [m/s], |omega| [rad/s]", fontsize=8)
                    ax2.tick_params(labelsize=7)
                    ax2.grid(True, color=(0.75, 0.75, 0.75), alpha=0.25, lw=0.4)
                    ax2.legend(fontsize=7, loc="upper right", frameon=False)
                    ax2.set_title(f"spawn tilt {float(tilt_deg[i]):.0f}°  |v|_end {v[-1]:.2f}  "
                                  f"|w|_end {w[-1]:.2f}", fontsize=9)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
                out_path = fig_dir / f"trajectory_grid_{letter}_xzyz_timeseries_REPRODUCTION.png"
                fig.savefig(out_path)
                plt.close(fig)
                repro.append({
                    "letter": f"{letter}-xzyz", "path": str(out_path), "note": note,
                    "episode_indices": idxs,
                    "producer": "REPRODUCTION — this module. No shipped function emits an xz/yz + "
                                "time-series grid (04_eval §3.5 leaves it unspecified); the suptitle "
                                "begins 'REPRODUCTION (not a shipped grid)'.",
                })
                print(f"wrote {out_path}", flush=True)

    return {
        "strata": strata,
        "shipped": shipped,
        "reproduction": repro,
        "panels_per_figure": int(PANELS_PER_FIGURE),
        "note": "24 requested panel slots against the shipped 8-per-figure convention, so THREE "
                "shipped-format figures are emitted rather than two; overlap between strata is not "
                "removed and the membership is recorded above.",
    }
