from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import torch

from src._version import __version__
from src.common.value_net import make_h_fn


ARENA_COLOR = "black"
OBSTACLE_FACE = (0.65, 0.65, 0.65, 0.7)
OBSTACLE_EDGE = (0.35, 0.35, 0.35, 0.9)
START_COLOR = "green"
GOAL_COLOR = "red"
TRAJECTORY_INACTIVE = "black"
TRAJECTORY_ACTIVE = (0.5, 0.5, 0.5)
GRID_COLOR = (0.75, 0.75, 0.75)
CONTROL_SAFE = ((0.0, 0.31, 0.72), (0.84, 0.19, 0.15))
CONTROL_NOMINAL_ALPHA = 0.38
BASELINE_ALPHA = 0.4
FIG_DPI = 150
FIG_SIZE = (14.0, 10.8)
CONTOUR_FIG_SIZE = (15.0, 9.0)
TITLE_FONT_SIZE = 15
PANEL_TITLE_FONT_SIZE = 11
CONTROL_LABEL_FONT_SIZE = 9
LEGEND_FONT_SIZE = 11
LINE_WIDTH = 1.7
BASELINE_LINE_WIDTH = 1.5
CONTROL_LINE_WIDTH = 1.6
BOUND_LINE_WIDTH = 1.0
START_MARKER = "o"
GOAL_MARKER = "*"
COLLISION_MARKER = "x"
START_MARKER_SIZE = 7.0
GOAL_MARKER_SIZE = 11.0
COLLISION_MARKER_SIZE = 75
LEGEND_MARKER_SIZE = 8.0
LEGEND_HANDLE_LENGTH = 2.5
FONT_FAMILY = "DejaVu Sans"
PANELS_PER_FIGURE = 8
EPISODES_PER_REPORT = 16
CONTOUR_CMAP = "coolwarm"
CONTOUR_LEVELS = np.linspace(-1.0, 1.0, 41)
CONTOUR_ZERO_LINE_WIDTH = 1.6
CONTOUR_OBSTACLE_LINE_WIDTH = 0.8
CONTOUR_VELOCITY_ARROW_SCALE = 0.9
CONTOUR_VELOCITY_ARROW_LINE_WIDTH = 1.9
CONTOUR_VELOCITY_ARROW_OUTLINE_WIDTH = 4.0
CONTOUR_VELOCITY_ARROW_HEAD_SIZE = 14.0
CONTOUR_ZERO_DOT_SIZE = 34.0


@dataclass(frozen=True)
class TrajectorySpec:
    states: Any
    intervention_mask: Any | None = None
    baseline: bool = False


@dataclass(frozen=True)
class EpisodeControlSpec:
    scene: Any
    pool_index: int
    outcome: str
    event_step: int
    filtered_states: Any
    intervention_mask: Any
    u_nom: Any
    u_safe: Any
    nominal_states: Any | None = None


@dataclass(frozen=True)
class CBFContourResult:
    path: Path
    h_min: float
    h_max: float
    zero_contour_panels: int
    velocity_arrow_count: int
    zero_velocity_dot_count: int


def plot_trajectory_control_grid(
    episodes: list[EpisodeControlSpec],
    output_path: Path,
    config: Mapping[str, Any],
    role: str,
    system_name: str,
    letter: str,
    u_bounds: Any,
    *,
    total_selected: int,
    shortfall: int,
    drop_nominal_control: bool = False,
) -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    world_lim = float(config["env"]["world_lim"])
    bounds = _to_numpy(u_bounds).astype(np.float64)

    fig, axes = plt.subplots(4, 4, figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.suptitle(
        f"{__version__} · {role} · {system_name} · Figure {letter} "
        f"(selected {min(total_selected, EPISODES_PER_REPORT)}/16 from "
        f"{total_selected} interventions, shortfall {shortfall})",
        fontsize=TITLE_FONT_SIZE,
        fontweight="normal",
    )

    for episode_slot in range(PANELS_PER_FIGURE):
        row = episode_slot // 2
        col = (episode_slot % 2) * 2
        traj_axis = axes[row, col]
        ctrl_axis = axes[row, col + 1]
        if episode_slot < len(episodes):
            episode = episodes[episode_slot]
            _plot_trajectory_panel(traj_axis, episode, world_lim)
            _plot_control_panel(
                ctrl_axis,
                episode,
                bounds,
                drop_nominal=drop_nominal_control,
            )
        else:
            traj_axis.axis("off")
            ctrl_axis.axis("off")

    trajectory_legend, control_legend = _legend_handles()
    fig.legend(
        handles=trajectory_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.041),
        ncol=6,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
    )
    fig.legend(
        handles=control_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.png")
    fig.savefig(temp_path, metadata={"Software": "matplotlib"})
    plt.close(fig)
    temp_path.replace(output_path)


def plot_cbf_contours(
    scenes: list[Any],
    output_path: Path,
    config: Mapping[str, Any],
    system: Any,
    value_net: Any,
    *,
    resolution: int = 150,
    role: str = "CBF contour",
) -> CBFContourResult:
    if len(scenes) < 2:
        raise ValueError("CBF contour plot requires at least two scenes.")
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    plt.rcParams["font.family"] = FONT_FAMILY
    world_lim = float(config["env"]["world_lim"])
    device, dtype = _module_device_dtype(value_net, system)
    # v2.5.2 fix: plot the DEPLOYED filter h (build_safety_h_fn) not the trainer's value_net. In a
    # frozen-channel run (exact_m0/maneuver/cpi) value_net is fresh-init (K_V forced to 0), so make_h_fn
    # would render a random field; build_safety_h_fn returns the actual deployed certificate. Value mode
    # (default) -> build_safety_h_fn returns make_h_fn(value_net), so it is unchanged there.
    from src.common.maneuver_value import build_safety_h_fn
    from types import SimpleNamespace
    h_fn = build_safety_h_fn(system, config, value_net)

    x_axis = torch.linspace(-world_lim, world_lim, resolution, device=device, dtype=dtype)
    y_axis = torch.linspace(-world_lim, world_lim, resolution, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x_axis, y_axis, indexing="xy")
    positions = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    x_np = _to_numpy(x_axis)
    y_np = _to_numpy(y_axis)

    velocity_columns = _contour_velocity_columns(system)
    fig, axes = plt.subplots(2, 3, figsize=CONTOUR_FIG_SIZE, dpi=FIG_DPI)
    fig.suptitle(
        f"{__version__} · {role} · {system.name}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="normal",
    )

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    h_min = float("inf")
    h_max = -float("inf")
    zero_contour_panels = 0
    velocity_arrow_count = 0
    zero_velocity_dot_count = 0
    contour_mappable = None

    for row, scene in enumerate(scenes[:2]):
        # tensorised scene (obstacle arrays as [K,*] tensors) so the deployed h_fn — value-net OR the
        # frozen exact_m0/maneuver channels — receives torch obstacles; both broadcast [K,*] against the grid.
        scene_t = SimpleNamespace(
            obstacle_centers=torch.as_tensor(scene.obstacle_centers, dtype=dtype, device=device),
            obstacle_radii=torch.as_tensor(scene.obstacle_radii, dtype=dtype, device=device),
            obstacle_active=torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=device),
            goal=torch.as_tensor(np.asarray(scene.goal), dtype=dtype, device=device))
        for col, (velocity_label, velocity_value) in enumerate(velocity_columns):
            axis = axes[row, col]
            velocity = velocity_value.to(device=device, dtype=dtype)
            states = _contour_states(system, positions, velocity)
            with torch.no_grad():
                h_values = torch.clamp(h_fn(states, scene_t), -1.0, 1.0)
            h_grid = _to_numpy(h_values.reshape(resolution, resolution))
            h_min = min(h_min, float(np.min(h_grid)))
            h_max = max(h_max, float(np.max(h_grid)))

            contour_mappable = axis.contourf(
                x_np,
                y_np,
                h_grid,
                levels=CONTOUR_LEVELS,
                cmap=CONTOUR_CMAP,
                norm=norm,
                extend="both",
            )
            if float(np.min(h_grid)) <= 0.0 <= float(np.max(h_grid)):
                zero_contour = axis.contour(
                    x_np,
                    y_np,
                    h_grid,
                    levels=[0.0],
                    colors=ARENA_COLOR,
                    linewidths=CONTOUR_ZERO_LINE_WIDTH,
                )
                if zero_contour.allsegs and zero_contour.allsegs[0]:
                    zero_contour_panels += 1
            _draw_obstacle_outlines(axis, scene)
            velocity_vector = _contour_velocity_vector(system, velocity)
            if _draw_contour_velocity_indicator(axis, velocity_vector):
                velocity_arrow_count += 1
            else:
                zero_velocity_dot_count += 1
            axis.plot(
                scene.goal[0],
                scene.goal[1],
                marker=GOAL_MARKER,
                color=GOAL_COLOR,
                markersize=GOAL_MARKER_SIZE,
                linestyle="None",
            )
            axis.set_title(
                f"scene {row:02d} · {velocity_label}",
                fontsize=PANEL_TITLE_FONT_SIZE,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-world_lim, world_lim)
            axis.set_ylim(-world_lim, world_lim)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color(ARENA_COLOR)

    if contour_mappable is None:
        raise RuntimeError("No contour panels were rendered.")
    fig.subplots_adjust(
        left=0.055,
        right=0.84,
        bottom=0.105,
        top=0.89,
        wspace=0.20,
        hspace=0.26,
    )
    colorbar_axis = fig.add_axes([0.875, 0.18, 0.018, 0.66])
    colorbar = fig.colorbar(contour_mappable, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=CONTROL_LABEL_FONT_SIZE)
    colorbar.set_label("h", fontsize=CONTROL_LABEL_FONT_SIZE)

    fig.legend(
        handles=_contour_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.022),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.png")
    fig.savefig(temp_path, metadata={"Software": "matplotlib"})
    plt.close(fig)
    temp_path.replace(output_path)
    return CBFContourResult(
        path=output_path,
        h_min=h_min,
        h_max=h_max,
        zero_contour_panels=zero_contour_panels,
        velocity_arrow_count=velocity_arrow_count,
        zero_velocity_dot_count=zero_velocity_dot_count,
    )


def plot_quadrotor_cbf_contour(
    scenes: list[Any],
    output_path: Path,
    config: Mapping[str, Any],
    system: Any,
    value_net: Any,
    *,
    resolution: int = 160,
    role: str = "CBF contour",
) -> Path:
    """6D-appropriate CBF contour for the planar quadrotor (the 2D-velocity-slice contour does not apply).
    Renders the deployed h(x)=V_hat on the POSITION plane (px,py) for up to two scenes across three
    approach-speed slices s=v^T Re (theta=0, omega=0 -> Re=(0,1), v=(0,s)) — the h_star velocity channel.
    Convention (matches the gate gate_in=sigmoid(-h)): h<0 safe (blue), h=0 boundary (black), h>0 unsafe."""
    plt.rcParams["font.family"] = FONT_FAMILY
    device, dtype = _module_device_dtype(value_net, system)
    h_fn = make_h_fn(value_net, system)
    world_lim = float(config["env"]["world_lim"])
    goal_radius = float(config["env"].get("goal_radius", 0.15))
    slices = [(-2.0, "v.Re=-2"), (0.0, "hover v.Re=0"), (2.0, "v.Re=+2")]
    use_scenes = scenes[:2] if len(scenes) >= 2 else scenes[:1]
    ax_lin = torch.linspace(-world_lim, world_lim, resolution, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(ax_lin, ax_lin, indexing="xy")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    axn = _to_numpy(ax_lin)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    nrow, ncol = len(use_scenes), len(slices)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.4 * nrow), dpi=FIG_DPI, squeeze=False)
    fig.suptitle(f"{__version__} · {role} · quadrotor_planar · h<0 safe (blue), h=0 boundary, h>0 unsafe",
                 fontsize=TITLE_FONT_SIZE, fontweight="normal")
    mappable = None
    from types import SimpleNamespace
    for r, sc in enumerate(use_scenes):
        active = _to_numpy(torch.as_tensor(np.asarray(sc.obstacle_active))).astype(bool)
        centers = np.asarray(sc.obstacle_centers)[active]
        radii = np.asarray(sc.obstacle_radii)[active]
        goal = np.asarray(sc.goal)
        scene_t = SimpleNamespace(
            obstacle_centers=torch.as_tensor(np.asarray(sc.obstacle_centers), dtype=dtype, device=device),
            obstacle_radii=torch.as_tensor(np.asarray(sc.obstacle_radii), dtype=dtype, device=device),
            obstacle_active=torch.as_tensor(np.asarray(sc.obstacle_active), dtype=torch.bool, device=device),
            goal=torch.as_tensor(goal, dtype=dtype, device=device))
        for c, (s, label) in enumerate(slices):
            axis = axes[r][c]
            x = torch.zeros(pos.shape[0], 6, device=device, dtype=dtype)
            x[:, :2] = pos
            x[:, 4] = s                                              # vy = s (theta=0 -> v^T Re = vy)
            with torch.no_grad():
                h = torch.clamp(h_fn(x, scene_t), -1.0, 1.0).reshape(resolution, resolution)
            h = _to_numpy(h)
            mappable = axis.contourf(axn, axn, h, levels=CONTOUR_LEVELS, cmap=CONTOUR_CMAP, norm=norm, extend="both")
            if float(np.min(h)) <= 0.0 <= float(np.max(h)):
                axis.contour(axn, axn, h, levels=[0.0], colors="black", linewidths=CONTOUR_ZERO_LINE_WIDTH)
            for ctr, rad in zip(centers, radii):
                axis.add_patch(Circle((ctr[0], ctr[1]), rad, fill=False, ec="k", lw=1.2, ls="--"))
                axis.add_patch(Circle((ctr[0], ctr[1]), rad, fill=True, fc="k", alpha=0.12))
            axis.plot(goal[0], goal[1], marker="*", ms=15, mfc="gold", mec="k", mew=1.0, zorder=5)
            axis.add_patch(Circle((goal[0], goal[1]), goal_radius, fill=False, ec="gold", lw=1.0))
            axis.set_xlim(-world_lim, world_lim); axis.set_ylim(-world_lim, world_lim)
            axis.set_aspect("equal")
            axis.set_title(f"scene{r} ({int(active.sum())} obs) · {label}", fontsize=9)
            if c == 0:
                axis.set_ylabel("py")
            if r == nrow - 1:
                axis.set_xlabel("px")
    if mappable is not None:
        fig.subplots_adjust(right=0.90, top=0.86, hspace=0.22, wspace=0.16)
        cax = fig.add_axes([0.92, 0.12, 0.014, 0.72])
        fig.colorbar(mappable, cax=cax, label="h(x)=V_hat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scene_grid(
    scenes: list[Any],
    output_path: Path,
    config: Mapping[str, Any],
    role: str,
    system_name: str,
    letter: str,
    start_index: int,
    trajectories: list[list[TrajectorySpec]] | None = None,
    outcomes: list[str] | None = None,
    event_steps: list[int] | None = None,
    draw_start_velocity: bool = False,
    draw_final_velocity: bool = False,
) -> None:
    _ = draw_start_velocity, draw_final_velocity
    plt.rcParams["font.family"] = FONT_FAMILY
    world_lim = float(config["env"]["world_lim"])
    hi = start_index + len(scenes) - 1

    fig, axes = plt.subplots(4, 4, figsize=(10.0, 10.8), dpi=FIG_DPI)
    fig.suptitle(
        f"{__version__} · {role} · {system_name} · Figure {letter} "
        f"(scenes {start_index:02d}–{hi:02d})",
        fontsize=TITLE_FONT_SIZE,
        fontweight="normal",
    )
    trajectories = trajectories or [[] for _ in scenes]
    outcomes = outcomes or [None for _ in scenes]
    event_steps = event_steps or [-1 for _ in scenes]

    for offset, axis in enumerate(axes.flat):
        if offset >= len(scenes):
            axis.axis("off")
            continue
        episode = EpisodeControlSpec(
            scene=scenes[offset],
            pool_index=start_index + offset,
            outcome=outcomes[offset] or "",
            event_step=event_steps[offset],
            filtered_states=trajectories[offset][-1].states
            if trajectories[offset]
            else np.zeros((0, 2)),
            intervention_mask=trajectories[offset][-1].intervention_mask
            if trajectories[offset]
            else None,
            u_nom=np.zeros((0, 2)),
            u_safe=np.zeros((0, 2)),
            nominal_states=trajectories[offset][0].states
            if trajectories[offset] and trajectories[offset][0].baseline
            else None,
        )
        _plot_trajectory_panel(axis, episode, world_lim)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.png")
    fig.savefig(temp_path, metadata={"Software": "matplotlib"})
    plt.close(fig)
    temp_path.replace(output_path)


def _plot_trajectory_panel(
    axis: plt.Axes,
    episode: EpisodeControlSpec,
    world_lim: float,
) -> None:
    _draw_arena(axis, world_lim)
    _draw_obstacles(axis, episode.scene)

    if episode.nominal_states is not None:
        _draw_baseline(axis, _to_numpy(episode.nominal_states))
    filtered_states = _to_numpy(episode.filtered_states)
    if filtered_states.size:
        _draw_filtered_trajectory(
            axis,
            filtered_states,
            _intervention_mask(episode.intervention_mask, max(0, filtered_states.shape[0] - 1)),
        )

    axis.plot(
        episode.scene.start[0],
        episode.scene.start[1],
        marker=START_MARKER,
        color=START_COLOR,
        markersize=START_MARKER_SIZE,
        linestyle="None",
    )
    axis.plot(
        episode.scene.goal[0],
        episode.scene.goal[1],
        marker=GOAL_MARKER,
        color=GOAL_COLOR,
        markersize=GOAL_MARKER_SIZE,
        linestyle="None",
    )

    if episode.outcome == "collision" and episode.event_step >= 0 and filtered_states.size:
        if episode.event_step < filtered_states.shape[0]:
            axis.scatter(
                filtered_states[episode.event_step, 0],
                filtered_states[episode.event_step, 1],
                marker=COLLISION_MARKER,
                color=GOAL_COLOR,
                s=COLLISION_MARKER_SIZE,
                linewidths=1.0,
            )

    axis.set_title(
        _panel_title(episode.pool_index, episode.outcome),
        fontsize=PANEL_TITLE_FONT_SIZE,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-world_lim, world_lim)
    axis.set_ylim(-world_lim, world_lim)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(ARENA_COLOR)


def _plot_control_panel(
    axis: plt.Axes,
    episode: EpisodeControlSpec,
    bounds: np.ndarray,
    *,
    drop_nominal: bool,
) -> None:
    u_safe = _to_numpy(episode.u_safe).astype(np.float64)
    u_nom = _to_numpy(episode.u_nom).astype(np.float64)
    if u_safe.ndim != 2:
        raise ValueError(f"u_safe must have shape [T, action_dim], got {u_safe.shape}.")
    time = np.arange(u_safe.shape[0])
    action_dim = min(u_safe.shape[1], 2)

    for idx in range(action_dim):
        color = CONTROL_SAFE[idx]
        if not drop_nominal and u_nom.shape == u_safe.shape:
            axis.plot(
                time,
                u_nom[:, idx],
                color=color,
                alpha=CONTROL_NOMINAL_ALPHA,
                linewidth=CONTROL_LINE_WIDTH,
                linestyle="--",
            )
        axis.plot(
            time,
            u_safe[:, idx],
            color=color,
            alpha=1.0,
            linewidth=CONTROL_LINE_WIDTH,
            linestyle="-",
        )
        axis.axhline(
            bounds[idx, 0],
            color=ARENA_COLOR,
            linewidth=BOUND_LINE_WIDTH,
            linestyle="--",
            alpha=0.75,
        )
        axis.axhline(
            bounds[idx, 1],
            color=ARENA_COLOR,
            linewidth=BOUND_LINE_WIDTH,
            linestyle="--",
            alpha=0.75,
        )

    axis.set_title("control", fontsize=PANEL_TITLE_FONT_SIZE)
    axis.set_xlabel("step", fontsize=CONTROL_LABEL_FONT_SIZE)
    axis.set_ylabel("u", fontsize=CONTROL_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=CONTROL_LABEL_FONT_SIZE - 1, width=0.6)
    axis.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.4)
    _set_control_limits(axis, u_safe, u_nom, bounds, drop_nominal)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(ARENA_COLOR)


def _draw_arena(axis: plt.Axes, world_lim: float) -> None:
    axis.add_patch(
        Rectangle(
            (-world_lim, -world_lim),
            2.0 * world_lim,
            2.0 * world_lim,
            fill=False,
            edgecolor=ARENA_COLOR,
            linewidth=0.8,
        )
    )
    ticks = np.arange(-int(world_lim), int(world_lim) + 1)
    for tick in ticks:
        axis.axhline(tick, color=GRID_COLOR, linewidth=0.3, alpha=0.3, zorder=0)
        axis.axvline(tick, color=GRID_COLOR, linewidth=0.3, alpha=0.3, zorder=0)


def _draw_obstacles(axis: plt.Axes, scene: Any) -> None:
    for center, radius in zip(
        scene.obstacle_centers[scene.obstacle_active],
        scene.obstacle_radii[scene.obstacle_active],
        strict=True,
    ):
        axis.add_patch(
            Circle(
                center,
                float(radius),
                facecolor=OBSTACLE_FACE,
                edgecolor=OBSTACLE_EDGE,
                linewidth=0.45,
            )
        )


def _draw_obstacle_outlines(axis: plt.Axes, scene: Any) -> None:
    for center, radius in zip(
        scene.obstacle_centers[scene.obstacle_active],
        scene.obstacle_radii[scene.obstacle_active],
        strict=True,
    ):
        axis.add_patch(
            Circle(
                center,
                float(radius),
                facecolor="none",
                edgecolor=OBSTACLE_EDGE,
                linewidth=CONTOUR_OBSTACLE_LINE_WIDTH,
            )
        )


def _draw_baseline(axis: plt.Axes, states: np.ndarray) -> None:
    if states.shape[0] < 2:
        return
    positions = states[:, :2]
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color=TRAJECTORY_INACTIVE,
        linestyle=":",
        linewidth=BASELINE_LINE_WIDTH,
        alpha=BASELINE_ALPHA,
    )


def _draw_filtered_trajectory(
    axis: plt.Axes,
    states: np.ndarray,
    intervention: np.ndarray,
) -> None:
    positions = states[:, :2]
    if positions.shape[0] < 2:
        return
    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    colors = [
        TRAJECTORY_ACTIVE if active else TRAJECTORY_INACTIVE
        for active in intervention
    ]
    axis.add_collection(
        LineCollection(
            segments,
            colors=colors,
            linewidths=LINE_WIDTH,
            linestyles="solid",
        )
    )


def _set_control_limits(
    axis: plt.Axes,
    u_safe: np.ndarray,
    u_nom: np.ndarray,
    bounds: np.ndarray,
    drop_nominal: bool,
) -> None:
    values = [u_safe.reshape(-1), bounds[:, :].reshape(-1)]
    if not drop_nominal and u_nom.shape == u_safe.shape:
        values.append(u_nom.reshape(-1))
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        axis.set_ylim(-1.0, 1.0)
        return
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    margin = max(0.1, 0.08 * (hi - lo if hi > lo else 1.0))
    axis.set_ylim(lo - margin, hi + margin)


def _panel_title(pool_index: int, outcome: str | None) -> str:
    if not outcome:
        return f"{pool_index:02d}"
    return f"{pool_index:02d} · {_outcome_word(outcome)}"


def _outcome_word(outcome: str) -> str:
    words = {
        "goal": "Reach",
        "collision": "Collision",
        "oob": "OOB",
        "stuck": "Stuck",
        "timeout": "Timeout",
    }
    return words[outcome]


def _legend_handles() -> tuple[list[Line2D], list[Line2D]]:
    trajectory = [
        Line2D(
            [0],
            [0],
            marker=START_MARKER,
            color="none",
            markerfacecolor=START_COLOR,
            markeredgecolor=START_COLOR,
            markersize=LEGEND_MARKER_SIZE,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker=GOAL_MARKER,
            color="none",
            markerfacecolor=GOAL_COLOR,
            markeredgecolor=GOAL_COLOR,
            markersize=LEGEND_MARKER_SIZE,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_INACTIVE,
            linewidth=LINE_WIDTH,
            label="Trajectory (CBF inactive)",
        ),
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_ACTIVE,
            linewidth=LINE_WIDTH,
            label="Trajectory (CBF active)",
        ),
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_INACTIVE,
            linestyle=":",
            linewidth=BASELINE_LINE_WIDTH,
            alpha=BASELINE_ALPHA,
            label="Nominal-only (dotted)",
        ),
        Line2D(
            [0],
            [0],
            marker=COLLISION_MARKER,
            color=GOAL_COLOR,
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            label="Collision",
        ),
    ]
    control = [
        Line2D([0], [0], color=CONTROL_SAFE[0], linewidth=CONTROL_LINE_WIDTH, label="u₁ safe"),
        Line2D([0], [0], color=CONTROL_SAFE[1], linewidth=CONTROL_LINE_WIDTH, label="u₂ safe"),
        Line2D(
            [0],
            [0],
            color=CONTROL_SAFE[0],
            linewidth=CONTROL_LINE_WIDTH,
            linestyle="--",
            alpha=CONTROL_NOMINAL_ALPHA,
            label="u₁ nominal",
        ),
        Line2D(
            [0],
            [0],
            color=CONTROL_SAFE[1],
            linewidth=CONTROL_LINE_WIDTH,
            linestyle="--",
            alpha=CONTROL_NOMINAL_ALPHA,
            label="u₂ nominal",
        ),
        Line2D(
            [0],
            [0],
            color=ARENA_COLOR,
            linewidth=BOUND_LINE_WIDTH,
            linestyle="--",
            label="Control bound",
        ),
    ]
    return trajectory, control


def _contour_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=ARENA_COLOR,
            linewidth=CONTOUR_VELOCITY_ARROW_LINE_WIDTH,
            marker=">",
            markersize=LEGEND_MARKER_SIZE,
            label="Fixed velocity",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ARENA_COLOR,
            markeredgecolor="white",
            markersize=LEGEND_MARKER_SIZE,
            label="Zero velocity",
        ),
    ]


def _intervention_mask(mask: Any | None, n_steps: int) -> np.ndarray:
    if mask is None:
        return np.zeros(n_steps, dtype=np.bool_)
    values = _to_numpy(mask).astype(np.bool_).reshape(-1)
    if values.shape[0] != n_steps:
        raise ValueError(f"Intervention mask length {values.shape[0]} != {n_steps}.")
    return values


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _module_device_dtype(value_net: Any, system: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(value_net.parameters())
        return parameter.device, parameter.dtype
    except (AttributeError, StopIteration):
        bounds = system.u_bounds
        dtype = torch.float32 if bounds.dtype == torch.float64 else bounds.dtype
        return bounds.device, dtype


def _contour_velocity_columns(system: Any) -> list[tuple[str, torch.Tensor]]:
    settings = {
        "double_integrator": [
            ("vx=-1.5, vy=0.5", torch.tensor([-1.5, 0.5])),
            ("vx=0, vy=0", torch.tensor([0.0, 0.0])),
            ("vx=1.5, vy=-0.5", torch.tensor([1.5, -0.5])),
        ],
        "unicycle": [
            ("v=1, theta=0", torch.tensor([1.0, 0.0])),
            ("v=0, theta=0", torch.tensor([0.0, 0.0])),
            ("v=1, theta=pi", torch.tensor([1.0, torch.pi])),
        ],
    }
    try:
        return settings[system.name]
    except KeyError as exc:
        raise ValueError(f"No CBF contour velocity settings for {system.name!r}.") from exc


def _contour_states(system: Any, positions: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    builders = {
        "double_integrator": _double_integrator_contour_states,
        "unicycle": _unicycle_contour_states,
    }
    try:
        return builders[system.name](positions, velocity)
    except KeyError as exc:
        raise ValueError(f"No CBF contour state builder for {system.name!r}.") from exc


def _contour_velocity_vector(system: Any, velocity: torch.Tensor) -> torch.Tensor:
    builders = {
        "double_integrator": _double_integrator_velocity_vector,
        "unicycle": _unicycle_velocity_vector,
    }
    try:
        return builders[system.name](velocity)
    except KeyError as exc:
        raise ValueError(f"No CBF contour velocity vector for {system.name!r}.") from exc


def _draw_contour_velocity_indicator(
    axis: plt.Axes,
    velocity_vector: torch.Tensor,
) -> bool:
    vector = _to_numpy(velocity_vector).astype(np.float64)
    if float(np.linalg.norm(vector)) <= 1.0e-12:
        axis.scatter(
            [0.0],
            [0.0],
            s=CONTOUR_ZERO_DOT_SIZE,
            facecolor=ARENA_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=8,
        )
        return False

    endpoint = CONTOUR_VELOCITY_ARROW_SCALE * vector
    arrow = FancyArrowPatch(
        (0.0, 0.0),
        (float(endpoint[0]), float(endpoint[1])),
        arrowstyle="-|>",
        mutation_scale=CONTOUR_VELOCITY_ARROW_HEAD_SIZE,
        linewidth=CONTOUR_VELOCITY_ARROW_LINE_WIDTH,
        color=ARENA_COLOR,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=8,
        path_effects=[
            path_effects.withStroke(
                linewidth=CONTOUR_VELOCITY_ARROW_OUTLINE_WIDTH,
                foreground="white",
            )
        ],
    )
    axis.add_patch(arrow)
    return True


def _double_integrator_contour_states(
    positions: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([positions, velocity.unsqueeze(0).expand(positions.shape[0], -1)], dim=1)


def _unicycle_contour_states(
    positions: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    speed = velocity[0].expand(positions.shape[0])
    theta = velocity[1].expand(positions.shape[0])
    return torch.stack([positions[:, 0], positions[:, 1], theta, speed], dim=1)


def _double_integrator_velocity_vector(velocity: torch.Tensor) -> torch.Tensor:
    return velocity


def _unicycle_velocity_vector(velocity: torch.Tensor) -> torch.Tensor:
    speed = velocity[0]
    theta = velocity[1]
    return torch.stack([speed * torch.cos(theta), speed * torch.sin(theta)])
