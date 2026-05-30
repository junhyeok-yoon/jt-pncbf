from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms
from src.common.value_net import make_h_fn
from src.envs.scene_init import Scene
from src.envs.scene_init_eval import sample_eval_scene
from src.eval.build_pools import EvaluationPool
from src.eval.evaluate import active_action_steps
from src.eval.evaluate import evaluate
from src.eval.evaluate import first_physical_event_step
from src.common.outcomes import step_outcomes
from src.eval.run_full import _load_framework


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = (
    REPO_ROOT
    / "data/v2.0.1__20260529-171057__seed42/checkpoints/best.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/diagnostics"


@dataclass(frozen=True)
class EpisodeSummary:
    episode_idx: int
    saved_role: str
    outcome: str
    event_step: int
    active_steps: int
    min_clearance: float
    event_clearance: float
    event_h: float
    event_lg_norm: float
    event_closing_speed: float
    event_brake_margin: float
    event_empty_halfspace_box: bool
    pre10_mean_h: float
    pre10_max_h: float
    pre10_min_clearance: float
    pre10_mean_closing_speed: float
    pre10_min_brake_margin: float
    pre10_mean_projection: float
    pre10_saturation_frac: float
    pre10_intervention_frac: float
    pre10_empty_frac: float
    obstacle_count: int
    nearest_pair_gap: float
    obstacle_goal_clearance: float
    obstacle_line_min_clearance: float
    start_goal_dist: float
    mechanism: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=45678)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--goal-controls", type=int, default=120)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.n < 2000:
        raise ValueError("--n must be at least 2000 for this diagnostic.")

    framework, config, checkpoint = _load_framework(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    system_name = str(config["run"]["system"])
    scenes = [sample_eval_scene(rng, config, system_name) for _ in range(args.n)]
    pool = EvaluationPool(
        name=f"failure_dissect_n{args.n}_seed{args.seed}",
        system=system_name,
        n_scenes=args.n,
        seed=args.seed,
        scenes=scenes,
    )

    out_dir = args.output_root / (
        f"v2.0.1__20260529-171057__failure_dissect_"
        f"n{args.n}_seed{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    episode_dir = out_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    eval_result = evaluate(
        framework,
        pool,
        config,
        mode="diagnostic_failure_dissect",
        step=int(checkpoint["step"]),
        ckpt_name=args.checkpoint.name,
        include_lqr_baseline=False,
        eval_batch_size=args.batch_size,
    )

    outcome_indices: dict[str, list[int]] = {}
    for idx, episode in enumerate(eval_result.trajectories):
        outcome_indices.setdefault(episode.filtered_outcome, []).append(idx)

    failure_indices = outcome_indices.get("collision", []) + outcome_indices.get("stuck", [])
    control_indices = _matched_goal_controls(
        eval_result.trajectories,
        failure_indices,
        outcome_indices.get("goal", []),
        args.goal_controls,
    )
    saved_indices = set(failure_indices) | set(control_indices)

    h_fn = make_h_fn(framework.value_net, framework.system, use_target=False)
    summaries: list[EpisodeSummary] = []
    for episode_idx in sorted(saved_indices):
        episode = eval_result.trajectories[episode_idx]
        role = (
            "failure"
            if episode.filtered_outcome in {"collision", "stuck"}
            else "goal_control"
        )
        trace = _episode_trace(
            framework=framework,
            h_fn=h_fn,
            config=config,
            scene=episode.scene,
            result=episode.filtered,
            outcome=episode.filtered_outcome,
            event_step=int(episode.filtered_event_step),
        )
        mechanism = _classify_mechanism(episode.filtered_outcome, trace)
        summary = _episode_summary(
            episode_idx=episode_idx,
            role=role,
            outcome=episode.filtered_outcome,
            event_step=int(episode.filtered_event_step),
            trace=trace,
            scene=episode.scene,
            mechanism=mechanism,
        )
        summaries.append(summary)
        _write_episode_npz(episode_dir, episode_idx, episode.filtered_outcome, role, trace)

    _write_csv(out_dir / "episode_summary.csv", summaries)
    _write_mechanism_csv(out_dir / "collision_mechanisms.csv", summaries, "collision")
    _write_mechanism_csv(out_dir / "stuck_mechanisms.csv", summaries, "stuck")
    _write_summary_json(
        out_dir / "summary.json",
        args=args,
        checkpoint_step=int(checkpoint["step"]),
        eval_row=eval_result.eval_row,
        outcome_indices=outcome_indices,
        failure_indices=failure_indices,
        control_indices=control_indices,
        summaries=summaries,
    )
    _write_representative_traces(out_dir / "representative_traces.md", summaries, episode_dir)
    _write_analysis_md(out_dir / "analysis.md", eval_result.eval_row, summaries)

    with (out_dir / "sampled_scenes.pkl").open("wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(out_dir)
    print(
        "outcomes="
        + ", ".join(f"{k}:{len(v)}" for k, v in sorted(outcome_indices.items()))
    )
    print(f"saved_failures={len(failure_indices)} saved_controls={len(control_indices)}")


def _matched_goal_controls(
    trajectories: list[Any],
    failure_indices: list[int],
    goal_indices: list[int],
    max_controls: int,
) -> list[int]:
    if not goal_indices or not failure_indices:
        return goal_indices[:max_controls]
    failure_features = np.stack([
        _scene_match_features(trajectories[idx].scene) for idx in failure_indices
    ])
    goal_features = np.stack([
        _scene_match_features(trajectories[idx].scene) for idx in goal_indices
    ])
    chosen: list[int] = []
    used: set[int] = set()
    for f in failure_features:
        d = np.linalg.norm((goal_features - f).astype(np.float64), axis=1)
        for order_idx in np.argsort(d):
            goal_idx = goal_indices[int(order_idx)]
            if goal_idx not in used:
                chosen.append(goal_idx)
                used.add(goal_idx)
                break
        if len(chosen) >= max_controls:
            break
    if len(chosen) < min(max_controls, len(goal_indices)):
        for goal_idx in goal_indices:
            if goal_idx not in used:
                chosen.append(goal_idx)
                used.add(goal_idx)
            if len(chosen) >= max_controls:
                break
    return chosen


def _scene_match_features(scene: Scene) -> np.ndarray:
    centers = np.asarray(scene.obstacle_centers, dtype=np.float64)
    radii = np.asarray(scene.obstacle_radii, dtype=np.float64)
    active = np.asarray(scene.obstacle_active, dtype=bool)
    start = np.asarray(scene.start, dtype=np.float64)
    goal = np.asarray(scene.goal, dtype=np.float64)
    active_centers = centers[active]
    active_radii = radii[active]
    start_goal = np.linalg.norm(goal - start)
    if active_centers.size == 0:
        return np.asarray([start_goal, 999.0, 999.0, 0.0], dtype=np.float64)
    start_clearance = np.min(np.linalg.norm(active_centers - start[None, :], axis=1) - active_radii)
    goal_clearance = np.min(np.linalg.norm(active_centers - goal[None, :], axis=1) - active_radii)
    return np.asarray([start_goal, start_clearance, goal_clearance, float(active.sum())])


def _episode_trace(
    *,
    framework: Any,
    h_fn: Any,
    config: dict[str, Any],
    scene: Scene,
    result: Any,
    outcome: str,
    event_step: int,
) -> dict[str, np.ndarray | float | str | int]:
    system = framework.system
    states = result.states[:, 0, :].detach()
    u_nom = result.u_nom[:, 0, :].detach()
    u_safe = result.u_safe[:, 0, :].detach()
    steps = u_safe.shape[0]
    x = states[:-1]
    if x.shape[0] != steps:
        raise ValueError("state/action length mismatch")

    h, lf_h, lg_h = _cbf_terms(system, h_fn, x, scene, u_nom)
    alpha = torch.where(
        h <= 0.0,
        torch.full_like(h, float(config["filter"]["alpha_safe"])),
        torch.full_like(h, float(config["filter"]["alpha_unsafe"])),
    )
    row_upper = -lf_h - alpha * h
    row_lhs_nom = torch.sum(lg_h * u_nom, dim=1)
    row_lhs_safe = torch.sum(lg_h * u_safe, dim=1)
    cbf_violation_nom = torch.relu(row_lhs_nom - row_upper)
    cbf_violation_safe = torch.relu(row_lhs_safe - row_upper)
    lg_norm = torch.linalg.norm(lg_h, dim=1)
    hdot_model = lf_h + row_lhs_safe
    dt = float(config["env"]["dt"])
    h_next = h_fn(states[1:], scene).reshape(-1).detach()
    hdot_fd = (h_next - h) / dt
    projection = torch.linalg.norm(u_safe - u_nom, dim=1)
    intervention = projection > 1.0e-3
    bounds = system.u_bounds.to(device=u_safe.device, dtype=u_safe.dtype)
    saturated = torch.any(
        torch.minimum(torch.abs(u_safe - bounds[:, 0]), torch.abs(u_safe - bounds[:, 1]))
        <= 1.0e-3,
        dim=1,
    )
    empty_box = _empty_halfspace_box(lg_h, row_upper, bounds)

    positions = system.position(x)
    velocities = _velocity(system, x)
    geom = _nearest_obstacle_geometry(scene, positions, velocities)
    goal_geom = _goal_geometry(scene, positions, velocities)

    masks = step_outcomes(result.states, scene, system, config)
    physical_event = int(first_physical_event_step(masks)[0].item())
    active_steps = active_action_steps(physical_event, steps)

    return {
        "outcome": outcome,
        "event_step": int(event_step),
        "physical_event_step": int(physical_event),
        "active_steps": int(active_steps),
        "dt": float(dt),
        "states": _np(states),
        "u_nom": _np(u_nom),
        "u_safe": _np(u_safe),
        "projection_norm": _np(projection),
        "intervention": _np(intervention),
        "infeasible": _np(result.infeasible[:, 0].detach()),
        "h": _np(h),
        "h_next": _np(h_next),
        "h_dot_model": _np(hdot_model),
        "h_dot_fd": _np(hdot_fd),
        "lf_h": _np(lf_h),
        "lg_h": _np(lg_h),
        "lg_norm": _np(lg_norm),
        "row_upper": _np(row_upper),
        "row_lhs_nom": _np(row_lhs_nom),
        "row_lhs_safe": _np(row_lhs_safe),
        "cbf_violation_nom": _np(cbf_violation_nom),
        "cbf_violation_safe": _np(cbf_violation_safe),
        "empty_halfspace_box": _np(empty_box),
        "saturated": _np(saturated),
        "nearest_obstacle_idx": geom["idx"],
        "nearest_obstacle_center": geom["center"],
        "nearest_obstacle_radius": geom["radius"],
        "nearest_rel_pos": geom["rel_pos"],
        "nearest_rel_vel": geom["rel_vel"],
        "nearest_distance": geom["distance"],
        "clearance": geom["clearance"],
        "closing_speed": geom["closing_speed"],
        "bearing_to_velocity_cos": geom["bearing_to_velocity_cos"],
        "brake_margin": geom["brake_margin"],
        "goal_rel_pos": goal_geom["rel_pos"],
        "goal_distance": goal_geom["distance"],
        "goal_closing_speed": goal_geom["closing_speed"],
        "obstacle_centers": np.asarray(scene.obstacle_centers, dtype=np.float32),
        "obstacle_radii": np.asarray(scene.obstacle_radii, dtype=np.float32),
        "obstacle_active": np.asarray(scene.obstacle_active, dtype=bool),
        "start": np.asarray(scene.start, dtype=np.float32),
        "goal": np.asarray(scene.goal, dtype=np.float32),
        "initial_velocity": np.asarray(scene.initial_velocity, dtype=np.float32),
        "scene_geometry": _scene_geometry(scene),
    }


def _velocity(system: Any, states: torch.Tensor) -> torch.Tensor:
    if system.name == "double_integrator":
        return states[:, 2:4]
    speed = system.speed(states)
    theta = states[:, 2]
    return torch.stack([speed * torch.cos(theta), speed * torch.sin(theta)], dim=1)


def _nearest_obstacle_geometry(
    scene: Scene,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> dict[str, np.ndarray]:
    device = positions.device
    dtype = positions.dtype
    centers = torch.as_tensor(scene.obstacle_centers, device=device, dtype=dtype)
    radii = torch.as_tensor(scene.obstacle_radii, device=device, dtype=dtype)
    active = torch.as_tensor(scene.obstacle_active, device=device, dtype=torch.bool)
    active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
    active_centers = centers[active]
    active_radii = radii[active]
    rel = positions.unsqueeze(1) - active_centers.unsqueeze(0)
    distance = torch.linalg.norm(rel, dim=2).clamp_min(1.0e-8)
    clearance_all = distance - active_radii.unsqueeze(0)
    local_idx = torch.argmin(clearance_all, dim=1)
    idx = active_idx[local_idx]
    batch = torch.arange(positions.shape[0], device=device)
    rel_pos = rel[batch, local_idx]
    nearest_distance = distance[batch, local_idx]
    nearest_radius = active_radii[local_idx]
    nearest_center = active_centers[local_idx]
    outward = rel_pos / nearest_distance.unsqueeze(1)
    radial_v_out = torch.sum(velocities * outward, dim=1)
    closing_speed = torch.relu(-radial_v_out)
    speed_norm = torch.linalg.norm(velocities, dim=1).clamp_min(1.0e-8)
    bearing_to_velocity_cos = torch.sum(velocities * (-outward), dim=1) / speed_norm
    clearance = nearest_distance - nearest_radius
    u_max = 2.0
    brake_distance = closing_speed * closing_speed / (2.0 * u_max)
    brake_margin = clearance - brake_distance
    return {
        "idx": _np(idx),
        "center": _np(nearest_center),
        "radius": _np(nearest_radius),
        "rel_pos": _np(rel_pos),
        "rel_vel": _np(velocities),
        "distance": _np(nearest_distance),
        "clearance": _np(clearance),
        "closing_speed": _np(closing_speed),
        "bearing_to_velocity_cos": _np(bearing_to_velocity_cos),
        "brake_margin": _np(brake_margin),
    }


def _goal_geometry(
    scene: Scene,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> dict[str, np.ndarray]:
    goal = torch.as_tensor(scene.goal, device=positions.device, dtype=positions.dtype)
    rel = goal.unsqueeze(0) - positions
    dist = torch.linalg.norm(rel, dim=1).clamp_min(1.0e-8)
    direction = rel / dist.unsqueeze(1)
    closing = torch.sum(velocities * direction, dim=1)
    return {"rel_pos": _np(rel), "distance": _np(dist), "closing_speed": _np(closing)}


def _empty_halfspace_box(
    normal: torch.Tensor,
    row_upper: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    minimizing_corner = torch.where(normal >= 0.0, bounds[:, 0], bounds[:, 1])
    min_lhs = torch.sum(normal * minimizing_corner, dim=1)
    return min_lhs > row_upper + 1.0e-9


def _scene_geometry(scene: Scene) -> dict[str, float | int]:
    centers = np.asarray(scene.obstacle_centers, dtype=np.float64)
    radii = np.asarray(scene.obstacle_radii, dtype=np.float64)
    active = np.asarray(scene.obstacle_active, dtype=bool)
    c = centers[active]
    r = radii[active]
    start = np.asarray(scene.start, dtype=np.float64)
    goal = np.asarray(scene.goal, dtype=np.float64)
    if len(c) == 0:
        return {
            "obstacle_count": 0,
            "nearest_pair_gap": float("inf"),
            "obstacle_goal_clearance": float("inf"),
            "obstacle_line_min_clearance": float("inf"),
            "start_goal_dist": float(np.linalg.norm(goal - start)),
        }
    pair_gap = float("inf")
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            pair_gap = min(pair_gap, float(np.linalg.norm(c[i] - c[j]) - r[i] - r[j]))
    goal_clearance = float(np.min(np.linalg.norm(c - goal[None, :], axis=1) - r))
    line_clearance = float(np.min([
        _point_segment_distance(center, start, goal) - radius
        for center, radius in zip(c, r)
    ]))
    return {
        "obstacle_count": int(len(c)),
        "nearest_pair_gap": pair_gap,
        "obstacle_goal_clearance": goal_clearance,
        "obstacle_line_min_clearance": line_clearance,
        "start_goal_dist": float(np.linalg.norm(goal - start)),
    }


def _point_segment_distance(point: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    seg = goal - start
    denom = float(np.dot(seg, seg))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    tau = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
    closest = start + tau * seg
    return float(np.linalg.norm(point - closest))


def _classify_mechanism(outcome: str, trace: dict[str, Any]) -> str:
    if outcome == "collision":
        return _classify_collision(trace)
    if outcome == "stuck":
        return _classify_stuck(trace)
    return "goal_control"


def _classify_collision(trace: dict[str, Any]) -> str:
    event = _event_action_index(trace)
    window = _window(event, len(trace["h"]), before=10)
    h = trace["h"]
    projection = trace["projection_norm"]
    sat = trace["saturated"]
    brake_margin = trace["brake_margin"]
    lg_norm = trace["lg_norm"]
    empty = trace["empty_halfspace_box"]
    clearance = trace["clearance"]
    closing = trace["closing_speed"]
    danger_candidates = np.where((clearance < 0.35) | (brake_margin < 0.0))[0]
    danger = int(danger_candidates[0]) if len(danger_candidates) else int(window.start)
    if bool(empty[window].mean() > 0.3):
        return "box-infeasible-cbf-constraint"
    if float(np.nanmin(lg_norm[window])) < 5.0e-4:
        return "degenerate-Lg-control-authority"
    if float(brake_margin[danger]) < 0.0 and float(h[danger]) <= 0.0:
        return "late-hazard-recognition-false-safe"
    if float(projection[window].mean()) > 1.0 and float(sat[window].mean()) > 0.5:
        return "committed-high-speed-filter-saturated"
    if float(closing[window].mean()) > 1.0 and float(brake_margin[window].min()) < 0.05:
        return "committed-high-speed-near-obstacle"
    return "mixed-collision"


def _classify_stuck(trace: dict[str, Any]) -> str:
    event = _event_action_index(trace)
    window = _window(event, len(trace["h"]), before=60)
    h = trace["h"]
    projection = trace["projection_norm"]
    sat = trace["saturated"]
    u_nom = trace["u_nom"]
    u_safe = trace["u_safe"]
    clearance = trace["clearance"]
    if float(np.linalg.norm(u_nom[window], axis=1).mean()) < 0.25 and float(projection[window].mean()) < 0.25:
        return "policy-stagnation-low-action"
    if float(projection[window].mean()) > 1.0 and float(sat[window].mean()) > 0.5:
        if float(clearance[window].min()) < 0.35:
            return "near-obstacle-projection-saturation-trap"
        return "overconservative-projection-saturation-trap"
    sign_changes = _action_sign_change_rate(u_safe[window])
    if sign_changes > 0.35 and float(np.linalg.norm(u_safe[window], axis=1).mean()) > 0.5:
        return "oscillatory-chattering"
    if float(np.nanmax(h[window])) > 0.0 and float(projection[window].mean()) > 0.5:
        return "positive-h-blocking-progress"
    return "geometry-or-low-progress-mixed"


def _event_action_index(trace: dict[str, Any]) -> int:
    event = int(trace["event_step"])
    steps = len(trace["h"])
    if event < 0:
        return steps - 1
    if str(trace.get("outcome", "")) in {"collision", "goal", "oob"} and event > 0:
        return max(0, min(event - 1, steps - 1))
    return max(0, min(event, steps - 1))


def _window(event: int, length: int, *, before: int) -> slice:
    return slice(max(0, event - before + 1), min(length, event + 1))


def _action_sign_change_rate(actions: np.ndarray) -> float:
    if actions.shape[0] < 2:
        return 0.0
    signs = np.sign(actions)
    changes = np.any(signs[1:] * signs[:-1] < 0, axis=1)
    return float(changes.mean())


def _episode_summary(
    *,
    episode_idx: int,
    role: str,
    outcome: str,
    event_step: int,
    trace: dict[str, Any],
    scene: Scene,
    mechanism: str,
) -> EpisodeSummary:
    event = _event_action_index(trace)
    pre = _window(event, len(trace["h"]), before=10)
    geom = trace["scene_geometry"]
    return EpisodeSummary(
        episode_idx=episode_idx,
        saved_role=role,
        outcome=outcome,
        event_step=event_step,
        active_steps=int(trace["active_steps"]),
        min_clearance=float(np.min(trace["clearance"][: max(1, int(trace["active_steps"]))])),
        event_clearance=float(trace["clearance"][event]),
        event_h=float(trace["h"][event]),
        event_lg_norm=float(trace["lg_norm"][event]),
        event_closing_speed=float(trace["closing_speed"][event]),
        event_brake_margin=float(trace["brake_margin"][event]),
        event_empty_halfspace_box=bool(trace["empty_halfspace_box"][event]),
        pre10_mean_h=float(np.mean(trace["h"][pre])),
        pre10_max_h=float(np.max(trace["h"][pre])),
        pre10_min_clearance=float(np.min(trace["clearance"][pre])),
        pre10_mean_closing_speed=float(np.mean(trace["closing_speed"][pre])),
        pre10_min_brake_margin=float(np.min(trace["brake_margin"][pre])),
        pre10_mean_projection=float(np.mean(trace["projection_norm"][pre])),
        pre10_saturation_frac=float(np.mean(trace["saturated"][pre])),
        pre10_intervention_frac=float(np.mean(trace["intervention"][pre])),
        pre10_empty_frac=float(np.mean(trace["empty_halfspace_box"][pre])),
        obstacle_count=int(geom["obstacle_count"]),
        nearest_pair_gap=float(geom["nearest_pair_gap"]),
        obstacle_goal_clearance=float(geom["obstacle_goal_clearance"]),
        obstacle_line_min_clearance=float(geom["obstacle_line_min_clearance"]),
        start_goal_dist=float(geom["start_goal_dist"]),
        mechanism=mechanism,
    )


def _write_episode_npz(
    episode_dir: Path,
    episode_idx: int,
    outcome: str,
    role: str,
    trace: dict[str, Any],
) -> None:
    path = episode_dir / f"{role}_{episode_idx:05d}_{outcome}.npz"
    arrays = {
        key: value
        for key, value in trace.items()
        if isinstance(value, np.ndarray)
    }
    metadata = {
        key: value
        for key, value in trace.items()
        if not isinstance(value, np.ndarray)
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _write_csv(path: Path, summaries: list[EpisodeSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def _write_mechanism_csv(
    path: Path,
    summaries: list[EpisodeSummary],
    outcome: str,
) -> None:
    rows = [s for s in summaries if s.outcome == outcome]
    counts: dict[str, list[EpisodeSummary]] = {}
    for row in rows:
        counts.setdefault(row.mechanism, []).append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "outcome",
            "mechanism",
            "count",
            "mean_event_h",
            "mean_event_clearance",
            "mean_event_closing_speed",
            "mean_event_brake_margin",
            "mean_pre10_projection",
            "mean_pre10_saturation",
            "representatives",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mechanism, group in sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            writer.writerow({
                "outcome": outcome,
                "mechanism": mechanism,
                "count": len(group),
                "mean_event_h": np.mean([g.event_h for g in group]),
                "mean_event_clearance": np.mean([g.event_clearance for g in group]),
                "mean_event_closing_speed": np.mean([g.event_closing_speed for g in group]),
                "mean_event_brake_margin": np.mean([g.event_brake_margin for g in group]),
                "mean_pre10_projection": np.mean([g.pre10_mean_projection for g in group]),
                "mean_pre10_saturation": np.mean([g.pre10_saturation_frac for g in group]),
                "representatives": " ".join(str(g.episode_idx) for g in group[:3]),
            })


def _write_summary_json(
    path: Path,
    *,
    args: argparse.Namespace,
    checkpoint_step: int,
    eval_row: dict[str, Any],
    outcome_indices: dict[str, list[int]],
    failure_indices: list[int],
    control_indices: list[int],
    summaries: list[EpisodeSummary],
) -> None:
    mechanism_counts: dict[str, dict[str, int]] = {}
    for summary in summaries:
        if summary.outcome not in {"collision", "stuck"}:
            continue
        mechanism_counts.setdefault(summary.outcome, {})
        mechanism_counts[summary.outcome][summary.mechanism] = (
            mechanism_counts[summary.outcome].get(summary.mechanism, 0) + 1
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "n_rollouts": args.n,
        "seed": args.seed,
        "eval_row": eval_row,
        "outcome_counts": {key: len(value) for key, value in outcome_indices.items()},
        "failure_count": len(failure_indices),
        "goal_control_count": len(control_indices),
        "mechanism_counts": mechanism_counts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_representative_traces(
    path: Path,
    summaries: list[EpisodeSummary],
    episode_dir: Path,
) -> None:
    lines = ["# Representative Failure Traces", ""]
    for outcome in ("collision", "stuck"):
        lines.append(f"## {outcome.title()}")
        groups: dict[str, list[EpisodeSummary]] = {}
        for summary in summaries:
            if summary.outcome == outcome:
                groups.setdefault(summary.mechanism, []).append(summary)
        for mechanism, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"### {mechanism} ({len(group)})")
            for summary in group[:2]:
                npz = next(episode_dir.glob(f"*_{summary.episode_idx:05d}_{outcome}.npz"))
                trace = np.load(npz, allow_pickle=False)
                event = max(0, min(summary.event_step, len(trace["h"]) - 1))
                rows = []
                for t in range(max(0, event - 5), event + 1):
                    rows.append(
                        "| "
                        + " | ".join(
                            [
                                str(t),
                                f"{trace['clearance'][t]:.3f}",
                                f"{trace['closing_speed'][t]:.3f}",
                                f"{trace['brake_margin'][t]:.3f}",
                                f"{trace['h'][t]:.3f}",
                                f"{trace['lf_h'][t]:.3f}",
                                f"{np.linalg.norm(trace['lg_h'][t]):.4f}",
                                f"{trace['projection_norm'][t]:.3f}",
                                str(bool(trace["saturated"][t])),
                            ]
                        )
                        + " |"
                    )
                lines.extend(
                    [
                        f"- episode `{summary.episode_idx}`, event step `{summary.event_step}`, file `{npz.name}`",
                        "",
                        "| t | clearance | closing | brake_margin | h | Lf h | ||Lg h|| | proj | sat |",
                        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                        *rows,
                        "",
                    ]
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis_md(
    path: Path,
    eval_row: dict[str, Any],
    summaries: list[EpisodeSummary],
) -> None:
    lines = [
        "# JT Failure Dissection",
        "",
        "## Aggregate rollout",
        "",
        f"- n_scenes: `{eval_row['n_scenes']}`",
        f"- cps: `{float(eval_row['cps']):.6f}`",
        f"- reach: `{float(eval_row['reach']):.6f}`",
        f"- collision: `{float(eval_row['collision']):.6f}`",
        f"- stuck: `{float(eval_row['stuck']):.6f}`",
        f"- timeout: `{float(eval_row['timeout']):.6f}`",
        f"- infeasibility: `{float(eval_row['infeasibility']):.6f}`",
        f"- saturation_rate: `{float(eval_row['saturation_rate']):.6f}`",
        "",
    ]
    for outcome in ("collision", "stuck"):
        rows = [s for s in summaries if s.outcome == outcome]
        lines.extend([f"## {outcome.title()} mechanisms", ""])
        groups: dict[str, list[EpisodeSummary]] = {}
        for row in rows:
            groups.setdefault(row.mechanism, []).append(row)
        lines.append("| mechanism | count | event_h | event_clearance | closing | brake_margin | pre10_projection | pre10_sat |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mechanism, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(
                f"| {mechanism} | {len(group)} | "
                f"{np.mean([g.event_h for g in group]):.3f} | "
                f"{np.mean([g.event_clearance for g in group]):.3f} | "
                f"{np.mean([g.event_closing_speed for g in group]):.3f} | "
                f"{np.mean([g.event_brake_margin for g in group]):.3f} | "
                f"{np.mean([g.pre10_mean_projection for g in group]):.3f} | "
                f"{np.mean([g.pre10_saturation_frac for g in group]):.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    main()
