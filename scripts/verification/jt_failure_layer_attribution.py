from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DIAGNOSTIC_DIR = Path(
    "data/diagnostics/"
    "v2.0.1__20260529-171057__failure_dissect_n2500_seed45678_20260529-194915"
)
K_OBS = 5
WORLD_LIM = 4.0
PATH_MARGIN = 0.15
GRID_STEP = 0.08
LARGE_DETOUR_RATIO = 1.30
LARGE_DETOUR_ANGLE_DEG = 60.0
LOCAL_DETOUR_RATIO = 1.20
LOCAL_DETOUR_ANGLE_DEG = 45.0


@dataclass(frozen=True)
class Attribution:
    episode_idx: int
    outcome: str
    layer: str
    representative: bool
    culprit_observed: bool
    culprit_topk_frac: float
    all_near_obstacles_topk_frac: float
    active_obstacles: int
    culprit_obstacle_idx: int
    line_clearance: float
    start_goal_dist: float
    start_goal_path_len: float
    start_goal_detour_ratio: float
    start_goal_initial_turn_deg: float
    event_goal_path_len: float
    event_goal_detour_ratio: float
    event_goal_initial_turn_deg: float
    large_detour_scene: bool
    large_detour_from_event: bool
    empty_onset_step: int
    brake_onset_step: int
    event_action_step: int
    steps_empty_to_event: int
    steps_brake_to_event: int
    h_at_empty_onset: float
    h_at_brake_onset: float
    h_at_event_action: float
    clearance_at_event_action: float
    closing_at_event_action: float
    brake_margin_at_event_action: float
    pre_window_projection: float
    pre_window_saturation: float
    stuck_sign_change_rate: float
    stuck_net_displacement: float
    stuck_path_length: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    args = parser.parse_args()
    diag_dir = args.diagnostic_dir
    episode_dir = diag_dir / "episodes"
    if not episode_dir.exists():
        raise FileNotFoundError(episode_dir)

    rows: list[Attribution] = []
    for path in sorted(episode_dir.glob("failure_*.npz")):
        trace = np.load(path, allow_pickle=False)
        outcome = _metadata(trace)["outcome"]
        if outcome not in {"collision", "stuck"}:
            continue
        rows.append(_attribute_episode(path, trace, outcome))

    rows.sort(key=lambda row: (row.outcome, row.episode_idx))
    _write_csv(diag_dir / "layer_attribution.csv", rows)
    _write_summary(diag_dir / "layer_attribution_summary.md", rows)
    print(diag_dir / "layer_attribution_summary.md")
    print(_counts_line(rows))


def _attribute_episode(path: Path, trace: Any, outcome: str) -> Attribution:
    episode_idx = int(path.name.split("_")[1])
    event_action = _event_action_step(trace, outcome)
    active_obstacles = int(np.asarray(trace["obstacle_active"], dtype=bool).sum())

    if outcome == "collision":
        culprit_idx = int(trace["nearest_obstacle_idx"][event_action])
        obs = _topk_stats(trace, culprit_idx, slice(max(0, event_action - 10 + 1), event_action + 1))
    else:
        w = _window(event_action, len(trace["h"]), before=60)
        culprit_idx = _dominant_nearest_idx(trace, w)
        obs = _topk_stats(trace, culprit_idx, w)

    scene_detour = _path_stats(
        trace["start"],
        trace["goal"],
        trace["obstacle_centers"],
        trace["obstacle_radii"],
        trace["obstacle_active"],
        PATH_MARGIN,
    )
    event_position = trace["states"][event_action, :2]
    event_detour = _path_stats(
        event_position,
        trace["goal"],
        trace["obstacle_centers"],
        trace["obstacle_radii"],
        trace["obstacle_active"],
        PATH_MARGIN,
    )
    line_clearance = _line_clearance(trace)

    empty_steps = np.where(trace["empty_halfspace_box"][: event_action + 1])[0]
    brake_steps = np.where(trace["brake_margin"][: event_action + 1] < 0.0)[0]
    empty_onset = int(empty_steps[0]) if empty_steps.size else -1
    brake_onset = int(brake_steps[0]) if brake_steps.size else -1

    pre = _window(event_action, len(trace["h"]), before=10 if outcome == "collision" else 60)
    sign_rate = _action_sign_change_rate(trace["u_safe"][pre])
    net_disp = _net_displacement(trace["states"][pre, :2])
    path_len = _path_length(trace["states"][pre, :2])

    large_scene = _large_detour(scene_detour, line_clearance)
    large_event = _large_detour(event_detour, line_clearance=None)
    layer, note = _assign_layer(
        outcome=outcome,
        observed=obs["culprit_observed"],
        large_scene=large_scene,
        large_event=large_event,
        empty_onset=empty_onset,
        brake_onset=brake_onset,
        event_action=event_action,
        sign_rate=sign_rate,
        pre_projection=float(np.mean(trace["projection_norm"][pre])),
        pre_saturation=float(np.mean(trace["saturated"][pre])),
    )

    return Attribution(
        episode_idx=episode_idx,
        outcome=outcome,
        layer=layer,
        representative=False,
        culprit_observed=bool(obs["culprit_observed"]),
        culprit_topk_frac=float(obs["culprit_topk_frac"]),
        all_near_obstacles_topk_frac=float(obs["all_near_obstacles_topk_frac"]),
        active_obstacles=active_obstacles,
        culprit_obstacle_idx=culprit_idx,
        line_clearance=float(line_clearance),
        start_goal_dist=float(np.linalg.norm(trace["goal"] - trace["start"])),
        start_goal_path_len=float(scene_detour["path_len"]),
        start_goal_detour_ratio=float(scene_detour["ratio"]),
        start_goal_initial_turn_deg=float(scene_detour["initial_turn_deg"]),
        event_goal_path_len=float(event_detour["path_len"]),
        event_goal_detour_ratio=float(event_detour["ratio"]),
        event_goal_initial_turn_deg=float(event_detour["initial_turn_deg"]),
        large_detour_scene=bool(large_scene),
        large_detour_from_event=bool(large_event),
        empty_onset_step=empty_onset,
        brake_onset_step=brake_onset,
        event_action_step=event_action,
        steps_empty_to_event=int(event_action - empty_onset) if empty_onset >= 0 else -1,
        steps_brake_to_event=int(event_action - brake_onset) if brake_onset >= 0 else -1,
        h_at_empty_onset=float(trace["h"][empty_onset]) if empty_onset >= 0 else float("nan"),
        h_at_brake_onset=float(trace["h"][brake_onset]) if brake_onset >= 0 else float("nan"),
        h_at_event_action=float(trace["h"][event_action]),
        clearance_at_event_action=float(trace["clearance"][event_action]),
        closing_at_event_action=float(trace["closing_speed"][event_action]),
        brake_margin_at_event_action=float(trace["brake_margin"][event_action]),
        pre_window_projection=float(np.mean(trace["projection_norm"][pre])),
        pre_window_saturation=float(np.mean(trace["saturated"][pre])),
        stuck_sign_change_rate=float(sign_rate),
        stuck_net_displacement=float(net_disp),
        stuck_path_length=float(path_len),
        note=note,
    )


def _metadata(trace: Any) -> dict[str, Any]:
    return json.loads(str(np.asarray(trace["metadata_json"]).item()))


def _event_action_step(trace: Any, outcome: str) -> int:
    event = int(_metadata(trace)["event_step"])
    steps = len(trace["h"])
    if outcome in {"collision", "goal", "oob"} and event > 0:
        return max(0, min(event - 1, steps - 1))
    if event < 0:
        return steps - 1
    return max(0, min(event, steps - 1))


def _window(event: int, length: int, *, before: int) -> slice:
    return slice(max(0, event - before + 1), min(length, event + 1))


def _dominant_nearest_idx(trace: Any, window: slice) -> int:
    values, counts = np.unique(trace["nearest_obstacle_idx"][window], return_counts=True)
    return int(values[np.argmax(counts)])


def _topk_stats(trace: Any, culprit_idx: int, window: slice) -> dict[str, float | bool]:
    positions = trace["states"][:-1, :2]
    centers = trace["obstacle_centers"]
    radii = trace["obstacle_radii"]
    active = trace["obstacle_active"].astype(bool)
    active_indices = np.nonzero(active)[0]
    surface = np.linalg.norm(
        centers[active][None, :, :] - positions[:, None, :],
        axis=2,
    ) - radii[active][None, :]
    order_local = np.argsort(surface, axis=1, kind="stable")[:, :K_OBS]
    order_global = active_indices[order_local]
    w_indices = np.arange(len(positions))[window]
    culprit_in = np.any(order_global[w_indices] == int(culprit_idx), axis=1)

    near_sets = []
    for t in w_indices:
        near = set(np.nonzero((surface[t] < 0.35))[0])
        if not near:
            near = {int(np.argmin(surface[t]))}
        near_global = {int(active_indices[idx]) for idx in near}
        top_global = set(int(x) for x in order_global[t])
        near_sets.append(float(near_global.issubset(top_global)))
    return {
        "culprit_topk_frac": float(culprit_in.mean()) if culprit_in.size else 0.0,
        "culprit_observed": bool(culprit_in.mean() >= 0.8) if culprit_in.size else False,
        "all_near_obstacles_topk_frac": float(np.mean(near_sets)) if near_sets else 0.0,
    }


def _path_stats(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    active: np.ndarray,
    margin: float,
) -> dict[str, float]:
    straight = float(np.linalg.norm(goal - start))
    path = _astar_path(start, goal, centers[active.astype(bool)], radii[active.astype(bool)], margin)
    if path is None:
        return {
            "path_len": float("inf"),
            "ratio": float("inf"),
            "initial_turn_deg": float("inf"),
        }
    path_len = _path_length(np.asarray(path, dtype=np.float64))
    direct = goal - start
    if len(path) >= 2 and np.linalg.norm(direct) > 1.0e-9:
        first = np.asarray(path[min(3, len(path) - 1)], dtype=np.float64) - start
        angle = _angle_deg(first, direct)
    else:
        angle = 0.0
    return {
        "path_len": path_len,
        "ratio": path_len / straight if straight > 1.0e-9 else 1.0,
        "initial_turn_deg": angle,
    }


def _astar_path(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    margin: float,
) -> list[tuple[float, float]] | None:
    xs = np.arange(-WORLD_LIM, WORLD_LIM + 0.5 * GRID_STEP, GRID_STEP)
    ys = np.arange(-WORLD_LIM, WORLD_LIM + 0.5 * GRID_STEP, GRID_STEP)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([grid_x, grid_y], axis=-1)
    free = np.ones(grid_x.shape, dtype=bool)
    if len(centers):
        dist = np.linalg.norm(pts[:, :, None, :] - centers[None, None, :, :], axis=-1)
        free &= np.all(dist >= (radii + margin)[None, None, :], axis=-1)
    start_idx = _nearest_free_index(start, xs, ys, free)
    goal_idx = _nearest_free_index(goal, xs, ys, free)
    if start_idx is None or goal_idx is None:
        return None
    if start_idx == goal_idx:
        return [(float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))]
    neighbors = [
        (-1, -1, np.sqrt(2.0) * GRID_STEP),
        (-1, 0, GRID_STEP),
        (-1, 1, np.sqrt(2.0) * GRID_STEP),
        (0, -1, GRID_STEP),
        (0, 1, GRID_STEP),
        (1, -1, np.sqrt(2.0) * GRID_STEP),
        (1, 0, GRID_STEP),
        (1, 1, np.sqrt(2.0) * GRID_STEP),
    ]
    heap = [(0.0, 0.0, start_idx)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start_idx: 0.0}
    max_i, max_j = free.shape
    while heap:
        _, current_cost, node = heapq.heappop(heap)
        if node == goal_idx:
            break
        if current_cost > cost.get(node, float("inf")) + 1.0e-12:
            continue
        for di, dj, step_cost in neighbors:
            nxt = (node[0] + di, node[1] + dj)
            if not (0 <= nxt[0] < max_i and 0 <= nxt[1] < max_j):
                continue
            if not free[nxt]:
                continue
            new_cost = current_cost + step_cost
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came[nxt] = node
                heuristic = float(np.linalg.norm(_coord(nxt, xs, ys) - _coord(goal_idx, xs, ys)))
                heapq.heappush(heap, (new_cost + heuristic, new_cost, nxt))
    if goal_idx not in cost:
        return None
    idx_path = [goal_idx]
    while idx_path[-1] != start_idx:
        idx_path.append(came[idx_path[-1]])
    idx_path.reverse()
    coords = [(float(start[0]), float(start[1]))]
    coords.extend(tuple(float(v) for v in _coord(idx, xs, ys)) for idx in idx_path[1:-1])
    coords.append((float(goal[0]), float(goal[1])))
    return coords


def _nearest_free_index(
    point: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    free: np.ndarray,
) -> tuple[int, int] | None:
    i = int(np.clip(np.round((point[0] - xs[0]) / GRID_STEP), 0, len(xs) - 1))
    j = int(np.clip(np.round((point[1] - ys[0]) / GRID_STEP), 0, len(ys) - 1))
    if free[i, j]:
        return (i, j)
    free_idx = np.argwhere(free)
    if free_idx.size == 0:
        return None
    coords = np.stack([xs[free_idx[:, 0]], ys[free_idx[:, 1]]], axis=1)
    nearest = int(np.argmin(np.linalg.norm(coords - point[None, :], axis=1)))
    return (int(free_idx[nearest, 0]), int(free_idx[nearest, 1]))


def _coord(idx: tuple[int, int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.asarray([xs[idx[0]], ys[idx[1]]], dtype=np.float64)


def _line_clearance(trace: Any) -> float:
    start = trace["start"].astype(np.float64)
    goal = trace["goal"].astype(np.float64)
    centers = trace["obstacle_centers"].astype(np.float64)
    radii = trace["obstacle_radii"].astype(np.float64)
    active = trace["obstacle_active"].astype(bool)
    vals = [
        _point_segment_distance(center, start, goal) - radius
        for center, radius in zip(centers[active], radii[active])
    ]
    return float(np.min(vals)) if vals else float("inf")


def _point_segment_distance(point: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    seg = goal - start
    denom = float(np.dot(seg, seg))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    tau = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + tau * seg)))


def _large_detour(stats: dict[str, float], line_clearance: float | None) -> bool:
    if not np.isfinite(stats["path_len"]):
        return True
    if stats["ratio"] >= LARGE_DETOUR_RATIO:
        return True
    if stats["initial_turn_deg"] >= LARGE_DETOUR_ANGLE_DEG:
        return True
    if line_clearance is not None and line_clearance < -0.30 and stats["ratio"] >= LOCAL_DETOUR_RATIO:
        return True
    return False


def _assign_layer(
    *,
    outcome: str,
    observed: bool,
    large_scene: bool,
    large_event: bool,
    empty_onset: int,
    brake_onset: int,
    event_action: int,
    sign_rate: float,
    pre_projection: float,
    pre_saturation: float,
) -> tuple[str, str]:
    if not observed:
        return "observation-limit", "culprit obstacle absent from Top-K in relevant window"
    if outcome == "collision":
        if large_scene:
            return (
                "dynamics-doomed-needs-planning",
                "locally box-infeasible after entering a large-detour/line-blocked scene",
            )
        return (
            "fixable-upstream-policy",
            "culprit observed and detour estimate modest; earlier braking/turning was available before local infeasibility",
        )
    if large_event and pre_projection > 1.0 and pre_saturation > 0.5:
        return (
            "planning-limit-geometric-trap",
            "event-to-goal path requires large detour while filter is saturated near obstacles",
        )
    if sign_rate > 0.35 and pre_projection > 1.0 and pre_saturation > 0.5:
        return (
            "fixable-stuck-chatter",
            "observed near-obstacle trap with high projection/saturation and oscillatory actions",
        )
    return "fixable-stuck-chatter", "local stuck without observation or planning limit"


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _net_displacement(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def _action_sign_change_rate(actions: np.ndarray) -> float:
    if actions.shape[0] < 2:
        return 0.0
    signs = np.sign(actions)
    changes = np.any(signs[1:] * signs[:-1] < 0, axis=1)
    return float(changes.mean())


def _write_csv(path: Path, rows: list[Attribution]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_summary(path: Path, rows: list[Attribution]) -> None:
    lines = ["# Layer Attribution", ""]
    lines.append(
        "Observation membership is reconstructed exactly from saved states and full obstacle "
        "layout using `k_obs=5`; the original NPZ traces did not serialize the observation tensor."
    )
    lines.append("")
    for outcome in ("collision", "stuck"):
        subset = [r for r in rows if r.outcome == outcome]
        lines.extend([f"## {outcome.title()}", ""])
        layer_counts: dict[str, list[Attribution]] = {}
        for row in subset:
            layer_counts.setdefault(row.layer, []).append(row)
        lines.append("| layer | count | fraction | representatives |")
        lines.append("|---|---:|---:|---|")
        for layer, group in sorted(layer_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            reps = ", ".join(str(r.episode_idx) for r in group[:5])
            lines.append(f"| {layer} | {len(group)} | {len(group)/len(subset):.3f} | {reps} |")
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|---|---:|")
        lines.append(f"| culprit observed | {_frac([r.culprit_observed for r in subset]):.3f} |")
        lines.append(f"| mean culprit Top-K fraction | {np.mean([r.culprit_topk_frac for r in subset]):.3f} |")
        lines.append(f"| mean all-near-obstacles Top-K fraction | {np.mean([r.all_near_obstacles_topk_frac for r in subset]):.3f} |")
        active_counts = [r.active_obstacles for r in subset]
        lines.append(
            f"| active obstacles mean/min/max | {np.mean(active_counts):.3f}/{min(active_counts)}/{max(active_counts)} |"
        )
        lines.append(f"| large start-goal detour | {_frac([r.large_detour_scene for r in subset]):.3f} |")
        lines.append(f"| large event-goal detour | {_frac([r.large_detour_from_event for r in subset]):.3f} |")
        lines.append(f"| mean start-goal detour ratio | {_mean_finite([r.start_goal_detour_ratio for r in subset]):.3f} |")
        lines.append(f"| mean start-goal initial turn deg | {_mean_finite([r.start_goal_initial_turn_deg for r in subset]):.3f} |")
        lines.append(f"| mean event-goal detour ratio | {_mean_finite([r.event_goal_detour_ratio for r in subset]):.3f} |")
        lines.append(f"| mean event-goal initial turn deg | {_mean_finite([r.event_goal_initial_turn_deg for r in subset]):.3f} |")
        lines.append("")
    lines.append("## Representative Rows")
    lines.append("")
    lines.append("| episode | outcome | layer | observed | line_clearance | sg_ratio | sg_turn | event_ratio | event_turn | empty_onset | brake_onset | event | h_event | note |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in _representatives(rows):
        lines.append(
            f"| {row.episode_idx} | {row.outcome} | {row.layer} | {row.culprit_observed} | "
            f"{row.line_clearance:.3f} | {row.start_goal_detour_ratio:.3f} | "
            f"{row.start_goal_initial_turn_deg:.1f} | {row.event_goal_detour_ratio:.3f} | "
            f"{row.event_goal_initial_turn_deg:.1f} | {row.empty_onset_step} | "
            f"{row.brake_onset_step} | {row.event_action_step} | {row.h_at_event_action:.3f} | "
            f"{row.note} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _representatives(rows: list[Attribution]) -> list[Attribution]:
    reps = []
    seen = set()
    for row in rows:
        key = (row.outcome, row.layer)
        if key not in seen:
            reps.append(row)
            seen.add(key)
    return reps


def _frac(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("inf")


def _counts_line(rows: list[Attribution]) -> str:
    parts = []
    for outcome in ("collision", "stuck"):
        subset = [r for r in rows if r.outcome == outcome]
        counts: dict[str, int] = {}
        for row in subset:
            counts[row.layer] = counts.get(row.layer, 0) + 1
        parts.append(f"{outcome}=" + ",".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    return " ".join(parts)


if __name__ == "__main__":
    main()
