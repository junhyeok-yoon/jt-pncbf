from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.scene_init import (  # noqa: E402
    Scene,
    _SceneModeParams,
    _initial_velocity_vector,
    _passes_unavoidable_collision_filter,
    sample_train_scene,
)
from src.envs.scene_init_eval import sample_eval_scene  # noqa: E402
from src.envs.double_integrator import DoubleIntegrator  # noqa: E402
from src.common.outcomes import resolve_outcome, step_outcomes  # noqa: E402
from src.eval.plotting import TrajectorySpec, plot_scene_grid  # noqa: E402
from src.eval.rollout import rollout_lqr  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "data" / "verification"
OUTPUT_PATHS = (
    OUTPUT_DIR / "train_scenes_A.png",
    OUTPUT_DIR / "train_scenes_B.png",
    OUTPUT_DIR / "eval_scenes_A.png",
    OUTPUT_DIR / "eval_scenes_B.png",
)
SYSTEM = "double_integrator"
N_SCENES_PER_FIGURE = 16
DTYPE = torch.float64


def main() -> int:
    config = _load_config(REPO_ROOT / "src" / "configs" / "base_config.yaml")
    train_params = _mode_params(config, "train")
    eval_params = _mode_params(config, "eval")
    system = DoubleIntegrator(config)

    train_rng = np.random.default_rng(seed=99999)
    eval_rng = np.random.default_rng(seed=88888)

    train_scenes_a = _sample_and_validate_scenes(
        train_rng, config, train_params, sample_train_scene, 0
    )
    train_scenes_b = _sample_and_validate_scenes(
        train_rng, config, train_params, sample_train_scene, N_SCENES_PER_FIGURE
    )
    eval_scenes_a = _sample_and_validate_scenes(
        eval_rng, config, eval_params, sample_eval_scene, 0
    )
    eval_scenes_b = _sample_and_validate_scenes(
        eval_rng, config, eval_params, sample_eval_scene, N_SCENES_PER_FIGURE
    )

    train_traj_a, train_outcomes_a, train_steps_a = _rollout_scenes(
        system, train_scenes_a, config
    )
    train_traj_b, train_outcomes_b, train_steps_b = _rollout_scenes(
        system, train_scenes_b, config
    )
    eval_traj_a, eval_outcomes_a, eval_steps_a = _rollout_scenes(
        system, eval_scenes_a, config
    )
    eval_traj_b, eval_outcomes_b, eval_steps_b = _rollout_scenes(
        system, eval_scenes_b, config
    )

    plot_scene_grid(
        train_scenes_a,
        OUTPUT_PATHS[0],
        config,
        "Train scenes",
        SYSTEM,
        "A",
        0,
        trajectories=train_traj_a,
        outcomes=train_outcomes_a,
        event_steps=train_steps_a,
        draw_start_velocity=True,
        draw_final_velocity=True,
    )
    plot_scene_grid(
        train_scenes_b,
        OUTPUT_PATHS[1],
        config,
        "Train scenes",
        SYSTEM,
        "B",
        N_SCENES_PER_FIGURE,
        trajectories=train_traj_b,
        outcomes=train_outcomes_b,
        event_steps=train_steps_b,
        draw_start_velocity=True,
        draw_final_velocity=True,
    )
    plot_scene_grid(
        eval_scenes_a,
        OUTPUT_PATHS[2],
        config,
        "Eval scenes",
        SYSTEM,
        "A",
        0,
        trajectories=eval_traj_a,
        outcomes=eval_outcomes_a,
        event_steps=eval_steps_a,
        draw_start_velocity=True,
        draw_final_velocity=True,
    )
    plot_scene_grid(
        eval_scenes_b,
        OUTPUT_PATHS[3],
        config,
        "Eval scenes",
        SYSTEM,
        "B",
        N_SCENES_PER_FIGURE,
        trajectories=eval_traj_b,
        outcomes=eval_outcomes_b,
        event_steps=eval_steps_b,
        draw_start_velocity=True,
        draw_final_velocity=True,
    )

    missing_paths = [path for path in OUTPUT_PATHS if not path.exists()]
    if missing_paths:
        missing = [str(path) for path in missing_paths]
        raise RuntimeError(f"Missing output files: {missing}")

    for path in OUTPUT_PATHS:
        print(path.relative_to(REPO_ROOT))

    return 0


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping):
        raise TypeError(f"Expected mapping config at {path}")
    return config


def _mode_params(config: Mapping[str, Any], mode: str) -> _SceneModeParams:
    if mode == "train":
        return _SceneModeParams(
            mode=mode,
            min_start_goal_dist=float(config["scene_train"]["min_start_goal_dist"]),
            start_goal_clearance=float(config["scene_train"]["start_goal_clearance"]),
        )
    if mode == "eval":
        return _SceneModeParams(
            mode=mode,
            min_start_goal_dist=float(config["eval"]["scene"]["min_start_goal_dist"]),
            start_goal_clearance=float(config["eval"]["scene"]["start_goal_clearance"]),
        )
    raise ValueError(f"Unsupported mode: {mode!r}")


def _sample_and_validate_scenes(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    params: _SceneModeParams,
    sampler: Any,
    start_index: int,
) -> list[Scene]:
    scenes = []
    for offset in range(N_SCENES_PER_FIGURE):
        scene = sampler(rng, config, SYSTEM)
        _validate_scene(scene, config, params, start_index + offset)
        scenes.append(scene)
    return scenes


def _rollout_scenes(
    system: DoubleIntegrator,
    scenes: list[Scene],
    config: Mapping[str, Any],
) -> tuple[list[list[TrajectorySpec]], list[str], list[int]]:
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    trajectories = []
    outcomes = []
    event_steps = []
    for scene in scenes:
        states = rollout_lqr(system, scene, _initial_state(scene), max_steps, dt)
        masks = step_outcomes(states, scene, system, config)
        resolved = resolve_outcome(masks)
        trajectories.append([TrajectorySpec(states=states[:, 0, :])])
        outcomes.append(resolved.outcome[0])
        event_steps.append(int(resolved.event_step[0]))
    return trajectories, outcomes, event_steps


def _validate_scene(
    scene: Scene,
    config: Mapping[str, Any],
    params: _SceneModeParams,
    scenario_index: int,
) -> None:
    active = scene.obstacle_active
    active_centers = scene.obstacle_centers[active]
    active_radii = scene.obstacle_radii[active]
    obstacle_cfg = config["obstacle"]
    n_active = int(np.count_nonzero(active))

    if np.linalg.norm(scene.goal - scene.start) < params.min_start_goal_dist:
        _fail("start-goal distance below minimum", scene, scenario_index)

    start_clearance = float(
        np.min(np.linalg.norm(active_centers - scene.start, axis=1) - active_radii)
    )
    if start_clearance < params.start_goal_clearance:
        _fail("start clearance below minimum", scene, scenario_index)

    goal_clearance = float(
        np.min(np.linalg.norm(active_centers - scene.goal, axis=1) - active_radii)
    )
    if goal_clearance < params.start_goal_clearance:
        _fail("goal clearance below minimum", scene, scenario_index)

    if not int(obstacle_cfg["n_min"]) <= n_active <= int(obstacle_cfg["n_max"]):
        _fail("active obstacle count out of bounds", scene, scenario_index)

    if np.any(active_radii < float(obstacle_cfg["r_min"])) or np.any(
        active_radii > float(obstacle_cfg["r_max"])
    ):
        _fail("active obstacle radius out of bounds", scene, scenario_index)

    velocity = _initial_velocity_vector(scene)
    if scene.system == "double_integrator":
        v_init_max = float(config["env"]["v_init_max"])
        if float(np.max(np.abs(velocity))) > v_init_max:
            _fail("initial velocity out of bounds", scene, scenario_index)

    if not _passes_unavoidable_collision_filter(scene, config, params):
        _fail("unavoidable-collision filter failed", scene, scenario_index)


def _initial_state(scene: Scene) -> torch.Tensor:
    if scene.initial_velocity is None:
        raise ValueError("Double Integrator scene is missing initial_velocity.")
    state = np.concatenate([scene.start, scene.initial_velocity])
    return torch.as_tensor(state, dtype=DTYPE).unsqueeze(0)


def _fail(message: str, scene: Scene, scenario_index: int) -> None:
    payload = {
        "message": message,
        "scenario_index": scenario_index,
        "scene": _scene_to_jsonable(scene),
    }
    raise AssertionError(json.dumps(payload, indent=2, sort_keys=True))


def _scene_to_jsonable(scene: Scene) -> dict[str, Any]:
    payload = asdict(scene)
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
