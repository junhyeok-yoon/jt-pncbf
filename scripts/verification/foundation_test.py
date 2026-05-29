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

from src.common.outcomes import (  # noqa: E402
    StepOutcomeMasks,
    resolve_outcome,
    step_outcomes,
)
from src.common.rk4 import rk4_step  # noqa: E402
from src.envs.double_integrator import DoubleIntegrator  # noqa: E402
from src.envs.scene_init import Scene, sample_train_scene  # noqa: E402
from src.envs.unicycle import Unicycle  # noqa: E402
from src.eval.plotting import TrajectorySpec, plot_scene_grid  # noqa: E402
from src.eval.rollout import rollout_lqr  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "data" / "verification"
OUTPUT_PATHS = (
    OUTPUT_DIR / "lqr_di_A.png",
    OUTPUT_DIR / "lqr_di_B.png",
    OUTPUT_DIR / "lqr_unicycle_A.png",
    OUTPUT_DIR / "lqr_unicycle_B.png",
)
N_SCENES = 32
N_SCENES_PER_FIGURE = 16
ACTION_TOL = 1.0e-6
ANGLE_TOL = 1.0e-6
DTYPE = torch.float64


def main() -> int:
    config = _load_config(REPO_ROOT / "src" / "configs" / "base_config.yaml")
    dt = float(config["env"]["dt"])
    max_steps = int(config["eval"]["max_steps"])

    di = DoubleIntegrator(config)
    unicycle = Unicycle(config)
    _run_synthetic_battery(config, di, unicycle)

    di_scenes, di_states, di_outcomes, di_steps = _sample_rollout_validate(
        di, config, dt, max_steps
    )
    uni_scenes, uni_states, uni_outcomes, uni_steps = _sample_rollout_validate(
        unicycle, config, dt, max_steps
    )

    _plot_lqr(
        di_scenes[:16],
        di_states[:16],
        di_outcomes[:16],
        di_steps[:16],
        config,
        di.name,
        "A",
        OUTPUT_PATHS[0],
        0,
    )
    _plot_lqr(
        di_scenes[16:],
        di_states[16:],
        di_outcomes[16:],
        di_steps[16:],
        config,
        di.name,
        "B",
        OUTPUT_PATHS[1],
        16,
    )
    _plot_lqr(
        uni_scenes[:16],
        uni_states[:16],
        uni_outcomes[:16],
        uni_steps[:16],
        config,
        unicycle.name,
        "A",
        OUTPUT_PATHS[2],
        0,
    )
    _plot_lqr(
        uni_scenes[16:],
        uni_states[16:],
        uni_outcomes[16:],
        uni_steps[16:],
        config,
        unicycle.name,
        "B",
        OUTPUT_PATHS[3],
        16,
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


def _run_synthetic_battery(
    config: Mapping[str, Any],
    di: DoubleIntegrator,
    unicycle: Unicycle,
) -> None:
    dt = float(config["env"]["dt"])
    _assert_di_zero_control(di, dt)
    _assert_di_constant_acceleration(di, dt)
    _assert_unicycle_straight_line(unicycle, dt)
    _assert_unicycle_pure_turn(unicycle, dt)
    _assert_observation_dimensions_and_topk(config, di, unicycle)
    _assert_observation_invariance(config, di, unicycle)
    _assert_lqr_pull_toward_goal(config, di, unicycle)
    _assert_outcome_resolver_reference()


def _assert_di_zero_control(di: DoubleIntegrator, dt: float) -> None:
    steps = 80
    velocity = torch.tensor([[0.7, -0.4]], dtype=DTYPE)
    x = torch.tensor([[0.0, 0.0, velocity[0, 0], velocity[0, 1]]], dtype=DTYPE)
    u = torch.zeros((1, di.action_dim), dtype=DTYPE)
    for _ in range(steps):
        x = rk4_step(di, x, u, dt)

    elapsed = steps * dt
    expected = torch.tensor(
        [[velocity[0, 0] * elapsed, velocity[0, 1] * elapsed, 0.7, -0.4]],
        dtype=DTYPE,
    )
    _assert_close("DI zero-control dynamics", x, expected, 1.0e-6)


def _assert_di_constant_acceleration(di: DoubleIntegrator, dt: float) -> None:
    steps = 80
    acceleration = torch.tensor([[0.6, -0.3]], dtype=DTYPE)
    x = torch.zeros((1, di.state_dim), dtype=DTYPE)
    for _ in range(steps):
        x = rk4_step(di, x, acceleration, dt)

    elapsed = steps * dt
    expected = torch.tensor(
        [
            [
                0.5 * acceleration[0, 0] * elapsed**2,
                0.5 * acceleration[0, 1] * elapsed**2,
                acceleration[0, 0] * elapsed,
                acceleration[0, 1] * elapsed,
            ]
        ],
        dtype=DTYPE,
    )
    _assert_close("DI constant-acceleration dynamics", x, expected, 1.0e-6)


def _assert_unicycle_straight_line(unicycle: Unicycle, dt: float) -> None:
    steps = 80
    speed = 0.8
    x = torch.tensor([[0.0, 0.0, 0.0, speed]], dtype=DTYPE)
    u = torch.zeros((1, unicycle.action_dim), dtype=DTYPE)
    for _ in range(steps):
        x = rk4_step(unicycle, x, u, dt)

    expected = torch.tensor([[speed * steps * dt, 0.0, 0.0, speed]], dtype=DTYPE)
    _assert_close("Unicycle straight-line dynamics", x, expected, 1.0e-5)


def _assert_unicycle_pure_turn(unicycle: Unicycle, dt: float) -> None:
    steps = 220
    omega = 1.2
    x = torch.zeros((1, unicycle.state_dim), dtype=DTYPE)
    u = torch.tensor([[0.0, omega]], dtype=DTYPE)
    for _ in range(steps):
        x = rk4_step(unicycle, x, u, dt)

    expected_theta = _wrap_angle_np(omega * steps * dt)
    expected = torch.tensor([[0.0, 0.0, expected_theta, 0.0]], dtype=DTYPE)
    _assert_close("Unicycle pure-turn dynamics", x, expected, 1.0e-5)


def _assert_observation_dimensions_and_topk(
    config: Mapping[str, Any],
    di: DoubleIntegrator,
    unicycle: Unicycle,
) -> None:
    scene = _handbuilt_topk_scene(config)
    x_di = torch.tensor([[0.0, 0.0, 0.1, -0.2]], dtype=DTYPE)
    x_uni = torch.tensor([[0.0, 0.0, 0.0, 0.3]], dtype=DTYPE)
    obs_di = di.observation(x_di, scene)
    obs_uni = unicycle.observation(x_uni, scene)

    if obs_di.shape != (1, 19):
        _fail("DI observation dimension mismatch", {"shape": tuple(obs_di.shape)})
    if obs_uni.shape != (1, 18):
        _fail(
            "Unicycle observation dimension mismatch",
            {"shape": tuple(obs_uni.shape)},
        )

    expected = _expected_topk_block(
        scene,
        position=np.zeros(2),
        k=int(config["env"]["k_obs"]),
    )
    di_block = obs_di[0, 4:].reshape(-1, 3).detach().cpu().numpy()
    uni_block = obs_uni[0, 3:].reshape(-1, 3).detach().cpu().numpy()
    if not np.allclose(di_block, expected, atol=1.0e-6):
        _fail("DI Top-K block mismatch", {"actual": di_block, "expected": expected})
    if not np.allclose(uni_block, expected, atol=1.0e-6):
        _fail(
            "Unicycle Top-K block mismatch",
            {"actual": uni_block, "expected": expected},
        )


def _assert_observation_invariance(
    config: Mapping[str, Any],
    di: DoubleIntegrator,
    unicycle: Unicycle,
) -> None:
    rng = np.random.default_rng(seed=123)
    scene = _handbuilt_topk_scene(config)
    translation = rng.uniform(-1.0, 1.0, size=2)

    x_di = torch.tensor([[0.2, -0.4, 0.3, -0.1]], dtype=DTYPE)
    scene_shifted = _transform_scene(scene, translation=translation)
    x_di_shifted = x_di.clone()
    x_di_shifted[:, :2] += torch.as_tensor(translation, dtype=DTYPE)
    _assert_close(
        "DI translation invariance",
        di.observation(x_di, scene),
        di.observation(x_di_shifted, scene_shifted),
        1.0e-6,
    )

    angle = 0.73
    x_uni = torch.tensor([[0.3, -0.2, 0.4, 0.5]], dtype=DTYPE)
    scene_rotated = _transform_scene(scene, translation=translation, rotation=angle)
    rotated_position = _rotate_points(x_uni[0, :2].numpy(), angle) + translation
    x_uni_rotated = x_uni.clone()
    x_uni_rotated[0, :2] = torch.as_tensor(rotated_position, dtype=DTYPE)
    x_uni_rotated[0, 2] = torch.as_tensor(_wrap_angle_np(float(x_uni[0, 2]) + angle))
    _assert_close(
        "Unicycle rotation+translation invariance",
        unicycle.observation(x_uni, scene),
        unicycle.observation(x_uni_rotated, scene_rotated),
        1.0e-6,
    )


def _assert_lqr_pull_toward_goal(
    config: Mapping[str, Any],
    di: DoubleIntegrator,
    unicycle: Unicycle,
) -> None:
    dt = float(config["env"]["dt"])
    max_steps = int(config["eval"]["max_steps"])
    scene_di = _single_far_obstacle_scene(config, "double_integrator")
    x0_di = torch.tensor([[-3.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    states_di = rollout_lqr(di, scene_di, x0_di, max_steps, dt)[:, 0, :]
    _assert_final_closer("DI LQR pull", states_di[:, :2], scene_di.goal)

    scene_uni = _single_far_obstacle_scene(config, "unicycle")
    x0_uni = torch.tensor([[-3.0, 0.0, 0.0, 0.1]], dtype=DTYPE)
    states_uni = rollout_lqr(unicycle, scene_uni, x0_uni, max_steps, dt)[:, 0, :]
    _assert_final_closer("Unicycle LQR pull", states_uni[:, :2], scene_uni.goal)


def _assert_outcome_resolver_reference() -> None:
    _assert_outcome_mask_priority()
    _assert_outcome_predicates()


def _assert_outcome_mask_priority() -> None:
    shape = (5, 4)
    collided = torch.zeros(shape, dtype=torch.bool)
    goal = torch.zeros(shape, dtype=torch.bool)
    oob = torch.zeros(shape, dtype=torch.bool)
    collided[2, 0] = True
    goal[2, 0] = True
    goal[1, 1] = True
    oob[1, 1] = True
    oob[3, 2] = True

    stuck = torch.zeros_like(collided)
    window_displacement = torch.full_like(collided, float("nan"), dtype=torch.float64)
    masks = StepOutcomeMasks(
        collided=collided,
        goal_reached=goal,
        oob=oob,
        stuck=stuck,
        window_displacement=window_displacement,
    )
    resolved = resolve_outcome(masks)
    expected_outcomes, expected_steps = _inline_reference_outcomes(masks)
    if resolved.outcome != expected_outcomes:
        _fail(
            "Outcome labels mismatch",
            {"actual": resolved.outcome, "expected": expected_outcomes},
        )
    if not torch.equal(resolved.event_step.cpu(), expected_steps):
        _fail(
            "Outcome event steps mismatch",
            {"actual": resolved.event_step, "expected": expected_steps},
        )


def _assert_outcome_predicates() -> None:
    config = _load_config(REPO_ROOT / "src" / "configs" / "base_config.yaml")
    system = DoubleIntegrator(config)
    scene = _outcome_battery_scene()
    states = torch.zeros((5, 4, system.state_dim), dtype=DTYPE)
    states[:, :, 0:2] = torch.tensor(
        [
            [[-2.0, 0.0], [0.0, 0.0], [4.0, 4.0], [-2.0, 2.0]],
            [[-1.0, 0.0], [0.6, 0.0], [5.0, 4.0], [-1.8, 2.0]],
            [[0.0, 0.0], [1.0, 0.0], [8.1, 0.0], [-1.6, 2.0]],
            [[1.0, 0.0], [1.4, 0.0], [8.2, 0.0], [-1.4, 2.0]],
            [[1.0, 0.0], [1.8, 0.0], [8.3, 0.0], [-1.2, 2.0]],
        ],
        dtype=DTYPE,
    )
    states[:, 0, 2:4] = 0.0
    states[:, 1, 2:4] = torch.tensor([1.0, 0.0], dtype=DTYPE)
    states[:, 2, 2:4] = 0.0
    states[:, 3, 2:4] = torch.tensor([0.2, 0.0], dtype=DTYPE)

    resolved = resolve_outcome(step_outcomes(states, scene, system, config))
    expected_outcomes = ["goal", "collision", "oob", "timeout"]
    expected_steps = torch.tensor([2, 2, 2, -1], dtype=torch.long)
    if resolved.outcome != expected_outcomes:
        _fail(
            "Predicate outcome labels mismatch",
            {"actual": resolved.outcome, "expected": expected_outcomes},
        )
    if not torch.equal(resolved.event_step.cpu(), expected_steps):
        _fail(
            "Predicate outcome event steps mismatch",
            {"actual": resolved.event_step, "expected": expected_steps},
        )


def _inline_reference_outcomes(
    masks: StepOutcomeMasks,
) -> tuple[list[str], torch.Tensor]:
    n_steps, batch_size = masks.collided.shape
    outcomes = ["timeout"] * batch_size
    event_steps = torch.full((batch_size,), -1, dtype=torch.long)
    for batch_idx in range(batch_size):
        for step in range(n_steps):
            if bool(masks.collided[step, batch_idx]):
                outcomes[batch_idx] = "collision"
                event_steps[batch_idx] = step
                break
            if bool(masks.goal_reached[step, batch_idx]):
                outcomes[batch_idx] = "goal"
                event_steps[batch_idx] = step
                break
            if bool(masks.oob[step, batch_idx]):
                outcomes[batch_idx] = "oob"
                event_steps[batch_idx] = step
                break
            if bool(masks.stuck[step, batch_idx]):
                outcomes[batch_idx] = "stuck"
                event_steps[batch_idx] = step
                break
    return outcomes, event_steps


def _sample_rollout_validate(
    system: Any,
    config: Mapping[str, Any],
    dt: float,
    max_steps: int,
) -> tuple[list[Scene], list[torch.Tensor], list[str], list[int]]:
    rng = np.random.default_rng(seed=99999)
    scenes = []
    states_by_scene = []
    outcomes = []
    event_steps = []
    for scenario_index in range(N_SCENES):
        scene = sample_train_scene(rng, config, system.name)
        x0 = _initial_state(scene)
        with torch.no_grad():
            states = rollout_lqr(system, scene, x0, max_steps, dt)[:, 0, :]
        outcome_result = _validate_scene_rollout(
            system,
            scene,
            states,
            config,
            scenario_index,
        )
        scenes.append(scene)
        states_by_scene.append(states)
        outcomes.append(outcome_result.outcome[0])
        event_steps.append(int(outcome_result.event_step[0]))
    return scenes, states_by_scene, outcomes, event_steps


def _validate_scene_rollout(
    system: Any,
    scene: Scene,
    states: torch.Tensor,
    config: Mapping[str, Any],
    scenario_index: int,
) -> Any:
    oob_limit = float(config["env"]["oob_limit"])
    positions = states[:, :2]
    if not torch.isfinite(states).all():
        _fail_scene("state contains NaN or Inf", scene, scenario_index)
    if torch.any(torch.abs(positions) > oob_limit + ACTION_TOL):
        _fail_scene("trajectory left OOB assertion bounds", scene, scenario_index)

    goal = torch.as_tensor(scene.goal, dtype=states.dtype).unsqueeze(0)
    actions = system.lqr_action(states[:-1], goal.expand(states.shape[0] - 1, -1))
    if not torch.isfinite(actions).all():
        _fail_scene("action contains NaN or Inf", scene, scenario_index)

    bounds = system.u_bounds.to(dtype=states.dtype)
    below = actions < bounds[:, 0] - ACTION_TOL
    above = actions > bounds[:, 1] + ACTION_TOL
    if torch.any(below | above):
        _fail_scene("action exceeded control bounds", scene, scenario_index)

    if system.name == "unicycle":
        theta = states[:, 2]
        below_min = torch.any(theta < -torch.pi - ANGLE_TOL)
        above_max = torch.any(theta > torch.pi + ANGLE_TOL)
        if below_min or above_max:
            _fail_scene("unicycle theta outside [-pi, pi]", scene, scenario_index)

    masks = step_outcomes(states.unsqueeze(1), scene, system, config)
    return resolve_outcome(masks)


def _plot_lqr(
    scenes: list[Scene],
    states_by_scene: list[torch.Tensor],
    outcomes: list[str],
    event_steps: list[int],
    config: Mapping[str, Any],
    system_name: str,
    letter: str,
    output_path: Path,
    start_index: int,
) -> None:
    trajectories = [[TrajectorySpec(states=states)] for states in states_by_scene]
    plot_scene_grid(
        scenes,
        output_path,
        config,
        "LQR-only",
        system_name,
        letter,
        start_index,
        trajectories=trajectories,
        outcomes=outcomes,
        event_steps=event_steps,
        draw_final_velocity=True,
    )


def _initial_state(scene: Scene) -> torch.Tensor:
    if scene.system == "double_integrator":
        if scene.initial_velocity is None:
            raise ValueError("DI scene missing initial_velocity.")
        state = np.concatenate([scene.start, scene.initial_velocity])
        return torch.as_tensor(state, dtype=DTYPE).unsqueeze(0)

    if scene.system == "unicycle":
        if scene.initial_heading is None or scene.initial_speed is None:
            raise ValueError("Unicycle scene missing initial heading or speed.")
        state = np.array(
            [
                scene.start[0],
                scene.start[1],
                scene.initial_heading,
                scene.initial_speed,
            ],
            dtype=np.float64,
        )
        return torch.as_tensor(state, dtype=DTYPE).unsqueeze(0)

    raise ValueError(f"Unsupported scene system: {scene.system!r}")


def _handbuilt_topk_scene(config: Mapping[str, Any]) -> Scene:
    n_max = int(config["obstacle"]["n_max"])
    centers = np.zeros((n_max, 2), dtype=np.float64)
    radii = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.bool_)
    centers[:7] = np.array(
        [
            [3.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, -1.4],
            [-2.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    radii[:7] = np.array([0.2, 0.1, 0.2, 0.4, 0.3, 0.8, 0.2], dtype=np.float64)
    active[:5] = True
    active[6] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([0.0, 0.0], dtype=np.float64),
        goal=np.array([2.5, -1.0], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _single_far_obstacle_scene(config: Mapping[str, Any], system: str) -> Scene:
    n_max = int(config["obstacle"]["n_max"])
    centers = np.zeros((n_max, 2), dtype=np.float64)
    radii = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.bool_)
    centers[0] = np.array([0.0, 3.5], dtype=np.float64)
    radii[0] = 0.2
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-3.0, 0.0], dtype=np.float64),
        goal=np.array([3.0, 0.0], dtype=np.float64),
        system=system,
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64)
        if system == "double_integrator"
        else None,
        initial_speed=0.1 if system == "unicycle" else None,
        initial_heading=0.0 if system == "unicycle" else None,
    )


def _outcome_battery_scene() -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([1.0, 0.0], dtype=np.float64)
    radii[0] = 0.3
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-2.0, 0.0], dtype=np.float64),
        goal=np.array([0.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _expected_topk_block(
    scene: Scene,
    position: np.ndarray,
    k: int,
) -> np.ndarray:
    active_indices = np.where(scene.obstacle_active)[0]
    rel = scene.obstacle_centers[active_indices] - position
    radii = scene.obstacle_radii[active_indices]
    surface = np.linalg.norm(rel, axis=1) - radii
    ranked_local = np.argsort(surface, kind="stable")[:k]
    block = np.concatenate([rel[ranked_local], radii[ranked_local, None]], axis=1)
    if block.shape[0] == k:
        return block
    return np.pad(block, ((0, k - block.shape[0]), (0, 0)))


def _transform_scene(
    scene: Scene,
    translation: np.ndarray,
    rotation: float | None = None,
) -> Scene:
    if rotation is None:
        centers = scene.obstacle_centers + translation
        goal = scene.goal + translation
        start = scene.start + translation
    else:
        centers = _rotate_points(scene.obstacle_centers, rotation) + translation
        goal = _rotate_points(scene.goal, rotation) + translation
        start = _rotate_points(scene.start, rotation) + translation

    return Scene(
        obstacle_centers=centers,
        obstacle_radii=scene.obstacle_radii.copy(),
        obstacle_active=scene.obstacle_active.copy(),
        start=start,
        goal=goal,
        system=scene.system,
        mode=scene.mode,
        initial_velocity=None
        if scene.initial_velocity is None
        else scene.initial_velocity.copy(),
        initial_speed=scene.initial_speed,
        initial_heading=scene.initial_heading,
    )


def _rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return np.asarray(points) @ rotation.T


def _assert_final_closer(name: str, positions: torch.Tensor, goal: np.ndarray) -> None:
    goal_tensor = torch.as_tensor(goal, dtype=positions.dtype)
    initial_distance = torch.linalg.norm(positions[0] - goal_tensor)
    final_distance = torch.linalg.norm(positions[-1] - goal_tensor)
    if not final_distance < initial_distance:
        _fail(
            f"{name} failed",
            {
                "initial_distance": float(initial_distance),
                "final_distance": float(final_distance),
            },
        )


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        _fail(
            f"{name} failed",
            {
                "actual": actual.detach().cpu().numpy(),
                "expected": expected.detach().cpu().numpy(),
                "atol": atol,
            },
        )


def _fail_scene(message: str, scene: Scene, scenario_index: int) -> None:
    _fail(
        message,
        {
            "scenario_index": scenario_index,
            "scene": _scene_to_jsonable(scene),
        },
    )


def _fail(message: str, payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"message": message, "payload": _jsonable(payload)},
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise AssertionError(message)


def _scene_to_jsonable(scene: Scene) -> dict[str, Any]:
    return _jsonable(asdict(scene))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _wrap_angle_np(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        raise SystemExit(1) from exc
