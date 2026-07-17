from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scene:
    obstacle_centers: Array
    obstacle_radii: Array
    obstacle_active: npt.NDArray[np.bool_]
    start: Array
    goal: Array
    system: str
    mode: str
    initial_velocity: Array | None = None
    initial_speed: float | None = None
    initial_heading: float | None = None
    initial_attitude: float | None = None      # quadrotor_planar: body attitude theta
    initial_omega: float | None = None         # quadrotor_planar: body angular rate omega


@dataclass(frozen=True)
class _SceneModeParams:
    mode: str
    min_start_goal_dist: float
    start_goal_clearance: float


_MAX_RETRIES = 1000
_START_GOAL_ARENA_MARGIN = 0.3
_ZERO_SPEED_EPS = 1.0e-12


def sample_train_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
) -> Scene:
    params = _SceneModeParams(
        mode="train",
        min_start_goal_dist=float(config["scene_train"]["min_start_goal_dist"]),
        start_goal_clearance=float(config["scene_train"]["start_goal_clearance"]),
    )
    return _sample_scene(rng, config, system, params)


def _sample_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
    params: _SceneModeParams,
) -> Scene:
    if system not in {"double_integrator", "unicycle", "quadrotor_planar"}:
        raise ValueError(f"Unsupported system: {system!r}")

    for _ in range(_MAX_RETRIES):
        start, goal = _sample_start_goal(rng, config, params)
        centers, radii, active = _sample_obstacles(rng, config, start, goal)
        scene = _make_scene(
            rng,
            config,
            system,
            params.mode,
            centers,
            radii,
            active,
            start,
            goal,
        )

        if not _has_start_goal_clearance(scene, params.start_goal_clearance):
            continue
        if not _passes_unavoidable_collision_filter(scene, config, params):
            continue
        if not _passes_recoverability_filter(scene, config):   # v2.6.2 amendment 3 (quadrotor only, additive)
            continue
        return scene

    raise RuntimeError(
        f"Scene sampling exceeded retry cap ({_MAX_RETRIES}) for mode={params.mode!r}, "
        f"system={system!r}."
    )


def _sample_start_goal(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    params: _SceneModeParams,
) -> tuple[Array, Array]:
    world_lim = float(config["env"]["world_lim"])
    coord_lim = world_lim - _START_GOAL_ARENA_MARGIN

    for _ in range(_MAX_RETRIES):
        start = rng.uniform(-coord_lim, coord_lim, size=2).astype(np.float64)
        goal = rng.uniform(-coord_lim, coord_lim, size=2).astype(np.float64)
        if np.linalg.norm(goal - start) >= params.min_start_goal_dist:
            return start, goal

    raise RuntimeError(
        "Start/goal sampling exceeded retry cap "
        f"({_MAX_RETRIES}) for mode={params.mode!r}."
    )


def _sample_obstacles(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    start: Array,
    goal: Array,
) -> tuple[Array, Array, npt.NDArray[np.bool_]]:
    obstacle_cfg = config["obstacle"]
    n_min = int(obstacle_cfg["n_min"])
    n_max = int(obstacle_cfg["n_max"])
    r_min = float(obstacle_cfg["r_min"])
    r_max = float(obstacle_cfg["r_max"])

    n_active = int(rng.integers(n_min, n_max + 1))
    centers = np.zeros((n_max, 2), dtype=np.float64)
    radii = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.bool_)

    radii[:n_active] = rng.uniform(r_min, r_max, size=n_active)
    active[:n_active] = True

    if rng.uniform() < float(obstacle_cfg["p_corridor"]):
        centers[:n_active] = _sample_corridor_centers(
            rng, config, start, goal, n_active
        )
    else:
        world_lim = float(config["env"]["world_lim"])
        centers[:n_active] = rng.uniform(-world_lim, world_lim, size=(n_active, 2))

    return centers, radii, active


def _sample_corridor_centers(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    start: Array,
    goal: Array,
    n_active: int,
) -> Array:
    world_lim = float(config["env"]["world_lim"])
    obstacle_cfg = config["obstacle"]
    segment = goal - start
    length = float(np.linalg.norm(segment))
    if length <= _ZERO_SPEED_EPS:
        return rng.uniform(-world_lim, world_lim, size=(n_active, 2)).astype(np.float64)

    direction = segment / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    midpoint = 0.5 * (start + goal)
    along_half = 0.5 * float(obstacle_cfg["corridor_along_factor"]) * length
    perp_half = 0.5 * float(obstacle_cfg["corridor_perp_factor"]) * length

    centers = np.zeros((n_active, 2), dtype=np.float64)
    for idx in range(n_active):
        for _ in range(_MAX_RETRIES):
            along = rng.uniform(-along_half, along_half)
            perp = rng.uniform(-perp_half, perp_half)
            center = midpoint + along * direction + perp * normal
            if np.all(np.abs(center) <= world_lim):
                centers[idx] = center
                break
        else:
            raise RuntimeError(
                f"Corridor obstacle sampling exceeded retry cap ({_MAX_RETRIES})."
            )

    return centers


def _make_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
    mode: str,
    centers: Array,
    radii: Array,
    active: npt.NDArray[np.bool_],
    start: Array,
    goal: Array,
) -> Scene:
    v_init_max = float(config["env"]["v_init_max"])

    if system == "double_integrator":
        initial_velocity = rng.uniform(-v_init_max, v_init_max, size=2).astype(
            np.float64
        )
        return Scene(
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_active=active,
            start=start,
            goal=goal,
            system=system,
            mode=mode,
            initial_velocity=initial_velocity,
        )

    if system == "quadrotor_planar":
        # v2.6.0 §4 IC set: v ~ U(disk radius v_init), theta ~ U[-pi,pi], omega ~ U[-omega_init_max, .].
        q = config["env"]["quadrotor_planar"]
        v_init = float(q["v_init_max"])
        omega_init = float(q["omega_init_max"])
        radius = v_init * float(np.sqrt(rng.uniform()))       # uniform-in-disk
        angle = float(rng.uniform(-np.pi, np.pi))
        initial_velocity = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64
        )
        return Scene(
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_active=active,
            start=start,
            goal=goal,
            system=system,
            mode=mode,
            initial_velocity=initial_velocity,
            initial_attitude=float(rng.uniform(-np.pi, np.pi)),
            initial_omega=float(rng.uniform(-omega_init, omega_init)),
        )

    initial_speed = float(rng.uniform(-v_init_max, v_init_max))
    initial_heading = float(rng.uniform(-np.pi, np.pi))
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=start,
        goal=goal,
        system=system,
        mode=mode,
        initial_speed=initial_speed,
        initial_heading=initial_heading,
    )


def _passes_recoverability_filter(scene: Scene, config: Mapping[str, Any]) -> bool:
    """v2.6.2 amendment 3: ANALYTIC attitude-aware IC recoverability test (quadrotor_planar ONLY; additive to
    the position/velocity filter above). Rejects born-doomed ICs — where the thrust axis cannot rotate onto
    the obstacle normal and brake the inward closure before impact. No-op (returns True) for DI/unicycle and
    if the recov_margin key is absent. See src/common/quadrotor_recoverability.py."""
    if scene.system != "quadrotor_planar":
        return True
    q = config["env"].get("quadrotor_planar", {})
    if "recov_margin" not in q:
        return True
    from src.common.quadrotor_recoverability import is_recoverable, plant_params
    return is_recoverable(
        p0=np.asarray(scene.start, dtype=np.float64),
        v0=_initial_velocity_vector(scene),
        theta0=float(scene.initial_attitude if scene.initial_attitude is not None else 0.0),
        omega0=float(scene.initial_omega if scene.initial_omega is not None else 0.0),
        centers=scene.obstacle_centers, radii=scene.obstacle_radii, active=scene.obstacle_active,
        plant=plant_params(config), margin=float(q["recov_margin"]))


def _has_start_goal_clearance(scene: Scene, clearance: float) -> bool:
    active_centers = scene.obstacle_centers[scene.obstacle_active]
    active_radii = scene.obstacle_radii[scene.obstacle_active]
    start_clearance = np.min(
        np.linalg.norm(active_centers - scene.start, axis=1) - active_radii
    )
    goal_clearance = np.min(
        np.linalg.norm(active_centers - scene.goal, axis=1) - active_radii
    )
    return bool(start_clearance >= clearance and goal_clearance >= clearance)


def _passes_unavoidable_collision_filter(
    scene: Scene,
    config: Mapping[str, Any],
    params: _SceneModeParams,
) -> bool:
    velocity = _initial_velocity_vector(scene)
    active_centers = scene.obstacle_centers[scene.obstacle_active]
    active_radii = scene.obstacle_radii[scene.obstacle_active]

    acceleration_bound = _acceleration_bound(config, scene.system)
    feasibility_margin = float(config["scene_train"]["init_feasibility_margin"])

    for center, radius in zip(active_centers, active_radii):
        center_delta = center - scene.start
        distance = float(np.linalg.norm(center_delta))
        if distance <= _ZERO_SPEED_EPS:
            stopping_distance = 0.0
        else:
            # v2.3.1 fix: inward speed = projected SPEED (m/s) toward the obstacle,
            # i.e. dot(velocity_m/s, unit(center - start)) -- NOT the prior projected
            # distance dot(unit(velocity), center - start). Units now match the
            # v^2/(2 a) max-braking stopping distance (03_train Sec 1.3 intent).
            u_dir = center_delta / distance
            inward_speed = max(0.0, float(np.dot(velocity, u_dir)))
            stopping_distance = inward_speed**2 / (2.0 * acceleration_bound)

        required_clearance = max(
            params.start_goal_clearance,
            stopping_distance + feasibility_margin,
        )
        if distance < float(radius) + required_clearance:
            return False

    return True


def _initial_velocity_vector(scene: Scene) -> Array:
    if scene.system == "double_integrator":
        if scene.initial_velocity is None:
            raise ValueError("Double Integrator scene is missing initial_velocity.")
        return scene.initial_velocity

    if scene.system == "unicycle":
        if scene.initial_speed is None or scene.initial_heading is None:
            raise ValueError(
                "Unicycle scene is missing initial_speed or initial_heading."
            )
        return np.array(
            [
                scene.initial_speed * np.cos(scene.initial_heading),
                scene.initial_speed * np.sin(scene.initial_heading),
            ],
            dtype=np.float64,
        )

    if scene.system == "quadrotor_planar":
        if scene.initial_velocity is None:
            raise ValueError("Quadrotor scene is missing initial_velocity.")
        return scene.initial_velocity

    raise ValueError(f"Unsupported system: {scene.system!r}")


def _acceleration_bound(config: Mapping[str, Any], system: str) -> float:
    if system == "double_integrator":
        return float(config["env"]["bounds"]["double_integrator"]["u_max"])
    if system == "unicycle":
        return float(config["env"]["bounds"]["unicycle"]["a_max"])
    if system == "quadrotor_planar":
        # Heuristic init-feasibility pre-filter only. The quadrotor is underactuated (instantaneous
        # accel direction is attitude-fixed), so a scalar bound is approximate; gravity g is the net
        # vertical authority magnitude. IC speeds <= v_init 1.5 make stopping_distance ~ 0.11 m, so
        # this bound barely gates. Recorded as PROTOCOL FOLLOW-UP (approximate underactuated filter).
        return float(config["env"]["quadrotor_planar"]["gravity"])
    raise ValueError(f"Unsupported system: {system!r}")
