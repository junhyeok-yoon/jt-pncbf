from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from src.envs.scene_init import (
    Array,
    Scene,
    _has_start_goal_clearance,
    _make_scene,
    _passes_unavoidable_collision_filter,
    _sample_start_goal,
    _MAX_RETRIES,
    _SceneModeParams,
)


def sample_train_fixed_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
) -> Scene:
    params = _SceneModeParams(
        mode="train",
        min_start_goal_dist=float(config["scene_train"]["min_start_goal_dist"]),
        start_goal_clearance=float(config["scene_train"]["start_goal_clearance"]),
    )
    return _sample_fixed_scene(rng, config, system, params)


def sample_eval_fixed_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
) -> Scene:
    params = _SceneModeParams(
        mode="eval",
        min_start_goal_dist=float(config["eval"]["scene"]["min_start_goal_dist"]),
        start_goal_clearance=float(config["eval"]["scene"]["start_goal_clearance"]),
    )
    return _sample_fixed_scene(rng, config, system, params)


def _sample_fixed_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
    params: _SceneModeParams,
) -> Scene:
    if system not in {"double_integrator", "unicycle"}:
        raise ValueError(f"Unsupported system: {system!r}")

    centers, radii, active = _fixed_centered_obstacles(config)
    for _ in range(_MAX_RETRIES):
        start, goal = _sample_start_goal(rng, config, params)
        scene = _make_scene(
            rng,
            config,
            system,
            params.mode,
            centers.copy(),
            radii.copy(),
            active.copy(),
            start,
            goal,
        )
        if not _has_start_goal_clearance(scene, params.start_goal_clearance):
            continue
        if not _passes_unavoidable_collision_filter(scene, config, params):
            continue
        return scene

    raise RuntimeError(
        "Fixed-centered scene sampling exceeded retry cap "
        f"({_MAX_RETRIES}) for mode={params.mode!r}, system={system!r}."
    )


def _fixed_centered_obstacles(
    config: Mapping[str, Any],
) -> tuple[Array, Array, npt.NDArray[np.bool_]]:
    n_max = int(config["obstacle"]["n_max"])
    radius = float(config["obstacle"]["r_max"])
    centers = np.zeros((n_max, 2), dtype=np.float64)
    radii = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.bool_)
    radii[0] = radius
    active[0] = True
    return centers, radii, active
