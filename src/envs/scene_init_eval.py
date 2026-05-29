from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.envs.scene_init import Scene, _SceneModeParams, _sample_scene


def sample_eval_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
) -> Scene:
    params = _SceneModeParams(
        mode="eval",
        min_start_goal_dist=float(config["eval"]["scene"]["min_start_goal_dist"]),
        start_goal_clearance=float(config["eval"]["scene"]["start_goal_clearance"]),
    )
    return _sample_scene(rng, config, system, params)
