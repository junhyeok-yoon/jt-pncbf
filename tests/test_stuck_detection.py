from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.outcomes import resolve_outcome, step_outcomes
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_stuck_detection_on_crafted_batch() -> None:
    config = _load_config()
    config["env"]["stuck_window_steps"] = 60
    config["env"]["stuck_radius"] = 0.10
    system = DoubleIntegrator(config)
    scene = _empty_scene()
    states = torch.zeros((71, 3, system.state_dim), dtype=DTYPE)

    states[:, 0, 0] = torch.linspace(0.0, 2.0, 71, dtype=DTYPE)
    states[:, 1, 0] = 0.02 * torch.sin(torch.linspace(0.0, 12.0, 71, dtype=DTYPE))
    states[:, 1, 1] = 0.02 * torch.cos(torch.linspace(0.0, 12.0, 71, dtype=DTYPE))

    masks = step_outcomes(states, scene, system, config)
    resolved = resolve_outcome(masks)

    assert resolved.outcome == ["timeout", "stuck", "stuck"]
    assert int(resolved.event_step[1]) == 60
    assert int(resolved.event_step[2]) == 60
    assert float(resolved.min_window_displacement[0]) > 0.10
    assert float(resolved.min_window_displacement[1]) <= 0.05
    assert float(resolved.min_window_displacement[2]) == 0.0


def _empty_scene() -> Scene:
    return Scene(
        obstacle_centers=np.zeros((12, 2), dtype=np.float64),
        obstacle_radii=np.zeros(12, dtype=np.float64),
        obstacle_active=np.zeros(12, dtype=np.bool_),
        start=np.zeros(2, dtype=np.float64),
        goal=np.array([3.0, 3.0], dtype=np.float64),
        system="double_integrator",
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _load_config() -> dict[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
