from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.eval.build_pools import EvaluationPool
from src.eval.evaluate import (
    EVAL_EPISODE_COLUMNS,
    EVAL_METRIC_COLUMNS,
    _eval_row,
    evaluate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


class IdentityFramework:
    def __init__(self, system: DoubleIntegrator) -> None:
        self.system = system

    def policy(self, x: torch.Tensor, scene: Any) -> torch.Tensor:
        return torch.zeros(
            (x.shape[0], self.system.action_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def filter(
        self,
        x: torch.Tensor,
        u_nom: torch.Tensor,
        scene: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)


def test_evaluate_returns_schema_and_hand_computed_cps() -> None:
    config = _load_config()
    config["eval"]["max_steps"] = 2
    config["eval"]["bootstrap"]["n_resample"] = 64
    system = DoubleIntegrator(config)
    pool = EvaluationPool(
        name="unit",
        system="double_integrator",
        n_scenes=2,
        seed=11,
        scenes=[_goal_scene(), _collision_scene()],
    )

    result = evaluate(
        IdentityFramework(system),
        pool,
        config,
        mode="in_loop",
        step=3,
        ckpt_name="unit.pt",
    )

    assert list(result.eval_row.keys()) == EVAL_METRIC_COLUMNS
    assert list(result.episode_rows[0].keys()) == EVAL_EPISODE_COLUMNS
    assert result.episode_rows[0]["outcome"] == "goal"
    assert result.episode_rows[0]["n_steps"] == 0
    assert result.episode_rows[1]["outcome"] == "collision"
    assert result.episode_rows[1]["n_steps"] == 0
    collision_states = result.trajectories[1].filtered.states
    assert torch.allclose(
        collision_states,
        collision_states[:1].expand_as(collision_states),
    )
    assert torch.count_nonzero(result.trajectories[1].filtered.u_safe) == 0
    assert result.eval_row["reach"] == 0.5
    assert result.eval_row["collision"] == 0.5
    assert result.eval_row["stuck"] == 0.0
    assert result.eval_row["infeasibility"] == 0.0
    assert 0.0 <= result.eval_row["saturation_rate"] <= 1.0
    assert result.eval_row["cps"] == -0.5
    assert result.eval_row["cps_ci_lo"] <= result.eval_row["cps"]
    assert result.eval_row["cps_ci_hi"] >= result.eval_row["cps"]
    for row in result.episode_rows:
        assert row["stuck"] == 0.0
        assert np.isnan(row["min_window_displacement"])
        assert 0.0 <= row["saturation_step_frac"] <= 1.0


def test_eval_row_uses_five_outcome_cps_and_stuck_bins() -> None:
    config = _load_config()
    config["eval"]["bootstrap"]["n_resample"] = 64
    pool = EvaluationPool(
        name="unit",
        system="double_integrator",
        n_scenes=5,
        seed=11,
        scenes=[_goal_scene()] * 5,
    )
    rows = [
        _episode_record("goal", 0.04),
        _episode_record("collision", 0.07),
        _episode_record("oob", 0.12),
        _episode_record("stuck", 0.17),
        _episode_record("timeout", 0.35),
    ]

    eval_row = _eval_row(
        mode="final",
        step=10,
        ckpt_name="unit.pt",
        pool=pool,
        n_scenes=5,
        episode_rows=rows,
        config=config,
    )

    assert eval_row["reach"] == 0.2
    assert eval_row["collision"] == 0.2
    assert eval_row["oob"] == 0.2
    assert eval_row["stuck"] == 0.2
    assert eval_row["timeout"] == 0.2
    assert eval_row["infeasibility"] == 0.2
    assert abs(eval_row["cps"] - (-0.66)) < 1.0e-12
    assert eval_row["stuck_bin_00_05"] == 0.2
    assert eval_row["stuck_bin_05_10"] == 0.2
    assert eval_row["stuck_bin_10_15"] == 0.2
    assert eval_row["stuck_bin_15_20"] == 0.2
    assert eval_row["stuck_bin_20_25"] == 0.0
    assert eval_row["stuck_bin_25_30"] == 0.0


def _goal_scene() -> Scene:
    centers, radii, active = _obstacles()
    centers[0] = np.array([3.0, 3.0], dtype=np.float64)
    radii[0] = 0.2
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([0.0, 0.0], dtype=np.float64),
        goal=np.array([0.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _collision_scene() -> Scene:
    centers, radii, active = _obstacles()
    centers[0] = np.array([0.0, 0.0], dtype=np.float64)
    radii[0] = 0.5
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([0.0, 0.0], dtype=np.float64),
        goal=np.array([3.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _obstacles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((12, 2), dtype=np.float64),
        np.zeros(12, dtype=np.float64),
        np.zeros(12, dtype=np.bool_),
    )


def _episode_record(outcome: str, min_window_displacement: float) -> dict[str, float | str]:
    reach = 1.0 if outcome == "goal" else 0.0
    collision = 1.0 if outcome == "collision" else 0.0
    oob = 1.0 if outcome == "oob" else 0.0
    stuck = 1.0 if outcome == "stuck" else 0.0
    timeout = 1.0 if outcome == "timeout" else 0.0
    infeasible = 0.2
    return {
        "outcome": outcome,
        "reach": reach,
        "collision": collision,
        "oob": oob,
        "stuck": stuck,
        "timeout": timeout,
        "infeasible_step_frac": infeasible,
        "saturation_step_frac": 0.0,
        "min_window_displacement": min_window_displacement,
    }


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
