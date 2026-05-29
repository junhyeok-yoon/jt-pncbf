from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.envs.scene_init_fixed import sample_eval_fixed_scene, sample_train_fixed_scene
from src.eval.build_pools import build_pool, pool_stem, PoolSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_centered_sampler_has_one_active_origin_obstacle() -> None:
    config = _load_config()
    rng = np.random.default_rng(seed=123)
    scene = sample_train_fixed_scene(rng, config, "double_integrator")

    assert int(np.sum(scene.obstacle_active)) == 1
    assert bool(scene.obstacle_active[0])
    assert np.allclose(scene.obstacle_centers[0], np.zeros(2))
    assert scene.obstacle_radii[0] == float(config["obstacle"]["r_max"])
    assert not np.any(scene.obstacle_active[1:])
    assert np.allclose(scene.obstacle_centers[1:], 0.0)
    assert np.allclose(scene.obstacle_radii[1:], 0.0)
    assert _clearance(scene.start, scene) >= float(config["scene_train"]["start_goal_clearance"])
    assert _clearance(scene.goal, scene) >= float(config["scene_train"]["start_goal_clearance"])


def test_fixed_eval_pool_uses_fixed_filename_and_scenes() -> None:
    config = _load_config()
    config["env"]["obstacle_distribution"] = "fixed_centered"
    pool = build_pool(
        config,
        "double_integrator",
        PoolSpec(name="inloop", n_scenes=4, seed=12345),
    )

    assert pool_stem("inloop", "double_integrator", 200, 12345, "fixed_centered") == (
        "eval_inloop_di_fixed_n200_seed12345"
    )
    assert len(pool.scenes) == 4
    for scene in pool.scenes:
        assert int(np.sum(scene.obstacle_active)) == 1
        assert bool(scene.obstacle_active[0])
        assert np.allclose(scene.obstacle_centers[0], np.zeros(2))
        assert scene.obstacle_radii[0] == float(config["obstacle"]["r_max"])


def test_fixed_eval_sampler_uses_eval_clearance() -> None:
    config = _load_config()
    rng = np.random.default_rng(seed=321)
    scene = sample_eval_fixed_scene(rng, config, "double_integrator")

    assert _clearance(scene.start, scene) >= float(config["eval"]["scene"]["start_goal_clearance"])
    assert _clearance(scene.goal, scene) >= float(config["eval"]["scene"]["start_goal_clearance"])


def _clearance(position: np.ndarray, scene: Any) -> float:
    centers = scene.obstacle_centers[scene.obstacle_active]
    radii = scene.obstacle_radii[scene.obstacle_active]
    return float(np.min(np.linalg.norm(centers - position, axis=1) - radii))


def _load_config() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
