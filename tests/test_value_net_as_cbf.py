from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.filter_cbfqp import CBFQPFilter
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.envs.unicycle import Unicycle


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_value_net_h_fn_drives_cbf_qp_for_both_systems() -> None:
    config = _load_config()
    torch.manual_seed(1234)

    for system in [DoubleIntegrator(config), Unicycle(config)]:
        value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
        h_fn = make_h_fn(value_net, system)
        filter_qp = CBFQPFilter(system, h_fn, config)
        scene = _tiny_scene(system.name)
        x = _state_batch(system.name)
        goal = torch.as_tensor(scene.goal, dtype=DTYPE).unsqueeze(0)
        goal = goal.expand(x.shape[0], -1)
        u_nom = system.lqr_action(x, goal)

        grad = torch.autograd.grad(h_fn(x.requires_grad_(True), scene).sum(), x)[0]
        assert torch.isfinite(grad).all()
        if system.name == "double_integrator":
            assert torch.any(torch.abs(grad[:, 2:4]) > 1.0e-10)
        else:
            assert torch.any(torch.abs(grad[:, 3]) > 1.0e-10)

        u_safe, infeasible, slack = filter_qp(x.detach(), scene, u_nom.detach())
        assert u_safe.shape == u_nom.shape
        assert infeasible.shape == (x.shape[0],)
        assert slack.shape == (x.shape[0],)
        assert torch.isfinite(u_safe).all()
        assert torch.isfinite(slack).all()

        bounds = system.u_bounds.to(dtype=DTYPE)
        assert torch.all(u_safe >= bounds[:, 0] - 1.0e-6)
        assert torch.all(u_safe <= bounds[:, 1] + 1.0e-6)


def _state_batch(system_name: str) -> torch.Tensor:
    if system_name == "double_integrator":
        return torch.tensor(
            [
                [-1.0, 0.0, 0.25, -0.15],
                [-0.6, 0.2, -0.10, 0.20],
                [0.1, -0.3, 0.15, 0.05],
            ],
            dtype=DTYPE,
        )
    return torch.tensor(
        [
            [-1.0, 0.0, 0.1, 0.25],
            [-0.6, 0.2, -0.2, 0.30],
            [0.1, -0.3, 0.4, -0.20],
        ],
        dtype=DTYPE,
    )


def _tiny_scene(system_name: str) -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([1.0, 0.5])
    radii[0] = 0.25
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-1.0, 0.0], dtype=np.float64),
        goal=np.array([1.5, 0.0], dtype=np.float64),
        system=system_name,
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64)
        if system_name == "double_integrator"
        else None,
        initial_speed=0.1 if system_name == "unicycle" else None,
        initial_heading=0.0 if system_name == "unicycle" else None,
    )


def _load_config() -> Mapping[str, Any]:
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
