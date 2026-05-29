from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.control_net import ControlNet
from src.common.filter_hardnet import HardNetFilter
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_policy_bptt_path_detaches_value_network() -> None:
    config = _load_config()
    torch.manual_seed(2026)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    control_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)

    with torch.no_grad():
        for parameter in value_net.parameters():
            parameter.zero_()

    # The JT policy step freezes V_S while retaining h(x) gradients through state.
    for parameter in value_net.parameters():
        parameter.requires_grad_(False)

    hardnet = HardNetFilter(system, make_h_fn(value_net, system), config)
    scene = _tiny_scene()
    x = torch.tensor([[-1.2, -0.4, 0.15, 0.05]], dtype=DTYPE)
    goal = torch.as_tensor(scene.goal, dtype=DTYPE).unsqueeze(0)
    dt = float(config["env"]["dt"])
    loss = torch.zeros((), dtype=DTYPE)

    for _ in range(4):
        obs = system.observation(x, scene)
        u_nom = control_net(obs)
        u_safe, _ = hardnet(x, scene, u_nom)
        position_error = system.position(x) - goal
        loss = loss + torch.sum(position_error * position_error)
        loss = loss + 1.0e-3 * torch.sum(u_safe * u_safe)
        x = rk4_step(system, x, u_safe, dt)

    loss.backward()

    assert _grad_norm(value_net.parameters()) < 1.0e-12
    assert _grad_norm(control_net.parameters()) > 0.0


def _tiny_scene() -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([1.6, 1.0], dtype=np.float64)
    radii[0] = 0.25
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-1.2, -0.4], dtype=np.float64),
        goal=np.array([1.4, 0.8], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.array([0.15, 0.05], dtype=np.float64),
    )


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad))
    return float(total**0.5)


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
