from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.control_net import ControlNet
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.collection import make_replay_buffers
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss
from src.frameworks.oc_pncbf.collection import OCReplayBuffer


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_jt_policy_bptt_routes_gradients_to_policy_only() -> None:
    config = _load_config()
    config["training"]["jt"]["bptt_T"] = 3
    torch.manual_seed(1234)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    buffer = OCReplayBuffer(capacity=8)
    scene = _tiny_scene()
    states = torch.tensor(
        [
            [-1.0, -0.4, 0.10, 0.05],
            [-0.9, -0.35, 0.10, 0.05],
            [-0.8, -0.30, 0.10, 0.05],
            [-0.7, -0.25, 0.10, 0.05],
        ],
        dtype=DTYPE,
    )
    h = torch.tensor([-0.4, -0.3, -0.2, -0.1], dtype=DTYPE)
    buffer.append(scene, states, h)
    buffers = make_replay_buffers(capacity=8)
    buffers.policy = buffer
    batch = buffers.policy.sample_tensor_batch(batch_size=3)

    value_net.zero_grad(set_to_none=True)
    policy_net.zero_grad(set_to_none=True)
    loss = policy_bptt_loss(
        system=system,
        policy_net=policy_net,
        value_net=value_net,
        batch=batch,
        config=config,
    )
    loss.total.backward()

    assert grad_norm(value_net.parameters()) < 1.0e-12
    assert grad_norm(policy_net.parameters()) > 0.0


def test_adaptive_sigma_hook_moves_up_when_unsafe_signal_is_low() -> None:
    from src.frameworks.jt_pncbf.collection import adaptive_sigma_update

    config = _load_config()
    sigma = float(config["schedules"]["sigma"]["init"])
    updated = adaptive_sigma_update(sigma, unsafe_fraction=0.0, config=config)
    assert updated > sigma


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
        start=np.array([-1.0, -0.4], dtype=np.float64),
        goal=np.array([1.4, 0.8], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.array([0.10, 0.05], dtype=np.float64),
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
