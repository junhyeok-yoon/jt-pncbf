from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from src.common.brake_rollout import brake_policy
from src.common.control_net import ControlNet
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.collection import make_replay_buffers
from src.frameworks.jt_pncbf.losses import grad_norm, value_targets
from src.frameworks.jt_pncbf.recovery_policy import RecoveryPolicy, recovery_bptt_loss
from src.frameworks.oc_pncbf.collection import OCReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEVICE = torch.device("cpu")
U_MAX = 2.0
EPS_V = 0.05


def _scene(center, radius, start, goal, v0) -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.asarray(center, dtype=np.float64)
    radii[0] = float(radius)
    active[0] = True
    return Scene(
        obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
        start=np.asarray(start, dtype=np.float64), goal=np.asarray(goal, dtype=np.float64),
        system="double_integrator", mode="synthetic",
        initial_velocity=np.asarray(v0, dtype=np.float64),
    )


def _mk_recovery(system, config):
    torch.manual_seed(0)
    return RecoveryPolicy(system.obs_dim, system, config, U_MAX, EPS_V).to(dtype=DTYPE)


def _batch_from(system, scene, states, h, bs):
    buf = OCReplayBuffer(capacity=16)
    buf.append(scene, states, h)
    buffers = make_replay_buffers(capacity=16)
    buffers.value = buf
    return buffers.value.sample_tensor_batch(batch_size=bs)


def test_zero_init_equals_brake_and_target_matches_brake_target() -> None:
    # (i) at init the residual is zero => pi_b == pi_brake; and the learned_recovery value target
    # equals the brake value target on a fixed batch (pi_b_target == pi_brake at init).
    config = _config()
    system = DoubleIntegrator(config)
    rec = _mk_recovery(system, config)

    x = torch.randn(64, 4, dtype=DTYPE)
    scene = batch_scenes([_scene([0.6, 0.2], 0.2, [0, 0], [1, 1], [0, 0])] * 64,
                         device=DEVICE, dtype=DTYPE)
    obs = system.observation(x, scene)
    assert torch.allclose(rec(x, obs), brake_policy(x, U_MAX, EPS_V), atol=1.0e-6)

    torch.manual_seed(1)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    sc = _scene([0.6, 0.0], 0.2, [-1.0, 0.0], [1.4, 0.0], [0.10, 0.0])
    states = torch.tensor([[-1.0, 0.0, 0.10, 0.0], [0.2, 0.0, 1.4, 0.0], [-0.4, 0.1, 0.0, 0.0]],
                          dtype=DTYPE)
    batch = _batch_from(system, sc, states, torch.tensor([-0.4, 0.1, -0.3], dtype=DTYPE), 3)

    y_rec = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                          lambda_disc=1.0, target_rhs=0.0, config=config, recovery_policy=rec)
    cfg_brake = _deep_merge(config, {"value_target": {"conditioning": "brake"}})
    y_brake = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                            lambda_disc=1.0, target_rhs=0.0, config=cfg_brake)
    assert torch.allclose(y_rec, y_brake, atol=1.0e-6)


def test_avoid_loss_routes_gradients_to_recovery_only() -> None:
    # (ii) one L_b backward: grad reaches xi only; V_S and pi_theta get no gradient.
    config = _config()
    system = DoubleIntegrator(config)
    rec = _mk_recovery(system, config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)

    sc = _scene([0.5, 0.0], 0.15, [0.0, 0.0], [1.4, 0.0], [1.4, 0.0])
    batch = _batch_from(system, sc, torch.tensor(
        [[0.0, 0.0, 1.4, 0.0], [0.1, 0.0, 1.3, 0.0]], dtype=DTYPE),
        torch.tensor([0.1, 0.05], dtype=DTYPE), 2)

    for net in (rec, value_net, policy_net):
        net.zero_grad(set_to_none=True)
    loss, _ = recovery_bptt_loss(system=system, recovery_policy=rec, batch=batch, config=config)
    loss.backward()
    assert grad_norm(rec.parameters()) > 0.0
    assert grad_norm(value_net.parameters()) < 1.0e-12
    assert grad_norm(policy_net.parameters()) < 1.0e-12


def test_avoid_learning_decreases_loss_and_produces_lateral_action() -> None:
    # (iii) doomed-by-braking but swerve-recoverable state: ~200 L_b steps strictly decrease L_b
    # and yield nonzero lateral (perpendicular-to-approach) action at the start state.
    config = _config()
    system = DoubleIntegrator(config)
    rec = _mk_recovery(system, config)
    sc = _scene([0.5, 0.0], 0.15, [0.0, 0.0], [2.0, 0.0], [1.4, 0.0])  # c0=0.35 < stop 0.49
    x0 = torch.tensor([[0.0, 0.0, 1.4, 0.0]], dtype=DTYPE)
    batch = _batch_from(system, sc, x0.repeat(2, 1), torch.tensor([0.1, 0.1], dtype=DTYPE), 2)

    opt = torch.optim.AdamW(rec.parameters(), lr=3.0e-3)
    loss0 = None
    for i in range(200):
        opt.zero_grad(set_to_none=True)
        loss, _ = recovery_bptt_loss(system=system, recovery_policy=rec, batch=batch, config=config)
        if i == 0:
            loss0 = float(loss.item())
        loss.backward()
        opt.step()
    lossN = float(loss.item())
    assert lossN < loss0 - 1.0e-3

    # lateral action = y-component at the head-on (x-axis) approach start state
    scene1 = batch_scenes([sc], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        u0 = rec(x0, system.observation(x0, scene1))
    assert float(u0[0, 1].abs()) > 1.0e-2


def test_target_conditioned_output_is_detached_and_ignores_online_policy() -> None:
    # (iv) value_targets(learned_recovery) carries no grad_fn and depends ONLY on pi_b_target
    # (perturbing the online pi_b does not change it).
    config = _config()
    system = DoubleIntegrator(config)
    online = _mk_recovery(system, config)
    target = deepcopy(online)
    target.requires_grad_(False)
    torch.manual_seed(2)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    sc = _scene([0.6, 0.0], 0.2, [-1.0, 0.0], [1.4, 0.0], [0.10, 0.0])
    batch = _batch_from(system, sc, torch.tensor(
        [[-1.0, 0.0, 0.10, 0.0], [0.2, 0.0, 1.4, 0.0]], dtype=DTYPE),
        torch.tensor([-0.4, 0.1], dtype=DTYPE), 2)

    y1 = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                       lambda_disc=1.0, target_rhs=0.0, config=config, recovery_policy=target)
    assert y1.grad_fn is None
    with torch.no_grad():
        for p in online.parameters():
            p.add_(5.0)
    y2 = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                       lambda_disc=1.0, target_rhs=0.0, config=config, recovery_policy=target)
    assert torch.allclose(y1, y2, atol=1.0e-12)


def test_flag_guard_recovery_not_invoked_off_and_invoked_on() -> None:
    # (v) conditioning brake/task_stored never touches the recovery policy; learned_recovery does.
    config = _config()
    system = DoubleIntegrator(config)
    torch.manual_seed(3)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    sc = _scene([0.6, 0.2], 0.2, [-1.0, -0.4], [1.4, 0.8], [0.10, 0.05])
    batch = _batch_from(system, sc, torch.tensor(
        [[-1.0, -0.4, 0.10, 0.05], [-0.8, -0.30, 0.14, 0.07]], dtype=DTYPE),
        torch.tensor([-0.4, -0.2], dtype=DTYPE), 2)

    def _raise(*a, **k):
        raise AssertionError("recovery policy must not be invoked when conditioning != learned_recovery")

    for cond in ("brake", "task_stored"):
        cfg = _deep_merge(config, {"value_target": {"conditioning": cond}})
        value_targets(system=system, target_value_net=target_vnet, batch=batch,
                      lambda_disc=1.0, target_rhs=0.0, config=cfg, recovery_policy=_raise)

    with pytest.raises(AssertionError, match="must not be invoked"):
        value_targets(system=system, target_value_net=target_vnet, batch=batch,
                      lambda_disc=1.0, target_rhs=0.0, config=config, recovery_policy=_raise)


def _config() -> dict[str, Any]:
    import yaml
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    # these tests exercise the recovery feature; pin conditioning so they are independent of the
    # exp_config default (which changes across v2.4.0 steps).
    return _deep_merge(_deep_merge(base, exp), {"value_target": {"conditioning": "learned_recovery"}})


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
