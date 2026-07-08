from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import yaml

from src.common.control_net import ControlNet
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.collection import make_replay_buffers
from src.frameworks.jt_pncbf.losses import grad_norm, value_loss, value_targets
from src.frameworks.oc_pncbf.collection import OCReplayBuffer
from src.frameworks.oc_pncbf.value_target import (
    compute_disc_avoid_terms,
    gamma_from_lambda,
    lambda_schedule_value,
    pncbf_target,
    rpcbf_target,
    schedule_value,
    target_from_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def test_pncbf_target_matches_hand_reference() -> None:
    h_seq = torch.tensor([-0.8, -0.2, 0.4, -0.5, 0.1], dtype=DTYPE)
    lambda_disc = torch.tensor(-torch.log(torch.tensor(0.5, dtype=DTYPE)), dtype=DTYPE)
    bootstrap_tail = torch.tensor(0.8, dtype=DTYPE)

    pure = pncbf_target(h_seq, lambda_disc, 1.0, 0.0, bootstrap_tail)
    full_mix = pncbf_target(h_seq, lambda_disc, 1.0, 1.0, bootstrap_tail)

    expected_pure = torch.tensor([-0.35, 0.1, 0.4, -0.2, 0.1], dtype=DTYPE)
    expected_full = torch.tensor([-0.35, 0.1, 0.4, 0.15, 0.8], dtype=DTYPE)
    assert torch.allclose(pure, expected_pure, atol=1.0e-6)
    assert torch.allclose(full_mix, expected_full, atol=1.0e-6)

    unclamped = torch.tensor([-2.0, 1.5], dtype=DTYPE)
    target = pncbf_target(unclamped, 0.1, 1.0, 1.0, torch.tensor(3.0, dtype=DTYPE))
    assert torch.all(target >= -1.0)
    assert torch.all(target <= 1.0)


def test_disc_avoid_terms_match_reference_form() -> None:
    costs = torch.tensor([[-0.8, -0.3], [-0.2, 0.1], [0.4, -0.4]], dtype=DTYPE)
    lambda_disc = -torch.log(torch.tensor(0.5, dtype=DTYPE))
    lhs, int_rhs, discount_rhs = compute_disc_avoid_terms(costs, lambda_disc, dt=1.0)

    expected_lhs = torch.tensor([[-0.35, -0.1], [0.1, 0.1], [0.4, -0.4]], dtype=DTYPE)
    expected_int = torch.tensor([[-0.45, -0.125], [-0.1, 0.05], [0.0, 0.0]], dtype=DTYPE)
    expected_discount = torch.tensor([[0.25, 0.25], [0.5, 0.5], [1.0, 1.0]], dtype=DTYPE)
    assert torch.allclose(lhs, expected_lhs, atol=1.0e-6)
    assert torch.allclose(int_rhs, expected_int, atol=1.0e-6)
    assert torch.allclose(discount_rhs, expected_discount, atol=1.0e-6)


def test_rpcbf_target_is_normalized_smooth_max() -> None:
    window = torch.tensor([[-0.6, -0.2, 0.0, 0.4]], dtype=DTYPE)
    betas = [1.0, 5.0, 20.0, 100.0]
    targets = torch.stack([rpcbf_target(window, beta) for beta in betas])

    mean_value = torch.mean(window, dim=1)
    max_value = torch.max(window, dim=1).values
    assert torch.all(targets >= mean_value - 1.0e-12)
    assert torch.all(targets <= max_value + 1.0e-12)
    assert torch.all(targets[1:] >= targets[:-1] - 1.0e-12)
    assert torch.allclose(targets[-1], max_value, atol=2.0e-2)


def test_rpcbf_horizon_invariance_and_stability() -> None:
    beta = 5.0
    base = torch.tensor([[-0.5, 0.2, 0.4]], dtype=DTYPE)
    padded = torch.tensor([[-0.5, 0.2, 0.4, 0.4]], dtype=DTYPE)
    base_target = rpcbf_target(base, beta)
    padded_target = rpcbf_target(padded, beta)
    assert torch.abs(padded_target - base_target) < 0.06

    max_value = torch.max(base, dim=1).values
    assert torch.allclose(rpcbf_target(base, 200.0), max_value, atol=1.0e-2)
    assert torch.allclose(rpcbf_target(padded, 200.0), max_value, atol=1.0e-2)

    large = torch.tensor([[-50.0, 0.0, 50.0]], dtype=DTYPE)
    stable = rpcbf_target(large, beta)
    assert torch.isfinite(stable).all()
    assert torch.all(stable >= -1.0)
    assert torch.all(stable <= 1.0)


def test_dispatcher_and_schedule_value() -> None:
    config = _load_config()
    h_windows = torch.tensor([[-0.4, 0.1, 0.2]], dtype=DTYPE)
    rpcbf_config = _deep_merge(config, {"value_target": {"type": "rpcbf"}})
    dispatched = target_from_config(rpcbf_config, h_windows=h_windows)
    expected = rpcbf_target(h_windows, float(config["value_target"]["rpcbf_beta"]))
    assert torch.allclose(dispatched, expected)

    schedule = config["schedules"]["gamma_disc"]
    lambda_value = lambda_schedule_value(
        schedule,
        epoch_index=0,
        total_epochs=int(config["training"]["oc_pncbf"]["epochs"]),
        dt=float(config["env"]["dt"]),
    )
    gamma_value = gamma_from_lambda(lambda_value, float(config["env"]["dt"]))
    assert abs(gamma_value - 0.95) < 1.0e-12

    rhs_schedule = config["schedules"]["target_rhs"]
    rhs_value = schedule_value(
        rhs_schedule,
        n_sched=450,
        n_steps=int(config["training"]["oc_pncbf"]["epochs"]),
    )
    # asserts the LIVE exp_config target_rhs schedule at n_sched=450 (frac 0.5): 0.5*final.
    # target_rhs.final=0.9 (baseline carry) => 0.45. (Tracks target_rhs.final; config-coupled.)
    assert abs(rhs_value - 0.45) < 1.0e-12


# --- v2.4.2 task_raw_lagged conditioning ---------------------------------------------------

def _rl_scene(center, radius, start, goal, v0) -> Scene:
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


def _rl_batch(system, scene, states, h, bs):
    buf = OCReplayBuffer(capacity=16)
    buf.append(scene, states, h)
    buffers = make_replay_buffers(capacity=16)
    buffers.value = buf
    return buffers.value.sample_tensor_batch(batch_size=bs)


def _rl_config() -> dict[str, Any]:
    return _deep_merge(_load_config(), {"value_target": {"conditioning": "task_raw_lagged"}})


def test_raw_lagged_label_matches_manual_unfiltered_reroll() -> None:
    # (a) the task_raw_lagged label equals a hand-rolled pncbf_target of the UNFILTERED pi_b
    # h-sequence (no HardNet, no noise, deterministic), tail = target_h at the rolled tail state.
    config = _rl_config()
    torch.manual_seed(0)
    system = DoubleIntegrator(config)
    pi_b = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    pi_b.requires_grad_(False)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)

    sc = _rl_scene([0.6, 0.0], 0.20, [-1.0, 0.0], [1.4, 0.0], [0.10, 0.0])
    states = torch.tensor(
        [[-1.0, 0.0, 0.10, 0.0], [0.2, 0.0, 1.4, 0.0], [-0.4, 0.1, 0.0, 0.0]], dtype=DTYPE)
    batch = _rl_batch(system, sc, states, torch.tensor([-0.4, 0.1, -0.3], dtype=DTYPE), 3)

    lambda_disc, target_rhs = 1.0, 0.0
    got = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                        lambda_disc=lambda_disc, target_rhs=target_rhs, config=config,
                        lagged_policy=pi_b)

    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])
    T_b = int(config["value_target"]["raw_lagged"]["T_b"])
    ub = system.u_bounds.to(dtype=DTYPE)
    lo, hi = ub[:, 0], ub[:, 1]
    with torch.no_grad():
        x = batch.states
        h_list = [signed_h(system.position(x), batch.scene, h_scale)]
        for _ in range(T_b):
            u = torch.clamp(pi_b(system.observation(x, batch.scene)), min=lo, max=hi)
            x = rk4_step(system, x, u, dt)
            h_list.append(signed_h(system.position(x), batch.scene, h_scale))
        h_seq = torch.stack(h_list, dim=1)
        tail = target_vnet.target_h(system.observation(x, batch.scene))
    ref = pncbf_target(h_seq.transpose(0, 1).contiguous(), lambda_disc, dt,
                       target_rhs, tail).detach()[0]
    assert got.shape == (3,)
    assert torch.allclose(got, ref, atol=1.0e-9)
    assert torch.all(got >= -1.0) and torch.all(got <= 1.0)


def test_raw_lagged_value_loss_no_grad_to_policy_or_pi_b() -> None:
    # (b) a value-loss backward reaches V_S only; neither the task policy nor pi_b receives grad
    # (the re-roll runs under no_grad and the label is detached).
    config = _rl_config()
    torch.manual_seed(1)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)
    pi_b = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)  # grad ON to prove no leak

    sc = _rl_scene([0.5, 0.0], 0.15, [0.0, 0.0], [1.4, 0.0], [1.4, 0.0])
    batch = _rl_batch(system, sc, torch.tensor(
        [[0.0, 0.0, 1.4, 0.0], [0.1, 0.0, 1.3, 0.0]], dtype=DTYPE),
        torch.tensor([0.1, 0.05], dtype=DTYPE), 2)

    for net in (value_net, policy_net, pi_b):
        net.zero_grad(set_to_none=True)
    res = value_loss(system=system, value_net=value_net, target_value_net=target_vnet,
                     batch=batch, lambda_disc=1.0, target_rhs=0.0, config=config, lagged_policy=pi_b)
    res.total.backward()
    assert grad_norm(value_net.parameters()) > 0.0
    assert all(p.grad is None for p in policy_net.parameters())
    assert all(p.grad is None for p in pi_b.parameters())


def test_task_stored_output_unchanged_and_independent_of_lagged_policy() -> None:
    # (c) flag-off parity: with conditioning=task_stored the output equals the verbatim task_stored
    # formula and is invariant to lagged_policy (the new branch/param never perturbs task_stored).
    config = _deep_merge(_load_config(), {"value_target": {"conditioning": "task_stored"}})
    torch.manual_seed(2)
    system = DoubleIntegrator(config)
    target_vnet = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    pi_b = ControlNet(system.obs_dim, system, config).to(dtype=DTYPE)

    sc = _rl_scene([0.6, 0.2], 0.20, [-1.0, -0.4], [1.4, 0.8], [0.10, 0.05])
    states = torch.tensor(
        [[-1.0, -0.4, 0.10, 0.05], [-0.8, -0.30, 0.14, 0.07], [-0.7, -0.25, 0.16, 0.08]], dtype=DTYPE)
    batch = _rl_batch(system, sc, states, torch.tensor([-0.4, -0.2, -0.1], dtype=DTYPE), 3)

    y_none = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                           lambda_disc=1.0, target_rhs=0.0, config=config)
    y_lag = value_targets(system=system, target_value_net=target_vnet, batch=batch,
                          lambda_disc=1.0, target_rhs=0.0, config=config, lagged_policy=pi_b)

    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_vnet.target_h(tail_obs)
    ref = pncbf_target(batch.h_sequence, 1.0, float(config["env"]["dt"]), 0.0, bootstrap_tail).detach()
    ref = ref.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_none, ref, atol=1.0e-9)
    assert torch.allclose(y_lag, ref, atol=1.0e-9)


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
