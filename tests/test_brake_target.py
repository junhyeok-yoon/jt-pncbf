from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from src.common.brake_rollout import brake_h_rollout, brake_policy
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene
from src.frameworks.jt_pncbf.collection import make_replay_buffers
from src.frameworks.jt_pncbf.losses import value_targets
from src.frameworks.oc_pncbf.collection import OCReplayBuffer
from src.frameworks.oc_pncbf.value_target import pncbf_target


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEVICE = torch.device("cpu")
U_MAX = 2.0
EPS_V = 0.05
T_B = 30
DT = 0.05
H_SCALE = 0.35


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


def _batched(scene: Scene, batch: int):
    return batch_scenes([scene] * batch, device=DEVICE, dtype=DTYPE)


def test_brake_rest_far_from_obstacles_stays_at_rest() -> None:
    # (i) A rest state far from any obstacle: the brake command is exactly zero, the state does
    # not move under RK4, and the signed-h sequence is constant and safe (< 0).
    system = DoubleIntegrator(_config())
    rest = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    scene = _batched(_scene([3.0, 3.0], 0.20, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]), 1)

    assert torch.allclose(brake_policy(rest, U_MAX, EPS_V), torch.zeros_like(rest[:, 2:4]))
    stepped = rk4_step(system, rest, brake_policy(rest, U_MAX, EPS_V), DT)
    assert torch.allclose(stepped, rest, atol=1.0e-9)

    h_seq, _ = brake_h_rollout(rest, scene, system, system.observation,
                               T_B, U_MAX, EPS_V, DT, H_SCALE)
    assert h_seq.shape == (1, T_B + 1)
    assert float(h_seq.max() - h_seq.min()) < 1.0e-9   # constant
    assert float(h_seq.max()) < 0.0                    # safe / far


def test_brake_doomed_fast_approach_labels_unsafe() -> None:
    # (ii) A fast head-on approach whose braking distance exceeds the clearance is doomed: the
    # rolled h-sequence contains h > 0, and the pncbf_target (as value_targets computes it,
    # index 0 of the brake rollout) is > 0.
    system = DoubleIntegrator(_config())
    scene = _batched(_scene([0.6, 0.0], 0.15, [0.0, 0.0], [2.0, 0.0], [1.4, 0.0]), 1)
    state = torch.tensor([[0.0, 0.0, 1.4, 0.0]], dtype=DTYPE)  # v^2/(2a)=0.49 > clearance 0.45

    h_seq, tail_obs = brake_h_rollout(state, scene, system, system.observation,
                                      T_B, U_MAX, EPS_V, DT, H_SCALE)
    assert float(h_seq.max()) > 0.0

    time_major = h_seq.transpose(0, 1).contiguous()             # [T_b+1, B]
    y = pncbf_target(time_major, 1.0, DT, 0.0, torch.tensor([-1.0], dtype=DTYPE))[0]
    assert float(y) > 0.0


def test_brake_recoverable_min_clearance_matches_analytic() -> None:
    # (iii) A recoverable head-on approach: rolling the brake policy, the closest-approach
    # clearance equals the analytic (clearance0 - v_in^2 / (2 u_max)) within tolerance.
    system = DoubleIntegrator(_config())
    center = np.array([0.6, 0.0])
    radius = 0.15
    v0 = 1.0
    x = torch.tensor([[0.0, 0.0, v0, 0.0]], dtype=DTYPE)
    clearance0 = float(np.linalg.norm(center - np.array([0.0, 0.0])) - radius)   # 0.45
    analytic_min = clearance0 - v0 ** 2 / (2.0 * U_MAX)                           # 0.45 - 0.25 = 0.20

    c = torch.tensor(center, dtype=DTYPE)
    min_clear = float("inf")
    for _ in range(T_B + 1):
        clr = float(torch.linalg.norm(x[0, :2] - c)) - radius
        min_clear = min(min_clear, clr)
        x = rk4_step(system, x, brake_policy(x, U_MAX, EPS_V), DT)

    # Tolerance: RK4 at dt=0.05 plus the eps_v=0.05 deadband (decel tapers below |v|<eps_v,
    # adding a small crawl beyond the ideal constant-decel stop) shift the realized stop by
    # < ~0.03 m; 0.05 m covers it. Recoverable => positive min clearance.
    assert min_clear > 0.0
    assert abs(min_clear - analytic_min) < 0.05


def test_flag_off_is_byte_identical_and_flag_on_dispatches(monkeypatch) -> None:
    # (iv) With conditioning=task_stored (default), the brake module is NEVER invoked and the
    # output equals the pre-change reference. With conditioning=brake it IS invoked.
    import src.frameworks.jt_pncbf.losses as losses_mod

    config = _config()
    torch.manual_seed(0)
    system = DoubleIntegrator(config)
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    target_value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)

    scene = _scene([0.6, 0.2], 0.20, [-1.0, -0.4], [1.4, 0.8], [0.10, 0.05])
    states = torch.tensor(
        [[-1.0, -0.4, 0.10, 0.05], [-0.9, -0.35, 0.12, 0.06],
         [-0.8, -0.30, 0.14, 0.07], [-0.7, -0.25, 0.16, 0.08]], dtype=DTYPE)
    h = torch.tensor([-0.4, -0.3, -0.2, -0.1], dtype=DTYPE)
    buf = OCReplayBuffer(capacity=8)
    buf.append(scene, states, h)
    buffers = make_replay_buffers(capacity=8)
    buffers.value = buf
    batch = buffers.value.sample_tensor_batch(batch_size=3)

    lambda_disc, target_rhs = 1.0, 0.0

    def _raise(*args, **kwargs):
        raise AssertionError("brake_h_rollout must not be called when conditioning != 'brake'")

    monkeypatch.setattr(losses_mod, "brake_h_rollout", _raise)

    cfg_off = _deep_merge(config, {"value_target": {"conditioning": "task_stored"}})
    got = value_targets(system=system, target_value_net=target_value_net, batch=batch,
                        lambda_disc=lambda_disc, target_rhs=target_rhs, config=cfg_off)

    # pre-change reference (the task_stored formula, verbatim)
    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_value_net.target_h(tail_obs)
    ref = pncbf_target(batch.h_sequence, lambda_disc, float(config["env"]["dt"]),
                       target_rhs, bootstrap_tail).detach()
    ref = ref.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)
    assert torch.allclose(got, ref, atol=1.0e-9)

    # flag ON: the monkeypatched brake must now be reached (proves dispatch)
    cfg_on = _deep_merge(config, {"value_target": {"conditioning": "brake"}})
    with pytest.raises(AssertionError, match="brake_h_rollout must not be called"):
        value_targets(system=system, target_value_net=target_value_net, batch=batch,
                      lambda_disc=lambda_disc, target_rhs=target_rhs, config=cfg_on)


def test_flag_on_end_to_end_targets_finite_and_clamped() -> None:
    # brake branch runs end-to-end (no monkeypatch): finite targets in [-1, 1], shape [B].
    config = _deep_merge(_config(), {"value_target": {"conditioning": "brake"}})
    torch.manual_seed(0)
    system = DoubleIntegrator(config)
    target_value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=DTYPE)
    scene = _scene([0.6, 0.0], 0.20, [-1.0, 0.0], [1.4, 0.0], [0.10, 0.0])
    states = torch.tensor(
        [[-1.0, 0.0, 0.10, 0.0], [-0.6, 0.0, 0.8, 0.0], [0.2, 0.0, 1.4, 0.0], [-0.4, 0.1, 0.0, 0.0]],
        dtype=DTYPE)
    h = torch.tensor([-0.4, -0.2, 0.1, -0.3], dtype=DTYPE)
    buf = OCReplayBuffer(capacity=8)
    buf.append(scene, states, h)
    buffers = make_replay_buffers(capacity=8)
    buffers.value = buf
    batch = buffers.value.sample_tensor_batch(batch_size=4)

    y = value_targets(system=system, target_value_net=target_value_net, batch=batch,
                      lambda_disc=1.0, target_rhs=0.0, config=config)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert torch.all(y >= -1.0) and torch.all(y <= 1.0)


def _config() -> dict[str, Any]:
    import yaml
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
