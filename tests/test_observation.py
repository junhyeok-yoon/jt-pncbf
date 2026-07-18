from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.envs.quadrotor_planar import QuadrotorPlanar
from src.envs.scene_init import Scene
from src.envs.unicycle import Unicycle


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_inactive_slots_remain_zero_padded_regardless_of_raw_values() -> None:
    config = _load_config()
    system = DoubleIntegrator(config)
    scene = Scene(
        obstacle_centers=np.full((12, 2), 100.0, dtype=np.float64),
        obstacle_radii=np.full(12, 0.8, dtype=np.float64),
        obstacle_active=np.zeros(12, dtype=np.bool_),
        start=np.zeros(2, dtype=np.float64),
        goal=np.array([1.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )
    x = torch.zeros((1, system.state_dim), dtype=DTYPE)

    obstacle_block = system.observation(x, scene)[0, 4:].reshape(-1, 3)

    assert torch.equal(obstacle_block, torch.zeros_like(obstacle_block))


def test_di_uses_raw_world_relative_obstacle_features() -> None:
    config = _load_config()
    system = DoubleIntegrator(config)
    scene = _ranked_scene("double_integrator")
    x = torch.tensor([[0.5, -0.25, 0.0, 0.0]], dtype=DTYPE)

    feature = system.observation(x, scene)[0, 4:7]

    assert torch.allclose(
        feature,
        torch.tensor([2.5, 0.25, 0.2], dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_unicycle_uses_raw_body_frame_obstacle_features() -> None:
    config = _load_config()
    system = Unicycle(config)
    scene = _ranked_scene("unicycle")
    x = torch.tensor([[0.0, 0.0, np.pi / 2.0, 0.0]], dtype=DTYPE)

    feature = system.observation(x, scene)[0, 3:6]

    assert torch.allclose(
        feature,
        torch.tensor([0.0, -3.0, 0.2], dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_topk_uses_active_mask_and_low_index_tiebreak() -> None:
    config = _load_config()
    system = DoubleIntegrator(config)
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([0.0, 0.0], dtype=np.float64)
    centers[1] = np.array([1.0, 0.0], dtype=np.float64)
    centers[2] = np.array([-1.0, 0.0], dtype=np.float64)
    radii[:3] = 0.2
    active[1:3] = True
    scene = Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.zeros(2, dtype=np.float64),
        goal=np.array([2.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )

    block = system.observation(torch.zeros((1, system.state_dim), dtype=DTYPE), scene)[
        0,
        4:,
    ].reshape(-1, 3)

    assert torch.allclose(block[0], torch.tensor([1.0, 0.0, 0.2], dtype=DTYPE))
    assert torch.allclose(block[1], torch.tensor([-1.0, 0.0, 0.2], dtype=DTYPE))
    assert torch.equal(block[2:], torch.zeros_like(block[2:]))


# ---- v2.7.0: quadrotor obs (sin θ, cos θ) append (01_env §3.3, dim 20 → 22) + DI/unicycle bit-parity ----
# Golden observations captured from the PRE-edit builder on a fixed scene + states (see _golden_scene).
# DI/unicycle goldens guard bit-identity across the quadrotor-only change; the quadrotor golden is the
# first-20 head (byte-identical to the pre-v2.7.0 layout); indices 20,21 must equal (sin θ, cos θ).
_QUAD_HEAD_GOLDEN = [0.1006091187, -0.3462337436, 0.4, 0.3541866749, -0.8213110247, 1.3808254558,
                     -0.378577681, 0.3, 0.0726408061, 1.6385125307, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0]
_DI_GOLDEN = [0.3, -0.2, 0.8, -0.4, 1.3, 0.6, 0.3, -1.0, 1.3, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_UNI_GOLDEN = [0.5, 0.6460611086, -0.618550761, 1.4192495599, 0.1890256248, 0.3, -0.5711602205,
               1.5374576425, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _golden_scene(system: str) -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([1.5, 0.5]); centers[1] = np.array([-0.8, 1.2])
    radii[0] = 0.3; radii[1] = 0.45; active[:2] = True
    return Scene(
        obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
        start=np.zeros(2, dtype=np.float64), goal=np.array([1.0, -0.5], dtype=np.float64),
        system=system, mode="unit",
        initial_velocity=np.array([0.3, -0.2]) if system != "unicycle" else None,
        initial_speed=0.5 if system == "unicycle" else None,
        initial_heading=0.3 if system == "unicycle" else None,
        initial_attitude=0.7 if system == "quadrotor_planar" else None,
        initial_omega=0.4 if system == "quadrotor_planar" else None,
    )


def test_quadrotor_obs_appends_sin_cos_theta_dim22() -> None:
    config = _load_config()
    system = QuadrotorPlanar(config)
    assert system.obs_dim == 22                                       # 2+1+2+3*k_obs(15)+2
    theta = 0.7
    x = torch.tensor([[0.2, -0.1, theta, 0.3, -0.2, 0.4]], dtype=DTYPE)
    obs = system.observation(x, _golden_scene("quadrotor_planar"))
    assert obs.shape == (1, 22)
    # first 20 components byte-identical to the pre-v2.7.0 layout
    assert torch.allclose(obs[0, :20], torch.tensor(_QUAD_HEAD_GOLDEN, dtype=DTYPE), atol=1e-9, rtol=0.0)
    # appended attitude = (sin θ, cos θ) at indices 20, 21
    assert torch.allclose(obs[0, 20:22],
                          torch.tensor([np.sin(theta), np.cos(theta)], dtype=DTYPE), atol=1e-12, rtol=0.0)


def test_di_unicycle_obs_bit_identical_parity() -> None:
    config = _load_config()
    di = DoubleIntegrator(config)
    assert di.obs_dim == 19
    di_obs = di.observation(torch.tensor([[0.2, -0.1, 0.3, -0.2]], dtype=DTYPE), _golden_scene("double_integrator"))
    assert torch.allclose(di_obs[0], torch.tensor(_DI_GOLDEN, dtype=DTYPE), atol=1e-9, rtol=0.0)
    uni = Unicycle(config)
    assert uni.obs_dim == 18
    uni_obs = uni.observation(torch.tensor([[0.2, -0.1, 0.3, 0.5]], dtype=DTYPE), _golden_scene("unicycle"))
    assert torch.allclose(uni_obs[0], torch.tensor(_UNI_GOLDEN, dtype=DTYPE), atol=1e-9, rtol=0.0)


def _ranked_scene(system: str) -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([3.0, 0.0], dtype=np.float64)
    centers[1] = np.array([4.0, 0.0], dtype=np.float64)
    radii[0] = 0.2
    radii[1] = 0.2
    active[:2] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.zeros(2, dtype=np.float64),
        goal=np.array([2.0, 0.0], dtype=np.float64),
        system=system,
        mode="unit",
        initial_velocity=np.zeros(2, dtype=np.float64)
        if system == "double_integrator"
        else None,
        initial_speed=0.0 if system == "unicycle" else None,
        initial_heading=0.0 if system == "unicycle" else None,
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
