from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import yaml

from src.common.maneuver_value import (
    _deadband_brake,
    _perp_directions,
    make_maneuver_h_fn,
    maneuver_maxh,
    maneuver_policy,
    maneuver_value,
    t_stop,
)
from src.common.rk4 import rk4_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import Scene

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _config() -> dict[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        merged[k] = _deep_merge(merged[k], v) if isinstance(v, Mapping) and isinstance(merged.get(k), Mapping) else v
    return merged


def _scene(center, radius, start, goal) -> Scene:
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
        initial_velocity=np.asarray([0.0, 0.0], dtype=np.float64),
    )


def _batched(scene, b):
    return batch_scenes([scene] * b, device=DEVICE, dtype=DTYPE)


def test_a_library_min_monotonicity() -> None:
    # (a) V_M(superset library) <= V_brake (brake-only) pointwise (min over a superset).
    config = _config()
    system = DoubleIntegrator(config)
    torch.manual_seed(0)
    b = 64
    scene = _batched(_scene([0.6, 0.0], 0.20, [-1.0, 0.0], [1.4, 0.0]), b)
    x = torch.cat([
        torch.empty(b, 2, dtype=DTYPE).uniform_(-1.5, 1.5),
        torch.empty(b, 2, dtype=DTYPE).uniform_(-2.0, 2.0),
    ], dim=1)
    v_brake = maneuver_value(x, scene, system, config, lateral_js=())
    v_lib = maneuver_value(x, scene, system, config, lateral_js=[4, 8])
    assert torch.all(v_lib <= v_brake + 1.0e-9)


def test_b_deadband_brake_stops_and_holds() -> None:
    # (b) deadband brake reaches rest within T_stop steps and holds position per-component after.
    config = _config()
    system = DoubleIntegrator(config)
    dt = float(config["env"]["dt"])
    u_max = float(config["env"]["bounds"]["double_integrator"]["u_max"])
    T = t_stop(config, system)
    x = torch.tensor([[0.0, 0.0, 2.0, -1.3], [0.3, -0.2, -1.7, 0.9]], dtype=DTYPE)
    for _ in range(T):
        x = rk4_step(system, x, _deadband_brake(x[:, 2:4], u_max, dt), dt)
    assert torch.all(x[:, 2:4].abs() < 1.0e-9)             # at rest within T_stop
    p_rest = x[:, :2].clone()
    for _ in range(5):
        x = rk4_step(system, x, _deadband_brake(x[:, 2:4], u_max, dt), dt)
    assert torch.allclose(x[:, :2], p_rest, atol=1.0e-12)  # holds position


def test_c_replanning_monotonicity() -> None:
    # (c) Prop-mcvalid(b): V_M(F(x, pi_M(x))) <= V_M(x) + 1e-6 on the SHIFT-CLOSED brake library
    # (lateral_js=() = arm A0). Exact: brake-from-x1 is the shift of brake-from-x, so max_{k>=1} h
    # <= max_{k>=0} h. The sparse-lateral library (shift(lateral_j)=lateral_{j-1}, absent) is NOT
    # shift-closed -> exact per-step monotonicity holds only for the brake component (build-log).
    config = _config()
    system = DoubleIntegrator(config)
    dt = float(config["env"]["dt"])
    torch.manual_seed(1)
    b = 128
    scene = _batched(_scene([0.5, 0.1], 0.25, [-1.0, 0.0], [1.6, 0.0]), b)
    x = torch.cat([
        torch.empty(b, 2, dtype=DTYPE).uniform_(-1.5, 1.5),
        torch.empty(b, 2, dtype=DTYPE).uniform_(-2.0, 2.0),
    ], dim=1)
    v0 = maneuver_value(x, scene, system, config, lateral_js=())
    u0 = maneuver_policy(x, scene, system, config, lateral_js=())
    x1 = rk4_step(system, x, u0, dt)
    v1 = maneuver_value(x1, scene, system, config, lateral_js=())
    assert torch.all(v1 <= v0 + 1.0e-6)


def test_c2_library_min_leq_brake_invariant() -> None:
    # (c2) the safety witness for the (non-shift-closed) lateral arms: {V_M<=0} subset {V_brake<=0}
    # since V_M = min over a superset containing brake => V_M <= V_brake pointwise (already (a));
    # here assert the SET inclusion on a random batch (every V_M<=0 state is V_brake<=0... trivially
    # false direction) -> assert V_M<=0 implies some maneuver reaches rest safely == maxh(argmin)<=0.
    config = _config()
    system = DoubleIntegrator(config)
    torch.manual_seed(3)
    b = 256
    scene = _batched(_scene([0.5, 0.1], 0.25, [-1.0, 0.0], [1.6, 0.0]), b)
    maxh = maneuver_maxh(x_rand := torch.cat([
        torch.empty(b, 2, dtype=DTYPE).uniform_(-1.5, 1.5),
        torch.empty(b, 2, dtype=DTYPE).uniform_(-2.0, 2.0)], dim=1),
        scene, system, config, lateral_js=[4, 8])
    v = maneuver_value(x_rand, scene, system, config, lateral_js=[4, 8])
    # V_M<=0 iff the best maneuver's whole-horizon worst-h <= 0 (open-loop safe-to-rest witness)
    assert torch.all((v <= 0.0) == (maxh.min(dim=1).values <= 0.0))


def test_d_autograd_matches_finite_difference() -> None:
    # (d) autograd gradient of V_M matches central finite differences on 5 states (rtol 1e-3), on the
    # brake-only library (no min-over-library kink), head-on states penetrating mid-ramp (stable
    # max-over-time argmax, signed_h in its smooth linear regime).
    config = _config()
    system = DoubleIntegrator(config)
    scene = _batched(_scene([0.6, 0.0], 0.20, [-1.0, 0.0], [1.4, 0.0]), 5)
    h_fn = make_maneuver_h_fn(system, config, lateral_js=())
    # SLOW mid-ramp states moving toward the obstacle (|v|_i <= u_max*dt = 0.1): the deadband brake is
    # entirely in the smooth -v/dt branch (stops in one step; NO sign() kink), so V_M is smooth in v
    # and autograd == central-FD. (In the full-decel -u_max*sign(v) regime autograd returns a valid
    # SUBGRADIENT that differs from FD by the brake-switching-time term — recorded in the build-log,
    # a property not a bug; the filter uses the subgradient.)
    x = torch.tensor([
        [0.20, 0.00, 0.08, 0.02],
        [0.18, 0.05, 0.06, -0.03],
        [0.22, -0.04, 0.09, 0.01],
        [0.16, 0.02, 0.05, 0.04],
        [0.24, -0.02, 0.07, -0.02],
    ], dtype=DTYPE, requires_grad=True)
    v = h_fn(x, scene)
    g_auto = torch.autograd.grad(v.sum(), x)[0]
    eps = 1.0e-6
    g_fd = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(4):
            xp = x.detach().clone(); xp[:, i] += eps
            xm = x.detach().clone(); xm[:, i] -= eps
            g_fd[:, i] = (h_fn(xp, scene) - h_fn(xm, scene)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, rtol=1.0e-3, atol=1.0e-4)


def test_e_perp_fallback_at_goal() -> None:
    # (e) e_perp world-axis fallback when the state sits on the goal (||goal - p|| < 1e-6).
    config = _config()
    system = DoubleIntegrator(config)
    scene = _batched(_scene([3.0, 3.0], 0.2, [1.0, 1.0], [1.0, 1.0]), 2)
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.3, 0.0]], dtype=DTYPE)
    perp = _perp_directions(x, scene, system)
    assert torch.allclose(perp, torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=DTYPE), atol=1.0e-9)


def test_f_brake_stops_and_holds_on_001_grid() -> None:
    # deploy-rate: dt_Vm=0.01 brake stops within T_stop(0.01) and holds at rest (regression for the
    # time-parameterized rollout; maxh must equal the k=0 signed_h once at rest and never grow).
    config = _config()
    system = DoubleIntegrator(config)
    dt = 0.01
    u_max = float(config["env"]["bounds"]["double_integrator"]["u_max"])
    v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"])
    import math
    t_stop_001 = math.ceil(v_max / (u_max * dt))
    scene = _batched(_scene([5.0, 5.0], 0.20, [-1.0, 0.0], [1.4, 0.0]), 4)  # far obstacle: h flat
    x = torch.tensor([[0.0, 0.0, 2.0, -1.5], [0.0, 0.0, -2.4, 0.8],
                      [0.0, 0.0, 0.05, 0.05], [0.0, 0.0, 1.0, 1.0]], dtype=DTYPE)
    # roll the deadband brake on the 0.01 grid and check it reaches rest within T_stop and holds
    xr = x.clone()
    for _ in range(t_stop_001):
        u = _deadband_brake(xr[:, 2:4], u_max, dt)
        xr = rk4_step(system, xr, u, dt)
    assert torch.all(xr[:, 2:4].abs() <= 1.0e-6)                      # stopped within T_stop(0.01)
    xr2 = rk4_step(system, xr, _deadband_brake(xr[:, 2:4], u_max, dt), dt)
    assert torch.allclose(xr2[:, :2], xr[:, :2], atol=1.0e-9)         # holds position at rest


def test_g_vm_dt005_reproduces_default() -> None:
    # regression guard: V_M with dt_override=0.05 (== config dt) and durations 0.05..0.40s (= j=1..8)
    # reproduces the un-overridden Stage-B library values EXACTLY on a probe batch.
    config = _config()
    system = DoubleIntegrator(config)
    torch.manual_seed(3)
    b = 48
    scene = _batched(_scene([0.6, 0.0], 0.20, [-1.0, 0.0], [1.4, 0.0]), b)
    x = torch.cat([torch.empty(b, 2, dtype=DTYPE).uniform_(-1.2, 1.2),
                   torch.empty(b, 2, dtype=DTYPE).uniform_(-2.0, 2.0)], dim=1)
    js = list(range(1, 9))
    v_default = maneuver_value(x, scene, system, config, lateral_js=js)                 # config dt=0.05
    v_dt = maneuver_value(x, scene, system, config, lateral_js=js, dt_override=0.05)    # explicit dt_Vm
    assert torch.allclose(v_dt, v_default, atol=1.0e-12)
