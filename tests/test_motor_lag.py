"""Tests for the first-order rotor-thrust lag (v2.7.4 preparation, Part A).

The last test (tau -> 0 recovers instantaneous thrust) is the invariant that lets this axis be ablated
against v2.7.3: with tau -> 0 the lagged plant reduces to today's direct-thrust plant to fp tolerance.
"""
import math

import torch

from src.common.motor_lag import (
    F_MAX,
    F_MIN,
    TAU_DEFAULT,
    lag_alpha,
    motor_lag_jacobians,
    motor_lag_step,
)

DT = 0.05          # current control interval
TAU = TAU_DEFAULT  # 0.02 s  -> dt/tau = 2.5 (where explicit Euler would diverge)


def _rand_box(shape, gen):
    return F_MIN + (F_MAX - F_MIN) * torch.rand(shape, generator=gen, dtype=torch.float64)


def test_one_step_fraction_and_monotone_convergence():
    """One step closes exactly (1 - exp(-dt/tau)) of the gap; iterating converges monotonically to f_cmd."""
    gen = torch.Generator().manual_seed(0)
    f = _rand_box((256, 4), gen)
    f_cmd = _rand_box((256, 4), gen)
    frac = 1.0 - math.exp(-DT / TAU)

    f1 = motor_lag_step(f, f_cmd, DT, TAU)
    # fractional approach (f1 - f) / (f_cmd - f) == 1 - exp(-dt/tau) on every channel
    approached = (f1 - f) / (f_cmd - f)
    assert torch.allclose(approached, torch.full_like(approached, frac), atol=1e-12)

    # monotone convergence: the gap |f - f_cmd| strictly shrinks each step and -> 0
    fk = f.clone()
    prev_gap = torch.linalg.norm(fk - f_cmd, dim=1)
    for _ in range(200):
        fk = motor_lag_step(fk, f_cmd, DT, TAU)
        gap = torch.linalg.norm(fk - f_cmd, dim=1)
        assert torch.all(gap <= prev_gap + 1e-15)          # never grows (monotone)
        prev_gap = gap
    assert torch.allclose(fk, f_cmd, atol=1e-9)            # converged to the command


def test_constant_command_is_fixed_point():
    """f == f_cmd is a fixed point at any (dt, tau)."""
    gen = torch.Generator().manual_seed(1)
    f_cmd = _rand_box((128, 4), gen)
    for dt, tau in [(DT, TAU), (0.02, 0.02), (0.2, 0.01), (0.001, 0.5)]:
        f_next = motor_lag_step(f_cmd, f_cmd, dt, tau)
        assert torch.allclose(f_next, f_cmd, atol=1e-15)


def test_box_invariance_for_any_command_in_box():
    """f_next never leaves [F_MIN, F_MAX] for any f, f_cmd in the box, across a wide dt/tau sweep."""
    gen = torch.Generator().manual_seed(2)
    for dt_over_tau in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]:
        tau, dt = 0.02, 0.02 * dt_over_tau
        f = _rand_box((512, 4), gen)
        f_cmd = _rand_box((512, 4), gen)
        # include the box corners explicitly
        f[:4] = torch.tensor([F_MIN, F_MAX, F_MIN, F_MAX], dtype=torch.float64)
        f_cmd[:4] = torch.tensor([F_MAX, F_MIN, F_MIN, F_MAX], dtype=torch.float64)
        f_next = motor_lag_step(f, f_cmd, dt, tau)
        assert torch.all(f_next >= F_MIN - 1e-12)
        assert torch.all(f_next <= F_MAX + 1e-12)


def test_shipped_jacobians_match_autograd():
    """The analytic (alpha, 1 - alpha) Jacobians match autograd elementwise."""
    gen = torch.Generator().manual_seed(3)
    for dt, tau in [(DT, TAU), (0.02, 0.02), (0.2, 0.05), (0.005, 0.05)]:
        d_df, d_dfcmd = motor_lag_jacobians(dt, tau)
        f = _rand_box((16, 4), gen).requires_grad_(True)
        f_cmd = _rand_box((16, 4), gen).requires_grad_(True)
        f_next = motor_lag_step(f, f_cmd, dt, tau)
        # sum-reduce so each input element's grad equals its own diagonal Jacobian entry
        (grad_f, grad_fcmd) = torch.autograd.grad(f_next.sum(), (f, f_cmd))
        assert torch.allclose(grad_f, torch.full_like(grad_f, d_df), atol=1e-12)
        assert torch.allclose(grad_fcmd, torch.full_like(grad_fcmd, d_dfcmd), atol=1e-12)
        # and the two Jacobians partition unity (convex combination)
        assert abs((d_df + d_dfcmd) - 1.0) < 1e-15


def test_stable_over_dt_over_tau_range():
    """No blow-up / NaN for dt/tau in [0.1, 10]; the retention factor stays in (0,1) so iteration contracts."""
    gen = torch.Generator().manual_seed(4)
    for dt_over_tau in [0.1, 0.3, 1.0, 2.5, 6.0, 10.0]:
        tau, dt = 0.02, 0.02 * dt_over_tau
        alpha = lag_alpha(dt, tau)
        assert 0.0 < alpha < 1.0
        f = _rand_box((64, 4), gen)
        f_cmd = _rand_box((64, 4), gen)
        for _ in range(500):
            f = motor_lag_step(f, f_cmd, dt, tau)
            assert torch.all(torch.isfinite(f))
        assert torch.allclose(f, f_cmd, atol=1e-6)          # contracted to the command, no divergence


def test_tau_to_zero_recovers_instantaneous_thrust():
    """tau -> 0 recovers today's direct-thrust plant (f_next == f_cmd) to fp tolerance — the ablation anchor."""
    gen = torch.Generator().manual_seed(5)
    f = _rand_box((256, 4), gen)
    f_cmd = _rand_box((256, 4), gen)
    for tau in [1e-6, 1e-9, 1e-12]:
        f_next = motor_lag_step(f, f_cmd, DT, tau)
        assert torch.allclose(f_next, f_cmd, atol=1e-12), f"tau={tau} did not recover instantaneous thrust"
    # exact tau == 0 sentinel also yields the instantaneous command
    assert torch.allclose(motor_lag_step(f, f_cmd, DT, 0.0), f_cmd, atol=0.0)
