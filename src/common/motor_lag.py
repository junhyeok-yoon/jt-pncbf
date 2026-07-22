"""First-order rotor-thrust lag (v2.7.4 preparation, Part A — NEW file, imported by nothing yet).

Real motors do not deliver commanded thrust instantly: a first-order lag with time constant ``tau``
relates the delivered per-rotor thrust ``f`` to the commanded ``f_cmd``,

    df/dt = (f_cmd - f) / tau .

Holding ``f_cmd`` constant over the control interval ``[t, t + dt]`` (zero-order hold, which is what a
digital controller actually does), this ODE has the EXACT closed-form solution

    f_next = f_cmd + (f - f_cmd) * exp(-dt / tau) .

That is the map advanced here. It is unconditionally stable: the retention factor ``alpha = exp(-dt/tau)``
lies in ``(0, 1]`` for every ``dt >= 0``, so the state contracts monotonically toward the command with no
overshoot at any timestep. Explicit Euler ``f_next = f + (dt/tau) * (f_cmd - f)`` is NOT: its multiplier
``1 - dt/tau`` has magnitude ``> 1`` once ``dt/tau > 2`` (at the current ``dt/tau = 0.05/0.02 = 2.5`` it is
``-1.5``), so it oscillates and diverges — and because the lag sits inside a differentiable rollout, that
divergence would surface as a training collapse, not as an obvious numerical error. The ZOH form is used
precisely to avoid that trap.

``f`` and ``f_cmd`` are per-rotor thrusts in the box ``[F_MIN, F_MAX] = [0, 4.905] N`` (F_MAX is the
per-rotor bound ``f_rotor_max`` used elsewhere in v2.7.3). Because ``f_next`` is a convex combination of
``f`` and ``f_cmd`` (weights ``alpha`` and ``1 - alpha``, both in ``[0, 1]`` and summing to 1), it stays in
the box whenever both inputs do — so NO clamp is applied. Clamping is deliberately omitted: without it the
map is exactly affine in ``(f, f_cmd)`` and its Jacobians are the constants

    d f_next / d f     = alpha        = exp(-dt / tau)
    d f_next / d f_cmd = 1 - alpha    = 1 - exp(-dt / tau) ,

which are shipped analytically below. The map is batched, on-device (device/dtype follow the inputs),
differentiable, and contains no Python loops.
"""
from __future__ import annotations

import math

from torch import Tensor

F_MIN: float = 0.0
F_MAX: float = 4.905
TAU_DEFAULT: float = 0.02


def lag_alpha(dt: float, tau: float = TAU_DEFAULT) -> float:
    """ZOH retention factor ``alpha = exp(-dt / tau) = d f_next / d f``. In ``(0, 1]`` for ``dt >= 0``."""
    if tau <= 0.0:
        return 0.0                                  # tau -> 0+: instantaneous thrust (alpha -> 0, f_next -> f_cmd)
    return math.exp(-float(dt) / float(tau))


def motor_lag_step(f: Tensor, f_cmd: Tensor, dt: float, tau: float = TAU_DEFAULT) -> Tensor:
    """Advance the rotor-thrust lag state one exact ZOH step.

    ``f`` (current delivered thrust) and ``f_cmd`` (commanded thrust) are broadcastable tensors with a
    trailing per-rotor dimension of size 4; any leading batch dims are preserved. Returns ``f_next`` on the
    same device/dtype. Differentiable w.r.t. both ``f`` and ``f_cmd``; no in-place ops, no Python loops.
    ``f_next`` is a convex combination of the two inputs, so it never leaves ``[F_MIN, F_MAX]`` when they
    do not — no clamp is applied (keeping the map affine with the constant Jacobians in ``motor_lag_jacobians``).
    """
    alpha = lag_alpha(dt, tau)                       # scalar; convex-combination weight on the current state
    return f_cmd + (f - f_cmd) * alpha


def motor_lag_jacobians(dt: float, tau: float = TAU_DEFAULT) -> tuple[float, float]:
    """Analytic (constant) Jacobians ``(d f_next / d f, d f_next / d f_cmd) = (alpha, 1 - alpha)``.

    Elementwise/diagonal: each rotor channel is independent, so the full Jacobian is ``alpha * I`` w.r.t.
    ``f`` and ``(1 - alpha) * I`` w.r.t. ``f_cmd``. These scalars are exact for the ZOH map at any ``dt``,
    ``tau`` and match autograd to floating-point tolerance.
    """
    alpha = lag_alpha(dt, tau)
    return alpha, 1.0 - alpha
