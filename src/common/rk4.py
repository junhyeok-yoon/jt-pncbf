from __future__ import annotations

import torch

from src.common.system import System


Tensor = torch.Tensor


def rk4_step(system: System, x: Tensor, u: Tensor, dt: float) -> Tensor:
    half_dt = 0.5 * dt
    k1 = system.dynamics(x, u)
    k2 = system.dynamics(x + half_dt * k1, u)
    k3 = system.dynamics(x + half_dt * k2, u)
    k4 = system.dynamics(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return system.wrap_state(x_next)
