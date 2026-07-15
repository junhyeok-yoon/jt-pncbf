"""The single brake-to-rest primitive (m_0) and its stopping-horizon / bounds accessors, system-dispatched.

One primitive, all consumers (labels `m0_value_raw`, `family_value`, the `exact_m0` channel, the S' shield).
Replaces the three inlined copies that assumed `v = x[:, 2:4]` and read the literal `"double_integrator"`
bounds (v2.5.1). Dispatch is on `system.name`:

- **double_integrator** — componentwise deadband brake on the velocity 2-vector `v = x[:, 2:4]`:
  `u_i = -u_max·sgn(v_i)` if `|v_i| > u_max·dt` else `-v_i/dt`.
- **unicycle** — longitudinal brake on the scalar speed `v = x[:, 3]`: `a = -a_max·sgn(v)` if
  `|v| > a_max·dt` else `-v/dt`, and **`omega = 0`** (the heading is held — an exact-channel SPEC, not a
  tuning choice: any nonzero turn rule would add a design degree of freedom the exactness argument lacks).

NO DEFAULT is ever supplied for a bounds key: a base-only config missing `unicycle.v_max` must raise a loud
`KeyError`, never silently substitute a value (a wrong `v_max` silently corrupts `T_stop` and the shield
`thresh` — see changes.md §3 item 1).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

Tensor = torch.Tensor


def brake_accel_bound(system, config: Mapping[str, Any]) -> float:
    """The brake's acceleration bound: DI `u_max`, unicycle `a_max`. No default (loud KeyError if absent)."""
    b = config["env"]["bounds"][system.name]
    return float(b["u_max"] if system.name == "double_integrator" else b["a_max"])


def speed_max(system, config: Mapping[str, Any]) -> float:
    """The speed cap `v_max` (dynamics constant; the wrap_state clamp). No default (loud KeyError if absent)."""
    return float(config["env"]["bounds"][system.name]["v_max"])


def t_stop(system, config: Mapping[str, Any], dt: float) -> int:
    """Brake-to-rest horizon `T_stop = ceil(v_max / (brake_bound · dt))`. DI and unicycle both = 25 at the
    canonical scale (v_max 2.5, bound 2.0, dt 0.05). Derived, never a literal."""
    return int(math.ceil(speed_max(system, config) / (brake_accel_bound(system, config) * dt)))


def deadband_brake(x: Tensor, system, config: Mapping[str, Any], dt: float) -> Tensor:
    """m_0 deadband brake ACTION for state `x`. Returns u [B, action_dim] toward rest."""
    umax = brake_accel_bound(system, config)
    if system.name == "double_integrator":
        v = x[:, 2:4]
        return torch.where(v.abs() > umax * dt, -umax * torch.sign(v), -v / dt)
    if system.name == "unicycle":
        v = x[:, 3]
        a = torch.where(v.abs() > umax * dt, -umax * torch.sign(v), -v / dt)
        return torch.stack([a, torch.zeros_like(a)], dim=1)   # omega = 0 (heading held)
    raise ValueError(f"deadband_brake: unsupported system {system.name!r}")
