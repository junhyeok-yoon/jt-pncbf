"""Analytic braking conditioning-policy rollout for the value target (v2.4.0 axis).

Framework-agnostic (imports only from src.common / src.envs; never from src.frameworks).

The value target's conditioning policy is decoupled from the behavior policy: instead of
running the max-over-time avoid recurrence over the *stored* noised-policy trajectory, it is
run over a fixed analytic braking rollout from each minibatch state. With a FIXED conditioning
policy the discounted recurrence is a stationary contraction, so the unsafe-label support does
not drift as the task policy improves.

Double-integrator state layout is [px, py, vx, vy]; the brake policy is per-axis maximal
deceleration, which is the per-axis time-optimal stop and is box-feasible by construction
(|u_i| <= u_max), so no safety filter is needed.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.system import System


Tensor = torch.Tensor


def brake_policy(states: Tensor, u_max: float, eps_v: float) -> Tensor:
    """Per-axis maximal-deceleration brake for the double integrator.

    For velocity component v_i: u_i = -u_max * clip(v_i / eps_v, -1, 1). The eps_v deadband
    tapers the command linearly to zero as the axis speed drops below eps_v, avoiding chatter
    at rest; outside the deadband the command saturates to full decel -sign(v_i) * u_max.
    Box-feasible by construction. Batched over the leading dim; DI state layout [px,py,vx,vy].
    """
    velocity = states[:, 2:4]
    return -u_max * torch.clamp(velocity / eps_v, min=-1.0, max=1.0)


def brake_h_rollout(
    states: Tensor,
    scene: Any,
    system: System,
    obs_fn: Callable[[Tensor, Any], Tensor],
    T_b: int,
    u_max: float,
    eps_v: float,
    dt: float,
    h_scale: float,
    policy_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Roll the fixed brake policy T_b steps from each minibatch state and return the
    signed-h sequence plus the tail observation.

    - `states`: [B, state_dim] minibatch start states (full state, pos+vel).
    - `scene`: the batched scene carrying obstacles + goal (obstacle_centers/radii/active,
      goal); consumed by the canonical `signed_h` and `system.observation` unchanged, so h and
      obs match collection exactly (no re-implementation).
    - Integration uses the SAME `rk4_step` (incl. velocity clamp via system.wrap_state) as the
      policy BPTT rollout; no HardNet filter and no exploration noise are applied.
    - Returns `(h_seq [B, T_b+1], tail_obs [B, obs_dim])`. h_seq[:, k] = signed_h at rollout
      step k (k=0 is the minibatch state itself). tail_obs is the observation at the final
      rolled state (which is at rest for T_b >= max stop steps), for the target bootstrap.

    The brake policy is analytic in velocity only, so per-step observations are not needed to
    drive it; obs is recomputed only at the tail (its sole consumer, the target bootstrap).
    Fully batched; the whole computation is under torch.no_grad().

    `policy_fn` (v2.4.0 Step 2): when None, the analytic `brake_policy` is used and the rollout is
    byte-identical to Step 1. When given, it is a callable `(x, obs) -> u` (e.g. the Polyak target
    recovery policy pi_b_target); obs is then recomputed each step and passed to it. The value
    target form is unchanged either way; only the conditioning policy that generates the rollout
    differs.
    """
    if T_b < 1:
        raise ValueError(f"T_b must be >= 1, got {T_b}.")
    with torch.no_grad():
        x = states
        h_list = [signed_h(system.position(x), scene, h_scale)]
        for _ in range(int(T_b)):
            if policy_fn is None:
                u = brake_policy(x, u_max, eps_v)
            else:
                u = policy_fn(x, obs_fn(x, scene))
            x = rk4_step(system, x, u, dt)
            h_list.append(signed_h(system.position(x), scene, h_scale))
        h_seq = torch.stack(h_list, dim=1)          # [B, T_b+1]
        tail_obs = obs_fn(x, scene)                 # obs at final (rested) state
    return h_seq, tail_obs
