from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from src.common.outcomes import step_outcomes
from src.common.rk4 import rk4_step
from src.common.system import System


Tensor = torch.Tensor


@dataclass(frozen=True)
class RolloutResult:
    states: Tensor
    u_nom: Tensor
    u_safe: Tensor
    intervention_mask: Tensor
    infeasible: Tensor
    empty: Tensor | None = None          # v2.7.1 S1d: per-step empty_intersection (split from infeasible)
    singular: Tensor | None = None       # v2.7.1 S1d: per-step singular (split); additive, not the combined flag


def rollout(
    system: System,
    policy_fn: Callable[[Tensor, Any], Tensor],
    filter_fn: Callable[[Tensor, Tensor, Any], tuple[Tensor, Tensor]],
    scene: Any,
    x0: Tensor,
    max_steps: int,
    dt: float,
) -> RolloutResult:
    if x0.ndim != 2:
        raise ValueError(f"x0 must have shape [B, state_dim], got {tuple(x0.shape)}")
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative, got {max_steps}.")

    x = system.wrap_state(x0)
    _assert_finite("Initial state", x)

    states = [x]
    u_nom_steps = []
    u_safe_steps = []
    infeasible_steps = []
    empty_steps: list[Tensor] = []          # v2.7.1 S1d: split empty/singular (HardNet filter, if present)
    singular_steps: list[Tensor] = []
    _fobj = getattr(getattr(filter_fn, "__self__", None), "_filter", None)

    for _ in range(max_steps):
        u_nom = policy_fn(x, scene)
        if u_nom.shape != (x.shape[0], system.action_dim):
            raise ValueError(
                "policy_fn returned shape "
                f"{tuple(u_nom.shape)}, expected {(x.shape[0], system.action_dim)}."
            )
        _assert_finite("Nominal action", u_nom)

        u_safe, infeasible = filter_fn(x, u_nom, scene)
        if u_safe.shape != u_nom.shape:
            raise ValueError(
                f"filter_fn returned u_safe shape {tuple(u_safe.shape)}, "
                f"expected {tuple(u_nom.shape)}."
            )
        if infeasible.shape != (x.shape[0],):
            raise ValueError(
                f"filter_fn returned infeasible shape {tuple(infeasible.shape)}, "
                f"expected {(x.shape[0],)}."
            )
        _assert_finite("Executed action", u_safe)

        x = rk4_step(system, x, u_safe, dt)
        _assert_finite("State after RK4 step", x)

        states.append(x)
        u_nom_steps.append(u_nom)
        u_safe_steps.append(u_safe)
        infeasible_steps.append(infeasible.to(device=x.device, dtype=torch.bool))
        _le = getattr(_fobj, "last_empty", None); _ls = getattr(_fobj, "last_singular", None)
        empty_steps.append(_le.to(device=x.device, dtype=torch.bool) if _le is not None
                           else infeasible.to(device=x.device, dtype=torch.bool))
        singular_steps.append(_ls.to(device=x.device, dtype=torch.bool) if _ls is not None
                              else torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))

    if max_steps == 0:
        empty_action = x.new_empty((0, x.shape[0], system.action_dim))
        empty_bool = torch.empty((0, x.shape[0]), dtype=torch.bool, device=x.device)
        return RolloutResult(
            states=torch.stack(states, dim=0),
            u_nom=empty_action,
            u_safe=empty_action.clone(),
            intervention_mask=empty_bool,
            infeasible=empty_bool.clone(),
        )

    u_nom_tensor = torch.stack(u_nom_steps, dim=0)
    u_safe_tensor = torch.stack(u_safe_steps, dim=0)
    intervention = torch.linalg.norm(u_safe_tensor - u_nom_tensor, dim=-1) > 1.0e-3
    return RolloutResult(
        states=torch.stack(states, dim=0),
        u_nom=u_nom_tensor,
        u_safe=u_safe_tensor,
        intervention_mask=intervention,
        infeasible=torch.stack(infeasible_steps, dim=0),
        empty=torch.stack(empty_steps, dim=0),
        singular=torch.stack(singular_steps, dim=0),
    )


def rollout_lqr(
    system: System,
    scene: Any,
    x0: Tensor,
    max_steps: int,
    dt: float,
) -> Tensor:
    def policy_fn(x: Tensor, policy_scene: Any) -> Tensor:
        goal = torch.as_tensor(policy_scene.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, goal)

    def filter_fn(x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        infeasible = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return u_nom, infeasible

    return rollout(system, policy_fn, filter_fn, scene, x0, max_steps, dt).states


def rollout_eval(
    system: System,
    policy_fn: Callable[[Tensor, Any], Tensor],
    filter_fn: Callable[[Tensor, Tensor, Any], tuple[Tensor, Tensor]],
    scene: Any,
    x0: Tensor,
    max_steps: int,
    dt: float,
    config: Mapping[str, Any],
) -> RolloutResult:
    if x0.ndim != 2:
        raise ValueError(f"x0 must have shape [B, state_dim], got {tuple(x0.shape)}")
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative, got {max_steps}.")

    x = system.wrap_state(x0)
    _assert_finite("Initial state", x)

    states = [x]
    u_nom_steps = []
    u_safe_steps = []
    infeasible_steps = []
    empty_steps: list[Tensor] = []          # v2.7.1 S1d: split empty/singular from the HardNet filter, if present
    singular_steps: list[Tensor] = []
    _fobj = getattr(getattr(filter_fn, "__self__", None), "_filter", None)
    physical_done = _physical_done_mask(system, scene, torch.stack(states, dim=0), config)

    for _ in range(max_steps):
        u_nom = policy_fn(x, scene)
        if u_nom.shape != (x.shape[0], system.action_dim):
            raise ValueError(
                "policy_fn returned shape "
                f"{tuple(u_nom.shape)}, expected {(x.shape[0], system.action_dim)}."
            )
        _assert_finite("Nominal action", u_nom)

        u_safe, infeasible = filter_fn(x, u_nom, scene)
        if u_safe.shape != u_nom.shape:
            raise ValueError(
                f"filter_fn returned u_safe shape {tuple(u_safe.shape)}, "
                f"expected {tuple(u_nom.shape)}."
            )
        if infeasible.shape != (x.shape[0],):
            raise ValueError(
                f"filter_fn returned infeasible shape {tuple(infeasible.shape)}, "
                f"expected {(x.shape[0],)}."
            )
        _assert_finite("Executed action", u_safe)

        done_action = physical_done.unsqueeze(1)
        u_nom = torch.where(done_action, torch.zeros_like(u_nom), u_nom)
        u_safe = torch.where(done_action, torch.zeros_like(u_safe), u_safe)
        infeasible = torch.where(
            physical_done,
            torch.zeros_like(infeasible, dtype=torch.bool),
            infeasible.to(device=x.device, dtype=torch.bool),
        )

        x_next = rk4_step(system, x, u_safe, dt)
        _assert_finite("State after RK4 step", x_next)
        x = torch.where(done_action, x, x_next)

        states.append(x)
        u_nom_steps.append(u_nom)
        u_safe_steps.append(u_safe)
        infeasible_steps.append(infeasible)
        _le = getattr(_fobj, "last_empty", None); _ls = getattr(_fobj, "last_singular", None)
        _eb = (_le.to(device=x.device, dtype=torch.bool) if _le is not None else infeasible.clone())
        _sb = (_ls.to(device=x.device, dtype=torch.bool) if _ls is not None else torch.zeros_like(infeasible))
        empty_steps.append(torch.where(physical_done, torch.zeros_like(_eb), _eb))
        singular_steps.append(torch.where(physical_done, torch.zeros_like(_sb), _sb))
        physical_done = physical_done | _physical_done_mask(
            system,
            scene,
            torch.stack(states, dim=0),
            config,
        )

    if max_steps == 0:
        empty_action = x.new_empty((0, x.shape[0], system.action_dim))
        empty_bool = torch.empty((0, x.shape[0]), dtype=torch.bool, device=x.device)
        return RolloutResult(
            states=torch.stack(states, dim=0),
            u_nom=empty_action,
            u_safe=empty_action.clone(),
            intervention_mask=empty_bool,
            infeasible=empty_bool.clone(),
        )

    u_nom_tensor = torch.stack(u_nom_steps, dim=0)
    u_safe_tensor = torch.stack(u_safe_steps, dim=0)
    intervention = torch.linalg.norm(u_safe_tensor - u_nom_tensor, dim=-1) > 1.0e-3
    return RolloutResult(
        states=torch.stack(states, dim=0),
        u_nom=u_nom_tensor,
        u_safe=u_safe_tensor,
        intervention_mask=intervention,
        infeasible=torch.stack(infeasible_steps, dim=0),
        empty=torch.stack(empty_steps, dim=0),
        singular=torch.stack(singular_steps, dim=0),
    )


def _physical_done_mask(
    system: System,
    scene: Any,
    states: Tensor,
    config: Mapping[str, Any],
) -> Tensor:
    masks = step_outcomes(states, scene, system, config)
    return masks.collided[-1] | masks.goal_reached[-1] | masks.oob[-1]


def _assert_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")
