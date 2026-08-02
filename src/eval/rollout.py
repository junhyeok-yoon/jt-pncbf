from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from src.common.motor_lag import motor_lag_step
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

    # v2.7.5 deploy control-rate axis: decouple the control period (dt_ctrl) from the integration step
    # (dt_sim == `dt`). Control (policy + filter) is recomputed once per control period and HELD (ZOH) across
    # `substeps` integration steps. EVAL PATH ONLY; `rollout()` and every training path are untouched. Default
    # dt_ctrl = dt_sim -> substeps = 1 -> bit-identical to the pre-v2.7.5 behaviour.
    dt_ctrl = float(config.get("eval", {}).get("dt_ctrl", dt)) if isinstance(config, Mapping) else dt
    if dt_ctrl <= 0.0:
        raise ValueError(f"eval.dt_ctrl must be positive, got {dt_ctrl}.")
    substeps = int(round(dt_ctrl / dt))
    if substeps < 1 or abs(substeps * dt - dt_ctrl) > 1e-9 * max(1.0, dt_ctrl):
        raise ValueError(
            f"eval.dt_ctrl ({dt_ctrl}) must be a positive integer multiple of dt_sim ({dt}); "
            f"got ratio {dt_ctrl / dt}."
        )

    # v2.8.0 Phase-2 D4 (C5): optional first-order rotor-thrust lag on the SIM grid, flag-gated and default
    # off. When eval.actuator_lag_tau <= 0 the applied action IS the commanded action (bit-identical to the
    # pre-D4 path). When > 0 the plant applies a ZOH-lagged thrust while the barrier is still enforced on the
    # commanded action — the transfer D4 probes.
    _lag_tau = float(config.get("eval", {}).get("actuator_lag_tau", 0.0)) if isinstance(config, Mapping) else 0.0
    _u_applied: Tensor | None = None

    states = [x]
    u_nom_steps = []
    u_safe_steps = []
    infeasible_steps = []
    empty_steps: list[Tensor] = []          # v2.7.1 S1d: split empty/singular from the HardNet filter, if present
    singular_steps: list[Tensor] = []
    _fobj = getattr(getattr(filter_fn, "__self__", None), "_filter", None)
    physical_done = _physical_done_mask(system, scene, torch.stack(states, dim=0), config)

    _held: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
    for _i in range(max_steps):
        if _i % substeps == 0:                       # recompute control at the start of each control period
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
            # capture the filter's empty/singular flags at compute time (the filter is NOT re-invoked during a
            # hold, so these are held alongside the action).
            _le = getattr(_fobj, "last_empty", None); _ls = getattr(_fobj, "last_singular", None)
            _eb_c = (_le.to(device=x.device, dtype=torch.bool) if _le is not None else infeasible.clone().bool())
            _sb_c = (_ls.to(device=x.device, dtype=torch.bool) if _ls is not None else torch.zeros_like(infeasible, dtype=torch.bool))
            # v2.7.5 graph-retention fix: drop this control computation's autograd graph. The eval needs no
            # gradient THROUGH the rollout; the filter builds (and frees) its own graph internally for grad-h
            # via _cbf_terms, which re-establishes a leaf on a detached input
            # (`x_req = x.detach().clone(); x_req.requires_grad_(True)`), so grad-h is unaffected. NOT a blanket
            # torch.no_grad(): that would disable grad globally and silently break grad-h.
            u_nom = u_nom.detach()
            u_safe = u_safe.detach()
            _held = (u_nom, u_safe, infeasible, _eb_c, _sb_c)
        else:                                        # hold the last computed control (ZOH) across the substep
            u_nom, u_safe, infeasible, _eb_c, _sb_c = _held

        done_action = physical_done.unsqueeze(1)
        u_nom = torch.where(done_action, torch.zeros_like(u_nom), u_nom)
        u_safe = torch.where(done_action, torch.zeros_like(u_safe), u_safe)
        infeasible = torch.where(
            physical_done,
            torch.zeros_like(infeasible, dtype=torch.bool),
            infeasible.to(device=x.device, dtype=torch.bool),
        )

        if _lag_tau > 0.0:                           # plant applies the ZOH-lagged thrust; filter enforced on u_safe
            _u_applied = u_safe if _u_applied is None else motor_lag_step(_u_applied, u_safe, dt, _lag_tau)
            _u_applied = torch.where(done_action, torch.zeros_like(_u_applied), _u_applied)
            u_plant = _u_applied
        else:
            u_plant = u_safe
        x_next = rk4_step(system, x, u_plant, dt)
        _assert_finite("State after RK4 step", x_next)
        # v2.7.5 graph-retention fix (the DOMINANT leak): detach the state carried into the next step.
        # u_nom = policy(obs(x)) graphs back to x, so without this x_{t+1} = rk4(x_t, u_safe) accumulates a
        # graph that grows every step and is retained by `states` — host RSS then scales with step count and
        # OOMs at 1000 steps. Detaching here is numerically inert (forward values are unchanged).
        x = torch.where(done_action, x, x_next).detach()

        states.append(x)
        u_nom_steps.append(u_nom)
        u_safe_steps.append(u_safe)
        infeasible_steps.append(infeasible)
        # use the flags captured at control-compute time (held across substeps; at substeps=1 captured fresh
        # each step, so bit-identical to reading _fobj.last_* under the pre-v2.7.5 behaviour).
        _eb = _eb_c
        _sb = _sb_c
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
