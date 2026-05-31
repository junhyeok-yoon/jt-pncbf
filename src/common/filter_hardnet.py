from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping

import torch

from src.common.rk4 import rk4_step
from src.common.system import System


Tensor = torch.Tensor

_SINGULAR_LG_THRESHOLD = 5.0e-4
_FEASIBILITY_TOL = 1.0e-9
_DENOM_TOL = 1.0e-12


@dataclass(frozen=True)
class _HardNetParams:
    epsilon: float
    box_aware: bool
    alpha_safe: float
    alpha_unsafe: float
    lookahead_enabled: bool
    lookahead_n: int
    lookahead_beta: float
    lookahead_delta: float
    dt: float


class HardNetFilter:
    def __init__(
        self,
        system: System,
        h_fn: Callable[[Tensor, Any], Tensor],
        config: Mapping[str, Any],
        *,
        policy_fn: Callable[[Tensor, Any], Tensor] | None = None,
    ) -> None:
        self.system = system
        self.h_fn = h_fn
        self.policy_fn = policy_fn
        self.params = _hardnet_params(config)

    def __call__(
        self,
        x: Tensor,
        scene: Any,
        u_nom: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        if x.shape[0] != u_nom.shape[0]:
            raise ValueError("x and u_nom batch sizes must match.")

        h, lf_h, lg_h = _cbf_terms(
            self.system,
            self.h_fn,
            x,
            scene,
            u_nom,
            create_graph=True,
        )
        alpha = _base_alpha(h, self.params)
        if (
            self.params.lookahead_enabled
            and self.params.lookahead_n > 0
            and self.params.lookahead_beta != 0.0
        ):
            if self.policy_fn is None:
                raise ValueError("HardNet lookahead requires a policy_fn.")
            h_peak = _lookahead_peak_h(
                system=self.system,
                h_fn=self.h_fn,
                policy_fn=self.policy_fn,
                x=x,
                scene=scene,
                params=self.params,
            )
            gap = torch.relu((h_peak - h.detach()) / self.params.lookahead_delta)
            alpha = alpha * (1.0 + self.params.lookahead_beta * gap)
        row_upper = -lf_h - alpha * h
        bounds = self.system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)

        projected = _base_projection(u_nom, lg_h, row_upper, bounds, self.params)
        lg_norm = torch.linalg.norm(lg_h, dim=1)
        singular = lg_norm < _SINGULAR_LG_THRESHOLD

        if not self.params.box_aware:
            return projected, singular

        box_projected, empty_intersection = _box_aware_projection(
            u_nom,
            projected,
            lg_h,
            row_upper,
            bounds,
        )
        return box_projected, singular | empty_intersection


def _cbf_terms(
    system: System,
    h_fn: Callable[[Tensor, Any], Tensor],
    x: Tensor,
    scene: Any,
    u_nom: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    x_req = x if x.requires_grad and create_graph else x.detach().clone()
    x_req = x_req.requires_grad_(True)
    with torch.enable_grad():
        h = h_fn(x_req, scene).reshape(-1)
        grad_h = torch.autograd.grad(
            h.sum(),
            x_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        zero_u = torch.zeros_like(u_nom)
        f_x = system.dynamics(x_req, zero_u)
        g_columns = []
        for action_idx in range(u_nom.shape[1]):
            basis = torch.zeros_like(u_nom)
            basis[:, action_idx] = 1.0
            g_columns.append(system.dynamics(x_req, basis) - f_x)
        g_x = torch.stack(g_columns, dim=2)

        lf_h = torch.sum(grad_h * f_x, dim=1)
        lg_h = torch.einsum("bs,bsa->ba", grad_h, g_x)
    return h, lf_h, lg_h


def _base_projection(
    u_nom: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
    params: _HardNetParams,
) -> Tensor:
    lhs = torch.sum(halfspace_normal * u_nom, dim=1)
    violation = torch.relu(lhs - row_upper)
    denom = torch.sum(halfspace_normal * halfspace_normal, dim=1)
    denom = denom + params.epsilon**2
    correction = halfspace_normal * (violation / denom).unsqueeze(1)
    return _clamp_to_bounds(u_nom - correction, bounds)


def _box_aware_projection(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
) -> tuple[Tensor, Tensor]:
    candidates = _candidate_actions(
        u_nom,
        base_projected,
        halfspace_normal,
        row_upper,
        bounds,
    )
    lhs = torch.einsum("ba,bna->bn", halfspace_normal, candidates)
    feasible = lhs <= row_upper.unsqueeze(1) + _FEASIBILITY_TOL
    distance_sq = torch.sum((candidates - u_nom.unsqueeze(1)) ** 2, dim=2)

    inf = torch.full_like(distance_sq, torch.inf)
    feasible_scores = torch.where(feasible, distance_sq, inf)
    feasible_idx = torch.argmin(feasible_scores, dim=1)

    violation = torch.relu(lhs - row_upper.unsqueeze(1))
    least_bad_scores = violation + _FEASIBILITY_TOL * distance_sq
    least_bad_idx = torch.argmin(least_bad_scores, dim=1)

    has_feasible = torch.any(feasible, dim=1)
    selected_idx = torch.where(has_feasible, feasible_idx, least_bad_idx)
    batch_idx = torch.arange(u_nom.shape[0], device=u_nom.device)
    selected = candidates[batch_idx, selected_idx]

    empty_intersection = _empty_halfspace_box(
        halfspace_normal,
        row_upper,
        bounds,
    )
    return selected, empty_intersection


def _candidate_actions(
    u_nom: Tensor,
    base_projected: Tensor,
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
) -> Tensor:
    # The box-aware enumerator is specified for the current 2-D action systems.
    action_dim = u_nom.shape[1]
    low = bounds[:, 0]
    high = bounds[:, 1]
    clamped_nom = _clamp_to_bounds(u_nom, bounds)
    candidates = [clamped_nom, _clamp_to_bounds(base_projected, bounds)]

    for corner_bits in product((0, 1), repeat=action_dim):
        corner_values = [
            high[idx] if bit else low[idx] for idx, bit in enumerate(corner_bits)
        ]
        corner = torch.stack(corner_values).to(device=u_nom.device, dtype=u_nom.dtype)
        candidates.append(corner.unsqueeze(0).expand(u_nom.shape[0], -1))

    for fixed_idx in range(action_dim):
        for fixed_side in (0, 1):
            fixed_value = high[fixed_idx] if fixed_side else low[fixed_idx]
            for solve_idx in range(action_dim):
                if solve_idx == fixed_idx:
                    continue
                cand = _replace_column(
                    clamped_nom,
                    fixed_idx,
                    fixed_value.expand(u_nom.shape[0]),
                )
                current_lhs = torch.sum(halfspace_normal * cand, dim=1)
                residual = (
                    row_upper
                    - current_lhs
                    + halfspace_normal[:, solve_idx] * cand[:, solve_idx]
                )
                denom = halfspace_normal[:, solve_idx]
                solved = residual / torch.where(
                    torch.abs(denom) > _DENOM_TOL,
                    denom,
                    torch.ones_like(denom),
                )
                solved_column = torch.where(
                    torch.abs(denom) > _DENOM_TOL,
                    solved,
                    cand[:, solve_idx],
                )
                cand = _replace_column(cand, solve_idx, solved_column)
                candidates.append(_clamp_to_bounds(cand, bounds))

    return torch.stack(candidates, dim=1)


def _empty_halfspace_box(
    halfspace_normal: Tensor,
    row_upper: Tensor,
    bounds: Tensor,
) -> Tensor:
    low = bounds[:, 0]
    high = bounds[:, 1]
    minimizing_corner = torch.where(halfspace_normal >= 0.0, low, high)
    min_lhs = torch.sum(halfspace_normal * minimizing_corner, dim=1)
    return min_lhs > row_upper + _FEASIBILITY_TOL


def _clamp_to_bounds(u: Tensor, bounds: Tensor) -> Tensor:
    return torch.clamp(u, min=bounds[:, 0], max=bounds[:, 1])


def _replace_column(tensor: Tensor, column_idx: int, values: Tensor) -> Tensor:
    columns = [tensor[:, idx] for idx in range(tensor.shape[1])]
    columns[column_idx] = values
    return torch.stack(columns, dim=1)


def _hardnet_params(config: Mapping[str, Any]) -> _HardNetParams:
    hardnet_cfg = config["filter"]["hardnet"]
    lookahead_cfg = config["filter"].get("lookahead", {})
    return _HardNetParams(
        epsilon=float(hardnet_cfg["epsilon"]),
        box_aware=bool(hardnet_cfg["box_aware"]),
        alpha_safe=float(config["filter"]["alpha_safe"]),
        alpha_unsafe=float(config["filter"]["alpha_unsafe"]),
        lookahead_enabled=bool(lookahead_cfg.get("enabled", False)),
        lookahead_n=int(lookahead_cfg.get("N", 0)),
        lookahead_beta=float(lookahead_cfg.get("beta", 0.0)),
        lookahead_delta=max(float(lookahead_cfg.get("delta", 0.1)), 1.0e-8),
        dt=float(config.get("env", {}).get("dt", 0.05)),
    )


def _base_alpha(h: Tensor, params: _HardNetParams) -> Tensor:
    return torch.where(
        h <= 0.0,
        torch.full_like(h, params.alpha_safe),
        torch.full_like(h, params.alpha_unsafe),
    )


def _lookahead_peak_h(
    *,
    system: System,
    h_fn: Callable[[Tensor, Any], Tensor],
    policy_fn: Callable[[Tensor, Any], Tensor],
    x: Tensor,
    scene: Any,
    params: _HardNetParams,
) -> Tensor:
    horizon = max(0, int(params.lookahead_n))
    x_la = system.wrap_state(x.detach())
    with torch.no_grad():
        h_peak = h_fn(x_la, scene).reshape(-1).detach()
    bounds = system.u_bounds.to(device=x_la.device, dtype=x_la.dtype)

    for _ in range(horizon):
        with torch.no_grad():
            u_nom = policy_fn(x_la, scene).detach()
        h, lf_h, lg_h = _cbf_terms(
            system,
            h_fn,
            x_la,
            scene,
            u_nom,
            create_graph=False,
        )
        alpha = _base_alpha(h, params)
        row_upper = -lf_h - alpha * h
        projected = _base_projection(u_nom, lg_h, row_upper, bounds, params)
        if params.box_aware:
            projected, _ = _box_aware_projection(
                u_nom,
                projected,
                lg_h,
                row_upper,
                bounds,
            )
        with torch.no_grad():
            x_la = rk4_step(system, x_la, projected.detach(), params.dt)
            h_next = h_fn(x_la, scene).reshape(-1).detach()
            h_peak = torch.maximum(h_peak, h_next)
    return h_peak
