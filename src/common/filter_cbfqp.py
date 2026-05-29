from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import proxsuite
import torch

from src.common.system import System


Tensor = torch.Tensor


@dataclass(frozen=True)
class QPSolveResult:
    u_safe: np.ndarray
    slack: float
    status: Any
    primal_residual: float
    dual_residual: float
    duality_gap: float
    row_lhs: float
    row_upper: float
    solution: np.ndarray


class CBFQPFilter:
    def __init__(
        self,
        system: System,
        h_fn: Callable[[Tensor, Any], Tensor],
        config: Mapping[str, Any],
    ) -> None:
        self.system = system
        self.h_fn = h_fn
        self.params = _qp_params(config)

    def __call__(
        self,
        x: Tensor,
        scene: Any,
        u_nom: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        if x.shape[0] != u_nom.shape[0]:
            raise ValueError("x and u_nom batch sizes must match.")

        h, lf_h, lg_h = _cbf_terms(self.system, self.h_fn, x, scene, u_nom)
        bounds = self.system.u_bounds.to(dtype=x.dtype)
        results = solve_cbf_qp_batch(u_nom, h, lf_h, lg_h, bounds, self.params)

        u_safe_np = np.stack([result.u_safe for result in results], axis=0)
        slack_np = np.asarray([result.slack for result in results], dtype=np.float64)
        u_safe = torch.as_tensor(u_safe_np, dtype=u_nom.dtype, device=u_nom.device)
        slack = torch.as_tensor(slack_np, dtype=u_nom.dtype, device=u_nom.device)
        infeasible = slack > 1.0e-4
        return u_safe, infeasible, slack


def solve_cbf_qp_batch(
    u_nom: Tensor,
    h: Tensor,
    lf_h: Tensor,
    lg_h: Tensor,
    bounds: Tensor,
    params: Mapping[str, float],
) -> list[QPSolveResult]:
    row_data = torch.cat(
        [u_nom, h.unsqueeze(1), lf_h.unsqueeze(1), lg_h],
        dim=1,
    )
    row_data_np = row_data.detach().cpu().double().numpy()
    action_dim = u_nom.shape[1]
    u_nom_np = row_data_np[:, :action_dim]
    h_np = row_data_np[:, action_dim]
    lf_h_np = row_data_np[:, action_dim + 1]
    lg_h_np = row_data_np[:, action_dim + 2 :]
    bounds_np = bounds.detach().cpu().double().numpy()

    cases = [
        (u_nom_np[idx], h_np[idx], lf_h_np[idx], lg_h_np[idx], bounds_np, params)
        for idx in range(u_nom_np.shape[0])
    ]
    if len(cases) == 1:
        return [_solve_cbf_qp_case(*cases[0])]

    # proxsuite dense QP has no vectorized API here, so rows are solved in CPU threads.
    max_workers = min(len(cases), int(params["max_workers"]))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda args: _solve_cbf_qp_case(*args), cases))


def solve_cbf_qp_case(
    u_nom: np.ndarray,
    h: float,
    lf_h: float,
    lg_h: np.ndarray,
    bounds: np.ndarray,
    params: Mapping[str, float],
) -> QPSolveResult:
    return _solve_cbf_qp_case(u_nom, h, lf_h, lg_h, bounds, params)


def _cbf_terms(
    system: System,
    h_fn: Callable[[Tensor, Any], Tensor],
    x: Tensor,
    scene: Any,
    u_nom: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    h = h_fn(x_req, scene).reshape(-1)
    grad_h = torch.autograd.grad(h.sum(), x_req, create_graph=False)[0]

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
    return h.detach(), lf_h.detach(), lg_h.detach()


def _solve_cbf_qp_case(
    u_nom: np.ndarray,
    h: float,
    lf_h: float,
    lg_h: np.ndarray,
    bounds: np.ndarray,
    params: Mapping[str, float],
) -> QPSolveResult:
    action_dim = int(u_nom.shape[0])
    alpha = params["alpha_safe"] if h <= 0.0 else params["alpha_unsafe"]
    row_upper = -float(lf_h) - alpha * (
        float(h) + params["v_shift"] + params["gamma_margin"]
    )

    hessian = np.diag([1.0] * action_dim + [params["penalty"]])
    gradient = np.concatenate([-np.asarray(u_nom, dtype=np.float64), [0.0]])
    constraint = np.zeros((action_dim + 2, action_dim + 1), dtype=np.float64)
    constraint[0, :action_dim] = np.asarray(lg_h, dtype=np.float64)
    constraint[0, action_dim] = -1.0
    for idx in range(action_dim):
        constraint[idx + 1, idx] = 1.0
    constraint[-1, action_dim] = 1.0

    lower = np.full(action_dim + 2, -np.inf, dtype=np.float64)
    upper = np.full(action_dim + 2, np.inf, dtype=np.float64)
    lower[1 : action_dim + 1] = bounds[:, 0]
    upper[1 : action_dim + 1] = bounds[:, 1]
    lower[-1] = -params["relax_eps1"]
    upper[0] = row_upper

    qp = proxsuite.proxqp.dense.QP(action_dim + 1, 0, action_dim + 2)
    qp.settings.eps_abs = float(params["eps_abs"])
    qp.settings.eps_rel = float(params["eps_abs"])
    qp.settings.max_iter = int(params["max_iter"])
    qp.settings.max_iter_in = int(params["max_iter_in"])
    qp.settings.verbose = False
    qp.init(
        H=hessian,
        g=gradient,
        C=constraint,
        l=lower,
        u=upper,
    )
    qp.solve()

    solution = np.asarray(qp.results.x, dtype=np.float64)
    u_safe = solution[:action_dim]
    slack = float(solution[action_dim])
    row_lhs = float(np.dot(lg_h, u_safe) - slack)
    info = qp.results.info
    if "SOLVED" not in str(info.status):
        raise RuntimeError(f"proxsuite CBF-QP solve failed with status {info.status}.")
    return QPSolveResult(
        u_safe=u_safe,
        slack=slack,
        status=info.status,
        primal_residual=float(info.pri_res),
        dual_residual=float(info.dua_res),
        duality_gap=float(info.duality_gap),
        row_lhs=row_lhs,
        row_upper=float(row_upper),
        solution=solution,
    )


def _qp_params(config: Mapping[str, Any]) -> dict[str, float]:
    cbf_cfg = config["filter"]["cbf_qp"]
    return {
        "penalty": float(cbf_cfg["penalty"]),
        "relax_eps1": float(cbf_cfg["relax_eps1"]),
        "v_shift": float(cbf_cfg["v_shift"]),
        "max_iter": float(cbf_cfg["max_iter"]),
        "max_iter_in": float(cbf_cfg["max_iter_in"]),
        "eps_abs": float(cbf_cfg["eps_abs"]),
        "max_workers": int(cbf_cfg["max_workers"]),
        "gamma_margin": float(config["filter"]["gamma_margin"]),
        "alpha_safe": float(config["filter"]["alpha_safe"]),
        "alpha_unsafe": float(config["filter"]["alpha_unsafe"]),
    }
