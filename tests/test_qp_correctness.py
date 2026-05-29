from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import torch
import yaml

from src.common.filter_cbfqp import solve_cbf_qp_case


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cbf_qp_matches_cvxpy_reference() -> None:
    config = _load_config()
    params = _qp_params(config)
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)

    cases = [
        {
            "name": "feasible_no_correction",
            "h": -0.5,
            "lf_h": 0.0,
            "lg_h": np.array([1.0, 0.0], dtype=np.float64),
            "u_nom": np.array([0.0, 0.0], dtype=np.float64),
        },
        {
            "name": "requires_slack",
            "h": 0.2,
            "lf_h": 0.0,
            "lg_h": np.array([-1.0, 0.0], dtype=np.float64),
            "u_nom": np.array([0.0, 0.0], dtype=np.float64),
        },
        {
            "name": "near_singular",
            "h": 0.05,
            "lf_h": 0.1,
            "lg_h": np.array([1.0e-8, -1.0e-8], dtype=np.float64),
            "u_nom": np.array([0.4, -0.3], dtype=np.float64),
        },
        {
            "name": "box_binding",
            "h": -0.8,
            "lf_h": 0.0,
            "lg_h": np.array([0.0, 1.0], dtype=np.float64),
            "u_nom": np.array([5.0, -5.0], dtype=np.float64),
        },
    ]

    for case in cases:
        result = solve_cbf_qp_case(
            u_nom=case["u_nom"],
            h=case["h"],
            lf_h=case["lf_h"],
            lg_h=case["lg_h"],
            bounds=bounds,
            params=params,
        )
        reference = _solve_reference(case, bounds, params)
        residual = max(abs(result.primal_residual), abs(result.dual_residual))
        assert residual < 1.0e-6, case["name"]
        assert result.row_lhs <= result.row_upper + 1.0e-6, case["name"]
        assert np.all(result.u_safe >= bounds[:, 0] - 1.0e-6), case["name"]
        assert np.all(result.u_safe <= bounds[:, 1] + 1.0e-6), case["name"]
        assert result.slack >= -params["relax_eps1"] - 1.0e-6, case["name"]
        assert np.allclose(result.solution, reference, atol=1.0e-5), case["name"]


def _solve_reference(
    case: Mapping[str, Any],
    bounds: np.ndarray,
    params: Mapping[str, float],
) -> np.ndarray:
    action_dim = case["u_nom"].shape[0]
    z = cp.Variable(action_dim + 1)
    u = z[:action_dim]
    slack = z[action_dim]
    alpha = params["alpha_safe"] if case["h"] <= 0.0 else params["alpha_unsafe"]
    row_upper = -case["lf_h"] - alpha * (
        case["h"] + params["v_shift"] + params["gamma_margin"]
    )
    objective = 0.5 * cp.sum_squares(u - case["u_nom"]) + 0.5 * params[
        "penalty"
    ] * cp.square(slack)
    constraints = [
        case["lg_h"] @ u - slack <= row_upper,
        u >= bounds[:, 0],
        u <= bounds[:, 1],
        slack >= -params["relax_eps1"],
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    value = problem.solve(solver=cp.OSQP, eps_abs=1.0e-10, eps_rel=1.0e-10)
    assert value is not None
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return np.asarray(z.value, dtype=np.float64)


def _load_config() -> Mapping[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
