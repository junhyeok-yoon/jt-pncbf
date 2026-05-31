from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import torch

from src.common.filter_hardnet import HardNetFilter


DTYPE = torch.float64


@dataclass(frozen=True)
class LinearScene:
    h_grad: torch.Tensor


class LinearDynamicsSystem:
    state_dim = 2
    action_dim = 2
    obs_dim = 2
    name = "linear_test"

    def __init__(self, f_x: torch.Tensor, bounds: torch.Tensor) -> None:
        self.f_x = f_x
        self.u_bounds = bounds

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.f_x.to(device=x.device, dtype=x.dtype).unsqueeze(0) + u

    def observation(self, x: torch.Tensor, scene: Any) -> torch.Tensor:
        return x

    def position(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2]

    def lqr_action(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.action_dim), dtype=x.dtype, device=x.device)

    def wrap_state(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_hardnet_matches_l2_qp_with_inactive_box() -> None:
    bounds = torch.tensor([[-100.0, 100.0], [-100.0, 100.0]], dtype=DTYPE)
    config = {
        "filter": {
            "hardnet": {"epsilon": 0.0, "box_aware": False},
            "alpha_safe": 2.0,
            "alpha_unsafe": 100.0,
        }
    }
    cases = [
        {
            "normal": np.array([1.0, 0.0], dtype=np.float64),
            "upper": 0.2,
            "u_nom": np.array([1.0, 0.1], dtype=np.float64),
        },
        {
            "normal": np.array([1.0, 1.0], dtype=np.float64),
            "upper": 0.0,
            "u_nom": np.array([2.0, 0.0], dtype=np.float64),
        },
        {
            "normal": np.array([-0.5, 1.5], dtype=np.float64),
            "upper": -0.1,
            "u_nom": np.array([0.6, 0.8], dtype=np.float64),
        },
    ]

    for case in cases:
        normal = torch.as_tensor(case["normal"], dtype=DTYPE)
        f_x = -case["upper"] * normal / torch.sum(normal * normal)
        system = LinearDynamicsSystem(f_x, bounds)
        scene = LinearScene(h_grad=normal)
        h_fn = lambda x, linear_scene: x @ linear_scene.h_grad
        hardnet = HardNetFilter(system, h_fn, config)

        x = torch.zeros((1, 2), dtype=DTYPE)
        u_nom = torch.as_tensor(case["u_nom"], dtype=DTYPE).unsqueeze(0)
        u_safe, infeasible = hardnet(x, scene, u_nom)
        reference = _l2_qp_reference(
            case["normal"],
            case["upper"],
            case["u_nom"],
            bounds.numpy(),
        )

        assert not bool(infeasible.item()), case
        assert np.allclose(u_safe.detach().numpy()[0], reference, atol=1.0e-6), case


def test_hardnet_lookahead_disabled_and_zero_beta_are_identity() -> None:
    bounds = torch.tensor([[-100.0, 100.0], [-100.0, 100.0]], dtype=DTYPE)
    system = LinearDynamicsSystem(torch.zeros(2, dtype=DTYPE), bounds)
    scene = LinearScene(h_grad=torch.tensor([1.0, 0.0], dtype=DTYPE))
    h_fn = lambda x, linear_scene: x @ linear_scene.h_grad
    base_config = {
        "env": {"dt": 0.05},
        "filter": {
            "hardnet": {"epsilon": 0.0, "box_aware": False},
            "alpha_safe": 2.0,
            "alpha_unsafe": 100.0,
        },
    }
    lookahead_config = {
        "env": {"dt": 0.05},
        "filter": {
            "hardnet": {"epsilon": 0.0, "box_aware": False},
            "alpha_safe": 2.0,
            "alpha_unsafe": 100.0,
            "lookahead": {"enabled": True, "N": 5, "beta": 0.0, "delta": 0.1},
        },
    }
    x = torch.tensor([[0.2, 0.0]], dtype=DTYPE)
    u_nom = torch.tensor([[1.0, 0.25]], dtype=DTYPE)
    policy_fn = lambda policy_x, policy_scene: u_nom.expand(policy_x.shape[0], -1)

    base_u, base_infeasible = HardNetFilter(system, h_fn, base_config)(x, scene, u_nom)
    lookahead_u, lookahead_infeasible = HardNetFilter(
        system,
        h_fn,
        lookahead_config,
        policy_fn=policy_fn,
    )(x, scene, u_nom)

    assert torch.allclose(lookahead_u, base_u)
    assert torch.equal(lookahead_infeasible, base_infeasible)


def _l2_qp_reference(
    normal: np.ndarray,
    upper: float,
    u_nom: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    u = cp.Variable(2)
    objective = cp.Minimize(0.5 * cp.sum_squares(u - u_nom))
    constraints = [
        normal @ u <= upper,
        u >= bounds[:, 0],
        u <= bounds[:, 1],
    ]
    problem = cp.Problem(objective, constraints)
    value = problem.solve(solver=cp.OSQP, eps_abs=1.0e-10, eps_rel=1.0e-10)
    assert value is not None
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return np.asarray(u.value, dtype=np.float64)
