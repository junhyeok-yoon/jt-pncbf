from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.linalg import solve_continuous_are
import torch

from src.common.observation import (
    scene_goal_tensor,
    scene_obstacle_tensors,
    top_k_obstacles,
)


Tensor = torch.Tensor


class DoubleIntegrator:
    state_dim = 4
    action_dim = 2
    obs_dim = 19
    name = "double_integrator"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.k_obs = int(config["env"]["k_obs"])
        u_max = float(config["env"]["bounds"]["double_integrator"]["u_max"])
        self.v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"])
        self.u_bounds = torch.tensor(
            [[-u_max, u_max], [-u_max, u_max]],
            dtype=torch.float64,
        )
        self._lqr_gain = _double_integrator_lqr_gain(
            config["lqr"]["double_integrator"]["Q"],
            config["lqr"]["double_integrator"]["R"],
        )

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        return torch.cat([x[:, 2:4], u], dim=1)

    def observation(self, x: Tensor, scene: Any) -> Tensor:
        positions = x[:, :2]
        goal = scene_goal_tensor(scene, x)
        centers, radii, active = scene_obstacle_tensors(scene, x.device, x.dtype)
        top_rel, top_radii = top_k_obstacles(
            positions,
            centers,
            radii,
            active,
            self.k_obs,
        )

        obstacle_block = torch.cat([top_rel, top_radii.unsqueeze(-1)], dim=-1)
        return torch.cat(
            [
                x[:, 2:4],
                goal - positions,
                obstacle_block.reshape(x.shape[0], -1),
            ],
            dim=1,
        )

    def position(self, x: Tensor) -> Tensor:
        return x[..., :2]

    def speed(self, x: Tensor) -> Tensor:
        return torch.linalg.norm(x[..., 2:4], dim=-1)

    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        goal = _batched_goal(goal, x)
        target = torch.cat([goal, torch.zeros_like(goal)], dim=1)
        gain = self._lqr_gain.to(device=x.device, dtype=x.dtype)
        u = -(x - target) @ gain.T
        return self._clamp_action(u)

    def wrap_state(self, x: Tensor) -> Tensor:
        speed = torch.linalg.norm(x[:, 2:4], dim=1).clamp_min(1.0e-6)
        scale = torch.clamp(self.v_max / speed, max=1.0)
        velocity = x[:, 2:4] * scale.unsqueeze(1)
        return torch.cat([x[:, :2], velocity], dim=1)

    def _clamp_action(self, u: Tensor) -> Tensor:
        bounds = self.u_bounds.to(device=u.device, dtype=u.dtype)
        return torch.clamp(u, min=bounds[:, 0], max=bounds[:, 1])


def _batched_goal(goal: Tensor, x: Tensor) -> Tensor:
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    return goal


def _double_integrator_lqr_gain(q_diag: list[float], r_diag: list[float]) -> Tensor:
    a_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    b_matrix = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q_matrix = np.diag(np.asarray(q_diag, dtype=np.float64))
    r_matrix = np.diag(np.asarray(r_diag, dtype=np.float64))
    p_matrix = solve_continuous_are(a_matrix, b_matrix, q_matrix, r_matrix)
    gain = np.linalg.solve(r_matrix, b_matrix.T @ p_matrix)
    return torch.as_tensor(gain, dtype=torch.float64)
