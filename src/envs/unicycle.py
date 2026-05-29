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


class Unicycle:
    state_dim = 4
    action_dim = 2
    obs_dim = 18
    name = "unicycle"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.k_obs = int(config["env"]["k_obs"])
        bounds_cfg = config["env"]["bounds"]["unicycle"]
        a_max = float(bounds_cfg["a_max"])
        omega_max = float(bounds_cfg["omega_max"])
        self.u_bounds = torch.tensor(
            [[-a_max, a_max], [-omega_max, omega_max]],
            dtype=torch.float64,
        )
        self.v_min = float(config["lqr"]["unicycle"]["v_min"])
        self._lqr_gain = _double_integrator_lqr_gain(
            config["lqr"]["unicycle"]["Q"],
            config["lqr"]["unicycle"]["R"],
        )

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        theta = x[:, 2]
        speed = x[:, 3]
        acceleration = u[:, 0]
        omega = u[:, 1]
        return torch.stack(
            [
                speed * torch.cos(theta),
                speed * torch.sin(theta),
                omega,
                acceleration,
            ],
            dim=1,
        )

    def observation(self, x: Tensor, scene: Any) -> Tensor:
        positions = x[:, :2]
        theta = x[:, 2]
        goal = scene_goal_tensor(scene, x)
        centers, radii, active = scene_obstacle_tensors(scene, x.device, x.dtype)
        top_rel, top_radii = top_k_obstacles(
            positions,
            centers,
            radii,
            active,
            self.k_obs,
        )

        goal_body = _rotate_world_to_body(goal - positions, theta)
        obstacle_body = _rotate_world_to_body(top_rel, theta)
        obstacle_block = torch.cat([obstacle_body, top_radii.unsqueeze(-1)], dim=-1)
        return torch.cat(
            [
                x[:, 3:4],
                goal_body,
                obstacle_block.reshape(x.shape[0], -1),
            ],
            dim=1,
        )

    def position(self, x: Tensor) -> Tensor:
        return x[..., :2]

    def speed(self, x: Tensor) -> Tensor:
        return torch.abs(x[..., 3])

    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        goal = _batched_goal(goal, x)
        theta = x[:, 2]
        speed = x[:, 3]
        virtual_state = torch.stack(
            [
                x[:, 0],
                x[:, 1],
                speed * torch.cos(theta),
                speed * torch.sin(theta),
            ],
            dim=1,
        )
        target = torch.cat([goal, torch.zeros_like(goal)], dim=1)
        gain = self._lqr_gain.to(device=x.device, dtype=x.dtype)
        virtual_acceleration = -(virtual_state - target) @ gain.T

        heading = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        lateral = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=1)
        acceleration = torch.sum(virtual_acceleration * heading, dim=1)

        speed_sign = torch.where(
            speed >= 0.0,
            torch.ones_like(speed),
            -torch.ones_like(speed),
        )
        speed_safe = speed_sign * torch.clamp(torch.abs(speed), min=self.v_min)
        omega = torch.sum(virtual_acceleration * lateral, dim=1) / speed_safe
        return self._clamp_action(torch.stack([acceleration, omega], dim=1))

    def wrap_state(self, x: Tensor) -> Tensor:
        wrapped = x.clone()
        wrapped[:, 2] = _wrap_angle(wrapped[:, 2])
        return wrapped

    def _clamp_action(self, u: Tensor) -> Tensor:
        bounds = self.u_bounds.to(device=u.device, dtype=u.dtype)
        return torch.clamp(u, min=bounds[:, 0], max=bounds[:, 1])


def _rotate_world_to_body(rel: Tensor, theta: Tensor) -> Tensor:
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    while cos_theta.ndim < rel[..., 0].ndim:
        cos_theta = cos_theta.unsqueeze(-1)
        sin_theta = sin_theta.unsqueeze(-1)

    x_body = cos_theta * rel[..., 0] + sin_theta * rel[..., 1]
    y_body = -sin_theta * rel[..., 0] + cos_theta * rel[..., 1]
    return torch.stack([x_body, y_body], dim=-1)


def _batched_goal(goal: Tensor, x: Tensor) -> Tensor:
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    return goal


def _wrap_angle(theta: Tensor) -> Tensor:
    return torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi


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
