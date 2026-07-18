"""Planar quadrotor with DIRECT input (f_thr, tau) — v2.6.0 Stage 0.

Theory note Def 2.1 (planar, SO(2)) realized as a `System` (src/common/system.py). The control is the
scalar thrust f_thr along the body thrust axis Re=(-sin theta, cos theta) and the scalar torque tau,
entering the dynamics DIRECTLY (R1, note Sec 8.2.5): never (a_x,a_y)-then-PD. Underactuation (Def 2.2):
thrust enters dot v only through Re; torque enters dot omega only.

State  x = [px, py, theta, vx, vy, omega]      (6D; v is WORLD-frame linear velocity)
Input  u = [f_thr, tau]                         (2D; asymmetric box [f_min,f_max] x [-tau_max,tau_max])

Dynamics (Def 2.1, planar):
    dot p     = v
    dot theta = omega
    dot v     = (1/m) f_thr Re - g e_g,   Re=(-sin th, cos th),  e_g=(0,1) (gravity pulls -g in +y)
    dot omega = tau / J                    (planar: omega x J omega = 0)

Hover equilibrium: theta=0 => Re=(0,1); f_thr=m g => dot v = (0, g)-(0,g)=0, tau=0 => dot omega=0.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from src.common.observation import (
    scene_goal_tensor,
    scene_obstacle_tensors,
    top_k_obstacles,
)

Tensor = torch.Tensor


class QuadrotorPlanar:
    state_dim = 6
    action_dim = 2
    name = "quadrotor_planar"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.k_obs = int(config["env"]["k_obs"])
        # body v (2), omega (1), goal_body (2), K*(rel2+r1), body-frame gravity dir (sin θ, cos θ) (2)
        self.obs_dim = 2 + 1 + 2 + 3 * self.k_obs + 2  # v2.7.0: +2 attitude (01_env §3.3), dim 22
        phys = config["env"]["quadrotor_planar"]
        self.mass = float(phys["mass"])
        self.inertia = float(phys["inertia"])
        self.gravity = float(phys["gravity"])
        b = config["env"]["bounds"]["quadrotor_planar"]
        self.f_min = float(b["f_min"])
        self.f_max = float(b["f_max"])
        self.tau_max = float(b["tau_max"])
        self.v_max = float(b["v_max"])
        self.omega_max = float(b["omega_max"])
        self.u_bounds = torch.tensor(
            [[self.f_min, self.f_max], [-self.tau_max, self.tau_max]],
            dtype=torch.float64,
        )
        lqr = config["lqr"]["quadrotor_planar"]
        self.kp_pos = float(lqr["kp_pos"])
        self.kd_pos = float(lqr["kd_pos"])
        self.kp_att = float(lqr["kp_att"])
        self.kd_att = float(lqr["kd_att"])

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        theta = x[:, 2]
        vx = x[:, 3]
        vy = x[:, 4]
        omega = x[:, 5]
        f_thr = u[:, 0]
        tau = u[:, 1]
        inv_m = 1.0 / self.mass
        dvx = inv_m * f_thr * (-torch.sin(theta))
        dvy = inv_m * f_thr * torch.cos(theta) - self.gravity
        domega = tau / self.inertia
        return torch.stack([vx, vy, omega, dvx, dvy, domega], dim=1)

    def observation(self, x: Tensor, scene: Any) -> Tensor:
        positions = x[:, :2]
        theta = x[:, 2]
        vel = x[:, 3:5]
        omega = x[:, 5]
        goal = scene_goal_tensor(scene, x)
        centers, radii, active = scene_obstacle_tensors(scene, x.device, x.dtype)
        top_rel, top_radii = top_k_obstacles(positions, centers, radii, active, self.k_obs)

        vel_body = _rotate_world_to_body(vel, theta)
        goal_body = _rotate_world_to_body(goal - positions, theta)
        obstacle_body = _rotate_world_to_body(top_rel, theta)
        obstacle_block = torch.cat([obstacle_body, top_radii.unsqueeze(-1)], dim=-1)
        # v2.7.0 (01_env §3.3): append the body-frame gravity direction (sin θ, cos θ) AFTER the obstacle
        # block — the minimal restoration that de-aliases upright vs inverted (theory note sec:obs). The
        # first (2+1+2+3*k_obs) components are byte-identical to the pre-v2.7.0 layout.
        attitude = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        return torch.cat(
            [
                vel_body,
                omega.unsqueeze(-1),
                goal_body,
                obstacle_block.reshape(x.shape[0], -1),
                attitude,
            ],
            dim=1,
        )

    def position(self, x: Tensor) -> Tensor:
        return x[..., :2]

    def speed(self, x: Tensor) -> Tensor:
        return torch.linalg.norm(x[..., 3:5], dim=-1)

    def thrust_axis(self, x: Tensor) -> Tensor:
        """Re = (-sin theta, cos theta), the body thrust axis (note Def 2.1 / Thm 5.3)."""
        theta = x[..., 2]
        return torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)

    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        """Cascaded PD nominal (A1): outer position loop -> desired force -> desired attitude ->
        inner attitude PD -> (f_thr, tau). Output is clamped into the asymmetric box (admissible)."""
        goal = _batched_goal(goal, x)
        p = x[:, :2]
        theta = x[:, 2]
        vel = x[:, 3:5]
        omega = x[:, 5]

        a_des = -self.kp_pos * (p - goal) - self.kd_pos * vel  # desired world accel (2D)
        e_up = torch.zeros_like(a_des)
        e_up[:, 1] = self.gravity
        f_des_vec = self.mass * (a_des + e_up)  # F_des = m (a_des + g e_up); gravity feedforward
        f_thr = torch.linalg.norm(f_des_vec, dim=1)
        theta_des = torch.atan2(-f_des_vec[:, 0], f_des_vec[:, 1])  # Re aligned with F_des
        tau = self.inertia * (-self.kp_att * _wrap_angle(theta - theta_des) - self.kd_att * omega)
        return self._clamp_action(torch.stack([f_thr, tau], dim=1))

    def wrap_state(self, x: Tensor) -> Tensor:
        # Functional (no in-place index writes) so it is safe to differentiate under BPTT:
        # wrap theta to (-pi, pi]; clamp ||v|| <= v_max (scale, preserving direction); clamp |omega|.
        theta = _wrap_angle(x[:, 2])
        vel = x[:, 3:5]
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        scale = torch.clamp(self.v_max / torch.clamp(speed, min=1e-12), max=1.0)
        vel_clamped = vel * scale
        omega = torch.clamp(x[:, 5], min=-self.omega_max, max=self.omega_max)
        return torch.stack(
            [x[:, 0], x[:, 1], theta, vel_clamped[:, 0], vel_clamped[:, 1], omega], dim=1
        )

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
