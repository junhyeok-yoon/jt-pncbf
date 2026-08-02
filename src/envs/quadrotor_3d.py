"""3D quadrotor with DIRECT input (f_thr, tau) — v2.7.2 bring-up (01_env §3.4).

State  x = [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]   (13D; q unit body->world, v WORLD linear vel)
Input  u = [f_thr, tau_x, tau_y, tau_z]                  (4D; box [0,f_max] x [-tau_max,tau_max]^3)

Dynamics (01_env §3.4):
    dot p     = v
    dot q     = 1/2 q (x) (0, omega)
    dot v     = (1/m) f_thr R(q) e3 - g e3      (e3 = world up = (0,0,1))
    dot omega = J^{-1} (tau - omega x J omega)
RK4 integration with per-step quaternion renormalization (double-cover safe).

Obstacles: infinite vertical cylinders (center c_xy, radius r). phi = ||p_xy - c_xy|| - r.
Observation (dim 32, full body frame R(q)^T): [v^b(3), omega^b(3), goal^b(3), g^b(3), Top-5 x (c_off^b(3), r)],
g^b = R(q)^T(-e3), c_off^b = R(q)^T(Delta c_xy, 0).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from src.common.observation import scene_goal_tensor, scene_obstacle_tensors, top_k_obstacles

Tensor = torch.Tensor


class QuadrotorQuad3D:
    state_dim = 13
    action_dim = 4
    name = "quadrotor_3d"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.k_obs = int(config["env"]["k_obs"])
        # v^b(3), omega^b(3), goal^b(3), g^b(3), K*(c_off^b 3 + r 1) = 12 + 4K ; K=5 -> 32
        # v2.7.6: obs_band_z appends absolute p_z, v_z so the band branch psi=|p_z|-4 is observable (32 -> 34).
        # Both are xy-translation- and yaw-invariant (the reduced symmetry group is preserved). Code default
        # False so a checkpoint whose config predates the flag (e.g. v2.7.4, dim-32) loads at its own dim;
        # obs_dim is thus determined by the checkpoint's own config (05_code, checkpoint compatibility).
        self.obs_band_z = bool(config.get("env", {}).get("obs_band_z", False))
        self.obs_dim = 3 + 3 + 3 + 3 + 4 * self.k_obs + (2 if self.obs_band_z else 0)
        phys = config["env"]["quadrotor_3d"]
        self.mass = float(phys["mass"])
        self.gravity = float(phys["gravity"])
        self.inertia = torch.tensor([float(phys["Jx"]), float(phys["Jy"]), float(phys["Jz"])], dtype=torch.float64)
        # v2.7.3 M0b: TRUE per-rotor actuator set. Control u = (f_1..f_4), each in [0, f_rotor_max] N.
        # X-config (rotors at 45/135/225/315 deg, alternating spin), fixed mixer wrench = M u:
        #   f_thr = sum f_i;  tau_x = l(f1+f2-f3-f4);  tau_y = l(-f1+f2+f3-f4);  tau_z = c(f1-f2+f3-f4)
        # with moment arm l = L/sqrt(2) and torque/force ratio c = k_M/k_F. Rows of M are orthogonal.
        self.arm_L = float(phys["arm_L"]); self.moment_arm = self.arm_L / math.sqrt(2.0)
        self.c_moment = float(phys["c_moment"])
        b = config["env"]["bounds"]["quadrotor_3d"]
        self.f_rotor_min = float(b["f_rotor_min"]); self.f_rotor_max = float(b["f_rotor_max"])
        self.v_max = float(b["v_max"]); self.omega_max = float(b["omega_max"])
        _l = self.moment_arm; _c = self.c_moment
        self.mixer = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [_l, _l, -_l, -_l], [-_l, _l, _l, -_l], [_c, -_c, _c, -_c]],
            dtype=torch.float64,
        )
        # orthogonal rows -> M^{-1} = M^T diag(1/||row_j||^2); norms^2 = (4, 4l^2, 4l^2, 4c^2)
        self.mixer_inv = self.mixer.t() @ torch.diag(
            1.0 / torch.tensor([4.0, 4.0 * _l * _l, 4.0 * _l * _l, 4.0 * _c * _c], dtype=torch.float64)
        )
        self.u_bounds = torch.tensor(
            [[self.f_rotor_min, self.f_rotor_max]] * 4, dtype=torch.float64
        )
        lqr = config["lqr"]["quadrotor_3d"]
        self.kp_pos = float(lqr["kp_pos"]); self.kd_pos = float(lqr["kd_pos"])
        self.kp_att = float(lqr["kp_att"]); self.kd_att = float(lqr["kd_att"])

    # ---- dynamics ----
    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        q = x[:, 3:7]; v = x[:, 7:10]; omega = x[:, 10:13]
        wrench = u @ self.mixer.to(x.device, x.dtype).t()     # per-rotor (f1..f4) -> (f_thr, tau) (constant linear)
        f_thr = wrench[:, 0]; tau = wrench[:, 1:4]
        R = _quat_to_R(q)                                     # [B,3,3] body->world
        e3 = R[:, :, 2]                                       # R e3 = 3rd column (body up in world)
        g_vec = torch.zeros_like(v); g_vec[:, 2] = self.gravity
        dv = (f_thr.unsqueeze(-1) / self.mass) * e3 - g_vec
        dq = 0.5 * _quat_mul(q, _pure_quat(omega))
        J = self.inertia.to(x.device, x.dtype)
        domega = (tau - torch.cross(omega, omega * J, dim=1)) / J
        return torch.cat([v, dq, dv, domega], dim=1)

    # ---- observation ----
    def observation(self, x: Tensor, scene: Any) -> Tensor:
        p = x[:, :3]; q = x[:, 3:7]; v = x[:, 7:10]; omega = x[:, 10:13]
        Rt = _quat_to_R(q).transpose(1, 2)                   # world->body = R(q)^T
        goal = scene_goal_tensor(scene, x)                   # [B,3]
        centers, radii, active = scene_obstacle_tensors(scene, x.device, x.dtype)  # centers [B,K,3] or [B,K,2]
        c_xy = centers[..., :2]                              # cylinder xy-centers
        p_xy = p[:, :2]
        top_rel_xy, top_radii = top_k_obstacles(p_xy, c_xy, radii, active, self.k_obs)   # xy surface-dist Top-K
        # cylinder offset feature: (Delta c_xy, 0) rotated into body
        c_off_world = torch.cat([top_rel_xy, torch.zeros_like(top_rel_xy[..., :1])], dim=-1)  # [B,K,3]
        v_b = _rot(Rt, v)                                    # v is WORLD linear velocity -> body
        omega_b = omega                                      # omega is ALREADY body-frame (q_dot=0.5 q(x)(0,w))
        goal_b = _rot(Rt, goal - p)
        g_world = torch.zeros_like(v); g_world[:, 2] = -1.0
        g_b = _rot(Rt, g_world)                              # R(q)^T(-e3)
        c_off_b = _rot_k(Rt, c_off_world)                    # [B,K,3]
        obstacle_block = torch.cat([c_off_b, top_radii.unsqueeze(-1)], dim=-1).reshape(x.shape[0], -1)
        base = torch.cat([v_b, omega_b, goal_b, g_b, obstacle_block], dim=1)
        if not self.obs_band_z:
            return base
        # v2.7.6: absolute altitude p_z and vertical velocity v_z (world; v is world linear velocity), appended
        # so the band branch psi=|p_z|-4 (+ c_z v_z) is an affine function of the input. Both invariant under
        # xy-translation and yaw. v_z equals -(v_b . g_b) but is added raw so the net need not synthesize it.
        return torch.cat([base, p[:, 2:3], v[:, 2:3]], dim=1)

    def position(self, x: Tensor) -> Tensor:
        return x[..., :3]

    def speed(self, x: Tensor) -> Tensor:
        return torch.linalg.norm(x[..., 7:10], dim=-1)

    def angular_rate(self, x: Tensor) -> Tensor:
        return torch.linalg.norm(x[..., 10:13], dim=-1)      # body angular rate ||omega||

    def thrust_axis(self, x: Tensor) -> Tensor:
        """Body up-axis in world = R(q) e3 (3rd column)."""
        return _quat_to_R(x[..., 3:7])[..., :, 2]

    def horizontal_velocity(self, x: Tensor) -> Tensor:
        """Obstacle-plane (horizontal) linear-velocity vector, for the situational obstacle-approach loss
        (03_train). quadrotor_3d: the xy components of the world velocity (vx, vy) — cylinders are infinite
        vertical, so only the horizontal closure matters."""
        return x[..., 7:9]

    def approach_barrier(self, x: Tensor, scene: Any, h_scale: float) -> Tensor:
        """h_star approach augmentation (system interface, v2.7.2): h_star = phi + c_gain * (v_xy . r_hat),
        r_hat = unit horizontal direction TOWARD the nearest active cylinder (c_xy - p_xy). A positive value
        is the CLOSING speed onto that cylinder, so the term inflates the avoid-cost when approaching fast
        (predictive lead, analogous to the planar thrust-axis term but obstacle-directed for cylinders).
        Selection matches phi's nearest-obstacle argmax; ties/near-center regularized by the 1e-9 floor."""
        p_xy = x[..., :2]                                    # [...,2]
        v_xy = x[..., 7:9]                                   # [...,2]
        centers, radii, active = scene_obstacle_tensors(scene, x.device, x.dtype)
        c_xy = centers[..., :2]                              # [...,K,2] or [K,2] (broadcasts over batch dims)
        toward = c_xy - p_xy.unsqueeze(-2)                   # [...,K,2] agent -> cylinder center
        surf = torch.linalg.norm(toward, dim=-1) - radii     # [...,K] surface distance (signed_h ordering)
        surf = surf.masked_fill(~active, torch.inf)
        idx = torch.argmin(surf, dim=-1, keepdim=True)       # [...,1] nearest active cylinder
        toward_sel = torch.gather(toward, -2, idx.unsqueeze(-1).expand(*idx.shape, 2)).squeeze(-2)  # [...,2]
        r_hat = toward_sel / torch.linalg.norm(toward_sel, dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.sum(v_xy * r_hat, dim=-1)               # [...]

    # ---- nominal: cascaded hover PD -> (f_thr, tau_des) -> mixer inverse -> per-rotor forces (clipped) ----
    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        goal = _batched_goal(goal, x)
        p = x[:, :3]; q = x[:, 3:7]; v = x[:, 7:10]; omega = x[:, 10:13]
        a_des = -self.kp_pos * (p - goal) - self.kd_pos * v
        e_up = torch.zeros_like(a_des); e_up[:, 2] = self.gravity
        f_des = self.mass * (a_des + e_up)                   # desired world force [B,3]
        R = _quat_to_R(q); b3 = R[:, :, 2]                   # current body-up in world
        f_thr = (f_des * b3).sum(dim=1).clamp(min=0.0)       # project onto current thrust axis
        # desired body-up = f_des direction; attitude error = b3 x b3_des (world), mapped to body
        b3_des = f_des / torch.clamp(torch.linalg.norm(f_des, dim=1, keepdim=True), min=1e-9)
        e_att_world = torch.cross(b3, b3_des, dim=1)         # small-angle attitude error axis (world)
        e_att_body = _rot(R.transpose(1, 2), e_att_world)
        J = self.inertia.to(x.device, x.dtype)
        tau = J * (self.kp_att * e_att_body - self.kd_att * omega)
        wrench = torch.cat([f_thr.unsqueeze(-1), tau], dim=1)          # desired (f_thr, tau)
        f_rotor = wrench @ self.mixer_inv.to(x.device, x.dtype).t()    # per-rotor forces (mixer inverse)
        return self._clamp_action(f_rotor)                            # per-rotor box clip [0, f_rotor_max]

    def wrap_state(self, x: Tensor) -> Tensor:
        p = x[:, :3]
        q = _quat_normalize(x[:, 3:7])
        v = x[:, 7:10]
        speed = torch.linalg.norm(v, dim=1, keepdim=True)
        v = v * torch.clamp(self.v_max / torch.clamp(speed, min=1e-12), max=1.0)
        omega = torch.clamp(x[:, 10:13], min=-self.omega_max, max=self.omega_max)
        return torch.cat([p, q, v, omega], dim=1)

    def _clamp_action(self, u: Tensor) -> Tensor:
        b = self.u_bounds.to(device=u.device, dtype=u.dtype)
        return torch.clamp(u, min=b[:, 0], max=b[:, 1])


# ---- quaternion helpers ([w,x,y,z]; R(q) body->world) ----
def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _pure_quat(omega: Tensor) -> Tensor:
    return torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)


def _quat_normalize(q: Tensor) -> Tensor:
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
    q = q / n
    return q * torch.sign(q[..., :1] + (q[..., :1] == 0).to(q.dtype))   # canonical w>=0 (double-cover)


def _quat_to_R(q: Tensor) -> Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def _rot(R: Tensor, v: Tensor) -> Tensor:          # [B,3,3] @ [B,3] -> [B,3]
    return torch.einsum("bij,bj->bi", R, v)


def _rot_k(R: Tensor, v: Tensor) -> Tensor:        # [B,3,3] applied to [B,K,3] -> [B,K,3]
    return torch.einsum("bij,bkj->bki", R, v)


def _batched_goal(goal: Tensor, x: Tensor) -> Tensor:
    return goal.unsqueeze(0).expand(x.shape[0], -1) if goal.ndim == 1 else goal
