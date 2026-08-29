"""Depth-FOUR analytic HOCBF for quadrotor_3d: the torque channel, with the thrust held.

The depth-two cascade (`filter_hocbf_quad3d.py`) is near-powerless at hover because its coefficient
-(n . R e3)/m vanishes there. Differentiating twice more reaches the TORQUES: the surviving term is
(grad_p h) . (T/m) R (Jinv tau x e3), whose row is

    A4 = (T/m) * Jinv (e3 x b),    b = R^T (grad_p h) = -R^T n

At hover T = mg, so A4 does NOT vanish -- re-pointing the thrust axis is the only physical route to
horizontal authority there, and this is the construction that commands it. Verified against nested
autograd through `system.dynamics`; Stage 1 measures |A4| at 975.6 median inside the 0-5 degree tilt
bin against the depth-two 0.0279.

THE HELD-THRUST ASSUMPTION ENTERS AT `_drift` BELOW (the `(T/m) R e3` term folded into the drift)
and at `collective_thrust`. At 20 Hz under a zero-order hold T is constant across the step, so
Tdot = Tddot = 0 within the interval and T is a measured parameter rather than a state. No dynamic
extension is performed and the rotor box stays a box.

THE SMOOTHED SIGNAL. Depth four needs h in C4; the deployed h is not C1 (a min over cylinders and a
clamp). The cascade's DERIVATIVE CHAIN ONLY uses
    d_soft = -(1/beta) logsumexp(-beta * d_i)          soft-min over active cylinders, C-infinity
    h_sm   = -L * tanh(d_soft / L)                     C-infinity saturating map, zero-preserving
Collisions are STILL SCORED by the deployed geometric predicate, untouched: this signal never
reaches `step_outcomes`.

THE INPUT SET. At the held thrust the rotor box admits a polytope of torques; the largest symmetric
axis-aligned box inside it is used so the shipped projection applies unchanged. Its conservatism is
reported as a volume ratio.

NOT ONE LINE OF ANY EXISTING FILE IS CHANGED, and nothing under src/frameworks/ppo/ is touched.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from src.common.filter_hardnet import (_SINGULAR_LG_THRESHOLD, _base_projection,
                                       _box_aware_projection, _empty_halfspace_box,
                                       _hardnet_params)

BETA_SOFTMIN = 40.0
L_SAT = 1.0


def soft_signal(x: Tensor, cen: Tensor, rad: Tensor, act: Tensor,
                beta: float = BETA_SOFTMIN, L: float = L_SAT) -> Tensor:
    """h_sm = -L tanh(softmin_i(d_i)/L); C-infinity, zero-preserving, positive-unsafe."""
    p = x[:, :2]
    rel = p[:, None, :] - cen
    d = torch.linalg.norm(rel, dim=-1) - rad
    d = torch.where(act, d, torch.full_like(d, 1.0e6))
    d_soft = -(-beta * d).logsumexp(dim=1) / beta
    return -L * torch.tanh(d_soft / L)


class HOCBFFilterD4:
    def __init__(self, system, config: Mapping[str, Any], a: float,
                 beta: float = BETA_SOFTMIN, L: float = L_SAT) -> None:
        if getattr(system, "name", "") != "quadrotor_3d":
            raise ValueError("HOCBFFilterD4 is for quadrotor_3d only.")
        if not a > 0.0:
            raise ValueError(f"gain must be positive, got {a}")
        self.system = system
        self.a = float(a)
        self.beta, self.L = float(beta), float(L)
        self._proj = _hardnet_params(config)
        self.M = system.mixer.clone()
        self.Minv = torch.linalg.inv(self.M)
        self.fmax = float(system.u_bounds[0, 1])
        self.last_empty = None
        self.last_singular = None
        self.last_a_norm = None
        self.last_tilt = None
        self.last_box = None

    # ---- the held thrust ---------------------------------------------------------------------
    def collective_thrust(self, u_nom: Tensor) -> Tensor:
        """THE HELD-THRUST PARAMETER: the collective the NOMINAL commands this step (ZOH)."""
        return (u_nom @ self.M.to(device=u_nom.device, dtype=u_nom.dtype).t())[:, 0]

    def _drift(self, x: Tensor, T: Tensor) -> Tensor:
        """f~(x) = f(x) + g_T(x) T with the thrust HELD -- the only remaining control is tau."""
        from src.envs.quadrotor_3d import _quat_to_R, _quat_mul, _pure_quat
        q, v, om = x[:, 3:7], x[:, 7:10], x[:, 10:13]
        R = _quat_to_R(q)
        gv = torch.zeros_like(v); gv[:, 2] = self.system.gravity
        dv = (T.unsqueeze(-1) / float(self.system.mass)) * R[:, :, 2] - gv
        dq = 0.5 * _quat_mul(q, _pure_quat(om))
        J = self.system.inertia.to(device=x.device, dtype=x.dtype)
        dom = (-torch.cross(om, om * J, dim=1)) / J
        return torch.cat([v, dq, dv, dom], dim=1)

    def torque_box(self, T: Tensor, n_bisect: int = 16) -> Tensor:
        """Largest SYMMETRIC axis-aligned torque box inside the polytope, by exact feasibility.

        u = M^{-1}[T; tau] is exact because M is invertible, so feasibility is 0 <= u <= fmax.
        """
        Minv = self.Minv.to(device=T.device, dtype=T.dtype)
        M = self.M.to(device=T.device, dtype=T.dtype)
        B = T.shape[0]
        half = []
        for k in (1, 2, 3):
            c = float(M[k].abs().max())
            up = c * (2.0 * torch.minimum(T, torch.full_like(T, 2 * self.fmax)) - T)
            dn = c * (2.0 * torch.clamp(T - 2 * self.fmax, min=0.0) - T)
            half.append(torch.minimum(up.abs(), dn.abs()))
        base = torch.stack(half, dim=1)
        signs = torch.tensor([[sx, sy, sz] for sx in (-1., 1.) for sy in (-1., 1.)
                              for sz in (-1., 1.)], dtype=T.dtype, device=T.device)
        lo = torch.zeros(B, dtype=T.dtype, device=T.device)
        hi = torch.ones(B, dtype=T.dtype, device=T.device)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            ok = torch.ones(B, dtype=torch.bool, device=T.device)
            for sg in signs:
                w = torch.cat([T.unsqueeze(-1), base * mid.unsqueeze(-1) * sg.unsqueeze(0)], dim=-1)
                u = torch.einsum('ij,bj->bi', Minv, w)
                ok &= ((u >= -1e-9) & (u <= self.fmax + 1e-9)).all(dim=1)
            lo = torch.where(ok, mid, lo); hi = torch.where(ok, hi, mid)
        return base * lo.unsqueeze(-1)

    # ---- the cascade -------------------------------------------------------------------------
    def row(self, x: Tensor, scene: Any, T: Tensor):
        """(h, A4, b4) with the four drift Lie derivatives taken by autograd along f~."""
        from src.common.observation import scene_obstacle_tensors
        cen, rad, actm = scene_obstacle_tensors(scene, x.device, x.dtype)
        cen = cen[..., :2] if cen.ndim == 3 else cen[None, ..., :2]
        rad = rad if rad.ndim == 2 else rad[None, :]
        actm = actm.to(torch.bool)
        actm = actm if actm.ndim == 2 else actm[None, :]
        xr = x.detach().clone().requires_grad_(True)
        h = soft_signal(xr, cen, rad, actm, self.beta, self.L)

        def L_f(field):
            g, = torch.autograd.grad(field.sum(), xr, create_graph=True)
            return (g * self._drift(xr, T)).sum(dim=1)

        l1 = L_f(h); l2 = L_f(l1); l3 = L_f(l2)
        g3, = torch.autograd.grad(l3.sum(), xr, create_graph=False)
        # tau enters only the omega block
        J = self.system.inertia.to(device=x.device, dtype=x.dtype)
        A4 = g3[:, 10:13] / J
        # drift 4th derivative
        l4 = (g3 * self._drift(xr, T).detach()).sum(dim=1)
        a = self.a
        # elementary symmetric coefficients of (a,a,a,a)
        b4 = -(l4.detach() + 4 * a * l3.detach() + 6 * a * a * l2.detach()
               + 4 * a ** 3 * l1.detach() + a ** 4 * h.detach())
        return h.detach(), A4.detach(), b4

    def __call__(self, x: Tensor, scene: Any, u_nom: Tensor):
        T = self.collective_thrust(u_nom)
        h, A4, b4 = self.row(x, scene, T)
        half = self.torque_box(T)
        bounds3 = torch.stack([-half, half], dim=-1)            # [B,3,2] per-row box
        tau_nom = (u_nom @ self.M.to(device=u_nom.device, dtype=u_nom.dtype).t())[:, 1:4]
        # the shipped projection expects a shared [dim,2] box; solve per-row by scaling to a unit box
        s = half.clamp_min(1e-9)
        A_s = A4 * s                                            # tau = s * z, z in [-1,1]^3
        z_nom = tau_nom / s
        unit = torch.tensor([[-1.0, 1.0]] * 3, dtype=x.dtype, device=x.device)
        base = _base_projection(z_nom, A_s, b4, unit, self._proj)
        z_safe, empty = _box_aware_projection(z_nom, base, A_s, b4, unit,
                                              self._proj.empty_mode, self._proj.empty_prox_temp)
        tau_safe = z_safe * s
        w = torch.cat([T.unsqueeze(-1), tau_safe], dim=-1)
        u = torch.einsum('ij,bj->bi', self.Minv.to(device=x.device, dtype=x.dtype), w)
        u = torch.clamp(u, 0.0, self.fmax)
        singular = torch.linalg.norm(A_s, dim=1) < _SINGULAR_LG_THRESHOLD
        from src.envs.quadrotor_3d import _quat_to_R
        self.last_empty = empty.detach()
        self.last_singular = singular.detach()
        self.last_a_norm = torch.linalg.norm(A4, dim=1).detach()
        self.last_tilt = torch.arccos(torch.clamp(_quat_to_R(x[:, 3:7])[:, 2, 2], -1, 1)).detach()
        self.last_box = half.detach()
        return u, (singular | empty)
