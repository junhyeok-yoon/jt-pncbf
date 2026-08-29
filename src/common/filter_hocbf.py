"""Analytic HIGH-ORDER CBF (HOCBF) filter — double integrator, unicycle, planar quadrotor.
v2.9.3, ADDITIVE.

STATUS LINE (must accompany every table built from this module):

    hand-built analytic high-order CBF (relative degree 2), single-row QP, no learning

This module contains NO learned object. The certificate is the raw geometric clearance to the
nearest ACTIVE obstacle, written in closed form; both Lie derivatives are written in closed form
below and are verified against autograd by `verify_affine_in_input` before anything is scored.

--------------------------------------------------------------------------------------------------
WHICH SYSTEMS, AND WHERE THE CONSTRUCTION STOPS  (v2.9.3 item 2 extension)
--------------------------------------------------------------------------------------------------
The hazard is position-only, so the relative degree is a property of each PLANT and is established
below per plant, not assumed. `L_g h == 0` on all three supported systems (the clearance depends on
the state only through p, and no input enters dot p on any of them), so the relative degree is at
least 2 everywhere and the plain first-order row of `filter_hardnet` is inapplicable to this hazard
on every one of them.

  double_integrator   u = (ax, ay)      relative degree 2, BOTH input channels. Cascade depth 2.
  unicycle            u = (a, omega)    relative degree 2, BOTH input channels: the acceleration
                                        enters through dot v and the turn rate enters through the
                                        heading, and BOTH appear at the second differentiation.
                                        Cascade depth 2.
  quadrotor_planar    u = (f_thr, tau)  relative degree 2 with respect to the THRUST channel only.
                                        L_g L_f h = [-(n.Re)/m, 0]: the torque column is
                                        IDENTICALLY ZERO. Cascade depth 2 is the correct depth for
                                        the scalar output, and it is what is built, but the row it
                                        produces constrains ONE of the two inputs.
  quadrotor_3d        NOT SUPPORTED — raises. See below.

WHERE IT STOPS, PART 1 — the torque channel of the planar quadrotor has NO finite relative degree.
Every drift Lie derivative L_f^k h of the planar quadrotor depends on the state only through (p, v):
L_f h = -n.v and L_f^2 h = -(||v||^2-(n.v)^2)/rho + g n_y contain neither theta nor omega, and the
drift field restricted to (p, v) is (v, -g e_y), which is theta-free, so by induction L_f^k h is
theta-free and omega-free for EVERY k. Since tau enters only dot omega, L_tau L_f^k h == 0 for every
k >= 0: the torque never appears in the drift Lie chain at any depth. Torque reaches the clearance
only through the BILINEAR product tau * f_thr along the trajectory (attitude matters only while the
vehicle is thrusting), which is not a relative degree at all. Under the standard dynamic extension
(promote f_thr to a state, take dot f_thr as the new input) the vector relative degree becomes
(3, 4) — 3 for the thrust rate, 4 for the torque — so a FOUR-LEVEL cascade is what it would take to
put torque in the row. That is not built here.

WHERE IT STOPS, PART 2 — the 3-D quadrotor. Same chain, and worse in three separate ways. Its hazard
is the clearance to a VERTICAL cylinder, so n is HORIZONTAL, and (i) the only channel that reaches
the hazard in two differentiations is the COLLECTIVE THRUST, with coefficient -(n . R e3)/m — the
horizontal component of the body-up axis, resolved along n. Since |n . R e3| <= sin(tilt) with
equality only when the lean is along n, that coefficient VANISHES IDENTICALLY on the hover manifold
{R e3 = e3}, i.e. at exactly the attitude the task's steady state requires: the depth-2 row's single
usable direction disappears where the vehicle must live. (ii) The input is four ROTOR FORCES, so the
depth-2 row constrains only their sum; three of the four input dimensions are not in it. (iii) The
three torque combinations have no finite relative degree at all, for the same reason as the planar
case, and under the dynamic extension the position hazard is FOUR levels from torque. A hand-built
four-level cascade is not a comparator this repository is prepared to defend; it is NOT attempted,
and this module raises on `quadrotor_3d`. That the analytic route reaches the double integrator, the
unicycle and (in one of its two inputs) the planar quadrotor, and does not reach the 3-D quadrotor
at all, is a MEASUREMENT OF THE CONSTRUCTION and is reported as one — the numbers behind (i) are in
`data/runs/v2.9.3/hocbf_extend/quadrotor_3d_obstruction.json`.

--------------------------------------------------------------------------------------------------
SIGN CONVENTION — POSITIVE-UNSAFE, matching the repository
--------------------------------------------------------------------------------------------------
`src/common/signed_h.py::signed_h` returns +1 inside a cylinder and -1 far away: the safe set is
{h <= 0}. The deployed row (`filter_hardnet._row_upper`) is `L_g h . u <= b`, `b = -L_f h - alpha*h`,
i.e. it enforces `hdot <= -alpha*h`. This module keeps that convention exactly, so its row composes
with the same projection and the same scoring path.

The hazard here is the RAW clearance, NOT `signed_h`. `signed_h` is the CLIPPED, h_scale-normalized
ramp `1 - 2*clamp(clearance/h_scale, 0, 1)`; it is flat (gradient identically zero) beyond
`clearance = h_scale` and again inside the cylinder, so its second derivative is zero almost
everywhere and a two-level cascade built on it would be vacuous exactly where it must act. The raw
clearance is derived here from `scene.obstacle_centers` / `obstacle_radii` / `obstacle_active`,
the same three arrays `signed_h` and `outcomes._collided_exact` read:

    rho_i(p) = ||p - c_i||,   d_i(p) = rho_i - r_i          (signed distance to cylinder i's surface)
    i*(x)    = argmax over ACTIVE i of (r_i - rho_i)         (the NEAREST active obstacle by clearance,
                                                              the same selection rule signed_h's
                                                              max-over-obstacles performs)
    h(x)     = r_{i*} - rho_{i*} = -d_{i*}(p)                (POSITIVE-UNSAFE: h > 0 <=> inside)

h > 0 is exactly the scored collision predicate (`_collided_exact` fires iff distance < radius on an
active obstacle), so the certificate and the scorer agree on the unsafe set by construction.

The max over obstacles is non-smooth at ties; the row is built from the argmax obstacle only, which
is the standard single-nearest-obstacle practice and is what the dispatch specifies ("clearance to
the nearest obstacle"). Ties are measure-zero in the pool and are NOT smoothed here.

--------------------------------------------------------------------------------------------------
THE TWO LIE DERIVATIVES, IN CLOSED FORM — (A) DOUBLE INTEGRATOR
--------------------------------------------------------------------------------------------------
State x = [p, v] in R^4, p = x[:, 0:2], v = x[:, 2:4]. The plant is control-affine with

    f(x) = [v, 0] in R^4,      g(x) = [[0, 0], [0, 0], [1, 0], [0, 1]] in R^{4x2}   (CONSTANT),

because `DoubleIntegrator.dynamics` returns `cat([x[:, 2:4], u])`: the input enters the velocity
block linearly with an identity gain and enters nowhere else.

Write n = (p - c_{i*}) / rho_{i*}, the OUTWARD unit normal of the nearest cylinder at p (||n|| = 1).

FIRST LEVEL.  h(x) = r_{i*} - ||p - c_{i*}||, so grad_p h = -n and grad_v h = 0. Hence

    L_f h   = grad h . f = -n . v            ("minus the closing rate": the rate at which the
                                              clearance is being consumed, sign-flipped),
    L_g h   = grad_v h  = 0                  (IDENTICALLY ZERO -> relative degree is 2, not 1).

`L_g h = 0` is what makes the plain first-order row of `filter_hardnet` inapplicable to this hazard:
its half-space {L_g h . u <= b} is either everything or nothing. That is the whole reason a
high-order cascade is needed, and it is verified numerically, not asserted.

SECOND LEVEL.  Let phi(x) = L_f h(x) = -n(p) . v. Then

    grad_p phi = -( I - n n^T ) v / rho_{i*} = -( v - n (n.v) ) / rho_{i*},
    grad_v phi = -n,

so, with f = [v, 0] and g constant,

    L_f^2 h     = grad phi . f = -( ||v||^2 - (n.v)^2 ) / rho_{i*}
                = -||v_perp||^2 / rho_{i*}      (v_perp = the component of v tangential to the
                                                 cylinder; the centripetal term, always <= 0, i.e.
                                                 tangential motion always OPENS clearance),
    L_g L_f h   = grad_v phi = -n^T             (a UNIT-NORM row, for every state with an active
                                                 obstacle -> the `||L_g h|| < 5e-4` singular clause
                                                 of the deployed filter is structurally vacuous here).

--------------------------------------------------------------------------------------------------
THE TWO LIE DERIVATIVES, IN CLOSED FORM — (B) UNICYCLE
--------------------------------------------------------------------------------------------------
State x = [px, py, theta, v] in R^4, input u = [a, omega]. `Unicycle.dynamics` returns
`[v cos theta, v sin theta, omega, a]`, so with the heading frame

    e(theta)      = ( cos theta,  sin theta)      (the heading direction; dot p = v e)
    e_perp(theta) = (-sin theta,  cos theta)      (dot e = omega e_perp)

the plant is control-affine with f(x) = [v e, 0, 0] and the CONSTANT
g = [[0,0],[0,0],[0,1],[1,0]] (column 1 = the acceleration channel, entering dot v; column 2 = the
turn-rate channel, entering dot theta). Write, with n the outward unit normal as above,

    s = n . e        (cosine of the angle between the normal and the heading; s in [-1, 1])
    t = n . e_perp   (the tangential cosine;  s^2 + t^2 = 1 exactly, since (e, e_perp) is an
                      orthonormal basis of R^2 and ||n|| = 1)

FIRST LEVEL.  grad_p h = -n, grad_theta h = 0, grad_v h = 0, so

    L_f h = -n . (v e) = -v s        (again minus the closing rate),
    L_g h = [0, 0]                   (IDENTICALLY ZERO -> relative degree is at least 2).

SECOND LEVEL.  With phi = L_f h = -v s and dot n = (I - n n^T) dot p / rho = v (e - n s)/rho,

    d/dt (n . e) = dot n . e + n . dot e = v (1 - s^2)/rho + omega t,

so, since dot v = a,

    phi_dot = -a s - v [ v (1 - s^2)/rho + omega t ]
            = ( -v^2 t^2 / rho )  +  [ -s, -v t ] . [a, omega]

    L_f^2 h   = -v^2 t^2 / rho = -(||dot p||^2 - (n . dot p)^2)/rho    (the SAME centripetal term as
                                  the double integrator, always <= 0),
    L_g L_f h = [ -s, -v t ]    (BOTH input channels appear -> relative degree is exactly 2 in BOTH
                                 inputs; the row is a genuine two-input row).

NOTE the row's norm: ||L_g L_f h||^2 = s^2 + v^2 t^2, which is NOT 1 as it is on the double
integrator. It degenerates to 0 exactly on {v = 0 and s = 0} — standing still, heading tangential to
the cylinder — the nonholonomic degeneracy of the unicycle. On that set no input moves the clearance
at second order, the `||L_g L_f h|| < 5e-4` singular clause is LIVE, and its measured mass is
reported rather than assumed vacuous.

--------------------------------------------------------------------------------------------------
THE TWO LIE DERIVATIVES, IN CLOSED FORM — (C) PLANAR QUADROTOR
--------------------------------------------------------------------------------------------------
State x = [px, py, theta, vx, vy, omega] in R^6, input u = [f_thr, tau].
`QuadrotorPlanar.dynamics` returns `[v, omega, (f/m) Re - g e_g, tau/J]` with

    Re(theta) = (-sin theta, cos theta)   (the body thrust axis),   e_g = (0, 1),

so f(x) = [v, omega, -g e_g, 0] and g(x) = [[0,0],[0,0],[0,0],[-sin theta/m, 0],[cos theta/m, 0],
[0, 1/J]] — control-affine, but NOT constant: the thrust column rotates with the attitude.

FIRST LEVEL.  grad_p h = -n, every other block of grad h is zero, so

    L_f h = -n . v          (minus the closing rate),
    L_g h = [0, 0]          (IDENTICALLY ZERO -> relative degree is at least 2).

SECOND LEVEL.  With phi = -n . v and dot n = (v - n (n.v))/rho,

    phi_dot = -(||v||^2 - (n.v)^2)/rho - n . [ (f/m) Re - g e_g ]

    L_f^2 h   = -(||v||^2 - (n.v)^2)/rho + g n_y     (centripetal term PLUS a gravity term: unlike
                                                      the double integrator this is NOT sign-definite
                                                      — free fall closes clearance on any obstacle
                                                      below the vehicle, n_y < 0),
    L_g L_f h = [ -(n . Re)/m , 0 ]                  (the TORQUE COLUMN IS EXACTLY ZERO).

Two consequences, both measured rather than asserted:
  (i)  the deployed row constrains the THRUST MAGNITUDE ONLY. The filter can push harder or ease off
       along the current thrust axis; it cannot re-point that axis, because re-pointing is a torque
       action and torque is not in the row at this depth (see WHERE IT STOPS, PART 1).
  (ii) ||L_g L_f h|| = |n . Re|/m vanishes whenever the thrust axis is TANGENTIAL to the cylinder.
       The `||L_g L_f h|| < 5e-4` singular clause is therefore LIVE on this system — structurally,
       not numerically — and on such rows feasibility reduces to the sign of b alone.

--------------------------------------------------------------------------------------------------
THE CASCADE AND THE ENFORCED CONSTRAINT
--------------------------------------------------------------------------------------------------
Standard two-level HOCBF (Xiao & Belta), written in the repository's POSITIVE-UNSAFE convention with
LINEAR class-K functions alpha_1(s) = a1*s, alpha_2(s) = a2*s (a1, a2 > 0):

    psi_0(x) = h(x)                                        <= 0 on the safe set
    psi_1(x) = psi_0dot + alpha_1(psi_0) = L_f h + a1 h     <= 0 on the level-1 set
    enforce:   psi_1dot + alpha_2(psi_1) <= 0

Because psi_1dot = L_f^2 h + (L_g L_f h) . u + a1 L_f h, the enforced constraint expands to

    (L_g L_f h) . u  <=  -L_f^2 h - (a1 + a2) L_f h - a1 a2 h        ... (ROW)

which is the SAME shape as the deployed row `A . u <= b` with

    A(x) = L_g L_f h  =  -n^T            (double integrator)
                      =  [-s, -v t]      (unicycle)
                      =  [-(n.Re)/m, 0]  (planar quadrotor)
    b(x) = -L_f^2 h - (a1 + a2) L_f h - a1 a2 h .

The cascade itself is IDENTICAL on all three systems; only the two closed forms above change. Depth
2 is the correct depth on each (see WHICH SYSTEMS above); nothing is forced onto a plant that does
not have it, and the one plant that does not have it — `quadrotor_3d` — is refused.

AFFINE IN THE INPUT. A and b are functions of the STATE ALONE on every supported system, and
`L_f^2 h`, `L_f h`, `h` contain no u. The constraint is therefore exactly affine in u, and the
program below is a QP (Euclidean projection of u_nom onto one half-space intersected with a box).
This is not left as an assertion: `verify_affine_in_input` differentiates psi_1 through the true
plant with autograd, forms psi_1dot(x, u) for arbitrary u, and checks (i) it equals A.u + c in closed
form and (ii) its second difference in u is zero, before any scoring runs.

In clearance terms (d = -h, so d > 0 is safe, d_dot = -hdot) the row reads
`d_ddot >= -(a1 + a2) d_dot - a1 a2 d`; on the boundary the closed loop is the linear second-order
system with roots -a1 and -a2, so d decays to zero without overshoot and never crosses it. a1 is the
gain that bounds the ADMISSIBLE APPROACH SPEED (level-1 set = {d_dot >= -a1 d}, i.e. closing speed at
most a1*d), so SMALLER a1 is the MORE CONSERVATIVE certificate; a2 sets how hard psi_1 is pushed
back to the level-1 set.

THE ONE-PARAMETER FAMILY (v2.9.3 item 3). With linear class-K functions the row depends on (a1, a2)
ONLY through their elementary symmetric functions a1+a2 and a1*a2, so the two gains are not two
degrees of freedom of the deployed filter's shape — they are the two roots of one characteristic
polynomial and any transposed pair is the SAME filter. Setting a1 = a2 = a places the closed loop's
double root at -a, i.e. CRITICALLY DAMPED, and reduces the row to

    b(x) = -L_f^2 h - 2 a L_f h - a^2 h ,

a single-parameter family. Small a is the CONSERVATIVE end (the level-1 set {d_dot >= -a d} admits
only a slow approach and SHRINKS as a -> 0, so more states start outside it); large a is the
permissive end (b -> +inf away from the surface and the row becomes vacuous).

--------------------------------------------------------------------------------------------------
THE PROGRAM
--------------------------------------------------------------------------------------------------
    u_safe(x) = argmin_u ||u - u_nom(x)||^2   s.t.   A(x) . u <= b(x),  u in U (the actuator box)

solved EXACTLY (not iteratively) by `filter_hardnet._dual_solve_projection`, the same closed-form
single-row box projection the deployed filter uses. NO SLACK VARIABLE is added. When the half-space
and the box do not intersect the program is INFEASIBLE; that row is COUNTED (the returned
`infeasible` flag) and the deployed command is the LEAST-VIOLATING admissible candidate selected by
`filter_hardnet._box_aware_projection`, i.e. the box point minimizing `relu(A.u - b)`. The registered
cell's `empty_fallback = {kstep, phases 1, k 3}` action substitution is deliberately NOT applied
here: it is a rollout-based search, and grafting it onto an analytic baseline would make the row a
hybrid. That divergence is named in the build-log.

`u_nom` is the caller's; nothing in this module touches the nominal policy or the actuator box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD,
    _base_projection,
    _dual_solve_projection,
    _hardnet_params,
)
from src.common.system import System


Tensor = torch.Tensor

_RHO_FLOOR = 1.0e-9      # guards n = (p - c)/rho at the cylinder axis; never binds on the pool
_INACTIVE = -1.0e6       # finite stand-in for "obstacle switched off" in the max over obstacles
_VACUOUS_B = 1.0e6       # RHS on a row with no active obstacle at all: constraint trivially holds


@dataclass(frozen=True)
class HOCBFParams:
    a1: float            # class-K gain of level 1: psi_1 = L_f h + a1 h
    a2: float            # class-K gain of level 2: psi_1dot + a2 psi_1 <= 0


SUPPORTED_SYSTEMS = ("double_integrator", "unicycle", "quadrotor_planar")

# Cascade depth that the position-only hazard actually requires on each plant, and the input
# channels the depth-2 row reaches. Reported, not assumed; every entry is verified numerically by
# `verify_affine_in_input` / `verify_relative_degree` before anything is scored.
RELATIVE_DEGREE = {
    "double_integrator": {"depth": 2, "channels_reached": ("ax", "ay"), "channels_missed": ()},
    "unicycle": {"depth": 2, "channels_reached": ("a", "omega"), "channels_missed": ()},
    "quadrotor_planar": {"depth": 2, "channels_reached": ("f_thr",), "channels_missed": ("tau",),
                         "depth_for_missed": 4, "note": "tau has NO finite drift relative degree; "
                                                        "4 is the depth under dynamic extension"},
    "quadrotor_3d": {"depth": None, "channels_reached": (), "channels_missed": ("f1..f4",),
                     "depth_for_missed": 4,
                     "note": "REFUSED: the depth-2 coefficient is (n . R e3)/m, identically zero on "
                             "the hover manifold; torque is 4 levels away under dynamic extension"},
}


class HOCBFFilter:
    """Two-level analytic HOCBF row + exact single-row box QP.

    Supported: double_integrator, unicycle, quadrotor_planar (see the module docstring for the
    closed form of each plant's two Lie derivatives and for why quadrotor_3d is refused)."""

    def __init__(self, system: System, config: Mapping[str, Any], a1: float, a2: float) -> None:
        name = getattr(system, "name", "")
        if name not in SUPPORTED_SYSTEMS:
            raise ValueError(
                "HOCBFFilter is defined for "
                + ", ".join(SUPPORTED_SYSTEMS)
                + " only (the position-only hazard has relative degree exactly 2 on each of them, "
                "and on quadrotor_3d the depth-2 coefficient (n . R e3)/m vanishes identically on "
                "the hover manifold while torque is four levels away); got system "
                f"{name or type(system).__name__!r}."
            )
        if not (a1 > 0.0 and a2 > 0.0):
            raise ValueError(f"class-K gains must be positive, got a1={a1}, a2={a2}.")
        self.system = system
        self.name = name
        self.params = HOCBFParams(a1=float(a1), a2=float(a2))
        # Only `epsilon` and `lg_reg_eps` are read off this block, and only to build the SAME
        # unconstrained half-space projection candidate the deployed filter feeds its box-aware
        # selection. No alpha, no lookahead, no empty_fallback of the deployed cell is used.
        self._proj_params = _hardnet_params(config)
        self.last_empty: Tensor | None = None
        self.last_singular: Tensor | None = None
        self.last_a: Tensor | None = None
        self.last_b: Tensor | None = None
        self.last_h: Tensor | None = None
        self.last_psi1: Tensor | None = None
        self.last_rho: Tensor | None = None
        self.last_idx: Tensor | None = None
        self.n_no_active_obstacle = 0

    # ---- the certificate and its two Lie derivatives (closed form; see the module docstring) ----
    def row(self, x: Tensor, scene: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return (h, L_f h, L_f^2 h, A = L_g L_f h, b) for the batch, all in closed form."""
        p = self.system.position(x)                       # [B, 2]
        centers = torch.as_tensor(scene.obstacle_centers, dtype=p.dtype, device=p.device)
        radii = torch.as_tensor(scene.obstacle_radii, dtype=p.dtype, device=p.device)
        active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=p.device)
        if centers.ndim == 2:                             # unbatched scene -> broadcast over the batch
            centers = centers.unsqueeze(0).expand(p.shape[0], -1, -1)
            radii = radii.unsqueeze(0).expand(p.shape[0], -1)
            active = active.unsqueeze(0).expand(p.shape[0], -1)
        centers = centers[..., : p.shape[-1]]

        rel = p.unsqueeze(-2) - centers                   # [B, M, 2]
        rho_all = torch.linalg.norm(rel, dim=-1)          # [B, M]
        h_all = radii - rho_all                           # POSITIVE-UNSAFE per obstacle
        h_all = torch.where(active, h_all, torch.full_like(h_all, _INACTIVE))
        idx = torch.argmax(h_all, dim=-1)                 # nearest ACTIVE obstacle, by clearance
        rows = torch.arange(p.shape[0], device=p.device)

        rho = rho_all[rows, idx].clamp_min(_RHO_FLOOR)    # [B]
        n = rel[rows, idx] / rho.unsqueeze(-1)            # [B, 2] outward unit normal, ||n|| = 1
        h = radii[rows, idx] - rho                        # [B]
        lf_h, lf2_h, a_row = self._lie(x, n, rho)
        a1, a2 = self.params.a1, self.params.a2
        b_row = -lf2_h - (a1 + a2) * lf_h - a1 * a2 * h

        # A scene with NO active obstacle carries no hazard: the row is switched off (A = 0, b large)
        # rather than left pointing at an inactive cylinder. The registered DI pool always has at
        # least one active obstacle, so this branch is expected to be dead; the counter proves it.
        any_active = active.any(dim=-1)
        if not bool(any_active.all()):
            self.n_no_active_obstacle += int((~any_active).sum().item())
            a_row = torch.where(any_active.unsqueeze(-1), a_row, torch.zeros_like(a_row))
            b_row = torch.where(any_active, b_row, torch.full_like(b_row, _VACUOUS_B))
            h = torch.where(any_active, h, torch.full_like(h, _INACTIVE))
            lf_h = torch.where(any_active, lf_h, torch.zeros_like(lf_h))
            lf2_h = torch.where(any_active, lf2_h, torch.zeros_like(lf2_h))
        self.last_rho, self.last_idx = rho.detach(), idx.detach()
        return h, lf_h, lf2_h, a_row, b_row

    def _lie(self, x: Tensor, n: Tensor, rho: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(L_f h, L_f^2 h, A = L_g L_f h) for the batch, in closed form, PER PLANT.

        This is the only place the three systems differ; the cascade, the row, the projection and
        the fallback above and below are shared verbatim. Every expression here is the one written
        out in the module docstring, and every one is checked against autograd through the TRUE
        plant by `verify_affine_in_input` before any scoring runs."""
        name = self.name
        if name == "double_integrator":
            # f = [v, 0], g = [[0],[I]] constant.  A = -n^T, ||A|| == 1 identically.
            v = x[:, 2:4]
            n_dot_v = torch.sum(n * v, dim=-1)
            lf_h = -n_dot_v
            lf2_h = -(torch.sum(v * v, dim=-1) - n_dot_v ** 2) / rho        # -|v_perp|^2 / rho
            a_row = -n
            return lf_h, lf2_h, a_row
        if name == "unicycle":
            # x = [px, py, theta, v];  u = [a, omega];  dot p = v e(theta).
            theta, speed = x[:, 2], x[:, 3]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            s = n[:, 0] * cos_t + n[:, 1] * sin_t                           # s = n . e
            t = -n[:, 0] * sin_t + n[:, 1] * cos_t                          # t = n . e_perp
            lf_h = -speed * s                                               # L_f h  = -v s
            lf2_h = -(speed ** 2) * (t ** 2) / rho                          # -v^2 t^2 / rho
            a_row = torch.stack([-s, -speed * t], dim=-1)                   # [-s, -v t]
            return lf_h, lf2_h, a_row
        if name == "quadrotor_planar":
            # x = [px, py, theta, vx, vy, omega];  u = [f_thr, tau];  dot v = (f/m) Re - g e_g.
            theta = x[:, 2]
            v = x[:, 3:5]
            n_dot_v = torch.sum(n * v, dim=-1)
            re = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)  # body thrust axis
            n_dot_re = torch.sum(n * re, dim=-1)
            g = float(self.system.gravity)
            m = float(self.system.mass)
            lf_h = -n_dot_v
            # centripetal term PLUS the gravity term: L_f^2 h is NOT sign-definite on this plant.
            lf2_h = -(torch.sum(v * v, dim=-1) - n_dot_v ** 2) / rho + g * n[:, 1]
            a_row = torch.stack([-n_dot_re / m, torch.zeros_like(n_dot_re)], dim=-1)
            return lf_h, lf2_h, a_row
        raise ValueError(f"unsupported system {name!r}")            # unreachable: gated in __init__

    def __call__(self, x: Tensor, scene: Any, u_nom: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        if x.shape[0] != u_nom.shape[0]:
            raise ValueError("x and u_nom batch sizes must match.")
        with torch.no_grad():
            h, lf_h, _lf2_h, a_row, b_row = self.row(x, scene)
            bounds = self.system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)
            base = _base_projection(u_nom, a_row, b_row, bounds, self._proj_params)
            u_safe, empty = _dual_solve_projection(u_nom, base, a_row, b_row, bounds)
            singular = torch.linalg.norm(a_row, dim=1) < _SINGULAR_LG_THRESHOLD
            self.last_empty = empty
            self.last_singular = singular
            self.last_a, self.last_b = a_row, b_row
            self.last_h = h
            self.last_psi1 = lf_h + self.params.a1 * h
        return u_safe, empty | singular


def verify_affine_in_input(
    system: System,
    filt: HOCBFFilter,
    x: Tensor,
    scene: Any,
    u_samples: Tensor,
) -> dict[str, float]:
    """Confirm, numerically, that the enforced constraint is AFFINE in u and that the two closed-form
    Lie derivatives are the true ones.

    (1) `L_g h == 0` and `L_f h` from autograd match the closed forms (relative degree really is 2).
    (2) psi_1dot(x, u), obtained by differentiating psi_1 = L_f h + a1 h through the TRUE plant with
        autograd and contracting with `system.dynamics(x, u)`, equals `A . u + (L_f^2 h + a1 L_f h)`.
    (3) the second difference of psi_1dot in u vanishes: for u_a, u_b and t in (0,1),
        psi_1dot(t u_a + (1-t) u_b) - t psi_1dot(u_a) - (1-t) psi_1dot(u_b) == 0.
    (4) the autograd L_g L_f h matches the closed-form A COLUMN BY COLUMN, so a structurally-zero
        column (the planar quadrotor's torque) is confirmed zero rather than assumed zero.
    Returns the max absolute residual of each check. All must be at machine precision.

    `L_g h` is formed the way `filter_hardnet._cbf_terms` forms it — g's j-th column is
    `dynamics(x, e_j) - dynamics(x, 0)`, exact for a control-affine plant — so this routine is
    plant-agnostic and reduces to the previous constant-[0; I] reading on the double integrator.
    """
    a1 = filt.params.a1
    x = x.detach().clone().double().requires_grad_(True)
    u_samples = u_samples.double()

    def h_of(state: Tensor) -> Tensor:
        _h, _, _, _, _ = filt.row(state, scene)
        return _h

    h = h_of(x)
    grad_h = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    zero_u = torch.zeros(x.shape[0], system.action_dim, dtype=x.dtype, device=x.device)
    f = system.dynamics(x, zero_u)
    g_cols = []
    for j in range(system.action_dim):
        basis = torch.zeros_like(zero_u)
        basis[:, j] = 1.0
        g_cols.append(system.dynamics(x, basis) - f)
    g_x = torch.stack(g_cols, dim=2)                     # [B, state_dim, action_dim]
    lg_h = torch.einsum("bs,bsa->ba", grad_h, g_x)       # plant-agnostic L_g h
    lf_h_auto = torch.sum(grad_h * f, dim=1)

    psi1 = lf_h_auto + a1 * h
    grad_psi1 = torch.autograd.grad(psi1.sum(), x, retain_graph=True)[0]
    lg_lf_h_auto = torch.einsum("bs,bsa->ba", grad_psi1, g_x)     # = L_g L_f h (a1 h adds nothing)

    with torch.no_grad():
        h_c, lf_h_c, lf2_h_c, a_row, _b = filt.row(x.detach(), scene)
        c_term = lf2_h_c + a1 * lf_h_c

    res = {
        "max_abs_Lg_h": float(lg_h.abs().max().item()),
        "max_abs_err_Lf_h": float((lf_h_auto.detach() - lf_h_c).abs().max().item()),
        "max_abs_err_h": float((h.detach() - h_c).abs().max().item()),
    }
    for j in range(system.action_dim):
        res[f"max_abs_err_LgLf_h_col{j}"] = float(
            (lg_lf_h_auto[:, j].detach() - a_row[:, j]).abs().max().item())
        res[f"max_abs_LgLf_h_col{j}_autograd"] = float(lg_lf_h_auto[:, j].detach().abs().max().item())
    err_affine, err_second = 0.0, 0.0
    for k in range(u_samples.shape[0]):
        u = u_samples[k].unsqueeze(0).expand(x.shape[0], -1)
        psi1dot = torch.sum(grad_psi1 * system.dynamics(x.detach(), u), dim=1)
        closed = torch.sum(a_row * u, dim=1) + c_term
        err_affine = max(err_affine, float((psi1dot - closed).abs().max().item()))
    for k in range(u_samples.shape[0] - 1):
        ua = u_samples[k].unsqueeze(0).expand(x.shape[0], -1)
        ub = u_samples[k + 1].unsqueeze(0).expand(x.shape[0], -1)
        for t in (0.25, 0.5, 0.75):
            um = t * ua + (1.0 - t) * ub
            pm = torch.sum(grad_psi1 * system.dynamics(x.detach(), um), dim=1)
            pa = torch.sum(grad_psi1 * system.dynamics(x.detach(), ua), dim=1)
            pb = torch.sum(grad_psi1 * system.dynamics(x.detach(), ub), dim=1)
            err_second = max(err_second, float((pm - t * pa - (1.0 - t) * pb).abs().max().item()))
    res["max_abs_err_psi1dot_vs_A_u_plus_c"] = err_affine
    res["max_abs_second_difference_in_u"] = err_second
    return res


def verify_relative_degree(
    system: System,
    filt: HOCBFFilter,
    x: Tensor,
    scene: Any,
    depth: int = 4,
) -> dict[str, Any]:
    """Establish the RELATIVE DEGREE of the position-only hazard on `system`, numerically.

    Builds the drift Lie chain y_k = L_f^k h by repeated autograd differentiation through the TRUE
    plant (y_0 = h, y_{k+1} = grad y_k . f with f = dynamics(x, 0)) and reports, for each k and each
    input channel j, max |L_{g_j} y_k| over the batch. The relative degree of channel j is the
    SMALLEST k+1 at which that quantity is nonzero; a channel whose whole column stays at machine
    zero through `depth` has no finite drift relative degree at that depth, and the depth-2 row does
    not reach it. This is what makes the statements in the module docstring measurements.
    """
    x = x.detach().clone().double().requires_grad_(True)
    zero_u = torch.zeros(x.shape[0], system.action_dim, dtype=x.dtype, device=x.device)
    f = system.dynamics(x, zero_u)
    g_cols = []
    for j in range(system.action_dim):
        basis = torch.zeros_like(zero_u)
        basis[:, j] = 1.0
        g_cols.append(system.dynamics(x, basis) - f)
    g_x = torch.stack(g_cols, dim=2)

    y, out = filt.row(x, scene)[0], {}
    for k in range(depth):
        grad_y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        lg = torch.einsum("bs,bsa->ba", grad_y, g_x)
        out[f"max_abs_Lg_Lf^{k}_h"] = [float(lg[:, j].abs().max().item())
                                       for j in range(system.action_dim)]
        y = torch.sum(grad_y * f, dim=1)
    out["system"] = getattr(system, "name", "")
    out["depth_probed"] = depth
    return out
