"""The two-level analytic HOCBF cascade EXTENDED to quadrotor_3d, collective thrust only.

`src/common/filter_hocbf.py` refuses quadrotor_3d (`SUPPORTED_SYSTEMS`, :298; the raise at :325-333)
and records why in `data/runs/v2.9.3/hocbf_extend/quadrotor_3d_obstruction.json`. That refusal was a
judgement about defensibility, not an impossibility: the obstruction record itself states that the
depth-2 row reaches the COLLECTIVE THRUST with coefficient -(n . R e3)/m and constrains the rotor
SUM. That is exactly the structure `quadrotor_planar` already ships with -- thrust reached, torque
column zero -- so the same cascade extends, and this module builds it.

NOT ONE LINE OF ANY EXISTING FILE IS CHANGED. `filter_hocbf.py` is imported, never edited; nothing in
jt_pncbf, oc_pncbf, envs, eval or common imports it, so no framework path moves.

WHAT IS REUSED, VERBATIM, FROM `filter_hocbf.py`
  the cascade itself           psi_0 = h, psi_1 = L_f h + a1 psi_0, row (L_g L_f h).u <= b with
                               b = -L_f^2 h - (a1+a2) L_f h - a1 a2 h          (:206-221)
  the nearest-obstacle choice  argmax over active of (r_i - rho_i), i.e. argmin surface distance
                               (:366-369)
  the row solve and the least-violating fallback, the projection, the SINGULAR test and the EMPTY
                               test -- inherited by subclassing `HOCBFFilter`, so `__call__` is the
                               shipped one and only `_lie` is overridden.

WHAT THE 3-D PLANT REQUIRED, AND WHY (four differences, and only four)
  (1) THE NORMAL IS HORIZONTAL. The cylinders are vertical, so h depends on p_xy only and
      n = (p_xy - c_xy)/rho is a 2-vector embedded in 3-D as (n_xy, 0).
  (2) THE GRAVITY TERM DISAPPEARS FROM L_f^2 h. On quadrotor_planar `_lie` carries `+ g*n[:,1]`
      (:430) because there the normal has a component along gravity. Here gravity is -g e3 and n is
      horizontal, so n . g_vec = 0 identically and the drift second derivative is the centripetal
      term alone:
            L_f^2 h = -(|v_xy|^2 - (n.v_xy)^2) / rho
      This is a genuine simplification, not an omission, and it makes L_f^2 h SIGN-DEFINITE here
      (<= 0) where on the planar plant it is not.
  (3) THE ROW IS THE SAME FOR EVERY ROTOR. The mixer's collective row is exactly ones(4)
      (verified: f_thr = sum(u)), so
            L_g L_f h = -(n . R e3)/m * ones(4)
      -- rank one in a 4-dimensional input, the same coefficient on each rotor. The three torque
      combinations never appear, exactly as tau does not on the planar plant, and their column is
      zero there too.
  (4) THE POSITION IS PROJECTED TO ITS XY FOOTPRINT before the parent's `row` sees it -- see
      `_XYPositionProxy`. The cylinders are infinite vertical, so the clearance is horizontal.

THE DEGENERACY IS THE POINT AND IS MEASURED, NOT ASSUMED. |n . R e3| <= sin(tilt), with equality iff
the lean is along n, so the row's single usable direction vanishes identically at hover -- the
attitude the task's steady state requires. This module records the coefficient norm at every live
step so the share below the SINGULAR threshold and its distribution against tilt are measured.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from src.common.filter_hocbf import HOCBFFilter, HOCBFParams
from src.common.system import System

SUPPORTED = ("quadrotor_3d",)


class _XYPositionProxy:
    """`system` with `position()` returning the XY FOOTPRINT, everything else forwarded.

    The 4th and last 3-D difference. `HOCBFFilter.row` (filter_hocbf.py:353-364) takes
    `self.system.position(x)` and slices the cylinder centres to ITS dimension; on a 3-D position
    against 2-D centres that mismatches, and slicing the other way would put p_z into a distance the
    cylinders are infinite along. Projecting the position instead is the correct reading -- the
    hazard is the horizontal clearance to a vertical cylinder -- and it lets the parent's `row`,
    projection, SINGULAR test, EMPTY test and fallback be reused WHOLE rather than copied.
    """

    def __init__(self, system):
        self._s = system

    def position(self, x):
        return self._s.position(x)[..., :2]

    def __getattr__(self, k):
        return getattr(self._s, k)


class HOCBFFilterQuad3D(HOCBFFilter):
    """`HOCBFFilter` with `_lie` extended to quadrotor_3d; everything else is inherited."""

    def __init__(self, system: System, config: Mapping[str, Any], a1: float, a2: float) -> None:
        name = getattr(system, "name", "")
        if name not in SUPPORTED:
            raise ValueError(f"HOCBFFilterQuad3D is for {SUPPORTED}, got {name!r}.")
        if not (a1 > 0.0 and a2 > 0.0):
            raise ValueError(f"class-K gains must be positive, got a1={a1}, a2={a2}.")
        # The parent's SUPPORTED_SYSTEMS gate is bypassed; EVERY other field it sets is set here,
        # with the same expressions (filter_hocbf.py:335-350), so `row`, `__call__`, the projection
        # and the fallback behave identically.
        from src.common.filter_hocbf import _hardnet_params
        self.full_system = system
        self.system = _XYPositionProxy(system)      # see the proxy's docstring: difference (4)
        self.name = name
        self.params = HOCBFParams(a1=float(a1), a2=float(a2))
        self._proj_params = _hardnet_params(config)
        self.last_empty = None
        self.last_singular = None
        self.last_a = None
        self.last_b = None
        self.last_h = None
        self.last_psi1 = None
        self.last_rho = None
        self.last_idx = None
        self.n_no_active_obstacle = 0
        # instrumentation added here only (does not enter the row)
        self.last_a_norm = None
        self.last_tilt = None

    def _lie(self, x: Tensor, n: Tensor, rho: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(L_f h, L_f^2 h, A = L_g L_f h) for quadrotor_3d. See the module docstring for the derivation.

        x = [p(3), q(4), v(3), omega(3)];  u = (f1..f4);  dot v = (sum(u)/m) R e3 - g e3.
        n arrives as the HORIZONTAL outward normal (the parent slices centers to the position dim).
        """
        from src.envs.quadrotor_3d import _quat_to_R
        v_xy = x[:, 7:9]
        n_xy = n[:, :2]
        n_dot_v = torch.sum(n_xy * v_xy, dim=-1)
        R = _quat_to_R(x[:, 3:7])
        re3 = R[:, :, 2]                                   # body-up axis in world
        n_dot_re3 = torch.sum(n_xy * re3[:, :2], dim=-1)   # n is horizontal: only the xy part enters
        m = float(self.system.mass)

        lf_h = -n_dot_v
        # NO gravity term: n is horizontal, gravity is vertical, so n . g_vec = 0 identically.
        lf2_h = -(torch.sum(v_xy * v_xy, dim=-1) - n_dot_v ** 2) / rho
        # the collective row is ones(4): f_thr = sum(u), so every rotor carries the SAME coefficient
        coef = -n_dot_re3 / m
        a_row = coef.unsqueeze(-1).expand(-1, int(self.system.action_dim)).contiguous()

        # instrumentation for the degeneracy measurement (does not enter the row)
        self.last_a_norm = torch.linalg.norm(a_row, dim=-1).detach()
        self.last_tilt = torch.arccos(torch.clamp(R[:, 2, 2], -1.0, 1.0)).detach()
        return lf_h, lf2_h, a_row
