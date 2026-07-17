"""v2.6.2 amendment 3 — ANALYTIC attitude-aware IC recoverability test for the planar quadrotor.

Model-independent (plant constants ONLY, no V_hat / no learned quantity). The existing position/velocity
stopping-distance filter ignores ATTITUDE: an underactuated quadrotor must first ROTATE the thrust axis
Re(theta)=(-sin theta, cos theta) onto the obstacle's outward normal n_k before it can brake the inward
closure; during that alignment the closure continues (and gravity may feed it). An IC is UNRECOVERABLE
w.r.t. obstacle k iff the surface distance is smaller than the inward distance travelled during alignment
+ the post-alignment braking distance:

    d_k < s_k*t_align + 0.5*a_adv*t_align^2 + (s_k + a_adv*t_align)^2 / (2*a_brake) + margin

  s_k     = relu(-v_0 . n_k)                inward closure speed (0 if receding)
  d_k     = ||p_0 - c_k|| - r_k             SURFACE distance
  n_k     = (p_0 - c_k)/||.||               OUTWARD normal
  t_align = min-time rotation of Re(theta_0) onto n_k under alpha_max=tau_max/J, omega_max, from omega_0
            (double-integrator bang-bang; omega_0 helps if toward the target, adverse correction if away)
  a_adv   = g*max(0, n_k_y)                 gravity feeds the closure while the obstacle is below the body
  a_brake = f_max/m - g                     direction-independent guaranteed net authority

Pure / deterministic (seeded pools reproducible). Reject iff ANY active obstacle fails.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def plant_params(config: Mapping[str, Any]) -> dict:
    q = config["env"]["quadrotor_planar"]
    b = config["env"]["bounds"]["quadrotor_planar"]
    m = float(q["mass"]); g = float(q["gravity"]); J = float(q["inertia"])
    f_max = float(b["f_max"]); tau_max = float(b["tau_max"]); omega_max = float(b["omega_max"])
    return dict(m=m, g=g, alpha_max=tau_max / J, omega_max=omega_max,
                a_brake=f_max / m - g)


def _rest_to_rest_time(d: float, A: float, W: float) -> float:
    """Min time to traverse distance d>=0 from rest to rest, |accel|<=A, |vel|<=W (triangular/trapezoidal)."""
    if d <= 0.0:
        return 0.0
    v_peak = math.sqrt(A * d)                      # triangular peak
    if v_peak <= W:
        return 2.0 * v_peak / A
    return d / W + W / A                           # trapezoidal (speed-capped)


def _min_time_align(delta: float, omega0: float, A: float, W: float) -> float:
    """Min time from (theta=0, omega=omega0) to (theta=delta, omega=0), |alpha|<=A, |omega|<=W. omega0 in the
    TARGET direction helps; adverse omega0 (away) adds a reversal cost. delta signed, omega0 signed."""
    d = abs(delta)
    if d < 1.0e-12 and abs(omega0) < 1.0e-12:
        return 0.0
    s = 1.0 if delta >= 0.0 else -1.0
    w = omega0 * s                                  # velocity component toward the target
    t0 = 0.0
    if w < 0.0:                                     # adverse: decelerate to 0 first, moving backward
        t0 = -w / A
        d = d + w * w / (2.0 * A)                   # extra distance from the backward excursion
        w = 0.0
    # from (pos 0, vel w>=0) to (pos d, vel 0)
    if w > 0.0 and (w * w) / (2.0 * A) > d:         # overshoot: must decelerate past d, then return
        t_over = w / A
        d_back = (w * w) / (2.0 * A) - d
        return t0 + t_over + _rest_to_rest_time(d_back, A, W)
    v_peak = math.sqrt(A * d + 0.5 * w * w)         # peak vel of the accel(+A)->decel(-A) profile
    if v_peak <= W:
        return t0 + (v_peak - w) / A + v_peak / A
    # speed-capped trapezoid: up w->W, cruise, down W->0
    t_up = (W - w) / A; t_dn = W / A
    d_up = (W * W - w * w) / (2.0 * A); d_dn = (W * W) / (2.0 * A)
    return t0 + t_up + max(0.0, d - d_up - d_dn) / W + t_dn


def is_recoverable(p0: np.ndarray, v0: np.ndarray, theta0: float, omega0: float,
                   centers: np.ndarray, radii: np.ndarray, active: np.ndarray,
                   plant: dict, margin: float) -> bool:
    """True iff the IC is recoverable w.r.t. EVERY active obstacle (analytic attitude-aware test)."""
    A = plant["alpha_max"]; W = plant["omega_max"]; g = plant["g"]; a_brake = plant["a_brake"]
    re = np.array([-math.sin(theta0), math.cos(theta0)])          # thrust axis
    for c, r, act in zip(centers, radii, active):
        if not bool(act):
            continue
        rel = p0 - c                                              # obstacle -> body
        dist = float(np.linalg.norm(rel))
        if dist < 1.0e-9:
            return False
        n = rel / dist                                            # outward normal
        d_k = dist - float(r)                                     # surface distance
        s_k = max(0.0, -float(np.dot(v0, n)))                     # inward closure speed
        if s_k <= 0.0:
            continue                                              # receding: this obstacle cannot doom
        # signed rotation to align Re with n (rotate theta by delta rotates Re by delta)
        cross = re[0] * n[1] - re[1] * n[0]
        dot = float(np.dot(re, n))
        delta = math.atan2(cross, dot)
        t_align = _min_time_align(delta, float(omega0), A, W)
        a_adv = g * max(0.0, float(n[1]))                         # gravity feeds closure if body above obstacle
        v_end = s_k + a_adv * t_align                             # inward speed at end of alignment
        travel = s_k * t_align + 0.5 * a_adv * t_align * t_align + (v_end * v_end) / (2.0 * a_brake)
        if d_k < travel + margin:
            return False
    return True


def recoverability_detail(p0, v0, theta0, omega0, centers, radii, active, plant, margin):
    """Diagnostic decomposition (v2.6.2 FP anatomy): per-obstacle criterion parts + the BINDING obstacle
    (max requirement-violation). Returns flagged(bool) and the binding parts (d_k, s_k, t_align, delta_theta,
    align_advance, brake_dist, d_req, slack=d_k-d_req). Mirrors is_recoverable exactly."""
    A = plant["alpha_max"]; W = plant["omega_max"]; g = plant["g"]; a_brake = plant["a_brake"]
    re = np.array([-math.sin(theta0), math.cos(theta0)])
    best = None  # (violation, parts)
    for _i, (c, r, act) in enumerate(zip(centers, radii, active)):
        if not bool(act):
            continue
        rel = p0 - c; dist = float(np.linalg.norm(rel))
        if dist < 1.0e-9:
            return dict(flagged=True, d_k=0.0, s_k=0.0, v_perp=0.0, t_align=0.0, delta_theta=0.0,
                        align_advance=0.0, brake_dist=0.0, d_req=float("inf"), slack=-float("inf"),
                        obs_idx=int(_i), n=np.zeros(2), center=np.asarray(c, np.float64).copy(), radius=float(r))
        n = rel / dist; d_k = dist - float(r); s_k = max(0.0, -float(np.dot(v0, n)))
        if s_k <= 0.0:
            continue
        v_perp = float(np.linalg.norm(v0 - float(np.dot(v0, n)) * n))     # tangential (lateral) speed
        cross = re[0] * n[1] - re[1] * n[0]; dot = float(np.dot(re, n)); delta = math.atan2(cross, dot)
        t_align = _min_time_align(delta, float(omega0), A, W)
        a_adv = g * max(0.0, float(n[1])); v_end = s_k + a_adv * t_align
        align_adv = s_k * t_align + 0.5 * a_adv * t_align * t_align
        brake = (v_end * v_end) / (2.0 * a_brake)
        d_req = align_adv + brake + margin
        viol = d_req - d_k
        parts = dict(d_k=d_k, s_k=s_k, v_perp=v_perp, t_align=t_align, delta_theta=abs(delta),
                     align_advance=align_adv, brake_dist=brake, d_req=d_req, slack=d_k - d_req,
                     obs_idx=int(_i), n=n.copy(), center=np.asarray(c, np.float64).copy(), radius=float(r))
        if best is None or viol > best[0]:
            best = (viol, parts)
    if best is None:            # no closing obstacle -> not flagged
        return dict(flagged=False, d_k=float("inf"), s_k=0.0, v_perp=0.0, t_align=0.0, delta_theta=0.0,
                    align_advance=0.0, brake_dist=0.0, d_req=0.0, slack=float("inf"),
                    obs_idx=-1, n=np.zeros(2), center=np.zeros(2), radius=0.0)
    parts = best[1]; parts["flagged"] = best[0] > 0.0
    return parts
