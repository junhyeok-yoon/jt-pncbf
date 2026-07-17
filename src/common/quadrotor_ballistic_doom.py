"""v2.6.2 §doom census — ballistic closed-form doom test on the RELAXED double-integrator system.

SOUNDNESS FRAME (stated verbatim in the report). The true per-axis acceleration is (f/m)*Re(theta) - g*e_y
with f in [0, f_max], so the true acceleration set is contained in the ball of radius f_max/m + g = 3g about
0. The RELAXED system -- a point mass p_ddot = u, ||u|| <= A := f_max/m + g (= 3g), NO attitude, NO velocity
clamp -- is therefore AT LEAST AS CAPABLE as the plant. Hence relaxed-system inevitable penetration for SOME
single obstacle k  =>  true doom in the full scene (capability containment + constraints-only-shrink). This
test is an UNDER-approximation of true doom: ZERO false exclusion by construction. What it CANNOT certify is
doom that exists only through the attitude/alignment cost (that boundary is stated, not certified).

REACHABLE-SET DOOM CERTIFICATE. From (p0, v0) under ||u|| <= A the position reachable at time t is the disc
centered at the ballistic point b(t) = p0 + v0*t with radius rho(t) = 0.5*A*t^2 (full authority, any
direction). If at some t the WHOLE reachable disc lies inside an obstacle disc (radius R, center c), every
admissible trajectory is inside at t -> collision inevitable -> doom:

    exists t >= 0 :   ||b(t) - c|| + 0.5*A*t^2  <  R           (reachable disc subset of obstacle disc)

This is SUFFICIENT for doom (sound); the exact relaxed doom is the 4D HJ value (census part B), so the
closed-form flag set is a SUBSET of the HJ flag set. Head-on special case (v0 = -s_k*n, n = outward normal):
g(t) = (d - s_k*t) + 0.5*A*t^2 is minimized at t* = s_k/A, giving d_k < s_k^2/(2A) -- normal closure vs
braking distance, with a lateral-escape allowance built in off-axis. g(t) = ||b(t)-c|| + 0.5*A*t^2 is convex
(norm of an affine map + convex quadratic) -> unique minimum by ternary search on [0, sqrt(2R/A)] (for
t > sqrt(2R/A), 0.5*A*t^2 >= R so g >= R, cannot flag).

Radius is rounded DOWN to shared buckets (smaller obstacle = easier to avoid = fewer flags = conservative)
so the closed form and the HJ census use the identical R_k^- and the A-subset-of-B gate is exact.

Plant constants only; pure / deterministic.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

BUCKET_LO = 0.15
BUCKET_HI = 0.80
BUCKET_STEP = 0.05
EPS_A = 1.0e-9                     # strict-penetration guard (analytic test has no discretisation error)


def accel_bound(config: Mapping[str, Any]) -> float:
    q = config["env"]["quadrotor_planar"]
    b = config["env"]["bounds"]["quadrotor_planar"]
    return float(b["f_max"]) / float(q["mass"]) + float(q["gravity"])          # = 3g


def radius_bucket_down(r: float) -> float:
    """Round a radius DOWN to the shared bucket grid (conservative: smaller obstacle)."""
    r = float(r)
    if r <= BUCKET_LO:
        return BUCKET_LO
    k = math.floor((min(r, BUCKET_HI) - BUCKET_LO) / BUCKET_STEP + 1e-9)
    return round(BUCKET_LO + k * BUCKET_STEP, 10)


def min_approach(p0: np.ndarray, v0: np.ndarray, center: np.ndarray, A: float, R: float) -> float:
    """min_{t>=0} ( ||p0 + v0*t - c|| + 0.5*A*t^2 ).  Convex; ternary search on [0, sqrt(2R/A)]."""
    rel0 = np.asarray(p0, np.float64) - np.asarray(center, np.float64)
    v = np.asarray(v0, np.float64)

    def g(t: float) -> float:
        b = rel0 + v * t
        return float(math.hypot(b[0], b[1]) + 0.5 * A * t * t)

    hi = math.sqrt(2.0 * R / A) if A > 0 else 0.0
    lo = 0.0
    for _ in range(80):                                   # (2/3)^80 ~ 1e-14 relative
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if g(m1) < g(m2):
            hi = m2
        else:
            lo = m1
    t = 0.5 * (lo + hi)
    return min(g(t), g(0.0))                               # include the t=0 endpoint


def is_doomed_ballistic(p0: np.ndarray, v0: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                        active: np.ndarray, A: float, use_buckets: bool = True) -> bool:
    """True iff SOME active obstacle guarantees penetration of the relaxed system (sound doom certificate)."""
    p0 = np.asarray(p0, np.float64)
    v0 = np.asarray(v0, np.float64)
    for c, r, act in zip(np.asarray(centers, np.float64), np.asarray(radii, np.float64),
                         np.asarray(active, bool)):
        if not bool(act):
            continue
        R = radius_bucket_down(float(r)) if use_buckets else float(r)
        if min_approach(p0, v0, c, A, R) < R - EPS_A:
            return True
    return False
