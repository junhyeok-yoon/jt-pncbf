"""v2.7.6 R6.3 M0 — attitude-aware vertical-recovery feasibility instrument.

R6.2's 3g-relaxed ballistic is attitude-blind (returned 0/2000 on the R6.1 pool tilt made adverse). This
models the vertical fall during an attitude recovery, using the SAME plant constants as prop:hold.

MODEL (stated exactly; conservative choices noted). Per IC (z0, v_z0, theta0=initial tilt, |omega0|):
  - Attitude is a 1-DOF tilt theta in [0, pi] with theta_dot in [-omega_max, +omega_max], theta_ddot in
    [-alpha_max, +alpha_max]. alpha_max = tau_max / Jx with tau_max = 2 * (arm_L/sqrt2) * f_rotor_max (the
    plant's max roll/pitch torque from the X-mixer tau_x=l(f1+f2-f3-f4)); omega_max = 4.0 (plant clamp).
  - WORST-CASE initial spin: theta_dot(0) = +min(|omega0|, omega_max) (the full initial rate acts to
    INCREASE tilt) -> the recovery must first null an adverse spin. Conservative.
  - The vehicle rotates time-optimally (bang-bang, velocity-clamped) toward upright theta=0 — the fastest
    way to regain altitude authority.
  - BEST-EFFORT vertical accel at each instant: a_z(theta) = g*(2*max(cos theta, 0) - 1). While inverted
    (theta>90deg) the best is zero thrust -> free fall a_z=-g (thrusting down is worse); once theta<90deg
    full thrust gives a_z=g(2cos theta-1), reaching +g upright, 0 at the 60deg holding limit. This is the
    true altitude-optimal policy, so the model is close to the actual best recovery (mildly optimistic on
    the thrust/torque allocation coupling, which is neglected — stated), kept sound by the adverse-spin and
    omega_max worst cases.
  - Integrate (theta, theta_dot, z, v_z) at dt=1e-3 s until arrested (v_z>=0 with theta<=60deg) or z<-8.
  FLAGGED = the modelled deepest z crosses -oob_limit (-8 m) before the vehicle can arrest.

Sound direction: flagged ICs are ones even this best-effort vertical recovery cannot save -> excluding them
from eval removes non-policy failures. Read-only on pools. No git, no securing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
G = float(REG["plant"]["gravity"])                 # 9.81
OOB = float(REG["plant"]["oob_limit"])             # 8.0
TWR = 2.0
# plant constants (exp_config env.quadrotor_3d / bounds)
ARM_L = 0.17; F_ROTOR_MAX = 4.905; JX = 0.01; OMEGA_MAX = 4.0
L_ARM = ARM_L / math.sqrt(2.0)
TAU_MAX = 2.0 * L_ARM * F_ROTOR_MAX                # max roll/pitch torque, N*m
ALPHA_MAX = TAU_MAX / JX                           # rad/s^2
THETA_HOLD = math.acos(1.0 / TWR)                  # 60 deg
DT = 1.0e-3; MAXSTEPS = 4000

MODEL_CONSTANTS = {"g": G, "TWR": TWR, "arm_L": ARM_L, "l_moment": round(L_ARM, 6),
                   "f_rotor_max": F_ROTOR_MAX, "Jx": JX, "tau_max": round(TAU_MAX, 6),
                   "alpha_max": round(ALPHA_MAX, 4), "omega_max": OMEGA_MAX,
                   "theta_hold_deg": round(math.degrees(THETA_HOLD), 4), "dt": DT, "maxsteps": MAXSTEPS,
                   "oob_limit": OOB}


def tilt_and_spin(scene):
    q = np.asarray(scene.initial_attitude_quat, float)
    cos_t = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2      # R(q)[2,2]
    theta = math.acos(min(1.0, max(-1.0, float(cos_t))))
    omega = float(np.linalg.norm(np.asarray(scene.initial_omega_vec, float)))
    return theta, omega


def flag_scenes(theta0, omega0, z0, vz0):
    """Vectorized over N ICs. Returns (flagged bool[N], minz float[N])."""
    N = theta0.shape[0]
    theta = theta0.astype(np.float64).copy()
    thdot = np.minimum(omega0, OMEGA_MAX).astype(np.float64).copy()   # adverse (+ increases tilt)
    z = z0.astype(np.float64).copy(); vz = vz0.astype(np.float64).copy()
    minz = z.copy()
    done = np.zeros(N, bool)
    for _ in range(MAXSTEPS):
        # time-optimal bang-bang toward theta=0 (double integrator, velocity clamp)
        sw = theta + 0.5 * thdot * np.abs(thdot) / ALPHA_MAX
        u = np.where(sw > 0.0, -ALPHA_MAX, ALPHA_MAX)
        thdot = np.clip(thdot + u * DT, -OMEGA_MAX, OMEGA_MAX)
        theta = theta + thdot * DT
        # tilt physically in [0, pi]; upright is an absorbing floor for the 1-DOF proxy
        under = theta <= 0.0
        theta = np.where(under, 0.0, theta)
        thdot = np.where(under & (thdot < 0.0), 0.0, thdot)
        theta = np.minimum(theta, math.pi)
        az = G * (2.0 * np.maximum(np.cos(theta), 0.0) - 1.0)
        vz = vz + az * DT
        z = z + vz * DT
        minz = np.minimum(minz, z)
        arrested = (vz >= 0.0) & (theta <= THETA_HOLD)
        done = done | arrested | (minz < -OOB)
        if done.all():
            break
    flagged = minz < -OOB
    return flagged, minz


# ============================================================================================
# v2.7.6 Stage-1 M1 — D_down: altitude LOST while righting to upright, integrated against the BAND
# floor -4 (not the -8 oob box), with NO more authority than the plant grants. Corrections vs the R6.3
# best-effort model (flag_scenes above):
#   (a) allocation coupling — torque and thrust share the per-rotor box (quadrotor_3d.py:52-55), so while
#       max righting torque is applied (a "bang" phase) the available total thrust is m*g, i.e.
#       TWR_eff = 2 - |tau|/tau_max = 1; on the velocity-clamp COAST phase (|tau|~0) TWR_eff = 2; inside the
#       holding cone (theta<=theta_hold) the vehicle eases torque and uses full thrust (TWR_eff=2).
#   (b) one ZOH control step of delay (dt=0.05): for the first step no corrective torque/thrust acts — the
#       vehicle free-falls (a_z=-g) while attitude drifts under the initial spin.
#   (c) gyroscopic term omega x J omega at worst-case sign: reduces the net righting angular accel to
#       ALPHA_EFF = (tau_max - tau_gyro_max)/Jx, tau_gyro_max = |Jz-Jy|*omega_max^2 on the roll/pitch axis.
#   (d) R6.3 worst-case adverse initial spin: theta_dot(0) = +min(|omega0|, omega_max).
# Best-effort vertical accel each step: a_z = g*(TWR_eff*max(cos theta,0) - 1). D_down = -min_t z(t) from
# z(0)=0. Deterministic, pure-python scalar (called per scene at build; ~O(1e3) steps). D_up is closed form.
DT_CTRL = 0.05                                         # one ZOH control step
JZ = 0.02; JY = 0.01                                   # inertia (exp_config env.quadrotor_3d)
TAU_GYRO_MAX = abs(JZ - JY) * OMEGA_MAX ** 2           # worst-case roll/pitch gyroscopic torque, N*m
ALPHA_EFF = (TAU_MAX - TAU_GYRO_MAX) / JX              # net righting angular accel, rad/s^2
DT_D = 5.0e-3; MAXSTEPS_D = 1400                        # integration step / cap (7 s) for D_down

MODEL_CONSTANTS.update({"dt_ctrl": DT_CTRL, "tau_gyro_max": round(TAU_GYRO_MAX, 6),
                        "alpha_eff": round(ALPHA_EFF, 4), "dt_D": DT_D, "maxsteps_D": MAXSTEPS_D,
                        "band_floor": -4.0, "band_ceiling": 4.0})


def D_up(vz0):
    """Closed-form ballistic rise against the band ceiling +4: max(0, v_z)^2 / (2 g)."""
    up = max(0.0, float(vz0))
    return up * up / (2.0 * G)


def D_down_single(theta0, vz0, omega_mag):
    """Altitude lost while righting to upright, plant-coupled (a)-(d). Returns D_down >= 0 (= -min z from
    z0=0). Pure python; deterministic."""
    theta = float(theta0)
    thdot = min(float(omega_mag), OMEGA_MAX)           # (d) adverse initial spin (+ increases tilt)
    z = 0.0; vz = float(vz0); minz = 0.0; t = 0.0
    for _ in range(MAXSTEPS_D):
        if t < DT_CTRL:                                # (b) ZOH delay: no corrective control this step
            u = 0.0; twr = 0.0
        else:
            sw = theta + 0.5 * thdot * abs(thdot) / ALPHA_EFF   # min-time bang-bang toward theta=0
            u = -ALPHA_EFF if sw > 0.0 else ALPHA_EFF
            if theta > THETA_HOLD:                     # righting regime
                thdot_unc = thdot + u * DT_D
                twr = 2.0 if abs(thdot_unc) > OMEGA_MAX else 1.0   # (a) coast=full thrust, bang=half
            else:                                      # inside holding cone: ease torque, full thrust
                twr = 2.0
        thdot = min(OMEGA_MAX, max(-OMEGA_MAX, thdot + u * DT_D))
        theta = theta + thdot * DT_D
        if theta <= 0.0:
            theta = 0.0
            if thdot < 0.0:
                thdot = 0.0
        elif theta > math.pi:
            theta = math.pi
        az = G * (twr * max(math.cos(theta), 0.0) - 1.0)
        vz = vz + az * DT_D
        z = z + vz * DT_D
        if z < minz:
            minz = z
        t += DT_D
        if vz >= 0.0 and theta <= THETA_HOLD:          # arrested and able to hold
            break
    return -minz


def band_z_interval(theta, vz, omega_mag, floor=-4.0, ceiling=4.0):
    """Admissible start_z interval [floor + D_down, ceiling - D_up] for one IC. Empty iff lo > hi."""
    return floor + D_down_single(theta, vz, omega_mag), ceiling - D_up(vz)


def check_pool(pool):
    th = np.array([0.0] * len(pool.scenes)); om = th.copy(); z0 = th.copy(); vz0 = th.copy()
    for i, s in enumerate(pool.scenes):
        t, o = tilt_and_spin(s); th[i] = t; om[i] = o
        z0[i] = float(s.start[2]); vz0[i] = float(s.initial_velocity[2])
    flagged, minz = flag_scenes(th, om, z0, vz0)
    idx = np.where(flagged)[0]
    flagged_ICs = [{"episode_idx": int(i), "z": round(float(z0[i]), 3), "v_z": round(float(vz0[i]), 3),
                    "tilt_deg": round(float(math.degrees(th[i])), 1), "omega": round(float(om[i]), 3),
                    "model_min_z": round(float(minz[i]), 3)} for i in idx[:60]]
    return {"n": len(pool.scenes), "flagged_total": int(flagged.sum()),
            "flagged_fraction": round(float(flagged.mean()), 6),
            "model_min_z": {"min": round(float(minz.min()), 3), "p1": round(float(np.percentile(minz, 1)), 3),
                            "median": round(float(np.median(minz)), 3)},
            "flagged_ICs": flagged_ICs}


if __name__ == "__main__":
    # M0 VALIDATION GATE: the extended instrument must flag NONZERO on the R6.1 eval-IC pool (where the
    # attitude-blind R6.2 returned 0), and be discriminating (not flag everything) on full-range/canonical.
    from src.eval.build_pools import load_pool
    POOLS = REPO / "data/runs/v2.7.6/pools"
    OUT = REPO / "data/runs/v2.7.6/r63_m0"; OUT.mkdir(parents=True, exist_ok=True)
    targets = {"R6.1_eval_ic_seed42": "eval_evalicz_quadrotor-3d-d2r_n2000_seed42",
               "full_range_seed42": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42",
               "canonical_seed23456": "eval_full_quadrotor-3d-d2r_n2000_seed23456"}
    out = {"model_constants": MODEL_CONSTANTS, "pools": {}}
    for key, stem in targets.items():
        rec = check_pool(load_pool(POOLS / f"{stem}.pkl"))
        out["pools"][key] = rec
        print(f"[{key}] flagged {rec['flagged_total']}/{rec['n']} (frac {rec['flagged_fraction']}) "
              f"model_min_z: min={rec['model_min_z']['min']} p1={rec['model_min_z']['p1']}")
    r61 = out["pools"]["R6.1_eval_ic_seed42"]["flagged_total"]
    out["validation_gate"] = {
        "R6.1_pool_flagged": r61,
        "attitude_blind_R6.2_flagged_on_same_pool": 0,
        "PASS": bool(r61 > 0),
        "note": "gate requires nonzero on the R6.1 pool where the attitude-blind instrument returned 0"}
    (OUT / "m0_validation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[M0 VALIDATION GATE] R6.1 pool flagged {r61} (attitude-blind gave 0) -> "
          f"{'PASS' if r61 > 0 else 'FAIL — instrument still blind, STOP'}")
