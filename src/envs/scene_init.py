from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scene:
    obstacle_centers: Array
    obstacle_radii: Array
    obstacle_active: npt.NDArray[np.bool_]
    start: Array
    goal: Array
    system: str
    mode: str
    initial_velocity: Array | None = None
    initial_speed: float | None = None
    initial_heading: float | None = None
    initial_attitude: float | None = None      # quadrotor_planar: body attitude theta
    initial_omega: float | None = None         # quadrotor_planar: body angular rate omega
    # quadrotor_3d (v2.7.2): start/goal are FULL 3D positions (single source of truth); the xy-clearance
    # filters slice the first 2 coords (cylinders are infinite-vertical). Attitude/rates carried here.
    initial_attitude_quat: Array | None = None  # unit quaternion [w,x,y,z], body->world (6-DOF IC)
    initial_omega_vec: Array | None = None       # body angular rate omega (3,)


@dataclass(frozen=True)
class _SceneModeParams:
    mode: str
    min_start_goal_dist: float
    start_goal_clearance: float


_MAX_RETRIES = 1000
_START_GOAL_ARENA_MARGIN = 0.3
_ZERO_SPEED_EPS = 1.0e-12

# v2.7.6 Stage-1 M2: band-feasible eval-IC redraw diagnostics (eval-only, quadrotor_3d floor_mode=="band").
# Incremented only inside that branch; read/reset by the pool builder. Training path never touches this.
_BAND_STATS = {"scenes": 0, "attempts": 0, "empty_interval_redraws": 0, "obstacle_scene_rejects": 0}


def band_stats() -> dict:
    return dict(_BAND_STATS)


def reset_band_stats() -> None:
    for _k in _BAND_STATS:
        _BAND_STATS[_k] = 0


def sample_train_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
) -> Scene:
    params = _SceneModeParams(
        mode="train",
        min_start_goal_dist=float(config["scene_train"]["min_start_goal_dist"]),
        start_goal_clearance=float(config["scene_train"]["start_goal_clearance"]),
    )
    return _sample_scene(rng, config, system, params)


def _sample_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
    params: _SceneModeParams,
) -> Scene:
    if system not in {"double_integrator", "unicycle", "quadrotor_planar", "quadrotor_3d"}:
        raise ValueError(f"Unsupported system: {system!r}")

    for _ in range(_MAX_RETRIES):
        start, goal = _sample_start_goal(rng, config, params)
        centers, radii, active = _sample_obstacles(rng, config, start, goal, system)
        scene = _make_scene(
            rng,
            config,
            system,
            params.mode,
            centers,
            radii,
            active,
            start,
            goal,
        )

        if not _has_start_goal_clearance(scene, params.start_goal_clearance):
            continue
        if not _passes_unavoidable_collision_filter(scene, config, params):
            continue
        if not _passes_recoverability_filter(scene, config):   # v2.6.2 amendment 3 (quadrotor only, additive)
            continue
        if not _passes_band_obstacle_screen(scene, config, params):   # v2.7.6 Stage-1 (eval band mode only)
            _BAND_STATS["obstacle_scene_rejects"] += 1
            continue
        return scene

    raise RuntimeError(
        f"Scene sampling exceeded retry cap ({_MAX_RETRIES}) for mode={params.mode!r}, "
        f"system={system!r}."
    )


def _sample_start_goal(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    params: _SceneModeParams,
) -> tuple[Array, Array]:
    world_lim = float(config["env"]["world_lim"])
    coord_lim = world_lim - _START_GOAL_ARENA_MARGIN

    for _ in range(_MAX_RETRIES):
        start = rng.uniform(-coord_lim, coord_lim, size=2).astype(np.float64)
        goal = rng.uniform(-coord_lim, coord_lim, size=2).astype(np.float64)
        if np.linalg.norm(goal - start) >= params.min_start_goal_dist:
            return start, goal

    raise RuntimeError(
        "Start/goal sampling exceeded retry cap "
        f"({_MAX_RETRIES}) for mode={params.mode!r}."
    )


def _resolve_obstacle_cfg(config: Mapping[str, Any], system: str) -> Mapping[str, Any]:
    """Per-system obstacle spec = `obstacle.per_system[system]` merged OVER the shared `obstacle` block.
    An ABSENT override returns the shared block unchanged (byte-identical sampling — v2.7.3 M0b invariant).
    The reserved keys `per_system` and `variant` are never read as obstacle parameters."""
    base = config["obstacle"]
    override = base.get("per_system", {}).get(system, {})
    if not override:
        return base
    merged = {k: v for k, v in base.items() if k != "per_system"}
    merged.update({k: v for k, v in override.items() if k != "variant"})
    return merged


def _sample_obstacles(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    start: Array,
    goal: Array,
    system: str,
) -> tuple[Array, Array, npt.NDArray[np.bool_]]:
    obstacle_cfg = _resolve_obstacle_cfg(config, system)
    n_min = int(obstacle_cfg["n_min"])
    n_max = int(obstacle_cfg["n_max"])
    r_min = float(obstacle_cfg["r_min"])
    r_max = float(obstacle_cfg["r_max"])

    n_active = int(rng.integers(n_min, n_max + 1))
    centers = np.zeros((n_max, 2), dtype=np.float64)
    radii = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.bool_)

    radii[:n_active] = rng.uniform(r_min, r_max, size=n_active)
    active[:n_active] = True

    if rng.uniform() < float(obstacle_cfg["p_corridor"]):
        centers[:n_active] = _sample_corridor_centers(
            rng, config, start, goal, n_active
        )
    else:
        world_lim = float(config["env"]["world_lim"])
        centers[:n_active] = rng.uniform(-world_lim, world_lim, size=(n_active, 2))

    return centers, radii, active


def _sample_corridor_centers(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    start: Array,
    goal: Array,
    n_active: int,
) -> Array:
    world_lim = float(config["env"]["world_lim"])
    obstacle_cfg = config["obstacle"]
    segment = goal - start
    length = float(np.linalg.norm(segment))
    if length <= _ZERO_SPEED_EPS:
        return rng.uniform(-world_lim, world_lim, size=(n_active, 2)).astype(np.float64)

    direction = segment / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    midpoint = 0.5 * (start + goal)
    along_half = 0.5 * float(obstacle_cfg["corridor_along_factor"]) * length
    perp_half = 0.5 * float(obstacle_cfg["corridor_perp_factor"]) * length

    centers = np.zeros((n_active, 2), dtype=np.float64)
    for idx in range(n_active):
        for _ in range(_MAX_RETRIES):
            along = rng.uniform(-along_half, along_half)
            perp = rng.uniform(-perp_half, perp_half)
            center = midpoint + along * direction + perp * normal
            if np.all(np.abs(center) <= world_lim):
                centers[idx] = center
                break
        else:
            raise RuntimeError(
                f"Corridor obstacle sampling exceeded retry cap ({_MAX_RETRIES})."
            )

    return centers


def _make_scene(
    rng: np.random.Generator,
    config: Mapping[str, Any],
    system: str,
    mode: str,
    centers: Array,
    radii: Array,
    active: npt.NDArray[np.bool_],
    start: Array,
    goal: Array,
) -> Scene:
    v_init_max = float(config["env"]["v_init_max"])

    if system == "double_integrator":
        initial_velocity = rng.uniform(-v_init_max, v_init_max, size=2).astype(
            np.float64
        )
        return Scene(
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_active=active,
            start=start,
            goal=goal,
            system=system,
            mode=mode,
            initial_velocity=initial_velocity,
        )

    if system == "quadrotor_planar":
        # v2.6.0 §4 IC set: v ~ U(disk radius v_init), theta ~ U[-pi,pi], omega ~ U[-omega_init_max, .].
        q = config["env"]["quadrotor_planar"]
        v_init = float(q["v_init_max"])
        omega_init = float(q["omega_init_max"])
        radius = v_init * float(np.sqrt(rng.uniform()))       # uniform-in-disk
        angle = float(rng.uniform(-np.pi, np.pi))
        initial_velocity = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64
        )
        return Scene(
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_active=active,
            start=start,
            goal=goal,
            system=system,
            mode=mode,
            initial_velocity=initial_velocity,
            initial_attitude=float(rng.uniform(-np.pi, np.pi)),
            initial_omega=float(rng.uniform(-omega_init, omega_init)),
        )

    if system == "quadrotor_3d":
        # v2.7.2 §3.4 6-DOF IC: xy start/goal + independent altitudes; ||v0|| from the planar speed
        # distribution with uniform-3D direction; attitude q = R_a(phi_tilt) R_z(psi); per-axis omega.
        q = config["env"]["quadrotor_3d"]
        if bool(q.get("ic_so3", False)):
            # v2.7.4 M0c: full-SO(3) IC. Attitude uniform on SO(3) (Shoemake), angular + linear velocity
            # direction-uniform on S^2 with magnitudes U[0, ic_omega_max] / U[0, ic_v_max]. The obstacle
            # geometry (centers/radii/active) and xy start/goal are already sampled above (shared with the
            # v2.7.3 path); only the 6-DOF IC differs here.
            world_lim = float(config["env"]["world_lim"])
            # v2.7.6: eval-only recoverable-set z-bounds conditioned on v_z (prop:hold; changes.md).
            # Gated on mode=='eval' AND an ic_eval_z config block (opt-in) — NOT on ic_so3. The TRAINING
            # path (mode!='eval') and the DEFAULT eval path (no ic_eval_z block) keep the exact v2.7.5 draw
            # order (start_z, goal_z, v_speed, vdir, quat, w_mag, wdir) and stay byte-identical, so the
            # canonical seed23456 full-range pool still reproduces. The eval-IC path REORDERS: velocity
            # first to obtain v_z, then start_z conditioned on it.
            ic_eval_z = q.get("ic_eval_z") if mode == "eval" else None
            if ic_eval_z is not None and str(ic_eval_z.get("floor_mode", "vz")) == "band":
                # v2.7.6 Stage-1 M2: BAND-feasible eval IC. Draw attitude, omega_0, v_0 FIRST; admit start_z
                # only within [-4 + D_down, +4 - D_up], so no IC is charged for a band exit it cannot avoid.
                # Resample z ONLY (Haar SO(3) preserved exactly); redraw the whole IC ONLY when the z-interval
                # is empty or the R6.2 obstacle screen dooms it. goal_z ~ U(-4,+4) unchanged. Eval-only — the
                # training path (mode!="eval") never reaches here. D_down instrument lazily imported.
                from scripts.analysis.v276_attitude_feasibility import D_down_single, D_up as _D_up
                floor_b = float(ic_eval_z.get("band_floor", -world_lim))
                ceil_b = float(ic_eval_z.get("band_ceiling", world_lim))
                # v2.7.6 Stage-1c: apply the existing start-goal clearance to the band surfaces |z|=4 (Stage 2
                # makes them hazard surfaces in h_star). delta_eval is READ from config, never chosen here:
                # goal_z clears the band by delta_eval, and the start clears the ceiling by delta_eval too.
                delta_eval = float(config["eval"]["scene"]["start_goal_clearance"])
                # The R6.2 obstacle screen is applied at scene level (_passes_band_obstacle_screen), AFTER the
                # start-goal clearance filter, so it never sees a raw start-inside-obstacle scene; the inner
                # loop here handles only the z-interval (resample z; redraw the IC only if the interval is empty).
                for _ in range(_MAX_RETRIES):
                    _BAND_STATS["attempts"] += 1
                    quat = _shoemake_quat(rng)                                      # attitude first (Haar law)
                    w_mag = float(q["ic_omega_max"]) * float(rng.uniform())
                    wdir = rng.normal(size=3); wdir = wdir / max(float(np.linalg.norm(wdir)), _ZERO_SPEED_EPS)
                    omega_vec = (w_mag * wdir).astype(np.float64)
                    v_speed = float(q["ic_v_max"]) * float(rng.uniform())
                    vdir = rng.normal(size=3); vdir = vdir / max(float(np.linalg.norm(vdir)), _ZERO_SPEED_EPS)
                    initial_velocity = (v_speed * vdir).astype(np.float64)
                    cos_theta = float(quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2 + quat[3] ** 2)  # R(q)[2,2]
                    theta = float(np.arccos(min(1.0, max(-1.0, cos_theta))))
                    vz = float(initial_velocity[2]); omega_mag = float(np.linalg.norm(omega_vec))
                    z_lo = floor_b + D_down_single(theta, vz, omega_mag)           # floor unchanged (D_down > delta)
                    z_hi = min(ceil_b - delta_eval, ceil_b - _D_up(vz))            # start clears the ceiling too
                    if z_lo > z_hi:                                                 # empty interval -> redraw IC
                        _BAND_STATS["empty_interval_redraws"] += 1
                        continue
                    start_z = float(rng.uniform(z_lo, z_hi))                        # resample z ONLY
                    goal_z = float(rng.uniform(floor_b + delta_eval, ceil_b - delta_eval))  # goal clears the band
                    _BAND_STATS["scenes"] += 1
                    return Scene(
                        obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
                        start=np.array([start[0], start[1], start_z], dtype=np.float64),
                        goal=np.array([goal[0], goal[1], goal_z], dtype=np.float64),
                        system=system, mode=mode, initial_velocity=initial_velocity,
                        initial_attitude_quat=quat, initial_omega_vec=omega_vec,
                    )
                raise RuntimeError("band-feasible eval IC: exceeded redraw cap")
            if ic_eval_z is not None and str(ic_eval_z.get("floor_mode", "vz")) == "tilt":
                # v2.7.6 R6.3: TILT-conditioned floor (replaces the R6.1 v_z floor, which was falsified).
                # Attitude drawn BEFORE position (reorder); attitude law (Haar SO(3)) unchanged.
                # deficit(theta) = max(0, 1/TWR - cos theta); z_init_min = tilt_floor_base + k*deficit, so the
                # floor never goes below tilt_floor_base (-4) — it only RISES for tilted ICs. Ceiling and goal
                # per changes.md. Self-contained early return.
                quat = _shoemake_quat(rng)                                 # attitude first (law unchanged)
                w_mag = float(q["ic_omega_max"]) * float(rng.uniform())
                wdir = rng.normal(size=3); wdir = wdir / max(float(np.linalg.norm(wdir)), _ZERO_SPEED_EPS)
                omega_vec = (w_mag * wdir).astype(np.float64)
                v_speed = float(q["ic_v_max"]) * float(rng.uniform())
                vdir = rng.normal(size=3); vdir = vdir / max(float(np.linalg.norm(vdir)), _ZERO_SPEED_EPS)
                initial_velocity = (v_speed * vdir).astype(np.float64)
                v_z_up = max(0.0, float(initial_velocity[2]))
                cos_theta = float(quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2 + quat[3] ** 2)   # R(q)[2,2]
                twr = float(ic_eval_z.get("twr", 2.0))
                deficit = max(0.0, 1.0 / twr - cos_theta)
                z_ceiling_base = float(ic_eval_z["z_ceiling_base"])
                z_init_max = min(z_ceiling_base, z_ceiling_base - float(ic_eval_z["z_ceiling_v_up_slope"]) * v_z_up)
                z_init_min = float(ic_eval_z["tilt_floor_base"]) + float(ic_eval_z["tilt_floor_k"]) * deficit
                start_z = float(rng.uniform(z_init_min, z_init_max))
                goal_half = float(ic_eval_z["goal_z_half"])
                goal_z = float(rng.uniform(-goal_half, goal_half))
                return Scene(
                    obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
                    start=np.array([start[0], start[1], start_z], dtype=np.float64),
                    goal=np.array([goal[0], goal[1], goal_z], dtype=np.float64),
                    system=system, mode=mode, initial_velocity=initial_velocity,
                    initial_attitude_quat=quat, initial_omega_vec=omega_vec,
                )
            if ic_eval_z is not None:
                v_speed = float(q["ic_v_max"]) * float(rng.uniform())      # |v0| ~ U[0, ic_v_max]
                vdir = rng.normal(size=3); vdir = vdir / max(float(np.linalg.norm(vdir)), _ZERO_SPEED_EPS)
                initial_velocity = (v_speed * vdir).astype(np.float64)
                v_z = float(initial_velocity[2])
                v_z_down = max(0.0, -v_z); v_z_up = max(0.0, v_z)
                z_ceiling_base = float(ic_eval_z["z_ceiling_base"])
                z_init_max = min(z_ceiling_base,                            # ceiling: lowered only when ascending
                                 z_ceiling_base - float(ic_eval_z["z_ceiling_v_up_slope"]) * v_z_up)
                z_init_min = (float(ic_eval_z["z_floor_base"])             # floor: recovery budget vs oob at -8
                              + float(ic_eval_z["z_floor_v_down_slope"]) * v_z_down)
                start_z = float(rng.uniform(z_init_min, z_init_max))
                goal_half = float(ic_eval_z["goal_z_half"])
                goal_z = float(rng.uniform(-goal_half, goal_half))          # settling => low speed at goal
            else:
                if mode == "train" and config["env"].get("band_hazard", {}).get("enabled", False):
                    # v2.7.6 Stage-2: training clears the band surfaces |z|=4 by delta_train — the same
                    # start-goal clearance every other hazard gets. Rollouts still continue through the surface
                    # (M0a), so this only removes born-on-surface starts, not doomed descents (valid V_hat data).
                    d_tr = float(config["scene_train"]["start_goal_clearance"])
                    start_z = float(rng.uniform(-world_lim + d_tr, world_lim - d_tr))
                    goal_z = float(rng.uniform(-world_lim + d_tr, world_lim - d_tr))
                else:                                                        # eval default / band off -> unchanged
                    start_z = float(rng.uniform(-world_lim, world_lim))
                    goal_z = float(rng.uniform(-world_lim, world_lim))
                v_speed = float(q["ic_v_max"]) * float(rng.uniform())      # |v0| ~ U[0, ic_v_max]
                vdir = rng.normal(size=3); vdir = vdir / max(float(np.linalg.norm(vdir)), _ZERO_SPEED_EPS)
                initial_velocity = (v_speed * vdir).astype(np.float64)
            start3 = np.array([start[0], start[1], start_z], dtype=np.float64)
            goal3 = np.array([goal[0], goal[1], goal_z], dtype=np.float64)
            quat = _shoemake_quat(rng)                                     # uniform SO(3)
            w_mag = float(q["ic_omega_max"]) * float(rng.uniform())        # |omega| ~ U[0, ic_omega_max]
            wdir = rng.normal(size=3); wdir = wdir / max(float(np.linalg.norm(wdir)), _ZERO_SPEED_EPS)
            omega_vec = (w_mag * wdir).astype(np.float64)
            return Scene(
                obstacle_centers=centers,
                obstacle_radii=radii,
                obstacle_active=active,
                start=start3,
                goal=goal3,
                system=system,
                mode=mode,
                initial_velocity=initial_velocity,
                initial_attitude_quat=quat,
                initial_omega_vec=omega_vec,
            )
        v_init = float(q["v_init_max"])
        omega_init = float(q["omega_init_max"])
        world_lim = float(config["env"]["world_lim"])
        start_z = float(rng.uniform(-world_lim, world_lim))
        goal_z = float(rng.uniform(-world_lim, world_lim))
        start3 = np.array([start[0], start[1], start_z], dtype=np.float64)   # full 3D start
        goal3 = np.array([goal[0], goal[1], goal_z], dtype=np.float64)       # full 3D goal
        speed = v_init * float(np.sqrt(rng.uniform()))         # planar disk-radius marginal
        vdir = rng.normal(size=3)
        vdir = vdir / max(float(np.linalg.norm(vdir)), _ZERO_SPEED_EPS)  # uniform on S^2
        initial_velocity = (speed * vdir).astype(np.float64)
        psi = float(rng.uniform(-np.pi, np.pi))                # yaw
        alpha = float(rng.uniform(-np.pi, np.pi))              # tilt-axis azimuth (uniform horizontal)
        phi_tilt = float(rng.uniform(0.0, 2.0 * np.pi / 3.0))  # tilt magnitude (past-horizontal, no inversion)
        quat = _compose_ic_quat(psi, alpha, phi_tilt)
        omega_vec = rng.uniform(-omega_init, omega_init, size=3).astype(np.float64)
        return Scene(
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_active=active,
            start=start3,
            goal=goal3,
            system=system,
            mode=mode,
            initial_velocity=initial_velocity,
            initial_attitude_quat=quat,
            initial_omega_vec=omega_vec,
        )

    initial_speed = float(rng.uniform(-v_init_max, v_init_max))
    initial_heading = float(rng.uniform(-np.pi, np.pi))
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=start,
        goal=goal,
        system=system,
        mode=mode,
        initial_speed=initial_speed,
        initial_heading=initial_heading,
    )


def _quat_mul_np(a: Array, b: Array) -> Array:
    """Hamilton product [w,x,y,z], matching src/envs/quadrotor_3d._quat_mul (R(a b) = R(a) R(b))."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _shoemake_quat(rng) -> Array:
    """Uniform unit quaternion on SO(3) by Shoemake (1992). Three U[0,1) draws give a point uniform on S^3
    (= Haar-uniform on SO(3) via the double cover), returned as [w,x,y,z], canonicalized to w>=0. This is
    NOT uniform Euler angles, which is a different, non-Haar measure. Under this sampler cos(tilt)=R(q)[2,2]
    is uniform on [-1,1] (pinned by test), the property uniform-Euler badly violates."""
    u1, u2, u3 = (float(x) for x in rng.uniform(size=3))
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    two_pi = 2.0 * np.pi
    q = np.array([b * np.cos(two_pi * u3),      # w
                  a * np.sin(two_pi * u2),      # x
                  a * np.cos(two_pi * u2),      # y
                  b * np.sin(two_pi * u3)],     # z
                 dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), _ZERO_SPEED_EPS)
    if q[0] < 0.0:
        q = -q
    return q


def _compose_ic_quat(psi: float, alpha: float, phi_tilt: float) -> Array:
    """6-DOF IC attitude q = R_a(phi_tilt) R_z(psi) (body->world, [w,x,y,z]): yaw psi about world-z, then
    tilt phi_tilt about the horizontal axis a=(cos alpha, sin alpha, 0). Canonicalized to w>=0 (double-cover)."""
    cy, sy = np.cos(0.5 * psi), np.sin(0.5 * psi)
    q_yaw = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    ct, st = np.cos(0.5 * phi_tilt), np.sin(0.5 * phi_tilt)
    q_tilt = np.array([ct, st * np.cos(alpha), st * np.sin(alpha), 0.0], dtype=np.float64)
    q = _quat_mul_np(q_tilt, q_yaw)
    q = q / max(float(np.linalg.norm(q)), _ZERO_SPEED_EPS)
    if q[0] < 0.0:
        q = -q
    return q


def _passes_recoverability_filter(scene: Scene, config: Mapping[str, Any]) -> bool:
    """v2.6.2 amendment 3: ANALYTIC attitude-aware IC recoverability test (quadrotor_planar ONLY; additive to
    the position/velocity filter above). Rejects born-doomed ICs — where the thrust axis cannot rotate onto
    the obstacle normal and brake the inward closure before impact. No-op (returns True) for DI/unicycle and
    if the recov_margin key is absent. See src/common/quadrotor_recoverability.py."""
    if scene.system != "quadrotor_planar":
        return True
    q = config["env"].get("quadrotor_planar", {})
    if "recov_margin" not in q:
        return True
    from src.common.quadrotor_recoverability import is_recoverable, plant_params
    return is_recoverable(
        p0=np.asarray(scene.start, dtype=np.float64),
        v0=_initial_velocity_vector(scene),
        theta0=float(scene.initial_attitude if scene.initial_attitude is not None else 0.0),
        omega0=float(scene.initial_omega if scene.initial_omega is not None else 0.0),
        centers=scene.obstacle_centers, radii=scene.obstacle_radii, active=scene.obstacle_active,
        plant=plant_params(config), margin=float(q["recov_margin"]))


def _passes_band_obstacle_screen(scene: Scene, config: Mapping[str, Any], params: _SceneModeParams) -> bool:
    """v2.7.6 Stage-1: R6.2 3g-relaxed obstacle-doom screen as a joint admission condition, applied ONLY to
    eval band-mode quadrotor_3d scenes. No-op (True) for every other system / mode / config, so the training
    path and the canonical / full-range / non-band eval paths are untouched (inertness). Runs AFTER the
    start-goal clearance filter, so the start is already >= clearance outside every obstacle (never the raw
    start-in-obstacle case)."""
    if params.mode != "eval" or scene.system != "quadrotor_3d":
        return True
    q = config.get("env", {}).get("quadrotor_3d", {})
    ic_eval_z = q.get("ic_eval_z")
    if not ic_eval_z or str(ic_eval_z.get("floor_mode", "vz")) != "band":
        return True
    from src.common.quadrotor_ballistic_doom import is_doomed_ballistic
    accel_a = 4.0 * float(config["env"]["bounds"]["quadrotor_3d"]["f_rotor_max"]) / float(q["mass"]) + float(q["gravity"])
    v0 = np.asarray(scene.initial_velocity, dtype=np.float64)
    return not is_doomed_ballistic(scene.start[:2], v0[:2], scene.obstacle_centers,
                                   scene.obstacle_radii, scene.obstacle_active, accel_a)


def _has_start_goal_clearance(scene: Scene, clearance: float) -> bool:
    active_centers = scene.obstacle_centers[scene.obstacle_active]
    active_radii = scene.obstacle_radii[scene.obstacle_active]
    d = active_centers.shape[-1]                                 # obstacle coord dim (xy for cylinders)
    start_clearance = np.min(
        np.linalg.norm(active_centers - scene.start[:d], axis=1) - active_radii
    )
    goal_clearance = np.min(
        np.linalg.norm(active_centers - scene.goal[:d], axis=1) - active_radii
    )
    return bool(start_clearance >= clearance and goal_clearance >= clearance)


def _passes_unavoidable_collision_filter(
    scene: Scene,
    config: Mapping[str, Any],
    params: _SceneModeParams,
) -> bool:
    velocity = _initial_velocity_vector(scene)
    active_centers = scene.obstacle_centers[scene.obstacle_active]
    active_radii = scene.obstacle_radii[scene.obstacle_active]
    start_xy = scene.start[: active_centers.shape[-1]]           # obstacle-plane start (xy for cylinders)

    acceleration_bound = _acceleration_bound(config, scene.system)
    feasibility_margin = float(config["scene_train"]["init_feasibility_margin"])

    for center, radius in zip(active_centers, active_radii):
        center_delta = center - start_xy
        distance = float(np.linalg.norm(center_delta))
        if distance <= _ZERO_SPEED_EPS:
            stopping_distance = 0.0
        else:
            # v2.3.1 fix: inward speed = projected SPEED (m/s) toward the obstacle,
            # i.e. dot(velocity_m/s, unit(center - start)) -- NOT the prior projected
            # distance dot(unit(velocity), center - start). Units now match the
            # v^2/(2 a) max-braking stopping distance (03_train Sec 1.3 intent).
            u_dir = center_delta / distance
            inward_speed = max(0.0, float(np.dot(velocity, u_dir)))
            stopping_distance = inward_speed**2 / (2.0 * acceleration_bound)

        required_clearance = max(
            params.start_goal_clearance,
            stopping_distance + feasibility_margin,
        )
        if distance < float(radius) + required_clearance:
            return False

    return True


def _initial_velocity_vector(scene: Scene) -> Array:
    if scene.system == "double_integrator":
        if scene.initial_velocity is None:
            raise ValueError("Double Integrator scene is missing initial_velocity.")
        return scene.initial_velocity

    if scene.system == "unicycle":
        if scene.initial_speed is None or scene.initial_heading is None:
            raise ValueError(
                "Unicycle scene is missing initial_speed or initial_heading."
            )
        return np.array(
            [
                scene.initial_speed * np.cos(scene.initial_heading),
                scene.initial_speed * np.sin(scene.initial_heading),
            ],
            dtype=np.float64,
        )

    if scene.system == "quadrotor_planar":
        if scene.initial_velocity is None:
            raise ValueError("Quadrotor scene is missing initial_velocity.")
        return scene.initial_velocity

    if scene.system == "quadrotor_3d":
        if scene.initial_velocity is None:
            raise ValueError("Quadrotor-3D scene is missing initial_velocity.")
        # xy velocity only: the collision pre-filter is a horizontal-braking heuristic (cylinders are infinite).
        return np.asarray(scene.initial_velocity, dtype=np.float64)[:2]

    raise ValueError(f"Unsupported system: {scene.system!r}")


def _acceleration_bound(config: Mapping[str, Any], system: str) -> float:
    if system == "double_integrator":
        return float(config["env"]["bounds"]["double_integrator"]["u_max"])
    if system == "unicycle":
        return float(config["env"]["bounds"]["unicycle"]["a_max"])
    if system == "quadrotor_planar":
        # Heuristic init-feasibility pre-filter only. The quadrotor is underactuated (instantaneous
        # accel direction is attitude-fixed), so a scalar bound is approximate; gravity g is the net
        # vertical authority magnitude. IC speeds <= v_init 1.5 make stopping_distance ~ 0.11 m, so
        # this bound barely gates. Recorded as PROTOCOL FOLLOW-UP (approximate underactuated filter).
        return float(config["env"]["quadrotor_planar"]["gravity"])
    if system == "quadrotor_3d":
        # Same approximate underactuated bound as planar: gravity is the net vertical authority magnitude
        # (horizontal braking heuristic only; PROTOCOL FOLLOW-UP shared with planar).
        return float(config["env"]["quadrotor_3d"]["gravity"])
    raise ValueError(f"Unsupported system: {system!r}")


# ---- v2.7.0 iteration-2 (label coverage): cell-targeted IC oversampling for collection ----
def _cell_tilted(scene: "Scene") -> bool:
    """Injection cell: |theta_0| > pi/2 (tilted band). Uses the quadrotor body attitude; None-safe."""
    a = scene.initial_attitude
    if a is None:
        return False
    aw = (float(a) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(aw) > (np.pi / 2.0)


def sample_cell_state_scene(sc, rng, config, max_tries=1000):
    """v2.7.1 corridor-cell STATE injection (changes.md §4). Return a NEW Scene = the fresh scene `sc`
    (obstacles + goal unchanged) with its IC fields overridden to a state sampled inside the failure corridor:
      - position at surface distance d ~ U[0.02, h_scale] from a uniformly chosen ACTIVE obstacle, along a
        uniform direction, REJECTED if it lands inside any active obstacle;
      - attitude |theta| ~ U[pi/2, pi], random sign;
      - speed ||v|| ~ U[0.5, 1.5], direction within +-60 deg of the bearing TOWARD that obstacle;
      - omega from the fresh scene's standard IC.
    Quadrotor-only; contamination-safe (operates on a FRESH sampler scene, never a pool object).
    """
    import dataclasses
    h_scale = float(config["env"]["h_scale"])
    q = config["env"]["quadrotor_planar"]
    centers = np.asarray(sc.obstacle_centers, float)
    radii = np.asarray(sc.obstacle_radii, float)
    active = np.asarray(sc.obstacle_active, bool)
    act_idx = np.nonzero(active)[0]
    if act_idx.size == 0:                                                    # no active obstacle -> standard IC
        return sc
    p = None; bearing = 0.0
    for _ in range(max_tries):
        oi = int(act_idx[int(rng.integers(act_idx.size))])
        c = centers[oi]; r = float(radii[oi])
        d = float(rng.uniform(0.02, h_scale))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        dirv = np.array([np.cos(phi), np.sin(phi)])
        cand = c + (r + d) * dirv
        if np.any(np.linalg.norm(centers[active] - cand, axis=1) - radii[active] < 0.0):
            continue                                                         # inside another obstacle -> reject
        p = cand; bearing = phi + np.pi                                      # bearing TOWARD the obstacle (= -dirv)
        break
    if p is None:
        return sc                                                            # exhausted -> fall back to standard IC
    theta = float(rng.choice(np.array([-1.0, 1.0])) * rng.uniform(np.pi / 2.0, np.pi))
    spd = float(rng.uniform(0.5, 1.5))
    va = bearing + float(rng.uniform(-np.pi / 3.0, np.pi / 3.0))             # within +-60 deg of the bearing
    vel = np.array([spd * np.cos(va), spd * np.sin(va)], float)
    omega = sc.initial_omega if sc.initial_omega is not None else float(
        rng.uniform(-float(q["omega_init_max"]), float(q["omega_init_max"])))
    return dataclasses.replace(sc, start=p.astype(float), initial_velocity=vel,
                               initial_attitude=theta, initial_omega=float(omega))


def sample_scenes(scene_sampler, rng, n_episodes, *, inject_frac=0.0, system_name=None, config=None, max_tries=1000):
    """Sample n_episodes FRESH scenes from scene_sampler(rng). If inject_frac > 0 (quadrotor only), the first
    round(inject_frac*n_episodes) are INJECTED:
      - v2.7.1 (config given): corridor-cell STATE injection — a fresh scene whose IC is overridden to a
        near-surface high-tilt inward-speed state (`sample_cell_state_scene`, changes.md §4);
      - v2.7.0 iteration-2 (config=None): tilted-cell IC rejection to |theta_0| > pi/2 (legacy/refuted lever).
    inject_frac <= 0 (or a non-quadrotor system) is the BIT-PARITY path — rng consumption is identical to
    `[scene_sampler(rng) for _ in range(n_episodes)]`.

    CONTAMINATION BAN (v2.7.0 iteration-2): scene_sampler MUST be the fresh scene-sampler callable. Passing an
    eval pool, a pool manifest, or a list of scenes is a hard error — injected ICs come ONLY from the standard
    scene + IC samplers, never from any eval/diagnostic artifact.
    """
    if (not callable(scene_sampler)) or isinstance(scene_sampler, (list, tuple)) or hasattr(scene_sampler, "scenes"):
        raise TypeError(
            "sample_scenes requires the FRESH scene-sampler callable; a pool / scene-list / manifest is "
            "forbidden on the injection path (v2.7.0 iteration-2 contamination ban)."
        )
    n = int(n_episodes)
    frac = float(inject_frac)
    if frac <= 0.0 or system_name != "quadrotor_planar":
        return [scene_sampler(rng) for _ in range(n)]                       # bit-parity
    n_inject = int(round(frac * n))
    scenes = []
    for i in range(n):
        if i < n_inject:
            if config is not None:                                          # v2.7.1 corridor-cell STATE injection
                scenes.append(sample_cell_state_scene(scene_sampler(rng), rng, config, max_tries=max_tries))
            else:                                                           # v2.7.0 iter-2 tilted-IC rejection
                sc = scene_sampler(rng)
                tries = 0
                while (not _cell_tilted(sc)) and tries < max_tries:
                    sc = scene_sampler(rng)
                    tries += 1
                scenes.append(sc)
        else:
            scenes.append(scene_sampler(rng))
    return scenes
