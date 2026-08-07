"""v2.8.2 PPO revision-10 training-IC CURRICULUM (Researcher B3).

Reach was never sampled (Part A: 27 goal episodes across all quadrotor training; no IC satisfies the reach
predicate at t=0 — start-goal distance >= 1.02 m vs 0.15 m, ||v0|| <= 1.5 vs 0.30, ||omega0|| <= 5.0 vs 0.30),
and model-free RL cannot learn a terminal it never experiences. This mixes a fraction of EASY starts (near the
goal, low speed, low angular rate, near-upright) into the TRAINING sampler so the +30 terminal fires and enters
the value function, then anneals that fraction to zero so the policy finishes on the true distribution.

DELIBERATE ASYMMETRY (recorded in the build-log, not hidden): the JT policy loss receives analytic settling
gradient through BPTT from w_settle / w_settle_ang and therefore never needs to sample a reach; PPO (model-free)
must experience one. That asymmetry is a finding about what the differentiable-dynamics approach buys.

Gate-2 SAFE: only the PPO TRAINING sampler changes. The EVALUATION IC distribution and the scored pools are
untouched, and src/envs/scene_init.py (read by the JT path) is NOT modified — easy scenes are built here by
overriding the IC fields of a normally-sampled Scene.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from src.envs.scene_init import Scene, sample_train_scene


def easy_frac_at(iteration: int, n_iterations: int, f0: float, anneal_frac: float) -> float:
    """Linear anneal from f0 (iter 0) to 0 at iteration = anneal_frac * n_iterations, then 0 (true distribution)."""
    if f0 <= 0.0 or anneal_frac <= 0.0:
        return 0.0
    t = iteration / max(1, int(anneal_frac * n_iterations))
    return float(max(0.0, f0 * (1.0 - t)))


def _small_tilt_quat(rng: np.random.Generator, max_tilt: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-9)
    angle = float(rng.uniform(0.0, max_tilt))
    w = np.cos(angle / 2.0)
    xyz = np.sin(angle / 2.0) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


def _clear_of_obstacles(start_xy: np.ndarray, scene: Scene, margin: float) -> bool:
    centers = np.asarray(scene.obstacle_centers, dtype=np.float64)
    radii = np.asarray(scene.obstacle_radii, dtype=np.float64)
    active = np.asarray(scene.obstacle_active, dtype=bool)
    if centers.size == 0:
        return True
    d = np.linalg.norm(centers[:, :2] - start_xy[:2], axis=-1)
    return bool(np.all(~active | (d > radii + margin)))


def make_easy_scene(base: Scene, rng: np.random.Generator, config: Mapping[str, Any], system_name: str,
                    easy: Mapping[str, float]) -> Scene:
    """Return an easy near-goal / low-velocity / low-omega / near-upright IC on the base scene's geometry, or the
    base scene unchanged if a clear near-goal start cannot be placed (rare)."""
    goal = np.asarray(base.goal, dtype=np.float64)
    world_lim = float(config["env"]["world_lim"])
    band = float(config["env"].get("band_collision_limit", 0.0)) or world_lim
    d_lo, d_hi = float(easy["d_lo"]), float(easy["d_hi"])
    v_max, om_max, max_tilt = float(easy["v_max"]), float(easy["om_max"]), float(easy["max_tilt"])
    dim = goal.shape[0]
    start = None
    for _ in range(8):
        direction = rng.normal(size=dim)
        direction /= (np.linalg.norm(direction) + 1e-9)
        cand = goal + float(rng.uniform(d_lo, d_hi)) * direction
        cand[:2] = np.clip(cand[:2], -world_lim + 0.3, world_lim - 0.3)
        if dim >= 3:
            cand[2] = np.clip(cand[2], -(band - 0.5), band - 0.5)
        if _clear_of_obstacles(cand, base, margin=0.10):
            start = cand
            break
    if start is None:
        return base
    if system_name == "quadrotor_3d":
        vdir = rng.normal(size=3); vdir /= (np.linalg.norm(vdir) + 1e-9)
        v0 = (float(rng.uniform(0.0, v_max)) * vdir).astype(np.float64)
        omdir = rng.normal(size=3); omdir /= (np.linalg.norm(omdir) + 1e-9)
        om0 = (float(rng.uniform(0.0, om_max)) * omdir).astype(np.float64)
        return replace(base, start=start.astype(np.float64), initial_velocity=v0,
                       initial_attitude_quat=_small_tilt_quat(rng, max_tilt), initial_omega_vec=om0)
    if system_name == "double_integrator":
        v0 = rng.normal(size=2)
        v0 *= (float(rng.uniform(0.0, v_max)) / (np.linalg.norm(v0) + 1e-9))
        return replace(base, start=start[:2].astype(np.float64), initial_velocity=v0.astype(np.float64))
    return base  # other systems: no curriculum (unused here)


def sample_curriculum_scenes(rng: np.random.Generator, config: Mapping[str, Any], system_name: str, n: int,
                             easy_frac: float, easy: Mapping[str, float]) -> list[Scene]:
    """Sample n training scenes; replace a random easy_frac of them with easy near-goal ICs."""
    scenes = [sample_train_scene(rng, config, system_name) for _ in range(int(n))]
    if easy_frac > 0.0:
        k = int(round(easy_frac * n))
        if k > 0:
            idx = rng.choice(n, size=min(k, n), replace=False)
            for i in idx:
                scenes[i] = make_easy_scene(scenes[i], rng, config, system_name, easy)
    return scenes
