"""v2.8.4 — the literature-configuration ladder's runner. MEASUREMENT ONLY.

Runs the mission's STRUCTURAL GATE — the obstacle-free straight-line scene of
`src/frameworks/mppi/smoke_cascade_rate.py:66`, closed loop on the FULL RK4 rotor-thrust plant — for a
named cell of `lit_rungs.LitFlags`, and writes `data/runs/v2.8.4/mppi_lit/**`.

The episode loop is `smoke_cascade_rate.check_straight_line_planner:199-202` entry for entry, which is
also what `reach_ladder.py` ran, so the ALL-FLAGS-OFF cell must reproduce the recorded smoke number

    data/runs/v2.8.4/mppi_cascaded/smoke.json  checks.b2_straight_line_planner.N1024.closest_approach_m
        = 1.0560340480879828

to the last digit. That equality is the OFF-state bit-parity check and is MEASURED, never asserted.

TWO REACH COLUMNS on every cell, both labelled and neither substituted for the other:
  * `reach_deployed`  — the DEPLOYED terminal, the conjunction `src.common.outcomes.step_outcomes`
    resolves: `d <= env.goal_radius AND ||v|| <= env.goal_speed_radius AND ||w|| <= env.goal_angrate_radius`,
    all three radii read from the effective config. The shipped predicate is CALLED, not replicated.
  * `reach_pos`       — POSITION ONLY at `env.goal_radius`. It exists so the reference's numbers are
    legible against ours; it is NOT our criterion and never replaces the first column.

K_seed IS ZERO ON EVERY CELL and is read back off the constructed controller's flags for every row.

NOTHING here touches a pool unless `--pool` is passed, and no cell is selected, ranked, promoted or
registered. No number produced by this file may share a table with a rotor-direct, seeded or reference
number without a variant column.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from src._version import __version__
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes
from src.frameworks.mppi import smoke_cascade_rate as smoke
from src.frameworks.mppi.evaluate_mppi import REPO, effective_config, load_mppi_config
from src.frameworks.mppi.lit_rungs import LitFlags, ReferenceCost, build_lit_controller


Tensor = torch.Tensor

OUT_DIR = REPO / "data/runs/v2.8.4/mppi_lit"

# READ-ONLY, recorded before and after. The mission's own ten.
READONLY_DIRS = tuple(
    REPO / f"data/runs/v2.8.4/{name}" for name in (
        "mppi_screen", "mppi_screen_v2", "mppi_screen_v3", "mppi_v3", "mppi_diag",
        "mppi_v4", "mppi_v5", "mppi_cascaded", "mppi_parity", "mppi_reach",
    )
)

RECORDED_SMOKE = REPO / "data/runs/v2.8.4/mppi_cascaded/smoke.json"

# The reference's own wall-time figures, for the compute-asymmetry column (arXiv:2605.02147 Table III,
# Quadrotor row, MPPI column; confirmed against the paper's arXiv HTML).
REFERENCE_MS_PER_ITERATION = 67.6
REFERENCE_MS_PER_CONTROL_STEP = 338.0


def recorded_closest() -> float:
    payload = json.loads(RECORDED_SMOKE.read_text(encoding="utf-8"))
    return float(payload["checks"]["b2_straight_line_planner"]["N1024"]["closest_approach_m"])


def proc_state() -> dict[str, Any]:
    pid = os.getpid()
    state = ""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
                break
    except OSError:                                                          # pragma: no cover
        state = "unavailable"
    return {"pid": pid, "ppid": os.getppid(), "proc_state": state,
            "launched_with": "plain subprocess; NO setsid and NO nohup"}


def dir_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in READONLY_DIRS:
        st = path.stat()
        deep = max((q.stat().st_mtime_ns for q in path.rglob("*")), default=0)
        out[str(path.relative_to(REPO))] = {
            "mtime_ns": st.st_mtime_ns,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "deepest_child_mtime_ns": deep,
            "n_entries": sum(1 for _ in path.rglob("*")),
        }
    return out


def gpu_state() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "note": "cells run on CPU/float64, as the parity target does"}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True, "free_MiB": free // 1024 // 1024,
        "used_MiB": (total - free) // 1024 // 1024, "total_MiB": total // 1024 // 1024,
    }


def write_report(name: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if path.exists():
        path = path.with_name(f"{path.stem}__{time.strftime('%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}", flush=True)
    return path


def band(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    t = torch.as_tensor(values, dtype=torch.float64)
    return {"min": float(t.min()), "p50": float(t.median()), "p95": float(t.quantile(0.95)),
            "max": float(t.max()), "mean": float(t.mean()), "n": int(t.numel())}


# ---- the ladder. Each entry is ONE cell; L3 / L4 / n_samples are construction-time selections of shipped
#      MPPIParams fields, so a cell that sets only those still runs the shipped `act`. -------------------
BETA = 8.054                    # the reference's inverse temperature (brief, Table VII, MPPI column)
REF_H = 60                      # the reference's horizon
REF_N = 500                     # the reference's samples per iteration (Table VII)
REF_ITERS = 5                   # the reference's iterations per control step (Table VII)
REF_N_TABLE_III = 2000          # Table III's note; the discrepancy is REPORTED, not resolved

CELLS: dict[str, dict[str, Any]] = {
    # --- the OFF state, and the check that my rollout override is inert -------------------------------
    "L0_all_flags_off":        {"flags": LitFlags()},
    "L0_parity_track_closest": {"flags": LitFlags(track_closest=True)},
    # --- L1: iterations per control step ---------------------------------------------------------
    "L1_iter2":                {"flags": LitFlags(l1_iterations=2, track_closest=True)},
    "L1_iter3":                {"flags": LitFlags(l1_iterations=3, track_closest=True)},
    "L1_iter5":                {"flags": LitFlags(l1_iterations=REF_ITERS, track_closest=True)},
    "L1_iter8":                {"flags": LitFlags(l1_iterations=8, track_closest=True)},
    # --- L2: cost form ---------------------------------------------------------------------------
    "L2_ref_cost":             {"flags": LitFlags(l2_cost="reference", track_closest=True)},
    "L2_ref_cost_noterm":      {"flags": LitFlags(l2_cost="reference", l2_terminal="none",
                                                  track_closest=True)},
    "L2_settling_off_only":    {"flags": LitFlags(l2_terminal="none", track_closest=True)},
    "L2_ref_cost_hoverthrust": {"flags": LitFlags(l2_cost="reference", l2_thrust_ref="hover_relative",
                                                  track_closest=True)},
    # --- L3: horizon -----------------------------------------------------------------------------
    "L3_horizon60":            {"flags": LitFlags(l3_horizon=REF_H)},
    # --- L4: temperature convention ---------------------------------------------------------------
    "L4_absolute_beta":        {"flags": LitFlags(l4_temperature="absolute", l4_beta=BETA)},
    # --- pairwise isolations, because L1 alone MOVES THE NOMINAL WITHOUT REFINING IT (see the build
    #     log): at ESS p50 ~ 920/1024 each iteration adds a near-unbiased displacement, so the question
    #     is whether L4's sharp temperature turns the same iterations into a refinement. ---------------
    "L1_L4":                   {"flags": LitFlags(l1_iterations=REF_ITERS, l4_temperature="absolute",
                                                  l4_beta=BETA, track_closest=True)},
    "L2_L4":                   {"flags": LitFlags(l2_cost="reference", l2_terminal="none",
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  track_closest=True)},
    # --- the cumulative ladder --------------------------------------------------------------------
    "L1_L2":                   {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", track_closest=True)},
    "L1_L2_L3":                {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  track_closest=True)},
    "L1_L2_L3_L4":             {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  track_closest=True)},
    # THE REFERENCE CONFIGURATION, at their own sample count and with our deployed terminal restored so
    # the GATE is judged under it.
    "REFERENCE_N500":          {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, track_closest=True)},
    "REFERENCE_N500_settling": {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="settling", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, track_closest=True)},
    "REFERENCE_N2000_tableIII": {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                   l2_terminal="none", l3_horizon=REF_H,
                                                   l4_temperature="absolute", l4_beta=BETA,
                                                   n_samples=REF_N_TABLE_III, track_closest=True)},
    # --- L2 MECHANISM PROBE. L1_L2 ends "stuck" at speed p50 0.404 m/s. The reference cost's thrust
    #     term is 0.366 of the sample cost there, and transcribed literally it charges w_thrust * T^2 at
    #     T ~ m*g (both read off the system object), i.e. it charges for EXISTING. This cell re-runs the
    #     same configuration with the thrust measured from hover instead, which is an INTERPRETATION of
    #     the reference cost and is labelled as one wherever it appears.
    "L1_L2_hoverthrust":       {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l2_thrust_ref="hover_relative",
                                                  track_closest=True)},
    # --- L5: proposal quality (only if L1-L4 do not reach) -----------------------------------------
    "L5_colored2":             {"flags": LitFlags(l5_noise="colored", l5_colored_exponent=2.0,
                                                  track_closest=True)},
    "L5_colored4":             {"flags": LitFlags(l5_noise="colored", l5_colored_exponent=4.0,
                                                  track_closest=True)},
    "L1_L2_L3_L4_L5":          {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, l5_noise="colored",
                                                  l5_colored_exponent=2.0, track_closest=True)},
    # L5 crosses the POSITION radius (0.0836 m) but not the DEPLOYED conjunction, which needs
    # ||v|| <= 0.3 and ||w|| <= 0.3 at the same step and which the reference running cost has no term
    # for. L2's own settling sub-flag is the thing the mission asks to measure both ways, so the same
    # configuration is run with our deployed settling terminal restored. No new rung is invented.
    "L5_REFERENCE_settling":   {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="settling", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, l5_noise="colored",
                                                  l5_colored_exponent=2.0, track_closest=True)},
    "L5_REFERENCE_c4":         {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, l5_noise="colored",
                                                  l5_colored_exponent=4.0, track_closest=True)},
    "L5_REFERENCE_c4_settling": {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                   l2_terminal="settling", l3_horizon=REF_H,
                                                   l4_temperature="absolute", l4_beta=BETA,
                                                   n_samples=REF_N, l5_noise="colored",
                                                   l5_colored_exponent=4.0, track_closest=True)},
    "L5_REFERENCE_N1024":      {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="settling", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=1024, l5_noise="colored",
                                                  l5_colored_exponent=2.0, track_closest=True)},
    # --- the (T, tau) VARIANT — a SEPARATE column, never the same row as cascaded_rate --------------
    "VAR_torque_off":          {"flags": LitFlags(track_closest=True), "variant": "thrust_torque"},
    "VAR_torque_reference":    {"flags": LitFlags(l1_iterations=REF_ITERS, l2_cost="reference",
                                                  l2_terminal="none", l3_horizon=REF_H,
                                                  l4_temperature="absolute", l4_beta=BETA,
                                                  n_samples=REF_N, track_closest=True),
                                "variant": "thrust_torque"},
}


@torch.no_grad()
def run_cell(
    name: str, spec: dict[str, Any], mppi_config: dict[str, Any], config: dict[str, Any],
    *, offset: list[float] | None = None, reference_cost: ReferenceCost | None = None,
) -> dict[str, Any]:
    """One cell on ONE obstacle-free straight-line scene — THE GATE."""
    device, dtype = torch.device("cpu"), torch.float64
    flags: LitFlags = spec["flags"]
    variant = str(spec.get("variant", "cascaded_rate"))
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    seed = int(mppi_config["cascaded"]["scale"]["seed"])
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])

    system, controller = build_lit_controller(
        mppi_config, config, flags=flags, reference_cost=reference_cost, variant=variant,
        device=device, dtype=dtype,
    )
    offset = [float(v) for v in (smoke_cfg["straight_line"]["offset"] if offset is None else offset)]
    scene = smoke.straight_line_scene(system, offset, dtype)
    batched = batch_scenes([scene], device=device, dtype=dtype)
    x = smoke.level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    seconds = float(smoke_cfg["straight_line"]["seconds"])
    n_steps = int(round(seconds / dt))

    controller.reset(1)
    controller.generator.manual_seed(seed)

    t0 = time.time()
    dist, ess, lam, speed, angrate, abs_z = [], [], [], [], [], []
    states = []
    for _ in range(n_steps):
        u = controller.act(x, batched)
        x = rk4_step(system, x, u, dt)
        states.append(x.clone())
        dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
        ess.append(float(controller.last_ess[0].item()))
        lam.append(float(controller.last_lam_eff[0].item()))
        speed.append(float(system.speed(x).item()))
        angrate.append(float(system.angular_rate(x).item()))
        abs_z.append(float(system.position(x)[0, 2].abs().item()))
    wall = time.time() - t0

    # ---- THE TWO REACH COLUMNS. The first CALLS the deployed predicate; the second is position-only.
    traj = torch.stack(states, dim=0)                                          # [T, 1, state_dim]
    masks = step_outcomes(traj, scene, system, config)
    outcome = resolve_outcome(masks)
    reach_deployed = bool(masks.goal_reached.any().item())
    reach_pos = bool(min(dist) <= goal_radius)
    closest = min(dist)
    from src.frameworks.mppi.recovery import tilt_cos
    final_tilt = float(torch.arccos(torch.clamp(tilt_cos(system, x), -1.0, 1.0)).item())

    def diag(rows: list[list[float]]) -> list[dict[str, float] | None]:
        return [band(row) for row in rows]

    record = {
        "cell": name,
        "variant": variant,
        "offset_m": offset,
        "is_the_gate_scene": offset == [float(v) for v in smoke_cfg["straight_line"]["offset"]],
        "flags": flags.record(),
        "K_seed_read_back": controller.flags.record()["K_seed"],
        "lit": controller.lit_record(),
        "N_samples": int(controller.params.n_samples),
        "horizon": int(controller.params.horizon),
        "lam_mode": str(controller.params.lam_mode),
        "lam": float(controller.params.lam),
        "iterations_per_control_step": int(flags.l1_iterations),
        "dt": controller.dt,
        "n_control_steps": n_steps,
        "seconds": seconds,
        "goal_radius_read_from_config": goal_radius,
        "goal_speed_radius_read_from_config": float(config["env"]["goal_speed_radius"]),
        "goal_angrate_radius_read_from_config": float(config["env"]["goal_angrate_radius"]),
        # --- the two labelled reach columns -------------------------------------------------------
        "reach_deployed_terminal": reach_deployed,
        "reach_deployed_terminal_definition":
            "src.common.outcomes.step_outcomes: d <= env.goal_radius AND ||v|| <= env.goal_speed_radius "
            "AND ||w|| <= env.goal_angrate_radius, all three read from the effective config. OUR criterion.",
        "reach_pos_only": reach_pos,
        "reach_pos_only_definition":
            "POSITION ONLY at env.goal_radius. Present so the reference's numbers are legible against "
            "ours; NOT our criterion and never a substitute for the column above.",
        "outcome": outcome.outcome[0],
        "outcome_event_step": int(outcome.event_step[0].item()),
        "collision_cause": outcome.collision_cause[0] if outcome.collision_cause else "",
        "closest_approach_m": closest,
        "final_distance_m": dist[-1],
        "final_speed_m_s": speed[-1],
        "final_angular_rate_rad_s": angrate[-1],
        # the tilt from level, through the shipped `recovery.tilt_cos`. L6's objective needs an
        # "orientation error" and our task carries no attitude reference; this is the nearest quantity the
        # system object defines and it is labelled as an interpretation wherever it is used.
        "final_tilt_rad": final_tilt,
        # --- wall time, against the reference's own figures ----------------------------------------
        "wall_s": round(wall, 3),
        "ms_per_control_step": round(1000.0 * wall / n_steps, 3),
        "ms_per_iteration": round(1000.0 * wall / n_steps / max(1, int(flags.l1_iterations)), 3),
        "reference_ms_per_iteration": REFERENCE_MS_PER_ITERATION,
        "reference_ms_per_control_step": REFERENCE_MS_PER_CONTROL_STEP,
        "wall_time_note": "single-scene CPU/float64, torch threads as launched; the reference's figures "
                          "are JAX-vectorised GPU. The comparison is ON RECORD, not like-for-like.",
        # --- bands ---------------------------------------------------------------------------------
        "ess": band(ess), "lam_eff": band(lam), "speed_m_s": band(speed),
        "angular_rate_rad_s": band(angrate), "abs_z_m": band(abs_z),
        "degenerate_steps": int(controller.degenerate_steps[0].item()),
        # --- L1's per-iteration diagnostics --------------------------------------------------------
        "per_iteration": {
            "what": "index i is iteration i of the control step, banded over the episode's control steps",
            "nominal_increment_norm": diag(controller.iter_increment),
            "best_sample_closest_approach_m": diag(controller.iter_best_closest),
            "min_over_samples_closest_approach_m": diag(controller.iter_min_closest),
            "ess": diag(controller.iter_ess),
            "lam_eff": diag(controller.iter_lam),
            "min_sample_cost": diag(controller.iter_cost_min),
        },
        "cost_term_totals": dict(controller.term_totals) if controller.term_totals else None,
        "preclip_over_this_leg": controller.preclip_record(),
        # THE DEPLOYED CONJUNCTION, step by step: how close the three legs came to closing together.
        # `step_outcomes`' own `goal_reached` is the conjunction; these are its three arguments.
        "deployed_conjunction_margin": {
            "what": "over the steps where the POSITION leg closes (d <= goal_radius), the speed and "
                    "angular rate at those steps. The deployed terminal needs all three simultaneously.",
            "n_steps_position_leg_closes": int(sum(1 for d in dist if d <= goal_radius)),
            "speed_at_those_steps": band([s for d, s in zip(dist, speed) if d <= goal_radius]),
            "angrate_at_those_steps": band([w for d, w in zip(dist, angrate) if d <= goal_radius]),
        },
        "per_step_distance_m": dist,
        "per_step_speed_m_s": speed,
        "per_step_angular_rate_rad_s": angrate,
    }
    return record


# =================================================================================================
# THE SCENE-DENSITY CONVERSION — our pool expressed in the reference's units
# =================================================================================================
# The reference's three settings, as the brief states them (arXiv:2605.02147 Appendix C-C; the
# Easy/Medium/Hard descriptions were independently confirmed against the paper's arXiv HTML).
REFERENCE_SETTINGS: dict[str, dict[str, Any]] = {
    "Easy":   {"n": 50,  "r_min": 0.3, "r_max": 0.6, "spread_x": 10.0, "spread_y": 5.0,
               "height_lo": 0.3, "height_hi": 5.0, "mppi_success": 1.00},
    "Medium": {"n": 100, "r_min": 0.3, "r_max": 0.8, "spread_x": 6.0,  "spread_y": 3.0,
               "height_lo": 0.3, "height_hi": 4.0, "mppi_success": 0.60},
    "Hard":   {"n": 100, "r_min": 0.4, "r_max": 0.9, "spread_x": 3.0,  "spread_y": 2.0,
               "height_lo": 0.5, "height_hi": 3.5, "mppi_success": 0.19},
}

POOL_OF_RECORD = "data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl"


def density_conversion(config: dict[str, Any], *, n_scenes: int = 400) -> dict[str, Any]:
    """Express the pool of record's first `n_scenes` scenes in the reference's units, beside its three
    settings. READ-ONLY on the pool. Every arena constant is read from the effective env config."""
    import numpy as np
    from src.eval.build_pools import load_pool

    env = config["env"]
    world = float(env["world_lim"])                     # read, never typed
    band = float(env["band_collision_limit"])           # read, never typed
    area = (2.0 * world) ** 2
    pool = load_pool(REPO / POOL_OF_RECORD)
    scenes = pool.scenes[:n_scenes]
    n_active = np.array([int(s.obstacle_active.sum()) for s in scenes])
    radii = np.concatenate([s.obstacle_radii[s.obstacle_active] for s in scenes])
    footprint = np.array([float(np.pi * (s.obstacle_radii[s.obstacle_active] ** 2).sum()) for s in scenes])
    starts = np.array([s.start for s in scenes])
    goals = np.array([s.goal for s in scenes])
    traverse = np.linalg.norm(starts - goals, axis=1)

    theirs = {}
    for name, spec in REFERENCE_SETTINGS.items():
        spread = spec["spread_x"] * spec["spread_y"]
        height = spec["height_hi"] - spec["height_lo"]
        r_mean = 0.5 * (spec["r_min"] + spec["r_max"])
        theirs[name] = {
            "obstacles_per_scene": spec["n"],
            "radii_m": [spec["r_min"], spec["r_max"]],
            "spread_m": [spec["spread_x"], spec["spread_y"]],
            "spread_area_m2": spread,
            "height_band_m": [spec["height_lo"], spec["height_hi"]],
            "height_band_thickness_m": height,
            "obstacles_per_m2": spec["n"] / spread,
            "obstacles_intersecting_an_altitude_slice_per_m2": spec["n"] * (2.0 * r_mean) / height / spread,
            "sphere_volume_fraction_if_spread_is_a_hard_box":
                spec["n"] * (4.0 / 3.0) * float(np.pi) * r_mean ** 3 / (spread * height),
            "mppi_success_rate_reported": spec["mppi_success"],
        }

    return {
        "what": "the pool of record expressed in the reference's units, beside its Easy / Medium / Hard",
        "theirs": theirs,
        "ours": {
            "pool": POOL_OF_RECORD,
            "pool_sha256_8": "3682a4e3",
            "n_scenes_measured": len(scenes),
            "which_scenes": f"the FIRST {len(scenes)} of the pool of record, read-only",
            "arena_m": [2.0 * world, 2.0 * world],
            "arena_area_m2": area,
            "arena_source": "env.world_lim, read from the effective config",
            "height_band_m": [-band, band],
            "height_band_thickness_m": 2.0 * band,
            "height_band_source": "env.band_collision_limit, read from the effective config",
            "obstacles_per_scene": {"min": int(n_active.min()), "p50": float(np.median(n_active)),
                                    "mean": float(n_active.mean()), "max": int(n_active.max())},
            "radii_m": {"min": float(radii.min()), "mean": float(radii.mean()), "max": float(radii.max())},
            "obstacles_per_m2": {"mean": float(n_active.mean() / area),
                                 "p50": float(np.median(n_active) / area),
                                 "max": float(n_active.max() / area)},
            "obstacles_intersecting_an_altitude_slice_per_m2": {
                "mean": float(n_active.mean() / area),
                "note": "IDENTICAL to obstacles_per_m2 — our obstacles are infinite vertical cylinders "
                        "spanning the whole band, so every one of them intersects every altitude",
            },
            "planar_footprint_coverage": {"mean": float((footprint / area).mean()),
                                          "max": float((footprint / area).max())},
            "traverse_start_to_goal_m": {"min": float(traverse.min()), "p50": float(np.median(traverse)),
                                         "mean": float(traverse.mean()), "max": float(traverse.max())},
        },
        "limits_of_the_conversion": [
            "CYLINDERS vs SPHERES. Ours are infinite vertical cylinders spanning the whole band, so every "
            "obstacle blocks at every altitude and there is no fly-over degree of freedom. Theirs are "
            "finite bodies inside a height band, so the per-altitude-slice row is the fairer comparison; "
            "on it our mean 0.128/m^2 sits below their Easy's 0.191/m^2 by a factor of 1.5, not the "
            "factor of 7.8 the raw obstacles-per-m^2 row suggests.",
            "ARENA vs CORRIDOR. Their spread is the region obstacles are placed in on a traverse of about "
            "that length; ours is a closed arena with start and goal drawn anywhere inside it, so the "
            "traverse is short relative to the arena and a route around the field is often available.",
            "THE SPREAD PARAMETER IS NOT STATED TO BE A HARD BOX. Taking it as one makes their Medium and "
            "Hard sphere volume fractions exceed 1 (1.05 and 6.39), i.e. the obstacles would have to "
            "overlap heavily. If the spread is a Gaussian scale rather than a box, their effective areal "
            "densities are LOWER than the obstacles-per-m^2 row states. The row is computed the only way "
            "the stated numbers allow and the ambiguity is left visible, not resolved.",
            "RADII. Our upper radius (0.800 m) matches their Medium; our lower (0.150 m) is below all "
            "three of their ranges, so our field is more heterogeneous in size than any of theirs.",
        ],
    }


# =================================================================================================
# L6 — WEIGHT TUNING. LAST, and only because L1-L5 did not reach.
# =================================================================================================
# The reference's six weights were Optuna-tuned on ITS scenes, so transplanting them is not guaranteed to
# fit ours. This is a small random search over multipliers on those six weights, in the reference's own
# objective form.
#
# THE HELD-OUT SET. The search runs on CONSTRUCTED obstacle-free straight-line scenes at offsets that are
# DISJOINT from the gate offset. No number reported anywhere in this work is measured on them, and the
# selected point is then evaluated on the GATE offset, which the search never saw. No pool scene is
# touched by the search at all.
L6_HELDOUT_OFFSETS = [[-2.5, 1.5, 0.0], [0.0, 0.0, 2.5], [1.2, 0.8, 0.5]]
L6_BASE_CELL = "L1_L2_L3_L4_L5"          # the best-performing configuration L1-L5 produced
L6_LOG10_RANGE = 1.0                     # multipliers log-uniform in [10^-1, 10^+1] on each weight
L6_DRAWS = 16                            # plus draw 0, which is the untouched reference point


def l6_objective(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The reference's own objective form: `5 * mean position error + 2 * mean orientation error
    - 10 * success rate`, lower is better.

    INTERPRETED, and labelled as such wherever it appears:
      * position error  := the final distance to the goal, averaged over the held-out scenes;
      * orientation error := the final TILT in radians (the angle between the body up-axis and world up,
        through `recovery.tilt_cos`). Our task carries NO attitude reference — it is yaw-blind and has no
        desired attitude — so the reference's "orientation error" has no exact counterpart. Tilt from
        level is the nearest quantity our system object defines and it is what is used;
      * success rate := the DEPLOYED terminal latch, `step_outcomes`, on the held-out scenes.
    """
    import numpy as np
    pos = float(np.mean([r["final_distance_m"] for r in rows]))
    ori = float(np.mean([r["final_tilt_rad"] for r in rows]))
    success = float(np.mean([1.0 if r["reach_deployed_terminal"] else 0.0 for r in rows]))
    return {"position_error_m": pos, "orientation_error_rad": ori, "success_rate": success,
            "objective": 5.0 * pos + 2.0 * ori - 10.0 * success}


def run_l6(mppi_config: dict[str, Any], config: dict[str, Any], *, draws: int) -> dict[str, Any]:
    import numpy as np
    base_flags: LitFlags = CELLS[L6_BASE_CELL]["flags"]
    rng = np.random.default_rng(int(mppi_config["cascaded"]["scale"]["seed"]))
    reference = ReferenceCost()
    names = ["w_goal", "w_obstacle", "w_thrust", "w_torque", "w_velocity", "w_height"]

    trials: list[dict[str, Any]] = []
    for draw in range(draws + 1):
        if draw == 0:
            multipliers = {n: 1.0 for n in names}
        else:
            multipliers = {
                n: float(10.0 ** rng.uniform(-L6_LOG10_RANGE, L6_LOG10_RANGE)) for n in names
            }
        candidate = ReferenceCost(**{n: getattr(reference, n) * multipliers[n] for n in names})
        rows = []
        for offset in L6_HELDOUT_OFFSETS:
            row = run_cell(
                f"L6_draw{draw}", {"flags": base_flags}, mppi_config, config,
                offset=offset, reference_cost=candidate,
            )
            rows.append(row)
        score = l6_objective(rows)
        trials.append({
            "draw": draw, "multipliers": multipliers, "weights": candidate.record(),
            **score,
            "per_offset": [
                {"offset_m": r["offset_m"], "closest_approach_m": r["closest_approach_m"],
                 "final_distance_m": r["final_distance_m"], "final_tilt_rad": r["final_tilt_rad"],
                 "reach_deployed_terminal": r["reach_deployed_terminal"],
                 "reach_pos_only": r["reach_pos_only"], "outcome": r["outcome"]}
                for r in rows
            ],
        })
        print(f"[l6] draw {draw}: objective {score['objective']:.4f} "
              f"(pos {score['position_error_m']:.3f}, ori {score['orientation_error_rad']:.3f}, "
              f"success {score['success_rate']:.2f})", flush=True)

    best = min(trials, key=lambda t: t["objective"])
    selected = ReferenceCost(**{n: reference.__getattribute__(n) * best["multipliers"][n] for n in names})
    # THE REPORTED NUMBER: the selected point on the GATE offset, which the search never saw.
    gate = run_cell(
        "L6_selected_on_gate", {"flags": base_flags}, mppi_config, config,
        offset=None, reference_cost=selected,
    )
    return {
        "what": "L6 — a small random search over the reference's six cost weights, on a HELD-OUT set of "
                "constructed obstacle-free straight-line scenes, in the reference's own objective form",
        "why_last": "L1-L5 did not reach the deployed terminal; the reference's weights were Optuna-tuned "
                    "on its scenes and transplanting them is not guaranteed to fit ours",
        "base_configuration": L6_BASE_CELL,
        "base_flags": base_flags.record(),
        "search_space": {
            "parameters": names,
            "form": "an independent multiplier on each of the six reference weights",
            "distribution": f"log-uniform in [10^-{L6_LOG10_RANGE:g}, 10^+{L6_LOG10_RANGE:g}]",
            "seed": int(mppi_config["cascaded"]["scale"]["seed"]),
        },
        "budget": {"draws": draws, "plus_the_untouched_reference_point": True,
                   "held_out_scenes_per_draw": len(L6_HELDOUT_OFFSETS),
                   "episodes_total": (draws + 1) * len(L6_HELDOUT_OFFSETS) + 1},
        "held_out_offsets_m": L6_HELDOUT_OFFSETS,
        "held_out_note": "CONSTRUCTED obstacle-free straight-line scenes, DISJOINT from the gate offset. "
                         "No reported number in this work is measured on them and no pool scene is "
                         "touched by the search.",
        "objective": "5 * mean position error + 2 * mean orientation error - 10 * success rate, lower is "
                     "better. INTERPRETED: position error = final distance to goal; orientation error = "
                     "final tilt in radians (our task has NO attitude reference, so the reference's term "
                     "has no exact counterpart); success = the DEPLOYED terminal latch.",
        "trials": trials,
        "selected_point": {"draw": best["draw"], "multipliers": best["multipliers"],
                           "weights": selected.record(), "held_out_objective": best["objective"]},
        "selected_point_on_the_gate_scene": gate,
        "no_selection_registered": "this is a SEARCH RESULT, not a cell selection. Nothing is promoted, "
                                   "nothing is registered, no ledger row is written.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l6", action="store_true", help="run L6's weight search (held-out scenes only)")
    parser.add_argument("--l6-draws", type=int, default=L6_DRAWS)
    parser.add_argument("--density", action="store_true",
                        help="write the scene-density conversion and exit. READ-ONLY on the pool.")
    parser.add_argument("--cells", type=str, default="L0_all_flags_off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument("--offsets", type=str, default="")
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    if args.density:
        before = dir_state()
        report = density_conversion(config)
        report["version"] = __version__
        report["process"] = proc_state()
        report["readonly_before"] = before
        report["readonly_after"] = dir_state()
        report["readonly_unchanged"] = report["readonly_before"] == report["readonly_after"]
        write_report("density_conversion.json", report)
        return 0
    if args.l6:
        before = dir_state()
        report = run_l6(mppi_config, config, draws=int(args.l6_draws))
        report["version"] = __version__
        report["process"] = proc_state()
        report["device"] = "cpu"
        report["dtype"] = "torch.float64"
        report["readonly_before"] = before
        report["readonly_after"] = dir_state()
        report["readonly_unchanged"] = report["readonly_before"] == report["readonly_after"]
        report["wall_s"] = round(time.time() - t0, 2)
        write_report("l6_search.json", report)
        return 0
    names = [n.strip() for n in args.cells.split(",") if n.strip()]
    unknown = [n for n in names if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; known: {sorted(CELLS)}")

    before = dir_state()
    target = recorded_closest()
    results: list[dict[str, Any]] = []
    parity: dict[str, bool | None] = {}
    offsets: list[list[float] | None] = [None]
    if args.offsets:
        offsets = [[float(v) for v in chunk.split(",")] for chunk in args.offsets.split(";")]

    for name in names:
        for offset in offsets:
            print(f"[lit] {name} offset={offset} ...", flush=True)
            row = run_cell(name, CELLS[name], mppi_config, config, offset=offset)
            if name.startswith("L0_") and offset is None:
                row["off_state_bit_parity"] = {
                    "target_source": "data/runs/v2.8.4/mppi_cascaded/smoke.json "
                                     "checks.b2_straight_line_planner.N1024.closest_approach_m",
                    "target_closest_approach_m": target,
                    "measured_closest_approach_m": row["closest_approach_m"],
                    "identical": row["closest_approach_m"] == target,
                    "abs_difference": abs(row["closest_approach_m"] - target),
                }
                parity[name] = row["off_state_bit_parity"]["identical"]
            print(
                f"[lit] {name} {row['offset_m']}: closest {row['closest_approach_m']:.6f} m  "
                f"reach_deployed {row['reach_deployed_terminal']}  reach_pos {row['reach_pos_only']}  "
                f"outcome {row['outcome']}  ({row['wall_s']} s, "
                f"{row['ms_per_control_step']:.1f} ms/control step)",
                flush=True,
            )
            results.append(row)

    report = {
        "what": "v2.8.4 literature-configuration ladder — the obstacle-free straight-line STRUCTURAL "
                "GATE, one row per cell",
        "version": __version__,
        "subject": "src/frameworks/mppi/cascade_rate.py, SUBCLASSED and not edited "
                   "(src/frameworks/mppi/lit_rungs.py). cascade.py is not the subject.",
        "gate": "nothing advances to the obstacle pool until this scene is REACHED under the DEPLOYED "
                "terminal (src.common.outcomes.step_outcomes, radii read from the config)",
        "K_seed": 0,
        "K_seed_note": "structurally zero — lit_rungs.py contains no seeded / geometric rollout code and "
                       "does not import reach_rungs.py",
        "process": proc_state(),
        "gpu": gpu_state(),
        "device": "cpu", "dtype": "torch.float64", "torch_threads": int(args.threads),
        "reference_configuration": {
            "source": "arXiv:2605.02147 Appendix C-C / Table VII (MPPI column), transcribed from the "
                      "mission brief",
            "horizon": REF_H, "iterations": REF_ITERS, "samples_per_iteration": REF_N,
            "inverse_temperature_beta": BETA, "lambda_absolute": 1.0 / BETA,
            "cost_weights": ReferenceCost().record(),
            "table_III_vs_table_VII_discrepancy":
                "Table III's caption states the quadrotor MPPI uses 2000 proposals; Table VII states 500 "
                "per iteration. Table VII (the experiment-details table) is PRIMARY here, per the brief. "
                "The 2000-proposal figure is run as its own labelled cell, REFERENCE_N2000_tableIII. The "
                "discrepancy is recorded, not resolved.",
        },
        "readonly_before": before,
        "no_selection": "MEASUREMENT ONLY. No pool unless --pool, no cell selection, no ledger row, no "
                        "promotion, no final score. Cascaded, rotor-direct, seeded, (T,tau) and reference "
                        "numbers never share a table without a variant column.",
        "rows": results,
        "parity": parity,
        "any_cell_reaches_deployed": any(r.get("reach_deployed_terminal") for r in results),
        "any_cell_reaches_pos": any(r.get("reach_pos_only") for r in results),
    }
    report["readonly_after"] = dir_state()
    report["readonly_unchanged"] = report["readonly_before"] == report["readonly_after"]
    report["wall_s"] = round(time.time() - t0, 2)
    suffix = f"__{args.tag}" if args.tag else ""
    write_report(f"ladder{suffix}.json", report)
    print(f"readonly unchanged: {report['readonly_unchanged']}", flush=True)
    if args.require_parity and not all(v is True for v in parity.values()):
        print("OFF-STATE PARITY FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":                                                   # pragma: no cover
    raise SystemExit(main())
