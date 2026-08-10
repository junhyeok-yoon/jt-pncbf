"""v2.8.4 MPPI goal-attraction redesign (charter "v4", G1-G4) — the smoke.

Two stages, written to `data/runs/v2.8.4/mppi_v4/`:

  --stage cpu    (NO GPU work, seconds) -> smoke_goal_v4.json
      a. g1_progress  — the G1 term is POSITIVE progress / NEGATIVE cost on a constructed APPROACHING
                        trajectory and NEGATIVE progress / POSITIVE cost on a RECEDING one, and exactly
                        zero on a frozen (repeated) state. Both signs are reported explicitly, because
                        "the progress term is positive when approaching" refers to the approach
                        increment while the COST contribution is its negation.
      b. g3_linear    — the G3 terminal is linear in distance: halving the distance halves the value, to
                        floating-point tolerance, at several distances and against the exact expression
                        w_terminal * d with w_terminal READ from config.
      c. g4_zero      — the G4 term is exactly zero at omega = 0 and strictly positive off it, and it is
                        the only difference between the G4-on and G4-off stage costs on the same states.
      d. ou_unchanged — the OU perturbation statistics are UNCHANGED from v3: the shipped v3 check
                        `src.frameworks.mppi.cpu_smoke.check_ou_statistics` is re-run verbatim (same
                        coefficient, same test) and compared field for field against the recorded
                        `data/runs/v2.8.4/mppi_screen_v3/cpu_smoke_v3.json:g_ou_statistics`.
      e. defaults_off — with all four switches off the stage and terminal costs equal the pre-v4
                        expressions recomputed independently, to 0 ULP, on a batch of real pool states.

  --stage repro  (GPU) -> repro_v3.json
      f. v3_repro     — the BRIEF-FIXED reference cell `N1024_lam0.05_C100000_m1_H40_sig1` re-run with
                        all four v4 switches OFF and compared METRIC FOR METRIC against the retained
                        artifact `data/runs/v2.8.4/mppi_v3/cell__N1024_lam0.05_C100000_m1_H40_sig1.json`.
                        A mismatch is a STOP. This cell is a REPRODUCTION of an already-recorded v3 row,
                        not one of the redesign's scored cells.

The reference cell is fixed by the task brief and is NOT a Researcher cell selection: no selection has
ever been registered for any MPPI screen. Every coefficient — the cell coordinates, w_p, w_omega,
w_terminal, theta_ref, the radii, dt — is READ from `src/frameworks/mppi/config.yaml` or from the
effective shipped config; none is typed here.

Run:  CUDA_VISIBLE_DEVICES="" python -m src.frameworks.mppi.smoke_goal_v4 --stage cpu
      python -m src.frameworks.mppi.smoke_goal_v4 --stage repro
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.frameworks.mppi.cost import CostParams, progress_cost, stage_cost, terminal_cost
from src.frameworks.mppi.cpu_smoke import check_ou_statistics
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
    run_cell,
)
from src.frameworks.mppi.screen_recovery import spawn_tilt_deg


OUT_DIR = REPO / "data/runs/v2.8.4/mppi_v4"
V3_SMOKE = REPO / "data/runs/v2.8.4/mppi_screen_v3/cpu_smoke_v3.json"
V3_DIR = REPO / "data/runs/v2.8.4/mppi_v3"
DEVICE = torch.device("cpu")
DTYPE = torch.float64          # the CPU checks run in float64; the GPU cells run float32

IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
    REPO / "data/runs/v2.8.4/mppi_v3",
    REPO / "data/runs/v2.8.4/mppi_diag",
)


def dir_state() -> dict[str, Any]:
    return {
        str(p.relative_to(REPO)): {
            "mtime": p.stat().st_mtime,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
            "n_entries": len(list(p.iterdir())),
        }
        for p in IMMUTABLE_DIRS
    }


def vram() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def gpu_poll(note: str) -> dict[str, Any]:
    """Poll for other GPU python processes of this project family, and for the raw VRAM state."""
    procs = subprocess.run(["pgrep", "-af", "v284_|mppi_"], capture_output=True, text=True)
    lines = [ln for ln in procs.stdout.strip().splitlines() if ln.strip()]
    mine = str(os.getpid())
    others = [ln for ln in lines if not ln.startswith(mine + " ")]
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "t": time.strftime("%H:%M:%S"),
        "note": note,
        "pgrep_v284_mppi_matches": others,
        "n_matches": len(others),
        "nvidia_smi_compute_apps": compute,
        "vram": vram(),
    }


def _state_batch(system, positions: np.ndarray, omegas: np.ndarray | None = None) -> torch.Tensor:
    """A batch of level-attitude states at the given positions (and optional body rates), built on the
    13-D quadrotor_3d layout (p, q, v, omega) the shipped code uses throughout — the same slicing
    `src.frameworks.mppi.cpu_smoke` and the diagnosis script use — and passed through the system's own
    `wrap_state`. Every value written here is a STATE, never a coefficient. The constructed states are
    read back through `system.position` / `system.angular_rate` in each check, so a layout mistake would
    surface as a failed check rather than as a silent pass."""
    n = positions.shape[0]
    x = torch.zeros((n, int(system.state_dim)), dtype=DTYPE)
    x[:, 0:3] = torch.as_tensor(positions, dtype=DTYPE)
    x[:, 3] = 1.0                                             # unit quaternion, level attitude
    if omegas is not None:
        x[:, 10:13] = torch.as_tensor(omegas, dtype=DTYPE)
    return system.wrap_state(x)


def check_g1_progress(config, mppi_config) -> dict:
    """(a) G1's approach term: positive progress / negative cost while approaching, and the reverse."""
    _, framework, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        g1=True, g2=False, g3=False, g4=False, device=DEVICE, dtype=DTYPE,
    )
    system = framework.system
    cost = framework.controller.cost
    w_p = float(cost.w_p_effective)

    goal = torch.zeros((1, 3), dtype=DTYPE)
    # a straight approaching leg, a straight receding leg, and a frozen (repeated) state
    approach = np.array([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    recede = approach[::-1].copy()
    legs: dict[str, Any] = {}
    for name, path in (("approaching", approach), ("receding", recede)):
        x = _state_batch(system, path)
        increments, cost_terms = [], []
        for k in range(1, x.shape[0]):
            term = progress_cost(system, x[k - 1 : k], x[k : k + 1], goal, cost)
            d_prev = float(torch.linalg.norm(system.position(x[k - 1 : k]) - goal))
            d_now = float(torch.linalg.norm(system.position(x[k : k + 1]) - goal))
            increments.append(d_prev - d_now)
            cost_terms.append(float(term))
        legs[name] = {
            "approach_increments_sum_m": float(np.sum(increments)),
            "cost_contribution_sum": float(np.sum(cost_terms)),
            "per_step_cost": cost_terms,
            "all_increments_same_sign": bool(np.all(np.sign(increments) == np.sign(increments[0]))),
        }
    frozen = _state_batch(system, np.array([[2.0, 1.0, -0.5]]))
    frozen_term = float(progress_cost(system, frozen, frozen, goal, cost))
    # the telescoping identity, verified numerically rather than only asserted in the docstring
    x_app = _state_batch(system, approach)
    d0 = float(torch.linalg.norm(system.position(x_app[0:1]) - goal))
    dH = float(torch.linalg.norm(system.position(x_app[-1:]) - goal))
    telescope = w_p * (dH - d0)
    passed = bool(
        legs["approaching"]["approach_increments_sum_m"] > 0.0
        and legs["approaching"]["cost_contribution_sum"] < 0.0
        and legs["receding"]["approach_increments_sum_m"] < 0.0
        and legs["receding"]["cost_contribution_sum"] > 0.0
        and frozen_term == 0.0
        and abs(legs["approaching"]["cost_contribution_sum"] - telescope) <= 1e-12
    )
    return {
        "term": "-w_p_eff * (||p_{k-1}-g|| - ||p_k-g||), charged per physical rollout step",
        "w_p_effective_read_from_config": w_p,
        "sign_convention": "approach INCREMENT positive while approaching; the COST contribution is its "
                           "negation, so the cost falls while approaching and rises while receding",
        "legs": legs,
        "frozen_sample_term": frozen_term,
        "telescoped_sum_w_p_times_(d_H - d_0)": telescope,
        "PASS": passed,
    }


def check_g3_linear(config, mppi_config) -> dict:
    """(b) G3's terminal is first-power: halving the distance halves the value."""
    _, framework, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        g1=False, g2=False, g3=True, g4=False, device=DEVICE, dtype=DTYPE,
    )
    system = framework.system
    cost = framework.controller.cost
    _, framework_off, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        g1=False, g2=False, g3=False, g4=False, device=DEVICE, dtype=DTYPE,
    )
    cost_off = framework_off.controller.cost

    distances = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    x = _state_batch(system, np.stack([distances, np.zeros_like(distances),
                                       np.zeros_like(distances)], axis=1))
    goal = torch.zeros((distances.size, 3), dtype=DTYPE)
    phi = terminal_cost(system, x, goal, cost).numpy()
    phi_settling = terminal_cost(system, x, goal, cost_off).numpy()
    expected = float(cost.w_terminal) * distances
    ratios = (phi[1:] / phi[:-1]).tolist()                   # each distance is half the previous one
    passed = bool(
        np.allclose(phi, expected, rtol=0.0, atol=1e-12)
        and np.allclose(np.array(ratios), 0.5, rtol=0.0, atol=1e-12)
    )
    return {
        "terminal_expression": "w_terminal * ||p_H - g||",
        "w_terminal_read_from_config": float(cost.w_terminal),
        "distances_m": distances.tolist(),
        "phi_G3": phi.tolist(),
        "phi_expected_w_terminal_times_distance": expected.tolist(),
        "max_abs_error": float(np.abs(phi - expected).max()),
        "ratio_at_each_halving": ratios,
        "phi_settling_same_states_for_contrast": phi_settling.tolist(),
        "linear_terminal_flag": bool(cost.linear_terminal),
        "PASS": passed,
    }


def check_g4_zero(config, mppi_config) -> dict:
    """(c) G4's term is zero at omega = 0, positive off it, and is the ONLY difference it makes."""
    _, framework_on, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        g1=False, g2=False, g3=False, g4=True, device=DEVICE, dtype=DTYPE,
    )
    _, framework_off, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        g1=False, g2=False, g3=False, g4=False, device=DEVICE, dtype=DTYPE,
    )
    system = framework_on.system
    cost_on = framework_on.controller.cost
    cost_off = framework_off.controller.cost
    w_omega = float(cost_on.g4_w_omega)

    omegas = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, -2.0, 0.0],
                       [1.0, 2.0, -3.0], [4.0, 4.0, 4.0]])
    positions = np.array([[2.0, 1.0, 0.5]] * omegas.shape[0])
    x = _state_batch(system, positions, omegas)
    goal = torch.zeros((x.shape[0], 3), dtype=DTYPE)
    on = stage_cost(system, x, goal, cost_on, None, x_prev=x).numpy()
    off = stage_cost(system, x, goal, cost_off, None, x_prev=x).numpy()
    difference = on - off
    rate = system.angular_rate(x).numpy()
    expected = w_omega * rate ** 2
    passed = bool(
        difference[0] == 0.0
        and np.all(difference[1:] > 0.0)
        and np.allclose(difference, expected, rtol=0.0, atol=1e-12)
    )
    return {
        "term": "w_omega * ||omega||^2, always on (no near-goal gate, no tilt condition)",
        "w_omega_read_from_config": w_omega,
        "omega_rows": omegas.tolist(),
        "angular_rate_norm": rate.tolist(),
        "stage_cost_G4_on": on.tolist(),
        "stage_cost_G4_off": off.tolist(),
        "difference": difference.tolist(),
        "expected_w_omega_times_norm_squared": expected.tolist(),
        "difference_at_omega_zero": float(difference[0]),
        "max_abs_error": float(np.abs(difference - expected).max()),
        "PASS": passed,
    }


def check_ou_unchanged(config, mppi_config) -> dict:
    """(d) the OU noise is untouched: the shipped v3 test re-run and compared to the v3 record."""
    now = check_ou_statistics(config, mppi_config)
    recorded = json.loads(V3_SMOKE.read_text(encoding="utf-8"))["g_ou_statistics"]
    # the DECLARED coefficients must be identical (that is the "same coefficient" half of the claim);
    # the MEASURED statistics are compared both for exact identity and against the test's own tolerance.
    declared = ("alpha_declared", "correlation_steps", "sigma_per_channel", "channel_scale",
                "sigma_per_rotor_equivalent_N", "space", "noise", "n_sequences", "horizon")
    measured = ("empirical_lag1_autocorr_per_channel", "empirical_std_by_k",
                "max_lag1_relative_error", "max_std_relative_error_over_all_k")
    declared_identical = {k: bool(now.get(k) == recorded.get(k)) for k in declared}
    measured_identical = {k: bool(now.get(k) == recorded.get(k)) for k in measured}
    lag1_now = np.asarray(now["empirical_lag1_autocorr_per_channel"], dtype=np.float64)
    lag1_v3 = np.asarray(recorded["empirical_lag1_autocorr_per_channel"], dtype=np.float64)
    max_lag1_gap = float(np.abs(lag1_now - lag1_v3).max())
    tolerance = float(now["tolerance_relative"]) * float(now["alpha_declared"])
    return {
        "what": "src.frameworks.mppi.cpu_smoke.check_ou_statistics re-run VERBATIM under the v4 code "
                "(same coefficient, same test) and compared to the recorded v3 result",
        "v3_record": str(V3_SMOKE.relative_to(REPO)) + ":g_ou_statistics",
        "rerun": now,
        "recorded_alpha": recorded.get("alpha_declared"),
        "recorded_lag1": recorded.get("empirical_lag1_autocorr_per_channel"),
        "declared_field_identical": declared_identical,
        "measured_field_identical": measured_identical,
        "all_declared_identical": bool(all(declared_identical.values())),
        "all_measured_identical": bool(all(measured_identical.values())),
        "max_abs_lag1_gap_vs_v3": max_lag1_gap,
        "lag1_gap_tolerance": tolerance,
        "rerun_PASS": bool(now["PASS"]),
        "PASS": bool(now["PASS"] and all(declared_identical.values()) and max_lag1_gap <= tolerance),
    }


def check_defaults_off(config, mppi_config, scenes) -> dict:
    """(e) with all four switches off the v4 code reproduces the pre-v4 expressions to 0 ULP."""
    _, framework, _, _ = build_framework(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3,
        device=DEVICE, dtype=DTYPE,          # no v4 kwargs at all: the config values, i.e. all off
    )
    system = framework.system
    cost = framework.controller.cost
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)
    x = system.wrap_state(initial_states_from_batch(batched))
    goal = torch.as_tensor(np.stack([np.asarray(s.goal) for s in scenes]), dtype=DTYPE)

    stage = stage_cost(system, x, goal, cost, None, x_prev=x)
    distance = torch.linalg.norm(system.position(x) - goal, dim=-1)
    gate = torch.exp(-(distance / cost.gate_rho).square())
    expected_stage = (
        cost.w_goal * distance.square()
        + cost.w_vel * gate * torch.clamp(system.speed(x) - cost.goal_speed_radius, min=0.0).square()
        + cost.w_angrate * gate
        * torch.clamp(system.angular_rate(x) - cost.goal_angrate_radius, min=0.0).square()
    )
    phi = terminal_cost(system, x, goal, cost)
    expected_phi = cost.w_terminal * (
        torch.clamp(distance - cost.goal_radius, min=0.0).square()
        + torch.clamp(system.speed(x) - cost.goal_speed_radius, min=0.0).square()
        + torch.clamp(system.angular_rate(x) - cost.goal_angrate_radius, min=0.0).square()
    )
    stage_exact = bool(torch.equal(stage, expected_stage))
    phi_exact = bool(torch.equal(phi, expected_phi))
    return {
        "n_states": int(x.shape[0]),
        "switches_resolved_from_config": cost.goal_v4_record(),
        "stage_cost_bit_identical_to_pre_v4_expression": stage_exact,
        "terminal_cost_bit_identical_to_pre_v4_expression": phi_exact,
        "max_abs_stage_difference": float((stage - expected_stage).abs().max()),
        "max_abs_terminal_difference": float((phi - expected_phi).abs().max()),
        "PASS": bool(stage_exact and phi_exact),
    }


def run_cpu(args) -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    pool_path = REPO / mppi_config["screen"]["pool"]
    scenes = load_pool(pool_path).scenes[:6]

    before = dir_state()
    report: dict[str, Any] = {
        "what": "charter-'v4' goal-attraction redesign (G1-G4) — CPU smoke. NO GPU work.",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "device": str(DEVICE), "dtype": str(DTYPE),
        "cuda_visible_to_process": bool(torch.cuda.is_available()),
        "config_read_back": {
            "goal_v4": mppi_config["goal_v4"],
            "cost": mppi_config["cost"],
            "deployed_radii": {
                "goal_radius": float(config["env"]["goal_radius"]),
                "goal_speed_radius": float(config["env"]["goal_speed_radius"]),
                "goal_angrate_radius": float(config["env"]["goal_angrate_radius"]),
            },
        },
        "a_g1_progress": check_g1_progress(config, mppi_config),
        "b_g3_linear_terminal": check_g3_linear(config, mppi_config),
        "c_g4_zero_at_omega_zero": check_g4_zero(config, mppi_config),
        "d_ou_unchanged_from_v3": check_ou_unchanged(config, mppi_config),
        "e_defaults_off_reproduce_pre_v4": check_defaults_off(config, mppi_config, scenes),
    }
    checks = ["a_g1_progress", "b_g3_linear_terminal", "c_g4_zero_at_omega_zero",
              "d_ou_unchanged_from_v3", "e_defaults_off_reproduce_pre_v4"]
    report["ALL_PASS"] = bool(all(report[k]["PASS"] for k in checks))
    report["checks"] = checks
    after = dir_state()
    report["immutable_dirs"] = {"before": before, "after": after, "unchanged": before == after}
    report["wall_s"] = round(time.time() - started, 2)
    (OUT_DIR / "smoke_goal_v4.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in checks:
        print(f"{key}: PASS={report[key]['PASS']}", flush=True)
    print(f"ALL_PASS={report['ALL_PASS']}  ({report['wall_s']}s)", flush=True)
    return 0 if report["ALL_PASS"] else 1


# ---- the GPU stage: the v3 reproduction --------------------------------------------------------------
# The metrics compared, field for field, against the retained v3 artifact. Every one of them is a number
# the shared scorer produced for that cell; none is a summary invented here.
REPRO_FIELDS = (
    "cps", "reach", "collision", "coll_obstacle", "coll_band_lower", "coll_band_upper",
    "oob", "stuck", "timeout", "saturation_rate", "cps_ci_lo", "cps_ci_hi",
    "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
)


def run_repro(args) -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA is not available to this process.")
    print(f"pid {os.getpid()} ppid {os.getppid()}", flush=True)
    mppi_config = load_mppi_config()
    screen = mppi_config["screen"]
    v4 = mppi_config["goal_v4"]
    reference = v4["reference_cell"]
    label = str(reference["label"])
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])

    artifact_path = V3_DIR / f"cell__{label}.json"
    recorded = json.loads(artifact_path.read_text(encoding="utf-8"))
    n_scenes = int(recorded["pool"]["n_scenes_scored"])
    ebs = int(recorded["pool"]["ebs"])
    pool_path = REPO / screen["pool"]
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if pool_sha[:8] != str(screen["pool_sha8"]):
        raise SystemExit(f"STOP: pool sha8 {pool_sha[:8]} != {screen['pool_sha8']}.")

    before = dir_state()
    poll = gpu_poll("before the v3 reproduction cell")
    print(f"poll {poll['t']} others={poll['n_matches']} free={poll['vram']['free_mib']:.0f} MiB",
          flush=True)
    if poll["vram"]["free_mib"] < float(args.min_free_mib):
        raise SystemExit(f"STOP: free VRAM {poll['vram']['free_mib']:.0f} MiB "
                         f"< {float(args.min_free_mib):.0f} MiB.")
    cap_mib = 0.5 * poll["vram"]["free_mib"]

    tilt = spawn_tilt_deg(pool_path, n_scenes, mppi_config)
    cell = run_cell(
        pool_path=pool_path, n_scenes=n_scenes, ebs=ebs,
        n_samples=int(reference["n_samples"]), horizon=int(reference["horizon"]),
        lam=float(reference["lam"]), c_crash=float(reference["c_crash"]),
        sigma=float(reference["sigma"]), seed=int(mppi_config["sampling"]["seed"]),
        sample_chunk=int(args.sample_chunk),
        control_hold=int(reference["control_hold"]),
        b1=bool(reference["b1"]), b2=bool(reference["b2"]), b3=bool(reference["b3"]),
        g1=False, g2=False, g3=False, g4=False,           # ALL FOUR v4 SWITCHES OFF
        tilt_deg=tilt, tilt_split_deg=theta_ref,
        label=f"v3repro__{label}", out_dir=OUT_DIR,
        device=torch.device("cuda"), dtype=torch.float32,
        mppi_config=mppi_config,
    )
    poll_after = gpu_poll("after the v3 reproduction cell")

    comparison = {
        field: {
            "v3_recorded": recorded[field], "v4_switches_off": cell[field],
            "abs_diff": abs(float(recorded[field]) - float(cell[field])),
            "identical": bool(float(recorded[field]) == float(cell[field])),
        }
        for field in REPRO_FIELDS
    }
    counts_identical = recorded["outcome_counts"] == cell["outcome_counts"]
    degenerate_identical = recorded["degenerate"] == cell["degenerate"]
    bands_identical = {
        band: bool(recorded["bands"][band]["reach"] == cell["bands"][band]["reach"]
                   and recorded["bands"][band]["collision"] == cell["bands"][band]["collision"]
                   and recorded["bands"][band]["n"] == cell["bands"][band]["n"])
        for band in ("ALL", "tilt_le_ref", "tilt_gt_ref")
    }
    all_identical = bool(
        all(v["identical"] for v in comparison.values())
        and counts_identical and degenerate_identical and all(bands_identical.values())
    )
    report = {
        "what": "charter-'v4' v3 REPRODUCTION gate — the brief-fixed reference cell re-run with all four "
                "v4 switches OFF and compared metric for metric against the retained v3 artifact. This "
                "is a reproduction of an already-recorded row, NOT one of the redesign's scored cells.",
        "reference_cell_status": str(reference["status"]),
        "pid": os.getpid(), "ppid": os.getppid(),
        "artifact": str(artifact_path.relative_to(REPO)),
        "n_scenes": n_scenes, "ebs": ebs,
        "pool_sha8": pool_sha[:8],
        "switches": cell["goal_v4"],
        "comparison": comparison,
        "outcome_counts": {"v3": recorded["outcome_counts"], "v4_off": cell["outcome_counts"],
                           "identical": counts_identical},
        "degenerate": {"v3": recorded["degenerate"], "v4_off": cell["degenerate"],
                       "identical": degenerate_identical},
        "bands_identical": bands_identical,
        "bands": {band: {"v3": {k: recorded["bands"][band][k] for k in ("n", "reach", "collision",
                                                                        "timeout", "cps")},
                         "v4_off": {k: cell["bands"][band][k] for k in ("n", "reach", "collision",
                                                                        "timeout", "cps")}}
                  for band in ("ALL", "tilt_le_ref", "tilt_gt_ref")},
        "ess_p50": {"v3": recorded["ess"]["active"]["p50"], "v4_off": cell["ess"]["active"]["p50"],
                    "identical": bool(recorded["ess"]["active"]["p50"]
                                      == cell["ess"]["active"]["p50"])},
        "wall_s": {"v3": recorded["wall_s"], "v4_off": cell["wall_s"]},
        "vram": {"poll_before": poll, "poll_after": poll_after,
                 "half_free_margin_cap_mib": round(cap_mib, 1),
                 "peak_cuda_alloc_mib": cell["peak_cuda_alloc_mib"],
                 "peak_cuda_reserved_mib": cell["peak_cuda_reserved_mib"],
                 "under_cap": bool(float(cell["peak_cuda_reserved_mib"]) <= cap_mib)},
        "immutable_dirs": {"before": before, "after": dir_state(),
                           "unchanged": before == dir_state()},
        "PASS": all_identical,
        "total_wall_s": round(time.time() - started, 2),
    }
    (OUT_DIR / "repro_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: comparison[k]["identical"] for k in comparison}, indent=2), flush=True)
    print(f"v3 reproduction PASS={all_identical}  peak reserved "
          f"{cell['peak_cuda_reserved_mib']} MiB (cap {cap_mib:.0f})", flush=True)
    return 0 if all_identical else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="charter-'v4' G1-G4 smoke.")
    parser.add_argument("--stage", type=str, default="cpu", choices=["cpu", "repro"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--sample-chunk", type=int, default=0)
    parser.add_argument("--min-free-mib", type=float, default=6144.0)
    args = parser.parse_args()
    return run_cpu(args) if args.stage == "cpu" else run_repro(args)


if __name__ == "__main__":
    raise SystemExit(main())
