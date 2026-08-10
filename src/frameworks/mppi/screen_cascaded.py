"""Driver for the literature-standard (T, omega_des) cascaded MPPI: the CPU smoke, then the 8-cell screen.

STRICT ORDER, ENFORCED BY THIS FILE. The smoke runs on CPU and must be complete and written before the
screen may touch the GPU; `screen` refuses to start if `smoke.json` does not exist.

THE SCREEN STOPS AFTER THE TABLE. No cell is selected, ranked or registered here. The full 8-cell table
goes to the Researcher and the selection is theirs; there is no `final` stage in this file, and the
n = 2000 score it would run is deliberately absent.

NO COEFFICIENT IS TYPED HERE. The grid, the scale, the low-pass, the channel-scale mode and the smoke's
own probe sizes come from `src/frameworks/mppi/config.yaml` (`mppi.cascaded`); the held base
configuration comes from `mppi.v5.base_cell` through `stages_v5.base_kwargs`, so the two blocks cannot
drift apart; the plant constants, dt, the PD gains, the box, omega_max and the three deployed radii come
from the shipped effective config and the System object. The only bare numbers below are the dispatch's
GPU floor and the wait interval, both launch conditions rather than coefficients of any controller.

WRITE SCOPE. `data/runs/v2.8.4/mppi_cascaded/**` and `docs/versions/v2.8.4/mppi_cascaded.md` only. The
build log is APPEND-ONLY. The seven rotor-direct / v5 artifact directories are recorded before and after
every stage and are never written.

Run:  CUDA_VISIBLE_DEVICES="" python -m src.frameworks.mppi.screen_cascaded --stage smoke
      python -m src.frameworks.mppi.screen_cascaded --stage screen
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

from src._version import __version__
from src.eval.build_pools import load_pool
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
    run_cell,
)
from src.frameworks.mppi.rounds_goal_v4 import TABLE_COLUMNS as V4_TABLE_COLUMNS
from src.frameworks.mppi.screen_recovery import spawn_tilt_deg
from src.frameworks.mppi.stages_v5 import (
    SWITCHES,
    base_kwargs,
    markdown_table,
    switch_readback,
    table_row,
)


OUT_DIR = REPO / "data/runs/v2.8.4/mppi_cascaded"
BUILD_LOG = REPO / "docs/versions/v2.8.4/mppi_cascaded.md"

# IMMUTABLE. The rotor-direct record is CLOSED; the v5 a_des cascade is likewise a retained row. All
# seven are recorded before and after every stage and none is ever written by this file.
IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
    REPO / "data/runs/v2.8.4/mppi_v3",
    REPO / "data/runs/v2.8.4/mppi_diag",
    REPO / "data/runs/v2.8.4/mppi_v4",
    REPO / "data/runs/v2.8.4/mppi_v5",
)

# Launch conditions, not coefficients of any controller.
VRAM_FLOOR_MIB = 6144.0
BLOCKED_POLL_WAIT_S = 90.0
POLL_PATTERN = "v284_|mppi_|gate_"

# The v4 format, with an explicit VARIANT column first (a cascaded row may never sit in a table without
# one) and the charter's degenerate-rate column appended. `reach_all` and `d_min_p50` lead the metrics,
# the tilt-60 split carries its own denominators, and nothing is sorted or ranked.
TABLE_COLUMNS = ("variant",) + V4_TABLE_COLUMNS + ("degen_step_frac", "ess_p50")


# =================================================================================================
# environment probes
# =================================================================================================
def vram() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def concurrency_poll() -> dict[str, Any]:
    """Other processes of the concurrent task families. Another CPU agent may be running and has EQUAL
    priority: if free VRAM would fall below the floor we WAIT rather than proceed."""
    proc = subprocess.run(["pgrep", "-af", POLL_PATTERN], capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if "pgrep" not in ln]
    mine = {str(os.getpid()), str(os.getppid())}
    others = [ln for ln in lines if ln.split(" ", 1)[0] not in mine]
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "pgrep_pattern": POLL_PATTERN,
        "matches": lines,
        "matches_excluding_self": others,
        "n_other": len(others),
        "nvidia_smi_compute_apps": compute,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def guarded_poll(cell_id: str, polls: list[dict[str, Any]]) -> dict[str, float]:
    """Poll before a cell. If free VRAM is under the floor, WAIT and poll again; two consecutive blocked
    polls is a STOP. Every poll — blocked or not — is appended to `polls` and reported."""
    for attempt in (1, 2):
        poll = concurrency_poll()
        free = vram()
        blocked = free["free_mib"] < VRAM_FLOOR_MIB
        polls.append({"cell": cell_id, "attempt": attempt, "blocked": blocked, "vram": free, **poll})
        print(f"[{cell_id}] poll {attempt}: {poll['n_other']} other matching process(es) "
              f"{poll['matches_excluding_self']}", flush=True)
        print(f"[{cell_id}] compute apps: {poll['nvidia_smi_compute_apps']!r}", flush=True)
        print(f"[{cell_id}] VRAM before {free}  blocked={blocked}", flush=True)
        if not blocked:
            return free
        if attempt == 1:
            print(f"[{cell_id}] free {free['free_mib']:.0f} MiB < floor {VRAM_FLOOR_MIB:.0f} — the "
                  f"concurrent agents have EQUAL priority, so WAITING {BLOCKED_POLL_WAIT_S:.0f}s rather "
                  f"than proceeding.", flush=True)
            time.sleep(BLOCKED_POLL_WAIT_S)
    raise SystemExit(
        f"STOP: two consecutive blocked polls before cell {cell_id} "
        f"(free VRAM under the {VRAM_FLOOR_MIB:.0f} MiB floor twice)."
    )


def dir_state() -> dict[str, Any]:
    return {
        str(p.relative_to(REPO)): {
            "mtime": p.stat().st_mtime,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))
            + f".{p.stat().st_mtime_ns % 1_000_000_000:09d}",
            "n_entries": len(list(p.iterdir())),
        }
        for p in IMMUTABLE_DIRS
    }


def proc_state() -> dict[str, Any]:
    """This process's own PID and /proc state. This process IS the child the operator launched — no
    setsid and no nohup anywhere."""
    pid = os.getpid()
    state = ""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
                break
    except OSError:                                                            # pragma: no cover
        state = "unavailable"
    return {"pid": pid, "ppid": os.getppid(), "proc_state": state,
            "launched_with": "plain subprocess; NO setsid and NO nohup"}


# =================================================================================================
# shared
# =================================================================================================
INTERFACE_VERBATIM = (
    "Literature-standard quadrotor MPPI (L1-MPPI, Pravitra et al. 2020; PA-MPPI, RA-L 2026): the "
    "planner samples in (collective thrust T, desired body rates omega_des) — attitude dynamics are "
    "delegated to an inner loop, exactly as those works prescribe. Two asymmetries vs our method, both "
    "stated wherever this baseline is reported: (1) privileged full-state + obstacle-field access; "
    "(2) the per-rotor box is NOT handled by the planner — the inner loop allocates rotor thrusts and "
    "clips to [0, 4.905], so actuator constraints are respected only after allocation, not during "
    "planning. The rotor-direct variant (v1-v5, reach 0.0000 across 25+ configurations and N up to "
    "8192) remains the same-interface comparison and is reported as its own row."
)


def cascade_kwargs(mppi_config: dict[str, Any], rate_gain_factor: float) -> dict[str, Any]:
    """The `rate_cascade` mapping one cell's controller is constructed from."""
    cfg = mppi_config["cascaded"]
    return {
        "plan": str(cfg["plan"]),
        "plan_dim": int(cfg["plan_dim"]),
        "channel_scale": str(cfg["channel_scale"]),
        "rate_lowpass": dict(cfg["rate_lowpass"]),
        "rate_gain_factor": float(rate_gain_factor),
    }


def append_log(section: str) -> None:
    """APPEND-ONLY. A table once written is never edited and never deleted."""
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open("a", encoding="utf-8") as handle:
        handle.write(section.rstrip("\n") + "\n\n")


def write_report(path: Path, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(f"{path.stem}__{time.strftime('%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def preamble(mppi_config: dict[str, Any], stage: str, *, gpu: bool) -> dict[str, Any]:
    cfg = mppi_config["cascaded"]
    scale = cfg["scale"]
    pool_path = REPO / mppi_config["screen"]["pool"]
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if pool_sha[:8] != str(mppi_config["screen"]["pool_sha8"]):
        raise SystemExit(f"STOP: pool sha8 {pool_sha[:8]} != {mppi_config['screen']['pool_sha8']}.")
    report: dict[str, Any] = {
        "what": f"literature-standard (T, omega_des) cascaded MPPI — {stage}",
        "version": __version__,
        "interface_verbatim": INTERFACE_VERBATIM,
        "variant": str(cfg["variant"]),
        "variant_status": str(cfg["variant_status"]),
        "process": proc_state(),
        "base_configuration_source": (
            "mppi.v5.base_cell — the v3 base held for every cell (C_crash 1e5, m 1, H 40, sigma 1.0, "
            "OU noise, hover centring, settling terminal, B1-B3 off, G1-G4 off). Read, not duplicated."
        ),
        "base_cell": dict(mppi_config["v5"]["base_cell"]),
        "cascaded_config": {k: v for k, v in cfg.items() if k != "smoke"},
        "scale": {"n_scenes": int(scale["n_scenes"]), "ebs": int(scale["ebs"]),
                  "seed": int(scale["seed"])},
        "pool": {"path": str(pool_path), "sha256": pool_sha, "sha8": pool_sha[:8],
                 "order": f"the FIRST {int(scale['n_scenes'])} scenes of the pool, in pool order"},
        "immutable_before": dir_state(),
        "no_selection": (
            "MEASUREMENT ONLY. No threshold is registered, no cell is ranked and no cell is selected "
            "anywhere in this lineage; the screen stops after the table and the selection is the "
            "Researcher's."
        ),
    }
    if gpu:
        launch = vram()
        if launch["free_mib"] < VRAM_FLOOR_MIB:
            raise SystemExit(
                f"STOP: launch free VRAM {launch['free_mib']:.0f} MiB < {VRAM_FLOOR_MIB:.0f}."
            )
        report["launch_vram"] = launch
        report["peak_cap_mib"] = 0.5 * launch["free_mib"]
        report["vram_floor_mib"] = VRAM_FLOOR_MIB
    return report


def finish(report: dict[str, Any], t0: float) -> None:
    report["immutable_after"] = dir_state()
    report["immutable_unchanged"] = report["immutable_before"] == report["immutable_after"]
    report["wall_s"] = round(time.time() - t0, 2)
    print(f"immutable dirs unchanged: {report['immutable_unchanged']}", flush=True)


# =================================================================================================
# STAGE: the CPU smoke
# =================================================================================================
def stage_smoke(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    from src.frameworks.mppi import smoke_cascade_rate as smoke

    t0 = time.time()
    report = preamble(mppi_config, "CPU smoke (before any GPU cell)", gpu=False)
    torch.set_num_threads(int(args.threads))
    cfg = mppi_config["cascaded"]
    smoke_cfg = cfg["smoke"]
    config = effective_config(mppi_config)
    base = mppi_config["v5"]["base_cell"]
    device = torch.device("cpu")
    dtype = torch.float64
    seed = int(cfg["scale"]["seed"])
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])

    # the smoke runs the FACTOR-1 controller: factor 1 is the backup-CBF value itself, which is what
    # makes check (f)'s algebraic identity with the shipped PD testable.
    smoke_kwargs = base_kwargs(mppi_config)
    smoke_kwargs["horizon"] = int(smoke_cfg["straight_line"]["mppi_horizon"])
    planner_budgets = [int(v) for v in smoke_cfg["straight_line"]["mppi_n_samples"]]

    smoke_factor = float(smoke_cfg["rate_gain_factor"])

    def make(n_samples: int):
        return build_framework(
            config, mppi_config, n_samples=int(n_samples), sample_chunk=0, seed=seed,
            rate_cascade=cascade_kwargs(mppi_config, smoke_factor),
            device=device, dtype=dtype, **smoke_kwargs,
        )

    system, framework, _, _ = make(planner_budgets[0])
    controller = framework.controller

    scene = smoke.straight_line_scene(
        system, [float(v) for v in smoke_cfg["straight_line"]["offset"]], dtype
    )
    seconds = float(smoke_cfg["straight_line"]["seconds"])
    pool_scenes = load_pool(REPO / mppi_config["screen"]["pool"]).scenes[
        : int(smoke_cfg["predicate_probe"]["n_scenes"])
    ]

    checks: dict[str, Any] = {}
    print("[smoke] (a) hover fixed point ...", flush=True)
    checks["a_hover_fixed_point"] = smoke.check_hover(
        system, controller, dt, int(smoke_cfg["hover_steps"])
    )
    print("[smoke] (b1) straight line, deterministic outer law ...", flush=True)
    checks["b1_straight_line_outer_law"] = smoke.check_straight_line_outer(
        system, controller, scene, dt, seconds, goal_radius
    )
    print("[smoke] (c) box and the asymmetry evidence ...", flush=True)
    checks["c_box_and_asymmetry"] = smoke.check_box(
        system, controller, int(smoke_cfg["box_probe"]["n_states"]),
        int(smoke_cfg["box_probe"]["n_sampled_plans"]), seed,
    )
    print("[smoke] (d) collision indicator vs step_outcomes ...", flush=True)
    checks["d_collision_indicator"] = smoke.check_predicate(
        system, controller, config, pool_scenes,
        int(smoke_cfg["predicate_probe"]["n_rollout_steps"]), seed,
    )
    print("[smoke] (e) sampled-rate low-pass ...", flush=True)
    checks["e_rate_lowpass"] = smoke.check_lowpass(
        controller, int(smoke_cfg["lowpass_probe"]["n_sequences"]), seed
    )
    print("[smoke] (f) the interface reproduces the shipped cascaded PD ...", flush=True)
    checks["f_interface_reproduces_shipped_pd"] = smoke.check_interface_reproduces_shipped(
        system, controller, int(smoke_cfg["box_probe"]["n_states"]), seed
    )
    print("[smoke] (g) planner model vs the full plant ...", flush=True)
    checks["g_planner_model_fidelity"] = smoke.check_planner_fidelity(
        system, controller, dt, int(smoke_cfg["fidelity_probe"]["n_states"]),
        seed + int(smoke_cfg["fidelity_probe"]["seed_offset"]),
    )
    checks["b2_straight_line_planner"] = {}
    for n_samples in planner_budgets:
        print(f"[smoke] (b2) straight line, the (T, omega) planner itself, N {n_samples} ...", flush=True)
        _, leg_framework, _, _ = make(n_samples)
        checks["b2_straight_line_planner"][f"N{n_samples}"] = smoke.check_straight_line_planner(
            system, leg_framework, scene, dt, seconds, goal_radius
        )

    report.update({
        "device": "cpu", "dtype": str(dtype),
        "dt_read_from_config": dt,
        "goal_radius_read_from_config": goal_radius,
        "controller": controller.cascade_record(),
        "sampler": controller.sampler_record(),
        "smoke_config": dict(smoke_cfg),
        "checks": checks,
        "verdicts": {
            "a_hover_max_position_drift_m": checks["a_hover_fixed_point"]["max_position_drift_m"],
            "a_hover_max_tilt_deg": checks["a_hover_fixed_point"]["max_tilt_deg"],
            "b1_reaches_position_radius":
                checks["b1_straight_line_outer_law"]["reaches_position_radius"],
            "b2_reaches_position_radius": {
                key: leg["reaches_position_radius"]
                for key, leg in checks["b2_straight_line_planner"].items()
            },
            "b2_closest_approach_m": {
                key: leg["closest_approach_m"]
                for key, leg in checks["b2_straight_line_planner"].items()
            },
            "g_position_divergence_p50_at_horizon_end":
                checks["g_planner_model_fidelity"]["position_divergence_m"]["p50"][-1],
            "c_aggressive_all_postclip_inside_box":
                checks["c_box_and_asymmetry"]["aggressive_command"]["all_postclip_inside_box"],
            "c_sampled_preclip_outside_box":
                checks["c_box_and_asymmetry"]["sampled_plans"]["n_preclip_outside_box"],
            "c_sampled_preclip_denominator":
                checks["c_box_and_asymmetry"]["sampled_plans"]["n_entries"],
            "d_collision_indicator_agreement_rate": checks["d_collision_indicator"]["agreement_rate"],
            "e_rate_autocorrelation_increased":
                checks["e_rate_lowpass"]["rate_channels_autocorrelation_increased"],
            "f_max_abs_rotor_difference_N":
                checks["f_interface_reproduces_shipped_pd"]["max_abs_rotor_difference_N"],
        },
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "smoke.json", report)

    hover = checks["a_hover_fixed_point"]
    b1 = checks["b1_straight_line_outer_law"]
    b2 = checks["b2_straight_line_planner"]
    box = checks["c_box_and_asymmetry"]
    pred = checks["d_collision_indicator"]
    low = checks["e_rate_lowpass"]
    ident = checks["f_interface_reproduces_shipped_pd"]
    fid = checks["g_planner_model_fidelity"]
    lines = [
        "# v2.8.4 — literature-standard (T, omega_des) cascaded MPPI: build log",
        "",
        "APPEND-ONLY. No table written below is ever edited or deleted.",
        "",
        "## What this baseline is (verbatim)",
        "",
        INTERFACE_VERBATIM,
        "",
        "The bracketed per-rotor interval above is the charter's own text; every machine-readable "
        "record this lineage emits formats the same sentence with the interval read off "
        "`system.u_bounds`, so the words and the numbers cannot drift apart.",
        "",
        "## How this differs from the charter-\"v5\" cascade",
        "",
        "They are different interfaces and separate rows. `src/frameworks/mppi/cascade.py` (v5 Stage 2) "
        "plans the desired WORLD ACCELERATION `a_des` in R^3 — `(T, q_des)` being read off "
        "`System.lqr_action`'s own lines because that shipped outer law derives both from one "
        "world-force vector and is yaw-blind — and its rollout integrates THE FULL PLANT with the "
        "shipped cascaded PD closed inside the loop, so a rotor command and a per-rotor clip exist at "
        "every rollout step. This baseline plans `v = [T, w_x, w_y, w_z]`: the planner propagates "
        "translational dynamics plus attitude KINEMATICS with the sampled body rate as the input "
        "(attitude dynamics are delegated and do not appear in the prediction), and the inner loop is a "
        "body-RATE tracking law rather than an attitude-error PD. No rotor command is formed anywhere "
        "in the planner.",
        "",
        "Context, measured and not re-derived here: the v5 `a_des` cascade scored reach 0.0000 on all "
        "four cells at n = 400 while driving `omega_p50` from about 4.89 rad/s down to 0.20-0.29. A "
        "reach of 0.0000 from this cascade is consistent with that record, not a new discovery.",
        "",
        "## CPU smoke (run before any GPU cell)",
        "",
        "Measurement only; no threshold is registered and no cell is selected. float64 on CPU; the "
        f"screen cells run float32 on the GPU. dt = {dt} and every radius, gain, bound and plant "
        "constant below is read from the shipped config or the System object.",
        "",
        "| check | measurement | read against |",
        "|---|---|---|",
        f"| (a) hover fixed point, {hover['steps']} steps at `T = m*g`, `omega_des = 0`, on the FULL "
        f"plant | position drift {hover['max_position_drift_m']:.3e} m, speed "
        f"{hover['max_speed_m_s']:.3e} m/s, rate {hover['max_angular_rate_rad_s']:.3e} rad/s, tilt "
        f"{hover['max_tilt_deg']:.3e} deg | exact stationarity; the rotor command differs from the "
        f"system's own trim by {hover['max_abs_difference_from_trim_N']:.3e} N |",
        f"| (b1) straight line, deterministic outer law | closest approach "
        f"{b1['closest_approach_m']:.4f} m, final speed {b1['final_speed_m_s']:.4f} m/s, rate-limit "
        f"engagements {b1['rate_limit_engagements']} | `env.goal_radius` = {goal_radius}, reaches: "
        f"{b1['reaches_position_radius']} |",
        *[f"| (b2) straight line, the (T, omega) MPPI planner itself (N {leg['N']}, H {leg['H']}) | "
          f"closest approach {leg['closest_approach_m']:.4f} m, final distance "
          f"{leg['final_distance_m']:.4f} m | `env.goal_radius` = {goal_radius}, reaches: "
          f"{leg['reaches_position_radius']} |" for leg in b2.values()],
        f"| (c) inner loop on a constructed aggressive command | post-clip range "
        f"[{box['aggressive_command']['postclip_min_N']:.4f}, "
        f"{box['aggressive_command']['postclip_max_N']:.4f}] N over "
        f"{box['aggressive_command']['n_entries']} entries | the box read from `system.u_bounds` = "
        f"{box['box_read_from_system']}, all inside: "
        f"{box['aggressive_command']['all_postclip_inside_box']} |",
        f"| (c) ASYMMETRY EVIDENCE: sampled plans mapping outside the box BEFORE the clip | "
        f"{box['sampled_plans']['n_preclip_outside_box']} of {box['sampled_plans']['n_entries']} "
        f"per-rotor entries "
        f"({box['sampled_plans']['frac_preclip_outside_box']:.4f}); "
        f"{box['sampled_plans']['n_plans_with_any_entry_outside_box']} of "
        f"{box['sampled_plans']['n_plans']} sampled plans have at least one | pre-clip range "
        f"[{box['sampled_plans']['preclip_min_N']:.4f}, {box['sampled_plans']['preclip_max_N']:.4f}] N "
        f"against the box {box['box_read_from_system']} |",
        f"| (d) the cost's collision indicator vs `outcomes.step_outcomes` | agreement "
        f"{pred['agreement_rate']:.6f} over {pred['n_states_compared']} states "
        f"({pred['n_disagreements']} disagreements) | the harness's own predicate, including "
        f"{pred['n_boundary_probes']} states placed exactly on the cylinder and band boundaries |",
        f"| (e) the sampled-rate low-pass | lag-1 autocorrelation before "
        f"{[round(v, 4) for v in low['lag1_before'][1:]]} -> after "
        f"{[round(v, 4) for v in low['lag1_after'][1:]]} | increased on every rate channel: "
        f"{low['rate_channels_autocorrelation_increased']}; the unfiltered collective channel is the "
        f"control |",
        f"| (f) the interface expresses the shipped cascaded PD | max rotor difference "
        f"{ident['max_abs_rotor_difference_N']:.3e} N over {ident['n_probe_states']} random states | "
        f"`system.lqr_action` itself, forces up to {ident['max_abs_rotor_force_N']:.4f} N |",
        f"| (g) the planner's model vs the FULL plant over one horizon | position divergence p50 "
        f"{[round(v, 4) for v in fid['position_divergence_m']['p50']]} m at horizon indices "
        f"{fid['position_divergence_m']['at_horizon_index']} (p95 "
        f"{[round(v, 4) for v in fid['position_divergence_m']['p95']]}) | the error the delegation "
        f"assumption makes, over {fid['n_states']} states; measurement only |",
        "",
        "(b1) is the check the charter's smoke item names: the obstacle-free straight-line scene is "
        "reached under the full cascade on the real plant. (b2) runs the same scene through the (T, "
        "omega) MPPI PLANNER at the screen's own two sample budgets and is reported beside it whatever "
        "it says — it is a wiring test on one constructed scene, not a second screen, and no threshold "
        "or selection is taken against it.",
        "",
        "(c) is the evidence for asymmetry (2), not prose: the planner has no per-rotor variable to "
        "bound, so its own sampled plans map to pre-clip rotor commands outside the box at the rate "
        "reported above, and the box is applied only after allocation, inside the inner loop, on the "
        "executed command.",
        "",
        "(f) states the algebraic identity that makes the inner loop a reuse rather than a "
        "re-invention: with the rate setpoint `omega_des = (kp_att / kd_att) e_att_body` and the "
        "inner-loop gain `k_rate = kd_att`, the cascade's torque is `J (kp_att e_att_body - kd_att "
        "omega)`, which is `System.lqr_action`'s own line. Both gains are read from "
        "`config['lqr'][run.system]` — the gains `src/common/filter_backup.py` names in its docstring "
        "and reaches through `override_gains`.",
        "",
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. CPU only, no GPU "
        f"was used. Immutable dirs unchanged: {report['immutable_unchanged']}. "
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


# =================================================================================================
# STAGE: the 8-cell screen
# =================================================================================================
def stage_screen(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    t0 = time.time()
    if not (OUT_DIR / "smoke.json").exists():
        raise SystemExit(
            "STOP: the charter orders the smoke BEFORE any GPU cell; run --stage smoke first "
            f"({OUT_DIR / 'smoke.json'} does not exist)."
        )
    report = preamble(mppi_config, "the 8-cell screen", gpu=True)
    torch.set_num_threads(int(args.threads))
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    cfg = mppi_config["cascaded"]
    scale = cfg["scale"]
    grid = cfg["grid"]
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])
    tilt = spawn_tilt_deg(
        REPO / mppi_config["screen"]["pool"], int(scale["n_scenes"]), mppi_config
    )
    report["tilt_split"] = {
        "theta_ref_deg": theta_ref, "convention": "<= / >",
        "definition": "degrees(arccos(clip(R(q0)[2,2], -1, 1))) — the documented fallback; the pool "
                      "manifest carries no per-scene spawn-tilt field",
        "policy": "born-inverted and high-tilt episodes are NEVER filtered out of a headline; the bands "
                  "are ADDITIONAL columns with their own denominators",
        "n_le": int((tilt <= theta_ref).sum()), "n_gt": int((tilt > theta_ref).sum()),
        "tilt_min": float(tilt.min()), "tilt_median": float(np.median(tilt)),
        "tilt_max": float(tilt.max()),
    }
    print(f"pid {os.getpid()} ppid {os.getppid()} state {report['process']['proc_state']}", flush=True)
    print(f"pool sha8 {report['pool']['sha8']} OK", flush=True)
    print(f"launch VRAM {report['launch_vram']}  cap {report['peak_cap_mib']:.0f} MiB", flush=True)
    print(f"tilt split at {theta_ref} deg: n_le {report['tilt_split']['n_le']} "
          f"n_gt {report['tilt_split']['n_gt']}", flush=True)

    polls: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    kwargs = base_kwargs(mppi_config)

    # CONFIG ORDER: N outer, lambda, rate-gain factor innermost — exactly the nesting the charter writes.
    for n_samples in [int(v) for v in grid["n_samples"]]:
        for lam in [float(v) for v in grid["lam"]]:
            for factor in [float(v) for v in grid["rate_gain_factors"]]:
                cell_id = f"N{n_samples}_lam{lam:g}_kfac{factor:g}"
                chunk = int(cfg["sample_chunk"][n_samples])
                before = guarded_poll(cell_id, polls)
                cell_kwargs = dict(kwargs)
                cell_kwargs["lam"] = lam
                try:
                    cell = run_cell(
                        pool_path=REPO / mppi_config["screen"]["pool"],
                        n_scenes=int(scale["n_scenes"]),
                        ebs=int(scale["ebs"]),
                        seed=int(scale["seed"]),
                        n_samples=n_samples,
                        sample_chunk=chunk,
                        tilt_deg=tilt,
                        tilt_split_deg=theta_ref,
                        label=cell_id,
                        out_dir=OUT_DIR,
                        device=device,
                        dtype=dtype,
                        mppi_config=mppi_config,
                        v4_columns=True,
                        rate_cascade=cascade_kwargs(mppi_config, factor),
                        **cell_kwargs,
                    )
                except torch.cuda.OutOfMemoryError as exc:                     # pragma: no cover
                    raise SystemExit(f"STOP: CUDA OOM on cell {cell_id}: {exc}")
                after = vram()
                peak = float(cell["peak_cuda_reserved_mib"])
                print(f"[{cell_id}] VRAM after {after}  peak alloc {cell['peak_cuda_alloc_mib']} MiB "
                      f"reserved {peak} MiB (cap {report['peak_cap_mib']:.0f})", flush=True)
                if peak > report["peak_cap_mib"]:
                    raise SystemExit(
                        f"STOP: peak reserved {peak:.0f} MiB over cap {report['peak_cap_mib']:.0f} MiB."
                    )
                readback = switch_readback(cell)
                if not readback["all_seven_false"]:
                    raise SystemExit(
                        f"STOP: cell {cell_id} did not resolve all seven switches false: "
                        f"{readback['switches']}"
                    )
                row = table_row(cell_id, cell)
                row["variant"] = "cascaded_rate"
                row["ess_p50"] = cell["ess"]["active"].get("p50")
                row["degen_step_frac"] = cell["degenerate"]["step_frac_over_active_window"]
                rows.append(row)
                records.append({
                    "cell_id": cell_id,
                    "N": n_samples, "lam": lam, "rate_gain_factor": factor,
                    "k_rate": cell["cascade"]["inner_loop"]["k_rate"],
                    "sample_chunk": chunk,
                    "variant": "cascaded_rate",
                    "switch_readback": readback,
                    "outcome_counts": cell["outcome_counts"],
                    "collision_by_class": {
                        "obstacle": cell["coll_obstacle"],
                        "band_lower": cell["coll_band_lower"],
                        "band_upper": cell["coll_band_upper"],
                    },
                    "ess": cell["ess"]["active"],
                    "degenerate": cell["degenerate"],
                    "preclip_evidence": cell["cascade"]["preclip_evidence"],
                    "sigma_per_channel": cell["cascade"]["sigma_per_channel"],
                    "vram": {"before": before, "after": after,
                             "peak_cuda_alloc_mib": cell["peak_cuda_alloc_mib"],
                             "peak_cuda_reserved_mib": peak, "cap_mib": report["peak_cap_mib"]},
                    "wall_s": cell["wall_s"],
                    "n_control_steps": cell["n_control_steps"],
                    "row": row,
                })
                print("[" + cell_id + "] " + json.dumps(row), flush=True)

    table = markdown_table(rows, TABLE_COLUMNS)
    print("\n" + table, flush=True)
    report.update({
        "polls": polls, "columns": list(TABLE_COLUMNS), "table_markdown": table,
        "rows": rows, "cells": records,
        "separation": "every row above carries an explicit `variant` column reading `cascaded_rate`. "
                      "These numbers are a SEPARATE baseline: they must never be blended with, averaged "
                      "with or substituted for the rotor-direct (vanilla) MPPI numbers or the v5 a_des "
                      "cascade's, and no table may mix them without that column.",
        "stopped_after_the_screen": (
            "STOP HONOURED. No cell was selected, ranked or registered, and the n = 2000 final score was "
            "NOT run: the charter hands the full 8-cell table to the Researcher and the selection is "
            "theirs."
        ),
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "screen.json", report)

    lines = [
        "## The 8-cell screen",
        "",
        "SEPARATE BASELINE ROW. Every row carries an explicit `variant` column reading `cascaded_rate`. "
        "These numbers must never be blended with, averaged with or substituted for the rotor-direct "
        "(vanilla) MPPI numbers or the charter-\"v5\" `a_des` cascade's, and no table may put them side "
        "by side without that column.",
        "",
        f"N x lambda x inner-loop rate gain at n = {report['scale']['n_scenes']} "
        f"(the first {report['scale']['n_scenes']} fullcb scenes of pool `{report['pool']['sha8']}`, in "
        f"pool order), ebs {report['scale']['ebs']}, seed {report['scale']['seed']}. Everything else is "
        "the v3 base configuration read from `mppi.v5.base_cell`: C_crash 1e5, m = 1, H = 40, sigma "
        "1.0, OU noise, hover centring on T, settling terminal with the deployed terminal's constants, "
        "B1-B3 off, G1-G4 off — all seven switches read back false on every cell. Rows are in CONFIG "
        "ORDER; nothing is sorted, ranked or selected.",
        "",
        table,
        "",
        "Columns are the v4 format with an explicit variant column first and the degenerate-decision "
        "rate appended. `reach_all` and `d_min_p50` lead the metrics. `inside_r05_share` is the v4 "
        "TIME-inside share of the flown trajectory against the reporting probe radius read from "
        "`mppi.goal_v4.report.probe_radius`; it is a reporting probe, not a threshold. The tilt bands "
        f"split at `mppi.recovery.theta_ref_deg` = {theta_ref:g} deg with `<=` / `>`, carry their own "
        f"denominators (`n_le60` = {report['tilt_split']['n_le']}, `n_gt60` = "
        f"{report['tilt_split']['n_gt']}), and are ADDITIONAL columns: no high-tilt or born-inverted "
        "episode is filtered out of any headline. Spawn tilt is `degrees(arccos(clip(R(q0)[2,2], -1, "
        "1)))` — the documented fallback, since the pool manifest carries no per-scene tilt field.",
        "",
        "### Cell configuration, collision by class, and the asymmetry count per cell",
        "",
        "| cell | N | lambda | rate-gain factor | k_rate | coll obstacle | coll band lower | "
        "coll band upper | pre-clip entries outside the box / total |",
        "|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['cell_id']} | {r['N']} | {r['lam']:g} | {r['rate_gain_factor']:g} | {r['k_rate']:g} | "
          f"{r['collision_by_class']['obstacle']:.4f} | {r['collision_by_class']['band_lower']:.4f} | "
          f"{r['collision_by_class']['band_upper']:.4f} | "
          f"{r['preclip_evidence']['n_outside_box']} / {r['preclip_evidence']['n_entries']} "
          f"({r['preclip_evidence']['frac_outside_box']:.4f}) |" for r in records],
        "",
        "The last column counts, over every EXECUTED command of the cell, the per-rotor entries of the "
        "pre-clip allocation `wrench @ system.mixer_inv.T` that fall outside the box read from "
        "`system.u_bounds`. It is the running measurement of asymmetry (2): the per-rotor box is NOT "
        "handled by the planner — the inner loop allocates rotor thrusts and clips to the box, so "
        "actuator constraints are respected only after allocation, not during planning.",
        "",
        "`infeasibility` and `mean_proj_mag` are STRUCTURALLY INAPPLICABLE for this arm and are recorded "
        "as such in every cell JSON: it enters the shared eval path with an identity filter and carries "
        "no certificate, so there is no QP and nothing can be infeasible. The zeros are a property of "
        "the wiring, not a measurement of the controller.",
        "",
        "### Chunking and VRAM, per cell",
        "",
        "| cell | N | sample_chunk | peak alloc MiB | peak reserved MiB | cap MiB | free before | "
        "free after | wall s | wall ms/step |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['cell_id']} | {r['N']} | {r['sample_chunk']} | "
          f"{r['vram']['peak_cuda_alloc_mib']:.1f} | {r['vram']['peak_cuda_reserved_mib']:.1f} | "
          f"{r['vram']['cap_mib']:.0f} | {r['vram']['before']['free_mib']:.0f} | "
          f"{r['vram']['after']['free_mib']:.0f} | {r['wall_s']:.1f} | "
          f"{r['row']['wall_ms_per_step']:.1f} |" for r in records],
        "",
        "### Stop",
        "",
        "The screen STOPS here. No cell is selected, ranked or registered, and the n = 2000 final score "
        "was not run: the full 8-cell table above goes to the Researcher and the selection is theirs.",
        "",
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. "
        f"Launch free VRAM {report['launch_vram']['free_mib']:.0f} MiB, peak cap "
        f"{report['peak_cap_mib']:.0f} MiB. Immutable dirs unchanged: "
        f"{report['immutable_unchanged']}. Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


# =================================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="literature-standard (T, omega_des) cascaded MPPI: smoke, then the 8-cell screen."
    )
    parser.add_argument("--stage", type=str, required=True, choices=["smoke", "screen"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    mppi_config = load_mppi_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return {"smoke": stage_smoke, "screen": stage_screen}[args.stage](args, mppi_config)


if __name__ == "__main__":
    raise SystemExit(main())
