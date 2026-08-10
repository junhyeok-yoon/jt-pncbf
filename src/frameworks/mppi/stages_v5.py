"""v2.8.4 charter-"v5" — the three-stage driver: sample budget, cascaded MPPI, relaxed companion.

STRICT ORDER, ENFORCED BY THIS FILE. Stage 1 must be complete and its table written to the build log
before Stage 2 may begin; Stage 2 must be complete and written before Stage 3. Each stage refuses to
run until the previous stage's artifact exists, so the ordering is a property of the code and not of
the operator's discipline.

THE TWO STRUCTURAL GATES, AND NOTHING ELSE.

  * AFTER STAGE 1 — binding in BOTH directions. If ANY Stage-1 cell attains reach_all > 0, the run
    STOPS after Stage 1: standard MPPI works at scale and Stage 2 is not needed. If all three cells
    read reach_all = 0.0000, the run proceeds. `stage2` re-reads the Stage-1 artifact and refuses to
    run if the gate went the other way, so the gate cannot be ignored by invoking the stages by hand.
  * AFTER STAGE 2 — none. Stage 3 runs regardless; it is a re-scoring.

NO OTHER THRESHOLD EXISTS ANYWHERE IN THE v5 LINEAGE. Nothing is ranked, nothing is selected, nothing
is registered. The r in {0.5, 1.0} of Stage 1 are dispatch-given REPORTING PROBES read from
`mppi.v5.stage1.endpoint_probe_radii`, and the 0.15 of Stage 3 is the DEPLOYED position radius read
from `env.goal_radius`. Neither is a gate.

NO COEFFICIENT IS TYPED HERE. The base cell, the scale, the N list, the chunk sizes, the cascade
configuration, the probe radii and the tilt-split theta_ref are read from `src/frameworks/mppi/
config.yaml`; the plant constants, dt, the PD gains and the three deployed radii come from the shipped
effective config through `evaluate_mppi` and the System object. The only bare numbers below are the
dispatch's GPU floor and the wait interval, both launch conditions rather than coefficients of any
controller.

WRITE SCOPE. `data/runs/v2.8.4/mppi_v5/**` and `docs/versions/v2.8.4/mppi_v5.md`, and nothing else.
The build log is APPEND-ONLY: every stage appends its section and no stage ever rewrites one.

Run:  python -m src.frameworks.mppi.stages_v5 --stage 0     # base-config verification (preflight)
      python -m src.frameworks.mppi.stages_v5 --stage 1
      python -m src.frameworks.mppi.stages_v5 --stage 2smoke
      python -m src.frameworks.mppi.stages_v5 --stage 2
      python -m src.frameworks.mppi.stages_v5 --stage 3
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
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
    run_cell,
)
from src.frameworks.mppi.relaxed_v5 import load_rollouts, save_rollouts, score_two_terminals
from src.frameworks.mppi.rounds_goal_v4 import TABLE_COLUMNS as V4_TABLE_COLUMNS
from src.frameworks.mppi.screen_recovery import spawn_tilt_deg


OUT_DIR = REPO / "data/runs/v2.8.4/mppi_v5"
BUILD_LOG = REPO / "docs/versions/v2.8.4/mppi_v5.md"

IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
    REPO / "data/runs/v2.8.4/mppi_v3",
    REPO / "data/runs/v2.8.4/mppi_diag",
    REPO / "data/runs/v2.8.4/mppi_v4",
)

# The dispatch's own GPU floor, in MiB, and how long to wait on a blocked poll before re-polling.
# Launch conditions, not coefficients of any controller.
VRAM_FLOOR_MIB = 6144.0
BLOCKED_POLL_WAIT_S = 90.0

# The concurrency poll pattern the dispatch names. Other agents run in
# data/runs/v2.8.4/smooth_diag/ and data/runs/v2.8.4/{spawn_signal,gate_screen}/ and have EQUAL
# priority: if free VRAM would fall below the floor we WAIT rather than proceed.
POLL_PATTERN = "v284_|mppi_|smooth|spawn"

# The seven switches the base configuration requires to resolve FALSE, read back off every
# constructed cell rather than assumed.
SWITCHES = ("B1", "B2", "B3", "G1", "G2", "G3", "G4")

# Stage 1's extra columns, appended AFTER the v4 format's own columns so the v4 format is intact.
STAGE1_EXTRA = ("ess_p50", "best_end_in_r0.5", "best_end_in_r1", "n_decisions")
# Stage 2 carries an explicit VARIANT column: a cascaded row may never sit in a table without one.
STAGE2_COLUMNS = ("variant",) + V4_TABLE_COLUMNS + ("ess_p50",)


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
    """Other processes of the concurrent task families, polled before EVERY cell."""
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
    """Poll before a cell. If free VRAM is under the floor, WAIT and poll again; two consecutive
    blocked polls is a STOP. Every poll — blocked or not — is appended to `polls` and reported."""
    for attempt in (1, 2):
        poll = concurrency_poll()
        free = vram()
        blocked = free["free_mib"] < VRAM_FLOOR_MIB
        record = {"cell": cell_id, "attempt": attempt, "blocked": blocked, "vram": free, **poll}
        polls.append(record)
        print(f"[{cell_id}] poll {attempt}: {poll['n_other']} other matching process(es) "
              f"{poll['matches_excluding_self']}", flush=True)
        print(f"[{cell_id}] compute apps: {poll['nvidia_smi_compute_apps']!r}", flush=True)
        print(f"[{cell_id}] VRAM before {free}  blocked={blocked}", flush=True)
        if not blocked:
            return free
        if attempt == 1:
            print(f"[{cell_id}] free {free['free_mib']:.0f} MiB < floor {VRAM_FLOOR_MIB:.0f} — the "
                  f"concurrent agents have EQUAL priority, so WAITING {BLOCKED_POLL_WAIT_S:.0f}s "
                  f"rather than proceeding.", flush=True)
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
    """This process's own PID and /proc state — the dispatch asks for the CHILD PID and its state, and
    this process IS the child the operator launched (no setsid, no nohup, anywhere)."""
    pid = os.getpid()
    state = ""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
                break
    except OSError:                                                       # pragma: no cover
        state = "unavailable"
    return {"pid": pid, "ppid": os.getppid(), "proc_state": state,
            "launched_with": "plain subprocess; NO setsid and NO nohup"}


# =================================================================================================
# shared cell construction
# =================================================================================================
def base_kwargs(mppi_config: dict[str, Any]) -> dict[str, Any]:
    """The BASE CONFIGURATION every v5 stage shares, read from `mppi.v5.base_cell`. See that block for
    the naming resolution between the two identically-named retained rows."""
    base = mppi_config["v5"]["base_cell"]
    return {
        "horizon": int(base["horizon"]),
        "lam": float(base["lam"]),
        "c_crash": float(base["c_crash"]),
        "sigma": float(base["sigma"]),
        "control_hold": int(base["control_hold"]),
        "space": str(base["space"]),
        "noise": str(base["noise"]),
        "lam_mode": str(base["lam_mode"]),
        "center": str(base["center"]),
        "terminal": str(base["terminal"]),
        "b1": bool(base["b1"]), "b2": bool(base["b2"]), "b3": bool(base["b3"]),
        "g1": bool(base["g1"]), "g2": bool(base["g2"]),
        "g3": bool(base["g3"]), "g4": bool(base["g4"]),
    }


def switch_readback(cell: dict[str, Any]) -> dict[str, Any]:
    """The seven switches READ BACK off the constructed cell record, and whether all seven are false."""
    values = {name: bool(cell["cell"][name]) for name in SWITCHES}
    return {"switches": values, "all_seven_false": not any(values.values())}


def table_row(cell_id: str, cell: dict[str, Any]) -> dict[str, Any]:
    """The v4 format's own columns, in its exact order, imported from `rounds_goal_v4.TABLE_COLUMNS`
    so the two formats cannot drift apart."""
    columns = cell["v4_columns"]
    bands = cell["bands"]
    le, gt = bands["tilt_le_ref"], bands["tilt_gt_ref"]
    steps = max(1, int(cell["n_control_steps"]))
    return {
        "cell_id": cell_id,
        "reach_all": cell["reach"],
        "d_min_p50": columns["ALL"]["d_min_p50"],
        "inside_r05_share": columns["ALL"]["inside_r_share"],
        "coll_all": cell["collision"],
        "timeout_all": cell["timeout"],
        "omega_p50": columns["ALL"]["omega_p50"],
        "smooth_mean_du": columns["ALL"]["smooth_mean_du"],
        "reach_le60": le.get("reach"),
        "coll_le60": le.get("collision"),
        "n_le60": le["n"],
        "reach_gt60": gt.get("reach"),
        "coll_gt60": gt.get("collision"),
        "n_gt60": gt["n"],
        "wall_ms_per_step": 1000.0 * float(cell["wall_s"]) / steps,
    }


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    def fmt(key: str, value: Any) -> str:
        if value is None:
            return "—"
        if key in ("cell_id", "variant", "terminal"):
            return str(value)
        if key in ("n_le60", "n_gt60", "n_decisions", "n", "relaxed_only_n"):
            return str(int(value))
        if key == "deployed_rederivation_agrees":
            return str(bool(value)).lower()
        if key == "wall_ms_per_step":
            return f"{float(value):.1f}"
        if key == "ess_p50":
            return f"{float(value):.1f}"
        return f"{float(value):.4f}"

    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(fmt(c, row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def append_log(section: str) -> None:
    """APPEND-ONLY. A table once written is never edited and never deleted."""
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open("a", encoding="utf-8") as handle:
        handle.write(section.rstrip("\n") + "\n\n")


def write_report(path: Path, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(f"{path.stem}__{time.strftime('%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def preamble(mppi_config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Everything every stage records before it touches the GPU."""
    v5 = mppi_config["v5"]
    scale = v5["scale"]
    pool_path = REPO / mppi_config["screen"]["pool"]
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if pool_sha[:8] != str(mppi_config["screen"]["pool_sha8"]):
        raise SystemExit(f"STOP: pool sha8 {pool_sha[:8]} != {mppi_config['screen']['pool_sha8']}.")
    launch = vram()
    if launch["free_mib"] < VRAM_FLOOR_MIB:
        raise SystemExit(f"STOP: launch free VRAM {launch['free_mib']:.0f} MiB < {VRAM_FLOOR_MIB:.0f}.")
    return {
        "what": f"charter-'v5' STAGE {stage}",
        "version": __version__,
        "process": proc_state(),
        "base_cell": dict(v5["base_cell"]),
        "base_config_naming_resolution": (
            "The charter names 'the v3 timeout-dominant cell' and ALSO fixes B1-B3 off / G1-G4 off. "
            "With the switches OFF that configuration is the charter-'v2' controller, retained at "
            "data/runs/v2.8.4/mppi_screen_v3/cell__N1024_lam0.05_C100000_m1_H40_sig1.json "
            "(reach 0.0000, collision 0.2400, timeout 0.7600 at n=400), NOT the same-named row at "
            "data/runs/v2.8.4/mppi_v3/cell__N1024_lam0.05_C100000_m1_H40_sig1.json "
            "(reach 0.0000, collision 0.2225, timeout 0.7775 at n=400), which has B1=B2=B3=true. "
            "THE SWITCHES ARE DECISIVE, so the mppi_screen_v3 row is the base."
        ),
        "scale": {"n_scenes": int(scale["n_scenes"]), "ebs": int(scale["ebs"]),
                  "seed": int(scale["seed"])},
        "pool": {"path": str(pool_path), "sha256": pool_sha, "sha8": pool_sha[:8],
                 "order": f"the FIRST {int(scale['n_scenes'])} scenes of the pool, in pool order"},
        "launch_vram": launch,
        "peak_cap_mib": 0.5 * launch["free_mib"],
        "vram_floor_mib": VRAM_FLOOR_MIB,
        "immutable_before": dir_state(),
    }


def run_one(
    *,
    cell_id: str,
    mppi_config: dict[str, Any],
    tilt: np.ndarray,
    theta_ref: float,
    device: torch.device,
    dtype: torch.dtype,
    cap_mib: float,
    polls: list[dict[str, Any]],
    n_samples: int,
    sample_chunk: int,
    lam: float | None = None,
    cascade: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one cell with the base configuration, guarded by the poll / floor / cap rules."""
    v5 = mppi_config["v5"]
    scale = v5["scale"]
    kwargs = base_kwargs(mppi_config)
    if lam is not None:
        kwargs["lam"] = float(lam)
    before = guarded_poll(cell_id, polls)
    capture: dict[str, Any] = {}
    try:
        cell = run_cell(
            pool_path=REPO / mppi_config["screen"]["pool"],
            n_scenes=int(scale["n_scenes"]),
            ebs=int(scale["ebs"]),
            seed=int(scale["seed"]),
            n_samples=int(n_samples),
            sample_chunk=int(sample_chunk),
            tilt_deg=tilt,
            tilt_split_deg=theta_ref,
            label=cell_id,
            out_dir=OUT_DIR,
            device=device,
            dtype=dtype,
            mppi_config=mppi_config,
            v4_columns=True,
            cascade=cascade,
            endpoint_probe_radii=[float(r) for r in v5["stage1"]["endpoint_probe_radii"]],
            capture=capture,
            **kwargs,
        )
    except torch.cuda.OutOfMemoryError as exc:                            # pragma: no cover
        raise SystemExit(f"STOP: CUDA OOM on cell {cell_id}: {exc}")
    after = vram()
    peak = float(cell["peak_cuda_reserved_mib"])
    print(f"[{cell_id}] VRAM after {after}  peak alloc {cell['peak_cuda_alloc_mib']} MiB "
          f"reserved {peak} MiB (cap {cap_mib:.0f})", flush=True)
    if peak > cap_mib:
        raise SystemExit(f"STOP: peak reserved {peak:.0f} MiB over cap {cap_mib:.0f} MiB.")

    # Stage 3 re-scores THESE rollouts; they are stored now because a rollout cannot be recovered
    # after the fact and Stage 3 is forbidden from producing a second set.
    stored = save_rollouts(
        OUT_DIR / f"rollouts__{cell_id}.npz",
        capture["system"], capture["result"].trajectories, capture["result"].episode_rows,
        tilt_deg=tilt,
    )
    readback = switch_readback(cell)
    if not readback["all_seven_false"]:
        raise SystemExit(
            f"STOP: cell {cell_id} did not resolve all seven switches false: {readback['switches']}"
        )
    record = {
        "cell_id": cell_id,
        "N": int(n_samples),
        "sample_chunk": int(sample_chunk),
        "sample_chunk_note": (
            "0 = all N rolled at once. Samples are INDEPENDENT, so chunking is EXACTLY equivalent — "
            "it trades wall time for peak VRAM and changes nothing else."
        ),
        "lam": float(kwargs["lam"]),
        "variant": "cascaded" if cascade is not None else "vanilla",
        "switch_readback": readback,
        "outcome_counts": cell["outcome_counts"],
        "ess": cell["ess"]["active"],
        "endpoint_probe": cell["endpoint_probe"],
        "degenerate_step_frac": cell["degenerate"]["step_frac_over_active_window"],
        "v4_columns": cell["v4_columns"],
        "cascade": cell.get("cascade"),
        "rollouts": stored,
        "vram": {"before": before, "after": after,
                 "peak_cuda_alloc_mib": cell["peak_cuda_alloc_mib"],
                 "peak_cuda_reserved_mib": peak, "cap_mib": cap_mib},
        "wall_s": cell["wall_s"],
        "n_control_steps": cell["n_control_steps"],
    }
    return cell, record


def setup(mppi_config: dict[str, Any], stage: str, device_name: str, dtype_name: str, threads: int):
    torch.set_num_threads(int(threads))
    device = torch.device(device_name)
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    report = preamble(mppi_config, stage)
    print(f"pid {os.getpid()} ppid {os.getppid()} state {report['process']['proc_state']}", flush=True)
    print(f"pool sha8 {report['pool']['sha8']} OK", flush=True)
    print(f"launch VRAM {report['launch_vram']}  cap {report['peak_cap_mib']:.0f} MiB", flush=True)
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])
    tilt = spawn_tilt_deg(
        REPO / mppi_config["screen"]["pool"], int(mppi_config["v5"]["scale"]["n_scenes"]), mppi_config
    )
    report["tilt_split"] = {
        "theta_ref_deg": theta_ref, "convention": "<= / >",
        "definition": "degrees(arccos(clip(R(q0)[2,2], -1, 1))) — the documented fallback; the pool "
                      "manifest carries no per-scene spawn-tilt field",
        "policy": "born-inverted episodes are NEVER filtered out of a headline; the bands are "
                  "ADDITIONAL columns with their own denominators",
        "n_le": int((tilt <= theta_ref).sum()), "n_gt": int((tilt > theta_ref).sum()),
        "tilt_min": float(tilt.min()), "tilt_median": float(np.median(tilt)),
        "tilt_max": float(tilt.max()),
    }
    print(f"tilt split at {theta_ref} deg: n_le {report['tilt_split']['n_le']} "
          f"n_gt {report['tilt_split']['n_gt']}", flush=True)
    return report, device, dtype, theta_ref, tilt


def finish(report: dict[str, Any], t0: float) -> None:
    report["immutable_after"] = dir_state()
    report["immutable_unchanged"] = report["immutable_before"] == report["immutable_after"]
    report["wall_s"] = round(time.time() - t0, 2)
    print(f"immutable dirs unchanged: {report['immutable_unchanged']}", flush=True)


# =================================================================================================
# STAGE 0 — base-config verification (a reproduction check, NOT a charter cell)
# =================================================================================================
def stage0(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    """Re-run the BASE cell at the charter's scale and compare it, metric for metric, against the
    retained mppi_screen_v3 row the naming resolution selects.

    THIS IS NOT A CHARTER CELL AND IS NEVER TABULATED WITH THE STAGE-1 CELLS. It exists to (i) show
    the seven switches resolving false on a constructed controller, (ii) evidence that the base
    resolution picked the row it claims, and (iii) MEASURE that the Stage-1 endpoint probe, which runs
    with the flag ON here, changes no number — the probe's inertness is measured, not asserted.
    """
    t0 = time.time()
    report, device, dtype, theta_ref, tilt = setup(
        mppi_config, "0 (base-config verification)", args.device, args.dtype, args.threads
    )
    base = mppi_config["v5"]["base_cell"]
    polls: list[dict[str, Any]] = []
    cell, record = run_one(
        cell_id="s0_base_repro_N1024", mppi_config=mppi_config, tilt=tilt, theta_ref=theta_ref,
        device=device, dtype=dtype, cap_mib=report["peak_cap_mib"], polls=polls,
        n_samples=int(base["n_samples"]), sample_chunk=0,
    )
    retained = json.loads((REPO / str(base["retained_row"])).read_text(encoding="utf-8"))
    sibling = json.loads((REPO / str(base["sibling_row"])).read_text(encoding="utf-8"))
    keys = ("reach", "collision", "timeout", "oob", "stuck", "cps",
            "coll_obstacle", "coll_band_lower", "coll_band_upper", "saturation_rate")
    comparison = {
        key: {"v5_rerun": float(cell[key]), "retained": float(retained[key]),
              "abs_diff": abs(float(cell[key]) - float(retained[key]))}
        for key in keys
    }
    report.update({
        "status": "NOT A CHARTER CELL — a base-config verification and probe-inertness measurement. "
                  "It is never tabulated with the Stage-1 cells.",
        "polls": polls,
        "cell": record,
        "switch_readback": record["switch_readback"],
        "base_row_used": {
            "path": str(base["retained_row"]),
            "reach": retained["reach"], "collision": retained["collision"],
            "timeout": retained["timeout"], "cell": retained["cell"],
        },
        "base_row_rejected_sibling": {
            "path": str(base["sibling_row"]),
            "reach": sibling["reach"], "collision": sibling["collision"],
            "timeout": sibling["timeout"], "cell": sibling["cell"],
            "why_rejected": "B1 = B2 = B3 = true; the charter fixes B1-B3 OFF and the switches are "
                            "decisive.",
        },
        "reproduction_vs_retained": comparison,
        "reproduction_exact": all(v["abs_diff"] == 0.0 for v in comparison.values()),
        "probe_inertness": (
            "this re-run had the Stage-1 endpoint probe ON. Every metric above matching the retained "
            "row is the MEASUREMENT that the probe changes nothing; it draws no random numbers and "
            "feeds nothing back into the control."
        ),
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "stage0_base_verification.json", report)
    lines = [
        "# v2.8.4 charter-\"v5\" build log — sample budget, cascaded MPPI, relaxed companion",
        "",
        "APPEND-ONLY. No table written below is ever edited or deleted.",
        "",
        "Artifacts: `data/runs/v2.8.4/mppi_v5/`. Pool "
        f"`{report['pool']['sha8']}`, first {report['scale']['n_scenes']} scenes in pool order, "
        f"seed {report['scale']['seed']}, ebs {report['scale']['ebs']}.",
        "",
        "## Stage 0 — base-config verification (NOT a charter cell)",
        "",
        "### The base-config naming resolution",
        "",
        report["base_config_naming_resolution"],
        "",
        "| candidate row | B1 | B2 | B3 | reach | collision | timeout | used as base |",
        "|---|---|---|---|---|---|---|---|",
        f"| `{base['retained_row']}` | false | false | false | "
        f"{retained['reach']:.4f} | {retained['collision']:.4f} | {retained['timeout']:.4f} | YES |",
        f"| `{base['sibling_row']}` | true | true | true | "
        f"{sibling['reach']:.4f} | {sibling['collision']:.4f} | {sibling['timeout']:.4f} | no |",
        "",
        "(The mppi_screen_v3 record predates the B1/B2/B3 fields, so they are absent from its `cell` "
        "block; absent is off — that screen shipped with all three defaulting false.)",
        "",
        "### The seven switches, read back off the constructed controller",
        "",
        "| " + " | ".join(SWITCHES) + " | all seven false |",
        "|" + "|".join(["---"] * (len(SWITCHES) + 1)) + "|",
        "| " + " | ".join(str(record["switch_readback"]["switches"][s]).lower() for s in SWITCHES)
        + f" | {str(record['switch_readback']['all_seven_false']).lower()} |",
        "",
        "### Reproduction of the retained base row (probe ON)",
        "",
        "| metric | v5 re-run | retained | abs diff |",
        "|---|---|---|---|",
        *[f"| {k} | {v['v5_rerun']:.4f} | {v['retained']:.4f} | {v['abs_diff']:.2e} |"
          for k, v in comparison.items()],
        "",
        f"Exact on every metric: **{report['reproduction_exact']}**. The re-run carried the Stage-1 "
        "endpoint probe switched ON, so this is the measurement that the added instrumentation is "
        "inert.",
        "",
        f"VRAM: launch free {report['launch_vram']['free_mib']:.0f} MiB, cap "
        f"{report['peak_cap_mib']:.0f} MiB, cell peak reserved "
        f"{record['vram']['peak_cuda_reserved_mib']:.0f} MiB. "
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. "
        f"Immutable dirs unchanged: {report['immutable_unchanged']}.",
        "",
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


# =================================================================================================
# STAGE 1 — sample budget
# =================================================================================================
def stage1(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    t0 = time.time()
    report, device, dtype, theta_ref, tilt = setup(
        mppi_config, "1 (sample budget)", args.device, args.dtype, args.threads
    )
    v5 = mppi_config["v5"]
    radii = [float(r) for r in v5["stage1"]["endpoint_probe_radii"]]
    polls: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for spec in v5["stage1"]["cells"]:
        n_samples, chunk = int(spec["n_samples"]), int(spec["sample_chunk"])
        cell_id = f"s1_N{n_samples}"
        cell, record = run_one(
            cell_id=cell_id, mppi_config=mppi_config, tilt=tilt, theta_ref=theta_ref,
            device=device, dtype=dtype, cap_mib=report["peak_cap_mib"], polls=polls,
            n_samples=n_samples, sample_chunk=chunk,
        )
        row = table_row(cell_id, cell)
        probe = cell["endpoint_probe"]
        row["ess_p50"] = cell["ess"]["active"].get("p50")
        row["best_end_in_r0.5"] = probe["shares_inside_r"][f"{radii[0]:g}"]
        row["best_end_in_r1"] = probe["shares_inside_r"][f"{radii[1]:g}"]
        row["n_decisions"] = probe["n_decision_steps"]
        rows.append(row)
        records.append({**record, "row": row})
        print("[" + cell_id + "] " + json.dumps(row), flush=True)

    columns = V4_TABLE_COLUMNS + STAGE1_EXTRA
    table = markdown_table(rows, columns)
    print("\n" + table, flush=True)

    # ---- THE GATE, binding in BOTH directions ---------------------------------------------------
    any_reach = [r["cell_id"] for r in rows if float(r["reach_all"]) > 0.0]
    gate = {
        "rule": "if ANY Stage-1 cell attains reach_all > 0, STOP after Stage 1 and report — standard "
                "MPPI works at scale and Stage 2 is not needed. If all three read reach_all = 0.0000, "
                "proceed.",
        "cells_with_reach_above_zero": any_reach,
        "decision": "STOP after Stage 1" if any_reach else "PROCEED to Stage 2",
        "proceed": not any_reach,
        "reach_all_by_cell": {r["cell_id"]: float(r["reach_all"]) for r in rows},
    }
    report.update({"polls": polls, "columns": list(columns), "table_markdown": table,
                   "rows": rows, "cells": records, "gate": gate,
                   "table_note": "the v4 format's own columns, imported from "
                                 "rounds_goal_v4.TABLE_COLUMNS, then the charter's four Stage-1 "
                                 "additions. `inside_r05_share` is the v4 TIME-inside share of the "
                                 "flown trajectory; `best_end_in_r*` is the charter's separate "
                                 "best-cost-sample ENDPOINT share over decisions. They are different "
                                 "quantities and are never conflated."})
    finish(report, t0)
    path = write_report(OUT_DIR / "stage1_sample_budget.json", report)
    lines = [
        "## Stage 1 — sample budget (the last card inside standard MPPI)",
        "",
        "Base configuration held fixed (lambda 0.05, C_crash 1e5, m=1, H=40, OU noise, hover "
        "centring, settling terminal, B1-B3 off, G1-G4 off, sigma 1.0, seed 42); N is the only axis. "
        f"n = {report['scale']['n_scenes']}, ebs = {report['scale']['ebs']}. All seven switches read "
        "back false on every cell.",
        "",
        table,
        "",
        "`inside_r05_share` is the v4 format's TIME-inside share of the flown trajectory. "
        "`best_end_in_r0.5` / `best_end_in_r1` are the charter's separate column: the share of "
        "DECISIONS whose argmin-cost sample ENDS inside r of the goal, over the active decision steps "
        "of every scored episode. Both radii are reporting probes read from "
        "`mppi.v5.stage1.endpoint_probe_radii`; no gate, ranking or selection is taken against them.",
        "",
        "### Chunking and VRAM, per cell",
        "",
        "| cell | N | sample_chunk | peak alloc MiB | peak reserved MiB | cap MiB | free before | "
        "free after | wall s |",
        "|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['cell_id']} | {r['N']} | {r['sample_chunk']} | "
          f"{r['vram']['peak_cuda_alloc_mib']:.1f} | {r['vram']['peak_cuda_reserved_mib']:.1f} | "
          f"{r['vram']['cap_mib']:.0f} | {r['vram']['before']['free_mib']:.0f} | "
          f"{r['vram']['after']['free_mib']:.0f} | {r['wall_s']:.1f} |" for r in records],
        "",
        "Chunking is EXACTLY equivalent — samples are independent, so it trades wall time for peak "
        "memory and changes nothing else. The N = 8192 cell is chunked as the charter requires.",
        "",
        "### The gate",
        "",
        gate["rule"],
        "",
        "reach_all by cell: "
        + ", ".join(f"`{k}` {v:.4f}" for k, v in gate["reach_all_by_cell"].items())
        + f". Cells with reach_all > 0: {gate['cells_with_reach_above_zero'] or 'none'}.",
        "",
        f"**Gate decision: {gate['decision']}.**",
        "",
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. "
        f"Immutable dirs unchanged: {report['immutable_unchanged']}. "
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    print(f"GATE: {gate['decision']}", flush=True)
    return 0


# =================================================================================================
# STAGE 2 — cascaded MPPI
# =================================================================================================
def require_stage1(kind: str) -> dict[str, Any]:
    path = OUT_DIR / "stage1_sample_budget.json"
    if not path.exists():
        raise SystemExit(
            f"STOP: Stage {kind} may not begin before Stage 1 is complete and written "
            f"({path} does not exist). The charter's order is strict."
        )
    stage1_report = json.loads(path.read_text(encoding="utf-8"))
    gate = stage1_report["gate"]
    if not gate["proceed"]:
        raise SystemExit(
            f"STOP: the Stage-1 gate went the other way — {gate['cells_with_reach_above_zero']} "
            f"attained reach_all > 0, so the charter STOPS after Stage 1 and Stage 2 is not needed."
        )
    return stage1_report


def stage2_smoke(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    """The charter's CPU smoke, run BEFORE the Stage-2 screen."""
    from src.frameworks.mppi.cascade import cascade_smoke

    t0 = time.time()
    require_stage1("2 (smoke)")
    report = preamble(mppi_config, "2 smoke (cascaded MPPI, CPU)")
    v5 = mppi_config["v5"]
    stage2 = v5["stage2"]
    torch.set_num_threads(int(args.threads))
    config = effective_config(mppi_config)
    base = v5["base_cell"]
    system, framework, _, _ = build_framework(
        config, mppi_config,
        n_samples=int(base["n_samples"]), sample_chunk=0, seed=int(v5["scale"]["seed"]),
        cascade=stage2, device=torch.device("cpu"), dtype=torch.float64,
        **base_kwargs(mppi_config),
    )
    smoke = cascade_smoke(system, config, framework.controller, stage2["smoke"])
    report.update({
        "variant_status": str(stage2["variant_status"]),
        "smoke": smoke,
        "verdicts": {
            "hover_fixed_point_stable": {
                "max_position_drift_m": smoke["a_hover_fixed_point"]["max_position_drift_m"],
                "max_speed_m_s": smoke["a_hover_fixed_point"]["max_speed_m_s"],
                "max_angular_rate_rad_s": smoke["a_hover_fixed_point"]["max_angular_rate_rad_s"],
                "max_tilt_deg": smoke["a_hover_fixed_point"]["max_tilt_deg"],
            },
            "straight_line_open_loop_reaches_position_radius":
                smoke["b_straight_line_open_loop"]["reaches_position_radius"],
            "straight_line_closed_outer_loop_reaches_position_radius":
                smoke["c_straight_line_closed_outer_loop"]["reaches_position_radius"],
            "wrap_max_abs_rotor_difference_N":
                smoke["d_wrap_reproduces_shipped_controller"]["max_abs_rotor_difference_N"],
            "hover_command_max_abs_difference_N":
                smoke["e_hover_command"]["max_abs_difference_N"],
        },
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "stage2_smoke.json", report)
    hover = smoke["a_hover_fixed_point"]
    opened = smoke["b_straight_line_open_loop"]
    closed = smoke["c_straight_line_closed_outer_loop"]
    wrap = smoke["d_wrap_reproduces_shipped_controller"]
    lines = [
        "## Stage 2 — CASCADED MPPI: CPU smoke",
        "",
        "**THIS IS A CASCADED VARIANT.** " + str(stage2["variant_status"]).strip(),
        "",
        "Plan variable: the desired WORLD acceleration `a_des` in R^3. The charter's "
        "`(total thrust T, desired attitude q_des)` are read off the shipped cascade's own lines — "
        "`T = clamp(f_des . b3, 0)` and `q_des`'s body-up axis `= f_des / ||f_des||` with "
        "`f_des = mass (a_des + gravity e3)` — because both are functions of the same `f_des` and the "
        "shipped attitude error `b3 x b3_des` is yaw-blind. Inner loop: `System.lqr_action`, the PD "
        "attitude controller `src/common/filter_backup.py` names in its first sentence, reached "
        "through `filter_backup.override_gains`; the attitude gains are never touched and every gain "
        "is read from `config['lqr']`.",
        "",
        "| check | measurement | read against |",
        "|---|---|---|",
        f"| (a) hover fixed point, {hover['steps']} steps at `a_des = 0` | position drift "
        f"{hover['max_position_drift_m']:.3e} m, speed {hover['max_speed_m_s']:.3e} m/s, rate "
        f"{hover['max_angular_rate_rad_s']:.3e} rad/s, tilt {hover['max_tilt_deg']:.3e} deg | "
        "exact stationarity |",
        f"| (b) straight line, OPEN LOOP | closest approach "
        f"{opened['closest_approach_m']:.4f} m, final speed {opened['final_speed_m_s']:.4f} m/s | "
        f"`env.goal_radius` = {opened['goal_radius_read_from_config']} → reaches: "
        f"**{opened['reaches_position_radius']}** |",
        f"| (c) straight line, closed outer loop (supporting) | closest approach "
        f"{closed['closest_approach_m']:.4f} m | reaches: {closed['reaches_position_radius']} |",
        f"| (d) wrap vs `system.lqr_action` on {wrap['n_probe_states']} random states | max rotor "
        f"difference {wrap['max_abs_rotor_difference_N']:.3e} N | forces up to "
        f"{wrap['max_abs_rotor_force_N']} N |",
        f"| (e) hover command | {smoke['e_hover_command']['max_abs_difference_N']:.3e} N from "
        f"`mass*gravity/rotor_dim` = {smoke['e_hover_command']['expected_trim_per_rotor_N']} N | "
        "the system's own trim |",
        "",
        "(b) is open loop in the OUTER loop: the `a_des` sequence is fixed at t = 0 and never "
        "re-computed from the state, while the inner attitude PD is closed — that is the cascade's own "
        "structure. (c) is reported as supporting evidence and never as a substitute for (b).",
        "",
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. CPU only, no GPU "
        f"was used. Immutable dirs unchanged: {report['immutable_unchanged']}. "
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


def stage2(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    t0 = time.time()
    require_stage1("2")
    if not (OUT_DIR / "stage2_smoke.json").exists():
        raise SystemExit("STOP: the charter orders 'smoke first'; run --stage 2smoke before --stage 2.")
    report, device, dtype, theta_ref, tilt = setup(
        mppi_config, "2 (cascaded MPPI screen)", args.device, args.dtype, args.threads
    )
    v5 = mppi_config["v5"]
    stage2_cfg = v5["stage2"]
    polls: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for spec in stage2_cfg["cells"]:
        n_samples, lam = int(spec["n_samples"]), float(spec["lam"])
        cell_id = f"s2_cascade_N{n_samples}_lam{lam:g}"
        cell, record = run_one(
            cell_id=cell_id, mppi_config=mppi_config, tilt=tilt, theta_ref=theta_ref,
            device=device, dtype=dtype, cap_mib=report["peak_cap_mib"], polls=polls,
            n_samples=n_samples, sample_chunk=int(spec["sample_chunk"]), lam=lam,
            cascade=stage2_cfg,
        )
        row = table_row(cell_id, cell)
        row["variant"] = "cascaded"
        row["ess_p50"] = cell["ess"]["active"].get("p50")
        rows.append(row)
        records.append({**record, "row": row})
        print("[" + cell_id + "] " + json.dumps(row), flush=True)

    table = markdown_table(rows, STAGE2_COLUMNS)
    print("\n" + table, flush=True)
    report.update({
        "polls": polls, "columns": list(STAGE2_COLUMNS), "table_markdown": table,
        "rows": rows, "cells": records,
        "variant_status": str(stage2_cfg["variant_status"]),
        "separation": "every row above carries an explicit `variant` column reading `cascaded`. These "
                      "numbers are a SEPARATE baseline and must never be blended with, averaged with "
                      "or substituted for vanilla MPPI numbers.",
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "stage2_cascade_screen.json", report)
    lines = [
        "## Stage 2 — CASCADED MPPI: the 4-cell screen",
        "",
        "**THIS IS A CASCADED VARIANT — A SEPARATE BASELINE ROW.** "
        + str(stage2_cfg["variant_status"]).strip(),
        "",
        "Every row below carries an explicit `variant` column. These numbers must never be blended "
        "with, averaged with, or substituted for vanilla MPPI numbers, and a cascaded row may never "
        "sit in a table with a vanilla row unless that column is present.",
        "",
        f"N x lambda at n = {report['scale']['n_scenes']}; everything else is the base configuration "
        "(C_crash 1e5, m=1, H=40, OU noise, hover centring, settling terminal, sigma 1.0, seed 42, "
        "B1-B3 off, G1-G4 off — all seven read back false on every cell).",
        "",
        table,
        "",
        "### Chunking and VRAM, per cell",
        "",
        "| cell | N | lambda | sample_chunk | peak alloc MiB | peak reserved MiB | cap MiB | "
        "free before | free after | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['cell_id']} | {r['N']} | {r['lam']:g} | {r['sample_chunk']} | "
          f"{r['vram']['peak_cuda_alloc_mib']:.1f} | {r['vram']['peak_cuda_reserved_mib']:.1f} | "
          f"{r['vram']['cap_mib']:.0f} | {r['vram']['before']['free_mib']:.0f} | "
          f"{r['vram']['after']['free_mib']:.0f} | {r['wall_s']:.1f} |" for r in records],
        "",
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. "
        f"Immutable dirs unchanged: {report['immutable_unchanged']}. "
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


# =================================================================================================
# STAGE 3 — relaxed-settling companion
# =================================================================================================
def stage3(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    t0 = time.time()
    require_stage1("3")
    stage2_path = OUT_DIR / "stage2_cascade_screen.json"
    if not stage2_path.exists():
        raise SystemExit(
            "STOP: Stage 3 may not begin before Stage 2 is complete and written. The charter's order "
            "is strict, and Stage 3's own gate is 'none — Stage 3 runs regardless', not 'Stage 3 may "
            "run early'."
        )
    report = preamble(mppi_config, "3 (relaxed-settling companion)")
    torch.set_num_threads(int(args.threads))
    v5 = mppi_config["v5"]
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])
    config = effective_config(mppi_config)
    from src.frameworks.jt_pncbf.train import make_system
    system = make_system(config)

    # every rollout set produced by Stages 0-2, in the order they were produced, then the retained
    # external set. NOTHING is re-rolled: this stage only reads stored arrays.
    sources: list[dict[str, Any]] = []
    for stage_file, tag in (("stage0_base_verification.json", "stage0"),
                            ("stage1_sample_budget.json", "stage1"),
                            ("stage2_cascade_screen.json", "stage2")):
        path = OUT_DIR / stage_file
        if not path.exists():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        cells = blob.get("cells") or ([blob["cell"]] if "cell" in blob else [])
        for record in cells:
            sources.append({
                "stage": tag,
                "cell_id": record["cell_id"],
                "variant": record.get("variant", "vanilla"),
                "path": record["rollouts"]["path"],
                "origin": "produced by this v5 run",
            })
    for spec in v5["stage3"]["external_rollouts"]:
        path = REPO / str(spec["path"])
        if not path.exists():
            sources.append({"stage": "external", "cell_id": str(spec["label"]), "variant": "vanilla",
                            "path": str(path), "origin": "RETAINED", "missing": True,
                            "note": str(spec["note"])})
            continue
        sources.append({
            "stage": "external", "cell_id": str(spec["label"]), "variant": "vanilla",
            "path": str(path), "origin": "RETAINED, IMMUTABLE, READ-ONLY", "note": str(spec["note"]),
        })

    scored: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for source in sources:
        if source.get("missing"):
            print(f"[{source['cell_id']}] rollout set absent: {source['path']}", flush=True)
            continue
        rollouts = load_rollouts(Path(source["path"]))
        record = score_two_terminals(system, config, rollouts, tilt_split_deg=theta_ref)
        scored.append({**source, "scoring": record})
        block = record["ALL"]
        rows.append({
            "cell_id": source["cell_id"],
            "variant": source["variant"],
            "n": block["n"],
            "reach_deployed": block["reach_deployed"],
            "reach_relaxed": block["reach_relaxed"],
            "relaxed_only_n": block["relaxed_only_n"],
            "d_min_p50": block["d_min_p50"],
            "d_min_min": block["d_min_min"],
            "deployed_rederivation_agrees": block["deployed_rederivation_agrees_with_harness"],
        })
        print(f"[{source['cell_id']}] deployed {block['reach_deployed']:.4f} "
              f"relaxed {block['reach_relaxed']:.4f} relaxed-only {block['relaxed_only_n']}",
              flush=True)

    columns = ("cell_id", "variant", "n", "reach_deployed", "reach_relaxed", "relaxed_only_n",
               "d_min_p50", "d_min_min")
    table = markdown_table(rows, columns)
    print("\n" + table, flush=True)

    relaxed_only_total = sum(int(r["relaxed_only_n"]) for r in rows)
    any_relaxed = sum(int(round(float(r["reach_relaxed"]) * int(r["n"]))) for r in rows)
    report.update({
        "no_new_rollouts": "Stage 3 integrates no plant, draws no random number and constructs no "
                           "controller. Every array it reads was produced by Stage 0/1/2 or is a "
                           "retained, read-only rollout set.",
        "sources": sources,
        "columns": list(columns),
        "table_markdown": table,
        "rows": rows,
        "scored": scored,
        "relaxed_only_episodes_total": relaxed_only_total,
        "episodes_reaching_relaxed_total": any_relaxed,
    })
    finish(report, t0)
    path = write_report(OUT_DIR / "stage3_relaxed_companion.json", report)

    detail: list[str] = []
    for entry in scored:
        block = entry["scoring"]["ALL"]
        if block["relaxed_only_n"] == 0:
            continue
        stats = block["relaxed_only_at_first_position_satisfying_step"]
        detail += [
            f"#### `{entry['cell_id']}` ({entry['variant']}) — {block['relaxed_only_n']} relaxed-only "
            "episodes, at the first position-satisfying step",
            "",
            "| quantity | n | min | p05 | p25 | p50 | p75 | p95 | max | deployed radius | "
            "frac inside |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *[f"| {name} | {d['n']} | {d['min']:.4f} | {d['p05']:.4f} | {d['p25']:.4f} | "
              f"{d['p50']:.4f} | {d['p75']:.4f} | {d['p95']:.4f} | {d['max']:.4f} | "
              f"{d['deployed_radius_read_from_config']} | {d['frac_inside_deployed_radius']:.4f} |"
              for name, d in (("\\|v\\|", stats["speed_abs_v"]),
                              ("\\|omega\\|", stats["angular_rate_abs_omega"]))],
            "",
        ]
    if not detail:
        detail = [
            "No episode in any scored cell reaches under the relaxed terminal but not the deployed "
            "one, so there is no |v| / |omega| distribution to report. Stated plainly, as the charter "
            "asks.",
            "",
        ]

    lines = [
        "## Stage 3 — relaxed-settling companion (re-scoring only, no new rollouts)",
        "",
        "One rollout set, two predicates. Stage 3 integrates no plant, draws no random number and "
        "constructs no controller: every array below was produced by Stage 0/1/2 or is a retained, "
        "read-only rollout set.",
        "",
        "- **DEPLOYED** (reported first, never omitted) — the eval harness's own `outcome == 'goal'`: "
        "`||p-g|| <= env.goal_radius` AND `||v|| <= env.goal_speed_radius` AND "
        "`||omega|| <= env.goal_angrate_radius`, with collision preempting goal at the same step.",
        "- **RELAXED** (annotated companion only) — `||p-g|| <= env.goal_radius` at any active step; "
        "the position radius is UNCHANGED and read from `env.goal_radius`, the velocity and "
        "angular-rate conditions are REMOVED. Never reported without the deployed number beside it "
        "and never a substitute for it.",
        "",
        f"All three radii read from the merged config: goal_radius "
        f"{config['env']['goal_radius']}, goal_speed_radius {config['env']['goal_speed_radius']}, "
        f"goal_angrate_radius {config['env']['goal_angrate_radius']}. None is typed in code.",
        "",
        table,
        "",
        "`v3_sibling_B1B2B3_on__n100` is the RETAINED `mppi_diag` rollout set: the base cell's "
        "B1-B3-**ON** sibling at n = 100, not the v5 base. It is read-only and is labelled as the "
        "sibling wherever it appears.",
        "",
        "### |v| and |omega| for the relaxed-only reachers",
        "",
        *detail,
        f"pid {report['process']['pid']} state `{report['process']['proc_state']}`. No GPU work. "
        f"Immutable dirs unchanged: {report['immutable_unchanged']}. "
        f"Artifact: `{path.relative_to(REPO)}`.",
    ]
    append_log("\n".join(lines))
    print(f"wrote {path}", flush=True)
    return 0


# =================================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="charter-'v5' staged driver.")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["0", "1", "2smoke", "2", "3"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    mppi_config = load_mppi_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "0": stage0, "1": stage1, "2smoke": stage2_smoke, "2": stage2, "3": stage3,
    }[args.stage](args, mppi_config)


if __name__ == "__main__":
    raise SystemExit(main())
