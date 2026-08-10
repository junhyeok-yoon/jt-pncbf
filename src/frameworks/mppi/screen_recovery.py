"""v2.8.4 MPPI baseline — the charter-"v3" screen and ablations for B1 / B2 / B3.

LINEAGE — the directory names do NOT match the charter's labels. Stated in full so nothing is conflated:

    charter "v1"  = the ORIGINAL 16-cell grid (N x H x lam x C_crash), 8 cells run
                    -> data/runs/v2.8.4/mppi_screen/          IMMUTABLE (this package calls it S1)
    (unlabelled)  = the R1-R4 amendment's 12-cell grid (sigma x lam_rel x H)
                    -> data/runs/v2.8.4/mppi_screen_v2/       IMMUTABLE (S2)
    charter "v2"  = the hover-centred / settling-terminal / control-hold 16-cell grid
                    -> data/runs/v2.8.4/mppi_screen_v3/       IMMUTABLE (S3)
    charter "v3"  = THIS work
                    -> data/runs/v2.8.4/mppi_v3/, docs/versions/v2.8.4/mppi_v3.md

WHAT THE CHARTER'S "v2" MEASURED. Not what the charter's context says. Verbatim from
`data/runs/v2.8.4/mppi_screen_v3/screen_rows.json`: reach is 0.0000 in all 16 cells and in BOTH tilt
bands of every one of them, `goal` appears in no cell's outcome counts, the lowest-collision cell is
0.2040 at tilt < 90 deg against 0.2764 at tilt >= 90 deg (so the low-tilt band fails almost as badly),
and all eight m=4 cells carry identical headline numbers because the degenerate branch fired on
0.9997-1.0000 of decisions. B1/B2/B3 are being applied to a controller that reached the goal at NO tilt.

WHAT THIS SCRIPT RUNS

  SCREEN — the same 16-cell grid as the charter's "v2" (`mppi.screen.grid_v3`, config order,
  N x lam_rel x C_crash x m with H = 40 and sigma = 1.0), with B1, B2 and B3 all ON, n = 400 scenes in
  pool order, ebs 200, seed 42, float32, CUDA, serial, one process.

  ABLATION, REGISTERED — four rows (v2 baseline / +B1 / +B1+B2 / +B1+B2+B3) at the charter's fallback
  cell, since no Researcher cell selection was supplied: the CONFIG-ORDER FIRST cell with {N=1024, m=4},
  i.e. N=1024, lam=0.05, C_crash=1e3, m=4. That cell is inside the m=4 corner where the degenerate branch
  fired on ~100% of decisions, i.e. the one configuration in which MPPI provably never runs its update.

  ABLATION, DIAGNOSTIC — the identical four rows at the config-order first cell with {N=1024, m=1}.
  NON-REGISTERED, run in addition so the components are also measured where the optimiser is live. The
  two tables are reported separately; neither substitutes for the other.

Every reported row carries THREE metric sets — overall, spawn tilt <= theta_ref, spawn tilt > theta_ref —
each with its own episode count, so every rate has its denominator. theta_ref is read from
`mppi.recovery.theta_ref_deg` (60 deg), the same field B1 and B2 act on.

SPAWN-TILT SOURCE — A DEVIATION FROM THE CHARTER, FLAGGED. The charter requires the split use the spawn
tilt recorded in the pool manifest. `data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_
seed823456.manifest.json` carries no per-scene attitude field of any kind (its leaves are the sampler
parameters, the screening record and the removed-scene lists), and the pickle holds `Scene` objects whose
only attitude entry is the raw `initial_attitude_quat`. There is no recorded tilt to read. The documented
fallback is therefore used — `cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2]`,
`tilt_deg = degrees(arccos(clip(cos_tilt, -1, 1)))`, the established definition at
`scripts/analysis/v282_alpha_tilt_screen.py:93-94` — and the deviation is recorded in every artifact.
Note also that the charter's "v2" screen split its bands at 90 deg and this one splits at 60 deg, so the
two screens' bands are NOT directly comparable.

CONCURRENCY — THE REGISTERED GAMMA RE-SCORE HAS ABSOLUTE PRIORITY. Before every cell:
  (a) is any `v284_gamma_*` python process alive (`pgrep -f`, self-excluded);
  (b) is `data/runs/v2.8.4/gamma_rescore/` advancing (file count and newest mtime).
If a registered process is alive AND the projected peak would take free VRAM below the floor, this
script WAITS rather than proceeding. Two consecutive polls with progress blocked and no headroom STOP
the run. Every poll is written into the screen JSON.

VRAM. Launch only if free >= --min-free-mib; the measured peak must stay under HALF the free margin at
launch (--sample-chunk caps the rollout batch along the sample axis if it ever does not, which is exactly
equivalent — samples are independent); free/used/total are logged before AND after every cell; any CUDA
OOM stops the run.

NO SELECTION, NO RANKING. The tables are emitted in config order and nothing is sorted, scored against a
bar, or chosen. Selection is the Researcher's.

Run:  python -m src.frameworks.mppi.screen_recovery --out-dir data/runs/v2.8.4/mppi_v3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.quadrotor_3d import _quat_to_R
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.frameworks.jt_pncbf.train import make_system
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    effective_config,
    load_mppi_config,
    run_cell,
)


GAMMA_RESCORE_DIR = REPO / "data/runs/v2.8.4/gamma_rescore"
GAMMA_PROCESS_PATTERN = "v284_gamma"
IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
)


# ---- VRAM ----------------------------------------------------------------------------------------
def free_vram_mib() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


# ---- the registered gamma re-score ----------------------------------------------------------------
def gamma_processes() -> list[dict[str, str]]:
    """Live `v284_gamma_*` python processes, this process and its own shell excluded.

    `pgrep -f` matches the whole command line, so a shell whose command line merely CONTAINS the pattern
    would match itself. Both are filtered: our own pid/ppid, and any line that is not a python process."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", GAMMA_PROCESS_PATTERN], capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        return []
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines():
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in mine or "python" not in command:
            continue
        found.append({"pid": pid, "command": command[:200]})
    return found


def gamma_progress() -> dict[str, Any]:
    """File count and newest mtime of the registered re-score's output directory."""
    if not GAMMA_RESCORE_DIR.exists():
        return {"exists": False, "n_files": 0, "newest_mtime": None, "newest_name": None}
    files = sorted(p for p in GAMMA_RESCORE_DIR.iterdir() if p.is_file())
    newest = max(files, key=lambda p: p.stat().st_mtime, default=None)
    return {
        "exists": True,
        "n_files": len(files),
        "newest_mtime": newest.stat().st_mtime if newest else None,
        "newest_name": newest.name if newest else None,
        "dir_mtime": GAMMA_RESCORE_DIR.stat().st_mtime,
    }


def poll_gamma(projected_peak_mib: float, min_free_mib: float, note: str) -> dict[str, Any]:
    """One priority poll. `blocked` is True only when a registered process is alive AND the projected
    peak would take free VRAM below the floor — i.e. only when yielding is actually required."""
    processes = gamma_processes()
    vram = free_vram_mib()
    headroom = vram["free_mib"] - projected_peak_mib
    blocked = bool(processes) and headroom < min_free_mib
    return {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "gamma_processes_alive": len(processes),
        "gamma_processes": processes,
        "gamma_rescore_progress": gamma_progress(),
        "vram": vram,
        "projected_peak_mib": round(projected_peak_mib, 1),
        "headroom_after_projection_mib": round(headroom, 1),
        "min_free_mib": min_free_mib,
        "blocked": blocked,
    }


# ---- the tilt array -------------------------------------------------------------------------------
def spawn_tilt_deg(pool_path: Path, n_scenes: int, mppi_config: dict) -> np.ndarray:
    """Spawn tilt of the pool's first `n_scenes` initial states, in degrees.

    The pool manifest carries NO per-scene spawn-tilt field, so this is the documented FALLBACK, flagged
    as a deviation from the charter wherever it is reported:

        cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2]
        tilt_deg = degrees(arccos(clip(cos_tilt, -1, 1)))

    the established definition at `scripts/analysis/v282_alpha_tilt_screen.py:93-94`. `x0` is built
    exactly as the eval harness builds it, so the array is aligned scene-for-scene with the episode rows.
    """
    config = effective_config(mppi_config)
    system = make_system(config)
    scenes = load_pool(pool_path).scenes[:n_scenes]
    batched = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float64)
    x0 = system.wrap_state(initial_states_from_batch(batched))
    cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2].numpy()
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def manifest_has_tilt_field(pool_path: Path) -> dict[str, Any]:
    """Whether the pool's manifest records a per-scene spawn tilt. Recorded so the fallback is evidenced
    rather than asserted."""
    manifest_path = pool_path.with_suffix("").with_suffix(".manifest.json")
    if not manifest_path.exists():
        manifest_path = pool_path.parent / (pool_path.stem + ".manifest.json")
    if not manifest_path.exists():
        return {"manifest_found": False, "path": str(manifest_path), "has_tilt_field": False}
    text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(text)

    def leaves(node: Any, path: str = "") -> list[str]:
        if isinstance(node, dict):
            return [k for key, value in node.items() for k in leaves(value, f"{path}/{key}")]
        if isinstance(node, list):
            return [path]
        return [path]

    keys = leaves(manifest)
    hits = [k for k in keys
            if any(token in k.lower() for token in ("tilt", "quat", "attitude", "roll", "pitch"))]
    return {
        "manifest_found": True,
        "path": str(manifest_path.relative_to(REPO)),
        "n_leaf_entries": len(keys),
        "attitude_like_keys": hits,
        "has_tilt_field": bool(hits),
        "conclusion": "no per-scene spawn-tilt field is recorded in the manifest, so the reporting "
                      "split uses the documented fallback definition — a DEVIATION from the charter"
        if not hits else "the manifest records an attitude-like field; inspect before using the fallback",
    }


# ---- the grids ------------------------------------------------------------------------------------
def screen_cells(screen: dict) -> list[dict]:
    """The 16-cell grid in CONFIG ORDER, nested exactly as `mppi.screen.grid_v3` writes it."""
    grid = screen["grid_v3"]
    return [
        {"N": int(n), "lam": float(lam), "C_crash": float(c), "m": int(hold),
         "H": int(h), "sigma": float(s)}
        for n, lam, c, hold, h, s in product(
            grid["n_samples"], grid["lam"], grid["c_crash"], grid["control_hold"],
            grid["horizon"], grid["sigma"],
        )
    ]


def cell_label(cell: dict) -> str:
    return (f"N{cell['N']}_lam{cell['lam']:g}_C{cell['C_crash']:g}_m{cell['m']}"
            f"_H{cell['H']}_sig{cell['sigma']:g}")


def ablation_cell(block: dict) -> dict:
    spec = block["cell"]
    return {"N": int(spec["n_samples"]), "lam": float(spec["lam"]), "C_crash": float(spec["c_crash"]),
            "m": int(spec["control_hold"]), "H": int(spec["horizon"]), "sigma": float(spec["sigma"])}


# ---- the run --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="charter-'v3' MPPI screen and ablations.")
    parser.add_argument("--out-dir", type=str, default=str(REPO / "data/runs/v2.8.4/mppi_v3"))
    parser.add_argument("--n-scenes", type=int, default=None)
    parser.add_argument("--ebs", type=int, default=None)
    parser.add_argument("--sample-chunk", type=int, default=0)
    parser.add_argument("--min-free-mib", type=float, default=6144.0)
    parser.add_argument("--wait-s", type=float, default=60.0,
                        help="seconds to yield to the registered gamma work before re-polling")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "screen", "ablation"])
    args = parser.parse_args()

    mppi_config = load_mppi_config()
    screen = mppi_config["screen"]
    recovery_cfg = mppi_config["recovery"]
    ablation_cfg = mppi_config["ablation"]
    theta_ref = float(recovery_cfg["theta_ref_deg"])
    n_scenes = int(args.n_scenes if args.n_scenes is not None else screen["n_scenes"])
    ebs = int(args.ebs if args.ebs is not None else screen["ebs"])
    pool_path = REPO / screen["pool"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(int(args.threads))
    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA is not available to this process.")
    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if pool_sha[:8] != str(screen["pool_sha8"]):
        raise SystemExit(
            f"STOP: pool sha256[:8] is {pool_sha[:8]}, expected {screen['pool_sha8']}."
        )

    immutable_before = {
        str(p.relative_to(REPO)): {"mtime": p.stat().st_mtime,
                                   "mtime_iso": time.strftime(
                                       "%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
                                   "n_files": len(list(p.iterdir()))}
        for p in IMMUTABLE_DIRS
    }

    launch_vram = free_vram_mib()
    if launch_vram["free_mib"] < args.min_free_mib:
        raise SystemExit(
            f"STOP: free VRAM {launch_vram['free_mib']:.0f} MiB < required {args.min_free_mib:.0f} MiB."
        )
    alloc_cap_mib = 0.5 * launch_vram["free_mib"]

    # the projection that the priority poll uses before the first cell is MEASURED, not typed: it is the
    # largest peak the charter's "v2" screen recorded on this same grid and pool.
    reference_rows = json.loads(
        (REPO / "data/runs/v2.8.4/mppi_screen_v3/screen_rows.json").read_text(encoding="utf-8")
    )["rows"]
    projected_peak = max(float(r["peak_cuda_reserved_mib"]) for r in reference_rows)

    tilt = spawn_tilt_deg(pool_path, n_scenes, mppi_config)
    tilt_source = manifest_has_tilt_field(pool_path)
    bins = {"n_le_theta_ref": int((tilt <= theta_ref).sum()),
            "n_gt_theta_ref": int((tilt > theta_ref).sum())}

    print(f"pid {os.getpid()}  ppid {os.getppid()}  free {launch_vram['free_mib']:.0f} MiB  "
          f"alloc cap {alloc_cap_mib:.0f} MiB  projected peak {projected_peak:.0f} MiB  "
          f"n {n_scenes}  ebs {ebs}  theta_ref {theta_ref:g} deg  bins {bins}", flush=True)
    print(f"spawn-tilt source: {tilt_source['conclusion']}", flush=True)

    polls: list[dict[str, Any]] = [poll_gamma(projected_peak, args.min_free_mib, "baseline, pre-launch")]
    print(f"poll {polls[-1]['t']} gamma_alive={polls[-1]['gamma_processes_alive']} "
          f"free={polls[-1]['vram']['free_mib']:.0f} blocked={polls[-1]['blocked']}", flush=True)

    state = {"stop_reason": None, "stopped_at": None, "projected_peak": projected_peak}

    def guarded_cell(label: str, cell: dict, *, b1: bool, b2: bool, b3: bool,
                     table: str) -> dict | None:
        """Poll, wait if the registered work needs the GPU, then score one cell. Returns None on STOP."""
        consecutive_blocked = 0
        while True:
            poll = poll_gamma(state["projected_peak"], args.min_free_mib, f"before {table}/{label}")
            polls.append(poll)
            print(f"poll {poll['t']} gamma_alive={poll['gamma_processes_alive']} "
                  f"free={poll['vram']['free_mib']:.0f} headroom={poll['headroom_after_projection_mib']:.0f} "
                  f"blocked={poll['blocked']}", flush=True)
            if not poll["blocked"]:
                break
            consecutive_blocked += 1
            if consecutive_blocked >= 2:
                state["stop_reason"] = (
                    f"two consecutive polls with progress blocked and no VRAM headroom before "
                    f"{table}/{label}: free {poll['vram']['free_mib']:.0f} MiB, projected peak "
                    f"{state['projected_peak']:.0f} MiB, floor {args.min_free_mib:.0f} MiB, with "
                    f"{poll['gamma_processes_alive']} registered gamma process(es) alive. The "
                    f"registered work has absolute priority."
                )
                state["stopped_at"] = f"{table}/{label}"
                return None
            print(f"  yielding {args.wait_s:.0f}s to the registered gamma work", flush=True)
            time.sleep(float(args.wait_s))

        vram_before = free_vram_mib()
        if vram_before["free_mib"] < args.min_free_mib:
            state["stop_reason"] = (
                f"free VRAM fell to {vram_before['free_mib']:.0f} MiB before {table}/{label}"
            )
            state["stopped_at"] = f"{table}/{label}"
            return None
        started = time.time()
        try:
            row = run_cell(
                pool_path=pool_path, n_scenes=n_scenes, ebs=ebs,
                n_samples=int(cell["N"]), horizon=int(cell["H"]), lam=float(cell["lam"]),
                c_crash=float(cell["C_crash"]), sigma=float(cell["sigma"]),
                seed=int(screen["seed"]), sample_chunk=int(args.sample_chunk),
                label=label, out_dir=out_dir, device=device, dtype=dtype,
                mppi_config=mppi_config, control_hold=int(cell["m"]),
                b1=b1, b2=b2, b3=b3,
                tilt_deg=tilt, tilt_split_deg=theta_ref,
            )
        except torch.cuda.OutOfMemoryError as error:
            state["stop_reason"] = f"CUDA OOM during {table}/{label}: {error}"
            state["stopped_at"] = f"{table}/{label}"
            return None
        vram_after = free_vram_mib()
        row["table"] = table
        row["switches"] = {"B1": b1, "B2": b2, "B3": b3}
        row["vram_free_before_mib"] = vram_before["free_mib"]
        row["vram_free_after_mib"] = vram_after["free_mib"]
        row["vram_used_before_mib"] = vram_before["used_mib"]
        row["vram_used_after_mib"] = vram_after["used_mib"]
        row["vram_total_mib"] = vram_before["total_mib"]
        row["alloc_cap_mib"] = round(alloc_cap_mib, 1)
        state["projected_peak"] = max(state["projected_peak"], float(row["peak_cuda_reserved_mib"]))

        low = row["bands"]["tilt_le_ref"]
        high = row["bands"]["tilt_gt_ref"]
        print(
            f"[{table}] {label} B1={int(b1)} B2={int(b2)} B3={int(b3)}  "
            f"reach {row['reach']:.4f} coll {row['collision']:.4f} to {row['timeout']:.4f} "
            f"stuck {row['stuck']:.4f} oob {row['oob']:.4f} cps {row['cps']:.4f} "
            f"degen_ep {row['degenerate']['episode_frac_with_any']:.4f} "
            f"b3_ep {row['recovery_events']['b3_event_episode_frac']:.4f} "
            f"b2_ep {row['recovery_events']['b2_seed_episode_frac']:.4f} "
            f"| <=60: reach {low.get('reach', float('nan')):.4f} coll "
            f"{low.get('collision', float('nan')):.4f} (n {low.get('n', 0)}) "
            f"| >60: reach {high.get('reach', float('nan')):.4f} coll "
            f"{high.get('collision', float('nan')):.4f} (n {high.get('n', 0)}) "
            f"| ESS p50 {row['ess']['active'].get('p50', float('nan')):.2f} "
            f"peak {row['peak_cuda_reserved_mib']:.0f} MiB free {vram_before['free_mib']:.0f}->"
            f"{vram_after['free_mib']:.0f}  {time.time() - started:.1f}s",
            flush=True,
        )
        if row["peak_cuda_reserved_mib"] > alloc_cap_mib:
            state["stop_reason"] = (
                f"{table}/{label} peak reserved {row['peak_cuda_reserved_mib']:.0f} MiB exceeded the "
                f"half-free-margin cap {alloc_cap_mib:.0f} MiB"
            )
            state["stopped_at"] = f"{table}/{label}"
            return None
        torch.cuda.empty_cache()
        return row

    header = {
        "what": "v2.8.4 MPPI baseline — charter 'v3': the recovery components B1 / B2 / B3, screened on "
                "the charter-'v2' 16-cell grid and ablated at two cells. No selection, no ranking.",
        "lineage": {
            "charter_v1": "data/runs/v2.8.4/mppi_screen/ — the ORIGINAL 16-cell grid "
                          "(N x H x lam x C_crash), 8 cells run. IMMUTABLE. This package calls it S1.",
            "amendment_12_cell": "data/runs/v2.8.4/mppi_screen_v2/ — the R1-R4 amendment's 12-cell grid "
                                 "(sigma x lam_rel x H). IMMUTABLE. S2.",
            "charter_v2": "data/runs/v2.8.4/mppi_screen_v3/ — the hover-centred / settling-terminal / "
                          "control-hold 16-cell grid. IMMUTABLE. S3.",
            "charter_v3": "THIS work — data/runs/v2.8.4/mppi_v3/, docs/versions/v2.8.4/mppi_v3.md",
        },
        "correction_to_the_charter_context": {
            "charter_says": "v2 fixed the reach collapse; the residual failure concentrates at high "
                            "spawn tilt.",
            "what_is_on_disk": {
                "reach": "0.0000 in ALL 16 charter-'v2' cells and in BOTH tilt bands of every cell; "
                         "'goal' appears in no cell's outcome_counts",
                "tilt_concentration": "the lowest-collision cell (N1024_lam0.05_C100000_m1_H40_sig1) is "
                                      "0.2040 at tilt < 90 deg against 0.2764 at tilt >= 90 deg — worse "
                                      "at high tilt, but the low-tilt band fails almost as badly and "
                                      "reach is zero in both",
                "m4_cells": "all eight m=4 cells carry identical headline numbers (collision 0.9500, "
                            "oob 0.0500, cps -1.9250) because the degenerate branch fired on "
                            "0.9997-1.0000 of decisions: at the 8 s lookahead every sampled rollout "
                            "collides inside the horizon, the MPPI update is never applied, and the "
                            "vehicle flies the carried hover-trim plan open-loop",
            },
            "consequence": "B1/B2/B3 are applied to a controller that did not reach the goal at ANY "
                           "tilt. Nothing here is a polish of a residual.",
        },
        "components": "see src/frameworks/mppi/recovery.py; all three default OFF and the shipped config "
                      "reproduces the charter's 'v2' controller",
        "component_config": recovery_cfg,
        "reporting_split": {
            "theta_ref_deg": theta_ref,
            "convention": "three metric sets per row: overall, spawn tilt <= theta_ref, spawn tilt > "
                          "theta_ref; each carries its own episode count so every rate has its "
                          "denominator. Born-inverted and high-tilt episodes are NEVER removed from a "
                          "headline number.",
            "not_comparable_to_charter_v2": "the charter-'v2' screen split at 90 deg with < / >=; this "
                                            "one splits at 60 deg with <= / >. The bands are not "
                                            "directly comparable across the two screens.",
            "source": tilt_source,
            "deviation": "DEVIATION FROM THE CHARTER — the charter requires the split use the spawn tilt "
                         "recorded in the pool manifest. The manifest records none, so the documented "
                         "fallback definition is used (see `spawn_tilt_deg`).",
            "n_le_theta_ref": bins["n_le_theta_ref"], "n_gt_theta_ref": bins["n_gt_theta_ref"],
            "tilt_min_deg": float(tilt.min()), "tilt_median_deg": float(np.median(tilt)),
            "tilt_max_deg": float(tilt.max()),
        },
        "pool": {"path": str(pool_path.relative_to(REPO)), "sha256": pool_sha, "sha8": pool_sha[:8],
                 "expected_sha8": str(screen["pool_sha8"]), "n_scenes": n_scenes, "ebs": ebs,
                 "order": "the FIRST n scenes of the pool, in pool order"},
        "seed": int(screen["seed"]), "dtype": args.dtype, "device": str(device),
        "pid": os.getpid(), "ppid": os.getppid(),
        "launch_vram": launch_vram, "alloc_cap_mib": round(alloc_cap_mib, 1),
        "min_free_mib": args.min_free_mib,
        "initial_projected_peak_mib": projected_peak,
        "initial_projected_peak_source": "the largest peak_cuda_reserved_mib the charter-'v2' screen "
                                         "recorded on this same grid and pool — measured, not typed",
        "immutable_dirs_before": immutable_before,
        "selection": "NONE. The tables are emitted in config order, nothing is ranked or sorted, and no "
                     "cell is chosen. Selection is the Researcher's.",
    }

    def write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    screen_rows: list[dict] = []
    ablation_tables: dict[str, dict] = {}

    if args.stage in ("all", "screen"):
        cells = screen_cells(screen)
        print(f"--- SCREEN: {len(cells)} cells, B1+B2+B3 all ON ---", flush=True)
        for index, cell in enumerate(cells):
            label = cell_label(cell)
            row = guarded_cell(label, cell, b1=True, b2=True, b3=True, table="screen")
            if row is None:
                break
            row["grid_index"] = index
            screen_rows.append(row)
            write(out_dir / "screen_rows.json", {
                **header, "table": "screen",
                "switches": {"B1": True, "B2": True, "B3": True},
                "grid": "mppi.screen.grid_v3 — the same 16-cell grid as the charter's 'v2', config order",
                "n_cells_planned": len(cells), "n_cells_completed": len(screen_rows),
                "stop_reason": state["stop_reason"], "stopped_at": state["stopped_at"],
                "polls": polls, "rows": screen_rows,
            })

    if args.stage in ("all", "ablation") and state["stop_reason"] is None:
        for key, filename in (("registered", "ablation_registered.json"),
                              ("diagnostic", "ablation_diagnostic.json")):
            block = ablation_cfg[key]
            cell = ablation_cell(block)
            base_label = cell_label(cell)
            print(f"--- ABLATION ({key}): {base_label} — {block['status']} ---", flush=True)
            rows: list[dict] = []
            for spec in ablation_cfg["rows"]:
                label = f"abl_{key}__{base_label}__{spec['label']}"
                row = guarded_cell(
                    label, cell, b1=bool(spec["b1"]), b2=bool(spec["b2"]), b3=bool(spec["b3"]),
                    table=f"ablation_{key}",
                )
                if row is None:
                    break
                row["ablation_row"] = spec["label"]
                rows.append(row)
                ablation_tables[key] = {
                    **header, "table": f"ablation_{key}", "status": block["status"],
                    "cell": cell, "cell_label": base_label,
                    "row_order": [s["label"] for s in ablation_cfg["rows"]],
                    "n_rows_planned": len(ablation_cfg["rows"]), "n_rows_completed": len(rows),
                    "stop_reason": state["stop_reason"], "stopped_at": state["stopped_at"],
                    "polls": polls, "rows": rows,
                }
                write(out_dir / filename, ablation_tables[key])
            if state["stop_reason"] is not None:
                break

    final_poll = poll_gamma(state["projected_peak"], args.min_free_mib, "final")
    polls.append(final_poll)
    immutable_after = {
        str(p.relative_to(REPO)): {"mtime": p.stat().st_mtime,
                                   "mtime_iso": time.strftime(
                                       "%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
                                   "n_files": len(list(p.iterdir()))}
        for p in IMMUTABLE_DIRS
    }
    unchanged = {
        name: bool(immutable_before[name] == immutable_after[name]) for name in immutable_before
    }
    tail = {
        "immutable_dirs_after": immutable_after,
        "immutable_dirs_unchanged": unchanged,
        "final_poll": final_poll,
        "polls": polls,
        "stop_reason": state["stop_reason"], "stopped_at": state["stopped_at"],
        "final_projected_peak_mib": state["projected_peak"],
    }
    if screen_rows:
        payload = json.loads((out_dir / "screen_rows.json").read_text(encoding="utf-8"))
        write(out_dir / "screen_rows.json", {**payload, **tail})
    for key, filename in (("registered", "ablation_registered.json"),
                          ("diagnostic", "ablation_diagnostic.json")):
        if key in ablation_tables:
            payload = json.loads((out_dir / filename).read_text(encoding="utf-8"))
            write(out_dir / filename, {**payload, **tail})

    print(f"immutable dirs unchanged: {unchanged}", flush=True)
    if state["stop_reason"]:
        print(f"STOPPED: {state['stop_reason']}", flush=True)
        return 1
    print("complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
