"""v2.8.4 charter-"v4" goal-attraction redesign — the scored rounds (probe / adjust / confirm).

Runs one ROUND of the dispatch's bounded schedule and writes its complete table. Every cell is scored by
`src.frameworks.mppi.evaluate_mppi.run_cell`, i.e. by the same shared eval path (`src.eval.evaluate`), on
the same pool, terminal predicate and outcome resolution as every other v2.8.4 row; the round-table
columns (d_min, the time-inside share, |omega| and the control smoothness) are read off the rollouts that
call already produced by `evaluate_mppi.v4_report_columns`.

    Round 1  probe    the dispatch's four switch combinations at the BRIEF-FIXED reference cell, n = 50
    Round 2  adjust   the Round-1-selected combination with its two named coefficients moved one value
                      down and one up (4 cells), n = 50. The two factors are
                      `mppi.goal_v4.round2.factor_down` / `factor_up`; no swept value is typed here.
    Round 3  confirm  the selected configuration at n = 400 with the full tilt split.

NO COEFFICIENT IS TYPED IN THIS FILE. The reference cell (N, lambda, C_crash, m, H, sigma, B1/B2/B3), the
probe and confirm scales, the four Round-1 combinations, the Round-2 factors, the reporting probe radius
and the tilt-split theta_ref are all read from `src/frameworks/mppi/config.yaml`; the plant constants, dt
and the three deployed radii come from the shipped effective config through `evaluate_mppi`.

NO THRESHOLD is registered, computed or applied anywhere in this file, and nothing here ranks or selects
a cell: which combination Round 2 moves on and which two coefficients it sweeps are given on the command
line by the Researcher after reading the Round-1 table, and the reason is recorded in the build log.

Run:  python -m src.frameworks.mppi.rounds_goal_v4 --round 1
      python -m src.frameworks.mppi.rounds_goal_v4 --round 2 --combo g1,g2,g3,g4 --sweep w_p,w_omega
      python -m src.frameworks.mppi.rounds_goal_v4 --round 3 --combo g1,g2,g3,g4
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
from src.frameworks.mppi.evaluate_mppi import REPO, load_mppi_config, run_cell
from src.frameworks.mppi.screen_recovery import spawn_tilt_deg

OUT_DIR = REPO / "data/runs/v2.8.4/mppi_v4"

IMMUTABLE_DIRS = (
    REPO / "data/runs/v2.8.4/mppi_screen",
    REPO / "data/runs/v2.8.4/mppi_screen_v2",
    REPO / "data/runs/v2.8.4/mppi_screen_v3",
    REPO / "data/runs/v2.8.4/mppi_v3",
    REPO / "data/runs/v2.8.4/mppi_diag",
)

# The dispatch's own GPU floor, in MiB. A launch condition, not a coefficient of any controller.
VRAM_FLOOR_MIB = 6144.0

# The round table's columns, in the dispatch's exact order.
TABLE_COLUMNS = (
    "cell_id", "reach_all", "d_min_p50", "inside_r05_share", "coll_all", "timeout_all",
    "omega_p50", "smooth_mean_du", "reach_le60", "coll_le60", "n_le60",
    "reach_gt60", "coll_gt60", "n_gt60", "wall_ms_per_step",
)

# The two v4 coefficients `run_cell` can move, and the config path each default is read from.
SWEEPABLE = {
    "w_p": ("g1", "w_p"),
    "w_omega": ("g4", "w_omega"),
}


def vram() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def gpu_python_poll() -> dict[str, Any]:
    """Other GPU python processes of this task family, polled before every cell."""
    proc = subprocess.run(["pgrep", "-af", "v284_|mppi_"], capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if "pgrep" not in ln]
    mine = {str(os.getpid()), str(os.getppid())}
    others = [ln for ln in lines if ln.split(" ", 1)[0] not in mine]
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "pgrep_pattern": "v284_|mppi_",
        "matches": lines,
        "matches_excluding_self": others,
        "n_other": len(others),
        "nvidia_smi_compute_apps": compute,
    }


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


def table_row(cell_id: str, cell: dict[str, Any]) -> dict[str, Any]:
    """The dispatch's columns, in its exact order, out of a scored cell record."""
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


def markdown_table(rows: list[dict[str, Any]]) -> str:
    def fmt(key: str, value: Any) -> str:
        if value is None:
            return "—"
        if key in ("cell_id",):
            return str(value)
        if key in ("n_le60", "n_gt60"):
            return str(int(value))
        if key == "wall_ms_per_step":
            return f"{float(value):.1f}"
        if key == "smooth_mean_du":
            return f"{float(value):.4f}"
        return f"{float(value):.4f}"

    head = "| " + " | ".join(TABLE_COLUMNS) + " |"
    rule = "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|"
    body = [
        "| " + " | ".join(fmt(c, row[c]) for c in TABLE_COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def build_cells(args: argparse.Namespace, v4: dict[str, Any]) -> list[dict[str, Any]]:
    """The cells of the requested round. Switch names and factors only — no coefficient is typed."""
    if args.round == 1:
        return [
            {"id": row["id"],
             "switches": {k: bool(row[k]) for k in ("g1", "g2", "g3", "g4")},
             "overrides": {}}
            for row in v4["round1"]
        ]

    combo = {k: (k in {s.strip() for s in args.combo.split(",") if s.strip()})
             for k in ("g1", "g2", "g3", "g4")}
    combo_id = "".join(k.upper() for k in ("g1", "g2", "g3", "g4") if combo[k]) or "v3ref"

    if args.round == 3:
        return [{"id": f"R3_{combo_id}", "switches": combo, "overrides": dict(args.fixed_overrides)}]

    swept = [s.strip() for s in args.sweep.split(",") if s.strip()]
    if len(swept) != 2:
        raise SystemExit(f"STOP: Round 2 sweeps exactly two coefficients; got {swept}.")
    for name in swept:
        if name not in SWEEPABLE:
            raise SystemExit(f"STOP: {name!r} is not a config-read v4 coefficient {sorted(SWEEPABLE)}.")
    down = float(v4["round2"]["factor_down"])
    up = float(v4["round2"]["factor_up"])
    cells = []
    for name in swept:
        block, key = SWEEPABLE[name]
        base = float(v4[block][key])
        for tag, factor in (("down", down), ("up", up)):
            cells.append({
                "id": f"R2_{combo_id}_{name}_{tag}",
                "switches": combo,
                "overrides": {**dict(args.fixed_overrides), name: base * factor},
                "swept": {"coefficient": name, "config_value": base, "factor": factor,
                          "value": base * factor},
            })
    return cells


def main() -> int:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run one scored round of the charter-'v4' redesign.")
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--combo", type=str, default="",
                        help="comma-separated switch names ON for rounds 2/3, e.g. 'g1,g2,g3,g4'")
    parser.add_argument("--sweep", type=str, default="",
                        help="Round 2 only: the two coefficients to move, e.g. 'w_p,w_omega'")
    parser.add_argument("--w-p", type=float, default=None,
                        help="Round 3 only: hold w_p at this value (default: the config value)")
    parser.add_argument("--w-omega", type=float, default=None,
                        help="Round 3 only: hold w_omega at this value (default: the config value)")
    parser.add_argument("--reason", type=str, default="",
                        help="the comparison that drove this round's choice, recorded in the run JSON")
    parser.add_argument("--plots", action="store_true",
                        help="Round 3: emit the shipped grids A/B/C and the xz/yz + time-series "
                             "REPRODUCTIONS from THESE rollouts (no second rollout set)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    args.fixed_overrides = {
        k: v for k, v in (("w_p", args.w_p), ("w_omega", args.w_omega)) if v is not None
    }

    print(f"pid {os.getpid()} ppid {os.getppid()}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    mppi_config = load_mppi_config()
    v4 = mppi_config["goal_v4"]
    reference = v4["reference_cell"]
    scale = v4["probe"] if args.round in (1, 2) else v4["confirm"]
    n_scenes, ebs = int(scale["n_scenes"]), int(scale["ebs"])
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])
    seed = int(mppi_config["sampling"]["seed"])

    pool_path = REPO / mppi_config["screen"]["pool"]
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if pool_sha[:8] != str(mppi_config["screen"]["pool_sha8"]):
        raise SystemExit(f"STOP: pool sha8 {pool_sha[:8]} != {mppi_config['screen']['pool_sha8']}.")
    print(f"pool sha8 {pool_sha[:8]} OK · n_scenes {n_scenes} ebs {ebs} seed {seed} "
          f"theta_ref {theta_ref}", flush=True)

    immutable_before = dir_state()
    launch_vram = vram()
    print(f"launch VRAM {launch_vram}", flush=True)
    if launch_vram["free_mib"] < VRAM_FLOOR_MIB:
        raise SystemExit(f"STOP: free VRAM {launch_vram['free_mib']:.0f} MiB < {VRAM_FLOOR_MIB:.0f}.")
    cap_mib = 0.5 * launch_vram["free_mib"]
    print(f"peak cap (half the launch free margin) {cap_mib:.0f} MiB", flush=True)

    tilt = spawn_tilt_deg(pool_path, n_scenes, mppi_config)
    n_le = int((tilt <= theta_ref).sum())
    n_gt = int((tilt > theta_ref).sum())
    print(f"tilt split at {theta_ref} deg: n_le {n_le} n_gt {n_gt} "
          f"(min {tilt.min():.2f} median {np.median(tilt):.2f} max {tilt.max():.2f})", flush=True)

    cells = build_cells(args, v4)
    print(f"round {args.round}: {len(cells)} cells -> {[c['id'] for c in cells]}", flush=True)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for spec in cells:
        poll = gpu_python_poll()
        before = vram()
        print(f"\n[{spec['id']}] poll: {poll['n_other']} other matching process(es) "
              f"{poll['matches_excluding_self']}", flush=True)
        print(f"[{spec['id']}] compute apps: {poll['nvidia_smi_compute_apps']!r}", flush=True)
        print(f"[{spec['id']}] VRAM before {before}", flush=True)
        if before["free_mib"] < VRAM_FLOOR_MIB:
            raise SystemExit(f"STOP: free VRAM {before['free_mib']:.0f} MiB < {VRAM_FLOOR_MIB:.0f}.")
        capture: dict[str, Any] | None = {} if args.plots else None
        try:
            cell = run_cell(
                pool_path=pool_path,
                n_scenes=n_scenes,
                ebs=ebs,
                n_samples=int(reference["n_samples"]),
                horizon=int(reference["horizon"]),
                lam=float(reference["lam"]),
                c_crash=float(reference["c_crash"]),
                sigma=float(reference["sigma"]),
                seed=seed,
                control_hold=int(reference["control_hold"]),
                b1=bool(reference["b1"]), b2=bool(reference["b2"]), b3=bool(reference["b3"]),
                g1=spec["switches"]["g1"], g2=spec["switches"]["g2"],
                g3=spec["switches"]["g3"], g4=spec["switches"]["g4"],
                w_p=spec["overrides"].get("w_p"),
                w_omega=spec["overrides"].get("w_omega"),
                tilt_deg=tilt,
                tilt_split_deg=theta_ref,
                label=spec["id"],
                out_dir=OUT_DIR,
                device=device,
                dtype=dtype,
                mppi_config=mppi_config,
                v4_columns=True,
                capture=capture,
            )
        except torch.cuda.OutOfMemoryError as exc:                       # pragma: no cover
            raise SystemExit(f"STOP: CUDA OOM on cell {spec['id']}: {exc}")
        after = vram()
        peak = float(cell["peak_cuda_reserved_mib"])
        print(f"[{spec['id']}] VRAM after {after}  peak alloc {cell['peak_cuda_alloc_mib']} MiB "
              f"reserved {peak} MiB (cap {cap_mib:.0f})", flush=True)
        if peak > cap_mib:
            raise SystemExit(f"STOP: peak reserved {peak:.0f} MiB over cap {cap_mib:.0f} MiB.")
        figures = None
        if capture:
            from src.frameworks.mppi.plots_goal_v4 import emit_grids
            figures = emit_grids(
                system=capture["system"], config=capture["config"],
                trajectories=capture["result"].trajectories,
                episode_rows=capture["result"].episode_rows,
                tilt_deg=tilt, label=spec["id"], fig_dir=OUT_DIR / "figures",
            )
        row = table_row(spec["id"], cell)
        rows.append(row)
        records.append({
            "figures": figures,
            "cell_id": spec["id"],
            "switches": spec["switches"],
            "overrides": spec["overrides"],
            "swept": spec.get("swept"),
            "goal_v4": cell["goal_v4"],
            "row": row,
            "outcome_counts": cell["outcome_counts"],
            "degenerate_step_frac": cell["degenerate"]["step_frac_over_active_window"],
            "ess_p50": cell["ess"]["active"].get("p50"),
            "v4_columns": cell["v4_columns"],
            "gpu_poll": poll,
            "vram": {"before": before, "after": after,
                     "peak_cuda_alloc_mib": cell["peak_cuda_alloc_mib"],
                     "peak_cuda_reserved_mib": peak},
            "wall_s": cell["wall_s"],
            "n_control_steps": cell["n_control_steps"],
        })
        print("[" + spec["id"] + "] " + json.dumps(row), flush=True)

    immutable_after = dir_state()
    table = markdown_table(rows)
    print("\n" + table, flush=True)
    report = {
        "what": f"charter-'v4' goal-attraction redesign — ROUND {args.round} "
                f"({'probe' if args.round == 1 else 'adjust' if args.round == 2 else 'confirm'})",
        "version": __version__,
        "round": args.round,
        "reason_for_this_round": args.reason,
        "pid": os.getpid(), "ppid": os.getppid(),
        "reference_cell": dict(reference),
        "scale": {"n_scenes": n_scenes, "ebs": ebs, "seed": seed},
        "pool": {"path": str(pool_path), "sha256": pool_sha, "sha8": pool_sha[:8],
                 "order": f"the FIRST {n_scenes} scenes of the pool, in pool order"},
        "tilt_split": {"theta_ref_deg": theta_ref, "convention": "<= / >",
                       "definition": "degrees(arccos(clip(R(q0)[2,2], -1, 1))) — the documented "
                                     "fallback; the pool manifest carries no per-scene tilt field",
                       "n_le": n_le, "n_gt": n_gt,
                       "tilt_min": float(tilt.min()), "tilt_median": float(np.median(tilt)),
                       "tilt_max": float(tilt.max())},
        "columns": list(TABLE_COLUMNS),
        "table_markdown": table,
        "rows": rows,
        "cells": records,
        "launch_vram": launch_vram,
        "peak_cap_mib": cap_mib,
        "immutable_dirs": {"before": immutable_before, "after": immutable_after,
                           "unchanged": immutable_before == immutable_after},
        "wall_s": round(time.time() - t0, 2),
    }
    path = OUT_DIR / f"round{args.round}.json"
    if path.exists():
        path = OUT_DIR / f"round{args.round}__{time.strftime('%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"immutable dirs unchanged: {report['immutable_dirs']['unchanged']}", flush=True)
    print(f"wrote {path}  ({report['wall_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
