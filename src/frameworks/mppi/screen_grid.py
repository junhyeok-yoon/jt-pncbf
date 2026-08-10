"""v2.8.4 MPPI baseline — the S3 16-cell screen, run SERIALLY in one process.

LINEAGE (get the names right; three screens exist):
    S1 = the ORIGINAL 16-cell grid (N x H x lam x C_crash), 8 of 16 cells run before a false-positive
         stop, artifacts `data/runs/v2.8.4/mppi_screen/`. The S3 dispatch calls it "v1". IMMUTABLE.
    S2 = the R1-R4 amendment's 12-cell grid (sigma x lam_rel x H), artifacts
         `data/runs/v2.8.4/mppi_screen_v2/`. IMMUTABLE.
    S3 = THIS screen, artifacts `data/runs/v2.8.4/mppi_screen_v3/`. The dispatch calls it "v2"; S3 is
         NOT the `mppi_screen_v2` directory. `--grid v3` (the default) runs it.

S3 grid (config order, sorted by nothing), nested exactly as the dispatch writes it:

    N in {256, 1024}  x  lam_rel in {0.05, 0.2}  x  C_crash in {1e3, 1e5}  x  m in {1, 4},
    H fixed at 40 DECISION entries, sigma fixed at 1.0 (the S1 value),

at n 400 on the fullcb pool, ebs 200, seed 42, with the S3 controller: the R-amendment sampler (R1
wrench-space plan allocated through `system.mixer_inv`, R2 stationary OU perturbations, R3 lambda
relative to the running std of the sample costs) plus S3's hover-centered sampling, settling terminal
cost and control hold m (effective lookahead H*m*dt = 2.0 s or 8.0 s). Every cell row additionally
carries the metric decomposition over the spawn-tilt bands (< 90 deg and >= 90 deg) — the born-inverted
class is REPORTED, never filtered: the headline numbers are always over all 400 scenes.

The superseded grids stay reachable with `--grid v1` / `--grid v2`; reproducing their MEASUREMENTS also
needs their controller switches (`--center none --control-hold 1 --terminal distance`, plus
`--space rotor --noise iid --lam-mode absolute` for S1), which the reproducibility gate exercises.

SELECTION: by Researcher decision from the full screen table. No threshold is registered, and this
script neither picks a cell nor ranks the table. All cells are reported as one table in config order.

CONCURRENCY GUARDS — the registered gamma-arm training run has ABSOLUTE priority:

  * launch only if nvidia-smi reports free VRAM >= --min-free-mib (default 6144 MiB);
  * the measured MPPI allocation must stay under half the free margin at launch; --sample-chunk caps
    the rollout batch along the sample axis to hold that (exactly equivalent to no chunking — samples
    are independent; asserted by cpu_smoke.py check (e)). A cell whose measured peak exceeds the cap
    STOPS the screen;
  * poll the gamma arm before and after every cell (see the R4 rule below);
  * ANY CUDA OOM anywhere STOPS the screen immediately.

POLL RULE (amendment R4) — the fix for the first screen's FALSE POSITIVE stop. That screen stopped after
cell 7 on three identical step counters; the arm was not stalled, it was inside its step-4500 in-loop
evaluation, during which `status.json` is not refreshed. A step-counter-only rule cannot tell an
in-loop-eval pause from a stall. The amended rule:

    frozen step counter AND no eval marker, on two consecutive polls  =>  STOP
    frozen step counter WITH an eval marker                           =>  NOT a stall, keep going

The eval marker is EITHER of:

  * the shared in-loop-eval flock on `data/runs/v2.8.4/eval_gate.lock` being held by someone else. The
    probe is NON-DESTRUCTIVE: open the file read-only, try `flock(LOCK_EX | LOCK_NB)`; EAGAIN/EACCES
    means an eval is active, and if the lock IS acquired it is released immediately (LOCK_UN) and the fd
    closed. The lock is never held for longer than the probe and the file is NEVER written;
  * the mtime of the run's eval outputs advancing — `eval_metrics.csv` / `eval_episodes.csv`.

Every poll (timestamp, current_step, pid_alive, lock_held, eval csv mtimes, frozen, eval_active, the
stall counter) is written into the screen's JSON, so the rule's behaviour is auditable after the fact.

Run:  python -m src.frameworks.mppi.screen_grid --grid v3 --out-dir data/runs/v2.8.4/mppi_screen_v3
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import subprocess
import time
from itertools import product
from pathlib import Path

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


GAMMA_PID = 1797033
GAMMA_RUN = REPO / (
    "data/runs/v2.8.4/set__20260809-002018__seed42/v2.8.4__jt__20260809-002018__seed42"
)
GAMMA_STATUS = GAMMA_RUN / "status.json"
EVAL_GATE_LOCK = REPO / "data/runs/v2.8.4/eval_gate.lock"
EVAL_OUTPUTS = (GAMMA_RUN / "eval_metrics.csv", GAMMA_RUN / "eval_episodes.csv")


def free_vram_mib() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def eval_lock_held(path: Path = EVAL_GATE_LOCK) -> bool | None:
    """NON-DESTRUCTIVE probe of the shared in-loop-eval flock. True = held by someone else.

    Opens read-only and tries a non-blocking exclusive flock. EAGAIN/EACCES/EWOULDBLOCK means an eval is
    active. If the lock IS acquired it is released IMMEDIATELY and the fd closed — the probe never holds
    it beyond that, and the file is never written. Returns None if the probe itself could not run."""
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return True
            return None
        fcntl.flock(fd, fcntl.LOCK_UN)          # release IMMEDIATELY; never held past the probe
        return False
    finally:
        os.close(fd)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def gamma_poll() -> dict:
    status = json.loads(GAMMA_STATUS.read_text(encoding="utf-8")) if GAMMA_STATUS.exists() else {}
    return {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid_alive": pid_alive(GAMMA_PID),
        "current_step": status.get("current_step"),
        "best_step": status.get("best_step"),
        "phase": status.get("phase"),
        "updated_at": status.get("updated_at"),
        "status_mtime": _mtime(GAMMA_STATUS),
        "lock_held": eval_lock_held(),
        "eval_csv_mtime": {p.name: _mtime(p) for p in EVAL_OUTPUTS},
    }


def classify(previous: dict, current: dict) -> dict:
    """Apply the R4 rule to two consecutive polls; returns the annotation stored on `current`."""
    frozen = current["current_step"] == previous["current_step"]
    csv_advanced = any(
        current["eval_csv_mtime"].get(name) is not None
        and current["eval_csv_mtime"].get(name) != previous["eval_csv_mtime"].get(name)
        for name in current["eval_csv_mtime"]
    )
    eval_active = bool(current["lock_held"]) or csv_advanced
    return {
        "frozen": bool(frozen),
        "eval_csv_mtime_advanced": bool(csv_advanced),
        "eval_marker_active": bool(eval_active),
        "counts_as_stall": bool(frozen and not eval_active),
    }


def spawn_tilt_deg(pool_path: Path, n_scenes: int, mppi_config: dict) -> np.ndarray:
    """Spawn tilt of the pool's first `n_scenes` initial states, in degrees.

    THE ESTABLISHED DEFINITION, read from the pool's own initial states rather than re-derived
    (scripts/analysis/v282_alpha_tilt_screen.py:93-94):

        cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2]
        tilt_deg = degrees(arccos(clip(cos_tilt, -1, 1)))

    i.e. the angle between the body z axis and world up. `x0` is built exactly as the eval harness builds
    it — `initial_states_from_batch` on the batched scenes, then `system.wrap_state` — so the array is
    aligned scene-for-scene with the episode rows, which `evaluate` emits in pool order.
    """
    config = effective_config(mppi_config)
    system = make_system(config)
    scenes = load_pool(pool_path).scenes[:n_scenes]
    batched = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float64)
    x0 = system.wrap_state(initial_states_from_batch(batched))
    cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2].numpy()
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def build_cells(screen: dict, which: str) -> list[dict]:
    """The grid, in CONFIG ORDER, sorted by nothing."""
    if which == "v3":
        grid = screen["grid_v3"]
        # nesting EXACTLY as the dispatch writes it: N x lam x C_crash x m (H and sigma are fixed)
        return [
            {"sigma": float(s), "lam": float(lam), "H": int(h), "N": int(n), "C_crash": float(c),
             "m": int(hold)}
            for n, lam, c, hold, h, s in product(
                grid["n_samples"], grid["lam"], grid["c_crash"], grid["control_hold"],
                grid["horizon"], grid["sigma"],
            )
        ]
    if which == "v2":
        grid = screen["grid_v2"]
        return [
            {"sigma": float(s), "lam": float(lam), "H": int(h),
             "N": int(n), "C_crash": float(c), "m": None}
            for s, lam, h, n, c in product(
                grid["sigma"], grid["lam"], grid["horizon"], grid["n_samples"], grid["c_crash"]
            )
        ]
    grid = screen["grid_v1_superseded"]
    return [
        {"sigma": None, "lam": float(lam), "H": int(h), "N": int(n), "C_crash": float(c), "m": None}
        for n, h, lam, c in product(
            grid["n_samples"], grid["horizon"], grid["lam"], grid["c_crash"]
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the S3 16-cell MPPI screen.")
    parser.add_argument("--grid", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--n-scenes", type=int, default=None)
    parser.add_argument("--ebs", type=int, default=None)
    parser.add_argument("--sample-chunk", type=int, default=0)
    parser.add_argument("--min-free-mib", type=float, default=6144.0)
    parser.add_argument("--out-dir", type=str, default=str(REPO / "data/runs/v2.8.4/mppi_screen_v3"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--center", type=str, default=None, choices=["hover", "none"],
                        help="override mppi.sampling.center (S3 default: hover)")
    parser.add_argument("--terminal", type=str, default=None, choices=["settling", "distance"],
                        help="override mppi.cost.terminal (S3 default: settling)")
    parser.add_argument("--space", type=str, default=None, choices=["wrench", "rotor"])
    parser.add_argument("--noise", type=str, default=None, choices=["ou", "iid"])
    parser.add_argument("--lam-mode", type=str, default=None, choices=["relative", "absolute"])
    args = parser.parse_args()

    mppi_config = load_mppi_config()
    screen = mppi_config["screen"]
    sampling = mppi_config["sampling"]
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

    launch_vram = free_vram_mib()
    if launch_vram["free_mib"] < args.min_free_mib:
        raise SystemExit(
            f"STOP: free VRAM {launch_vram['free_mib']:.0f} MiB < required {args.min_free_mib:.0f} MiB."
        )
    alloc_cap_mib = 0.5 * launch_vram["free_mib"]

    cells = build_cells(screen, args.grid)
    # The tilt array is computed ONCE, from the pool's own initial states, and handed to every cell so
    # every row's split rests on the same numbers (S3 reporting split).
    tilt = spawn_tilt_deg(pool_path, n_scenes, mppi_config)
    tilt_bands = {"lt_90": int((tilt < 90.0).sum()), "ge_90": int((tilt >= 90.0).sum())}
    print(f"pid {os.getpid()}  launch: free {launch_vram['free_mib']:.0f} MiB, alloc cap "
          f"{alloc_cap_mib:.0f} MiB, grid {args.grid}, {len(cells)} cells, n {n_scenes}, ebs {ebs}, "
          f"sample_chunk {args.sample_chunk}, sampler {args.space or sampling['space']}/"
          f"{args.noise or sampling['noise']}/{args.lam_mode or sampling['lam_mode']}, center "
          f"{args.center or sampling['center']}, terminal "
          f"{args.terminal or mppi_config['cost']['terminal']}, tilt bands {tilt_bands}", flush=True)

    polls = [gamma_poll()]
    polls[-1]["rule"] = {"frozen": False, "eval_marker_active": bool(polls[-1]["lock_held"]),
                         "counts_as_stall": False, "note": "baseline poll, nothing to compare against"}
    polls[-1]["stall_count"] = 0
    print(f"gamma poll {polls[-1]}", flush=True)
    stalls = 0
    rows: list[dict] = []
    stop_reason = None
    stopped_at = None

    for index, cell in enumerate(cells):
        if cell["m"] is not None:
            label = (
                f"N{cell['N']}_lam{cell['lam']:g}_C{cell['C_crash']:g}_m{cell['m']}"
                f"_H{cell['H']}_sig{cell['sigma']:g}"
            )
        elif cell["sigma"] is not None:
            label = f"sig{cell['sigma']:g}_lam{cell['lam']:g}_H{cell['H']}_N{cell['N']}_C{cell['C_crash']:g}"
        else:
            label = f"N{cell['N']}_H{cell['H']}_lam{cell['lam']:g}_C{cell['C_crash']:g}"
        vram_before = free_vram_mib()
        if vram_before["free_mib"] < args.min_free_mib:
            stop_reason = f"free VRAM fell to {vram_before['free_mib']:.0f} MiB before cell {label}"
            stopped_at = label
            break
        started = time.time()
        try:
            row = run_cell(
                pool_path=pool_path, n_scenes=n_scenes, ebs=ebs,
                n_samples=int(cell["N"]), horizon=int(cell["H"]), lam=float(cell["lam"]),
                c_crash=float(cell["C_crash"]), sigma=cell["sigma"], seed=int(screen["seed"]),
                sample_chunk=int(args.sample_chunk), label=label, out_dir=out_dir,
                device=device, dtype=dtype, mppi_config=mppi_config,
                control_hold=cell["m"], center=args.center, terminal=args.terminal,
                space=args.space, noise=args.noise, lam_mode=args.lam_mode,
                tilt_deg=tilt,
            )
        except torch.cuda.OutOfMemoryError as error:
            stop_reason = f"CUDA OOM during cell {label}: {error}"
            stopped_at = label
            break
        vram_after = free_vram_mib()
        row["vram_free_before_mib"] = vram_before["free_mib"]
        row["vram_free_after_mib"] = vram_after["free_mib"]
        row["vram_used_before_mib"] = vram_before["used_mib"]
        row["vram_used_after_mib"] = vram_after["used_mib"]
        row["vram_total_mib"] = vram_before["total_mib"]
        row["alloc_cap_mib"] = round(alloc_cap_mib, 1)
        row["grid_index"] = index
        rows.append(row)
        ess = row["ess"]["active"]
        bands = row.get("bands", {})
        lo_band, hi_band = bands.get("tilt_lt_90", {}), bands.get("tilt_ge_90", {})
        print(
            f"[{index:2d}/{len(cells)}] {label}  reach {row['reach']:.4f} coll {row['collision']:.4f} "
            f"to {row['timeout']:.4f} stuck {row['stuck']:.4f} oob {row['oob']:.4f} "
            f"cps {row['cps']:.4f} degen {row['degenerate']['episode_frac_with_any']:.4f} "
            f"| reach<90 {lo_band.get('reach', float('nan')):.4f} (n {lo_band.get('n', 0)}) "
            f"reach>=90 {hi_band.get('reach', float('nan')):.4f} (n {hi_band.get('n', 0)}) "
            f"| ESS p50 {ess.get('p50', float('nan')):.1f} mean {ess.get('mean', float('nan')):.1f} "
            f"peak {row['peak_cuda_reserved_mib']:.0f} MiB  free {vram_before['free_mib']:.0f}->"
            f"{vram_after['free_mib']:.0f} MiB  {time.time() - started:.1f}s",
            flush=True,
        )

        if row["peak_cuda_reserved_mib"] > alloc_cap_mib:
            stop_reason = (
                f"cell {label} peak reserved {row['peak_cuda_reserved_mib']:.0f} MiB exceeded the "
                f"half-free-margin cap {alloc_cap_mib:.0f} MiB"
            )
            stopped_at = label
            break

        poll = gamma_poll()
        poll["rule"] = classify(polls[-1], poll)
        if not poll["pid_alive"]:
            poll["stall_count"] = stalls
            polls.append(poll)
            print(f"gamma poll {poll}", flush=True)
            stop_reason = f"gamma arm PID {GAMMA_PID} is no longer alive (after cell {label})"
            stopped_at = label
            break
        # R4: a frozen step counter is a stall ONLY when no eval marker is active.
        stalls = stalls + 1 if poll["rule"]["counts_as_stall"] else 0
        poll["stall_count"] = stalls
        polls.append(poll)
        print(f"gamma poll {poll}", flush=True)
        if stalls >= 2:
            stop_reason = (
                f"gamma arm step counter frozen at {poll['current_step']} with NO eval marker for two "
                f"consecutive polls (after cell {label})"
            )
            stopped_at = label
            break

        torch.cuda.empty_cache()

    summary = {
        "what": f"v2.8.4 MPPI baseline — screen S3, grid {args.grid}, config order, no threshold, "
                "no selection.",
        "lineage": {
            "S1": "the ORIGINAL 16-cell grid (N x H x lam x C_crash); artifacts "
                  "data/runs/v2.8.4/mppi_screen/; the S3 dispatch calls it 'v1'; IMMUTABLE",
            "S2": "the R1-R4 amendment's 12-cell grid (sigma x lam_rel x H); artifacts "
                  "data/runs/v2.8.4/mppi_screen_v2/; IMMUTABLE",
            "S3": "THIS screen; artifacts data/runs/v2.8.4/mppi_screen_v3/; the dispatch calls it 'v2' "
                  "— S3 is NOT the mppi_screen_v2 directory",
        },
        "s3_changes": {
            "1_sampling_center": "u = u_hover + u_plan + eps with u_hover the fixed hover-trim anchor "
                                 "read from the system object; u_plan is the deviation the update moves",
            "2_terminal_cost": "w_terminal * [relu(d - goal_radius)^2 + relu(|v| - goal_speed_radius)^2 "
                               "+ relu(|w| - goal_angrate_radius)^2], all three radii read from the "
                               "effective deployed config; zero exactly on the deployed reach predicate",
            "3_control_hold": "each of the H decision entries is applied for m physical steps; the "
                              "rollout spans H*m rk4 steps and re-planning happens every m control steps",
            "4_reporting_split": "every cell also carries the full decomposition over spawn tilt < 90 "
                                 "deg and >= 90 deg; born-inverted episodes are NEVER filtered out of a "
                                 "headline number",
        },
        "held_coefficients": {
            "sigma": "1.0 — the S1 value and the config default; NOT a grid dimension here, so the "
                     "delta against S1's table is readable. S2 swept sigma in {0.1,0.3,0.6}, so S2 and "
                     "S3 are not directly comparable.",
            "lam_mode": "relative — R3 fixed a demonstrated defect (S1 ESS median 1.0001); reverting to "
                        "absolute would reintroduce it. The lam values are lam_rel.",
            "space/noise": "wrench / ou — the dispatch's 'first-order AR smoothing with coefficient in "
                           "config' IS the R2 stationary OU, coefficient config-read with the load-time "
                           "consistency assert.",
        },
        "amendment": {
            "R1_control_parameterization": "plan in body-wrench space, allocated through system.mixer_inv, "
                                           "clipped per rotor to system.u_bounds AFTER allocation",
            "R2_noise": "mean = hover trim wrench from system constants; stationary OU along the horizon",
            "R3_grid_and_lambda": "sigma x lam_rel x H, N=256, C_crash=1e5 fixed; lam_eff = "
                                  "max(lam_rel * std_n(S_n), lam_eps_abs) per (scene, control step)",
            "R4_poll_rule": "a frozen step counter with an eval marker (eval_gate.lock held, or the eval "
                            "CSVs' mtime advancing) is NOT a stall; two frozen polls with no marker STOP",
        },
        "selection": "by Researcher decision from the full screen table; no threshold is registered",
        "sampler": {k: sampling[k] for k in
                    ("space", "noise", "lam_mode", "center", "control_hold", "channel_scale", "trim",
                     "ou", "lam_eps_abs", "sigma")},
        "sampler_overrides_from_cli": {
            "center": args.center, "terminal": args.terminal, "space": args.space,
            "noise": args.noise, "lam_mode": args.lam_mode,
        },
        "terminal_cost_mode": args.terminal or mppi_config["cost"]["terminal"],
        "tilt_split": {
            "definition": "cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2]; tilt_deg = "
                          "degrees(arccos(clip(cos_tilt, -1, 1))) — the established definition at "
                          "scripts/analysis/v282_alpha_tilt_screen.py:93-94, read from the pool's own "
                          "initial states",
            "split_deg": 90.0,
            "n_lt_90": tilt_bands["lt_90"], "n_ge_90": tilt_bands["ge_90"],
            "tilt_median_deg": float(np.median(tilt)),
            "tilt_min_deg": float(tilt.min()), "tilt_max_deg": float(tilt.max()),
        },
        "pool": str(pool_path.relative_to(REPO)), "pool_sha8_expected": screen["pool_sha8"],
        "n_scenes": n_scenes, "ebs": ebs, "sample_chunk": int(args.sample_chunk),
        "seed": int(screen["seed"]), "dtype": args.dtype, "pid": os.getpid(),
        "launch_vram": launch_vram, "alloc_cap_mib": round(alloc_cap_mib, 1),
        "min_free_mib": args.min_free_mib,
        "gamma_arm": {
            "pid": GAMMA_PID,
            "status": str(GAMMA_STATUS.relative_to(REPO)),
            "eval_gate_lock": str(EVAL_GATE_LOCK.relative_to(REPO)),
            "eval_outputs": [str(p.relative_to(REPO)) for p in EVAL_OUTPUTS],
            "polls": polls,
        },
        "n_cells_planned": len(cells), "n_cells_completed": len(rows),
        "stop_reason": stop_reason, "stopped_at_cell": stopped_at,
        "rows": rows,
    }
    (out_dir / "screen_rows.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if stop_reason:
        print(f"STOPPED: {stop_reason}", flush=True)
        return 1
    print("screen complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
