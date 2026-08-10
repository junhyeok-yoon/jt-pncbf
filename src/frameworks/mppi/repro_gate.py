"""v2.8.4 MPPI baseline — the BACKWARD-REPRODUCIBILITY GATE for the amended sampler.

The v2.8.4 amendment replaces the sampler (R1 wrench-space plan + system allocation, R2 OU noise,
R3 relative lambda) and makes the new path the default. The SUPERSEDED first screen is retained as
measurement under `data/runs/v2.8.4/mppi_screen/`, and it must stay exactly reproducible through the
legacy switches `--space rotor --noise iid --lam-mode absolute`. S3 (hover-centered sampling, the
settling terminal cost, the control hold) adds three more switches, and the gate pins those to their
legacy values too (`center none`, `control_hold 1`, `terminal distance`, all read from
`mppi.repro_gate`), so the fresh run is the S1 controller and nothing else. The COMPARISON is unchanged:
S3's new leaves all live under `sampler.` / `ess.`, which the ADDED_OK list already excused, and the
tilt-band block is emitted only when a tilt array is supplied — the gate supplies none.

This script re-runs ONE cell of the first screen through the legacy path and compares EVERY metric of
the fresh run against the persisted artifact. Identity of the metrics is the gate; anything else is a
STOP and the amended screen must not run.

What is compared: every leaf of the retained cell JSON except the run-to-run quantities that are not
metrics — wall clock, per-step wall clock, the CUDA peak allocation, and the cell's own `label`. The
`sampler` / `ess` / `infeasibility_status` blocks and `degenerate.step_frac_over_active_window` are new
fields that the retained artifact predates; they are reported as ADDED, never silently dropped, and they
are not part of the identity claim because the old artifact carries nothing to compare them against.

Run:  python -m src.frameworks.mppi.repro_gate --out-dir data/runs/v2.8.4/mppi_screen_v3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    _resolve_device,
    _resolve_dtype,
    load_mppi_config,
    run_cell,
)


# not metrics: they legitimately vary between two runs of the same computation
EXCLUDED = {
    "wall_s", "wall_per_control_step_s", "peak_cuda_alloc_mib", "peak_cuda_reserved_mib", "label",
}
# fields the amendment added; the retained artifact predates them, so there is nothing to compare
ADDED_OK = {
    "sampler", "ess", "infeasibility_status",
    "cell.lambda_mode", "cell.space", "cell.noise",
    "degenerate.step_frac_over_active_window",
}


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Leaves of a nested dict as `a.b.c -> value`; lists are compared whole (as leaves)."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def compare(reference: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    ref_flat = _flatten(reference)
    new_flat = _flatten(fresh)
    ref_keys = {k for k in ref_flat if k.split(".")[0] not in EXCLUDED and k not in EXCLUDED}
    compared, mismatches, missing = {}, [], []
    for key in sorted(ref_keys):
        if key not in new_flat:
            missing.append(key)
            continue
        same = ref_flat[key] == new_flat[key]
        compared[key] = {"reference": ref_flat[key], "fresh": new_flat[key], "identical": bool(same)}
        if not same:
            mismatches.append(key)
    added = sorted(
        k for k in new_flat
        if k not in ref_flat
        and k.split(".")[0] not in EXCLUDED
        and not any(k == a or k.startswith(a + ".") for a in ADDED_OK)
    )
    return {
        "n_leaves_compared": len(compared),
        "n_mismatches": len(mismatches),
        "mismatched_keys": mismatches,
        "missing_in_fresh": missing,
        "unexpected_new_keys": added,
        "leaves": compared,
        "PASS": bool(not mismatches and not missing and not added),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MPPI backward-reproducibility gate.")
    parser.add_argument("--out-dir", type=str, default=str(REPO / "data/runs/v2.8.4/mppi_screen_v3"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    device, dtype = _resolve_device(args.device), _resolve_dtype(args.dtype)
    mppi_config = load_mppi_config()
    gate = mppi_config["repro_gate"]
    screen = mppi_config["screen"]
    artifact = REPO / gate["artifact"]
    reference = json.loads(artifact.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell = reference["cell"]
    started = time.time()
    fresh = run_cell(
        pool_path=Path(reference["pool"]["path"]),
        n_scenes=int(reference["pool"]["n_scenes_scored"]),
        ebs=int(reference["pool"]["ebs"]),
        n_samples=int(cell["N"]), horizon=int(cell["H"]), lam=float(cell["lambda"]),
        c_crash=float(cell["C_crash"]), sigma=float(cell["sigma"]), seed=int(cell["seed"]),
        sample_chunk=int(cell["sample_chunk"]),
        space=str(gate["space"]), noise=str(gate["noise"]), lam_mode=str(gate["lam_mode"]),
        center=str(gate["center"]), control_hold=int(gate["control_hold"]),
        terminal=str(gate["terminal"]),
        label=str(reference["label"]), out_dir=out_dir, device=device, dtype=dtype,
        mppi_config=mppi_config, write=False,
    )
    result = compare(reference, fresh)
    record = {
        "what": "v2.8.4 MPPI amendment — backward-reproducibility gate (D). The SUPERSEDED first "
                "screen's sampler must still reproduce its retained measurement exactly. Re-run after "
                "the S3 changes (hover centre, settling terminal, control hold) with every S3 switch "
                "pinned to its legacy value; the comparison itself is unchanged.",
        "legacy_switches": {
            "space": gate["space"], "noise": gate["noise"], "lam_mode": gate["lam_mode"],
            "center": gate["center"], "control_hold": gate["control_hold"],
            "terminal": gate["terminal"],
        },
        "reference_artifact": str(artifact),
        "reference_artifact_mtime": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(artifact.stat().st_mtime)
        ),
        "cell": cell,
        "pool_sha8": reference["pool"]["sha8"],
        "device": str(device), "dtype": str(dtype),
        "wall_s": round(time.time() - started, 2),
        "excluded_from_comparison": sorted(EXCLUDED),
        "new_fields_not_in_the_reference": sorted(ADDED_OK),
        "comparison": result,
        "ALL_PASS": result["PASS"],
    }
    path = out_dir / "repro_gate.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "comparison"}, indent=2))
    print(
        f"repro gate: {result['n_leaves_compared']} leaves compared, "
        f"{result['n_mismatches']} mismatches, {len(result['missing_in_fresh'])} missing, "
        f"{len(result['unexpected_new_keys'])} unexpected new -> "
        f"{'ALL_PASS' if result['PASS'] else 'ALL_FAIL'}  ({path})",
        flush=True,
    )
    if not result["PASS"]:
        print("MISMATCHES:", json.dumps(
            {k: result["leaves"][k] for k in result["mismatched_keys"]}, indent=2))
    _ = screen
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
