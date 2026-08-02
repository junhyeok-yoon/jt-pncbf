"""v2.8.0 R1 — one fresh-process run of either the integrated three-cell driver or the standalone
per-cell path, on the dual deliverable. Writes full-precision tilt60/bandopen cps + per-episode rows so
the two lineages' clusters and the per-episode flip list can be characterized. Invoke once per process
(the whole point is a fresh process per run):

    python -m scripts.analysis.v280_r1_run --mode driver     --out data/runs/v2.8.0/r1/driver_1
    python -m scripts.analysis.v280_r1_run --mode standalone --out data/runs/v2.8.0/r1/standalone_1
"""
from __future__ import annotations
import argparse, copy, csv, json
from pathlib import Path
import torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework, run_final_cells, EVAL_EPISODE_COLUMNS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL_D2R = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
POOL_NAV = REPO / "data/secured_data/pools/eval_navcone_quadrotor-3d-d2r_n2000_seed34567.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _standalone_cell(pool, band_terminates):
    """Exactly the s4_proofs per-cell fresh-load path (the sidecar-generating lineage)."""
    ck = torch.load(str(CK), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])
    filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    filt["projection"] = "dual_solve"
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": 0.48, "band_terminates": band_terminates},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = _load_framework(str(CK), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(DEV)
    return evaluate(fw, pool, cfg, mode="final", step=int(ck2["step"]), ckpt_name="r1",
                    max_scenes=None, include_lqr_baseline=False)


def _write_eps(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_EPISODE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["driver", "standalone"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "driver":
        # exactly the W4/W2 integrated three-cell driver invocation (mixed eval runs first)
        over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                        "goal_angrate_radius": 0.48},
                "eval": {"max_steps": 200, "dt_ctrl": 0.05, "final": {"cells": ["mixed", "tilt60", "bandopen"]}},
                "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
        fw, cfg, ck = _load_framework(str(CK), config_overrides=over)
        run_final_cells(out, fw, cfg, ck, CK)
        def _cps(cell):
            with open(out / "eval" / cell / "eval_metrics.csv") as f:
                return float(list(csv.DictReader(f))[0]["cps"])
        tilt60, bandopen = _cps("tilt60"), _cps("bandopen")
        # per-cell eval_episodes.csv already written by run_final_cells under out/eval/<cell>/
    else:
        t = _standalone_cell(POOL_NAV, True)
        b = _standalone_cell(POOL_D2R, False)
        tilt60, bandopen = float(t.eval_row["cps"]), float(b.eval_row["cps"])
        _write_eps(out / "tilt60_episodes.csv", t.episode_rows)
        _write_eps(out / "bandopen_episodes.csv", b.episode_rows)

    res = {"mode": args.mode, "tilt60_cps": tilt60, "bandopen_cps": bandopen}
    (out / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    print("R1RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
