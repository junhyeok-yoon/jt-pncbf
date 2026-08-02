"""v2.8.0 Phase-2 C5 scoring — inertness proof + D4 prediction/falsifier from the per-cell sidecars.

D4 prediction: as tau grows, enumerate's collision rate rises FASTER than dual_solve's, and the gap widens
with control rate. Falsifier: the two degrade at the same rate in tau (gap ~ 0, no rate trend). X1: the
comparison requires BOTH projections at each (rate, tau); cells missing a projection are reported, not scored.
Writes inertness_proof.json and c5_summary.json. D4 is only declared scored when the full grid is present."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
C5 = REPO / "data/runs/v2.8.0/c5"
RATE_TO_ARM = {20: "A", 100: "C", 500: "D"}
TAUS = [0.0, 0.005, 0.01, 0.02, 0.05]
RATES = [20, 100, 500]
PROJS = ["dual_solve", "enumerate"]
EQ_BAND = 0.005


def load_cells():
    cells = {}
    for f in glob.glob(str(C5 / "cell_*.json")):
        c = json.loads(Path(f).read_text())
        cells[(c["rate_hz"], c["proj"], float(c["tau"]))] = c
    return cells


def main():
    cells = load_cells()
    # ---- inertness ----
    inert = {}
    for rate in RATES:
        for proj in PROJS:
            c = cells.get((rate, proj, 0.0))
            rec_p = REPO / f"data/runs/v2.8.0/s3_eval/rate_{RATE_TO_ARM[rate]}_{proj}.json"
            if c is None or not rec_p.exists():
                continue
            rec = float(json.loads(rec_p.read_text())["outcome"]["cps"])
            got = float(c["outcome"]["cps"]); d = got - rec
            inert[f"{rate}hz_{proj}"] = {"tau0_cps": got, "recorded_cps": rec, "delta": d, "within_band": abs(d) <= EQ_BAND}
    (C5 / "inertness_proof.json").write_text(json.dumps(inert, indent=2) + "\n")
    all_inert = bool(inert) and all(v["within_band"] for v in inert.values())

    # ---- collision(tau) slope per (rate, proj), X1: need both projections at a (rate,tau) ----
    slopes, curves, missing = {}, {}, []
    for rate in RATES:
        for proj in PROJS:
            pts = sorted((t, cells[(rate, proj, t)]["outcome"]["collision"]) for t in TAUS if (rate, proj, t) in cells)
            curves[f"{rate}hz_{proj}"] = pts
            if len(pts) >= 2:
                ts = np.array([p[0] for p in pts]); cs = np.array([p[1] for p in pts])
                slopes[f"{rate}hz_{proj}"] = float(np.polyfit(ts, cs, 1)[0])
    # X1 completeness: both projections present for each (rate,tau) actually run
    for rate in RATES:
        for t in TAUS:
            have = [(rate, p, t) in cells for p in PROJS]
            if any(have) and not all(have):
                missing.append({"rate": rate, "tau": t, "have": [p for p in PROJS if (rate, p, t) in cells]})

    gap = {}
    for rate in RATES:
        de, dd = slopes.get(f"{rate}hz_enumerate"), slopes.get(f"{rate}hz_dual_solve")
        gap[f"{rate}hz"] = (de - dd) if (de is not None and dd is not None) else None

    grid_complete = len(cells) == len(RATES) * len(PROJS) * len(TAUS)
    # verdict logic (only meaningful on the complete grid with X1 intact)
    verdict = "not scored (grid incomplete)"
    if grid_complete and not missing:
        gaps = [gap[f"{r}hz"] for r in RATES if gap.get(f"{r}hz") is not None]
        enum_faster = all(g > 0 for g in gaps)
        widens = len(gaps) >= 2 and all(gaps[i] <= gaps[i + 1] for i in range(len(gaps) - 1))
        if enum_faster and widens:
            verdict = "PREDICTION MET (enumerate collision rises faster; gap widens with rate)"
        elif all(abs(g) < 0.01 for g in gaps):
            verdict = "FALSIFIED (projections degrade at the same rate; gap ~ 0)"
        else:
            verdict = "MIXED (enum-minus-dual gap present but not monotonically widening with rate)"

    viol = {k: {"realized_median": c["realized_violation"].get("realized_violation_median"),
                "realized_p90": c["realized_violation"].get("realized_violation_p90"),
                "bound_median": c["realized_violation"].get("bound_median"),
                "realized_le_bound_frac": c["realized_violation"].get("realized_le_bound_frac")}
            for k, c in {f"{r}hz_{p}_tau{t}": cells[(r, p, t)] for (r, p, t) in cells}.items()}

    summary = {
        "grid_cells_present": len(cells), "grid_cells_expected": len(RATES) * len(PROJS) * len(TAUS),
        "grid_complete": grid_complete, "x1_incomplete_pairs": missing,
        "inertness_all_within_band": all_inert, "inertness": inert,
        "collision_vs_tau_slope": slopes, "collision_curves": curves,
        "enum_minus_dual_slope_gap_by_rate": gap,
        "D4_verdict": verdict, "realized_violation_vs_bound": viol,
        "cells": {f"{r}hz_{p}_tau{t}": {"cps": cells[(r, p, t)]["outcome"]["cps"],
                                        "collision": cells[(r, p, t)]["outcome"]["collision"]}
                  for (r, p, t) in cells}}
    (C5 / "c5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"cells present: {len(cells)}/{summary['grid_cells_expected']}  grid_complete={grid_complete}")
    print(f"inertness all within +-{EQ_BAND}: {all_inert}  ({ {k: round(v['delta'],5) for k,v in inert.items()} })")
    print(f"collision-vs-tau slopes: { {k: round(v,4) for k,v in slopes.items()} }")
    print(f"enum-minus-dual gap by rate: { {k: (round(v,4) if v is not None else None) for k,v in gap.items()} }")
    print(f"D4 verdict: {verdict}")
    if missing:
        print(f"X1 INCOMPLETE PAIRS (a projection missing at a run tau): {missing}")


if __name__ == "__main__":
    main()
