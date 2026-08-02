"""v2.8.0 B — aggregate the 8 rate x projection cells: cell table, B1/B2 slope fits, falsifier scoring.

B1: slope of log(TV/s median) vs log(1/h) over A,C,D (1/h = 1/dt_ctrl = 20,100,500 Hz). Prediction:
    dual_solve slope < 0.25 (Lipschitz -> TV/s bounded); enumerate slope > 0.25 (chatters across ties).
    Falsifier: the two slopes are equal within their fit uncertainty.
B2 (null): inter-sample V_hat rise (p90) falls at first order in h in BOTH realizations -> slope of
    log(p90) vs log(1/h) ~ -1 for both, and they do not differ. Falsifier: the realizations differ in rate.
Arm B (fine grid, 20 Hz control) is reported separately, never folded into the A/C/D slope.
Reads s3_eval/rate_<arm>_<proj>.json. Writes s3_eval/rate_aggregate.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
S3 = REPO / "data/runs/v2.8.0/s3_eval"
HZ = {"A": 20.0, "B": 20.0, "C": 100.0, "D": 500.0}      # 1/dt_ctrl
SLOPE_ARMS = ["A", "C", "D"]


def load(arm, proj):
    p = S3 / f"rate_{arm}_{proj}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    coef, cov = np.polyfit(x, y, 1, cov=True)
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1])


cells = {}
for arm in "ABCD":
    for proj in ("enumerate", "dual_solve"):
        d = load(arm, proj)
        if d is not None:
            cells[f"{arm}_{proj}"] = d

rep = {"cells": {}, "b1": {}, "b2": {}}
for key, d in cells.items():
    o = d["outcome"]
    rep["cells"][key] = {
        "verify_config": d["verify_config"],
        "cps": o["cps"], "cps_ci": [o["cps_ci_lo"], o["cps_ci_hi"]], "reach": o["reach"],
        "collision": o["collision"], "collision_ci": [o["collision_ci_lo"], o["collision_ci_hi"]],
        "obstacle": d["collision_decomposition"]["obstacle"], "floor": d["collision_decomposition"]["band_lower"],
        "ceiling": d["collision_decomposition"]["band_upper"],
        "oob": o["oob"], "stuck": o["stuck"], "timeout": o["timeout"], "infeasibility": o["infeasibility"],
        "tv_per_s_median": d["tv_per_s"]["median"], "tv_per_s_p90": d["tv_per_s"]["p90"],
        "intersample_median": d["intersample_Vhat_rise"]["median"], "intersample_p90": d["intersample_Vhat_rise"]["p90"],
        "tv_empty_branch_frac": d["tv_branch_split"]["empty_branch_fraction_of_total_tv"],
    }

# ---- B1: log(TV/s median) vs log(1/h) over A,C,D ----
for proj in ("enumerate", "dual_solve"):
    if all(f"{a}_{proj}" in cells for a in SLOPE_ARMS):
        logx = [np.log(HZ[a]) for a in SLOPE_ARMS]
        logy = [np.log(max(cells[f'{a}_{proj}']['tv_per_s']['median'], 1e-12)) for a in SLOPE_ARMS]
        s, se, _ = fit(logx, logy)
        rep["b1"][proj] = {"slope": s, "slope_se": se, "tv_per_s_median": {a: cells[f'{a}_{proj}']['tv_per_s']['median'] for a in SLOPE_ARMS}}
if "enumerate" in rep["b1"] and "dual_solve" in rep["b1"]:
    sd, sed = rep["b1"]["dual_solve"]["slope"], rep["b1"]["dual_solve"]["slope_se"]
    se_, see = rep["b1"]["enumerate"]["slope"], rep["b1"]["enumerate"]["slope_se"]
    gap = se_ - sd; gap_se = float(np.hypot(sed, see))
    rep["b1"]["prediction_dual_below_0.25"] = sd < 0.25
    rep["b1"]["prediction_enum_above_0.25"] = se_ > 0.25
    rep["b1"]["slopes_equal_within_uncertainty"] = abs(gap) <= gap_se        # the FALSIFIER condition
    rep["b1"]["verdict"] = ("FALSIFIED (slopes equal within fit uncertainty)" if abs(gap) <= gap_se
                            else ("MET" if (sd < 0.25 and se_ > 0.25) else "PARTIAL (slopes differ but not both sides of 0.25)"))

# ---- B2: log(inter-sample p90) vs log(1/h) over A,C,D ; expect slope ~ -1 both, not differing ----
for proj in ("enumerate", "dual_solve"):
    if all(f"{a}_{proj}" in cells for a in SLOPE_ARMS):
        logx = [np.log(HZ[a]) for a in SLOPE_ARMS]
        p90 = [cells[f'{a}_{proj}']["intersample_Vhat_rise"]["p90"] for a in SLOPE_ARMS]
        logy = [np.log(max(v, 1e-12)) for v in p90]
        s, se, _ = fit(logx, logy)
        rep["b2"][proj] = {"slope_log_p90_vs_log_invh": s, "slope_se": se, "p90": {a: p90[i] for i, a in enumerate(SLOPE_ARMS)}}
if "enumerate" in rep["b2"] and "dual_solve" in rep["b2"]:
    sd, sed = rep["b2"]["dual_solve"]["slope_log_p90_vs_log_invh"], rep["b2"]["dual_solve"]["slope_se"]
    se_, see = rep["b2"]["enumerate"]["slope_log_p90_vs_log_invh"], rep["b2"]["enumerate"]["slope_se"]
    rep["b2"]["both_first_order"] = bool(abs(sd + 1.0) < 0.3 and abs(se_ + 1.0) < 0.3)   # slope ~ -1
    rep["b2"]["realizations_differ_in_rate"] = abs(se_ - sd) > float(np.hypot(sed, see))  # FALSIFIER
    rep["b2"]["verdict"] = ("FALSIFIED (realizations differ in rate)" if rep["b2"]["realizations_differ_in_rate"]
                            else "HOLDS (both fall ~first-order, not differing)")

(S3 / "rate_aggregate.json").write_text(json.dumps(rep, indent=2) + "\n")
print("=== cells ===")
for k, c in rep["cells"].items():
    print(f"  {k:16s} cps {c['cps']:.4f} reach {c['reach']:.4f} coll {c['collision']:.4f} "
          f"TV/s med {c['tv_per_s_median']:.2f} interV_p90 {c['intersample_p90']:.2e} TVempty {c['tv_empty_branch_frac']}")
print("=== B1 ===", json.dumps(rep["b1"], indent=2))
print("=== B2 ===", json.dumps(rep["b2"], indent=2))
