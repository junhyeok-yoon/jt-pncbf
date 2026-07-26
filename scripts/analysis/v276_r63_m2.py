"""v2.7.6 R6.3 M2 — re-screen R6.1 with the tilt-floor k* pool vs the full-range baseline.

Reuses the R6.1 harness (load_fw / per_episode / agg / diff_ci / METRICS from v276_r61_screen) and the R6.1
full-range baseline (baseline_seed42.json, already computed). Runs the v2.7.4 checkpoint on the k*-tilt pool
at seed 42 and reports the same table with 95% CI + separation, plus the registered prediction / falsifier /
null adjudication. Single seed. No git, no securing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analysis.v276_r61_screen import load_fw, per_episode, agg, diff_ci, METRICS, POOLS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SCR = REPO / "data/runs/v2.7.6/r61_screen"
OUT = REPO / "data/runs/v2.7.6/r63_m2"; OUT.mkdir(parents=True, exist_ok=True)
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
KSTAR = json.loads((REPO / "data/runs/v2.7.6/r63_kscreen/kscreen_summary.json").read_text())["k_star"]
KSTEM = f"eval_tiltk{str(KSTAR).replace('.', 'p')}_quadrotor-3d-d2r_n2000_seed42"


def main():
    base = json.load(open(SCR / "baseline_seed42.json"))       # R6.1 full-range baseline (reused)
    arr_b = {m: np.array(base["arr"][m]) for m in METRICS}

    fw, cfg, ck = load_fw()
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    recs = per_episode(fw, cfg, ck, KSTEM)
    t_agg, arr_t = agg(recs, n_boot, bseed)
    json.dump({"key": f"treatment_tiltk{KSTAR}", "stem": KSTEM, "n": len(recs), "agg": t_agg,
               "arr": {m: arr_t[m].tolist() for m in METRICS}}, open(OUT / "treatment_kstar.json", "w"), indent=2)

    diff = diff_ci(arr_b, arr_t, n_boot, bseed)               # treatment(k*) - baseline(full-range)
    fl = diff["oob_floor"]; co = diff["collision"]; ti = diff["timeout"]; re = diff["reach"]
    floor_rose = fl["delta_treat_minus_base"] > 0 and fl["ci95"][0] > 0
    floor_fell = fl["delta_treat_minus_base"] < 0 and fl["ci95"][1] < 0
    coll_worse = co["delta_treat_minus_base"] > 0 and co["ci95"][0] > 0
    time_worse = ti["delta_treat_minus_base"] > 0 and ti["ci95"][0] > 0
    reach_rose = re["delta_treat_minus_base"] > 0 and re["ci95"][0] > 0
    # prediction: oob(floor) does NOT rise. falsifier: it rises CI-separated.
    if floor_rose:
        verdict = "falsified"
    elif not (coll_worse or time_worse):
        verdict = "confirmed"
    else:
        verdict = "inconclusive"
    null_reading = ("floor fell AND reach rose -> competence recovered" if (floor_fell and reach_rose) else
                    "floor fell but reach did not rise -> relabeling excluded losses, not recovered competence"
                    if floor_fell else "floor did not fall; null model n/a (prediction is non-rise, not fall)")
    summary = {"k_star": KSTAR, "treatment_stem": KSTEM,
               "baseline_agg": base["agg"], "treatment_agg": t_agg,
               "difference_treat_minus_base": diff,
               "R6.3_M2_adjudication": {"floor_rose_CI_separated": floor_rose, "floor_fell_CI_separated": floor_fell,
                                        "collision_worse": coll_worse, "timeout_worse": time_worse,
                                        "reach_rose_CI_separated": reach_rose,
                                        "verdict": verdict, "null_reading": null_reading},
               "escalation_note": "single seed 42; escalate to {99,12345} only if favorable and verdict-grade"}
    (OUT / "r63_m2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def row(tag, a):
        print(f"{tag:16s} cps={a['cps']['mean']:.5f} reach={a['reach']['mean']:.4f} coll={a['collision']['mean']:.4f} "
              f"timeout={a['timeout']['mean']:.4f} oob_floor={a['oob_floor']['mean']:.4f}")
    row("baseline", base["agg"]); row(f"k*={KSTAR}", t_agg)
    print("\n=== diff (k* - baseline), 95% unpaired bootstrap ===")
    for m in ("oob_floor", "oob", "collision", "timeout", "reach", "cps"):
        print(f"  {m:12s} d={diff[m]['delta_treat_minus_base']:+.4f} CI{diff[m]['ci95']} excl0={diff[m]['excludes_zero']}")
    print(f"\nR6.3 M2 verdict: {verdict}\nnull: {null_reading}")


if __name__ == "__main__":
    main()
