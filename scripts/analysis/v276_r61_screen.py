"""v2.7.6 R6.1 — eval-only screen of the v2.7.4 checkpoint on the full-range vs eval-IC pools.

Same checkpoint, same eval settings (v2.7.4 canonical deployed default: dt=dt_ctrl=0.05, max_steps=200,
stuck_window=60, empty_fallback=none), three pools:
  (gate)      seed23456 full-range = canonical -> cps MUST reproduce 0.8508367 within 1e-3 (harness trust).
  (baseline)  fullrange seed 42.
  (treatment) eval-IC   seed 42.
Per pool: outcome fractions + cps with within-seed bootstrap CIs, and an oob split into floor / ceiling /
wall by terminal position (registered rule: floor iff terminal z <= -oob_limit + tol). Baseline-vs-treatment
difference of every fraction with an UNPAIRED scene-bootstrap 95% CI (the two pools hold different scenes).
Adjudication of R6.1 (prediction + null) is emitted but verdict language stays confirmed/falsified/
inconclusive. Resume-safe: a complete per-pool JSON is reused. No training, no git, no securing.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
POOLS = REPO / "data/runs/v2.7.6/pools"
OUT = REPO / "data/runs/v2.7.6/r61_screen"; OUT.mkdir(parents=True, exist_ok=True)
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
OOB = float(REG["plant"]["oob_limit"]); TOL = 1e-6
REG_CPS = 0.8508367

PLAN = {
    "gate_seed23456":  "eval_full_quadrotor-3d-d2r_n2000_seed23456",
    "baseline_seed42": "eval_fullrange_quadrotor-3d-d2r_n2000_seed42",
    "treatment_seed42": "eval_evalicz_quadrotor-3d-d2r_n2000_seed42",
}
METRICS = ("cps", "reach", "collision", "stuck", "timeout", "oob",
           "oob_floor", "oob_ceiling", "oob_wall", "infeasibility")


def load_fw():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05},
            "filter": copy.deepcopy(ck["config"]["filter"])}
    over["filter"]["empty_fallback"] = {"mode": "none",
                                        "k": int(over["filter"].get("empty_fallback", {}).get("k", 10))}
    fw, cfg, _ = load_framework_from_checkpoint(CKPT, config_overrides=over)
    assert abs(float(cfg["env"]["dt"]) - 0.05) < 1e-12 and int(cfg["eval"]["max_steps"]) == 200
    assert cfg["filter"]["empty_fallback"]["mode"] == "none"
    return fw, cfg, ck


def per_episode(fw, cfg, ck, stem):
    res = evaluate(fw, POOLS / f"{stem}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name=CKPT.name, max_scenes=None, include_lqr_baseline=False)
    rows = res.episode_rows
    recs = []
    for r, tr in zip(rows, res.trajectories):
        n = int(float(r["n_steps"]))
        st = tr.filtered.states
        idx = min(n, st.shape[0] - 1)
        z_term = float(st[idx, 0, 2])
        oob = float(r["oob"])
        floor = 1.0 if (oob > 0.5 and z_term <= -OOB + TOL) else 0.0
        ceiling = 1.0 if (oob > 0.5 and z_term >= OOB - TOL) else 0.0
        wall = 1.0 if (oob > 0.5 and floor == 0.0 and ceiling == 0.0) else 0.0
        recs.append({"cps": float(r["cps_episode"]), "reach": float(r["reach"]),
                     "collision": float(r["collision"]), "stuck": float(r["stuck"]),
                     "timeout": float(r["timeout"]), "oob": float(r["oob"]),
                     "infeasibility": float(r["infeasible_step_frac"]),
                     "oob_floor": floor, "oob_ceiling": ceiling, "oob_wall": wall,
                     "z_term": z_term, "episode_idx": int(r["episode_idx"])})
    return recs


def agg(recs, n_boot, seed):
    arr = {m: np.array([r[m] for r in recs], float) for m in METRICS}
    rng = np.random.default_rng(seed)
    n = len(recs)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for m in METRICS:
        a = arr[m]; means = a[idx].mean(axis=1)
        out[m] = {"mean": round(float(a.mean()), 6),
                  "ci95": [round(float(np.percentile(means, 2.5)), 6),
                           round(float(np.percentile(means, 97.5)), 6)]}
    return out, arr


def diff_ci(arr_a, arr_b, n_boot, seed):
    """UNPAIRED bootstrap of (treatment - baseline) for each metric (independent resamples per pool)."""
    rng = np.random.default_rng(seed)
    na, nb = len(next(iter(arr_a.values()))), len(next(iter(arr_b.values())))
    ia = rng.integers(0, na, size=(n_boot, na)); ib = rng.integers(0, nb, size=(n_boot, nb))
    out = {}
    for m in METRICS:
        da = arr_a[m][ia].mean(axis=1); db = arr_b[m][ib].mean(axis=1)
        d = db - da                                       # treatment - baseline
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        out[m] = {"delta_treat_minus_base": round(float(arr_b[m].mean() - arr_a[m].mean()), 6),
                  "ci95": [round(lo, 6), round(hi, 6)], "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def run_pool(fw, cfg, ck, key, stem, n_boot, seed):
    p = OUT / f"{key}.json"
    if p.exists():
        d = json.loads(p.read_text()); print(f"[{key}] REUSED (cps={d['agg']['cps']['mean']:.6f})"); return d
    recs = per_episode(fw, cfg, ck, stem)
    a, arr = agg(recs, n_boot, seed)
    d = {"key": key, "stem": stem, "n": len(recs), "agg": a,
         "arr": {m: arr[m].tolist() for m in METRICS}}
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(f"[{key}] cps={a['cps']['mean']:.6f} reach={a['reach']['mean']:.4f} "
          f"coll={a['collision']['mean']:.4f} timeout={a['timeout']['mean']:.4f} "
          f"oob={a['oob']['mean']:.4f} (floor={a['oob_floor']['mean']:.4f} "
          f"ceil={a['oob_ceiling']['mean']:.4f} wall={a['oob_wall']['mean']:.4f})")
    return d


def main():
    fw, cfg, ck = load_fw()
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])

    gate = run_pool(fw, cfg, ck, "gate_seed23456", PLAN["gate_seed23456"], n_boot, bseed)
    gd = gate["agg"]["cps"]["mean"] - REG_CPS
    print(f"[HARNESS GATE] seed23456 cps {gate['agg']['cps']['mean']:.6f} vs registered {REG_CPS:.6f} "
          f"delta {gd:+.6f}  {'PASS' if abs(gd) <= 1e-3 else 'FAIL'}")
    if abs(gd) > 1e-3:
        raise SystemExit(f"HALT: harness does not reproduce the v2.7.4 headline (delta {gd:+.6f}).")

    base = run_pool(fw, cfg, ck, "baseline_seed42", PLAN["baseline_seed42"], n_boot, bseed)
    treat = run_pool(fw, cfg, ck, "treatment_seed42", PLAN["treatment_seed42"], n_boot, bseed)

    arr_b = {m: np.array(base["arr"][m]) for m in METRICS}
    arr_t = {m: np.array(treat["arr"][m]) for m in METRICS}
    diff = diff_ci(arr_b, arr_t, n_boot, bseed)

    # adjudication (R6.1): prediction + null. Vocabulary limited to confirmed/falsified/inconclusive.
    fl = diff["oob_floor"]; co = diff["collision"]; ti = diff["timeout"]; re = diff["reach"]
    floor_fell = fl["delta_treat_minus_base"] < 0 and fl["ci95"][1] < 0
    coll_worse = co["delta_treat_minus_base"] > 0 and co["ci95"][0] > 0
    time_worse = ti["delta_treat_minus_base"] > 0 and ti["ci95"][0] > 0
    if floor_fell and not (coll_worse or time_worse):
        verdict = "confirmed"
    elif (not floor_fell) or coll_worse or time_worse:
        verdict = "falsified" if (fl["ci95"][0] > 0 or coll_worse or time_worse) else "inconclusive"
    else:
        verdict = "inconclusive"
    reach_rose = re["delta_treat_minus_base"] > 0 and re["ci95"][0] > 0
    null_reading = ("reach rose with the floor-exit drop -> competence recovered, not just relabeled"
                    if reach_rose else
                    "reach did NOT rise (CI includes zero or negative) -> floor-exit change is relabeling "
                    "excluded losses, not recovered competence")

    summary = {"harness_gate_cps": gate["agg"]["cps"]["mean"], "harness_gate_delta": round(gd, 6),
               "baseline_stem": base["stem"], "treatment_stem": treat["stem"],
               "baseline_agg": base["agg"], "treatment_agg": treat["agg"],
               "difference_treat_minus_base": diff,
               "R6.1_adjudication": {"floor_exit_fell_CI_separated": floor_fell,
                                     "collision_worse_CI_separated": coll_worse,
                                     "timeout_worse_CI_separated": time_worse,
                                     "reach_rose_CI_separated": reach_rose,
                                     "verdict": verdict, "null_reading": null_reading},
               "escalation_note": ("seed-42 screen only; escalate to {99,12345} iff verdict-grade "
                                   "(CI-separated floor drop, no adverse collision/timeout).")}
    (OUT / "r61_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== R6.1 difference (treatment - baseline), 95% unpaired bootstrap ===")
    for m in ("oob_floor", "oob", "collision", "timeout", "reach", "cps"):
        print(f"  {m:12s} d={diff[m]['delta_treat_minus_base']:+.4f} CI{diff[m]['ci95']} "
              f"excl0={diff[m]['excludes_zero']}")
    print(f"\nR6.1 verdict: {verdict}\nnull: {null_reading}")


if __name__ == "__main__":
    main()
