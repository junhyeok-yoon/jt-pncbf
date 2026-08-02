"""v2.8.0 S3 amendment A — P3 as a PAIRED difference across the two arms.

Both arms are evaluated on the same pool, so every scene index appears in both. Scene difficulty
dominates the marginal variance and cancels in the per-scene difference, so the paired statistic is a
much tighter estimator of the SAME quantity the marginal comparison targets. For each scene present in
both arms:  d_i = metric_i^dual - metric_i^enum.  Report mean(d) with a bootstrap CI over scenes, on the
full pool AND the hold-feasible subset (theta0<=60), for cps AND collision_obstacle separately. Confirm
the scene indices align; if not -> HALT (the pairing is the whole point). Report the marginal comparison
(independent per-arm bootstrap CIs) as well, and state which is which; if paired and marginal disagree in
sign, report the disagreement rather than choosing.

Inputs: data/runs/v2.8.0/s3_eval/m4_dual.json, m4_enum.json (each with full per-episode `episode_cause`),
scratch tilt.npy (per-IC tilt, global order). Sidecar: data/runs/v2.8.0/p3_paired.json"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SCRATCH = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
S3 = REPO / "data/runs/v2.8.0/s3_eval"
BOOT_SEED, NBOOT = 20260508, 10000

_ap = argparse.ArgumentParser()
_ap.add_argument("--dual-tag", default="m4_dual")
_ap.add_argument("--enum-tag", default="m4_enum")
_ap.add_argument("--out", default="p3_paired.json")
_A = _ap.parse_args()


def load_arm(tag):
    d = json.loads((S3 / f"{tag}.json").read_text())
    ec = sorted(d["episode_cause"], key=lambda e: e["episode_idx"])
    idx = np.array([e["episode_idx"] for e in ec])
    cps = np.array([float(e["cps_episode"]) for e in ec])
    obst = np.array([1.0 if e.get("collision_cause", "") == "obstacle" else 0.0 for e in ec])
    return idx, cps, obst


idx_d, cps_d, obst_d = load_arm(_A.dual_tag)
idx_e, cps_e, obst_e = load_arm(_A.enum_tag)

# --- scene-index alignment gate ---
if idx_d.shape != idx_e.shape or not np.array_equal(idx_d, idx_e):
    print("HALT: scene indices do not align between the two arms -> pairing invalid", file=sys.stderr)
    raise SystemExit(1)
N = idx_d.shape[0]

# hold-feasible subset (theta0 <= 60), from scratch tilt.npy in global order
tilt = np.load(SCRATCH / "tilt.npy")
assert tilt.shape[0] == N, f"tilt length {tilt.shape[0]} != n {N}"
# cross-check against floor_feasibility per-IC tilt (same computation should agree)
ff = json.loads((REPO / "data/runs/v2.8.0/floor_feasibility.json").read_text())
tilt_ff = np.array([d["tilt_deg"] for d in ff["ic"]])
maxdiff = float(np.abs(tilt - tilt_ff).max())
hold = tilt <= 60.0

rng = np.random.default_rng(BOOT_SEED)


def paired(dv, ev, mask):
    """paired per-scene difference dual-enum over mask, with bootstrap CI over scenes."""
    d = (dv - ev)[mask]
    n = d.shape[0]
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return {"n": int(n), "mean_diff": float(d.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "dual_mean": float(dv[mask].mean()), "enum_mean": float(ev[mask].mean())}


def marginal(dv, ev, mask):
    """independent per-arm bootstrap CIs (the registered reading), plus the difference of marginals."""
    d, e = dv[mask], ev[mask]; n = d.shape[0]
    bd = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    be = np.array([e[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return {"n": int(n),
            "dual": {"mean": float(d.mean()), "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]},
            "enum": {"mean": float(e.mean()), "ci": [float(np.percentile(be, 2.5)), float(np.percentile(be, 97.5))]},
            "diff_of_marginals": float(d.mean() - e.mean()),
            "ci_overlap": not (np.percentile(bd, 2.5) > np.percentile(be, 97.5) or
                               np.percentile(be, 2.5) > np.percentile(bd, 97.5))}


def sign(x, tol=1e-9):
    return 0 if abs(x) < tol else (1 if x > 0 else -1)


rep = {"n": N, "nboot": NBOOT, "boot_seed": BOOT_SEED,
       "scene_index_alignment": "confirmed (idx_dual == idx_enum)", "tilt_vs_ff_maxdiff": maxdiff,
       "note": "paired = tighter estimator of the same quantity; marginal = registered reading. "
               "positive diff (dual-enum) means dual scores higher (cps) or collides more (obstacle)."}
for metric, dv, ev in (("cps", cps_d, cps_e), ("collision_obstacle", obst_d, obst_e)):
    for sub, mask in (("full", np.ones(N, bool)), ("hold_feasible", hold)):
        p = paired(dv, ev, mask); m = marginal(dv, ev, mask)
        disagree = sign(p["mean_diff"]) != sign(m["diff_of_marginals"]) and sign(p["mean_diff"]) != 0 and sign(m["diff_of_marginals"]) != 0
        rep[f"{metric}__{sub}"] = {"paired": p, "marginal": m,
                                   "sign_disagreement_paired_vs_marginal": bool(disagree)}
rep["dual_tag"] = _A.dual_tag; rep["enum_tag"] = _A.enum_tag
(REPO / "data/runs/v2.8.0" / _A.out).write_text(json.dumps(rep, indent=2) + "\n")

print(f"P3 paired [{_A.dual_tag} vs {_A.enum_tag}] (n={N}, tilt-vs-ff maxdiff {maxdiff:.2e}):")
for metric in ("cps", "collision_obstacle"):
    for sub in ("full", "hold_feasible"):
        r = rep[f"{metric}__{sub}"]; p, m = r["paired"], r["marginal"]
        print(f"  {metric:18s} {sub:13s} n={p['n']:4d} | PAIRED d(dual-enum)={p['mean_diff']:+.4f} "
              f"[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] | MARGINAL diff={m['diff_of_marginals']:+.4f} "
              f"(overlap={m['ci_overlap']})" + ("  <SIGN-DISAGREE>" if r["sign_disagreement_paired_vs_marginal"] else ""))
