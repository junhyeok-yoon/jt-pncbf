"""v2.8.0 S3 — JOINT training arm for quadrotor_3d under the settled terminal, ablating filter.projection.

Follows the v2.7.6 Stage-2 JT recipe EXACTLY (collector=continuing, inject_frac=0, bptt_T=30,
n_steps=50000, sat_excess_threshold=[4.905]x4, seed 42, policy fresh, band hazard in h_star, banded
in-loop selection), value-init from the SHARED v2.7.6 OC value (filter-independent: OC uses CBFQPFilter,
not filter.projection — M1). The NEW TERMINAL (goal_angrate_radius=0.48, w_settle_ang=0.39) is inherited
from the committed config. The ONLY per-arm difference is `filter.projection`. empty_fallback is forced to
none for TRAINING (the {kstep,phases1,k3} shipped fallback is eval-only, applied at M4). P2 instrumentation
(loss.policy.log_jac_classes) is on for BOTH arms (identical) and is detached/no-effect. Prints the config
diff, ABORTS on any out-of-scope key, and persists the P2 series to jac_classes.csv."""
import argparse, csv
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training
from src.frameworks.jt_pncbf import losses as L

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OC_VALUE = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/checkpoints/best.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--projection", required=True, choices=["dual_solve", "enumerate"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--stage", default="full", choices=["smoke", "full"])
ap.add_argument("--resume-ckpt", default=None,
                help="v2.8.0 S3 R1: resume from a step_NNNNNN.pt (restores nets+optimizers+step; cold buffer/RNG). "
                     "Mutually exclusive with the OC value-init.")
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["run"]["system"] = "quadrotor_3d"
    c["collection"]["collector"] = "continuing"
    c["collection"]["inject_frac"] = 0.0
    c["loss"]["policy"]["sat_excess_threshold"] = [4.905, 4.905, 4.905, 4.905]
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 4.0
    c["filter"]["projection"] = a.projection                       # THE AXIS (only per-arm difference)
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}       # eval-only kstep off during training
    c["loss"]["policy"]["log_jac_classes"] = True                   # P2 instrumentation (both arms; no-effect)
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"run.system", "collection.collector", "collection.inject_frac", "loss.policy.sat_excess_threshold",
           "env.band_hazard.enabled", "env.band_hazard.limit", "env.band_collision_limit",
           "filter.projection", "filter.empty_fallback.mode", "filter.empty_fallback.k",
           "filter.empty_fallback.phases", "filter.empty_fallback.quadrotor_3d.mode",
           "filter.empty_fallback.quadrotor_3d.phases", "filter.empty_fallback.quadrotor_3d.k",
           "loss.policy.log_jac_classes"}
bad = []
print(f"=== v2.8.0 S3 JT config diff (arm projection={a.projection}) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
q = _patched()
pol = q["loss"]["policy"]; env = q["env"]
print(f"  NEW TERMINAL: goal_angrate_radius={env.get('goal_angrate_radius')} w_settle={pol['w_settle']} "
      f"w_settle_ang={pol.get('w_settle_ang')} settle_rho={pol['settle_rho']} w_terminal={pol['w_terminal']} "
      f"w_terminal_v={pol['w_terminal_v']}", flush=True)
print(f"  filter.projection={q['filter']['projection']} empty_fallback(train)={q['filter']['empty_fallback']}", flush=True)
print(f"  value_init={OC_VALUE.name} (v2.7.6 OC, filter-independent shared value)", flush=True)
# guard the fixed registered values
assert abs(float(env.get("goal_angrate_radius", -1)) - 0.48) < 1e-12, "goal_angrate_radius != 0.48"
assert abs(float(pol.get("w_settle_ang", -1)) - 0.39) < 1e-12, "w_settle_ang != 0.39"
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
if not OC_VALUE.exists():
    raise SystemExit(f"ABORT: shared OC value not found: {OC_VALUE}")
print("CONFIG DIFF OK.", flush=True)

L._JAC_CLASS_LOG.clear()
T.load_effective_config = _patched
if a.resume_ckpt:
    print(f"RESUME from {a.resume_ckpt} (restores nets+optimizers+step; cold buffer + fresh RNG)", flush=True)
    r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed,
                     resume_ckpt=Path(a.resume_ckpt), n_steps_override=a.steps,
                     output_root=REPO / "data", device="auto")
else:
    r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed,
                     value_init_ckpt=OC_VALUE, n_steps_override=a.steps,
                     output_root=REPO / "data", device="auto")
run_dir = Path(getattr(r, "run_dir", "data"))
# persist P2 series
if L._JAC_CLASS_LOG:
    cols = list(L._JAC_CLASS_LOG[0].keys())
    with (run_dir / "jac_classes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for row in L._JAC_CLASS_LOG:
            w.writerow(row)
    print(f"P2 series: {len(L._JAC_CLASS_LOG)} rows -> {run_dir/'jac_classes.csv'}", flush=True)
else:
    print("WARNING: P2 series empty (log_jac_classes did not fire)", flush=True)
print(f"S3 JT DONE arm={a.projection} run_dir={run_dir} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
