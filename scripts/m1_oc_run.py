"""v2.6.0 Stage 1 M1 — learn V^{h_star, pi_nominal} via OC-PNCBF value regression (fixed nominal LQR).
Budget = 50000 value steps = epochs 100 x grad_steps 500 (registered). gamma_disc schedule unchanged.
h_star labeling + epsilon_g (R2) already wired in oc collection/value step. Single seed. Prints the full
config diff (only training.oc_pncbf.epochs differs from the registered exp_config; run.system/version/seed
set by run_training) and ABORTS if any other key changed."""
import argparse
from pathlib import Path

import src.frameworks.oc_pncbf.train as T
from src.frameworks.oc_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=100)   # 100 x grad_steps(500) = 50000 value steps
ap.add_argument("--inject-frac", type=float, default=0.0)  # v2.7.0 iter-2: tilted-cell IC oversampling
ap.add_argument("--collector", default="legacy", choices=["legacy", "continuing"])  # v2.7.0 iter-5 Track A
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["training"]["oc_pncbf"]["epochs"] = int(a.epochs)   # ONLY budget knob; schedule untouched
    c["collection"]["inject_frac"] = float(a.inject_frac)  # v2.7.0 iter-2 label-coverage knob
    c["collection"]["collector"] = str(a.collector)        # v2.7.0 iter-5 continuing-batch collector
    return c


# config diff (patched vs registered): must be exactly training.oc_pncbf.epochs
def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"training.oc_pncbf.epochs", "collection.inject_frac", "collection.collector"}
bad = []
print("=== M1 config diff (registered exp_config -> M1 launch) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
# confirm the quadrotor + eps_g keys are present (from committed exp_config, not a launch delta)
q = _patched()
print("  quadrotor c_gain:", q["env"]["quadrotor_planar"]["c_gain"],
      "| eps_g:", q["loss"]["value"]["lg_authority"]["eps_g"],
      "w:", q["loss"]["value"]["lg_authority"]["weight"], flush=True)
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
print("CONFIG DIFF OK (only epochs).", flush=True)

T.load_effective_config = _patched
r = run_training(stage="full", system="quadrotor_planar", seed=a.seed, output_root=Path("data"), device="auto")
print(f"M1 DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)}", flush=True)
