"""v2.7.2 M3 — learn V^{h_star, pi_nominal} for quadrotor_3d via OC-PNCBF value regression (fixed nominal
cascaded-PD). Budget = 50000 value steps = epochs 100 x grad_steps 500 (registered; adopted defaults:
collector=continuing, inject_frac=0). h_star labeling via the generalized value_target_barrier (system
interface). eps_g/R2 lg_authority is quadrotor_planar-only (6-state near-B0 sampler) and is INACTIVE for
quadrotor_3d — the M3 authority gate measures L_g V_hat directly. Prints the config diff vs registered
exp_config and ABORTS if any out-of-scope key changed."""
import argparse
from pathlib import Path

import src.frameworks.oc_pncbf.train as T
from src.frameworks.oc_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=100)          # 100 x grad_steps(500) = 50000 value steps
ap.add_argument("--inject-frac", type=float, default=0.0)
ap.add_argument("--collector", default="continuing", choices=["legacy", "continuing"])
ap.add_argument("--stage", default="full", choices=["smoke", "full"])
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["run"]["system"] = "quadrotor_3d"
    c["training"]["oc_pncbf"]["epochs"] = int(a.epochs)
    c["collection"]["inject_frac"] = float(a.inject_frac)
    c["collection"]["collector"] = str(a.collector)
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"run.system", "training.oc_pncbf.epochs", "collection.inject_frac", "collection.collector"}
bad = []
print("=== M3 config diff (registered exp_config -> M3 launch) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
q = _patched()
print("  quadrotor_3d c_gain:", q["env"]["quadrotor_3d"]["c_gain"],
      "| lg_authority weight:", q["loss"]["value"]["lg_authority"]["weight"],
      "(INACTIVE for quadrotor_3d — planar-only near-B0 sampler)", flush=True)
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
print("CONFIG DIFF OK.", flush=True)

T.load_effective_config = _patched
r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed, output_root=Path("data"), device="auto")
print(f"M3 DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)}", flush=True)
