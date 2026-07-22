"""v2.7.2 M5 — value-policy JOINT training (jt_pncbf) for quadrotor_3d, warm-started from the M3 OC value
V^{h_star, pi_nominal}. Co-trains the policy by BPTT through the HardNet map on V_hat (h_star + situational
running/terminal loss, generalized to the horizontal obstacle plane). Adopted defaults: continuing collector,
inject 0, bptt_T 30, n_steps 50000, seed 42. sat_excess_threshold is extended to the 4-channel box
[f_thr, tau_x, tau_y, tau_z] = [19.62, 1, 1, 1] (frozen per-channel planar values tiled to the 3 torques).
Prints the launch config delta and ABORTS on any out-of-scope key change."""
import argparse
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--value-init", required=True)                 # M3 OC value best.pt
ap.add_argument("--collector", default="continuing", choices=["legacy", "continuing"])
ap.add_argument("--inject-frac", type=float, default=0.0)
ap.add_argument("--stage", default="full", choices=["smoke", "full"])
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["run"]["system"] = "quadrotor_3d"
    c["collection"]["collector"] = str(a.collector)
    c["collection"]["inject_frac"] = float(a.inject_frac)
    # v2.7.3: per-rotor box [0, 4.905]^4 -> saturation threshold = per-rotor max on every channel.
    c["loss"]["policy"]["sat_excess_threshold"] = [4.905, 4.905, 4.905, 4.905]
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"run.system", "collection.collector", "collection.inject_frac",
           "loss.policy.sat_excess_threshold"}
bad = []
print("=== M5 config diff (registered exp_config -> M5 launch) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
q = _patched()["loss"]["policy"]
print(f"  situational: w_settle {q['w_settle']} rho {q['settle_rho']} w_appr {q['w_appr']} tau_brake "
      f"{q['tau_brake']} w_terminal {q['w_terminal']} w_terminal_v {q['w_terminal_v']}", flush=True)
print(f"  [OK] training.jt.n_steps -> {a.steps} (via override); value_init {a.value_init}", flush=True)
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
print("CONFIG DIFF OK.", flush=True)

T.load_effective_config = _patched
r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=a.steps,
                 output_root=Path("data"), device="auto")
print(f"M5 DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
