"""v2.6.0 Stage 2 M4 — value-policy JOINT training (jt_pncbf), warm-started from the M1 OC value
V^{h_star, pi_nominal}. Co-trains the policy by BPTT through the HardNet map on V_hat (h_star + eps_g
already wired). Single seed 42, registered budget n_steps=50000 (schedule_n_steps unchanged). Prints the
launch config delta vs the registered exp_config (only training.jt.n_steps + value_init_ckpt); any other
changed key = error, stop."""
import argparse
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training, load_effective_config

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--value-init", default="data/v2.6.0__20260715-010357__seed42/checkpoints/best.pt")
a = ap.parse_args()

# effective config delta preview (registered -> M4 launch): only n_steps + value_init_ckpt
c0 = load_effective_config()
n0 = int(c0["training"]["jt"]["n_steps"])
print("=== M4 launch config delta (registered exp_config -> M4) ===", flush=True)
print(f"  [OK] run.system: {c0['run']['system']} (default; quadrotor_planar)", flush=True)
print(f"  [OK] training.jt.n_steps: {n0} -> {a.steps}  (registered budget; via n_steps_override)", flush=True)
print(f"  [OK] training.jt.value_init_ckpt: (none) -> {a.value_init}  (M1 V_hat warm-start)", flush=True)
print(f"  [OK] run.seed: {a.seed}", flush=True)
print(f"  schedule_n_steps: {c0['training']['jt'].get('schedule_n_steps')} (UNCHANGED)", flush=True)
print(f"  h_star c_gain: {c0['env']['quadrotor_planar']['c_gain']} | eps_g: {c0['loss']['value']['lg_authority']['eps_g']} "
      f"w: {c0['loss']['value']['lg_authority']['weight']} (committed; not a launch delta)", flush=True)

r = run_training(stage="full", system="quadrotor_planar", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=a.steps,
                 output_root=Path("data"), device="auto")
print(f"M4 DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
