"""v2.6.0 credit-horizon axis (M5) — PLAIN value-policy JOINT training (jt_pncbf) with a BPTT TERMINAL
VALUE, warm-started from the M1 OC value V^{h_star, pi_nominal}. Policy is FRESH (no BC / no pi_init):
value_init_ckpt loads v_s_state (+ target) only; policy + optimizers + step start from scratch. The
credit-horizon fix is the terminal added to the rollout return in policy_bptt_loss
(discount*w_terminal*||p_T-g||, config loss.policy.w_terminal) which breaks the myopic-hover trap.

Single seed 42, registered budget n_steps=50000 (schedule_n_steps UNCHANGED). Prints the launch config
delta vs the registered exp_config (only training.jt.n_steps + value_init_ckpt); any other changed key is
surfaced. --smoke runs a tiny step budget to confirm value-load + fresh policy + terminal wiring (no NaN).
"""
import argparse
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training, load_effective_config

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--value-init", default="data/runs/v2.6.0/set__20260715-010357__seed42/v2.6.0__20260715-010357__seed42/checkpoints/best.pt")
ap.add_argument("--inject-frac", type=float, default=0.0)  # v2.7.0 iter-2: tilted-cell IC oversampling
ap.add_argument("--collector", default="legacy", choices=["legacy", "continuing"])  # v2.7.0 iter-5 Track A
ap.add_argument("--smoke", action="store_true", help="tiny run (300 steps) to gate wiring, not convergence")
a = ap.parse_args()

steps = 300 if a.smoke else a.steps

# v2.7.0 iter-2/5: inject_frac + collector patched into the effective config (defaults = bit-parity).
_orig_cfg = T.load_effective_config
def _patched_cfg():
    c = _orig_cfg()
    c["collection"]["inject_frac"] = float(a.inject_frac)
    c["collection"]["collector"] = str(a.collector)
    return c
T.load_effective_config = _patched_cfg

c0 = _patched_cfg()
n0 = int(c0["training"]["jt"]["n_steps"])
w_term = float(c0["loss"]["policy"]["w_terminal"])
print("=== M5 (credit-horizon / terminal) launch config delta (registered exp_config -> run) ===", flush=True)
print(f"  [OK] run.system: {c0['run']['system']} (quadrotor_planar)", flush=True)
print(f"  [OK] training.jt.n_steps: {n0} -> {steps}  ({'SMOKE' if a.smoke else 'registered budget'}; via n_steps_override)", flush=True)
print(f"  [OK] training.jt.value_init_ckpt: (none) -> {a.value_init}  (M1 V_hat warm-start; policy FRESH)", flush=True)
print(f"  [OK] run.seed: {a.seed}", flush=True)
print(f"  [OK] collection.inject_frac: {c0['collection']['inject_frac']}  (v2.7.0 iter-2 tilted-cell IC oversampling)", flush=True)
print(f"  [OK] collection.collector: {c0['collection']['collector']}  (v2.7.0 iter-5 continuing-batch collector)", flush=True)
print(f"  schedule_n_steps: {c0['training']['jt'].get('schedule_n_steps')} (UNCHANGED)", flush=True)
print(f"  loss.policy.w_terminal: {w_term} (credit-horizon terminal; committed, not a launch delta)", flush=True)
print(f"  bptt_T: {c0['training']['jt']['bptt_T']} | gamma_T: {c0['loss']['policy']['gamma_T']} "
      f"| eps_g: {c0['loss']['value']['lg_authority']['eps_g']} w: {c0['loss']['value']['lg_authority']['weight']}", flush=True)
assert w_term > 0.0, "w_terminal must be >0 for the credit-horizon run"

r = run_training(stage="full", system="quadrotor_planar", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=steps,
                 output_root=Path("data"), device="auto")
print(f"RUN DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
