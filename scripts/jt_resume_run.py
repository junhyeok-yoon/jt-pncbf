"""v2.6.0 M6 extension — FULL joint resume of a jt_pncbf run (pi + V_s + target + both optimizers + step),
to continue training past the registered budget when cps is still rising. resume_ckpt is a full
continuation (NOT a warm-start): schedules stay saturated (start_step >= schedule_n_steps=42000), no
vs_warmup re-entry (start_step >> 2000), terminal + detach_filter_coeffs carried by the live exp_config.

Usage: jt_resume_run.py --resume <ckpt.pt> --total-steps <N>   (N = TOTAL step budget, e.g. 100000 to add
50k on top of a 50k run). Prints the resume delta; asserts the ckpt carries the optimizer states.
"""
import argparse
from pathlib import Path

import torch

from src.frameworks.jt_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--resume", required=True, help="full joint step checkpoint to resume from")
ap.add_argument("--total-steps", type=int, required=True, help="TOTAL step budget (start_step + extra)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

ck = torch.load(a.resume, map_location="cpu", weights_only=False)
need = {"pi_state", "v_s_state", "v_s_target_state", "opt_pi_state", "opt_vs_state", "step"}
missing = need - set(ck.keys())
assert not missing, f"resume ckpt missing keys for a clean joint resume: {missing}"
start = int(ck["step"])
assert a.total_steps > start, f"--total-steps {a.total_steps} must exceed the ckpt step {start}"
print("=== M6 resume (full joint continuation) ===", flush=True)
print(f"  resume_ckpt: {a.resume}  (step {start})", flush=True)
print(f"  total-steps: {a.total_steps}  -> extra {a.total_steps - start} steps", flush=True)
print(f"  seed: {a.seed} | detach_filter_coeffs + w_terminal carried by exp_config", flush=True)

r = run_training(stage="full", system="quadrotor_planar", seed=a.seed,
                 resume_ckpt=Path(a.resume), n_steps_override=a.total_steps,
                 output_root=Path("data"), device="auto")
print(f"RESUME DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
