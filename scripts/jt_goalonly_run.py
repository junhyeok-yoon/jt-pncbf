"""v2.6.2 M5 — GOAL-ONLY ablation arm of the situational velocity objective: same as the FULL M4 JT run but
with loss.policy.w_appr = 0 (goal-quadratic + goal-gated settling ON, obstacle-gated approach OFF). Isolates
the obstacle-approach component (changes.md §3/§5). Patches load_effective_config to force w_appr=0 (race-free
config override, no exp_config mutation); everything else identical to the FULL run. Value warm-start from the
reused v2.6.1 M1; policy fresh; n_steps=50000.
"""
import argparse
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--value-init", required=True)
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["loss"]["policy"]["w_appr"] = 0.0          # ablation: obstacle-approach OFF (settling stays on)
    return c


T.load_effective_config = _patched

c0 = _patched()
print("=== M5 GOAL-ONLY ablation launch ===", flush=True)
print(f"  w_settle={c0['loss']['policy']['w_settle']}  w_appr={c0['loss']['policy']['w_appr']} (OVERRIDDEN 1.0->0.0)  "
      f"w_terminal_v={c0['loss']['policy']['w_terminal_v']}", flush=True)
print(f"  value_init={a.value_init}  seed={a.seed}  steps={a.steps}", flush=True)
assert c0["loss"]["policy"]["w_appr"] == 0.0

r = run_training(stage="full", system="quadrotor_planar", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=a.steps,
                 output_root=Path("data"), device="auto")
print(f"M5 GOAL-ONLY DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"best_cps={getattr(r,'best_cps',None)}", flush=True)
