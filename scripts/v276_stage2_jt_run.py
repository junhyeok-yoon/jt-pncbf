"""v2.7.6 Stage-2 M4(c) — JOINT training (jt_pncbf) for quadrotor_3d under the Stage-2 h_star, value-init
from the Stage-2 OC value (a). Follows the v2.7.4 JT recipe EXACTLY (jt3d_run: collector=continuing,
inject_frac=0, bptt_T=30, n_steps=50000, sat_excess_threshold=[4.905]x4, seed 42, policy fresh). The ONLY
differences from v2.7.4 JT are (i) the band hazard in h_star (band_hazard.enabled), inherited from the
value-init, and (ii) band_collision_limit=4.0 in the IN-LOOP eval so best.pt is selected on banded cps
(04_eval s1 / 03_train s9 — a band-blind selection would prefer the band-ignoring checkpoint). 03_train s4.7
cps_floor is a RESERVED halt the JT trainer does not wire; the run relies on the active NaN/Inf and
V_S-gradient-leak halts. Prints the config diff and ABORTS on any out-of-scope key. No git, no securing."""
import argparse
from pathlib import Path

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--value-init", required=True)                 # Stage-2 OC value best.pt (a)
ap.add_argument("--collector", default="continuing", choices=["legacy", "continuing"])
ap.add_argument("--inject-frac", type=float, default=0.0)
ap.add_argument("--stage", default="full", choices=["smoke", "full"])
ap.add_argument("--beta", type=float, default=None)     # v2.8.1 S1: soft-rank beta (/m); must match the value-init cell
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["run"]["system"] = "quadrotor_3d"
    c["collection"]["collector"] = str(a.collector)
    c["collection"]["inject_frac"] = float(a.inject_frac)
    c["loss"]["policy"]["sat_excess_threshold"] = [4.905, 4.905, 4.905, 4.905]
    # v2.7.6 Stage-2: band hazard in h_star (inherited via value-init) + banded in-loop selection.
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 4.0
    # v2.8.1 S1 (Researcher item 2): TRAIN at the empty_fallback the v2.7.6/v2.8.0 value was trained at —
    # {mode:none, k:10}, the value recorded in 043415/015517 config.yaml. The per-system quadrotor_3d
    # {kstep, phases 1, k 3} block (exp_config B6.4) POSTDATES those launches and is consumed by the training
    # rollout + BPTT policy loss (_hardnet_params <- collection.py / losses.py), so it is NOT eval-only; only
    # mode:none is bit-parity with the pre-fallback filter (02_control s4). The shipped {kstep,...} is applied
    # at SCORING only. Committed exp_config.yaml is left untouched — the pin lives in this patch (whitelisted).
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    if a.beta is not None:                               # v2.8.1 S1: encoder beta must match the value-init cell
        c.setdefault("obs", {}).setdefault("quadrotor_3d", {})["beta"] = float(a.beta)
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"run.system", "collection.collector", "collection.inject_frac", "loss.policy.sat_excess_threshold",
           "env.band_hazard.enabled", "env.band_hazard.limit", "env.band_collision_limit",
           "filter.empty_fallback.quadrotor_3d.mode", "filter.empty_fallback.quadrotor_3d.phases",
           "filter.empty_fallback.quadrotor_3d.k", "obs.quadrotor_3d.beta"}
bad = []
print("=== Stage-2 M4(c) JT config diff (registered exp_config -> Stage-2 JT launch) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
q = _patched()["loss"]["policy"]
print(f"  situational: w_settle {q['w_settle']} rho {q['settle_rho']} w_appr {q['w_appr']} tau_brake "
      f"{q['tau_brake']} w_terminal {q['w_terminal']} w_terminal_v {q['w_terminal_v']}", flush=True)
print(f"  band_hazard: {_patched()['env']['band_hazard']} | band_collision_limit (in-loop selection): "
      f"{_patched()['env']['band_collision_limit']}", flush=True)
print(f"  [OK] training.jt.n_steps -> {a.steps} (via override); value_init {a.value_init}", flush=True)
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
print("CONFIG DIFF OK.", flush=True)

T.load_effective_config = _patched
r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=a.steps,
                 output_root=Path("data"), device="auto")
print(f"Stage-2 JT DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
