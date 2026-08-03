"""v2.7.6 Stage-2 M4(a) — OC value V^{h_star,pi} under the STAGE-2 h_star (band branch on), following the
083533 recipe EXACTLY (collector=continuing, inject_frac=0, epochs=100 = 50000 value steps, seed 42). The
ONLY difference from 083533 is the band hazard: env.band_hazard.enabled=true (h_star vertical branch) and
env.band_collision_limit=4.0 (|z|>=4 collision surface). Prints the config diff vs registered exp_config and
ABORTS if any out-of-scope key changed. No git, no securing."""
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
ap.add_argument("--beta", type=float, default=None)     # v2.8.1 S1 beta-screen: soft-rank beta (/m). Unset -> 2.0 default
a = ap.parse_args()

_orig = T.load_effective_config


def _patched():
    c = _orig()
    c["run"]["system"] = "quadrotor_3d"
    c["training"]["oc_pncbf"]["epochs"] = int(a.epochs)
    c["collection"]["inject_frac"] = float(a.inject_frac)
    c["collection"]["collector"] = str(a.collector)
    # v2.7.6 Stage-2: the band hazard in h_star — the sole intended difference from 083533. The OC value
    # regresses this band-aware h_star (band_hazard). band_collision stays 0.0 IN TRAINING: this stage's
    # nominal is a FIXED cascaded-PD that cannot learn to defend the band, so banded cps is structurally
    # immobile for the whole run and the cps_floor halt would measure a quantity the stage cannot move
    # (03_train s4.7). Banded numbers are reported at M4(b) eval instead.
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 0.0
    # v2.8.1 S1 (Researcher item 2): TRAIN at the empty_fallback the v2.7.6/v2.8.0 value was trained at —
    # {mode:none, k:10}, the value recorded in 043415/015517 config.yaml. The per-system quadrotor_3d
    # {kstep, phases 1, k 3} block (exp_config B6.4) POSTDATES those launches and is consumed by the training
    # rollout + BPTT policy loss (_hardnet_params <- collection.py / losses.py), so it is NOT eval-only; only
    # mode:none is bit-parity with the pre-fallback filter (02_control s4). The shipped {kstep,...} is applied
    # at SCORING only. Committed exp_config.yaml is left untouched — the pin lives in this patch (whitelisted).
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    if a.beta is not None:                               # v2.8.1 S1 beta-screen: the ONLY per-cell difference
        c.setdefault("obs", {}).setdefault("quadrotor_3d", {})["beta"] = float(a.beta)
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(_flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


base = _flat(_orig()); patched = _flat(_patched())
allowed = {"run.system", "training.oc_pncbf.epochs", "collection.inject_frac", "collection.collector",
           "env.band_hazard.enabled", "env.band_hazard.limit", "env.band_collision_limit",
           "filter.empty_fallback.quadrotor_3d.mode", "filter.empty_fallback.quadrotor_3d.phases",
           "filter.empty_fallback.quadrotor_3d.k", "obs.quadrotor_3d.beta"}
bad = []
print("=== Stage-2 M4(a) OC config diff (registered exp_config -> Stage-2 OC launch) ===", flush=True)
for k in sorted(set(base) | set(patched)):
    if base.get(k) != patched.get(k):
        ok = k in allowed
        print(f"  [{'OK' if ok else 'ERROR'}] {k}: {base.get(k)!r} -> {patched.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
q = _patched()
print("  quadrotor_3d c_gain:", q["env"]["quadrotor_3d"]["c_gain"],
      "| h_scale:", q["env"]["h_scale"], "| band_hazard:", q["env"]["band_hazard"],
      "| band_collision_limit:", q["env"]["band_collision_limit"], flush=True)
if bad:
    raise SystemExit(f"ABORT: out-of-scope config keys changed: {bad}")
print("CONFIG DIFF OK.", flush=True)

T.load_effective_config = _patched
r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed, output_root=Path("data"), device="auto")
print(f"Stage-2 OC DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)}", flush=True)
