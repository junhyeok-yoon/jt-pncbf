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
# v2.8.2 S1 conditions (CTRL/M1/M2/M3) — per-condition infeasibility machinery (s1_premeasure.md §6). None => exp_config default (OFF).
ap.add_argument("--empty-mode", default=None, choices=["argmin", "prox"])   # M1: continuous empty branch
ap.add_argument("--empty-prox-temp", type=float, default=None)             # M1: prox continuity scale gamma (=s̄=2.200)
ap.add_argument("--w-infeas", type=float, default=None)                    # M2: infeasibility-margin cost weight
ap.add_argument("--infeas-delta", type=float, default=None)               # M2: relu headroom delta (required iff w_infeas>0)
ap.add_argument("--w-du", type=float, default=None)                        # M3: filtered-command rate weight
ap.add_argument("--alpha-unsafe", type=float, default=None)                # v2.8.2 iter: alpha_unsafe axis (filter.alpha_unsafe)
ap.add_argument("--infeas-clip", action="store_true")                      # v2.8.2 I3: clip w_infeas term to min(relu,delta)
ap.add_argument("--train-empty-fallback", default=None)                    # v2.8.2 FLOOR: "mode:phases:k" (e.g. kstep:2:5) -> TRAIN with the k-step empty fallback ACTIVE (per-system quad sub-key); default None = CTRL {none,k10} inert
ap.add_argument("--w-floor-recov", type=float, default=None)               # v2.8.2 cond(c): floor-recovery shaping weight (loss.policy.w_floor_recov); None => exp_config default (OFF)
ap.add_argument("--floor-recov-zthr", type=float, default=None)            # v2.8.2 cond(c): floor-recovery altitude threshold z_thr (loss.policy.floor_recov_zthr); None => default 0.5
ap.add_argument("--dry-run", action="store_true")                          # print the config diff and exit (no training)
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
    if a.train_empty_fallback is not None:               # v2.8.2 FLOOR: TRAIN with the k-step fallback ACTIVE (per-system quad override; consumed by the training rollout/BPTT per lines 46-51)
        _m, _p, _k = a.train_empty_fallback.split(":")
        c["filter"]["empty_fallback"]["quadrotor_3d"] = {"mode": str(_m), "phases": int(_p), "k": int(_k)}
    if a.beta is not None:                               # v2.8.1 S1: encoder beta must match the value-init cell
        c.setdefault("obs", {}).setdefault("quadrotor_3d", {})["beta"] = float(a.beta)
    # v2.8.2 S1 conditions: per-condition infeasibility machinery (s1_premeasure.md §6; None => exp_config default OFF)
    if a.empty_mode is not None:
        c["filter"]["empty_mode"] = str(a.empty_mode)
    if a.empty_prox_temp is not None:
        c["filter"]["empty_prox_temp"] = float(a.empty_prox_temp)
    if a.w_infeas is not None:
        c["loss"]["policy"]["w_infeas"] = float(a.w_infeas)
    if a.infeas_delta is not None:
        c["loss"]["policy"]["infeas_delta"] = float(a.infeas_delta)
    if a.w_du is not None:
        c["loss"]["policy"]["w_du"] = float(a.w_du)
    if a.w_floor_recov is not None:                      # v2.8.2 cond(c): floor-recovery shaping weight (default OFF)
        c["loss"]["policy"]["w_floor_recov"] = float(a.w_floor_recov)
    if a.floor_recov_zthr is not None:                   # v2.8.2 cond(c): floor-recovery altitude threshold z_thr
        c["loss"]["policy"]["floor_recov_zthr"] = float(a.floor_recov_zthr)
    if a.alpha_unsafe is not None:                        # v2.8.2 iter: alpha_unsafe axis (only diff from CTRL)
        c["filter"]["alpha_unsafe"] = float(a.alpha_unsafe)
    if a.infeas_clip:                                     # v2.8.2 I3: clipped w_infeas term
        c["loss"]["policy"]["infeas_clip"] = True
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
           "filter.empty_fallback.quadrotor_3d.k", "obs.quadrotor_3d.beta",
           "filter.empty_mode", "filter.empty_prox_temp",                          # v2.8.2 M1
           "loss.policy.w_infeas", "loss.policy.infeas_delta", "loss.policy.w_du",  # v2.8.2 M2/M3
           "filter.alpha_unsafe",                                                    # v2.8.2 iter: alpha_unsafe axis
           "loss.policy.infeas_clip",                                                # v2.8.2 I3: clipped term
           "loss.policy.w_floor_recov", "loss.policy.floor_recov_zthr"}              # v2.8.2 cond(c): floor-recovery shaping
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
if a.dry_run:
    raise SystemExit(0)

T.load_effective_config = _patched
r = run_training(stage=a.stage, system="quadrotor_3d", seed=a.seed,
                 value_init_ckpt=Path(a.value_init), n_steps_override=a.steps,
                 output_root=Path("data"), device="auto")
print(f"Stage-2 JT DONE run_dir={getattr(r,'run_dir',r)} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)}", flush=True)
