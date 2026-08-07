"""v2.8.2 PPO baseline launcher (DI gate + quadrotor_3d).

Assembles the PPO effective config (shared base+exp env/network unchanged + additive `ppo:` block), prints the
Gate-2 config diff against the CTRL run's recorded config.yaml classifying every difference as ENVIRONMENTAL
(a defect to fix) or ALGORITHMIC (expected — JT machinery PPO lacks, or the ppo block), prints the Gate-2
environment checklist, then runs training. --dry-run stops after the diff.

Examples:
  # Gate 1 (DI, short budget):
  python scripts/ppo_run.py --system double_integrator --num-envs 512 --n-iterations 120 --eval-cadence-iters 10
  # Gate 2 dry-run diff (quadrotor):
  python scripts/ppo_run.py --system quadrotor_3d --dry-run
  # Quadrotor full budget (30000 update steps = 300 * 4 * 25):
  python scripts/ppo_run.py --system quadrotor_3d --num-envs 1024 --n-iterations 300 --eval-cadence-iters 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.eval.build_pools import resolve_pool_or_raise
from src.frameworks.ppo.config import load_ppo_config
from src.frameworks.ppo.train import run_ppo_training

CTRL_CONFIG = Path("data/runs/v2.8.2/set__20260803-063606__seed42/"
                   "v2.8.2__jt__20260803-063606__seed42/config.yaml")

# Environment-defining blocks: a difference here vs CTRL is an ENVIRONMENTAL defect. Everything else is
# algorithmic (framework machinery / the additive ppo block).
_ENV_PREFIXES = ("env.", "obstacle.", "obs.", "scene_train.", "network.", "lqr.",
                 "eval.max_steps", "eval.scene", "eval.in_loop", "eval.full", "eval.dt")

INLOOP_POOLS = {
    "double_integrator": "eval_inloop_di_n500_seed12345",
    "quadrotor_3d": "eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678",
}


def _flat(d, p=""):
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            o.update(_flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o


def _is_env(key: str) -> bool:
    return any(key == pfx or key.startswith(pfx) for pfx in _ENV_PREFIXES)


def gate2_diff(config: dict, system: str) -> int:
    if not CTRL_CONFIG.exists():
        print(f"[gate2] CTRL config not found at {CTRL_CONFIG}; skipping diff.")
        return 0
    ctrl = yaml.safe_load(CTRL_CONFIG.read_text(encoding="utf-8"))
    mine, base = _flat(config), _flat(ctrl)
    keys = sorted(set(mine) | set(base))
    env_defects, algo = [], []
    print("=== Gate-2 config diff (PPO launch vs CTRL v2.8.2__jt__20260803-063606__seed42) ===", flush=True)
    for k in keys:
        mv, bv = mine.get(k, "<absent>"), base.get(k, "<absent>")
        if mv == bv:
            continue
        env = _is_env(k)
        tag = "ENVIRONMENTAL" if env else "algorithmic"
        (env_defects if env else algo).append(k)
        print(f"  [{tag:13s}] {k}: CTRL={bv!r} -> PPO={mv!r}", flush=True)
    print(f"--- {len(env_defects)} environmental diff(s), {len(algo)} algorithmic diff(s) ---", flush=True)

    e = config["env"]
    system_obj_dim = {"double_integrator": 19, "quadrotor_3d": 34}
    print("=== Gate-2 environment checklist ===", flush=True)
    print(f"  system={config['run']['system']}  encoder={config.get('obs', {}).get(system, {}).get('encoder', config.get('obs', {}).get('encoder'))}", flush=True)
    print(f"  obs_band_z={e.get('obs_band_z')}  expected_obs_dim={system_obj_dim.get(system)}", flush=True)
    print(f"  band_collision_limit={e.get('band_collision_limit')}  band_terminates={e.get('band_terminates', True)}  "
          f"band_hazard={e.get('band_hazard')}", flush=True)
    print(f"  goal_angrate_radius={e.get('goal_angrate_radius')}  goal_radius={e.get('goal_radius')}  "
          f"goal_speed_radius={e.get('goal_speed_radius')}", flush=True)
    print(f"  dt={e.get('dt')}  max_steps={config['eval']['max_steps']}  oob_limit={e.get('oob_limit')}  "
          f"stuck_window_steps={e.get('stuck_window_steps')}  stuck_radius={e.get('stuck_radius')}", flush=True)
    if system == "quadrotor_3d":
        q = e.get("quadrotor_3d", {})
        print(f"  IC: ic_so3={q.get('ic_so3')}  ic_v_max={q.get('ic_v_max')}  ic_omega_max={q.get('ic_omega_max')}  "
              f"(doomed starts included: scene filters unchanged from base/exp)", flush=True)
        ov = config.get("obstacle", {}).get("per_system", {}).get("quadrotor_3d", {})
        print(f"  obstacle variant={ov.get('variant')} n_min={ov.get('n_min')} n_max={ov.get('n_max')}", flush=True)
    if env_defects:
        print(f"[gate2] ENVIRONMENTAL DEFECTS present: {env_defects}", flush=True)
    return len(env_defects)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["double_integrator", "quadrotor_3d"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--n-iterations", type=int, default=None)
    ap.add_argument("--eval-cadence-iters", type=int, default=None)
    ap.add_argument("--inloop-pool", default=None, help="pool stem override for the in-loop eval")
    ap.add_argument("--reach-floor", type=float, default=0.02,
                    help="1/3-budget in-loop-reach guard floor; set < 0 to disable (full budget).")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    overrides: dict = {"ppo": {}}
    if a.num_envs is not None:
        overrides["ppo"]["num_envs"] = int(a.num_envs)
    if a.n_iterations is not None:
        overrides["ppo"]["n_iterations"] = int(a.n_iterations)
    if a.eval_cadence_iters is not None:
        overrides["ppo"]["eval_cadence_iters"] = int(a.eval_cadence_iters)

    config = load_ppo_config(a.system, quadrotor_env_stage2=(a.system == "quadrotor_3d"), overrides=overrides)
    env_defects = gate2_diff(config, a.system)

    p = config["ppo"]
    mb = int(p.get("minibatch_size", 0) or 0)
    horizon = int(p["horizon"]) if p.get("horizon") else int(config["eval"]["max_steps"])
    mb_desc = (f"minibatch_size={mb} (n_minibatches = ceil(N/{mb}) per iter)" if mb > 0
               else f"{p['n_minibatches']} minibatches")
    print(f"=== PPO budget: {p['n_iterations']} iters * {p['epochs']} epochs * [{mb_desc}]  "
          f"num_envs={p['num_envs']}  horizon={horizon}(=eval.max_steps). Actual interactions/updates/wall "
          f"reported per-iteration and at the end (NOT a step-count symmetry with JT). ===", flush=True)
    if a.dry_run:
        print("DRY RUN: config + gate-2 diff only; no training.", flush=True)
        return
    if env_defects and a.system == "quadrotor_3d":
        raise SystemExit(f"ABORT: environmental defects vs CTRL: fix before launching quadrotor training.")

    pool_stem = a.inloop_pool or INLOOP_POOLS[a.system]
    inloop_pool_path = resolve_pool_or_raise(pool_stem)
    print(f"[ppo] in-loop pool: {inloop_pool_path}", flush=True)

    run_ppo_training(config, seed=a.seed, inloop_pool_path=inloop_pool_path,
                     one_third_reach_floor=a.reach_floor)


if __name__ == "__main__":
    main()
