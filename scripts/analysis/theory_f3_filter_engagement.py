"""v2.7.4 theory F3 — filter engagement over training (leg (iii) of conj:inflation, previously unmeasured).

Registered prediction (BEFORE data): if leg (iii) holds, the intervention rate and the mean projection
magnitude rise with training step. Falsified if either is flat or falls.

Method: the same 34 saved cadence checkpoints of the M5 JT run rolled on the FROZEN d2r in-loop pool (n=500)
through the existing eval path (rollout_eval). Per checkpoint, over ACTIVE steps only (states freeze after an
episode terminates, so the tail would otherwise be counted):
  - intervention rate  = mean of rollout's intervention_mask (||u_safe - u_nom|| > 1e-3, the codebase tolerance)
  - mean projection    = mean ||u_safe - u_nom|| over INTERVENING steps only
  - empty-branch rate  = mean of the rollout's per-step `empty` flag
Eval-only; no training, no checkpoint written.
"""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np
import torch

from src.common.outcomes import step_outcomes, resolve_outcome
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import initial_states_from_batch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

RUN = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42")
POOL = EVAL_POOLS_DIR / "eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)

scenes = load_pool(POOL).scenes
ckpts = sorted(RUN.glob("checkpoints/step_*.pt"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
rows = []
for cp in ckpts:
    step = int(re.search(r"step_(\d+)", cp.name).group(1))
    fw, cfg, _ = load_framework_from_checkpoint(cp)
    system = fw.system
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    dtype, device = _tensor_options(system, fw)
    bs = make_batched_scene(scenes, device=device, dtype=dtype)
    res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, initial_states_from_batch(bs),
                       max_steps=max_steps, dt=dt, config=cfg)
    ev = resolve_outcome(step_outcomes(res.states, bs, system, cfg)).event_step.numpy()
    T = int(res.u_safe.shape[0]); N = int(res.u_safe.shape[1])
    ev = np.clip(np.where(ev < 0, T, ev), 1, T)
    # active-step mask [T, N]
    idx = np.arange(T)[:, None]
    active = idx < ev[None, :]
    proj = torch.linalg.norm(res.u_safe - res.u_nom, dim=-1).detach().to(torch.float64).numpy()   # [T,N]
    interv = res.intervention_mask.detach().cpu().numpy().astype(bool)                            # [T,N]
    empty = (res.empty.detach().cpu().numpy().astype(bool) if res.empty is not None
             else np.zeros_like(interv))
    n_active = int(active.sum())
    interv_a = interv & active
    rows.append({
        "step": step, "n_episodes": N, "n_active_steps": n_active,
        "intervention_rate": round(float(interv_a.sum() / max(n_active, 1)), 6),
        "mean_projection_on_intervening": round(float(proj[interv_a].mean()) if interv_a.any() else 0.0, 6),
        "mean_projection_all_active": round(float(proj[active].mean()), 6),
        "empty_branch_rate": round(float((empty & active).sum() / max(n_active, 1)), 6),
    })
    r = rows[-1]
    print(f"step {step:6d} interv_rate={r['intervention_rate']:.4f} "
          f"mean_proj_interv={r['mean_projection_on_intervening']:.4f} "
          f"empty={r['empty_branch_rate']:.5f} n_active={n_active}", flush=True)

st = np.array([r["step"] for r in rows], float)
def trend(key):
    y = np.array([r[key] for r in rows], float)
    sl = float(np.polyfit(st, y, 1)[0])
    return {"first": y[0], "last": y[-1], "delta": round(float(y[-1] - y[0]), 6),
            "ols_slope_per_10k_steps": round(sl * 1e4, 6),
            "frac_increasing": round(float((np.diff(y) > 0).mean()), 4)}

out = {"pool": POOL.name, "run": str(RUN), "n_checkpoints": len(rows),
       "intervention_tolerance": "||u_safe-u_nom|| > 1e-3 (codebase rollout tolerance)",
       "per_checkpoint": rows,
       "trend": {k: trend(k) for k in ("intervention_rate", "mean_projection_on_intervening",
                                       "mean_projection_all_active", "empty_branch_rate")}}
(OUT / "f3_filter_engagement.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["trend"], indent=2))
