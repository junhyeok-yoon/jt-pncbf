"""v2.7.4 theory T2 — realized closed-loop speed over training (leg (i) of conj:inflation).

Registered prediction (BEFORE data): mean |v_xy| over the in-loop eval rises monotonically-in-trend across
training steps. Falsified if it is flat or falls.

Method: for each saved cadence checkpoint of the M5 JT run, roll the deployed loop (policy + filter) on the
FROZEN d2r in-loop pool through the existing eval path (rollout_eval — no new rollout written) and report mean
and p90 of |v_xy|, mean |v_z|, and mean realized path length / straight-line start-goal distance. Statistics
are taken over ACTIVE steps only (states are frozen after an episode terminates, so including the tail would
weight the frozen terminal velocity). Eval-only; no checkpoint written.
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

ckpts = sorted(RUN.glob("checkpoints/step_*.pt"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
scenes = load_pool(POOL).scenes
rows = []
for cp in ckpts:
    step = int(re.search(r"step_(\d+)", cp.name).group(1))
    fw, cfg, _ = load_framework_from_checkpoint(cp)
    system = fw.system
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    dtype, device = _tensor_options(system, fw)
    bs = make_batched_scene(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bs)
    res = rollout_eval(system, fw.policy, _filter_adapter(fw), bs, x0, max_steps=max_steps, dt=dt, config=cfg)
    S = res.states.detach().to(torch.float64)                        # [T+1, N, 13]
    ev = resolve_outcome(step_outcomes(res.states, bs, system, cfg)).event_step.numpy()
    Sn = S.numpy(); T1, N = Sn.shape[0], Sn.shape[1]
    ev = np.clip(np.where(ev < 0, T1 - 1, ev), 1, T1 - 1)

    vxy_all, vz_all, ratios = [], [], []
    start = Sn[0, :, 0:3]; goal = bs.goal.detach().to(torch.float64).numpy()
    for i in range(N):
        k = int(ev[i])
        v = Sn[:k, i, 7:10]                                          # ACTIVE steps only
        vxy_all.append(np.hypot(v[:, 0], v[:, 1])); vz_all.append(np.abs(v[:, 2]))
        p = Sn[:k + 1, i, 0:3]
        path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        sl = float(np.linalg.norm(goal[i] - start[i]))
        ratios.append(path / max(sl, 1e-9))
    vxy_c = np.concatenate(vxy_all); vz_c = np.concatenate(vz_all)
    rows.append({"step": step, "n_episodes": N,
                 "mean_vxy": round(float(vxy_c.mean()), 5), "p90_vxy": round(float(np.percentile(vxy_c, 90)), 5),
                 "mean_vz": round(float(vz_c.mean()), 5),
                 "mean_pathlen_over_straightline": round(float(np.mean(ratios)), 5)})
    print(f"step {step:6d}  mean|v_xy|={rows[-1]['mean_vxy']:.4f}  p90={rows[-1]['p90_vxy']:.4f}  "
          f"mean|v_z|={rows[-1]['mean_vz']:.4f}  path/SL={rows[-1]['mean_pathlen_over_straightline']:.4f}", flush=True)

steps = np.array([r["step"] for r in rows], float)
mv = np.array([r["mean_vxy"] for r in rows], float)
slope = float(np.polyfit(steps, mv, 1)[0]) if len(rows) > 1 else float("nan")
out = {"pool": POOL.name, "n_checkpoints": len(rows), "run": str(RUN),
       "per_checkpoint": rows,
       "trend": {"mean_vxy_first": mv[0], "mean_vxy_last": mv[-1], "delta": round(float(mv[-1] - mv[0]), 5),
                 "ols_slope_per_step": slope, "ols_slope_per_10k_steps": round(slope * 1e4, 5),
                 "spearman_like_frac_increasing": round(float((np.diff(mv) > 0).mean()), 4)}}
(OUT / "t2_speed_over_training.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["trend"], indent=2))
