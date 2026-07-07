"""v2.4.1 Stage C (C1-add) — saturation decomposition on the frozen n2000 pool.

For a checkpoint, replicate the eval's per-episode saturation_rate (evaluate.py: a step is saturated
iff any u_safe component is within 1e-3 of a box bound, averaged over active_steps up to the physical
event, then averaged over episodes) and split it into:
    sat_task   = saturated steps with the deficit INACTIVE (||delta_u||<=1e-6 or singular / filter idle)
    sat_safety = saturated AND deficit-active
where delta_u = u_cbf_raw (raw box-free CBF projection) - u_safe (deployed box-aware output).
sat_safety is the deficit channel's ceiling on saturation_rate. Also reports mean episode speed
(mean system.speed over active steps, per episode then averaged) as the motion control (C3-add).
Read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch

from src.common.filter_hardnet import HardNetFilter
from src.common.outcomes import step_outcomes
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool
from src.eval.evaluate import (
    _filter_adapter, _slice_rollout, _tensor_options, active_action_steps,
    first_physical_event_step,
)
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes as make_batched_scene, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--gsr", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.1")
    args = ap.parse_args()

    framework, config, _ = _load_framework(args.ckpt)
    cfg = copy.deepcopy(dict(config))
    cfg["env"] = {**cfg["env"], "goal_speed_radius": float(args.gsr)}
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    hn = HardNetFilter(system, make_h_fn(framework.value_net, system), cfg)
    bounds = system.u_bounds.to(device=device, dtype=dtype)
    max_steps = int(cfg["eval"]["max_steps"])
    dt = float(cfg["env"]["dt"])
    scenes = load_pool(POOL).scenes

    per_ep = []   # (sat_frac, sat_task_frac, sat_safety_frac, mean_speed)
    for start in range(0, len(scenes), args.batch):
        bs = scenes[start:start + args.batch]
        bscene = make_batched_scene(bs, device=device, dtype=dtype)
        x0 = initial_states_from_batch(bscene)
        with torch.no_grad():
            filt = rollout_eval(system, framework.policy, _filter_adapter(framework),
                                bscene, x0, max_steps=max_steps, dt=dt, config=cfg)
            T = filt.u_safe.shape[0]
            dact = []
            for t in range(T):
                us, _, ucbf, sing = hn(filt.states[t], bscene, filt.u_nom[t], return_deficit_aux=True)
                du = torch.linalg.norm(ucbf - us, dim=1)
                dact.append((~sing) & (du > 1.0e-6))
            dact = torch.stack(dact)                                   # [T, B]
            ld = torch.abs(filt.u_safe - bounds[:, 0]); ud = torch.abs(filt.u_safe - bounds[:, 1])
            sat = torch.any(torch.minimum(ld, ud) <= 1.0e-3, dim=-1)   # [T, B]
            speed = system.speed(filt.states)                          # [T+1, B]
        for i, scene in enumerate(bs):
            fi = _slice_rollout(filt, i)
            masks = step_outcomes(fi.states, scene, system, cfg)
            pes = int(first_physical_event_step(masks)[0].item())
            a = active_action_steps(pes, fi.u_safe.shape[0])
            if a <= 0:
                per_ep.append((0.0, 0.0, 0.0, float(speed[0, i])))
                continue
            s = sat[:a, i]; d = dact[:a, i]
            per_ep.append((float(s.double().mean()),
                           float((s & ~d).double().mean()),
                           float((s & d).double().mean()),
                           float(speed[:a, i].mean())))

    res = {"ckpt": str(args.ckpt), "gsr": args.gsr, "n_episodes": len(per_ep),
           "saturation_rate": statistics.mean(p[0] for p in per_ep),
           "sat_task": statistics.mean(p[1] for p in per_ep),
           "sat_safety": statistics.mean(p[2] for p in per_ep),
           "mean_episode_speed": statistics.mean(p[3] for p in per_ep)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.satdecomp.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.tag}] n={res['n_episodes']} saturation_rate={res['saturation_rate']:.4f} "
          f"= sat_task {res['sat_task']:.4f} + sat_safety {res['sat_safety']:.4f} "
          f"(deficit ceiling); mean_speed={res['mean_episode_speed']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
