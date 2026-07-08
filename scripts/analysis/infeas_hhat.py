"""v2.4.2 decline forensics B addon — deployed h-hat distribution at infeasible steps (read-only).

For a checkpoint on a frozen pool, roll the deployed path, classify each active step as
singular / empty-intersection (reusing filter_hardnet logic), and report quantiles of the deployed
h-hat at infeasible steps overall and split by cause. Reading: empty-intersection h-hat concentrated
near 0 = band-consistent (H1); spread wide in h-hat = large eps0 (H2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD, _base_alpha, _cbf_terms, _empty_halfspace_box,
)
from src.common.outcomes import step_outcomes
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options, first_physical_event_step
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"


def _q(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "frac_within_0.1_of_0": float((np.abs(a) <= 0.1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()

    framework, config, checkpoint = _load_framework(a.ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    scenes = load_pool(a.pool).scenes
    if a.max_scenes:
        scenes = scenes[: a.max_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework),
                           batched, x0, max_steps, dt, config=config)
    masks = step_outcomes(res.states, batched, system, config)
    event = first_physical_event_step(masks)
    active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
    t_idx = torch.arange(max_steps, device=device).unsqueeze(1)
    active = t_idx < active_n.unsqueeze(0)

    hf = framework._filter
    params, h_fn = hf.params, hf.h_fn
    bounds = system.u_bounds.to(device=device, dtype=dtype)

    hh, sing_m, empty_m, act_m = [], [], [], []
    with torch.no_grad():
        for t in range(max_steps):
            x = res.states[t]
            u = res.u_nom[t]
            h, lf, lg = _cbf_terms(system, h_fn, x, batched, u, create_graph=False)
            alpha = _base_alpha(h, params)
            row_upper = -lf - alpha * h
            hh.append(h_fn(x, batched).reshape(-1))
            sing_m.append(torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD)
            empty_m.append(_empty_halfspace_box(lg, row_upper, bounds))
            act_m.append(active[t])
    hh = torch.stack(hh).cpu().numpy()
    sing = torch.stack(sing_m).cpu().numpy()
    empty = torch.stack(empty_m).cpu().numpy()
    act = torch.stack(act_m).cpu().numpy()
    infeas = (sing | empty) & act
    empty_only = empty & ~sing & act
    sing_a = sing & act

    out = {
        "checkpoint": str(a.ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
        "pool": str(a.pool), "n_scenes": len(scenes),
        "hhat_at_infeasible": _q(hh[infeas]),
        "hhat_at_empty_intersection": _q(hh[empty_only]),
        "hhat_at_singular": _q(hh[sing_a]),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
