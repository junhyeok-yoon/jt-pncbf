"""v2.4.2 forensics B band test — deployed h-hat distribution at INFEASIBLE steps (read-only).

Reproduces the deployed HardNet rollout (same as infeas_attribution) and collects deployed h-hat at
every active INFEASIBLE step (singular OR empty-intersection). Reports quantiles + a coarse
histogram + the fraction with |h-hat|<0.05. Reading: h-hat concentrated near 0 => band-consistent
(H1); h-hat spread wide => large eps0 (H2).
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
POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"


def _band(ckpt, pool, max_scenes):
    fw, cfg, ck = _load_framework(ckpt)
    system = fw.system
    dtype, device = _tensor_options(system, fw)
    max_steps = int(cfg["eval"]["max_steps"])
    dt = float(cfg["env"]["dt"])
    scenes = load_pool(pool).scenes
    if max_scenes:
        scenes = scenes[:max_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, fw.policy, _filter_adapter(fw), batched, x0, max_steps, dt, config=cfg)
    masks = step_outcomes(res.states, batched, system, cfg)
    event = first_physical_event_step(masks)
    active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
    t_idx = torch.arange(max_steps, device=device).unsqueeze(1)
    active = t_idx < active_n.unsqueeze(0)
    hf = fw._filter
    params, h_fn = hf.params, hf.h_fn
    bounds = system.u_bounds.to(device=device, dtype=dtype)
    hh = []
    with torch.no_grad():
        for t in range(max_steps):
            x, u = res.states[t], res.u_nom[t]
            h, lf, lg = _cbf_terms(system, h_fn, x, batched, u, create_graph=False)
            sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            alpha = _base_alpha(h, params)
            row = -lf - alpha * h
            empty = _empty_halfspace_box(lg, row, bounds)
            inf = (sing | empty) & active[t]
            if inf.any():
                hh.append(h[inf].detach().cpu())
    hh = torch.cat(hh).numpy() if hh else np.array([])
    qs = {str(q): float(np.percentile(hh, q)) for q in (5, 10, 25, 50, 75, 90, 95)} if hh.size else {}
    edges = [-1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0]
    hist, _ = np.histogram(hh, bins=edges) if hh.size else (np.zeros(7, int), None)
    return {
        "checkpoint_step": int(ck.get("step", -1)),
        "n_infeasible_steps": int(hh.size),
        "hhat_mean": float(hh.mean()) if hh.size else None,
        "hhat_quantiles": qs,
        "hist_bins": ["[-1,-.5)", "[-.5,-.2)", "[-.2,-.05)", "[-.05,.05)", "[.05,.2)", "[.2,.5)", "[.5,1]"],
        "hist_counts": [int(c) for c in hist],
        "frac_abs_h_lt_0.05": float((np.abs(hh) < 0.05).mean()) if hh.size else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="h-hat at infeasible steps (read-only).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()
    stats = _band(a.ckpt, a.pool, a.max_scenes)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag}.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
