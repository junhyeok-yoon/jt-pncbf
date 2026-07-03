"""HardNet projection-infeasibility attribution (read-only).

For a checkpoint evaluated on a frozen pool, roll episodes exactly as run_full's deployed path
(policy + HardNet box-aware filter), and for every per-step infeasibility event classify:
  - CAUSE: singular (||L_g h|| < 5e-4) vs empty box-halfspace intersection (reusing the exact
    src/common/filter_hardnet.py logic), and
  - STATE: recoverable vs doomed via the losses.py unavoidable-collision predicate
    (surf_clr < v_close^2/(2 u_max) + delta_feas, nearest active obstacle).

Aggregated over ACTIVE filtered steps (before the first physical event), matching the eval
infeasibility definition. Emits a JSON + markdown table. Nothing is trained or modified.

Usage:
  python -m scripts.analysis.infeas_attribution --ckpt PATH [--pool PATH] [--max-scenes N]
         [--tag NAME] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD,
    _base_alpha,
    _cbf_terms,
    _empty_halfspace_box,
)
from src.common.observation import top_k_obstacles
from src.common.outcomes import step_outcomes
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options, first_physical_event_step
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"


def attribute(ckpt: Path, pool_path: Path, max_scenes: int | None):
    framework, config, checkpoint = _load_framework(ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])
    delta_feas = float(config["scene_train"]["init_feasibility_margin"])

    scenes = load_pool(pool_path).scenes
    if max_scenes is not None:
        scenes = scenes[:max_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)

    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework),
                           batched, x0, max_steps, dt, config=config)
    deployed_infeasible = res.infeasible.bool()                       # [T,B] singular|empty

    # active steps: strictly before the first physical event (goal/collision/oob)
    masks = step_outcomes(res.states, batched, system, config)
    event = first_physical_event_step(masks)                         # [B], -1 if none
    active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
    t_idx = torch.arange(max_steps, device=device).unsqueeze(1)
    active = t_idx < active_n.unsqueeze(0)                           # [T,B]

    hf = framework._filter
    params = hf.params
    h_fn = hf.h_fn
    bounds = system.u_bounds.to(device=device, dtype=dtype)
    eps = 1.0e-9

    sing_l, empty_l, doom_l = [], [], []
    with torch.no_grad():
        for t in range(max_steps):
            x = res.states[t]
            u = res.u_nom[t]
            h, lf, lg = _cbf_terms(system, h_fn, x, batched, u, create_graph=False)
            alpha = _base_alpha(h, params)
            row_upper = -lf - alpha * h
            sing_l.append(torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD)
            empty_l.append(_empty_halfspace_box(lg, row_upper, bounds))
            # recoverable vs doomed (losses.py:197-206), nearest active obstacle
            p = system.position(x)
            v = x[:, 2:4]
            top_rel, top_radii = top_k_obstacles(
                p, batched.obstacle_centers, batched.obstacle_radii, batched.obstacle_active, 1)
            cp = top_rel[:, 0, :]
            rr = top_radii[:, 0]
            dist = torch.linalg.norm(cp, dim=1)
            surf_clr = dist - rr
            v_close = torch.relu(torch.sum(v * cp, dim=1) / dist.clamp_min(eps))
            d_stop = v_close * v_close / (2.0 * u_max)
            doom_l.append(surf_clr < (d_stop + delta_feas))
    sing = torch.stack(sing_l, dim=0)
    empty = torch.stack(empty_l, dim=0)
    doomed = torch.stack(doom_l, dim=0)
    recomputed = sing | empty

    A = active
    n_active = float(A.sum().item())
    def rate(mask):
        return float((mask & A).sum().item()) / n_active if n_active else 0.0

    inf = recomputed & A
    n_inf = float(inf.sum().item())
    def inf_share(mask):
        return float((mask & inf).sum().item()) / n_inf if n_inf else 0.0

    return {
        "checkpoint": str(ckpt),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "pool": str(pool_path),
        "n_scenes": len(scenes),
        "active_steps": int(n_active),
        "deployed_infeasibility_rate": float((deployed_infeasible & A).sum().item()) / n_active if n_active else 0.0,
        "recomputed_infeasibility_rate": rate(recomputed),
        "singular_rate": rate(sing),
        "empty_intersection_rate": rate(empty),
        "recoverable_infeasibility_rate": rate(recomputed & ~doomed),
        "doomed_infeasibility_rate": rate(recomputed & doomed),
        "infeas_share_recoverable": inf_share(~doomed),
        "infeas_share_doomed": inf_share(doomed),
        "infeas_share_singular": inf_share(sing),
        "infeas_share_empty_intersection": inf_share(empty & ~sing),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HardNet infeasibility attribution (read-only).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0/attribution")
    args = ap.parse_args()

    stats = attribute(args.ckpt, args.pool, args.max_scenes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    keys = ["n_scenes", "active_steps", "deployed_infeasibility_rate",
            "recomputed_infeasibility_rate", "singular_rate", "empty_intersection_rate",
            "recoverable_infeasibility_rate", "doomed_infeasibility_rate",
            "infeas_share_recoverable", "infeas_share_doomed",
            "infeas_share_singular", "infeas_share_empty_intersection"]
    md = [f"# infeasibility attribution — {args.tag}", "",
          f"checkpoint: `{stats['checkpoint']}` (step {stats['checkpoint_step']})",
          f"pool: `{stats['pool']}`", "", "| metric | value |", "|---|---:|"]
    for k in keys:
        v = stats[k]
        md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    (args.out / f"{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"OUT_JSON={args.out / f'{args.tag}.json'}")
    for k in keys:
        print(f"  {k} = {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
