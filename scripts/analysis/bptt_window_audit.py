"""BPTT window-event contamination audit (read-only) — v2.4.0 Step 3/4 collapse forensics.

The policy BPTT rollout (losses.py policy_bptt_loss) integrates the task cost for the FULL bptt_T
steps with NO goal-arrival / collision / out-of-arena masking or termination and no terminal
value. This script quantifies how much of the differentiable return is accrued in physically
TERMINAL states, and how that worsens from T=30 to T=60.

Method: reconstruct a representative set of BPTT start states by deterministically rolling the
in-loop pool under pi(ckpt)+HardNet and sampling visited active states (CAVEAT: training's D_pi is
collected with sigma_pi exploration noise; this no-noise proxy slightly under-samples off-policy
states — it is a lower bound on contamination). Then roll T=60 windows no_grad under pi+HardNet
from those start states and, per window, find the first physical event (goal / collision /
out-of-arena) and the discounted-cost share accrued at/after it. T=30 metrics are the first-30-step
slice of the same windows (identical start states).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.outcomes import _collided_exact
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
N_START = 1024


def _events(system, x, scene, config):
    """Per-state physical-terminal flags: goal-arrival, collision, out-of-arena."""
    p = system.position(x)
    goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
    dist = torch.linalg.norm(p - goal, dim=-1)
    speed = system.speed(x)
    goal_reached = (dist <= float(config["env"]["goal_radius"])) & (speed <= float(config["env"]["goal_speed_radius"]))
    collided = _collided_exact(p, scene)
    oob = torch.any(torch.abs(p) > float(config["env"]["oob_limit"]), dim=-1)
    return goal_reached, collided, oob


def audit(ckpt: Path, pool_path: Path, seed_offset: int):
    framework, config, checkpoint = _load_framework(ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    dt = float(config["env"]["dt"])
    gamma_t = float(config["loss"]["policy"]["gamma_T"])
    lambda_v = float(config["loss"]["policy"]["lambda_v"])
    mu_u = float(config["loss"]["policy"]["mu_u"])
    T_full, T_short = 60, 30

    scenes = load_pool(pool_path).scenes
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    filter_fn = _filter_adapter(framework)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, filter_fn, batched, x0,
                           int(config["eval"]["max_steps"]), dt, config=config)
    # collect visited states (all steps, all scenes) and sample N_START (state, scene_idx) pairs
    T, B, _ = res.states.shape
    gen = torch.Generator(device="cpu").manual_seed(1234 + seed_offset)
    flat_t = torch.randint(0, T, (N_START,), generator=gen)
    flat_b = torch.randint(0, B, (N_START,), generator=gen)
    start = res.states[flat_t, flat_b].to(device)                        # [N,4]
    start_scenes = [scenes[int(b)] for b in flat_b]
    sbatch = batch_scenes(start_scenes, device=device, dtype=dtype)

    # roll T_full windows no_grad under pi+HardNet from the start states
    with torch.no_grad():
        x = system.wrap_state(start)
        first_event = torch.full((N_START,), -1, dtype=torch.long, device=device)
        cost_steps = []
        discount = 1.0
        for t in range(T_full):
            u_nom = framework.policy(x, sbatch)
            u_safe, _ = filter_fn(x, u_nom, sbatch)
            x = rk4_step(system, x, u_safe, dt)
            goal_r, coll, oob = _events(system, x, sbatch, config)
            ev = goal_r | coll | oob
            newly = ev & (first_event < 0)
            first_event = torch.where(newly, torch.full_like(first_event, t), first_event)
            p = system.position(x)
            gpos = torch.as_tensor(sbatch.goal, dtype=dtype, device=device)
            d2 = torch.sum((p - gpos) ** 2, dim=1)
            v2 = system.speed(x) ** 2
            u2 = torch.sum(u_safe * u_safe, dim=1)
            cost_steps.append(discount * (d2 + lambda_v * v2 + mu_u * u2))
            discount *= gamma_t
        cost = torch.stack(cost_steps, dim=0)                            # [T_full, N]

    def summarize(T):
        c = cost[:T]
        fe = first_event.clone()
        has = (fe >= 0) & (fe < T)
        total = c.sum(dim=0).clamp_min(1e-12)
        idx = torch.arange(T, device=device).unsqueeze(1)
        post_mask = has.unsqueeze(0) & (idx >= fe.clamp(min=0).unsqueeze(0))
        post_share = (c * post_mask).sum(dim=0) / total
        fe_valid = fe[has].float()
        return {
            "T": T,
            "window_has_event_frac": float(has.to(dtype).mean()),
            "mean_post_event_cost_share": float(post_share.mean()),
            "mean_first_event_step": float(fe_valid.mean()) if fe_valid.numel() else None,
            "median_first_event_step": float(fe_valid.median()) if fe_valid.numel() else None,
        }

    # per-event-type window fractions over the full window
    with torch.no_grad():
        gr_any = torch.zeros(N_START, dtype=torch.bool, device=device)
        co_any = torch.zeros(N_START, dtype=torch.bool, device=device)
        ob_any = torch.zeros(N_START, dtype=torch.bool, device=device)
        x = system.wrap_state(start)
        for t in range(T_full):
            u_nom = framework.policy(x, sbatch)
            u_safe, _ = filter_fn(x, u_nom, sbatch)
            x = rk4_step(system, x, u_safe, dt)
            gr, co, ob = _events(system, x, sbatch, config)
            gr_any |= gr; co_any |= co; ob_any |= ob

    return {
        "checkpoint": str(ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
        "n_start_states": N_START, "gamma_T": gamma_t,
        "goal_arrival_window_frac_T60": float(gr_any.to(dtype).mean()),
        "collision_window_frac_T60": float(co_any.to(dtype).mean()),
        "oob_window_frac_T60": float(ob_any.to(dtype).mean()),
        "T30": summarize(T_short), "T60": summarize(T_full),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="BPTT window-event contamination audit (read-only).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=INLOOP_POOL)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0")
    args = ap.parse_args()
    stats = audit(args.ckpt, args.pool, args.seed_offset)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.bpttwin.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    md = [f"# BPTT window-event audit — {args.tag}", "",
          f"ckpt `{stats['checkpoint']}` step {stats['checkpoint_step']}  N={stats['n_start_states']}  gamma_T={stats['gamma_T']}", "",
          f"window contains (T=60): goal {stats['goal_arrival_window_frac_T60']:.3f} | "
          f"collision {stats['collision_window_frac_T60']:.3f} | oob {stats['oob_window_frac_T60']:.3f}", "",
          "| horizon | window_has_event | post_event_cost_share | mean_first_event_step |",
          "|---|---:|---:|---:|"]
    for k in ("T30", "T60"):
        s = stats[k]
        md.append(f"| {s['T']} | {s['window_has_event_frac']:.3f} | {s['mean_post_event_cost_share']:.3f} | "
                  f"{s['mean_first_event_step'] if s['mean_first_event_step'] is None else round(s['mean_first_event_step'],1)} |")
    (args.out / f"{args.tag}.bpttwin.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OUT={args.out / f'{args.tag}.bpttwin.json'}")
    print(f"  window (T60): goal {stats['goal_arrival_window_frac_T60']:.3f} coll {stats['collision_window_frac_T60']:.3f} oob {stats['oob_window_frac_T60']:.3f}")
    for k in ("T30", "T60"):
        print(f"  {k}: {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
