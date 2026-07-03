"""Near-goal stuck audit (read-only) — v2.4.0 Step 4 collapse forensics.

For a checkpoint on a pool, roll the deployed path and, for every STUCK episode, characterize the
TERMINAL state to separate final-approach/classification failures from true policy stalls:
  terminal ||p-g||, terminal ||v||, (||p-g|| - reach_radius), goal-to-nearest-obstacle surface
  clearance, V_S(goal_pos, v=0 | scene) sign+value, filter-active fraction over the final 30
  steps, terminal-phase mean|u| and per-axis sign-flip rate (dead parking vs dithering).
Buckets (priority order): GOAL_BLOCKED (V_S(goal,v=0)>0 or terminal filter-active>0.5),
NEAR_GOAL_PARKED (terminal ||p-g|| in [reach_radius, reach_radius+0.3]), DITHER (sign-flip>0.4),
FAR_FIELD_STALL (else). Emits JSON + markdown; run per checkpoint and compare externally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.outcomes import resolve_outcome, step_outcomes
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
INLOOP_POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"
FINAL_N = 30          # terminal-phase window (steps)
FILTER_TOL = 1.0e-3
SIGNFLIP_DITHER = 0.40


def audit(ckpt: Path, pool_path: Path):
    framework, config, checkpoint = _load_framework(ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    reach_radius = float(config["env"]["goal_radius"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])

    scenes = load_pool(pool_path).scenes
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework),
                           batched, x0, max_steps, dt, config=config)
    masks = step_outcomes(res.states, batched, system, config)
    resolved = resolve_outcome(masks)
    outcomes, event_step = resolved.outcome, resolved.event_step
    du = torch.linalg.norm(res.u_safe - res.u_nom, dim=-1)          # [T,B] filter modification

    # V_S(goal_pos, v=0 | scene) for every scene, batched
    goal = batched.goal                                            # [B,2]
    goal_state = torch.cat([goal, torch.zeros_like(goal)], dim=1)  # [B,4]
    with torch.no_grad():
        v_goal = framework.value_net.deployed_h(system.observation(goal_state, batched)).reshape(-1)

    centers, radii, active = batched.obstacle_centers, batched.obstacle_radii, batched.obstacle_active
    eps = 1.0e-9
    details, buckets = [], {"GOAL_BLOCKED": 0, "NEAR_GOAL_PARKED": 0, "DITHER": 0, "FAR_FIELD_STALL": 0}
    for b, o in enumerate(outcomes):
        if o != "stuck":
            continue
        te = max(1, min(int(event_step[b].item()), res.states.shape[0] - 1))
        xT = res.states[te, b]
        p, v = xT[:2], xT[2:4]
        g = goal[b]
        dist_goal = float(torch.linalg.norm(p - g))
        vnorm = float(torch.linalg.norm(v))
        # goal-to-nearest-obstacle surface clearance
        act = active[b]
        gc = torch.linalg.norm(g.unsqueeze(0) - centers[b][act], dim=1) - radii[b][act]
        goal_clear = float(gc.min())
        vg = float(v_goal[b])
        lo = max(0, te - FINAL_N)
        filt_active = float((du[lo:te, b] > FILTER_TOL).to(dtype).mean()) if te > lo else 0.0
        u_tail = res.u_safe[lo:te, b]                              # [<=30, 2]
        mean_abs_u = float(u_tail.abs().mean())
        if u_tail.shape[0] > 1:
            sign = torch.sign(u_tail)
            flips = (sign[1:] != sign[:-1]).to(dtype).mean()
            signflip = float(flips)
        else:
            signflip = 0.0
        near_goal = dist_goal <= reach_radius + 0.3
        blocked = (vg > 0.0) or (filt_active > 0.5)
        if blocked:
            bucket = "GOAL_BLOCKED"
        elif signflip > SIGNFLIP_DITHER:
            bucket = "DITHER"
        elif near_goal:
            bucket = "NEAR_GOAL_PARKED"
        else:
            bucket = "FAR_FIELD_STALL"
        buckets[bucket] += 1
        details.append({
            "scene": b, "terminal_dist_goal": dist_goal, "dist_minus_reach": dist_goal - reach_radius,
            "terminal_speed": vnorm, "goal_obstacle_clearance": goal_clear,
            "V_S_goal_v0": vg, "V_S_goal_positive": bool(vg > 0.0),
            "terminal_filter_active_frac": filt_active, "terminal_mean_abs_u": mean_abs_u,
            "terminal_signflip_rate": signflip, "near_goal": bool(near_goal), "bucket": bucket,
        })

    n = len(details)
    def frac(key):
        return float(sum(1 for d in details if d[key]) / n) if n else 0.0
    def mean(key):
        return float(sum(d[key] for d in details) / n) if n else 0.0
    stats = {
        "checkpoint": str(ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
        "pool": str(pool_path), "n_scenes": len(scenes), "n_stuck": n,
        "stuck_rate": n / len(scenes) if scenes else 0.0,
        "near_goal_frac_of_stuck": frac("near_goal"),
        "V_S_goal_positive_frac_of_stuck": frac("V_S_goal_positive"),
        "mean_terminal_dist_goal": mean("terminal_dist_goal"),
        "mean_terminal_speed": mean("terminal_speed"),
        "mean_terminal_filter_active": mean("terminal_filter_active_frac"),
        "mean_terminal_signflip": mean("terminal_signflip_rate"),
        "buckets": buckets,
        "bucket_frac": {k: (v / n if n else 0.0) for k, v in buckets.items()},
    }
    return stats, details


def main() -> int:
    ap = argparse.ArgumentParser(description="Near-goal stuck audit (read-only).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=INLOOP_POOL)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0")
    args = ap.parse_args()
    stats, details = audit(args.ckpt, args.pool)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.stuckterm.json").write_text(
        json.dumps({"stats": stats, "details": details}, indent=2) + "\n", encoding="utf-8")
    keys = ["n_scenes", "n_stuck", "stuck_rate", "near_goal_frac_of_stuck",
            "V_S_goal_positive_frac_of_stuck", "mean_terminal_dist_goal", "mean_terminal_speed",
            "mean_terminal_filter_active", "mean_terminal_signflip"]
    md = [f"# stuck terminal audit — {args.tag}", "",
          f"ckpt `{stats['checkpoint']}` step {stats['checkpoint_step']}  pool `{args.pool.name}`", "",
          "| metric | value |", "|---|---:|"]
    for k in keys:
        v = stats[k]
        md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md += ["", "| bucket | count | frac |", "|---|---:|---:|"]
    for k, v in stats["buckets"].items():
        md.append(f"| {k} | {v} | {stats['bucket_frac'][k]:.3f} |")
    (args.out / f"{args.tag}.stuckterm.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OUT={args.out / f'{args.tag}.stuckterm.json'}")
    for k in keys:
        print(f"  {k} = {stats[k]}")
    print("  buckets =", stats["buckets"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
