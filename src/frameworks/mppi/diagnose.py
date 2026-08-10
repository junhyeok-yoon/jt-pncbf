"""v2.8.4 MPPI baseline — screen diagnostics. MEASUREMENT ONLY, no selection, no threshold.

Two questions the 16-cell table raises but cannot answer on its own:

  1. WHY is reach 0? The deployed reach terminal is a conjunction
     (dist <= 0.15 AND speed <= 0.30 AND ||omega|| <= 0.30). This records, per episode, the closest
     approach and which legs held there, so the failing leg is named rather than guessed.

  2. WHY are lambda and C_crash nearly inert? The exponential weighting is a soft argmin only when
     lambda is comparable to the SPREAD of the sample costs S. This records, per control step, the
     spread (max - min, std) of S and the effective sample size  ESS = 1 / sum_n w_n^2  of the weights.
     ESS ~ 1 means the softmax has collapsed to a hard argmin, in which case lambda changes nothing and
     C_crash changes nothing unless it flips the argmin.

Writes `data/runs/v2.8.4/mppi_screen/diagnose__<label>.json`.

Run:  python -m src.frameworks.mppi.diagnose --n-samples 1024 --horizon 40 --lam 0.2 --c-crash 1e3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.frameworks.mppi.cost import collision_mask
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
    _resolve_device,
    _resolve_dtype,
)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="MPPI screen diagnostics.")
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--c-crash", type=float, default=1e3)
    parser.add_argument("--sigma", type=float, default=None,
                        help="override mppi.sampling.sigma (MEASUREMENT ONLY — sigma is not in the "
                             "declared screening grid and no cell is selected from this probe)")
    parser.add_argument("--n-scenes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    device, dtype = _resolve_device(args.device), _resolve_dtype(args.dtype)
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    system, framework, params, cost = build_framework(
        config, mppi_config, n_samples=args.n_samples, horizon=args.horizon, lam=args.lam,
        c_crash=args.c_crash, sigma=args.sigma, device=device, dtype=dtype,
    )
    controller = framework.controller

    pool = load_pool(REPO / mppi_config["screen"]["pool"])
    scenes = pool.scenes[: args.n_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x = system.wrap_state(initial_states_from_batch(batched)).to(dtype)
    goal = batched.goal.to(dtype)
    dt = float(config["env"]["dt"])

    # instrument the weighting: recompute S and the weights exactly as `act` does, then read the spread
    spread, ess_all, cost_std = [], [], []
    original_rollout = controller._rollout_cost

    def instrumented(x_in, sampled, goal_in, centers, radii, active):
        sample_cost, all_dead = original_rollout(x_in, sampled, goal_in, centers, radii, active)
        shifted = sample_cost - sample_cost.min(dim=1, keepdim=True).values
        # mirror the controller's lambda rule exactly, so the reported ESS is the ESS actually used
        lam_eff = (
            params.lam if params.lam_mode == "absolute"
            else torch.clamp(params.lam * sample_cost.std(dim=1, keepdim=True), min=params.lam_eps_abs)
        )
        weight = torch.exp(-shifted / lam_eff)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        spread.append((sample_cost.max(dim=1).values - sample_cost.min(dim=1).values).cpu().numpy())
        cost_std.append(sample_cost.std(dim=1).cpu().numpy())
        ess_all.append((1.0 / weight.square().sum(dim=1)).cpu().numpy())
        return sample_cost, all_dead

    controller._rollout_cost = instrumented

    controller.reset(x.shape[0])
    distance, speed, angrate, dead = [], [], [], torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    for _ in range(args.steps):
        u = controller.act(x, batched)
        x = rk4_step(system, x, u, dt)
        collided = collision_mask(
            system.position(x).unsqueeze(1), batched.obstacle_centers, batched.obstacle_radii,
            batched.obstacle_active, cost,
        ).squeeze(1)
        dead = dead | collided
        distance.append(torch.linalg.norm(system.position(x) - goal, dim=-1).cpu().numpy())
        speed.append(system.speed(x).cpu().numpy())
        angrate.append(system.angular_rate(x).cpu().numpy())

    distance = np.stack(distance)          # [T, B]
    speed = np.stack(speed)
    angrate = np.stack(angrate)
    closest = distance.argmin(axis=0)
    rows = np.arange(distance.shape[1])
    d_min = distance[closest, rows]
    v_at = speed[closest, rows]
    w_at = angrate[closest, rows]

    r_g = float(config["env"]["goal_radius"])
    v_g = float(config["env"]["goal_speed_radius"])
    w_g = float(config["env"]["goal_angrate_radius"])
    leg_d = d_min <= r_g
    leg_v = speed <= v_g
    leg_w = angrate <= w_g
    leg_pos = distance <= r_g

    label = args.label or (
        f"N{args.n_samples}_H{args.horizon}_lam{args.lam:g}_C{args.c_crash:g}_sig{params.sigma:g}"
    )
    out = {
        "what": "v2.8.4 MPPI screen diagnostics. MEASUREMENT ONLY; no selection, no threshold.",
        "cell": {"N": params.n_samples, "H": params.horizon, "lambda": params.lam,
                 "lambda_mode": params.lam_mode, "space": params.space, "noise": params.noise,
                 "C_crash": cost.c_crash, "sigma": params.sigma, "seed": params.seed},
        "population": {"n_scenes": len(scenes), "steps": args.steps,
                       "pool": mppi_config["screen"]["pool"], "device": str(device), "dtype": str(dtype)},
        "terminal_radii": {"goal_radius": r_g, "goal_speed_radius": v_g, "goal_angrate_radius": w_g},
        "why_reach_is_low": {
            "closest_approach_m": {"mean": float(d_min.mean()), "median": float(np.median(d_min)),
                                   "min": float(d_min.min()), "p90": float(np.percentile(d_min, 90))},
            "episodes_entering_the_goal_ball": int(leg_d.sum()),
            "episode_frac_entering_the_goal_ball": float(leg_d.mean()),
            "at_closest_approach": {
                "speed_mean": float(v_at.mean()), "speed_median": float(np.median(v_at)),
                "angrate_mean": float(w_at.mean()), "angrate_median": float(np.median(w_at)),
                "frac_speed_leg_held": float((v_at <= v_g).mean()),
                "frac_angrate_leg_held": float((w_at <= w_g).mean()),
                "frac_all_three_legs_held": float((leg_d & (v_at <= v_g) & (w_at <= w_g)).mean()),
            },
            "step_level_leg_rates_over_all_steps": {
                "position_leg": float(leg_pos.mean()), "speed_leg": float(leg_v.mean()),
                "angrate_leg": float(leg_w.mean()),
                "all_three": float((leg_pos & leg_v & leg_w).mean()),
            },
            "speed_over_all_steps": {"mean": float(speed.mean()), "median": float(np.median(speed))},
            "angrate_over_all_steps": {"mean": float(angrate.mean()),
                                       "median": float(np.median(angrate))},
        },
        "why_lambda_and_C_crash_are_nearly_inert": {
            "note": "ESS = 1 / sum_n w_n^2 over the N sample weights. ESS ~ 1 means exp(-S/lambda) has "
                    "collapsed to a hard argmin: lambda then changes nothing, and C_crash changes nothing "
                    "unless it moves the argmin. The comparison to make is lambda vs the cost spread.",
            "lambda": params.lam,
            "cost_spread_max_minus_min": {"mean": float(np.mean(spread)),
                                          "median": float(np.median(spread)),
                                          "min": float(np.min(spread))},
            "cost_std_across_samples": {"mean": float(np.mean(cost_std)),
                                        "median": float(np.median(cost_std))},
            "lambda_over_cost_std_median": float(params.lam / max(np.median(cost_std), 1e-12)),
            "ESS": {"mean": float(np.mean(ess_all)), "median": float(np.median(ess_all)),
                    "max": float(np.max(ess_all)),
                    "frac_steps_ESS_below_1p01": float(np.mean(np.array(ess_all) < 1.01))},
            "N": params.n_samples,
        },
        "collision_frac_within_the_window": float(dead.float().mean().item()),
        "degenerate_steps_per_episode": {
            "mean": float(controller.degenerate_steps.double().mean().item()),
            "max": int(controller.degenerate_steps.max().item()),
        },
    }
    path = REPO / f"data/runs/v2.8.4/mppi_screen/diagnose__{label}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
