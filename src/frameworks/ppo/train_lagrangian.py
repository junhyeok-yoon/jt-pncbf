"""PPO-LAGRANGIAN: the shipped PPO with its fixed collision penalty replaced by a learned multiplier.

A VARIANT MODULE BESIDE `train.py`, NOT AN EDIT TO IT. **NOT ONE LINE OF ANY EXISTING FILE IS
CHANGED.** Every function this file needs is imported from `train.py`; the cost signal needs the
per-lane terminal index and outcomes, which `compute_rewards_and_gae` computes internally and does
not return, so `cost_buffers` below RECOMPUTES them with the same expressions rather than widening
that function's contract. The recomputation is a second `step_outcomes`/`resolve_outcome` pass over
the same states -- cheap beside the rollout, and it keeps the shipped PPO byte-for-byte reproducible.
Nothing outside `src/frameworks/ppo/` imports this package (verified by grep over jt_pncbf, oc_pncbf,
common, envs and eval), so stop condition 2 cannot be reached from here.

THE SINGLE AXIS.
  shipped:      terminal reward map carries collision = -100 (the registered run's own config value;
                `train.py`'s in-code default is -5, and the registered run overrides it).
  lagrangian:   collision REMOVED from the terminal reward map (set to 0.0), and instead
                  cost c_t = 1 at the collision terminal transition, 0 everywhere else
                  a cost critic V_c with the SAME architecture (`nets.PPOValue`) and the SAME
                    gamma / gae_lambda as the reward critic
                  policy objective advantage  A = A_r - lambda * A_c
                  lambda >= 0 by projected dual ascent on (J_c - d)

Everything else is identical: the same observation, the same potential shaping, the same per-step
penalty, the same attitude deadband, the same goal bonus, the same smoothness terms, the same network
sizes, the same total environment steps, the same curriculum, the same seed.

TWO CHOICES, MADE HERE AND REPORTED.
  d (target collision rate) = 0.02.
      The shipped PPO scores collision 0.0820 on the registered pool; the jointly trained pair scores
      0.0225. d = 0.02 therefore asks the constrained agent for roughly the safety the CERTIFICATE
      buys, which is the comparison worth making. A looser d (0.05) is already met by the shipped
      agent's own training-time rate at some points and would leave the constraint inactive; a
      tighter one (0.005) is below anything either method has shown and would pin lambda at its
      bound, which stop condition 3 forbids.
  dual step size eta = 0.5, with lambda clipped to [0, LAMBDA_MAX = 50].
      The gap (J_c - d) is a rate difference, order 0.05-0.1 once training is past the initial
      collapse, so eta = 0.5 moves lambda by ~0.03/iteration: slow against PPO's own 4-epoch update
      so the two do not race, fast enough to reach O(1) within ~40 iterations of 3000. LAMBDA_MAX is
      a diagnostic bound, not a tuning knob -- saturating it for more than a quarter of training is
      the dispatch's stop condition 3 and is checked at the end of the run.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.common.outcomes import resolve_outcome, step_outcomes
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.ppo import train as T
from src.frameworks.ppo.nets import PPOValue

Tensor = torch.Tensor

TARGET_COST_RATE_D = 0.02
DUAL_STEP_ETA = 0.5
LAMBDA_MAX = 50.0
LAMBDA_INIT = 0.0
COST_EMA_BETA = 0.0     # 0 => J_c is the current iteration's collision rate, no smoothing


def cost_buffers(roll, scene, system, config, *, gamma, lam, cost_value) -> dict[str, Tensor]:
    """Cost signal, cost GAE and cost returns on exactly the transitions `compute_rewards_and_gae` keeps.

    c_t = 1 on the transition that terminates in a COLLISION, 0 on every other transition. The active
    mask, the terminal index and the flattening order are recomputed with the SAME expressions
    `train.compute_rewards_and_gae` uses, so `keep` is index-for-index identical.
    """
    states = roll["states"]
    t_steps, b = roll["val"].shape
    device = states.device
    resolved = resolve_outcome(step_outcomes(states, scene, system, config))
    event_step = resolved.event_step.to(device)
    outcomes = resolved.outcome
    term_t = torch.where(event_step >= 1, event_step - 1,
                         torch.where(event_step < 0, torch.full_like(event_step, t_steps - 1),
                                     torch.full_like(event_step, -1)))
    t_idx = torch.arange(t_steps, device=device).unsqueeze(1)
    active = (t_idx <= term_t.unsqueeze(0)) & (term_t.unsqueeze(0) >= 0)

    is_coll = torch.tensor([o == "collision" for o in outcomes], dtype=torch.bool, device=device)
    cost = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    valid = term_t >= 0
    lanes = torch.nonzero(valid & is_coll, as_tuple=False).flatten()
    if lanes.numel():
        cost[term_t[lanes], lanes] = 1.0
    cost = cost * active

    with torch.no_grad():
        vc = cost_value(roll["obs"].reshape(t_steps * b, -1)).reshape(t_steps, b)

    next_nonterminal = (t_idx < term_t.unsqueeze(0)).to(states.dtype)
    advc = torch.zeros((t_steps, b), dtype=states.dtype, device=device)
    last = torch.zeros(b, dtype=states.dtype, device=device)
    for t in range(t_steps - 1, -1, -1):
        m = next_nonterminal[t]
        nv = vc[t + 1] * m if t + 1 < t_steps else torch.zeros(b, dtype=states.dtype, device=device)
        delta = cost[t] + gamma * nv - vc[t]
        last = delta + gamma * lam * m * last
        advc[t] = last
    retc = advc + vc

    keep = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
    n_valid = int(valid.sum().item())
    return {"adv_c": advc.reshape(-1)[keep], "ret_c": retc.reshape(-1)[keep],
            "val_c": vc.reshape(-1)[keep],
            "episode_cost_rate": (float(is_coll[valid].to(states.dtype).mean().item())
                                  if n_valid else 0.0),
            "n_valid_lanes": n_valid}


def lagrangian_update(policy, value, cost_value, opt_pi, opt_vf, opt_vc, buf, ppo_cfg, torch_gen,
                      lam_dual: float) -> dict[str, float]:
    """`train.ppo_update` with A = A_r - lambda*A_c and a third optimizer for the cost critic.

    The reward-side arithmetic (clipping, value clipping, entropy, grad clipping, minibatching,
    diagnostics) is `train.ppo_update`'s, reached by handing it a buffer whose `adv` is already the
    combined advantage. The cost critic is fitted in its own pass with the same clipped-value form.
    """
    combined = dict(buf)
    combined["adv"] = buf["adv"] - lam_dual * buf["adv_c"]
    stats = T.ppo_update(policy, value, opt_pi, opt_vf, combined, ppo_cfg, torch_gen)

    # cost critic: same epochs / minibatching / clip form as the reward critic
    n = buf["obs"].shape[0]
    epochs = int(ppo_cfg["epochs"])
    clip_vloss = bool(ppo_cfg["clip_vloss"])
    clip = float(ppo_cfg["clip_ratio"])
    max_grad_norm = float(ppo_cfg["max_grad_norm"])
    mb_target = int(ppo_cfg.get("minibatch_size", 0) or 0)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=torch_gen, device=buf["obs"].device)
        mb = max(1, min(n, mb_target)) if mb_target > 0 else max(1, -(-n // int(ppo_cfg["n_minibatches"])))
        for s in range(0, n, mb):
            idx = perm[s:s + mb]
            if idx.numel() == 0:
                continue
            pred = cost_value(buf["obs"][idx]).squeeze(-1)
            ret_mb, old_mb = buf["ret_c"][idx], buf["val_c"][idx]
            if clip_vloss:
                un = (pred - ret_mb) ** 2
                cl = (old_mb + torch.clamp(pred - old_mb, -clip, clip) - ret_mb) ** 2
                loss = 0.5 * torch.max(un, cl).mean()
            else:
                loss = 0.5 * ((pred - ret_mb) ** 2).mean()
            opt_vc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cost_value.parameters(), max_grad_norm)
            opt_vc.step()
            losses.append(float(loss.item()))
    stats["vc_loss"] = float(np.mean(losses)) if losses else 0.0
    return stats


def dual_step(lam_dual: float, cost_rate: float, d: float, eta: float) -> float:
    """Projected dual ASCENT: lambda <- clip(lambda + eta*(J_c - d), 0, LAMBDA_MAX)."""
    return float(min(LAMBDA_MAX, max(0.0, lam_dual + eta * (cost_rate - d))))
