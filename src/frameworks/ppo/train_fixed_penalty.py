"""FIXED-lambda collision pricing: PPO with a constant Lagrange multiplier, no dual ascent.

Stage 2 of the PPO-Lagrangian axis. Stage 1's dual ran away and halted under its stop condition;
here lambda is a CONSTANT held for the whole run, and the frontier is traced by varying it across
runs. There is no dual variable in this module at all.

TWO CORRECTIONS TO STAGE 1, both inside the axis.

(a) THE REFERENCE POINT IS -100.0. The shipped PPO's persisted config carries
    `ppo.reward.collision = -100.0`; the -5.0 at `train.py:215` is the in-code default the shipped
    run overrides. Every baseline reference in this module and its report is -100.0.

(b) NORMALIZE THE TWO ADVANTAGES SEPARATELY, THEN COMBINE.
    Stage 1 formed `A_r - lambda*A_c` and handed it to `train.ppo_update`, whose `normalize_adv`
    then rescaled the COMBINED quantity to unit variance -- dividing lambda's magnitude out and
    leaving it a direction weight rather than a price. Here:

        A_r_hat = (A_r - mean) / std          [reward advantage, normalized alone]
        A_c_hat = (A_c - mean) / std          [cost advantage, normalized alone]
        A       = (A_r_hat - lambda * A_c_hat) / (1 + lambda)

    THE PUBLISHED FORM. This is the PPO-Lagrangian penalized objective of Ray, Achiam and Amodei,
    "Benchmarking Safe Exploration in Deep Reinforcement Learning" (2019), whose reference
    implementation (`safety-starter-agents`, the PPO-Lagrangian agent) forms the surrogate as
    (ratio*adv - penalty*ratio*cadv) / (1 + penalty) with the reward and cost advantages centred and
    scaled separately beforehand. THE TWO AGREE: their surrogate is
    ratio * [(A_r_hat - p*A_c_hat) / (1+p)], and PPO's clipped surrogate is linear in the advantage,
    so factoring the combined advantage out of the ratio -- which is what the line below does -- is
    the same objective term for term. The 1/(1+lambda) keeps the effective step size bounded as
    lambda grows: at lambda = 0 the objective is exactly the reward advantage, and as lambda -> inf
    it tends to -A_c_hat, with every intermediate lambda a genuine convex weight rather than a
    rescaled direction.

WHAT IS UNCHANGED. Same observation, potential shaping, per-step penalty, attitude deadband, goal
bonus, smoothness terms, network sizes, environment steps, curriculum, seed. The collision terminal
is 0.0 in the reward for every run; collision reaches the policy only through the cost channel.

NOT ONE LINE OF ANY EXISTING FILE IS CHANGED. `cost_buffers` is imported from the stage-1 module,
which itself recomputes the terminal index rather than widening `compute_rewards_and_gae`.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
import torch

from src.frameworks.ppo import train as T
from src.frameworks.ppo.train_lagrangian import cost_buffers  # noqa: F401  (re-exported for drivers)

Tensor = torch.Tensor

SHIPPED_COLLISION_TERMINAL = -100.0     # correction (a): the shipped run's own config value


def _norm(a: Tensor) -> Tensor:
    return (a - a.mean()) / (a.std() + 1e-8) if a.numel() > 1 else a


def combined_advantage(adv_r: Tensor, adv_c: Tensor, lam: float) -> Tensor:
    """(A_r_hat - lambda*A_c_hat) / (1 + lambda), each advantage normalized ALONE first.

    Correction (b). At lambda = 0 this is exactly the normalized reward advantage, so the lambda = 0
    run is the shipped PPO minus its collision terminal and nothing else.
    """
    return (_norm(adv_r) - float(lam) * _norm(adv_c)) / (1.0 + float(lam))


def fixed_penalty_update(policy, value, cost_value, opt_pi, opt_vf, opt_vc, buf, ppo_cfg, torch_gen,
                         lam: float) -> dict[str, float]:
    """`train.ppo_update` with the combined advantage precomputed and its own normalization OFF.

    `normalize_adv` is switched off in a COPY of the config for this call only -- the file is not
    touched -- because the combining above has already normalized each advantage on its own. Letting
    ppo_update normalize again is exactly the stage-1 defect.
    """
    cfg = dict(ppo_cfg)
    cfg["normalize_adv"] = False
    combined = dict(buf)
    combined["adv"] = combined_advantage(buf["adv"], buf["adv_c"], lam)
    stats = T.ppo_update(policy, value, opt_pi, opt_vf, combined, cfg, torch_gen)

    n = buf["obs"].shape[0]
    epochs = int(ppo_cfg["epochs"])
    clip = float(ppo_cfg["clip_ratio"])
    max_grad_norm = float(ppo_cfg["max_grad_norm"])
    clip_vloss = bool(ppo_cfg["clip_vloss"])
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
