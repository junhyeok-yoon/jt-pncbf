"""DENSE cost signal for the fixed-lambda penalty route: the certificate's own hazard, every step.

The asymmetry this module removes
---------------------------------
Stage 2 (`train_fixed_penalty`) swept a fixed multiplier against a cost that fires ONLY on the
transition that terminates in a collision -- one nonzero number per colliding episode, and nothing
at all on an episode that misses an obstacle by a centimetre. The certificate this paper deploys is
fitted to `h` at EVERY step: its target is sup_t h(x_t) along the closed-loop rollout, so `h` is
read on every state of every episode. The penalty route was therefore swept hard on a signal far
poorer than the one the proposed method receives.

Here the cost is built from `common.quadrotor_barrier.value_target_barrier` -- the SAME function,
called the SAME way, that the value-labelling sites call. Nothing else moves.

The cost map, and why this one
------------------------------
    c_t = (h(x_{t+1}) + 1) / 2

evaluated on the state each transition LANDS in, which is the state the sparse indicator scores
(it fires when x_{t+1} is a collision state), so the two costs are read at the same place and only
their shape differs.

`(h+1)/2` is AFFINE and strictly increasing, so it carries exactly the information `h` carries: no
more, and -- the point -- no less. The alternatives in the dispatch both destroy information in the
safe region, which is the region the whole question lives in:

  * `max(h, 0)` collapses every safe state to the single value 0. On the training distribution
    (measured, 185708 live transitions of the lambda = 2 policy at easy_frac 0) it is nonzero on
    2.49% of transitions. That is denser than the terminal indicator, which is nonzero on about
    0.10% of transitions, but it is still the terminal indicator's defect moved one step earlier:
    an episode that clears every cylinder by 3 cm and one that clears them by 3 m receive the same
    cost, and telling those apart is the entire job.
  * `max(h + m, 0)` restores a graded band of width `m` but introduces a free parameter that would
    have to be swept, and the run cap is 20 across all stages with 7 already spent. `(h+1)/2` is the
    `m -> infinity` limit of that family with the clamp never binding, so it needs no sweep.

Values on the certificate's hazard (`geom_form` exp, `ell` 0.125, `c_gain` 0.0 -- L363's own):
  at an obstacle contact, h = +1  ->  c = 1.0
  on the zero level set,  h =  0  ->  c = 0.5
  far from every cylinder and away from the bands, h -> -1  ->  c -> 0 (measured min 0.00000)
The map is NOT clamped above. `h` exceeds +1 only on the vertical branch at high |v_z|, on 0.17% of
live transitions (measured max c = 1.4879); clamping there would flatten the one part of the signal
that is about the ceiling and the floor.

Which hazard config
-------------------
`value_target_barrier` reads its hazard parameters off the config it is handed. The shipped PPO's
persisted config and the deployed certificate L363's differ in exactly two of them:

    env.quadrotor_3d.c_gain :  0.3   (PPO)  vs  0.0                          (L363)
    hazard                  :  absent -> 'clip'  vs  {geom_form: exp, ell: 0.125}  (L363)

`h_scale` 0.35, `band_hazard` {enabled, limit 4.0}, `r_max` 0.8 and `omega_max` 4.0 are already
identical. `hazard_config` below overlays those two keys onto a COPY of the training config, so the
cost is the certificate's own function of state. This is deliberate and is the axis: the point is to
hand the baseline the proposed method's signal, not the baseline's own idea of a hazard. No PPO key
moves -- the overlay is used for the cost channel only and never reaches the reward, the shaping,
the scenes, the dynamics or the outcome predicates.

NOT ONE LINE OF ANY EXISTING FILE IS CHANGED. The reward-side arithmetic, the combined advantage
`(A_r_hat - lambda*A_c_hat)/(1+lambda)` and the cost-critic pass are `train_fixed_penalty`'s,
imported and reused; only the cost tensor handed to the GAE recursion is different.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_barrier import value_target_barrier
from src.frameworks.ppo.train_fixed_penalty import (          # noqa: F401  (re-exported for drivers)
    combined_advantage,
    fixed_penalty_update,
)

Tensor = torch.Tensor

# L363 COLDABL's own hazard parameters, read off its persisted checkpoint config.
CERT_HAZARD = {"geom_form": "exp", "ell": 0.125}
CERT_C_GAIN = 0.0


def hazard_config(config: Mapping[str, Any]) -> dict:
    """A COPY of `config` carrying the deployed certificate's two hazard keys. Nothing else moves."""
    cfg = copy.deepcopy(dict(config))
    cfg["hazard"] = dict(CERT_HAZARD)
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["quadrotor_3d"] = dict(cfg["env"]["quadrotor_3d"])
    cfg["env"]["quadrotor_3d"]["c_gain"] = float(CERT_C_GAIN)
    return cfg


def dense_cost(system, states: Tensor, scene, cfg_h: Mapping[str, Any]) -> Tensor:
    """c = (h + 1)/2 on the LANDED state of every transition. `states` is [T+1, B, D]; returns [T, B]."""
    h = value_target_barrier(system, states[1:], scene, cfg_h)
    return 0.5 * (h + 1.0)


def dense_cost_buffers(roll, scene, system, config, *, gamma, lam, cost_value,
                       cfg_h: Mapping[str, Any] | None = None) -> dict[str, Tensor]:
    """`train_lagrangian.cost_buffers` with the dense cost in place of the collision indicator.

    The active mask, the terminal index and the flattening order are recomputed with the SAME
    expressions `train.compute_rewards_and_gae` uses, so `keep` is index-for-index identical to the
    reward buffers and to the sparse module's. `episode_cost_rate` is kept under its old name and
    old meaning -- the fraction of terminated lanes that ended in a collision -- so the metric is
    comparable across the sparse and dense runs; the dense signal is reported beside it as
    `mean_step_cost`.
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

    with torch.no_grad():
        cost = dense_cost(system, states, scene, cfg_h if cfg_h is not None else hazard_config(config))
    cost = cost.to(states.dtype) * active

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
    is_coll = torch.tensor([o == "collision" for o in outcomes], dtype=torch.bool, device=device)
    valid = term_t >= 0
    n_valid = int(valid.sum().item())
    n_active = int(active.sum().item())
    return {"adv_c": advc.reshape(-1)[keep], "ret_c": retc.reshape(-1)[keep],
            "val_c": vc.reshape(-1)[keep],
            "episode_cost_rate": (float(is_coll[valid].to(states.dtype).mean().item())
                                  if n_valid else 0.0),
            "mean_step_cost": (float(cost[active].mean().item()) if n_active else 0.0),
            "max_step_cost": (float(cost[active].max().item()) if n_active else 0.0),
            "n_valid_lanes": n_valid}
