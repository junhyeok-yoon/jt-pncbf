"""Filter-free evaluation framework for the PPO baseline.

The shipped scoring path is  policy -> filter (HardNet projection + empty_fallback) -> action box -> plant ->
outcomes. The PPO baseline has NO certificate and NO filter, so "the shipped scoring config MINUS the filter"
reduces to  policy -> action box -> plant -> outcomes: the filter becomes the identity (u_safe == u_nom,
infeasible == all-False), which is exactly what this framework provides. Because the pools, the band
terminate/open switch, the reach terminal, and the outcome predicates are untouched, this stays the same
comparison as every JT/OC row — only the filter is removed.

By construction the eval then reports infeasibility == 0 (no filter can be infeasible), mean_proj_mag == 0
(no projection), and cps = reach - 2*collision - stuck - 0.5*(oob + timeout) - 0.3*0.
"""
from __future__ import annotations

from typing import Any

import torch

from src.common.system import System
from src.frameworks.ppo.nets import PPOPolicy, PPOValue


Tensor = torch.Tensor


class FilterFreeFramework:
    """Adapts a trained PPO policy to the src.eval.evaluate framework protocol with an identity filter."""

    def __init__(self, system: System, policy: PPOPolicy, value: PPOValue | None = None) -> None:
        self.system = system
        self.policy_net = policy
        # evaluate._tensor_options reads framework.value_net to resolve rollout dtype/device; the policy carries
        # the same dtype/device, so alias it when no value net is supplied (the value net is otherwise unused).
        self.value_net = value if value is not None else policy

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        with torch.no_grad():
            return self.policy_net.deterministic_action(self.system.observation(x, scene))

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        # Identity filter: the executed action IS the policy's action box output; nothing can be infeasible.
        return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
