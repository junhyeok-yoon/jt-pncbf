from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import nn

from src.common.system import System


Tensor = torch.Tensor


class ValueNetEnsemble(nn.Module):
    def __init__(self, obs_dim: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        value_cfg = config["network"]["value"]
        n_vs = int(value_cfg["n_vs"])
        hidden = int(value_cfg["hidden"])
        n_layers = int(value_cfg["n_layers"])
        beta = float(value_cfg["softplus_beta"])
        # v2.8.4 — the certificate's codomain UPPER bound (the one config key of the ceiling axis).
        # Default 1.0 reproduces the shipped read-out `clamp(., -1.0, 1.0)` bit for bit. The LOWER
        # bound stays at -1.0 and is deliberately not configurable: the safe set {V_hat <= 0} and the
        # whole region below it are untouched, so only the resolution of the unsafe region changes.
        self.ceiling = float(value_cfg.get("ceiling", 1.0))
        self.members = nn.ModuleList(
            [_make_value_member(obs_dim, hidden, n_layers, beta) for _ in range(n_vs)]
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.forward_all(obs)

    def forward_all(self, obs: Tensor) -> Tensor:
        return torch.cat([member(obs) for member in self.members], dim=1)

    def value_all(self, obs: Tensor) -> Tensor:
        return torch.clamp(self.forward_all(obs), min=-1.0, max=self.ceiling)

    def value(self, obs: Tensor) -> Tensor:
        return torch.mean(self.value_all(obs), dim=1)

    def target_h(self, obs: Tensor) -> Tensor:
        return torch.min(self.value_all(obs), dim=1).values

    def deployed_h(self, obs: Tensor) -> Tensor:
        return self.value(obs)


def make_h_fn(
    value_net: ValueNetEnsemble,
    system: System,
    use_target: bool = False,
) -> Callable[[Tensor, Any], Tensor]:
    def h_fn(x: Tensor, scene: Any) -> Tensor:
        obs = system.observation(x, scene)
        if use_target:
            return value_net.target_h(obs)
        return value_net.deployed_h(obs)

    return h_fn


def _make_value_member(
    obs_dim: int,
    hidden: int,
    n_layers: int,
    beta: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = obs_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Softplus(beta=beta))
        in_dim = hidden
    output = nn.Linear(in_dim, 1)
    nn.init.normal_(output.weight, mean=0.0, std=0.01)
    nn.init.constant_(output.bias, 0.0)
    layers.append(output)
    return nn.Sequential(*layers)
