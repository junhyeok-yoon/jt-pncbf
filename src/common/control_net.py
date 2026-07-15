from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.common.system import System


Tensor = torch.Tensor


class ControlNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        system: System,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        control_cfg = config["network"]["control"]
        hidden = int(control_cfg["hidden"])
        n_layers = int(control_cfg["n_layers"])
        activation = str(control_cfg["activation"])
        self.output = str(control_cfg["output"])
        self.output_gain = float(control_cfg.get("output_gain", 1.0 / 0.7))  # clamp_tanh over-range gain
        self.u_bounds = system.u_bounds.detach().clone()
        self.last_pretanh: Tensor | None = None

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(_activation(activation))
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, system.action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        self.last_pretanh = None
        preactivation = self.head(self.trunk(obs))
        self.last_pretanh = preactivation
        if self.output == "softsign":
            bounded = torch.nn.functional.softsign(preactivation)
        elif self.output == "tanh":
            bounded = torch.tanh(preactivation)
        elif self.output == "clamp_tanh":
            # v2.6.0: over-range gain then clamp, so the box boundary is REACHED inside the
            # differentiable region (softsign/tanh are open (-1,1) and never saturate the box, which
            # left the quadrotor unable to command |tau|=tau_max). g=1/0.7: tanh reaches +/-1/g=+/-0.7
            # of its range at |z|=atanh(0.7)=0.867, then clamp holds at the boundary (dead-gradient tail
            # by design: at max control the target IS the boundary, no gradient needed).
            gain = float(self.output_gain)
            bounded = torch.clamp(gain * torch.tanh(preactivation), -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported control output: {self.output!r}")

        bounds = self.u_bounds.to(device=obs.device, dtype=obs.dtype)
        center = 0.5 * (bounds[:, 0] + bounds[:, 1])
        half_width = 0.5 * (bounds[:, 1] - bounds[:, 0])
        return center + half_width * bounded


def _activation(name: str) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported control activation: {name!r}")
