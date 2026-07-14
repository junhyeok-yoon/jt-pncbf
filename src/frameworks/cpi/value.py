"""CPI value trunk (02_control §3.2): 3 hidden layers x 256, Softplus(beta=20), raw linear head.

Single member (n_vs=1); no read-out clip in this version (conservatism lives in the pinball tau and the
conformal eps_q, not in an ensemble-min or a clip). PyTorch default init. Input = the dim-19 DI observation.
Forward returns the raw scalar V_hat per state.
"""
from __future__ import annotations

import torch
from torch import nn

Tensor = torch.Tensor


class CPIValue(nn.Module):
    def __init__(self, obs_dim: int = 19, hidden: int = 256, n_layers: int = 3, beta: float = 20.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Softplus(beta=beta))
            d = hidden
        layers.append(nn.Linear(d, 1))                                   # raw linear head, no clip
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)
