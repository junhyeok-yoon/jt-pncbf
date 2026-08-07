"""PPO policy and value networks.

Design constraints (v2.8.2 mission):
  * Policy capacity MUST NOT be a confound: the policy reuses src.common.control_net.ControlNet — the exact
    class (and therefore trunk depth/width/activation) the JT policy uses. The only additions are a
    state-independent diagonal log-std (standard for continuous PPO) and a tanh squash to the action box.
  * The value net uses the SAME trunk architecture (network.control: n_layers x hidden, leaky_relu) with a
    SCALAR head and SEPARATE parameters — no trunk sharing, so the value loss can never perturb the policy.

Action mapping: the policy is a tanh-squashed diagonal Gaussian. The network head produces a pre-tanh mean
mu(obs); a sample is z = mu + std * eps, the executed control is u = center + halfwidth * tanh(z) (the per-rotor
action box). Log-probabilities carry the tanh Jacobian (SAC-standard, stable form). The constant halfwidth
Jacobian is dropped — it is identical between collection and update, so it cancels in every PPO ratio.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from src.common.control_net import ControlNet
from src.common.system import System


Tensor = torch.Tensor

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_LOG_2PI = math.log(2.0 * math.pi)


def _tanh_log_det(z: Tensor) -> Tensor:
    """log(1 - tanh(z)^2) per element, stable (SAC form): 2*(log2 - z - softplus(-2z))."""
    return 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))


def _gaussian_logp(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Diagonal-Gaussian log density of z, summed over action dims."""
    std = log_std.exp()
    return (-0.5 * ((z - mean) / std) ** 2 - log_std - 0.5 * _LOG_2PI).sum(dim=-1)


class PPOPolicy(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy over the action box, built on a JT ControlNet trunk+head."""

    def __init__(self, obs_dim: int, system: System, config: Mapping[str, Any], init_log_std: float = -0.5) -> None:
        super().__init__()
        # Reuse the exact JT policy class => identical learnable capacity (trunk + head). We consume only its
        # pre-tanh preactivation (control.head(control.trunk(obs))) and apply our own tanh squash, so the
        # network's parameters ARE the JT policy's parameters, one-for-one.
        self.control = ControlNet(obs_dim, system, config)
        self.action_dim = int(system.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))
        bounds = system.u_bounds.detach().clone().to(dtype=torch.float32)
        self.register_buffer("u_center", 0.5 * (bounds[:, 0] + bounds[:, 1]))
        self.register_buffer("u_halfwidth", 0.5 * (bounds[:, 1] - bounds[:, 0]))

    def mean_pre(self, obs: Tensor) -> Tensor:
        """Pre-tanh Gaussian mean mu(obs) from the ControlNet trunk+head."""
        return self.control.head(self.control.trunk(obs))

    def _to_box(self, a_unit: Tensor) -> Tensor:
        return self.u_center + self.u_halfwidth * a_unit

    def deterministic_action(self, obs: Tensor) -> Tensor:
        """Greedy action u = center + halfwidth * tanh(mu(obs)) — used for evaluation/deployment."""
        return self._to_box(torch.tanh(self.mean_pre(obs)))

    @torch.no_grad()
    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action for rollout. Returns (u_box, z_pretanh, logp). z is the value stored in the buffer
        and re-scored during the PPO update (the tanh-Gaussian re-parameterization)."""
        mean = self.mean_pre(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        z = mean + std * eps
        logp = _gaussian_logp(z, mean, log_std) - _tanh_log_det(z).sum(dim=-1)
        return self._to_box(torch.tanh(z)), z, logp

    def evaluate(self, obs: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Re-score stored pre-tanh samples z under the current policy. Returns (logp, entropy)."""
        mean = self.mean_pre(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        logp = _gaussian_logp(z, mean, log_std) - _tanh_log_det(z).sum(dim=-1)
        # Pre-squash diagonal-Gaussian differential entropy (standard continuous-PPO exploration bonus).
        entropy = (log_std + 0.5 * (_LOG_2PI + 1.0)).sum(dim=-1).expand(obs.shape[0])
        return logp, entropy


class PPOValue(nn.Module):
    """State-value V(obs). SAME trunk architecture as the policy (network.control), scalar head, own params."""

    def __init__(self, obs_dim: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        control_cfg = config["network"]["control"]
        hidden = int(control_cfg["hidden"])
        n_layers = int(control_cfg["n_layers"])
        activation = str(control_cfg["activation"])
        if activation != "leaky_relu":
            raise ValueError(f"PPOValue expects leaky_relu (matching ControlNet), got {activation!r}.")
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, obs: Tensor) -> Tensor:
        return self.head(self.trunk(obs)).squeeze(-1)
