"""CPI safety channel: a frozen learned certificate V_hat as the deployed CBF h_fn.

make_cpi_h_fn(checkpoint, system) -> h_fn(x, scene) = RAW net forward on the standard dim-19 observation
(no [-1,1] clip; raw range ~[-5, +1]). Weights come from the given CPIValue checkpoint and are frozen
(requires_grad_(False)); state-gradients flow via autograd (the filter's _cbf_terms differentiates h_fn
w.r.t. x). Matches the learned-value channel exactly (make_h_fn -> deployed_h -> raw value(obs)); the
filter/infeasibility math of 02_control §5-§7 is unchanged and carries no |h|<=1 assumption (audited:
_base_alpha uses only sign(h), the projection is h-magnitude-agnostic, only the control is bound-clamped).

h_eff = h + V_SHIFT + gamma_margin per 02_control §5.1; here gamma_margin = 0.0 (conservatism lives in
tau=0.9) and V_SHIFT (1e-3) is unwired in the codebase (deployed_h is raw), so h_eff == h (raw). The
adapter returns the RAW forward so it is bit-identical to a direct net call (parity test).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.frameworks.cpi.value import CPIValue

Tensor = torch.Tensor


def load_frozen_cpi_net(checkpoint_path, obs_dim: int) -> CPIValue:
    ck = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    net = CPIValue(obs_dim=obs_dim)
    net.load_state_dict(ck["model_state"])
    net.requires_grad_(False)
    net.eval()
    return net


def make_exact_m0_h_fn(system, config):
    """exact_m0 safety channel: h(x) = UNCLIPPED single-backup certificate V_m0(x) — the family labeler's
    j=0 member (the differentiable 25-step deadband-brake rollout, running-max of the unclipped signed_h
    position ramp). No library, backup only; carries NO learned parameters and NO clip; x-gradients flow via
    autograd (as the v2.5.0 maneuver channel differentiates V_M). gamma_margin 0.0. By construction the
    value is bit-identical to the labeler's m_0 path (m0_value_raw), so the deploy filter and the label
    certificate agree exactly (removes the P1 learned-filter semantic gap)."""
    from src.frameworks.cpi.labels import m0_value_raw

    dt = float(config["env"]["dt"])

    def h_fn(x: Tensor, scene: Any) -> Tensor:
        return m0_value_raw(x, scene.obstacle_centers, scene.obstacle_radii, scene.obstacle_active,
                            system, config, dt)

    return h_fn


def make_cpi_h_fn(checkpoint_path, system):
    """Deployed h_fn for the frozen CPI certificate. Frozen weights; x-gradients via autograd; raw output."""
    net = load_frozen_cpi_net(checkpoint_path, system.obs_dim)
    state = {"net": net}

    def h_fn(x: Tensor, scene: Any) -> Tensor:
        n = state["net"]
        p = next(n.parameters())
        if p.device != x.device or p.dtype != x.dtype:
            n = n.to(device=x.device, dtype=x.dtype); state["net"] = n
        obs = system.observation(x, scene)
        return n(obs)                                                   # RAW; no clip

    return h_fn
