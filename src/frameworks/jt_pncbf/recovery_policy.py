"""Learned avoid-optimal recovery policy pi_b for value-target conditioning (v2.4.0 Step 2).

Step 1 conditioned the value target on an analytic per-axis brake, giving a value that
approximates the braking envelope -(clearance - v_in^2/(2 u_max)); its gradient has only
radial/approach structure, so the HardNet filter can brake but not steer, and corridor scenes
get stuck. Step 2 makes the conditioning policy a LEARNED recovery policy that starts exactly at
the analytic brake (zero-initialized residual) and learns to swerve under an avoid-only
objective. The certified set {V^{h,pi_b} <= 0} then grows toward the maximal control-invariant
set and grad V acquires lateral components from pi_b's swerving label trajectories.

Parameterization (residual around the brake, same MLP family as pi_theta):
    r_xi(x, obs) = u_max * softsign(MLP_xi([x, obs]))          # last layer zero-init -> r==0 at init
    pi_b(x, obs) = clamp(pi_brake(x) + r_xi(x, obs), -u_max, u_max)
The clamp saturates the braking axis but leaves the lateral axis (v_i ~ 0) unsaturated, which is
where swerve gradients live.

Avoid-only objective (no task term, no V_S, no pi_theta): differentiable T_b-step rollout of the
ONLINE pi_b, softmax-over-time of the RAW (unclamped) signed-h. Minimizing it monotonically
drives pi_b toward avoid-optimal recovery; drift is one-directional and bounded, and the value
target consumes a Polyak-averaged copy pi_b_target to bound the per-step drift rate.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.common.brake_rollout import brake_policy
from src.common.rk4 import rk4_step
from src.common.system import System

Tensor = torch.Tensor


class RecoveryPolicy(nn.Module):
    def __init__(self, obs_dim: int, system: System, config: Mapping[str, Any],
                 u_max: float, eps_v: float) -> None:
        super().__init__()
        control_cfg = config["network"]["control"]
        hidden = int(control_cfg["hidden"])
        n_layers = int(control_cfg["n_layers"])
        activation = str(control_cfg["activation"])
        if activation != "leaky_relu":
            raise ValueError(f"Unsupported recovery activation: {activation!r}")
        self.u_max = float(u_max)
        self.eps_v = float(eps_v)
        self.u_bounds = system.u_bounds.detach().clone()
        self.last_residual: Tensor | None = None

        in_dim = system.state_dim + obs_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, system.action_dim)
        # zero-init the last layer so r_xi == 0 at init => pi_b == pi_brake exactly (Step 1 continuity)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, obs: Tensor) -> Tensor:
        pre = self.head(self.trunk(torch.cat([x, obs], dim=-1)))
        residual = self.u_max * torch.nn.functional.softsign(pre)
        self.last_residual = residual
        u_brake = brake_policy(x, self.u_max, self.eps_v)
        return torch.clamp(u_brake + residual, min=-self.u_max, max=self.u_max)


def _raw_signed_h(p: Tensor, scene: Any, h_scale: float) -> Tensor:
    """UNCLAMPED signed-h (same geometry as src.common.signed_h but no clamps), so gradients flow
    through clearance -> position -> actions in the avoid-only BPTT. Max over active obstacles."""
    centers = torch.as_tensor(scene.obstacle_centers, dtype=p.dtype, device=p.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=p.dtype, device=p.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=p.device)
    rel = p.unsqueeze(-2) - centers
    distance = torch.linalg.norm(rel, dim=-1)
    clearance = distance - radii
    h_all = 1.0 - 2.0 * (clearance / h_scale)                     # no clamp
    h_all = torch.where(active, h_all, torch.full_like(h_all, -1.0e9))
    return torch.max(h_all, dim=-1).values


def recovery_bptt_loss(
    *,
    system: System,
    recovery_policy: RecoveryPolicy,
    batch: Any,
    config: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """Avoid-only differentiable rollout of the ONLINE pi_b. Returns (L_b, residual_norm).

    L_b = mean_batch[ (1/beta) * log( (1/(T_b+1)) sum_k exp(beta * h_k) ) ]  over the T_b+1 raw h.
    Gradients flow h -> states -> actions -> xi through RK4; V_S and pi_theta are not in the graph.
    """
    rec_cfg = config["value_target"]["recovery"]
    T_b = int(rec_cfg["T_b"])
    beta = float(rec_cfg["beta"])
    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])

    x = system.wrap_state(batch.states.detach())
    scene = batch.scene
    h_list = [_raw_signed_h(system.position(x), scene, h_scale)]
    res_list: list[Tensor] = []
    for _ in range(T_b):
        obs = system.observation(x, scene)
        u = recovery_policy(x, obs)
        res_list.append(torch.linalg.norm(recovery_policy.last_residual, dim=1))
        x = rk4_step(system, x, u, dt)
        h_list.append(_raw_signed_h(system.position(x), scene, h_scale))
    H = torch.stack(h_list, dim=0)                                # [T_b+1, B]
    lme = (torch.logsumexp(beta * H, dim=0) - torch.log(H.new_tensor(float(H.shape[0])))) / beta
    loss = lme.mean()
    residual_norm = torch.stack(res_list, dim=0).mean() if res_list else H.new_zeros(())
    return loss, residual_norm
