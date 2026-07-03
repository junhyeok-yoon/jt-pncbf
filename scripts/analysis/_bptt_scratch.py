"""Scratch copy of the policy-BPTT TASK return (per-sample) with autograd-surgery hooks, for the
gradient-explosion attribution (Stages B/C). The training loss file is untouched.

Focus = the TASK return term gate_in * sum_k discount_k (d2 + lambda_v v2 + mu_u u2). Stage 3a of
step4_audit.md established this term is ~99% of L_pi and the explosion source; the small per-step
regularizers are excluded so the ablations act on the exploding term only.

The filtered action reproduces HardNetFilter.__call__ (non-lookahead path, which is the run config)
by calling the SAME _base_projection / _box_aware_projection with the CBF coefficients (h, L_f h,
L_g h) computed by the SAME _cbf_terms. Hooks:
  detach_filter_jac : detach (h, L_f h, L_g h) so d(u_safe)/dx flows only through u_nom (C1).
  singular_mask_tau : stop gradient through u_safe at steps with ||L_g h|| < tau (C2).
"""
from __future__ import annotations

from typing import Any

import torch

from src.common.filter_hardnet import (
    _base_alpha, _base_projection, _box_aware_projection, _cbf_terms, _hardnet_params,
)
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.frameworks.jt_pncbf.losses import _scene_goal, frozen_params

Tensor = torch.Tensor


def _filtered(system, h_fn, params, x, scene, u_nom, *, detach_filter_jac, singular_mask_tau):
    h, lf_h, lg_h = _cbf_terms(system, h_fn, x, scene, u_nom, create_graph=True)
    if detach_filter_jac:
        h, lf_h, lg_h = h.detach(), lf_h.detach(), lg_h.detach()
    alpha = _base_alpha(h, params)
    row_upper = -lf_h - alpha * h
    bounds = system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)
    projected = _base_projection(u_nom, lg_h, row_upper, bounds, params)
    if params.box_aware:
        u_safe, _ = _box_aware_projection(u_nom, projected, lg_h, row_upper, bounds)
    else:
        u_safe = projected
    if singular_mask_tau is not None:
        sing = (torch.linalg.norm(lg_h, dim=1) < float(singular_mask_tau)).unsqueeze(1)
        u_safe = torch.where(sing, u_safe.detach(), u_safe)
    return u_safe


def task_return_persample(
    *, system, policy_net, value_net, batch, config,
    bptt_t: int, gamma_t: float | None = None,
    detach_filter_jac: bool = False, singular_mask_tau: float | None = None,
) -> Tensor:
    policy_cfg = config["loss"]["policy"]
    dt = float(config["env"]["dt"])
    if gamma_t is None:
        gamma_t = float(policy_cfg["gamma_T"])
    lambda_v = float(policy_cfg["lambda_v"])
    mu_u = float(policy_cfg["mu_u"])
    tau_gate = float(policy_cfg["tau_gate"])
    params = _hardnet_params(config)
    h_fn = make_h_fn(value_net, system)

    with frozen_params(value_net):
        x = system.wrap_state(batch.states.detach())
        scene = batch.scene
        v_now = value_net.deployed_h(system.observation(x, scene)).detach()
        gate_in = torch.sigmoid(-v_now / tau_gate)
        task_cost = x.new_zeros(x.shape[0])
        discount = 1.0
        for _ in range(bptt_t):
            u_nom = policy_net(system.observation(x, scene))
            u_safe = _filtered(system, h_fn, params, x, scene, u_nom,
                               detach_filter_jac=detach_filter_jac, singular_mask_tau=singular_mask_tau)
            x = rk4_step(system, x, u_safe, dt)
            goal = _scene_goal(scene, x)
            d2 = torch.sum((system.position(x) - goal) ** 2, dim=1)
            v2 = system.speed(x) ** 2
            u2 = torch.sum(u_safe * u_safe, dim=1)
            task_cost = task_cost + discount * (d2 + lambda_v * v2 + mu_u * u2)
            discount *= gamma_t
        return gate_in * task_cost               # [B] per-sample


def persample_grad_norms(loss_vec: Tensor, params) -> Tensor:
    """Per-sample policy-gradient L2 norms via one batched backward (is_grads_batched)."""
    B = loss_vec.shape[0]
    eye = torch.eye(B, device=loss_vec.device, dtype=loss_vec.dtype)
    grads = torch.autograd.grad(loss_vec, list(params), grad_outputs=eye,
                                is_grads_batched=True, retain_graph=False, allow_unused=True)
    sq = None
    for g in grads:
        if g is None:
            continue
        gi = g.reshape(B, -1)
        sq = gi.pow(2).sum(dim=1) if sq is None else sq + gi.pow(2).sum(dim=1)
    return sq.clamp_min(0).sqrt()
