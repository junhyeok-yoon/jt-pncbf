from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from src.common.filter_hardnet import HardNetFilter
from src.common.rk4 import rk4_step
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import BatchedScene
from src.frameworks.oc_pncbf.collection import TensorTransitionBatch
from src.frameworks.oc_pncbf.value_target import pncbf_target


Tensor = torch.Tensor


@dataclass(frozen=True)
class ValueLossResult:
    total: Tensor
    reach: Tensor
    targets: Tensor


@dataclass(frozen=True)
class PolicyLossResult:
    total: Tensor
    task: Tensor
    action_norm: Tensor
    smoothness: Tensor
    saturation_excess: Tensor
    pretanh: Tensor
    outside: Tensor
    grad_leak: float
    action_abs_mean: Tensor
    action_abs_max: Tensor
    satfrac_a_phi: Tensor


def value_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> ValueLossResult:
    obs = system.observation(batch.states, batch.scene)
    targets = value_targets(
        system=system,
        target_value_net=target_value_net,
        batch=batch,
        lambda_disc=lambda_disc,
        target_rhs=target_rhs,
        config=config,
    )
    prediction = value_net(obs)
    reach = torch.mean((prediction - targets.unsqueeze(1)) ** 2)
    total = float(config["loss"]["value"]["lambda_R"]) * reach
    return ValueLossResult(total=total, reach=reach, targets=targets)


def value_targets(
    *,
    system: System,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> Tensor:
    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_value_net.target_h(tail_obs)
    targets = pncbf_target(
        batch.h_sequence,
        lambda_disc,
        float(config["env"]["dt"]),
        target_rhs,
        bootstrap_tail,
    ).detach()
    return targets.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)


def policy_bptt_loss(
    *,
    system: System,
    policy_net: nn.Module,
    value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    config: Mapping[str, Any],
) -> PolicyLossResult:
    policy_cfg = config["loss"]["policy"]
    dt = float(config["env"]["dt"])
    bptt_t = int(config["training"]["jt"]["bptt_T"])
    gamma_t = float(policy_cfg["gamma_T"])
    lambda_v = float(policy_cfg["lambda_v"])
    mu_u = float(policy_cfg["mu_u"])
    tau_gate = float(policy_cfg["tau_gate"])
    hardnet = HardNetFilter(system, make_h_fn(value_net, system), config)

    _zero_grads(value_net.parameters())
    with frozen_params(value_net):
        x = system.wrap_state(batch.states.detach())
        scene = batch.scene
        v_now = value_net.deployed_h(system.observation(x, scene)).detach()
        gate_in = torch.sigmoid(-v_now / tau_gate)
        gate_out = torch.sigmoid(v_now / tau_gate)
        task_cost = x.new_zeros(x.shape[0])
        discount = 1.0
        nominal_actions: list[Tensor] = []
        safe_actions: list[Tensor] = []
        pretanh_values: list[Tensor] = []
        sat_excess_values: list[Tensor] = []
        pretanh_penalties: list[Tensor] = []

        for _ in range(bptt_t):
            obs = system.observation(x, scene)
            u_nom = policy_net(obs)
            nominal_actions.append(u_nom)
            if getattr(policy_net, "last_pretanh", None) is not None:
                z = policy_net.last_pretanh
                pretanh_values.append(z)
                excess_z = (z.abs() - float(policy_cfg["z_target"])).clamp_min(0.0)
                sq_excess = (excess_z * excess_z).mean(dim=1)
                if bool(policy_cfg["vs_gated_pretanh"]):
                    v_gate = value_net.deployed_h(system.observation(x, scene)).detach()
                    gate = torch.sigmoid((-v_gate - 0.02) / float(policy_cfg["vs_gate_tau"]))
                    sq_excess = sq_excess * gate
                pretanh_penalties.append(sq_excess.mean())

            sat_excess = (
                u_nom.abs() - float(policy_cfg["sat_excess_threshold"])
            ).clamp_min(0.0)
            sat_excess_values.append((sat_excess * sat_excess).sum(dim=1).mean())

            u_safe, _ = hardnet(x, scene, u_nom)
            safe_actions.append(u_safe)
            x = rk4_step(system, x, u_safe, dt)
            goal = _scene_goal(scene, x)
            pos_error = system.position(x) - goal
            d2 = torch.sum(pos_error * pos_error, dim=1)
            v2 = system.speed(x) * system.speed(x)
            u2 = torch.sum(u_safe * u_safe, dim=1)
            task_cost = task_cost + discount * (d2 + lambda_v * v2 + mu_u * u2)
            discount *= gamma_t

        action_stack = torch.stack(nominal_actions, dim=0)
        safe_stack = torch.stack(safe_actions, dim=0)
        action_norm = torch.mean(torch.sum(action_stack * action_stack, dim=2))
        if action_stack.shape[0] > 1:
            smoothness = torch.mean(
                torch.sum((action_stack[1:] - action_stack[:-1]) ** 2, dim=2)
            )
        else:
            smoothness = action_stack.new_zeros(())
        saturation_excess = torch.stack(sat_excess_values).mean()
        pretanh = (
            torch.stack(pretanh_penalties).mean()
            if pretanh_penalties
            else action_stack.new_zeros(())
        )
        obs0 = system.observation(batch.states.detach(), batch.scene)
        u0 = policy_net(obs0)
        x_next_unfiltered = rk4_step(system, batch.states.detach(), u0, dt)
        v_next = value_net.deployed_h(system.observation(x_next_unfiltered, batch.scene))
        outside = (gate_out * v_next).mean()
        task = (gate_in * task_cost).mean()
        total = (
            task
            + float(policy_cfg["lambda_a"]) * action_norm
            + float(policy_cfg["lambda_s"]) * smoothness
            + float(policy_cfg["lambda_sat"]) * saturation_excess
            + float(policy_cfg["lambda_pretanh"]) * pretanh
            + float(policy_cfg["w_outside"]) * outside
        )

    return PolicyLossResult(
        total=total,
        task=task.detach(),
        action_norm=action_norm.detach(),
        smoothness=smoothness.detach(),
        saturation_excess=saturation_excess.detach(),
        pretanh=pretanh.detach(),
        outside=outside.detach(),
        grad_leak=grad_norm(value_net.parameters()),
        action_abs_mean=safe_stack.abs().mean().detach(),
        action_abs_max=safe_stack.abs().max().detach(),
        satfrac_a_phi=_satfrac(action_stack, system).detach(),
    )


@contextmanager
def frozen_params(module: nn.Module) -> Iterator[None]:
    states = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, state in zip(module.parameters(), states, strict=True):
            param.requires_grad_(state)


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).cpu().item())
    return float(total**0.5)


def _zero_grads(parameters: Any) -> None:
    for parameter in parameters:
        parameter.grad = None


def _scene_goal(scene: BatchedScene, x: Tensor) -> Tensor:
    goal = scene.goal.to(device=x.device, dtype=x.dtype)
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    return goal


def _satfrac(actions: Tensor, system: System) -> Tensor:
    bounds = system.u_bounds.to(device=actions.device, dtype=actions.dtype)
    lower_dist = torch.abs(actions - bounds[:, 0])
    upper_dist = torch.abs(actions - bounds[:, 1])
    saturated = torch.any(torch.minimum(lower_dist, upper_dist) <= 1.0e-3, dim=-1)
    return saturated.to(dtype=actions.dtype).mean()
