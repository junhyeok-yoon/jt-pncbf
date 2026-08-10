from __future__ import annotations

from typing import Any, Mapping

import math

import torch


Tensor = torch.Tensor


def pncbf_target(
    h_seq: Tensor,
    lambda_disc: Tensor | float,
    dt: float,
    target_rhs: Tensor | float,
    bootstrap_tail: Tensor,
    ceiling: float = 1.0,
) -> Tensor:
    # v2.8.4 ceiling axis: `ceiling` is the codomain's UPPER bound, shared with the read-out via the
    # single key network.value.ceiling. The default 1.0 leaves every existing call byte-identical.
    # The LOWER bound stays at -1.0 at both clamp sites and is not configurable.
    if h_seq.shape[0] < 1:
        raise ValueError("h_seq must contain at least one state.")

    squeeze_output = h_seq.ndim == 1
    costs = torch.clamp(h_seq.unsqueeze(1) if squeeze_output else h_seq, min=-1.0, max=ceiling)
    rhs = _as_tensor(target_rhs, costs)
    tail = _as_tensor(bootstrap_tail, costs)
    lhs, int_rhs, discount_rhs = compute_disc_avoid_terms(costs, lambda_disc, dt)
    rhs_full = int_rhs + discount_rhs * tail.unsqueeze(0)
    mixed = lhs + rhs * torch.relu(rhs_full - lhs)
    result = torch.clamp(mixed, min=-1.0, max=ceiling)
    return result.squeeze(1) if squeeze_output else result

def compute_disc_avoid_terms(
    costs: Tensor,
    lambda_disc: Tensor | float,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if costs.ndim != 2:
        raise ValueError(f"costs must have shape [T, B], got {tuple(costs.shape)}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    gamma = torch.exp(-_as_tensor(lambda_disc, costs) * float(dt))
    one_minus_gamma = 1.0 - gamma
    time_dim = costs.shape[0]
    t_range = torch.arange(time_dim, device=costs.device, dtype=costs.dtype)
    discount_rhs = gamma ** (time_dim - 1 - t_range)
    discount_rhs = discount_rhs.unsqueeze(1).expand_as(costs)

    int_rhs = torch.zeros_like(costs)
    for idx in range(time_dim - 2, -1, -1):
        int_rhs[idx] = one_minus_gamma * costs[idx] + gamma * int_rhs[idx + 1]

    lhs = torch.zeros_like(costs)
    lhs[-1] = costs[-1]
    for idx in range(time_dim - 2, -1, -1):
        candidate = one_minus_gamma * costs[idx] + gamma * lhs[idx + 1]
        lhs[idx] = torch.maximum(costs[idx], candidate)

    return lhs, int_rhs, discount_rhs


def rpcbf_target(h_windows: Tensor, beta: float) -> Tensor:
    if h_windows.shape[-1] < 2:
        raise ValueError("h_windows must contain H + 1 entries with H >= 1.")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")

    values = torch.clamp(h_windows, min=-1.0, max=1.0)
    max_values = torch.max(values, dim=-1).values
    shifted = beta * (values - max_values.unsqueeze(-1))
    log_mean_exp = torch.logsumexp(shifted, dim=-1)
    log_mean_exp = log_mean_exp - torch.log(
        values.new_tensor(float(values.shape[-1]))
    )
    target = max_values + log_mean_exp / beta
    return torch.clamp(target, min=-1.0, max=1.0)


def target_from_config(
    config: Mapping[str, Any],
    *,
    h_seq: Tensor | None = None,
    h_windows: Tensor | None = None,
    gamma_disc: Tensor | float | None = None,
    lambda_disc: Tensor | float | None = None,
    target_rhs: Tensor | float | None = None,
    bootstrap_tail: Tensor | None = None,
) -> Tensor:
    target_cfg = config["value_target"]
    target_type = str(target_cfg["type"])
    if target_type == "pncbf":
        if h_seq is None or target_rhs is None:
            raise ValueError("PNCBF target requires h_seq and target_rhs.")
        if bootstrap_tail is None:
            raise ValueError("PNCBF target requires bootstrap_tail.")
        lambda_value = lambda_disc if lambda_disc is not None else gamma_disc
        if lambda_value is None:
            raise ValueError("PNCBF target requires lambda_disc.")
        return pncbf_target(
            h_seq,
            lambda_value,
            float(config["env"]["dt"]),
            target_rhs,
            bootstrap_tail,
        )
    if target_type == "rpcbf":
        if h_windows is None:
            raise ValueError("RPCBF target requires h_windows.")
        return rpcbf_target(h_windows, float(target_cfg["rpcbf_beta"]))
    raise ValueError(f"Unsupported value target type: {target_type!r}")


def schedule_value(
    schedule_cfg: Mapping[str, float],
    n_sched: int,
    n_steps: int,
    vs_warmup_steps: int = 0,
) -> float:
    n_eff = n_steps - vs_warmup_steps
    if n_eff <= 0:
        raise ValueError("n_steps must exceed vs_warmup_steps.")

    init = float(schedule_cfg["init"])
    final = float(schedule_cfg["final"])
    warm_end = float(schedule_cfg["f_warm"]) * n_eff
    phase_end = float(schedule_cfg["f_phase1"]) * n_eff
    step = float(max(0, n_sched))

    if step < warm_end:
        return init
    if step >= phase_end:
        return final
    if phase_end <= warm_end:
        return final

    frac = (step - warm_end) / (phase_end - warm_end)
    return init + frac * (final - init)


def lambda_schedule_value(
    schedule_cfg: Mapping[str, float],
    epoch_index: int,
    total_epochs: int,
    dt: float,
) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    horizon_init = float(schedule_cfg["horizon_init"])
    horizon_final = float(schedule_cfg["horizon_final"])
    lambda_floor = float(schedule_cfg["lambda_floor"])
    warm_end = int(round(total_epochs * float(schedule_cfg["f_warm"])))
    phase_end = int(round(total_epochs * float(schedule_cfg["f_phase1"])))
    epoch = int(epoch_index)
    lambda_phase1_end = _lambda_from_horizon(horizon_final, dt)

    if epoch < warm_end:
        return _lambda_from_horizon(horizon_init, dt)
    if epoch < phase_end:
        frac = (epoch - warm_end) / max(phase_end - warm_end, 1)
        horizon = horizon_init + frac * (horizon_final - horizon_init)
        return _lambda_from_horizon(horizon, dt)

    frac = (epoch - phase_end) / max(total_epochs - phase_end, 1)
    return lambda_phase1_end + frac * (lambda_floor - lambda_phase1_end)


def gamma_from_lambda(lambda_disc: float, dt: float) -> float:
    return float(math.exp(-float(lambda_disc) * float(dt)))


def _truncated_avoid(costs: Tensor, gamma: Tensor) -> Tensor:
    values = [costs[-1]]
    next_value = costs[-1]
    for idx in range(costs.shape[0] - 2, -1, -1):
        gamma_t = _gamma_at(gamma, idx)
        cost_t = costs[idx]
        next_value = torch.maximum(
            cost_t,
            (1.0 - gamma_t) * cost_t + gamma_t * next_value,
        )
        values.append(next_value)
    values.reverse()
    return torch.stack(values, dim=0)


def _bootstrapped_avoid(costs: Tensor, gamma: Tensor, tail: Tensor) -> Tensor:
    values = []
    next_value = tail
    for idx in range(costs.shape[0] - 1, -1, -1):
        gamma_t = _gamma_at(gamma, idx)
        cost_t = costs[idx]
        next_value = torch.maximum(
            cost_t,
            (1.0 - gamma_t) * cost_t + gamma_t * next_value,
        )
        values.append(next_value)
    values.reverse()
    return torch.stack(values, dim=0)


def _gamma_at(gamma: Tensor, idx: int) -> Tensor:
    if gamma.ndim == 0:
        return gamma
    return gamma[idx]


def _as_tensor(value: Tensor | float, like: Tensor) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return like.new_tensor(float(value))


def _lambda_from_horizon(horizon_steps: float, dt: float) -> float:
    if horizon_steps <= 1.0:
        raise ValueError(f"horizon_steps must exceed 1, got {horizon_steps}.")
    return float(-math.log(1.0 - 1.0 / float(horizon_steps)) / float(dt))
