from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.frameworks.oc_pncbf.value_target import (
    compute_disc_avoid_terms,
    gamma_from_lambda,
    lambda_schedule_value,
    pncbf_target,
    rpcbf_target,
    schedule_value,
    target_from_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_pncbf_target_matches_hand_reference() -> None:
    h_seq = torch.tensor([-0.8, -0.2, 0.4, -0.5, 0.1], dtype=DTYPE)
    lambda_disc = torch.tensor(-torch.log(torch.tensor(0.5, dtype=DTYPE)), dtype=DTYPE)
    bootstrap_tail = torch.tensor(0.8, dtype=DTYPE)

    pure = pncbf_target(h_seq, lambda_disc, 1.0, 0.0, bootstrap_tail)
    full_mix = pncbf_target(h_seq, lambda_disc, 1.0, 1.0, bootstrap_tail)

    expected_pure = torch.tensor([-0.35, 0.1, 0.4, -0.2, 0.1], dtype=DTYPE)
    expected_full = torch.tensor([-0.35, 0.1, 0.4, 0.15, 0.8], dtype=DTYPE)
    assert torch.allclose(pure, expected_pure, atol=1.0e-6)
    assert torch.allclose(full_mix, expected_full, atol=1.0e-6)

    unclamped = torch.tensor([-2.0, 1.5], dtype=DTYPE)
    target = pncbf_target(unclamped, 0.1, 1.0, 1.0, torch.tensor(3.0, dtype=DTYPE))
    assert torch.all(target >= -1.0)
    assert torch.all(target <= 1.0)


def test_disc_avoid_terms_match_reference_form() -> None:
    costs = torch.tensor([[-0.8, -0.3], [-0.2, 0.1], [0.4, -0.4]], dtype=DTYPE)
    lambda_disc = -torch.log(torch.tensor(0.5, dtype=DTYPE))
    lhs, int_rhs, discount_rhs = compute_disc_avoid_terms(costs, lambda_disc, dt=1.0)

    expected_lhs = torch.tensor([[-0.35, -0.1], [0.1, 0.1], [0.4, -0.4]], dtype=DTYPE)
    expected_int = torch.tensor([[-0.45, -0.125], [-0.1, 0.05], [0.0, 0.0]], dtype=DTYPE)
    expected_discount = torch.tensor([[0.25, 0.25], [0.5, 0.5], [1.0, 1.0]], dtype=DTYPE)
    assert torch.allclose(lhs, expected_lhs, atol=1.0e-6)
    assert torch.allclose(int_rhs, expected_int, atol=1.0e-6)
    assert torch.allclose(discount_rhs, expected_discount, atol=1.0e-6)


def test_rpcbf_target_is_normalized_smooth_max() -> None:
    window = torch.tensor([[-0.6, -0.2, 0.0, 0.4]], dtype=DTYPE)
    betas = [1.0, 5.0, 20.0, 100.0]
    targets = torch.stack([rpcbf_target(window, beta) for beta in betas])

    mean_value = torch.mean(window, dim=1)
    max_value = torch.max(window, dim=1).values
    assert torch.all(targets >= mean_value - 1.0e-12)
    assert torch.all(targets <= max_value + 1.0e-12)
    assert torch.all(targets[1:] >= targets[:-1] - 1.0e-12)
    assert torch.allclose(targets[-1], max_value, atol=2.0e-2)


def test_rpcbf_horizon_invariance_and_stability() -> None:
    beta = 5.0
    base = torch.tensor([[-0.5, 0.2, 0.4]], dtype=DTYPE)
    padded = torch.tensor([[-0.5, 0.2, 0.4, 0.4]], dtype=DTYPE)
    base_target = rpcbf_target(base, beta)
    padded_target = rpcbf_target(padded, beta)
    assert torch.abs(padded_target - base_target) < 0.06

    max_value = torch.max(base, dim=1).values
    assert torch.allclose(rpcbf_target(base, 200.0), max_value, atol=1.0e-2)
    assert torch.allclose(rpcbf_target(padded, 200.0), max_value, atol=1.0e-2)

    large = torch.tensor([[-50.0, 0.0, 50.0]], dtype=DTYPE)
    stable = rpcbf_target(large, beta)
    assert torch.isfinite(stable).all()
    assert torch.all(stable >= -1.0)
    assert torch.all(stable <= 1.0)


def test_dispatcher_and_schedule_value() -> None:
    config = _load_config()
    h_windows = torch.tensor([[-0.4, 0.1, 0.2]], dtype=DTYPE)
    rpcbf_config = _deep_merge(config, {"value_target": {"type": "rpcbf"}})
    dispatched = target_from_config(rpcbf_config, h_windows=h_windows)
    expected = rpcbf_target(h_windows, float(config["value_target"]["rpcbf_beta"]))
    assert torch.allclose(dispatched, expected)

    schedule = config["schedules"]["gamma_disc"]
    lambda_value = lambda_schedule_value(
        schedule,
        epoch_index=0,
        total_epochs=int(config["training"]["oc_pncbf"]["epochs"]),
        dt=float(config["env"]["dt"]),
    )
    gamma_value = gamma_from_lambda(lambda_value, float(config["env"]["dt"]))
    assert abs(gamma_value - 0.95) < 1.0e-12

    rhs_schedule = config["schedules"]["target_rhs"]
    rhs_value = schedule_value(
        rhs_schedule,
        n_sched=450,
        n_steps=int(config["training"]["oc_pncbf"]["epochs"]),
    )
    assert abs(rhs_value - 0.45) < 1.0e-12


def _load_config() -> Mapping[str, Any]:
    base = yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO_ROOT / "src/configs/exp_config.yaml").read_text())
    return _deep_merge(base, exp)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
