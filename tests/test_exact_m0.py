"""v2.5.1 E1 — exact_m0 safety channel (unclipped single-backup certificate V_m0 as the filter h_fn):
value parity with the family labeler's j=0 member (<=1e-6), finite+nonzero x-gradients (autograd through
the 25-step brake rollout), and the frozen-channel trainer gate riding on type 'exact_m0'."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from src.common.maneuver_value import build_safety_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.frameworks.cpi.labels import m0_value_raw

REPO = Path(__file__).resolve().parents[1]


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    cfg = m(b, e)
    cfg["safety_channel"] = {"type": "exact_m0", "gamma_margin": 0.0}
    return cfg


def _batch(system):
    torch.manual_seed(0)
    B, K = 64, 12
    x = torch.empty(B, 4, dtype=torch.float32)
    x[:, :2].uniform_(-4.0, 4.0); x[:, 2:].uniform_(-2.5, 2.5)
    C = torch.zeros(B, K, 2, dtype=torch.float32); R = torch.zeros(B, K, dtype=torch.float32)
    A = torch.zeros(B, K, dtype=torch.bool)
    C[:, 0] = torch.tensor([1.0, 0.5]); R[:, 0] = 0.4; A[:, 0] = True
    C[:, 1] = torch.tensor([-1.5, -1.0]); R[:, 1] = 0.3; A[:, 1] = True
    scene = SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A, goal=torch.zeros(B, 2))
    return x, C, R, A, scene


def test_exact_m0_value_parity_with_labeler_j0():
    cfg = _cfg(); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    h_fn = build_safety_h_fn(system, cfg, None)
    x, C, R, A, scene = _batch(system)
    hv = h_fn(x, scene)
    ref = m0_value_raw(x, C, R, A, system, cfg, float(cfg["env"]["dt"]))       # labeler j=0 member
    assert torch.allclose(hv, ref, atol=1e-6), float((hv - ref).abs().max())


def test_exact_m0_xgrad_finite_and_nonzero():
    cfg = _cfg(); system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(torch.float32)
    h_fn = build_safety_h_fn(system, cfg, None)
    x, C, R, A, scene = _batch(system)
    x = x.clone().requires_grad_(True)
    h = h_fn(x, scene).sum()
    h.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def test_exact_m0_frozen_gate_ride(tmp_path: Path):
    from src.frameworks.jt_pncbf.train import run_training
    r = run_training(stage="smoke", output_root=tmp_path, seed=5, smoke_eval_scenes=1,
                     device="cpu", safety_channel="exact_m0")
    assert not r.halted
    # frozen channel => no value learning (K_V=0): the value loss column is a hard 0.0.
    assert r.last_value_loss == 0.0
    assert r.last_pi_grad_norm > 0.0
