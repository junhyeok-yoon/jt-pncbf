"""CPI safety channel: frozen weights (no h_fn parameter requires grad; x-gradients flow) and adapter
parity (h_fn == direct raw net forward within 1e-6)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.frameworks.cpi.channel import load_frozen_cpi_net, make_cpi_h_fn
from src.frameworks.cpi.value import CPIValue

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/secured_data/v2.5.1/seed42/checkpoints/best.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="secured v2.5.1 seed42 checkpoint required")


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def test_cpi_channel_frozen_and_xgrad():
    cfg = _cfg(); system = DoubleIntegrator(cfg)
    h_fn = make_cpi_h_fn(CKPT, system)
    net = load_frozen_cpi_net(CKPT, system.obs_dim)
    assert all(not p.requires_grad for p in net.parameters()), "cpi net parameters must be frozen"
    from src.frameworks.cpi.labels import sample_scenes, label_streams
    scenes = sample_scenes(1, label_streams(cfg)["scene"], cfg)
    from types import SimpleNamespace
    import numpy as np
    sc = SimpleNamespace(obstacle_centers=torch.as_tensor(scenes[0].obstacle_centers, dtype=torch.float32),
                         obstacle_radii=torch.as_tensor(scenes[0].obstacle_radii, dtype=torch.float32),
                         obstacle_active=torch.as_tensor(np.asarray(scenes[0].obstacle_active), dtype=torch.bool),
                         goal=torch.as_tensor(np.asarray(scenes[0].goal), dtype=torch.float32))
    x = torch.zeros(4, 4, requires_grad=True); x.data.uniform_(-2, 2)
    h = h_fn(x, sc)
    g = torch.autograd.grad(h.sum(), x)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0, "state-gradients must flow through the frozen net"


def test_cpi_adapter_parity():
    cfg = _cfg(); system = DoubleIntegrator(cfg)
    h_fn = make_cpi_h_fn(CKPT, system)
    net = load_frozen_cpi_net(CKPT, system.obs_dim).to(torch.float32)
    from src.frameworks.cpi.labels import sample_scenes, label_streams, stack_scene_obstacles
    from types import SimpleNamespace
    dev = torch.device("cpu")
    scenes = sample_scenes(2, label_streams(cfg)["scene"], cfg)
    C, R, A, G = stack_scene_obstacles(scenes, dev, torch.float32)
    torch.manual_seed(0)
    sid = torch.arange(2).repeat_interleave(64)
    x = torch.empty(128, 4); x[:, :2].uniform_(-4, 4); x[:, 2:].uniform_(-2.5, 2.5)
    scn = SimpleNamespace(obstacle_centers=C[sid], obstacle_radii=R[sid], obstacle_active=A[sid], goal=G[sid])
    with torch.no_grad():
        h1 = h_fn(x, scn)
        h2 = net(system.observation(x, scn))
    assert float((h1 - h2).abs().max()) <= 1e-6
