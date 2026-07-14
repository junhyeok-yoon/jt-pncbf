"""Family labeler: the j=0 member equals the iteration-0 brake value bitwise; adding a transport gives
V_1 <= V_0 elementwise (family-retention monotonicity, for any transport)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import yaml

from src.envs.double_integrator import DoubleIntegrator
from src.frameworks.cpi.channel import make_cpi_h_fn
from src.frameworks.cpi.family import family_value
from src.frameworks.cpi.labels import label_streams, m0_value_raw, sample_scenes, stack_scene_obstacles

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/secured_data/v2.5.1/seed42/checkpoints/best.pt"
pytestmark = pytest.mark.skipif(not (CKPT.exists() and torch.cuda.is_available()),
                                reason="secured v2.5.1 seed42 checkpoint + cuda required")


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def test_family_j0_identity_and_monotonicity():
    cfg = _cfg(); dev = torch.device("cuda"); DT = torch.float32
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    scenes = sample_scenes(16, label_streams(cfg)["scene"], cfg)
    C, R, A, G = stack_scene_obstacles(scenes, dev, DT)
    torch.manual_seed(0)
    per = 64; S = len(scenes); sid = torch.arange(S, device=dev).repeat_interleave(per)
    x = torch.empty(S * per, 4, device=dev, dtype=DT); x[:, :2].uniform_(-4, 4); x[:, 2:].uniform_(-2.5, 2.5)
    dt = float(cfg["env"]["dt"])
    v0 = m0_value_raw(x, C[sid], R[sid], A[sid], system, cfg, dt)
    # j=0 identity: empty family == m_0
    vf0 = family_value(x, C[sid], R[sid], A[sid], G[sid], transports=[], system=system, config=cfg, t_bailout=40)
    assert torch.equal(vf0, v0), "empty-family value must equal the m_0 brake value bitwise"
    # monotonicity with a transport (cpi filter o policy). Use the secured policy as the transport policy.
    from src.common.control_net import ControlNet
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    pol = ControlNet(system.obs_dim, system, cfg).to(device=dev, dtype=DT).eval()
    h_fn = make_cpi_h_fn(CKPT, system)
    v1 = family_value(x, C[sid], R[sid], A[sid], G[sid], transports=[(pol, h_fn)], system=system, config=cfg, t_bailout=40)
    assert bool((v1 <= v0 + 1e-6).all()), "V_1 must be <= V_0 elementwise (family retention)"
