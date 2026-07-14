"""v2.5.1 A1 cumulative family retention: build_transports constructs the FULL cumulative transport list
{T_1,...,T_k}, and the bail-out family value is monotone NON-INCREASING as transports are appended
(V over {m_0} >= V over {m_0,T_1} >= ... >= V over {m_0,T_1,...,T_k}). This is the property single-step
retention violated (family_k dropped T_1..T_{k-2}, admitting V_k > V_{k-1})."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from src.envs.double_integrator import DoubleIntegrator
from src.frameworks.cpi.family import build_transports, family_value
from src.frameworks.cpi.labels import label_streams, m0_value_raw, sample_scenes, stack_scene_obstacles

REPO = Path(__file__).resolve().parents[1]
REG_LIVE = REPO / "data/secured_data/v2.5.1/certificate_chain/family_registry.json"
# fixture fallback (cumulative transports) if the secured registry is absent — SECURED artifacts only, so
# the test never references the previous_runs archive. The monotonicity property holds for any transports;
# here V_hat_0/V_hat_1 certificates paired with two exact_backup policies.
_FALLBACK = [("data/secured_data/v2.5.1/seed42/checkpoints/best.pt", "data/secured_data/v2.5.1/exact_backup/pi_exactm0_seed42/best.pt"),
             ("data/secured_data/v2.5.1/certificate_chain/V_hat_1/best.pt", "data/secured_data/v2.5.1/exact_backup/pi_exactm0_seed99/best.pt")]


def _transport_pairs():
    if REG_LIVE.exists():
        reg = json.load(REG_LIVE.open())
        return [(REPO / reg[t]["vhat"], REPO / reg[t]["pi"]) for t in ("T_1", "T_2", "T_3")]
    return [(REPO / v, REPO / p) for v, p in _FALLBACK]


def _inputs_ready():
    return torch.cuda.is_available() and all(v.exists() and p.exists() for v, p in _transport_pairs())


pytestmark = pytest.mark.skipif(not _inputs_ready(), reason="cuda + transport checkpoints required")


def _cfg() -> dict[str, Any]:
    import yaml
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def test_build_transports_cumulative_and_monotone():
    cfg = _cfg(); dev = torch.device("cuda"); DT = torch.float32
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    pairs = _transport_pairs()
    transports = build_transports(pairs, system, cfg)
    assert len(transports) == len(pairs), "cumulative family must retain every transport T_1..T_k"

    scenes = sample_scenes(16, label_streams(cfg)["scene"], cfg)
    C, R, A, G = stack_scene_obstacles(scenes, dev, DT)
    torch.manual_seed(0)
    per = 48; S = len(scenes); sid = torch.arange(S, device=dev).repeat_interleave(per)
    x = torch.empty(S * per, 4, device=dev, dtype=DT)
    x[:, :2].uniform_(-4, 4); x[:, 2:].uniform_(-2.5, 2.5)
    dt = float(cfg["env"]["dt"])

    v0 = m0_value_raw(x, C[sid], R[sid], A[sid], system, cfg, dt)
    prev = v0
    for k in range(1, len(pairs) + 1):
        vk = family_value(x, C[sid], R[sid], A[sid], G[sid], transports[:k], system, cfg, t_bailout=40)
        assert bool((vk <= prev + 1e-6).all()), f"family_{k} must be <= family_{k-1} elementwise (cumulative retention)"
        prev = vk
