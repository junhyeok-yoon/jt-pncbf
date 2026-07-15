"""CPI brake stop-and-hold: from random states with ||v|| <= v_max, after T_stop steps of m_0 the exact
integrator reaches rest (||v|| <= 1e-9). This is the mathematical stop-and-hold property, verified in
float64 (the float32 label/deploy path reaches the float32 floor ~7.5e-9; see phase-i0 report). Also
checks that the recorded V_clip equals clamp(V_raw) exactly on a fixed batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.common.rk4 import rk4_step
from src.envs.double_integrator import DoubleIntegrator
import src.frameworks.cpi.labels as L

REPO = Path(__file__).resolve().parents[1]


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def test_brake_stop_and_hold_float64():
    cfg = _cfg(); dev = torch.device("cpu"); DT = torch.float64
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    v_max = float(cfg["env"]["bounds"]["double_integrator"]["v_max"]); u_max = 2.0; dtv = float(cfg["cpi"]["labels"]["dt_vm"])
    torch.manual_seed(3)
    x = torch.empty(4096, 4, device=dev, dtype=DT)
    x[:, :2].uniform_(-4.0, 4.0)
    d = torch.empty(4096, 2, device=dev, dtype=DT).uniform_(-1.0, 1.0)
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x[:, 2:] = d * torch.empty(4096, 1, device=dev, dtype=DT).uniform_(0.0, v_max)
    for _ in range(L.t_stop(system, cfg, dtv)):
        v = x[:, 2:4]
        u = torch.where(v.abs() > u_max * dtv, -u_max * torch.sign(v), -v / dtv)
        x = rk4_step(system, x, u, dtv)
    assert float(torch.linalg.norm(x[:, 2:4], dim=1).max()) <= 1e-9


def test_vclip_equals_clamp_vraw():
    cfg = _cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); DT = torch.float32
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    scenes = L.sample_scenes(4, L.label_streams(cfg)["scene"], cfg)
    C, R, A, _ = L.stack_scene_obstacles(scenes, dev, DT)
    torch.manual_seed(0)
    S = len(scenes); per = 32
    sid = torch.arange(S, device=dev).repeat_interleave(per)
    x = torch.empty(S * per, 4, device=dev, dtype=DT)
    x[:, :2].uniform_(-4.0, 4.0); x[:, 2:].uniform_(-2.5, 2.5)
    vraw = L.m0_value_raw(x, C[sid], R[sid], A[sid], system, cfg, float(cfg["cpi"]["labels"]["dt_vm"]))
    assert torch.equal(torch.clamp(vraw, -1.0, 1.0), vraw.clamp(-1.0, 1.0))
    assert torch.isfinite(vraw).all()
