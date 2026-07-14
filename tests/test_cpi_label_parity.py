"""CPI label parity: clip(V_raw) from the cpi label path equals maneuver_value REFERENCE restricted to
{m_0} within 1e-6, on a fixed 512-state batch across 8 fixed scenes. float32 (the label/deploy dtype).

No member-subset hook was added to maneuver_value (its public API already evaluates {m_0} via
lateral_js=()), so the additional default-path-unchanged assertion is not applicable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import yaml

from src.common.maneuver_value import maneuver_maxh
from src.envs.double_integrator import DoubleIntegrator
import src.frameworks.cpi.labels as L

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the cuda maneuver path")


def _cfg() -> dict[str, Any]:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def test_label_parity_m0():
    cfg = _cfg(); dev = torch.device("cuda"); DT = torch.float32
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    dtv = float(cfg["cpi"]["labels"]["dt_vm"])
    scenes = L.sample_scenes(8, L.label_streams(cfg)["scene"], cfg)
    C, R, A, _ = L.stack_scene_obstacles(scenes, dev, DT)
    torch.manual_seed(0)
    per = 64  # 8 scenes * 64 = 512 states
    S = len(scenes)
    sid = torch.arange(S, device=dev).repeat_interleave(per)
    x = torch.empty(S * per, 4, device=dev, dtype=DT)
    x[:, :2].uniform_(-4.0, 4.0)
    x[:, 2:].uniform_(-2.5, 2.5)
    vraw = L.m0_value_raw(x, C[sid], R[sid], A[sid], system, cfg, dtv)
    vclip = torch.clamp(vraw, -1.0, 1.0)
    maxdiff = 0.0
    for s in range(S):
        ref = maneuver_maxh(x[sid == s], scenes[s], system, cfg, lateral_js=(), dt_override=dtv, fast=False)[:, 0]
        maxdiff = max(maxdiff, float((vclip[sid == s] - ref).abs().max()))
    assert maxdiff <= 1e-6, f"clip(V_raw) vs maneuver {{m_0}} max|diff| {maxdiff:.3e}"
