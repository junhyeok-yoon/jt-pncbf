"""CPI sampler: positions outside all active obstacles, speeds <= v_max, disjoint scene splits, and
seeded determinism of the first states."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

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


def _sample(cfg, n_scenes=16, per=60):
    """Sample scenes + uniform states from the master-seed streams (SeedSequence(master_seed).spawn(4))."""
    dev = torch.device("cpu"); DT = torch.float32
    streams = L.label_streams(cfg)
    scenes = L.sample_scenes(n_scenes, streams["scene"], cfg)
    obs = L.stack_scene_obstacles(scenes, dev, DT)
    v_max = float(cfg["env"]["bounds"]["double_integrator"]["v_max"]); world = float(cfg["env"]["world_lim"])
    pos, vel, sid = L.sample_uniform_states(obs, streams["state"], per, v_max, world, dev, DT)
    return obs, pos, vel, sid, v_max


def test_positions_outside_and_speed_bounded():
    cfg = _cfg()
    obs, pos, vel, sid, v_max = _sample(cfg)
    C, R, A, _ = obs
    outside = L.outside_all_obstacles(pos, C[sid], R[sid], A[sid])
    assert bool(outside.all()), "some sampled positions are inside an active obstacle"
    assert float(torch.linalg.norm(vel, dim=1).max()) <= v_max + 1e-6


def test_splits_pairwise_disjoint():
    cfg = _cfg()
    parts = L.split_by_scene(2000, cfg["cpi"]["split"]["fractions"], L.label_streams(cfg)["split"])
    names = list(parts)
    allids = set()
    total = 0
    for n in names:
        total += len(parts[n])
        allids |= parts[n]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert parts[names[i]].isdisjoint(parts[names[j]]), f"{names[i]} & {names[j]} overlap"
    assert len(allids) == total == 2000


def test_master_seed_determinism_first_states():
    """Two runs with the same master_seed (via SeedSequence.spawn) produce identical first 100 states."""
    cfg = _cfg()
    _, p1, v1, s1, _ = _sample(cfg)
    _, p2, v2, s2, _ = _sample(cfg)
    assert torch.equal(p1[:100], p2[:100]) and torch.equal(v1[:100], v2[:100]) and torch.equal(s1[:100], s2[:100])
