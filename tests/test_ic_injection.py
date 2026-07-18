"""v2.7.0 iteration-2 — cell-targeted IC oversampling (sample_scenes). Bit-parity at inject_frac=0,
correct tilted-cell injection at inject_frac>0, and the contamination-ban guard."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from src.envs.scene_init import sample_scenes, sample_train_scene, _cell_tilted

REPO = Path(__file__).resolve().parents[1]


def _cfg() -> dict[str, Any]:
    base = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    return _merge(base, exp)


def _merge(b: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
    m = dict(b)
    for k, v in o.items():
        m[k] = _merge(m[k], v) if isinstance(v, Mapping) and isinstance(m.get(k), Mapping) else v
    return m


def _sampler(cfg, system="quadrotor_planar"):
    return lambda rng: sample_train_scene(rng, cfg, system)


def _scene_key(sc):
    return (tuple(np.asarray(sc.start).ravel()), tuple(np.asarray(sc.goal).ravel()),
            float(sc.initial_attitude if sc.initial_attitude is not None else 0.0),
            tuple(np.asarray(sc.initial_velocity if sc.initial_velocity is not None else [0, 0]).ravel()),
            tuple(np.asarray(sc.obstacle_centers).ravel()))


def test_inject_frac_zero_is_bit_identical() -> None:
    cfg = _cfg(); s = _sampler(cfg); n = 128
    base = [s(np.random.default_rng(7)) for _ in range(n)] if False else None
    # baseline: the exact pre-change expression
    rng_a = np.random.default_rng(20270717)
    baseline = [s(rng_a) for _ in range(n)]
    rng_b = np.random.default_rng(20270717)
    got = sample_scenes(s, rng_b, n, inject_frac=0.0, system_name="quadrotor_planar")
    assert len(got) == n
    assert [_scene_key(a) for a in baseline] == [_scene_key(b) for b in got]
    # non-quadrotor is bit-parity even if inject_frac>0 (injection is quadrotor-scoped)
    rng_c = np.random.default_rng(11); rng_d = np.random.default_rng(11)
    di = _sampler(cfg, "double_integrator")
    b2 = [di(rng_c) for _ in range(n)]
    g2 = sample_scenes(di, rng_d, n, inject_frac=0.25, system_name="double_integrator")
    assert [_scene_key(a) for a in b2] == [_scene_key(b) for b in g2]


def test_inject_frac_ten_percent_fills_tilted_cell_with_fresh_scenes() -> None:
    cfg = _cfg(); s = _sampler(cfg); n = 200; frac = 0.10
    scenes = sample_scenes(s, np.random.default_rng(3), n, inject_frac=frac, system_name="quadrotor_planar")
    assert len(scenes) == n
    n_inject = round(frac * n)                                            # 20
    injected = scenes[:n_inject]
    # every injected episode is in the tilted cell |theta0| > pi/2, and is a fresh Scene (has an IC)
    assert all(_cell_tilted(sc) for sc in injected)
    assert all(abs(sc.initial_attitude) > np.pi / 2 for sc in injected)
    assert all(sc.initial_velocity is not None for sc in injected)
    # the injected block guarantees a >= inject_frac tilted-cell floor over the whole batch
    cell_frac = np.mean([_cell_tilted(sc) for sc in scenes])
    assert cell_frac >= frac                                             # floor from the injected block
    # injected scenes are distinct (fresh draws, not a repeated fixed IC)
    assert len({_scene_key(sc) for sc in injected}) > 1


def test_contamination_ban_guard_rejects_pool_like_objects() -> None:
    cfg = _cfg(); s = _sampler(cfg)
    scenes = [s(np.random.default_rng(1)) for _ in range(4)]

    class _FakePool:                                                      # a pool exposes `.scenes`
        def __init__(self, sc):
            self.scenes = sc

    with pytest.raises(TypeError):
        sample_scenes(scenes, np.random.default_rng(0), 4, inject_frac=0.1, system_name="quadrotor_planar")
    with pytest.raises(TypeError):
        sample_scenes(_FakePool(scenes), np.random.default_rng(0), 4, inject_frac=0.1,
                      system_name="quadrotor_planar")
