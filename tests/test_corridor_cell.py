"""v2.7.1 M1 — corridor-cell STATE injection (changes.md §4): inject_frac=0 bit-parity, cell-distribution
ranges, contamination guard, and R1-R3 / per-segment labeling parity for injected episodes (no special-casing)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.quadrotor_barrier import value_target_barrier
from src.envs.quadrotor_planar import QuadrotorPlanar
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import sample_cell_state_scene, sample_scenes, sample_train_scene
from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
from src.frameworks.oc_pncbf.collection import OCReplayBuffer

REPO = Path(__file__).resolve().parents[1]
DT = torch.float32


def _cfg() -> dict[str, Any]:
    base = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    exp = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    return _merge(base, exp)


def _merge(b: Mapping[str, Any], o: Mapping[str, Any]) -> dict[str, Any]:
    m = dict(b)
    for k, v in o.items():
        m[k] = _merge(m[k], v) if isinstance(v, Mapping) and isinstance(m.get(k), Mapping) else v
    return m


def _key(sc) -> tuple:
    return (tuple(np.asarray(sc.start).ravel()), tuple(np.asarray(sc.goal).ravel()),
            tuple(np.asarray(sc.obstacle_centers).ravel()))


def _sampler(cfg):
    return lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")


def test_t1_inject_frac_zero_bit_parity() -> None:
    cfg = _cfg(); s = _sampler(cfg); n = 40
    base = [_key(s(rng)) for rng in [np.random.default_rng(9)] for _ in range(n)]  # deterministic driver below
    b = [s(np.random.default_rng(9)) for _ in range(1)]  # noqa: F841 (shape check only)
    # inject_frac=0 with config must reproduce the plain fresh-sampler draw (rng consumption identical)
    got0 = sample_scenes(s, np.random.default_rng(9), n, inject_frac=0.0, system_name="quadrotor_planar", config=cfg)
    ref = [s(r) for r in [np.random.default_rng(9)] for _ in range(n)]
    assert [_key(a) for a in got0] == [_key(a) for a in ref]
    # config passed but inject_frac=0 == config=None (both bit-parity)
    gotN = sample_scenes(s, np.random.default_rng(9), n, inject_frac=0.0, system_name="quadrotor_planar", config=None)
    assert [_key(a) for a in got0] == [_key(a) for a in gotN]


def test_t2_cell_distribution_ranges() -> None:
    cfg = _cfg(); s = _sampler(cfg); h_scale = float(cfg["env"]["h_scale"]); eps = 1e-6
    rng = np.random.default_rng(11)
    inj = [sample_cell_state_scene(s(rng), rng, cfg) for _ in range(300)]
    n_moved = 0
    for sc in inj:
        p = np.asarray(sc.start, float); th = float(sc.initial_attitude); v = np.asarray(sc.initial_velocity, float)
        cen = np.asarray(sc.obstacle_centers, float); rad = np.asarray(sc.obstacle_radii, float)
        act = np.asarray(sc.obstacle_active, bool)
        if not act.any():
            continue
        n_moved += 1
        surf = np.linalg.norm(cen[act] - p, axis=1) - rad[act]
        assert (surf >= -1e-6).all(), "position inside an obstacle (rejection failed)"
        athe = abs((th + np.pi) % (2 * np.pi) - np.pi)
        assert np.pi / 2 - eps <= athe <= np.pi + eps, f"|theta|={athe} out of [pi/2,pi]"
        spd = np.linalg.norm(v)
        assert 0.5 - eps <= spd <= 1.5 + eps, f"speed {spd} out of [0.5,1.5]"
        # exists an active obstacle at surface dist in [0.02, h_scale] toward which v points within 60 deg
        found = False
        for j in np.nonzero(act)[0]:
            d = float(np.linalg.norm(cen[j] - p) - rad[j])
            if 0.02 - 1e-6 <= d <= h_scale + 1e-6:
                bear = np.arctan2(cen[j][1] - p[1], cen[j][0] - p[0])
                vang = np.arctan2(v[1], v[0])
                if abs((vang - bear + np.pi) % (2 * np.pi) - np.pi) <= np.pi / 3 + 1e-6:
                    found = True; break
        assert found, "no active obstacle satisfies d in [0.02,h_scale] + 60deg inward cone"
    assert n_moved >= 250


def test_t3_contamination_guard() -> None:
    cfg = _cfg()
    class _Pool:  # pool-like object carrying .scenes
        scenes = [1, 2, 3]
    for bad in ([_cfg], [1, 2], _Pool()):
        try:
            sample_scenes(bad, np.random.default_rng(0), 4, inject_frac=0.10,
                          system_name="quadrotor_planar", config=cfg)
            assert False, "contamination guard did not fire"
        except TypeError:
            pass


def test_t4_injected_follow_R1R3_and_segment_labels() -> None:
    cfg = _merge(_cfg(), {"collection": {"continuing": {"episode_timeout": 40}}})
    system = QuadrotorPlanar(cfg); buf = OCReplayBuffer(capacity=1_000_000)
    s = _sampler(cfg)

    def hbatch(states_g, bscene):
        return value_target_barrier(system, states_g, bscene, cfg)

    fall = lambda x, bs: torch.zeros(x.shape[0], 2, dtype=DT)   # spans/triggers like any episode
    stt = ContinuingState.create(system, s, np.random.default_rng(4), 16, cfg, torch.device("cpu"), DT,
                                 inject_frac=0.10, system_name="quadrotor_planar")
    # injected rows are ordinary rows: create + advance with inject_frac>0, no special-casing anywhere
    for i in range(3):
        advance_round(stt, round_length=30, step_fn=fall, h_batch_fn=hbatch, scene_sampler=s,
                      rng=np.random.default_rng(5 + i), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                      inject_frac=0.10, system_name="quadrotor_planar")
    assert len(buf._trajectories) > 0
    # per-segment labels are the SAME value_target_barrier recursion (no label surgery on injected states)
    maxerr = 0.0
    for tr in buf._trajectories[:8]:
        S = tr.states.unsqueeze(1)
        bs = batch_scenes([tr.scene], device=torch.device("cpu"), dtype=DT)
        hre = value_target_barrier(system, S, bs, cfg).reshape(-1)
        maxerr = max(maxerr, float((hre - tr.h).abs().max()))
    assert maxerr < 1e-5, f"stored labels diverge from value_target_barrier recompute (maxerr={maxerr})"
