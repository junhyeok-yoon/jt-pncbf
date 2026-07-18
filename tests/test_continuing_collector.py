"""v2.7.0 iteration-5 — continuing-batch collector (vectorized + batched append): legacy parity,
segment isolation, R1/R2/R3 firing, round-boundary carry-over, mid-round-refill carry-over."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.quadrotor_barrier import value_target_barrier
from src.envs.quadrotor_planar import QuadrotorPlanar
from src.envs.scene_init import Scene, sample_train_scene
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


def _scene(start, goal, att=0.0, vel=(0.0, 0.0)) -> Scene:
    c = np.zeros((12, 2)); r = np.zeros(12); a = np.zeros(12, bool)
    c[0] = [3.0, 3.0]; r[0] = 0.3; a[0] = True
    return Scene(obstacle_centers=c, obstacle_radii=r, obstacle_active=a,
                 start=np.asarray(start, float), goal=np.asarray(goal, float),
                 system="quadrotor_planar", mode="unit",
                 initial_velocity=np.asarray(vel, float), initial_attitude=float(att), initial_omega=0.0)


def _hbatch(system, cfg):
    def h(states_g, bscene):                                # [L, G, D] -> [L, G]
        return value_target_barrier(system, states_g, bscene, cfg)
    return h


def _state1(system, scene, cfg) -> ContinuingState:
    w = int(cfg["collection"]["continuing"]["stationary_window"])
    x0 = system.wrap_state(torch.tensor(
        [[*scene.start, scene.initial_attitude, *scene.initial_velocity, 0.0]], dtype=DT))
    p0 = system.position(x0)
    return ContinuingState(system=system, scenes=[scene], x=x0.clone(),
        t_row=np.zeros(1, np.int64), ep_id=np.zeros(1, np.int64), goal_hit_countup=np.full(1, -1, np.int64),
        pos_ring=p0.unsqueeze(1).repeat(1, w + 1, 1).detach().clone(), gstep=0, next_ep_id=1,
        device=torch.device("cpu"), dtype=DT, w=w)


def test_t1_legacy_default() -> None:
    # committed default is legacy (bit-parity, exercised by the existing collection smoke tests).
    assert _cfg()["collection"]["collector"] == "legacy"


def test_t3_R1_R2_R3_fire() -> None:
    cfg0 = _cfg(); system = QuadrotorPlanar(cfg0); mg = cfg0["env"]["quadrotor_planar"]["gravity"]

    def _run(scene, step_fn, cont, round_len=20):
        cfg = _merge(cfg0, {"collection": {"continuing": {**cfg0["collection"]["continuing"], **cont}}})
        buf = OCReplayBuffer(capacity=100000)
        stt = _state1(system, scene, cfg)
        return advance_round(stt, round_length=round_len, step_fn=step_fn, h_batch_fn=_hbatch(system, cfg),
                             scene_sampler=lambda rng: _scene([-1, -1], [1, 1]), rng=np.random.default_rng(0),
                             config=cfg, buffer=buf, dt=cfg["env"]["dt"], system_name="quadrotor_planar")

    hover = lambda x, bs: torch.tensor([[mg, 0.0]], dtype=DT)
    fall = lambda x, bs: torch.zeros(1, 2, dtype=DT)
    s1 = _run(_scene([0.0, 0.0], [0.0, 0.0]), hover,
              {"k_hover": 3, "stationary_window": 50, "episode_timeout": 200}, round_len=5)
    assert s1.r1_count == 1 and s1.r2_count == 0 and s1.r3_count == 0
    s2 = _run(_scene([2.0, 2.0], [0.0, 0.0]), hover,
              {"k_hover": 50, "stationary_window": 5, "episode_timeout": 200}, round_len=7)
    assert s2.r2_count == 1 and s2.r1_count == 0 and s2.r3_count == 0
    s3 = _run(_scene([2.0, 2.0], [0.0, -8.0], vel=(0.5, -0.5)), fall,
              {"k_hover": 50, "stationary_window": 50, "episode_timeout": 8}, round_len=9)
    assert s3.r3_count == 1 and s3.r1_count == 0 and s3.r2_count == 0


def test_t2_segment_isolation() -> None:
    cfg = _merge(_cfg(), {"collection": {"continuing": {"episode_timeout": 8}}})
    system = QuadrotorPlanar(cfg); buf = OCReplayBuffer(capacity=100000)
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")
    stt = ContinuingState.create(system, sampler, np.random.default_rng(1), 4, cfg, torch.device("cpu"), DT,
                                 system_name="quadrotor_planar")
    fall = lambda x, bs: torch.zeros(x.shape[0], 2, dtype=DT)
    advance_round(stt, round_length=10, step_fn=fall, h_batch_fn=_hbatch(system, cfg), scene_sampler=sampler,
                  rng=np.random.default_rng(2), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                  system_name="quadrotor_planar")
    n_after_first = len(buf._trajectories)
    h_snapshot = [t.h.clone() for t in buf._trajectories]
    advance_round(stt, round_length=10, step_fn=fall, h_batch_fn=_hbatch(system, cfg), scene_sampler=sampler,
                  rng=np.random.default_rng(3), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                  system_name="quadrotor_planar")
    for i in range(n_after_first):                          # first round's stored labels unchanged by round 2
        assert torch.equal(buf._trajectories[i].h, h_snapshot[i])
    assert len(buf._trajectories) > n_after_first


def test_t4_and_t5_carryover() -> None:
    cfg = _merge(_cfg(), {"collection": {"continuing": {"episode_timeout": 200}}})  # no R3 within the round
    system = QuadrotorPlanar(cfg); buf = OCReplayBuffer(capacity=100000)
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")
    stt = ContinuingState.create(system, sampler, np.random.default_rng(4), 4, cfg, torch.device("cpu"), DT,
                                 system_name="quadrotor_planar")
    fall = lambda x, bs: torch.zeros(x.shape[0], 2, dtype=DT)   # never goal/stationary/timeout -> spans rounds
    ep_before = stt.ep_id.copy()
    s0 = advance_round(stt, round_length=15, step_fn=fall, h_batch_fn=_hbatch(system, cfg), scene_sampler=sampler,
                       rng=np.random.default_rng(5), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                       system_name="quadrotor_planar")
    # t4: episodes span the round boundary — no refill, t_row advanced by round length, boundary segments stored,
    # same episode continues (ep_id unchanged).
    assert s0.refill_count == 0 and s0.boundary_segments == 4
    assert np.array_equal(stt.ep_id, ep_before) and np.all(stt.t_row == 15)
    n0 = len(buf._trajectories)
    advance_round(stt, round_length=15, step_fn=fall, h_batch_fn=_hbatch(system, cfg), scene_sampler=sampler,
                  rng=np.random.default_rng(6), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                  system_name="quadrotor_planar")
    # t5: the still-incomplete episode's later-time states reach the buffer in round 2's boundary segment.
    assert np.all(stt.t_row == 30) and len(buf._trajectories) == n0 + 4


def test_semantic_equivalence_ref_vs_new() -> None:
    # The perf fix (incremental append + _cat_pad_time) must produce IDENTICAL buffer contents to the frozen
    # reference implementation (full-rebuild path). Deterministic nominal-LQR step, small timeout to force
    # refills across rounds; large capacity so no FIFO eviction confounds the comparison.
    from src.frameworks.jt_pncbf._continuing_collector_ref import ContinuingStateRef, advance_round_ref
    cfg = _merge(_cfg(), {"collection": {"continuing": {"episode_timeout": 12, "stationary_window": 8}}})
    system = QuadrotorPlanar(cfg)
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")
    mg = cfg["env"]["quadrotor_planar"]["gravity"]

    def _lqr(x, bs):                                        # deterministic (no noise) -> reproducible
        goal = torch.as_tensor(bs.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, goal)

    def _run(state_cls, adv):
        buf = OCReplayBuffer(capacity=10_000_000)
        st = state_cls.create(system, sampler, np.random.default_rng(123), 8, cfg, torch.device("cpu"), DT,
                              system_name="quadrotor_planar")
        for _ in range(4):
            adv(st, round_length=20, step_fn=_lqr, h_batch_fn=_hbatch(system, cfg), scene_sampler=sampler,
                rng=np.random.default_rng(999), config=cfg, buffer=buf, dt=cfg["env"]["dt"],
                system_name="quadrotor_planar")
        return buf

    ref = _run(ContinuingStateRef, advance_round_ref)
    new = _run(ContinuingState, advance_round)
    assert len(ref._trajectories) == len(new._trajectories)
    for a, b in zip(ref._trajectories, new._trajectories):
        assert torch.equal(a.states, b.states) and torch.equal(a.h, b.h)
        assert np.array_equal(np.asarray(a.scene.goal), np.asarray(b.scene.goal))
    # derived views (padded h-sequence + tails) must match too
    assert torch.allclose(ref._tensor_traj_h, new._tensor_traj_h, atol=1e-6)
    assert torch.equal(ref._tensor_traj_tail_states, new._tensor_traj_tail_states)
    assert torch.equal(ref._tensor_states, new._tensor_states)
