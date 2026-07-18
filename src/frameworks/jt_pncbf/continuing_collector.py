"""v2.7.0 iteration-5 Track A — continuing-batch collector (vectorized + batched-append).

Persistent per-row rollout state carried ACROSS collection rounds. Each round advances every row by
`round_length` steps (= the existing collection rollout length, unchanged). Rows carry over: an episode cut
by a round boundary continues next round from the same state. A row is TRUNCATED and REFILLED (fresh
scene+IC, t_row=0, new episode_id) ONLY on:
  R1  goal criterion (eval goal test: dist<=goal_radius AND speed<=goal_speed_radius) + K_HOVER more steps
  R2  stationarity: net position displacement over W consecutive steps < STAT_THRESH metres
  R3  per-episode timeout: t_row == EPISODE_TIMEOUT (= eval max_steps)
NO collision rule — collision episodes keep the legacy pass-through behaviour end-to-end (those states are
deliberate value anchors; never masked, truncated, or relabelled).

Labels are PER-SEGMENT: a segment is the contiguous span within one round for one row between refills / round
boundaries; both a round-boundary cut and an R1/R2/R3 truncation close a segment. Each closed segment is
stored as its OWN buffer trajectory, so the existing value-target recursion (which bootstraps at the stored
trajectory's tail with the target net) closes per-segment — no recursion crosses an episode boundary.

Performance (05_code GPU-utilization): the per-step inner work is VECTORISED over the batch (goal / stationary
/ timeout masks are batch tensors; only the sparse set of TRIGGERED rows is handled in Python). Segments are
reconstructed at round end from a per-step batch-state list (`round_states`) plus the sparse fresh-IC overrides
and appended to the buffer GROUPED BY LENGTH via `append_batch`, so `_rebuild_tensor_views` is amortised to
~O(#distinct-segment-lengths) per round rather than O(#segments).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene, sample_scenes

Tensor = torch.Tensor


@dataclass
class ContinuingStats:
    r1_count: int = 0
    r2_count: int = 0
    r3_count: int = 0
    refill_count: int = 0
    segments: int = 0
    boundary_segments: int = 0
    unsafe_segments: int = 0
    steps: int = 0                  # total row-steps advanced this round (= B * round_length)
    append_batches: int = 0         # number of length-group append calls (amortisation diagnostic)
    timings: dict = field(default_factory=lambda: dict(  # permanent cheap per-phase wall-clock (seconds)
        sim=0.0, trigger=0.0, refill=0.0, segment=0.0, append=0.0, hlabel=0.0, other=0.0))


@dataclass
class ContinuingState:
    system: Any
    scenes: list[Scene]
    x: Tensor                                  # [B, state_dim] live per-row state
    t_row: np.ndarray                          # [B] step within the current episode
    ep_id: np.ndarray                          # [B] episode id per row
    goal_hit_countup: np.ndarray               # [B] steps since goal test first held (-1 if not held)
    pos_ring: Tensor                           # [B, W+1, 2] ring buffer of recent positions (for R2)
    gstep: int                                 # persistent global step counter (ring index)
    next_ep_id: int
    device: Any
    dtype: Any
    w: int                                     # stationary window W
    cum: dict = None                           # cumulative instrumentation across rounds (set in create)

    @staticmethod
    def create(system, scene_sampler, rng, n_rows, config, device, dtype, inject_frac=0.0, system_name=None):
        w = int(config["collection"]["continuing"]["stationary_window"])
        scenes = sample_scenes(scene_sampler, rng, n_rows, inject_frac=inject_frac, system_name=system_name,
                               config=config)
        bs = batch_scenes(scenes, device=device, dtype=dtype)
        x = system.wrap_state(initial_states_from_batch(bs).to(dtype))
        p0 = system.position(x)                                       # [B,2]
        pos_ring = p0.unsqueeze(1).repeat(1, w + 1, 1).detach().clone()
        return ContinuingState(
            system=system, scenes=scenes, x=x,
            t_row=np.zeros(n_rows, np.int64), ep_id=np.arange(n_rows, dtype=np.int64),
            goal_hit_countup=np.full(n_rows, -1, np.int64),
            pos_ring=pos_ring, gstep=0, next_ep_id=n_rows, device=device, dtype=dtype, w=w,
            cum=dict(rounds=0, r1=0, r2=0, r3=0, refills=0, segments=0, boundary=0, steps=0, seconds=0.0))


def _goals_tensor(system, scenes, device, dtype) -> Tensor:
    return torch.as_tensor(np.stack([np.asarray(s.goal, np.float64) for s in scenes]), device=device, dtype=dtype)


def advance_round(
    state: ContinuingState,
    *,
    round_length: int,
    step_fn: Callable[[Tensor, Any], Tensor],
    h_batch_fn: Callable[[Tensor, Any], Tensor],       # (states [L+1,G,D], batched_scene) -> h [L+1,G]
    scene_sampler,
    rng: np.random.Generator,
    config: Any,
    buffer: Any,
    dt: float,
    inject_frac: float = 0.0,
    system_name: str | None = None,
    sync: bool = False,
) -> ContinuingStats:
    system = state.system
    B = len(state.scenes)
    cc = config["collection"]["continuing"]
    k_hover = int(cc["k_hover"]); w = int(cc["stationary_window"])
    stat_thresh = float(cc["stationary_thresh"]); ep_timeout = int(cc["episode_timeout"])
    goal_radius = float(config["env"]["goal_radius"]); goal_speed = float(config["env"]["goal_speed_radius"])
    st = ContinuingStats()
    _do_sync = (sync and torch.cuda.is_available())

    def _t() -> float:
        if _do_sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    t_setup = time.perf_counter()
    batched = batch_scenes(state.scenes, device=state.device, dtype=state.dtype)
    goals = _goals_tensor(system, state.scenes, state.device, state.dtype)
    scene_dirty = False

    round_states: list[Tensor] = [state.x.detach().clone()]           # index 0 = each row's segment start
    seg_start = np.zeros(B, np.int64)                                 # round_states index where segment started
    seg_first: list[Tensor | None] = [None] * B                      # fresh-IC override for a refill-started seg
    # deferred segment descriptors: (row, start_idx, end_idx, fresh-IC override or None, scene). Segments are
    # reconstructed at round end by a SINGLE batched gather on `round_states` — no per-row Python stacking.
    closed: list[tuple[int, int, int, Tensor | None, Scene]] = []
    st.timings["other"] += time.perf_counter() - t_setup

    for _ in range(round_length):
        tr = _t()
        if scene_dirty:                                              # rebuild the full batched scene after refills
            batched = batch_scenes(state.scenes, device=state.device, dtype=state.dtype)
            goals = _goals_tensor(system, state.scenes, state.device, state.dtype)
            scene_dirty = False
        st.timings["refill"] += _t() - tr
        t0 = _t()
        with torch.no_grad():
            u_safe = step_fn(state.x, batched)
            state.x = rk4_step(system, state.x, u_safe, dt).detach()
        st.steps += B
        round_states.append(state.x.detach().clone())               # trigger states preserved (pre-refill)
        idx = len(round_states) - 1
        state.gstep += 1
        state.t_row += 1
        t1 = _t(); st.timings["sim"] += t1 - t0

        with torch.no_grad():
            p = system.position(state.x)                            # [B,2]
            dist = torch.linalg.norm(p - goals, dim=1)              # [B]
            spd = system.speed(state.x)                            # [B]
            state.pos_ring[:, state.gstep % (w + 1)] = p
            lag = state.pos_ring[:, (state.gstep - w) % (w + 1)]
            disp = torch.linalg.norm(p - lag, dim=1)               # [B]
        # one host<->device sync per step: stack the two boolean predicates and pull them together
        both = torch.stack([(dist <= goal_radius) & (spd <= goal_speed), disp < stat_thresh], dim=0).cpu().numpy()
        goal_now, disp_hit = both[0], both[1]
        cu = state.goal_hit_countup
        cu[:] = np.where(cu >= 0, cu + 1, np.where(goal_now, 0, -1))
        r1 = cu >= k_hover
        r2 = (state.t_row > w) & disp_hit
        r3 = state.t_row >= ep_timeout
        trig = r1 | r2 | r3
        trig_rows = np.nonzero(trig)[0]
        t2 = _t(); st.timings["trigger"] += t2 - t1
        if trig_rows.size:
            for b in trig_rows.tolist():
                closed.append((int(b), int(seg_start[b]), idx, seg_first[b], state.scenes[b]))
                if r1[b]:
                    st.r1_count += 1
                elif r2[b]:
                    st.r2_count += 1
                else:
                    st.r3_count += 1
            ts = _t(); st.timings["segment"] += ts - t2
            st.refill_count += int(trig_rows.size)
            new_scenes = sample_scenes(scene_sampler, rng, int(trig_rows.size),
                                       inject_frac=inject_frac, system_name=system_name, config=config)
            nb = batch_scenes(new_scenes, device=state.device, dtype=state.dtype)
            nx = system.wrap_state(initial_states_from_batch(nb).to(state.dtype))   # [n, D]
            np0 = system.position(nx)                                              # [n, 2]
            for j, b in enumerate(trig_rows.tolist()):
                state.scenes[b] = new_scenes[j]
                state.x[b] = nx[j]
                state.t_row[b] = 0
                cu[b] = -1
                seg_start[b] = idx
                seg_first[b] = nx[j].detach().clone()
                state.pos_ring[b, :] = np0[j]                                      # reset ring to fresh IC
            scene_dirty = True
            st.timings["refill"] += _t() - ts

    # round boundary: close every row's current segment; carry the boundary state as next round's segment start
    tb = _t()
    final = len(round_states) - 1
    for b in range(B):
        if final - int(seg_start[b]) + 1 >= 2:
            closed.append((b, int(seg_start[b]), final, seg_first[b], state.scenes[b]))
            st.boundary_segments += 1
    tb2 = _t(); st.timings["segment"] += tb2 - tb

    # reconstruct ALL segments with one batched gather on the stacked round states, grouped by length; compute h
    # per length-group with the batched barrier and append via the fast incremental path (one rebuild/group).
    tg = _t()
    RS = torch.stack(round_states, dim=0)                                          # [T+1, B, D] (one GPU op)
    by_len: dict[int, list[tuple[int, int, Tensor | None, Scene]]] = {}
    for b, start, end, ov, scene in closed:
        L = end - start + 1
        if L < 2:
            continue
        by_len.setdefault(L, []).append((b, start, ov, scene))
    st.timings["segment"] += _t() - tg
    unsafe_counts = []
    for L, group in by_len.items():
        tgg = _t()
        rows = torch.tensor([g[0] for g in group], device=state.device, dtype=torch.long)      # [G]
        starts = torch.tensor([g[1] for g in group], device=state.device, dtype=torch.long)    # [G]
        time_idx = starts.unsqueeze(0) + torch.arange(L, device=state.device).unsqueeze(1)     # [L, G]
        states_g = RS[time_idx, rows.unsqueeze(0).expand(L, -1)]                    # [L, G, D] (single gather)
        ov_rows = [k for k, g in enumerate(group) if g[2] is not None]             # refill-started segments
        if ov_rows:                                                                # first state = fresh IC
            states_g[0, ov_rows] = torch.stack([group[k][2] for k in ov_rows], dim=0)
        scenes_g = [g[3] for g in group]
        bscene = batch_scenes(scenes_g, device=state.device, dtype=state.dtype)
        st.timings["segment"] += _t() - tgg
        th = _t()
        with torch.no_grad():
            h_g = h_batch_fn(states_g, bscene)                                     # [L, G] (batched barrier)
        th2 = _t(); st.timings["hlabel"] += th2 - th
        # fast incremental append (flat-tensor _cat, variable length via _cat_pad_time) — no full rebuild
        buffer.append_batch(scenes_g, bscene, states_g.detach(), h_g.detach())
        st.append_batches += 1
        st.segments += len(group)
        unsafe_counts.append((h_g.max(dim=0).values > 0.0).sum())
        st.timings["append"] += _t() - th2
    if unsafe_counts:
        st.unsafe_segments += int(torch.stack(unsafe_counts).sum().item())         # one sync for the round
    if state.cum is not None:                                                       # cumulative instrumentation
        c = state.cum
        c["rounds"] += 1; c["r1"] += st.r1_count; c["r2"] += st.r2_count; c["r3"] += st.r3_count
        c["refills"] += st.refill_count; c["segments"] += st.segments
        c["boundary"] += st.boundary_segments; c["steps"] += st.steps
    return st
