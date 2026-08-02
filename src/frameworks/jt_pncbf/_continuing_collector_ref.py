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

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene, sample_scenes
from src.frameworks.oc_pncbf.collection import TrajectoryRecord, TransitionRecord

Tensor = torch.Tensor


def _append_record_no_rebuild(buffer, states: Tensor, scene: Scene, h: Tensor) -> None:
    """Append one variable-length trajectory (record + transitions) to `buffer` WITHOUT rebuilding the tensor
    views — same record structure as OCReplayBuffer.append, but the caller batches one _rebuild per round so
    the cost is amortised. The buffer's _rebuild_tensor_views handles variable length via _pad_record_h."""
    tid = buffer._next_trajectory_id
    rec = TrajectoryRecord(trajectory_id=tid, scene=scene, states=states.detach().clone(),
                           h=h.detach().clone(), tail_obs=None)
    buffer._next_trajectory_id += 1
    buffer._trajectories.append(rec)
    for k in range(rec.states.shape[0] - 1):
        buffer._transitions.append(TransitionRecord(
            trajectory_id=tid, step_idx=k, scene=scene, state=rec.states[k], next_state=rec.states[k + 1],
            h=rec.h[k], next_h=rec.h[k + 1], obs=None))


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
    append_batches: int = 0         # number of append_batch calls (amortisation diagnostic)


@dataclass
class ContinuingStateRef:
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
        scenes = sample_scenes(scene_sampler, rng, n_rows, inject_frac=inject_frac, system_name=system_name)
        bs = batch_scenes(scenes, device=device, dtype=dtype)
        x = system.wrap_state(initial_states_from_batch(bs).to(dtype))
        p0 = system.position(x)                                       # [B,2]
        pos_ring = p0.unsqueeze(1).repeat(1, w + 1, 1).detach().clone()
        return ContinuingStateRef(
            system=system, scenes=scenes, x=x,
            t_row=np.zeros(n_rows, np.int64), ep_id=np.arange(n_rows, dtype=np.int64),
            goal_hit_countup=np.full(n_rows, -1, np.int64),
            pos_ring=pos_ring, gstep=0, next_ep_id=n_rows, device=device, dtype=dtype, w=w,
            cum=dict(rounds=0, r1=0, r2=0, r3=0, refills=0, segments=0, boundary=0, steps=0, seconds=0.0))


def _goals_tensor(system, scenes, device, dtype) -> Tensor:
    return torch.as_tensor(np.stack([np.asarray(s.goal, np.float64) for s in scenes]), device=device, dtype=dtype)


def advance_round_ref(
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
) -> ContinuingStats:
    system = state.system
    B = len(state.scenes)
    cc = config["collection"]["continuing"]
    k_hover = int(cc["k_hover"]); w = int(cc["stationary_window"])
    stat_thresh = float(cc["stationary_thresh"]); ep_timeout = int(cc["episode_timeout"])
    goal_radius = float(config["env"]["goal_radius"]); goal_speed = float(config["env"]["goal_speed_radius"])
    goal_angrate = float(config["env"].get("goal_angrate_radius", float("inf")))   # v2.8.0: angular reach term
    st = ContinuingStats()

    batched = batch_scenes(state.scenes, device=state.device, dtype=state.dtype)
    goals = _goals_tensor(system, state.scenes, state.device, state.dtype)
    scene_dirty = False

    round_states: list[Tensor] = [state.x.detach().clone()]           # index 0 = each row's segment start
    seg_start = np.zeros(B, np.int64)                                 # round_states index where segment started
    seg_first: list[Tensor | None] = [None] * B                      # fresh-IC override for a refill-started seg
    closed: list[tuple[Tensor, Scene]] = []                          # (states [L+1,D], scene) awaiting group-append

    def _slice(b: int, start: int, end: int) -> Tensor:
        if seg_first[b] is not None:
            rows = [seg_first[b]] + [round_states[t][b] for t in range(start + 1, end + 1)]
        else:
            rows = [round_states[t][b] for t in range(start, end + 1)]
        return torch.stack(rows, dim=0)                              # [L+1, D]

    for _ in range(round_length):
        if scene_dirty:
            batched = batch_scenes(state.scenes, device=state.device, dtype=state.dtype)
            goals = _goals_tensor(system, state.scenes, state.device, state.dtype)
            scene_dirty = False
        with torch.no_grad():
            u_safe = step_fn(state.x, batched)
            state.x = rk4_step(system, state.x, u_safe, dt).detach()
        st.steps += B
        round_states.append(state.x.detach().clone())               # trigger states preserved (pre-refill)
        idx = len(round_states) - 1
        state.gstep += 1
        state.t_row += 1

        with torch.no_grad():
            p = system.position(state.x)                            # [B,2]
            dist = torch.linalg.norm(p - goals, dim=1)              # [B]
            spd = system.speed(state.x)                            # [B]
            arate = system.angular_rate(state.x)                   # [B] (structural 0 on DI/unicycle)
            state.pos_ring[:, state.gstep % (w + 1)] = p
            lag = state.pos_ring[:, (state.gstep - w) % (w + 1)]
            disp = torch.linalg.norm(p - lag, dim=1)               # [B]
        goal_now = ((dist <= goal_radius) & (spd <= goal_speed) & (arate <= goal_angrate)).cpu().numpy()
        disp_np = disp.cpu().numpy()
        cu = state.goal_hit_countup
        cu[:] = np.where(cu >= 0, cu + 1, np.where(goal_now, 0, -1))
        r1 = cu >= k_hover
        r2 = (state.t_row > w) & (disp_np < stat_thresh)
        r3 = state.t_row >= ep_timeout
        trig = r1 | r2 | r3
        trig_rows = np.nonzero(trig)[0]
        if trig_rows.size:
            for b in trig_rows.tolist():
                closed.append((_slice(int(b), int(seg_start[b]), idx), state.scenes[b]))
                if r1[b]:
                    st.r1_count += 1
                elif r2[b]:
                    st.r2_count += 1
                else:
                    st.r3_count += 1
            st.refill_count += int(trig_rows.size)
            new_scenes = sample_scenes(scene_sampler, rng, int(trig_rows.size),
                                       inject_frac=inject_frac, system_name=system_name)
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

    # round boundary: close every row's current segment; carry the boundary state as next round's segment start
    end = len(round_states) - 1
    for b in range(B):
        seg = _slice(b, int(seg_start[b]), end)
        if seg.shape[0] >= 2:
            closed.append((seg, state.scenes[b]))
            st.boundary_segments += 1

    # group by length and append_batch (amortised rebuild); compute h per group with the batched barrier
    by_len: dict[int, list[tuple[Tensor, Scene]]] = {}
    for seg, scene in closed:
        if seg.shape[0] < 2:
            continue
        by_len.setdefault(seg.shape[0], []).append((seg, scene))
    for L, group in by_len.items():
        states_g = torch.stack([s for s, _ in group], dim=1)                       # [L, G, D]
        scenes_g = [sc for _, sc in group]
        bscene = batch_scenes(scenes_g, device=state.device, dtype=state.dtype)
        with torch.no_grad():
            h_g = h_batch_fn(states_g, bscene)                                     # [L, G] (batched barrier)
        for j, (seg, scene) in enumerate(group):
            _append_record_no_rebuild(buffer, seg, scene, h_g[:, j])
        st.append_batches += 1
        st.segments += len(group)
        st.unsafe_segments += int((h_g.max(dim=0).values > 0.0).sum().item())
    if by_len:                                                                     # one amortised rebuild/round
        buffer._evict_fifo()
        buffer._rebuild_tensor_views()
    if state.cum is not None:                                                       # cumulative instrumentation
        c = state.cum
        c["rounds"] += 1; c["r1"] += st.r1_count; c["r2"] += st.r2_count; c["r3"] += st.r3_count
        c["refills"] += st.refill_count; c["segments"] += st.segments
        c["boundary"] += st.boundary_segments; c["steps"] += st.steps
    return st
