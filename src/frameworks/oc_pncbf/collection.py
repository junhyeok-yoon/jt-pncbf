from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.common.signed_h import signed_h
from src.common.system import System
from src.envs.scene_batch import BatchedScene, batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene
from src.eval.rollout import rollout_lqr


Tensor = torch.Tensor


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: int
    scene: Scene
    states: Tensor
    h: Tensor
    tail_obs: Tensor | None = None


@dataclass(frozen=True)
class TransitionRecord:
    trajectory_id: int
    step_idx: int
    scene: Scene
    state: Tensor
    next_state: Tensor
    h: Tensor
    next_h: Tensor
    obs: Tensor | None = None


@dataclass(frozen=True)
class AnchorBatch:
    states: Tensor
    h: Tensor
    trajectory_indices: Tensor
    step_indices: Tensor
    collision_window: Tensor
    near_unsafe: Tensor
    future_h: Tensor


@dataclass(frozen=True)
class AnchorSets:
    unsafe: AnchorBatch
    safe: AnchorBatch


@dataclass(frozen=True)
class ValueMinibatch:
    transition_states: Tensor
    transition_next_states: Tensor
    transition_h: Tensor
    transition_next_h: Tensor
    transition_scenes: list[Scene]
    unsafe_states: Tensor
    safe_states: Tensor
    source_indices: dict[str, np.ndarray]


@dataclass(frozen=True)
class TensorTransitionBatch:
    states: Tensor
    next_states: Tensor
    h: Tensor
    next_h: Tensor
    trajectory_ids: Tensor
    trajectory_rows: Tensor
    step_indices: Tensor
    scene: BatchedScene
    h_sequence: Tensor
    tail_states: Tensor
    tail_scene: BatchedScene


class OCReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self._trajectories: list[TrajectoryRecord] = []
        self._transitions: list[TransitionRecord] = []
        self._tensor_states: Tensor | None = None
        self._tensor_next_states: Tensor | None = None
        self._tensor_h: Tensor | None = None
        self._tensor_next_h: Tensor | None = None
        self._tensor_trajectory_ids: Tensor | None = None
        self._tensor_step_indices: Tensor | None = None
        self._tensor_centers: Tensor | None = None
        self._tensor_radii: Tensor | None = None
        self._tensor_active: Tensor | None = None
        self._tensor_goals: Tensor | None = None
        self._tensor_traj_ids_unique: Tensor | None = None
        self._tensor_traj_h: Tensor | None = None
        self._tensor_traj_tail_states: Tensor | None = None
        self._tensor_traj_centers: Tensor | None = None
        self._tensor_traj_radii: Tensor | None = None
        self._tensor_traj_active: Tensor | None = None
        self._tensor_traj_goals: Tensor | None = None
        self._next_trajectory_id = 0

    @property
    def traj_view(self) -> list[TrajectoryRecord]:
        return list(self._trajectories)

    @property
    def transition_view(self) -> list[TransitionRecord]:
        if self._transitions:
            return list(self._transitions)
        return [
            TransitionRecord(
                trajectory_id=trajectory.trajectory_id,
                step_idx=step_idx,
                scene=trajectory.scene,
                state=trajectory.states[step_idx],
                next_state=trajectory.states[step_idx + 1],
                h=trajectory.h[step_idx],
                next_h=trajectory.h[step_idx + 1],
            )
            for trajectory in self._trajectories
            for step_idx in range(trajectory.states.shape[0] - 1)
        ]

    def __len__(self) -> int:
        if self._tensor_states is not None:
            return int(self._tensor_states.shape[0])
        if self._transitions:
            return len(self._transitions)
        return sum(max(0, trajectory.states.shape[0] - 1) for trajectory in self._trajectories)

    def append(
        self,
        scene: Scene,
        states: Tensor,
        h: Tensor,
        observations: Tensor | None = None,
        tail_obs: Tensor | None = None,
    ) -> TrajectoryRecord:
        if states.ndim != 2:
            raise ValueError(f"states must have shape [T+1, state_dim], got {states.shape}")
        if h.shape != (states.shape[0],):
            raise ValueError(f"h shape {tuple(h.shape)} does not match states.")
        if states.shape[0] < 2:
            raise ValueError("trajectory must contain at least one transition.")
        if observations is not None and observations.shape[0] != states.shape[0] - 1:
            raise ValueError(
                "observations must have one row per transition, got "
                f"{observations.shape[0]} for {states.shape[0] - 1} transitions."
            )
        if tail_obs is not None and tail_obs.ndim != 1:
            raise ValueError(f"tail_obs must be rank-1, got {tail_obs.shape}.")
        if not torch.isfinite(states).all() or not torch.isfinite(h).all():
            raise FloatingPointError("trajectory contains NaN or Inf.")

        record = TrajectoryRecord(
            trajectory_id=self._next_trajectory_id,
            scene=scene,
            states=states.detach().clone(),
            h=h.detach().clone(),
            tail_obs=None if tail_obs is None else tail_obs.detach().clone(),
        )
        self._next_trajectory_id += 1
        self._trajectories.append(record)
        for step_idx in range(states.shape[0] - 1):
            self._transitions.append(
                TransitionRecord(
                    trajectory_id=record.trajectory_id,
                    step_idx=step_idx,
                    scene=scene,
                    state=record.states[step_idx],
                    next_state=record.states[step_idx + 1],
                    h=record.h[step_idx],
                    next_h=record.h[step_idx + 1],
                    obs=None
                    if observations is None
                    else observations[step_idx].detach().clone(),
                )
            )
        self._evict_fifo()
        self._rebuild_tensor_views()
        return record

    def append_batch(
        self,
        scenes: list[Scene],
        batched_scene: BatchedScene,
        states: Tensor,
        h: Tensor,
    ) -> list[TrajectoryRecord]:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [T+1, B, D], got {states.shape}.")
        if h.shape != states.shape[:2]:
            raise ValueError(f"h shape {tuple(h.shape)} does not match states.")
        if len(scenes) != states.shape[1]:
            raise ValueError("Number of scenes must match state batch size.")

        records = []
        for batch_idx, scene in enumerate(scenes):
            trajectory_id = self._next_trajectory_id
            record = TrajectoryRecord(
                trajectory_id=trajectory_id,
                scene=scene,
                states=states[:, batch_idx, :].detach().clone(),
                h=h[:, batch_idx].detach().clone(),
                tail_obs=None,
            )
            self._next_trajectory_id += 1
            self._trajectories.append(record)
            records.append(record)

        if records:
            self._append_tensor_batch(
                batched_scene,
                states,
                h,
                [r.trajectory_id for r in records],
            )
        self._evict_fifo()
        if self._tensor_states is None:
            self._rebuild_tensor_views()
        return records

    def sample_tensor_batch(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> TensorTransitionBatch:
        if self._tensor_states is None:
            self._rebuild_tensor_views()
        if self._tensor_states is None or self._tensor_states.shape[0] == 0:
            raise ValueError("Cannot sample from an empty transition buffer.")

        device = self._tensor_states.device
        indices = torch.randint(
            self._tensor_states.shape[0],
            (batch_size,),
            generator=generator,
            device=device,
        )
        transition_scene = BatchedScene(
            obstacle_centers=self._tensor_centers.index_select(0, indices),
            obstacle_radii=self._tensor_radii.index_select(0, indices),
            obstacle_active=self._tensor_active.index_select(0, indices),
            start=torch.zeros((batch_size, 2), dtype=self._tensor_states.dtype, device=device),
            goal=self._tensor_goals.index_select(0, indices),
            system=self._trajectories[0].scene.system,
            mode=self._trajectories[0].scene.mode,
        )
        trajectory_ids = self._tensor_trajectory_ids.index_select(0, indices)
        if (
            self._tensor_traj_ids_unique is None
            or self._tensor_traj_h is None
        ):
            raise ValueError("Trajectory tensor cache is unavailable for target assembly.")
        trajectory_rows = torch.searchsorted(self._tensor_traj_ids_unique, trajectory_ids)
        if not torch.equal(
            self._tensor_traj_ids_unique.index_select(0, trajectory_rows),
            trajectory_ids,
        ):
            raise ValueError("Transition trajectory ids are inconsistent with trajectory cache.")
        tail_scene = BatchedScene(
            obstacle_centers=self._tensor_traj_centers.index_select(0, trajectory_rows),
            obstacle_radii=self._tensor_traj_radii.index_select(0, trajectory_rows),
            obstacle_active=self._tensor_traj_active.index_select(0, trajectory_rows),
            start=torch.zeros((batch_size, 2), dtype=self._tensor_states.dtype, device=device),
            goal=self._tensor_traj_goals.index_select(0, trajectory_rows),
            system=self._trajectories[0].scene.system,
            mode=self._trajectories[0].scene.mode,
        )

        return TensorTransitionBatch(
            states=self._tensor_states.index_select(0, indices),
            next_states=self._tensor_next_states.index_select(0, indices),
            h=self._tensor_h.index_select(0, indices),
            next_h=self._tensor_next_h.index_select(0, indices),
            trajectory_ids=trajectory_ids,
            trajectory_rows=trajectory_rows,
            step_indices=self._tensor_step_indices.index_select(0, indices),
            scene=transition_scene,
            h_sequence=self._tensor_traj_h.index_select(0, trajectory_rows).transpose(0, 1),
            tail_states=self._tensor_traj_tail_states.index_select(0, trajectory_rows),
            tail_scene=tail_scene,
        )

    def _evict_fifo(self) -> None:
        if len(self._trajectories) <= self.capacity:
            return

        evicted_ids = []
        evicted_transition_count = 0
        original_trajectory_count = len(self._trajectories)
        cache_can_slice = (
            self._tensor_states is not None
            and self._tensor_traj_ids_unique is not None
            and self._tensor_traj_ids_unique.shape[0] == original_trajectory_count
        )
        while len(self._trajectories) > self.capacity:
            evicted = self._trajectories.pop(0)
            evicted_ids.append(evicted.trajectory_id)
            n_removed = max(0, evicted.states.shape[0] - 1)
            evicted_transition_count += n_removed

        if evicted_ids:
            if self._transitions:
                evicted_set = set(evicted_ids)
                self._transitions = [
                    transition
                    for transition in self._transitions
                    if transition.trajectory_id not in evicted_set
                ]
            if cache_can_slice:
                self._slice_tensor_views(
                    evicted_transition_count,
                    len(evicted_ids),
                )
            else:
                self._rebuild_tensor_views()

    def _slice_tensor_views(
        self,
        transition_count: int,
        trajectory_count: int,
    ) -> None:
        self._tensor_states = _slice_front(self._tensor_states, transition_count)
        self._tensor_next_states = _slice_front(self._tensor_next_states, transition_count)
        self._tensor_h = _slice_front(self._tensor_h, transition_count)
        self._tensor_next_h = _slice_front(self._tensor_next_h, transition_count)
        self._tensor_trajectory_ids = _slice_front(
            self._tensor_trajectory_ids,
            transition_count,
        )
        self._tensor_step_indices = _slice_front(self._tensor_step_indices, transition_count)
        self._tensor_centers = _slice_front(self._tensor_centers, transition_count)
        self._tensor_radii = _slice_front(self._tensor_radii, transition_count)
        self._tensor_active = _slice_front(self._tensor_active, transition_count)
        self._tensor_goals = _slice_front(self._tensor_goals, transition_count)

        self._tensor_traj_ids_unique = _slice_front(
            self._tensor_traj_ids_unique,
            trajectory_count,
        )
        self._tensor_traj_h = _slice_front(self._tensor_traj_h, trajectory_count)
        self._tensor_traj_tail_states = _slice_front(
            self._tensor_traj_tail_states,
            trajectory_count,
        )
        self._tensor_traj_centers = _slice_front(
            self._tensor_traj_centers,
            trajectory_count,
        )
        self._tensor_traj_radii = _slice_front(
            self._tensor_traj_radii,
            trajectory_count,
        )
        self._tensor_traj_active = _slice_front(
            self._tensor_traj_active,
            trajectory_count,
        )
        self._tensor_traj_goals = _slice_front(
            self._tensor_traj_goals,
            trajectory_count,
        )

    def _append_tensor_batch(
        self,
        scene: BatchedScene,
        states: Tensor,
        h: Tensor,
        trajectory_ids: list[int],
    ) -> None:
        n_steps = states.shape[0] - 1
        batch_size = states.shape[1]
        flat_states = states[:-1].permute(1, 0, 2).reshape(batch_size * n_steps, -1)
        flat_next_states = states[1:].permute(1, 0, 2).reshape(batch_size * n_steps, -1)
        flat_h = h[:-1].transpose(0, 1).reshape(batch_size * n_steps)
        flat_next_h = h[1:].transpose(0, 1).reshape(batch_size * n_steps)
        traj_ids = torch.as_tensor(
            trajectory_ids,
            dtype=torch.long,
            device=states.device,
        ).repeat_interleave(n_steps)
        step_ids = torch.arange(n_steps, dtype=torch.long, device=states.device)
        step_ids = step_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        centers = scene.obstacle_centers.repeat_interleave(n_steps, dim=0)
        radii = scene.obstacle_radii.repeat_interleave(n_steps, dim=0)
        active = scene.obstacle_active.repeat_interleave(n_steps, dim=0)
        goals = scene.goal.repeat_interleave(n_steps, dim=0)

        self._tensor_states = _cat_optional(self._tensor_states, flat_states)
        self._tensor_next_states = _cat_optional(self._tensor_next_states, flat_next_states)
        self._tensor_h = _cat_optional(self._tensor_h, flat_h)
        self._tensor_next_h = _cat_optional(self._tensor_next_h, flat_next_h)
        self._tensor_trajectory_ids = _cat_optional(self._tensor_trajectory_ids, traj_ids)
        self._tensor_step_indices = _cat_optional(self._tensor_step_indices, step_ids)
        self._tensor_centers = _cat_optional(self._tensor_centers, centers)
        self._tensor_radii = _cat_optional(self._tensor_radii, radii)
        self._tensor_active = _cat_optional(self._tensor_active, active)
        self._tensor_goals = _cat_optional(self._tensor_goals, goals)
        self._tensor_traj_ids_unique = _cat_optional(
            self._tensor_traj_ids_unique,
            torch.as_tensor(trajectory_ids, dtype=torch.long, device=states.device),
        )
        self._tensor_traj_h = _cat_optional(
            self._tensor_traj_h,
            h[:-1].transpose(0, 1).contiguous(),
        )
        self._tensor_traj_tail_states = _cat_optional(
            self._tensor_traj_tail_states,
            states[-1],
        )
        self._tensor_traj_centers = _cat_optional(
            self._tensor_traj_centers,
            scene.obstacle_centers,
        )
        self._tensor_traj_radii = _cat_optional(
            self._tensor_traj_radii,
            scene.obstacle_radii,
        )
        self._tensor_traj_active = _cat_optional(
            self._tensor_traj_active,
            scene.obstacle_active,
        )
        self._tensor_traj_goals = _cat_optional(self._tensor_traj_goals, scene.goal)

    def _rebuild_tensor_views(self) -> None:
        if not self._transitions:
            self._tensor_states = None
            self._tensor_next_states = None
            self._tensor_h = None
            self._tensor_next_h = None
            self._tensor_trajectory_ids = None
            self._tensor_step_indices = None
            self._tensor_centers = None
            self._tensor_radii = None
            self._tensor_active = None
            self._tensor_goals = None
            self._tensor_traj_ids_unique = None
            self._tensor_traj_h = None
            self._tensor_traj_tail_states = None
            self._tensor_traj_centers = None
            self._tensor_traj_radii = None
            self._tensor_traj_active = None
            self._tensor_traj_goals = None
            return
        states = torch.stack([transition.state for transition in self._transitions], dim=0)
        device = states.device
        dtype = states.dtype
        self._tensor_states = states
        self._tensor_next_states = torch.stack(
            [transition.next_state for transition in self._transitions],
            dim=0,
        )
        self._tensor_h = torch.stack([transition.h for transition in self._transitions], dim=0)
        self._tensor_next_h = torch.stack(
            [transition.next_h for transition in self._transitions],
            dim=0,
        )
        self._tensor_trajectory_ids = torch.as_tensor(
            [transition.trajectory_id for transition in self._transitions],
            dtype=torch.long,
            device=device,
        )
        self._tensor_step_indices = torch.as_tensor(
            [transition.step_idx for transition in self._transitions],
            dtype=torch.long,
            device=device,
        )
        self._tensor_centers = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_centers, dtype=dtype, device=device)
                for t in self._transitions
            ],
            dim=0,
        )

        self._tensor_traj_ids_unique = torch.as_tensor(
            [trajectory.trajectory_id for trajectory in self._trajectories],
            dtype=torch.long,
            device=device,
        )
        self._tensor_traj_h = _pad_record_h(self._trajectories, dtype, device)
        self._tensor_traj_tail_states = torch.stack(
            [trajectory.states[-1] for trajectory in self._trajectories],
            dim=0,
        )
        self._tensor_traj_centers = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_centers, dtype=dtype, device=device)
                for t in self._trajectories
            ],
            dim=0,
        )
        self._tensor_traj_radii = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_radii, dtype=dtype, device=device)
                for t in self._trajectories
            ],
            dim=0,
        )
        self._tensor_traj_active = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_active, dtype=torch.bool, device=device)
                for t in self._trajectories
            ],
            dim=0,
        )
        self._tensor_traj_goals = torch.stack(
            [
                torch.as_tensor(t.scene.goal, dtype=dtype, device=device)
                for t in self._trajectories
            ],
            dim=0,
        )
        self._tensor_radii = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_radii, dtype=dtype, device=device)
                for t in self._transitions
            ],
            dim=0,
        )
        self._tensor_active = torch.stack(
            [
                torch.as_tensor(t.scene.obstacle_active, dtype=torch.bool, device=device)
                for t in self._transitions
            ],
            dim=0,
        )
        self._tensor_goals = torch.stack(
            [
                torch.as_tensor(t.scene.goal, dtype=dtype, device=device)
                for t in self._transitions
            ],
            dim=0,
        )


def collect(
    system: System,
    scene_sampler: Callable[[np.random.Generator], Scene],
    rng: np.random.Generator,
    n_episodes: int,
    max_steps: int,
    dt: float,
    buffer: OCReplayBuffer | None = None,
    h_scale: float = 1.0,
    storage_device: torch.device | None = None,
    storage_dtype: torch.dtype | None = None,
) -> OCReplayBuffer:
    if buffer is None:
        capacity = max(1, n_episodes * max(1, max_steps))
        buffer = OCReplayBuffer(capacity=capacity)

    scenes = [scene_sampler(rng) for _ in range(n_episodes)]
    dtype = storage_dtype or system.u_bounds.dtype
    device = storage_device or system.u_bounds.device
    batched_scene = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched_scene)
    with torch.no_grad():
        states = rollout_lqr(system, batched_scene, x0, max_steps, dt)
        h_values = signed_h(system.position(states), batched_scene, h_scale)
    buffer.append_batch(
        scenes,
        batched_scene,
        states.detach(),
        h_values.detach(),
    )
    return buffer


def label_anchors(
    traj_view: list[TrajectoryRecord],
    value_net: Any,
    labeling_cfg: dict[str, Any],
) -> AnchorSets:
    _ = value_net
    k_a = int(labeling_cfg["k_A"])
    m_phys = float(labeling_cfg["m_phys"])
    k_safe = int(labeling_cfg["k_safe"])
    delta_c = float(labeling_cfg["delta_C"])

    unsafe_items = []
    safe_items = []
    for traj_pos, trajectory in enumerate(traj_view):
        h = trajectory.h
        collision_steps = torch.nonzero(h > 0.0, as_tuple=False).flatten()
        collision_free = collision_steps.numel() == 0
        for step_idx in range(h.shape[0]):
            tau_collision = _time_to_next_collision(collision_steps, step_idx)
            in_collision_window = tau_collision <= k_a
            near_unsafe = bool(m_phys > 0.0 and h[step_idx] >= -m_phys)
            future_h = torch.max(h[step_idx : min(h.shape[0], step_idx + k_safe + 1)])

            if in_collision_window or near_unsafe:
                unsafe_items.append(
                    _anchor_item(
                        trajectory,
                        traj_pos,
                        step_idx,
                        in_collision_window,
                        near_unsafe,
                        future_h,
                    )
                )
            if collision_free and bool(future_h <= -delta_c):
                safe_items.append(
                    _anchor_item(
                        trajectory,
                        traj_pos,
                        step_idx,
                        False,
                        False,
                        future_h,
                    )
                )

    state_dim = traj_view[0].states.shape[1] if traj_view else 0
    dtype = traj_view[0].states.dtype if traj_view else torch.float64
    device = traj_view[0].states.device if traj_view else torch.device("cpu")
    return AnchorSets(
        unsafe=_make_anchor_batch(unsafe_items, state_dim, dtype, device),
        safe=_make_anchor_batch(safe_items, state_dim, dtype, device),
    )


def sample_value_minibatch(
    buffer: OCReplayBuffer,
    anchors: AnchorSets,
    batch_size: int,
    rng: np.random.Generator,
) -> ValueMinibatch:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    transitions = buffer.transition_view
    if not transitions:
        raise ValueError("Cannot sample from an empty transition buffer.")

    transition_indices = rng.integers(0, len(transitions), size=batch_size)
    selected = [transitions[int(idx)] for idx in transition_indices]
    anchor_count = batch_size // 4
    unsafe_states, unsafe_indices = _sample_anchor_states(
        anchors.unsafe,
        anchor_count,
        rng,
    )
    safe_states, safe_indices = _sample_anchor_states(
        anchors.safe,
        anchor_count,
        rng,
    )

    return ValueMinibatch(
        transition_states=torch.stack([item.state for item in selected], dim=0),
        transition_next_states=torch.stack(
            [item.next_state for item in selected],
            dim=0,
        ),
        transition_h=torch.stack([item.h for item in selected], dim=0),
        transition_next_h=torch.stack([item.next_h for item in selected], dim=0),
        transition_scenes=[item.scene for item in selected],
        unsafe_states=unsafe_states,
        safe_states=safe_states,
        source_indices={
            "transition": transition_indices,
            "unsafe": unsafe_indices,
            "safe": safe_indices,
        },
    )


def _time_to_next_collision(collision_steps: Tensor, step_idx: int) -> int:
    future = collision_steps[collision_steps >= step_idx]
    if future.numel() == 0:
        return 10**9
    return int(future[0]) - step_idx


def _anchor_item(
    trajectory: TrajectoryRecord,
    traj_pos: int,
    step_idx: int,
    collision_window: bool,
    near_unsafe: bool,
    future_h: Tensor,
) -> dict[str, Any]:
    return {
        "state": trajectory.states[step_idx],
        "h": trajectory.h[step_idx],
        "trajectory_idx": traj_pos,
        "step_idx": step_idx,
        "collision_window": collision_window,
        "near_unsafe": near_unsafe,
        "future_h": future_h,
    }


def _make_anchor_batch(
    items: list[dict[str, Any]],
    state_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> AnchorBatch:
    if not items:
        empty_states = torch.empty((0, state_dim), dtype=dtype, device=device)
        empty_float = torch.empty((0,), dtype=dtype, device=device)
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
        return AnchorBatch(
            states=empty_states,
            h=empty_float,
            trajectory_indices=empty_long,
            step_indices=empty_long,
            collision_window=empty_bool,
            near_unsafe=empty_bool,
            future_h=empty_float,
        )

    return AnchorBatch(
        states=torch.stack([item["state"] for item in items], dim=0),
        h=torch.stack([item["h"] for item in items], dim=0),
        trajectory_indices=torch.as_tensor(
            [item["trajectory_idx"] for item in items],
            dtype=torch.long,
            device=device,
        ),
        step_indices=torch.as_tensor(
            [item["step_idx"] for item in items],
            dtype=torch.long,
            device=device,
        ),
        collision_window=torch.as_tensor(
            [item["collision_window"] for item in items],
            dtype=torch.bool,
            device=device,
        ),
        near_unsafe=torch.as_tensor(
            [item["near_unsafe"] for item in items],
            dtype=torch.bool,
            device=device,
        ),
        future_h=torch.stack([item["future_h"] for item in items], dim=0),
    )


def _sample_anchor_states(
    anchors: AnchorBatch,
    count: int,
    rng: np.random.Generator,
) -> tuple[Tensor, np.ndarray]:
    if count <= 0 or anchors.states.shape[0] == 0:
        return anchors.states[:0], np.empty((0,), dtype=np.int64)
    indices = rng.integers(0, anchors.states.shape[0], size=count)
    index_tensor = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=anchors.states.device,
    )
    return anchors.states.index_select(0, index_tensor), indices


def _cat_optional(existing: Tensor | None, new: Tensor) -> Tensor:
    value = new.detach().clone()
    if existing is None:
        return value
    return torch.cat([existing, value], dim=0)


def _pad_record_h(
    trajectories: list[TrajectoryRecord],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    max_len = max(max(0, trajectory.h.shape[0] - 1) for trajectory in trajectories)
    padded = []
    for trajectory in trajectories:
        h = trajectory.h[:-1].to(device=device, dtype=dtype)
        if h.shape[0] < max_len:
            h = torch.cat(
                [h, h[-1].expand(max_len - h.shape[0])],
                dim=0,
            )
        padded.append(h)
    return torch.stack(padded, dim=0)


def _slice_front(value: Tensor | None, count: int) -> Tensor | None:
    if value is None:
        return None
    if count <= 0:
        return value
    return value[count:].contiguous()
