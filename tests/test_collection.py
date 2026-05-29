from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.common.outcomes import resolve_outcome, step_outcomes
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene, sample_train_scene
from src.eval.rollout import rollout, rollout_lqr
from src.frameworks.oc_pncbf.collection import (
    OCReplayBuffer,
    TrajectoryRecord,
    collect,
    label_anchors,
    sample_value_minibatch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def test_generic_filtered_rollout_shapes_and_masks() -> None:
    config = _load_base_config()
    system = DoubleIntegrator(config)
    scene = _scene()
    x0 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]], dtype=DTYPE)

    def policy_fn(x: torch.Tensor, policy_scene: Scene) -> torch.Tensor:
        return torch.ones((x.shape[0], system.action_dim), dtype=x.dtype)

    def filter_fn(
        x: torch.Tensor,
        u_nom: torch.Tensor,
        filter_scene: Scene,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_safe = u_nom.clone()
        u_safe[:, 0] = u_safe[:, 0] + 0.01
        infeasible = torch.tensor([False, True], device=x.device)
        return u_safe, infeasible

    result = rollout(system, policy_fn, filter_fn, scene, x0, max_steps=3, dt=0.05)
    assert result.states.shape == (4, 2, system.state_dim)
    assert result.u_nom.shape == (3, 2, system.action_dim)
    assert result.u_safe.shape == (3, 2, system.action_dim)
    assert result.intervention_mask.shape == (3, 2)
    assert result.infeasible.shape == (3, 2)
    assert torch.all(result.intervention_mask)
    assert torch.equal(result.infeasible[:, 1], torch.ones(3, dtype=torch.bool))


def test_eval_rollout_keeps_full_length_after_collision() -> None:
    config = _load_base_config()
    system = DoubleIntegrator(config)
    scene = _collision_path_scene()
    batched_scene = batch_scenes([scene], device=torch.device("cpu"), dtype=DTYPE)
    x0 = initial_states_from_batch(batched_scene)
    max_steps = 80

    states = rollout_lqr(
        system,
        batched_scene,
        x0,
        max_steps=max_steps,
        dt=float(config["env"]["dt"]),
    )
    resolved = resolve_outcome(step_outcomes(states, scene, system, config))

    assert states.shape[0] == max_steps + 1
    assert resolved.outcome == ["collision"]
    assert 0 <= int(resolved.event_step[0]) < max_steps


def test_buffer_fifo_eviction_keeps_views_consistent() -> None:
    buffer = OCReplayBuffer(capacity=1)
    scene = _scene()
    first_states = torch.zeros((3, 4), dtype=DTYPE)
    second_states = torch.ones((3, 4), dtype=DTYPE)
    h = torch.tensor([-0.5, -0.4, -0.3], dtype=DTYPE)

    first = buffer.append(scene, first_states, h)
    second = buffer.append(scene, second_states, h)

    assert first.trajectory_id != second.trajectory_id
    assert len(buffer.traj_view) == 1
    assert buffer.traj_view[0].trajectory_id == second.trajectory_id
    assert len(buffer.transition_view) == 2
    assert all(item.trajectory_id == second.trajectory_id for item in buffer.transition_view)


def test_collection_has_nontrivial_unsafe_signal() -> None:
    config = _load_base_config()
    system = DoubleIntegrator(config)
    rng = np.random.default_rng(seed=202605)

    def sampler(local_rng: np.random.Generator) -> Scene:
        return sample_train_scene(local_rng, config, system.name)

    buffer = collect(
        system,
        sampler,
        rng,
        n_episodes=80,
        max_steps=200,
        dt=float(config["env"]["dt"]),
        h_scale=float(config["env"]["h_scale"]),
    )
    fraction = _collision_fraction(buffer.traj_view)
    assert fraction > 0.05


def test_collection_keeps_post_terminal_states_for_value_targets() -> None:
    config = _load_base_config()
    system = DoubleIntegrator(config)
    scenes = [_collision_path_scene(), _oob_path_scene()]
    scene_iter = iter(scenes)

    def sampler(local_rng: np.random.Generator) -> Scene:
        _ = local_rng
        return next(scene_iter)

    buffer = collect(
        system,
        sampler,
        np.random.default_rng(seed=7),
        n_episodes=len(scenes),
        max_steps=80,
        dt=float(config["env"]["dt"]),
        h_scale=float(config["env"]["h_scale"]),
    )

    assert len(buffer.traj_view) == 2
    for scene, record in zip(scenes, buffer.traj_view, strict=True):
        full_batch = batch_scenes([scene], device=torch.device("cpu"), dtype=DTYPE)
        full_states = rollout_lqr(
            system,
            full_batch,
            initial_states_from_batch(full_batch),
            max_steps=80,
            dt=float(config["env"]["dt"]),
        )
        resolved = resolve_outcome(step_outcomes(full_states, scene, system, config))
        event_step = int(resolved.event_step[0])

        assert resolved.outcome[0] in {"collision", "oob"}
        assert 0 <= event_step < 80
        assert record.states.shape[0] == 81
        assert record.h.shape[0] == 81
        assert record.states.shape[0] > event_step + 1
        record_transitions = [
            t for t in buffer.transition_view if t.trajectory_id == record.trajectory_id
        ]
        assert len(record_transitions) == 80
        if resolved.outcome[0] == "collision":
            assert bool(torch.any(record.h[event_step + 1 :] > 0.0))

    sample = buffer.sample_tensor_batch(batch_size=4, generator=torch.Generator().manual_seed(3))
    assert sample.h_sequence.shape[1] == 4


def test_label_anchors_and_minibatch_composition() -> None:
    labeling_cfg = {"k_A": 2, "m_phys": 0.1, "k_safe": 1, "delta_C": 0.1}
    scene = _scene()
    collision_record = _record(
        trajectory_id=0,
        scene=scene,
        h_values=[-0.5, -0.2, -0.05, 0.1, 0.2],
    )
    safe_record = _record(
        trajectory_id=1,
        scene=scene,
        h_values=[-0.6, -0.5, -0.4, -0.3],
    )

    anchors = label_anchors([collision_record, safe_record], None, labeling_cfg)
    assert anchors.unsafe.states.shape[0] > 0
    assert anchors.safe.states.shape[0] > 0
    unsafe_valid = anchors.unsafe.collision_window | (
        anchors.unsafe.h >= -labeling_cfg["m_phys"]
    )
    assert torch.all(unsafe_valid)
    assert torch.all(anchors.safe.future_h <= -labeling_cfg["delta_C"])

    buffer = OCReplayBuffer(capacity=20)
    buffer.append(scene, collision_record.states, collision_record.h)
    buffer.append(scene, safe_record.states, safe_record.h)
    minibatch = sample_value_minibatch(
        buffer,
        anchors,
        batch_size=8,
        rng=np.random.default_rng(seed=11),
    )
    assert minibatch.transition_states.shape == (8, 4)
    assert minibatch.unsafe_states.shape == (2, 4)
    assert minibatch.safe_states.shape == (2, 4)
    assert minibatch.source_indices["transition"].shape == (8,)
    assert minibatch.source_indices["unsafe"].shape == (2,)
    assert minibatch.source_indices["safe"].shape == (2,)

    empty_safe = label_anchors([collision_record], None, labeling_cfg)
    degraded = sample_value_minibatch(
        buffer,
        empty_safe,
        batch_size=8,
        rng=np.random.default_rng(seed=12),
    )
    assert degraded.unsafe_states.shape[0] == 2
    assert degraded.safe_states.shape[0] == 0


def _record(
    trajectory_id: int,
    scene: Scene,
    h_values: list[float],
) -> TrajectoryRecord:
    states = torch.zeros((len(h_values), 4), dtype=DTYPE)
    states[:, 0] = torch.arange(len(h_values), dtype=DTYPE)
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        scene=scene,
        states=states,
        h=torch.tensor(h_values, dtype=DTYPE),
    )


def _collision_fraction(records: list[TrajectoryRecord]) -> float:
    collided = [bool(torch.any(record.h > 0.0)) for record in records]
    return sum(collided) / len(collided)


def _scene() -> Scene:
    centers = np.zeros((12, 2), dtype=np.float64)
    radii = np.zeros(12, dtype=np.float64)
    active = np.zeros(12, dtype=np.bool_)
    centers[0] = np.array([2.0, 2.0], dtype=np.float64)
    radii[0] = 0.2
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([0.0, 0.0], dtype=np.float64),
        goal=np.array([1.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _collision_path_scene() -> Scene:
    centers, radii, active = _obstacles()
    centers[0] = np.array([0.0, 0.0], dtype=np.float64)
    radii[0] = 0.3
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([-1.0, 0.0], dtype=np.float64),
        goal=np.array([1.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.zeros(2, dtype=np.float64),
    )


def _oob_path_scene() -> Scene:
    centers, radii, active = _obstacles()
    centers[0] = np.array([20.0, 20.0], dtype=np.float64)
    radii[0] = 0.1
    active[0] = True
    return Scene(
        obstacle_centers=centers,
        obstacle_radii=radii,
        obstacle_active=active,
        start=np.array([7.5, 0.0], dtype=np.float64),
        goal=np.array([20.0, 0.0], dtype=np.float64),
        system="double_integrator",
        mode="synthetic",
        initial_velocity=np.array([2.0, 0.0], dtype=np.float64),
    )


def _obstacles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((12, 2), dtype=np.float64),
        np.zeros(12, dtype=np.float64),
        np.zeros(12, dtype=np.bool_),
    )


def _load_base_config() -> Mapping[str, Any]:
    return yaml.safe_load((REPO_ROOT / "src/configs/base_config.yaml").read_text())
