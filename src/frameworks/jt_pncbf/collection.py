from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from src.common.filter_cbfqp import CBFQPFilter
from src.common.filter_hardnet import HardNetFilter
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import BatchedScene, batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene
from src.frameworks.oc_pncbf.collection import OCReplayBuffer


Tensor = torch.Tensor


@dataclass(frozen=True)
class CollectionStats:
    n_episodes: int
    unsafe_fraction: float
    sigma_before: float
    sigma_after: float
    mean_projection: float
    infeasible_fraction: float


@dataclass
class JTReplayBuffers:
    value: OCReplayBuffer
    policy: OCReplayBuffer


def make_replay_buffers(capacity: int) -> JTReplayBuffers:
    return JTReplayBuffers(
        value=OCReplayBuffer(capacity=capacity),
        policy=OCReplayBuffer(capacity=capacity),
    )


def collect_policy_rollouts(
    *,
    system: System,
    policy_net: torch.nn.Module,
    value_net: ValueNetEnsemble,
    scene_sampler: Callable[[np.random.Generator], Scene],
    rng: np.random.Generator,
    torch_generator: torch.Generator | None,
    n_episodes: int,
    max_steps: int,
    dt: float,
    buffer: OCReplayBuffer,
    config: Mapping[str, object],
    sigma: float,
    storage_device: torch.device,
    storage_dtype: torch.dtype,
    collection_filter: str = "hardnet",
) -> CollectionStats:
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}.")

    scenes = [scene_sampler(rng) for _ in range(n_episodes)]
    batched_scene = batch_scenes(scenes, device=storage_device, dtype=storage_dtype)
    x0 = initial_states_from_batch(batched_scene)
    filter_layer = _make_collection_filter(
        collection_filter,
        system,
        value_net,
        config,
    )
    states, projection, infeasible = _rollout_for_collection(
        system=system,
        policy_net=policy_net,
        filter_layer=filter_layer,
        scene=batched_scene,
        x0=x0,
        max_steps=max_steps,
        dt=dt,
        sigma=sigma,
        torch_generator=torch_generator,
    )
    with torch.no_grad():
        h_values = signed_h(
            system.position(states),
            batched_scene,
            float(config["env"]["h_scale"]),  # type: ignore[index]
        )
    buffer.append_batch(scenes, batched_scene, states.detach(), h_values.detach())
    unsafe_fraction = _unsafe_episode_fraction(h_values)
    sigma_after = adaptive_sigma_update(float(sigma), unsafe_fraction, config)
    return CollectionStats(
        n_episodes=n_episodes,
        unsafe_fraction=unsafe_fraction,
        sigma_before=float(sigma),
        sigma_after=sigma_after,
        mean_projection=float(projection.mean().detach().cpu().item())
        if projection.numel()
        else 0.0,
        infeasible_fraction=float(infeasible.float().mean().detach().cpu().item())
        if infeasible.numel()
        else 0.0,
    )


def collect_jt(
    *,
    system: System,
    policy_net: torch.nn.Module,
    value_net: ValueNetEnsemble,
    scene_sampler: Callable[[np.random.Generator], Scene],
    rng: np.random.Generator,
    torch_generator: torch.Generator | None,
    buffers: JTReplayBuffers,
    n_episodes: int,
    max_steps: int,
    dt: float,
    config: Mapping[str, object],
    sigma_v: float,
    sigma_pi: float,
    storage_device: torch.device,
    storage_dtype: torch.dtype,
    collection_filter: str = "hardnet",
) -> tuple[CollectionStats, CollectionStats]:
    value_stats = collect_policy_rollouts(
        system=system,
        policy_net=policy_net,
        value_net=value_net,
        scene_sampler=scene_sampler,
        rng=rng,
        torch_generator=torch_generator,
        n_episodes=n_episodes,
        max_steps=max_steps,
        dt=dt,
        buffer=buffers.value,
        config=config,
        sigma=sigma_v,
        storage_device=storage_device,
        storage_dtype=storage_dtype,
        collection_filter=collection_filter,
    )
    policy_stats = collect_policy_rollouts(
        system=system,
        policy_net=policy_net,
        value_net=value_net,
        scene_sampler=scene_sampler,
        rng=rng,
        torch_generator=torch_generator,
        n_episodes=n_episodes,
        max_steps=max_steps,
        dt=dt,
        buffer=buffers.policy,
        config=config,
        sigma=sigma_pi,
        storage_device=storage_device,
        storage_dtype=storage_dtype,
        collection_filter=collection_filter,
    )
    return value_stats, policy_stats


def adaptive_sigma_update(
    sigma: float,
    unsafe_fraction: float,
    config: Mapping[str, object],
) -> float:
    sigma_cfg = config["schedules"]["sigma"]  # type: ignore[index]
    sigma_min = float(sigma_cfg["sigma_min"])  # type: ignore[index]
    sigma_max = float(sigma_cfg["sigma_max"])  # type: ignore[index]
    beta = float(sigma_cfg.get("beta_sigma", 0.05))  # type: ignore[union-attr]
    target = float(sigma_cfg.get("rho_target", 0.10))  # type: ignore[union-attr]
    if unsafe_fraction < target:
        sigma_target = sigma_max
    else:
        sigma_target = sigma_min
    updated = (1.0 - beta) * float(sigma) + beta * sigma_target
    return float(min(max(updated, sigma_min), sigma_max))


def _rollout_for_collection(
    *,
    system: System,
    policy_net: torch.nn.Module,
    filter_layer: HardNetFilter | CBFQPFilter,
    scene: BatchedScene,
    x0: Tensor,
    max_steps: int,
    dt: float,
    sigma: float,
    torch_generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    x = system.wrap_state(x0.detach())
    states = [x]
    projection_steps = []
    infeasible_steps = []
    bounds = system.u_bounds.to(device=x.device, dtype=x.dtype)

    for _ in range(max_steps):
        with torch.no_grad():
            obs = system.observation(x, scene)
            u_base = policy_net(obs)
            if sigma > 0.0:
                noise = torch.randn(
                    u_base.shape,
                    generator=torch_generator,
                    device=u_base.device,
                    dtype=u_base.dtype,
                ) * float(sigma)
            else:
                noise = torch.zeros_like(u_base)
            u_tilde = torch.clamp(u_base + noise, min=bounds[:, 0], max=bounds[:, 1])

        with torch.enable_grad():
            filtered = filter_layer(x.detach(), scene, u_tilde.detach())
            u_safe = filtered[0]
            infeasible = filtered[1]

        u_safe = u_safe.detach()
        infeasible_steps.append(infeasible.detach())
        projection_steps.append(torch.linalg.norm(u_safe - u_tilde, dim=-1).detach())
        with torch.no_grad():
            x = rk4_step(system, x, u_safe, dt).detach()
        states.append(x)

    return (
        torch.stack(states, dim=0),
        torch.stack(projection_steps, dim=0)
        if projection_steps
        else x.new_empty((0, x.shape[0])),
        torch.stack(infeasible_steps, dim=0)
        if infeasible_steps
        else torch.empty((0, x.shape[0]), dtype=torch.bool, device=x.device),
    )


def _unsafe_episode_fraction(h_values: Tensor) -> float:
    if h_values.ndim != 2:
        raise ValueError(f"h_values must have shape [T+1, B], got {tuple(h_values.shape)}.")
    collided = torch.any(h_values > 0.0, dim=0)
    return float(collided.float().mean().detach().cpu().item())


def _make_collection_filter(
    collection_filter: str,
    system: System,
    value_net: ValueNetEnsemble,
    config: Mapping[str, object],
) -> HardNetFilter | CBFQPFilter:
    h_fn = make_h_fn(value_net, system)
    if collection_filter == "hardnet":
        return HardNetFilter(system, h_fn, config)
    if collection_filter == "cbf_qp":
        return CBFQPFilter(system, h_fn, config)
    raise ValueError(f"Unsupported JT collection filter: {collection_filter!r}")
