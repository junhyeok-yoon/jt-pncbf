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
from src.common.quadrotor_barrier import value_target_barrier
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import BatchedScene, batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene, sample_scenes
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


@dataclass(frozen=True)
class PrecursorStats:
    n: int
    rollout_len: int
    target_unsafe_frac: float
    target_mean: float


@dataclass
class JTReplayBuffers:
    value: OCReplayBuffer
    policy: OCReplayBuffer
    precursor: OCReplayBuffer  # v2.2.2: collision-precursor states, populated once per collection


def make_replay_buffers(
    capacity: int, policy_capacity: int | None = None, store_state_seq: bool = False
) -> JTReplayBuffers:
    # v2.3.0: D_pi (policy buffer) may be capped independently of D_V / precursor.
    # policy_capacity=None -> D_pi shares `capacity`, bit-identical to prior behavior.
    # When set, only D_pi's FIFO cap differs; D_V and precursor keep `capacity`.
    policy_cap = capacity if policy_capacity is None else int(policy_capacity)
    # v2.8.2: only D_V feeds the value target, so only it needs the state sequence (flag-gated).
    return JTReplayBuffers(
        value=OCReplayBuffer(capacity=capacity, store_state_seq=store_state_seq),
        policy=OCReplayBuffer(capacity=policy_cap),
        precursor=OCReplayBuffer(capacity=capacity),
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

    inject_frac = float(config.get("collection", {}).get("inject_frac", 0.0))  # type: ignore[union-attr]
    if str(config.get("collection", {}).get("collector", "legacy")) == "continuing":  # type: ignore[union-attr]
        return _collect_policy_rollouts_continuing(
            system=system, policy_net=policy_net, value_net=value_net, scene_sampler=scene_sampler,
            rng=rng, torch_generator=torch_generator, n_episodes=n_episodes, max_steps=max_steps, dt=dt,
            buffer=buffer, config=config, sigma=sigma, storage_device=storage_device,
            storage_dtype=storage_dtype, collection_filter=collection_filter, inject_frac=inject_frac,
        )

    scenes = sample_scenes(
        scene_sampler, rng, n_episodes,
        inject_frac=inject_frac,
        system_name=system.name,
    )
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
        obs_deficit=bool(config.get("loss", {}).get("policy", {}).get("obs_deficit_feedback", False)),  # type: ignore[union-attr]
    )
    with torch.no_grad():
        # v2.6.0: the value target barrier is h_star (phi + c v^T Re) for the quadrotor, phi for DI/uni.
        h_values = value_target_barrier(system, states, batched_scene, config)
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


def _collect_policy_rollouts_continuing(
    *, system, policy_net, value_net, scene_sampler, rng, torch_generator, n_episodes, max_steps, dt,
    buffer, config, sigma, storage_device, storage_dtype, collection_filter, inject_frac,
) -> CollectionStats:
    """v2.7.0 iter-5: continuing-batch JT collection round (see continuing_collector). Persistent per-buffer
    rows carried across rounds; per-segment bootstrap labels via the SAME value_target_barrier (B=1 path)."""
    from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
    if bool(config.get("loss", {}).get("policy", {}).get("obs_deficit_feedback", False)):  # type: ignore[union-attr]
        raise NotImplementedError("continuing collector does not support obs_deficit_feedback (frozen off).")
    filter_layer = _make_collection_filter(collection_filter, system, value_net, config)
    bounds = system.u_bounds.to(device=storage_device, dtype=storage_dtype)

    def step_fn(x: Tensor, batched_scene) -> Tensor:
        obs = system.observation(x, batched_scene)
        u_base = policy_net(obs)
        if sigma > 0.0:
            noise = torch.randn(u_base.shape, generator=torch_generator, device=u_base.device,
                                dtype=u_base.dtype) * float(sigma)
        else:
            noise = torch.zeros_like(u_base)
        u_tilde = torch.clamp(u_base + noise, min=bounds[:, 0], max=bounds[:, 1])
        filtered = filter_layer(x.detach(), batched_scene, u_tilde.detach())
        return filtered[0].detach()

    def h_batch_fn(states_g: Tensor, batched_scene) -> Tensor:  # states_g [L, G, D] -> h [L, G]
        return value_target_barrier(system, states_g, batched_scene, config)

    state = getattr(buffer, "_cont_state", None)
    if state is None:
        state = ContinuingState.create(system, scene_sampler, rng, n_episodes, config,
                                       storage_device, storage_dtype, inject_frac=inject_frac,
                                       system_name=system.name)
        buffer._cont_state = state
    st = advance_round(state, round_length=max_steps, step_fn=step_fn, h_batch_fn=h_batch_fn,
                       scene_sampler=scene_sampler, rng=rng, config=config, buffer=buffer, dt=dt,
                       inject_frac=inject_frac, system_name=system.name)
    unsafe_fraction = (st.unsafe_segments / st.segments) if st.segments else 0.0
    sigma_after = adaptive_sigma_update(float(sigma), unsafe_fraction, config)
    buffer._cont_last_stats = st                                # instrumentation for the caller/report
    return CollectionStats(n_episodes=n_episodes, unsafe_fraction=unsafe_fraction,
                           sigma_before=float(sigma), sigma_after=sigma_after,
                           mean_projection=0.0, infeasible_fraction=0.0)


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
    inj_cfg = config.get("loss", {}).get("value", {}).get("precursor_injection", {})  # type: ignore[union-attr]
    if bool(inj_cfg.get("enabled", False)):
        collect_precursors(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            scene_sampler=scene_sampler,
            rng=rng,
            torch_generator=torch_generator,
            buffer=buffers.precursor,
            n_precursors=n_episodes,
            max_steps=max_steps,
            dt=dt,
            config=config,
            storage_device=storage_device,
            storage_dtype=storage_dtype,
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
    obs_deficit: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    x = system.wrap_state(x0.detach())
    states = [x]
    projection_steps = []
    infeasible_steps = []
    bounds = system.u_bounds.to(device=x.device, dtype=x.dtype)
    # v2.4.1 Exp 2: deployed deficit-observation channel. delta_u_{t-1} (init 0) is appended to the
    # policy observation; it is a detached feature (collection is no_grad), never a gradient path.
    prev_deficit = (
        x.new_zeros((x.shape[0], system.action_dim)) if obs_deficit else None
    )

    for _ in range(max_steps):
        with torch.no_grad():
            obs = system.observation(x, scene)
            if obs_deficit:
                obs = torch.cat([obs, prev_deficit], dim=1)
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
            if obs_deficit:
                filtered = filter_layer(x.detach(), scene, u_tilde.detach(), return_deficit_aux=True)
                u_safe = filtered[0]
                infeasible = filtered[1]
                prev_deficit = (filtered[2] - filtered[0]).detach()
            else:
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


def _construct_precursors(
    system: System,
    scene: BatchedScene,
    config: Mapping[str, object],
    generator: torch.Generator | None,
) -> Tensor:
    """v2.2.2: synthetic collision-precursor initial states, vectorized on-device, placed around a
    random ACTIVE obstacle of each scene (clearance d in [d_lo,d_hi], inward speed s in [s_lo,s_hi])."""
    cfg = config["loss"]["value"]["precursor_injection"]  # type: ignore[index]
    d_lo, d_hi = float(cfg.get("d_lo", 0.0)), float(cfg.get("d_hi", 0.2))  # type: ignore[union-attr]
    s_lo, s_hi = float(cfg.get("s_lo", 0.3)), float(cfg.get("s_hi", 1.0))  # type: ignore[union-attr]
    lateral_frac = float(cfg.get("lateral_frac", 0.5))  # type: ignore[union-attr]
    v_max = float(config["env"]["bounds"]["double_integrator"]["v_max"])  # type: ignore[index]
    centers = scene.obstacle_centers
    radii = scene.obstacle_radii
    active = scene.obstacle_active
    B = centers.shape[0]
    device, dtype = centers.device, centers.dtype
    eps = 1.0e-9
    weights = active.to(dtype)
    weights = torch.where(weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights))
    sel = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    bidx = torch.arange(B, device=device)
    c0 = centers[bidx, sel]
    r0 = radii[bidx, sel]
    g = torch.randn(B, 2, generator=generator, device=device, dtype=dtype)
    u_rad = g / torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)
    d = d_lo + (d_hi - d_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    p = c0 + (r0 + d).unsqueeze(1) * u_rad
    s = s_lo + (s_hi - s_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    v_in = -s.unsqueeze(1) * u_rad
    perp = torch.stack([-u_rad[:, 1], u_rad[:, 0]], dim=1)
    lat = (lateral_frac * s) * (2.0 * torch.rand(B, generator=generator, device=device, dtype=dtype) - 1.0)
    v = v_in + lat.unsqueeze(1) * perp
    if system.name == "unicycle":
        # Unicycle analog of the DI inward-velocity precursor: same inward(+lateral) 2D velocity,
        # expressed as (heading, signed speed). theta points toward the obstacle (+ lateral spread);
        # speed = |v| capped at the unicycle v_max. wrap_state re-wraps theta / re-clamps speed.
        u_v_max = float(config["env"]["bounds"]["unicycle"]["v_max"])  # type: ignore[index]
        speed = torch.linalg.norm(v, dim=1).clamp(max=u_v_max)
        theta = torch.atan2(v[:, 1], v[:, 0])
        return system.wrap_state(torch.stack([p[:, 0], p[:, 1], theta, speed], dim=1))
    vnorm = torch.linalg.norm(v, dim=1, keepdim=True)
    v = torch.where(vnorm > v_max, v * (v_max / vnorm.clamp_min(eps)), v)
    return system.wrap_state(torch.cat([p, v], dim=1))


def collect_precursors(
    *,
    system: System,
    policy_net: torch.nn.Module,
    value_net: ValueNetEnsemble,
    scene_sampler: Callable[[np.random.Generator], Scene],
    rng: np.random.Generator,
    torch_generator: torch.Generator | None,
    buffer: OCReplayBuffer,
    n_precursors: int,
    max_steps: int,
    dt: float,
    config: Mapping[str, object],
    storage_device: torch.device,
    storage_dtype: torch.dtype,
) -> PrecursorStats:
    """v2.2.2 collection-time precursor injection: construct precursors around freshly-sampled scenes,
    roll out the CURRENT policy ONCE via the existing batched-GPU HardNet-filtered rollout (sigma=0,
    full fixed horizon -- see NOTE), and append the precursor trajectories to the precursor buffer as
    ordinary (state, target) transitions. Targets are the standard max-over-time signed_h, labeled at
    collection time exactly like every other buffer state.

    NOTE: collision early-stop is NOT applied -- the OCReplayBuffer incremental tensor append
    (_append_tensor_batch) requires a FIXED trajectory length across collection batches, which a
    variable-length early-stopped rollout violates. The full-horizon rollout gives the identical
    (exact) max-over-time target, and the dominant cost reduction is already achieved by rolling out
    ONCE per collection (not K_V times per value update)."""
    h_scale = float(config["env"]["h_scale"])  # type: ignore[index]
    scenes = [scene_sampler(rng) for _ in range(n_precursors)]
    batched_scene = batch_scenes(scenes, device=storage_device, dtype=storage_dtype)
    x0 = _construct_precursors(system, batched_scene, config, torch_generator)
    filter_layer = _make_collection_filter("hardnet", system, value_net, config)
    states_t, _, _ = _rollout_for_collection(
        system=system, policy_net=policy_net, filter_layer=filter_layer, scene=batched_scene, x0=x0,
        max_steps=max_steps, dt=dt, sigma=0.0, torch_generator=torch_generator,
        obs_deficit=bool(config.get("loss", {}).get("policy", {}).get("obs_deficit_feedback", False)),  # type: ignore[union-attr]
    )
    with torch.no_grad():
        h_values = value_target_barrier(system, states_t, batched_scene, config)
    buffer.append_batch(scenes, batched_scene, states_t.detach(), h_values.detach())
    max_h = h_values.max(dim=0).values  # max-over-time per precursor (target proxy, pre-bootstrap)
    return PrecursorStats(
        n=n_precursors,
        rollout_len=int(states_t.shape[0] - 1),
        target_unsafe_frac=float((max_h >= 0.0).to(max_h.dtype).mean().item()),
        target_mean=float(max_h.mean().item()),
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
    # v2.5.0 Stage B (scope-extended): the safety channel may be the analytic maneuver barrier V_M
    # instead of the learned make_h_fn(value_net) — one builder shared with the policy-BPTT filter, so
    # collection and BPTT filter with the SAME h_fn. Value path (default) is bit-identical.
    from src.common.maneuver_value import build_safety_h_fn
    h_fn = build_safety_h_fn(system, config, value_net)
    if collection_filter == "hardnet":
        return HardNetFilter(system, h_fn, config)
    if collection_filter == "cbf_qp":
        return CBFQPFilter(system, h_fn, config)
    raise ValueError(f"Unsupported JT collection filter: {collection_filter!r}")
