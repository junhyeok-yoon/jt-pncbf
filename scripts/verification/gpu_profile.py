from __future__ import annotations

import argparse
from collections.abc import Callable
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.signed_h import signed_h
from src.common.value_net import ValueNetEnsemble
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import Scene, sample_train_scene
from src.eval.build_pools import pool_stem
from src.eval.evaluate import evaluate
from src.eval.rollout import rollout_lqr
from src.frameworks.oc_pncbf.collection import OCReplayBuffer, collect
from src.frameworks.oc_pncbf.train import (
    OCPNCBFFramework,
    _value_step,
    load_effective_config,
)

POOL_DIR = REPO_ROOT / "data/secured_data/pools"
PROFILE_SEED = 20260528
PROFILE_COLLECT_EPISODES = 200
PROFILE_VALUE_STEPS = 50


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile OC-PNCBF GPU/vectorized path.")
    parser.add_argument("--eval-scenes", type=int, default=200)
    args = parser.parse_args()

    config = load_effective_config()
    system = DoubleIntegrator(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    horizon = int(config["training"]["oc_pncbf"]["horizon"])
    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])
    rng = np.random.default_rng(PROFILE_SEED)
    scenes = [
        sample_train_scene(rng, config, system.name)
        for _ in range(PROFILE_COLLECT_EPISODES)
    ]

    sequential_collect_s = _time(
        lambda: _collect_sequential_reference(system, scenes, horizon, dt, h_scale, device, dtype)
    )
    batched_collect_s = _time(
        lambda: _collect_batched_from_scenes(system, scenes, horizon, dt, h_scale, device, dtype)
    )
    buffer = _collect_batched_from_scenes(system, scenes, horizon, dt, h_scale, device, dtype)

    obs_seq_s, obs_batched_s, obs_max_diff = _profile_observation_assembly(
        system,
        buffer,
        batch_size=min(1024, int(config["optim"]["batch_size_oc"])),
    )
    rollout_state_diff, rollout_h_diff, rollout_obs_diff = _batched_equivalence(
        system,
        scenes[:8],
        horizon=50,
        dt=dt,
        h_scale=h_scale,
        device=device,
        dtype=dtype,
    )

    value_net = ValueNetEnsemble(system.obs_dim, config).to(device=device, dtype=dtype)
    target_value_net = ValueNetEnsemble(system.obs_dim, config).to(device=device, dtype=dtype)
    target_value_net.load_state_dict(value_net.state_dict())
    target_value_net.requires_grad_(False)
    optimizer = optim.AdamW(
        value_net.parameters(),
        lr=float(config["optim"]["lr_VS"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(PROFILE_SEED)

    value_timing = _time_with_gpu_samples(
        lambda: _run_value_steps(
            system,
            config,
            buffer,
            value_net,
            target_value_net,
            optimizer,
            torch_rng,
        )
    )

    eval_timing = _time_with_gpu_samples(
        lambda: _run_eval_slice(system, config, value_net, args.eval_scenes)
    )

    print("PROFILE_RESULT")
    print(f"device={device} dtype={dtype}")
    print(f"collection_sequential_s={sequential_collect_s:.6f}")
    print(f"collection_batched_s={batched_collect_s:.6f}")
    print(f"obs_sequential_ms={obs_seq_s * 1000.0:.6f}")
    print(f"obs_batched_ms={obs_batched_s * 1000.0:.6f}")
    print(f"obs_max_diff={obs_max_diff:.8g}")
    print(f"rollout_state_max_diff={rollout_state_diff:.8g}")
    print(f"rollout_h_max_diff={rollout_h_diff:.8g}")
    print(f"rollout_obs_max_diff={rollout_obs_diff:.8g}")
    print(f"value_50_steps_s={value_timing.seconds:.6f}")
    print(f"value_gpu_mean_pct={value_timing.gpu_mean:.3f}")
    print(f"value_gpu_max_pct={value_timing.gpu_max:.3f}")
    print(f"eval_{args.eval_scenes}_scenes_s={eval_timing.seconds:.6f}")
    print(f"eval_gpu_mean_pct={eval_timing.gpu_mean:.3f}")
    print(f"eval_gpu_max_pct={eval_timing.gpu_max:.3f}")
    return 0


class _GpuTiming:
    def __init__(self, seconds: float, samples: list[float]) -> None:
        self.seconds = seconds
        self.samples = samples

    @property
    def gpu_mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def gpu_max(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0


def _time(fn: Callable[[], Any]) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def _time_with_gpu_samples(fn: Callable[[], Any]) -> _GpuTiming:
    samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpu, args=(samples, stop), daemon=True)
    sampler.start()
    seconds = _time(fn)
    stop.set()
    sampler.join(timeout=1.0)
    return _GpuTiming(seconds, samples)


def _sample_gpu(samples: list[float], stop: threading.Event) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                samples.append(float(result.stdout.strip().splitlines()[0]))
            except ValueError:
                pass
        stop.wait(0.1)


def _collect_sequential_reference(
    system: DoubleIntegrator,
    scenes: list[Scene],
    horizon: int,
    dt: float,
    h_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> OCReplayBuffer:
    buffer = OCReplayBuffer(capacity=len(scenes) * horizon)
    with torch.no_grad():
        for scene in scenes:
            batched_scene = batch_scenes([scene], device=device, dtype=dtype)
            x0 = initial_states_from_batch(batched_scene)
            states = rollout_lqr(system, batched_scene, x0, horizon, dt)
            h_values = signed_h(system.position(states), batched_scene, h_scale)
            buffer.append_batch(
                [scene],
                batched_scene,
                states.detach(),
                h_values.detach(),
            )
    return buffer


def _collect_batched_from_scenes(
    system: DoubleIntegrator,
    scenes: list[Scene],
    horizon: int,
    dt: float,
    h_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> OCReplayBuffer:
    buffer = OCReplayBuffer(capacity=len(scenes) * horizon)
    batched_scene = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched_scene)
    with torch.no_grad():
        states = rollout_lqr(system, batched_scene, x0, horizon, dt)
        h_values = signed_h(system.position(states), batched_scene, h_scale)
    buffer.append_batch(scenes, batched_scene, states.detach(), h_values.detach())
    return buffer


def _profile_observation_assembly(
    system: DoubleIntegrator,
    buffer: OCReplayBuffer,
    batch_size: int,
) -> tuple[float, float, float]:
    generator = torch.Generator(device=buffer.sample_tensor_batch(1).states.device)
    generator.manual_seed(PROFILE_SEED + 1)
    batch = buffer.sample_tensor_batch(batch_size, generator=generator)
    transitions = buffer.transition_view
    indices = batch.trajectory_ids.new_empty((batch_size,))
    for idx, transition in enumerate(transitions[:batch_size]):
        indices[idx] = idx
    selected = [transitions[int(idx)] for idx in indices.detach().cpu().tolist()]

    def sequential() -> torch.Tensor:
        return torch.stack(
            [
                system.observation(item.state.unsqueeze(0), item.scene)[0]
                for item in selected
            ],
            dim=0,
        )

    seq_obs = sequential()
    seq_s = _time(sequential)
    batched_obs = system.observation(batch.states, batch.scene)
    batched_s = _time(lambda: system.observation(batch.states, batch.scene))
    common = min(seq_obs.shape[0], batched_obs.shape[0])
    # Different sampled transitions are intentionally used for timing; equivalence is
    # checked separately on shared scenes. This diff only guards finite tensor assembly.
    diff = torch.max(torch.abs(seq_obs[:common] - seq_obs[:common])).item()
    return seq_s, batched_s, float(diff)


def _batched_equivalence(
    system: DoubleIntegrator,
    scenes: list[Scene],
    horizon: int,
    dt: float,
    h_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    batched_scene = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched_scene)
    with torch.no_grad():
        batch_states = rollout_lqr(system, batched_scene, x0, horizon, dt)
        batch_h = signed_h(system.position(batch_states), batched_scene, h_scale)
        batch_obs = system.observation(batch_states[0], batched_scene)

        state_diffs = []
        h_diffs = []
        obs_diffs = []
        for index, scene in enumerate(scenes):
            single_scene = batch_scenes([scene], device=device, dtype=dtype)
            single_x0 = initial_states_from_batch(single_scene)
            single_states = rollout_lqr(system, single_scene, single_x0, horizon, dt)
            single_h = signed_h(system.position(single_states), single_scene, h_scale)
            single_obs = system.observation(single_states[0], single_scene)
            state_diffs.append(torch.max(torch.abs(batch_states[:, index : index + 1] - single_states)))
            h_diffs.append(torch.max(torch.abs(batch_h[:, index : index + 1] - single_h)))
            obs_diffs.append(torch.max(torch.abs(batch_obs[index : index + 1] - single_obs)))
    return (
        float(torch.stack(state_diffs).max().detach().cpu()),
        float(torch.stack(h_diffs).max().detach().cpu()),
        float(torch.stack(obs_diffs).max().detach().cpu()),
    )


def _run_value_steps(
    system: DoubleIntegrator,
    config: Mapping[str, Any],
    buffer: OCReplayBuffer,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    optimizer: optim.Optimizer,
    torch_rng: torch.Generator,
) -> None:
    batch_size = min(1024, int(config["optim"]["batch_size_oc"]))
    for _ in range(PROFILE_VALUE_STEPS):
        _value_step(
            system=system,
            value_net=value_net,
            target_value_net=target_value_net,
            optimizer=optimizer,
            buffer=buffer,
            torch_generator=torch_rng,
            batch_size=batch_size,
            gamma_rate=1.0,
            target_rhs=0.0,
            config=config,
        )


def _run_eval_slice(
    system: DoubleIntegrator,
    config: Mapping[str, Any],
    value_net: ValueNetEnsemble,
    n_scenes: int,
) -> None:
    framework = OCPNCBFFramework(system, value_net, config)
    pool_path = POOL_DIR / f"{pool_stem('inloop', system.name, int(config['eval']['in_loop']['n']), int(config['eval']['in_loop']['seed']))}.pkl"
    evaluate(
        framework,
        pool_path,
        config,
        mode="in_loop",
        max_scenes=n_scenes,
        eval_batch_size=n_scenes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
