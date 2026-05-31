from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn, optim
import yaml

from src._version import __version__
from src.common.control_net import ControlNet
from src.common.filter_hardnet import HardNetFilter
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_init import sample_train_scene
from src.envs.scene_init_fixed import sample_train_fixed_scene
from src.envs.unicycle import Unicycle
from src.eval.evaluate import (
    EVAL_EPISODE_COLUMNS,
    EVAL_METRIC_COLUMNS,
    EvaluationResult,
    evaluate,
)
from src.eval.build_pools import load_pool
from src.frameworks.jt_pncbf.collection import (
    CollectionStats,
    JTReplayBuffers,
    collect_policy_rollouts,
    collect_jt,
    make_replay_buffers,
)
from src.frameworks.jt_pncbf.losses import grad_norm, policy_bptt_loss, value_loss
from src.frameworks.oc_pncbf.value_target import (
    gamma_from_lambda,
    lambda_schedule_value,
    schedule_value,
)


Tensor = torch.Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG_PATH = REPO_ROOT / "src/configs/base_config.yaml"
EXP_CONFIG_PATH = REPO_ROOT / "src/configs/exp_config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data"
POOL_DIR = REPO_ROOT / "data/secured_data/pools"

METRIC_COLUMNS = [
    "step",
    "wallclock_s",
    "schedule_step",
    "lambda_disc_active",
    "gamma_disc_active",
    "target_rhs_active",
    "sigma",
    "sigma_pi",
    "rho_unsafe_v",
    "rho_unsafe_pi",
    "collect_proj_mag",
    "collect_infeasible",
    "L_R",
    "L_V_total",
    "L_in_task",
    "L_anorm",
    "L_smooth",
    "L_satex",
    "L_pretanh",
    "L_out",
    "L_pi_total",
    "grad_norm_VS",
    "grad_norm_pi",
    "grad_leak_VS_from_Lpi",
    "abs_action_mean",
    "abs_action_max",
    "satfrac_a_phi",
    "probe_h_min",
    "probe_h_max",
    "probe_h_mean",
]


@dataclass(frozen=True)
class JTTrainingResult:
    run_dir: Path
    halted: bool
    halt_reason: str | None
    last_value_loss: float
    last_policy_loss: float
    last_vs_grad_norm: float
    last_pi_grad_norm: float
    max_policy_grad_leak: float
    last_unsafe_fraction: float
    last_sigma: float


class JTPNCBFFramework:
    def __init__(
        self,
        system: System,
        value_net: ValueNetEnsemble,
        policy_net: ControlNet,
        config: Mapping[str, Any],
    ) -> None:
        self.system = system
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self._filter = HardNetFilter(
            system,
            make_h_fn(value_net, system),
            config,
            policy_fn=self.policy,
        )

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        return self.policy_net(self.system.observation(x, scene))

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        return self._filter(x, scene, u_nom)

    def value(self, x: Tensor, scene: Any) -> Tensor:
        return make_h_fn(self.value_net, self.system)(x, scene)


def run_training(
    *,
    stage: str = "full",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int | None = None,
    smoke_eval_scenes: int = 2,
    device: str = "auto",
    train_dtype_name: str = "float32",
    obstacle_distribution: str | None = None,
    n_steps_override: int | None = None,
    value_batch_size_override: int | None = None,
    schedule_n_steps_override: int | None = None,
) -> JTTrainingResult:
    config = load_effective_config()
    config["run"]["version"] = __version__
    config["run"]["framework"] = "jt_pncbf"
    if seed is not None:
        config["run"]["seed"] = int(seed)
    if obstacle_distribution is not None:
        config["env"]["obstacle_distribution"] = obstacle_distribution
    if n_steps_override is not None:
        if n_steps_override <= 0:
            raise ValueError(f"n_steps_override must be positive, got {n_steps_override}.")
        config["run"]["n_steps_override"] = int(n_steps_override)
    if value_batch_size_override is not None:
        if value_batch_size_override <= 0:
            raise ValueError(
                "value_batch_size_override must be positive, "
                f"got {value_batch_size_override}."
            )
        config["run"]["value_batch_size_override"] = int(value_batch_size_override)
    if schedule_n_steps_override is not None:
        if schedule_n_steps_override <= 0:
            raise ValueError(
                "schedule_n_steps_override must be positive, "
                f"got {schedule_n_steps_override}."
            )
        config["run"]["schedule_n_steps_override"] = int(schedule_n_steps_override)

    run_seed = int(config["run"]["seed"])
    np_rng = np.random.default_rng(run_seed)
    torch.manual_seed(run_seed)
    system = make_system(config)
    train_device = _resolve_train_device(device)
    train_dtype = _resolve_train_dtype(train_dtype_name)
    torch_rng = torch.Generator(device=train_device)
    torch_rng.manual_seed(run_seed)

    value_net = ValueNetEnsemble(system.obs_dim, config).to(
        device=train_device,
        dtype=train_dtype,
    )
    target_value_net = deepcopy(value_net)
    target_value_net.requires_grad_(False)
    policy_net = ControlNet(system.obs_dim, system, config).to(
        device=train_device,
        dtype=train_dtype,
    )
    opt_vs = optim.AdamW(
        value_net.parameters(),
        lr=float(config["optim"]["lr_VS"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    opt_pi = optim.AdamW(
        policy_net.parameters(),
        lr=float(config["optim"]["lr_pi"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    jt_cfg = dict(config["training"]["jt"])
    collection_cfg = dict(config["collection"]["jt"])
    base_n_steps = int(jt_cfg["n_steps"])
    n_steps = int(n_steps_override) if n_steps_override is not None else base_n_steps
    config_schedule_n_steps = int(jt_cfg.get("schedule_n_steps", base_n_steps))
    schedule_n_steps = (
        int(schedule_n_steps_override)
        if schedule_n_steps_override is not None
        else config_schedule_n_steps
    )
    config["training"]["jt"]["n_steps"] = n_steps
    config["training"]["jt"]["schedule_n_steps"] = schedule_n_steps
    vs_warmup_steps = int(jt_cfg["vs_warmup_steps"])
    k_v = int(jt_cfg["K_V"])
    k_pi = int(jt_cfg["K_pi"])
    bptt_t = int(jt_cfg["bptt_T"])
    collect_every = int(collection_cfg["collect_every"])
    episodes_per_collect = int(collection_cfg["episodes_per_collect"])
    horizon = int(config["training"]["oc_pncbf"]["horizon"])
    policy_batch_size = int(config["optim"]["batch_size_jt"])
    value_batch_size = (
        int(value_batch_size_override)
        if value_batch_size_override is not None
        else policy_batch_size
    )
    config["optim"]["batch_size_jt_value"] = value_batch_size
    metrics_log_every = int(config["optim"].get("metrics_log_every", 1))
    eval_cadence = int(config.get("eval", {}).get("cadence", 20000))
    eval_max_scenes = None
    if stage == "smoke":
        n_steps = min(n_steps, 4)
        vs_warmup_steps = min(vs_warmup_steps, 1)
        k_v = 1
        k_pi = 1
        bptt_t = min(bptt_t, 3)
        config["training"]["jt"]["bptt_T"] = bptt_t
        collect_every = 1
        episodes_per_collect = min(episodes_per_collect, 4)
        horizon = min(horizon, 8)
        policy_batch_size = min(policy_batch_size, 8)
        value_batch_size = min(value_batch_size, 8)
        eval_cadence = n_steps
        eval_max_scenes = smoke_eval_scenes

    buffers = make_replay_buffers(capacity=int(collection_cfg["buffer_cap"]))
    run_dir = _create_run_dir(output_root, config, run_seed)
    _initialize_run_dir(run_dir, config)
    writer = _make_summary_writer(run_dir / "tensorboard")
    scene_sampler = _train_scene_sampler(config)
    probe_scene = _make_probe_scene(config, system, train_device, train_dtype)
    sigma = float(config["schedules"]["sigma"]["init"])
    sigma_pi_base = float(config["schedules"]["sigma_pi"]["init"])
    start_time = time.time()
    best_cps = -float("inf")
    best_step = 0
    halt_reason = None
    last_row: dict[str, Any] | None = None
    last_logged_step = 0
    last_value_loss = float("nan")
    last_policy_loss = 0.0
    last_vs_grad = 0.0
    last_pi_grad = 0.0
    max_policy_leak = 0.0
    last_v_stats = CollectionStats(0, 0.0, sigma, sigma, 0.0, 0.0)
    last_pi_stats = CollectionStats(0, 0.0, sigma_pi_base, sigma_pi_base, 0.0, 0.0)

    _write_status(
        run_dir,
        stage=stage,
        phase="training",
        current_step=0,
        best_step=0,
        best_cps=best_cps,
        halt_reason=None,
    )

    for step in range(1, n_steps + 1):
        if step == 1 or step % max(1, collect_every) == 0:
            sigma_pi = _sigma_pi_at(config, sigma_pi_base, step)
            last_v_stats, last_pi_stats = collect_jt(
                system=system,
                policy_net=policy_net,
                value_net=target_value_net,
                scene_sampler=lambda rng: scene_sampler(rng, config, system.name),
                rng=np_rng,
                torch_generator=torch_rng,
                buffers=buffers,
                n_episodes=episodes_per_collect,
                max_steps=horizon,
                dt=float(config["env"]["dt"]),
                config=config,
                sigma_v=sigma,
                sigma_pi=sigma_pi,
                storage_device=train_device,
                storage_dtype=train_dtype,
            )
            sigma = last_v_stats.sigma_after

        schedule_step = max(0, step - vs_warmup_steps)
        schedule_total_steps = schedule_n_steps
        effective_steps = max(1, schedule_total_steps - vs_warmup_steps)
        schedule_step_clamped = min(schedule_step, effective_steps)
        lambda_disc = lambda_schedule_value(
            config["schedules"]["gamma_disc"],
            schedule_step_clamped,
            effective_steps,
            float(config["env"]["dt"]),
        )
        gamma_disc = gamma_from_lambda(lambda_disc, float(config["env"]["dt"]))
        target_rhs = schedule_value(
            config["schedules"]["target_rhs"],
            schedule_step,
            effective_steps,
        )

        value_scalars = _value_updates(
            system=system,
            value_net=value_net,
            target_value_net=target_value_net,
            optimizer=opt_vs,
            buffers=buffers,
            torch_generator=torch_rng,
            batch_size=value_batch_size,
            n_updates=k_v,
            lambda_disc=lambda_disc,
            target_rhs=target_rhs,
            config=config,
        )
        last_value_loss = value_scalars["L_V_total"]
        last_vs_grad = value_scalars["grad_norm_VS"]
        if not np.isfinite(last_value_loss):
            halt_reason = "nan_or_inf_L_V"
            break

        policy_scalars = _zero_policy_scalars()
        if step > vs_warmup_steps and k_pi > 0:
            policy_scalars = _policy_updates(
                system=system,
                policy_net=policy_net,
                value_net=value_net,
                optimizer=opt_pi,
                buffers=buffers,
                torch_generator=torch_rng,
                batch_size=policy_batch_size,
                n_updates=k_pi,
                config=config,
            )
            last_policy_loss = policy_scalars["L_pi_total"]
            last_pi_grad = policy_scalars["grad_norm_pi"]
            max_policy_leak = max(max_policy_leak, policy_scalars["grad_leak_VS_from_Lpi"])
            if not np.isfinite(last_policy_loss):
                halt_reason = "nan_or_inf_L_pi"
                break
            if policy_scalars["grad_leak_VS_from_Lpi"] > float(
                config["halt"]["vs_grad_leak_threshold"]
            ):
                halt_reason = "policy_gradient_leak"
                break

        row = _metrics_row(
            step=step,
            wallclock_s=time.time() - start_time,
            schedule_step=schedule_step_clamped,
            lambda_disc=lambda_disc,
            gamma_disc=gamma_disc,
            target_rhs=target_rhs,
            sigma=sigma,
            sigma_pi=_sigma_pi_at(config, sigma_pi_base, step),
            value_scalars=value_scalars,
            policy_scalars=policy_scalars,
            value_stats=last_v_stats,
            policy_stats=last_pi_stats,
            probe_spread=_probe_h_spread(system, value_net, probe_scene),
        )
        last_row = row
        if step % metrics_log_every == 0:
            _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [row])
            _write_tb_scalars(writer, "train", row, step)
            last_logged_step = step

        if step == n_steps or step % max(1, eval_cadence) == 0:
            eval_result = evaluate(
                JTPNCBFFramework(system, value_net, policy_net, config),
                _pool_path("inloop", config, system.name),
                config,
                mode="in_loop",
                step=step,
                ckpt_name=f"step_{step:06d}.pt",
                max_scenes=eval_max_scenes,
                include_lqr_baseline=True,
            )
            _record_eval(
                run_dir,
                writer,
                eval_result,
                step,
                config,
                system,
                value_net,
            )
            cps = float(eval_result.eval_row["cps"])
            if cps > best_cps + float(config["halt"]["early_stop_min_delta"]):
                best_cps = cps
                best_step = step
                _save_checkpoint(
                    run_dir / "checkpoints/best.pt",
                    value_net,
                    target_value_net,
                    policy_net,
                    opt_vs,
                    opt_pi,
                    config,
                    step,
                    best_cps,
                    best_step,
                )
            _save_checkpoint(
                run_dir / f"checkpoints/step_{step:06d}.pt",
                value_net,
                target_value_net,
                policy_net,
                opt_vs,
                opt_pi,
                config,
                step,
                best_cps,
                best_step,
            )
        _write_status(
            run_dir,
            stage=stage,
            phase="training",
            current_step=step,
            best_step=best_step,
            best_cps=best_cps,
            halt_reason=None,
        )

    final_step = step if "step" in locals() else 0
    if last_row is not None and last_logged_step != final_step:
        _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [last_row])
        _write_tb_scalars(writer, "train", last_row, final_step)
    _save_checkpoint(
        run_dir / "checkpoints/final.pt",
        value_net,
        target_value_net,
        policy_net,
        opt_vs,
        opt_pi,
        config,
        final_step,
        best_cps,
        best_step,
    )
    if not (run_dir / "checkpoints/best.pt").exists():
        _save_checkpoint(
            run_dir / "checkpoints/best.pt",
            value_net,
            target_value_net,
            policy_net,
            opt_vs,
            opt_pi,
            config,
            final_step,
            best_cps,
            best_step,
        )
    _write_status(
        run_dir,
        stage=stage,
        phase="halted" if halt_reason is not None else "done",
        current_step=final_step,
        best_step=best_step,
        best_cps=best_cps,
        halt_reason=halt_reason,
    )
    _write_report(
        run_dir,
        run_id=run_dir.name,
        git_commit=(run_dir / "git_commit.txt").read_text(encoding="utf-8").strip(),
        wallclock_s=time.time() - start_time,
        halt_reason=halt_reason,
    )
    writer.close()
    _auto_run_final_eval(
        run_dir,
        max_scenes=eval_max_scenes if stage == "smoke" else None,
    )
    _write_status(
        run_dir,
        stage=stage,
        phase="halted" if halt_reason is not None else "done",
        current_step=final_step,
        best_step=best_step,
        best_cps=best_cps,
        halt_reason=halt_reason,
    )
    _write_report(
        run_dir,
        run_id=run_dir.name,
        git_commit=(run_dir / "git_commit.txt").read_text(encoding="utf-8").strip(),
        wallclock_s=time.time() - start_time,
        halt_reason=halt_reason,
    )
    return JTTrainingResult(
        run_dir=run_dir,
        halted=halt_reason is not None,
        halt_reason=halt_reason,
        last_value_loss=last_value_loss,
        last_policy_loss=last_policy_loss,
        last_vs_grad_norm=last_vs_grad,
        last_pi_grad_norm=last_pi_grad,
        max_policy_grad_leak=max_policy_leak,
        last_unsafe_fraction=last_v_stats.unsafe_fraction,
        last_sigma=sigma,
    )


def run_value_refinement(
    *,
    checkpoint_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    refine_steps: int = 4000,
    seed: int | None = None,
    device: str = "auto",
    train_dtype_name: str = "float32",
    collection_filter: str = "hardnet",
    sigma_mode: str = "adaptive",
    fixed_sigma: float | None = None,
) -> JTTrainingResult:
    if refine_steps <= 0:
        raise ValueError(f"refine_steps must be positive, got {refine_steps}.")
    if collection_filter not in {"hardnet", "cbf_qp"}:
        raise ValueError(f"Unsupported value-refine collection filter: {collection_filter!r}")
    if sigma_mode not in {"adaptive", "fixed"}:
        raise ValueError(f"Unsupported value-refine sigma mode: {sigma_mode!r}")
    if sigma_mode == "fixed":
        if fixed_sigma is None:
            raise ValueError("fixed_sigma is required when sigma_mode='fixed'.")
        if fixed_sigma < 0.0:
            raise ValueError(f"fixed_sigma must be nonnegative, got {fixed_sigma}.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["run"] = dict(config["run"])
    config["run"]["version"] = __version__
    config["run"]["framework"] = "jt_pncbf"
    config["run"]["parent_checkpoint"] = str(checkpoint_path)
    config["run"]["value_refine_collection_filter"] = collection_filter
    config["run"]["value_refine_sigma_mode"] = sigma_mode
    if fixed_sigma is not None:
        config["run"]["value_refine_fixed_sigma"] = float(fixed_sigma)
    if seed is not None:
        config["run"]["seed"] = int(seed)

    run_seed = int(config["run"]["seed"])
    np_rng = np.random.default_rng(run_seed)
    torch.manual_seed(run_seed)
    train_device = _resolve_train_device(device)
    train_dtype = _resolve_train_dtype(train_dtype_name)
    torch_rng = torch.Generator(device=train_device)
    torch_rng.manual_seed(run_seed)
    system = make_system(config)

    value_net = ValueNetEnsemble(system.obs_dim, config).to(
        device=train_device,
        dtype=train_dtype,
    )
    value_net.load_state_dict(_state_to_dtype(checkpoint["v_s_state"], train_dtype))
    target_value_net = ValueNetEnsemble(system.obs_dim, config).to(
        device=train_device,
        dtype=train_dtype,
    )
    target_state = checkpoint.get("v_s_target_state", checkpoint["v_s_state"])
    target_value_net.load_state_dict(_state_to_dtype(target_state, train_dtype))
    target_value_net.requires_grad_(False)
    policy_net = ControlNet(system.obs_dim, system, config).to(
        device=train_device,
        dtype=train_dtype,
    )
    policy_net.load_state_dict(_state_to_dtype(checkpoint["pi_state"], train_dtype))
    policy_net.requires_grad_(False)
    policy_net.eval()

    opt_vs = optim.AdamW(
        value_net.parameters(),
        lr=float(config["optim"]["lr_VS"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    if "opt_vs_state" in checkpoint:
        opt_vs.load_state_dict(checkpoint["opt_vs_state"])
        _move_optimizer_state(opt_vs, train_device, train_dtype)
    opt_pi = optim.AdamW(
        policy_net.parameters(),
        lr=float(config["optim"]["lr_pi"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    if "opt_pi_state" in checkpoint:
        opt_pi.load_state_dict(checkpoint["opt_pi_state"])
        _move_optimizer_state(opt_pi, train_device, train_dtype)

    collection_cfg = dict(config["collection"]["jt"])
    collect_every = int(collection_cfg["collect_every"])
    episodes_per_collect = int(collection_cfg["episodes_per_collect"])
    horizon = int(config["training"]["oc_pncbf"]["horizon"])
    batch_size = int(config["optim"]["batch_size_jt"])
    k_v = int(config["training"]["jt"]["K_V"])
    metrics_log_every = int(config["optim"].get("metrics_log_every", 1))
    eval_cadence = int(config.get("eval", {}).get("cadence", 2000))
    source_step = int(checkpoint.get("step", 0))
    source_best_cps = float(checkpoint.get("best_cps", -float("inf")))
    source_best_step = int(checkpoint.get("best_step", source_step))
    original_n_steps = int(config["training"]["jt"]["n_steps"])
    vs_warmup_steps = int(config["training"]["jt"]["vs_warmup_steps"])
    effective_steps = max(1, original_n_steps - vs_warmup_steps)

    buffers = make_replay_buffers(capacity=int(collection_cfg["buffer_cap"]))
    run_dir = _create_run_dir(output_root, config, run_seed)
    _initialize_run_dir(run_dir, config)
    writer = _make_summary_writer(run_dir / "tensorboard")
    scene_sampler = _train_scene_sampler(config)
    probe_scene = _make_probe_scene(config, system, train_device, train_dtype)
    sigma = (
        float(fixed_sigma)
        if sigma_mode == "fixed"
        else float(config["schedules"]["sigma"]["init"])
    )
    sigma_pi = 0.0
    start_time = time.time()
    best_cps = -float("inf")
    best_step = 0
    halt_reason = None
    last_row: dict[str, Any] | None = None
    last_logged_step = 0
    last_value_loss = float("nan")
    last_vs_grad = 0.0
    last_v_stats = CollectionStats(0, 0.0, sigma, sigma, 0.0, 0.0)
    policy_stats = CollectionStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    _write_status(
        run_dir,
        stage="value_refine",
        phase="training",
        current_step=source_step,
        best_step=source_best_step,
        best_cps=source_best_cps,
        halt_reason=None,
    )

    for local_step in range(1, refine_steps + 1):
        global_step = source_step + local_step
        if local_step == 1 or local_step % max(1, collect_every) == 0:
            last_v_stats = collect_policy_rollouts(
                system=system,
                policy_net=policy_net,
                value_net=target_value_net,
                scene_sampler=lambda rng: scene_sampler(rng, config, system.name),
                rng=np_rng,
                torch_generator=torch_rng,
                n_episodes=episodes_per_collect,
                max_steps=horizon,
                dt=float(config["env"]["dt"]),
                buffer=buffers.value,
                config=config,
                sigma=sigma,
                storage_device=train_device,
                storage_dtype=train_dtype,
                collection_filter=collection_filter,
            )
            if sigma_mode == "fixed":
                sigma = float(fixed_sigma)
                last_v_stats = CollectionStats(
                    n_episodes=last_v_stats.n_episodes,
                    unsafe_fraction=last_v_stats.unsafe_fraction,
                    sigma_before=sigma,
                    sigma_after=sigma,
                    mean_projection=last_v_stats.mean_projection,
                    infeasible_fraction=last_v_stats.infeasible_fraction,
                )
            else:
                sigma = last_v_stats.sigma_after

        schedule_step = max(0, global_step - vs_warmup_steps)
        schedule_step_clamped = min(schedule_step, effective_steps)
        lambda_disc = lambda_schedule_value(
            config["schedules"]["gamma_disc"],
            schedule_step_clamped,
            effective_steps,
            float(config["env"]["dt"]),
        )
        gamma_disc = gamma_from_lambda(lambda_disc, float(config["env"]["dt"]))
        target_rhs = schedule_value(
            config["schedules"]["target_rhs"],
            schedule_step_clamped,
            effective_steps,
        )

        value_scalars = _value_updates(
            system=system,
            value_net=value_net,
            target_value_net=target_value_net,
            optimizer=opt_vs,
            buffers=buffers,
            torch_generator=torch_rng,
            batch_size=batch_size,
            n_updates=k_v,
            lambda_disc=lambda_disc,
            target_rhs=target_rhs,
            config=config,
        )
        last_value_loss = value_scalars["L_V_total"]
        last_vs_grad = value_scalars["grad_norm_VS"]
        if not np.isfinite(last_value_loss):
            halt_reason = "nan_or_inf_L_V"
            break

        row = _metrics_row(
            step=global_step,
            wallclock_s=time.time() - start_time,
            schedule_step=schedule_step_clamped,
            lambda_disc=lambda_disc,
            gamma_disc=gamma_disc,
            target_rhs=target_rhs,
            sigma=sigma,
            sigma_pi=sigma_pi,
            value_scalars=value_scalars,
            policy_scalars=_zero_policy_scalars(),
            value_stats=last_v_stats,
            policy_stats=policy_stats,
            probe_spread=_probe_h_spread(system, value_net, probe_scene),
        )
        last_row = row
        if global_step % metrics_log_every == 0:
            _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [row])
            _write_tb_scalars(writer, "train", row, global_step)
            last_logged_step = global_step

        if local_step == refine_steps or global_step % max(1, eval_cadence) == 0:
            eval_result = evaluate(
                JTPNCBFFramework(system, value_net, policy_net, config),
                _pool_path("inloop", config, system.name),
                config,
                mode="in_loop",
                step=global_step,
                ckpt_name=f"step_{global_step:06d}.pt",
                include_lqr_baseline=True,
            )
            _record_eval(
                run_dir,
                writer,
                eval_result,
                global_step,
                config,
                system,
                value_net,
            )
            cps = float(eval_result.eval_row["cps"])
            if cps > best_cps + float(config["halt"]["early_stop_min_delta"]):
                best_cps = cps
                best_step = global_step
                _save_checkpoint(
                    run_dir / "checkpoints/best.pt",
                    value_net,
                    target_value_net,
                    policy_net,
                    opt_vs,
                    opt_pi,
                    config,
                    global_step,
                    best_cps,
                    best_step,
                )
            _save_checkpoint(
                run_dir / f"checkpoints/step_{global_step:06d}.pt",
                value_net,
                target_value_net,
                policy_net,
                opt_vs,
                opt_pi,
                config,
                global_step,
                best_cps,
                best_step,
            )
        _write_status(
            run_dir,
            stage="value_refine",
            phase="training",
            current_step=global_step,
            best_step=best_step,
            best_cps=best_cps,
            halt_reason=None,
        )

    final_step = source_step + local_step if "local_step" in locals() else source_step
    if last_row is not None and last_logged_step != final_step:
        _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [last_row])
        _write_tb_scalars(writer, "train", last_row, final_step)
    _save_checkpoint(
        run_dir / "checkpoints/final.pt",
        value_net,
        target_value_net,
        policy_net,
        opt_vs,
        opt_pi,
        config,
        final_step,
        best_cps,
        best_step,
    )
    if not (run_dir / "checkpoints/best.pt").exists():
        _save_checkpoint(
            run_dir / "checkpoints/best.pt",
            value_net,
            target_value_net,
            policy_net,
            opt_vs,
            opt_pi,
            config,
            final_step,
            best_cps,
            best_step,
        )
    _write_status(
        run_dir,
        stage="value_refine",
        phase="halted" if halt_reason is not None else "done",
        current_step=final_step,
        best_step=best_step,
        best_cps=best_cps,
        halt_reason=halt_reason,
    )
    _write_report(
        run_dir,
        run_id=run_dir.name,
        git_commit=(run_dir / "git_commit.txt").read_text(encoding="utf-8").strip(),
        wallclock_s=time.time() - start_time,
        halt_reason=halt_reason,
    )
    writer.close()
    return JTTrainingResult(
        run_dir=run_dir,
        halted=halt_reason is not None,
        halt_reason=halt_reason,
        last_value_loss=last_value_loss,
        last_policy_loss=0.0,
        last_vs_grad_norm=last_vs_grad,
        last_pi_grad_norm=0.0,
        max_policy_grad_leak=0.0,
        last_unsafe_fraction=last_v_stats.unsafe_fraction,
        last_sigma=sigma,
    )


def _value_updates(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    optimizer: optim.Optimizer,
    buffers: JTReplayBuffers,
    torch_generator: torch.Generator | None,
    batch_size: int,
    n_updates: int,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    totals = []
    reaches = []
    grad_norms = []
    for _ in range(n_updates):
        batch = buffers.value.sample_tensor_batch(batch_size, generator=torch_generator)
        result = value_loss(
            system=system,
            value_net=value_net,
            target_value_net=target_value_net,
            batch=batch,
            lambda_disc=lambda_disc,
            target_rhs=target_rhs,
            config=config,
        )
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        grad = nn.utils.clip_grad_norm_(
            value_net.parameters(),
            max_norm=float(config["optim"]["grad_clip"]),
        )
        optimizer.step()
        _polyak_update(
            target_value_net,
            value_net,
            tau=float(config["optim"]["tau_polyak"]),
        )
        totals.append(result.total.detach())
        reaches.append(result.reach.detach())
        grad_norms.append(torch.as_tensor(grad, device=result.total.device, dtype=result.total.dtype))
    return _host_scalars(
        {
            "L_R": torch.stack(reaches).mean(),
            "L_V_total": torch.stack(totals).mean(),
            "grad_norm_VS": torch.stack(grad_norms).mean(),
        }
    )


def _policy_updates(
    *,
    system: System,
    policy_net: ControlNet,
    value_net: ValueNetEnsemble,
    optimizer: optim.Optimizer,
    buffers: JTReplayBuffers,
    torch_generator: torch.Generator | None,
    batch_size: int,
    n_updates: int,
    config: Mapping[str, Any],
) -> dict[str, float]:
    accum: dict[str, list[Tensor]] = {
        "L_in_task": [],
        "L_anorm": [],
        "L_smooth": [],
        "L_satex": [],
        "L_pretanh": [],
        "L_out": [],
        "L_pi_total": [],
        "grad_norm_pi": [],
        "abs_action_mean": [],
        "abs_action_max": [],
        "satfrac_a_phi": [],
    }
    leaks = []
    for _ in range(n_updates):
        batch = buffers.policy.sample_tensor_batch(batch_size, generator=torch_generator)
        optimizer.zero_grad(set_to_none=True)
        value_net.zero_grad(set_to_none=True)
        result = policy_bptt_loss(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            batch=batch,
            config=config,
        )
        result.total.backward()
        leak = grad_norm(value_net.parameters())
        grad = nn.utils.clip_grad_norm_(
            policy_net.parameters(),
            max_norm=float(config["optim"]["grad_clip"]),
        )
        optimizer.step()
        leaks.append(leak)
        accum["L_in_task"].append(result.task)
        accum["L_anorm"].append(result.action_norm)
        accum["L_smooth"].append(result.smoothness)
        accum["L_satex"].append(result.saturation_excess)
        accum["L_pretanh"].append(result.pretanh)
        accum["L_out"].append(result.outside)
        accum["L_pi_total"].append(result.total.detach())
        accum["grad_norm_pi"].append(torch.as_tensor(grad, device=result.total.device, dtype=result.total.dtype))
        accum["abs_action_mean"].append(result.action_abs_mean)
        accum["abs_action_max"].append(result.action_abs_max)
        accum["satfrac_a_phi"].append(result.satfrac_a_phi)
    scalars = _host_scalars({key: torch.stack(values).mean() for key, values in accum.items()})
    scalars["grad_leak_VS_from_Lpi"] = max(leaks) if leaks else 0.0
    return scalars


def _zero_policy_scalars() -> dict[str, float]:
    return {
        "L_in_task": 0.0,
        "L_anorm": 0.0,
        "L_smooth": 0.0,
        "L_satex": 0.0,
        "L_pretanh": 0.0,
        "L_out": 0.0,
        "L_pi_total": 0.0,
        "grad_norm_pi": 0.0,
        "grad_leak_VS_from_Lpi": 0.0,
        "abs_action_mean": 0.0,
        "abs_action_max": 0.0,
        "satfrac_a_phi": 0.0,
    }


def load_effective_config() -> dict[str, Any]:
    base = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    exp = yaml.safe_load(EXP_CONFIG_PATH.read_text(encoding="utf-8"))
    return _deep_merge(base, exp)


def make_system(config: Mapping[str, Any]) -> System:
    system_name = str(config["run"]["system"])
    if system_name == "double_integrator":
        return DoubleIntegrator(config)
    if system_name == "unicycle":
        return Unicycle(config)
    raise ValueError(f"Unsupported system: {system_name!r}")


def _resolve_train_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported training device: {device!r}")


def _resolve_train_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported training dtype: {dtype_name!r}")


def load_framework_from_checkpoint(
    checkpoint_path: Path,
    config_overrides: Mapping[str, Any] | None = None,
) -> tuple[JTPNCBFFramework, Mapping[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = deepcopy(checkpoint["config"])
    if config_overrides:
        config = _deep_merge(config, config_overrides)
    system = make_system(config)
    first_tensor = next(iter(checkpoint["v_s_state"].values()))
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=first_tensor.dtype)
    value_net.load_state_dict(checkpoint["v_s_state"])
    value_net.eval()
    policy_net = ControlNet(system.obs_dim, system, config).to(dtype=first_tensor.dtype)
    policy_net.load_state_dict(checkpoint["pi_state"])
    policy_net.eval()
    return JTPNCBFFramework(system, value_net, policy_net, config), config, checkpoint


def _state_to_dtype(
    state: Mapping[str, Tensor],
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    return {
        key: value.to(dtype=dtype) if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in state.items()
    }


def _move_optimizer_state(
    optimizer: optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            target_dtype = dtype if value.is_floating_point() else value.dtype
            state[key] = value.to(device=device, dtype=target_dtype)


def _metrics_row(
    *,
    step: int,
    wallclock_s: float,
    schedule_step: int,
    lambda_disc: float,
    gamma_disc: float,
    target_rhs: float,
    sigma: float,
    sigma_pi: float,
    value_scalars: Mapping[str, float],
    policy_scalars: Mapping[str, float],
    value_stats: CollectionStats,
    policy_stats: CollectionStats,
    probe_spread: Mapping[str, float],
) -> dict[str, Any]:
    row = {column: 0.0 for column in METRIC_COLUMNS}
    row.update(
        {
            "step": int(step),
            "wallclock_s": float(wallclock_s),
            "schedule_step": int(schedule_step),
            "lambda_disc_active": float(lambda_disc),
            "gamma_disc_active": float(gamma_disc),
            "target_rhs_active": float(target_rhs),
            "sigma": float(sigma),
            "sigma_pi": float(sigma_pi),
            "rho_unsafe_v": float(value_stats.unsafe_fraction),
            "rho_unsafe_pi": float(policy_stats.unsafe_fraction),
            "collect_proj_mag": float(value_stats.mean_projection),
            "collect_infeasible": float(value_stats.infeasible_fraction),
            **{key: float(value) for key, value in value_scalars.items()},
            **{key: float(value) for key, value in policy_scalars.items()},
            **{key: float(value) for key, value in probe_spread.items()},
        }
    )
    return row


def _make_probe_scene(
    config: Mapping[str, Any],
    system: System,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch

    pool = load_pool(_pool_path("inloop", config, system.name))
    scenes = pool.scenes[: min(64, len(pool.scenes))]
    scene = batch_scenes(scenes, device=device, dtype=dtype)
    return scene, initial_states_from_batch(scene)


def _probe_h_spread(
    system: System,
    value_net: ValueNetEnsemble,
    probe: Any,
) -> dict[str, float]:
    scene, x = probe
    with torch.no_grad():
        h = value_net.deployed_h(system.observation(x, scene))
    return {
        "probe_h_min": float(h.min().detach().cpu().item()),
        "probe_h_max": float(h.max().detach().cpu().item()),
        "probe_h_mean": float(h.mean().detach().cpu().item()),
    }


def _save_checkpoint(
    path: Path,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    policy_net: ControlNet,
    opt_vs: optim.Optimizer,
    opt_pi: optim.Optimizer,
    config: Mapping[str, Any],
    step: int,
    best_cps: float,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": __version__,
            "framework": "jt_pncbf",
            "step": int(step),
            "best_cps": float(best_cps),
            "best_step": int(best_step),
            "v_s_state": value_net.state_dict(),
            "v_s_target_state": target_value_net.state_dict(),
            "pi_state": policy_net.state_dict(),
            "opt_vs_state": opt_vs.state_dict(),
            "opt_pi_state": opt_pi.state_dict(),
            "config": _plain_data(config),
        },
        path,
    )


def _create_run_dir(output_root: Path, config: Mapping[str, Any], seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{config['run']['version']}__{timestamp}__seed{seed}"
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}_{suffix}"
        suffix += 1
    return run_dir


def _initialize_run_dir(run_dir: Path, config: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "tensorboard").mkdir()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(_plain_data(config), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(_git_commit_text() + "\n", encoding="utf-8")
    _init_csv(run_dir / "metrics.csv", METRIC_COLUMNS)
    _init_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS)
    _init_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS)
    _write_pool_manifest_copy(run_dir, config)


def _record_eval(
    run_dir: Path,
    writer: Any,
    eval_result: EvaluationResult,
    step: int,
    config: Mapping[str, Any],
    system: System,
    value_net: ValueNetEnsemble,
) -> None:
    _append_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS, [eval_result.eval_row])
    _append_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, eval_result.episode_rows)
    from src.eval.run_full import (
        log_png_to_tensorboard,
        write_cbf_contour_figure,
        write_trajectory_figures,
    )

    write_trajectory_figures(
        run_dir=run_dir,
        eval_result=eval_result,
        config=config,
        system_name=system.name,
        u_bounds=system.u_bounds,
        role="JT in-loop eval",
        output_dir=run_dir / "figures/inloop",
        filename_template=f"step_{step:06d}_grid_{{letter}}.png",
    )
    contour_path = write_cbf_contour_figure(
        eval_result=eval_result,
        config=config,
        system=system,
        value_net=value_net,
        output_path=run_dir / "figures/inloop" / f"step_{step:06d}_cbf_contour.png",
        role="JT in-loop eval CBF contour",
    )
    log_png_to_tensorboard(
        writer,
        f"eval/in_loop/step_{step:06d}_cbf_contour",
        contour_path,
        step,
    )
    _write_tb_scalars(writer, f"eval/{eval_result.eval_row['mode']}", eval_result.eval_row, step)


def _write_pool_manifest_copy(run_dir: Path, config: Mapping[str, Any]) -> None:
    system_name = str(config["run"]["system"])
    manifests: dict[str, Any] = {}
    for pool_name in ("inloop", "full"):
        path = _pool_path(pool_name, config, system_name).with_suffix(".manifest.json")
        manifests[pool_name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )
    (run_dir / "pool_manifest.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pool_path(pool_name: str, config: Mapping[str, Any], system_name: str) -> Path:
    from src.eval.build_pools import obstacle_distribution_name, pool_stem

    if pool_name == "inloop":
        n_scenes = int(config["eval"]["in_loop"]["n"])
        seed = int(config["eval"]["in_loop"]["seed"])
    elif pool_name == "full":
        n_scenes = int(config["eval"]["full"]["n"])
        seed = int(config["eval"]["full"]["seed"])
    else:
        raise ValueError(f"Unknown pool: {pool_name!r}")
    return POOL_DIR / (
        f"{pool_stem(pool_name, system_name, n_scenes, seed, obstacle_distribution_name(config))}.pkl"
    )


def _train_scene_sampler(config: Mapping[str, Any]) -> Any:
    from src.eval.build_pools import obstacle_distribution_name

    distribution = obstacle_distribution_name(config)
    if distribution == "random":
        return sample_train_scene
    if distribution == "fixed_centered":
        return sample_train_fixed_scene
    raise ValueError(f"Unsupported obstacle distribution: {distribution!r}")


def _write_status(
    run_dir: Path,
    *,
    stage: str,
    phase: str,
    current_step: int,
    best_step: int,
    best_cps: float,
    halt_reason: str | None,
) -> None:
    status = {
        "stage": stage,
        "phase": phase,
        "current_step": int(current_step),
        "best_step": int(best_step),
        "best_cps": float(best_cps),
        "halt_reason": halt_reason,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(
    run_dir: Path,
    *,
    run_id: str,
    git_commit: str,
    wallclock_s: float,
    halt_reason: str | None,
) -> None:
    eval_rows = _read_csv(run_dir / "eval_metrics.csv")
    inloop_rows = [row for row in eval_rows if row.get("mode") == "in_loop"]
    peak = (
        max((float(row["cps"]), int(row["step"])) for row in inloop_rows)
        if inloop_rows
        else (float("nan"), 0)
    )
    lines = [
        "# JT-PNCBF Run Report",
        "",
        "## Run identity",
        f"- run_id: `{run_id}`",
        f"- version: `{__version__}`",
        f"- git_commit: `{git_commit}`",
        f"- wallclock_s: `{wallclock_s:.3f}`",
        "",
        "## In-loop eval",
        f"- peak_cps: `{peak[0]:.6f}`",
        f"- peak_step: `{peak[1]}`",
        "",
        "## Halt status",
        f"- halt_reason: `{halt_reason}`",
        "",
        "## File index",
        "- metrics: `metrics.csv`",
        "- eval_metrics: `eval_metrics.csv`",
        "- eval_episodes: `eval_episodes.csv`",
        "- checkpoints: `checkpoints/best.pt`, `checkpoints/final.pt`",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sigma_pi_at(config: Mapping[str, Any], sigma_pi_base: float, step: int) -> float:
    decay_steps = int(config["schedules"]["sigma_pi"].get("decay_steps", 1))
    if decay_steps <= 0:
        return 0.0
    frac = max(0.0, 1.0 - float(step) / float(decay_steps))
    return float(sigma_pi_base) * frac


def _polyak_update(
    target_value_net: ValueNetEnsemble,
    value_net: ValueNetEnsemble,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target_value_net.parameters(),
            value_net.parameters(),
            strict=True,
        ):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _host_scalars(values: Mapping[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu().item())
        for key, value in values.items()
    }


def _init_csv(path: Path, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        csv.DictWriter(file_obj, fieldnames=columns).writeheader()


def _append_csv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _auto_run_final_eval(run_dir: Path, *, max_scenes: int | None = None) -> None:
    # SIGINT/user-killed runs are out of scope for trainer-side final eval; use
    # `python -m src.eval.run_full --run-dir ... --ckpt ...` for those manual cases.
    best_ckpt = run_dir / "checkpoints/best.pt"
    if _has_final_eval_row(run_dir / "eval_metrics.csv", best_ckpt.name):
        print(f"Final eval already present for {best_ckpt.name}; skipping trainer-side final eval.")
        return

    from src.eval.run_full import run_full_eval

    run_full_eval(run_dir, ckpt_path=best_ckpt, max_scenes=max_scenes)


def _has_final_eval_row(path: Path, ckpt_name: str) -> bool:
    if not path.exists():
        return False
    for row in _read_csv(path):
        if row.get("mode") == "final" and row.get("ckpt_name") == ckpt_name:
            return True
    return False


def _write_tb_scalars(writer: Any, prefix: str, row: Mapping[str, Any], step: int) -> None:
    for key, value in row.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"{prefix}/{key}", float(value), int(step))


def _make_summary_writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


class _NullSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_image(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _git_commit_text() -> str:
    commit = _git_command(["git", "rev-parse", "HEAD"])
    dirty = _git_command(["git", "status", "--porcelain"])
    suffix = " DIRTY" if dirty else ""
    return f"{commit or 'unknown'}{suffix}"


def _git_command(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Train JT-PNCBF.")
    parser.add_argument("--stage", choices=["smoke", "full", "value_refine"], default="full")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-eval-scenes", type=int, default=2)
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--refine-steps", type=int, default=4000)
    parser.add_argument("--jt-n-steps", type=int, default=None)
    parser.add_argument("--value-batch-size", type=int, default=None)
    parser.add_argument("--schedule-n-steps", type=int, default=None)
    parser.add_argument(
        "--value-refine-collection-filter",
        choices=["hardnet", "cbf_qp"],
        default="hardnet",
    )
    parser.add_argument(
        "--value-refine-sigma-mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
    )
    parser.add_argument("--value-refine-fixed-sigma", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--train-dtype",
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--obstacle-distribution",
        choices=["random", "fixed_centered"],
        default=None,
    )
    args = parser.parse_args()
    if args.stage == "value_refine":
        if args.init_ckpt is None:
            parser.error("--stage value_refine requires --init-ckpt")
        result = run_value_refinement(
            checkpoint_path=args.init_ckpt,
            output_root=args.output_root,
            refine_steps=args.refine_steps,
            seed=args.seed,
            device=args.device,
            train_dtype_name=args.train_dtype,
            collection_filter=args.value_refine_collection_filter,
            sigma_mode=args.value_refine_sigma_mode,
            fixed_sigma=args.value_refine_fixed_sigma,
        )
    else:
        result = run_training(
            stage=args.stage,
            output_root=args.output_root,
            seed=args.seed,
            smoke_eval_scenes=args.smoke_eval_scenes,
            device=args.device,
            train_dtype_name=args.train_dtype,
            obstacle_distribution=args.obstacle_distribution,
            n_steps_override=args.jt_n_steps,
            value_batch_size_override=args.value_batch_size,
            schedule_n_steps_override=args.schedule_n_steps,
        )
    print(result.run_dir)
    return 0 if result.halt_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
