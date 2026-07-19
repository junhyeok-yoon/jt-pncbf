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
from src.common.filter_cbfqp import CBFQPFilter
from src.common.system import System
from src.common.quadrotor_barrier import lg_authority_loss
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import BatchedScene
from src.envs.scene_init import sample_train_scene
from src.envs.scene_init_fixed import sample_train_fixed_scene
from src.envs.unicycle import Unicycle
from src.eval.build_pools import obstacle_distribution_name, pool_stem
from src.eval.evaluate import (
    EVAL_EPISODE_COLUMNS,
    EVAL_METRIC_COLUMNS,
    EvaluationResult,
    evaluate,
)
from src.frameworks.oc_pncbf.collection import OCReplayBuffer, collect
from src.frameworks.oc_pncbf.value_target import (
    gamma_from_lambda,
    lambda_schedule_value,
    pncbf_target,
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
    "n_sched",
    "lambda_disc_active",
    "gamma_disc",
    "gamma_disc_active",
    "target_rhs",
    "sigma",
    "sigma_pi",
    "L_R",
    "L_A",
    "L_C",
    "L_V_total",
    "L_in_task",
    "L_anorm",
    "L_smooth",
    "L_satex",
    "L_pretanh",
    "L_out",
    "L_pi_total",
    "grad_norm_VS",
    "L_lg_raw",              # v2.6.0 epsilon_g (R2): weighted-in raw penalty + ||L_g V_hat|| gate stats
    "lg_min",
    "lg_median",
    "lg_degen_frac",
    "grad_norm_pi",
    "grad_leak_VS_from_Lpi",
    "proj_mag_ema",
    "sigma_pin_counter",
    "abs_action_mean",
    "abs_action_max",
    "satfrac_a_phi",
]


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    smoke_grad_norm: float | None
    best_cps: float
    best_step: int
    final_l_r: float
    halted: bool
    halt_reason: str | None


class OCPNCBFFramework:
    def __init__(
        self,
        system: System,
        value_net: ValueNetEnsemble,
        config: Mapping[str, Any],
    ) -> None:
        self.system = system
        self.value_net = value_net
        self.config = config
        self._filter = CBFQPFilter(system, self._h_fn, config)

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        goal = _scene_goal_batch(scene, x)
        return self.system.lqr_action(x, goal)

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor, Tensor]:
        return self._filter(x, scene, u_nom)

    def value(self, x: Tensor, scene: Any) -> Tensor:
        return self._h_fn(x, scene)

    def _h_fn(self, x: Tensor, scene: Any) -> Tensor:
        if isinstance(scene, BatchedScene):
            obs = self.system.observation(x, scene)
            return self.value_net.deployed_h(obs)
        if isinstance(scene, list):
            obs = torch.cat(
                [
                    self.system.observation(x[index : index + 1], item)
                    for index, item in enumerate(scene)
                ],
                dim=0,
            )
            return self.value_net.deployed_h(obs)
        return make_h_fn(self.value_net, self.system)(x, scene)


def _scene_goal_batch(scene: Any, x: Tensor) -> Tensor:
    if isinstance(scene, BatchedScene):
        return scene.goal
    if isinstance(scene, list):
        goals = [
            torch.as_tensor(item.goal, dtype=x.dtype, device=x.device)
            for item in scene
        ]
        return torch.stack(goals, dim=0)
    goal = torch.as_tensor(scene.goal, dtype=x.dtype, device=x.device)
    return goal.unsqueeze(0).expand(x.shape[0], -1)


def run_training(
    *,
    stage: str = "full",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int | None = None,
    eval_every_epochs: int | None = None,
    smoke_eval_scenes: int = 20,
    device: str = "auto",
    train_dtype_name: str = "float32",
    system: str | None = None,
    obstacle_distribution: str | None = None,
) -> TrainingResult:
    config = load_effective_config()
    config["run"]["version"] = __version__
    config["run"]["framework"] = "oc_pncbf"
    if system is not None:           # v2.6.0: explicit system override (exp_config default is quadrotor_planar)
        config["run"]["system"] = system
    if seed is not None:
        config["run"]["seed"] = int(seed)
    if obstacle_distribution is not None:
        config["env"]["obstacle_distribution"] = obstacle_distribution
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
    optimizer = optim.AdamW(
        value_net.parameters(),
        lr=float(config["optim"]["lr_VS"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    run_dir = _create_run_dir(output_root, config, run_seed)
    _initialize_run_dir(run_dir, config)
    writer = _make_summary_writer(run_dir / "tensorboard")
    start_time = time.time()

    train_cfg = config["training"]["oc_pncbf"]
    collection_cfg = config["collection"]["oc_pncbf"]
    epochs = int(train_cfg["epochs"])
    grad_steps_per_epoch = int(train_cfg["grad_steps_per_epoch"])
    collect_size = int(collection_cfg["collect_size"])
    batch_size = int(config["optim"]["batch_size_oc"])
    metrics_log_every = int(config["optim"].get("metrics_log_every", 1))
    if metrics_log_every <= 0:
        raise ValueError("optim.metrics_log_every must be a positive integer.")
    horizon = int(train_cfg["horizon"])
    schedule_epochs = int(config["schedules"]["gamma_disc"].get("total_epochs", epochs))
    if eval_every_epochs is None:
        cadence_steps = int(config.get("eval", {}).get("cadence", 2000))
        eval_every_epochs = max(1, cadence_steps // max(1, grad_steps_per_epoch))
    eval_max_scenes = None
    if stage == "smoke":
        epochs = min(epochs, 2)
        grad_steps_per_epoch = min(grad_steps_per_epoch, 5)
        collect_size = min(collect_size, 20)
        batch_size = min(batch_size, 64)
        horizon = min(horizon, 50)
        eval_every_epochs = epochs
        eval_max_scenes = smoke_eval_scenes

    buffer = OCReplayBuffer(capacity=int(collection_cfg["buffer_capacity"]))
    global_step = 0
    best_cps = -float("inf")
    best_step = 0
    final_l_r = float("nan")
    smoke_grad_norm = None
    halt_reason = None
    inloop_pool_path = _pool_path("inloop", config, system.name)
    scene_sampler = _train_scene_sampler(config)
    last_metrics_row: dict[str, Any] | None = None
    last_logged_metrics_step = 0

    _write_status(
        run_dir,
        stage=stage,
        phase="training",
        current_step=0,
        best_step=0,
        best_cps=best_cps,
        halt_reason=None,
    )

    for epoch in range(1, epochs + 1):
        schedule_epoch = epoch - 1
        lambda_disc = lambda_schedule_value(
            config["schedules"]["gamma_disc"],
            schedule_epoch,
            schedule_epochs,
            float(config["env"]["dt"]),
        )
        gamma_disc = gamma_from_lambda(lambda_disc, float(config["env"]["dt"]))
        target_rhs = schedule_value(
            config["schedules"]["target_rhs"],
            schedule_epoch,
            schedule_epochs,
        )
        collect(
            system=system,
            scene_sampler=lambda rng: scene_sampler(rng, config, system.name),
            rng=np_rng,
            n_episodes=collect_size,
            max_steps=horizon,
            dt=float(config["env"]["dt"]),
            buffer=buffer,
            h_scale=float(config["env"]["h_scale"]),
            storage_device=train_device,
            storage_dtype=train_dtype,
            config=config,
        )

        for _ in range(grad_steps_per_epoch):
            step_result = _value_step(
                system=system,
                value_net=value_net,
                target_value_net=target_value_net,
                optimizer=optimizer,
                buffer=buffer,
                torch_generator=torch_rng,
                batch_size=batch_size,
                lambda_disc=lambda_disc,
                target_rhs=target_rhs,
                config=config,
            )
            value_scalars = _value_step_scalars(step_result)
            final_l_r = value_scalars["L_R"]
            if not np.isfinite(value_scalars["L_V_total"]):
                halt_reason = "nan_or_inf_L_V"
                break
            if smoke_grad_norm is None:
                smoke_grad_norm = value_scalars["grad_norm_VS"]
                if stage == "smoke" and smoke_grad_norm <= 0.0:
                    halt_reason = "smoke_value_grad_zero"
                    break
            _polyak_update(
                target_value_net,
                value_net,
                tau=float(config["optim"]["tau_polyak"]),
            )
            global_step += 1
            metrics_row = _metrics_row(
                step=global_step,
                wallclock_s=time.time() - start_time,
                n_sched=schedule_epoch,
                lambda_disc=lambda_disc,
                gamma_disc=gamma_disc,
                target_rhs=target_rhs,
                step_result=value_scalars,
            )
            last_metrics_row = metrics_row
            if global_step % metrics_log_every == 0:
                _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [metrics_row])
                _write_tb_scalars(writer, "train", metrics_row, global_step)
                last_logged_metrics_step = global_step
        if halt_reason is not None:
            break

        should_eval = epoch == epochs or epoch % max(1, eval_every_epochs) == 0
        if should_eval:
            framework = OCPNCBFFramework(system, value_net, config)
            eval_result = evaluate(
                framework,
                inloop_pool_path,
                config,
                mode="in_loop",
                step=global_step,
                ckpt_name=f"step_{global_step:06d}.pt",
                max_scenes=eval_max_scenes,
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
                    optimizer,
                    config,
                    epoch,
                    global_step,
                    best_cps,
                    best_step,
                )
            _save_checkpoint(
                run_dir / f"checkpoints/step_{global_step:06d}.pt",
                value_net,
                target_value_net,
                optimizer,
                config,
                epoch,
                global_step,
                best_cps,
                best_step,
            )
            if (
                stage != "smoke"
                and epoch >= max(1, eval_every_epochs)
                and cps < float(config["halt"]["cps_floor"])
            ):
                halt_reason = "cps_floor"
            patience = int(config["halt"]["early_stop_patience"])
            if patience > 0 and global_step - best_step >= patience:
                halt_reason = "early_stop_patience"
            _write_status(
                run_dir,
                stage=stage,
                phase="training" if halt_reason is None else "halted",
                current_step=global_step,
                best_step=best_step,
                best_cps=best_cps,
                halt_reason=halt_reason,
            )
        if halt_reason is not None:
            break

    if last_metrics_row is not None and last_logged_metrics_step != global_step:
        _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS, [last_metrics_row])
        _write_tb_scalars(writer, "train", last_metrics_row, global_step)

    _save_checkpoint(
        run_dir / "checkpoints/final.pt",
        value_net,
        target_value_net,
        optimizer,
        config,
        epoch if "epoch" in locals() else 0,
        global_step,
        best_cps,
        best_step,
    )
    if not (run_dir / "checkpoints/best.pt").exists():
        _save_checkpoint(
            run_dir / "checkpoints/best.pt",
            value_net,
            target_value_net,
            optimizer,
            config,
            epoch if "epoch" in locals() else 0,
            global_step,
            best_cps,
            best_step,
        )
    _write_status(
        run_dir,
        stage=stage,
        phase="halted" if halt_reason is not None else "done",
        current_step=global_step,
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
    return TrainingResult(
        run_dir=run_dir,
        smoke_grad_norm=smoke_grad_norm,
        best_cps=best_cps,
        best_step=best_step,
        final_l_r=final_l_r,
        halted=halt_reason is not None,
        halt_reason=halt_reason,
    )


def load_framework_from_checkpoint(
    checkpoint_path: Path,
    config_overrides: Mapping[str, Any] | None = None,
) -> tuple[OCPNCBFFramework, Mapping[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = deepcopy(checkpoint["config"])
    if config_overrides:
        config = _deep_merge(config, config_overrides)
    system = make_system(config)
    first_tensor = next(iter(checkpoint["v_s_state"].values()))
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=first_tensor.dtype)
    value_net.load_state_dict(checkpoint["v_s_state"])
    value_net.eval()
    return OCPNCBFFramework(system, value_net, config), config, checkpoint


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
    if system_name == "quadrotor_planar":
        from src.envs.quadrotor_planar import QuadrotorPlanar
        return QuadrotorPlanar(config)
    if system_name == "quadrotor_3d":
        from src.envs.quadrotor_3d import QuadrotorQuad3D
        return QuadrotorQuad3D(config)
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


def _value_step(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    optimizer: optim.Optimizer,
    buffer: OCReplayBuffer,
    torch_generator: torch.Generator | None,
    batch_size: int,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> dict[str, Tensor]:
    batch = buffer.sample_tensor_batch(batch_size, generator=torch_generator)
    obs = system.observation(batch.states, batch.scene)
    targets = _targets_for_tensor_batch(
        system=system,
        target_value_net=target_value_net,
        batch=batch,
        lambda_disc=lambda_disc,
        target_rhs=target_rhs,
        config=config,
    )

    optimizer.zero_grad(set_to_none=True)
    prediction = value_net(obs)
    loss_r = torch.mean((prediction - targets.unsqueeze(1)) ** 2)
    loss_v = float(config["loss"]["value"]["lambda_R"]) * loss_r
    # v2.6.0 R2 epsilon_g: protect ||L_g V_hat|| on B_0 (note O3). Quadrotor only; weight 0 => skipped.
    lg_auth_weight = float(config["loss"]["value"].get("lg_authority", {}).get("weight", 0.0))
    lg_diag = {"lg_min": 0.0, "lg_median": 0.0, "lg_degen_frac": 0.0}
    lg_raw = loss_r.detach().new_zeros(())
    if lg_auth_weight > 0.0 and getattr(system, "name", None) == "quadrotor_planar":
        lg_loss, lg_diag = lg_authority_loss(system, value_net, config, torch_generator)
        loss_v = loss_v + lg_auth_weight * lg_loss
        lg_raw = lg_loss.detach()
    loss_v.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        value_net.parameters(),
        max_norm=float(config["optim"]["grad_clip"]),
    )
    optimizer.step()
    return {
        "L_R": loss_r.detach(),
        "L_A": loss_r.detach().new_zeros(()),
        "L_C": loss_r.detach().new_zeros(()),
        "L_V_total": loss_v.detach(),
        "L_lg_raw": lg_raw,
        "lg_min": torch.as_tensor(lg_diag["lg_min"], device=loss_v.device, dtype=loss_v.dtype),
        "lg_median": torch.as_tensor(lg_diag["lg_median"], device=loss_v.device, dtype=loss_v.dtype),
        "lg_degen_frac": torch.as_tensor(lg_diag["lg_degen_frac"], device=loss_v.device, dtype=loss_v.dtype),
        "grad_norm_VS": torch.as_tensor(
            grad_norm,
            device=loss_v.device,
            dtype=loss_v.dtype,
        ).detach(),
    }


def _targets_for_tensor_batch(
    *,
    system: System,
    target_value_net: ValueNetEnsemble,
    batch: Any,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
) -> Tensor:
    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_value_net.target_h(tail_obs)
    targets = pncbf_target(
        batch.h_sequence,
        lambda_disc,
        float(config["env"]["dt"]),
        target_rhs,
        bootstrap_tail,
    ).detach()
    return targets.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)


def _value_step_scalars(step_result: Mapping[str, Tensor]) -> dict[str, float]:
    keys = ["L_R", "L_A", "L_C", "L_V_total", "grad_norm_VS"]
    if "lg_min" in step_result:                     # v2.6.0 epsilon_g diagnostics (quadrotor)
        keys = keys + ["L_lg_raw", "lg_min", "lg_median", "lg_degen_frac"]
    values = torch.stack([step_result[key].reshape(()) for key in keys])
    host_values = values.detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, host_values, strict=True)}


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


def _metrics_row(
    *,
    step: int,
    wallclock_s: float,
    n_sched: int,
    lambda_disc: float,
    gamma_disc: float,
    target_rhs: float,
    step_result: Mapping[str, float],
) -> dict[str, Any]:
    row = {column: 0.0 for column in METRIC_COLUMNS}
    row.update(
        {
            "step": int(step),
            "wallclock_s": float(wallclock_s),
            "n_sched": int(n_sched),
            "lambda_disc_active": float(lambda_disc),
            "gamma_disc": float(gamma_disc),
            "gamma_disc_active": float(gamma_disc),
            "target_rhs": float(target_rhs),
            "sigma": 0.0,
            "sigma_pi": 0.0,
            "L_R": float(step_result["L_R"]),
            "L_A": 0.0,
            "L_C": 0.0,
            "L_V_total": float(step_result["L_V_total"]),
            "grad_norm_VS": float(step_result["grad_norm_VS"]),
        }
    )
    for k in ("L_lg_raw", "lg_min", "lg_median", "lg_degen_frac"):   # v2.6.0 epsilon_g diagnostics
        if k in step_result:
            row[k] = float(step_result[k])
    return row


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
        role="In-loop eval",
        output_dir=run_dir / "figures/inloop",
        filename_template=f"step_{step:06d}_grid_{{letter}}.png",
    )
    # v2.6.1: CBF contour saved every eval for ALL systems (write_cbf_contour_figure dispatches on
    # system.name — quadrotor gets a position-plane approach-speed contour; None only on failure). Viz-only.
    contour_path = write_cbf_contour_figure(
        eval_result=eval_result,
        config=config,
        system=system,
        value_net=value_net,
        output_path=run_dir / "figures/inloop" / f"step_{step:06d}_cbf_contour.png",
        role="In-loop eval CBF contour",
    )
    if contour_path is not None:
        log_png_to_tensorboard(
            writer,
            f"eval/in_loop/step_{step:06d}_cbf_contour",
            contour_path,
            step,
        )
    _write_tb_scalars(
        writer,
        f"eval/{eval_result.eval_row['mode']}",
        eval_result.eval_row,
        step,
    )


def _save_checkpoint(
    path: Path,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    optimizer: optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
    step: int,
    best_cps: float,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": __version__,
            "framework": "oc_pncbf",
            "epoch": int(epoch),
            "step": int(step),
            "best_cps": float(best_cps),
            "best_step": int(best_step),
            "v_s_state": value_net.state_dict(),
            "v_s_target_state": target_value_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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
    git_text = _git_commit_text()
    (run_dir / "git_commit.txt").write_text(git_text + "\n", encoding="utf-8")
    _init_csv(run_dir / "metrics.csv", METRIC_COLUMNS)
    _init_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS)
    _init_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS)
    _write_pool_manifest_copy(run_dir, config)


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
    final_rows = [row for row in eval_rows if row.get("mode") == "final"]
    inloop_rows = [row for row in eval_rows if row.get("mode") == "in_loop"]
    peak = (
        max((float(row["cps"]), int(row["step"])) for row in inloop_rows)
        if inloop_rows
        else (float("nan"), 0)
    )
    final = final_rows[-1] if final_rows else None
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    system_name = str(run_config["run"]["system"])
    inloop_pool = _pool_path("inloop", run_config, system_name).stem
    full_pool = _pool_path("full", run_config, system_name).stem
    lines = [
        "# OC-PNCBF Run Report",
        "",
        "## Run identity",
        f"- run_id: `{run_id}`",
        f"- version: `{__version__}`",
        f"- git_commit: `{git_commit}`",
        f"- wallclock_s: `{wallclock_s:.3f}`",
        "",
        "## Final eval results",
    ]
    if final is None:
        lines.append("- final eval: not run")
    else:
        lines.extend(
            [
                f"- cps: `{float(final['cps']):.6f}`",
                f"- reach: `{float(final['reach']):.6f}`",
                f"- collision: `{float(final['collision']):.6f}`",
                f"- infeasibility: `{float(final['infeasibility']):.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## In-loop training curve",
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
            "- figures: `figures/trajectory_grid_A.png`, "
            "`figures/trajectory_grid_B.png`, `figures/cbf_contour.png`",
            "",
            "## Pool identities",
            f"- in_loop: `{inloop_pool}`",
            f"- full: `{full_pool}`",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Train OC-PNCBF.")
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-every-epochs", type=int, default=None)
    parser.add_argument("--smoke-eval-scenes", type=int, default=20)
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
    result = run_training(
        stage=args.stage,
        output_root=args.output_root,
        seed=args.seed,
        eval_every_epochs=args.eval_every_epochs,
        smoke_eval_scenes=args.smoke_eval_scenes,
        device=args.device,
        train_dtype_name=args.train_dtype,
        obstacle_distribution=args.obstacle_distribution,
    )
    print(result.run_dir)
    return 0 if result.halt_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
