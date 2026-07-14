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
from src.frameworks.jt_pncbf.recovery_policy import RecoveryPolicy, recovery_bptt_loss
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
from src.frameworks.jt_pncbf.losses import (
    cbf_deriv_feasibility_loss,
    grad_norm,
    grad_sq_norm,
    policy_bptt_loss,
    spline_sobolev_loss,
    value_loss,
)
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
    "value_target_unsafe_frac",
    "rho_unsafe_label",
    "label_mean",
    "tail_push_mean",
    "tail_exceed_frac",
    "loss_pi_b",
    "recovery_residual_norm",
    "L_feas_raw",
    "L_feas_weighted",
    "gate_active_frac",
    "gate_n_constructed",
    "gate_n_kept",
    "gate_doomed_frac",
    "lg_norm_gate_mean",
    "lg_norm_gate_p50",
    "injected_fraction_actual",
    "injected_target_mean",
    "injected_target_unsafe_frac",
    "L_sobolev_raw",
    "L_sobolev_weighted",
    "sob_grad_label_v_frac",
    "sob_grad_label_v_mean",
    "sob_dhdv_pred_mean",
    "sob_v_spline_mean",
    "sob_gate_mean",
    "sob_gate_frac_active",
    "sob_gate_dhdv_pred",
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
    "L_rate_raw",
    "L_rate_weighted",
    "mean_abs_du",
    "L_u_raw",
    "L_u_weighted",
    "L_deficit",
    "mean_deficit_active",
    "deficit_active_frac",
    "deficit_clip_frac",
    "mean_abs_deficit_feature",
    "L_friction",
    "proj_mag_bptt",
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


def _obs_deficit_on(config: Mapping[str, Any]) -> bool:
    return bool(config["loss"]["policy"].get("obs_deficit_feedback", False))


def _build_control_net(system: System, config: Mapping[str, Any]) -> ControlNet:
    # v2.4.1 Exp 2: when obs_deficit_feedback is on, the POLICY (only) consumes an augmented
    # observation [obs (dim obs_dim), delta_u_{t-1} (dim action_dim)]; the two new input columns are
    # zero-initialized so the policy ignores the deficit feature at init (clean continuity). The value
    # network observation is unchanged. Flag off => dim-obs_dim policy, byte-identical baseline.
    extra = system.action_dim if _obs_deficit_on(config) else 0
    net = ControlNet(system.obs_dim + extra, system, config)
    if extra:
        with torch.no_grad():
            net.trunk[0].weight[:, system.obs_dim:].zero_()
    return net


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
        # v2.4.1 Exp 2: deployed deficit-observation channel. delta_u_{t-1} is written by filter() and
        # read by policy() on the next step; reset per rollout (a fresh framework is built per eval).
        self._obs_deficit = _obs_deficit_on(config)
        self._prev_deficit: Tensor | None = None
        from src.common.maneuver_value import build_safety_h_fn
        self._filter = HardNetFilter(
            system,
            build_safety_h_fn(system, config, value_net),   # v2.5.0 Stage B: V_M or learned h_fn
            config,
            policy_fn=self.policy,
        )

    def reset_deficit_state(self) -> None:
        self._prev_deficit = None

    def _policy_obs(self, x: Tensor, scene: Any) -> Tensor:
        obs = self.system.observation(x, scene)
        if not self._obs_deficit:
            return obs
        b = obs.shape[0]
        if self._prev_deficit is None or self._prev_deficit.shape[0] != b:
            feat = obs.new_zeros((b, self.system.action_dim))
        else:
            feat = self._prev_deficit
        return torch.cat([obs, feat], dim=1)

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        return self.policy_net(self._policy_obs(x, scene))

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        if not self._obs_deficit:
            return self._filter(x, scene, u_nom)
        u_safe, infeasible, u_cbf_only, _ = self._filter(
            x, scene, u_nom, return_deficit_aux=True)
        self._prev_deficit = (u_cbf_only - u_safe).detach()
        return u_safe, infeasible

    def value(self, x: Tensor, scene: Any) -> Tensor:
        from src.common.maneuver_value import build_safety_h_fn
        return build_safety_h_fn(self.system, self.config, self.value_net)(x, scene)


class _ManeuverEvalFramework:
    """v2.5.0 Stage B eval-only framework: trained policy + analytic V_M HardNet filter computed with
    create_graph=False (first-order L_g only). u_safe is IDENTICAL to the deployed create_graph=True
    filter (create_graph affects only the unused 2nd-order graph), but no graph accumulates across the
    ~200-step eval rollout -> no OOM. Mirrors the Stage-A projection path (mc_stage_a.StageAFramework)."""

    def __init__(self, system: System, policy_net: ControlNet, config: Mapping[str, Any]) -> None:
        from src.common.filter_hardnet import _hardnet_params
        from src.common.maneuver_value import build_safety_h_fn
        self.system = system
        self.config = config
        self.policy_net = policy_net
        # evaluate._tensor_options reads framework.value_net's dtype/device (its ONLY use of value_net);
        # alias the trained policy so the eval runs on the policy's device/dtype (float32, cuda).
        self.value_net = policy_net
        self.h_fn = build_safety_h_fn(system, config, None)   # maneuver V_M + gamma_m
        self.params = _hardnet_params(config)

    def policy(self, x: Tensor, scene: Any) -> Tensor:
        return self.policy_net(self.system.observation(x, scene))

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        from src.common.filter_hardnet import (
            _SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
        )
        h, lf, lg = _cbf_terms(self.system, self.h_fn, x, scene, u_nom, create_graph=False)
        h, lf, lg = h.detach(), lf.detach(), lg.detach()
        with torch.no_grad():
            alpha = _base_alpha(h, self.params)
            row_upper = -lf - alpha * h
            bounds = self.system.u_bounds.to(device=u_nom.device, dtype=u_nom.dtype)
            projected = _base_projection(u_nom, lg, row_upper, bounds, self.params)
            singular = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
            if self.params.box_aware:
                u_safe, empty = _box_aware_projection(u_nom, projected, lg, row_upper, bounds)
                infeasible = singular | empty
            else:
                u_safe, infeasible = projected, singular
        return u_safe.detach(), infeasible.detach()


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
    resume_ckpt: Path | None = None,
    pi_init_ckpt: Path | None = None,
    safety_channel: str | None = None,
    w_friction: float | None = None,
) -> JTTrainingResult:
    config = load_effective_config()
    config["run"]["version"] = __version__
    config["run"]["framework"] = "jt_pncbf"
    if safety_channel is not None:   # v2.5.0 Stage B: --safety-channel maneuver activates the analytic V_M
        config.setdefault("safety_channel", {})["type"] = safety_channel
    if w_friction is not None:       # v2.5.0 Stage B-2: filter-friction weight override
        config["loss"]["policy"]["w_friction"] = float(w_friction)
    if pi_init_ckpt is not None:     # v2.5.1 A2(a): policy warm-start (pi_state only; see the loader below)
        config["training"]["jt"]["pi_init_ckpt"] = str(pi_init_ckpt)
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
    policy_net = _build_control_net(system, config).to(
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

    # v2.5.1 A2(b): horizon-summary critic W (PERFORMANCE CHANNEL ONLY — never touches h_fn, the filter,
    # the shield, or the labels). W = house value trunk (CPIValue: 3x256 Softplus(beta=20) + raw linear
    # head) on the dim-19 base observation, own parameters; a Polyak target net for its bootstrap. Built
    # ONLY when training.jt.horizon_critic.enabled is true; enabled=false leaves everything below inert
    # (critic_net=None threads through to policy_bptt_loss => byte-identical to baseline).
    hc_cfg = dict(config["training"]["jt"].get("horizon_critic", {}))
    horizon_critic_enabled = bool(hc_cfg.get("enabled", False))
    critic_net = None
    critic_target = None
    opt_w = None
    if horizon_critic_enabled:
        from src.frameworks.cpi.value import CPIValue
        if bool(config["loss"]["policy"].get("obs_deficit_feedback", False)):
            raise ValueError("horizon_critic is not supported with obs_deficit_feedback (dim-19 obs assumed).")
        critic_net = CPIValue(obs_dim=system.obs_dim).to(device=train_device, dtype=train_dtype)
        critic_target = deepcopy(critic_net)
        critic_target.requires_grad_(False)
        opt_w = optim.AdamW(
            critic_net.parameters(),
            lr=float(hc_cfg.get("lr", 1.0e-3)),
            weight_decay=float(config["optim"]["weight_decay"]),
        )

    # v2.4.0 Step 2: learned recovery conditioning policy pi_b (residual around the analytic brake),
    # trained by an avoid-only BPTT loss and consumed (via a Polyak target copy) by the value target.
    # Only built when value_target.conditioning == "learned_recovery"; otherwise all-None (the brake /
    # task_stored paths are unaffected).
    conditioning = str(config["value_target"].get("conditioning", "task_stored"))
    recovery_enabled = conditioning == "learned_recovery"
    recovery_policy: RecoveryPolicy | None = None
    recovery_target: RecoveryPolicy | None = None
    opt_b: optim.Optimizer | None = None
    k_b = 0
    if recovery_enabled:
        rec_u_max = float(config["env"]["bounds"][system.name]["u_max"])
        rec_eps_v = float(config["value_target"]["brake"]["eps_v"])
        recovery_policy = RecoveryPolicy(system.obs_dim, system, config, rec_u_max, rec_eps_v).to(
            device=train_device, dtype=train_dtype,
        )
        recovery_target = deepcopy(recovery_policy)
        recovery_target.requires_grad_(False)
        opt_b = optim.AdamW(
            recovery_policy.parameters(),
            lr=float(config["optim"]["lr_pi"]),
            weight_decay=float(config["optim"]["weight_decay"]),
        )
        k_b = int(config["value_target"]["recovery"].get("K_b", 1))

    # Joint resume: continue training from a saved checkpoint's nets + optimizer state, starting at its
    # step (the loop and schedule then continue from there; schedule_step is clamped so it stays at the
    # final saturated value when start_step >= schedule_n_steps). start_step=0 == fresh run (unchanged).
    start_step = 0
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=train_device, weights_only=False)
        value_net.load_state_dict(ckpt["v_s_state"])
        target_value_net.load_state_dict(ckpt["v_s_target_state"])
        policy_net.load_state_dict(ckpt["pi_state"])
        opt_vs.load_state_dict(ckpt["opt_vs_state"])
        opt_pi.load_state_dict(ckpt["opt_pi_state"])
        start_step = int(ckpt["step"])
        config["run"]["resume_ckpt"] = str(resume_ckpt)
        config["run"]["resume_from_step"] = start_step

    # v2.5.1 A2(a): policy WARM-START. Loads ONLY pi_state into policy_net; the optimizer, LR/sigma
    # schedules, step counter (start_step stays 0), value net, target net, AND the certificate safety
    # channel all start fresh — unlike resume_ckpt (a full joint resume). pi_init_ckpt=null (default)
    # never opens the loader, so a fresh run is byte-identical. Mutually exclusive with resume_ckpt. The
    # source path + sha256 are stamped into config["run"] for provenance.
    pi_init_ckpt_cfg = config["training"]["jt"].get("pi_init_ckpt")
    if pi_init_ckpt_cfg is not None:
        if resume_ckpt is not None:
            raise ValueError("pi_init_ckpt (warm-start) and resume_ckpt (full resume) are mutually exclusive.")
        import hashlib
        pi_init_path = Path(pi_init_ckpt_cfg)
        pi_init_sha = hashlib.sha256(pi_init_path.read_bytes()).hexdigest()
        pi_ckpt = torch.load(pi_init_path, map_location=train_device, weights_only=False)
        policy_net.load_state_dict(pi_ckpt["pi_state"])
        config["run"]["pi_init_ckpt"] = str(pi_init_path)
        config["run"]["pi_init_ckpt_sha256"] = pi_init_sha

    # v2.4.2: task_raw_lagged conditioning. pi_b is a Polyak-lagged, grad-free copy of the task policy
    # (in no optimizer), initialized from the starting policy_net. The value target conditions its label
    # on UNFILTERED, noise-free deterministic re-rolls of pi_b (losses.py value_targets branch); after
    # every policy optimizer step theta_b <- (1-tau_b) theta_b + tau_b theta_pi. During vs_warmup there
    # are no policy steps so pi_b stays at init (expected). Only built for this conditioning; else None.
    raw_lagged_enabled = conditioning == "task_raw_lagged"
    lagged_policy: ControlNet | None = None
    raw_lagged_tau_b = 0.0
    if raw_lagged_enabled:
        lagged_policy = deepcopy(policy_net)
        lagged_policy.requires_grad_(False)
        raw_lagged_tau_b = float(config["value_target"]["raw_lagged"]["tau_b"])

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

    # v2.5.0 Stage B: analytic-maneuver safety channel. The value net is not learned (K_V=0), no warmup,
    # and detach_filter_coeffs is forced on (first-order only through the 30-step V_M rollout). Applied
    # AFTER the smoke reduction so it wins over smoke's K_V=1. Value mode (default) is untouched.
    # Frozen safety channel (no value learning, K_V=0): analytic maneuver V_M OR a frozen learned CPI
    # certificate. Both deploy build_safety_h_fn and use the first-order _ManeuverEvalFramework for in-loop
    # eval (create_graph=False; no graph accumulation). v2.5.1 CPI loop extends this gate to 'cpi'.
    maneuver_mode = str(config.get("safety_channel", {}).get("type", "value")) in ("maneuver", "cpi", "exact_m0")
    if maneuver_mode:
        k_v = 0
        vs_warmup_steps = 0
        config["loss"]["policy"]["detach_filter_coeffs"] = True

    policy_buffer_cap = collection_cfg.get("policy_buffer_cap")
    buffers = make_replay_buffers(
        capacity=int(collection_cfg["buffer_cap"]),
        policy_capacity=int(policy_buffer_cap) if policy_buffer_cap is not None else None,
    )
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

    for step in range(start_step + 1, n_steps + 1):
        if step == start_step + 1 or step % max(1, collect_every) == 0:
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

        # Host-side scalar extraction (the probe forward + .item() syncs) is done ONLY on steps whose
        # metrics row is actually emitted; the optimizer math runs every step regardless. Safety halts
        # (NaN/Inf, grad-leak) are decided every step via single device->host bools (see _value_updates
        # / _policy_updates), so they fire on the same step as before.
        do_log = (step % metrics_log_every == 0) or (step == n_steps)

        if maneuver_mode:
            # Stage B: no value learning (K_V=0). Skip the value update entirely (K_V=0 would break the
            # empty-loop mean); emit zeroed value scalars so the metrics schema is unchanged (value cols
            # 0). target_value_net stays at init and is never queried (build_safety_h_fn ignores it).
            value_scalars = {"value_finite": True, "L_V_total": 0.0, "grad_norm_VS": 0.0, "L_R": 0.0}
        else:
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
                log=do_log,
                policy_net=policy_net,
                recovery_policy=recovery_target,
                lagged_policy=lagged_policy,
            )
        if not value_scalars.pop("value_finite"):
            halt_reason = "nan_or_inf_L_V"
            break

        # v2.4.0 Step 2: pi_b update AFTER the value updates and BEFORE the policy updates, active
        # from step 1 (does not depend on V_S or the vs_warmup window).
        if recovery_enabled:
            recovery_scalars = _recovery_updates(
                system=system,
                recovery_policy=recovery_policy,
                recovery_target=recovery_target,
                optimizer=opt_b,
                buffers=buffers,
                torch_generator=torch_rng,
                batch_size=policy_batch_size,
                n_updates=k_b,
                config=config,
                log=do_log,
            )
            if not recovery_scalars.pop("recovery_finite"):
                halt_reason = "nan_or_inf_L_pi_b"
                break
            if do_log:
                value_scalars.update(recovery_scalars)

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
                step=step,
                log=do_log,
                critic_net=critic_net,
            )
            pi_finite = bool(policy_scalars.pop("pi_finite"))
            leak_exceeds = bool(policy_scalars.pop("leak_exceeds"))
            leak_sq_max = policy_scalars.pop("leak_sq_max")
            if not pi_finite:
                halt_reason = "nan_or_inf_L_pi"
                break
            if leak_exceeds:
                max_policy_leak = float(leak_sq_max.sqrt().item())
                halt_reason = "policy_gradient_leak"
                break
            # v2.4.2: Polyak-lag pi_b toward the just-updated task policy (after every policy step;
            # k_pi=1 => once per macro step). Only active for task_raw_lagged (lagged_policy is None else).
            if lagged_policy is not None:
                _polyak_update(lagged_policy, policy_net, tau=raw_lagged_tau_b)
            if do_log:
                last_policy_loss = policy_scalars["L_pi_total"]
                last_pi_grad = policy_scalars["grad_norm_pi"]
                max_policy_leak = max(max_policy_leak, policy_scalars["grad_leak_VS_from_Lpi"])

        # v2.5.1 A2(b): one horizon-critic W regression step per macro step (independent of the vs_warmup
        # window). Grad-free 30-step rollout of the DEPLOYED (filtered) policy + Polyak-target bootstrap;
        # W never affects collection, the filter, or eval. Inert when horizon_critic.enabled is false.
        if horizon_critic_enabled:
            critic_scalars = _critic_updates(
                system=system,
                critic_net=critic_net,
                critic_target=critic_target,
                optimizer=opt_w,
                value_net=value_net,
                policy_net=policy_net,
                buffers=buffers,
                torch_generator=torch_rng,
                batch_size=policy_batch_size,
                config=config,
                log=do_log,
            )
            if not critic_scalars.pop("critic_finite"):
                halt_reason = "nan_or_inf_L_W"
                break
            if do_log:
                value_scalars.update(critic_scalars)

        if do_log:
            last_value_loss = value_scalars["L_V_total"]
            last_vs_grad = value_scalars["grad_norm_VS"]
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
            # v2.5.0 Stage B: the deployed HardNetFilter uses create_graph=True (for training BPTT); over
            # the ~200-step eval rollout each step's V_M graph chains into the next state and accumulates
            # to OOM (|M|=17 rollout). Eval only needs u_safe VALUES, identical under create_graph=False,
            # so maneuver-mode eval uses a first-order framework (no graph accumulation; the exact path
            # validated in Stage A). Value mode unchanged.
            eval_framework = (_ManeuverEvalFramework(system, policy_net, config) if maneuver_mode
                              else JTPNCBFFramework(system, value_net, policy_net, config))
            eval_result = evaluate(
                eval_framework,
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
    policy_net = _build_control_net(system, config).to(
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

    policy_buffer_cap = collection_cfg.get("policy_buffer_cap")
    buffers = make_replay_buffers(
        capacity=int(collection_cfg["buffer_cap"]),
        policy_capacity=int(policy_buffer_cap) if policy_buffer_cap is not None else None,
    )
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
            log=True,
            policy_net=None,
        )
        value_scalars.pop("value_finite", None)
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
    log: bool = True,
    policy_net: ControlNet | None = None,
    recovery_policy: Any = None,
    lagged_policy: Any = None,
) -> dict[str, Any]:
    # v2.2.1: box-feasibility CBF-derivative auxiliary term (value-side). weight=0 => skip the gate
    # entirely (no RNG draws, no _cbf_terms) so behavior is bit-identical to the baseline value loss.
    # The optimizer math (sample/loss/backward/clip/step/polyak) runs EVERY step; host-side scalar
    # extraction (.item()/.cpu()) and the L_feas diagnostics are computed ONLY when log=True, so a
    # non-logging step incurs a single device->host bool (the NaN/Inf halt check) instead of ~20 syncs.
    feas_weight = float(config["loss"]["value"].get("cbf_deriv", {}).get("weight", 0.0))
    feas_enabled = feas_weight > 0.0
    # v2.2.2 precursor injection (collection-time design): a ~inj_fraction share of each value minibatch
    # is sampled from the PRECURSOR buffer (populated once per collection in collect_jt), the rest from
    # the normal value buffer. NO rollout in the update loop. Flag off / empty precursor buffer (e.g.
    # value-refinement) => baseline path, value-side bit-identical to baseline.
    inj_cfg = config["loss"]["value"].get("precursor_injection", {})
    inj_fraction = float(inj_cfg.get("fraction", 0.0))
    n_inj = int(round(inj_fraction * batch_size))
    inj_enabled = bool(inj_cfg.get("enabled", False)) and len(buffers.precursor) > 0 and n_inj > 0
    n_normal = (batch_size - n_inj) if inj_enabled else batch_size
    # v2.2.2 Route D: RPCBF cubic-spline-max Sobolev gradient-matching on V_S (value-side). weight=0 /
    # disabled => skip all rollout/spline/grad work so the value step is bit-identical to the baseline.
    sob_cfg = config["loss"]["value"].get("sobolev", {})
    sob_weight = float(sob_cfg.get("weight", 0.0))
    sob_enabled = bool(sob_cfg.get("enabled", False)) and sob_weight > 0.0

    totals = []
    reaches = []
    grad_norms = []
    feas_logs: dict[str, list[float]] = {
        "L_feas_raw": [], "L_feas_weighted": [], "gate_active_frac": [],
        "gate_n_constructed": [], "gate_n_kept": [], "gate_doomed_frac": [],
        "lg_norm_gate_mean": [], "lg_norm_gate_p50": [],
    }
    inj_logs: dict[str, list[float]] = {
        "injected_fraction_actual": [], "injected_target_mean": [], "injected_target_unsafe_frac": [],
    }
    sob_logs: dict[str, list[float]] = {
        "L_sobolev_raw": [], "L_sobolev_weighted": [], "sob_grad_label_v_frac": [],
        "sob_grad_label_v_mean": [], "sob_dhdv_pred_mean": [], "sob_v_spline_mean": [],
        "sob_gate_mean": [], "sob_gate_frac_active": [], "sob_gate_dhdv_pred": [],
    }
    vt_unsafe: list[float] = []          # v2.4.0: fraction of value-minibatch targets y > 0
    vt_mean: list[float] = []            # v2.4.2: mean value-minibatch label y (label_mean)
    tp_push: list[float] = []            # v2.4.2 Exp 2: tail_push_mean (raw_lagged only; 0 otherwise)
    tp_exceed: list[float] = []          # v2.4.2 Exp 2: tail_exceed_frac
    for _ in range(n_updates):
        if inj_enabled:
            normal_batch = buffers.value.sample_tensor_batch(n_normal, generator=torch_generator)
            res_n = value_loss(system=system, value_net=value_net, target_value_net=target_value_net,
                               batch=normal_batch, lambda_disc=lambda_disc, target_rhs=target_rhs, config=config,
                               recovery_policy=recovery_policy, lagged_policy=lagged_policy)
            inj_batch = buffers.precursor.sample_tensor_batch(n_inj, generator=torch_generator)
            res_i = value_loss(system=system, value_net=value_net, target_value_net=target_value_net,
                               batch=inj_batch, lambda_disc=lambda_disc, target_rhs=target_rhs, config=config,
                               recovery_policy=recovery_policy, lagged_policy=lagged_policy)
            # single MSE over the mixed batch_size states: (sum_sq_normal + sum_sq_inj)/batch_size
            total_loss = (n_normal * res_n.total + n_inj * res_i.total) / batch_size
            reach_log = (n_normal * res_n.reach + n_inj * res_i.reach) / batch_size
            scene_for_feas = normal_batch.scene
            states_for_aux = normal_batch.states
            if log:
                tgt = res_i.targets.detach()
                inj_logs["injected_fraction_actual"].append(n_inj / batch_size)
                inj_logs["injected_target_mean"].append(float(tgt.mean().item()))
                inj_logs["injected_target_unsafe_frac"].append(float((tgt >= 0.0).to(tgt.dtype).mean().item()))
                vt_unsafe.append(float((res_n.targets.detach() > 0.0).to(res_n.targets.dtype).mean().item()))
                vt_mean.append(float(res_n.targets.detach().mean().item()))
                tp_push.append(float(res_n.tail_push_mean))
                tp_exceed.append(float(res_n.tail_exceed_frac))
        else:
            batch = buffers.value.sample_tensor_batch(batch_size, generator=torch_generator)
            result = value_loss(system=system, value_net=value_net, target_value_net=target_value_net,
                                batch=batch, lambda_disc=lambda_disc, target_rhs=target_rhs, config=config,
                                recovery_policy=recovery_policy, lagged_policy=lagged_policy)
            total_loss = result.total
            reach_log = result.reach
            scene_for_feas = batch.scene
            states_for_aux = batch.states
            if log:
                vt = result.targets.detach()
                vt_unsafe.append(float((vt > 0.0).to(vt.dtype).mean().item()))
                vt_mean.append(float(vt.mean().item()))
                tp_push.append(float(result.tail_push_mean))
                tp_exceed.append(float(result.tail_exceed_frac))
        if feas_enabled:
            feas = cbf_deriv_feasibility_loss(
                system=system,
                value_net=value_net,
                scene=scene_for_feas,
                config=config,
                generator=torch_generator,
                collect_diagnostics=log,
            )
            total_loss = total_loss + feas_weight * feas.loss_raw
            if log:
                loss_raw = float(feas.loss_raw.detach().item())
                feas_logs["L_feas_raw"].append(loss_raw)
                feas_logs["L_feas_weighted"].append(feas_weight * loss_raw)
                for key in ("gate_active_frac", "gate_n_constructed", "gate_n_kept",
                            "gate_doomed_frac", "lg_norm_gate_mean", "lg_norm_gate_p50"):
                    feas_logs[key].append(feas.diagnostics[key])
        if sob_enabled:
            sob = spline_sobolev_loss(
                system=system,
                value_net=value_net,
                policy_net=policy_net,
                states=states_for_aux,
                scene=scene_for_feas,
                config=config,
                collect_diagnostics=log,
            )
            total_loss = total_loss + sob_weight * sob.loss_raw
            if log:
                sob_raw = float(sob.loss_raw.detach().item())
                sob_logs["L_sobolev_raw"].append(sob_raw)
                sob_logs["L_sobolev_weighted"].append(sob_weight * sob_raw)
                for key in ("sob_grad_label_v_frac", "sob_grad_label_v_mean",
                            "sob_dhdv_pred_mean", "sob_v_spline_mean",
                            "sob_gate_mean", "sob_gate_frac_active", "sob_gate_dhdv_pred"):
                    sob_logs[key].append(sob.diagnostics[key])
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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
        totals.append(total_loss.detach())          # full optimized objective (mixed MSE [+ weighted L_feas])
        reaches.append(reach_log.detach())           # MSE reach term (mixed when injecting)
        grad_norms.append(torch.as_tensor(grad, device=total_loss.device, dtype=total_loss.dtype))

    mean_total = torch.stack(totals).mean()
    out: dict[str, Any] = {"value_finite": bool(torch.isfinite(mean_total))}  # per-step halt signal (1 sync)
    if log:
        out.update(_host_scalars({
            "L_R": torch.stack(reaches).mean(),
            "L_V_total": mean_total,
            "grad_norm_VS": torch.stack(grad_norms).mean(),
        }))
        if feas_enabled:
            out.update({key: float(np.mean(vals)) for key, vals in feas_logs.items()})
        else:
            out.update({key: 0.0 for key in feas_logs})
        if inj_enabled:
            out.update({key: float(np.mean(vals)) for key, vals in inj_logs.items()})
        else:
            out.update({key: 0.0 for key in inj_logs})
        if sob_enabled:
            out.update({key: float(np.mean(vals)) for key, vals in sob_logs.items()})
        else:
            out.update({key: 0.0 for key in sob_logs})
        out["value_target_unsafe_frac"] = float(np.mean(vt_unsafe)) if vt_unsafe else 0.0
        out["rho_unsafe_label"] = float(np.mean(vt_unsafe)) if vt_unsafe else 0.0
        out["label_mean"] = float(np.mean(vt_mean)) if vt_mean else 0.0
        out["tail_push_mean"] = float(np.mean(tp_push)) if tp_push else 0.0
        out["tail_exceed_frac"] = float(np.mean(tp_exceed)) if tp_exceed else 0.0
    return out


def _recovery_updates(
    *,
    system: System,
    recovery_policy: RecoveryPolicy,
    recovery_target: RecoveryPolicy,
    optimizer: optim.Optimizer,
    buffers: JTReplayBuffers,
    torch_generator: torch.Generator | None,
    batch_size: int,
    n_updates: int,
    config: Mapping[str, Any],
    log: bool = True,
) -> dict[str, Any]:
    # v2.4.0 Step 2: avoid-only BPTT updates of the learned recovery policy pi_b, sampled from D_V.
    # No V_S and no pi_theta are in the graph, so gradients reach only pi_b's params (routing is by
    # construction). After each optimizer step, the Polyak target pi_b_target is nudged (tau_b), and
    # the value target consumes ONLY that target copy. Finiteness is checked every step (halt signal).
    tau_b = float(config["value_target"]["recovery"]["tau_b"])
    grad_clip = float(config["optim"]["grad_clip"])
    losses: list[Tensor] = []
    residuals: list[Tensor] = []
    finite = True
    for _ in range(n_updates):
        batch = buffers.value.sample_tensor_batch(batch_size, generator=torch_generator)
        loss, residual_norm = recovery_bptt_loss(
            system=system, recovery_policy=recovery_policy, batch=batch, config=config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(recovery_policy.parameters(), max_norm=grad_clip)
        optimizer.step()
        _polyak_update(recovery_target, recovery_policy, tau=tau_b)
        finite = finite and bool(torch.isfinite(loss).item())
        losses.append(loss.detach())
        residuals.append(residual_norm.detach())
    out: dict[str, Any] = {"recovery_finite": finite}
    if log:
        out["loss_pi_b"] = float(torch.stack(losses).mean().item())
        out["recovery_residual_norm"] = float(torch.stack(residuals).mean().item())
    return out


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
    step: int = 0,
    log: bool = True,
    critic_net: nn.Module | None = None,
) -> dict[str, Any]:
    # Optimizer math runs EVERY step. The grad-leak SAFETY check is decided on-device every step
    # (grad_sq_norm -> single 'leak_exceeds' bool); host-side row scalars + the leak magnitude are
    # materialized only when log=True. Returns control signals (pi_finite/leak_exceeds/leak_sq_max)
    # always; row scalars only on logging steps.
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
        "L_rate_raw": [],
        "L_rate_weighted": [],
        "mean_abs_du": [],
        "L_u_raw": [],
        "L_u_weighted": [],
        "L_deficit": [],
        "mean_deficit_active": [],
        "deficit_active_frac": [],
        "deficit_clip_frac": [],
        "mean_abs_deficit_feature": [],
        "L_friction": [],
        "proj_mag_bptt": [],
    }
    pi_totals: list[Tensor] = []
    leak_sqs: list[Tensor] = []
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
            step=step,
            critic_net=critic_net,
        )
        result.total.backward()
        leak_sq = grad_sq_norm(value_net.parameters())          # on-device sum of squared value grads
        leak_sqs.append(leak_sq if leak_sq is not None else result.total.detach().new_zeros(()))
        grad = nn.utils.clip_grad_norm_(
            policy_net.parameters(),
            max_norm=float(config["optim"]["grad_clip"]),
        )
        optimizer.step()
        pi_totals.append(result.total.detach())
        if log:
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
            accum["L_rate_raw"].append(result.l_rate_raw)
            accum["L_rate_weighted"].append(result.l_rate_weighted)
            accum["mean_abs_du"].append(result.mean_abs_du)
            accum["L_u_raw"].append(result.l_u_raw)
            accum["L_u_weighted"].append(result.l_u_weighted)
            accum["L_deficit"].append(result.l_deficit)
            accum["mean_deficit_active"].append(result.mean_deficit_active)
            accum["deficit_active_frac"].append(result.deficit_active_frac)
            accum["deficit_clip_frac"].append(result.deficit_clip_frac)
            accum["mean_abs_deficit_feature"].append(result.mean_abs_deficit_feature)
            accum["L_friction"].append(result.friction_loss)
            accum["proj_mag_bptt"].append(result.proj_mag_bptt)

    leak_sq_max = torch.stack(leak_sqs).max()
    threshold = float(config["halt"]["vs_grad_leak_threshold"])
    out: dict[str, Any] = {
        "pi_finite": bool(torch.isfinite(torch.stack(pi_totals).mean())),
        "leak_exceeds": bool(leak_sq_max > threshold * threshold),  # equiv to ||leak|| > threshold
        "leak_sq_max": leak_sq_max,
    }
    if log:
        out.update(_host_scalars({key: torch.stack(values).mean() for key, values in accum.items()}))
        out["grad_leak_VS_from_Lpi"] = float(leak_sq_max.sqrt().item())
    return out


def _critic_updates(
    *,
    system: System,
    critic_net: nn.Module,
    critic_target: nn.Module,
    optimizer: optim.Optimizer,
    value_net: ValueNetEnsemble,
    policy_net: ControlNet,
    buffers: JTReplayBuffers,
    torch_generator: torch.Generator | None,
    batch_size: int,
    config: Mapping[str, Any],
    log: bool = True,
) -> dict[str, Any]:
    # v2.5.1 A2(b): one horizon-critic W regression step. W(obs(x_0)) regresses the n-step (n=30)
    # bootstrapped discounted stage-cost-to-go of the EXACT L_pi stage cost c = d2 + lambda_v*v2 + mu_u*u2
    # under the DEPLOYED (filtered) policy: target = sum_{t<n} gamma_c^t c_t + gamma_c^n * W_target(x_n),
    # entirely grad-free. Then Polyak the target net (house tau). Never touches collection/filter/eval.
    from src.common.maneuver_value import build_safety_h_fn
    from src.common.rk4 import rk4_step
    from src.frameworks.jt_pncbf.losses import _scene_goal

    hc_cfg = config["training"]["jt"]["horizon_critic"]
    gamma_c = float(hc_cfg["gamma"]); n_step = int(hc_cfg["n_step"])
    dt = float(config["env"]["dt"]); tau = float(config["optim"]["tau_polyak"])
    pc = config["loss"]["policy"]; lambda_v = float(pc["lambda_v"]); mu_u = float(pc["mu_u"])
    detach_coeffs = bool(pc.get("detach_filter_coeffs", False))
    hardnet = HardNetFilter(system, build_safety_h_fn(system, config, value_net), config)

    batch = buffers.policy.sample_tensor_batch(batch_size, generator=torch_generator)
    with torch.no_grad():
        x = system.wrap_state(batch.states.detach())
        scene = batch.scene
        obs0 = system.observation(x, scene)
        cost_to_go = x.new_zeros(x.shape[0])
        discount = 1.0
        for _ in range(n_step):
            u_nom = policy_net(system.observation(x, scene))
            u_safe, _ = hardnet(x, scene, u_nom, detach_coeffs=detach_coeffs)
            x = rk4_step(system, x, u_safe, dt)
            pos_error = system.position(x) - _scene_goal(scene, x)
            d2 = torch.sum(pos_error * pos_error, dim=1)
            v2 = system.speed(x) * system.speed(x)
            u2 = torch.sum(u_safe * u_safe, dim=1)
            cost_to_go = cost_to_go + discount * (d2 + lambda_v * v2 + mu_u * u2)
            discount *= gamma_c
        target = cost_to_go + (gamma_c ** n_step) * critic_target(system.observation(x, scene))

    pred = critic_net(obs0)
    loss = torch.mean((pred - target) * (pred - target))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=float(config["optim"]["grad_clip"]))
    optimizer.step()
    _polyak_update(critic_target, critic_net, tau=tau)
    out: dict[str, Any] = {"critic_finite": bool(torch.isfinite(loss).item())}
    if log:
        out["L_W"] = float(loss.detach().item())
        out["grad_norm_W"] = float(grad)
    return out


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
        "L_rate_raw": 0.0,
        "L_rate_weighted": 0.0,
        "mean_abs_du": 0.0,
        "L_u_raw": 0.0,
        "L_u_weighted": 0.0,
        "L_deficit": 0.0,
        "mean_deficit_active": 0.0,
        "deficit_active_frac": 0.0,
        "deficit_clip_frac": 0.0,
        "mean_abs_deficit_feature": 0.0,
        "L_friction": 0.0,
        "proj_mag_bptt": 0.0,
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
    policy_net = _build_control_net(system, config).to(dtype=first_tensor.dtype)
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
    # One device->host transfer for all scalars (stack + single .cpu()) instead of one .item() each.
    keys = list(values)
    if not keys:
        return {}
    stacked = torch.stack([values[key].detach().reshape(()) for key in keys]).cpu()
    return {key: float(stacked[idx]) for idx, key in enumerate(keys)}


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
    parser.add_argument("--resume-ckpt", type=Path, default=None,
                        help="Resume joint training from a checkpoint's nets+optimizer state at its step.")
    parser.add_argument("--safety-channel", choices=["value", "maneuver"], default=None,
                        help="v2.5.0 Stage B: 'maneuver' deploys the analytic V_M barrier (no value learning).")
    parser.add_argument("--w-friction", type=float, default=None,
                        help="v2.5.0 Stage B-2: filter-friction weight w_friction*||u_safe-u_nom||^2 (default config 0.0).")
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
            resume_ckpt=args.resume_ckpt,
            safety_channel=args.safety_channel,
            w_friction=args.w_friction,
        )
    print(result.run_dir)
    return 0 if result.halt_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
