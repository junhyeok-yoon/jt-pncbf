"""v2.9.3 ALT-PNCBF trainer — the alternating-block (policy-iteration) macro-step loop.

READ `src/frameworks/alt_pncbf/__init__.py` FIRST for what this framework is and what it deliberately
does not carry. This module is the LOOP and nothing else: every network, loss, collector, inner update,
evaluation call and artifact writer below is imported, never redefined.

The macro step is the joint trainer's macro step (03_train §5), with exactly one substitution: instead
of running K_V value updates AND K_pi policy updates, it runs the updates of the block the schedule says
is active, and freezes + audits the other network. Collection, the schedule clock, sigma, logging,
in-loop evaluation, checkpointing, status and the halt checks all advance PER MACRO STEP exactly as in
the joint trainer, in the joint trainer's order.

THE FOUR HALT CHECKS THIS FRAMEWORK ADDS OR KEEPS (invariant 4 — both directions, both measured):

  value block  ->  policy grads must not exist   `alt_value_block_policy_gradient_leak`
  value block  ->  policy params must not move   `alt_value_block_policy_drift`
  policy block ->  value grads must not exist    `policy_gradient_leak`   (the JT check, imported)
  policy block ->  value params must not move    `alt_policy_block_value_drift`

The joint trainer checks one direction only (`_policy_updates` measures grad_sq_norm on the value net,
jt_pncbf/train.py:1670); the reverse direction is structural there but unasserted
(docs/versions/v2.9.3/block_alternation_design.md §5.2). Here both directions are frozen with the
repository's own `frozen_params` idiom AND measured every macro step.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn, optim

from src._version import __version__
from src.common.control_net import ControlNet
from src.common.eval_gate import eval_gate
from src.common.train_instrument import PhaseTimers
from src.common.value_net import ValueNetEnsemble
from src.eval.evaluate import evaluate

# --- imported from the joint framework: losses, collectors, inner updates, artifact contract ---------
# Nothing in src/frameworks/jt_pncbf/ is modified; every name below is used exactly as the joint
# trainer uses it, so this framework's runs are byte-compatible with every existing instrument.
from src.frameworks.jt_pncbf.collection import (
    CollectionStats,
    collect_jt,
    collect_policy_rollouts,
    collect_precursors,
    make_replay_buffers,
)
from src.frameworks.jt_pncbf.losses import _zero_grads, frozen_params, grad_sq_norm
from src.frameworks.jt_pncbf.train import (
    DEFAULT_OUTPUT_ROOT,
    METRIC_COLUMNS,
    JTPNCBFFramework,
    _append_csv,
    _auto_run_final_eval,
    _build_control_net,
    _create_run_dir,
    _deep_merge,
    _flush_jac_classes,
    _init_csv,
    _initialize_run_dir,
    _make_probe_scene,
    _make_summary_writer,
    _metrics_row,
    _policy_updates,
    _pool_path,
    _probe_h_spread,
    _record_eval,
    _sigma_pi_at,
    _sigma_probe_halt_check,
    _state_to_dtype,
    _train_scene_sampler,
    _value_updates,
    _write_report,
    _write_status,
    _write_tb_scalars,
    _zero_policy_scalars,
    load_effective_config,
    make_system,
    _plain_data,
    _resolve_train_device,
    _resolve_train_dtype,
)
from src.frameworks.oc_pncbf.value_target import (
    gamma_from_lambda,
    lambda_schedule_value,
    schedule_value,
)

from src.frameworks.alt_pncbf.schedule import (
    BLOCK_CODE,
    COLLECT_ACTIVE_ONLY,
    COLLECT_BOTH,
    POLICY,
    VALUE,
    BlockSchedule,
    CollectPolicy,
    block_schedule_from_config,
    collect_policy_from_config,
    validate_against_loop,
    validate_collect_policy,
)

Tensor = torch.Tensor

FRAMEWORK = "alt_pncbf"

# Columns this framework appends to the joint trainer's METRIC_COLUMNS. All numeric, all appended at the
# END of the header by _append_csv(extend_header=True) on the first row, so every existing reader that
# selects columns by name is unaffected (jt_pncbf/train.py:2391-2420, the v2.8.3 D9 mechanism).
ALT_METRIC_COLUMNS = [
    "block",                    # 0.0 = value block, 1.0 = policy block  (invariant 6)
    "block_index",              # 0-based alternation cycle
    "k_v_active",               # value updates actually run this macro step
    "k_pi_active",              # policy updates actually run this macro step
    "grad_leak_pi_from_LV",     # ||grad on the HELD POLICY|| during a value block (must be 0)
    "held_param_drift",         # sum |theta_held - theta_held_at_block_start| (must be 0)
]

# `alt_collect.csv` — the per-collection record of WHICH passes ran, what the sigma controller did, and
# what each buffer looked like immediately before and after. Written ONLY when the collection gate is
# engaged (`CollectPolicy.engaged`) or when JT_ALT_COLLECT_LOG is set in the environment, so a default
# `collect: both / scale 1` run writes exactly the files ALTBLK writes and no others.
ALT_COLLECT_COLUMNS = [
    "step", "block", "block_index", "collect_mode", "n_episodes_active_scale",
    "trigger",                       # cadence | first_step | prime
    "dv_ran", "dpi_ran", "precursor_ran",
    "n_episodes_dv", "n_episodes_dpi",
    "sigma_v_before", "sigma_v_after", "sigma_v_held",
    "sigma_pi_used", "sigma_pi_source_is_schedule",
    "rho_unsafe_dv", "rho_unsafe_dpi",
    "dv_trajectories", "dv_transitions", "dv_next_traj_id", "dv_cont_gstep",
    "dpi_trajectories", "dpi_transitions", "dpi_next_traj_id", "dpi_cont_gstep",
    "precursor_trajectories", "precursor_transitions",
]


@dataclass(frozen=True)
class ALTTrainingResult:
    run_dir: Path
    halted: bool
    halt_reason: str | None
    last_value_loss: float
    last_policy_loss: float
    last_vs_grad_norm: float
    last_pi_grad_norm: float
    max_policy_grad_leak: float      # value grads seen during policy blocks (the JT-direction leak)
    max_value_grad_leak: float       # policy grads seen during value blocks (the reverse-direction leak)
    max_held_param_drift: float      # largest held-network parameter drift observed within a block
    n_value_blocks: int
    n_policy_blocks: int
    last_unsafe_fraction: float
    last_sigma: float


# ----------------------------------------------------------------------------------------------------
# Freeze auditing (invariant 4). Both helpers are one pass over a parameter list and cost one device->
# host sync each, the same budget the joint trainer already spends on its own per-step halt bools.
# ----------------------------------------------------------------------------------------------------

def _param_snapshot(module: nn.Module) -> list[Tensor]:
    """Detached clones of every parameter, taken once at the START of a block."""
    return [p.detach().clone() for p in module.parameters()]


def _param_drift(module: nn.Module, snapshot: list[Tensor]) -> Tensor | None:
    """Sum of |theta - theta_snapshot| as an ON-DEVICE scalar (no host sync here).

    Compared against exact zero by the caller: `not bool(drift == 0)` is True for any change AND for a
    NaN (nan == 0 is False), so a parameter that went non-finite is caught rather than missed by a
    `> 0` comparison.
    """
    total: Tensor | None = None
    for param, ref in zip(module.parameters(), snapshot, strict=True):
        term = (param.detach() - ref).abs().sum()
        total = term if total is None else total + term
    return total


def _collector_probe(buffer: Any) -> tuple[int, int, int, int]:
    """Everything about a replay buffer that a collection pass MOVES, as one comparable tuple.

    (trajectories, transitions, next_trajectory_id, continuing-collector global step). Read-only and
    allocation-free: `len(buffer._trajectories)` is the FIFO population the cap is applied to
    (oc_pncbf/collection.py:306), `len(buffer)` the transition count, `_next_trajectory_id` a monotone
    counter that keeps moving even when the FIFO is saturated (:218), and `_cont_state.gstep` the
    continuing collector's own step counter (continuing_collector.py:172), which keeps moving even when
    a round appends nothing. The last two are what make "the inactive buffer did not move" NON-VACUOUS
    at a saturated cap, where the first two are constant no matter what was written.
    """
    cont = getattr(buffer, "_cont_state", None)
    return (
        len(buffer._trajectories),
        len(buffer),
        int(buffer._next_trajectory_id),
        int(getattr(cont, "gstep", -1)) if cont is not None else -1,
    )


def _collect_for_block(
    *,
    block: str,
    collect_policy: CollectPolicy,
    system: Any,
    policy_net: nn.Module,
    value_net: ValueNetEnsemble,
    scene_sampler_fn: Any,
    np_rng: Any,
    torch_rng: torch.Generator,
    buffers: Any,
    episodes_per_collect: int,
    horizon: int,
    dt: float,
    config: Mapping[str, Any],
    sigma_v: float,
    sigma_pi: float,
    train_device: Any,
    train_dtype: Any,
) -> tuple[CollectionStats | None, CollectionStats | None, bool, int, int]:
    """One collection, running the passes `collect_policy` says this block runs.

    Returns (D_V stats or None, D_pi stats or None, precursor_ran, n_episodes_dv, n_episodes_dpi).
    A None stat means THE PASS DID NOT RUN — never that it ran and produced nothing — which is what the
    caller keys the sigma hold on.

    `collect: both / scale 1` takes the FIRST branch, which is the untouched `collect_jt` call the
    ALTBLK loop made, with the same arguments in the same order; nothing about that path is
    reimplemented here. The decomposed branch below reproduces `collect_jt`'s body
    (jt_pncbf/collection.py:226-277) pass for pass and in its order — D_V, then D_pi, then precursor —
    with the passes the block does not own omitted.
    """
    run_dv = collect_policy.runs_pass(VALUE, block)
    run_dpi = collect_policy.runs_pass(POLICY, block)
    n_dv = collect_policy.n_episodes_for(VALUE, block, episodes_per_collect)
    n_dpi = collect_policy.n_episodes_for(POLICY, block, episodes_per_collect)

    inj_cfg = ((config.get("loss") or {}).get("value") or {}).get("precursor_injection") or {}
    precursor_enabled = bool(inj_cfg.get("enabled", False))

    if run_dv and run_dpi and n_dv == episodes_per_collect and n_dpi == episodes_per_collect:
        v_stats, pi_stats = collect_jt(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            scene_sampler=scene_sampler_fn,
            rng=np_rng,
            torch_generator=torch_rng,
            buffers=buffers,
            n_episodes=episodes_per_collect,
            max_steps=horizon,
            dt=dt,
            config=config,
            sigma_v=sigma_v,
            sigma_pi=sigma_pi,
            storage_device=train_device,
            storage_dtype=train_dtype,
        )
        return v_stats, pi_stats, precursor_enabled, n_dv, n_dpi

    v_stats: CollectionStats | None = None
    pi_stats: CollectionStats | None = None
    if run_dv:
        v_stats = collect_policy_rollouts(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            scene_sampler=scene_sampler_fn,
            rng=np_rng,
            torch_generator=torch_rng,
            n_episodes=n_dv,
            max_steps=horizon,
            dt=dt,
            buffer=buffers.value,
            config=config,
            sigma=sigma_v,
            storage_device=train_device,
            storage_dtype=train_dtype,
        )
    if run_dpi:
        pi_stats = collect_policy_rollouts(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            scene_sampler=scene_sampler_fn,
            rng=np_rng,
            torch_generator=torch_rng,
            n_episodes=n_dpi,
            max_steps=horizon,
            dt=dt,
            buffer=buffers.policy,
            config=config,
            sigma=sigma_pi,
            storage_device=train_device,
            storage_dtype=train_dtype,
        )
    # The precursor pass writes ONLY into `buffers.precursor`, which only `_value_updates` reads
    # (jt_pncbf/train.py:1389,:1427). It therefore belongs to the value block and rides with the D_V
    # pass, at that pass's own episode count — `collect_jt` sets `n_precursors = n_episodes`
    # (jt_pncbf/collection.py:270), and that identity is preserved here.
    precursor_ran = False
    if run_dv and precursor_enabled:
        collect_precursors(
            system=system,
            policy_net=policy_net,
            value_net=value_net,
            scene_sampler=scene_sampler_fn,
            rng=np_rng,
            torch_generator=torch_rng,
            buffer=buffers.precursor,
            n_precursors=n_dv,
            max_steps=horizon,
            dt=dt,
            config=config,
            storage_device=train_device,
            storage_dtype=train_dtype,
        )
        precursor_ran = True
    return v_stats, pi_stats, precursor_ran, n_dv, n_dpi


def _held_value_scalars() -> dict[str, Any]:
    """The zeroed value-column dict for a POLICY block.

    Verbatim the dict the joint trainer already emits when it skips the value update
    (jt_pncbf/train.py:697): `_metrics_row` seeds every column to 0.0 before applying the scalar dicts
    (jt_pncbf/train.py:1930), so the metrics schema is unchanged. This exact dict has shipped in every
    maneuver/cpi run, so its schema consequence is proven in production data rather than argued.
    """
    return {"value_finite": True, "L_V_total": 0.0, "grad_norm_VS": 0.0, "L_R": 0.0}


def _refuse_unsupported(config: Mapping[str, Any]) -> None:
    """Refuse loudly every joint-training-only mechanism that has no meaning across a block boundary."""
    channel = str((config.get("safety_channel") or {}).get("type", "value"))
    if channel != "value":
        raise NotImplementedError(
            f"safety_channel.type={channel!r} replaces the LEARNED certificate with an analytic or "
            "frozen one. A policy block must improve the policy against the learned certificate held "
            "fixed, not against a different certificate; ALT-PNCBF supports safety_channel.type='value' "
            "only."
        )
    if bool((config["training"]["jt"].get("horizon_critic") or {}).get("enabled", False)):
        raise NotImplementedError(
            "training.jt.horizon_critic is a joint-training performance channel updated once per macro "
            "step irrespective of the block; it is not carried into ALT-PNCBF."
        )
    conditioning = str(config["value_target"].get("conditioning", "task_stored"))
    if conditioning not in ("task_stored", "brake"):
        raise NotImplementedError(
            f"value_target.conditioning={conditioning!r} maintains a second policy that is Polyak-stepped "
            "after every policy optimizer step ('task_raw_lagged') or trained alongside it "
            "('learned_recovery'). Neither has a defined meaning across a block boundary; ALT-PNCBF "
            "supports 'task_stored' and 'brake'."
        )
    if config["training"]["jt"].get("pi_init_ckpt") is not None:
        raise NotImplementedError(
            "training.jt.pi_init_ckpt (policy warm-start) is not carried into ALT-PNCBF; the alternation "
            "control inherits its parent joint run's certificate warm start (value_init_ckpt) only."
        )


def run_training(
    *,
    stage: str = "full",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int | None = None,
    smoke_eval_scenes: int = 2,
    device: str = "auto",
    train_dtype_name: str = "float32",
    system: str | None = None,
    obstacle_distribution: str | None = None,
    n_steps_override: int | None = None,
    value_batch_size_override: int | None = None,
    schedule_n_steps_override: int | None = None,
    value_block_override: int | None = None,
    policy_block_override: int | None = None,
    first_block_override: str | None = None,
    collect_override: str | None = None,
    n_episodes_active_scale_override: int | None = None,
) -> ALTTrainingResult:
    config = load_effective_config()
    config["run"]["version"] = __version__
    config["run"]["framework"] = FRAMEWORK
    if system is not None:
        config["run"]["system"] = system
    if seed is not None:
        config["run"]["seed"] = int(seed)
    if obstacle_distribution is not None:
        config["env"]["obstacle_distribution"] = obstacle_distribution
    for name, value in (
        ("n_steps_override", n_steps_override),
        ("value_batch_size_override", value_batch_size_override),
        ("schedule_n_steps_override", schedule_n_steps_override),
    ):
        if value is not None:
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
            config["run"][name] = int(value)

    # The block schedule is CONFIGURATION, in macro steps (invariant 5). It rides in through
    # `training.alt` — put there by the launcher's load_effective_config redirect — and any override
    # given here is written INTO the config before validation, so config.yaml is the single persisted
    # source of truth for what the blocks were.
    alt_cfg = dict(config.get("training", {}).get("alt", {}) or {})
    if value_block_override is not None or policy_block_override is not None or first_block_override is not None:
        alt_cfg["enabled"] = True
        if value_block_override is not None:
            alt_cfg["value_block"] = int(value_block_override)
        if policy_block_override is not None:
            alt_cfg["policy_block"] = int(policy_block_override)
        if first_block_override is not None:
            alt_cfg["first"] = str(first_block_override)
    # The two collection keys are overridable the same way, and like the block lengths the override is
    # written INTO the config before validation so config.yaml stays the single persisted source of
    # truth. Passing neither leaves `training.alt` exactly as the launcher supplied it.
    if collect_override is not None:
        alt_cfg["collect"] = str(collect_override)
    if n_episodes_active_scale_override is not None:
        alt_cfg["n_episodes_active_scale"] = int(n_episodes_active_scale_override)

    _refuse_unsupported(config)

    run_seed = int(config["run"]["seed"])
    np_rng = np.random.default_rng(run_seed)
    torch.manual_seed(run_seed)
    system = make_system(config)
    train_device = _resolve_train_device(device)
    train_dtype = _resolve_train_dtype(train_dtype_name)
    torch_rng = torch.Generator(device=train_device)
    torch_rng.manual_seed(run_seed)

    value_net = ValueNetEnsemble(system.obs_dim, config).to(device=train_device, dtype=train_dtype)
    target_value_net = deepcopy(value_net)
    target_value_net.requires_grad_(False)
    policy_net = _build_control_net(system, config).to(device=train_device, dtype=train_dtype)
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

    # Certificate warm start (v2.6.0 semantics, imported verbatim in behaviour): loads ONLY v_s_state
    # (+ target). Kept because every registered joint run carries it; without it an alternation run would
    # differ from its control on TWO axes (alternation AND cold start) instead of one.
    value_init_ckpt_cfg = config["training"]["jt"].get("value_init_ckpt")
    if value_init_ckpt_cfg is not None:
        import hashlib

        vi_path = Path(value_init_ckpt_cfg)
        vi_sha = hashlib.sha256(vi_path.read_bytes()).hexdigest()
        vi_ckpt = torch.load(vi_path, map_location=train_device, weights_only=False)
        value_net.load_state_dict(_state_to_dtype(vi_ckpt["v_s_state"], train_dtype))
        target_value_net.load_state_dict(
            _state_to_dtype(vi_ckpt.get("v_s_target_state", vi_ckpt["v_s_state"]), train_dtype))
        config["run"]["value_init_ckpt"] = str(vi_path)
        config["run"]["value_init_ckpt_sha256"] = vi_sha
        config["run"]["value_init_run_id"] = vi_path.parent.parent.name
        config["run"]["value_init_sha8"] = vi_sha[:8]

    jt_cfg = dict(config["training"]["jt"])
    collection_cfg = dict(config["collection"]["jt"])
    base_n_steps = int(jt_cfg["n_steps"])
    n_steps = int(n_steps_override) if n_steps_override is not None else base_n_steps
    config_schedule_n_steps = int(jt_cfg.get("schedule_n_steps", base_n_steps))
    schedule_n_steps = (
        int(schedule_n_steps_override) if schedule_n_steps_override is not None
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
        int(value_batch_size_override) if value_batch_size_override is not None else policy_batch_size
    )
    config["optim"]["batch_size_jt_value"] = value_batch_size
    metrics_log_every = int(config["optim"].get("metrics_log_every", 1))
    eval_cadence = int(config.get("eval", {}).get("cadence", 20000))
    eval_max_scenes = None
    if stage == "smoke":
        # The joint trainer's smoke reduction (jt_pncbf/train.py:576-590), with three additions that
        # exist so a smoke actually exercises the alternation rather than one block of it:
        #   * the budget is exactly TWO FULL CYCLES, so both blocks run twice and three boundaries are
        #     crossed (a JT smoke's flat 4 steps could sit inside one block);
        #   * one metrics row per macro step, so the block column and the two freeze audits are written
        #     at every step rather than once;
        #   * the eval/checkpoint cadence is the shorter block, so a checkpoint lands on every block
        #     boundary and the freeze is provable from the checkpoints alone, independently of the row
        #     the loop wrote (05_code §5, verification independence).
        alt_cfg.setdefault("enabled", True)
        alt_cfg.setdefault("value_block", 1)
        alt_cfg.setdefault("policy_block", 1)
        alt_cfg.setdefault("first", VALUE)
        _v_blk, _p_blk = int(alt_cfg["value_block"]), int(alt_cfg["policy_block"])
        n_steps = min(n_steps, 2 * (_v_blk + _p_blk))
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
        metrics_log_every = 1
        eval_cadence = min(_v_blk, _p_blk)
        eval_max_scenes = smoke_eval_scenes

    # Build and VALIDATE the schedule before a run directory exists, so a refused schedule leaves no
    # artifact behind.
    block_schedule = block_schedule_from_config(alt_cfg)
    collect_policy = collect_policy_from_config(alt_cfg)
    validate_against_loop(
        block_schedule,
        n_steps=n_steps,
        vs_warmup_steps=vs_warmup_steps,
        collect_every=collect_every,
        k_v=k_v,
        k_pi=k_pi,
    )
    validate_collect_policy(collect_policy, block_schedule, collect_every=collect_every)
    # `CollectPolicy.as_config` emits a key only when it is NOT at its default, so this assignment is
    # byte-identical to `block_schedule.as_config()` alone for a `both / 1` run — the ALTBLK config
    # subtree is reproduced exactly, and an ALTSEP config gains exactly the keys that differ.
    config["training"]["alt"] = {**block_schedule.as_config(), **collect_policy.as_config()}
    config["training"]["jt"]["vs_warmup_steps"] = vs_warmup_steps
    config["training"]["jt"]["K_V"] = k_v
    config["training"]["jt"]["K_pi"] = k_pi

    leak_threshold = float(config["halt"]["vs_grad_leak_threshold"])

    policy_buffer_cap = collection_cfg.get("policy_buffer_cap")
    buffers = make_replay_buffers(
        capacity=int(collection_cfg["buffer_cap"]),
        policy_capacity=int(policy_buffer_cap) if policy_buffer_cap is not None else None,
        store_state_seq=bool(((config.get('value_target') or {}).get('candidate_set') or {}).get('enabled', False)
                             or ((config.get('value_target') or {}).get('sigma_hazard') or {}).get('enabled', False)),
    )
    run_dir = _create_run_dir(output_root, config, run_seed, stage=stage)
    _initialize_run_dir(run_dir, config)
    _write_blocks_csv(run_dir, block_schedule, n_steps)
    writer = _make_summary_writer(run_dir / "tensorboard")
    scene_sampler = _train_scene_sampler(config)
    probe_scene = _make_probe_scene(config, system, train_device, train_dtype)
    sigma = float(config["schedules"]["sigma"]["init"])
    sigma_pi_base = float(config["schedules"]["sigma_pi"]["init"])
    start_time = time.time()
    timers = PhaseTimers()
    _eg = (config.get("eval", {}) or {}).get("gate", {}) or {}
    _eg_enabled = bool(_eg.get("enabled", True))
    _eg_timeout = float(_eg.get("timeout_s", 300.0))
    _eg_lock = Path(output_root) / "runs" / str(config["run"]["version"]) / "eval_gate.lock"
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
    max_value_leak = 0.0
    max_drift = 0.0
    n_value_blocks = 0
    n_policy_blocks = 0
    last_v_stats = CollectionStats(0, 0.0, sigma, sigma, 0.0, 0.0)
    last_pi_stats = CollectionStats(0, 0.0, sigma_pi_base, sigma_pi_base, 0.0, 0.0)
    prev_block: str | None = None
    held_snapshot: list[Tensor] = []
    held_target_snapshot: list[Tensor] = []
    # `alt_collect.csv` exists only when the collection gate is engaged, so a `both / 1` run writes the
    # ALTBLK file set exactly. JT_ALT_COLLECT_LOG turns the same OBSERVATION on for a default run without
    # touching the config schema or any training arithmetic; it is read once, here, and nowhere else.
    import os as _os
    collect_log_on = collect_policy.engaged or bool(_os.environ.get("JT_ALT_COLLECT_LOG"))

    _write_status(run_dir, stage=stage, phase="training", current_step=0, best_step=0,
                  best_cps=best_cps, halt_reason=None)

    for step in range(1, n_steps + 1):
        block = block_schedule.block_at(step)
        block_index = block_schedule.index_at(step)

        # --- INVARIANT 3, part 1: BOTH blocks collect, at the deployed cadence, under the policy and
        # certificate in force. The gate is the joint trainer's, on the MACRO step and on nothing else
        # (jt_pncbf/train.py:646), so a block's data is drawn under the networks that block holds.
        #
        # `collect: active_only` changes WHICH PASSES a collection runs, never WHEN a collection fires —
        # with ONE addition it cannot do without. Under `active_only` a buffer is fed only by its own
        # block, so the first macro step of the first block of a given kind finds that block's buffer
        # EMPTY unless the deployed cadence happens to fire on it, and `sample_tensor_batch` raises on an
        # empty buffer (oc_pncbf/collection.py:245). One PRIMING collection is run on that step, and on
        # no other, because after it that buffer is never empty again. Under `both` `prime` is
        # structurally False and the gate expression is the joint trainer's, unchanged.
        active_buffer = buffers.value if block == VALUE else buffers.policy
        prime = collect_policy.mode == COLLECT_ACTIVE_ONLY and len(active_buffer) == 0
        if step == 1 or step % max(1, collect_every) == 0 or prime:
            sigma_pi = _sigma_pi_at(config, sigma_pi_base, step)
            sigma_v_before = sigma
            pre_dv = _collector_probe(buffers.value)
            pre_dpi = _collector_probe(buffers.policy)
            _t = time.time()
            v_stats, pi_stats, precursor_ran, n_ep_dv, n_ep_dpi = _collect_for_block(
                block=block,
                collect_policy=collect_policy,
                system=system,
                policy_net=policy_net,
                value_net=target_value_net,
                scene_sampler_fn=lambda rng: scene_sampler(rng, config, system.name),
                np_rng=np_rng,
                torch_rng=torch_rng,
                buffers=buffers,
                episodes_per_collect=episodes_per_collect,
                horizon=horizon,
                dt=float(config["env"]["dt"]),
                config=config,
                sigma_v=sigma,
                sigma_pi=sigma_pi,
                train_device=train_device,
                train_dtype=train_dtype,
            )
            timers.t_collect += time.time() - _t
            post_dv = _collector_probe(buffers.value)
            post_dpi = _collector_probe(buffers.policy)

            # ==== THE HOLD ==============================================================
            # `sigma` (sigma_v) is the ONLY persistent collection controller in this loop: an EMA
            # advanced by `adaptive_sigma_update` from the D_V pass's unsafe fraction
            # (jt_pncbf/collection.py:280-296), read back at jt_pncbf/train.py:667. It is advanced ONLY
            # from a D_V pass THAT ACTUALLY RAN. When no D_V pass ran there is no fresh statistic, so the
            # controller is HELD — never stepped on the stale `last_v_stats` (which would re-apply an old
            # unsafe fraction and walk sigma toward a bound over a whole block) and never on the D_pi
            # pass's unsafe fraction (a statistic from the other buffer's distribution). `v_stats is
            # None` means THE PASS DID NOT RUN, not that it ran empty, so the two cases cannot be
            # confused.
            #
            # `sigma_pi` is NOT a controller in this framework and has no state to hold: it is
            # `_sigma_pi_at(config, sigma_pi_base, step)` (jt_pncbf/train.py:2291-2296), a pure
            # deterministic function of the macro step, recomputed above at every collection exactly as
            # the joint trainer recomputes it. `pi_stats.sigma_after` — the value `adaptive_sigma_update`
            # would give it — is DISCARDED here, as the joint trainer discards it (train.py:667 reads
            # `last_v_stats.sigma_after` only). Nothing below assigns a D_pi statistic to any sigma.
            sigma_v_held = v_stats is None
            if v_stats is not None:
                last_v_stats = v_stats
                sigma = v_stats.sigma_after
            if pi_stats is not None:
                last_pi_stats = pi_stats
            if sigma_v_held and not bool(sigma == sigma_v_before):
                raise RuntimeError(
                    f"alt sigma hold violated at macro step {step}: no D_V pass ran but sigma moved "
                    f"{sigma_v_before!r} -> {sigma!r}."
                )
            # A pass that did not run must have moved NOTHING. Checked on the buffers themselves rather
            # than on the code path, so a future edit that reintroduces a write is caught at the artifact
            # and not by reading this function.
            if v_stats is None and post_dv != pre_dv:
                raise RuntimeError(
                    f"alt collect hold violated at macro step {step}: no D_V pass ran but D_V moved "
                    f"{pre_dv} -> {post_dv} (trajectories, transitions, next_traj_id, cont_gstep)."
                )
            if pi_stats is None and post_dpi != pre_dpi:
                raise RuntimeError(
                    f"alt collect hold violated at macro step {step}: no D_pi pass ran but D_pi moved "
                    f"{pre_dpi} -> {post_dpi} (trajectories, transitions, next_traj_id, cont_gstep)."
                )

            if collect_log_on:
                pre_pre = _collector_probe(buffers.precursor)
                _collect_csv = run_dir / "alt_collect.csv"
                if not _collect_csv.exists():          # _append_csv writes no header to a new file
                    _init_csv(_collect_csv, ALT_COLLECT_COLUMNS)
                _append_csv(_collect_csv, ALT_COLLECT_COLUMNS, [{
                    "step": step,
                    "block": block,
                    "block_index": block_index,
                    "collect_mode": collect_policy.mode,
                    "n_episodes_active_scale": collect_policy.n_episodes_active_scale,
                    "trigger": ("first_step" if step == 1
                                else "cadence" if step % max(1, collect_every) == 0 else "prime"),
                    "dv_ran": int(v_stats is not None),
                    "dpi_ran": int(pi_stats is not None),
                    "precursor_ran": int(precursor_ran),
                    "n_episodes_dv": n_ep_dv if v_stats is not None else 0,
                    "n_episodes_dpi": n_ep_dpi if pi_stats is not None else 0,
                    "sigma_v_before": sigma_v_before,
                    "sigma_v_after": sigma,
                    "sigma_v_held": int(sigma_v_held),
                    "sigma_pi_used": sigma_pi,
                    "sigma_pi_source_is_schedule": 1,
                    "rho_unsafe_dv": (v_stats.unsafe_fraction if v_stats is not None else ""),
                    "rho_unsafe_dpi": (pi_stats.unsafe_fraction if pi_stats is not None else ""),
                    "dv_trajectories": post_dv[0], "dv_transitions": post_dv[1],
                    "dv_next_traj_id": post_dv[2], "dv_cont_gstep": post_dv[3],
                    "dpi_trajectories": post_dpi[0], "dpi_transitions": post_dpi[1],
                    "dpi_next_traj_id": post_dpi[2], "dpi_cont_gstep": post_dpi[3],
                    "precursor_trajectories": pre_pre[0], "precursor_transitions": pre_pre[1],
                }])

        # --- schedule clock: per macro step, byte-for-byte the joint trainer's (train.py:669-687) ---
        schedule_step = max(0, step - vs_warmup_steps)
        effective_steps = max(1, schedule_n_steps - vs_warmup_steps)
        schedule_step_clamped = min(schedule_step, effective_steps)
        lambda_disc = lambda_schedule_value(
            config["schedules"]["gamma_disc"], schedule_step_clamped, effective_steps,
            float(config["env"]["dt"]),
        )
        gamma_disc = gamma_from_lambda(lambda_disc, float(config["env"]["dt"]))
        target_rhs = schedule_value(config["schedules"]["target_rhs"], schedule_step, effective_steps)
        do_log = (step % metrics_log_every == 0) or (step == n_steps)

        # --- block entry: snapshot the network this block HOLDS, once per block --------------------
        if block != prev_block:
            if block == VALUE:
                held_snapshot = _param_snapshot(policy_net)
                held_target_snapshot = []
                n_value_blocks += 1
            else:
                held_snapshot = _param_snapshot(value_net)
                held_target_snapshot = _param_snapshot(target_value_net)
                n_policy_blocks += 1
            prev_block = block

        _t = time.time()
        leak_pi_from_lv = 0.0
        if block == VALUE:
            # ==== VALUE BLOCK: re-fit the certificate to rollouts collected under the HELD policy ====
            # INVARIANT 4, direction value->policy. The freeze is STRUCTURAL: `frozen_params` is the
            # repository's own idiom (jt_pncbf/losses.py:1284-1292), the one the joint trainer applies in
            # the other direction at losses.py:858. Policy grads are cleared on entry so the measurement
            # afterwards is of THIS block's macro step and nothing earlier.
            _zero_grads(policy_net.parameters())
            with frozen_params(policy_net):
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
                    recovery_policy=None,
                    lagged_policy=None,
                )
            if not value_scalars.pop("value_finite"):
                halt_reason = "nan_or_inf_L_V"
                break
            # INVARIANT 4, the MEASUREMENT the joint trainer has in one direction only. The freeze is
            # released by now (the context manager restored requires_grad on exit), so this reads the
            # .grad tensors themselves: any write onto the held policy during the block is caught.
            leak_sq = grad_sq_norm(policy_net.parameters())
            if leak_sq is not None:
                if bool(leak_sq > leak_threshold * leak_threshold):
                    max_value_leak = float(leak_sq.sqrt().item())
                    halt_reason = "alt_value_block_policy_gradient_leak"
                    break
                if do_log:
                    leak_pi_from_lv = float(leak_sq.sqrt().item())
                    max_value_leak = max(max_value_leak, leak_pi_from_lv)
            drift = _param_drift(policy_net, held_snapshot)
            policy_scalars = _zero_policy_scalars()
            if do_log:
                # Value scalars are recorded from VALUE blocks only; in a policy block the value columns
                # carry the held-zero dict, which is a block marker and not a measurement.
                last_value_loss = value_scalars["L_V_total"]
                last_vs_grad = value_scalars["grad_norm_VS"]
            k_v_active, k_pi_active = k_v, 0
        else:
            # ==== POLICY BLOCK: improve the policy against the certificate HELD FIXED for the block ===
            # INVARIANT 4, direction policy->value. The structural freeze is inside the imported loss:
            # `_zero_grads(value_net.parameters())` then `with frozen_params(value_net):` wrapping the
            # whole rollout and objective (jt_pncbf/losses.py:857-858). The measurement is inside the
            # imported update: `leak_sq = grad_sq_norm(value_net.parameters())` after backward, reduced
            # by max over the inner updates (jt_pncbf/train.py:1670, :1711) and compared against
            # halt.vs_grad_leak_threshold (:1715). NO outer frozen_params(value_net) is added here: it
            # would leave the value parameters requires_grad=False at backward time and make that
            # imported measurement vacuous. The freeze is not weakened — it is the same one, applied at
            # the site the joint trainer applies it, and its check is left able to fire.
            if step <= vs_warmup_steps:                      # unreachable: validate_against_loop refuses it
                raise RuntimeError(
                    f"policy block at macro step {step} inside the warmup (vs_warmup_steps="
                    f"{vs_warmup_steps}); the schedule validator should have refused this run."
                )
            value_scalars = _held_value_scalars()
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
                critic_net=None,
            )
            pi_finite = bool(policy_scalars.pop("pi_finite"))
            leak_exceeds = bool(policy_scalars.pop("leak_exceeds"))
            leak_sq_max = policy_scalars.pop("leak_sq_max")
            value_scalars.pop("value_finite")
            if not pi_finite:
                halt_reason = "nan_or_inf_L_pi"
                break
            if leak_exceeds:
                max_policy_leak = float(leak_sq_max.sqrt().item())
                halt_reason = "policy_gradient_leak"
                break
            if do_log:
                last_policy_loss = policy_scalars["L_pi_total"]
                last_pi_grad = policy_scalars["grad_norm_pi"]
                max_policy_leak = max(max_policy_leak, policy_scalars["grad_leak_VS_from_Lpi"])
            drift = _param_drift(value_net, held_snapshot)
            target_drift = _param_drift(target_value_net, held_target_snapshot)
            if target_drift is not None and not bool(target_drift == 0):
                max_drift = max(max_drift, float(target_drift.item()))
                halt_reason = "alt_policy_block_target_drift"
                break
            k_v_active, k_pi_active = 0, k_pi

        # --- INVARIANT 4, the strongest form of the freeze check: the held network's PARAMETERS have
        # not moved since this block began. `not (drift == 0)` is True for any change and also for NaN.
        held_drift = 0.0
        if drift is not None and not bool(drift == 0):
            halt_reason = ("alt_value_block_policy_drift" if block == VALUE
                           else "alt_policy_block_value_drift")
            max_drift = max(max_drift, float(drift.item()))
            break
        if do_log and drift is not None:
            held_drift = float(drift.item())
        timers.t_bptt += time.time() - _t

        if do_log:
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
                **timers.as_row(),
            )
            # INVARIANT 6: which block is active at this macro step, recorded as a column.
            row.update({
                "block": BLOCK_CODE[block],
                "block_index": float(block_index),
                "k_v_active": float(k_v_active),
                "k_pi_active": float(k_pi_active),
                "grad_leak_pi_from_LV": leak_pi_from_lv,
                "held_param_drift": held_drift,
            })
            last_row = row
            if step % metrics_log_every == 0:
                _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS + ALT_METRIC_COLUMNS, [row],
                            extend_header=True)
                _write_tb_scalars(writer, "train", row, step)
                _flush_jac_classes(run_dir)
                last_logged_step = step
                timers.reset()

        if step == n_steps or step % max(1, eval_cadence) == 0:
            with eval_gate(_eg_lock, enabled=_eg_enabled, timeout_s=_eg_timeout,
                           log=lambda m: print(m, flush=True)) as _gate_wait:
                timers.t_eval_wait += _gate_wait
                _t = time.time()
                # The DEPLOYED object is the joint framework's: same certificate net, same policy net,
                # same HardNet projection, same builder. Only the training loop differed.
                eval_framework = JTPNCBFFramework(system, value_net, policy_net, config)
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
                _record_eval(run_dir, writer, eval_result, step, config, system, value_net)
                timers.t_eval += time.time() - _t
            _sh_halt = _sigma_probe_halt_check(system, target_value_net, config, step, run_dir)
            if _sh_halt is not None:
                halt_reason = _sh_halt
                break
            if stage == "smoke":
                _cf = run_dir / "figures/inloop" / f"step_{step:06d}_cbf_contour.png"
                if not (_cf.exists() and _cf.stat().st_size > 0):
                    halt_reason = "smoke_contour_missing"
                    break
            cps = float(eval_result.eval_row["cps"])
            _t = time.time()
            if cps > best_cps + float(config["halt"]["early_stop_min_delta"]):
                best_cps = cps
                best_step = step
                _save_checkpoint(run_dir / "checkpoints/best.pt", value_net, target_value_net,
                                 policy_net, opt_vs, opt_pi, config, step, best_cps, best_step,
                                 block=block, block_index=block_index)
            _save_checkpoint(run_dir / f"checkpoints/step_{step:06d}.pt", value_net, target_value_net,
                             policy_net, opt_vs, opt_pi, config, step, best_cps, best_step,
                             block=block, block_index=block_index)
            timers.t_ckpt += time.time() - _t
        _write_status(run_dir, stage=stage, phase="training", current_step=step, best_step=best_step,
                      best_cps=best_cps, halt_reason=None)

    final_step = step if "step" in locals() else 0
    final_block = block_schedule.block_at(final_step) if final_step >= 1 else VALUE
    if last_row is not None and last_logged_step != final_step:
        _append_csv(run_dir / "metrics.csv", METRIC_COLUMNS + ALT_METRIC_COLUMNS, [last_row],
                    extend_header=True)
        _write_tb_scalars(writer, "train", last_row, final_step)
    _save_checkpoint(run_dir / "checkpoints/final.pt", value_net, target_value_net, policy_net,
                     opt_vs, opt_pi, config, final_step, best_cps, best_step,
                     block=final_block, block_index=max(0, block_schedule.index_at(max(1, final_step))))
    if not (run_dir / "checkpoints/best.pt").exists():
        _save_checkpoint(run_dir / "checkpoints/best.pt", value_net, target_value_net, policy_net,
                         opt_vs, opt_pi, config, final_step, best_cps, best_step,
                         block=final_block,
                         block_index=max(0, block_schedule.index_at(max(1, final_step))))
    phase = "halted" if halt_reason is not None else "done"
    _write_status(run_dir, stage=stage, phase=phase, current_step=final_step, best_step=best_step,
                  best_cps=best_cps, halt_reason=halt_reason)
    _write_report(run_dir, run_id=run_dir.name,
                  git_commit=(run_dir / "git_commit.txt").read_text(encoding="utf-8").strip(),
                  wallclock_s=time.time() - start_time, halt_reason=halt_reason)
    writer.close()
    _auto_run_final_eval(run_dir, max_scenes=eval_max_scenes if stage == "smoke" else None)
    _write_status(run_dir, stage=stage, phase=phase, current_step=final_step, best_step=best_step,
                  best_cps=best_cps, halt_reason=halt_reason)
    _write_report(run_dir, run_id=run_dir.name,
                  git_commit=(run_dir / "git_commit.txt").read_text(encoding="utf-8").strip(),
                  wallclock_s=time.time() - start_time, halt_reason=halt_reason)
    return ALTTrainingResult(
        run_dir=run_dir,
        halted=halt_reason is not None,
        halt_reason=halt_reason,
        last_value_loss=last_value_loss,
        last_policy_loss=last_policy_loss,
        last_vs_grad_norm=last_vs_grad,
        last_pi_grad_norm=last_pi_grad,
        max_policy_grad_leak=max_policy_leak,
        max_value_grad_leak=max_value_leak,
        max_held_param_drift=max_drift,
        n_value_blocks=n_value_blocks,
        n_policy_blocks=n_policy_blocks,
        last_unsafe_fraction=last_v_stats.unsafe_fraction,
        last_sigma=sigma,
    )


def _write_blocks_csv(run_dir: Path, schedule: BlockSchedule, n_steps: int) -> None:
    """One row per block: the alternation, written once, before any training happens.

    Additive artifact (no existing instrument reads it). The block at any macro step is ALSO recoverable
    from `config.yaml` alone, since `BlockSchedule.block_at` is a pure function of the persisted
    training.alt keys — this file only saves a reader the arithmetic.
    """
    import csv as _csv

    with (run_dir / "blocks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["block_ordinal", "cycle_index", "kind", "start_step", "end_step", "n_macro_steps"])
        for ordinal, (start, end, kind, cycle) in enumerate(schedule.blocks(n_steps)):
            w.writerow([ordinal, cycle, kind, start, end, end - start + 1])


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
    *,
    block: str,
    block_index: int,
) -> None:
    """Every key the joint trainer writes (jt_pncbf/train.py:1988-2017), same names, same order, so
    every checkpoint-reading instrument works unchanged — plus `framework: alt_pncbf` and the block the
    checkpoint was taken in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": __version__,
            "framework": FRAMEWORK,
            "step": int(step),
            "best_cps": float(best_cps),
            "best_step": int(best_step),
            "v_s_state": value_net.state_dict(),
            "v_s_target_state": target_value_net.state_dict(),
            "pi_state": policy_net.state_dict(),
            "opt_vs_state": opt_vs.state_dict(),
            "opt_pi_state": opt_pi.state_dict(),
            "config": _plain_data(config),
            "block": str(block),
            "block_index": int(block_index),
        },
        path,
    )


def load_framework_from_checkpoint(
    checkpoint_path: Path,
    config_overrides: Mapping[str, Any] | None = None,
) -> tuple[JTPNCBFFramework, Mapping[str, Any], dict[str, Any]]:
    """The registry entry point (src/eval/run_full.py). Returns the SAME deployed object the joint
    framework returns — certificate + policy + HardNet — because only the training loop differed."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ALT-PNCBF (alternating value/policy blocks).")
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-eval-scenes", type=int, default=2)
    parser.add_argument("--system", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--obstacle-distribution", choices=["random", "fixed_centered"], default=None)
    parser.add_argument("--alt-n-steps", type=int, default=None)
    parser.add_argument("--value-batch-size", type=int, default=None)
    parser.add_argument("--schedule-n-steps", type=int, default=None)
    parser.add_argument("--value-block", type=int, default=None,
                        help="macro steps per VALUE block (written into config.training.alt).")
    parser.add_argument("--policy-block", type=int, default=None,
                        help="macro steps per POLICY block (written into config.training.alt).")
    parser.add_argument("--first", choices=["value", "policy"], default=None)
    parser.add_argument("--collect", choices=[COLLECT_BOTH, COLLECT_ACTIVE_ONLY], default=None,
                        help="which of collect_jt's two passes a block runs "
                             "(written into config.training.alt.collect; default 'both').")
    parser.add_argument("--n-episodes-active-scale", type=int, default=None,
                        help="multiplier on n_episodes of the ACTIVE pass only "
                             "(written into config.training.alt.n_episodes_active_scale; default 1).")
    args = parser.parse_args()
    result = run_training(
        stage=args.stage,
        output_root=args.output_root,
        seed=args.seed,
        smoke_eval_scenes=args.smoke_eval_scenes,
        device=args.device,
        train_dtype_name=args.train_dtype,
        system=args.system,
        obstacle_distribution=args.obstacle_distribution,
        n_steps_override=args.alt_n_steps,
        value_batch_size_override=args.value_batch_size,
        schedule_n_steps_override=args.schedule_n_steps,
        value_block_override=args.value_block,
        policy_block_override=args.policy_block,
        first_block_override=args.first,
        collect_override=args.collect,
        n_episodes_active_scale_override=args.n_episodes_active_scale,
    )
    print(result.run_dir)
    return 0 if result.halt_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
