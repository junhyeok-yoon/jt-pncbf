from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from src.common.outcomes import (
    OutcomeResult,
    StepOutcomeMasks,
    resolve_outcome,
    step_outcomes,
)
from src.common.signed_h import hazard_geom, signed_h
from src.common.system import System
from src.envs.scene_batch import (
    batch_scenes as make_batched_scene,
    initial_states_from_batch,
)
from src.envs.scene_init import Scene
from src.eval.bootstrap import within_seed_ci
from src.eval.build_pools import EvaluationPool, load_pool, sha256_file
from src.eval.rollout import RolloutResult, rollout_eval


Tensor = torch.Tensor

EVAL_METRIC_COLUMNS = [
    "mode",
    "step",
    "ckpt_name",
    "pool_name",
    "pool_seed",
    "n_scenes",
    "cps",
    "reach",
    "collision",
    "oob",
    "stuck",
    "timeout",
    "infeasibility",
    "saturation_rate",
    "stuck_bin_00_05",
    "stuck_bin_05_10",
    "stuck_bin_10_15",
    "stuck_bin_15_20",
    "stuck_bin_20_25",
    "stuck_bin_25_30",
    "alpha_safe",
    "alpha_unsafe",
    "cps_ci_lo",
    "cps_ci_hi",
    "reach_ci_lo",
    "reach_ci_hi",
    "collision_ci_lo",
    "collision_ci_hi",
    "stuck_ci_lo",
    "stuck_ci_hi",
    "infeasibility_ci_lo",
    "infeasibility_ci_hi",
]

EVAL_EPISODE_COLUMNS = [
    "mode",
    "step",
    "ckpt_name",
    "episode_idx",
    "outcome",
    "n_steps",
    "cps_episode",
    "reach",
    "collision",
    "oob",
    "stuck",
    "timeout",
    "infeasible_step_frac",
    "empty_step_frac",
    "empty_source",
    "singular_step_frac",
    "saturation_step_frac",
    "min_window_displacement",
    "mean_proj_mag",
    "max_h",
    "traj_path_len",
    "angrate_at_reach",
    "collision_cause",
    "band_crossings",
    "first_crossing_step",
]


class EvaluatedFramework(Protocol):
    system: System

    def policy(self, x: Tensor, scene: Any) -> Tensor: ...

    def filter(self, x: Tensor, u_nom: Tensor, scene: Any) -> Any: ...


@dataclass(frozen=True)
class EpisodeTrajectory:
    scene: Scene
    filtered: RolloutResult
    filtered_outcome: str
    filtered_event_step: int
    lqr_states: Tensor | None = None
    lqr_outcome: str | None = None
    lqr_event_step: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    eval_row: dict[str, Any]
    episode_rows: list[dict[str, Any]]
    trajectories: list[EpisodeTrajectory]
    pool: EvaluationPool
    manifest: dict[str, Any] | None
    sha_verified: bool | None


def evaluate(
    framework: EvaluatedFramework,
    pool: EvaluationPool | str | Path,
    config: Mapping[str, Any],
    mode: str,
    *,
    step: int = 0,
    ckpt_name: str = "",
    max_scenes: int | None = None,
    include_lqr_baseline: bool = False,
    eval_batch_size: int | None = None,
) -> EvaluationResult:
    loaded_pool, manifest, sha_verified = _load_pool_with_manifest(pool)
    scenes = loaded_pool.scenes[:max_scenes] if max_scenes is not None else loaded_pool.scenes
    if not scenes:
        raise ValueError("Evaluation pool contains no scenes.")

    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    episode_rows: list[dict[str, Any]] = []
    trajectories: list[EpisodeTrajectory] = []

    batch_size = eval_batch_size or len(scenes)
    for batch_start in range(0, len(scenes), batch_size):
        batch_scenes = scenes[batch_start : batch_start + batch_size]
        dtype, device = _tensor_options(framework.system, framework)
        batched_scene = make_batched_scene(batch_scenes, device=device, dtype=dtype)
        x0 = initial_states_from_batch(batched_scene)
        filtered = rollout_eval(
            framework.system,
            framework.policy,
            _filter_adapter(framework),
            batched_scene,
            x0,
            max_steps=max_steps,
            dt=dt,
            config=config,
        )
        lqr_batch = None
        if include_lqr_baseline:
            with torch.no_grad():
                lqr_batch = _rollout_lqr_batch(
                    framework.system,
                    batched_scene,
                    x0,
                    max_steps=max_steps,
                    dt=dt,
                    config=config,
                )

        # v2.8.2 B1: outcome predicates computed ONCE per chunk on the batched [T,B,.] states
        # (step_outcomes / resolve_outcome / first_physical_event_step already accept the batch
        # dim; batched_scene carries per-scene obstacles + goal). ONE host transfer per chunk;
        # masks are sliced per scene in the loop. Bit-identical to the former per-scene calls (C1).
        chunk_masks = step_outcomes(filtered.states, batched_scene, framework.system, config)
        chunk_resolved = resolve_outcome(chunk_masks)
        chunk_phys = first_physical_event_step(chunk_masks)
        chunk_min_wd = eval_min_window_displacement_batched(chunk_masks, chunk_phys)
        chunk_band_step = chunk_masks.collided_band_lower | chunk_masks.collided_band_upper  # [T,B]
        event_step_cpu = chunk_resolved.event_step.detach().cpu()
        phys_cpu = chunk_phys.detach().cpu()
        min_wd_cpu = chunk_min_wd.detach().cpu()
        band_step_cpu = chunk_band_step.detach().cpu()
        causes_all = chunk_resolved.collision_cause

        # v2.9.0: the LQR-baseline outcome resolution, hoisted out of the per-scene loop exactly as
        # the policy path above already is. resolve_outcome walks the time axis ONCE for the whole
        # [T,B,.] chunk instead of once per scene, and the loop slices the result; the per-scene
        # calls performed ~360 blocking device->host synchronizations each. `lqr_outcome` and
        # `lqr_event_step` keep their present meaning and type for every consumer.
        lqr_resolved = None
        lqr_event_step_cpu = None
        if lqr_batch is not None:
            lqr_resolved = _resolve_states(
                lqr_batch,
                batched_scene,
                framework.system,
                config,
            )
            lqr_event_step_cpu = lqr_resolved.event_step.detach().cpu()

        for local_idx, scene in enumerate(batch_scenes):
            episode_idx = batch_start + local_idx
            filtered_i = _slice_rollout(filtered, local_idx)
            outcome = chunk_resolved.outcome[local_idx]
            event_step = int(event_step_cpu[local_idx].item())
            physical_event_step = int(phys_cpu[local_idx].item())
            active_steps = active_action_steps(
                physical_event_step,
                filtered_i.u_safe.shape[0],
            )
            min_window_displacement = float(min_wd_cpu[local_idx].item())
            # v2.8.0: per-episode band-crossing count + first crossing step (rising edges of the band
            # predicate over the active window). Always visible so the floor-permissive readout's use of
            # the permission is auditable; under band_terminates=true a crossing ends the episode so this
            # is 0/1, under permissive mode it counts every entry into the band during flight.
            band_step = band_step_cpu[:, local_idx]
            band_active = band_step[: active_steps] if active_steps > 0 else band_step[:0]
            if band_active.numel() > 0:
                prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=band_active.device),
                                  band_active[:-1]])
                band_crossings = int((band_active & ~prev).sum().item())
                nz = torch.nonzero(band_active)
                first_crossing_step = int(nz[0].item()) if nz.numel() > 0 else -1
            else:
                band_crossings, first_crossing_step = 0, -1
            lqr_states = None
            lqr_outcome = None
            lqr_event_step = None
            if lqr_batch is not None:
                lqr_states = lqr_batch[:, local_idx : local_idx + 1, :]
                lqr_outcome = lqr_resolved.outcome[local_idx]
                lqr_event_step = int(lqr_event_step_cpu[local_idx].item())

            episode_rows.append(
                _episode_row(
                    mode=mode,
                    step=step,
                    ckpt_name=ckpt_name,
                    episode_idx=episode_idx,
                    outcome=outcome,
                    event_step=event_step,
                    active_steps=active_steps,
                    min_window_displacement=min_window_displacement,
                    result=filtered_i,
                    scene=scene,
                    system=framework.system,
                    config=config,
                    collision_cause=(causes_all[local_idx] if causes_all else ""),
                    band_crossings=band_crossings,
                    first_crossing_step=first_crossing_step,
                )
            )
            trajectories.append(
                EpisodeTrajectory(
                    scene=scene,
                    filtered=filtered_i,
                    filtered_outcome=outcome,
                    filtered_event_step=event_step,
                    lqr_states=lqr_states,
                    lqr_outcome=lqr_outcome,
                    lqr_event_step=lqr_event_step,
                )
            )

    eval_row = _eval_row(
        mode=mode,
        step=step,
        ckpt_name=ckpt_name,
        pool=loaded_pool,
        n_scenes=len(scenes),
        episode_rows=episode_rows,
        config=config,
    )
    # v2.8.0 W1: dual-scoring in-loop split. When the pool manifest carries a per-IC provenance array
    # (the mixed pool), add the two provenance-half sub-scores beside the blended cps (which is unchanged
    # in name and remains the best.pt selection signal). Provenance-free pools add nothing -> today's row.
    prov = manifest.get("provenance") if hasattr(manifest, "get") else None
    if prov is not None and max_scenes is None and len(prov) == len(episode_rows):
        eval_row.update(provenance_half_scores(episode_rows, prov))
    return EvaluationResult(
        eval_row=eval_row,
        episode_rows=episode_rows,
        trajectories=trajectories,
        pool=loaded_pool,
        manifest=manifest,
        sha_verified=sha_verified,
    )


def initial_state(scene: Scene, system: System) -> Tensor:
    return initial_states([scene], system)


def initial_states(
    scenes: list[Scene],
    system: System,
    framework: EvaluatedFramework | None = None,
) -> Tensor:
    dtype, device = _tensor_options(system, framework)
    return initial_states_from_batch(make_batched_scene(scenes, device=device, dtype=dtype))


def _scene_initial_state_array(scene: Scene) -> np.ndarray:
    if scene.initial_velocity is not None:
        return np.concatenate([scene.start, scene.initial_velocity])
    if scene.initial_speed is None or scene.initial_heading is None:
        raise ValueError("Scene is missing initial speed or heading.")
    return np.array(
        [
            scene.start[0],
            scene.start[1],
            scene.initial_heading,
            scene.initial_speed,
        ],
        dtype=np.float64,
    )


def load_verified_pool(path: str | Path) -> tuple[EvaluationPool, dict[str, Any] | None, bool | None]:
    return _load_pool_with_manifest(Path(path))


def _tensor_options(
    system: System,
    framework: EvaluatedFramework | None,
) -> tuple[torch.dtype, torch.device]:
    value_net = getattr(framework, "value_net", None)
    if value_net is not None:
        try:
            parameter = next(value_net.parameters())
            return parameter.dtype, parameter.device
        except StopIteration:
            pass
    return system.u_bounds.dtype, system.u_bounds.device


def _slice_rollout(result: RolloutResult, batch_index: int) -> RolloutResult:
    return RolloutResult(
        states=result.states[:, batch_index : batch_index + 1, :],
        u_nom=result.u_nom[:, batch_index : batch_index + 1, :],
        u_safe=result.u_safe[:, batch_index : batch_index + 1, :],
        intervention_mask=result.intervention_mask[:, batch_index : batch_index + 1],
        infeasible=result.infeasible[:, batch_index : batch_index + 1],
        empty=None if result.empty is None else result.empty[:, batch_index : batch_index + 1],
        singular=None if result.singular is None else result.singular[:, batch_index : batch_index + 1],
        # v2.8.3 D1: the provenance flag must survive the per-episode slice, or every episode row
        # reports "alias" no matter what the batch rollout actually captured.
        empty_is_native=getattr(result, "empty_is_native", False),
    )


def _rollout_lqr_batch(
    system: System,
    scene: Any,
    x0: Tensor,
    max_steps: int,
    dt: float,
    config: Mapping[str, Any],
) -> Tensor:
    def policy_fn(x: Tensor, policy_scene: Any) -> Tensor:
        goal = torch.as_tensor(policy_scene.goal, dtype=x.dtype, device=x.device)
        return system.lqr_action(x, goal)

    def filter_fn(
        x: Tensor,
        u_nom: Tensor,
        filter_scene: Any,
    ) -> tuple[Tensor, Tensor]:
        return u_nom, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    return rollout_eval(
        system,
        policy_fn,
        filter_fn,
        scene,
        x0,
        max_steps,
        dt,
        config=config,
    ).states


def _load_pool_with_manifest(
    pool: EvaluationPool | str | Path,
) -> tuple[EvaluationPool, dict[str, Any] | None, bool | None]:
    if isinstance(pool, EvaluationPool):
        return pool, None, None

    pool_path = Path(pool)
    loaded_pool = load_pool(pool_path)
    manifest_path = pool_path.with_name(pool_path.name.replace(".pkl", ".manifest.json"))
    if not manifest_path.exists():
        print(f"Warning: pool manifest missing for {pool_path}.")
        return loaded_pool, None, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(pool_path)
    expected_sha = str(manifest.get("pool_sha256", ""))
    sha_verified = actual_sha == expected_sha
    if not sha_verified:
        print(
            "Warning: pool SHA mismatch for "
            f"{pool_path}: manifest={expected_sha}, actual={actual_sha}."
        )
    return loaded_pool, manifest, sha_verified


def _filter_adapter(
    framework: EvaluatedFramework,
) -> Any:
    def wrapped(x: Tensor, u_nom: Tensor, scene: Any) -> tuple[Tensor, Tensor]:
        filtered = framework.filter(x, u_nom, scene)
        if not isinstance(filtered, tuple):
            raise TypeError("framework.filter must return a tuple.")
        if len(filtered) < 2:
            raise ValueError("framework.filter must return at least (u_safe, infeasible).")
        return filtered[0], filtered[1].to(device=x.device, dtype=torch.bool)

    # v2.8.3 D1 ROOT REPAIR. rollout_eval resolves the filter object as
    # `filter_fn.__self__._filter` to read last_empty / last_singular. This adapter is a plain closure,
    # so `__self__` does not exist, the lookup returned None, and empty_steps fell back to `infeasible`
    # -- which is why empty_step_frac has been a strict ALIAS of infeasible_step_frac (D1). Attaching
    # the filter here restores the intended source without changing what the adapter computes.
    wrapped._filter = getattr(framework, "_filter", None)
    return wrapped


def _resolve_states(
    states: Tensor,
    scene: Scene,
    system: System,
    config: Mapping[str, Any],
) -> OutcomeResult:
    return resolve_outcome(step_outcomes(states, scene, system, config))


def _episode_row(
    *,
    mode: str,
    step: int,
    ckpt_name: str,
    episode_idx: int,
    outcome: str,
    event_step: int,
    active_steps: int,
    min_window_displacement: float,
    result: RolloutResult,
    scene: Scene,
    system: System,
    config: Mapping[str, Any],
    collision_cause: str = "",
    band_crossings: int = 0,
    first_crossing_step: int = -1,
) -> dict[str, Any]:
    reach = 1.0 if outcome == "goal" else 0.0
    collision = 1.0 if outcome == "collision" else 0.0
    oob = 1.0 if outcome == "oob" else 0.0
    stuck = 1.0 if outcome == "stuck" else 0.0
    timeout = 1.0 if outcome == "timeout" else 0.0
    infeasible_step_frac = active_bool_fraction(result.infeasible, active_steps)
    # v2.8.3 D1: provenance is explicit. `last_empty` = the filter's own empty_intersection flag;
    # `alias` = the legacy fallback where empty is indistinguishable from infeasible and the
    # empty leg of any prediction is NOT scoreable from this column.
    empty_source = "last_empty" if getattr(result, "empty_is_native", False) else "alias"
    empty_step_frac = active_bool_fraction(result.empty, active_steps) if result.empty is not None else infeasible_step_frac
    singular_step_frac = active_bool_fraction(result.singular, active_steps) if result.singular is not None else 0.0
    saturation_frac = saturation_step_fraction(result, system, active_steps=active_steps)
    cps_episode = (
        reach
        - 2.0 * collision
        - stuck
        - 0.5 * (oob + timeout)
        - 0.3 * infeasible_step_frac
    )
    projection = torch.linalg.norm(result.u_safe - result.u_nom, dim=-1)
    positions = system.position(result.states)
    # v2.8.5: the reported per-step h series / max_h is a DIAGNOSTIC (it enters no score); it follows
    # the configured geometric term so the diagnostic and the label agree. hazard.geom_form absent or
    # 'clip' -> bit-identical to v2.8.4.
    _geom_form, _ell = hazard_geom(config)
    h_values = signed_h(positions, scene, float(config["env"]["h_scale"]),
                        geom_form=_geom_form, ell=_ell)
    path_delta = positions[1:] - positions[:-1]
    # v2.8.0: angular rate at the reach instant (first goal step); nan otherwise.
    if outcome == "goal" and 0 <= event_step < result.states.shape[0]:
        angrate_at_reach = float(system.angular_rate(result.states[event_step])[0].item())
    else:
        angrate_at_reach = float("nan")
    return {
        "mode": mode,
        "step": int(step),
        "ckpt_name": ckpt_name,
        "episode_idx": int(episode_idx),
        "outcome": outcome,
        "n_steps": int(event_step if event_step >= 0 else result.u_safe.shape[0]),
        "cps_episode": float(cps_episode),
        "reach": reach,
        "collision": collision,
        "oob": oob,
        "stuck": stuck,
        "timeout": timeout,
        "infeasible_step_frac": infeasible_step_frac,
        "empty_step_frac": empty_step_frac,
        "empty_source": empty_source,
        "singular_step_frac": singular_step_frac,
        "saturation_step_frac": saturation_frac,
        "min_window_displacement": min_window_displacement,
        "mean_proj_mag": float(projection.mean().item()) if projection.numel() else 0.0,
        "max_h": float(h_values.max().item()),
        "traj_path_len": float(torch.linalg.norm(path_delta, dim=-1).sum().item()),
        "angrate_at_reach": angrate_at_reach,
        "collision_cause": collision_cause,
        "band_crossings": int(band_crossings),
        "first_crossing_step": int(first_crossing_step),
    }


def saturation_step_fraction(
    result: RolloutResult,
    system: System,
    *,
    active_steps: int | None = None,
) -> float:
    if result.u_safe.numel() == 0:
        return 0.0
    u_safe = result.u_safe if active_steps is None else result.u_safe[:active_steps]
    if u_safe.numel() == 0:
        return 0.0
    bounds = system.u_bounds.to(device=result.u_safe.device, dtype=result.u_safe.dtype)
    lower_dist = torch.abs(u_safe - bounds[:, 0])
    upper_dist = torch.abs(u_safe - bounds[:, 1])
    saturated = torch.any(torch.minimum(lower_dist, upper_dist) <= 1.0e-3, dim=-1)
    return float(saturated.double().mean().item())


def active_action_steps(physical_event_step: int, max_action_steps: int) -> int:
    if physical_event_step >= 0:
        return max(0, min(int(physical_event_step), int(max_action_steps)))
    return int(max_action_steps)


def active_bool_fraction(values: Tensor, active_steps: int) -> float:
    if values.numel() == 0 or active_steps <= 0:
        return 0.0
    return float(values[:active_steps].double().mean().item())


def eval_min_window_displacement(
    step_masks: StepOutcomeMasks,
    physical_event_step: int,
) -> Tensor:
    window = step_masks.window_displacement
    if physical_event_step >= 0:
        window = window[: physical_event_step + 1]
    finite_values = torch.where(
        torch.isfinite(window),
        window,
        torch.full_like(window, float("inf")),
    )
    min_value = finite_values.amin(dim=0)
    return torch.where(
        torch.isinf(min_value),
        torch.full_like(min_value, float("nan")),
        min_value,
    )


def eval_min_window_displacement_batched(
    step_masks: StepOutcomeMasks,
    physical_event_steps: Tensor,
) -> Tensor:
    """v2.8.2 B1: batched equivalent of eval_min_window_displacement over all B columns at once.
    Each column is cut at its OWN physical_event_step (>=0) or uses the full window (<0), then the
    finite min over the kept rows is taken -- bit-identical, per column, to the scalar-cut version."""
    window = step_masks.window_displacement                                  # [T, B]
    n_steps, _ = window.shape
    row = torch.arange(n_steps, device=window.device).unsqueeze(1)           # [T, 1]
    cut = torch.where(
        physical_event_steps >= 0,
        physical_event_steps.to(row.dtype),
        torch.full_like(physical_event_steps, n_steps - 1, dtype=row.dtype),
    )                                                                        # [B] (<0 -> full window)
    keep = row <= cut.unsqueeze(0)                                           # [T, B]
    finite_values = torch.where(
        torch.isfinite(window) & keep,
        window,
        torch.full_like(window, float("inf")),
    )
    min_value = finite_values.amin(dim=0)                                    # [B]
    return torch.where(
        torch.isinf(min_value),
        torch.full_like(min_value, float("nan")),
        min_value,
    )


def first_physical_event_step(step_masks: StepOutcomeMasks) -> Tensor:
    physical = step_masks.collided | step_masks.goal_reached | step_masks.oob
    if physical.ndim != 2:
        raise ValueError(f"Expected masks with shape [T, B], got {tuple(physical.shape)}")
    n_steps, batch_size = physical.shape
    event_step = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=physical.device,
    )
    unresolved = torch.ones(batch_size, dtype=torch.bool, device=physical.device)
    for step in range(n_steps):
        fired = unresolved & physical[step]
        if bool(fired.any()):
            event_step[fired] = step
            unresolved[fired] = False
        if not bool(unresolved.any()):
            break
    return event_step


def provenance_half_scores(
    episode_rows: list[Mapping[str, Any]], provenance: list[str]
) -> dict[str, float]:
    """v2.8.0 W1: split per-episode cps by provenance flag into `cps_full_half` / `cps_tilt60_half`.
    By construction the blended cps (mean over all episodes) equals the episode-weighted mean of the halves,
    so the two halves and the blended value are consistent. NaN for an absent group."""
    cps = np.array([float(e["cps_episode"]) for e in episode_rows], dtype=np.float64)
    prov = np.array(list(provenance))
    out: dict[str, float] = {}
    for tag, col in (("full", "cps_full_half"), ("tilt60", "cps_tilt60_half")):
        m = prov == tag
        out[col] = float(cps[m].mean()) if bool(m.any()) else float("nan")
    return out


def _eval_row(
    *,
    mode: str,
    step: int,
    ckpt_name: str,
    pool: EvaluationPool,
    n_scenes: int,
    episode_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap_cfg = config["eval"]["bootstrap"]
    ci_result = within_seed_ci(
        episode_rows,
        n_resample=int(bootstrap_cfg["n_resample"]),
        seed=int(bootstrap_cfg["seed"]),
    )
    mean = ci_result["mean"]
    ci = ci_result["ci"]
    saturation_rate = float(
        np.mean([float(row["saturation_step_frac"]) for row in episode_rows])
    )
    stuck_bins = stuck_bin_fractions(episode_rows)
    return {
        "mode": mode,
        "step": int(step),
        "ckpt_name": ckpt_name,
        "pool_name": pool.name,
        "pool_seed": int(pool.seed),
        "n_scenes": int(n_scenes),
        "cps": mean["cps"],
        "reach": mean["reach"],
        "collision": mean["collision"],
        "oob": mean["oob"],
        "stuck": mean["stuck"],
        "timeout": mean["timeout"],
        "infeasibility": mean["infeasibility"],
        "saturation_rate": saturation_rate,
        **stuck_bins,
        "alpha_safe": float(config["filter"]["alpha_safe"]),
        "alpha_unsafe": float(config["filter"]["alpha_unsafe"]),
        "cps_ci_lo": ci["cps"]["lo"],
        "cps_ci_hi": ci["cps"]["hi"],
        "reach_ci_lo": ci["reach"]["lo"],
        "reach_ci_hi": ci["reach"]["hi"],
        "collision_ci_lo": ci["collision"]["lo"],
        "collision_ci_hi": ci["collision"]["hi"],
        "stuck_ci_lo": ci["stuck"]["lo"],
        "stuck_ci_hi": ci["stuck"]["hi"],
        "infeasibility_ci_lo": ci["infeasibility"]["lo"],
        "infeasibility_ci_hi": ci["infeasibility"]["hi"],
    }


def stuck_bin_fractions(episode_rows: list[Mapping[str, Any]]) -> dict[str, float]:
    bins = {
        "stuck_bin_00_05": 0,
        "stuck_bin_05_10": 0,
        "stuck_bin_10_15": 0,
        "stuck_bin_15_20": 0,
        "stuck_bin_20_25": 0,
        "stuck_bin_25_30": 0,
    }
    if not episode_rows:
        return {key: 0.0 for key in bins}

    edges = [
        (0.00, 0.05, "stuck_bin_00_05"),
        (0.05, 0.10, "stuck_bin_05_10"),
        (0.10, 0.15, "stuck_bin_10_15"),
        (0.15, 0.20, "stuck_bin_15_20"),
        (0.20, 0.25, "stuck_bin_20_25"),
        (0.25, 0.30, "stuck_bin_25_30"),
    ]
    for row in episode_rows:
        value = float(row["min_window_displacement"])
        if not np.isfinite(value):
            continue
        for lo, hi, key in edges:
            if lo <= value < hi:
                bins[key] += 1
                break
    total = float(len(episode_rows))
    return {key: count / total for key, count in bins.items()}
