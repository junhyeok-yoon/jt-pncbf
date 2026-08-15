from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from src.common.system import System


Tensor = torch.Tensor


@dataclass(frozen=True)
class StepOutcomeMasks:
    collided: Tensor
    goal_reached: Tensor
    oob: Tensor
    stuck: Tensor
    window_displacement: Tensor
    # v2.8.0: collision CAUSE channels, recorded alongside `collided` (which stays bit-identical).
    # obstacle = xy-cylinder contact; band_lower/band_upper = floor/ceiling surface (p_z <= -band / >= +band).
    # None means the caller did not compute the cause (older/hand-built masks) -> cause "not_recorded".
    collided_obstacle: Tensor | None = None
    collided_band_lower: Tensor | None = None
    collided_band_upper: Tensor | None = None


@dataclass(frozen=True)
class OutcomeResult:
    outcome: list[str]
    event_step: Tensor
    min_window_displacement: Tensor
    # v2.8.0: per-episode collision cause at the collision step ("" if not a collision; "not_recorded"
    # if the cause channels were absent). Priority when several fire at the same step, in evaluation order:
    # obstacle > band_lower > band_upper.
    collision_cause: list[str] = field(default_factory=list)


def _reach_angular_rate(
    system: System,
    states: Tensor,
    actions: Tensor | None,
) -> Tensor:
    """v2.9.1 item 2 — the third leg of the reach predicate, for systems whose turn rate is a CONTROL
    INPUT rather than a state.

    `System.angular_rate(x)` reads the angular rate off the state. On the unicycle the turn rate is
    `u[1]` (`unicycle.dynamics`: dtheta/dt = omega = u[:,1]), so `angular_rate` is a hard structural
    zero and the third leg was vacuous there: a unicycle turning at omega_max could be scored as
    having arrived. A system may therefore expose `commanded_angular_rate(x, u)`; when it does AND the
    caller supplies the executed action stack, that is what the predicate reads.

    Alignment is BACKWARD (zero-order hold): state x_t was produced by the command u_{t-1} held over
    [t-1, t), so the turn rate the body carried on arriving at x_t is |u_{t-1}[1]|. The initial state
    x_0 precedes every command; its commanded rate is undefined and is taken as 0 — which is exactly
    the pre-v2.9.1 structural value at that one index. Backward alignment also keeps the outcome at
    step t causal (it depends only on controls applied up to t).

    Systems that do NOT define `commanded_angular_rate` (double_integrator, quadrotor_planar,
    quadrotor_3d) take the `system.angular_rate(states)` branch unchanged, whatever `actions` is —
    so their masks are bit-identical to the pre-v2.9.1 function by construction.
    """
    commanded = getattr(system, "commanded_angular_rate", None)
    if commanded is None or actions is None:
        return system.angular_rate(states)
    if actions.ndim != states.ndim:
        raise ValueError(
            f"actions must have the same rank as states, got {tuple(actions.shape)} "
            f"vs {tuple(states.shape)}."
        )
    n_states = states.shape[0]
    n_actions = actions.shape[0]
    if n_actions == n_states:
        aligned = actions
    elif n_actions == n_states - 1:
        aligned = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    else:
        raise ValueError(
            f"actions has {n_actions} steps, expected {n_states} or {n_states - 1} "
            f"for states with {n_states} steps."
        )
    return commanded(states, aligned.to(device=states.device, dtype=states.dtype))


def step_outcomes(
    states: Tensor,
    scene: Any,
    system: System,
    config: Mapping[str, Any],
    actions: Tensor | None = None,
) -> StepOutcomeMasks:
    positions = system.position(states)
    obstacle = _collided_exact(positions, scene)
    # v2.7.6 Stage-2: the arena band |p_z| >= band_collision_limit is a collision surface (01_env s1.6
    # priority; collision outranks goal/oob in resolve_outcome). Config-gated: 0.0 = OFF (legacy, pre-Stage-2
    # scoring, keeps the oob predicate as the only z outcome). 3D systems only. xy collision unchanged; the
    # oob predicate (|z|>8) is UNCHANGED (collision at |z|>=4 simply preempts it in z, still live in xy).
    # v2.8.0: the band surfaces are split at the predicate into lower/upper (floor/ceiling) rather than
    # inferred later from the sign of p_z (the abs() below would discard which surface was hit). The union
    # obstacle | band_lower | band_upper is bit-identical to the previous `collided | (|p_z| >= band_z)`.
    band_z = float(config["env"].get("band_collision_limit", 0.0))
    band_lower = torch.zeros_like(obstacle)
    band_upper = torch.zeros_like(obstacle)
    if band_z > 0.0 and positions.shape[-1] >= 3:
        band_lower = positions[..., 2] <= -band_z
        band_upper = positions[..., 2] >= band_z
    # v2.8.0 floor-permissive readout: env.band_terminates (default True) gates whether crossing the band
    # counts as a collision (and thereby ends the episode). When False, the band surfaces stay in the
    # dynamics and observation (unchanged system) but a crossing sets neither `collided` nor an
    # episode-ending condition; the band_lower/band_upper channels are still populated so a per-episode
    # crossing count remains visible. Default True is bit-identical to the pre-v2.8.0 predicate.
    band_terminates = bool(config["env"].get("band_terminates", True))
    collided = (obstacle | band_lower | band_upper) if band_terminates else obstacle

    goal = torch.as_tensor(scene.goal, dtype=states.dtype, device=states.device)
    goal_distance = torch.linalg.norm(positions - goal, dim=-1)
    speed = system.speed(states)
    angrate = _reach_angular_rate(system, states, actions)
    goal_angrate_radius = float(config["env"].get("goal_angrate_radius", float("inf")))
    goal_reached = (
        (goal_distance <= float(config["env"]["goal_radius"]))
        & (speed <= float(config["env"]["goal_speed_radius"]))
        & (angrate <= goal_angrate_radius)
    )

    oob_limit = float(config["env"]["oob_limit"])
    oob = torch.any(torch.abs(positions) > oob_limit, dim=-1)
    window_disp = window_displacement(
        positions,
        int(config["env"].get("stuck_window_steps", 60)),
    )
    stuck = window_disp <= float(config["env"].get("stuck_radius", 0.10))
    return StepOutcomeMasks(
        collided=collided,
        goal_reached=goal_reached,
        oob=oob,
        stuck=stuck,
        window_displacement=window_disp,
        collided_obstacle=obstacle,
        collided_band_lower=band_lower,
        collided_band_upper=band_upper,
    )


def _collided_exact(positions: Tensor, scene: Any) -> Tensor:
    centers = torch.as_tensor(
        scene.obstacle_centers,
        dtype=positions.dtype,
        device=positions.device,
    )
    radii = torch.as_tensor(
        scene.obstacle_radii,
        dtype=positions.dtype,
        device=positions.device,
    )
    active = torch.as_tensor(
        scene.obstacle_active,
        dtype=torch.bool,
        device=positions.device,
    )
    # Obstacles live in the first centers.shape[-1] position coordinates (xy footprint for infinite
    # vertical cylinders; a no-op when position dim == center dim, e.g. DI/unicycle/planar).
    pos_o = positions[..., : centers.shape[-1]]
    distance = torch.linalg.norm(pos_o.unsqueeze(-2) - centers, dim=-1)
    return ((distance < radii) & active).any(dim=-1)


# v2.8.0 D4/F1: the unfold in window_displacement materializes a [T-W, W+1, B, D] tensor; at a 500 Hz control
# rate (T=5000, W=1500, B=2000) that single intermediate is ~117 GiB and OOMs the shared eval path
# (rollout_eval -> _physical_done_mask -> step_outcomes -> here). Because every episode's window statistic is
# independent, processing the batch in episode chunks is BIT-IDENTICAL to the single-pass computation (proved
# in tests/test_window_displacement_chunk.py); this budget caps the intermediate's element count per chunk.
_WINDOW_ELEM_BUDGET = 3.0e8


def window_displacement(positions: Tensor, window_steps: int, batch_chunk: int | None = None) -> Tensor:
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError(
            "positions must have shape [T, B, D] with D >= 2, "
            f"got {tuple(positions.shape)}."
        )
    if window_steps < 1:
        raise ValueError(f"window_steps must be positive, got {window_steps}.")

    n_steps, batch_size, dim = positions.shape
    result = positions.new_full((n_steps, batch_size), float("nan"))
    if n_steps <= window_steps:
        return result

    if batch_chunk is None:
        elem_per_ep = (n_steps - window_steps) * (window_steps + 1) * dim
        batch_chunk = int(min(batch_size, max(1, _WINDOW_ELEM_BUDGET // max(elem_per_ep, 1))))
    batch_chunk = max(1, int(batch_chunk))
    for start in range(0, batch_size, batch_chunk):
        stop = min(start + batch_chunk, batch_size)
        windows = positions[:, start:stop].unfold(0, window_steps + 1, 1).permute(0, 3, 1, 2)
        anchor = windows[:, :1, :, :]
        result[window_steps:, start:stop] = torch.linalg.norm(windows - anchor, dim=-1).amax(dim=1)
    return result


def resolve_outcome(step_masks: StepOutcomeMasks) -> OutcomeResult:
    collided = step_masks.collided
    if collided.ndim != 2:
        raise ValueError(
            f"Expected masks with shape [T, B], got {tuple(collided.shape)}"
        )

    n_steps, batch_size = collided.shape
    outcomes = ["timeout"] * batch_size
    event_step = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=collided.device,
    )
    unresolved = torch.ones(batch_size, dtype=torch.bool, device=collided.device)
    min_window_disp = _min_window_displacement(step_masks.window_displacement)

    for step in range(n_steps):
        step_collision = unresolved & step_masks.collided[step]
        _record_events(
            outcomes,
            event_step,
            unresolved,
            step_collision,
            step,
            "collision",
        )

        step_goal = unresolved & step_masks.goal_reached[step]
        _record_events(outcomes, event_step, unresolved, step_goal, step, "goal")

        step_oob = unresolved & step_masks.oob[step]
        _record_events(outcomes, event_step, unresolved, step_oob, step, "oob")

        step_stuck = unresolved & step_masks.stuck[step]
        _record_events(outcomes, event_step, unresolved, step_stuck, step, "stuck")

        if not bool(unresolved.any()):
            break

    return OutcomeResult(
        outcome=outcomes,
        event_step=event_step,
        min_window_displacement=min_window_disp,
        collision_cause=_collision_causes(step_masks, outcomes, event_step),
    )


def _collision_causes(
    step_masks: StepOutcomeMasks, outcomes: list[str], event_step: Tensor
) -> list[str]:
    """v2.8.0: per-episode collision cause at the resolved collision step. Priority, in the order the
    predicate evaluates them: obstacle > band_lower > band_upper. "" for non-collision episodes;
    "not_recorded" when the cause channels were absent."""
    n = len(outcomes)
    if step_masks.collided_obstacle is None:
        return ["not_recorded" if o == "collision" else "" for o in outcomes]
    rows = torch.arange(n, device=event_step.device)
    es = event_step.clamp(min=0)
    obs = step_masks.collided_obstacle[es, rows].cpu().tolist()
    bl = step_masks.collided_band_lower[es, rows].cpu().tolist()
    bu = step_masks.collided_band_upper[es, rows].cpu().tolist()
    causes = []
    for b in range(n):
        if outcomes[b] != "collision":
            causes.append("")
        elif obs[b]:
            causes.append("obstacle")
        elif bl[b]:
            causes.append("band_lower")
        elif bu[b]:
            causes.append("band_upper")
        else:
            causes.append("unknown")           # unreachable: collided step always has a cause
    return causes


def _min_window_displacement(window_displacement_value: Tensor) -> Tensor:
    finite_values = torch.where(
        torch.isfinite(window_displacement_value),
        window_displacement_value,
        torch.full_like(window_displacement_value, float("inf")),
    )
    min_value = finite_values.amin(dim=0)
    return torch.where(
        torch.isinf(min_value),
        torch.full_like(min_value, float("nan")),
        min_value,
    )


def _record_events(
    outcomes: list[str],
    event_step: Tensor,
    unresolved: Tensor,
    mask: Tensor,
    step: int,
    outcome: str,
) -> None:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return
    event_step[indices] = step
    unresolved[indices] = False
    for index in indices.detach().cpu().tolist():
        outcomes[index] = outcome
