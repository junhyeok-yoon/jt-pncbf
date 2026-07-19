from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class OutcomeResult:
    outcome: list[str]
    event_step: Tensor
    min_window_displacement: Tensor


def step_outcomes(
    states: Tensor,
    scene: Any,
    system: System,
    config: Mapping[str, Any],
) -> StepOutcomeMasks:
    positions = system.position(states)
    collided = _collided_exact(positions, scene)

    goal = torch.as_tensor(scene.goal, dtype=states.dtype, device=states.device)
    goal_distance = torch.linalg.norm(positions - goal, dim=-1)
    speed = system.speed(states)
    goal_reached = (goal_distance <= float(config["env"]["goal_radius"])) & (
        speed <= float(config["env"]["goal_speed_radius"])
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


def window_displacement(positions: Tensor, window_steps: int) -> Tensor:
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError(
            "positions must have shape [T, B, D] with D >= 2, "
            f"got {tuple(positions.shape)}."
        )
    if window_steps < 1:
        raise ValueError(f"window_steps must be positive, got {window_steps}.")

    n_steps, batch_size, _ = positions.shape
    result = positions.new_full((n_steps, batch_size), float("nan"))
    if n_steps <= window_steps:
        return result

    windows = positions.unfold(0, window_steps + 1, 1).permute(0, 3, 1, 2)
    anchor = windows[:, :1, :, :]
    displacement = torch.linalg.norm(windows - anchor, dim=-1).amax(dim=1)
    result[window_steps:] = displacement
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
    )


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
