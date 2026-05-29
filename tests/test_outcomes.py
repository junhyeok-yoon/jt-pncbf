from __future__ import annotations

import torch

from src.common.outcomes import StepOutcomeMasks, resolve_outcome


def test_collision_wins_over_simultaneous_stuck() -> None:
    collided = torch.zeros((61, 1), dtype=torch.bool)
    goal = torch.zeros_like(collided)
    oob = torch.zeros_like(collided)
    stuck = torch.zeros_like(collided)
    window_displacement = torch.full((61, 1), float("nan"))

    collided[60, 0] = True
    stuck[60, 0] = True
    window_displacement[60, 0] = 0.0

    resolved = resolve_outcome(
        StepOutcomeMasks(
            collided=collided,
            goal_reached=goal,
            oob=oob,
            stuck=stuck,
            window_displacement=window_displacement,
        )
    )

    assert resolved.outcome == ["collision"]
    assert int(resolved.event_step[0]) == 60
    assert float(resolved.min_window_displacement[0]) == 0.0
