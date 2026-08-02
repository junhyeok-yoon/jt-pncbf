from __future__ import annotations

from typing import Any, Protocol

import torch


Tensor = torch.Tensor


class System(Protocol):
    state_dim: int
    action_dim: int
    obs_dim: int
    name: str
    u_bounds: Tensor

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """Return batched continuous-time state derivatives."""

    def observation(self, x: Tensor, scene: Any) -> Tensor:
        """Return batched observations for states in one scene."""

    def position(self, x: Tensor) -> Tensor:
        """Return batched planar positions."""

    def speed(self, x: Tensor) -> Tensor:
        """Return batched scalar speeds for goal/outcome predicates."""

    def angular_rate(self, x: Tensor) -> Tensor:
        """Return batched scalar angular-rate magnitude for the goal/outcome predicate;
        a structural zero on systems with no angular-rate state (the condition is vacuous there)."""

    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        """Return batched clamped nominal actions toward goal."""

    def wrap_state(self, x: Tensor) -> Tensor:
        """Return state after applying representation normalization."""
