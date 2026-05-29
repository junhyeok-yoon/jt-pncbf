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

    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor:
        """Return batched clamped nominal actions toward goal."""

    def wrap_state(self, x: Tensor) -> Tensor:
        """Return state after applying representation normalization."""
