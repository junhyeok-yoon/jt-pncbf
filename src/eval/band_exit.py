"""v2.7.6 Stage-1 M4 — band_exit diagnostic.

An episode "band-exits" iff its trajectory leaves the arena band z in [floor, ceiling] at any step within
the physical horizon (n_steps). This is a RECORDED DIAGNOSTIC ONLY: it is not a cps term and does not touch
the oob predicate (|z| > 8) or any cps component. Additive module; nothing in the core eval path imports it.
"""
from __future__ import annotations

from typing import Any

BAND_FLOOR = -4.0
BAND_CEILING = 4.0


def episode_band_exit(states: Any, n_steps: int, system: Any,
                      floor: float = BAND_FLOOR, ceiling: float = BAND_CEILING) -> float:
    """states: [T+1, 1, state_dim] rollout states for one episode. Returns 1.0 if the vertical position z
    leaves [floor, ceiling] within the physical horizon, else 0.0."""
    z = system.position(states)[..., 2].reshape(-1)
    k = min(int(n_steps), int(z.shape[0]) - 1)
    zt = z[: k + 1]
    return 1.0 if (float(zt.min()) < floor or float(zt.max()) > ceiling) else 0.0
