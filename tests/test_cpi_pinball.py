"""CPI pinball: the constant minimizing the empirical pinball loss at level tau equals the sample
tau-quantile (hand case + a larger sample vs numpy quantile)."""
from __future__ import annotations

import numpy as np
import torch

from src.frameworks.cpi.calib import pinball_loss


def _grid_min_constant(y, tau, grid):
    yt = torch.as_tensor(y, dtype=torch.float64)
    losses = [float(pinball_loss(yt, torch.full_like(yt, float(c)), tau)) for c in grid]
    return float(grid[int(np.argmin(losses))])


def test_pinball_median_hand_case():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    grid = np.linspace(0.0, 6.0, 601)
    c = _grid_min_constant(y, 0.5, grid)
    assert abs(c - 3.0) <= 0.02                       # median minimizes tau=0.5 pinball


def test_pinball_matches_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=20000)
    for tau in (0.5, 0.9):
        grid = np.linspace(-4.0, 4.0, 1601)
        c = _grid_min_constant(y, tau, grid)
        q = float(np.quantile(y, tau, method="linear"))
        assert abs(c - q) <= 0.05, f"tau={tau}: grid-min {c:.3f} vs quantile {q:.3f}"
