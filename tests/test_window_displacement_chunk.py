"""v2.8.0 D4/F1 — window_displacement episode-chunking must be BIT-IDENTICAL to the single-pass computation.

The 500 Hz eval path OOMs on the [T-W, W+1, B, D] unfold intermediate; window_displacement now processes the
batch in episode chunks. Every episode's window statistic is independent, so any chunk size must reproduce the
single-pass result exactly (torch.equal), including the NaN prefix rows."""
from __future__ import annotations
import torch

from src.common.outcomes import window_displacement


def _nan_equal(a, b):
    # torch.equal treats NaN != NaN; compare finite entries exactly and NaN masks identically.
    na, nb = torch.isnan(a), torch.isnan(b)
    return torch.equal(na, nb) and torch.equal(a[~na], b[~nb])


def test_chunk_equals_single_pass_various_chunks():
    torch.manual_seed(0)
    T, B, D = 40, 17, 3
    W = 12
    positions = torch.randn(T, B, D, dtype=torch.float64)
    full = window_displacement(positions, W, batch_chunk=B)          # single pass
    for chunk in (1, 2, 5, 8, 16, B, B + 5):
        got = window_displacement(positions, W, batch_chunk=chunk)
        assert _nan_equal(got, full), f"chunk={chunk} differs from single pass"


def test_adaptive_default_matches_single_pass():
    torch.manual_seed(1)
    T, B, D = 60, 23, 3
    W = 20
    positions = torch.randn(T, B, D, dtype=torch.float32)
    adaptive = window_displacement(positions, W)                      # default adaptive chunking
    single = window_displacement(positions, W, batch_chunk=10_000)    # forced single pass
    assert _nan_equal(adaptive, single)


def test_short_horizon_all_nan():
    positions = torch.randn(5, 4, 3)
    out = window_displacement(positions, 10)                          # n_steps <= window -> all NaN
    assert out.shape == (5, 4) and torch.isnan(out).all()
