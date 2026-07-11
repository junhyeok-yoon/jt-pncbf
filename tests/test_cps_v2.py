"""v2.5.0 cps-v2 metric amendment: the per-step infeasible_v2 flag reclassifies benign far-field-singular
steps (CBF row already satisfied) as FEASIBLE. Three-way discrimination unit test on synthetic rows."""
from __future__ import annotations

import torch

from scripts.analysis.deploy_rate_eval import infeasible_v2


def test_infeasible_v2_three_way_discrimination() -> None:
    # rows: 0 far-field flat + row SATISFIED (feasible); 1 flat + row VIOLATED (infeasible);
    #       2 empty-intersection (infeasible regardless of row/singular).
    singular = torch.tensor([True, True, False])
    empty = torch.tensor([False, False, True])
    row = torch.tensor([+0.5, -0.3, -0.9])          # row_upper = -L_f V - alpha*V (>=0 => satisfied)
    got = infeasible_v2(singular, empty, row)
    assert got.tolist() == [False, True, True]


def test_infeasible_v2_row_boundary_and_nonsingular() -> None:
    # row exactly 0 counts as SATISFIED (>=0); a NON-singular non-empty step is feasible even if row<0
    # (there L_g V != 0 so the base/box projection handles it — 'singular' is the gate for the row test).
    singular = torch.tensor([True, False, True])
    empty = torch.tensor([False, False, False])
    row = torch.tensor([0.0, -1.0, -1e-9])
    assert infeasible_v2(singular, empty, row).tolist() == [False, False, True]


def test_infeasible_v2_reduces_to_empty_when_no_singular() -> None:
    # with no singular steps (learned-filter regime), infeasible_v2 == empty exactly (== legacy for filters
    # whose V is not saturated-flat), so cps_v2 == legacy there.
    n = 16
    torch.manual_seed(0)
    empty = torch.rand(n) < 0.3
    row = torch.randn(n)
    singular = torch.zeros(n, dtype=torch.bool)
    assert torch.equal(infeasible_v2(singular, empty, row), empty)
