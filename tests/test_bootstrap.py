from __future__ import annotations

import numpy as np

from src.eval.bootstrap import (
    across_seed_aggregate,
    compare,
    within_seed_ci,
)


def test_within_seed_ci_brackets_known_mean_and_is_deterministic() -> None:
    records = (
        [{"outcome": "goal", "infeasibility": 0.1}] * 50
        + [{"outcome": "collision", "infeasibility": 0.1}] * 25
        + [{"outcome": "timeout", "infeasibility": 0.1}] * 25
    )
    expected_cps = 0.5 - 2.0 * 0.25 - 0.5 * 0.25 - 0.3 * 0.1

    result_a = within_seed_ci(records, n_resample=1000, seed=123)
    result_b = within_seed_ci(records, n_resample=1000, seed=123)

    assert result_a["ci"]["cps"]["lo"] <= expected_cps <= result_a["ci"]["cps"]["hi"]
    assert abs(result_a["mean"]["cps"] - expected_cps) < 1.0e-12
    assert result_a["mean"] == result_b["mean"]
    for metric, values in result_a["resamples"].items():
        assert np.array_equal(values, result_b["resamples"][metric])


def test_across_seed_aggregate_and_compare_verdicts() -> None:
    weak_records = [{"outcome": "timeout"} for _ in range(80)]
    strong_records = [{"outcome": "goal"} for _ in range(80)]
    identical_records = [{"outcome": "goal"} for _ in range(80)]

    weak = across_seed_aggregate(
        [
            within_seed_ci(weak_records, n_resample=300, seed=1),
            within_seed_ci(weak_records, n_resample=300, seed=2),
            within_seed_ci(weak_records, n_resample=300, seed=3),
        ]
    )
    strong = across_seed_aggregate(
        [
            within_seed_ci(strong_records, n_resample=300, seed=4),
            within_seed_ci(strong_records, n_resample=300, seed=5),
            within_seed_ci(strong_records, n_resample=300, seed=6),
        ]
    )
    identical = across_seed_aggregate(
        [
            within_seed_ci(identical_records, n_resample=300, seed=7),
            within_seed_ci(identical_records, n_resample=300, seed=8),
            within_seed_ci(identical_records, n_resample=300, seed=9),
        ]
    )

    assert strong["mean"]["cps"] == 1.0
    assert weak["mean"]["cps"] == -0.5
    assert compare(weak, strong) == "improved"
    assert compare(strong, weak) == "regressed"
    assert compare(strong, identical) == "no improvement detected"
