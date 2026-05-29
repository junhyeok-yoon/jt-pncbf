from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np


DEFAULT_N_RESAMPLE = 1000
DEFAULT_SEED = 20260508
METRICS = (
    "cps",
    "reach",
    "collision",
    "oob",
    "stuck",
    "timeout",
    "infeasibility",
)


def within_seed_ci(
    episode_records: Sequence[Mapping[str, Any]],
    n_resample: int = DEFAULT_N_RESAMPLE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if len(episode_records) == 0:
        raise ValueError("episode_records must be non-empty.")
    if n_resample <= 0:
        raise ValueError("n_resample must be positive.")

    components = _episode_components(episode_records)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(
        0,
        len(episode_records),
        size=(n_resample, len(episode_records)),
    )

    resamples: dict[str, np.ndarray] = {}
    for metric in METRICS:
        values = components[metric]
        resamples[metric] = np.mean(values[sample_idx], axis=1)

    mean = {metric: float(np.mean(components[metric])) for metric in METRICS}
    ci = {
        metric: {
            "lo": float(np.percentile(values, 2.5, method="linear")),
            "hi": float(np.percentile(values, 97.5, method="linear")),
        }
        for metric, values in resamples.items()
    }
    return {
        "n_episodes": len(episode_records),
        "mean": mean,
        "ci": ci,
        "resamples": resamples,
    }


def across_seed_aggregate(per_seed_records: Sequence[Any]) -> dict[str, Any]:
    if len(per_seed_records) == 0:
        raise ValueError("per_seed_records must be non-empty.")

    per_seed = [_as_within_seed_result(record) for record in per_seed_records]
    seed_means = {
        metric: np.asarray(
            [seed_result["mean"][metric] for seed_result in per_seed],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    pooled_resamples = {
        metric: np.concatenate(
            [
                np.asarray(seed_result["resamples"][metric], dtype=np.float64)
                for seed_result in per_seed
            ],
            axis=0,
        )
        for metric in METRICS
    }

    mean = {metric: float(np.mean(values)) for metric, values in seed_means.items()}
    sd = {
        metric: float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0
        for metric, values in seed_means.items()
    }
    pooled_ci = {
        metric: {
            "lo": float(np.percentile(values, 2.5, method="linear")),
            "hi": float(np.percentile(values, 97.5, method="linear")),
        }
        for metric, values in pooled_resamples.items()
    }
    return {
        "n_seeds": len(per_seed),
        "mean": mean,
        "sd": sd,
        "pooled_ci": pooled_ci,
        "seed_means": seed_means,
        "pooled_resamples": pooled_resamples,
    }


def compare(version_A: Any, version_B: Any) -> str:
    aggregate_a = _as_aggregate(version_A)
    aggregate_b = _as_aggregate(version_B)

    mean_a = aggregate_a["mean"]["cps"]
    mean_b = aggregate_b["mean"]["cps"]
    ci_a = aggregate_a["pooled_ci"]["cps"]
    ci_b = aggregate_b["pooled_ci"]["cps"]

    if mean_b > mean_a and ci_b["lo"] > ci_a["hi"]:
        return "improved"
    if mean_b < mean_a and ci_b["hi"] < ci_a["lo"]:
        return "regressed"
    return "no improvement detected"


def _as_aggregate(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "pooled_ci" in value:
        return dict(value)
    if isinstance(value, Sequence) and value and _looks_like_within_seed(value[0]):
        return across_seed_aggregate(value)
    if isinstance(value, Sequence):
        return across_seed_aggregate([within_seed_ci(value)])
    raise TypeError("Expected an aggregate, within-seed results, or episode records.")


def _as_within_seed_result(value: Any) -> dict[str, Any]:
    if _looks_like_within_seed(value):
        return dict(value)
    if isinstance(value, Sequence):
        return within_seed_ci(value)
    raise TypeError("Expected a within-seed result or episode records.")


def _looks_like_within_seed(value: Any) -> bool:
    return isinstance(value, Mapping) and "mean" in value and "resamples" in value


def _episode_components(
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    reach = np.asarray(
        [_outcome_component(record, "goal") for record in episode_records],
        dtype=np.float64,
    )
    collision = np.asarray(
        [_outcome_component(record, "collision") for record in episode_records],
        dtype=np.float64,
    )
    oob = np.asarray(
        [_outcome_component(record, "oob") for record in episode_records],
        dtype=np.float64,
    )
    timeout = np.asarray(
        [_outcome_component(record, "timeout") for record in episode_records],
        dtype=np.float64,
    )
    stuck = np.asarray(
        [_outcome_component(record, "stuck") for record in episode_records],
        dtype=np.float64,
    )
    infeasibility = np.asarray(
        [_infeasibility(record) for record in episode_records],
        dtype=np.float64,
    )
    cps = (
        reach
        - 2.0 * collision
        - stuck
        - 0.5 * (oob + timeout)
        - 0.3 * infeasibility
    )
    return {
        "cps": cps,
        "reach": reach,
        "collision": collision,
        "oob": oob,
        "stuck": stuck,
        "timeout": timeout,
        "infeasibility": infeasibility,
    }


def _outcome_component(record: Mapping[str, Any], outcome: str) -> float:
    key = "reach" if outcome == "goal" else outcome
    if key in record:
        return float(record[key])
    return 1.0 if str(record.get("outcome")) == outcome else 0.0


def _infeasibility(record: Mapping[str, Any]) -> float:
    for key in ("infeasibility", "infeasible_step_frac", "infeasible_fraction"):
        if key in record:
            return float(record[key])
    return 0.0
