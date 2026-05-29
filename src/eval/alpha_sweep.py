from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.common.filter_cbfqp import CBFQPFilter
from src.eval.evaluate import EVAL_EPISODE_COLUMNS, EVAL_METRIC_COLUMNS, evaluate
from src.eval.run_full import _append_csv, _full_pool_path, _load_oc_framework


ALPHA_UNSAFE_VALUES = (20.0, 50.0, 100.0, 200.0)
ALPHA_SAFE_VALUES = (2.0, 5.0, 10.0)


def run_alpha_sweep(
    run_dir: Path,
    *,
    ckpt_path: Path | None = None,
    max_scenes: int | None = None,
    append: bool = True,
) -> list[dict[str, Any]]:
    checkpoint_path = ckpt_path or run_dir / "checkpoints/best.pt"
    _, base_config, checkpoint = _load_oc_framework(checkpoint_path)
    pairs = _alpha_pairs(
        default_safe=float(base_config["filter"]["alpha_safe"]),
        default_unsafe=float(base_config["filter"]["alpha_unsafe"]),
    )

    eval_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for alpha_safe, alpha_unsafe in pairs:
        config = deepcopy(base_config)
        config["filter"]["alpha_safe"] = float(alpha_safe)
        config["filter"]["alpha_unsafe"] = float(alpha_unsafe)
        framework, _, _ = _load_oc_framework(checkpoint_path)
        framework.config = config
        framework._filter = CBFQPFilter(  # noqa: SLF001
            framework.system,
            framework._h_fn,  # noqa: SLF001
            config,
        )
        result = evaluate(
            framework,
            _full_pool_path(config),
            config,
            mode="final_alpha_sweep",
            step=int(checkpoint["step"]),
            ckpt_name=checkpoint_path.name,
            max_scenes=max_scenes,
            include_lqr_baseline=False,
        )
        eval_rows.append(result.eval_row)
        episode_rows.extend(result.episode_rows)

    if append:
        _append_csv(run_dir / "eval_metrics.csv", EVAL_METRIC_COLUMNS, eval_rows)
        _append_csv(run_dir / "eval_episodes.csv", EVAL_EPISODE_COLUMNS, episode_rows)
    return eval_rows


def _alpha_pairs(default_safe: float, default_unsafe: float) -> list[tuple[float, float]]:
    pairs = [(default_safe, value) for value in ALPHA_UNSAFE_VALUES]
    pairs.extend((value, default_unsafe) for value in ALPHA_SAFE_VALUES)
    unique: list[tuple[float, float]] = []
    seen = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval-time CBF-QP alpha sweep.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()

    rows = run_alpha_sweep(
        args.run_dir,
        ckpt_path=args.ckpt,
        max_scenes=args.max_scenes,
        append=not args.no_append,
    )
    for row in rows:
        print(
            "alpha_safe={alpha_safe} alpha_unsafe={alpha_unsafe} "
            "cps={cps} reach={reach} collision={collision} "
            "infeasibility={infeasibility} saturation_rate={saturation_rate}".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
