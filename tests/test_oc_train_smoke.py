from __future__ import annotations

import csv
from pathlib import Path

from src.frameworks.oc_pncbf.train import run_training


def test_oc_train_smoke_completes_and_writes_run_dir(tmp_path: Path) -> None:
    result = run_training(
        stage="smoke",
        output_root=tmp_path,
        seed=17,
        smoke_eval_scenes=1,
        system="double_integrator",   # v2.6.0: exp_config default is quadrotor_planar; pin the DI training path
    )

    assert not result.halted
    assert result.smoke_grad_norm is not None
    assert result.smoke_grad_norm > 0.0

    required_paths = [
        "config.yaml",
        "git_commit.txt",
        "tensorboard",
        "metrics.csv",
        "eval_metrics.csv",
        "eval_episodes.csv",
        "status.json",
        "pool_manifest.json",
        "checkpoints/best.pt",
        "checkpoints/final.pt",
        "report.md",
    ]
    for relative in required_paths:
        assert (result.run_dir / relative).exists(), relative

    metrics_rows = _read_csv(result.run_dir / "metrics.csv")
    eval_rows = _read_csv(result.run_dir / "eval_metrics.csv")
    episode_rows = _read_csv(result.run_dir / "eval_episodes.csv")
    assert metrics_rows
    assert eval_rows
    assert episode_rows
    assert [int(row["step"]) for row in metrics_rows] == [10]
    assert "L_R" in metrics_rows[0]
    assert "cps" in eval_rows[0]
    assert "outcome" in episode_rows[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
