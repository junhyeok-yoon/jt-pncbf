from __future__ import annotations

import csv
from pathlib import Path

from src.frameworks.jt_pncbf.train import run_training


def test_jt_train_smoke_completes_and_writes_artifacts(tmp_path: Path) -> None:
    result = run_training(
        stage="smoke",
        output_root=tmp_path,
        seed=23,
        smoke_eval_scenes=1,
        device="cpu",
    )

    assert not result.halted
    assert result.last_value_loss == result.last_value_loss
    assert result.last_policy_loss == result.last_policy_loss
    assert result.last_vs_grad_norm > 0.0
    assert result.last_pi_grad_norm > 0.0
    assert result.max_policy_grad_leak < 1.0e-9
    assert result.last_sigma > 0.0

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
    assert metrics_rows
    assert eval_rows
    row = metrics_rows[-1]
    assert "rho_unsafe_v" in row
    assert "target_rhs_active" in row
    assert "target_rhs" not in row
    assert "grad_leak_VS_from_Lpi" in row
    assert float(row["grad_leak_VS_from_Lpi"]) < 1.0e-9
    assert float(row["grad_norm_pi"]) > 0.0
    assert sum(row["mode"] == "final" for row in eval_rows) == 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
