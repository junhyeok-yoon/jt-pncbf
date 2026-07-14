"""v2.5.1 A2(a) policy warm-start (training.jt.pi_init_ckpt): loads ONLY pi_state; the optimizer, the LR/
sigma schedules, the step counter, the value net, and the certificate channel start fresh. Contract checks
via short smoke runs: (1) the run record stamps pi_init_ckpt + sha256 and is NOT a full resume (start_step=0,
metrics begin at step 1); (2) warm-start pulls the initial policy toward the donor even under a different
seed (fresh-init would be far); (3) warm-start and full-resume are mutually exclusive."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from src.frameworks.jt_pncbf.train import run_training


def _pi_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)["pi_state"]


def _pi_dist(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return float(sum((a[k] - b[k]).pow(2).sum() for k in a).sqrt())


def _ckpt_step(ckpt_path: Path) -> int:
    return int(torch.load(ckpt_path, map_location="cpu", weights_only=False)["step"])


def test_pi_warmstart_loads_pi_only_and_starts_fresh(tmp_path: Path) -> None:
    donor = run_training(stage="smoke", output_root=tmp_path, seed=7, smoke_eval_scenes=1, device="cpu")
    assert not donor.halted
    donor_ckpt = donor.run_dir / "checkpoints/final.pt"
    donor_pi = _pi_state(donor_ckpt)

    # warm-start from the donor under a DIFFERENT seed; and a fresh run at the same seed for contrast.
    warm = run_training(stage="smoke", output_root=tmp_path, seed=999, smoke_eval_scenes=1, device="cpu",
                        pi_init_ckpt=donor_ckpt)
    fresh = run_training(stage="smoke", output_root=tmp_path, seed=999, smoke_eval_scenes=1, device="cpu")
    assert not warm.halted and not fresh.halted

    # (1) provenance recorded; NOT a full resume.
    rec = yaml.safe_load((warm.run_dir / "config.yaml").read_text())["run"]
    assert rec.get("pi_init_ckpt") == str(donor_ckpt)
    assert rec.get("pi_init_ckpt_sha256") == hashlib.sha256(donor_ckpt.read_bytes()).hexdigest()
    assert "resume_from_step" not in rec and "resume_ckpt" not in rec
    # fresh step counter (start_step=0): warm-start ran a full fresh schedule, same final step as the donor
    # (a full resume from the donor would have started at the donor's step, not 0).
    assert _ckpt_step(warm.run_dir / "checkpoints/final.pt") == _ckpt_step(donor_ckpt)

    # (2) warm-start pulled the policy toward the donor despite the seed change; fresh-init is far.
    warm_pi = _pi_state(warm.run_dir / "checkpoints/final.pt")
    fresh_pi = _pi_state(fresh.run_dir / "checkpoints/final.pt")
    assert _pi_dist(warm_pi, donor_pi) < _pi_dist(fresh_pi, donor_pi)


def test_pi_warmstart_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    donor = run_training(stage="smoke", output_root=tmp_path, seed=3, smoke_eval_scenes=1, device="cpu")
    ckpt = donor.run_dir / "checkpoints/final.pt"
    with pytest.raises(ValueError):
        run_training(stage="smoke", output_root=tmp_path, seed=3, smoke_eval_scenes=1, device="cpu",
                     resume_ckpt=ckpt, pi_init_ckpt=ckpt)
