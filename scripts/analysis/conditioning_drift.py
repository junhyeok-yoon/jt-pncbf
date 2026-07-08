"""v2.4.2 decline forensics D — conditioning two-timescale drift (read-only, throwaway).

Across cadence checkpoints, L2 drift of pi_theta actions and L1 drift of deployed h-hat on a FIXED
probe-state set (initial states of the first n in-loop pool scenes). Also reports the configured
pi_b effective half-life from tau_b (and K_pi=1). Reading: pi drift settling while V-error rises =>
moving-target dominance (H2); both settling => benign (H1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.build_pools import load_pool
from src.eval.evaluate import _tensor_options
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"


def _probe(fw, scenes):
    system = fw.system
    dtype, device = _tensor_options(system, fw)
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x = initial_states_from_batch(batched)
    with torch.no_grad():
        u = fw.policy(x, batched).detach().cpu().numpy()
        h = fw._filter.h_fn(x, batched).reshape(-1).detach().cpu().numpy()
    return u, h


def main() -> int:
    ap = argparse.ArgumentParser(description="conditioning two-timescale drift (read-only).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--steps", type=str,
                    default="15000,18000,21000,24000,27000,30000,33000,36000,39000,42000")
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()

    steps = [int(s) for s in a.steps.split(",")]
    scenes = load_pool(a.pool).scenes[: a.n]

    prev_u = prev_h = None
    rows = []
    tau_b = None
    for st in steps:
        ckpt = a.run_dir / f"checkpoints/step_{st:06d}.pt"
        if not ckpt.exists():
            continue
        fw, cfg, _ = _load_framework(ckpt)
        if tau_b is None:
            tau_b = float(cfg["value_target"]["raw_lagged"]["tau_b"])
        u, h = _probe(fw, scenes)
        if prev_u is not None:
            du = float(np.sqrt(((u - prev_u) ** 2).sum(axis=1)).mean())
            dh = float(np.abs(h - prev_h).mean())
            rows.append({"step": st, "pi_drift_L2_per_cadence": du, "hhat_drift_L1_per_cadence": dh})
        prev_u, prev_h = u, h

    half_life = float(np.log(2.0) / tau_b) if tau_b else None
    out = {
        "run_dir": str(a.run_dir),
        "pi_b_tau_b": tau_b,
        "pi_b_halflife_policy_steps": half_life,
        "K_pi": 1,
        "pi_b_halflife_macro_steps": half_life,
        "cadence_macro_steps": 1500,
        "rows": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "conditioning_drift.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
