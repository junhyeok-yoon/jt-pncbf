"""v2.7.6 Stage-2 M0(b) — report h_scale, value clip floor, and max |h_star| on a v2.7.4 rollout batch.
Read-only. Rolls the deployed v2.7.4 policy+filter on the full-range pool and evaluates value_target_barrier
(the OBSTACLE-only h_star used to label V_hat) over all rollout states."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.common.quadrotor_barrier import value_target_barrier
from src.envs.scene_batch import batch_scenes
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt"
POOL = REPO / "data/runs/v2.7.6/pools/eval_fullrange_quadrotor-3d-d2r_n2000_seed42.pkl"
OUT = REPO / "data/runs/v2.7.6/stage2_m0"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    fw, cfg, ck = load_framework_from_checkpoint(CKPT, config_overrides=None)
    sys = fw.system
    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name="best.pt",
                   max_scenes=512, include_lqr_baseline=False)
    hvals, zvals = [], []
    with torch.no_grad():
        for tr in res.trajectories:
            st = tr.filtered.states                                   # [T+1, 1, D]
            bs = batch_scenes([tr.scene], device=st.device, dtype=st.dtype)
            h = value_target_barrier(sys, st, bs, cfg).numpy().reshape(-1)
            hvals.append(h); zvals.append(sys.position(st)[..., 2].numpy().reshape(-1))
    h = np.concatenate(hvals); zpos = np.concatenate(zvals)
    out = {"h_scale": float(cfg["env"]["h_scale"]),
           "value_clip": {"floor": -1.0, "ceiling": 1.0, "source": "src/common/value_net.py:33 clamp(-1,1)"},
           "hstar_note": "obstacle-only h_star (value_target_barrier) over 512 episodes x 201 steps",
           "hstar": {"max": round(float(h.max()), 4), "min": round(float(h.min()), 4),
                     "p99": round(float(np.percentile(h, 99)), 4), "p999": round(float(np.percentile(h, 99.9)), 4),
                     "frac_gt_1": round(float((h > 1.0).mean()), 6), "frac_lt_-1": round(float((h < -1.0).mean()), 6),
                     "abs_max": round(float(np.abs(h).max()), 4)},
           "z_range_observed": {"min": round(float(zpos.min()), 3), "max": round(float(zpos.max()), 3)},
           "psi_cap_r_max_from_config": float(cfg["obstacle"]["per_system"]["quadrotor_3d"]["r_max"]),
           "c_z_pi_over_omega_max": round(np.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"]), 4)}
    (OUT / "m0_hstar_scale.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
