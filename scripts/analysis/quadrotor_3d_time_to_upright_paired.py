"""v2.7.4 M6 amendment (paired) — pool-wide time-to-upright per initial-tilt stratum, PRE-JT vs JT side by
side (measurement only). Rolls both arms through the canonical rollout_eval path, computes per-episode
time-to-upright = first t with tilt(t)=arccos(R[2,2]) < 30 deg, and reports the distribution (p50/p90) per
stratum [0,60)/[60,120)/[120,180] for each arm with the NON-RECOVERING fraction stated separately. Writes a
JSON with both arms and their denominators. No verdict language."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.build_pools import load_pool, EVAL_POOLS_DIR, DEFAULT_OUTPUT_DIR
from src.eval.evaluate import _filter_adapter, _tensor_options, make_batched_scene
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R
from src.eval.run_full import _load_framework

ap = argparse.ArgumentParser()
ap.add_argument("--jt-ckpt", required=True)
ap.add_argument("--pre-ckpt", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--pool", default="eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
a = ap.parse_args()

pp = EVAL_POOLS_DIR / a.pool if (EVAL_POOLS_DIR / a.pool).exists() else DEFAULT_OUTPUT_DIR / a.pool
scenes = load_pool(pp).scenes


def roll_ttu(ckpt):
    fw, cfg, _ = _load_framework(Path(ckpt))
    system = fw.system
    dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    dtype, device = _tensor_options(system, fw)
    bscene = make_batched_scene(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(bscene)
    result = rollout_eval(system, fw.policy, _filter_adapter(fw), bscene, x0, max_steps=max_steps, dt=dt, config=cfg)
    S = result.states.detach()                                  # [T+1, N, 13]
    Tn, N = S.shape[0], S.shape[1]
    q = S[:, :, 3:7].reshape(-1, 4).to(torch.float64)
    tilt = torch.rad2deg(torch.arccos(torch.clamp(_quat_to_R(q)[:, 2, 2], -1.0, 1.0))).reshape(Tn, N).numpy()
    below = tilt < 30.0
    ttu = np.full(N, np.nan)
    for i in range(N):
        idx = np.where(below[:, i])[0]
        if idx.size:
            ttu[i] = idx[0] * dt
    return tilt[0], ttu, N


init_tilt_jt, ttu_jt, N = roll_ttu(a.jt_ckpt)
init_tilt_pre, ttu_pre, Np = roll_ttu(a.pre_ckpt)
assert N == Np and np.allclose(init_tilt_jt, init_tilt_pre, atol=1e-6), "arms must share ICs"
init_tilt = init_tilt_jt


def strat(ttu, mask):
    rec = ttu[mask]; got = rec[~np.isnan(rec)]
    return dict(
        n=int(mask.sum()), recovered_n=int(got.size),
        time_to_upright_p50_s=(round(float(np.percentile(got, 50)), 3) if got.size else None),
        time_to_upright_p90_s=(round(float(np.percentile(got, 90)), 3) if got.size else None),
        non_recovering_fraction=(round(float(np.isnan(rec).mean()), 4) if mask.sum() else None))


out = {"n": int(N), "tilt_below_deg": 30.0, "arms": ["pre_JT", "JT"], "strata": []}
for lo, hi in [(0, 60), (60, 120), (120, 180.001)]:
    m = (init_tilt >= lo) & (init_tilt < hi)
    out["strata"].append({"stratum_deg": f"[{int(lo)},{int(min(hi,180))})",
                          "pre_JT": strat(ttu_pre, m), "JT": strat(ttu_jt, m)})
outp = Path(a.run_dir) / "figures" / "flip_recovery" / "time_to_upright_paired.json"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
