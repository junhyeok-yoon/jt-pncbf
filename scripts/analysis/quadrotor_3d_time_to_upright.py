"""v2.7.4 M6 — pool-wide time-to-upright per initial-tilt stratum (measurement only).

Rolls the full d2r pool through the canonical rollout_eval path (bit-identical to the M6 eval), computes
per-episode time-to-upright = first t with tilt(t)=arccos(R(q_t)[2,2]) < 30 deg, and reports the distribution
(p50/p90) per initial-tilt stratum [0,60)/[60,120)/[120,180] with the NON-RECOVERING fraction (never <30 deg)
stated separately. Writes a JSON under the run dir. No verdict language.
"""
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
ap.add_argument("--ckpt", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--pool", default="eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl")
a = ap.parse_args()

pp = EVAL_POOLS_DIR / a.pool if (EVAL_POOLS_DIR / a.pool).exists() else DEFAULT_OUTPUT_DIR / a.pool
pool = load_pool(pp); scenes = pool.scenes
fw, cfg, _ = _load_framework(Path(a.ckpt))
system = fw.system
dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])

dtype, device = _tensor_options(system, fw)
bscene = make_batched_scene(scenes, device=device, dtype=dtype)
x0 = initial_states_from_batch(bscene)
result = rollout_eval(system, fw.policy, _filter_adapter(fw), bscene, x0, max_steps=max_steps, dt=dt, config=cfg)
S = result.states.detach()                                      # [T+1, N, 13]
Tn, N = S.shape[0], S.shape[1]
q = S[:, :, 3:7].reshape(-1, 4).to(torch.float64)
R = _quat_to_R(q)
tilt = torch.rad2deg(torch.arccos(torch.clamp(R[:, 2, 2], -1.0, 1.0))).reshape(Tn, N).numpy()   # [T+1, N]
init_tilt = tilt[0]                                              # initial tilt per episode
below = tilt < 30.0
ttu = np.full(N, np.nan)
for i in range(N):
    idx = np.where(below[:, i])[0]
    if idx.size:
        ttu[i] = idx[0] * dt

out = {"n": int(N), "tilt_below_deg": 30.0, "strata": []}
for lo, hi in [(0, 60), (60, 120), (120, 180.001)]:
    m = (init_tilt >= lo) & (init_tilt < hi)
    k = int(m.sum())
    rec = ttu[m]
    recovered = rec[~np.isnan(rec)]
    non_recover_frac = float(np.isnan(rec).mean()) if k else None
    out["strata"].append({
        "stratum_deg": f"[{int(lo)},{int(min(hi, 180))})", "n": k,
        "recovered_n": int(recovered.size),
        "time_to_upright_p50_s": (round(float(np.percentile(recovered, 50)), 3) if recovered.size else None),
        "time_to_upright_p90_s": (round(float(np.percentile(recovered, 90)), 3) if recovered.size else None),
        "non_recovering_fraction": (round(non_recover_frac, 4) if non_recover_frac is not None else None),
    })
outp = Path(a.run_dir) / "figures" / "flip_recovery" / "time_to_upright_by_stratum.json"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
