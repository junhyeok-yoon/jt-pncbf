"""v2.8.1 S1 — terminal-confound re-scoring (report-only, NON-GATING).

The v2.8.1 terminal (goal_angrate_radius 0.48 -> 0.30) and the encoder (hard_topk -> soft_topk beta=30) changed
TOGETHER across the lineage break, so a raw hard-vs-soft delta confounds the two. This isolates the terminal:
  - 043415 step-50000 (v2.7.6 hard OC value; config has NO obs block -> hard_topk) re-scored under the NEW 0.30
  - beta=30 OC value cell (soft) re-scored under the OLD 0.48
On the canonical pool (band-terminate cell), shipped fallback {kstep,phases 1,k 3}, dual_solve, n2000. Reports
collision / reach / cps / mean-episode-length only. Does NOT touch the JT deliverable's scored terminal."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL_D2R = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--angrate", required=True, type=float)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
filt["projection"] = filt.get("projection") or "dual_solve"
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": a.angrate, "band_terminates": True},
        "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, ck2 = load_fw(a.ckpt, config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None
pool = load_pool(POOL_D2R)
res = evaluate(fw, pool, cfg, mode="final", step=int(ck2["step"]), ckpt_name=a.label,
               max_scenes=None, include_lqr_baseline=False)
r = res.eval_row; eps = res.episode_rows
# mean episode length: find the per-episode step-count field
LENKEYS = ("n_steps", "steps", "length", "episode_length", "T", "horizon", "n_control_steps", "ctrl_steps")
lk = next((k for k in LENKEYS if eps and k in eps[0]), None)
mean_len = float(np.mean([e[lk] for e in eps])) if lk else None
enc = getattr(fw.system, "encoder", "?"); beta = float(getattr(fw.system, "soft_beta", 0.0))
rep = {"label": a.label, "ckpt": str(a.ckpt), "goal_angrate_radius": a.angrate, "encoder": enc, "beta": beta,
       "cps": float(r["cps"]), "reach": float(r["reach"]), "collision": float(r["collision"]),
       "oob": float(r.get("oob") or 0.0), "stuck": float(r.get("stuck") or 0.0),
       "timeout": float(r.get("timeout") or 0.0), "mean_episode_length": mean_len, "len_key": lk,
       "episode_row_keys": sorted(eps[0].keys()) if eps else []}
(OUT / f"confound_{a.label}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"[{a.label}] enc={enc} beta={beta} angrate={a.angrate} | cps {rep['cps']:.4f} reach {rep['reach']:.4f} "
      f"coll {rep['collision']:.4f} | mean_ep_len {mean_len} (key={lk})", flush=True)
