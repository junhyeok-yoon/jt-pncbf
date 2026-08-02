"""v2.7.7 M22 helper (Amdt 11) — deployed-arm per-episode dump via the real eval harness. Re-runs evaluate() on
the after checkpoint (09c33bf4) with the DEPLOYED configuration (banded scoring band_collision_limit=4.0, kstep
empty-branch fallback k=5) on the canonical pool, and dumps per-episode: outcome + reach/collision/oob/stuck/
timeout + per-episode infeasibility (from episode_rows) and min/max p_z + IC state (from the filtered trajectory).
Eval-only; no src or scoring-code edits — uses evaluate() and the recorded filter config exactly as
scripts/analysis/v276_canonical_eval.py did. Saves a json (episode rows + aggregate) + npz (min_pz/max_pz/IC)."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
import torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
JT42 = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(JT42, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "k": 5}                     # DEPLOYED fallback (matches recorded row)
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},  # banded
        "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, _ = load_fw(JT42, config_overrides=over)
for net in ("value_net", "policy_net"):
    mnet = getattr(fw, net, None)
    if mnet is not None:
        mnet.to(DEV)
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=JT42.name,
               max_scenes=None, include_lqr_baseline=False)
rows = list(res.episode_rows)
n = len(rows)
# per-episode min/max p_z + IC from the filtered trajectory
min_pz = np.full(n, np.inf); max_pz = np.full(n, -np.inf); IC = np.zeros((n, 13), np.float32)
for i, tr in enumerate(res.trajectories):
    S = tr.filtered.states.detach().cpu().numpy()                     # [T+1, statedim] (single episode)
    S = S.reshape(S.shape[0], -1)
    min_pz[i] = float(S[:, 2].min()); max_pz[i] = float(S[:, 2].max()); IC[i] = S[0, :13]
np.savez_compressed(SCR / "deployed_dump.npz", min_pz=min_pz, max_pz=max_pz, IC=IC)
(SCR / "deployed_rows.json").write_text(json.dumps({"eval_row": res.eval_row, "episode_rows": rows}, indent=2, default=float) + "\n")
# quick aggregate cross-check
o = np.array([r["outcome"] for r in rows])
inf = np.array([float(r.get("infeasible_step_frac", r.get("infeasibility", 0.0))) for r in rows])
reach = (o == "goal").mean(); coll = (o == "collision").mean(); stuck = (o == "stuck").mean()
oob = (o == "oob").mean(); to = (o == "timeout").mean()
cps = reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * inf.mean()
print(f"deployed eval done: n={n}")
print(f"  aggregate: cps {cps:.4f} reach {reach:.4f} coll {coll:.4f} oob {oob:.4f} stuck {stuck:.4f} to {to:.4f} infeas {inf.mean():.4f}")
print(f"  recorded : cps 0.8051 reach 0.9375 coll 0.0425 oob 0.0000 stuck 0.0000 to 0.0200 infeas 0.1247")
print(f"  eval_row cps {res.eval_row.get('cps')}, reach {res.eval_row.get('reach')}, collision {res.eval_row.get('collision')}, infeasibility {res.eval_row.get('infeasibility')}")
print(f"  episode_row keys: {sorted(rows[0].keys())}")
