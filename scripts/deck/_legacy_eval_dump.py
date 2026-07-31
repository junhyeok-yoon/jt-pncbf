"""v2.7.7 M27 helper (Amdt 15) — legacy (band NON-terminal) per-episode dump with tilt-at-crossing analysis. Runs
evaluate() on the after checkpoint (09c33bf4) with the WITHOUT-z-limit configuration (band_collision_limit=0, so a
floor crossing does NOT terminate the episode), kstep k=5, canonical n=2000 — reproduces the recorded legacy row
(0.9078). Per episode, from the filtered trajectory, computes: the without-z-limit outcome + infeasibility, whether
p_z crosses z=-4, the tilt at the first crossing, and whether the vehicle was tilted >=60 deg for the ENTIRE run up
to (and including) the crossing (never inside the 60 deg cone). Eval-only; no src/scoring edits. Saves a compact
npz + json for the re-scoring."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
import torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
JT42 = REPO / "data/previous_runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZLIM = 4.0


def tilt_deg(S):                                             # S [T+1, statedim] -> tilt per step (deg)
    q = S[:, 3:7]; q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    w, x, y, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R22 = 1 - 2 * (x * x + y * y)                            # body-up z-component = cos(tilt)
    return np.degrees(np.arccos(np.clip(R22, -1, 1)))


ck = torch.load(JT42, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "k": 5}
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 0.0},  # legacy: band NON-terminal
        "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, _ = load_fw(JT42, config_overrides=over)
for net in ("value_net", "policy_net"):
    m = getattr(fw, net, None)
    if m is not None:
        m.to(DEV)
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=JT42.name, max_scenes=None, include_lqr_baseline=False)
rows = list(res.episode_rows); n = len(rows)
outcome = np.array([r["outcome"] for r in rows]); infeas = np.array([float(r["infeasible_step_frac"]) for r in rows])
crossed = np.zeros(n, bool); cross_step = np.full(n, -1); tilt_at_cross = np.full(n, np.nan)
tilt_ge60_entire = np.zeros(n, bool); tilt0 = np.full(n, np.nan)
for i, tr in enumerate(res.trajectories):
    S = tr.filtered.states.detach().cpu().numpy(); S = S.reshape(S.shape[0], -1)
    pz = S[:, 2]; th = tilt_deg(S); tilt0[i] = th[0]
    below = np.nonzero(pz <= -ZLIM)[0]
    if len(below):
        c = int(below[0]); crossed[i] = True; cross_step[i] = c; tilt_at_cross[i] = th[c]
        tilt_ge60_entire[i] = bool(np.all(th[:c + 1] >= 60.0))
np.savez_compressed(SCR / "legacy_dump.npz", outcome=outcome, infeas=infeas, crossed=crossed, cross_step=cross_step,
                    tilt_at_cross=tilt_at_cross, tilt_ge60_entire=tilt_ge60_entire, tilt0=tilt0)
(SCR / "legacy_rows.json").write_text(json.dumps({"eval_row": res.eval_row}, indent=2, default=float) + "\n")
# quick aggregate cross-check (legacy cps should reproduce 0.9078)
reach = (outcome == "goal").mean(); coll = (outcome == "collision").mean(); stuck = (outcome == "stuck").mean()
oob = (outcome == "oob").mean(); to = (outcome == "timeout").mean()
cps = reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * infeas.mean()
print(f"legacy (band non-terminal) eval: n={n}; cps {cps:.4f} reach {reach:.4f} coll {coll:.4f} oob {oob:.4f} stuck {stuck:.4f} to {to:.4f} infeas {infeas.mean():.4f}")
print(f"  recorded legacy row: cps 0.9078 reach 0.9690 coll 0.0105 oob 0.0000 stuck 0.0005 to 0.0200 infeas 0.0997")
print(f"  crossings: {int(crossed.sum())}; of those tilt>=60 at crossing {int((crossed & (tilt_at_cross>=60)).sum())}, "
      f"tilt<60 {int((crossed & (tilt_at_cross<60)).sum())}; tilt>=60 entire-before {int((crossed & tilt_ge60_entire).sum())}")
