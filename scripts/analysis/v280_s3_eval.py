"""v2.8.0 S3 M4/M5 evaluation — a checkpoint on the canonical d2r pool at the shipped fallback under the
new terminal, with per-component CIs, the angrate_at_reach distribution, and the M1 collision decomposition
(collision_obstacle / collision_band_lower / collision_band_upper, each with its own bootstrap CI).

Usage: --ckpt <best.pt> --projection {dual_solve|enumerate} --tag <name>  [--angrate 0.48]
Persists to data/runs/v2.8.0/s3_eval/<tag>.json."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s3_eval"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_SEED = 20260508

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--projection", required=True, choices=["dual_solve", "enumerate"])
ap.add_argument("--tag", required=True)
ap.add_argument("--angrate", type=float, default=0.48)
ap.add_argument("--phases", type=int, default=1)                    # empty_fallback kstep phases (shipped 1)
ap.add_argument("--k", type=int, default=3)                         # empty_fallback kstep k (shipped 3)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "phases": a.phases, "k": a.k}
filt["projection"] = a.projection
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": a.angrate},
        "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
fw, cfg, ck2 = load_fw(a.ckpt, config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None

res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name=Path(a.ckpt).name,
               max_scenes=None, include_lqr_baseline=False)
r = res.eval_row
eps = res.episode_rows
outc = np.array([e["outcome"] for e in eps])
cause = np.array([e.get("collision_cause", "") for e in eps])
ang = np.array([e["angrate_at_reach"] for e in eps if e["outcome"] == "goal"
                and e["angrate_at_reach"] == e["angrate_at_reach"]])


def boot_rate(mask_vals, n=1000):
    rng = np.random.default_rng(BOOT_SEED); N = len(mask_vals)
    idx = rng.integers(0, N, size=(n, N))
    vals = [float(mask_vals[i].mean()) for i in idx]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]
CI = ["cps_ci_lo", "cps_ci_hi", "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
      "stuck_ci_lo", "stuck_ci_hi", "infeasibility_ci_lo", "infeasibility_ci_hi"]
rep = {"tag": a.tag, "ckpt": str(a.ckpt), "projection": a.projection, "pool": "0ef3751b",
       "fallback": "kstep phases1 k3", "goal_angrate_radius": a.angrate, "device": str(DEV),
       "step": int(ck2["step"]), "n": len(eps)}
rep["outcome"] = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP + CI}
# M1 collision decomposition (episode fractions of the whole pool; sum == collision), each with a CI
for c in ("obstacle", "band_lower", "band_upper"):
    m = (cause == c).astype(float)
    rep[f"collision_{c}"] = float(m.mean())
    rep[f"collision_{c}_ci"] = boot_rate(m)
rep["angrate_at_reach"] = {
    "n_reach": int(ang.size),
    "median": float(np.median(ang)) if ang.size else None,
    "p90": float(np.quantile(ang, 0.90)) if ang.size else None,
    "max": float(ang.max()) if ang.size else None,
    "frac_below_0.48": float((ang < 0.48).mean()) if ang.size else None,
}
# per-episode collision_cause + started-flag join is done later against the C3 sidecar; persist the raw
# per-episode (episode_idx, outcome, collision_cause) so the C3 cross can be computed without re-eval.
rep["episode_cause"] = [{"episode_idx": int(e["episode_idx"]), "outcome": e["outcome"],
                         "collision_cause": e.get("collision_cause", ""),
                         # v2.8.0 cone_split: full per-episode fields so a subset cps can be recomputed
                         "reach": float(e["reach"]), "collision": float(e["collision"]), "oob": float(e["oob"]),
                         "stuck": float(e["stuck"]), "timeout": float(e["timeout"]),
                         "infeasible_step_frac": float(e["infeasible_step_frac"]),
                         "cps_episode": float(e["cps_episode"]),
                         "angrate_at_reach": (float(e["angrate_at_reach"]) if e["angrate_at_reach"] == e["angrate_at_reach"] else None)}
                        for e in eps]
(OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"{a.tag}: cps {r['cps']:.4f} reach {r['reach']:.4f} coll {r['collision']:.4f} "
      f"(obstacle {rep['collision_obstacle']:.4f} / band_lower {rep['collision_band_lower']:.4f} / "
      f"band_upper {rep['collision_band_upper']:.4f}) stuck {r['stuck']:.4f} -> {OUT/(a.tag+'.json')}", flush=True)
