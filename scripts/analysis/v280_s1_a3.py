"""v2.8.0 S1 M4 A3 — paired outcome vector with per-component CIs, dual_solve vs enumerate, eval-only on
09c33bf4 / canonical pool 0ef3751b / deployed fallback (kstep k=5, banded). Registered null: infeasibility
does not move CI-separated. Persists to data/runs/v2.8.0/s1_projection/a3.json."""
from __future__ import annotations
import copy, json
from pathlib import Path
import torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
JT3D = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
POOL = REPO / "data/eval_pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s1_projection"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(proj):
    ck = torch.load(JT3D, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "k": 5}
    filt["projection"] = proj
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, _ = load_fw(JT3D, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None);  m.to(DEV) if m is not None else None
    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=JT3D.name,
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row
    keys = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate",
            "cps_ci_lo", "cps_ci_hi", "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
            "infeasibility_ci_lo", "infeasibility_ci_hi", "n_scenes"]
    return {k: (float(r[k]) if r.get(k) is not None else None) for k in keys if k in r}


rep = {"checkpoint": "09c33bf4", "pool": "0ef3751b", "fallback": "kstep k=5 (banded)", "device": str(DEV)}
for proj in ("enumerate", "dual_solve"):
    rep[proj] = run(proj)
    print(f"A3 {proj}: cps {rep[proj].get('cps'):.4f} reach {rep[proj].get('reach'):.4f} "
          f"coll {rep[proj].get('collision'):.4f} infeas {rep[proj].get('infeasibility'):.4f} "
          f"[{rep[proj].get('infeasibility_ci_lo')},{rep[proj].get('infeasibility_ci_hi')}]", flush=True)
# A3 null: infeasibility CI overlap
e, d = rep["enumerate"], rep["dual_solve"]
if all(k in e and e[k] is not None for k in ("infeasibility_ci_lo", "infeasibility_ci_hi")):
    overlap = not (e["infeasibility_ci_hi"] < d["infeasibility_ci_lo"] or d["infeasibility_ci_hi"] < e["infeasibility_ci_lo"])
    rep["A3_infeasibility_CI_overlap"] = bool(overlap)
    rep["A3_null_holds_infeasibility_not_CI_separated"] = bool(overlap)
(OUT / "a3.json").write_text(json.dumps(rep, indent=2) + "\n")
print("A3 DONE ->", OUT / "a3.json")
