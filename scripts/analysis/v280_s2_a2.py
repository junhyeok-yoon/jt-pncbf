"""v2.8.0 S2 A2 — planar angular rate at reach (the Stage-0 B0 probe on the planar bold checkpoint).

Loads 3b27d691 (v2.7.0 iter-5, the checkpoint the v2.7.1 planar bold row is measured on) on its
canonical pool, at the v2.7.1 recorded deployment (enumerate, empty_fallback kstep k=5), evaluates,
and reports the distribution of the planar angular rate |omega| = |x[...,5]| at the reach instant
(first step the current dist+speed predicate fires) over goal episodes. Measurement, not a gate;
omega_G is not chosen here. Artifact -> data/runs/v2.8.0/s2_terminal/a2_planar_b0.json."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/s2_terminal"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(CK, map_location="cpu", weights_only=False)
filt = copy.deepcopy(ck["config"]["filter"])
filt["empty_fallback"] = {"mode": "kstep", "k": 5}
filt["projection"] = "enumerate"                    # v2.7.1 recorded deployment
over = {"eval": {"max_steps": 200}, "filter": filt}
fw, cfg, _ = load_fw(CK, config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None); m.to(DEV) if m is not None else None

res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name=CK.name,
               max_scenes=None, include_lqr_baseline=False)
r = res.eval_row
omegas = []
for tr in res.trajectories:
    if tr.filtered_outcome != "goal":
        continue
    es = tr.filtered_event_step
    if es is None or es < 0:
        continue
    st = tr.filtered.states                      # [T+1, 1, 6]
    es = min(es, st.shape[0] - 1)
    omegas.append(abs(float(st[es, 0, 5])))       # planar angular rate = |x[5]|
om = np.array(omegas)
rep = {
    "checkpoint": "3b27d691", "pool": "eval_full_quadrotor-planar_n2000_seed23456",
    "deployment": "enumerate, empty_fallback kstep k=5", "device": str(DEV),
    "outcome_row": {k: (float(r[k]) if r.get(k) is not None else None)
                    for k in ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")},
    "n_reach": int(om.size),
    "omega_at_reach_radps": {
        "median": float(np.median(om)) if om.size else None,
        "p90": float(np.quantile(om, 0.90)) if om.size else None,
        "min": float(om.min()) if om.size else None,
        "max": float(om.max()) if om.size else None,
        "mean": float(om.mean()) if om.size else None,
    },
    "fraction_below": {str(t): float((om < t).mean()) if om.size else None
                       for t in (0.10, 0.25, 0.50, 1.00)},
    "omega_G_planar_if_ratio_matched": 0.48,     # v_G/v_max=0.12 x omega_max=4.0 (same as 3D); NOT chosen here
}
np.savez_compressed(OUT / "a2_planar_omega.npz", omega=om)
(OUT / "a2_planar_b0.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
