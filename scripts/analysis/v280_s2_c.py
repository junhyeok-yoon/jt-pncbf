"""v2.8.0 S2 Part C — the comparator (T1) and the shipped-fallback measurement (C2).

C1/T1: re-score 09c33bf4 (3D) and 3b27d691 (planar) on their canonical pools under the NEW terminal
(goal_angrate_radius=0.48), eval-only, at the shipped fallback. Report the full outcome vector with
per-component CIs and the angrate_at_reach distribution (now a native episode column, B5). Score T1's
stuck prediction (stuck rises >0.10 vs the OLD terminal; falsifier <0.02) by measuring both terminals
at the same config.
C2: the 3D canonical pool at {kstep,phases 1,k 3} (shipped) and {kstep,phases 2,k 5} (what every
recorded number used), both under the new terminal, both with CIs.
Artifacts -> data/runs/v2.8.0/s2_terminal/c_terminal.json."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.8.0/s2_terminal"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JT3D = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
POOL3D = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
PLANAR = REPO / "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt"
POOLPL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"

CI = ["cps_ci_lo", "cps_ci_hi", "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
      "stuck_ci_lo", "stuck_ci_hi", "infeasibility_ci_lo", "infeasibility_ci_hi"]
COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]


def run(ck, pool, fallback, projection, angrate):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(c["config"]["filter"]); filt["empty_fallback"] = fallback; filt["projection"] = projection
    env = {"goal_angrate_radius": angrate} if angrate is not None else {}
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0, **env},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = load_fw(ck, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    res = evaluate(fw, pool, cfg, mode="final", step=int(ck2["step"]), ckpt_name=ck.name,
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row
    ang = np.array([ep["angrate_at_reach"] for ep in res.episode_rows
                    if ep["outcome"] == "goal" and ep["angrate_at_reach"] == ep["angrate_at_reach"]])  # drop nan
    out = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP + CI}
    out["n_reach"] = int((np.array([ep["outcome"] for ep in res.episode_rows]) == "goal").sum())
    out["angrate_at_reach"] = {
        "n": int(ang.size),
        "median": float(np.median(ang)) if ang.size else None,
        "p90": float(np.quantile(ang, 0.90)) if ang.size else None,
        "max": float(ang.max()) if ang.size else None,
        "frac_below_omega_G_0.48": float((ang < 0.48).mean()) if ang.size else None,
    }
    return out


rep = {"checkpoints": {"3d": "09c33bf4", "planar": "3b27d691"}, "pool_3d": "0ef3751b", "omega_G": 0.48, "device": str(DEV)}

# ---- C1/T1: 3D at shipped {kstep,phases1,k3}, dual; OLD vs NEW terminal ----
fb_ship = {"mode": "kstep", "phases": 1, "k": 3}
rep["T1_3d_old_terminal"] = run(JT3D, POOL3D, fb_ship, "dual_solve", None)
rep["T1_3d_new_terminal"] = run(JT3D, POOL3D, fb_ship, "dual_solve", 0.48)
d_stuck_3d = rep["T1_3d_new_terminal"]["stuck"] - rep["T1_3d_old_terminal"]["stuck"]
rep["T1_3d_stuck_delta"] = d_stuck_3d
rep["T1_3d_prediction_stuck_rise_gt_0.10"] = bool(d_stuck_3d > 0.10)
rep["T1_3d_falsifier_stuck_rise_lt_0.02"] = bool(d_stuck_3d < 0.02)
print(f"T1 3D: stuck old {rep['T1_3d_old_terminal']['stuck']:.4f} -> new {rep['T1_3d_new_terminal']['stuck']:.4f} "
      f"(delta {d_stuck_3d:+.4f}); reach {rep['T1_3d_old_terminal']['reach']:.4f} -> {rep['T1_3d_new_terminal']['reach']:.4f}", flush=True)

# ---- C1/T1: planar at shipped {kstep,k5}, enumerate (recorded deployment); OLD vs NEW terminal ----
fb_pl = {"mode": "kstep", "k": 5}
rep["T1_planar_old_terminal"] = run(PLANAR, POOLPL, fb_pl, "enumerate", None)
rep["T1_planar_new_terminal"] = run(PLANAR, POOLPL, fb_pl, "enumerate", 0.48)
d_stuck_pl = rep["T1_planar_new_terminal"]["stuck"] - rep["T1_planar_old_terminal"]["stuck"]
rep["T1_planar_stuck_delta"] = d_stuck_pl
rep["T1_planar_prediction_stuck_rise_gt_0.10"] = bool(d_stuck_pl > 0.10)
rep["T1_planar_falsifier_stuck_rise_lt_0.02"] = bool(d_stuck_pl < 0.02)
print(f"T1 planar: stuck old {rep['T1_planar_old_terminal']['stuck']:.4f} -> new {rep['T1_planar_new_terminal']['stuck']:.4f} "
      f"(delta {d_stuck_pl:+.4f}); reach {rep['T1_planar_old_terminal']['reach']:.4f} -> {rep['T1_planar_new_terminal']['reach']:.4f}", flush=True)

# ---- C2: 3D new terminal at shipped {phases1,k3} (reuse) and at {phases2,k5} ----
rep["C2_3d_phases1_k3_new_terminal"] = rep["T1_3d_new_terminal"]
rep["C2_3d_phases2_k5_new_terminal"] = run(JT3D, POOL3D, {"mode": "kstep", "phases": 2, "k": 5}, "dual_solve", 0.48)
print(f"C2 3D {{p1,k3}} cps {rep['C2_3d_phases1_k3_new_terminal']['cps']:.4f} | "
      f"{{p2,k5}} cps {rep['C2_3d_phases2_k5_new_terminal']['cps']:.4f}", flush=True)

(OUT / "c_terminal.json").write_text(json.dumps(rep, indent=2) + "\n")
print("PART C DONE ->", OUT / "c_terminal.json")

# ---- ledger rows (eval_only, unbolded) ----
def row(ver, sysn, parent, arm, m):
    def g(k): return f"{m[k]:.4f}" if m.get(k) is not None else ""
    return (f"| {ver} | {sysn} | 2026-07-30 | {parent} | 42 | {g('cps')} | eval_only({arm}) | {g('reach')} | "
            f"{g('collision')} | {g('oob')} | {g('stuck')} | {g('timeout')} | {g('infeasibility')} | {g('saturation_rate')} | "
            f"{g('cps')} | EXPLORATORY; v2.8.0 S2 new terminal (omega_G=0.48); {arm}; unbolded |")
print("\n=== LEDGER ROWS (append) ===")
print(row("v2.8.0", "quadrotor_3d", "09c33bf4", "new terminal, shipped fallback kstep phases1 k3, canonical pool 0ef3751b", rep["C2_3d_phases1_k3_new_terminal"]))
print(row("v2.8.0", "quadrotor_3d", "09c33bf4", "new terminal, kstep phases2 k5 (recorded-config), canonical pool 0ef3751b", rep["C2_3d_phases2_k5_new_terminal"]))
print(row("v2.8.0", "quadrotor_planar", "3b27d691", "new terminal, kstep k5, enumerate, canonical planar pool", rep["T1_planar_new_terminal"]))
