"""v2.7.5 theory R15 — how much would a finer training grid (dt=0.01) change the label? Recorded artifacts only.

Decision rule / falsifier / null model registered in r15_decision_rule.json BEFORE this ran. Backward pathwise
discounted-avoid recursion on the recorded 0.01-grid trajectories of Arms B and C:
  V(x_T)=h_star(x_T); V(x_k)=max(h_star(x_k), (1-gamma)*h_star(x_k)+gamma*V(x_next))
coarse V_005: x_next=+5 steps, gamma=exp(-lambda*0.05); fine V_001: x_next=+1, gamma=exp(-lambda*0.01);
compared at coarse grid indices anchored at the terminal. lambda held at 0.10662751890126089. Undiscounted
running-max control alongside. Single-seed. Eval-only.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
import torch

from src.common.quadrotor_barrier import make_barrier_fn
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes
from src.eval.run_full import _load_framework

RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
LAM = 0.10662751890126089; ALPHA = 2.0; SUB = 5
G005 = float(np.exp(-LAM * 0.05)); G001 = float(np.exp(-LAM * 0.01))
fw, cfg, _ = _load_framework(CKPT); system = fw.system
h_star_fn = make_barrier_fn(float(cfg["env"][system.name]["c_gain"]), float(cfg["env"]["h_scale"]))
v_fn = make_h_fn(fw.value_net, system); scenes = load_pool(POOL).scenes

def load(a):
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    return {"S": z["states"], "n": s["n_steps"].astype(int), "dt": float(z["dt"]),
            "oc": {i: rows[i]["outcome"] for i in rows}}

def q(a, p): return round(float(np.percentile(a, p)), 6) if len(a) else None
def dist(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "median": q(a, 50), "p90": q(a, 90), "p99": q(a, 99), "max": q(a, 100)} if a.size else {"n": 0}

def run_arm(a):
    D = load(a)
    gap, v005c, v001c, vhatc, hstarc, speedc, c2c, outc_c = [], [], [], [], [], [], [], []
    ctrl_gap = []                       # undiscounted running-max gap (fine - coarse)
    for i in range(D["S"].shape[0]):
        T = int(min(D["n"][i], D["S"].shape[1] - 1))
        if T < SUB + 1: continue
        X = torch.tensor(D["S"][i, :T + 1], dtype=torch.float32)
        b = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            h = h_star_fn(X, b).numpy().reshape(-1)
            vhat = v_fn(X, b).numpy().reshape(-1)
        h = np.clip(h, -1.0, 1.0)
        # fine label V_001 at every step
        V1 = np.empty(T + 1); V1[T] = h[T]
        for k in range(T - 1, -1, -1):
            V1[k] = max(h[k], (1 - G001) * h[k] + G001 * V1[k + 1])
        # fine running-max
        M1 = np.empty(T + 1); M1[T] = h[T]
        for k in range(T - 1, -1, -1):
            M1[k] = max(h[k], M1[k + 1])
        # coarse grid anchored at terminal: indices T, T-5, ..., >=0
        cidx = list(range(T, -1, -SUB))[::-1]
        V5 = {cidx[-1]: h[cidx[-1]]}
        for c in cidx[-2::-1]:
            V5[c] = max(h[c], (1 - G005) * h[c] + G005 * V5[c + SUB])
        M5 = {cidx[-1]: h[cidx[-1]]}
        for c in cidx[-2::-1]:
            M5[c] = max(h[c], M5[c + SUB])
        for c in cidx:
            g = V1[c] - V5[c]
            gap.append(g); v005c.append(V5[c]); v001c.append(V1[c]); vhatc.append(float(vhat[c]))
            hstarc.append(float(h[c])); speedc.append(float(np.linalg.norm(D["S"][i, c, 7:10])))
            ctrl_gap.append(M1[c] - M5[c]); outc_c.append(D["oc"][i])
            # per-instant C2 (only where a full +5 window exists ahead): 2*excess(0.05)/0.05^2
            if c + SUB <= T:
                v0 = float(vhat[c]); vsub = float(vhat[c + SUB])
                c2c.append(2.0 * (vsub - v0 * (1 - ALPHA * 0.05)) / 0.05 ** 2)
            else:
                c2c.append(np.nan)
    gap = np.array(gap); v005 = np.array(v005c); v001 = np.array(v001c); vhat = np.array(vhatc)
    hstar = np.array(hstarc); speed = np.array(speedc); c2 = np.array(c2c); cg = np.array(ctrl_gap)
    outc = np.array(outc_c)
    n = gap.size
    # R15.2 sign flips
    R2 = {"n_states": int(n),
          "a_V005le0_lt_V001": round(float(((v005 <= 0) & (v001 > 0)).mean()), 6),
          "a_reverse_V001le0_lt_V005": round(float(((v001 <= 0) & (v005 > 0)).mean()), 6),
          "b_Vhatle0_lt_V001_DEPLOYMENT": round(float(((vhat <= 0) & (v001 > 0)).mean()), 6),
          "b_reverse_V001le0_lt_Vhat": round(float(((v001 <= 0) & (vhat > 0)).mean()), 6),
          "frac_discounted_gap_negative": round(float((gap < 0).mean()), 6),
          "frac_undiscounted_control_gap_negative": round(float((cg < -1e-9).mean()), 6)}
    # R15.3 gap distribution + by C2 quantile + by outcome
    band = np.abs(vhat) <= 0.05
    finite = np.isfinite(c2)
    by_c2 = {}
    if finite.sum() > 100:
        cq = np.percentile(c2[finite], [50, 90, 99])
        for lbl, lo, hi in [("<=p50", -np.inf, cq[0]), ("p50-p90", cq[0], cq[1]),
                             ("p90-p99", cq[1], cq[2]), (">p99", cq[2], np.inf)]:
            m = finite & (c2 > lo) & (c2 <= hi)
            by_c2[lbl] = {"n": int(m.sum()), "gap_median": q(gap[m], 50), "gap_p90": q(gap[m], 90)}
    by_out = {}
    for o in ("collision", "goal", "timeout", "oob", "stuck"):
        m = outc == o
        if m.sum() > 0: by_out[o] = {"n": int(m.sum()), "gap_median": q(gap[m], 50), "gap_p90": q(gap[m], 90)}
    R3 = {"gap_overall": dist(gap), "gap_within_absVhat_le0.05": dist(gap[band]),
          "undiscounted_control_gap": dist(cg),
          "gap_by_C2_quantile": by_c2, "gap_by_outcome": by_out,
          "R5_residual_reference": 0.0479,
          "gap_median_vs_R5_residual": round(float(np.median(gap)) / 0.0479, 3)}
    # R15.4 null model
    def r2(x, y):
        x = np.asarray(x); y = np.asarray(y); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10: return None
        c = np.corrcoef(x[m], y[m])[0, 1]; return round(float(c * c), 4)
    R4 = {"R2_gap_on_hstar": r2(hstar, gap), "R2_gap_on_speed": r2(speed, gap),
          "R2_gap_on_absVhat": r2(np.abs(vhat), gap)}
    return {"gamma_005": round(G005, 6), "gamma_001": round(G001, 6),
            "R15.2_sign_flips": R2, "R15.3_gap": R3, "R15.4_null_model": R4}

R = {"lambda_disc": LAM, "arm_B_train_period": run_arm("B_20Hz_fine"),
     "arm_C_deploy_loop": run_arm("C_100Hz_fine")}
(RD / "r15_grid_label.json").write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
