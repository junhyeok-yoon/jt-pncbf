"""v2.7.5 theory R13-R14 — heavy-tail crossing reading, and first-divergence causality. Recorded artifacts only.

R13: on Arm B feasible control instants, excess over the linear bound at p90/p99/p99.9/max, linear-vs-quadratic
fit per quantile; per-instant effective curvature C2(x_k)=2*excess(0.05)/0.05^2 distribution; restricted to the
3159 crossing instants, C2 distribution + fraction crossing by s=0.01. Falsifier: if crossing C2 matches
non-crossing or the high-quantile envelope stays quadratic, the heavy-tail reading fails; do not substitute.
R14: FIRST large jump per scene pair (separation still at RK4 scale); pre-jump separation, empty/feasible flag,
selected-candidate difference. Falsifier: if pre-jump sep >1e-3 or candidates agree, selection change is a
consequence and R11's localization is unsupported. Single-seed. Eval-only. alpha=2.0.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
import torch

from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes
from src.eval.run_full import _load_framework

RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
ALPHA = 2.0; SUB = 5
fw, cfg, _ = _load_framework(CKPT); system = fw.system
v_fn = make_h_fn(fw.value_net, system); scenes = load_pool(POOL).scenes

def load(a):
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    return {"S": z["states"], "em": z["empty"].astype(bool), "uc": s["u_cmd"].astype(np.float64), "n": s["n_steps"].astype(int), "dt": float(z["dt"])}
A, B = load("A_20Hz_coarse"), load("B_20Hz_fine"); N = A["S"].shape[0]; ids = list(range(N))
def q(a, p): return round(float(np.percentile(a, p)), 6) if len(a) else None
R = {}

# ================= R13 =================
s_sec = np.array([j * B["dt"] for j in range(1, SUB + 1)])
exc = {j: [] for j in range(1, SUB + 1)}
C2_all, C2_cross, C2_noncross = [], [], []
n_cross = 0; n_cross_first = 0; n_periods = 0
for i in ids:
    k = int(min(B["n"][i], B["S"].shape[1] - 1))
    if k < SUB: continue
    X = torch.tensor(B["S"][i, :k + 1], dtype=torch.float32)
    b = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad(): vh = v_fn(X, b).numpy().reshape(-1)
    for tk in range(0, k - SUB + 1, SUB):
        if B["em"][i, tk]: continue
        n_periods += 1; v0 = vh[tk]
        e = {}
        for j in range(1, SUB + 1):
            if tk + j <= k:
                e[j] = vh[tk + j] - v0 * (1.0 - ALPHA * j * B["dt"]); exc[j].append(e[j])
        c2 = 2.0 * e.get(SUB, 0.0) / (SUB * B["dt"]) ** 2
        C2_all.append(c2)
        crossed = v0 <= 0.0 and any(vh[tk + j] > 0.0 for j in range(1, SUB + 1) if tk + j <= k)
        first = v0 <= 0.0 and (tk + 1 <= k) and vh[tk + 1] > 0.0
        if crossed: n_cross += 1; C2_cross.append(c2)
        else: C2_noncross.append(c2)
        if first: n_cross_first += 1

def fit(x, y, p):
    bb = x ** p; C = float(np.sum(bb * y) / np.sum(bb * bb)); pred = C * bb
    sr = float(np.sum((y - pred) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    return round(C, 4), round(1 - sr / st, 5) if st > 0 else None
env = {}
for qq in (90, 99, 99.9, 100):
    yv = np.array([np.percentile(exc[j], qq) if qq < 100 else np.max(exc[j]) for j in range(1, SUB + 1)])
    C2q, r2q = fit(s_sec, yv, 2); C1q, r2l = fit(s_sec, yv, 1)
    env[f"q{qq}"] = {"excess_by_s": [round(float(v), 5) for v in yv],
                     "quad_C2": C2q, "quad_R2": r2q, "linear_C1": C1q, "linear_R2": r2l,
                     "implied_curvature_at_s0.05": round(2 * float(yv[-1]) / 0.05 ** 2, 2),
                     "implied_curvature_at_s0.01": round(2 * float(yv[0]) / 0.01 ** 2, 2)}
def cdist(a): a = np.asarray(a, float); return {"n": int(a.size), "median": q(a, 50), "p90": q(a, 90), "p99": q(a, 99), "max": q(a, 100)}
R["R13_heavy_tail"] = {
    "n_feasible_control_periods": n_periods,
    "excess_envelope_by_quantile": env,
    "per_instant_C2_all": cdist(C2_all),
    "per_instant_C2_crossing_instants": cdist(C2_cross),
    "per_instant_C2_noncrossing": cdist(C2_noncross),
    "n_crossing": n_cross, "n_crossing_by_first_substep": n_cross_first,
    "frac_crossings_reached_by_s0.01": round(n_cross_first / max(n_cross, 1), 4),
    "registered": "high-quantile envelope degrades from quadratic toward flat; crossing C2 >> non-crossing; most crossings by s=0.01",
    "falsifier": "crossing C2 ~ non-crossing OR high-quantile envelope still quadratic -> heavy-tail reading fails, dt^1.16 stays unexplained"}

# ================= R14 =================
RATIO, FLOOR = 100.0, 1e-9
pre_sep, jump_empty, jump_feas, cand_diff, out_over_in = [], 0, 0, 0, []
n_found = 0
for i in ids:
    kA = int(min(A["n"][i], A["S"].shape[1] - 1)); K = min(kA, (B["S"].shape[1] - 1) // SUB)
    if K < 3: continue
    pA = A["S"][i, :K + 1, :3]; pB = B["S"][i, :K * SUB + 1:SUB, :3][:K + 1]
    d = np.linalg.norm(pA - pB, axis=1)
    first = None
    for k in range(K):
        if d[k] > FLOOR and d[k + 1] / d[k] > RATIO:
            first = k; break
    if first is None: continue
    n_found += 1; k = first
    pre_sep.append(float(d[k]))
    eA = bool(A["em"][i, k]); eB = bool(B["em"][i, SUB * k: SUB * k + SUB].any())
    if eA or eB: jump_empty += 1
    else: jump_feas += 1
    out = float(np.linalg.norm(A["uc"][i, k] - B["uc"][i, SUB * k]))       # deployed-action (selected-candidate) diff
    if out > 1e-3: cand_diff += 1
    # full-state input separation (13-dim) at the control instant, and the output/input amplification
    xin = float(np.linalg.norm(A["S"][i, k] - B["S"][i, SUB * k]))
    out_over_in.append(out / max(xin, 1e-12))
pre = np.array(pre_sep)
R["R14_first_divergence"] = {
    "ratio_threshold": RATIO, "n_scene_pairs_with_a_jump": n_found,
    "pre_jump_separation_m": {"median": q(pre, 50), "p90": q(pre, 90), "max": q(pre, 100),
                              "frac_below_1e-5": round(float((pre < 1e-5).mean()), 4),
                              "frac_below_1e-3": round(float((pre < 1e-3).mean()), 4)},
    "first_jump_branch": {"empty_active": jump_empty, "feasible": jump_feas,
                          "frac_empty": round(jump_empty / max(n_found, 1), 4)},
    "selected_candidate_differs_at_first_jump": {"n": cand_diff, "frac": round(cand_diff / max(n_found, 1), 4)},
    "output_over_input_amplification_at_first_jump": {"median": q(np.array(out_over_in), 50),
        "p90": q(np.array(out_over_in), 90), "max": q(np.array(out_over_in), 100),
        "note": "||u_cmd_A - u_cmd_B|| / ||x_A - x_B|| at the control instant; O(1)/1e-5 => discontinuous selection"},
    "code_read_box_aware_projection": {
        "finding": ("SELECTION over a finite enumerated candidate set, NOT an exact convex-QP solve, for the "
                    "4-D quadrotor_3d action -> can jump discontinuously across ties"),
        "citations": [
            "src/common/filter_hardnet.py:233-285 _candidate_actions enumerates {clamped u_nom, base "
            "projection, all 2^action_dim box corners (line 247), fix-one-coord-solve-one-coord points "
            "(lines 254-283)}",
            "src/common/filter_hardnet.py:240 comment: 'The box-aware enumerator is specified for the current "
            "2-D action systems' -> exact only for 2-D; the 4-D quadrotor optimum (>=2 box coords active + "
            "halfspace) is NOT in the enumeration",
            "src/common/filter_hardnet.py:214,218,221 _box_aware_projection selects via argmin (feasible: "
            "argmin distance_sq; else argmin violation) -> a finite argmin, discontinuous at ties"],
        "registered": "at first divergence pre-jump sep ~1e-5 AND selected candidate already differs -> selection is the cause",
        "falsifier": "pre-jump sep already >1e-3 OR candidates agree at first jump -> selection is a consequence, R11 localization unsupported"}}

(RD / "r13r14.json").write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
