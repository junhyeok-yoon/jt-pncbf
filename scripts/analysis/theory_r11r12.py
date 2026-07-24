"""v2.7.5 theory R11-R12 — localize the discontinuity, measure boundary concentration. Recorded artifacts only.

R11: do the O(1) A-vs-B separation jumps localize on the empty branch? Cross-tab large single-step increases in
||x_A - x_B|| against empty-branch activity, with denominators; plus the direct tie-crossing count (selected
candidate differs between A and B on shared-empty control instants). Falsifier: large jumps at feasible steps at
a comparable per-step rate -> empty branch is NOT the discontinuity; then report whether feasible jumps coincide
with a change in the selected box face/corner.
R12: is the state distribution concentrated at V_hat=0? Empirical density of V_hat(x_k) at Arm B (and C) control
instants over [-0.5,0], histogram + power-law |V_hat|^(-beta) fit over the decade nearest zero. Registered
prediction beta ~ 0.42 (2(1-beta)=1.16). Falsifier: beta<=0 (flat/falling) -> concentration fails, dt^1.16
unexplained. Bound is SUFFICIENT-only. Single-seed. Eval-only.
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
UMAX = 4.905; SUB = 5
fw, cfg, _ = _load_framework(CKPT); system = fw.system
v_fn = make_h_fn(fw.value_net, system); scenes = load_pool(POOL).scenes

def load(a):
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    return {"S": z["states"], "em": z["empty"].astype(bool), "uc": s["u_cmd"].astype(np.float64),
            "n": s["n_steps"].astype(int)}
A, B, C = load("A_20Hz_coarse"), load("B_20Hz_fine"), load("C_100Hz_fine")
N = A["S"].shape[0]; ids = list(range(N))
R = {}

# ================= R11 =================
incs, emp_flag, tie_diff, tie_total = [], [], 0, 0
# also collect for feasible-jump face-change analysis
per = []
for i in ids:
    kA = int(min(A["n"][i], A["S"].shape[1] - 1)); K = min(kA, (B["S"].shape[1] - 1) // SUB)
    if K < 2: continue
    pA = A["S"][i, :K + 1, :3]; pB = B["S"][i, :K * SUB + 1:SUB, :3][:K + 1]
    d = np.linalg.norm(pA - pB, axis=1)
    for k in range(K):
        inc = float(d[k + 1] - d[k])
        eA = bool(A["em"][i, k]); eB = bool(B["em"][i, SUB * k: SUB * k + SUB].any())
        empty_active = eA or eB
        incs.append(inc); emp_flag.append(empty_active)
        per.append((i, k, inc, empty_active))
        # tie-crossing: both control-instant actions empty -> compare selected candidate (u_cmd)
        if eA and bool(B["em"][i, SUB * k]):
            tie_total += 1
            if float(np.linalg.norm(A["uc"][i, k] - B["uc"][i, SUB * k])) > 1e-3:
                tie_diff += 1
incs = np.array(incs); emp_flag = np.array(emp_flag)
thr_pct = float(np.percentile(incs, 99))                  # top-1% threshold
THR_ABS = 0.01                                            # 0.01 m single A-step jump
for name, thr in (("top1pct", thr_pct), ("abs_0.01m", THR_ABS)):
    big = incs > thr
    n_emp = int(emp_flag.sum()); n_fea = int((~emp_flag).sum())
    big_emp = int((big & emp_flag).sum()); big_fea = int((big & ~emp_flag).sum())
    R.setdefault("R11_jump_localization", {})[name] = {
        "threshold": round(thr, 5), "n_large_jumps": int(big.sum()),
        "empty_active_intervals": n_emp, "feasible_intervals": n_fea,
        "large_jumps_on_empty": big_emp, "large_jumps_on_feasible": big_fea,
        "rate_per_empty_interval": round(big_emp / max(n_emp, 1), 6),
        "rate_per_feasible_interval": round(big_fea / max(n_fea, 1), 6),
        "frac_of_large_jumps_that_are_empty": round(big_emp / max(int(big.sum()), 1), 4),
        "rate_ratio_empty_over_feasible": round((big_emp / max(n_emp, 1)) / max(big_fea / max(n_fea, 1), 1e-12), 2)}
# feasible-jump face-change check (for the falsifier branch): among large jumps on FEASIBLE intervals, how
# often did A's selected u_cmd change its active-bound set between k-1 and k (a box-face switch)?
def active_bounds(u): return tuple((np.isclose(u, 0.0, atol=1e-3) | np.isclose(u, UMAX, atol=1e-3)).astype(int))
fea_big = [(i, k) for (i, k, inc, ea) in per if inc > thr_pct and not ea and k >= 1]
face_change = sum(1 for (i, k) in fea_big if active_bounds(A["uc"][i, k]) != active_bounds(A["uc"][i, k - 1]))
R["R11_jump_localization"]["feasible_large_jumps_face_switch"] = {
    "n_feasible_large_jumps_top1pct": len(fea_big),
    "n_with_box_face_selection_change": face_change,
    "frac": round(face_change / max(len(fea_big), 1), 4)}
R["R11_jump_localization"]["tie_crossing_on_shared_empty_control_instants"] = {
    "n_shared_empty_instants": tie_total, "n_selected_candidate_differs": tie_diff,
    "frac": round(tie_diff / max(tie_total, 1), 4)}

# ================= R12 =================
def vhat_at_control_instants(arm, step_stride):
    vals = []
    for i in ids:
        k = int(min(arm["n"][i], arm["S"].shape[1] - 1))
        if k < 1: continue
        idx = list(range(0, k + 1, step_stride))
        X = torch.tensor(arm["S"][i, idx], dtype=torch.float32)
        b = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad(): vals.append(v_fn(X, b).numpy().reshape(-1))
    return np.concatenate(vals)

def density_powerlaw(vh, label):
    v = vh[(vh >= -0.5) & (vh <= 0.0)]
    absv = -v                                             # |V_hat| in [0, 0.5]
    # histogram in log-space over the decade nearest zero: |V_hat| in [1e-3, 1e-1]
    edges = np.logspace(-3, np.log10(0.5), 25)
    cnt, _ = np.histogram(absv, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:]); widths = np.diff(edges)
    dens = cnt / (widths * max(v.size, 1))                # normalized density
    # fit over the decade nearest zero: |V_hat| in [1e-2, 1e-1]
    m = (centers >= 1e-2) & (centers <= 1e-1) & (dens > 0)
    beta = beta_ci = r2 = None
    if m.sum() >= 3:
        x = np.log(centers[m]); y = np.log(dens[m])
        A_ = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A_, y, rcond=None)
        slope = float(coef[0]); beta = -slope             # dens ~ |V|^(slope) = |V|^(-beta)
        yhat = A_ @ coef; ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        n_ = m.sum(); se = float(np.sqrt(ss_res / max(n_ - 2, 1)) / np.sqrt(np.sum((x - x.mean()) ** 2)))
        beta_ci = [round(beta - 1.96 * se, 4), round(beta + 1.96 * se, 4)]
    return {"label": label, "n_control_instants_in[-0.5,0]": int(v.size),
            "histogram_centers_absV": [round(float(c), 5) for c in centers],
            "histogram_density": [round(float(x), 3) for x in dens],
            "counts": [int(c) for c in cnt],
            "powerlaw_fit_decade_[0.01,0.1]": {"beta": round(beta, 4) if beta is not None else None,
                "beta_95CI": beta_ci, "R2": round(r2, 4) if r2 is not None else None,
                "registered_prediction_beta": 0.42},
            "frac_within_0.05_of_zero": round(float((absv <= 0.05).mean()), 5),
            "frac_within_0.01_of_zero": round(float((absv <= 0.01).mean()), 5)}

vhB = vhat_at_control_instants(B, SUB)     # B: control instant every 5th step
vhC = vhat_at_control_instants(C, 1)       # C: every step is a control instant
R["R12_boundary_concentration"] = {
    "note": "density of V_hat(x_k) at control instants over [-0.5,0]; power-law |V_hat|^(-beta) over decade [0.01,0.1]",
    "falsifier": "beta<=0 (flat/falling) -> concentration explanation fails, dt^1.16 unexplained",
    "bound_is_sufficient_only": "a mismatch between the bound's dt^2 and the observed rate is NOT by itself a refutation of prop:zoh-layer",
    "arm_B": density_powerlaw(vhB, "B_20Hz_fine"), "arm_C": density_powerlaw(vhC, "C_100Hz_fine")}

(RD / "r11r12.json").write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
