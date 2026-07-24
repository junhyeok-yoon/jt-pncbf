"""v2.7.5 theory addendum R7-R10 — sampled-data follow-ups (measurement only, no verdict change).

R7 full A-vs-B paired table + cps churn decomposition (the integration-grid NULL MODEL for B-vs-C's churn).
R8 corrected inter-sample: (a) boundary band |V_hat(x_k)|<=0.05 linear-vs-quadratic; (b) excess over the
   linear bound term regressed on s^2; plus first-substep sign-crossing rate (the dt_ctrl=0.01 counterfactual).
R9 empty-branch gradient classification (clamped-nominal / corner / neither), B and C, with denominators.
R10 A-vs-B state separation growth (sensitive dependence vs branch-switch), agree and disagree groups.
Params from metrics.csv @39000: alpha=2.0. Single-seed. Eval-only.
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
OUT = Path("data/runs/v2.7.5/dt_ctrl_arms"); ALPHA = 2.0
OUTC = ["goal", "collision", "timeout", "stuck", "oob"]
UMAX = 4.905
fw, cfg, _ = _load_framework(CKPT); system = fw.system
v_fn = make_h_fn(fw.value_net, system)
scenes = load_pool(POOL).scenes

def load(a):
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    return {"S": z["states"], "em": z["empty"].astype(bool), "dt": float(z["dt"]),
            "uc": s["u_cmd"].astype(np.float64), "un": s["u_nom"].astype(np.float64),
            "n": s["n_steps"].astype(int), "oc": {i: rows[i]["outcome"] for i in rows},
            "cps": {i: float(rows[i]["cps_episode"]) for i in rows}}
A, B, C = load("A_20Hz_coarse"), load("B_20Hz_fine"), load("C_100Hz_fine")
N = A["S"].shape[0]; ids = list(range(N))
def q(a, p): return round(float(np.percentile(a, p)), 5) if len(a) else None
def dist(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "median": q(a, 50), "iqr": [q(a, 25), q(a, 75)], "p90": q(a, 90), "max": q(a, 100)} if a.size else {"n": 0}
R = {}

# ---------------- R7 ----------------
tab = {ob: {oc: 0 for oc in OUTC} for ob in OUTC}
for i in ids: tab[A["oc"][i]][B["oc"][i]] += 1
offdiag = int(sum(tab[a][b] for a in OUTC for b in OUTC if a != b))
d = np.array([B["cps"][i] - A["cps"][i] for i in ids])
gross_pos = float(d[d > 0].sum()); gross_neg = float(d[d < 0].sum()); net = float(d.sum())
# by outcome transition: sum of cps change grouped by (A->B) cell
by_cell = {}
for i in ids:
    key = f"{A['oc'][i]}->{B['oc'][i]}"
    by_cell[key] = by_cell.get(key, 0.0) + (B["cps"][i] - A["cps"][i])
top = sorted(((k, round(v, 3)) for k, v in by_cell.items() if abs(v) > 1.0), key=lambda x: -abs(x[1]))
R["R7_A_vs_B_full_table_and_churn"] = {
    "table_A_rows_B_cols": tab, "total_offdiagonal_scenes_changed": offdiag,
    "cps_churn_by_scene_sum": {"gross_positive": round(gross_pos, 3), "gross_negative": round(gross_neg, 3),
                               "net": round(net, 3), "total_movement_abs": round(gross_pos - gross_neg, 3)},
    "B_vs_C_reference_from_M5": {"changed": 118, "gross_positive": 139.5, "gross_negative": -81.5, "net": 58.0},
    "cps_change_by_transition_cell_over1": top,
    "registered_reading": ("if A-vs-B churn magnitude ~ B-vs-C's while its net ~0, churn is not specific to "
                           "control rate; only the net drift is")}

# ---------------- R8 ----------------
SUB = 5; s_sec = np.array([j * B["dt"] for j in range(1, SUB + 1)])
band_raw = {j: [] for j in range(1, SUB + 1)}; excess_all = {j: [] for j in range(1, SUB + 1)}
n_periods = 0; n_first_cross = 0; n_any_cross = 0
for i in ids:
    k = int(min(B["n"][i], B["S"].shape[1] - 1))
    if k < SUB: continue
    X = torch.tensor(B["S"][i, :k + 1], dtype=torch.float32)
    b = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad(): vh = v_fn(X, b).numpy().reshape(-1)
    for tk in range(0, k - SUB + 1, SUB):
        if B["em"][i, tk]: continue
        n_periods += 1; v0 = vh[tk]; in_band = abs(v0) <= 0.05
        any_c = False; first_c = False
        for j in range(1, SUB + 1):
            if tk + j > k: continue
            vs = vh[tk + j]; s = j * B["dt"]
            if in_band: band_raw[j].append(vs - v0)
            excess_all[j].append(vs - v0 * (1.0 - ALPHA * s))   # excess over the linear-bound term
            if vs > 0.0:
                any_c = True
                if j == 1: first_c = True
        if v0 <= 0.0 and any_c: n_any_cross += 1
        if v0 <= 0.0 and first_c: n_first_cross += 1

def fit_pow(x, y, p):
    bb = x ** p; C = float(np.sum(bb * y) / np.sum(bb * bb)); pred = C * bb
    sr = float(np.sum((y - pred) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    return C, (1 - sr / st if st > 0 else None)
band_p90 = np.array([np.percentile(band_raw[j], 90) if band_raw[j] else np.nan for j in range(1, SUB + 1)])
bC2, bR2q = fit_pow(s_sec, band_p90, 2); bC1, bR2l = fit_pow(s_sec, band_p90, 1)
# (b) excess ~ s^2, coefficient vs C2/2
exc_mean = np.array([np.mean(excess_all[j]) for j in range(1, SUB + 1)])
exc_p90 = np.array([np.percentile(excess_all[j], 90) for j in range(1, SUB + 1)])
eC_mean, eR2_mean = fit_pow(s_sec, exc_mean, 2)
eC_p90, eR2_p90 = fit_pow(s_sec, exc_p90, 2)
# linear R2 of excess (should be worse if excess is quadratic)
_, eR2_mean_lin = fit_pow(s_sec, exc_mean, 1)
R["R8_corrected_intersample"] = {
    "n_feasible_control_periods": n_periods,
    "a_boundary_band_absV0_le_0.05": {"n_in_band_note": "instants with |V_hat(x_k)|<=0.05",
        "p90_raw_excursion_by_s": [round(float(v), 6) for v in band_p90],
        "quadratic_C2": round(bC2, 4), "quadratic_R2": round(bR2q, 5) if bR2q is not None else None,
        "linear_C1": round(bC1, 4), "linear_R2": round(bR2l, 5) if bR2l is not None else None},
    "b_excess_over_linear_bound_vs_s2": {
        "excess_definition": "V_hat(x(t_k+s)) - V_hat(x_k)*(1 - alpha*s), alpha=2.0, s in seconds",
        "mean_excess_by_s": [round(float(v), 6) for v in exc_mean],
        "p90_excess_by_s": [round(float(v), 6) for v in exc_p90],
        "fit_mean_coeff_s2": round(eC_mean, 4), "fit_mean_R2_quadratic": round(eR2_mean, 5),
        "fit_mean_R2_linear_for_contrast": round(eR2_mean_lin, 5) if eR2_mean_lin is not None else None,
        "fit_p90_coeff_s2": round(eC_p90, 4), "fit_p90_R2_quadratic": round(eR2_p90, 5),
        "implied_C2_from_mean_fit_x2": round(2 * eC_mean, 4),
        "falsifier": "if excess not approximately quadratic (low R2 for s^2), the O(dt^2) layer reading FAILS"},
    "first_substep_sign_crossing": {
        "n_periods_v0_le0_cross_at_s_le_0.01": n_first_cross,
        "n_periods_v0_le0_cross_anywhere": n_any_cross,
        "frac_first_substep": round(n_first_cross / max(n_periods, 1), 6),
        "frac_any_substep": round(n_any_cross / max(n_periods, 1), 6),
        "registered_prediction_if_dt2": "~ 2.45%/25 ~ 0.1%", "full_period_rate_ref": 0.0245}}

# ---------------- R9 ----------------
def clamp_box(u): return np.clip(u, 0.0, UMAX)
def classify(arm):
    S = arm["S"]; em = arm["em"]; uc = arm["uc"]; un = arm["un"]; ns = arm["n"]; T = uc.shape[1]
    idx = np.arange(T)[None, :]; active = idx < ns[:, None]
    m = em & active
    UC = uc[m]; UN = clamp_box(un[m])
    is_clamp_nom = np.all(np.isclose(UC, UN, atol=1e-3), axis=1)
    at_bound = np.isclose(UC, 0.0, atol=1e-3) | np.isclose(UC, UMAX, atol=1e-3)
    is_corner = np.all(at_bound, axis=1) & ~is_clamp_nom
    is_neither = ~is_clamp_nom & ~is_corner
    n = int(m.sum())
    # for (iii): does u_cmd vary with u_nom? correlation of ||uc|| with ||un|| on the neither set
    corr = None
    if is_neither.sum() > 5:
        a2 = np.linalg.norm(UC[is_neither], axis=1); b2 = np.linalg.norm(uc[m][is_neither], axis=1)  # placeholder
        corr = round(float(np.corrcoef(np.linalg.norm(UC[is_neither], axis=1), np.linalg.norm(un[m][is_neither], axis=1))[0, 1]), 4)
    control_instants = None
    if arm["dt"] < 0.05:   # Arm B/C: empty flag held across substeps; count distinct control instants (every 5th)
        sub = int(round(0.05 / arm["dt"]))
        ci = m[:, ::sub]
        control_instants = int(ci.sum())
    return {"n_empty_active_integration_steps": n,
            "total_active_integration_steps": int(active.sum()),
            "empty_fraction_of_active_steps": round(n / max(int(active.sum()), 1), 5),
            "distinct_empty_control_instants": control_instants,
            "case_i_clamped_nominal_gradient_flows": {"n": int(is_clamp_nom.sum()), "frac": round(float(is_clamp_nom.mean()), 5)},
            "case_ii_all_components_at_bound_corner_grad_zero": {"n": int(is_corner.sum()), "frac": round(float(is_corner.mean()), 5)},
            "case_iii_neither_edge_or_projection": {"n": int(is_neither.sum()), "frac": round(float(is_neither.mean()), 5),
                                                     "corr_ucmd_norm_vs_unom_norm": corr}}
R["R9_empty_gradient_classification"] = {
    "denominator_note": ("the empty flag is per INTEGRATION step (captured at the control instant and HELD "
                         "across the 5 substeps in Arms B/C); 70908/59256 are empty integration steps, and the "
                         "distinct empty control instants are ~1/5 of those"),
    "B_20Hz_fine": classify(B), "C_100Hz_fine": classify(C)}

# ---------------- R10 ----------------
sub = int(round(A["dt"] / B["dt"]))   # 5: A step k aligns with B step 5k (same physical time)
def sep_stats(group):
    rates, t015, jumpy = [], [], []
    frac_reach = 0; below1e3 = 0
    for i in group:
        kA = int(min(A["n"][i], A["S"].shape[1] - 1))
        kBmax = B["S"].shape[1] - 1
        K = min(kA, kBmax // sub)
        if K < 5: continue
        pA = A["S"][i, :K + 1, :3]; pB = B["S"][i, :K * sub + 1:sub, :3][:K + 1]
        dsep = np.linalg.norm(pA - pB, axis=1)                  # position separation at A's times
        t = np.arange(K + 1) * A["dt"]
        # pre-saturation window: from first step sep>1e-6 up to sep<0.15
        pos = np.nonzero((dsep > 1e-6) & (dsep < 0.15))[0]
        if dsep.max() < 1e-3: below1e3 += 1
        if pos.size >= 3:
            w = pos
            slope = np.polyfit(t[w], np.log(dsep[w]), 1)[0]     # log-linear growth rate (1/s)
            rates.append(float(slope))
            # smooth vs jump: max single-step ratio of dsep
            ratios = dsep[1:] / np.maximum(dsep[:-1], 1e-12)
            jumpy.append(float(np.max(ratios[np.isfinite(ratios)])) if ratios.size else 1.0)
        reach = np.nonzero(dsep >= 0.15)[0]
        if reach.size:
            frac_reach += 1; t015.append(float(reach[0] * A["dt"]))
    return {"n_scenes": len(group), "n_analyzed": len(rates),
            "median_growth_rate_per_s": round(float(np.median(rates)), 4) if rates else None,
            "growth_rate_iqr": [round(float(np.percentile(rates, 25)), 4), round(float(np.percentile(rates, 75)), 4)] if rates else None,
            "frac_reaching_0.15m": round(frac_reach / max(len(group), 1), 4),
            "median_time_to_0.15m_s": round(float(np.median(t015)), 4) if t015 else None,
            "n_sep_below_1e-3_throughout": below1e3,
            "median_max_single_step_ratio": round(float(np.median(jumpy)), 3) if jumpy else None}
agree = [i for i in ids if A["oc"][i] == B["oc"][i]]
disagree = [i for i in ids if A["oc"][i] != B["oc"][i]]
R["R10_sensitive_dependence"] = {
    "alignment": "A step k vs B step 5k (same physical time); position separation ||p_A - p_B||",
    "registered_prediction": "rate>0 and separation reaches obstacle scale (0.15 m) within the episode",
    "falsifier_null": "if separation stays below ~1e-3 m throughout, sensitive dependence cannot explain reassignment",
    "agree_scenes": sep_stats(agree), "disagree_scenes": sep_stats(disagree),
    "branch_switch_note": "max_single_step_ratio >> smooth growth per step indicates a branch-discrete O(1) jump, not chaos"}

(OUT.parent.parent / "v2.7.4/theory" ).mkdir(parents=True, exist_ok=True)
(Path("data/runs/v2.7.5/dt_ctrl_arms") / "r7r10_sampled_data.json").write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
