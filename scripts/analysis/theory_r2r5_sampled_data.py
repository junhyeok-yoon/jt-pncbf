"""v2.7.5 theory addendum R2-R5 — sampled-data reading of the dt axis (measurement only, no verdict change).

Reads regenerated per-step artifacts (states_{A,B,C}.npz = states+intervention+empty; action_stream_*.npz;
per_episode_*.csv) plus the flip ids from m5_posthoc_results.json. Deployed label-operator params read from the
run's metrics.csv @39000: lambda_disc=0.10662751890126089, target_rhs=0.9 (final), dt_train=0.05; alpha=2.0.
Single-seed scope. Falsifiers registered per item in the printed/JSON output BEFORE the numbers. Eval-only.
"""
from __future__ import annotations

import csv, json
from pathlib import Path

import numpy as np
import torch

from src.common.quadrotor_barrier import make_barrier_fn
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.eval.build_pools import load_pool, EVAL_POOLS_DIR
from src.envs.scene_batch import batch_scenes
from src.eval.run_full import _load_framework
from src.frameworks.oc_pncbf.value_target import pncbf_target

RD = Path("data/runs/v2.7.5/dt_ctrl_arms")
POOL = EVAL_POOLS_DIR / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
CKPT = Path("data/runs/v2.7.4/set__20260720-083533__seed42/v2.7.4__20260720-091830__seed42/checkpoints/best.pt")
OUT = Path("data/runs/v2.7.4/theory"); OUT.mkdir(parents=True, exist_ok=True)
LAMBDA_DISC, TARGET_RHS, DT_TRAIN, ALPHA = 0.10662751890126089, 0.9, 0.05, 2.0
OUTC = ["goal", "collision", "timeout", "stuck", "oob"]
RNG = np.random.default_rng(20260723)

fw, cfg, _ = _load_framework(CKPT)
system = fw.system
h_star_fn = make_barrier_fn(float(cfg["env"][system.name]["c_gain"]), float(cfg["env"]["h_scale"]))
v_fn = make_h_fn(fw.value_net, system)

def load_arm(a):
    z = np.load(RD / f"states_{a}.npz"); s = np.load(RD / f"action_stream_{a}.npz")
    rows = {int(r["episode_idx"]): r for r in csv.DictReader(open(RD / f"per_episode_{a}.csv"))}
    return {"S": z["states"], "em": z["empty"].astype(bool), "iv": z["intervention"].astype(bool),
            "dt": float(z["dt"]), "uc": s["u_cmd"].astype(np.float64), "un": s["u_nom"].astype(np.float64),
            "n": s["n_steps"].astype(int), "oc": {i: rows[i]["outcome"] for i in rows}}
A, B, C = load_arm("A_20Hz_coarse"), load_arm("B_20Hz_fine"), load_arm("C_100Hz_fine")
N = A["S"].shape[0]; ids = list(range(N))
scenes = load_pool(POOL).scenes
M5 = json.loads((RD / "m5_posthoc_results.json").read_text())
T = M5["T1_contingency_B_rows_C_cols"]
fwd = [i for i in ids if B["oc"][i] == "collision" and C["oc"][i] == "goal"]
rev = [i for i in ids if C["oc"][i] == "collision" and B["oc"][i] == "goal"]

def vhat_traj(arm, i, k):
    X = torch.tensor(arm["S"][i, :k + 1], dtype=torch.float32)
    b = batch_scenes([scenes[i]] * X.shape[0], device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        return v_fn(X, b).numpy().reshape(-1)

def q(a, p): return round(float(np.percentile(a, p)), 5) if len(a) else None
def dist(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "median": q(a, 50), "p90": q(a, 90), "max": q(a, 100), "iqr": [q(a, 25), q(a, 75)]} if a.size else {"n": 0}

R = {"params": {"lambda_disc": LAMBDA_DISC, "target_rhs": TARGET_RHS, "dt_train": DT_TRAIN, "alpha": ALPHA}}

# ---------------- R2: A-vs-B paired 5x5 contingency ----------------
tabAB = {ob: {oc: 0 for oc in OUTC} for ob in OUTC}
for i in ids: tabAB[A["oc"][i]][B["oc"][i]] += 1
Acoll = sum(1 for i in ids if A["oc"][i] == "collision"); Bcoll = sum(1 for i in ids if B["oc"][i] == "collision")
Aoob = sum(1 for i in ids if A["oc"][i] == "oob"); Boob = sum(1 for i in ids if B["oc"][i] == "oob")
Bnew_coll = [i for i in ids if A["oc"][i] != "collision" and B["oc"][i] == "collision"]
src_counts = {o: sum(1 for i in Bnew_coll if A["oc"][i] == o) for o in OUTC}
R["R2_A_vs_B_contingency"] = {
    "registered_question": "do the +collisions in B come predominantly from scenes A recorded as oob (detection effect)?",
    "falsifier": "A-oob->B-collision cell small and gains from A-goal/A-timeout instead -> detection reading FAILS",
    "table_A_rows_B_cols": tabAB, "A_collisions": Acoll, "B_collisions": Bcoll, "A_oob": Aoob, "B_oob": Boob,
    "n_B_new_collisions_A_noncollision": len(Bnew_coll), "source_of_B_new_collisions_by_A_outcome": src_counts,
    "A_oob_to_B_collision_cell": tabAB["oob"]["collision"]}

# ---------------- R3: segment-straddle ----------------
def seg_min_clearance(i, k):
    """min over consecutive OUTSIDE-OUTSIDE segments of A's xy path of (dist_to_axis - radius); None if no seg."""
    s = scenes[i]; cen = np.asarray(s.obstacle_centers, float)[:, :2]; rad = np.asarray(s.obstacle_radii, float)
    act = np.asarray(s.obstacle_active, bool)
    if not act.any(): return None
    cen, rad = cen[act], rad[act]
    P = A["S"][i, :k + 1, :2]
    outside = (np.linalg.norm(P[:, None, :] - cen[None], axis=2) >= rad[None]).all(axis=1)   # [k+1] outside all
    best = np.inf
    for t in range(P.shape[0] - 1):
        if not (outside[t] and outside[t + 1]): continue
        ts = np.linspace(0, 1, 21)[:, None]
        seg = P[t] * (1 - ts) + P[t + 1] * ts                       # [21,2]
        cl = (np.linalg.norm(seg[:, None, :] - cen[None], axis=2) - rad[None]).min()
        best = min(best, float(cl))
    return None if best == np.inf else best
straddle = [i for i in ids if A["oc"][i] != "collision" and B["oc"][i] == "collision"]
control_agree = [i for i in ids if A["oc"][i] == B["oc"][i] == "goal"]
control_samp = list(RNG.choice(control_agree, size=min(200, len(control_agree)), replace=False))
def seg_group(g):
    vals = [seg_min_clearance(i, int(min(A["n"][i], A["S"].shape[1] - 1))) for i in g]
    vals = [v for v in vals if v is not None]
    return {"n_scenes": len(g), "n_with_outside_outside_segment": len(vals),
            "min_clearance": dist(vals), "n_negative_straddle": int(sum(v < 0 for v in vals))}
# max per-step displacement at dt=0.05 vs radius distribution
disp = []
for i in ids:
    k = int(min(A["n"][i], A["S"].shape[1] - 1))
    if k >= 1: disp.append(float(np.linalg.norm(np.diff(A["S"][i, :k + 1, :2], axis=0), axis=1).max()))
radii_all = np.concatenate([np.asarray(s.obstacle_radii, float)[np.asarray(s.obstacle_active, bool)] for s in scenes])
R["R3_segment_straddle"] = {
    "registered": "RK4 h=0.05 error ~1e-5; a real detection gap must be geometrically visible as a straddle",
    "falsifier": "if straddle count ~0, the O(dt) detection gap is not the mechanism",
    "A_noncollision_B_collision": seg_group(straddle),
    "control_A_B_agree_goal_sample": seg_group(control_samp),
    "max_per_step_xy_displacement_dt0.05": dist(disp),
    "obstacle_radius_distribution": {"median": round(float(np.median(radii_all)), 4),
        "p05": round(float(np.percentile(radii_all, 5)), 4), "p95": round(float(np.percentile(radii_all, 95)), 4),
        "min": round(float(radii_all.min()), 4)}}

# ---------------- R4: Arm B inter-sample excursion ----------------
SUB = 5   # dt_ctrl 0.05 / dt_sim 0.01
exc = {s: [] for s in range(1, SUB + 1)}
n_violation_periods = 0; n_periods = 0
for i in ids:
    k = int(min(B["n"][i], B["S"].shape[1] - 1))
    if k < SUB: continue
    vh = vhat_traj(B, i, k)                             # V_hat at every dt_sim step
    for tk in range(0, k - SUB + 1, SUB):                          # control instants
        if B["em"][i, tk]: continue                                # feasible instants only (exclude empty)
        n_periods += 1
        v0 = vh[tk]
        sub_pos = False
        for s in range(1, SUB + 1):
            if tk + s <= k:
                exc[s].append(vh[tk + s] - v0)
                if vh[tk + s] > 0.0: sub_pos = True
        if v0 <= 0.0 and sub_pos: n_violation_periods += 1
# fit quadratic to the upper envelope (p90) vs s
s_arr = np.array([s * B["dt"] for s in range(1, SUB + 1)])
p90 = np.array([np.percentile(exc[s], 90) for s in range(1, SUB + 1)])
mx = np.array([np.max(exc[s]) for s in range(1, SUB + 1)])
# quadratic-through-origin fit y = C2 s^2 (envelope), and linear y = C1 s, compare R^2
def fit_through_origin(x, y, power):
    b = (x ** power); C = float(np.sum(b * y) / np.sum(b * b)); pred = C * b
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return C, (1 - ss_res / ss_tot if ss_tot > 0 else None)
C2, r2_quad = fit_through_origin(s_arr, p90, 2)
C1, r2_lin = fit_through_origin(s_arr, p90, 1)
def thickness(dt): return C2 * dt * dt / (2.0 * (1.0 - ALPHA * dt))
R["R4_intersample_excursion"] = {
    "registered_prediction": "upper envelope grows quadratically in s (O(dt^2) protected-layer)",
    "falsifier": "envelope closer to linear than quadratic in s",
    "arm": "B (dt_sim 0.01, dt_ctrl 0.05); FEASIBLE control instants only (empty excluded)",
    "n_control_periods": n_periods,
    "excursion_by_substep_s": {f"{s*B['dt']:.2f}": dist(exc[s]) for s in range(1, SUB + 1)},
    "envelope_p90_values": [round(float(v), 6) for v in p90],
    "quadratic_fit_C2": round(C2, 4), "quadratic_R2": round(r2_quad, 5) if r2_quad is not None else None,
    "linear_fit_C1": round(C1, 4), "linear_R2": round(r2_lin, 5) if r2_lin is not None else None,
    "protected_layer_thickness_C2dt2_over_2(1-alpha*dt)": {
        "dt_0.05": round(thickness(0.05), 6), "dt_0.01": round(thickness(0.01), 6)},
    "n_periods_with_intersample_violation_vk_le0_substep_gt0": n_violation_periods,
    "frac_violation_periods": round(n_violation_periods / max(n_periods, 1), 6)}

# ---------------- R5: one-step label residual ----------------
def label_residual(arm, group_ids, cap_steps=None):
    res, absv, surf_all = [], [], []
    for i in group_ids:
        k = int(min(arm["n"][i], arm["S"].shape[1] - 1))
        if k < 1: continue
        steps = range(k) if cap_steps is None else range(0, k, max(1, k // cap_steps))
        Xk = torch.tensor(arm["S"][i, list(steps)], dtype=torch.float32)
        Uk = torch.tensor(arm["uc"][i, list(steps)], dtype=torch.float32)
        b = batch_scenes([scenes[i]] * Xk.shape[0], device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            xnext = system.wrap_state(rk4_step(system, Xk, Uk, DT_TRAIN))          # one TRAINING step, deployed action
            hk = h_star_fn(Xk, b).numpy().reshape(-1)
            hn = h_star_fn(xnext, b).numpy().reshape(-1)
            vk = v_fn(Xk, b).numpy().reshape(-1)
            vn = v_fn(xnext, b).numpy().reshape(-1)
            # surface distance at x_k
            cen = np.asarray(scenes[i].obstacle_centers, float)[:, :2]; rad = np.asarray(scenes[i].obstacle_radii, float)
            act = np.asarray(scenes[i].obstacle_active, bool)
            P = arm["S"][i, list(steps), :2]
            sd = (np.linalg.norm(P[:, None] - cen[None, act], axis=2) - rad[None, act]).min(axis=1) if act.any() else np.full(P.shape[0], np.inf)
        for j in range(len(hk)):
            hseq = torch.tensor([[hk[j]], [hn[j]]], dtype=torch.float32)           # 2-step h_seq
            tail = torch.tensor([vn[j]], dtype=torch.float32)
            tgt = float(pncbf_target(hseq, LAMBDA_DISC, DT_TRAIN, TARGET_RHS, tail).reshape(-1)[0])
            res.append(abs(vk[j] - tgt)); absv.append(abs(vk[j])); surf_all.append(float(sd[j]))
    return np.array(res), np.array(absv), np.array(surf_all)
noflip = [i for i in ids if i not in set(fwd) | set(rev)]
noflip_s = list(RNG.choice(noflip, size=200, replace=False))
groups5 = {"forward_flip": fwd, "reverse_flip": rev, "no_flip": noflip_s}
r5 = {}; nullmodel = {}
for gname, g in groups5.items():
    rB, aB, sB = label_residual(B, g, cap_steps=60)
    rC, aC, sC = label_residual(C, g, cap_steps=60)
    r5[gname] = {"B": dist(rB), "C": dist(rC)}
    # null model on the COMBINED B+C residuals
    r_all = np.concatenate([rB, rC]); a_all = np.concatenate([aB, aC]); s_all = np.concatenate([sB, sC])
    finite = np.isfinite(s_all)
    nullmodel[gname] = {
        "corr_residual_vs_absVhat": round(float(np.corrcoef(r_all, a_all)[0, 1]), 4) if r_all.size > 2 else None,
        "corr_residual_vs_surfdist": round(float(np.corrcoef(r_all[finite], s_all[finite])[0, 1]), 4) if finite.sum() > 2 else None}
R["R5_label_residual"] = {
    "registered_prediction": "reverse-flip scenes carry systematically larger residual than forward-flip",
    "falsifier": "distributions overlap or ordering reverses",
    "null_model": "if residual is essentially f(|V_hat|) or f(surface distance) alone, test is UNINFORMATIVE",
    "operator": "one TRAINING step (dt=0.05) with the deployed filtered action u_cmd; T_A via pncbf_target",
    "by_group": r5, "null_model_correlations": nullmodel,
    "no_flip_sample_n": len(noflip_s)}

(OUT / "r2r5_sampled_data.json").write_text(json.dumps(R, indent=2) + "\n")
print(json.dumps(R, indent=2))
