"""v2.2.0 Stage 2 — Diagnostics 2 & 3: stuck decomposition (with HOCBF contrast) + collision causes.

DIAGNOSTIC 2 (stuck): partition the N=2000 PNCBF-stuck into HOCBF-resolved (->goal) vs
HOCBF-also-stuck; for each, the single-step CBF+box escapable/blocked signature (PNCBF row) and the
HOCBF multi-row escapable test at the stall, plus trap-depth geometry. Output the three fractions:
filter-fixable / nominal-fixable / planning-hard. Also characterize the goal->stuck-under-HOCBF
regressions.

DIAGNOSTIC 3 (collision): attribute each PNCBF collision (23 deploy + 283 Stage-2 N=30000 NPZ) to a
dominant cause among {V_S value error, learned-invariance violation, singular/empty control
authority, discretization}, with the ground-truth danger = signed clearance ||p-c||-r (collision
<0). The learned-h CBF convention: safe={h<=0}, danger={h>0}; row A=L_g h, b=-L_f h-alpha*h,
cbf_violation_safe = A.u_safe - b (>0 = row violated). All recomputation is a forward pass on stored
states (verified bit-identical to the NPZ fields).

Reads the deploy traces cache (stage2_stuck_obs_grids.build_or_load_traces) and the N=30000 NPZs.
Read-only on checkpoints/pools/NPZs. Deterministic.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_stuck_collision_decomp.py
"""

from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage2_hocbf_deploy_n2000 as D  # noqa: E402
import stage2_stuck_obs_grids as G  # noqa: E402
from src.common.filter_hardnet import _base_alpha, _cbf_terms, _empty_halfspace_box, _hardnet_params  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402
from src.envs.scene_batch import batch_scenes  # noqa: E402
from src.envs.scene_init import Scene  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

OUT = D.OUT
LARGEN_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN/episodes"
U_MAX = 2.0
EPS_PROG = 0.1
STUCK_WINDOW = 60
SINGULAR_TOL = 5.0e-4
ROW_ENFORCED_TOL = 1.0e-2     # |cbf_violation_safe| <= this => CBF row satisfied on learned h
CORNERS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float) * U_MAX


def unit(v, axis=-1):
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9, None)


# ---- single-row CBF+box escapable test (matches stage2_stuck_mechanism) --------------

def best_goal_progress_singlerow(A, b, g):
    """max_{u in box, A.u<=b} g.u, per step. A,g [S,2], b [S]. nan if box∩halfspace empty."""
    S = A.shape[0]
    cands = [np.broadcast_to(c, (S, 2)).copy() for c in CORNERS]
    A0, A1 = A[:, 0], A[:, 1]
    for cx in (U_MAX, -U_MAX):
        uy = np.where(np.abs(A1) > 1e-12, (b - A0 * cx) / np.where(np.abs(A1) > 1e-12, A1, 1.0), np.nan)
        cands.append(np.stack([np.full(S, cx), uy], axis=1))
    for cy in (U_MAX, -U_MAX):
        ux = np.where(np.abs(A0) > 1e-12, (b - A1 * cy) / np.where(np.abs(A0) > 1e-12, A0, 1.0), np.nan)
        cands.append(np.stack([ux, np.full(S, cy)], axis=1))
    C = np.stack(cands, axis=1)
    Au = np.einsum("sd,skd->sk", A, C)
    ftol = 1e-4 * (1.0 + np.abs(b))[:, None]
    inbox = (np.abs(C[:, :, 0]) <= U_MAX + 1e-6) & (np.abs(C[:, :, 1]) <= U_MAX + 1e-6)
    feas = (Au <= b[:, None] + ftol) & inbox & np.isfinite(C).all(axis=2)
    val = np.where(feas, np.einsum("sd,skd->sk", g, C), -np.inf)
    best = val.max(axis=1)
    return np.where(np.isneginf(best), np.nan, best)


# ---- multi-row CBF+box escapable test (HOCBF rows + box) ------------------------------

def best_goal_progress_multirow(A, b, g):
    """max_{u in box, A_i.u<=b_i all i} g.u, per step. A [S,m,2], b [S,m], g [S,2]. nan if empty."""
    S, m, _ = A.shape
    boxA = np.broadcast_to(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float), (S, 4, 2))
    boxb = np.broadcast_to(np.array([U_MAX, U_MAX, U_MAX, U_MAX], float), (S, 4))
    AA = np.concatenate([A, boxA], axis=1)        # [S, m+4, 2]
    bb = np.concatenate([b, boxb], axis=1)
    M = m + 4
    cands = []
    for i, j in combinations(range(M), 2):
        a0, a1 = AA[:, i], AA[:, j]
        det = a0[:, 0] * a1[:, 1] - a0[:, 1] * a1[:, 0]
        ok = np.abs(det) > 1e-12
        dd = np.where(ok, det, 1.0)
        ux = (bb[:, i] * a1[:, 1] - bb[:, j] * a0[:, 1]) / dd
        uy = (a0[:, 0] * bb[:, j] - a1[:, 0] * bb[:, i]) / dd
        v = np.stack([ux, uy], axis=1)
        cands.append(np.where(ok[:, None], v, np.nan))
    C = np.stack(cands, axis=1)                    # [S,P,2]
    Au = np.einsum("smd,spd->spm", AA, C)
    feas = (Au <= bb[:, None, :] + 1e-6).all(axis=2) & np.isfinite(C).all(axis=2)
    val = np.where(feas, np.einsum("sd,spd->sp", g, C), -np.inf)
    best = val.max(axis=1)
    return np.where(np.isneginf(best), np.nan, best)


def hocbf_rows(states, scene, a1, a2, margin, k):
    """Pure-geometry HOCBF rows for the nearest active-k obstacles, per step. Returns A [S,k_eff,2], b."""
    p = states[:, :2]; v = states[:, 2:4]
    centers = np.asarray(scene.obstacle_centers, float); radii = np.asarray(scene.obstacle_radii, float)
    active = np.asarray(scene.obstacle_active, bool)
    rel = p[:, None, :] - centers[None, :, :]                      # [S,nmax,2]
    surf = np.linalg.norm(rel, axis=2) - radii[None, :]
    surf = np.where(active[None, :], surf, np.inf)
    keff = min(k, int(active.sum()))
    sel = np.argsort(surf, axis=1)[:, :keff]                       # [S,keff]
    si = np.arange(states.shape[0])[:, None]
    rel_k = rel[si, sel]                                           # [S,keff,2]
    radii_k = radii[sel]                                          # [S,keff]
    r_safe = radii_k + margin
    h0 = np.sum(rel_k * rel_k, axis=2) - r_safe * r_safe
    relv = np.sum(rel_k * v[:, None, :], axis=2)
    v2 = np.sum(v * v, axis=1, keepdims=True)
    psi1 = 2.0 * relv + a1 * h0
    A = -2.0 * rel_k
    b = 2.0 * v2 + 2.0 * a1 * relv + a2 * psi1
    return A, b


def episode_escapable(best_series, frac=0.5):
    fin = best_series[np.isfinite(best_series)]
    if fin.size == 0:
        return False, 0.0, 1.0
    esc = float(np.mean(best_series > EPS_PROG))                   # nan>0.1 -> False (blocked)
    empty = float(np.mean(~np.isfinite(best_series)))
    return esc >= frac, esc, empty


# ---- DIAGNOSTIC 2 --------------------------------------------------------------------

def diagnostic2(cache, scenes, fw, config, system, device, dtype):
    h_fn = make_h_fn(fw.value_net, system); params = _hardnet_params(config)
    p_idx = cache["pncbf_stuck_idx"]; p_states = cache["pncbf_stuck_states"]
    h_out = cache["hocbf_outcomes"]
    a1, a2, margin, k = D.HOCBF_A1, D.HOCBF_A2, D.R_MARGIN, D.K_DEPLOY

    recs = []
    for n in range(len(p_idx)):
        idx = int(p_idx[n]); st = p_states[n]; scene = scenes[idx]
        on = G.physical_onset(st)
        lo = on if on is not None else max(0, st.shape[0] - 1 - STUCK_WINDOW)
        hi = min(st.shape[0] - 1, lo + STUCK_WINDOW)
        phase = st[lo:hi]                                          # action-step states
        pos = phase[:, :2]; g = unit(np.asarray(scene.goal, float)[None, :] - pos)
        # PNCBF single-row escapable (recompute row on phase states)
        x = torch.as_tensor(phase, device=device, dtype=dtype)
        bscene = batch_scenes([scene] * x.shape[0], device=device, dtype=dtype)
        hh, lf, lg = _cbf_terms(system, h_fn, x, bscene, torch.zeros(x.shape[0], 2, device=device, dtype=dtype), create_graph=False)
        alpha = _base_alpha(hh, params); ru = (-lf - alpha * hh)
        A_p = lg.detach().cpu().numpy(); b_p = ru.detach().cpu().numpy()
        best_p = best_goal_progress_singlerow(A_p, b_p, g)
        esc_p, escfrac_p, _ = episode_escapable(best_p)
        # HOCBF multi-row escapable on SAME stuck-phase states (does a CBF-feasible progress exist?)
        A_h, b_h = hocbf_rows(phase, scene, a1, a2, margin, k)
        best_h = best_goal_progress_multirow(A_h, b_h, g)
        esc_h, escfrac_h, empty_h = episode_escapable(best_h)
        # geometry / trap depth
        geo = G.episode_geometry(st, scene)
        recs.append({"idx": idx, "hocbf_outcome": str(h_out[idx]),
                     "pncbf_row_escapable": bool(esc_p), "pncbf_escfrac": escfrac_p,
                     "hocbf_row_escapable": bool(esc_h), "hocbf_escfrac": escfrac_h, "hocbf_empty_frac": empty_h,
                     "final_dist_to_goal": geo["final_dist_to_goal"],
                     "clearance_at_stall": geo["clearance_at_stall"], "n_obs_within_R": geo["n_obs_within_R"]})

    resolved = [r for r in recs if r["hocbf_outcome"] == "goal"]
    also_stuck = [r for r in recs if r["hocbf_outcome"] == "stuck"]
    other = [r for r in recs if r["hocbf_outcome"] not in ("goal", "stuck")]
    n = len(recs)

    def stat(group, key):
        v = [r[key] for r in group]
        return {"n": len(group), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}

    # three fractions over PNCBF-stuck
    filter_fixable = len(resolved)
    planning_hard = sum(1 for r in also_stuck if not r["hocbf_row_escapable"])  # HOCBF also stuck AND myopically blocked
    nominal_fixable = sum(1 for r in also_stuck if r["hocbf_row_escapable"])    # also stuck but a progress dir exists
    # 'other' (HOCBF timeout etc.) folded by escapability
    planning_hard += sum(1 for r in other if not r["hocbf_row_escapable"])
    nominal_fixable += sum(1 for r in other if r["hocbf_row_escapable"])

    # goal -> stuck under HOCBF regressions
    p_out = cache["pncbf_outcomes"]; h_idx = cache["hocbf_stuck_idx"]; h_states = cache["hocbf_stuck_states"]
    h_pos_of = {int(h_idx[m]): m for m in range(len(h_idx))}
    reg = []
    for m in range(len(h_idx)):
        idx = int(h_idx[m])
        if str(p_out[idx]) != "goal":
            continue
        st = h_states[m]; scene = scenes[idx]
        on = G.physical_onset(st); lo = on if on is not None else max(0, st.shape[0] - 1 - STUCK_WINDOW)
        hi = min(st.shape[0] - 1, lo + STUCK_WINDOW); phase = st[lo:hi]
        pos = phase[:, :2]; g = unit(np.asarray(scene.goal, float)[None, :] - pos)
        A_h, b_h = hocbf_rows(phase, scene, D.HOCBF_A1, D.HOCBF_A2, D.R_MARGIN, D.K_DEPLOY)
        best_h = best_goal_progress_multirow(A_h, b_h, g)
        esc_h, escfrac_h, empty_h = episode_escapable(best_h)
        geo = G.episode_geometry(st, scene)
        reg.append({"idx": idx, "hocbf_row_escapable": bool(esc_h), "hocbf_escfrac": escfrac_h,
                    "hocbf_empty_frac": empty_h, "final_dist_to_goal": geo["final_dist_to_goal"],
                    "clearance_at_stall": geo["clearance_at_stall"], "n_obs_within_R": geo["n_obs_within_R"]})

    return {
        "n_pncbf_stuck": n,
        "partition": {"hocbf_resolved_goal": len(resolved), "hocbf_also_stuck": len(also_stuck),
                      "hocbf_other": len(other)},
        "pncbf_row_escapable_overall": float(np.mean([r["pncbf_row_escapable"] for r in recs])),
        "escapable_by_partition": {
            "resolved": {"pncbf_row_escapable_frac": float(np.mean([r["pncbf_row_escapable"] for r in resolved])) if resolved else None,
                         "hocbf_row_escapable_frac": float(np.mean([r["hocbf_row_escapable"] for r in resolved])) if resolved else None},
            "also_stuck": {"pncbf_row_escapable_frac": float(np.mean([r["pncbf_row_escapable"] for r in also_stuck])) if also_stuck else None,
                           "hocbf_row_escapable_frac": float(np.mean([r["hocbf_row_escapable"] for r in also_stuck])) if also_stuck else None}},
        "trap_depth_by_partition": {
            "resolved": {k2: stat(resolved, k2) for k2 in ("final_dist_to_goal", "clearance_at_stall", "n_obs_within_R")},
            "also_stuck": {k2: stat(also_stuck, k2) for k2 in ("final_dist_to_goal", "clearance_at_stall", "n_obs_within_R")}},
        "three_fractions_over_pncbf_stuck": {
            "filter_fixable_hocbf_resolves": filter_fixable, "filter_fixable_frac": filter_fixable / n,
            "nominal_fixable_alsostuck_but_myopically_escapable": nominal_fixable, "nominal_fixable_frac": nominal_fixable / n,
            "planning_hard_alsostuck_and_blocked": planning_hard, "planning_hard_frac": planning_hard / n},
        "goal_to_stuck_regressions": {
            "n": len(reg),
            "hocbf_row_escapable_frac": float(np.mean([r["hocbf_row_escapable"] for r in reg])) if reg else None,
            "blocked_planning_hard": sum(1 for r in reg if not r["hocbf_row_escapable"]),
            "escapable_overconservative": sum(1 for r in reg if r["hocbf_row_escapable"]),
            "trap_depth": {k2: stat(reg, k2) for k2 in ("final_dist_to_goal", "clearance_at_stall", "n_obs_within_R")}},
    }


# ---- DIAGNOSTIC 3 --------------------------------------------------------------------

def classify_collision(h, clearance, lg_norm, cvs, empty_box, saturated, h_dot_model, h_next, ev_action):
    """Dominant cause at the last actionable step before impact. Series are length-T arrays."""
    i = ev_action
    hi = float(h[i]); lgn = float(lg_norm[i]); cv = float(cvs[i])
    emp = bool(empty_box[i]); sat = bool(saturated[i]); sing = lgn < SINGULAR_TOL
    # alarm: clearance at which h first crosses 0 (learned safe-boundary location) + lead steps
    crossed = np.where(h[: i + 1] > 0.0)[0]
    if crossed.size:
        ac = int(crossed[0]); alarm_lead = i - ac; alarm_clear = float(clearance[ac])
    else:
        ac = -1; alarm_lead = 0; alarm_clear = float("nan")
    if hi <= 0.0:
        cause = "value_error"                              # V_S reports SAFE at last action
    elif sing:
        cause = "singular_gradient"                        # ||L_g h|| < 5e-4: filter has no authority
    elif emp:
        cause = "empty_authority"                          # required (alpha_unsafe) braking exceeds box-achievable
    elif cv <= ROW_ENFORCED_TOL:
        # row satisfied on learned h yet collided -> discretization vs invariance
        if hi <= 0.0 and float(h_next[i]) > 0.0:
            cause = "discretization"
        else:
            cause = "invariance_violation"
    else:
        cause = "other_unenforced"
    return {"cause": cause, "h": hi, "lg_norm": lgn, "cbf_violation_safe": cv, "empty_box": emp,
            "saturated": sat, "singular": sing, "alarm_lead": alarm_lead, "alarm_clearance": alarm_clear}


def diagnostic3_largeN():
    fs = sorted(glob.glob(str(LARGEN_DIR / "ep_*_collision.npz")))
    recs = []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        T = len(z["h"]); ev = int(md["event_step"]); ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        recs.append(classify_collision(
            np.asarray(z["h"], float), np.asarray(z["clearance"], float), np.asarray(z["lg_norm"], float),
            np.asarray(z["cbf_violation_safe"], float), np.asarray(z["empty_halfspace_box"]).astype(bool),
            np.asarray(z["saturated"]).astype(bool), np.asarray(z["h_dot_model"], float),
            np.asarray(z["h_next"], float), ea))
    return summarize_collisions(recs, "largeN_n30000")


def diagnostic3_deploy(cache, scenes, fw, config, system, device, dtype):
    h_fn = make_h_fn(fw.value_net, system); params = _hardnet_params(config)
    bounds = system.u_bounds.to(device=device, dtype=dtype)
    c_idx = cache["pncbf_col_idx"]; c_states = cache["pncbf_col_states"]; c_usafe = cache["pncbf_col_usafe"]
    p_ev = cache["pncbf_event"]
    recs = []
    for n in range(len(c_idx)):
        idx = int(c_idx[n]); scene = scenes[idx]
        ev = int(p_ev[idx]); T = c_states[n].shape[0] - 1
        ea = max(0, min(ev - 1, T - 1)) if ev > 0 else T - 1
        x = torch.as_tensor(c_states[n][:T], device=device, dtype=dtype)
        bscene = batch_scenes([scene] * T, device=device, dtype=dtype)
        hh, lf, lg = _cbf_terms(system, h_fn, x, bscene, torch.zeros(T, 2, device=device, dtype=dtype), create_graph=False)
        alpha = _base_alpha(hh, params); ru = -lf - alpha * hh
        lgn = torch.linalg.norm(lg, dim=1)
        usafe = torch.as_tensor(c_usafe[n][:T], device=device, dtype=dtype)
        cvs = (torch.sum(lg * usafe, dim=1) - ru)
        empty = _empty_halfspace_box(lg, ru, bounds)
        sat = (torch.abs(torch.abs(usafe) - U_MAX) <= 1e-3).any(dim=1)
        # h_next from h_fn on next states
        xn = torch.as_tensor(c_states[n][1:T + 1], device=device, dtype=dtype)
        hn = h_fn(xn, batch_scenes([scene] * T, device=device, dtype=dtype)).reshape(-1)
        # clearance per step from geometry
        pos = c_states[n][:T, :2]
        centers = np.asarray(scene.obstacle_centers, float); radii = np.asarray(scene.obstacle_radii, float)
        act = np.asarray(scene.obstacle_active, bool)
        surf = np.where(act[None, :], np.linalg.norm(pos[:, None, :] - centers[None, :, :], axis=2) - radii[None, :], np.inf)
        clearance = surf.min(axis=1)
        recs.append(classify_collision(
            hh.detach().cpu().numpy(), clearance, lgn.detach().cpu().numpy(), cvs.detach().cpu().numpy(),
            empty.detach().cpu().numpy().astype(bool), sat.detach().cpu().numpy().astype(bool),
            np.zeros(T), hn.detach().cpu().numpy(), ea))
    return summarize_collisions(recs, "deploy_n2000")


def summarize_collisions(recs, label):
    n = len(recs)
    causes = {}
    for r in recs:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    def med(key, pred=lambda r: True):
        v = [r[key] for r in recs if pred(r) and np.isfinite(r[key])]
        return float(np.median(v)) if v else None
    return {"label": label, "n": n,
            "cause_counts": causes, "cause_fracs": {k: v / n for k, v in causes.items()},
            "median_h_at_impact": med("h"), "median_lg_norm_at_impact": med("lg_norm"),
            "median_cbf_violation_safe": med("cbf_violation_safe"),
            "frac_empty_box_at_impact": float(np.mean([r["empty_box"] for r in recs])),
            "frac_singular_at_impact": float(np.mean([r["singular"] for r in recs])),
            "frac_saturated_at_impact": float(np.mean([r["saturated"] for r in recs])),
            "frac_v_s_says_safe": float(np.mean([r["h"] <= 0 for r in recs])),
            "median_alarm_lead_steps": med("alarm_lead"),
            "median_alarm_clearance": med("alarm_clearance"),
            "note": ("ground-truth danger = signed clearance ||p-c||-r (<0 collision); learned safe set "
                     "{h<=0}; alarm_clearance = clearance where h first crosses 0 (learned boundary "
                     "location); cbf_violation_safe = A.u_safe - row_upper (>0 row violated).")}


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    fw, config, _ = load_framework_from_checkpoint(D.CKPT)
    system = make_system(config)
    fw.value_net.to(device, dtype).eval(); fw.policy_net.to(device, dtype).eval()
    pool, _, _ = D.build_n2000_pool(config)
    scenes = pool.scenes

    cache = G.build_or_load_traces(fw, config, system, device, dtype)
    rep = {"hocbf_gains": [D.HOCBF_A1, D.HOCBF_A2], "r_margin": D.R_MARGIN, "k_deploy": D.K_DEPLOY,
           "eps_prog": EPS_PROG, "singular_tol": SINGULAR_TOL, "row_enforced_tol": ROW_ENFORCED_TOL}
    print("[diag2] stuck decomposition ...")
    rep["diagnostic2_stuck"] = diagnostic2(cache, scenes, fw, config, system, device, dtype)
    print("[diag3] collision causes (large-N N=30000 NPZs) ...")
    rep["diagnostic3_collision_largeN"] = diagnostic3_largeN()
    print("[diag3] collision causes (deploy N=2000, recomputed) ...")
    rep["diagnostic3_collision_deploy"] = diagnostic3_deploy(cache, scenes, fw, config, system, device, dtype)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stuck_collision_decomp_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("summary ->", OUT / "stuck_collision_decomp_summary.json")
    print("diag2 fractions:", rep["diagnostic2_stuck"]["three_fractions_over_pncbf_stuck"])
    print("diag3 largeN causes:", rep["diagnostic3_collision_largeN"]["cause_fracs"])
    print("diag3 deploy causes:", rep["diagnostic3_collision_deploy"]["cause_fracs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
