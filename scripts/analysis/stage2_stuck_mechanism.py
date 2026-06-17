"""v2.2.0 Stage 2 — stuck dynamics diagnosis: projection mechanism, escapable vs blocked (read-only).

For the near-obstacle-projection-saturation stuck trap: (A) what the HardNet projection does at the
stuck states, (B) whether a CBF+box-feasible goal-progress action EXISTS (escapable) or not
(blocked), (C) for escapable, the feasible direction's geometry and the algebraic recovered
goal-progress if the nominal were re-aimed along it. Pure algebra on the stored CBF row
(A = lg_h, b = row_upper) and the verified u_max; no env, no net, no Fisher. Read-only;
deterministic.

DI dynamics used (verified): u = acceleration, per-component box |u_i| <= u_max = 2.0; the CBF
row is A·u <= b with A = L_g h = [dh/dvx, dh/dvy], b = -L_f h - alpha*h (alpha_safe=2, dt=0.05).
Feasibility test (single-step, on the stored row at the visited stuck state): does there exist
u in [-2,2]^2 with A·u <= b and goal-directed acceleration g_hat·u > eps_prog? Solved exactly as a
2-variable LP by vertex enumeration of {box} ∩ {A·u <= b}. Justification: at a near-stationary
stuck state, a CBF+box-feasible goal-directed acceleration is what would START progress; the stored
A,b are exactly the constraint the filter applied there, so this is exact for the visited state (a
necessary local condition — infeasible ⇒ definitely trapped at that state; feasible ⇒ the filter
could accelerate goalward but isn't). It is a one-step local test, not a closed-loop guarantee.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_stuck_mechanism.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN"
EPISODE_DIR = OUT / "episodes"
LABELS = OUT / "stage2_failure_labels.csv"

U_MAX = 2.0
HN_EPS = 5.0e-4
STUCK_WINDOW = 60
STUCK_RADIUS = 0.10
EPS_PROG = 0.1                 # goal-directed accel deemed "progress" (report distribution too)
NOM_COLLAPSE = 0.25           # |u_nom| below this = policy proposes ~no action
SLACK_TOL = 1e-3              # CBF row considered binding/tight if (b - A.u_safe) <= this*(1+|b|)
DOMINANT = {"near-obstacle-projection-saturation-trap", "overconservative-projection-saturation-trap"}
CORNERS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float) * U_MAX


def load_stuck():
    labels = {int(r["episode_idx"]): r for r in csv.DictReader(LABELS.open())} if LABELS.exists() else {}
    eps = []
    for path in sorted(EPISODE_DIR.glob("ep_*_stuck.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        idx = int(path.stem.split("_")[1])
        eps.append({"idx": idx, "active_steps": int(md.get("active_steps", len(z["h"]))),
                    "n": int(len(z["h"])),
                    "lg_h": np.asarray(z["lg_h"], float), "row_upper": np.asarray(z["row_upper"], float),
                    "row_lhs_safe": np.asarray(z["row_lhs_safe"], float),
                    "u_nom": np.asarray(z["u_nom"], float), "u_safe": np.asarray(z["u_safe"], float),
                    "goal_rel_pos": np.asarray(z["goal_rel_pos"], float),
                    "nearest_rel_pos": np.asarray(z["nearest_rel_pos"], float),
                    "states": np.asarray(z["states"], float),
                    "saturated": np.asarray(z["saturated"], bool),
                    "projection_norm": np.asarray(z["projection_norm"], float),
                    "clearance": np.asarray(z["clearance"], float),
                    "mechanism": labels.get(idx, {}).get("mechanism", "n/a"),
                    "layer": labels.get(idx, {}).get("layer", "n/a")})
    return eps


def physical_onset(states):
    pos = states[:, :2]; T = pos.shape[0]
    for t in range(STUCK_WINDOW, T):
        if np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - pos[t - STUCK_WINDOW], axis=1).max() <= STUCK_RADIUS:
            return max(0, t - STUCK_WINDOW)
    return None


def stuck_steps(e):
    on = physical_onset(e["states"])
    hi = min(e["active_steps"], e["n"])
    if on is None:
        on = max(0, hi - STUCK_WINDOW)
    lo = max(0, on)
    hi = min(hi, on + STUCK_WINDOW)
    return np.arange(lo, max(lo + 1, hi))


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def best_goal_progress(A, b, g):
    """max over u in [-2,2]^2 with A.u<=b of (g.u); vectorized over rows. Returns (best_val[M], best_u[M,2])."""
    M = A.shape[0]
    cands = [np.broadcast_to(c, (M, 2)).copy() for c in CORNERS]   # 4 corners
    A0, A1 = A[:, 0], A[:, 1]
    for cx in (U_MAX, -U_MAX):                                     # x-edges: solve uy
        uy = np.where(np.abs(A1) > 1e-12, (b - A0 * cx) / np.where(np.abs(A1) > 1e-12, A1, 1.0), np.nan)
        cands.append(np.stack([np.full(M, cx), uy], axis=1))
    for cy in (U_MAX, -U_MAX):                                     # y-edges: solve ux
        ux = np.where(np.abs(A0) > 1e-12, (b - A1 * cy) / np.where(np.abs(A0) > 1e-12, A0, 1.0), np.nan)
        cands.append(np.stack([ux, np.full(M, cy)], axis=1))
    C = np.stack(cands, axis=1)                                    # [M,8,2]
    Au = np.einsum("md,mkd->mk", A, C)
    ftol = 1e-4 * (1.0 + np.abs(b))[:, None]
    inbox = (np.abs(C[:, :, 0]) <= U_MAX + 1e-6) & (np.abs(C[:, :, 1]) <= U_MAX + 1e-6)
    feasible = (Au <= b[:, None] + ftol) & inbox & np.isfinite(C).all(axis=2)
    val = np.einsum("md,mkd->mk", g, C)
    val = np.where(feasible, val, -np.inf)
    k = np.argmax(val, axis=1)
    best = val[np.arange(M), k]
    best_u = C[np.arange(M), k]
    best = np.where(np.isneginf(best), np.nan, best)              # nan = CBF-box empty (no feasible u)
    return best, best_u


def base_project(unom, A, b):
    lhs = np.einsum("md,md->m", A, unom)
    viol = np.maximum(lhs - b, 0.0)
    denom = np.einsum("md,md->m", A, A) + HN_EPS ** 2
    u = unom - A * (viol / denom)[:, None]
    return np.clip(u, -U_MAX, U_MAX)


def main():
    eps = load_stuck()
    dom = [e for e in eps if e["mechanism"] in DOMINANT or e["layer"] == "planning-limit-geometric-trap"]
    rep = {"n_stuck": len(eps), "n_dominant_subset": len(dom),
           "mechanism_counts": _count(eps, "mechanism"), "layer_counts": _count(eps, "layer"),
           "config": {"u_max": U_MAX, "alpha_safe": 2.0, "dt": 0.05, "hardnet_eps": HN_EPS},
           "eps_prog": EPS_PROG}

    # accumulate step-level over the dominant subset
    A_all, b_all, g_all, vsafe_all, vnom_all, sat_all, proj_all, slack_all, vel_all, near_all = ([] for _ in range(10))
    ep_records = []
    for e in dom:
        idx = stuck_steps(e)
        A = e["lg_h"][idx]; b = e["row_upper"][idx]
        g = unit(e["goal_rel_pos"][idx])
        un = e["u_nom"][idx]; us = e["u_safe"][idx]
        v = e["states"][idx, 2:4]
        near = unit(e["nearest_rel_pos"][idx])      # outward from obstacle (agent - center)
        slack = b - e["row_lhs_safe"][idx]
        best, best_u = best_goal_progress(A, b, g)
        escap = best > EPS_PROG
        gp_nom = np.einsum("md,md->m", un, g); gp_safe = np.einsum("md,md->m", us, g)
        ep_records.append({
            "idx": e["idx"], "mechanism": e["mechanism"], "layer": e["layer"], "nsteps": len(idx),
            "frac_escapable": float(np.mean(escap)),
            "best_progress_median": float(np.nanmedian(best)),
            "gp_safe_median": float(np.median(gp_safe)), "gp_nom_median": float(np.median(gp_nom)),
            "best_u": best_u, "g": g, "near": near, "escap": escap, "best": best,
            "A": A, "b": b})
        A_all.append(A); b_all.append(b); g_all.append(g); vsafe_all.append(us); vnom_all.append(un)
        sat_all.append(e["saturated"][idx]); proj_all.append(e["projection_norm"][idx])
        slack_all.append(slack); vel_all.append(v); near_all.append(near)

    A_all = np.concatenate(A_all); b_all = np.concatenate(b_all); g_all = np.concatenate(g_all)
    us = np.concatenate(vsafe_all); un = np.concatenate(vnom_all)
    sat = np.concatenate(sat_all); proj = np.concatenate(proj_all); slack = np.concatenate(slack_all)
    vel = np.concatenate(vel_all); near = np.concatenate(near_all)
    gp_nom = np.einsum("md,md->m", un, g_all); gp_safe = np.einsum("md,md->m", us, g_all)
    nrm_nom = np.linalg.norm(un, axis=1); nrm_safe = np.linalg.norm(us, axis=1)
    ang = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", unit(un), unit(us)), -1, 1)))
    speed = np.linalg.norm(vel, axis=1)
    brake = np.einsum("md,md->m", us, -unit(vel))     # >0 => opposing velocity (braking)

    # Part A — step-level mechanism categorization
    tight = slack <= SLACK_TOL * (1.0 + np.abs(b_all))
    collapse = nrm_nom < NOM_COLLAPSE
    cat = np.full(slack.shape, "other", dtype=object)
    cat[collapse] = "nominal_collapse"
    cat[(~collapse) & tight] = "cbf_binding"
    cat[(~collapse) & (~tight) & sat] = "box_saturation_only"
    rep["partA_mechanism"] = {
        "n_steps": int(slack.size),
        "angle_unom_usafe_median_deg": float(np.median(ang)),
        "u_nom_norm_median": float(np.median(nrm_nom)), "u_safe_norm_median": float(np.median(nrm_safe)),
        "gp_nom_median": float(np.median(gp_nom)), "gp_safe_median": float(np.median(gp_safe)),
        "frac_gp_nom_pos": float(np.mean(gp_nom > EPS_PROG)), "frac_gp_safe_pos": float(np.mean(gp_safe > EPS_PROG)),
        "saturated_frac": float(np.mean(sat)), "projection_norm_median": float(np.median(proj)),
        "speed_median": float(np.median(speed)), "brake_component_median": float(np.median(brake)),
        "cbf_tight_frac": float(np.mean(tight)),
        "category_fractions": {c: float(np.mean(cat == c)) for c in
                               ("cbf_binding", "nominal_collapse", "box_saturation_only", "other")}}

    # Part B — escapable vs blocked (episode-level: majority of stuck steps escapable)
    fr = np.array([r["frac_escapable"] for r in ep_records])
    escapable_ep = fr >= 0.5
    rep["partB_escapable"] = {
        "n_episodes": len(ep_records),
        "frac_escapable_steps_overall": float(np.mean(np.concatenate([r["escap"] for r in ep_records]))),
        "episode_escapable_majority": int(escapable_ep.sum()), "episode_blocked_majority": int((~escapable_ep).sum()),
        "episode_escapable_frac": float(np.mean(escapable_ep)),
        "best_progress_step_quantiles": _q(np.concatenate([r["best"] for r in ep_records])),
        "frac_steps_best_gt": {thr: float(np.mean(np.concatenate([r["best"] for r in ep_records]) > thr))
                               for thr in (0.1, 0.5, 1.0)},
        "escapable_by_layer": _split_count(ep_records, escapable_ep, "layer"),
        "escapable_by_mechanism": _split_count(ep_records, escapable_ep, "mechanism")}

    # Part C — feasible-direction geometry + re-aim recovered progress (escapable steps)
    esc_mask = np.concatenate([r["escap"] for r in ep_records])
    best_u_all = np.concatenate([r["best_u"] for r in ep_records])[esc_mask]
    g_esc = g_all[esc_mask]; near_esc = near[esc_mask]; A_esc = A_all[esc_mask]; b_esc = b_all[esc_mask]
    bu = unit(best_u_all)
    tang = np.stack([-near_esc[:, 1], near_esc[:, 0]], axis=1)     # tangent (perp to outward)
    ang_goal = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", bu, g_esc), -1, 1)))
    ang_tan = np.degrees(np.arccos(np.clip(np.abs(np.einsum("md,md->m", bu, tang)), -1, 1)))  # to tangent line
    ang_out = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", bu, near_esc), -1, 1)))
    # re-aim algebra: project a nominal aimed at goal vs aimed at feasible direction
    gp_proj_goal = np.einsum("md,md->m", base_project(U_MAX * g_esc, A_esc, b_esc), g_esc)
    gp_proj_feas = np.einsum("md,md->m", base_project(U_MAX * bu, A_esc, b_esc), g_esc)
    rep["partC_escape_geometry"] = {
        "n_escapable_steps": int(esc_mask.sum()),
        "best_u_angle_to_goal_median_deg": float(np.median(ang_goal)),
        "best_u_angle_to_obstacle_tangent_median_deg": float(np.median(ang_tan)),
        "best_u_angle_to_obstacle_outward_median_deg": float(np.median(ang_out)),
        "frac_more_tangential_than_goalward": float(np.mean(ang_tan < ang_goal)),
        "reaim_goal_projected_gp_median": float(np.median(gp_proj_goal)),
        "reaim_feasible_projected_gp_median": float(np.median(gp_proj_feas)),
        "actual_gp_safe_median_escsteps": float(np.median(gp_safe[esc_mask])),
        "note": "open-loop algebraic estimate on stored A,b; not a closed-loop guarantee."}

    # Part C.3 — blocked confirmation
    blk_mask = ~esc_mask
    if blk_mask.any():
        Ag = np.einsum("md,md->m", A_all[blk_mask], g_all[blk_mask])
        empty = np.isnan(np.concatenate([r["best"] for r in ep_records]))[blk_mask]
        rep["partC_blocked"] = {
            "n_blocked_steps": int(blk_mask.sum()),
            "frac_goaldir_in_forbidden_halfspace_Ag_gt0": float(np.mean(Ag > 0)),
            "A_dot_ghat_median": float(np.median(Ag)), "b_median": float(np.median(b_all[blk_mask])),
            "frac_cbf_box_empty": float(np.mean(empty))}

    (OUT / "stage2_stuck_mechanism_summary.json").write_text(json.dumps(_jsonable(rep), indent=2), encoding="utf-8")
    _figs(ang, gp_nom, gp_safe, fr, np.concatenate([r["best"] for r in ep_records]), ang_goal, ang_tan,
          gp_proj_goal, gp_proj_feas)
    _print(rep)
    return 0


def _count(eps, key):
    c = {}
    for e in eps:
        c[e[key]] = c.get(e[key], 0) + 1
    return c


def _split_count(recs, mask, key):
    out = {}
    for r, m in zip(recs, mask):
        out.setdefault(r[key], [0, 0])
        out[r[key]][0 if m else 1] += 1
    return {k: {"escapable": v[0], "blocked": v[1]} for k, v in out.items()}


def _q(a):
    a = a[np.isfinite(a)]
    return {"p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90))} if a.size else {}


def _figs(ang, gp_nom, gp_safe, fr, best, ang_goal, ang_tan, gp_pg, gp_pf):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)
    ax[0].hist(ang, bins=36, color="#1f77b4"); ax[0].set_title("angle(u_nom, u_safe) deg"); ax[0].set_xlabel("deg")
    ax[1].hist(gp_nom, bins=40, alpha=0.6, label="u_nom·ĝ", color="#2ca02c")
    ax[1].hist(gp_safe, bins=40, alpha=0.6, label="u_safe·ĝ", color="#d62728")
    ax[1].axvline(0, color="0.3", lw=1); ax[1].set_title("goal-directed acceleration"); ax[1].legend(fontsize=8, frameon=False)
    ax[2].hist(fr, bins=20, color="#7f7f7f"); ax[2].axvline(0.5, color="r", ls="--", lw=1)
    ax[2].set_title("per-episode frac stuck-steps escapable"); ax[2].set_xlabel("fraction")
    fig.tight_layout(); fig.savefig(OUT / "stage2_stuck_partAB.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)
    ax[0].hist(best[np.isfinite(best)], bins=40, color="#4c78a8"); ax[0].axvline(EPS_PROG, color="r", ls="--", lw=1)
    ax[0].set_title("max CBF+box-feasible goal accel (best_progress)"); ax[0].set_xlabel("g·u")
    ax[1].hist(ang_goal, bins=36, alpha=0.6, label="to goal", color="#2ca02c")
    ax[1].hist(ang_tan, bins=36, alpha=0.6, label="to obstacle tangent", color="#9467bd")
    ax[1].set_title("feasible direction geometry (deg)"); ax[1].legend(fontsize=8, frameon=False)
    ax[2].hist(gp_pg, bins=40, alpha=0.6, label="re-aim goal→proj·ĝ", color="#d62728")
    ax[2].hist(gp_pf, bins=40, alpha=0.6, label="re-aim feasible→proj·ĝ", color="#2ca02c")
    ax[2].axvline(0, color="0.3", lw=1); ax[2].set_title("re-aimed nominal recovered progress"); ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "stage2_stuck_partC.png"); plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _print(rep):
    print("n_stuck:", rep["n_stuck"], "dominant subset:", rep["n_dominant_subset"])
    a = rep["partA_mechanism"]
    print(f"\nPart A (n_steps={a['n_steps']}): angle(unom,usafe) med={a['angle_unom_usafe_median_deg']:.1f}deg "
          f"|unom|={a['u_nom_norm_median']:.2f} |usafe|={a['u_safe_norm_median']:.2f}")
    print(f"  gp_nom med={a['gp_nom_median']:.3f} (frac>0 {a['frac_gp_nom_pos']:.3f})  "
          f"gp_safe med={a['gp_safe_median']:.3f} (frac>0 {a['frac_gp_safe_pos']:.3f})")
    print(f"  saturated={a['saturated_frac']:.3f} proj_norm med={a['projection_norm_median']:.2f} "
          f"speed med={a['speed_median']:.3f} brake-comp med={a['brake_component_median']:.3f} cbf_tight={a['cbf_tight_frac']:.3f}")
    print(f"  categories: {a['category_fractions']}")
    b = rep["partB_escapable"]
    print(f"\nPart B: escapable steps overall={b['frac_escapable_steps_overall']:.3f}; "
          f"episodes escapable(>=50% steps)={b['episode_escapable_majority']}/{b['n_episodes']} "
          f"({b['episode_escapable_frac']:.3f}) blocked={b['episode_blocked_majority']}")
    print(f"  best_progress step quantiles: {b['best_progress_step_quantiles']}  frac steps best>: {b['frac_steps_best_gt']}")
    print(f"  by layer: {b['escapable_by_layer']}")
    c = rep["partC_escape_geometry"]
    print(f"\nPart C: feasible-dir angle-to-goal med={c['best_u_angle_to_goal_median_deg']:.1f} "
          f"angle-to-tangent med={c['best_u_angle_to_obstacle_tangent_median_deg']:.1f} "
          f"more-tangential-than-goalward={c['frac_more_tangential_than_goalward']:.3f}")
    print(f"  re-aim recovered gp: goal-aim proj={c['reaim_goal_projected_gp_median']:.3f}  "
          f"feasible-aim proj={c['reaim_feasible_projected_gp_median']:.3f}  actual usafe·ĝ={c['actual_gp_safe_median_escsteps']:.3f}")
    if "partC_blocked" in rep:
        bl = rep["partC_blocked"]
        print(f"\nPart C blocked: n={bl['n_blocked_steps']} frac A·ĝ>0={bl['frac_goaldir_in_forbidden_halfspace_Ag_gt0']:.3f} "
              f"A·ĝ med={bl['A_dot_ghat_median']:.3f} cbf-box-empty={bl['frac_cbf_box_empty']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
