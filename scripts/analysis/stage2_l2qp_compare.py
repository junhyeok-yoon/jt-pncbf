"""v2.2.0 Stage 2 — HardNet u_safe vs true joint L2-QP at stuck states (read-only).

Tests whether the HardNet projection (differentiable closed form + box-aware enumeration) differs
from the true Euclidean CBF-QP `argmin ||u-u_nom||_2 s.t. A u <= b, |u|<=u_max` at the stored stuck
states. Pure algebra on stored (u_nom, A=lg_h, b=row_upper) + an exact 2-var QP solve; no env, no
net, no Fisher. Read-only; deterministic.

The L2-QP is solved exactly by KKT active-set enumeration over the finite candidate set (covers all
active sets of a 2-var box + 1-halfspace QP): unconstrained u_nom; halfspace-line projection;
single-box-edge perpendicular foot; box corners; halfspace∩box-edge vertices. The exact solver is
cross-checked against cvxpy/OSQP on a random sample (KKT/optimality verified).

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_l2qp_compare.py
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
STUCK_WINDOW = 60
STUCK_RADIUS = 0.10
EPS_PROG = 0.1
FTOL = 1e-7
DOMINANT = {"near-obstacle-projection-saturation-trap", "overconservative-projection-saturation-trap"}
VERIFY_SEED = 20260616
VERIFY_N = 200


def load_stuck():
    labels = {int(r["episode_idx"]): r for r in csv.DictReader(LABELS.open())} if LABELS.exists() else {}
    eps = []
    for path in sorted(EPISODE_DIR.glob("ep_*_stuck.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        idx = int(path.stem.split("_")[1])
        lab = labels.get(idx, {})
        if not (lab.get("mechanism") in DOMINANT or lab.get("layer") == "planning-limit-geometric-trap"):
            continue
        eps.append({"idx": idx, "active_steps": int(md.get("active_steps", len(z["h"]))), "n": int(len(z["h"])),
                    "lg_h": np.asarray(z["lg_h"], float), "row_upper": np.asarray(z["row_upper"], float),
                    "u_nom": np.asarray(z["u_nom"], float), "u_safe": np.asarray(z["u_safe"], float),
                    "goal_rel_pos": np.asarray(z["goal_rel_pos"], float), "states": np.asarray(z["states"], float)})
    return eps


def physical_onset(states):
    pos = states[:, :2]; T = pos.shape[0]
    for t in range(STUCK_WINDOW, T):
        if np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - pos[t - STUCK_WINDOW], axis=1).max() <= STUCK_RADIUS:
            return max(0, t - STUCK_WINDOW)
    return None


def stuck_idx(e):
    on = physical_onset(e["states"]); hi = min(e["active_steps"], e["n"])
    if on is None:
        on = max(0, hi - STUCK_WINDOW)
    return np.arange(max(0, on), min(hi, on + STUCK_WINDOW))


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def l2qp_exact(unom, A, b):
    """Exact 2-var QP: argmin||u-unom||^2 s.t. A.u<=b, |u|<=U_MAX. Vectorized. Returns (u[M,2], feasible[M])."""
    M = unom.shape[0]
    A0, A1 = A[:, 0], A[:, 1]
    nA2 = A0 * A0 + A1 * A1
    cands = []
    cands.append(np.clip(unom, -U_MAX, U_MAX))                                   # box proj of nominal
    proj = unom - A * ((np.einsum("md,md->m", A, unom) - b) / np.clip(nA2, 1e-12, None))[:, None]
    cands.append(np.clip(proj, -U_MAX, U_MAX))                                   # halfspace-line proj, clipped
    # single-box-edge perpendicular foot (clip one comp, keep other = nominal)
    for i, val in ((0, U_MAX), (0, -U_MAX), (1, U_MAX), (1, -U_MAX)):
        c = unom.copy(); c[:, i] = val; cands.append(np.clip(c, -U_MAX, U_MAX))
    for cx in (U_MAX, -U_MAX):
        for cy in (U_MAX, -U_MAX):
            cands.append(np.broadcast_to(np.array([cx, cy], float), (M, 2)).copy())   # corners
    for cx in (U_MAX, -U_MAX):                                                   # halfspace ∩ x-edge
        uy = np.where(np.abs(A1) > 1e-12, (b - A0 * cx) / np.where(np.abs(A1) > 1e-12, A1, 1.0), np.nan)
        cands.append(np.stack([np.full(M, cx), uy], axis=1))
    for cy in (U_MAX, -U_MAX):                                                   # halfspace ∩ y-edge
        ux = np.where(np.abs(A0) > 1e-12, (b - A1 * cy) / np.where(np.abs(A0) > 1e-12, A0, 1.0), np.nan)
        cands.append(np.stack([ux, np.full(M, cy)], axis=1))
    C = np.stack(cands, axis=1)                                                  # [M,K,2]
    Au = np.einsum("md,mkd->mk", A, C)
    feas = (Au <= b[:, None] + FTOL * (1 + np.abs(b)[:, None])) & \
           (np.abs(C[:, :, 0]) <= U_MAX + 1e-6) & (np.abs(C[:, :, 1]) <= U_MAX + 1e-6) & np.isfinite(C).all(2)
    dist = np.sum((C - unom[:, None, :]) ** 2, axis=2)
    any_feas = feas.any(axis=1)
    dist_f = np.where(feas, dist, np.inf)
    k = np.argmin(dist_f, axis=1)
    u = C[np.arange(M), k]
    # infeasible rows: least-violating (min relu(A.u-b)) over box candidates
    if (~any_feas).any():
        viol = np.maximum(Au - b[:, None], 0.0)
        kk = np.argmin(np.where(np.isfinite(C).all(2) & (np.abs(C[:, :, 0]) <= U_MAX + 1e-6) &
                                (np.abs(C[:, :, 1]) <= U_MAX + 1e-6), viol, np.inf), axis=1)
        u[~any_feas] = C[np.arange(M), kk][~any_feas]
    return u, any_feas


def verify_with_cvxpy(unom, A, b, feasible):
    try:
        import cvxpy as cp
    except Exception:
        return {"cvxpy": "unavailable"}
    rng = np.random.default_rng(VERIFY_SEED)
    idx = rng.choice(np.where(feasible)[0], size=min(VERIFY_N, int(feasible.sum())), replace=False)
    u_mine, _ = l2qp_exact(unom[idx], A[idx], b[idx])
    maxdiff = 0.0
    for j, i in enumerate(idx):
        u = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - unom[i])),
                          [A[i] @ u <= b[i], u >= -U_MAX, u <= U_MAX])
        prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10)
        maxdiff = max(maxdiff, float(np.linalg.norm(u.value - u_mine[j])))
    return {"cvxpy": "ok", "n_verified": int(len(idx)), "max_abs_diff_exact_vs_cvxpy": maxdiff}


def main():
    eps = load_stuck()
    rows_u, rows_A, rows_b, rows_us, rows_g = [], [], [], [], []
    for e in eps:
        idx = stuck_idx(e)
        rows_u.append(e["u_nom"][idx]); rows_A.append(e["lg_h"][idx]); rows_b.append(e["row_upper"][idx])
        rows_us.append(e["u_safe"][idx]); rows_g.append(unit(e["goal_rel_pos"][idx]))
    unom = np.concatenate(rows_u); A = np.concatenate(rows_A); b = np.concatenate(rows_b)
    usafe = np.concatenate(rows_us); g = np.concatenate(rows_g)
    M = unom.shape[0]

    u_qp, feasible = l2qp_exact(unom, A, b)
    verify = verify_with_cvxpy(unom, A, b, feasible)

    diff = np.linalg.norm(u_qp - usafe, axis=1)
    ang_qp_hn = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", unit(u_qp), unit(usafe)), -1, 1)))
    gp_qp = np.einsum("md,md->m", u_qp, g); gp_hn = np.einsum("md,md->m", usafe, g)
    dnom_qp = np.linalg.norm(u_qp - unom, axis=1); dnom_hn = np.linalg.norm(usafe - unom, axis=1)
    rot_qp = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", unit(unom), unit(u_qp)), -1, 1)))
    rot_hn = np.degrees(np.arccos(np.clip(np.einsum("md,md->m", unit(unom), unit(usafe)), -1, 1)))

    # escapable / blocked split (single-step feasible goal progress > 0.1, on the QP-feasible region)
    # best feasible goal progress is from the QP-feasible candidate set; reuse: a step is escapable if
    # max feasible g.u > EPS_PROG. Approximate by gp at the goal-LP optimum:
    from numpy import inf  # noqa
    escap = np.array([_best_progress(A[i], b[i], g[i]) > EPS_PROG for i in range(M)])

    def stats(x, m=None):
        x = x if m is None else x[m]
        x = x[np.isfinite(x)]
        return {"n": int(x.size), "median": float(np.median(x)) if x.size else None,
                "p25": float(np.percentile(x, 25)) if x.size else None,
                "p75": float(np.percentile(x, 75)) if x.size else None,
                "mean": float(np.mean(x)) if x.size else None}

    rep = {
        "n_steps": M, "n_episodes": len(eps), "n_feasible": int(feasible.sum()), "n_infeasible": int((~feasible).sum()),
        "verify": verify,
        "partA_overall": {
            "u_qp_minus_u_safe_norm": stats(diff),
            "u_qp_minus_u_safe_norm_feasible": stats(diff, feasible),
            "u_qp_minus_u_safe_norm_infeasible": stats(diff, ~feasible),
            "angle_qp_vs_hardnet_deg": stats(ang_qp_hn, feasible),
            "goal_progress_qp": stats(gp_qp, feasible), "goal_progress_hardnet": stats(gp_hn, feasible),
            "goal_progress_qp_minus_hn": stats(gp_qp - gp_hn, feasible),
            "intervention_mag_qp": stats(dnom_qp, feasible), "intervention_mag_hardnet": stats(dnom_hn, feasible),
            "intervention_mag_hn_minus_qp": stats(dnom_hn - dnom_qp, feasible),
            "rotation_from_unom_qp_deg": stats(rot_qp, feasible), "rotation_from_unom_hardnet_deg": stats(rot_hn, feasible)},
        "partB_escapable": {
            "n_escapable": int(escap.sum()), "n_blocked": int((~escap).sum()),
            "gp_qp_minus_hn_escapable": stats(gp_qp - gp_hn, escap & feasible),
            "gp_qp_minus_hn_blocked": stats(gp_qp - gp_hn, (~escap)),
            "diff_norm_escapable": stats(diff, escap & feasible), "diff_norm_blocked": stats(diff, ~escap),
            "rotation_qp_escapable_deg": stats(rot_qp, escap & feasible),
            "rotation_hardnet_escapable_deg": stats(rot_hn, escap & feasible)},
    }
    (OUT / "stage2_l2qp_compare_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(gp_hn, gp_qp, ang_qp_hn, diff, dnom_hn, dnom_qp, feasible)
    _print(rep)
    return 0


def _best_progress(A, b, g):
    cx = [U_MAX, -U_MAX]
    cands = [np.array([sx * U_MAX, sy * U_MAX]) for sx in (1, -1) for sy in (1, -1)]
    A0, A1 = A
    for x in cx:
        if abs(A1) > 1e-12:
            cands.append(np.array([x, (b - A0 * x) / A1]))
    for y in cx:
        if abs(A0) > 1e-12:
            cands.append(np.array([(b - A1 * y) / A0, y]))
    best = -np.inf
    for u in cands:
        if np.all(np.abs(u) <= U_MAX + 1e-6) and (A @ u <= b + FTOL * (1 + abs(b))):
            best = max(best, float(g @ u))
    return best


def _figs(gp_hn, gp_qp, ang, diff, dnom_hn, dnom_qp, feas):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)
    ax[0].hist(gp_hn[feas], bins=40, alpha=0.6, label="HardNet u_safe·ĝ", color="#d62728")
    ax[0].hist(gp_qp[feas], bins=40, alpha=0.6, label="L2-QP u_qp·ĝ", color="#2ca02c")
    ax[0].axvline(0, color="0.3", lw=1); ax[0].set_title("goal progress: HardNet vs L2-QP"); ax[0].legend(fontsize=8, frameon=False)
    ax[1].hist(diff[feas], bins=40, color="#4c78a8"); ax[1].set_yscale("log")
    ax[1].set_title("||u_qp - u_safe|| (feasible steps)"); ax[1].set_xlabel("L2 distance")
    ax[2].hist(dnom_hn[feas], bins=40, alpha=0.6, label="HardNet ||u_safe-u_nom||", color="#d62728")
    ax[2].hist(dnom_qp[feas], bins=40, alpha=0.6, label="L2-QP ||u_qp-u_nom||", color="#2ca02c")
    ax[2].set_title("intervention magnitude"); ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "stage2_l2qp_compare.png"); plt.close(fig)


def _print(rep):
    print(f"steps={rep['n_steps']} episodes={rep['n_episodes']} feasible={rep['n_feasible']} infeasible={rep['n_infeasible']}")
    print("verify:", rep["verify"])
    pa = rep["partA_overall"]
    print("Part A (feasible steps):")
    print(f"  ||u_qp - u_safe||: median={pa['u_qp_minus_u_safe_norm_feasible']['median']:.3e} "
          f"p75={pa['u_qp_minus_u_safe_norm_feasible']['p75']:.3e}  (infeasible median={pa['u_qp_minus_u_safe_norm_infeasible']['median']})")
    print(f"  angle(u_qp,u_safe) median={pa['angle_qp_vs_hardnet_deg']['median']} deg")
    print(f"  goal progress: HardNet med={pa['goal_progress_hardnet']['median']:.3f}  L2QP med={pa['goal_progress_qp']['median']:.3f}  "
          f"QP-minus-HN med={pa['goal_progress_qp_minus_hn']['median']:.3e}")
    print(f"  intervention mag: HardNet med={pa['intervention_mag_hardnet']['median']:.3f}  L2QP med={pa['intervention_mag_qp']['median']:.3f}  "
          f"HN-minus-QP med={pa['intervention_mag_hn_minus_qp']['median']:.3e}")
    print(f"  rotation from u_nom: HardNet med={pa['rotation_from_unom_hardnet_deg']['median']:.1f}deg  "
          f"L2QP med={pa['rotation_from_unom_qp_deg']['median']:.1f}deg")
    pb = rep["partB_escapable"]
    print(f"Part B: escapable={pb['n_escapable']} blocked={pb['n_blocked']}")
    print(f"  gp(QP-HN) escapable med={pb['gp_qp_minus_hn_escapable']['median']:.3e}  blocked med={pb['gp_qp_minus_hn_blocked']['median']}")
    print(f"  rotation escapable: HardNet {pb['rotation_hardnet_escapable_deg']['median']:.1f} vs L2QP {pb['rotation_qp_escapable_deg']['median']:.1f} deg")


if __name__ == "__main__":
    raise SystemExit(main())
