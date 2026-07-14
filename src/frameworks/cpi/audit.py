"""CPI iteration-0 audits and figures.

8a gap audit (seed-independent, exact oracles): G = {V_M17 <= 0 < V_m0}; criterion C = closing speed
toward the nearest active obstacle AND surface distance <= ||v||^2/(2 u_max). 8b per-seed test audits:
coverage, zero-set IoU, sign accuracy, eps_q. 8c aggregation. Figures F1-F6.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def criterion_C(pos, vel, C, R, A, u_max):
    """C(x): v . u_hat > 0 (u_hat toward the nearest-by-surface-distance active obstacle center) AND
    surface distance <= ||v||^2/(2 u_max). pos/vel [N,2]; C[N,K,2] R[N,K] A[N,K]. Returns bool [N], plus
    the two component masks for the breakdown."""
    rel = C - pos[:, None, :]                                            # [N,K,2] toward center
    dist = np.linalg.norm(rel, axis=-1)                                  # [N,K]
    surf = dist - R
    surf_masked = np.where(A, surf, np.inf)
    j = np.argmin(surf_masked, axis=1)                                   # nearest active by surface dist
    n = np.arange(pos.shape[0])
    cvec = rel[n, j]; d0 = dist[n, j]
    u_hat = cvec / np.clip(d0[:, None], 1e-12, None)
    closing = np.sum(vel * u_hat, axis=1) > 0.0
    surf0 = surf[n, j]
    speed2 = np.sum(vel * vel, axis=1)
    reachable = surf0 <= speed2 / (2.0 * u_max)
    return closing & reachable, closing, reachable, surf0


def gap_audit(vm0, vm17, pos, vel, C, R, A, u_max) -> dict:
    """8a: gap set G = {V_M17 <= 0 < V_m0}; fraction satisfying C; certified-fraction ratio; breakdown."""
    vm0 = np.asarray(vm0); vm17 = np.asarray(vm17)
    G = (vm17 <= 0.0) & (vm0 > 0.0)
    nG = int(G.sum())
    frac_cert_m0 = float((vm0 <= 0.0).mean()); frac_cert_m17 = float((vm17 <= 0.0).mean())
    out = {"n_states": int(vm0.size), "n_gap": nG,
           "frac_cert_m0": frac_cert_m0, "frac_cert_m17": frac_cert_m17,
           "certified_fraction_ratio": (frac_cert_m0 / frac_cert_m17) if frac_cert_m17 > 0 else None}
    if nG:
        cmask, closing, reachable, _ = criterion_C(pos[G], vel[G], C[G], R[G], A[G], u_max)
        out["frac_C"] = float(cmask.mean())
        out["remainder_breakdown"] = {
            "not_closing_not_reachable": int((~closing & ~reachable).sum()),
            "closing_not_reachable": int((closing & ~reachable).sum()),
            "not_closing_reachable": int((~closing & reachable).sum()),
            "C_satisfied": int(cmask.sum())}
    else:
        out["frac_C"] = None
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool); u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def test_audit(vhat, vraw, band, eps_q10) -> dict:
    """8b: coverage within band, zero-set IoU, sign accuracy, conservative IoU, reliability curve."""
    vhat = np.asarray(vhat); vraw = np.asarray(vraw)
    in_band = np.abs(vhat) <= band
    cov = float((vraw[in_band] <= vhat[in_band] + eps_q10).mean()) if in_band.any() else None
    iou_zero = iou(vhat <= 0.0, vraw <= 0.0)
    iou_cons = iou(vhat <= -eps_q10, vraw <= 0.0)
    sign_acc = float(((vhat <= 0.0) == (vraw <= 0.0)).mean())
    # reliability: bin V_hat, report V_raw mean + 10/90 quantiles
    edges = np.linspace(np.percentile(vhat, 1), np.percentile(vhat, 99), 21)
    bins = np.digitize(vhat, edges); rel = []
    for bidx in range(1, len(edges)):
        m = bins == bidx
        if m.sum() >= 20:
            rel.append({"vhat_lo": float(edges[bidx - 1]), "vhat_hi": float(edges[bidx]),
                        "vraw_mean": float(vraw[m].mean()), "vraw_p10": float(np.percentile(vraw[m], 10)),
                        "vraw_p90": float(np.percentile(vraw[m], 90)), "n": int(m.sum())})
    return {"n_test": int(vhat.size), "n_test_band": int(in_band.sum()),
            "coverage_band_alpha0.10": cov, "iou_zeroset": iou_zero, "iou_conservative": iou_cons,
            "sign_accuracy": sign_acc, "reliability": rel}


def make_figures(fig_dir, vhat_test, vraw_test, cov_curve, eps_profile, gap, ratio_ci, slices=None):
    """F1-F6 (PNG). slices (optional, seed-42 verdict-grade) is a list of dicts with grid data for F3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    # F1 hexbin V_hat vs V_raw
    fig, ax = plt.subplots(figsize=(5, 5)); ax.hexbin(vhat_test, vraw_test, gridsize=60, bins="log", cmap="viridis")
    lo, hi = min(vhat_test.min(), vraw_test.min()), max(vhat_test.max(), vraw_test.max())
    ax.plot([lo, hi], [lo, hi], "r-", lw=1); ax.set_xlabel("V_hat"); ax.set_ylabel("V_raw"); ax.set_title("F1 V_hat vs V_raw")
    fig.tight_layout(); fig.savefig(fig_dir / "F1_hexbin.png", dpi=110); plt.close(fig)
    # F2 coverage vs nominal
    fig, ax = plt.subplots(figsize=(5, 4)); al = [c[0] for c in cov_curve]; cv = [c[1] for c in cov_curve]
    ax.plot([1 - a for a in al], cv, "o-"); ax.plot([0.8, 1.0], [0.8, 1.0], "k--", lw=0.8)
    ax.set_xlabel("nominal 1-alpha"); ax.set_ylabel("empirical coverage"); ax.set_title("F2 one-sided coverage")
    fig.tight_layout(); fig.savefig(fig_dir / "F2_coverage.png", dpi=110); plt.close(fig)
    # F3 certified-set slices
    if slices:
        n = len(slices); fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        for k, sl in enumerate(slices):
            ax = axs[0, k]; ext = sl["extent"]
            ax.contour(sl["gx"], sl["gy"], sl["vm0"], levels=[0.0], colors="tab:blue", linewidths=1.4)
            ax.contour(sl["gx"], sl["gy"], sl["vm17"], levels=[0.0], colors="tab:green", linewidths=1.4)
            ax.contour(sl["gx"], sl["gy"], sl["vhat"], levels=[0.0], colors="tab:red", linewidths=1.2, linestyles="--")
            ax.contour(sl["gx"], sl["gy"], sl["vhat"], levels=[-sl["eps"]], colors="tab:orange", linewidths=1.0, linestyles=":")
            ax.set_title(sl["title"], fontsize=8); ax.set_aspect("equal")
        fig.suptitle("F3 zero boundaries: V_m0(blue) V_M17(green) Vhat=0(red--) Vhat=-eps(orange:)", fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "F3_slices.png", dpi=110); plt.close(fig)
    # F4 eps_q vs alpha and band
    fig, ax = plt.subplots(figsize=(5, 4))
    for band, pts in eps_profile.items():
        al = [p[0] for p in pts]; eq = [p[1] for p in pts]
        ax.plot(al, eq, "o-", label=f"band {band}")
    ax.axhline(0.0625, color="k", ls="--", lw=0.8, label="delta_grid 0.0625"); ax.set_xlabel("alpha"); ax.set_ylabel("eps_q")
    ax.legend(fontsize=7); ax.set_title("F4 eps_q profile"); fig.tight_layout(); fig.savefig(fig_dir / "F4_epsq.png", dpi=110); plt.close(fig)
    # F5 gap breakdown + speed-clearance scatter
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    if gap.get("gap_scatter"):
        sc = gap["gap_scatter"]; col = ["tab:red" if c else "tab:gray" for c in sc["C"]]
        axs[0].scatter(sc["clearance"], sc["speed"], c=col, s=4, alpha=0.5)
        axs[0].set_xlabel("surface clearance"); axs[0].set_ylabel("speed"); axs[0].set_title("F5 gap set (red=C)")
    bd = gap.get("remainder_breakdown", {})
    if bd:
        axs[1].bar(range(len(bd)), list(bd.values())); axs[1].set_xticks(range(len(bd)))
        axs[1].set_xticklabels(list(bd.keys()), rotation=45, ha="right", fontsize=6); axs[1].set_title("F5 breakdown")
    fig.tight_layout(); fig.savefig(fig_dir / "F5_gap.png", dpi=110); plt.close(fig)
    # F6 certified-fraction ratio bar with CI
    fig, ax = plt.subplots(figsize=(3.5, 4)); r = ratio_ci["ratio"]; ci = ratio_ci.get("ci")
    ax.bar([0], [r], yerr=[[r - ci[0]], [ci[1] - r]] if ci else None, capsize=6, color="tab:purple")
    ax.axhline(1.0, color="k", ls="--", lw=0.8); ax.set_xticks([0]); ax.set_xticklabels(["m0/M17"])
    ax.set_ylabel("certified-fraction ratio"); ax.set_title("F6 cert ratio"); fig.tight_layout()
    fig.savefig(fig_dir / "F6_ratio.png", dpi=110); plt.close(fig)


def bootstrap_scene_ci(vhat, vraw, scene_id, band, eps_q10, n_resample=1000, seed=20260508):
    """Scene-bootstrap 95% CI of coverage + zero-set IoU (resample TEST scenes)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(scene_id); covs = []; ious = []
    idx_by_scene = {s: np.where(scene_id == s)[0] for s in uniq}
    for _ in range(n_resample):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        sel = np.concatenate([idx_by_scene[uniq[p]] for p in pick])
        vh = vhat[sel]; vr = vraw[sel]; ib = np.abs(vh) <= band
        covs.append((vr[ib] <= vh[ib] + eps_q10).mean() if ib.any() else np.nan)
        ious.append(iou(vh <= 0.0, vr <= 0.0))
    q = lambda a: [float(np.percentile(a, 2.5, method="linear")), float(np.percentile(a, 97.5, method="linear"))]
    return {"coverage_ci": q(np.asarray(covs)[~np.isnan(covs)]), "iou_ci": q(ious)}
