"""v2.2.0 Stage 1 Step 6 — unbiased full-pool decomposition (SCOD epistemic + CBF residual).

Reads ONLY the Step-5 full-pool NPZs (data/diagnostics/v2.2.0_stage1_scod/episodes/),
deterministic (pure numpy/sklearn/scipy; reruns to identical numbers). Reproduces the four
Stage-0 analyses on the UNBIASED full pool with SCOD as the epistemic signal, alongside
recomputed 2-member disagreement and the executed-action CBF residual.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage1_decomposition.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage1_scod/episodes"
OUT_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage1_scod"
LEADS = [1, 3, 5, 10, 15, 30]
PRE_WINDOW = 30

# Declared in advance: higher = more failure-like (+1) for all three.
SIGNALS = {"SCOD": "scod_epistemic", "2-member": "ensemble_disagreement",
           "residual": "cbf_violation_safe"}


def checked_auc(labels, scores):
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    fin = np.isfinite(scores); labels, scores = labels[fin], scores[fin]
    if len(np.unique(labels)) < 2:
        return float("nan")
    sk = float(roc_auc_score(labels, scores))
    pos, neg = scores[labels == 1], scores[labels == 0]
    d = pos[:, None] - neg[None, :]
    pw = float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / (pos.size * neg.size))
    if abs(sk - pw) > 1e-9:
        raise AssertionError(f"AUROC mismatch {sk} {pw}")
    return sk


def event_action_step(outcome, event_step, n):
    if outcome in {"collision", "goal", "oob"} and event_step > 0:
        return max(0, min(event_step - 1, n - 1))
    if event_step < 0:
        return n - 1
    return max(0, min(event_step, n - 1))


def load():
    eps = []
    for path in sorted(EPISODE_DIR.glob("ep_*.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"]); n = int(len(z["h"]))
        eps.append({
            "idx": int(path.stem.split("_")[1]), "outcome": outcome,
            "event_action": event_action_step(outcome, int(md["event_step"]), n),
            "active_steps": int(md.get("active_steps", n)), "n": n,
            "scod_epistemic": np.asarray(z["scod_epistemic"], float),
            "ensemble_disagreement": np.asarray(z["ensemble_disagreement"], float),
            "cbf_violation_safe": np.asarray(z["cbf_violation_safe"], float),
            "clearance": np.asarray(z["clearance"], float),
            "closing_speed": np.asarray(z["closing_speed"], float),
        })
    return eps


def emax(ep, key):
    lo = max(0, ep["event_action"] - PRE_WINDOW + 1)
    hi = min(ep["event_action"] + 1, ep["active_steps"], ep["n"])
    return float(np.max(ep[key][lo:hi])) if hi > lo else None


def lead(ep, key, k):
    i = ep["event_action"] - k
    return float(ep[key][i]) if (0 <= i < ep["n"] and i < ep["active_steps"]) else None


def active(ep, key):
    hi = min(ep["active_steps"], ep["n"])
    return ep[key][:hi]


def assoc(pos, neg, key, reducer, k=None):
    lab, sc = [], []
    for L, grp in ((1, pos), (0, neg)):
        for ep in grp:
            v = emax(ep, key) if reducer == "emax" else lead(ep, key, k)
            if v is not None:
                lab.append(L); sc.append(v)
    return checked_auc(lab, sc), int(sum(lab)), int(len(lab) - sum(lab))


def main():
    eps = load()
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("collision", "stuck", "goal", "timeout", "oob")}
    counts = {o: len(by[o]) for o in by if by[o]}
    rep = {"counts": counts, "note": "UNBIASED full pool (N=500); proper rates but tiny "
           f"collision positive class (n={len(by['collision'])})."}

    # (1) association
    a1 = {"episode": {}, "lead": {}}
    for name, key in SIGNALS.items():
        c = assoc(by["collision"], by["goal"], key, "emax")
        s = assoc(by["stuck"], by["goal"], key, "emax")
        a1["episode"][name] = {"collision": c[0], "stuck": s[0], "n_pos_c": c[1], "n_pos_s": s[1], "n_goal": c[2]}
        a1["lead"][name] = {"collision": {k: assoc(by["collision"], by["goal"], key, "lead", k)[0] for k in LEADS},
                            "stuck": {k: assoc(by["stuck"], by["goal"], key, "lead", k)[0] for k in LEADS}}
    rep["A1_association"] = a1

    # (2) independence
    a2 = {}
    for o in ("collision", "stuck", "goal"):
        s = np.concatenate([active(e, "scod_epistemic") for e in by[o]]) if by[o] else np.array([])
        r = np.concatenate([active(e, "cbf_violation_safe") for e in by[o]]) if by[o] else np.array([])
        a2[o] = {"n": int(s.size),
                 "pearson": float(pearsonr(s, r)[0]) if s.size > 2 and np.ptp(s) > 0 and np.ptp(r) > 0 else float("nan"),
                 "spearman": float(spearmanr(s, r)[0]) if s.size > 2 and np.ptp(s) > 0 and np.ptp(r) > 0 else float("nan")}
    rep["A2_independence"] = a2

    # (3) differential separation
    a3 = {}
    for name in SIGNALS:
        ce = a1["episode"][name]["collision"]; se = a1["episode"][name]["stuck"]
        cvs = assoc(by["collision"], by["stuck"], SIGNALS[name], "emax")[0]
        a3[name] = {"collision_vs_goal": ce, "stuck_vs_goal": se, "gap_c_minus_s": ce - se,
                    "collision_vs_stuck_direct": cvs}
    rep["A3_differential"] = a3

    # (4) low-low-but-failed (both <= goal per-step medians); no dissection labels for this set
    gs = np.concatenate([active(e, "scod_epistemic") for e in by["goal"]])
    gr = np.concatenate([active(e, "cbf_violation_safe") for e in by["goal"]])
    smed, rmed = float(np.median(gs)), float(np.median(gr))
    a4 = {"scod_goal_median": smed, "residual_goal_median": rmed,
          "labels_available": False, "classes": {}}
    for o in ("collision", "stuck"):
        ep_ids, ll_ts, tot = [], 0, 0
        ll_clear, ll_close = [], []
        for e in by[o]:
            s = active(e, "scod_epistemic"); r = active(e, "cbf_violation_safe")
            cl = active(e, "clearance"); cs = active(e, "closing_speed")
            mask = (s <= smed) & (r <= rmed)
            tot += mask.size; ll_ts += int(mask.sum())
            if mask.any():
                ep_ids.append(e["idx"])
                ll_clear.extend(cl[mask].tolist()); ll_close.extend(cs[mask].tolist())
        a4["classes"][o] = {"episodes_with_lowlow_step": len(ep_ids), "total_episodes": len(by[o]),
                            "lowlow_timesteps": ll_ts, "total_timesteps": tot,
                            "lowlow_clearance_median": float(np.median(ll_clear)) if ll_clear else None,
                            "lowlow_closing_speed_median": float(np.median(ll_close)) if ll_close else None}
    rep["A4_lowlow"] = a4

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decomposition_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(by, a1, smed, rmed)
    _print(rep)
    return 0


def _figs(by, a1, smed, rmed):
    # lead-K AUROC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=140, sharey=True)
    for ax, cls in zip(axes, ("collision", "stuck")):
        for name, st in (("SCOD", "o-"), ("2-member", "s--"), ("residual", "^:")):
            ax.plot(LEADS, [a1["lead"][name][cls][k] for k in LEADS], st, label=name)
        ax.axhline(0.5, color="0.6", lw=1, ls=":"); ax.set_title(f"{cls} vs goal (full pool)")
        ax.set_xlabel("lead K"); ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("AUROC"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "decomp_lead_auroc.png"); plt.close(fig)
    # SCOD vs residual scatter
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=140)
    col = {"goal": "#7f7f7f", "stuck": "#1f77b4", "collision": "#d62728"}
    for o in ("goal", "stuck", "collision"):
        s = np.concatenate([active(e, "scod_epistemic") for e in by[o]])
        r = np.concatenate([active(e, "cbf_violation_safe") for e in by[o]])
        if s.size > 4000:
            st = int(np.ceil(s.size / 4000)); s, r = s[::st], r[::st]
        ax.scatter(s, r, s=6, alpha=0.4, c=col[o], edgecolors="none", label=f"{o} (n~{s.size})")
    ax.axvline(smed, color="0.3", ls="--", lw=1); ax.axhline(rmed, color="0.3", ls="--", lw=1)
    ax.set_xlabel("SCOD epistemic (per active step)"); ax.set_ylabel("residual cbf_violation_safe")
    ax.set_title("Full-pool axes; dashed = goal per-step medians")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "decomp_axes_scatter.png"); plt.close(fig)


def _print(rep):
    print("counts:", rep["counts"])
    print("A1 episode AUROC (collision/stuck vs goal):")
    for n in SIGNALS:
        e = rep["A1_association"]["episode"][n]
        print(f"  {n:9s}: collision={e['collision']:.4f} stuck={e['stuck']:.4f} (n_pos_c={e['n_pos_c']} n_pos_s={e['n_pos_s']} n_goal={e['n_goal']})")
    print("A2 independence (pearson/spearman SCOD vs residual):",
          {k: (round(v["pearson"], 3), round(v["spearman"], 3)) for k, v in rep["A2_independence"].items()})
    print("A3 differential (collision-vs-goal minus stuck-vs-goal):",
          {k: round(v["gap_c_minus_s"], 3) for k, v in rep["A3_differential"].items()})
    a4 = rep["A4_lowlow"]
    print(f"A4 low-low (goal med SCOD {a4['scod_goal_median']:.2f}, residual {a4['residual_goal_median']:.3f}):")
    for o in ("collision", "stuck"):
        c = a4["classes"][o]
        print(f"  {o}: {c['episodes_with_lowlow_step']}/{c['total_episodes']} eps, ts {c['lowlow_timesteps']}/{c['total_timesteps']}, "
              f"lowlow clearance med={c['lowlow_clearance_median']}, closing med={c['lowlow_closing_speed_median']}")


if __name__ == "__main__":
    raise SystemExit(main())
