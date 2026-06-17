"""v2.2.0 Stage 2 — confusion-matrix / detection-rate analysis at operating points (read-only).

Reuses stage2_decomposition.py's exact episode loading + signal reduction (emax over the 30-step
pre-event window) + signal definitions (SCOD, 2-member, residual). Reads ONLY the Stage-2 NPZs +
labels; deterministic. Parts A-D: binary failure detection at the goal-median and fixed-5%-FPR
operating points; success-mispredicted-as-failure; failure-type confusion (collision vs stuck).

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_detection_rates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage2_decomposition as S2  # reuse load(), emax(), auc(), SIGNALS exactly  # noqa: E402

OUT = S2.OUT
TARGET_FPR = 0.05
SIGNALS = S2.SIGNALS


def episode_signal(eps, key):
    return np.array([v for v in (S2.emax(e, key) for e in eps) if v is not None], dtype=float)


def threshold_at_fpr(goal_scores, fpr):
    # smallest threshold flagging <= fpr fraction of goal (strictly-greater rule for ties up high)
    return float(np.quantile(goal_scores, 1.0 - fpr, method="higher"))


def rate(scores, thr):
    return float(np.mean(scores >= thr))


def main() -> int:
    eps = S2.load()
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("collision", "stuck", "goal", "timeout")}
    counts = {o: len(by[o]) for o in by}
    rep = {"counts": counts}

    # signal arrays per class
    sig = {name: {o: episode_signal(by[o], key) for o in by} for name, key in SIGNALS.items()}

    # self-consistency: reproduce stage2 episode-level AUROC vs goal
    check = {}
    for name in SIGNALS:
        for cls in ("collision", "stuck"):
            pos, neg = sig[name][cls], sig[name]["goal"]
            lab = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            check[f"{name}:{cls}"] = round(S2.auc(lab, np.concatenate([pos, neg])), 4)
    rep["self_consistency_auroc"] = check

    # Part A — confusion at two operating points
    partA = {}
    thresholds = {}
    for name in SIGNALS:
        g = sig[name]["goal"]
        thr_med = float(np.median(g))
        thr_fpr = threshold_at_fpr(g, TARGET_FPR)
        thresholds[name] = {"goal_median": thr_med, "fpr5": thr_fpr}
        partA[name] = {}
        for label, thr in (("goal_median", thr_med), ("fpr5", thr_fpr)):
            partA[name][label] = {
                "threshold": thr,
                "collision_TPR": rate(sig[name]["collision"], thr),
                "stuck_TPR": rate(sig[name]["stuck"], thr),
                "timeout_TPR": rate(sig[name]["timeout"], thr),
                "goal_FPR": rate(g, thr),
            }
    rep["thresholds"] = thresholds
    rep["partA_detection"] = partA

    # Part B — success mispredicted as failure (type split by SCOD epistemic axis)
    scod_med = thresholds["SCOD"]["goal_median"]
    scod_fpr = thresholds["SCOD"]["fpr5"]
    partB = {}
    for opname, gthr_lookup in (("goal_median", "goal_median"), ("fpr5", "fpr5")):
        # "flagged as failure" defined per signal's own threshold at this operating point;
        # type split uses SCOD vs SCOD goal-median (the stage2 type axis).
        sub = {}
        for name in SIGNALS:
            g = sig[name]["goal"]
            thr = thresholds[name][gthr_lookup]
            flagged = g >= thr
            gscod = sig["SCOD"]["goal"]
            pred_coll = flagged & (gscod >= scod_med)
            pred_stuck = flagged & (gscod < scod_med)
            sub[name] = {
                "P_goal_flagged_any": float(np.mean(flagged)),
                "P_goal_as_collision": float(np.mean(pred_coll)),
                "P_goal_as_stuck": float(np.mean(pred_stuck)),
            }
        partB[opname] = sub
    rep["partB_success_mispredicted"] = {
        "type_rule": "flagged AND SCOD>=goal-median => predicted collision; flagged AND SCOD<median => predicted stuck",
        "scod_goal_median": scod_med, "by_operating_point": partB}

    # Part C — failure-type confusion (collision vs stuck) on actual failures, SCOD-based type predictor
    sc = sig["SCOD"]
    pred_coll_c = float(np.mean(sc["collision"] >= scod_med))   # P(pred collision | actual collision)
    pred_coll_s = float(np.mean(sc["stuck"] >= scod_med))       # P(pred collision | actual stuck)
    n_c, n_s = sc["collision"].size, sc["stuck"].size
    cm = {
        "rule": "predict collision if SCOD episode-max >= goal-median (7.65), else stuck",
        "scod_goal_median": scod_med,
        "P_predColl_given_collision": pred_coll_c,
        "P_predStuck_given_collision": 1 - pred_coll_c,
        "P_predColl_given_stuck": pred_coll_s,
        "P_predStuck_given_stuck": 1 - pred_coll_s,
        "counts": {"coll_predColl": int(round(pred_coll_c * n_c)), "coll_predStuck": int(round((1 - pred_coll_c) * n_c)),
                   "stuck_predColl": int(round(pred_coll_s * n_s)), "stuck_predStuck": int(round((1 - pred_coll_s) * n_s))},
        "overall_type_accuracy": float((pred_coll_c * n_c + (1 - pred_coll_s) * n_s) / (n_c + n_s)),
    }
    # mechanism cross-ref for the confused cases (stuck predicted collision = SCOD-high stuck)
    mech_confused = {}
    for e in by["stuck"]:
        v = S2.emax(e, "scod_epistemic")
        if v is not None and v >= scod_med:
            mech_confused[e["mechanism"]] = mech_confused.get(e["mechanism"], 0) + 1
    cm["stuck_predColl_mechanism"] = mech_confused
    rep["partC_type_confusion"] = cm

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_detection_rates_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(sig, thresholds, cm)
    _print(rep)
    return 0


def _figs(sig, thresholds, cm):
    # TPR-at-FPR (ROC) for collision and stuck vs goal
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    fprs = np.linspace(0, 1, 200)
    for ax, cls in zip(axes, ("collision", "stuck")):
        for name in SIGNALS:
            g = np.sort(sig[name]["goal"]); pos = sig[name][cls]
            tpr = []
            for f in fprs:
                thr = np.quantile(g, 1 - f, method="higher") if f > 0 else g.max() + 1e-9
                tpr.append(np.mean(pos >= thr))
            ax.plot(fprs, tpr, label=name)
        ax.axvline(0.05, color="0.5", ls="--", lw=1, label="5% FPR")
        ax.plot([0, 1], [0, 1], color="0.8", lw=1)
        ax.set_title(f"{cls} vs goal"); ax.set_xlabel("goal FPR"); ax.set_ylabel("detection TPR"); ax.set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage2_roc.png"); plt.close(fig)

    # type-confusion matrix (rates)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
    M = np.array([[cm["P_predColl_given_collision"], cm["P_predStuck_given_collision"]],
                  [cm["P_predColl_given_stuck"], cm["P_predStuck_given_stuck"]]])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["pred collision", "pred stuck"]); ax.set_yticks([0, 1], ["actual collision", "actual stuck"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.5 else "black")
    ax.set_title(f"Type confusion (SCOD>={cm['scod_goal_median']:.2f} => collision)\noverall acc {cm['overall_type_accuracy']:.3f}", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "stage2_type_confusion.png"); plt.close(fig)


def _print(rep):
    print("counts:", rep["counts"])
    print("self-consistency AUROC (should match stage2: SCOD coll 0.9027, stuck 0.4211, ...):")
    print("  ", rep["self_consistency_auroc"])
    print("Part A — detection at operating points:")
    for name in SIGNALS:
        for op in ("goal_median", "fpr5"):
            d = rep["partA_detection"][name][op]
            print(f"  {name:9s} {op:11s} thr={d['threshold']:.3f}: collTPR={d['collision_TPR']:.3f} stuckTPR={d['stuck_TPR']:.3f} "
                  f"timeoutTPR={d['timeout_TPR']:.3f} goalFPR={d['goal_FPR']:.3f}")
    print("Part B — success mispredicted (fpr5 op):")
    for name in SIGNALS:
        d = rep["partB_success_mispredicted"]["by_operating_point"]["fpr5"][name]
        print(f"  {name:9s}: P(goal flagged)={d['P_goal_flagged_any']:.3f} as-coll={d['P_goal_as_collision']:.3f} as-stuck={d['P_goal_as_stuck']:.3f}")
    c = rep["partC_type_confusion"]
    print("Part C — type confusion (SCOD-based):")
    print(f"  P(predColl|coll)={c['P_predColl_given_collision']:.3f}  P(predStuck|stuck)={c['P_predStuck_given_stuck']:.3f}  acc={c['overall_type_accuracy']:.3f}")
    print(f"  stuck->predColl mechanisms: {c['stuck_predColl_mechanism']}")


if __name__ == "__main__":
    raise SystemExit(main())
