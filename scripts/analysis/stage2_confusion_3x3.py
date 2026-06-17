"""v2.2.0 Stage 2 — 3x3 confusion matrix via a two-filter classifier (read-only).

Actual {goal, collision, stuck} x predicted {goal, collision, stuck}. Two-filter classifier:
  filter 1 = CBF residual decides success vs failure (residual_max < tau_res => goal);
  filter 2 = SCOD decides collision vs stuck among predicted failures (scod_max >= tau_scod => collision).
Reuses stage2_decomposition.py's exact loading + emax reduction + signal definitions. Reads ONLY
the saved NPZs + labels; deterministic. Also fills two ablations (residual-only, SCOD-only) for
comparison, and cross-references mechanism labels for the type-error cells.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_confusion_3x3.py
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

import stage2_decomposition as S2  # exact load(), emax(), auc(), SIGNALS  # noqa: E402

OUT = S2.OUT
CLASSES = ["goal", "collision", "stuck"]


def emax_arr(eps, key):
    return np.array([S2.emax(e, key) for e in eps], dtype=float)


def best_balanced_threshold(pos, neg):
    cand = np.unique(np.concatenate([pos, neg]))
    scores = [(0.5 * ((pos >= t).mean() + (neg < t).mean()), float(t)) for t in cand]
    return max(scores)[1]


def matrix(actual_eps, predict_fn):
    """actual_eps: dict class->list[ep]; predict_fn(ep)->predicted label. Returns counts dict."""
    cm = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    for a in CLASSES:
        for ep in actual_eps[a]:
            cm[a][predict_fn(ep)] += 1
    return cm


def rates(cm):
    out = {}
    for a in CLASSES:
        tot = sum(cm[a].values())
        out[a] = {p: (cm[a][p] / tot if tot else 0.0) for p in CLASSES}
    return out


def summary_metrics(cm):
    total = sum(sum(cm[a].values()) for a in CLASSES)
    correct = sum(cm[a][a] for a in CLASSES)
    n_fail = sum(sum(cm[a].values()) for a in ("collision", "stuck"))
    miss = cm["collision"]["goal"] + cm["stuck"]["goal"]
    n_goal = sum(cm["goal"].values())
    false_alarm = cm["goal"]["collision"] + cm["goal"]["stuck"]
    detected_fail = n_fail - miss
    type_err = cm["collision"]["stuck"] + cm["stuck"]["collision"]
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "miss_rate": miss / n_fail if n_fail else 0.0,
        "false_alarm_rate": false_alarm / n_goal if n_goal else 0.0,
        "type_error_rate": type_err / detected_fail if detected_fail else 0.0,
        "recall": {a: (cm[a][a] / sum(cm[a].values()) if sum(cm[a].values()) else 0.0) for a in CLASSES},
    }


def main() -> int:
    eps = S2.load()
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("collision", "stuck", "goal", "timeout")}
    counts = {o: len(by[o]) for o in by}

    # per-episode signals
    res = {o: emax_arr(by[o], "cbf_violation_safe") for o in by}
    scod = {o: emax_arr(by[o], "scod_epistemic") for o in by}

    # self-consistency
    def auc_vs_goal(sig, cls):
        pos, neg = sig[cls], sig["goal"]
        return round(S2.auc(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg]), 4)
    check = {"SCOD:collision": auc_vs_goal(scod, "collision"), "SCOD:stuck": auc_vs_goal(scod, "stuck"),
             "residual:collision": auc_vs_goal(res, "collision"), "residual:stuck": auc_vs_goal(res, "stuck")}

    # thresholds
    tau_scod = float(np.median(scod["goal"]))                       # 7.65, stage2 type axis
    tau_res_5fpr = float(np.quantile(res["goal"], 0.95, method="higher"))
    tau_res_pos = 1e-12                                             # failure if any CBF violation
    fpr_5 = float((res["goal"] >= tau_res_5fpr).mean())
    fpr_pos = float((res["goal"] >= tau_res_pos).mean())
    # optimal type thresholds (collision pos vs stuck neg)
    tau_res_type = best_balanced_threshold(res["collision"], res["stuck"])
    tau_scod_type = best_balanced_threshold(scod["collision"], scod["stuck"])
    tau_scod_det = float(np.quantile(scod["goal"], 0.95, method="higher"))   # SCOD 5%FPR detection

    rep = {"counts": counts, "self_consistency_auroc": check,
           "thresholds": {"tau_scod_type_axis": tau_scod, "tau_res_5fpr": tau_res_5fpr, "goalFPR_5fpr": fpr_5,
                          "tau_res_pos": tau_res_pos, "goalFPR_pos": fpr_pos,
                          "tau_res_type_optimal": tau_res_type, "tau_scod_type_optimal": tau_scod_type,
                          "tau_scod_det_5fpr": tau_scod_det},
           "residual_type_separation": {
               "collision_min": float(res["collision"].min()), "collision_median": float(np.median(res["collision"])),
               "stuck_max": float(res["stuck"].max()), "stuck_median": float(np.median(res["stuck"])),
               "n_stuck_overlap_ge_collmin": int((res["stuck"] >= res["collision"].min()).sum())}}

    # predictors
    def two_filter(tau_res):
        def f(ep):
            r = S2.emax(ep, "cbf_violation_safe"); s = S2.emax(ep, "scod_epistemic")
            if r < tau_res:
                return "goal"
            return "collision" if s >= tau_scod else "stuck"
        return f

    def residual_only(tau_res):
        def f(ep):
            r = S2.emax(ep, "cbf_violation_safe")
            if r < tau_res:
                return "goal"
            return "collision" if r >= tau_res_type else "stuck"
        return f

    def scod_only(ep):
        s = S2.emax(ep, "scod_epistemic")
        if s < tau_scod_det:
            return "goal"
        return "collision" if s >= tau_scod_type else "stuck"

    matrices = {}
    for tag, tau_res in (("two_filter_res5fpr", tau_res_5fpr), ("two_filter_respos", tau_res_pos)):
        cm = matrix(by, two_filter(tau_res))
        matrices[tag] = {"counts": cm, "rates": rates(cm), "metrics": summary_metrics(cm),
                         "timeout_pred": _timeout_pred(by["timeout"], two_filter(tau_res))}
    for tag, fn in (("residual_only_res5fpr", residual_only(tau_res_5fpr)), ("scod_only", scod_only)):
        cm = matrix(by, fn)
        matrices[tag] = {"counts": cm, "rates": rates(cm), "metrics": summary_metrics(cm),
                         "timeout_pred": _timeout_pred(by["timeout"], fn)}
    rep["matrices"] = matrices

    # type-error mechanism cross-ref: stuck predicted collision under the two-filter (SCOD-high stuck)
    confused = {}
    for ep in by["stuck"]:
        if two_filter(tau_res_5fpr)(ep) == "collision":
            confused[ep["mechanism"]] = confused.get(ep["mechanism"], 0) + 1
    rep["type_error_stuck_to_collision_mechanism"] = confused

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_confusion_3x3_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(matrices)
    _print(rep)
    return 0


def _timeout_pred(timeouts, fn):
    out = {"goal": 0, "collision": 0, "stuck": 0}
    for ep in timeouts:
        out[fn(ep)] += 1
    return out


def _figs(matrices):
    for tag in ("two_filter_res5fpr", "residual_only_res5fpr", "scod_only"):
        cm = matrices[tag]["counts"]
        M = np.array([[matrices[tag]["rates"][a][p] for p in CLASSES] for a in CLASSES])
        fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), [f"pred {c}" for c in CLASSES])
        ax.set_yticks(range(3), [f"actual {c}" for c in CLASSES])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i,j]:.3f}\n({cm[CLASSES[i]][CLASSES[j]]})", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] > 0.5 else "black")
        m = matrices[tag]["metrics"]
        ax.set_title(f"{tag}\nacc {m['overall_accuracy']:.3f}  miss {m['miss_rate']:.3f}  "
                     f"FA {m['false_alarm_rate']:.3f}  typeErr {m['type_error_rate']:.3f}", fontsize=8)
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(OUT / f"stage2_confusion_{tag}.png"); plt.close(fig)


def _print(rep):
    print("counts:", rep["counts"])
    print("self-consistency:", rep["self_consistency_auroc"])
    t = rep["thresholds"]
    print(f"thresholds: tau_scod_type={t['tau_scod_type_axis']:.3f} tau_res_5fpr={t['tau_res_5fpr']:.2e}(FPR {t['goalFPR_5fpr']:.3f}) "
          f"tau_res_pos(FPR {t['goalFPR_pos']:.3f}) tau_res_type_opt={t['tau_res_type_optimal']:.2f} tau_scod_det={t['tau_scod_det_5fpr']:.2f}")
    rs = rep["residual_type_separation"]
    print(f"residual type sep: coll[min {rs['collision_min']:.1f}, med {rs['collision_median']:.1f}] "
          f"stuck[max {rs['stuck_max']:.1f}, med {rs['stuck_median']:.1f}] overlap {rs['n_stuck_overlap_ge_collmin']}")
    for tag, M in rep["matrices"].items():
        print(f"\n[{tag}]  metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in M["metrics"].items() if k != "recall"})
        print("  recall:", {k: round(v, 3) for k, v in M["metrics"]["recall"].items()})
        for a in CLASSES:
            print(f"   actual {a:9s}: " + " ".join(f"{p}={M['counts'][a][p]}" for p in CLASSES))
        print("  timeout pred:", M["timeout_pred"])
    print("\ntype-error stuck->collision mechanisms (two-filter):", rep["type_error_stuck_to_collision_mechanism"])


if __name__ == "__main__":
    raise SystemExit(main())
