"""v2.2.0 Stage 2 Step 4 — quantified two-axis decomposition on the adequate-N failure sample.

Reads ONLY the Step-2 NPZs (data/diagnostics/v2.2.0_stage2_largeN/episodes/) and the Step-3
label table (stage2_failure_labels.csv). Deterministic (pure numpy/sklearn/scipy; seeded
bootstrap). Five analyses: association (+95% CIs), independence, differential separation,
the 2x2 SCOD x residual quadrant decomposition (+ mechanism cross-tab), and lead-time.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_decomposition.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_stage2_largeN"
EPISODE_DIR = OUT / "episodes"
LABELS = OUT / "stage2_failure_labels.csv"

LEADS = [1, 3, 5, 10, 15, 30]
PRE_WINDOW = 30
STUCK_WINDOW = 60
STUCK_RADIUS = 0.10
BOOT = 1000
BOOT_SEED = 20260606
SIGNALS = {"SCOD": "scod_epistemic", "2-member": "ensemble_disagreement", "residual": "cbf_violation_safe"}


def auc(labels, scores):
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


def boot_ci_auc(pos, neg, seed):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        p = rng.choice(pos, size=pos.size, replace=True)
        n = rng.choice(neg, size=neg.size, replace=True)
        lab = np.concatenate([np.ones(p.size), np.zeros(n.size)])
        vals.append(auc(lab, np.concatenate([p, n])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def event_action_step(outcome, event_step, n):
    if outcome in {"collision", "goal", "oob"} and event_step > 0:
        return max(0, min(event_step - 1, n - 1))
    if event_step < 0:
        return n - 1
    return max(0, min(event_step, n - 1))


def load():
    labels = {}
    if LABELS.exists():
        for r in csv.DictReader(LABELS.open()):
            labels[int(r["episode_idx"])] = r
    eps = []
    for path in sorted(EPISODE_DIR.glob("ep_*.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"]); n = int(len(z["h"]))
        idx = int(path.stem.split("_")[1])
        eps.append({"idx": idx, "outcome": outcome,
                    "event_action": event_action_step(outcome, int(md["event_step"]), n),
                    "event_step": int(md["event_step"]), "active_steps": int(md.get("active_steps", n)), "n": n,
                    "scod_epistemic": np.asarray(z["scod_epistemic"], float),
                    "ensemble_disagreement": np.asarray(z["ensemble_disagreement"], float),
                    "cbf_violation_safe": np.asarray(z["cbf_violation_safe"], float),
                    "clearance": np.asarray(z["clearance"], float),
                    "closing_speed": np.asarray(z["closing_speed"], float),
                    "states": np.asarray(z["states"], float),
                    "mechanism": labels.get(idx, {}).get("mechanism", "n/a"),
                    "layer": labels.get(idx, {}).get("layer", "n/a")})
    return eps


def emax(ep, key):
    lo = max(0, ep["event_action"] - PRE_WINDOW + 1)
    hi = min(ep["event_action"] + 1, ep["active_steps"], ep["n"])
    return float(np.max(ep[key][lo:hi])) if hi > lo else None


def lead(ep, key, k):
    i = ep["event_action"] - k
    return float(ep[key][i]) if (0 <= i < ep["n"] and i < ep["active_steps"]) else None


def active(ep, key):
    hi = min(ep["active_steps"], ep["n"]); return ep[key][:hi]


def main():
    eps = load()
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("collision", "stuck", "goal", "timeout", "oob")}
    by = {o: v for o, v in by.items() if v}
    counts = {o: len(v) for o, v in by.items()}
    goal = by.get("goal", [])
    rep = {"counts": counts, "n_goal_reference": len(goal), "bootstrap": {"n": BOOT, "seed": BOOT_SEED}}

    # (1) association + CIs
    a1 = {}
    for name, key in SIGNALS.items():
        a1[name] = {}
        for cls in ("collision", "stuck"):
            pos = [v for v in (emax(e, key) for e in by.get(cls, [])) if v is not None]
            neg = [v for v in (emax(e, key) for e in goal) if v is not None]
            a1[name][cls] = {"auroc": auc([1]*len(pos)+[0]*len(neg), pos+neg),
                             "ci": boot_ci_auc(pos, neg, BOOT_SEED), "n_pos": len(pos), "n_neg": len(neg)}
        a1[name]["lead_collision"] = {}
        for k in LEADS:
            pos = [v for v in (lead(e, key, k) for e in by.get("collision", [])) if v is not None]
            neg = [v for v in (lead(e, key, k) for e in goal) if v is not None]
            a1[name]["lead_collision"][k] = {"auroc": auc([1]*len(pos)+[0]*len(neg), pos+neg),
                                             "ci": boot_ci_auc(pos, neg, BOOT_SEED + k)}
    rep["A1_association"] = a1

    # (2) independence
    a2 = {}
    for o in by:
        if o in ("timeout", "oob"):
            continue
        s = np.concatenate([active(e, "scod_epistemic") for e in by[o]])
        r = np.concatenate([active(e, "cbf_violation_safe") for e in by[o]])
        ok = s.size > 2 and np.ptp(s) > 0 and np.ptp(r) > 0
        a2[o] = {"n": int(s.size), "pearson": float(pearsonr(s, r)[0]) if ok else float("nan"),
                 "spearman": float(spearmanr(s, r)[0]) if ok else float("nan")}
    rep["A2_independence"] = a2

    # (3) differential separation (+CI on the gap via paired bootstrap on episodes)
    a3 = {}
    for name, key in SIGNALS.items():
        cg = a1[name]["collision"]["auroc"]; sg = a1[name]["stuck"]["auroc"]
        cvs_pos = [v for v in (emax(e, key) for e in by.get("collision", [])) if v is not None]
        cvs_neg = [v for v in (emax(e, key) for e in by.get("stuck", [])) if v is not None]
        a3[name] = {"collision_vs_goal": cg, "stuck_vs_goal": sg, "gap_c_minus_s": cg - sg,
                    "collision_vs_stuck": auc([1]*len(cvs_pos)+[0]*len(cvs_neg), cvs_pos+cvs_neg),
                    "collision_vs_stuck_ci": boot_ci_auc(cvs_pos, cvs_neg, BOOT_SEED + 100)}
    rep["A3_differential"] = a3

    # (4) 2x2 quadrant (episode-max; thresholds = goal-class episode-max medians)
    def emax_pair(e):
        return emax(e, "scod_epistemic"), emax(e, "cbf_violation_safe")
    gscod = [p[0] for p in (emax_pair(e) for e in goal) if p[0] is not None]
    gres = [p[1] for p in (emax_pair(e) for e in goal) if p[1] is not None]
    sthr, rthr = float(np.median(gscod)), float(np.median(gres))
    quad_outcome = {}
    quad_mech = {}
    for cls in ("collision", "stuck"):
        for e in by.get(cls, []):
            s, r = emax_pair(e)
            if s is None or r is None:
                continue
            q = ("Shi" if s >= sthr else "Slo") + "_" + ("Rhi" if r >= rthr else "Rlo")
            quad_outcome.setdefault(q, {}).setdefault(cls, 0)
            quad_outcome[q][cls] += 1
            if cls == "stuck":
                quad_mech.setdefault(q, {}).setdefault(e["mechanism"], 0)
                quad_mech[q][e["mechanism"]] += 1
    rep["A4_quadrant"] = {"scod_threshold": sthr, "residual_threshold": rthr,
                          "residual_threshold_degenerate": rthr <= 1e-9,
                          "quadrant_x_outcome": quad_outcome, "quadrant_x_stuck_mechanism": quad_mech}

    # (5) lead-time: collision actionable lead (steps before event signal first >= goal median)
    a5 = {"collision_actionable_lead": {}, "stuck_physical_onset": {}}
    for name, key in SIGNALS.items():
        thr = float(np.median([v for v in (emax(e, key) for e in goal) if v is not None]))
        leads = []
        for e in by.get("collision", []):
            ea = e["event_action"]; series = e[key][:ea + 1]
            crossed = np.where(series >= thr)[0]
            if crossed.size:
                run_start = crossed[0]
                for j in range(crossed[0], ea + 1):
                    if series[j] < thr:
                        run_start = j + 1
                leads.append(ea - run_start)
        a5["collision_actionable_lead"][name] = {
            "threshold": thr, "n_with_crossing": len(leads),
            "median_steps_before_event": float(np.median(leads)) if leads else None,
            "p25": float(np.percentile(leads, 25)) if leads else None,
            "p75": float(np.percentile(leads, 75)) if leads else None}
    # stuck physical stall onset (positions-derived): first step the W=60 window displacement <= r_stuck
    onsets = []
    for e in by.get("stuck", []):
        pos = e["states"][:, :2]
        T = pos.shape[0]
        first = None
        for t in range(STUCK_WINDOW, T):
            anchor = pos[t - STUCK_WINDOW]
            disp = np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - anchor, axis=1).max()
            if disp <= STUCK_RADIUS:
                first = t; break
        if first is not None:
            onsets.append({"event_step": e["event_step"], "window_trip": first,
                           "physical_onset": max(0, first - STUCK_WINDOW)})
    a5["stuck_physical_onset"] = {
        "n": len(onsets),
        "median_window_trip": float(np.median([o["window_trip"] for o in onsets])) if onsets else None,
        "median_physical_onset": float(np.median([o["physical_onset"] for o in onsets])) if onsets else None,
        "median_trip_minus_physical": float(np.median([o["window_trip"] - o["physical_onset"] for o in onsets])) if onsets else None,
        "note": "physical stall onset = window_trip - 60 (the window anchor); event_step lags the true stall by up to the 60-step window."}
    rep["A5_leadtime"] = a5

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_decomposition_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(by, a1, sthr, rthr, goal)
    _print(rep)
    return 0


def _figs(by, a1, sthr, rthr, goal):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=140, sharey=True)
    for ax, cls in zip(axes, ("collision", "stuck")):
        for name, st in (("SCOD", "o-"), ("2-member", "s--"), ("residual", "^:")):
            if cls == "collision":
                ax.plot(LEADS, [a1[name]["lead_collision"][k]["auroc"] for k in LEADS], st, label=name)
        ax.axhline(0.5, color="0.6", lw=1, ls=":"); ax.set_xlabel("lead K"); ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{cls}: collision lead-K AUROC" if cls == "collision" else "stuck (see episode AUROC)")
    axes[0].set_ylabel("AUROC"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage2_lead_auroc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=140)
    col = {"goal": "#7f7f7f", "stuck": "#1f77b4", "collision": "#d62728"}
    for o in ("goal", "stuck", "collision"):
        xs = [emax(e, "scod_epistemic") for e in by.get(o, [])]
        ys = [emax(e, "cbf_violation_safe") for e in by.get(o, [])]
        xs = [v for v in xs if v is not None]; ys = [v for v in ys if v is not None]
        ax.scatter(xs, ys, s=10, alpha=0.5, c=col[o], edgecolors="none", label=f"{o} (n={len(xs)})")
    ax.axvline(sthr, color="0.3", ls="--", lw=1); ax.axhline(rthr, color="0.3", ls="--", lw=1)
    ax.set_xlabel("episode-max SCOD"); ax.set_ylabel("episode-max residual cbf_violation_safe")
    ax.set_title("2x2 axes (episode-max); dashed = goal-class medians")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage2_quadrant_scatter.png"); plt.close(fig)


def _print(rep):
    print("counts:", rep["counts"], "| goal ref:", rep["n_goal_reference"])
    print("A1 association (episode AUROC [95% CI], n_pos):")
    for name in SIGNALS:
        for cls in ("collision", "stuck"):
            d = rep["A1_association"][name][cls]
            print(f"  {name:9s} {cls:9s}: {d['auroc']:.4f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] (n_pos={d['n_pos']}, n_goal={d['n_neg']})")
    print("A2 independence (pearson):", {k: round(v["pearson"], 3) for k, v in rep["A2_independence"].items()})
    print("A3 differential gap (collision-vs-goal − stuck-vs-goal):", {k: round(v["gap_c_minus_s"], 3) for k, v in rep["A3_differential"].items()})
    q = rep["A4_quadrant"]
    print(f"A4 quadrant thresholds: SCOD {q['scod_threshold']:.2f}, residual {q['residual_threshold']:.3f} (degenerate={q['residual_threshold_degenerate']})")
    print("  quadrant x outcome:", q["quadrant_x_outcome"])
    print("  quadrant x stuck-mechanism:", q["quadrant_x_stuck_mechanism"])
    print("A5 collision actionable lead (median steps before event):",
          {k: v["median_steps_before_event"] for k, v in rep["A5_leadtime"]["collision_actionable_lead"].items()})
    print("A5 stuck physical onset:", {k: rep["A5_leadtime"]["stuck_physical_onset"][k] for k in ("median_window_trip", "median_physical_onset", "median_trip_minus_physical")})


if __name__ == "__main__":
    raise SystemExit(main())
