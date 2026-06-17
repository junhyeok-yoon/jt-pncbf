"""v2.2.0 Stage 2 — residual lead-time & physical intervenability (per-step, read-only).

Decisive temporal question for the residual-only classifier (tau_det for success/failure,
tau_type=49.9 for stuck/collision): does the RUNNING residual reach each threshold early enough,
at a usable operating point, that intervention has time AND physical room? Uses the PER-STEP
residual series (not episode-max). Read-only; deterministic. Reuses the Stage-2 loading + signal
definitions. Parts: A collision detection-lead vs type-lead (+gap); B physical avoidability at
detection; C stuck lead vs physical stall onset (+type spike); D SCOD-vs-residual earliness;
E verdict.

Sustained-crossing rule (stated): the lead is `event - e*`, where `e*` is the earliest step such
that the running residual stays >= threshold for ALL steps in [e*, event] (i.e. it remains above
through impact / the reference step). Lead is 0 if the signal is below threshold at the reference
step. For stuck (no impact freeze) we use the FIRST crossing over the active trajectory, since the
reference is the physical stall onset.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_leadtime.py
"""

from __future__ import annotations

import csv
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

import stage2_decomposition as S2  # emax, event_action_step, auc  # noqa: E402

OUT = S2.OUT
EPISODE_DIR = OUT / "episodes"
LABELS = OUT / "stage2_failure_labels.csv"
U_MAX = 2.0
DT = 0.05
STUCK_WINDOW = 60
STUCK_RADIUS = 0.10
ACT_MIN, ACT_HI = 11, 29
RES = "cbf_violation_safe"
SCOD = "scod_epistemic"


def load():
    labels = {int(r["episode_idx"]): r for r in csv.DictReader(LABELS.open())} if LABELS.exists() else {}
    eps = []
    for path in sorted(EPISODE_DIR.glob("ep_*.npz")):
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"]); n = int(len(z["h"])); idx = int(path.stem.split("_")[1])
        eps.append({
            "idx": idx, "outcome": outcome,
            "event_action": S2.event_action_step(outcome, int(md["event_step"]), n),
            "active_steps": int(md.get("active_steps", n)), "n": n,
            RES: np.asarray(z[RES], float), SCOD: np.asarray(z[SCOD], float),
            "clearance": np.asarray(z["clearance"], float),
            "closing_speed": np.asarray(z["closing_speed"], float),
            "brake_margin": np.asarray(z["brake_margin"], float),
            "states": np.asarray(z["states"], float),
            "mechanism": labels.get(idx, {}).get("mechanism", "n/a"),
            "layer": labels.get(idx, {}).get("layer", "n/a")})
    return eps


def sustained_lead(series, ref, thr):
    """lead = ref - e*, e* = earliest step from which series stays >= thr through ref. (0, None) if off at ref."""
    if ref < 0 or ref >= series.size or series[ref] < thr:
        return 0, None
    j = ref
    while j - 1 >= 0 and series[j - 1] >= thr:
        j -= 1
    return ref - j, j


def first_crossing(series, hi, thr):
    idx = np.where(series[:hi] >= thr)[0]
    return int(idx[0]) if idx.size else None


def physical_onset(states):
    pos = states[:, :2]; T = pos.shape[0]
    for t in range(STUCK_WINDOW, T):
        if np.linalg.norm(pos[t - STUCK_WINDOW:t + 1] - pos[t - STUCK_WINDOW], axis=1).max() <= STUCK_RADIUS:
            return max(0, t - STUCK_WINDOW), t
    return None, None


def qstats(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    return {"n": int(a.size), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75))}


def main():
    eps = load()
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("collision", "stuck", "goal", "timeout")}
    rep = {"counts": {o: len(by[o]) for o in by}}

    def emax_arr(c, k): return np.array([S2.emax(e, k) for e in by[c]], float)
    gr = emax_arr("goal", RES); gs = emax_arr("goal", SCOD)
    check = {}
    for nm, k in (("SCOD", SCOD), ("residual", RES)):
        for cls in ("collision", "stuck"):
            pos = emax_arr(cls, k)
            check[f"{nm}:{cls}"] = round(S2.auc(np.r_[np.ones(pos.size), np.zeros(gr.size if k == RES else gs.size)],
                                                np.r_[pos, gr if k == RES else gs]), 4)
    rep["self_consistency_auroc"] = check

    tau_det_5fpr = float(np.quantile(gr, 0.95, method="higher"))
    tau_det_pos = 1e-12
    cand = np.unique(np.concatenate([emax_arr("collision", RES), emax_arr("stuck", RES)]))
    rc, rs = emax_arr("collision", RES), emax_arr("stuck", RES)
    tau_type = float(max(cand, key=lambda t: 0.5 * ((rc >= t).mean() + (rs < t).mean())))
    scod_5fpr = float(np.quantile(gs, 0.95, method="higher"))
    rep["thresholds"] = {"tau_det_5fpr": tau_det_5fpr, "tau_det_pos": tau_det_pos,
                         "tau_type": tau_type, "scod_5fpr": scod_5fpr}

    # Part A — collision detection-lead vs type-lead
    nC = len(by["collision"])
    partA = {}
    det_steps = {}
    for tname, tau in (("det_5fpr", tau_det_5fpr), ("det_pos", tau_det_pos), ("type_49.9", tau_type)):
        leads, rss = [], {}
        for e in by["collision"]:
            lead, rstart = sustained_lead(e[RES], e["event_action"], tau)
            if lead > 0:
                leads.append(lead); rss[e["idx"]] = rstart
        la = np.array(leads, float)
        partA[tname] = {"threshold": tau, "flagged_at_event_rate": len(leads) / nC, **qstats(la),
                        "frac_all_lead_ge11": float((la >= ACT_MIN).sum()) / nC,
                        "frac_all_lead_ge29": float((la >= ACT_HI).sum()) / nC}
        det_steps[tname] = rss
    # detection-vs-type gap (per collision, both sustained)
    gaps = []
    for e in by["collision"]:
        ld, _ = sustained_lead(e[RES], e["event_action"], tau_det_5fpr)
        lt, _ = sustained_lead(e[RES], e["event_action"], tau_type)
        if ld > 0:
            gaps.append(ld - lt)
    partA["detection_minus_type_gap"] = qstats(gaps)
    rep["partA_collision_leadtime"] = partA

    # Part B — physical avoidability at the det_5fpr detection step (lead>=11)
    eidx = {e["idx"]: e for e in by["collision"]}
    in_time = [(idx, rs) for idx, rs in det_steps["det_5fpr"].items()
               if (e := eidx[idx]) is not None and (e["event_action"] - rs) >= ACT_MIN]
    avoid_decel = brake_pos = 0
    cl, cs, bm, reqd = [], [], [], []
    for idx, rs in in_time:
        e = eidx[idx]
        clv, csv_, bmv = float(e["clearance"][rs]), float(e["closing_speed"][rs]), float(e["brake_margin"][rs])
        cl.append(clv); cs.append(csv_); bm.append(bmv)
        req = csv_ * csv_ / (2.0 * max(clv, 1e-9))    # required decel to stop within clearance
        reqd.append(req if clv > 0 else np.inf)
        if clv > 0 and req <= U_MAX:
            avoid_decel += 1
        if bmv > 0:
            brake_pos += 1
    rep["partB_intervenability"] = {
        "criterion": f"avoidable iff clearance>0 and required_decel=closing^2/(2*clearance) <= u_max={U_MAX} "
                     f"(equiv brake_margin>0); double-integrator, dt={DT}",
        "n_flagged_in_time_ge11": len(in_time),
        "frac_decel_feasible": avoid_decel / len(in_time) if in_time else None,
        "frac_brake_margin_pos": brake_pos / len(in_time) if in_time else None,
        "det_clearance_median": float(np.median(cl)) if cl else None,
        "det_closing_speed_median": float(np.median(cs)) if cs else None,
        "det_required_decel_median": float(np.median([r for r in reqd if np.isfinite(r)])) if reqd else None,
        "preventable_frac_of_all283": (avoid_decel / nC)}

    # Part C — stuck detection vs physical onset + type spike
    onsets = {e["idx"]: physical_onset(e["states"])[0] for e in by["stuck"]}
    n_onset = sum(1 for v in onsets.values() if v is not None)
    partC = {}
    for tname, tau in (("det_5fpr", tau_det_5fpr), ("det_pos", tau_det_pos)):
        flagged, leads, pos_lead = 0, [], 0
        for e in by["stuck"]:
            ph = onsets[e["idx"]]
            if ph is None:
                continue
            fc = first_crossing(e[RES], min(e["active_steps"], e["n"]), tau)
            if fc is None:
                continue
            flagged += 1; lead = ph - fc; leads.append(lead)
            if lead > 0:
                pos_lead += 1
        partC[tname] = {"threshold": tau, "n_stuck_with_onset": n_onset,
                        "flag_rate": flagged / n_onset if n_onset else 0.0,
                        **{f"lead_{k}": v for k, v in qstats(leads).items()},
                        "frac_positive_lead": pos_lead / flagged if flagged else None}
    spike = sum(1 for e in by["stuck"] if (e[RES][:min(e["active_steps"], e["n"])] >= tau_type).any())
    partC["running_residual_ge_tau_type"] = {"count": spike, "frac_of_stuck": spike / len(by["stuck"])}
    rep["partC_stuck_leadtime"] = partC

    # Part D — SCOD vs residual earliness (subset both flag)
    # collision: both sustained-flagged at impact (5% FPR)
    both_c = []
    for e in by["collision"]:
        ls, _ = sustained_lead(e[SCOD], e["event_action"], scod_5fpr)
        lr, _ = sustained_lead(e[RES], e["event_action"], tau_det_5fpr)
        if ls > 0 and lr > 0:
            both_c.append((ls, lr))
    # stuck: both first-cross over active traj (5% FPR); lead vs onset
    both_s = []
    for e in by["stuck"]:
        ph = onsets[e["idx"]]
        if ph is None:
            continue
        hi = min(e["active_steps"], e["n"])
        fcs = first_crossing(e[SCOD], hi, scod_5fpr); fcr = first_crossing(e[RES], hi, tau_det_5fpr)
        if fcs is not None and fcr is not None:
            both_s.append((ph - fcs, ph - fcr))
    rep["partD_scod_vs_residual"] = {
        "collision_both_flagged_n": len(both_c),
        "collision_scod_lead_median": float(np.median([a for a, _ in both_c])) if both_c else None,
        "collision_residual_lead_median": float(np.median([b for _, b in both_c])) if both_c else None,
        "collision_scod_minus_residual_median": float(np.median([a - b for a, b in both_c])) if both_c else None,
        "stuck_both_flagged_n": len(both_s),
        "stuck_scod_lead_median": float(np.median([a for a, _ in both_s])) if both_s else None,
        "stuck_residual_lead_median": float(np.median([b for _, b in both_s])) if both_s else None,
        "stuck_scod_minus_residual_median": float(np.median([a - b for a, b in both_s])) if both_s else None}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_leadtime_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _figs(by, tau_det_5fpr, tau_type, scod_5fpr)
    _print(rep)
    return 0


def _figs(by, tau_det, tau_type, scod_5fpr):
    # aligned per-step residual trajectories (sample collisions + stuck)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    for e in by["collision"][:40]:
        ea = e["event_action"]; lo = max(0, ea - 60)
        axes[0].plot(np.arange(lo - ea, 1), e[RES][lo:ea + 1], color="#d62728", alpha=0.25, lw=0.8)
    axes[0].axhline(tau_type, color="k", ls="--", lw=1, label="tau_type 49.9")
    axes[0].axhline(tau_det, color="0.5", ls=":", lw=1, label="tau_det 5%FPR")
    axes[0].set_title("collision: running residual aligned to impact (40 samples)")
    axes[0].set_xlabel("steps before impact"); axes[0].set_ylabel("residual"); axes[0].legend(frameon=False, fontsize=8)
    for e in by["stuck"][:40]:
        ph = physical_onset(e["states"])[0]
        if ph is None:
            continue
        lo = max(0, ph - 60); hi = min(e["n"] - 1, ph + 20)
        axes[1].plot(np.arange(lo - ph, hi - ph + 1), e[RES][lo:hi + 1], color="#1f77b4", alpha=0.2, lw=0.8)
    axes[1].axhline(tau_type, color="k", ls="--", lw=1); axes[1].axhline(tau_det, color="0.5", ls=":", lw=1)
    axes[1].axvline(0, color="0.3", lw=1, label="physical stall onset")
    axes[1].set_title("stuck: running residual aligned to physical onset (40 samples)")
    axes[1].set_xlabel("steps relative to physical onset"); axes[1].set_ylabel("residual"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage2_leadtime_trajectories.png"); plt.close(fig)

    # collision detection-lead vs type-lead distributions
    det_l, typ_l = [], []
    for e in by["collision"]:
        ld, _ = sustained_lead(e[RES], e["event_action"], tau_det); lt, _ = sustained_lead(e[RES], e["event_action"], tau_type)
        if ld > 0:
            det_l.append(ld)
        if lt > 0:
            typ_l.append(lt)
    fig, ax = plt.subplots(figsize=(7, 4.3), dpi=140)
    ax.hist(det_l, bins=range(0, 40, 2), alpha=0.55, color="#d62728", label=f"detection (tau_det), n={len(det_l)}")
    ax.hist(typ_l, bins=range(0, 40, 2), alpha=0.55, color="#7f0000", label=f"type (49.9), n={len(typ_l)}")
    ax.axvline(ACT_MIN, color="0.3", ls="--", lw=1, label="lead=11")
    ax.set_xlabel("sustained lead before impact (steps)"); ax.set_ylabel("collisions")
    ax.set_title("Collision: detection-lead vs type-decision-lead (residual)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage2_leadtime_det_vs_type.png"); plt.close(fig)


def _print(rep):
    print("counts:", rep["counts"])
    print("self-consistency:", rep["self_consistency_auroc"])
    print("thresholds:", {k: (round(v, 3) if v > 1e-6 else v) for k, v in rep["thresholds"].items()})
    print("\nPart A collision lead (residual):")
    for tn, d in rep["partA_collision_leadtime"].items():
        if tn == "detection_minus_type_gap":
            print(f"  GAP det-minus-type: median {d['median']} [p25 {d['p25']}, p75 {d['p75']}]"); continue
        print(f"  {tn:10s} thr={d['threshold']:.3g}: flagged@event={d['flagged_at_event_rate']:.3f} "
              f"lead med={d['median']} [p25 {d['p25']},p75 {d['p75']}] frac(all>=11)={d['frac_all_lead_ge11']:.3f} "
              f"frac(all>=29)={d['frac_all_lead_ge29']:.3f}")
    b = rep["partB_intervenability"]
    print(f"\nPart B (det_5fpr, lead>=11): n={b['n_flagged_in_time_ge11']} decel-feasible={b['frac_decel_feasible']} "
          f"brake_margin>0={b['frac_brake_margin_pos']} preventable_frac_all283={b['preventable_frac_of_all283']:.3f} "
          f"(det clearance med={b['det_clearance_median']}, closing med={b['det_closing_speed_median']})")
    print("\nPart C stuck (residual) lead vs physical onset:")
    for tn in ("det_5fpr", "det_pos"):
        d = rep["partC_stuck_leadtime"][tn]
        print(f"  {tn}: flag_rate={d['flag_rate']:.3f} lead med={d['lead_median']} frac_positive={d['frac_positive_lead']}")
    sp = rep["partC_stuck_leadtime"]["running_residual_ge_tau_type"]
    print(f"  stuck running residual >= tau_type (would momentarily mistype as collision): {sp['count']} ({sp['frac_of_stuck']:.3f})")
    d = rep["partD_scod_vs_residual"]
    print("\nPart D SCOD vs residual (subset both flag):")
    print(f"  collision both n={d['collision_both_flagged_n']}: SCOD lead med={d['collision_scod_lead_median']} "
          f"residual lead med={d['collision_residual_lead_median']} SCOD-minus-residual={d['collision_scod_minus_residual_median']}")
    print(f"  stuck both n={d['stuck_both_flagged_n']}: SCOD lead med={d['stuck_scod_lead_median']} "
          f"residual lead med={d['stuck_residual_lead_median']} SCOD-minus-residual={d['stuck_scod_minus_residual_median']}")


if __name__ == "__main__":
    raise SystemExit(main())
