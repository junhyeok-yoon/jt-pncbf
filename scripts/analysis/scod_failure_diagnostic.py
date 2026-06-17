"""v2.2.0 — SCOD-on-real-failures diagnostic (read-only).

Decides whether the grade-2 SCOD epistemic signal captures REAL collisions (vs only
out-of-manifold inputs), by measuring SCOD on the Stage-0 dissection NPZ set with the SAVED
Fisher's online closure. Reuses the Fisher; does not rebuild it. Read-only on the NPZs and the
Fisher; writes only this report's figures and (the caller writes) the report.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/scod_failure_diagnostic.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage1_scod_build as B  # SCOD online closure + checked_auc + device_dtype  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

DISSECT = REPO_ROOT / "data/diagnostics/v2.0.1__20260529-171057__failure_dissect_n2500_seed45678_20260529-194915"
EPISODE_DIR = DISSECT / "episodes"
FISHER_PATH = REPO_ROOT / "data/diagnostics/v2.2.0_stage1_scod/scod_fisher.pt"
OUT_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_scod_diagnostic"

LEADS = [1, 3, 5, 10, 15, 30]
PRE_WINDOW = 30
MEAN_TOL = 1.0e-6               # CPU float32 reconstruction noise floor (Stage-0 got ~3e-8 on CPU)
WORLD_LIM = 4.0
R_MAX = 0.80
V_MAX = 2.5
NEAR_TAIL = 0.10

# Manifold reference quantiles (deployed-policy rollout, from the Stage-1 build measurement).
MANIFOLD = {"min": -0.055, "p1": 0.141, "p5": 0.215, "median": 1.16, "frac_lt_010": 0.0053}

SIGNALS = {"SCOD": +1.0, "2-member |v1-v2|": +1.0, "residual cbf_violation_safe": +1.0}


def load_art() -> dict:
    f = torch.load(FISHER_PATH, map_location="cpu", weights_only=False)
    return {"sigma_out": f["sigma_out"], "epsilon": f["epsilon"],
            "members": [{"eigvals": m["eigvals"], "eigvecs": m["eigvecs"], "k": m["k"]} for m in f["members"]]}


def event_action_step(outcome: str, event_step: int, n: int) -> int:
    if outcome in {"collision", "goal", "oob"} and event_step > 0:
        return max(0, min(event_step - 1, n - 1))
    if event_step < 0:
        return n - 1
    return max(0, min(event_step, n - 1))


def load_episodes(deployed, system, art, device, dtype):
    episodes = []
    max_mismatch = 0.0
    for path in sorted(EPISODE_DIR.glob("*.npz")):
        if not path.name.startswith(("failure_", "goal_control_")):
            continue
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        outcome = str(md["outcome"])
        event_step = int(md["event_step"])
        active_steps = int(md.get("active_steps", len(z["h"])))
        states = torch.as_tensor(np.asarray(z["states"], dtype=np.float64), device=device, dtype=dtype)
        x = states[:-1]
        n = x.shape[0]
        scene = SimpleNamespace(
            goal=torch.as_tensor(np.asarray(z["goal"]), device=device, dtype=dtype),
            obstacle_centers=torch.as_tensor(np.asarray(z["obstacle_centers"]), device=device, dtype=dtype),
            obstacle_radii=torch.as_tensor(np.asarray(z["obstacle_radii"]), device=device, dtype=dtype),
            obstacle_active=torch.as_tensor(np.asarray(z["obstacle_active"]), device=device, dtype=torch.bool),
        )
        with torch.no_grad():
            obs = system.observation(x, scene)
            members = deployed.value_all(obs)               # [n, 2] clipped
            deployed_h = members.mean(dim=1).cpu().numpy()
            scod = B.scod_scalar(deployed, obs, art)[0].cpu().numpy()
        twomember = torch.abs(members[:, 0] - members[:, 1]).cpu().numpy()
        max_mismatch = max(max_mismatch, float(np.max(np.abs(deployed_h - np.asarray(z["h"], dtype=np.float64)))))
        ea = event_action_step(outcome, event_step, n)
        episodes.append({
            "idx": int(md.get("episode_idx", -1)) if "episode_idx" in md else _idx_from_name(path),
            "outcome": outcome, "event_action": ea, "active_steps": active_steps, "n": n,
            "scod": scod, "twomember": twomember,
            "residual": np.asarray(z["cbf_violation_safe"], dtype=np.float64),
            "clearance": np.asarray(z["clearance"], dtype=np.float64),
            "states": np.asarray(z["states"], dtype=np.float64),
            "obstacle_radii": np.asarray(z["obstacle_radii"], dtype=np.float64),
            "obstacle_centers": np.asarray(z["obstacle_centers"], dtype=np.float64),
            "obstacle_active": np.asarray(z["obstacle_active"], dtype=bool),
        })
    return episodes, max_mismatch


def _idx_from_name(path: Path) -> int:
    parts = path.stem.split("_")
    return int(parts[2]) if parts[0] == "goal" else int(parts[1])


def safe_idx(ep, i):
    return 0 <= i < ep["n"] and i < ep["active_steps"]


def episode_max(ep, key):
    lo = max(0, ep["event_action"] - PRE_WINDOW + 1)
    hi = min(ep["event_action"] + 1, ep["active_steps"], ep["n"])
    if hi <= lo:
        return None
    return float(np.max(ep[key][lo:hi]))


def lead_value(ep, key, k):
    i = ep["event_action"] - k
    return float(ep[key][i]) if safe_idx(ep, i) else None


def active_series(ep, key):
    hi = min(ep["active_steps"], ep["n"])
    return ep[key][:hi]


def assoc_episode(pos, neg, key):
    labels, scores = [], []
    for lab, grp in ((1, pos), (0, neg)):
        for ep in grp:
            v = episode_max(ep, key)
            if v is not None:
                labels.append(lab); scores.append(v)
    return B.checked_auc(np.array(labels), np.array(scores)), int(sum(labels)), int(len(labels) - sum(labels))


def assoc_lead(pos, neg, key, k):
    labels, scores = [], []
    for lab, grp in ((1, pos), (0, neg)):
        for ep in grp:
            v = lead_value(ep, key, k)
            if v is not None:
                labels.append(lab); scores.append(v)
    return B.checked_auc(np.array(labels), np.array(scores))


def main() -> int:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Run on CPU: matches the NPZ deployed-h (computed on CPU in the dissection) to ~3e-8,
    # reproducing the Stage-0 reconstruction check; dataset is small so CPU is cheap.
    device, dtype = torch.device("cpu"), torch.float32
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fw, config, _ = load_framework_from_checkpoint(B.SECURED_CKPT)
    system = make_system(config)
    deployed = fw.value_net.to(device=device, dtype=dtype).eval()
    art = load_art()

    episodes, max_mismatch = load_episodes(deployed, system, art, device, dtype)
    if max_mismatch > MEAN_TOL:
        raise SystemExit(f"STOP: obs reconstruction mismatch {max_mismatch:.3e} > {MEAN_TOL:.1e}")
    by = {o: [e for e in episodes if e["outcome"] == o] for o in ("collision", "stuck", "goal")}
    report: dict[str, Any] = {"recon_max_mismatch": max_mismatch,
                              "counts": {k: len(v) for k, v in by.items()}}

    # Q1 — association SCOD vs 2-member
    q1 = {"episode": {}, "lead": {}}
    for key in ("SCOD", "2-member |v1-v2|"):
        col, c_np, c_nn = assoc_episode(by["collision"], by["goal"], _k(key))
        stu, s_np, s_nn = assoc_episode(by["stuck"], by["goal"], _k(key))
        q1["episode"][key] = {"collision": col, "stuck": stu, "n_pos_c": c_np, "n_pos_s": s_np, "n_neg": c_nn}
        q1["lead"][key] = {
            "collision": {k: assoc_lead(by["collision"], by["goal"], _k(key), k) for k in LEADS},
            "stuck": {k: assoc_lead(by["stuck"], by["goal"], _k(key), k) for k in LEADS},
        }
    q1["scod_vs_2member_collision_gap"] = q1["episode"]["2-member |v1-v2|"]["collision"] - q1["episode"]["SCOD"]["collision"]
    report["Q1"] = q1
    _plot_lead(q1, OUT_DIR / "q1_lead_auroc.png")

    # Q2 — manifold bucketing of collisions
    report["Q2"] = bucket_collisions(by["collision"])
    _plot_q2(by["collision"], OUT_DIR / "q2_collision_clearance.png")

    # Q3 — deployment (goal-timestep) reference
    report["Q3"] = q3_reference(by, art)
    _plot_q3(by, OUT_DIR / "q3_scod_reference.png")

    # Q4 — independence + low-low under SCOD
    report["Q4"] = q4_decomposition(by)

    (OUT_DIR / "diagnostic_summary.json").write_text(json.dumps(B._jsonable(report), indent=2), encoding="utf-8")
    _print_summary(report)
    return 0


def _k(signal: str) -> str:
    return {"SCOD": "scod", "2-member |v1-v2|": "twomember", "residual cbf_violation_safe": "residual"}[signal]


def bucket_collisions(coll):
    rows = []
    for ep in coll:
        ea = ep["event_action"]
        surf = float(ep["clearance"][ea])
        surf_w = float(np.min(ep["clearance"][max(0, ea - 4): ea + 1]))   # lead-5 window min
        speed = float(np.linalg.norm(ep["states"][ea, 2:4]))
        act = ep["obstacle_active"]
        rmax = float(ep["obstacle_radii"][act].max()) if act.any() else 0.0
        cmax = float(np.abs(ep["obstacle_centers"][act]).max()) if act.any() else 0.0
        out_of_manifold = (rmax > R_MAX) or (cmax > WORLD_LIM) or (speed > V_MAX)
        if out_of_manifold:
            bucket = "b_out_of_manifold"
        elif surf < NEAR_TAIL:
            bucket = "a_in_manifold_near_tail"
        else:
            bucket = "c_typical_clearance"
        rows.append({"idx": ep["idx"], "surf_event": surf, "surf_window_min": surf_w,
                     "speed": speed, "rmax": rmax, "cmax": cmax, "scod_event": float(ep["scod"][ea]),
                     "bucket": bucket})
    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("a_in_manifold_near_tail", "b_out_of_manifold", "c_typical_clearance")}
    surfs = np.array([r["surf_event"] for r in rows])
    return {"n": len(rows), "bucket_counts": counts,
            "surf_event_quantiles": {"min": float(surfs.min()), "p5": float(np.percentile(surfs, 5)),
                                     "median": float(np.median(surfs)), "p95": float(np.percentile(surfs, 95)),
                                     "max": float(surfs.max())},
            "frac_surf_lt_010": float(np.mean(surfs < NEAR_TAIL)),
            "manifold_reference": MANIFOLD, "rows": rows}


def q3_reference(by, art):
    # deployment reference = all active goal-class timesteps (full-rollout states)
    goal_scod = np.concatenate([active_series(e, "scod") for e in by["goal"]])
    # collision near-event timesteps (lead-5 window)
    coll_scod = []
    for e in by["collision"]:
        ea = e["event_action"]
        coll_scod.append(e["scod"][max(0, ea - 4): ea + 1])
    coll_scod = np.concatenate(coll_scod)
    labels = np.concatenate([np.ones_like(coll_scod), np.zeros_like(goal_scod)])
    scores = np.concatenate([coll_scod, goal_scod])
    auc = B.checked_auc(labels, scores)
    return {"reference": "goal-class deployment timesteps (full rollout)",
            "goal_scod_median": float(np.median(goal_scod)), "goal_scod_p95": float(np.percentile(goal_scod, 95)),
            "collision_near_scod_median": float(np.median(coll_scod)),
            "collision_vs_goal_timestep_auroc": auc,
            "n_goal_steps": int(goal_scod.size), "n_collision_near_steps": int(coll_scod.size)}


def q4_decomposition(by):
    from scipy.stats import pearsonr, spearmanr
    mechanisms = _load_csv_map(DISSECT / "episode_summary.csv", "mechanism")
    layers = _load_csv_map(DISSECT / "layer_attribution.csv", "layer")

    indep = {}
    for o in ("collision", "stuck", "goal"):
        s = np.concatenate([active_series(e, "scod") for e in by[o]])
        r = np.concatenate([active_series(e, "residual") for e in by[o]])
        indep[o] = {"n": int(s.size),
                    "pearson": float(pearsonr(s, r)[0]) if s.size > 2 else float("nan"),
                    "spearman": float(spearmanr(s, r)[0]) if s.size > 2 else float("nan")}

    gs = np.concatenate([active_series(e, "scod") for e in by["goal"]])
    gr = np.concatenate([active_series(e, "residual") for e in by["goal"]])
    scod_med, res_med = float(np.median(gs)), float(np.median(gr))
    lowlow = {"scod_goal_median": scod_med, "residual_goal_median": res_med, "classes": {}}
    for o in ("collision", "stuck"):
        eps_with = []
        ll_ts = tot_ts = 0
        for e in by[o]:
            s = active_series(e, "scod"); r = active_series(e, "residual")
            mask = (s <= scod_med) & (r <= res_med)
            tot_ts += mask.size; ll_ts += int(mask.sum())
            if mask.any():
                eps_with.append(e["idx"])
        mech = _count([mechanisms.get(i, "?") for i in eps_with])
        lay = _count([layers.get(i, "?") for i in eps_with])
        lowlow["classes"][o] = {"episodes_with_lowlow_step": len(eps_with), "total_episodes": len(by[o]),
                                "lowlow_timesteps": ll_ts, "total_timesteps": tot_ts,
                                "mechanism_counts": mech, "layer_counts": lay}
    return {"independence": indep, "lowlow": lowlow}


def _load_csv_map(path, field):
    out = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        out[int(row["episode_idx"])] = row.get(field, "?")
    return out


def _count(items):
    c = {}
    for it in items:
        c[it] = c.get(it, 0) + 1
    return c


def _plot_lead(q1, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=140, sharey=True)
    for ax, cls in zip(axes, ("collision", "stuck")):
        for key, style in (("SCOD", "o-"), ("2-member |v1-v2|", "s--")):
            ys = [q1["lead"][key][cls][k] for k in LEADS]
            ax.plot(LEADS, ys, style, label=key)
        ax.axhline(0.5, color="0.6", lw=1, ls=":")
        ax.set_title(f"{cls} vs goal"); ax.set_xlabel("lead K"); ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("AUROC"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _plot_q2(coll, path):
    surf = np.array([ep["clearance"][ep["event_action"]] for ep in coll])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    ax.hist(surf, bins=20, color="#d62728", alpha=0.8, label=f"collision event surface dist (n={len(surf)})")
    ax.axvline(NEAR_TAIL, color="0.2", ls="--", lw=1, label="near-tail threshold 0.10")
    ax.axvline(MANIFOLD["median"], color="#1f77b4", ls="-", lw=1.2, label=f"manifold median {MANIFOLD['median']}")
    ax.axvline(MANIFOLD["p5"], color="#1f77b4", ls=":", lw=1, label=f"manifold p5 {MANIFOLD['p5']}")
    ax.set_xlabel("nearest-obstacle surface distance at the executed-action step")
    ax.set_ylabel("collision episodes")
    ax.set_title("Q2: real collisions sit in the in-manifold near-obstacle tail")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _plot_q3(by, path):
    goal = np.concatenate([active_series(e, "scod") for e in by["goal"]])
    coll = np.concatenate([e["scod"][max(0, e["event_action"] - 4): e["event_action"] + 1] for e in by["collision"]])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    bins = np.linspace(np.log10(max(1e-3, min(goal.min(), coll.min()))), np.log10(max(goal.max(), coll.max())), 40)
    ax.hist(np.log10(np.clip(goal, 1e-3, None)), bins=bins, density=True, alpha=0.6, color="#7f7f7f", label=f"goal timesteps (n={goal.size})")
    ax.hist(np.log10(np.clip(coll, 1e-3, None)), bins=bins, density=True, alpha=0.6, color="#d62728", label=f"collision near-event (n={coll.size})")
    ax.set_xlabel("log10 SCOD epistemic"); ax.set_ylabel("density")
    ax.set_title("Q3: SCOD vs the deployment (goal-timestep) reference")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _print_summary(r):
    print(f"recon max mismatch: {r['recon_max_mismatch']:.3e} (gate {MEAN_TOL:.1e})  counts={r['counts']}")
    print("Q1 episode AUROC (collision/stuck):")
    for key in ("SCOD", "2-member |v1-v2|"):
        e = r["Q1"]["episode"][key]
        print(f"  {key}: collision={e['collision']:.4f} stuck={e['stuck']:.4f}")
    print(f"  SCOD vs 2-member collision gap: {r['Q1']['scod_vs_2member_collision_gap']:+.4f}")
    print("Q2 collision manifold buckets:", r["Q2"]["bucket_counts"],
          "| surf<0.10 frac:", round(r["Q2"]["frac_surf_lt_010"], 3),
          "| surf median:", round(r["Q2"]["surf_event_quantiles"]["median"], 3))
    q3 = r["Q3"]
    print(f"Q3 deployment-ref: collision-vs-goal-timestep AUROC={q3['collision_vs_goal_timestep_auroc']:.4f} "
          f"(goal med {q3['goal_scod_median']:.2f}, coll med {q3['collision_near_scod_median']:.2f})")
    print("Q4 indep (pearson):", {k: round(v["pearson"], 3) for k, v in r["Q4"]["independence"].items()})
    for o in ("collision", "stuck"):
        c = r["Q4"]["lowlow"]["classes"][o]
        print(f"  Q4 low-low {o}: {c['episodes_with_lowlow_step']}/{c['total_episodes']} eps; layers={c['layer_counts']}")


if __name__ == "__main__":
    raise SystemExit(main())
