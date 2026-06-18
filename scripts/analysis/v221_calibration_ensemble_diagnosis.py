"""v2.2.1 — calibration & ensemble post-hoc diagnostic (READ-ONLY, deterministic, seeded).

Question (complements the remaining-collision diagnosis, which classified WHAT the collisions are):
WHY did V_S let them happen? An analytic HOCBF is also myopic yet collides zero because it is
structurally CONSERVATIVE (over-estimates danger). Collisions arise when V_S UNDER-estimates h (labels
an unsafe state safe) at the states that go on to collide, so the HardNet filter engages too late.
Two mechanisms with OPPOSITE fixes:
  (i)  net-level under-training / data gap (rare collision-precursor states stay under-estimated), or
  (ii) ENSEMBLE aggregation washes out danger: deployed h = MEAN of 2 clamped members, so one member
       flagging danger (h>0) is cancelled by the other under-estimating (h<0); target=min is also a
       train/deploy statistic mismatch.

Analyses (single rollout per checkpoint, cached; reuses the v2.2.0 harness jt_failure_dissect for
collision identification + geometry so definitions match prior diagnoses):
  D1 under-estimation at collision states: deployed-h trajectory pre-impact; first-flag lead time;
     fraction still declared SAFE (h<0) at the LAST brake-feasible step; matched deployed-h vs
     non-colliding states at the same (clearance, inward-speed) cell.
  D2 ensemble decomposition at the last brake-feasible (critical) state: per-member h0/h1, mean/min/max;
     mean-washout fraction (mean<0 but max>=0), both-members-wrong fraction (both<0), member
     disagreement collision-vs-safe, and the would-be-flagged fraction for deploy={mean,min,max}.
  D3 data-gap vs bias: are collision-precursor (clearance,speed) cells under-/over-/typically-visited
     on-pool (proxy for what max-over-time TD trained on)?

Deployed=mean, target=min, max are mean/min/max of the SAME clamped per-member values (value_net.value_all),
so the counterfactual "if deployed were min/max instead of mean" is exact.

Run (GPU; ~5-10 min, both checkpoints):
  python scripts/analysis/v221_calibration_ensemble_diagnosis.py
Smoke (CPU, tiny; SOTA for both tags):
  python scripts/analysis/v221_calibration_ensemble_diagnosis.py --smoke 4 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts/analysis"), str(REPO_ROOT / "scripts/verification")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jt_failure_dissect as D  # noqa: E402  (_episode_trace, _event_action_index)
from src.eval.build_pools import EvaluationPool, load_pool  # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402
from src.eval.run_full import _load_framework  # noqa: E402
from src.frameworks.jt_pncbf.train import make_system  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402

DEFAULT_TREATED = REPO_ROOT / "data/v2.2.1__20260618-001001__seed42/checkpoints/best.pt"
DEFAULT_SOTA = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
DEFAULT_POOL = REPO_ROOT / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl"
CACHE_ROOT = REPO_ROOT / "data/diagnostics/v2.2.1_calibration_ensemble"
DOC = REPO_ROOT / "docs/versions/v2.2.1/calibration_ensemble_diagnosis.md"

EVAL_BATCH = 250
SEED = 20260618          # same seed as the remaining-collision diagnosis -> identical collision set
PRE_WINDOW = 20          # K steps before impact
U_MAX = 2.0
COLLISION = "collision"
# (clearance, inward-speed) bins for matched comparison + visitation density
CLEAR_EDGES = [0.0, 0.10, 0.20, 0.35, 0.50, 1.0, 1e9]
SPEED_EDGES = [0.0, 0.30, 0.60, 1.0, 1.5, 2.5, 1e9]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ================================================================================================== #
# Rollout cache (single pass per checkpoint): deployed h + BOTH members + geometry, idempotent        #
# ================================================================================================== #
def build_cache(tag: str, ckpt_path: Path, pool_path: Path, device: torch.device, n: int | None) -> Path:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out = CACHE_ROOT / tag
    ep_dir = out / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    if index_path.exists():
        rows = list(csv.DictReader(index_path.open()))
        if n is None or len(rows) == n:
            print(f"[{tag}] cache present ({len(rows)} episodes) -> reuse")
            return out

    fw, config, checkpoint = _load_framework(ckpt_path)
    dtype = torch.float32
    system = make_system(config)
    fw.system = system
    fw.value_net.to(device=device, dtype=dtype).eval()
    fw.policy_net.to(device=device, dtype=dtype).eval()
    value_net = fw.value_net
    h_fn = make_h_fn(value_net, system, use_target=False)

    pool = load_pool(Path(pool_path))
    scenes = pool.scenes if n is None else pool.scenes[:n]
    use_pool = EvaluationPool(name=f"v221_calib_{tag}", system=pool.system,
                              n_scenes=len(scenes), seed=getattr(pool, "seed", SEED), scenes=scenes)
    result = evaluate(fw, use_pool, config, mode="diagnostic_v221_calibration",
                      step=int(checkpoint.get("step", 0)), ckpt_name=ckpt_path.name,
                      include_lqr_baseline=False, eval_batch_size=min(EVAL_BATCH, len(scenes)))

    index_rows = []
    for idx, ep in enumerate(result.trajectories):
        outcome = ep.filtered_outcome
        trace = D._episode_trace(framework=fw, h_fn=h_fn, config=config, scene=ep.scene,
                                 result=ep.filtered, outcome=outcome,
                                 event_step=int(ep.filtered_event_step))
        x = ep.filtered.states[:-1, 0, :].detach()
        with torch.no_grad():
            obs = system.observation(x, ep.scene)
            members = value_net.value_all(obs)                      # [T, n_vs] clamped per-member
        m = members.detach().cpu().numpy()
        np.savez_compressed(
            ep_dir / f"ep_{idx:05d}_{outcome}.npz",
            outcome=np.asarray(outcome), event_step=np.asarray(int(ep.filtered_event_step)),
            active_steps=np.asarray(int(trace["active_steps"])),
            h=np.asarray(trace["h"], np.float32),                   # deployed (mean of members)
            member0=m[:, 0].astype(np.float32), member1=m[:, 1].astype(np.float32),
            clearance=np.asarray(trace["clearance"], np.float32),
            closing_speed=np.asarray(trace["closing_speed"], np.float32),
            brake_margin=np.asarray(trace["brake_margin"], np.float32),
        )
        index_rows.append({"idx": idx, "outcome": outcome})
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "outcome"]); w.writeheader(); w.writerows(index_rows)
    (out / "eval_row.json").write_text(json.dumps({k: _num(v) for k, v in result.eval_row.items()}, indent=2))
    print(f"[{tag}] rolled out {len(index_rows)} episodes -> {out}")
    return out


def _num(v):
    return float(v) if isinstance(v, (np.floating, np.integer)) else v


def load_cache(tag: str):
    out = CACHE_ROOT / tag
    eps = []
    for path in sorted((out / "episodes").glob("ep_*.npz")):
        z = np.load(path, allow_pickle=True)
        eps.append({
            "idx": int(path.stem.split("_")[1]),
            "outcome": str(z["outcome"]), "event_step": int(z["event_step"]),
            "active_steps": int(z["active_steps"]),
            "h": np.asarray(z["h"], float), "member0": np.asarray(z["member0"], float),
            "member1": np.asarray(z["member1"], float), "clearance": np.asarray(z["clearance"], float),
            "closing_speed": np.asarray(z["closing_speed"], float),
            "brake_margin": np.asarray(z["brake_margin"], float),
        })
    eps.sort(key=lambda e: e["idx"])
    return eps, json.loads((out / "eval_row.json").read_text())


# ================================================================================================== #
# shared helpers
# ================================================================================================== #
def _event_idx(ep) -> int:
    return D._event_action_index({"event_step": ep["event_step"], "h": ep["h"], "outcome": ep["outcome"]})


def _last_brake_feasible(ep, event) -> int | None:
    bm = ep["brake_margin"][:event + 1]
    idx = np.where(bm > 0.0)[0]
    return int(idx[-1]) if idx.size else None


def _first_flag_lead(ep, event):
    """steps before impact that deployed h first reaches >=0 (first danger flag); None if never."""
    seg = ep["h"][:event + 1]
    cr = np.where(seg >= 0.0)[0]
    return (event - int(cr[0])) if cr.size else None


def _cell(clear, speed) -> tuple[int, int]:
    ci = int(np.digitize([clear], CLEAR_EDGES)[0] - 1)
    si = int(np.digitize([speed], SPEED_EDGES)[0] - 1)
    ci = min(max(ci, 0), len(CLEAR_EDGES) - 2)
    si = min(max(si, 0), len(SPEED_EDGES) - 2)
    return ci, si


# ================================================================================================== #
# D1 — under-estimation at collision states
# ================================================================================================== #
def analysis_d1(eps) -> dict:
    coll = [e for e in eps if e["outcome"] == COLLISION]
    leads, never, safe_at_lastbf, n_lastbf = [], 0, 0, 0
    h_at_lastbf = []
    for e in coll:
        ev = _event_idx(e)
        lead = _first_flag_lead(e, ev)
        if lead is None:
            never += 1
        else:
            leads.append(lead)
        lbf = _last_brake_feasible(e, ev)
        if lbf is not None:
            n_lastbf += 1
            h_lbf = float(e["h"][lbf])
            h_at_lastbf.append(h_lbf)
            if h_lbf < 0.0:
                safe_at_lastbf += 1
    # matched deployed-h vs non-colliding states at same (clearance, speed) cell, over pre-impact window
    coll_cells: dict[tuple, list] = {}
    for e in coll:
        ev = _event_idx(e)
        lo = max(0, ev - PRE_WINDOW + 1)
        for t in range(lo, ev + 1):
            coll_cells.setdefault(_cell(e["clearance"][t], e["closing_speed"][t]), []).append(float(e["h"][t]))
    safe_eps = [e for e in eps if e["outcome"] == "goal"]
    safe_cells: dict[tuple, list] = {}
    for e in safe_eps:
        a = e["active_steps"]
        for t in range(min(a, len(e["h"]))):
            safe_cells.setdefault(_cell(e["clearance"][t], e["closing_speed"][t]), []).append(float(e["h"][t]))
    matched = []
    for cell, ch in sorted(coll_cells.items()):
        if cell in safe_cells and len(ch) >= 3 and len(safe_cells[cell]) >= 3:
            matched.append({
                "clearance_bin": f"[{CLEAR_EDGES[cell[0]]:.2f},{CLEAR_EDGES[cell[0]+1]:.2f})",
                "speed_bin": f"[{SPEED_EDGES[cell[1]]:.2f},{SPEED_EDGES[cell[1]+1]:.2f})",
                "n_coll": len(ch), "n_safe": len(safe_cells[cell]),
                "coll_h_mean": float(np.mean(ch)), "safe_h_mean": float(np.mean(safe_cells[cell])),
                "coll_minus_safe": float(np.mean(ch) - np.mean(safe_cells[cell]))})
    return {
        "n_collisions": len(coll),
        "first_flag_lead": {"n_flagged": len(leads), "n_never_flagged": never,
                            "frac_never_flagged": (never / len(coll)) if coll else None,
                            "median_lead": float(np.median(leads)) if leads else None,
                            "p25": float(np.percentile(leads, 25)) if leads else None,
                            "p75": float(np.percentile(leads, 75)) if leads else None},
        "last_brake_feasible": {"n_with_brakeable_step": n_lastbf,
                                "frac_declared_safe_at_last_brakeable": (safe_at_lastbf / n_lastbf) if n_lastbf else None,
                                "median_deployed_h_at_last_brakeable": float(np.median(h_at_lastbf)) if h_at_lastbf else None},
        "matched_cells": matched}


# ================================================================================================== #
# D2 — ensemble decomposition at the last brake-feasible (critical) state
# ================================================================================================== #
def analysis_d2(eps) -> dict:
    coll = [e for e in eps if e["outcome"] == COLLISION]
    crit = []   # per collision: members at last-brake-feasible step
    for e in coll:
        ev = _event_idx(e)
        lbf = _last_brake_feasible(e, ev)
        if lbf is None:
            continue
        m0, m1 = float(e["member0"][lbf]), float(e["member1"][lbf])
        crit.append({"m0": m0, "m1": m1, "mean": 0.5 * (m0 + m1),
                     "min": min(m0, m1), "max": max(m0, m1), "disagree": abs(m0 - m1)})
    n = len(crit)
    if n == 0:
        return {"n_critical": 0}
    mean_safe = [c for c in crit if c["mean"] < 0.0]
    washout = sum(1 for c in mean_safe if c["max"] >= 0.0)            # mean<0 but a member flagged danger
    both_wrong = sum(1 for c in crit if c["m0"] < 0.0 and c["m1"] < 0.0)
    # member disagreement: collision critical states vs safe states
    safe_dis = []
    for e in eps:
        if e["outcome"] != "goal":
            continue
        a = min(e["active_steps"], len(e["member0"]))
        safe_dis.extend(np.abs(e["member0"][:a] - e["member1"][:a]).tolist())
    # counterfactual would-be-flagged-at-last-brakeable for each deploy statistic
    flagged = {stat: float(np.mean([c[stat] >= 0.0 for c in crit])) for stat in ("mean", "min", "max")}
    return {
        "n_critical": n,
        "frac_mean_declared_safe": len(mean_safe) / n,
        "frac_mean_washout_of_n": washout / n,                       # of ALL critical collisions
        "frac_mean_washout_of_mean_safe": (washout / len(mean_safe)) if mean_safe else None,
        "frac_both_members_wrong": both_wrong / n,
        "disagreement_collision": {"median": float(np.median([c["disagree"] for c in crit])),
                                   "mean": float(np.mean([c["disagree"] for c in crit]))},
        "disagreement_safe": {"median": float(np.median(safe_dis)) if safe_dis else None,
                              "mean": float(np.mean(safe_dis)) if safe_dis else None},
        "counterfactual_flag_rate_at_last_brakeable": flagged,
        "member_h_at_critical": {"m0_median": float(np.median([c["m0"] for c in crit])),
                                 "m1_median": float(np.median([c["m1"] for c in crit])),
                                 "mean_median": float(np.median([c["mean"] for c in crit])),
                                 "max_median": float(np.median([c["max"] for c in crit]))}}


# ================================================================================================== #
# D3 — data-gap vs bias: visitation density of collision-precursor cells
# ================================================================================================== #
def analysis_d3(eps) -> dict:
    nc, ns = len(CLEAR_EDGES) - 1, len(SPEED_EDGES) - 1
    visited = np.zeros((nc, ns), float)
    for e in eps:                                       # on-pool visited distribution (all active steps)
        a = min(e["active_steps"], len(e["clearance"]))
        for t in range(a):
            ci, si = _cell(e["clearance"][t], e["closing_speed"][t])
            visited[ci, si] += 1.0
    total = visited.sum()
    dens = visited / total if total > 0 else visited
    visited_cell_vals = dens[dens > 0]
    med_visited_density = float(np.median(visited_cell_vals)) if visited_cell_vals.size else 0.0
    coll = [e for e in eps if e["outcome"] == COLLISION]
    crit_dens, below = [], 0
    cells_seen = {}
    for e in coll:
        ev = _event_idx(e)
        lbf = _last_brake_feasible(e, ev)
        t = lbf if lbf is not None else ev
        ci, si = _cell(e["clearance"][t], e["closing_speed"][t])
        d = float(dens[ci, si])
        crit_dens.append(d)
        cells_seen[(ci, si)] = cells_seen.get((ci, si), 0) + 1
        if d < med_visited_density:
            below += 1
    cell_table = [{"clearance_bin": f"[{CLEAR_EDGES[ci]:.2f},{CLEAR_EDGES[ci+1]:.2f})",
                   "speed_bin": f"[{SPEED_EDGES[si]:.2f},{SPEED_EDGES[si+1]:.2f})",
                   "n_collision_critical": k, "visited_density": float(dens[ci, si])}
                  for (ci, si), k in sorted(cells_seen.items(), key=lambda kv: -kv[1])]
    return {
        "n_collision_critical": len(crit_dens),
        "median_visited_density_of_collision_cells": float(np.median(crit_dens)) if crit_dens else None,
        "median_visited_density_overall": med_visited_density,
        "frac_collision_cells_below_median_visitation": (below / len(crit_dens)) if crit_dens else None,
        "collision_critical_cells": cell_table}


# ================================================================================================== #
def _figs(tag, d1, d2, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # D2 counterfactual flag-rate bar (mean/min/max)
    cf = d2.get("counterfactual_flag_rate_at_last_brakeable")
    if cf:
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
        ax.bar(["min\n(=target)", "mean\n(=deployed)", "max\n(most cons.)"],
               [cf["min"], cf["mean"], cf["max"]], color=["#7f7f7f", "#d62728", "#2ca02c"])
        ax.set_ylabel("frac flagged danger at last brake-feasible step")
        ax.set_ylim(0, 1)
        ax.set_title(f"{tag}: would-be-flagged by deploy statistic")
        fig.tight_layout(); fig.savefig(out_dir / f"{tag}_d2_counterfactual.png"); plt.close(fig)


def write_doc(meta, results):
    L = ["# v2.2.1 — calibration & ensemble diagnosis (why V_S let collisions happen)\n"]
    L.append(f"_pool_ `{meta['pool']}` (N={meta['n']}) | seed={SEED} | device={meta['device']}  ")
    L.append(f"_TREATED_ `{meta['treated_ckpt']}` (step {meta['treated_step']})  ")
    L.append(f"_SOTA_ `{meta['sota_ckpt']}` (step {meta['sota_step']})\n")
    L.append("Read-only; reuses `jt_failure_dissect._episode_trace`/`_event_action_index` and the same "
             "rollout seed as the remaining-collision diagnosis, so the collision set matches.\n")
    L.append("Deployed h = MEAN of 2 clamped members; target = MIN; MAX = most danger-conservative "
             "(all from `value_net.value_all`). safe={h<=0}, danger={h>0}.\n")
    for tag in ("TREATED", "SOTA"):
        r = results[tag]; d1, d2, d3 = r["D1"], r["D2"], r["D3"]
        L.append(f"\n## {tag}\n")
        L.append(f"collisions: {d1['n_collisions']} | eval_row collision={r['eval_row'].get('collision')} cps={r['eval_row'].get('cps')}\n")
        fl = d1["first_flag_lead"]; lb = d1["last_brake_feasible"]
        L.append("### D1 under-estimation at collision states")
        L.append(f"- first danger-flag (deployed h>=0) lead: median {fl['median_lead']} steps "
                 f"[p25 {fl['p25']}, p75 {fl['p75']}]; never flagged: {fl['n_never_flagged']}/{d1['n_collisions']} "
                 f"(frac {fl['frac_never_flagged']}).")
        L.append(f"- **declared SAFE (h<0) at the LAST brake-feasible step: "
                 f"{lb['frac_declared_safe_at_last_brakeable']}** "
                 f"(n with a brakeable step={lb['n_with_brakeable_step']}, median deployed h there "
                 f"{lb['median_deployed_h_at_last_brakeable']}).")
        if d1["matched_cells"]:
            L.append("- matched deployed-h, collision-precursor vs non-colliding at same (clearance,speed):")
            L.append("\n  | clearance | speed | n_coll | n_safe | coll h | safe h | coll-safe |")
            L.append("  |---|---|---|---|---|---|---|")
            for mc in d1["matched_cells"]:
                L.append(f"  | {mc['clearance_bin']} | {mc['speed_bin']} | {mc['n_coll']} | {mc['n_safe']} | "
                         f"{mc['coll_h_mean']:.3f} | {mc['safe_h_mean']:.3f} | {mc['coll_minus_safe']:+.3f} |")
        L.append("\n### D2 ensemble decomposition (at last brake-feasible / critical state)")
        if d2.get("n_critical", 0) == 0:
            L.append("- no critical states (no brake-feasible collisions).")
        else:
            cf = d2["counterfactual_flag_rate_at_last_brakeable"]
            L.append(f"- n critical = {d2['n_critical']}; mean declared safe = {d2['frac_mean_declared_safe']:.3f}.")
            L.append(f"- **mean-washout (mean<0 but max>=0, a member flagged danger): {d2['frac_mean_washout_of_n']:.3f}** "
                     f"of all critical (={d2['frac_mean_washout_of_mean_safe']} of mean-safe).")
            L.append(f"- **both members wrong (both h<0): {d2['frac_both_members_wrong']:.3f}** (needs better learning).")
            L.append(f"- counterfactual would-be-flagged at last brake-feasible step: "
                     f"**min {cf['min']:.3f} | mean {cf['mean']:.3f} | max {cf['max']:.3f}**.")
            L.append(f"- member disagreement |h0-h1|: collision median {d2['disagreement_collision']['median']:.3f} "
                     f"vs safe median {d2['disagreement_safe']['median']}.")
            mh = d2["member_h_at_critical"]
            L.append(f"- median member h at critical: m0 {mh['m0_median']:.3f}, m1 {mh['m1_median']:.3f}, "
                     f"mean {mh['mean_median']:.3f}, max {mh['max_median']:.3f}.")
        L.append("\n### D3 data-gap vs bias")
        L.append(f"- collision-critical cells: median visited-density {d3['median_visited_density_of_collision_cells']} "
                 f"vs overall median {d3['median_visited_density_overall']}; "
                 f"frac of collision cells BELOW median visitation = {d3['frac_collision_cells_below_median_visitation']}.")
        if d3["collision_critical_cells"]:
            L.append("\n  | clearance | speed | n_coll_critical | visited_density |")
            L.append("  |---|---|---|---|")
            for c in d3["collision_critical_cells"][:8]:
                L.append(f"  | {c['clearance_bin']} | {c['speed_bin']} | {c['n_collision_critical']} | {c['visited_density']:.4f} |")
    L.append("\n## Verdict — ranked candidate fixes")
    L.append(results["verdict"])
    L.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[doc] -> {DOC} ({len(L)} lines)")


def _verdict(results) -> str:
    t = results["TREATED"]
    d2 = t["D2"]; d1 = t["D1"]; d3 = t["D3"]
    if d2.get("n_critical", 0) == 0:
        return "Insufficient brake-feasible collisions to rank fixes (see per-checkpoint tables)."
    washout = d2["frac_mean_washout_of_n"]
    both_wrong = d2["frac_both_members_wrong"]
    cf = d2["counterfactual_flag_rate_at_last_brakeable"]
    max_gain = cf["max"] - cf["mean"]
    below = d3["frac_collision_cells_below_median_visitation"]
    parts = []
    parts.append(f"At the last brake-feasible step, deployed(mean) flags {cf['mean']:.2f} of collisions; "
                 f"MAX would flag {cf['max']:.2f} (+{max_gain:.2f}), MIN {cf['min']:.2f}.")
    # rank
    rank = []
    if washout >= 0.15 or max_gain >= 0.15:
        rank.append(("A", f"change deploy aggregation mean->max/upper-quantile (mean-washout {washout:.2f}, "
                          f"max-flag gain +{max_gain:.2f}) -- cheapest, deploy-only"))
    if both_wrong >= 0.4:
        rank.append(("B" if (below is not None and below < 0.5) else "C",
                     f"both-members-wrong {both_wrong:.2f}: under-estimation is net-level, "
                     f"{'loss bias (states well-visited)' if (below is not None and below < 0.5) else 'data gap (under-visited)'}"))
    if below is not None and below >= 0.5:
        rank.append(("C", f"collision-critical cells under-visited ({below:.2f} below median) -> inject precursor states"))
    if not rank:
        rank.append(("B", "moderate signals; a conservative/asymmetric value loss is the safest single lever"))
    order = ", ".join(f"({tag}) {desc}" for tag, desc in rank)
    return parts[0] + " Ranked: " + order + "."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treated-ckpt", type=Path, default=DEFAULT_TREATED)
    ap.add_argument("--sota-ckpt", type=Path, default=DEFAULT_SOTA)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    global DOC
    if args.smoke > 0:
        device = torch.device("cpu")
        treated_ckpt = sota_ckpt = args.sota_ckpt
        n = args.smoke
        tags = {"TREATED": "treated_smoke", "SOTA": "sota_smoke"}
        DOC = Path("/tmp/v221_calibration_ensemble_smoke.md")
        print(f"[smoke] N={n} CPU; SOTA for both; doc -> {DOC}")
    else:
        treated_ckpt, sota_ckpt, n = args.treated_ckpt, args.sota_ckpt, args.n
        tags = {"TREATED": "treated", "SOTA": "sota"}

    ckpts = {"TREATED": treated_ckpt, "SOTA": sota_ckpt}
    results = {}
    for label in ("SOTA", "TREATED"):
        build_cache(tags[label], ckpts[label], args.pool, device, n)
        eps, eval_row = load_cache(tags[label])
        results[label] = {"D1": analysis_d1(eps), "D2": analysis_d2(eps), "D3": analysis_d3(eps),
                          "eval_row": eval_row}
        _figs(label, results[label]["D1"], results[label]["D2"], CACHE_ROOT / ("_smoke_figs" if args.smoke else "figures"))

    meta = {"pool": str(args.pool), "n": (n if n is not None else results["TREATED"]["D1"]["n_collisions"] and "full"),
            "device": str(device), "treated_ckpt": str(treated_ckpt), "sota_ckpt": str(sota_ckpt),
            "treated_step": _ckpt_step(treated_ckpt), "sota_step": _ckpt_step(sota_ckpt)}
    meta["n"] = "full(500)" if n is None else n
    results["verdict"] = _verdict(results)
    (CACHE_ROOT / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps({"meta": meta, **results}, indent=2, default=_num), encoding="utf-8")
    write_doc(meta, results)
    print("VERDICT:", results["verdict"])
    return 0


def _ckpt_step(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        return -1


if __name__ == "__main__":
    raise SystemExit(main())
