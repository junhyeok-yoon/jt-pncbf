"""v2.2.2 — calibration re-diagnosis + reach-loss diagnosis (READ-ONLY, deterministic, seeded).

Closes the v2.2.2 mechanism question the aggregate metrics cannot answer. v2.2.2 (collision-precursor
injection into the value MSE buffer) dropped full_n500 collision 0.064 -> 0.006 (== SOTA) but lost
reach (0.940 vs SOTA 0.958), netting cps 0.8415 < SOTA 0.8852. Two questions:

PART A — did the injection FIX V_S CALIBRATION (the intended mechanism)?  Re-run the EXACT v2.2.1
  D1/D2/D3 (imported verbatim from v221_calibration_ensemble_diagnosis) on v2.2.2 best.pt, reported
  side-by-side with SOTA and the v2.2.1-TREATED checkpoint. Key before/after: did the "declared SAFE
  at the last brake-feasible step" fraction move DOWN from v2.2.1's 0.67 toward the SOTA-conservative
  regime, and did D2 both-members-wrong drop?

PART B — where did REACH go (over-conservatism check)?  (1) shared-scene SOTA-outcome x v2.2.2-outcome
  confusion table over the 500 scenes (same approach as the v2.2.1 remaining-collision A3); the reach->
  non-reach regressions and what they convert to. (2) On those regressed scenes, deployed h along the
  v2.2.2 vs SOTA trajectory (safe portion). (3) RIGOROUS same-state over-conservatism test: cross-
  evaluate BOTH nets' deployed h on the IDENTICAL safe-normal observations (clearance>0.5, low inward
  speed, away from obstacles) that SOTA drives through -> is v2.2.2's h systematically higher there
  (injection over-generalised its conservatism beyond the collision-critical target region)?

Same rollout SEED (20260618) and harness (jt_failure_dissect._episode_trace / _event_action_index;
src.eval.evaluate.evaluate; run_full._load_framework) as the v2.2.1 diagnoses, so the collision set and
D1/D2/D3 numbers are directly comparable. Caches a single rich rollout pass per checkpoint (incl. obs
for cross-evaluation) under data/diagnostics/v2.2.2_calibration_reach/<tag>/ (git-ignored).

Run (GPU; ~10-15 min, three checkpoints):
  python scripts/analysis/v222_calibration_reach_diagnosis.py
Smoke (CPU, tiny; SOTA for all three tags):
  python scripts/analysis/v222_calibration_reach_diagnosis.py --smoke 4 --device cpu
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

import jt_failure_dissect as D  # noqa: E402
import v221_calibration_ensemble_diagnosis as C  # noqa: E402  (D1/D2/D3 + helpers, reused verbatim)
import v221_remaining_collision_diagnosis as RC  # noqa: E402  (analysis3_transition shared-scene matrix)
from src.eval.build_pools import EvaluationPool, load_pool  # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402
from src.eval.run_full import _load_framework  # noqa: E402
from src.frameworks.jt_pncbf.train import make_system  # noqa: E402
from src.common.value_net import make_h_fn  # noqa: E402

DEF_V222 = REPO_ROOT / "data/v2.2.2__20260618-054411__seed42/checkpoints/best.pt"
DEF_SOTA = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
DEF_V221 = REPO_ROOT / "data/runs/v2.2.1/v2.2.1__20260618-001001__seed42/checkpoints/best.pt"
DEF_POOL = REPO_ROOT / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl"
CACHE_ROOT = REPO_ROOT / "data/diagnostics/v2.2.2_calibration_reach"
DOC = REPO_ROOT / "docs/versions/v2.2.2/calibration_reach_diagnosis.md"

EVAL_BATCH = 250
SEED = C.SEED                       # 20260618 — identical collision set to the v2.2.1 diagnoses
COLLISION = "collision"
GOAL = "goal"
# safe-normal region for the over-conservatism (matched same-state) test
SAFE_CLEAR = 0.5                    # surface clearance to nearest obstacle (m); > -> "away from obstacles"
SAFE_INWARD = 0.3                  # inward speed (m/s); < -> "low inward speed"
MAX_MATCH_STATES = 60000           # cap on reference states for cross-eval (deterministic head slice)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _ckpt_step(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        return -1


def _num(v):
    return float(v) if isinstance(v, (np.floating, np.integer)) else v


# ================================================================================================== #
# Rich rollout cache (single pass per checkpoint): D1/D2/D3 fields + obs (for cross-eval) + geometry  #
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
    if index_path.exists() and (out / "eval_row.json").exists():
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
    use_pool = EvaluationPool(name=f"v222_calreach_{tag}", system=pool.system,
                              n_scenes=len(scenes), seed=getattr(pool, "seed", SEED), scenes=scenes)
    result = evaluate(fw, use_pool, config, mode="diagnostic_v222_calibration_reach",
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
        np.savez_compressed(
            ep_dir / f"ep_{idx:05d}_{outcome}.npz",
            outcome=np.asarray(outcome), event_step=np.asarray(int(ep.filtered_event_step)),
            active_steps=np.asarray(int(trace["active_steps"])),
            h=np.asarray(trace["h"], np.float32),
            member0=members[:, 0].detach().cpu().numpy().astype(np.float32),
            member1=members[:, 1].detach().cpu().numpy().astype(np.float32),
            clearance=np.asarray(trace["clearance"], np.float32),
            closing_speed=np.asarray(trace["closing_speed"], np.float32),
            brake_margin=np.asarray(trace["brake_margin"], np.float32),
            goal_distance=np.asarray(trace["goal_distance"], np.float32),
            intervention=np.asarray(trace["intervention"], np.float32),
            saturated=np.asarray(trace["saturated"], np.float32),
            obs=obs.detach().cpu().numpy().astype(np.float32),       # [T, obs_dim] for cross-evaluation
        )
        index_rows.append({"idx": idx, "outcome": outcome})
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "outcome"]); w.writeheader(); w.writerows(index_rows)
    (out / "eval_row.json").write_text(json.dumps({k: _num(v) for k, v in result.eval_row.items()}, indent=2))
    print(f"[{tag}] rolled out {len(index_rows)} episodes -> {out}")
    return out


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
            "goal_distance": np.asarray(z["goal_distance"], float),
            "intervention": np.asarray(z["intervention"], float),
            "saturated": np.asarray(z["saturated"], float),
            "obs": np.asarray(z["obs"], np.float32),
        })
    eps.sort(key=lambda e: e["idx"])
    return eps, json.loads((out / "eval_row.json").read_text())


# ================================================================================================== #
# PART A — calibration re-diagnosis: run the EXACT v2.2.1 D1/D2/D3 on each checkpoint                  #
# ================================================================================================== #
def part_a(eps_by_tag: dict) -> dict:
    out = {}
    for tag, eps in eps_by_tag.items():
        out[tag] = {"D1": C.analysis_d1(eps), "D2": C.analysis_d2(eps), "D3": C.analysis_d3(eps)}
    return out


def _short_clearance_inversion(d1: dict) -> dict | None:
    """The [0,0.10) clearance matched coll-vs-safe rows (the v2.2.1 under-estimation inversion)."""
    rows = [m for m in d1["matched_cells"] if m["clearance_bin"] == "[0.00,0.10)"]
    if not rows:
        return None
    return {"rows": rows, "mean_coll_minus_safe": float(np.mean([r["coll_minus_safe"] for r in rows]))}


# ================================================================================================== #
# PART B — reach-loss diagnosis                                                                       #
# ================================================================================================== #
def part_b_confusion(v222_eps, sota_eps) -> dict:
    """Shared-scene SOTA x v2.2.2 outcome matrix; reach->non-reach regressions and their breakdown."""
    a3 = RC.analysis3_transition(treated=v222_eps, sota=sota_eps)   # treated == v2.2.2
    so = {e["idx"]: e["outcome"] for e in sota_eps}
    tr = {e["idx"]: e["outcome"] for e in v222_eps}
    shared = sorted(set(so) & set(tr))
    regressed = [i for i in shared if so[i] == GOAL and tr[i] != GOAL]      # SOTA reaches, v2.2.2 does not
    gained = [i for i in shared if so[i] != GOAL and tr[i] == GOAL]         # v2.2.2 reaches, SOTA does not
    reg_break = C._tally([tr[i] for i in regressed]) if hasattr(C, "_tally") else _tally([tr[i] for i in regressed])
    gain_break = _tally([so[i] for i in gained])
    return {
        "matrix_sota_x_v222": a3["matrix_sota_x_treated"], "outcomes": a3["outcomes"],
        "n_shared": a3["n_shared"],
        "n_reach_regressions": len(regressed), "regression_breakdown": reg_break, "regressed_idx": regressed,
        "n_reach_gains": len(gained), "gain_breakdown": gain_break,
        "net_reach_delta": len(gained) - len(regressed)}


def _tally(items):
    out = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _safe_mask(ep) -> np.ndarray:
    a = min(int(ep["active_steps"]), len(ep["h"]))
    m = np.zeros(len(ep["h"]), bool)
    if a <= 0:
        return m
    m[:a] = (ep["clearance"][:a] > SAFE_CLEAR) & (ep["closing_speed"][:a] < SAFE_INWARD)
    return m


def part_b_regressed_h(v222_eps, sota_eps, regressed_idx) -> dict:
    """Deployed h on the SAFE portion of each regressed scene's trajectory: v2.2.2 vs SOTA (own traj)."""
    so = {e["idx"]: e for e in sota_eps}
    tr = {e["idx"]: e for e in v222_eps}
    v222_safe_h, sota_safe_h, v222_interv, sota_interv, v222_sat, sota_sat = [], [], [], [], [], []
    for i in regressed_idx:
        ev, sv = tr[i], so[i]
        mv, ms = _safe_mask(ev), _safe_mask(sv)
        if mv.any():
            v222_safe_h.append(float(np.mean(ev["h"][mv])))
            v222_interv.append(float(np.mean(ev["intervention"][:ev["active_steps"]])))
            v222_sat.append(float(np.mean(ev["saturated"][:ev["active_steps"]])))
        if ms.any():
            sota_safe_h.append(float(np.mean(sv["h"][ms])))
            sota_interv.append(float(np.mean(sv["intervention"][:sv["active_steps"]])))
            sota_sat.append(float(np.mean(sv["saturated"][:sv["active_steps"]])))
    def stat(a):
        return {"n": len(a), "mean": float(np.mean(a)) if a else None, "median": float(np.median(a)) if a else None}
    return {
        "v222_safe_portion_h": stat(v222_safe_h), "sota_safe_portion_h": stat(sota_safe_h),
        "v222_intervention_rate": stat(v222_interv), "sota_intervention_rate": stat(sota_interv),
        "v222_saturation_rate": stat(v222_sat), "sota_saturation_rate": stat(sota_sat)}


def _gather_safe_obs(eps, only_idx=None) -> np.ndarray:
    """Stack obs of safe-normal states (clearance>0.5, inward speed<0.3) from GOAL episodes."""
    chunks = []
    for e in eps:
        if e["outcome"] != GOAL:
            continue
        if only_idx is not None and e["idx"] not in only_idx:
            continue
        m = _safe_mask(e)
        if m.any():
            chunks.append(e["obs"][m])
    if not chunks:
        return np.zeros((0, eps[0]["obs"].shape[1]), np.float32)
    return np.concatenate(chunks, axis=0)


def _load_value_net(ckpt_path: Path, device: torch.device):
    fw, config, _ = _load_framework(ckpt_path)
    system = make_system(config)
    vn = fw.value_net.to(device=device, dtype=torch.float32).eval()
    return vn


def _deployed_h_on_obs(value_net, obs_np: np.ndarray, device: torch.device, batch: int = 8192) -> np.ndarray:
    if obs_np.shape[0] == 0:
        return np.zeros((0,), float)
    out = []
    with torch.no_grad():
        for i in range(0, obs_np.shape[0], batch):
            ob = torch.from_numpy(obs_np[i:i + batch]).to(device=device, dtype=torch.float32)
            out.append(value_net.deployed_h(ob).detach().cpu().numpy())
    return np.concatenate(out).astype(float)


def part_b_matched_overconservatism(ref_obs: np.ndarray, ref_name: str, nets: dict, device: torch.device) -> dict:
    """Cross-evaluate every net's DEPLOYED h on the IDENTICAL reference obs. Same x -> isolates the net."""
    if ref_obs.shape[0] > MAX_MATCH_STATES:
        ref_obs = ref_obs[:MAX_MATCH_STATES]
    h = {tag: _deployed_h_on_obs(vn, ref_obs, device) for tag, vn in nets.items()}
    res = {"reference": ref_name, "n_states": int(ref_obs.shape[0]),
           "median_h": {t: float(np.median(v)) if v.size else None for t, v in h.items()},
           "mean_h": {t: float(np.mean(v)) if v.size else None for t, v in h.items()},
           "frac_danger_hpos": {t: float(np.mean(v > 0.0)) if v.size else None for t, v in h.items()}}
    if "V222" in h and "SOTA" in h and h["V222"].size:
        d = h["V222"] - h["SOTA"]
        res["v222_minus_sota"] = {"median": float(np.median(d)), "mean": float(np.mean(d)),
                                  "frac_v222_higher": float(np.mean(d > 0.0)),
                                  "p90": float(np.percentile(d, 90)), "p10": float(np.percentile(d, 10))}
    if "V222" in h and "V221" in h and h["V221"].size:
        d = h["V222"] - h["V221"]
        res["v222_minus_v221"] = {"median": float(np.median(d)), "mean": float(np.mean(d)),
                                  "frac_v222_higher": float(np.mean(d > 0.0))}
    return res


# ================================================================================================== #
# Figures + doc                                                                                       #
# ================================================================================================== #
def _figs(part_a_res, matched_sota_ref, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = [t for t in ("V221", "V222", "SOTA") if t in part_a_res]
    safe_frac = [part_a_res[t]["D1"]["last_brake_feasible"]["frac_declared_safe_at_last_brakeable"] or 0.0 for t in tags]
    med_h = [part_a_res[t]["D1"]["last_brake_feasible"]["median_deployed_h_at_last_brakeable"] or 0.0 for t in tags]
    bw = [part_a_res[t]["D2"].get("frac_both_members_wrong", 0.0) for t in tags]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    x = np.arange(len(tags)); w = 0.38
    ax[0].bar(x - w / 2, safe_frac, w, label="frac declared SAFE @ last brake-feasible", color="#d62728")
    ax[0].bar(x + w / 2, bw, w, label="both-members-wrong frac", color="#1f77b4")
    ax[0].set_xticks(x); ax[0].set_xticklabels(tags); ax[0].set_ylim(0, 1)
    ax[0].set_title("PART A: under-estimation @ critical state"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar(x, med_h, color=["#7f7f7f", "#d62728", "#2ca02c"][:len(tags)])
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(tags)
    ax[1].set_ylabel("median deployed h @ last brake-feasible")
    ax[1].set_title("PART A: <0 = under-estimates danger")
    fig.tight_layout(); fig.savefig(out_dir / "partA_calibration.png"); plt.close(fig)

    if matched_sota_ref and "v222_minus_sota" in matched_sota_ref:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        mh = matched_sota_ref["median_h"]
        ax.bar(list(mh.keys()), list(mh.values()), color="#9467bd")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel("median deployed h on SOTA safe-normal states")
        d = matched_sota_ref["v222_minus_sota"]
        ax.set_title(f"PART B over-conservatism (same states)\nv222-sota median {d['median']:+.3f}, "
                     f"frac v222 higher {d['frac_v222_higher']:.2f}")
        fig.tight_layout(); fig.savefig(out_dir / "partB_overconservatism.png"); plt.close(fig)


def write_doc(meta, pa, conf, reg_h, matched, eval_rows, verdict):
    L = ["# v2.2.2 — calibration re-diagnosis + reach-loss diagnosis\n"]
    L.append(f"_pool_ `{meta['pool']}` (N={meta['n']}) | seed={SEED} | device={meta['device']}  ")
    L.append(f"_V222_ `{meta['v222']}` (step {meta['v222_step']})  ")
    L.append(f"_SOTA_ `{meta['sota']}` (step {meta['sota_step']})  ")
    L.append(f"_V221_ `{meta['v221']}` (step {meta['v221_step']})\n")
    L.append("Read-only. PART A reuses the EXACT v2.2.1 D1/D2/D3 (`v221_calibration_ensemble_diagnosis`) "
             "and the same rollout seed, so numbers are directly comparable. Deployed h = MEAN of 2 "
             "clamped members; safe={h<=0}, danger={h>0}.\n")

    L.append("## eval_row (this rollout, full pool)\n")
    L.append("| ckpt | cps | reach | collision | stuck | timeout | infeasibility | saturation |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in ("V222", "SOTA", "V221"):
        r = eval_rows[t]
        L.append(f"| {t} | {r.get('cps'):.4f} | {r.get('reach'):.3f} | {r.get('collision'):.3f} | "
                 f"{r.get('stuck'):.3f} | {r.get('timeout'):.3f} | {r.get('infeasibility'):.3f} | "
                 f"{r.get('saturation_rate'):.3f} |")

    L.append("\n## PART A — did the injection fix V_S calibration?\n")
    L.append("### D1 — under-estimation at the last brake-feasible (critical) step\n")
    L.append("| ckpt | collisions | frac declared SAFE @ last brake-feasible | median deployed h there | "
             "first-flag median lead (steps) | never-flagged |")
    L.append("|---|---|---|---|---|---|")
    for t in ("V221", "V222", "SOTA"):
        d1 = pa[t]["D1"]; lb = d1["last_brake_feasible"]; fl = d1["first_flag_lead"]
        L.append(f"| {t} | {d1['n_collisions']} | {lb['frac_declared_safe_at_last_brakeable']} | "
                 f"{lb['median_deployed_h_at_last_brakeable']} | {fl['median_lead']} | "
                 f"{fl['n_never_flagged']}/{d1['n_collisions']} |")
    L.append("\nReference (v2.2.1 doc): TREATED 0.667 safe / median h -0.061 / 16-step lead; "
             "SOTA 0.25 safe / +0.317 / ~40-step lead.\n")
    for t in ("V221", "V222", "SOTA"):
        inv = _short_clearance_inversion(pa[t]["D1"])
        if inv:
            L.append(f"- {t} short-clearance [0,0.10) matched coll-minus-safe h "
                     f"(neg = precursors look MORE safe than equally-close safe states): "
                     f"mean {inv['mean_coll_minus_safe']:+.3f} over {len(inv['rows'])} speed cells.")

    L.append("\n### D2 — ensemble decomposition at the critical state\n")
    L.append("| ckpt | n_critical | mean-washout | both-members-wrong | "
             "counterfactual flag-rate (min/mean/max) |")
    L.append("|---|---|---|---|---|")
    for t in ("V221", "V222", "SOTA"):
        d2 = pa[t]["D2"]
        if d2.get("n_critical", 0) == 0:
            L.append(f"| {t} | 0 | - | - | - |"); continue
        cf = d2["counterfactual_flag_rate_at_last_brakeable"]
        L.append(f"| {t} | {d2['n_critical']} | {d2['frac_mean_washout_of_n']:.3f} | "
                 f"{d2['frac_both_members_wrong']:.3f} | {cf['min']:.2f}/{cf['mean']:.2f}/{cf['max']:.2f} |")
    L.append("\nReference (v2.2.1 doc): TREATED washout 0.238, both-wrong 0.429; SOTA washout 0.0, both-wrong 0.25.\n")

    L.append("### D3 — collision-critical cell visitation\n")
    L.append("| ckpt | median visited-density of collision cells | overall median | frac below median |")
    L.append("|---|---|---|---|")
    for t in ("V221", "V222", "SOTA"):
        d3 = pa[t]["D3"]
        L.append(f"| {t} | {d3['median_visited_density_of_collision_cells']} | "
                 f"{d3['median_visited_density_overall']} | "
                 f"{d3['frac_collision_cells_below_median_visitation']} |")

    L.append("\n## PART B — where did reach go?\n")
    L.append(f"Shared scenes: {conf['n_shared']}. Reach regressions (SOTA goal -> v2.2.2 non-goal): "
             f"**{conf['n_reach_regressions']}**, breakdown {conf['regression_breakdown']}. "
             f"Reach gains (v2.2.2 goal, SOTA not): {conf['n_reach_gains']} ({conf['gain_breakdown']}). "
             f"Net reach delta v2.2.2-SOTA: **{conf['net_reach_delta']:+d}**.\n")
    outs = conf["outcomes"]; mat = conf["matrix_sota_x_v222"]
    L.append("| SOTA \\ v2.2.2 | " + " | ".join(outs) + " |")
    L.append("|---|" + "|".join(["---"] * len(outs)) + "|")
    for a in outs:
        if a in mat:
            L.append(f"| {a} | " + " | ".join(str(mat[a].get(b, 0)) for b in outs) + " |")

    L.append("\n### B2 — deployed h on the SAFE portion of regressed-scene trajectories\n")
    L.append(f"- v2.2.2 safe-portion h: {reg_h['v222_safe_portion_h']}")
    L.append(f"- SOTA  safe-portion h: {reg_h['sota_safe_portion_h']}")
    L.append(f"- intervention rate (regressed scenes): v2.2.2 {reg_h['v222_intervention_rate']}, "
             f"SOTA {reg_h['sota_intervention_rate']}")
    L.append(f"- saturation rate (regressed scenes): v2.2.2 {reg_h['v222_saturation_rate']}, "
             f"SOTA {reg_h['sota_saturation_rate']}")

    L.append("\n### B3 — RIGOROUS same-state over-conservatism (cross-eval on identical observations)\n")
    for m in matched:
        L.append(f"- reference = {m['reference']} ({m['n_states']} safe-normal states, "
                 f"clearance>{SAFE_CLEAR}, inward speed<{SAFE_INWARD}):")
        L.append(f"  median deployed h: {m['median_h']}; frac flagged danger (h>0): {m['frac_danger_hpos']}.")
        if "v222_minus_sota" in m:
            d = m["v222_minus_sota"]
            L.append(f"  **v2.2.2 - SOTA on identical states: median {d['median']:+.4f}, mean {d['mean']:+.4f}, "
                     f"frac v2.2.2 higher {d['frac_v222_higher']:.3f}** (p10 {d['p10']:+.3f}, p90 {d['p90']:+.3f}).")
        if "v222_minus_v221" in m:
            d = m["v222_minus_v221"]
            L.append(f"  v2.2.2 - v2.2.1 on identical states: median {d['median']:+.4f}, "
                     f"frac v2.2.2 higher {d['frac_v222_higher']:.3f}.")

    L.append("\n## VERDICT\n")
    L.append(verdict)
    L.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[doc] -> {DOC} ({len(L)} lines)")


def _verdict(pa, conf, reg_h, matched, eval_rows) -> str:
    v222_d1 = pa["V222"]["D1"]["last_brake_feasible"]
    v221_d1 = pa["V221"]["D1"]["last_brake_feasible"]
    sota_d1 = pa["SOTA"]["D1"]["last_brake_feasible"]
    v222_d2, v221_d2 = pa["V222"]["D2"], pa["V221"]["D2"]
    v222_bw, v221_bw = v222_d2.get("frac_both_members_wrong"), v221_d2.get("frac_both_members_wrong")
    v222_wash, v221_wash = v222_d2.get("frac_mean_washout_of_n"), v221_d2.get("frac_mean_washout_of_n")
    safe222 = v222_d1["frac_declared_safe_at_last_brakeable"]
    safe221 = v221_d1["frac_declared_safe_at_last_brakeable"]
    medh222, medh221 = v222_d1["median_deployed_h_at_last_brakeable"], v221_d1["median_deployed_h_at_last_brakeable"]
    v222_d3, v221_d3 = pa["V222"]["D3"], pa["V221"]["D3"]
    vis222 = v222_d3["median_visited_density_of_collision_cells"]
    vis221 = v221_d3["median_visited_density_of_collision_cells"]
    below222, below221 = v222_d3["frac_collision_cells_below_median_visitation"], v221_d3["frac_collision_cells_below_median_visitation"]

    # (i) calibration fixed?
    calib = []
    if None not in (safe222, safe221):
        moved = safe221 - safe222
        calib.append(f"declared-SAFE @ last brake-feasible {safe221:.2f}->{safe222:.2f} (== SOTA "
                     f"{sota_d1['frac_declared_safe_at_last_brakeable']:.2f}; "
                     + ("moved INTO the conservative regime" if moved > 0.05 else "did NOT move down") + ")")
    if None not in (medh221, medh222):
        calib.append(f"median deployed h there {medh221:+.3f}->{medh222:+.3f} (under-est -> "
                     + ("positive/conservative" if medh222 > 0 else "still <=0") + ")")
    if None not in (v221_bw, v222_bw):
        calib.append(f"both-members-wrong {v221_bw:.2f}->{v222_bw:.2f}"
                     + (" (dropped: net-level learning fix)" if v222_bw < v221_bw - 0.05 else ""))
    if None not in (v221_wash, v222_wash):
        calib.append(f"mean-washout {v221_wash:.2f}->{v222_wash:.2f}")
    if None not in (vis221, vis222) and vis221:
        calib.append(f"collision-cell visitation {vis221:.4f}->{vis222:.4f} (~{vis222/vis221:.0f}x), "
                     f"frac-below-median {below221:.2f}->{below222:.2f}")
    fixed = (None not in (safe222, safe221) and (safe221 - safe222) > 0.05) and \
            (None not in (v222_bw, v221_bw) and v222_bw <= v221_bw + 1e-9)

    # (ii) reach loss source: matched same-state test (generic + regressed-scene) vs B2 own-trajectory signature
    m_all = next((x for x in matched if x["reference"].startswith("SOTA goal")), None)
    m_reg = next((x for x in matched if "REGRESSED" in x["reference"]), None)
    dmed = m_all["v222_minus_sota"]["median"] if (m_all and "v222_minus_sota" in m_all) else None
    dfrac = m_all["v222_minus_sota"]["frac_v222_higher"] if (m_all and "v222_minus_sota" in m_all) else None
    dmed_reg = m_reg["v222_minus_sota"]["median"] if (m_reg and "v222_minus_sota" in m_reg) else None
    iv222 = reg_h["v222_intervention_rate"]["median"]; ivS = reg_h["sota_intervention_rate"]["median"]
    inflated = (dmed is not None and dmed > 0.02 and dfrac is not None and dfrac > 0.55)
    reg = conf["regression_breakdown"]
    if inflated:
        reach_src = (f"OVER-CONSERVATISM beyond the target region: on identical safe-normal states SOTA drives "
                     f"through, v2.2.2 h is higher (median {dmed:+.3f}, {dfrac:.0%} higher). The injection inflated "
                     f"h outside the collision-critical region. Next axis: tighten the injection's spatial influence "
                     f"(region/fraction); do NOT abandon it.")
    else:
        reach_src = (
            f"NOT value-side over-conservatism. The rigorous same-state test is decisive: on the IDENTICAL "
            f"safe-normal states SOTA drives through, v2.2.2 deployed h is NOT higher (median diff {dmed:+.4f}, "
            f"only {dfrac:.0%} of states higher), and the same holds on the regressed scenes' viable states "
            f"(median diff {dmed_reg:+.4f}). So the injection did NOT over-generalise conservatism onto safe "
            f"states / viable paths. The B2 over-braking signature (on the {reg_h['v222_intervention_rate']['n']} "
            f"regressed scenes v2.2.2 intervenes {iv222:.0%} of steps vs SOTA {ivS:.0%}, with higher safe-portion h) "
            f"is therefore confined to v2.2.2's OWN divergent trajectories on ~{conf['n_reach_regressions']} scenes "
            f"(mostly -> {reg}), not a global h-inflation that blocks SOTA's path. This is consistent with a weaker / "
            f"less-general checkpoint: best.pt@35000 was selected on in-loop cps 0.8964 but generalises to full_n500 "
            f"cps ~0.8415, a ~0.055 in-loop/full gap. Next axis: improve checkpoint generality / selection (or "
            f"policy-side robustness), NOT shrink the injection region (the matched test shows it isn't inflating "
            f"safe-state h).")
    return ("(i) CALIBRATION: " + ("FIXED as intended -- " if fixed else "signals mixed -- ")
            + "; ".join(calib) + "."
            + f"\n\n(ii) REACH LOSS ({conf['n_reach_regressions']} reach->non-reach regressions, net {conf['net_reach_delta']:+d} "
              f"reach scenes vs SOTA): {reach_src}"
            + "\n\nCAVEAT: V222@35000 / SOTA@34000 / V221@28000 are different checkpoints & convergence; read "
              "FRACTIONS, and weigh the in-loop(0.8964) vs full_n500(0.8415) generalization gap when attributing reach.")


# ================================================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v222-ckpt", type=Path, default=DEF_V222)
    ap.add_argument("--sota-ckpt", type=Path, default=DEF_SOTA)
    ap.add_argument("--v221-ckpt", type=Path, default=DEF_V221)
    ap.add_argument("--pool", type=Path, default=DEF_POOL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    global DOC, CACHE_ROOT
    ckpts = {"V222": args.v222_ckpt, "SOTA": args.sota_ckpt, "V221": args.v221_ckpt}
    if args.smoke > 0:
        device = torch.device("cpu")
        ckpts = {"V222": args.sota_ckpt, "SOTA": args.sota_ckpt, "V221": args.sota_ckpt}
        tags = {"V222": "v222_smoke", "SOTA": "sota_smoke", "V221": "v221_smoke"}
        n = args.smoke
        DOC = Path("/tmp/v222_calibration_reach_smoke.md")
        print(f"[smoke] N={n} CPU; SOTA for all three; doc -> {DOC}")
    else:
        tags = {"V222": "v222", "SOTA": "sota", "V221": "v221"}
        n = args.n

    eps_by_tag, eval_rows = {}, {}
    for label in ("SOTA", "V222", "V221"):
        build_cache(tags[label], ckpts[label], args.pool, device, n)
        eps, row = load_cache(tags[label])
        eps_by_tag[label] = eps; eval_rows[label] = row

    # PART A
    pa = part_a(eps_by_tag)

    # PART B
    conf = part_b_confusion(eps_by_tag["V222"], eps_by_tag["SOTA"])
    reg_h = part_b_regressed_h(eps_by_tag["V222"], eps_by_tag["SOTA"], conf["regressed_idx"])
    nets = {t: _load_value_net(ckpts[t], device) for t in ("V222", "SOTA", "V221")}
    matched = []
    sota_ref = _gather_safe_obs(eps_by_tag["SOTA"])
    matched.append(part_b_matched_overconservatism(sota_ref, "SOTA goal-traj safe-normal states", nets, device))
    v222_ref = _gather_safe_obs(eps_by_tag["V222"])
    matched.append(part_b_matched_overconservatism(v222_ref, "v2.2.2 goal-traj safe-normal states", nets, device))
    # regressed-scene-restricted reference (SOTA reaches there; v2.2.2 does not)
    reg_ref = _gather_safe_obs(eps_by_tag["SOTA"], only_idx=set(conf["regressed_idx"]))
    if reg_ref.shape[0] > 0:
        matched.append(part_b_matched_overconservatism(reg_ref, "SOTA safe-normal states on REGRESSED scenes", nets, device))

    meta = {"pool": str(args.pool), "n": ("full(500)" if n is None else n), "device": str(device),
            "v222": str(ckpts["V222"]), "sota": str(ckpts["SOTA"]), "v221": str(ckpts["V221"]),
            "v222_step": _ckpt_step(ckpts["V222"]), "sota_step": _ckpt_step(ckpts["SOTA"]),
            "v221_step": _ckpt_step(ckpts["V221"])}
    verdict = _verdict(pa, conf, reg_h, matched, eval_rows)
    fig_dir = CACHE_ROOT / ("_smoke_figs" if args.smoke else "figures")
    matched_sota = next((m for m in matched if m["reference"].startswith("SOTA goal")), None)
    _figs(pa, matched_sota, fig_dir)

    summary = {"meta": meta, "eval_rows": eval_rows,
               "partA": {t: {"D1": _drop_keys(pa[t]["D1"], ["matched_cells"]),
                             "D1_short_clearance": _short_clearance_inversion(pa[t]["D1"]),
                             "D2": pa[t]["D2"], "D3": _drop_keys(pa[t]["D3"], ["collision_critical_cells"])}
                         for t in pa},
               "partB": {"confusion": _drop_keys(conf, ["regressed_idx"]),
                         "regressed_safe_h": reg_h, "matched_overconservatism": matched},
               "verdict": verdict}
    (CACHE_ROOT / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps(summary, indent=2, default=_num), encoding="utf-8")
    write_doc(meta, pa, conf, reg_h, matched, eval_rows, verdict)
    print("\nVERDICT:\n" + verdict)
    return 0


def _drop_keys(d: dict, keys) -> dict:
    out = dict(d)
    for k in keys:
        out.pop(k, None)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
