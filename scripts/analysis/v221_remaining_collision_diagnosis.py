"""v2.2.1 — remaining-collision post-hoc diagnostic (READ-ONLY, deterministic, seeded).

Question: the h-band L_feas run raised gate dh/dv (~0.20 -> ~0.65) yet final collision (~0.05-0.07)
stayed far above SOTA's 0.006. Did the bottleneck STAY at fast-approach dh/dv (0.65 insufficient), or
MOVE to a mode dh/dv cannot fix (slow-approach / geometric trap / doomed)? This script classifies the
REMAINING collisions AFTER the L_feas treatment and contrasts them with the untreated same-seed SOTA.

It rolls out BOTH checkpoints on the standard full pool (N=500), caches per-episode traces, and runs
four analyses, reusing the v2.2.0 diagnostic harness verbatim (scripts/verification/jt_failure_dissect:
_episode_trace, _classify_mechanism, _event_action_index; src.eval.evaluate.evaluate; run_full._load_framework)
so this is a true before/after, NOT a re-derivation:

  A1 REMAINING-COLLISION RECLASSIFICATION  (TREATED vs SOTA): each collision -> {fast_approach_myopia,
     slow_approach, geometric_trap, doomed_at_entry}. Key output: did the fast-approach-myopia fraction
     drop SOTA->TREATED (prescription worked, bottleneck moved) or stay (0.65 dh/dv still insufficient)?
  A2 SPEED-BAND MATCH: histogram inward approach speed at the collision step (TREATED) vs the gate's
     training band [0.6, 2.5]; deployed ||L_g h|| as a function of inward-speed band over ALL eval
     rollout states (TREATED vs SOTA) -> does the strengthened gradient cover the speeds where eval
     collisions actually occur?
  A3 COLLISION->STUCK TRANSITION: per shared scene index, SOTA-outcome x TREATED-outcome matrix; count
     collision(SOTA)->stuck(TREATED) (over-braking) and the reverse.
  A4 INFEASIBILITY SIDE-EFFECT: HardNet projection-infeasibility / empty half-space-box / saturation,
     TREATED vs SOTA on the same pool (the cps tax the stronger dh/dv may have introduced).

Caches per-episode NPZ traces under data/diagnostics/v2.2.1_remaining_collision/<tag>/ (git-ignored);
writes docs/versions/v2.2.1/remaining_collision_diagnosis.md (verdicts) + summary JSON + figures.

Run (after the run completes; uses GPU):
  python scripts/analysis/v221_remaining_collision_diagnosis.py \
      --treated-ckpt data/v2.2.1__20260618-001001__seed42/checkpoints/best.pt \
      --sota-ckpt    data/secured_data/v2.0.1/seed42/checkpoints/best.pt \
      --pool         data/secured_data/pools/eval_full_di_n500_seed23456.pkl --device auto
Smoke (CPU, tiny, no GPU contention; SOTA used for both tags):
  python scripts/analysis/v221_remaining_collision_diagnosis.py --smoke 3 --device cpu
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

import jt_failure_dissect as D  # noqa: E402  (_episode_trace, _classify_mechanism, _event_action_index, _write_episode_npz)
from src.common.value_net import make_h_fn  # noqa: E402
from src.eval.build_pools import EvaluationPool, load_pool  # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402
from src.eval.run_full import _load_framework  # noqa: E402
from src.frameworks.jt_pncbf.train import make_system  # noqa: E402

# ---- fixed defaults (read-only inputs) -------------------------------------------------------------
DEFAULT_TREATED = REPO_ROOT / "data/v2.2.1__20260618-001001__seed42/checkpoints/best.pt"
DEFAULT_SOTA = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
DEFAULT_POOL = REPO_ROOT / "data/secured_data/pools/eval_full_di_n500_seed23456.pkl"
CACHE_ROOT = REPO_ROOT / "data/diagnostics/v2.2.1_remaining_collision"
DOC = REPO_ROOT / "docs/versions/v2.2.1/remaining_collision_diagnosis.md"

EVAL_BATCH = 250
SEED = 20260618

# ---- classification thresholds (documented; reuse v2.2.0 constants where they exist) ---------------
U_MAX = 2.0                 # double-integrator accel bound (m/s^2)
ACT_MIN = 11               # v2.2.0 actionable-lead floor (steps) for "early alarm / feasible brake in time"
PRE_WINDOW = 10            # pre-impact window (matches _classify_collision before=10)
DANGER_CLEAR = 0.35        # v2.2.0 danger-onset clearance (= h_scale)
SLOW_SPEED = 0.6           # inward-speed cutoff = gate s_lo; below this, dh/dv is not the bottleneck
GATE_S_LO, GATE_S_HI = 0.6, 2.5   # L_feas gate training inward-speed band
NEAR_RADIUS = 0.5          # surface-clearance radius for counting "nearby" active obstacles
MULTI_OBSTACLE = 2         # >= this many nearby active obstacles at danger onset => multi-obstacle
H_ALARM = -0.05            # deployed h "alarm" level (h crossing toward 0 from the safe side)
SPEED_BANDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 4.0]

COLLISION = "collision"
OUTCOME_ORDER = ["goal", "collision", "stuck", "timeout", "oob"]


# ================================================================================================== #
# Rollout + per-episode trace cache (the only heavy / GPU step; idempotent)                           #
# ================================================================================================== #
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_cache(tag: str, ckpt_path: Path, pool_path: Path, device: torch.device, n: int | None) -> Path:
    """Roll out one checkpoint on the pool; write per-episode trace NPZs + index.csv + eval_row.json."""
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out = CACHE_ROOT / tag
    ep_dir = out / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    done = index_path.exists() and (out / "eval_row.json").exists()
    if done:
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
    h_fn = make_h_fn(fw.value_net, fw.system, use_target=False)

    pool = load_pool(Path(pool_path))
    scenes = pool.scenes if n is None else pool.scenes[:n]
    use_pool = EvaluationPool(name=f"v221_rc_{tag}", system=pool.system,
                              n_scenes=len(scenes), seed=getattr(pool, "seed", SEED), scenes=scenes)
    result = evaluate(fw, use_pool, config, mode="diagnostic_v221_remaining_collision",
                      step=int(checkpoint.get("step", 0)), ckpt_name=ckpt_path.name,
                      include_lqr_baseline=False, eval_batch_size=min(EVAL_BATCH, len(scenes)))

    index_rows = []
    for idx, ep in enumerate(result.trajectories):
        outcome = ep.filtered_outcome
        trace = D._episode_trace(framework=fw, h_fn=h_fn, config=config, scene=ep.scene,
                                 result=ep.filtered, outcome=outcome,
                                 event_step=int(ep.filtered_event_step))
        D._write_episode_npz(ep_dir, idx, outcome, "ep", trace)
        index_rows.append({"idx": idx, "outcome": outcome, "event_step": int(ep.filtered_event_step)})

    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "outcome", "event_step"])
        w.writeheader(); w.writerows(index_rows)
    (out / "eval_row.json").write_text(json.dumps({k: _jsonable(v) for k, v in result.eval_row.items()},
                                                  indent=2), encoding="utf-8")
    print(f"[{tag}] rolled out {len(index_rows)} episodes -> {out}")
    return out


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def load_cache(tag: str):
    """Return (episodes:list[dict], eval_row:dict). Each episode dict has outcome/event_step + trace arrays."""
    out = CACHE_ROOT / tag
    ep_dir = out / "episodes"
    eval_row = json.loads((out / "eval_row.json").read_text())
    by_idx = {int(r["idx"]): r for r in csv.DictReader((out / "index.csv").open())}
    eps = []
    for path in sorted(ep_dir.glob("ep_*.npz")):
        idx = int(path.stem.split("_")[1])
        z = np.load(path, allow_pickle=False)
        md = json.loads(str(np.asarray(z["metadata_json"]).item()))
        ep = {k: np.asarray(z[k]) for k in z.files if k != "metadata_json"}
        ep.update({k: v for k, v in md.items()})
        ep["idx"] = idx
        ep["outcome"] = str(md.get("outcome", by_idx.get(idx, {}).get("outcome", "?")))
        ep["event_step"] = int(md.get("event_step", by_idx.get(idx, {}).get("event_step", -1)))
        eps.append(ep)
    eps.sort(key=lambda e: e["idx"])
    return eps, eval_row


# ================================================================================================== #
# Shared per-collision feature extraction + 4-way classifier                                          #
# ================================================================================================== #
def _danger_index(ep, event) -> int:
    clearance = ep["clearance"]; brake = ep["brake_margin"]
    cand = np.where((clearance < DANGER_CLEAR) | (brake < 0.0))[0]
    cand = cand[cand <= event]
    return int(cand[0]) if cand.size else max(0, event - PRE_WINDOW + 1)


def _n_nearby_obstacles(ep, step) -> int:
    pos = np.asarray(ep["states"])[step, :2]
    centers = np.asarray(ep["obstacle_centers"], float)
    radii = np.asarray(ep["obstacle_radii"], float)
    active = np.asarray(ep["obstacle_active"], bool)
    if active.sum() == 0:
        return 0
    surf = np.linalg.norm(centers[active] - pos[None, :], axis=1) - radii[active]
    return int(np.sum(surf < NEAR_RADIUS))


def _feasible_brake_in_time(ep, event) -> bool:
    """True if some pre-impact step with lead>=ACT_MIN had brake_margin>0 (avoidable with time to act)."""
    brake = ep["brake_margin"]
    hi = max(0, event - ACT_MIN + 1)            # steps with lead >= ACT_MIN are indices [0, event-ACT_MIN]
    return bool(np.any(brake[:hi] > 0.0)) if hi > 0 else False


def _alarm_lead(ep, event) -> int:
    """Sustained early-alarm lead: ref - e*, e* earliest step h stays >= H_ALARM through impact. 0 if off at impact."""
    h = ep["h"]
    if event < 0 or event >= h.size or h[event] < H_ALARM:
        return 0
    j = event
    while j - 1 >= 0 and h[j - 1] >= H_ALARM:
        j -= 1
    return event - j


def _pre_closing(ep, event) -> float:
    lo = max(0, event - PRE_WINDOW + 1)
    seg = ep["closing_speed"][lo:event + 1]
    return float(np.max(seg)) if seg.size else float(ep["closing_speed"][event])


def classify_collision_4way(ep) -> dict:
    """Classify one collision episode into fast_approach_myopia / slow_approach / geometric_trap /
    doomed_at_entry, returning the class + the sub-signals used (for transparency)."""
    event = D._event_action_index(ep)
    danger = _danger_index(ep, event)
    n_near = _n_nearby_obstacles(ep, danger)
    pre_close = _pre_closing(ep, event)
    feasible = _feasible_brake_in_time(ep, event)
    alarm = _alarm_lead(ep, event)
    geom = ep.get("scene_geometry", {})
    info = {"event": event, "danger": danger, "n_near": n_near, "pre_closing": pre_close,
            "feasible_brake_in_time": feasible, "alarm_lead": alarm,
            "event_clearance": float(ep["clearance"][event]), "event_h": float(ep["h"][event]),
            "event_lg_norm": float(ep["lg_norm"][event]),
            "obstacle_count": int(geom.get("obstacle_count", 0)) if isinstance(geom, dict) else 0,
            "mechanism_v220": D._classify_mechanism("collision", ep)}
    # decision tree (order matters; documented in the module header)
    if n_near >= MULTI_OBSTACLE:
        cls = "geometric_trap"
    elif not feasible:
        cls = "doomed_at_entry"          # brake was never feasible with >= ACT_MIN steps to act
    elif pre_close < SLOW_SPEED:
        cls = "slow_approach"            # low inward speed -> dh/dv not the bottleneck
    else:
        cls = "fast_approach_myopia"     # high speed, brake feasible earlier, dh/dv too weak
    info["class"] = cls
    return info


# ================================================================================================== #
# Analyses                                                                                            #
# ================================================================================================== #
def analysis1_reclassify(treated, sota) -> dict:
    res = {}
    for tag, eps in (("SOTA", sota), ("TREATED", treated)):
        cols = [classify_collision_4way(e) for e in eps if e["outcome"] == COLLISION]
        n = len(cols)
        counts = {c: 0 for c in ("fast_approach_myopia", "slow_approach", "geometric_trap", "doomed_at_entry")}
        for c in cols:
            counts[c["class"]] += 1
        res[tag] = {
            "n_collisions": n,
            "counts": counts,
            "fractions": {k: (v / n if n else 0.0) for k, v in counts.items()},
            "early_alarm_frac_of_fast": (
                float(np.mean([c["alarm_lead"] >= ACT_MIN for c in cols if c["class"] == "fast_approach_myopia"]))
                if any(c["class"] == "fast_approach_myopia" for c in cols) else None),
            "mean_pre_closing": float(np.mean([c["pre_closing"] for c in cols])) if n else None,
            "mean_event_lg_norm": float(np.mean([c["event_lg_norm"] for c in cols])) if n else None,
            "mechanism_v220": _tally([c["mechanism_v220"] for c in cols]),
            "per_episode": cols,
        }
    fa_sota = res["SOTA"]["fractions"]["fast_approach_myopia"]
    fa_treated = res["TREATED"]["fractions"]["fast_approach_myopia"]
    res["verdict"] = {
        "fast_approach_myopia_SOTA": fa_sota,
        "fast_approach_myopia_TREATED": fa_treated,
        "delta": fa_treated - fa_sota,
        "interpretation": (
            "fast-approach-myopia fraction DROPPED SOTA->TREATED: prescription worked, bottleneck moved"
            if fa_treated < fa_sota - 0.05 else
            "fast-approach-myopia fraction roughly UNCHANGED: 0.65 dh/dv still insufficient for the eval regime"
            if abs(fa_treated - fa_sota) <= 0.05 else
            "fast-approach-myopia fraction ROSE SOTA->TREATED")}
    return res


def analysis2_speed_band(treated, sota) -> dict:
    # collision-step inward speeds (TREATED) vs gate band [0.6, 2.5]
    coll = [e for e in treated if e["outcome"] == COLLISION]
    coll_speeds = [float(e["closing_speed"][D._event_action_index(e)]) for e in coll]
    in_gate = float(np.mean([(GATE_S_LO <= s <= GATE_S_HI) for s in coll_speeds])) if coll_speeds else None
    below_gate = float(np.mean([s < GATE_S_LO for s in coll_speeds])) if coll_speeds else None

    # deployed ||L_g h|| vs inward-speed band over ALL active rollout states
    def band_curve(eps):
        sp, lg = [], []
        for e in eps:
            a = int(e.get("active_steps", len(e["h"])))
            sp.append(np.asarray(e["closing_speed"])[:a])
            lg.append(np.asarray(e["lg_norm"])[:a])
        sp = np.concatenate(sp) if sp else np.array([])
        lg = np.concatenate(lg) if lg else np.array([])
        curve = []
        for lo, hi in zip(SPEED_BANDS[:-1], SPEED_BANDS[1:]):
            m = (sp >= lo) & (sp < hi)
            curve.append({"band": f"[{lo},{hi})", "n": int(m.sum()),
                          "lg_mean": float(lg[m].mean()) if m.any() else None,
                          "lg_p50": float(np.median(lg[m])) if m.any() else None})
        return curve
    return {
        "n_collisions_treated": len(coll),
        "collision_step_inward_speed": {
            "median": float(np.median(coll_speeds)) if coll_speeds else None,
            "p25": float(np.percentile(coll_speeds, 25)) if coll_speeds else None,
            "p75": float(np.percentile(coll_speeds, 75)) if coll_speeds else None,
            "frac_in_gate_band_0.6_2.5": in_gate,
            "frac_below_gate_floor_0.6": below_gate,
            "values": coll_speeds},
        "lg_vs_speed_TREATED": band_curve(treated),
        "lg_vs_speed_SOTA": band_curve(sota),
        "gate_training_band": [GATE_S_LO, GATE_S_HI]}


def analysis3_transition(treated, sota) -> dict:
    so = {e["idx"]: e["outcome"] for e in sota}
    tr = {e["idx"]: e["outcome"] for e in treated}
    shared = sorted(set(so) & set(tr))
    outcomes = OUTCOME_ORDER + sorted({o for o in list(so.values()) + list(tr.values()) if o not in OUTCOME_ORDER})
    mat = {a: {b: 0 for b in outcomes} for a in outcomes}
    for i in shared:
        mat.setdefault(so[i], {b: 0 for b in outcomes})
        mat[so[i]].setdefault(tr[i], 0)
        mat[so[i]][tr[i]] += 1
    coll_to_stuck = sum(1 for i in shared if so[i] == "collision" and tr[i] == "stuck")
    stuck_to_coll = sum(1 for i in shared if so[i] == "stuck" and tr[i] == "collision")
    coll_to_reach = sum(1 for i in shared if so[i] == "collision" and tr[i] == "goal")
    reach_to_coll = sum(1 for i in shared if so[i] == "goal" and tr[i] == "collision")
    return {"n_shared": len(shared), "outcomes": outcomes, "matrix_sota_x_treated": mat,
            "collision_to_stuck": coll_to_stuck, "stuck_to_collision": stuck_to_coll,
            "collision_to_reach": coll_to_reach, "reach_to_collision": reach_to_coll,
            "net_collision_to_stuck": coll_to_stuck - stuck_to_coll}


def analysis4_infeasibility(treated, sota, treated_row, sota_row) -> dict:
    def per_step(eps):
        emp, inf, sat, proj = [], [], [], []
        for e in eps:
            a = int(e.get("active_steps", len(e["h"])))
            emp.append(np.asarray(e["empty_halfspace_box"])[:a].astype(float))
            if "infeasible" in e:
                inf.append(np.asarray(e["infeasible"])[:a].astype(float))
            sat.append(np.asarray(e["saturated"])[:a].astype(float))
            proj.append(np.asarray(e["projection_norm"])[:a].astype(float))
        cat = lambda L: np.concatenate(L) if L else np.array([])
        emp, inf, sat, proj = cat(emp), cat(inf), cat(sat), cat(proj)
        return {"empty_halfspace_box_rate": float(emp.mean()) if emp.size else None,
                "infeasible_rate": float(inf.mean()) if inf.size else None,
                "saturated_rate": float(sat.mean()) if sat.size else None,
                "projection_mean": float(proj.mean()) if proj.size else None,
                "projection_p95": float(np.percentile(proj, 95)) if proj.size else None}
    return {
        "TREATED": {"eval_row": {k: treated_row.get(k) for k in ("infeasibility", "saturation_rate", "collision", "stuck", "reach", "cps")},
                    "per_step": per_step(treated)},
        "SOTA": {"eval_row": {k: sota_row.get(k) for k in ("infeasibility", "saturation_rate", "collision", "stuck", "reach", "cps")},
                 "per_step": per_step(sota)}}


def _tally(items):
    out = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ================================================================================================== #
# Figures + results doc                                                                               #
# ================================================================================================== #
def _figs(a1, a2, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = ["fast_approach_myopia", "slow_approach", "geometric_trap", "doomed_at_entry"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    x = np.arange(len(classes)); w = 0.38
    ax.bar(x - w / 2, [a1["SOTA"]["fractions"][c] for c in classes], w, label=f"SOTA (n={a1['SOTA']['n_collisions']})", color="#7f7f7f")
    ax.bar(x + w / 2, [a1["TREATED"]["fractions"][c] for c in classes], w, label=f"TREATED (n={a1['TREATED']['n_collisions']})", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_ylabel("fraction of collisions"); ax.set_title("A1 remaining-collision class: SOTA vs TREATED")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "a1_reclassify.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    sp = a2["collision_step_inward_speed"]["values"]
    ax[0].hist(sp, bins=np.linspace(0, 2.6, 27), color="#d62728", alpha=0.8)
    ax[0].axvspan(GATE_S_LO, GATE_S_HI, color="#1f77b4", alpha=0.15, label="gate band [0.6,2.5]")
    ax[0].axvline(GATE_S_LO, color="#1f77b4", ls="--", lw=1)
    ax[0].set_xlabel("inward speed at collision step"); ax[0].set_ylabel("collisions (TREATED)")
    ax[0].set_title("A2 collision speeds vs gate band"); ax[0].legend(frameon=False, fontsize=8)
    for tag, st, col in (("lg_vs_speed_SOTA", "s--", "#7f7f7f"), ("lg_vs_speed_TREATED", "o-", "#d62728")):
        centers = [0.5 * (SPEED_BANDS[i] + SPEED_BANDS[i + 1]) for i in range(len(SPEED_BANDS) - 1)]
        ys = [(b["lg_mean"] if b["lg_mean"] is not None else np.nan) for b in a2[tag]]
        ax[1].plot(centers, ys, st, color=col, label=tag.replace("lg_vs_speed_", ""))
    ax[1].axvspan(GATE_S_LO, GATE_S_HI, color="#1f77b4", alpha=0.12)
    ax[1].axhline(0.48, color="0.5", ls=":", lw=1, label="prior 0.48")
    ax[1].set_xlabel("inward-speed band"); ax[1].set_ylabel("deployed ||L_g h|| mean")
    ax[1].set_title("A2 dh/dv vs speed (eval states)"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "a2_speed_band.png"); plt.close(fig)


def write_doc(meta, a1, a2, a3, a4):
    L = []
    L.append("# v2.2.1 — remaining-collision diagnosis (post-L_feas-treatment)\n")
    L.append(f"_TREATED_: `{meta['treated_ckpt']}` (step {meta['treated_step']})  ")
    L.append(f"_SOTA (untreated, same-seed)_: `{meta['sota_ckpt']}` (step {meta['sota_step']})  ")
    L.append(f"_pool_: `{meta['pool']}` (N={meta['n']})  | seed={SEED} | device={meta['device']}\n")
    L.append("Read-only post-hoc; reuses the v2.2.0 harness (`jt_failure_dissect._episode_trace`, "
             "`_classify_mechanism`, `_event_action_index`; `src.eval.evaluate.evaluate`) so definitions match.\n")

    v = a1["verdict"]
    L.append("## Core verdict\n")
    L.append(f"- fast-approach-myopia fraction of collisions: **SOTA {v['fast_approach_myopia_SOTA']:.3f} -> "
             f"TREATED {v['fast_approach_myopia_TREATED']:.3f}** (delta {v['delta']:+.3f}).")
    L.append(f"- **{v['interpretation']}**\n")

    L.append("## A1 remaining-collision reclassification (SOTA vs TREATED)\n")
    L.append("| class | SOTA frac (n) | TREATED frac (n) |")
    L.append("|---|---|---|")
    for c in ("fast_approach_myopia", "slow_approach", "geometric_trap", "doomed_at_entry"):
        L.append(f"| {c} | {a1['SOTA']['fractions'][c]:.3f} ({a1['SOTA']['counts'][c]}) | "
                 f"{a1['TREATED']['fractions'][c]:.3f} ({a1['TREATED']['counts'][c]}) |")
    L.append(f"\n- total collisions: SOTA {a1['SOTA']['n_collisions']}, TREATED {a1['TREATED']['n_collisions']}.")
    L.append(f"- mean inward speed at collision (pre-window peak): SOTA {a1['SOTA']['mean_pre_closing']}, "
             f"TREATED {a1['TREATED']['mean_pre_closing']}.")
    L.append(f"- mean deployed ||L_g h|| at collision step: SOTA {a1['SOTA']['mean_event_lg_norm']}, "
             f"TREATED {a1['TREATED']['mean_event_lg_norm']}.")
    L.append(f"- v2.2.0 mechanism tally (TREATED): {a1['TREATED']['mechanism_v220']}")
    L.append(f"- v2.2.0 mechanism tally (SOTA): {a1['SOTA']['mechanism_v220']}\n")

    s = a2["collision_step_inward_speed"]
    L.append("## A2 speed-band match (did we strengthen dh/dv where collisions happen?)\n")
    L.append(f"- collision-step inward speed (TREATED): median {s['median']}, p25 {s['p25']}, p75 {s['p75']}.")
    L.append(f"- fraction of collisions inside the gate band [0.6,2.5]: **{s['frac_in_gate_band_0.6_2.5']}**; "
             f"below the gate floor 0.6: **{s['frac_below_gate_floor_0.6']}**.")
    L.append("\n| inward-speed band | n states | ||L_g h|| mean SOTA | ||L_g h|| mean TREATED |")
    L.append("|---|---|---|---|")
    for bt, bs in zip(a2["lg_vs_speed_TREATED"], a2["lg_vs_speed_SOTA"]):
        L.append(f"| {bt['band']} | {bt['n']} | {bs['lg_mean']} | {bt['lg_mean']} |")
    L.append("")

    L.append("## A3 collision->stuck transition (over-braking check)\n")
    L.append(f"- shared scenes: {a3['n_shared']}.")
    L.append(f"- collision(SOTA)->stuck(TREATED): **{a3['collision_to_stuck']}**; "
             f"stuck(SOTA)->collision(TREATED): {a3['stuck_to_collision']}; "
             f"net collision->stuck: **{a3['net_collision_to_stuck']:+d}**.")
    L.append(f"- collision(SOTA)->reach(TREATED): {a3['collision_to_reach']}; "
             f"reach(SOTA)->collision(TREATED): {a3['reach_to_collision']}.")
    outs = a3["outcomes"]
    L.append("\n| SOTA \\ TREATED | " + " | ".join(outs) + " |")
    L.append("|---|" + "|".join(["---"] * len(outs)) + "|")
    for a in outs:
        if a in a3["matrix_sota_x_treated"]:
            L.append(f"| {a} | " + " | ".join(str(a3["matrix_sota_x_treated"][a].get(b, 0)) for b in outs) + " |")
    L.append("")

    L.append("## A4 infeasibility side-effect (the cps tax)\n")
    L.append("| metric | SOTA | TREATED |")
    L.append("|---|---|---|")
    for k in ("collision", "stuck", "reach", "cps", "infeasibility", "saturation_rate"):
        L.append(f"| eval_row.{k} | {a4['SOTA']['eval_row'].get(k)} | {a4['TREATED']['eval_row'].get(k)} |")
    for k in ("empty_halfspace_box_rate", "infeasible_rate", "saturated_rate", "projection_mean", "projection_p95"):
        L.append(f"| per_step.{k} | {a4['SOTA']['per_step'][k]} | {a4['TREATED']['per_step'][k]} |")
    L.append("")
    L.append("Figures: `a1_reclassify.png`, `a2_speed_band.png` (under the cache dir).\n")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[doc] -> {DOC} ({len(L)} lines)")


# ================================================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treated-ckpt", type=Path, default=DEFAULT_TREATED)
    ap.add_argument("--sota-ckpt", type=Path, default=DEFAULT_SOTA)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=None, help="limit scenes (default: full pool)")
    ap.add_argument("--smoke", type=int, default=0, help="tiny CPU smoke on N scenes (SOTA used for both tags)")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    global DOC
    if args.smoke > 0:
        device = torch.device("cpu")
        treated_ckpt = sota_ckpt = args.sota_ckpt
        n = args.smoke
        treated_tag, sota_tag = "treated_smoke", "sota_smoke"
        DOC = Path("/tmp/v221_remaining_collision_diagnosis_smoke.md")
        print(f"[smoke] N={n} on CPU; SOTA ckpt for both tags; doc -> {DOC}")
    else:
        treated_ckpt, sota_ckpt, n = args.treated_ckpt, args.sota_ckpt, args.n
        treated_tag, sota_tag = "treated", "sota"

    build_cache(sota_tag, sota_ckpt, args.pool, device, n)
    build_cache(treated_tag, treated_ckpt, args.pool, device, n)
    treated, treated_row = load_cache(treated_tag)
    sota, sota_row = load_cache(sota_tag)

    a1 = analysis1_reclassify(treated, sota)
    a2 = analysis2_speed_band(treated, sota)
    a3 = analysis3_transition(treated, sota)
    a4 = analysis4_infeasibility(treated, sota, treated_row, sota_row)

    meta = {"treated_ckpt": str(treated_ckpt), "sota_ckpt": str(sota_ckpt), "pool": str(args.pool),
            "n": (n if n is not None else len(treated)), "device": str(device),
            "treated_step": _ckpt_step(treated_ckpt), "sota_step": _ckpt_step(sota_ckpt)}
    fig_dir = CACHE_ROOT / ("_smoke_figs" if args.smoke else "figures")
    _figs(a1, a2, fig_dir)
    summary = {"meta": meta, "A1": _drop(a1, "per_episode"), "A2": a2, "A3": a3, "A4": a4}
    (CACHE_ROOT / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    write_doc(meta, a1, a2, a3, a4)
    print("A1 verdict:", a1["verdict"]["interpretation"])
    return 0


def _ckpt_step(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        return -1


def _drop(d, key):
    out = json.loads(json.dumps(d, default=_jsonable))
    for v in out.values():
        if isinstance(v, dict):
            v.pop(key, None)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
