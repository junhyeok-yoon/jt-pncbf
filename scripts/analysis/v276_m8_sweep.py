"""v2.7.6 M8.2 — fallback cost sweep over phase count and k. Checkpoint = JT step 42000 (09c33bf4), pool =
full-range seed42, BANDED scoring. Arms: phases in {1,2} x k in {1,2,3,4,5} (ten) + mode=none floor. Per arm:
cps + component breakdown + scene-bootstrap 95% CIs, and the collision split cylinder / band (floor/ceiling).
Eval forced onto GPU (cps is device-invariant; the vectorized M8.0 kstep makes this tractable). k ascending
within each phase count; a clear plateau may end a phase early (reported). Selection rule applied at the end.
Latency is measured separately (M8.3). Eval-only. Artifacts under stage2_eval/."""
from __future__ import annotations

import copy, json, time
from pathlib import Path

import torch

from src.eval.evaluate import evaluate
from src.eval.bootstrap import within_seed_ci
from src.eval.run_full import _load_framework as load_fw
from scripts.analysis.v276_m7_fallback_trial import _split, POOLS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
STEM = "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"
JT42 = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
OUT = REPO / "data/runs/v2.7.6/stage2_eval"
COMPS = ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(mode, phases, k, scoring="banded"):
    ck = torch.load(JT42, map_location="cpu", weights_only=False)
    boot = ck["config"]["eval"]["bootstrap"]; n_boot = int(boot["n_resample"]); bseed = int(boot["seed"])
    filt = copy.deepcopy(ck["config"]["filter"])
    if mode == "none":
        filt["empty_fallback"] = {"mode": "none", "k": int(filt.get("empty_fallback", {}).get("k", 10))}
        label = "none"; n_cand = 0
    else:
        filt["empty_fallback"] = {"mode": "kstep", "k": k, "phases": phases}
        label = f"p{phases}_k{k}"; n_cand = phases * 25 * k
    bc = 0.0 if scoring == "legacy" else 4.0
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": bc},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, _ = load_fw(JT42, config_overrides=over)
    for net in ("value_net", "policy_net"):
        m = getattr(fw, net, None)
        if m is not None:
            m.to(DEV)
    t0 = time.perf_counter()
    res = evaluate(fw, POOLS / f"{STEM}.pkl", cfg, mode="final", step=int(ck["step"]),
                   ckpt_name="step_042000.pt", max_scenes=None, include_lqr_baseline=False)
    wall = round(time.perf_counter() - t0, 1)
    rows = list(res.episode_rows)
    ci = within_seed_ci(rows, n_resample=n_boot, seed=bseed)
    agg = {"label": label, "mode": mode, "phases": phases, "k": k, "scoring": scoring,
           "n_candidate_evals": n_cand, "n": len(rows), "wall_s": wall, "device": str(DEV)}
    for m in COMPS:
        agg[m] = round(float(ci["mean"][m]), 6); agg[m + "_ci"] = [round(float(ci["ci"][m]["lo"]), 6), round(float(ci["ci"][m]["hi"]), 6)]
    agg["split"] = _split(res, n_boot, bseed)
    s = agg["split"]
    print(f"[{label}/{scoring}] cps {agg['cps']:+.5f} {agg['cps_ci']} reach {agg['reach']:.4f} coll {agg['collision']:.4f} "
          f"band {s['band_total']['count']} (fl {s['band_floor']['count']}/cl {s['band_ceiling']['count']}) "
          f"cand {n_cand} wall {wall}s", flush=True)
    return agg


if __name__ == "__main__":
    cells = [run("none", 0, 0, "banded")]
    for phases in (1, 2):
        for k in (1, 2, 3, 4, 5):
            cells.append(run("kstep", phases, k, "banded"))
            (OUT / "m8_sweep.json").write_text(json.dumps({"pool": STEM, "scoring": "banded", "ckpt_sha8": "09c33bf4",
                "cells": cells}, indent=2) + "\n")
    # selection: cheapest (phases,then k asc) whose banded cps not CI-separated below two-phase k=5
    ref = next(c for c in cells if c["label"] == "p2_k5")
    ref_lo = ref["cps_ci"][0]
    cand = [c for c in cells if c["mode"] == "kstep"]
    cand.sort(key=lambda c: (c["phases"], c["k"]))                        # cheapest by (phases, then k)
    qualifying = [c for c in cand if not (c["cps_ci"][1] < ref_lo)]       # not CI-separated below ref
    sel = min(qualifying, key=lambda c: c["n_candidate_evals"]) if qualifying else ref
    ties = [c["label"] for c in qualifying if c["n_candidate_evals"] == sel["n_candidate_evals"]]
    rep = {"reference_p2_k5_cps": ref["cps"], "reference_ci": ref["cps_ci"],
           "selected_label": sel["label"], "selected_cps": sel["cps"], "selected_n_candidate_evals": sel["n_candidate_evals"],
           "ties_same_cost": ties, "rule": "cheapest by (phases,then k) asc whose banded cps not CI-separated below p2_k5"}
    print("\nSELECTION:", json.dumps(rep), flush=True)
    full = json.load(open(OUT / "m8_sweep.json")); full["selection"] = rep
    (OUT / "m8_sweep.json").write_text(json.dumps(full, indent=2) + "\n")
    print("M8.2 sweep done ->", OUT / "m8_sweep.json", flush=True)
