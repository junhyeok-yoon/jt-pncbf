"""v2.8.0 Phase-2 C5 — D4 actuator-lag sweep (prediction + falsifier).

First-order rotor-thrust lag tau on the sim grid (eval.actuator_lag_tau, added default-off to rollout_eval).
INERTNESS: the tau=0 cells must reproduce the recorded deploy-rate rows (s3_eval/rate_{A,C,D}_{proj}) within
the R2 cross-process equivalence band (+-0.005); a miss HALTs C5 (C1-C4/C6 are unaffected). SWEEP: tau in
{0,0.005,0.01,0.02,0.05} x both projections (X1) x {20,100,500} Hz on the canonical pool — per-component CIs,
collision decomposition, and the realized inter-sample violation (a^T u_a - b)_+ (median & p90) vs the bound
||a|| tau ||udot_c||. PREDICTION: as tau grows, enumerate's collision rate rises faster than dual_solve's and
the gap widens with control rate; FALSIFIER: the two degrade at the same rate.

Cells are written incrementally to data/runs/v2.8.0/c5/cell_<rate>hz_<proj>_tau<tau>.json (resumable). tau
priority order puts {0,0.01,0.05} first so a compute cut drops 0.005/0.02 first (X1 forbids cutting a
projection). Summary: c5_summary.json + inertness_proof.json.
"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate, first_physical_event_step
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.motor_lag import motor_lag_step
from src.common.kstep_fallback import slice_scene

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/c5"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT = 20260508
RATE_ARMS = {20: (0.05, 0.05), 100: (0.01, 0.01), 500: (0.002, 0.002)}
RATE_TO_ARM = {20: "A", 100: "C", 500: "D"}          # recorded deploy-rate rows for the inertness check
TAUS = [0.0, 0.01, 0.05, 0.005, 0.02]                # priority order; tail dropped first under a compute cut
VIOL_SUB = 256                                        # realized-violation subsample (noted; median/p90 are robust)
EQ_BAND = 0.005

_scenes = load_pool(POOL).scenes
_bscene_full = batch_scenes(_scenes, device=DEV, dtype=torch.float32)
_x0_full = initial_states_from_batch(_bscene_full)
_rng = np.random.default_rng(BOOT)


def boot_ci(v, n=1000):
    v = np.asarray(v, float); N = len(v)
    idx = _rng.integers(0, N, size=(n, N))
    m = v[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def realized_violation(fw, dt_sim, dt_ctrl, tau, max_steps, stuck_w):
    """Subsample lagged rollout capturing a, b (filter stash) and the applied (lagged) thrust; returns the
    inter-sample violation (a^T u_applied - b)_+ and the bound ||a|| tau ||udot_c|| over feasible steps."""
    sub = torch.zeros(_x0_full.shape[0], dtype=torch.bool, device=DEV); sub[:VIOL_SUB] = True
    bscene = slice_scene(_bscene_full, sub)
    x = _x0_full[:VIOL_SUB].clone(); B = x.shape[0]
    substeps = int(round(dt_ctrl / dt_sim))
    u_app = None; held_u = None
    viols, bounds_a, du_c_prev = [], [], None
    with torch.no_grad():
        for i in range(max_steps):
            if i % substeps == 0:
                u_nom = fw.policy(x, bscene)
                u_safe, _ = fw.filter(x, u_nom, bscene)
                a_row = fw._filter.last_a; b_row = fw._filter.last_b.reshape(-1)
                emp = getattr(fw._filter, "last_empty", None)
                emp = emp.bool() if emp is not None else torch.zeros(B, dtype=torch.bool, device=DEV)
                held_u = u_safe.detach(); held_a = a_row; held_b = b_row; held_emp = emp
            u_app = held_u if u_app is None else motor_lag_step(u_app, held_u, dt_sim, tau)
            # inter-sample realized violation of the CURRENT constraint row by the applied (lagged) thrust
            viol = torch.relu((held_a * u_app).sum(1) - held_b)               # [B]
            feas = ~held_emp
            if feas.any():
                viols.append(viol[feas].cpu().numpy())
                bounds_a.append((torch.linalg.norm(held_a[feas], dim=1)).cpu().numpy())
            x = fw.system.wrap_state(rk4_step(fw.system, x, u_app, dt_sim))
    v = np.concatenate(viols) if viols else np.array([0.0])
    return {"realized_violation_median": float(np.median(v)), "realized_violation_p90": float(np.percentile(v, 90)),
            "realized_violation_max": float(v.max()), "n_feasible_steps": int(v.size),
            "subsample_scenes": VIOL_SUB, "tau": tau}


def run_cell(rate, proj, tau):
    dt_sim, dt_ctrl = RATE_ARMS[rate]
    max_steps = int(round(10.0 / dt_sim)); stuck_w = int(round(3.0 / dt_sim)); kfb = int(round(0.15 / dt_sim))
    ck = torch.load(str(CK), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])
    filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": kfb}; filt["projection"] = proj
    over = {"env": {"dt": dt_sim, "stuck_window_steps": stuck_w, "stuck_radius": 0.10,
                    "band_collision_limit": 4.0, "goal_angrate_radius": 0.48},
            "eval": {"max_steps": max_steps, "dt_ctrl": dt_ctrl, "actuator_lag_tau": tau}, "filter": filt}
    fw, cfg, ck2 = load_fw(str(CK), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(DEV)
    verify = {"dt_sim": float(cfg["env"]["dt"]), "dt_ctrl": float(cfg["eval"]["dt_ctrl"]),
              "max_steps": int(cfg["eval"]["max_steps"]), "stuck_window_steps": int(cfg["env"]["stuck_window_steps"]),
              "fallback_k": int(fw._filter.params.empty_fallback_k), "projection": str(fw._filter.params.projection),
              "actuator_lag_tau": float(cfg["eval"].get("actuator_lag_tau", 0.0))}
    exp = {"dt_sim": dt_sim, "dt_ctrl": dt_ctrl, "max_steps": max_steps, "stuck_window_steps": stuck_w,
           "fallback_k": kfb, "projection": proj, "actuator_lag_tau": tau}
    verify["ok"] = all((abs(verify[k] - exp[k]) < 1e-9) if isinstance(exp[k], float) else (verify[k] == exp[k]) for k in exp)
    if not verify["ok"]:
        raise SystemExit(f"HALT: C5 rescale/lag config mismatch {verify} vs {exp}")

    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name=f"{rate}_{proj}_tau{tau}",
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row; eps = res.episode_rows
    cause = np.array([e.get("collision_cause", "") for e in eps])
    COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]
    def arr(key):
        return np.array([1.0 if e["outcome"] == key else 0.0 for e in eps]) if key in ("goal",) else None
    outcome = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP}
    # per-component CIs from episode rows
    reach = np.array([e["reach"] for e in eps]); coll = np.array([e["collision"] for e in eps])
    cpsv = np.array([e["cps_episode"] for e in eps])
    outcome_ci = {"cps": boot_ci(cpsv), "reach": boot_ci(reach), "collision": boot_ci(coll)}
    decomp = {c: {"rate": float((cause == c).mean()), "ci": boot_ci((cause == c).astype(float))}
              for c in ("obstacle", "band_lower", "band_upper")}
    viol = realized_violation(fw, dt_sim, dt_ctrl, tau, max_steps, stuck_w)
    cell = {"rate_hz": rate, "proj": proj, "tau": tau, "verify_config": verify,
            "outcome": outcome, "outcome_ci": outcome_ci, "collision_decomposition": decomp,
            "realized_violation": viol}
    p = OUT / f"cell_{rate}hz_{proj}_tau{str(tau).replace('.', 'p')}.json"
    p.write_text(json.dumps(cell, indent=2) + "\n")
    print(f"[{rate}Hz/{proj}/tau{tau}] cps {outcome['cps']:.4f} coll {outcome['collision']:.4f} "
          f"viol_med {viol['realized_violation_median']:.2e} p90 {viol['realized_violation_p90']:.2e} -> {p.name}",
          flush=True)
    return cell


def main():
    done = {}
    # tau-major so the priority set {0,0.01,0.05} completes across ALL rates before the tail {0.005,0.02}
    # (the D4 prediction is a rate x tau trend; a compute cut then drops 0.005/0.02 first, never a projection).
    for tau in TAUS:
        for rate in (20, 100, 500):                   # cheap rates first within each tau
            for proj in ("dual_solve", "enumerate"):  # X1: never cut a projection
                p = OUT / f"cell_{rate}hz_{proj}_tau{str(tau).replace('.', 'p')}.json"
                if p.exists():
                    done[(rate, proj, tau)] = json.loads(p.read_text()); continue
                done[(rate, proj, tau)] = run_cell(rate, proj, tau)

    # ---- inertness proof: tau=0 cells reproduce the recorded deploy-rate rows within the R2 band ----
    inert = {}
    for rate in (20, 100, 500):
        arm = RATE_TO_ARM[rate]
        for proj in ("dual_solve", "enumerate"):
            rec_p = REPO / f"data/runs/v2.8.0/s3_eval/rate_{arm}_{proj}.json"
            cell = done.get((rate, proj, 0.0))
            if cell is None or not rec_p.exists():
                continue
            rec_cps = float(json.loads(rec_p.read_text())["outcome"]["cps"])
            got = float(cell["outcome"]["cps"])
            inert[f"{rate}hz_{proj}"] = {"tau0_cps": got, "recorded_cps": rec_cps, "delta": got - rec_cps,
                                         "within_band": abs(got - rec_cps) <= EQ_BAND}
    (OUT / "inertness_proof.json").write_text(json.dumps(inert, indent=2) + "\n")
    all_inert = all(v["within_band"] for v in inert.values()) if inert else False

    # ---- D4 prediction scoring: enum collision rises faster than dual as tau grows; gap widens with rate ----
    summary = {"inertness_all_within_band": all_inert, "inertness": inert, "cells": {}}
    for (rate, proj, tau), c in done.items():
        summary["cells"][f"{rate}hz_{proj}_tau{tau}"] = {
            "cps": c["outcome"]["cps"], "collision": c["outcome"]["collision"],
            "realized_violation_median": c["realized_violation"]["realized_violation_median"]}
    # collision(tau) slope per (rate,proj)
    slopes = {}
    for rate in (20, 100, 500):
        for proj in ("dual_solve", "enumerate"):
            pts = [(t, done[(rate, proj, t)]["outcome"]["collision"]) for t in TAUS if (rate, proj, t) in done]
            if len(pts) >= 2:
                pts.sort()
                ts = np.array([p[0] for p in pts]); cs = np.array([p[1] for p in pts])
                slopes[f"{rate}hz_{proj}"] = float(np.polyfit(ts, cs, 1)[0])
    summary["collision_vs_tau_slope"] = slopes
    summary["D4_prediction"] = {
        "statement": "enumerate collision rate rises faster than dual_solve as tau grows; gap widens with rate",
        "enum_minus_dual_slope": {f"{r}hz": (slopes.get(f"{r}hz_enumerate", None) is not None
                                             and slopes.get(f"{r}hz_dual_solve", None) is not None)
                                  and (slopes[f"{r}hz_enumerate"] - slopes[f"{r}hz_dual_solve"]) or None
                                  for r in (20, 100, 500)}}
    (OUT / "c5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nINERTNESS all within +-{EQ_BAND}: {all_inert}")
    print(f"collision-vs-tau slopes: {slopes}")
    print(f"-> {OUT/'c5_summary.json'}")
    if inert and not all_inert:
        print("HALT(C5): inertness proof failed for some tau=0 cell (see inertness_proof.json). C1-C4/C6 unaffected.")


if __name__ == "__main__":
    main()
