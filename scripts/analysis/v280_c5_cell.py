"""v2.8.0 Phase-2 C5 (F2) — ONE lag-sweep cell per fresh process (rate_branch pattern).

Each invocation runs a single (rate, proj, tau) cell and writes one sidecar. Running one cell per process
(a) removes the cross-cell GPU accumulation that OOM'd the monolithic sweep, and (b) places every cell in the
same first-eval-order lineage as the recorded deploy-rate rows — so the tau=0 inertness bit-match is by
construction, not by accident. For tau=0 the cell self-checks against the recorded row and EXITS NON-ZERO if
it drifts >±0.005 (the driver's `set -e` then stops the sweep). Any OOM/error also exits non-zero.

Usage: --rate {20,100,500} --proj {dual_solve,enumerate} --tau <float> [--out <dir>]
Sidecar: <out>/cell_<rate>hz_<proj>_tau<tau>.json   (default out = data/runs/v2.8.0/c5)
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.motor_lag import motor_lag_step
from src.common.kstep_fallback import slice_scene

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT = 20260508
RATE_ARMS = {20: (0.05, 0.05), 100: (0.01, 0.01), 500: (0.002, 0.002)}
RATE_TO_ARM = {20: "A", 100: "C", 500: "D"}
VIOL_SUB = 256
EQ_BAND = 0.005


def realized_violation(fw, bscene_full, x0_full, dt_sim, dt_ctrl, tau, max_steps):
    sub = torch.zeros(x0_full.shape[0], dtype=torch.bool, device=DEV); sub[:VIOL_SUB] = True
    bscene = slice_scene(bscene_full, sub)
    x = x0_full[:VIOL_SUB].clone(); B = x.shape[0]
    substeps = int(round(dt_ctrl / dt_sim))
    u_app = None; held_u = held_a = held_b = held_emp = prev_u = None
    viols, bnds = [], []
    with torch.no_grad():
        for i in range(max_steps):
            if i % substeps == 0:
                u_nom = fw.policy(x, bscene)
                u_safe, _ = fw.filter(x, u_nom, bscene)
                held_a = fw._filter.last_a; held_b = fw._filter.last_b.reshape(-1)
                emp = getattr(fw._filter, "last_empty", None)
                held_emp = emp.bool() if emp is not None else torch.zeros(B, dtype=torch.bool, device=DEV)
                prev_u = held_u
                held_u = u_safe.detach()
            u_app = held_u if u_app is None else motor_lag_step(u_app, held_u, dt_sim, tau)
            feas = ~held_emp
            if feas.any():
                viol = torch.relu((held_a * u_app).sum(1) - held_b)
                viols.append(viol[feas].cpu().numpy())
                if prev_u is not None:
                    # analysis bound on the inter-sample violation: ||a|| * tau * ||udot_c||, udot_c the command
                    # derivative across the last control update (a check on the analysis; reported not explained).
                    udot = (held_u - prev_u) / dt_ctrl
                    bnd = torch.linalg.norm(held_a, dim=1) * tau * torch.linalg.norm(udot, dim=1)
                    bnds.append(bnd[feas].cpu().numpy())
            x = fw.system.wrap_state(rk4_step(fw.system, x, u_app, dt_sim))
    v = np.concatenate(viols) if viols else np.array([0.0])
    bd = np.concatenate(bnds) if bnds else np.array([0.0])
    return {"realized_violation_median": float(np.median(v)), "realized_violation_p90": float(np.percentile(v, 90)),
            "realized_violation_max": float(v.max()), "bound_median": float(np.median(bd)),
            "bound_p90": float(np.percentile(bd, 90)), "realized_le_bound_frac": float((v[:len(bd)] <= bd[:len(v)] + 1e-9).mean())
            if min(len(v), len(bd)) else 1.0, "n_feasible_steps": int(v.size), "subsample_scenes": VIOL_SUB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, required=True, choices=list(RATE_ARMS))
    ap.add_argument("--proj", required=True, choices=["dual_solve", "enumerate"])
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", default=str(REPO / "data/runs/v2.8.0/c5"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rate, proj, tau = a.rate, a.proj, a.tau
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
        print(f"HALT: rescale/lag config mismatch {verify} vs {exp}", flush=True); sys.exit(4)

    scenes = load_pool(POOL).scenes
    bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x0 = initial_states_from_batch(bscene)

    res = evaluate(fw, POOL, cfg, mode="final", step=int(ck2["step"]), ckpt_name=f"{rate}_{proj}_tau{tau}",
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row; eps = res.episode_rows
    cause = np.array([e.get("collision_cause", "") for e in eps])
    COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]
    outcome = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP}
    rng = np.random.default_rng(BOOT)
    def ci(v):
        v = np.asarray(v, float); N = len(v); idx = rng.integers(0, N, size=(1000, N))
        m = v[idx].mean(axis=1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    reach = np.array([e["reach"] for e in eps]); coll = np.array([e["collision"] for e in eps])
    cpsv = np.array([e["cps_episode"] for e in eps])
    outcome_ci = {"cps": ci(cpsv), "reach": ci(reach), "collision": ci(coll)}
    decomp = {c: {"rate": float((cause == c).mean()), "ci": ci((cause == c).astype(float))}
              for c in ("obstacle", "band_lower", "band_upper")}
    viol = realized_violation(fw, bscene, x0, dt_sim, dt_ctrl, tau, max_steps)

    cell = {"rate_hz": rate, "proj": proj, "tau": tau, "lineage": "driver_per_cell_fresh_process",
            "verify_config": verify, "outcome": outcome, "outcome_ci": outcome_ci,
            "collision_decomposition": decomp, "realized_violation": viol}

    # tau=0 inertness self-check against the recorded deploy-rate row (bit-match is by-construction lineage now)
    inert_status = "n/a"
    if tau == 0.0:
        rec_p = REPO / f"data/runs/v2.8.0/s3_eval/rate_{RATE_TO_ARM[rate]}_{proj}.json"
        if rec_p.exists():
            rec = float(json.loads(rec_p.read_text())["outcome"]["cps"])
            d = outcome["cps"] - rec
            cell["inertness"] = {"tau0_cps": outcome["cps"], "recorded_cps": rec, "delta": d, "within_band": abs(d) <= EQ_BAND}
            inert_status = f"delta {d:+.6f} {'OK' if abs(d) <= EQ_BAND else 'FAIL'}"

    p = out / f"cell_{rate}hz_{proj}_tau{str(tau).replace('.', 'p')}.json"
    p.write_text(json.dumps(cell, indent=2) + "\n")
    print(f"[{rate}Hz/{proj}/tau{tau}] cps {outcome['cps']:.6f} coll {outcome['collision']:.4f} "
          f"viol_med {viol['realized_violation_median']:.2e} p90 {viol['realized_violation_p90']:.2e} "
          f"inert[{inert_status}] -> {p.name}", flush=True)

    if tau == 0.0 and "inertness" in cell and not cell["inertness"]["within_band"]:
        print(f"HALT: inertness gate FAILED {rate}Hz/{proj} {inert_status}", flush=True); sys.exit(3)


if __name__ == "__main__":
    main()
