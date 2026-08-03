"""v2.8.1 S1 — dual-scoring three-cell final for the beta=30 JT deliverable (gate / cps_tilt60 / cps_bandopen).
Mirrors v280_m3_eval exactly EXCEPT: ANGRATE = 0.30 (the v2.8.1 S1 tightened terminal), no reproduction anchors
(new soft-encoder lineage), output under data/runs/v2.8.1/. The encoder (soft_topk, beta=30) resolves from the
checkpoint's own config; the eval overrides only the scoring env (terminal, band) + shipped fallback + dual_solve.
n=2000 seed42 GPU, shipped fallback {kstep,phases 1,k 3}."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL_D2R = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
POOL_NAV = REPO / "data/secured_data/pools/eval_navcone_quadrotor-3d-d2r_n2000_seed34567.pkl"
OUT = REPO / "data/runs/v2.8.1"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_SEED = 20260508
ANGRATE = 0.30                                         # v2.8.1 S1 terminal (goal_angrate_radius = 0.30)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--label", required=True)
a = ap.parse_args()
rng = np.random.default_rng(BOOT_SEED)


def boot(v, n=1000):
    N = len(v); idx = rng.integers(0, N, (n, N)); vv = v[idx].mean(axis=1)
    return [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))]


def run_cell(pool_path, band_terminates):
    from src.eval.build_pools import load_pool
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    filt["projection"] = filt.get("projection") or "dual_solve"
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": ANGRATE, "band_terminates": band_terminates},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = load_fw(a.ckpt, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    pool = load_pool(pool_path)
    res = evaluate(fw, pool, cfg, mode="final", step=int(ck2["step"]), ckpt_name=a.label,
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row; eps = res.episode_rows
    COMP = ["cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate"]
    CI = ["cps_ci_lo", "cps_ci_hi", "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi",
          "stuck_ci_lo", "stuck_ci_hi", "infeasibility_ci_lo", "infeasibility_ci_hi"]
    outcome = {k: (float(r[k]) if r.get(k) is not None else None) for k in COMP + CI}
    cause = np.array([e.get("collision_cause", "") for e in eps])
    decomp = {}
    for c in ("obstacle", "band_lower", "band_upper"):
        m = (cause == c).astype(float); decomp[c] = float(m.mean()); decomp[c + "_ci"] = boot(m)
    bc = np.array([int(e.get("band_crossings", 0)) for e in eps]); outc = np.array([e["outcome"] for e in eps])
    crossed = bc >= 1
    band = {"crossing_rate": float(crossed.mean()), "n_crossed": int(crossed.sum()),
            "reach_after_crossing_frac": float(((outc == "goal") & crossed).sum() / max(1, crossed.sum()))}
    encoder = getattr(fw.system, "encoder", "?"); beta = float(getattr(fw.system, "soft_beta", 0.0))
    return {"outcome": outcome, "collision_decomposition": decomp, "band": band, "n": len(eps),
            "encoder": encoder, "beta": beta}


rep = {"label": a.label, "ckpt": str(a.ckpt), "terminal_goal_angrate": ANGRATE,
       "pool_gate": "0ef3751b (canonical, band-terminate)", "pool_cps_tilt60": "navcone_seed34567 (tilt<=60)",
       "pool_cps_bandopen": "0ef3751b (canonical, band-open)"}
rep["gate"] = run_cell(POOL_D2R, True)
rep["cps_tilt60"] = run_cell(POOL_NAV, True)
rep["cps_bandopen"] = run_cell(POOL_D2R, False)
(OUT / f"dual_{a.label}.json").write_text(json.dumps(rep, indent=2) + "\n")
for cell in ("gate", "cps_tilt60", "cps_bandopen"):
    o = rep[cell]["outcome"]; d = rep[cell]["collision_decomposition"]; b = rep[cell]["band"]
    print(f"[{a.label}/{cell}] enc={rep[cell]['encoder']} beta={rep[cell]['beta']} | cps {o['cps']:.4f} "
          f"reach {o['reach']:.4f} coll {o['collision']:.4f} (o{d['obstacle']:.4f}/f{d['band_lower']:.4f}/"
          f"c{d['band_upper']:.4f}) | crossing_rate {b['crossing_rate']:.4f}", flush=True)
gr = rep["gate"]["outcome"]["reach"]
print(f"\nT2: gate-cell reach = {gr:.4f}  ({'PASS' if gr >= 0.8875 else 'FAIL'} vs >=0.8875)")
print(f"-> {OUT / f'dual_{a.label}.json'}")
