"""v2.8.0 M3 — evaluate one quadrotor_3d checkpoint in three cells (gate / cps_tilt60 / cps_bandopen).

gate        : pool -d2r, band terminating (existing scoring) -> reproduction gate, NOT a third definition
cps_tilt60  : pool navcone (tilt<=60), band terminating       -> definition 1 (navigation)
cps_bandopen: pool -d2r, band PERMISSIVE (band_terminates=false) -> definition 2 (floor-permissive)
--terminal new|old selects goal_angrate_radius = 0.48 (new) or absent/inert (old, A3 for pre-v2.8.0 ckpts).
All n=2000, seed42, GPU, shipped fallback (kstep phases1 k3), new terminal (omega_G=0.48).
Per cell: full outcome vector + per-component CIs + collision decomposition. Free cell additionally: band
crossing rate, crossings-per-episode distribution, and reach-after-crossing count. If the checkpoint has a
recorded canonical row, the canonical cell must reproduce it (HALT). Sidecar: data/runs/v2.8.0/m3_<label>.json"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL_D2R = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
POOL_NAV = REPO / "data/secured_data/pools/eval_navcone_quadrotor-3d-d2r_n2000_seed34567.pkl"
OUT = REPO / "data/runs/v2.8.0"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_SEED = 20260508
# recorded canonical rows (new terminal, shipped fallback) that the canonical cell must reproduce
ANCHORS = {"v2.8.0_dual": {"cps": 0.7919, "reach": 0.9295, "collision": 0.0440},
           "v2.8.0_enum": {"cps": 0.7778, "reach": 0.9165, "collision": 0.0430},
           "v2.7.6": {"cps": -0.6670, "reach": 0.1935, "collision": 0.0435}}

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--terminal", default="new", choices=["new", "old"])  # A3: old = goal_angrate_radius absent
a = ap.parse_args()
ANGRATE = 0.48 if a.terminal == "new" else 1e12       # 1e12 = angular conjunct inert = old terminal
rng = np.random.default_rng(BOOT_SEED)


def boot(v, n=1000):
    N = len(v); idx = rng.integers(0, N, (n, N)); vv = v[idx].mean(axis=1)
    return [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))]


def run_cell(pool, band_terminates):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    filt["projection"] = filt.get("projection") or "dual_solve"
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": ANGRATE, "band_terminates": band_terminates},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = load_fw(a.ckpt, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
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
    cell = {"outcome": outcome, "collision_decomposition": decomp, "n": len(eps)}
    # band-crossing statistics (meaningful for the free cell; computed always)
    bc = np.array([int(e.get("band_crossings", 0)) for e in eps])
    outc = np.array([e["outcome"] for e in eps])
    crossed = bc >= 1
    cell["band"] = {
        "crossing_rate": float(crossed.mean()), "crossing_rate_ci": boot(crossed.astype(float)),
        "n_crossed": int(crossed.sum()),
        "crossings_per_episode": {str(k): int((bc == k).sum()) for k in range(0, int(bc.max()) + 1)} if bc.size else {},
        "mean_crossings_given_crossed": float(bc[crossed].mean()) if crossed.any() else 0.0,
        "reached_goal_after_crossing": int(((outc == "goal") & crossed).sum()),
        "reach_after_crossing_frac": float(((outc == "goal") & crossed).sum() / max(1, crossed.sum())),
    }
    return cell


rep = {"label": a.label, "ckpt": str(a.ckpt), "terminal": a.terminal, "pool_gate": "0ef3751b",
       "pool_cps_tilt60": "navcone_seed34567"}
# cell names (v2.8.0 A2): gate = existing-scoring reproduction cell (NOT a third definition);
# cps_tilt60 = def-1 (tilt<=60 pool, band terminating); cps_bandopen = def-2 (full range, band permissive)
rep["gate"] = run_cell(POOL_D2R, True)
rep["cps_tilt60"] = run_cell(POOL_NAV, True)
rep["cps_bandopen"] = run_cell(POOL_D2R, False)

# reproduction gate on the gate cell (only where a recorded row exists for this terminal)
anchors = ANCHORS if a.terminal == "new" else {}          # new-terminal anchors do not apply under old terminal
if a.label in anchors:
    anc = anchors[a.label]; got = rep["gate"]["outcome"]
    ok = all(abs(got[k] - anc[k]) < 1e-4 for k in anc)
    rep["gate_reproduction"] = {"anchor": anc, "got": {k: got[k] for k in anc}, "pass": ok}
    print(f"[{a.label}/{a.terminal}] gate reproduction {'PASS' if ok else 'FAIL'}: {[(k, round(got[k],4), anc[k]) for k in anc]}", flush=True)
else:
    rep["gate_reproduction"] = {"anchor": None, "note": f"no recorded {a.terminal}-terminal canonical row at shipped fallback; fresh"}

tag = a.label if a.terminal == "new" else f"{a.label}_old"
(OUT / f"m3_{tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
for cell in ("gate", "cps_tilt60", "cps_bandopen"):
    o = rep[cell]["outcome"]; d = rep[cell]["collision_decomposition"]; b = rep[cell]["band"]
    print(f"[{a.label}/{a.terminal}/{cell}] cps {o['cps']:.4f} reach {o['reach']:.4f} coll {o['collision']:.4f} "
          f"(o{d['obstacle']:.4f}/f{d['band_lower']:.4f}/c{d['band_upper']:.4f}) | crossing_rate {b['crossing_rate']:.4f} "
          f"reach_after_cross {b['reached_goal_after_crossing']}", flush=True)
if a.label in anchors and not rep["gate_reproduction"]["pass"]:
    raise SystemExit(f"HALT: {a.label} gate cell did not reproduce its recorded {a.terminal}-terminal row")
print(f"-> {OUT / f'm3_{tag}.json'}")
