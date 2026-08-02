"""v2.8.0 S4 — three proofs for the dual-scoring standard, on the dual deliverable.

1. mixed full-provenance half (n=1000) CI-overlaps the canonical full pool (m4_dual, n=2000) — same scoring,
   different draw, so CI overlap (not bit-equality) is the check.
2. tilt60 cell bit-reproduces the completed cps_tilt60 (0.9206). HALT if not.
3. bandopen cell bit-reproduces the completed cps_bandopen (0.8818). HALT if not.
Sidecar: data/runs/v2.8.0/s4_proofs.json. Does not overwrite any m3_* sidecar."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, torch

from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL_D2R = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
POOL_NAV = REPO / "data/secured_data/pools/eval_navcone_quadrotor-3d-d2r_n2000_seed34567.pkl"
POOL_MIX = REPO / "data/secured_data/pools/eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678.pkl"
MIX_MAN = REPO / "data/secured_data/pools/eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678.manifest.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT = 20260508
# full-precision completed values from the m3 sidecar (the ledger 0.9206/0.8818 are 4-decimal roundings)
_M3 = json.loads((REPO / "data/runs/v2.8.0/m3_v2.8.0_dual.json").read_text())
RECORDED_TILT60 = float(_M3["cps_tilt60"]["outcome"]["cps"])
RECORDED_BANDOPEN = float(_M3["cps_bandopen"]["outcome"]["cps"])
CANON_CPS = 0.7919  # m4_dual (canonical full pool, band terminating)


def run(pool, band_terminates):
    ck = torch.load(str(CK), map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"]); filt["empty_fallback"] = {"mode": "kstep", "phases": 1, "k": 3}
    filt["projection"] = "dual_solve"
    over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                    "goal_angrate_radius": 0.48, "band_terminates": band_terminates},
            "eval": {"max_steps": 200, "dt_ctrl": 0.05}, "filter": filt}
    fw, cfg, ck2 = load_fw(str(CK), config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    return evaluate(fw, pool, cfg, mode="final", step=int(ck2["step"]), ckpt_name="s4", max_scenes=None,
                    include_lqr_baseline=False)


rng = np.random.default_rng(BOOT)
def boot(v, n=2000):
    N = len(v); idx = rng.integers(0, N, (n, N)); vv = v[idx].mean(axis=1)
    return [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))]


rep = {}
# Proof 2: tilt60 bit-repro
t = run(POOL_NAV, True); c_t = float(t.eval_row["cps"])
rep["proof2_tilt60"] = {"cps": c_t, "recorded": RECORDED_TILT60, "bit_repro": abs(c_t - RECORDED_TILT60) < 1e-9}
print(f"PROOF2 tilt60: {c_t:.6f} vs {RECORDED_TILT60} -> {'PASS' if rep['proof2_tilt60']['bit_repro'] else 'FAIL'}", flush=True)
# Proof 3: bandopen bit-repro
b = run(POOL_D2R, False); c_b = float(b.eval_row["cps"])
rep["proof3_bandopen"] = {"cps": c_b, "recorded": RECORDED_BANDOPEN, "bit_repro": abs(c_b - RECORDED_BANDOPEN) < 1e-9}
print(f"PROOF3 bandopen: {c_b:.6f} vs {RECORDED_BANDOPEN} -> {'PASS' if rep['proof3_bandopen']['bit_repro'] else 'FAIL'}", flush=True)
# Proof 1: mixed full-half CI-overlap vs canonical
prov = json.loads(MIX_MAN.read_text())["provenance"]
m = run(POOL_MIX, True); eps = m.episode_rows
cps_ep = np.array([float(e["cps_episode"]) for e in eps])
full_mask = np.array([p == "full" for p in prov])
full_cps = cps_ep[full_mask]; full_mean = float(full_cps.mean()); full_ci = boot(full_cps)
# canonical CI from m4_dual sidecar
m4 = json.loads((REPO / "data/runs/v2.8.0/s3_eval/m4_dual.json").read_text())["outcome"]
canon_ci = [m4["cps_ci_lo"], m4["cps_ci_hi"]]
overlap = not (full_ci[0] > canon_ci[1] or canon_ci[0] > full_ci[1])
rep["proof1_mixed_full_half"] = {"n_full": int(full_mask.sum()), "full_half_cps": full_mean, "full_half_ci": full_ci,
                                 "canonical_cps": CANON_CPS, "canonical_ci": canon_ci, "ci_overlap": overlap}
print(f"PROOF1 mixed full-half: cps {full_mean:.4f} {full_ci} vs canonical {CANON_CPS} {canon_ci} -> "
      f"CI-overlap {'YES' if overlap else 'NO'}", flush=True)
(REPO / "data/runs/v2.8.0/s4_proofs.json").write_text(json.dumps(rep, indent=2) + "\n")
if not (rep["proof2_tilt60"]["bit_repro"] and rep["proof3_bandopen"]["bit_repro"]):
    raise SystemExit("HALT: S4 bit-reproduction proof failed")
print("S4: proofs 2&3 bit-reproduce (HALTs pass); proof1 CI-overlap reported.")
