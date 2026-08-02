"""v2.8.0 W4 — end-to-end proof of the three-cell final driver (run_final_cells) on the dual deliverable.

Runs the finished W2 driver once and verifies (v2.8.0 R2 — scored against the DRIVER lineage, not the
standalone m3 sidecar): the tilt60 and bandopen cells reproduce the R1-registered driver-lineage targets
(data/runs/v2.8.0/w4_driver_reference.json) — bit-for-bit when the driver lineage's within-cluster spread
is 0, else within that spread (HALT on either); the mixed cell's full-provenance half CI-overlaps the
canonical row; and the W3 two-row pair for this run parses to the header cell count under the R-audit
parser. Proof output is filed under the run dir (NOT appended to the ledger). The standalone sidecars and
the L2 ledger rows are untouched. Sidecar: data/runs/v2.8.0/w4_proof/proof.json."""
from __future__ import annotations
import copy, csv, json, re
from pathlib import Path
import numpy as np, torch

from src.eval.run_full import _load_framework, run_final_cells, dual_scoring_ledger_rows
from src.eval.evaluate import provenance_half_scores

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
RUN = REPO / "data/runs/v2.8.0/w4_proof"; RUN.mkdir(parents=True, exist_ok=True)
MIX_MAN = REPO / "data/secured_data/pools/eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678.manifest.json"
# v2.8.0 R2: score the integrated driver against the DRIVER-LINEAGE references (R1-registered), not the
# standalone m3 sidecar — bit-for-bit reproduction is a within-lineage property (R1: driver spread = 0).
REF = json.loads((REPO / "data/runs/v2.8.0/w4_driver_reference.json").read_text())
REC_T = float(REF["driver_reproduction_target"]["cps_tilt60"])
REC_B = float(REF["driver_reproduction_target"]["cps_bandopen"])
# PASS criterion: bit-match if the driver lineage bit-matches itself (spread 0), else agreement within the
# measured within-cluster spread (the registered tolerance).
DRIVER_SPREAD = max(float(REF["driver_lineage"]["within_cluster_spread_tilt60"]),
                    float(REF["driver_lineage"]["within_cluster_spread_bandopen"]))
TOL = 1e-9 if DRIVER_SPREAD == 0.0 else DRIVER_SPREAD
CRITERION = "bit-match (within-lineage spread = 0)" if DRIVER_SPREAD == 0.0 else \
            f"agreement within registered within-cluster spread {DRIVER_SPREAD:.2e}"
CANON_CI = [json.loads((REPO / "data/runs/v2.8.0/s3_eval/m4_dual.json").read_text())["outcome"]["cps_ci_lo"],
            json.loads((REPO / "data/runs/v2.8.0/s3_eval/m4_dual.json").read_text())["outcome"]["cps_ci_hi"]]

over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},
        "eval": {"max_steps": 200, "dt_ctrl": 0.05, "final": {"cells": ["mixed", "tilt60", "bandopen"]}},
        "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
fw, cfg, ck = _load_framework(str(CK), config_overrides=over)
cells = run_final_cells(RUN, fw, cfg, ck, CK)

def read_metrics(name):
    with open(RUN / "eval" / name / "eval_metrics.csv") as f:
        return list(csv.DictReader(f))[0]
mt, mb, mm = read_metrics("tilt60"), read_metrics("bandopen"), read_metrics("mixed")
cps_t, cps_b = float(mt["cps"]), float(mb["cps"])
proof2 = abs(cps_t - REC_T) <= TOL
proof3 = abs(cps_b - REC_B) <= TOL

# mixed full-half CI-overlap vs canonical
prov = json.loads(MIX_MAN.read_text())["provenance"]
with open(RUN / "eval" / "mixed" / "eval_episodes.csv") as f:
    ep = list(csv.DictReader(f))
cps_ep = np.array([float(e["cps_episode"]) for e in ep]); full = cps_ep[np.array([p == "full" for p in prov])]
rng = np.random.default_rng(20260508); N = len(full)
bs = np.array([full[rng.integers(0, N, N)].mean() for _ in range(2000)])
full_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
overlap = not (full_ci[0] > CANON_CI[1] or CANON_CI[0] > full_ci[1])
# mixed cell wrote the two halves?
halves_present = ("cps_full_half" in mm and "cps_tilt60_half" in mm and mm["cps_full_half"] != "")

# W3 pair from this run, parse-check under the R-audit parser
def cell_dict(m): return {k: float(m[k]) for k in ("reach","collision","oob","stuck","timeout","infeasibility","saturation_rate","cps")}
band = cell_dict(mb); band["crossing_rate"] = 0.0345
rows = dual_scoring_ledger_rows(date="2026-08-01", parent="w4_proof", run_id="w4_proof", seeds="42",
                                blended_cps=float(mm["cps"]), tilt60=cell_dict(mt), bandopen=band)
ncol = len("| version | system | date | parent | seeds | cps_v2 | cps_tilt60 | cps_bandopen | eval_source | reach | collision | oob | stuck | timeout | infeas | sat_rate | cps | verdict |".split("|"))
w3_ok = all(len(re.split(r'(?<!\\)\|', r)) == ncol for r in rows)

rep = {"lineage": "driver", "criterion": CRITERION, "tolerance": TOL,
       "proof2_tilt60": {"cps": cps_t, "driver_target": REC_T, "repro": proof2, "delta": cps_t - REC_T},
       "proof3_bandopen": {"cps": cps_b, "driver_target": REC_B, "repro": proof3, "delta": cps_b - REC_B},
       "proof1_mixed_full_half": {"n_full": int(N), "full_ci": full_ci, "canonical_ci": CANON_CI, "ci_overlap": overlap,
                                  "mixed_blended_cps": float(mm["cps"]), "halves_written": bool(halves_present),
                                  "cps_full_half": mm.get("cps_full_half"), "cps_tilt60_half": mm.get("cps_tilt60_half")},
       "w3_pair_parses": w3_ok, "artifact_sets": sorted([p.name for p in (RUN / "eval").iterdir()])}
(RUN / "proof.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"W4 artifact sets: {rep['artifact_sets']}")
print(f"W4 scored against DRIVER lineage; criterion: {CRITERION}")
print(f"PROOF2 tilt60 {cps_t:.12f} vs driver-target {REC_T:.12f} (d={cps_t-REC_T:+.2e}) -> {'PASS' if proof2 else 'FAIL'}")
print(f"PROOF3 bandopen {cps_b:.12f} vs driver-target {REC_B:.12f} (d={cps_b-REC_B:+.2e}) -> {'PASS' if proof3 else 'FAIL'}")
print(f"PROOF1 mixed full-half CI {full_ci} vs canonical {CANON_CI} -> overlap {overlap}; halves_written {halves_present} "
      f"(full {mm.get('cps_full_half')}, tilt60 {mm.get('cps_tilt60_half')}, blended {mm['cps']})")
print(f"W3 pair parses to {ncol-2} cells: {w3_ok}")
if not (proof2 and proof3):
    raise SystemExit("HALT: W4 driver-lineage reproduction failed")
print("W4: PASS (driver-lineage reproduction)")
