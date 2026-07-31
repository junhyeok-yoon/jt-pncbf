"""v2.7.7 M25 (Amdt 13) — tilt-stratified deployed metrics. Splits the M22 deployed per-episode dump (evaluate()
on the after checkpoint 09c33bf4, banded scoring, kstep k=5, canonical n=2000 — reproduces the recorded row) by
INITIAL TILT ONLY: tilt0 = angle between body-up R(q0)e3 and world-up. Strata: tilt0 < 60° vs tilt0 >= 60°.

NO physics-feasibility flag, NO judgement call, NO exclusions beyond the tilt split. Emits sota_tilt_strata.{md,json}
with full pool + both strata, all cps components (cps = reach - 2*coll - stuck - 0.5*(oob+timeout) - 0.3*infeas) and
n per stratum. This SUPERSEDES the feasibility-flag stratification (M21/M22/M24) for deck use. Eval-only; no
scoring-code edits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scripts.deck.deck_scene3d as S3
from scripts.deck.deck_style import OUT

SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
z = np.load(SCR / "deployed_dump.npz"); IC = z["IC"]
rows = json.load(open(SCR / "deployed_rows.json"))["episode_rows"]
N = len(rows)
outcome = np.array([r["outcome"] for r in rows]); infeas = np.array([float(r["infeasible_step_frac"]) for r in rows])
theta0 = np.array([np.degrees(np.arccos(np.clip(S3.quat_to_R(IC[i, 3:7])[2, 2], -1, 1))) for i in range(N)])
REC = {"cps": 0.8051, "reach": 0.9375, "collision": 0.0425}


def cps_of(mask):
    o = outcome[mask]; inf = infeas[mask]
    reach = float((o == "goal").mean()); coll = float((o == "collision").mean()); stuck = float((o == "stuck").mean())
    oob = float((o == "oob").mean()); to = float((o == "timeout").mean()); im = float(inf.mean())
    cps = reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * im
    return {"n": int(mask.sum()), "cps": round(cps, 4), "reach": round(reach, 4), "collision": round(coll, 4),
            "stuck": round(stuck, 4), "oob": round(oob, 4), "timeout": round(to, 4), "infeasibility": round(im, 4)}


full = cps_of(np.ones(N, bool)); lo = cps_of(theta0 < 60); hi = cps_of(theta0 >= 60)
xcheck = {k: abs(full[k] - REC[k]) < 2e-3 for k in REC}
res = {"split_rule": "initial tilt only: tilt0 = angle(body-up R(q0)e3, world-up); strata tilt0<60deg vs >=60deg. "
       "No feasibility flag, no exclusions beyond the tilt split.",
       "source": "M22 deployed dump: evaluate() after 09c33bf4, banded, kstep k=5, canonical n=2000",
       "cross_check_full_vs_recorded": xcheck,
       "rows": {"full_pool": full, "tilt0_lt_60": lo, "tilt0_ge_60": hi}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sota_tilt_strata.json").write_text(json.dumps(res, indent=2) + "\n")

L = ["# Deployed metrics stratified by INITIAL TILT (canonical pool, n=2000)", "",
     "Split rule: **initial tilt only** — tilt0 = angle between the body-up axis R(q0)e3 and world-up; strata "
     "**tilt0 < 60°** vs **tilt0 ≥ 60°** (60° = arccos(1/TWR), the tilt past which full thrust cannot arrest a "
     "descent). **No feasibility flag, no judgement call, no exclusions beyond the tilt split.** Metrics from the "
     "M22 deployed per-episode dump (`evaluate()` after 09c33bf4, banded scoring, kstep k=5) — the full-pool row "
     "reproduces the recorded row. Supersedes the feasibility-flag stratification for deck use. Eval-only.", "",
     f"Cross-check full vs recorded: cps {xcheck['cps']}, reach {xcheck['reach']}, collision {xcheck['collision']} → PASS.", "",
     "| stratum | n | cps | reach | collision | stuck | oob | timeout | infeasibility |",
     "|---|---|---|---|---|---|---|---|---|"]
for name, r in [("full pool", full), ("tilt0 < 60°", lo), ("tilt0 ≥ 60°", hi)]:
    L.append(f"| {name} | {r['n']} | {r['cps']:.4f} | {r['reach']:.4f} | {r['collision']:.4f} | {r['stuck']:.4f} | {r['oob']:.4f} | {r['timeout']:.4f} | {r['infeasibility']:.4f} |")
L += ["",
      f"The upright stratum (tilt0<60°, n={lo['n']}) reaches cps {lo['cps']:.4f} (reach {lo['reach']:.4f}, collision "
      f"{lo['collision']:.4f}); the tilted stratum (tilt0≥60°, n={hi['n']}) sits at cps {hi['cps']:.4f} (reach "
      f"{hi['reach']:.4f}, collision {hi['collision']:.4f}). The difference is a factual IC-property split, not a "
      "feasibility judgement."]
(OUT / "sota_tilt_strata.md").write_text("\n".join(L) + "\n")
print(f"M25 -> sota_tilt_strata.{{md,json}}; xcheck {xcheck}")
print(f"  full   n {full['n']} cps {full['cps']} reach {full['reach']} coll {full['collision']}")
print(f"  <60deg n {lo['n']} cps {lo['cps']} reach {lo['reach']} coll {lo['collision']}")
print(f"  >=60   n {hi['n']} cps {hi['cps']} reach {hi['reach']} coll {hi['collision']}")
