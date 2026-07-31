"""v2.7.7 M27 (Amdt 15) — tilt-at-crossing re-scoring of the band z-limit. From the legacy (band NON-terminal)
per-episode dump (evaluate() after 09c33bf4, kstep k=5, canonical n=2000), re-score every episode:
  - never crosses z=-z_lim              -> recorded (without-z-limit) outcome
  - crosses with tilt < 60 deg          -> band collision
  - crosses with tilt >= 60 deg         -> NOT a collision; keep the episode's eventual outcome (from the
                                           band-non-terminal rollout: reach / collision / timeout / arena-floor oob)
Emits sota_tilt_at_crossing.{md,json}: full cps components, the three category counts, the tilt-at-crossing
distribution, and the sub-count of crossings tilted >=60 deg for the ENTIRE run up to the crossing (never inside the
cone) — a check on the 'policy created the tilt' reading. Sanity: the re-scored cps must lie BETWEEN the recorded
with-z-limit (0.8051) and without-z-limit (0.9078) rows. This SUPERSEDES M25. Eval-only; no scoring-code edits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.deck.deck_style import OUT

SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
WITH_Z, WITHOUT_Z = 0.8051, 0.9078
z = np.load(SCR / "legacy_dump.npz", allow_pickle=True)
outcome = z["outcome"].astype(object); infeas = z["infeas"]; crossed = z["crossed"]
tilt_at_cross = z["tilt_at_cross"]; tilt_ge60_entire = z["tilt_ge60_entire"]
N = len(outcome)

cross_lt60 = crossed & (tilt_at_cross < 60)
cross_ge60 = crossed & (tilt_at_cross >= 60)
never = ~crossed
# re-scored outcome: only upright (tilt<60) crossings become band collisions; everything else keeps its
# band-non-terminal (eventual) outcome.
rescored = outcome.copy()
rescored[cross_lt60] = "collision"


def cps_components(o, inf):
    reach = float((o == "goal").mean()); coll = float((o == "collision").mean()); stuck = float((o == "stuck").mean())
    oob = float((o == "oob").mean()); to = float((o == "timeout").mean()); im = float(inf.mean())
    return {"cps": round(reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * im, 4), "reach": round(reach, 4),
            "collision": round(coll, 4), "stuck": round(stuck, 4), "oob": round(oob, 4), "timeout": round(to, 4),
            "infeasibility": round(im, 4), "n": int(len(o))}


row = cps_components(rescored, infeas)
sanity_ok = WITH_Z <= row["cps"] <= WITHOUT_Z
# eventual outcome of the tilt>=60 crossings (the ones we forgive)
ev = {k: int((cross_ge60 & (outcome == k)).sum()) for k in ("goal", "collision", "timeout", "oob", "stuck")}
# tilt-at-crossing distribution
tac = tilt_at_cross[crossed]
tbins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180.01)]
tdist = {f"[{lo},{int(hi)})": int(((tac >= lo) & (tac < hi)).sum()) for lo, hi in tbins}
ge60_entire = int((cross_ge60 & tilt_ge60_entire).sum())
res = {"method": "tilt-at-crossing re-scoring (M27, Amdt 15); band non-terminal rollout, kstep k=5",
       "categories": {"never_crosses": int(never.sum()), "crosses_tilt_lt_60_band_collision": int(cross_lt60.sum()),
                      "crosses_tilt_ge_60_keep_eventual": int(cross_ge60.sum())},
       "tilt_ge60_crossings_eventual_outcome": ev,
       "tilt_ge60_crossings_tilted_ge60_entire_before_crossing": ge60_entire,
       "tilt_at_crossing_distribution_deg": tdist,
       "rescored_row": row,
       "sanity": {"with_z_limit": WITH_Z, "rescored": row["cps"], "without_z_limit": WITHOUT_Z, "in_range": sanity_ok}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sota_tilt_at_crossing.json").write_text(json.dumps(res, indent=2) + "\n")

nc = int(cross_ge60.sum())
L = ["# Band z-limit — tilt-at-crossing re-scoring (canonical pool, n=2000)", "",
     "From an eval-only rollout of the deployed 3-D checkpoint (09c33bf4, kstep k=5) with **band contact "
     "NON-terminal** (the without-z-limit configuration, so a floor crossing does not end the episode), logging "
     "p_z and tilt per step. Each episode is re-scored:", "",
     "- **never crosses z=−4** → recorded (without-z-limit) outcome",
     "- **crosses with tilt < 60°** → band collision (upright at the floor = a genuine band failure)",
     "- **crosses with tilt ≥ 60°** → NOT a collision; keep the episode's eventual outcome (reach / collision / "
     "timeout / arena-floor oob)", "",
     "Rationale: 60° = arccos(1/TWR), the tilt past which full thrust cannot arrest a descent, so a tilted floor "
     "dip is a transient attitude excursion, not an upright band failure. No scoring-code edits. Supersedes M25.", "",
     f"**Categories:** never crosses **{int(never.sum())}**; crosses tilt<60° (→ band collision) "
     f"**{int(cross_lt60.sum())}**; crosses tilt≥60° (→ keep eventual) **{nc}**.", "",
     "| row | n | cps | reach | collision | stuck | oob | timeout | infeasibility |",
     "|---|---|---|---|---|---|---|---|---|",
     f"| with-z-limit (recorded, banded) | 2000 | 0.8051 | 0.9375 | 0.0425 | 0.0000 | 0.0000 | 0.0200 | 0.1247 |",
     f"| **tilt-at-crossing re-scoring** | {row['n']} | **{row['cps']:.4f}** | {row['reach']:.4f} | {row['collision']:.4f} | {row['stuck']:.4f} | {row['oob']:.4f} | {row['timeout']:.4f} | {row['infeasibility']:.4f} |",
     f"| without-z-limit (recorded, legacy) | 2000 | 0.9078 | 0.9690 | 0.0105 | 0.0005 | 0.0000 | 0.0200 | 0.0997 |",
     "",
     f"**Sanity:** re-scored cps {row['cps']:.4f} lies between with-z-limit 0.8051 and without-z-limit 0.9078 → "
     f"**{'PASS' if sanity_ok else 'FAIL — reported, stopping'}**.", "",
     "Tilt-at-crossing distribution (deg): " + ", ".join(f"{k} {v}" for k, v in tdist.items()) + ".", "",
     f"Eventual outcome of the {nc} tilt≥60° crossings (the ones re-scored to their eventual fate): "
     + ", ".join(f"{k} {v}" for k, v in ev.items()) + ".", "",
     f"**'Policy created the tilt' check:** of the {nc} tilt≥60° crossings, **{ge60_entire}** were tilted ≥60° for "
     f"the ENTIRE run up to the crossing (never inside the 60° cone) — i.e. the tilt was present from the initial "
     f"condition, not created by the policy; the remaining **{nc - ge60_entire}** were inside the cone at some "
     f"point before tilting past it (consistent with the policy/dynamics creating the tilt)."]
(OUT / "sota_tilt_at_crossing.md").write_text("\n".join(L) + "\n")
print(f"M27 -> sota_tilt_at_crossing.{{md,json}}")
print(f"  categories: never {int(never.sum())}, cross<60 {int(cross_lt60.sum())}, cross>=60 {nc}")
print(f"  rescored cps {row['cps']} reach {row['reach']} coll {row['collision']} (sanity in [0.8051,0.9078]: {sanity_ok})")
print(f"  tilt>=60 crossings eventual: {ev}; tilted>=60 entire-before: {ge60_entire}")
print(f"  tilt-at-crossing dist: {tdist}")
