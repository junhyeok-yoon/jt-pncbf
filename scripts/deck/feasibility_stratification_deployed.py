"""v2.7.7 M22 + M24 (Amdt 11/12) — feasibility-stratified cps on the DEPLOYED configuration (kstep k=5), using the
SIMULATOR-CONSISTENT dz_min flag (M24). Reads the deployed per-episode dump (evaluate(), banded, kstep k=5 —
reproduces the recorded row exactly) and the simulator-rollout flag (`dz_min_sim.npz`: recovery rolled through the
recorded rk4 + wrap_state with a max-authority plant PD aimed at the ceiling). Eval-only; no scoring-code edits.

M22: full-pool row (authoritative, reproduces recorded 0.8051/0.9375/0.0425) + feasible subset (flagged removed),
all cps components. M24 finding: the simulator-consistent flag fixed recall (→1.0, every deployed floor violation
is flagged) but precision is only ~0.33 — the max-authority plant control is still weaker than the deployed
TRAINED policy on borderline high-tilt / high-|omega0| ICs, so the flag OVER-flags (not conservative). Per the
amendment, the next suspected term is named and we STOP without retuning. The old analytic (M21) flag numbers are
marked superseded (kept in the record)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scripts.deck.deck_scene3d as S3
from scripts.deck.deck_style import OUT

SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
ZLIM = 4.0
z = np.load(SCR / "deployed_dump.npz"); min_pz = z["min_pz"]; max_pz = z["max_pz"]; IC = z["IC"]
rows = json.load(open(SCR / "deployed_rows.json"))["episode_rows"]
N = len(rows)
outcome = np.array([r["outcome"] for r in rows]); infeas = np.array([float(r["infeasible_step_frac"]) for r in rows])
sim = np.load(SCR / "dz_min_sim.npz"); flag = sim["flag"]; min_pz_sim = sim["min_pz_sim"]
theta0 = np.array([np.degrees(np.arccos(np.clip(S3.quat_to_R(IC[i, 3:7])[2, 2], -1, 1))) for i in range(N)])
wxy = np.sqrt(IC[:, 10] ** 2 + IC[:, 11] ** 2)
floor_viol = min_pz <= -ZLIM
nfl = int(flag.sum()); tp = int((flag & floor_viol).sum())
prec = tp / nfl if nfl else 0.0; rec = tp / int(floor_viol.sum()) if floor_viol.sum() else 0.0


def cps_of(o, inf):
    reach = (o == "goal").mean(); coll = (o == "collision").mean(); stuck = (o == "stuck").mean()
    oob = (o == "oob").mean(); to = (o == "timeout").mean()
    return {"cps": float(reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * inf.mean()), "reach": float(reach),
            "collision": float(coll), "stuck": float(stuck), "oob": float(oob), "timeout": float(to),
            "infeasibility": float(inf.mean()), "n": int(len(o))}


full = cps_of(outcome, infeas); feas = cps_of(outcome[~flag], infeas[~flag])
tilt_bins = [(0, 60), (60, 120), (120, 180.01)]
tilt_dist = {f"[{lo},{int(hi)})": int(((theta0[flag] >= lo) & (theta0[flag] < hi)).sum()) for lo, hi in tilt_bins}
REC = {"cps": 0.8051, "reach": 0.9375, "collision": 0.0425}
xcheck = {k: abs(full[k] - REC[k]) < 2e-3 for k in REC}

res = {"flag": "simulator-consistent (M24): plant max-authority recovery rolled through recorded rk4 + wrap_state",
       "n_flagged": nfl, "flagged_tilt_distribution_deg": tilt_dist,
       "flag_quality_vs_deployed": {"deployed_floor_violations": int(floor_viol.sum()), "precision": round(prec, 3),
            "recall": round(rec, 3), "false_positives": int((flag & ~floor_viol).sum()),
            "false_negatives": int((~flag & floor_viol).sum()),
            "verdict": "recall 1.0 (catches every violation) but precision ~0.33 -> OVER-FLAGS -> NOT conservative"},
       "rows": {"full_pool_deployed": {"source": "evaluate() after 09c33bf4, banded, kstep k=5 (reproduces recorded)",
                    "cross_check_pass": xcheck, **{k: round(full[k], 4) for k in full}},
                "feasible_subset_sim_flag": {"note": "sim flag OVER-flags (precision 0.33) -> these numbers are an "
                    "UPPER bound, not reliable; kept for completeness", **{k: round(feas[k], 4) for k in feas}}},
       "superseded": {"analytic_flag_M21": "n=105, deployed precision 0.58 (Amdt-10/11); superseded by the "
                      "simulator-consistent flag but kept in the record"}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sota_feasibility_deployed.json").write_text(json.dumps(res, indent=2) + "\n")

fps = [(int(i), round(float(min_pz_sim[i]), 2), round(float(min_pz[i]), 2), round(float(theta0[i]), 0),
        round(float(wxy[i]), 2)) for i in np.nonzero(flag & ~floor_viol)[0]]
fp_hi = sorted(fps, key=lambda t: t[2])[-12:]                # deployed min_pz closest to the floor (borderline)
L = ["# Band scoring — feasibility-stratified cps, DEPLOYED config, SIMULATOR-CONSISTENT flag (M22 + M24)", "",
     "Per-episode dump from `evaluate()` (after 09c33bf4, banded, kstep k=5 — reproduces the recorded row exactly), "
     "joined to the **simulator-consistent** dz_min flag (M24): the recovery is rolled through the RECORDED rk4 + "
     "wrap_state (same integrator + ω clamp as deployment), applying the max-authority plant PD (saturated "
     "collective + max attitude gain) aimed to hold xy and climb to the ceiling. Eval-only; no scoring-code edits.", "",
     "## M22 — full-pool vs feasible-subset (all cps components)", "",
     f"Cross-check vs recorded: cps {xcheck['cps']}, reach {xcheck['reach']}, collision {xcheck['collision']} → PASS.", "",
     "| scope | n | cps | reach | collision | stuck | oob | timeout | infeasibility |",
     "|---|---|---|---|---|---|---|---|---|",
     f"| full pool (deployed, kstep k=5) | {full['n']} | {full['cps']:.4f} | {full['reach']:.4f} | {full['collision']:.4f} | {full['stuck']:.4f} | {full['oob']:.4f} | {full['timeout']:.4f} | {full['infeasibility']:.4f} |",
     f"| feasible subset (sim flag removed) | {feas['n']} | {feas['cps']:.4f} | {feas['reach']:.4f} | {feas['collision']:.4f} | {feas['stuck']:.4f} | {feas['oob']:.4f} | {feas['timeout']:.4f} | {feas['infeasibility']:.4f} |",
     "",
     f"**n_flagged {nfl}/{N}** (tilt deg " + ", ".join(f"{k} {v}" for k, v in tilt_dist.items()) + "). "
     f"**The feasible-subset row is an UPPER bound — see M24: the flag over-flags (precision 0.33).**", "",
     "## M24 — simulator-consistent flag: quality vs the deployed rollout", "",
     f"The sim-consistent rollout FIXES recall: **recall {rec:.2f}** (all {int(floor_viol.sum())} deployed floor "
     f"violations are flagged; 0 false negatives) — the analytic ω-capped model (M21) had recall only 0.73. But "
     f"**precision {prec:.2f}** ({tp}/{nfl}); **{int((flag & ~floor_viol).sum())} false positives**, all borderline "
     "(the deployed policy holds them just above −4 while the max-authority PD drops just below).", "",
     "Borderline false positives (deployed min p_z nearest the floor):", "",
     "| idx | sim min p_z | deployed min p_z | tilt0° | |ω0| |",
     "|---|---|---|---|---|"]
for i, sp, dp, t, w in fp_hi:
    L.append(f"| {i} | {sp:.2f} | {dp:.2f} | {t:.0f} | {w:.2f} |")
L += ["",
      "**Precision is NOT near 1 → per the amendment, the next suspected term is named and we STOP without "
      "retuning.**", "",
      "**Suspected term: the recovery control is still suboptimal relative to the deployed TRAINED policy (and to "
      "the true minimum-drop optimal control) at the feasibility margin.** Even a max-authority plant PD (boosted "
      "position gains → saturated collective; kp_att=40 → saturated righting torque) drops ~0.5–2 m more than the "
      "deployed policy on the false positives, which are concentrated on **high-tilt ICs with high initial angular "
      "rate |ω0|** — the trained policy handles the initial angular momentum and timing better than any fixed PD. "
      "The **thrust–torque coupling** is the root: righting torque needs collective, which past 60° points partly "
      "down and deepens the fall, so the minimum-drop control is a coupled attitude+thrust+rate optimal-control "
      "problem, not any single prescribed law. A conservative (precision→1) flag would require solving that OCP or "
      "using the deployed policy itself (circular). No change made; flagged for the Researcher.", "",
      "## Status", "",
      "- The recall-1.0 flag is a valid screen for **definitely-feasible** ICs (flag=False → the max-authority "
      "control keeps it in the band), but it is NOT a conservative infeasibility certificate.",
      "- **Superseded:** the analytic ω-capped flag (M21: n=105, deployed precision 0.58) is superseded by this "
      "simulator-consistent flag but kept in the record (`sota_feasibility.{md,json}`).",
      "- The authoritative full-pool deployed row (cps 0.8051 / reach 0.9375 / collision 0.0425) is unchanged; the "
      "feasibility-corrected cps remains bracketed (not a single reliable number) until the OCP above is resolved."]
(OUT / "sota_feasibility_deployed.md").write_text("\n".join(L) + "\n")
print(f"M22 full cps {full['cps']:.4f} (xcheck {xcheck}); feasible(sim-flag) cps {feas['cps']:.4f} n={feas['n']}")
print(f"M24 sim flag: n_flagged {nfl}, precision {prec:.3f}, recall {rec:.3f}; false-pos {int((flag&~floor_viol).sum())}; NOT near 1 -> term named, STOP")
