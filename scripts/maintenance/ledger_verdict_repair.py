"""One-shot repair: relocate every ledger verdict verbatim into docs/ledger_verdicts.md, then
compress the over-length verdict cells to at most five sentences.

Order of operations, so the companion is lossless by construction:
  1. read docs/ledger.md AS IT IS and write docs/ledger_verdicts.md from that text
  2. apply the compressed verdicts below, each guarded by a prefix assertion on the text it replaces
  3. verify -- companion round-trip, non-verdict cells byte-identical, every cell <= 5 sentences

Anchors are derived from the row's version, system and label (the alias cell), never from a line
number: slug(version)--slug(system)--slug(alias or 'unlabeled'), plus a 1-based occurrence index
when several rows share all three. Row order is fixed by the dispatch, so the index is stable.

Reads and writes docs/ only. Nothing under data/ is touched.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/maintenance"))
from ledger_verdict_lint import parse, count_sentences, MAX_SENTENCES  # noqa: E402

LEDGER = REPO / "docs" / "ledger.md"
COMPANION = REPO / "docs" / "ledger_verdicts.md"


def slug(s: str) -> str:
    s = s.replace("**", "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unlabeled"


def anchors(rows, cols):
    """-> {line_no: anchor}, stable under the dispatch's no-reorder constraint."""
    vi, ai = cols.index("verdict"), cols.index("alias")
    seen, out = {}, {}
    for ln, c in rows:
        base = f"{slug(c[0])}--{slug(c[1])}--{slug(c[ai])}"
        seen[base] = seen.get(base, 0) + 1
        out[ln] = base if seen[base] == 1 else f"{base}--{seen[base]}"
    # a base that occurs more than once must have EVERY occurrence indexed, including the first
    counts = {}
    for ln, c in rows:
        base = f"{slug(c[0])}--{slug(c[1])}--{slug(c[ai])}"
        counts[base] = counts.get(base, 0) + 1
    seen2 = {}
    for ln, c in rows:
        base = f"{slug(c[0])}--{slug(c[1])}--{slug(c[ai])}"
        seen2[base] = seen2.get(base, 0) + 1
        out[ln] = base if counts[base] == 1 else f"{base}--{seen2[base]}"
    assert len(set(out.values())) == len(out), "anchor collision"
    return out


# ---------------------------------------------------------------------------------------------
# The compressed verdicts. Key = ledger line number; `head` = a prefix of the text being replaced,
# asserted before the swap so a mis-keyed entry fails loudly instead of overwriting the wrong row.
# `{A}` is substituted with that row's companion anchor. Slot order per the dispatch:
#   1 what the row is | 2 headline result | 3 comparability | 4 bold/promotion/supersession | 5 pointer
# ---------------------------------------------------------------------------------------------
NEW: dict[int, tuple[str, str]] = {
 64: ("EXPERIMENT (user-directed u_max 2->20)",
   "EXPERIMENT (user-directed): one global actuator change, u_max 2 -> 20, against the v2.3.0 "
   "double_integrator baseline, with v_max held at 2.5. cps 0.9419 [0.9290, 0.9551] against "
   "v2.3.0's 0.8700. NOT comparable to any +/-2 row, since the actuator spec differs and the result "
   "is not +/-2-deployable; single seed, best.pt with no n2000 re-selection. SOTA UNCHANGED, no bold "
   "change, no promotion. Detail docs/versions/v2.3.0_results.md (Experiment: u_max widening) and "
   "docs/ledger_verdicts.md#{A}."),
 81: ("Training-free, NOT SOTA-comparable",
   "Training-free reach-witness follower, carrying no learned object. GATE PASSED at exhaustive "
   "coverage 0.98 (P-R4), the follower safe (coll 0.000) and reaching 0.9615 near-optimally, but "
   "the rescue hypothesis is REFUTED (P-R6) and the cost bar REFUTED (P-R7). NOT SOTA-comparable "
   "and not comparable to any filtered row: cps == cps_v2 only because there are no filter steps, "
   "so infeasibility is 0 by convention. No bold change, no promotion. Detail reach_witness.md "
   "Stage R-2 and docs/ledger_verdicts.md#{A}."),
 82: ("ARM-C (close-prep)",
   "ARM-C close-prep: the v2.3.0 learned-V HardNet filter re-evaluated at the D-fast deploy axis "
   "(dt 0.01) against its own D-slow score. GATE PASS on reproduction, and the move D-slow -> "
   "D-fast REGRESSES the learned filter by -0.0072 cps_v2 (stuck-driven) while V_M GAINS +0.0031 "
   "on the same move, so the D-fast gain is V_M-SPECIFIC and not shared. Eval-only on a changed "
   "deploy axis, so not comparable to D-slow rows; NOT a SOTA claim. No bold change, no promotion. "
   "Detail close_prep_batch.md T-A and docs/ledger_verdicts.md#{A}."),
 87: ("CPI iteration-3 (pathfinder, seed 42)",
   "CPI iteration-3 pathfinder: policy pi_3 under the frozen certificate V_hat_2. Arm-A cps_v2 "
   "0.660 REGRESSED against iteration-2's 0.721, failing the P-PI2 gate and halting P-I1-3 "
   "monotonicity at k=3. The line is REJECTED, so this row is a terminal record rather than a "
   "comparator; single seed. No bold change, no promotion. Detail phase_i1_loop_report.md I3 and "
   "docs/ledger_verdicts.md#{A}."),
 108: ("J-arc arm B POOLED",
   "J-arc arm B, pooled: the S' shield deployed on top of the learned filter. **P-J1 HALT-gate "
   "PASS** and **P-J2 PASS** (0.9171 > 0.8422), the shield transferring at NO cps cost and in fact "
   "a gain over arm A (0.8992 -> 0.9171, coll 0.0060 -> 0.0000), while **P-J4 is FALSIFIED** at "
   "ovr/ep 0.489 against the >1.0 predicted. A different deployment class (learned filter plus S' "
   "shield), so not comparable to unshielded rows. NOT SOTA-bold; flag Researcher. Detail "
   "phase_j_report.md and docs/ledger_verdicts.md#{A}."),
 112: ("J-arc arm B' POOLED",
   "J-arc arm B', pooled: the same shield as arm B at roughly twice the checks per episode (417 "
   "against 200). **P-J3 PASS** (reach 0.9548 >= arm A's 0.9533 - 0.05), with cps_v2 0.9174 "
   "essentially equal to arm B's 0.9171, so the ladder buys no cps here. A different deployment "
   "class, so not comparable to unshielded rows. NOT SOTA-bold; flag Researcher. Detail "
   "phase_j_report.md and docs/ledger_verdicts.md#{A}."),
 137: ("[M6] JT co-trained policy",
   "[M6] JT co-trained policy on the pre-per-rotor wrench action box -- the lineage bring-up "
   "headline. H1 CONFIRMED: cps 0.9329 [0.918, 0.949], CI-separated above the pre-JT 0.8481, with "
   "collision halved 0.020 -> 0.0095. NOT comparable to any current-plant result, since the v2.7.3 "
   "per-rotor plant replaced that action box and the checkpoint does not load on it, so these "
   "numbers describe a vehicle that cannot be built; no cross-dimension claim was made. SUPERSEDED "
   "by v2.8.0 (2.4 / 6.3) and RELEASED FROM BOLD, the row retained for provenance. Detail "
   "docs/ledger_verdicts.md#{A}."),
 232: ("CONTROL (v2.8.2), UNBOLDED",
   "CONTROL for v2.8.2: the v2.8.0 JT recipe with hard_topk at terminal omega_G 0.30, shared "
   "v2.7.6 value-init and a fresh policy, best.pt at step 24000, secured as a control and "
   "attribution snapshot. **T2 PASSES at gate reach 0.9205 >= 0.8875**, which attributes v2.8.1's "
   "T2 failure to the ENCODER and not the terminal, closing open discrepancy (2) of "
   "v2.8.1_results.md. NOT comparable to the standing 3-D bold 0.7919: neither step-matched (30000 "
   "against 50000) nor terminal-matched (0.30 against 0.48), and the run's own auto-final eval used "
   "the training-time fallback rather than the shipped one, so the dual three-cell is authoritative; "
   "single seed. UNBOLDED, no promotion, NOT a SOTA claim. Detail "
   "docs/versions/v2.8.2/s2_conditions.md and docs/ledger_verdicts.md#{A}."),
 236: ("STEP-MISMATCHED, NOT converged",
   "FLOOR axis (FB deployed-fallback co-adaptation), stopped on Researcher decision at best.pt step "
   "9000 of 10499. CONFOUND BROKEN: at matched fallback the floor gain vanishes (FB 0.0145 against "
   "CTRL 0.0150), so FB's E0 pass was the EVAL FALLBACK and not co-adaptation, leaving a clean "
   "training effect of +0.018 cps at step 9000 that narrows from +0.097 at 4500 -- training speed, "
   "not quality. STEP-MISMATCHED and NOT converged, so cps 0.7289 [0.696, 0.763] is an "
   "EARLY-checkpoint score NOT comparable to CTRL's 0.7785; FB trained under {kstep, phases 2, k 5} "
   "and is scored here at the shipped {kstep, phases 1, k 3}; single seed. UNBOLDED, no promotion. "
   "Detail docs/versions/v2.8.2/fb_floor.md and docs/ledger_verdicts.md#{A}."),
 249: ("PPO BASELINE (no certificate, no filter)",
   "PPO BASELINE on quadrotor_3d with no certificate and no filter, under a Researcher-registered "
   "shaped reward. It learns near-complete collision avoidance (obstacle 0.0455 / floor 0.0275 / "
   "ceiling 0.0035) and beats the LQR nominal at cps -0.6148 against -0.845, but reach is ~0 and "
   "the run is timeout-dominated at 0.9225, so it cannot achieve the tight settling terminal. "
   "Conditional on that shaping and NOT comparable to the certificate rows on equal terms -- "
   "smoothness is penalized here while JT's w_du is off, and the reach terminal is heavily "
   "gamma-discounted; single seed. UNBOLDED, no promotion. Detail "
   "docs/versions/v2.8.2/ppo_baseline.md and docs/ledger_verdicts.md#{A}."),
 309: ("the v2.9.1 JT-PNCBF arm on double_integrator",
   "the v2.9.1 JT-PNCBF arm on double_integrator, warm-started value-only from this system's OWN OC "
   "best.pt so the policy trains from scratch, run to its own terminal path. Against this system's "
   "OC row L307 on the SAME pool and cell the difference is +0.024298, and against the unfiltered "
   "LQR nominal (cps -0.009500) the certificate-plus-policy adds +0.914257. Single seed 42 on both "
   "sides with NO CI separation claimed in either direction; trained with the empty-branch fallback "
   "OFF, and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this system. EMPTY 0.2570 and "
   "SINGULAR 0.2305 are EPISODE-axis fractions and are never summed. No bold change, no promotion, "
   "no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md and docs/ledger_verdicts.md#{A}."),
 310: ("the v2.9.1 JT-PNCBF arm on unicycle",
   "the v2.9.1 JT-PNCBF arm on unicycle, warm-started value-only from this system's OWN OC best.pt "
   "so the policy trains from scratch, run to its own terminal path. Against this system's OC row "
   "L308 on the SAME pool and cell the difference is +0.202640, and against the unfiltered LQR "
   "nominal (cps 0.002000) the certificate-plus-policy adds +0.871126. Single seed 42 on both sides "
   "with NO CI separation claimed in either direction; trained with the empty-branch fallback OFF, "
   "and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this system. EMPTY 0.1830 and SINGULAR "
   "0.1150 are EPISODE-axis fractions and are never summed. No bold change, no promotion, no SOTA "
   "claim; detail docs/versions/v2.9.1/jt_planar.md and docs/ledger_verdicts.md#{A}."),
 311: ("the v2.9.1 JT-PNCBF arm on quadrotor_planar",
   "the v2.9.1 JT-PNCBF arm on quadrotor_planar, warm-started value-only from this system's OWN OC "
   "best.pt, and a RELAUNCH of the byte-identical JTPLANAR_V291_R3 that died of CUDA OOM while "
   "sharing the card. Against this system's OC row L306 on the SAME pool and cell the difference is "
   "+0.116590, and against the unfiltered LQR nominal (cps -0.206000) the certificate-plus-policy "
   "adds +0.981740. Single seed 42 with NO CI separation claimed in either direction; trained with "
   "the empty-branch fallback OFF, and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this "
   "system. EMPTY 0.3780 and SINGULAR 0.1085 are EPISODE-axis fractions and are never summed. No "
   "bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md and "
   "docs/ledger_verdicts.md#{A}."),
 312: ("the v2.9.1 JT-PNCBF arm on quadrotor_3d",
   "the v2.9.1 JT-PNCBF arm on quadrotor_3d, warm-started value-only from this system's OWN OC "
   "best.pt (L305) so the quadrotor_3d column is STRUCTURALLY IDENTICAL to the other three, its "
   "config being L304's with EXACTLY ONE key changed. Against L305 on the SAME pool and cell the "
   "difference is +0.186152; against L304 it is -0.004236, i.e. 0.51x the 0.0083 admissibility "
   "floor and inside L304's own 95% CI [0.8471, 0.8962], so NO separation is claimed in either "
   "direction. Single seed 42 and no CI computed by this row's producer; EMPTY 0.5830 and SINGULAR "
   "0.0955 are EPISODE-axis fractions and are never summed. Per the Researcher's disposition fixed "
   "in advance of this number, Table I takes THIS row for the quadrotor_3d JT cell and the BOLD "
   "STAYS AT L304; no promotion in the registering item. Detail "
   "docs/versions/v2.9.1/final_scoring.md 15 and docs/ledger_verdicts.md#{A}."),
 313: ("**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L132",
   "**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L132: the same checkpoint L132 names by digest "
   "(3b27d691, located at data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt and "
   "strict-loading under current code at obs_dim 22), scored on the current cell and pool. RESULT "
   "0.851927 against L132's 0.9036, i.e. -0.051673, BELOW the bold by 6.2x the 0.0083 admissibility "
   "floor. One documented addition beyond gate_overrides, env.goal_angrate_radius 0.3, makes the "
   "reach predicate strictly TIGHTER than the one that produced 0.9036, so this row is comparable to "
   "L306 and L311 on the same pool and cell but NOT to L132's own figure; single seed 42, no CI. "
   "BOLD MOVED HERE at the v2.9.1 close (Researcher decision, 2026-08-14), superseding L132, which "
   "keeps its content and loses bold -- a BASIS re-score of the same checkpoint, NOT a 04_eval 5 "
   "CI-separated beat and not an improvement claim, the natural successors L311 and L306 being "
   "bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md and "
   "docs/ledger_verdicts.md#{A}**"),
 314: ("**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62",
   "**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62, which names no checkpoint but a 3-SEED "
   "AGGREGATE; all three secured snapshots (seeds 42, 12345, 99) were located and strict-load under "
   "current code at obs_dim 19. RESULT 3-seed mean 0.868247 (sd 0.008178) against L62's 0.8698, i.e. "
   "-0.001553, BELOW the bold by 0.19x the 0.0083 admissibility floor, well inside the seed spread "
   "and NOT a separation. Scored on the v2.9.1 REGENERATED DI pool, which makes this row comparable "
   "to L307 and L309 and NOT to L62 itself or to any historical DI row, since that pool is not yet "
   "the pool of record. BOLD MOVED HERE at the v2.9.1 close (Researcher decision, 2026-08-14), "
   "superseding L62, which keeps its content and loses bold -- a BASIS re-score, not an improvement "
   "claim, the natural successor L309 being bold-INELIGIBLE per 06_workflow 6.3. Detail "
   "docs/versions/v2.9.1/bold_rescore.md and docs/ledger_verdicts.md#{A}**"),
 315: ("VALUE-ONLY CONTINUATION of L312",
   "VALUE-ONLY CONTINUATION of L312: 3000 further macro steps (9450 -> 12450) with the POLICY FROZEN "
   "via training.jt.K_pi 1 -> 0 and the value still learning, asking whether the certificate at step "
   "9450 had converged to that policy or was still lagging it. REGISTERED HYPOTHESIS FALSIFIED on "
   "both named columns and in the wrong direction -- infeasibility ROSE 0.079209 -> 0.081508 and the "
   "EMPTY episode share ROSE 0.5830 -> 0.5995 -- so freezing the policy did NOT make the filter's "
   "infeasible states rarer. cps 0.849048 is reported and is NOT the falsifier, since it mixes "
   "channels this axis does not target; against L312 it is -0.018939 and against the bold L304 "
   "-0.023175, exceeding neither. Single seed 42, no CI, and best_step is the FINAL step, the "
   "maximum of a 0.028-wide plateau rather than of a still-rising curve. No bold change, no "
   "promotion; detail docs/versions/v2.9.1/value_only_continuation.md and "
   "docs/ledger_verdicts.md#{A}."),
 316: ("v2.9.2 REGISTERED AXIS, OC arm",
   "v2.9.2 REGISTERED AXIS, OC arm: env.quadrotor_3d.c_gain 0.3 -> 0.0, the horizontal approach-term "
   "ablation, derived from L305's OWN persisted config with EXACTLY ONE flattened key changed of 329 "
   "and the term's absence proven functionally on 12 probe states. A1 HOLDS on this arm -- "
   "coll_obstacle 0.0310 -> 0.0500, 62 -> 100 episodes of 2000 -- while the vertical band channels "
   "are UNCHANGED to the episode, which is what the registered hypothesis predicted. cps 0.6194 "
   "against the parent L305's 0.6818 is reported and NOT claimed, since cps is not the registered "
   "observable and no CI exists on either row; single seed 42, and best_step is the final eval, so "
   "this checkpoint was still improving when the budget ended. NOT bold, NOT promoted as SOTA -- a "
   "06_workflow 6.3 experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md and "
   "docs/ledger_verdicts.md#{A}."),
 317: ("PRIVILEGED MODEL-BASED REFERENCE POINT",
   "PRIVILEGED MODEL-BASED REFERENCE POINT, NOT A PEER of a learned-certificate row: MPPI reads the "
   "full 13-D state and the exact obstacle field and rolls the true plant inside its own planner, "
   "where the learned arms see a 34-D observation and carry no plant model. Reach 0.2980 and "
   "collision 0.6525, of which obstacle 0.3280 and band_lower 0.3235, so the arena floor accounts "
   "for essentially half the collisions. NOT comparable on infeasibility or mean_proj_mag at all, "
   "which are STRUCTURALLY INAPPLICABLE rather than zero because it runs an identity filter; this "
   "is the FIRST MPPI row in the ledger and the first at n 2000, every prior figure on record being "
   "400 scenes; one run, single seed 42, no CI, no selection. NOT bold, NOT promoted; it bears on "
   "docs/versions/v2.9.2/mppi_root_cause.md 4-iv and on nothing else. Detail "
   "docs/versions/v2.9.2/mppi_registered.md and docs/ledger_verdicts.md#{A}."),
 318: ("v2.9.2 REGISTERED AXIS, JT arm",
   "v2.9.2 REGISTERED AXIS, JT arm: env.quadrotor_3d.c_gain 0.3 -> 0.0, derived from L312's OWN "
   "persisted config with EXACTLY TWO flattened keys changed of 328, the second repointing "
   "value_init_ckpt to this version's OC c_gain-0 cell so the arm is not warm-started from a "
   "c_gain-0.3 certificate. A1 HOLDS on this arm -- coll_obstacle 0.0075 -> 0.0090, 15 -> 18 "
   "episodes of 2000 -- and WITH THE OC ARM L316 IT HOLDS ON BOTH ARMS. cps 0.8495 against L312's "
   "0.8680 is reported and NOT claimed, since cps is not the registered observable and no CI exists "
   "on either row; INITIALIZER ASYMMETRY, a fact and not a defect, is that this arm warm-starts from "
   "a BUDGET-TRUNCATED checkpoint where L312 warm-started from a converged peak, the budget having "
   "been held equal; single seed 42. NOT bold, NOT promoted as SOTA -- a 06_workflow 6.3 "
   "experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md and "
   "docs/ledger_verdicts.md#{A}."),
 319: ("v2.9.2 MISMATCH PREDICTION TEST. V is V^pi",
   "v2.9.2 MISMATCH PREDICTION TEST, pairing L312's policy with L305's certificate. PREDICTION HOLDS "
   "on this cell: cps 0.6925 against its own policy's matched cell MATCHED_JT 0.8680, i.e. -0.1755, "
   "a degradation far beyond the 0.0083 floor, and coll_obstacle 0.0075 -> 0.0555 (15 -> 111 "
   "episodes of 2000) is where it goes while coll_band_lower is UNCHANGED. V is V^pi, so this is a "
   "WRONG CERTIFICATE rather than a controlled variation and NOT A VALID DEPLOYMENT, comparable only "
   "to the other three mismatch cells; single seed 42, no CI on any of the four. NOT bold, NOT "
   "promoted, and no configuration here is proposed, adopted or recommended. Detail "
   "docs/versions/v2.9.2/decomp.md and docs/ledger_verdicts.md#{A}."),
 320: ("v2.9.2 MISMATCH PREDICTION TEST. V is V^pi",
   "v2.9.2 MISMATCH PREDICTION TEST, pairing the shipped LQR nominal with L312's certificate. "
   "PREDICTION FALSIFIED on this cell, and this is the finding: cps 0.8136 against its own policy's "
   "matched cell MATCHED_OC 0.6818, i.e. +0.1317, so the mismatched certificate EXCEEDS the matched "
   "one by 15.9x the 0.0083 floor and on this system and pool the policy-dependence premise does NOT "
   "hold in this direction. V is V^pi, so this is a WRONG CERTIFICATE rather than a controlled "
   "variation and NOT A VALID DEPLOYMENT, comparable only to the other three mismatch cells; single "
   "seed 42, no CI on any of the four. NOT bold, NOT promoted, and no configuration here is "
   "proposed, adopted or recommended. Detail docs/versions/v2.9.2/decomp.md and "
   "docs/ledger_verdicts.md#{A}."),
}

# the eight horizon-400 rows share one template
H400 = {
 321: ("L305", "quadrotor_3d OC", "+0.051139", "[+0.038552, +0.063224]", 20, "140"),
 322: ("L306", "quadrotor_planar OC", "+0.037161", "[+0.025654, +0.049372]", 14, "145"),
 323: ("L307", "double_integrator OC", "+0.001764", "[+0.000214, +0.004028]", 14, "81"),
 324: ("L308", "unicycle OC", "+0.004828", "[+0.000181, +0.009899]", 14, "189"),
 325: ("L309", "double_integrator JT", "+0.000745", "[-0.000053, +0.002285]", 14, "47"),
 326: ("L310", "unicycle JT", "+0.000085", "[+0.000024, +0.000156]", 14, "41"),
 327: ("L311", "quadrotor_planar JT", "+0.013638", "[+0.006640, +0.021657]", 14, "100"),
 328: ("L312", "quadrotor_3d JT", "+0.017325", "[+0.010542, +0.024130]", 14, "41"),
}
for _ln, (_par, _what, _d, _ci, _nf, _ch) in H400.items():
    NEW[_ln] = ("NEW CELL at eval.max_steps 400",
        f"NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with "
        f"every other key identical to {_par}'s, on {_par}'s own pool and checkpoint ({_what}). "
        f"Paired against a same-instrument 200-step control on identical scenes, the per-episode cps "
        f"difference is {_d} with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE "
        f"ITSELF of {_ci} at the shipped defaults, {_ch} of 2000 episodes changed. NOT COMPARABLE TO "
        f"ANY 200-STEP ROW including its own counterpart {_par}, because the horizon moves both the "
        f"outcome resolution and the infeasibility denominator; that 200 control reproduced {_par} in "
        f"all {_nf} compared fields within 1e-12; single seed 42, the interval within-seed and "
        f"scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows "
        f"off-basis, which is the Researcher's decision and not this dispatch's. Detail "
        f"docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in "
        f"docs/ledger_verdicts.md#{{A}}.")


def build_companion(lines, cols, rows, anch) -> str:
    vi, ai = cols.index("verdict"), cols.index("alias")
    out = [
        "# Ledger verdicts — full text",
        "",
        "Companion to `docs/ledger.md`. The ledger's `verdict` column is capped at five sentences",
        "(`scripts/check_ledger.py` rule 11); this file carries the **full verdict text of every row**",
        "as it stood when the cap was installed, verbatim and unmodified, including bold markup.",
        "",
        "Nothing here is rewritten, summarized or reordered. Sections are in ledger table order.",
        "Anchors are derived from the row's version, system and label, never from a line number:",
        "`slug(version)--slug(system)--slug(alias)`, with a 1-based occurrence index appended when",
        "several rows share all three.",
        "",
        f"Rows: {len(rows)}.",
        "",
        "---",
        "",
    ]
    for ln, c in rows:
        a = anch[ln]
        ver = c[0].replace("**", "")
        sysm = c[1].replace("**", "")
        alias = c[ai].replace("**", "") or "-"
        out += [
            f'<a id="{a}"></a>',
            f"## {ver} · {sysm} · {alias}",
            "",
            f"`anchor: {a}` · ledger line {ln} at the time of writing · "
            f"cps `{c[cols.index('cps')]}` · date `{c[cols.index('date')]}`",
            "",
            c[vi],
            "",
        ]
    return "\n".join(out) + "\n"


def main() -> int:
    lines, hdr, cols, rows = parse()
    vi = cols.index("verdict")
    anch = anchors(rows, cols)
    original = {ln: c[vi] for ln, c in rows}

    COMPANION.write_text(build_companion(lines, cols, rows, anch))
    print(f"wrote {COMPANION.relative_to(REPO)} with {len(rows)} sections")

    changed = 0
    for ln, (head, new) in sorted(NEW.items()):
        cells = [c.strip() for c in lines[ln - 1].strip().strip("|").split("|")]
        assert len(cells) == len(cols), f"L{ln} column count"
        assert cells[vi].startswith(head), f"L{ln} head mismatch: {cells[vi][:80]!r}"
        text = new.replace("{A}", anch[ln])
        assert "|" not in text and "\n" not in text, f"L{ln} illegal char"
        k = count_sentences(text)
        assert k <= MAX_SENTENCES, f"L{ln} rewrite is {k} sentences"
        cells[vi] = text
        lines[ln - 1] = "| " + " | ".join(cells) + " |"
        changed += 1
    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"rewrote {changed} verdict cells")

    # ---- verification ------------------------------------------------------------------------
    _, _, cols2, rows2 = parse()
    assert cols2 == cols and len(rows2) == len(rows), "shape changed"
    comp = COMPANION.read_text()
    lossless = [ln for ln in original if original[ln] not in comp]
    print(f"losslessness: {len(original) - len(lossless)}/{len(original)} verdicts recoverable "
          f"verbatim from the companion; failures {lossless if lossless else 'NONE'}")
    diffs = []
    for (ln, a), (ln2, b) in zip(rows, rows2):
        assert ln == ln2
        for j, name in enumerate(cols):
            if name != "verdict" and a[j] != b[j]:
                diffs.append((ln, name))
    print(f"non-verdict cells differing: {diffs if diffs else 'NONE'}")
    over = [(ln, count_sentences(c[vi])) for ln, c in rows2 if count_sentences(c[vi]) > MAX_SENTENCES]
    print(f"verdicts still over {MAX_SENTENCES} sentences: {over if over else 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
