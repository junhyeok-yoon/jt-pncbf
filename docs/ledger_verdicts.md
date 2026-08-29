# Ledger verdicts — full text

Companion to `docs/ledger.md`. The ledger's `verdict` and `eval_source` columns are capped at
400 characters each (`scripts/check_ledger.py` rule 11, `MAX_CELL_CHARS`); this file carries the
**full text of every row**, including the pre-cap text of every cell the cap forced to be
compressed, verbatim except for the one substitution recorded below, and including bold markup.

**One global substitution, 2026-08-18 — `00_constitution` §3 Prohibition 5.** The banned term for
an experimental condition was replaced throughout this file and throughout `docs/ledger.md`:
`arm`/`arms` → `condition`/`conditions` (one ledger cell took `cell` instead, to stay inside the
400-character cap), and the uppercase label `ARM` → **`REF`**, which also renames the alias of
ledger rows L271 and L278 from `ARM (C=1.3)` to `REF (C=1.3)` and their anchors from
`…--arm-c-1-3--{1,2}` to `…--ref-c-1-3--{1,2}`. **No number, no date, no path and no bold marker
was altered**, and the substitution was verified to preserve the digit set of every cell.
Build-logs closed before that date still cite the old spellings; `ARM` there is this file's `REF`.
Wherever a subheading below says "pre-cap text, verbatim", read it as verbatim **under this one
substitution**.

Nothing else here is rewritten, summarized or reordered. Sections are in ledger table order.
Anchors are derived from the row's version, system and label, never from a line number:
`slug(version)--slug(system)--slug(alias)`, with a 1-based occurrence index appended when
several rows share all three.

Rows: 321.

---

<a id="v2-0-0--double-integrator--unlabeled--1"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--1` · ledger line 20 at the time of writing · cps `0.6498` · date `2026-05-28 13:21:10`

superseded (first full run)

<a id="v2-0-0--double-integrator--unlabeled--2"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--2` · ledger line 21 at the time of writing · cps `0.1714` · date `2026-05-28 14:38:09`

reverted (truncation regressed)

<a id="v2-0-0--double-integrator--unlabeled--3"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--3` · ledger line 22 at the time of writing · cps `0.6540` · date `2026-05-28 14:57:54`

superseded (pre-alignment baseline)

<a id="v2-0-0--double-integrator--unlabeled--4"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--4` · ledger line 23 at the time of writing · cps `0.9828` · date `2026-05-28 19:21:06`

diagnostic (fixed obstacle)

<a id="v2-0-0--double-integrator--unlabeled--5"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--5` · ledger line 24 at the time of writing · cps `0.4564` · date `2026-05-28 20:36:10`

reverted (obs saturation regressed)

<a id="v2-0-0--double-integrator--unlabeled--6"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--6` · ledger line 25 at the time of writing · cps `0.5420` · date `2026-05-28 22:45:06`

reverted (gamma regressed)

<a id="v2-0-0--double-integrator--unlabeled--7"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--7` · ledger line 26 at the time of writing · cps `0.7969` · date `2026-05-29 00:14:41`

adopted; below §2.4 eligibility pool size (full_n500 < n2000) — released from bold

<a id="v2-0-0--double-integrator--unlabeled--8"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--8` · ledger line 27 at the time of writing · cps `0.7706` · date `2026-05-29 01:02:01`

ablation (R=0.1)

<a id="v2-0-0--double-integrator--unlabeled--9"></a>
## v2.0.0 · double_integrator · -

`anchor: v2-0-0--double-integrator--unlabeled--9` · ledger line 28 at the time of writing · cps `0.7372` · date `2026-05-29 01:15:30`

ablation (R=5.0)

<a id="v2-0-1--double-integrator--unlabeled--1"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--1` · ledger line 29 at the time of writing · cps `0.7418` · date `2026-05-29 08:42:10`

projection filter viable (collision 0.002, but saturation/stuck up vs CBF-QP)

<a id="v2-0-1--double-integrator--unlabeled--2"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--2` · ledger line 30 at the time of writing · cps `0.6578` · date `2026-05-29 09:32:20`

JT first run - stable, policy learned; below OC (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--3"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--3` · ledger line 31 at the time of writing · cps `0.6608` · date `2026-05-29 11:49:34`

value-only refinement 4k - budget-limited; extended in follow-up (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--4"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--4` · ledger line 32 at the time of writing · cps `0.7633` · date `2026-05-29 12:01:27`

value-only refinement 42k V-updates - collision improved; roughness above OC (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--5"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--5` · ledger line 33 at the time of writing · cps `0.7520` · date `2026-05-29 12:14:39`

CBF-QP collection refinement - no roughness gain vs HardNet (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--6"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--6` · ledger line 34 at the time of writing · cps `0.6488` · date `2026-05-29 12:43:38`

JT OC-scale value batch - stable, roughness worsened, full-pool below first JT (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--7"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--7` · ledger line 35 at the time of writing · cps `0.7287` · date `2026-05-29 16:13:00`

JT slow schedule - stable, smoother V_S, collision still above OC (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--8"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--8` · ledger line 36 at the time of writing · cps `0.7573` · date `2026-05-29 16:49:24`

value-refine sigma=0 - unsafe held, collision improved, roughness unchanged (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--9"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--9` · ledger line 37 at the time of writing · cps `0.6636` · date `2026-05-29 16:55:56`

value-refine sigma=0.1 - unsafe held, eval regressed (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--10"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--10` · ledger line 38 at the time of writing · cps `0.8852` · date `2026-05-29 17:10:57`

JT slow schedule extended - stable, new v2.0.1 SOTA (cps_v2 @n2000 re-eval; legacy cell @n500); below §2.4 eligibility pool size (full_n500 < n2000) — released from bold

<a id="v2-0-1--double-integrator--unlabeled--11"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--11` · ledger line 39 at the time of writing · cps `0.8775` · date `2026-05-29 22:20:32`

multi-seed validation seed 12345 (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-0-1--double-integrator--unlabeled--12"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--12` · ledger line 40 at the time of writing · cps `0.8705` · date `2026-05-30 00:30:59`

multi-seed validation seed 99 (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-2-1--double-integrator--unlabeled--1"></a>
## v2.2.1 · double_integrator · -

`anchor: v2-2-1--double-integrator--unlabeled--1` · ledger line 41 at the time of writing · cps `0.4161` · date `2026-06-17 19:38:24`

L_feas aux loss (weight 0.1, seed 0, short 10k proof-of-mechanism); engaged, no collapse; not SOTA-comparable. SOTA UNCHANGED. See cbf_deriv_feasibility_build.md. (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-2-1--double-integrator--unlabeled--2"></a>
## v2.2.1 · double_integrator · -

`anchor: v2-2-1--double-integrator--unlabeled--2` · ledger line 42 at the time of writing · cps `0.6818` · date `2026-06-17 20:16:56`

L_feas weight 0.3 (seed 0, full budget); no collapse; cps 0.6818 < SOTA, gap is seed/init not the term. SOTA UNCHANGED. (cps_v2 @n2000 re-eval; legacy cell @n500)

<a id="v2-2-1--double-integrator--unlabeled--3"></a>
## v2.2.1 · double_integrator · -

`anchor: v2-2-1--double-integrator--unlabeled--3` · ledger line 43 at the time of writing · cps `0.6995` · date `2026-06-18 00:10:01`

h-band L_feas (weight 0.1, seed 42, best@28000); collision 0.064 >> SOTA, cps 0.6995. SOTA UNCHANGED. (cps_v2 @n2000 re-eval; legacy cell @n500 best@28000)

<a id="v2-2-2--double-integrator--unlabeled--1"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--1` · ledger line 44 at the time of writing · cps `0.8415` · date `2026-06-18 08:03:59`

Collision-precursor injection (L_feas off); collision 0.006 == SOTA but reach 0.940 < 0.958 -> cps 0.8415. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--2"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--2` · ledger line 45 at the time of writing · cps `0.8150` · date `2026-06-18`

Injection best.pt@35000 on n2000; collision 0.0085 < SOTA but reach 0.9305 -> cps 0.8150. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--3"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--3` · ledger line 46 at the time of writing · cps `0.8607` · date `2026-06-18`

Injection, ckpt@40000 sweep-selected (selection bias); cps 0.8607, collision 0.0045 but CI overlaps SOTA -> tie, not a beat. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--4"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--4` · ledger line 47 at the time of writing · cps `0.6386` · date `2026-06-18`

Input-rate dead-zone+curriculum on injection; cps 0.6386 << injection-only, du never suppressed -> input-rate axis FAILED. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--5"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--5` · ledger line 48 at the time of writing · cps `0.7715` · date `2026-06-18`

u-reg L_u (weight 0.5) on injection, best.pt@17000; n2000 0.7715 < injection-only, collision not reduced. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--6"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--6` · ledger line 49 at the time of writing · cps `0.8404` · date `2026-06-18`

u-reg w0.5 n2000 true-best@35000; cps 0.8404 < injection 0.8607 and SOTA, over-speed uncurbed -> u-reg REJECTED. SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--7"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--7` · ledger line 50 at the time of writing · cps `0.7770` · date `2026-06-19`

Sobolev gradient-matching (w0.07) on injection; mechanism curbed over-speed but reach cost -> cps 0.7770 regressed. SOTA UNCHANGED. See sobolev_build.md.

<a id="v2-2-2--double-integrator--unlabeled--8"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--8` · ledger line 51 at the time of writing · cps `0.0056` · date `2026-06-19`

RPCBF robust deploy filter on SOTA (no retrain); CATASTROPHIC -- collision 10x, 81% infeasible (worst-case value vs nominal-calibrated filter). SOTA UNCHANGED.

<a id="v2-2-2--double-integrator--unlabeled--9"></a>
## v2.2.2 · double_integrator · -

`anchor: v2-2-2--double-integrator--unlabeled--9` · ledger line 52 at the time of writing · cps `0.8239` · date `2026-06-19`

Gated velocity-only Sobolev (w2.2) on injection; reach preserved, over-speed curbed, but collision 0.0045->0.0200 -> cps 0.8239 < base. Sobolev family exhausted. SOTA UNCHANGED.

<a id="v2-0-1--double-integrator--unlabeled--13"></a>
## v2.0.1 · double_integrator · -

`anchor: v2-0-1--double-integrator--unlabeled--13` · ledger line 53 at the time of writing · cps `0.8568` · date `2026-06-18`

Prior DI SOTA (n2000 baseline of record); SUPERSEDED 2026-06-20 by v2.3.0 (3-seed 0.8698). Kept for history.

<a id="v2-2-2-uni--unicycle--unlabeled"></a>
## v2.2.2-uni · unicycle · -

`anchor: v2-2-2-uni--unicycle--unlabeled` · ledger line 54 at the time of writing · cps `0.4981` · date `2026-06-19`

First unicycle JT-PNCBF baseline (all add-ons off); cps 0.4981, deficit is navigation not safety; not DI-comparable. Bar for Step-2.

<a id="v2-2-2-uni-inj--unicycle--unlabeled--1"></a>
## v2.2.2-uni-inj · unicycle · -

`anchor: v2-2-2-uni-inj--unicycle--unlabeled--1` · ledger line 55 at the time of writing · cps `0.6243` · date `2026-06-19`

Best unicycle: injection ported; cps 0.4981->0.6243 (+0.126, non-overlapping CIs), gain is navigation. Secured v2.2.2 unicycle SOTA; not DI-comparable. See injection_build.md.

<a id="v2-2-2-uni-inj--unicycle--unlabeled--2"></a>
## v2.2.2-uni-inj · unicycle · -

`anchor: v2-2-2-uni-inj--unicycle--unlabeled--2` · ledger line 56 at the time of writing · cps `0.6720` · date `2026-06-20`

Same secured best.pt@28000 re-scored at goal_speed_radius 0.50 (eval-time); cps 0.6243->0.6720 (timeout recovered). Not DI-comparable.

<a id="v2-3-0--double-integrator--unlabeled--1"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--1` · ledger line 57 at the time of writing · cps `0.8702` · date `2026-06-20`

D_pi=5k/D_V=1M window, seed 42, ckpt@40500; cps 0.8702 (+0.013), CI includes 0.8568 -> tie, not a beat; no collapse. See dpi_policy_buffer_window.md. (cps_v2 @n2000 on best.pt; legacy cell @40500-reselected)

<a id="v2-3-0--double-integrator--unlabeled--2"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--2` · ledger line 58 at the time of writing · cps `0.8700` · date `2026-06-20`

D_pi=2k/D_V=1M, seed 42, ckpt@40500; cps 0.8700 == 5k run (D_pi recency saturates), CI overlaps -> tie; no collapse. See dpi_policy_buffer_window.md.

<a id="v2-3-0--double-integrator--unlabeled--3"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--3` · ledger line 59 at the time of writing · cps `0.8473` · date `2026-06-20`

D_V-shrink to 10k on D_pi=2k, seed 42; cps 0.8473 (-0.0227 vs D_V=1M), below SOTA; KEY: no collapse occurred. See dpi_policy_buffer_window.md. (cps_v2 @n2000 on best.pt; legacy cell @48000-reselected)

<a id="v2-3-0--double-integrator--unlabeled--4"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--4` · ledger line 60 at the time of writing · cps `0.8766` · date `2026-06-20`

Multi-seed (2/3) D_pi=2k/D_V=1M, seed 12345, ckpt@49500; cps 0.8766 (highest single-seed); no collapse; aggregate pending. See dpi_policy_buffer_window.md.

<a id="v2-3-0--double-integrator--unlabeled--5"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--5` · ledger line 61 at the time of writing · cps `0.8628` · date `2026-06-20`

Multi-seed (3/3) D_pi=2k/D_V=1M, seed 99, ckpt@45000; cps 0.8628, CI includes 0.8568; no collapse. See dpi_policy_buffer_window.md.

<a id="v2-3-0--double-integrator--unlabeled--6"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--6` · ledger line 62 at the time of writing · cps `0.8698` · date `2026-06-20`

New DI SOTA 0.8698 (3-seed mean); CI [0.8527,0.8869] overlaps prior 0.8568 -> point-estimate, not a CI-confirmed beat. See v2.3.0_results.md. FLAG (§2.4): cps_v2 reads '-', so this row is not rankable under the current definition — bold retained but pending re-scoring. SUPERSEDED at the v2.9.1 close (2026-08-14): bold moves to L314, the current-basis 3-seed re-score of this row's own checkpoint set. This row keeps its content; its 0.8698 remains unrankable under cps_v2.

### relocated from the `verdict` column (pre-cap text, verbatim)

New DI SOTA 0.8698 (3-seed mean); CI [0.8527,0.8869] overlaps prior 0.8568 -> point-estimate, not a CI-confirmed beat. See v2.3.0_results.md. FLAG (§2.4): cps_v2 reads '-', so this row is not rankable under the current definition — bold retained but pending re-scoring. SUPERSEDED at the v2.9.1 close (2026-08-14): bold moves to L314, the current-basis 3-seed re-score of this row's own checkpoint set. This row keeps its content; its 0.8698 remains unrankable under cps_v2.

<a id="v2-3-1--double-integrator--unlabeled"></a>
## v2.3.1 · double_integrator · -

`anchor: v2-3-1--double-integrator--unlabeled` · ledger line 63 at the time of writing · cps `0.8423` · date `2026-06-21`

v_init_max 0.5->2.0 fast-approach axis REGRESSED: n2000 0.8423 [0.8219,0.8642] vs v2.3.0 0.8700 (infeas 0.115 up vs 0.088, below SOTA; CI-overlap, single-seed). Diagnostic on a 2.0 pool: v2.3.0 0.8927 > v2.3.1 0.8566 with LOWER infeas -> worse policy IN-DISTRIBUTION, not pool-mismatch. SOTA UNCHANGED. See v2.3.1_results.md. (cps_v2 @n2000 on best.pt; legacy cell @40500-reselected)

<a id="v2-3-0-umax20--double-integrator--unlabeled"></a>
## v2.3.0_umax20 · double_integrator · -

`anchor: v2-3-0-umax20--double-integrator--unlabeled` · ledger line 64 at the time of writing · cps `0.9419` · date `2026-06-29`

EXPERIMENT (user-directed u_max 2->20): cps 0.9419 [0.9290,0.9551] vs v2.3.0 0.8700. Single global change u_max 2->20 (policy/LQR/dynamics/projection box all ±20; v_max still 2.5). Gains (reach 0.947->0.98, stuck 0.026->0.001, infeas 0.088->0.062) from ~10x accel authority + the ±20 box rarely binding (box-induced infeasibility removed). Not a ±2-deployable result; single-seed, best.pt (no n2000 re-selection). SOTA UNCHANGED (different actuator spec). See docs/versions/v2.3.0_results.md (Experiment: u_max widening section).

### relocated from the `verdict` column (pre-cap text, verbatim)

EXPERIMENT (user-directed): one global actuator change, u_max 2 -> 20, against the v2.3.0 double_integrator baseline, with v_max held at 2.5. cps 0.9419 [0.9290, 0.9551] against v2.3.0's 0.8700. NOT comparable to any +/-2 row, since the actuator spec differs and the result is not +/-2-deployable; single seed, best.pt with no n2000 re-selection. SOTA UNCHANGED, no bold change, no promotion. Detail docs/versions/v2.3.0_results.md (Experiment: u_max widening) and docs/ledger_verdicts.md#v2-3-0-umax20--double-integrator--unlabeled.

<a id="v2-4-0--double-integrator--unlabeled--1"></a>
## v2.4.0 · double_integrator · -

`anchor: v2-4-0--double-integrator--unlabeled--1` · ledger line 65 at the time of writing · cps `0.6810 [0.6513, 0.7116]` · date `2026-07-02`

REGRESSED. SOTA UNCHANGED

<a id="v2-4-0--double-integrator--unlabeled--2"></a>
## v2.4.0 · double_integrator · -

`anchor: v2-4-0--double-integrator--unlabeled--2` · ledger line 66 at the time of writing · cps `0.7092 [0.6795, 0.7384]` · date `2026-07-02`

REGRESSED. SOTA UNCHANGED (cps-v2 re-eval path diff +0.010)

<a id="v2-4-0--double-integrator--unlabeled--3"></a>
## v2.4.0 · double_integrator · -

`anchor: v2-4-0--double-integrator--unlabeled--3` · ledger line 67 at the time of writing · cps `-0.0667 [-0.1170, -0.0169]` · date `2026-07-02`

collapse. SOTA UNCHANGED

<a id="v2-4-0--double-integrator--unlabeled--4"></a>
## v2.4.0 · double_integrator · -

`anchor: v2-4-0--double-integrator--unlabeled--4` · ledger line 68 at the time of writing · cps `0.5540 (in-loop)` · date `2026-07-02`

collapse / audited. SOTA UNCHANGED (cps_v2 @n2000 re-eval of best.pt; legacy cell = in-loop@9000)

<a id="v2-4-0--double-integrator--unlabeled--5"></a>
## v2.4.0 · double_integrator · -

`anchor: v2-4-0--double-integrator--unlabeled--5` · ledger line 69 at the time of writing · cps `0.8002 [0.7734, 0.8252]` · date `2026-07-02`

hygiene PASS, horizon NEGATIVE. SOTA UNCHANGED

<a id="v2-4-1--double-integrator--unlabeled--1"></a>
## v2.4.1 · double_integrator · -

`anchor: v2-4-1--double-integrator--unlabeled--1` · ledger line 70 at the time of writing · cps `0.8109 [0.7887, 0.8343]` · date `2026-07-03`

target-refuted. SOTA UNCHANGED

<a id="v2-4-1--double-integrator--unlabeled--2"></a>
## v2.4.1 · double_integrator · -

`anchor: v2-4-1--double-integrator--unlabeled--2` · ledger line 71 at the time of writing · cps `0.5843 [0.5486, 0.6202]` · date `2026-07-03`

REGRESSED (conservatism). SOTA UNCHANGED

<a id="v2-4-1--double-integrator--unlabeled--3"></a>
## v2.4.1 · double_integrator · -

`anchor: v2-4-1--double-integrator--unlabeled--3` · ledger line 72 at the time of writing · cps `` · date `2026-07-03`

blocked (grad-tail). SOTA UNCHANGED

<a id="v2-4-1--double-integrator--unlabeled--4"></a>
## v2.4.1 · double_integrator · -

`anchor: v2-4-1--double-integrator--unlabeled--4` · ledger line 73 at the time of writing · cps `0.8781 [0.8599, 0.8955]` · date `2026-07-03`

target-refuted; NO conservatism echo. SOTA UNCHANGED

<a id="v2-4-2--double-integrator--unlabeled--1"></a>
## v2.4.2 · double_integrator · -

`anchor: v2-4-2--double-integrator--unlabeled--1` · ledger line 74 at the time of writing · cps `0.7448 [0.7138, 0.7734]` · date `2026-07-08`

REGRESSED (post-peak decline, tail-ratchet). SOTA UNCHANGED

<a id="v2-4-2--double-integrator--unlabeled--2"></a>
## v2.4.2 · double_integrator · -

`anchor: v2-4-2--double-integrator--unlabeled--2` · ledger line 75 at the time of writing · cps `0.7466 [0.7126, 0.7764]` · date `2026-07-08`

REGRESSED (tail-cap; decline mitigated not removed, ceiling unchanged = Exp 1). SOTA UNCHANGED

<a id="v2-5-0--double-integrator--unlabeled--1"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--1` · ledger line 76 at the time of writing · cps `0.5387 [0.5110, 0.5640]` · date `2026-07-08`

Training-free analytic V_M+LQR: SAFE (coll 0.010 vs LQR 0.34; empty-int@{V_M<=0} 0.007) but cps 0.539 << gate; infeas raw = far-field V_M-saturation singular (benign). NOT SOTA-comparable.

<a id="v2-5-0--double-integrator--unlabeled--2"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--2` · ledger line 77 at the time of writing · cps `0.5616 [0.5360, 0.5840]` · date `2026-07-08`

Headline condition; feasibility-witness empty@cert 0.015; cps 0.562 << gate 0.80, < OC 0.797 / v2.4.2 0.745 / v2.3.0 0.87. Training-free, NOT SOTA-comparable.

<a id="v2-5-0--double-integrator--unlabeled--3"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--3` · ledger line 78 at the time of writing · cps `0.5685 [0.5440, 0.5910]` · date `2026-07-08`

Largest library; P-A3 monotone (stuck 0.067>=0.055>=0.0535, cps up). cps 0.569 << gate. Training-free, NOT SOTA-comparable.

<a id="v2-5-0--double-integrator--unlabeled--4"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--4` · ledger line 79 at the time of writing · cps `0.5893 [0.5650, 0.6110]` · date `2026-07-09`

Structural safety CONFIRMED (P-B2: collision flat ~0.005, training-INDEPENDENT across all 20 ckpts). Liveness: empty-only cps 0.8265 > A1' 0.8088 & OC 0.797, but < P-B4 0.85 & reach 0.9225<0.94; canonical 0.5893 far-field-singular-capped. Policy peaks ~step9k then drifts down; proj_mag no decline. NOT SOTA (learned-V 0.8698); dual-convention.

<a id="v2-5-0--double-integrator--unlabeled--5"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--5` · ledger line 80 at the time of writing · cps `0.5893 [0.5620, 0.6140]` · date `2026-07-10`

legacy cps 0.5893 / cps-v2 0.8327 @D-slow (cps-v2 = row-satisfied-singular reclassification = empty-only). Marginal over Stage-B (cps_v2 0.8254->0.8327, stuck 0.049->0.043, structural safety P-F3 CONFIRMED) but P-F1 (proj_mag halving) REFUTED — friction did NOT reduce intervention (active-band proj_mag ~1.33 unchanged); P-F2 reach 0.925<0.94 & P-F4 cps_v2<0.85 REFUTED. P-L4: stuck reduction 80% open-area. NOT SOTA (v2.3.0 cps_v2 0.9010). Fast-vs-fast re-baselined.

### relocated from the `verdict` column (pre-cap text, verbatim)

legacy cps 0.5893 / cps-v2 0.8327 @D-slow (cps-v2 = row-satisfied-singular reclassification = empty-only). Marginal over Stage-B (cps_v2 0.8254->0.8327, stuck 0.049->0.043, structural safety P-F3 CONFIRMED) but P-F1 (proj_mag halving) REFUTED — friction did NOT reduce intervention (active-band proj_mag ~1.33 unchanged); P-F2 reach 0.925<0.94 & P-F4 cps_v2<0.85 REFUTED. P-L4: stuck reduction 80% open-area. NOT SOTA (v2.3.0 cps_v2 0.9010). Fast-vs-fast re-baselined.

<a id="v2-5-0--double-integrator--unlabeled--6"></a>
## v2.5.0 · double_integrator · -

`anchor: v2-5-0--double-integrator--unlabeled--6` · ledger line 81 at the time of writing · cps `0.9230` · date `2026-07-10`

Training-free, NOT SOTA-comparable (no learned object). cps==cps_v2 (no filter steps => infeasibility 0 by convention). GATE PASSED: exhaustive coverage(full grammar) 0.98 (P-R4). Follower SAFE (coll 0.000) + reaches 0.9615 near-optimally (median ttg-R(x0) 0.05s), but rescue hypothesis REFUTED (P-R6: rescues 0.14-0.24 of B-2 stall-onsets vs LQR 10/15 open-area; certified plan exists from only 0.19 of onsets — hard-geometry traps) and cost REFUTED (P-R7: beam @B=4096 3476ms >> 300ms bar). Coverage carried by capture=LQR terminal (config iii alone 0.944), which is also the cost driver. See reach_witness.md Stage R-2.

### relocated from the `verdict` column (pre-cap text, verbatim)

Training-free reach-witness follower, carrying no learned object. GATE PASSED at exhaustive coverage 0.98 (P-R4), the follower safe (coll 0.000) and reaching 0.9615 near-optimally, but the rescue hypothesis is REFUTED (P-R6) and the cost bar REFUTED (P-R7). NOT SOTA-comparable and not comparable to any filtered row: cps == cps_v2 only because there are no filter steps, so infeasibility is 0 by convention. No bold change, no promotion. Detail reach_witness.md Stage R-2 and docs/ledger_verdicts.md#v2-5-0--double-integrator--unlabeled--6.

<a id="v2-3-0--double-integrator--unlabeled--7"></a>
## v2.3.0 · double_integrator · -

`anchor: v2-3-0--double-integrator--unlabeled--7` · ledger line 82 at the time of writing · cps `0.8551` · date `2026-07-10`

REF-C (close-prep): v2.3.0 learned-V HardNet filter re-eval at D-fast (dt 0.01). GATE PASS — learned D-slow reproduces the rescore reference (legacy 0.8624 / cps_v2 0.9010) to numerical identity @batch250 (batch-200 fp jitter caused a spurious -0.010 miss, diagnosed). D-slow->D-fast (learned): cps_v2 0.9010->0.8938 (-0.0072), reach 0.955->0.9515, coll 0.009->0.0085, stuck 0.0195->0.027. KEY: the D-fast gain is V_M-SPECIFIC, NOT shared — V_M D-slow->D-fast cps_v2 +0.0031 (gains) while the learned filter REGRESSES -0.0072 (stuck-driven). coll CI [0.0045,0.013]. eval-only, changed deploy axis, parent v2.3.0; NOT a SOTA claim. See close_prep_batch.md T-A.

### relocated from the `verdict` column (pre-cap text, verbatim)

REF-C close-prep: the v2.3.0 learned-V HardNet filter re-evaluated at the D-fast deploy axis (dt 0.01) against its own D-slow score. GATE PASS on reproduction, and the move D-slow -> D-fast REGRESSES the learned filter by -0.0072 cps_v2 (stuck-driven) while V_M GAINS +0.0031 on the same move, so the D-fast gain is V_M-SPECIFIC and not shared. Eval-only on a changed deploy axis, so not comparable to D-slow rows; NOT a SOTA claim. No bold change, no promotion. Detail close_prep_batch.md T-A and docs/ledger_verdicts.md#v2-3-0--double-integrator--unlabeled--7.

<a id="v2-5-1--double-integrator--unlabeled--1"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--1` · ledger line 83 at the time of writing · cps `0.6780` · date `2026-07-11`

CPI iteration-1 (pathfinder, seed 42): pi_1 improved under frozen learned V_hat_0 as the deployed CBF. cps_v2==legacy (inf_canonical==inf_v2, no far-field-singular inflation). P-I1-1 (cps_v2>=0.85) REFUTED: the learned certificate admits 2.8% collisions (one-sided error). Single-seed. See phase_i1_loop_report.md I1.4.

<a id="v2-5-1--double-integrator--unlabeled--2"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--2` · ledger line 84 at the time of writing · cps `0.7378` · date `2026-07-11`

S' geometric shield on the it1 filter condition: collision 0.028->0.000, verified_start 1.000, overrides/ep 0.2745, 4.38x wall overhead. P-I1-2 PASS (0 verified-start collisions). Single-seed. See phase_i1_loop_report.md I1.4.

<a id="v2-5-1--double-integrator--unlabeled--3"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--3` · ledger line 85 at the time of writing · cps `0.7210` · date `2026-07-11`

CPI iteration-2 (pathfinder, seed 42): pi_2 under frozen V_hat_1. condition-A cps_v2 0.721 (+0.043 vs it1); collision 0.0245 caps it. P-PI2 gate 0.83 NOT met at it2 -> iteration 3. Single-seed. See phase_i1_loop_report.md I2.4.

<a id="v2-5-1--double-integrator--unlabeled--4"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--4` · ledger line 86 at the time of writing · cps `0.7845` · date `2026-07-11`

S' shield on it2 filter condition: collision 0.0245->0.000, verified_start 1.000, overrides/ep 0.375, 4.55x overhead. P-I1-2 PASS. Single-seed. See phase_i1_loop_report.md I2.4.

<a id="v2-5-1--double-integrator--unlabeled--5"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--5` · ledger line 87 at the time of writing · cps `0.6599` · date `2026-07-11`

CPI iteration-3 (pathfinder, seed 42): pi_3 under frozen V_hat_2. condition-A cps_v2 0.660 REGRESSED vs it2 0.721 (coll 0.0475). P-PI2 GATE FAILED (it2 0.721/it3 0.660 < 0.83) -> LINE REJECTED. P-I1-3 monotonicity HALT at k=3 (single-step family retention insufficient). Single-seed. See phase_i1_loop_report.md I3.

### relocated from the `verdict` column (pre-cap text, verbatim)

CPI iteration-3 pathfinder: policy pi_3 under the frozen certificate V_hat_2. Arm-A cps_v2 0.660 REGRESSED against iteration-2's 0.721, failing the P-PI2 gate and halting P-I1-3 monotonicity at k=3. The line is REJECTED, so this row is a terminal record rather than a comparator; single seed. No bold change, no promotion. Detail phase_i1_loop_report.md I3 and docs/ledger_verdicts.md#v2-5-1--double-integrator--unlabeled--5.

<a id="v2-5-1--double-integrator--unlabeled--6"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--6` · ledger line 88 at the time of writing · cps `0.7760` · date `2026-07-11`

S' shield on it3 filter condition: collision 0.0475->0.000, verified_start 1.000, overrides/ep 0.716, 4.45x overhead. P-I1-2 PASS (shield carries safety independent of V_hat). Single-seed. See phase_i1_loop_report.md I3.

<a id="v2-5-1--double-integrator--unlabeled--7"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--7` · ledger line 89 at the time of writing · cps `0.7378` · date `2026-07-12`

Deconfound (a): it2_ws seed 42 warm-started (pi_state only) from pathfinder it1, channel V_hat_1. condition-A cps_v2 0.738 vs pathfinder fresh-init it2 0.721 (+0.017) and it1 0.678 (paired +0.060 [+0.032,+0.090]). Single-seed row; 3-seed pooled + P-A2 in the close Note. See phase_i2_deconfound_report.md A5.

<a id="v2-5-1--double-integrator--unlabeled--8"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--8` · ledger line 90 at the time of writing · cps `0.7985` · date `2026-07-12`

S' shield on the it2_ws seed-42 filter condition: collision 0.024->0.000, verified_start 1.000, 0 verified-start collisions. P-I1-2 PASS. Single-seed. See phase_i2_deconfound_report.md A5.

<a id="v2-5-1--double-integrator--unlabeled--9"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--9` · ledger line 91 at the time of writing · cps `0.4090` · date `2026-07-12`

Deconfound (b): horizon-summary critic W (performance channel; enabled during training only). condition-A cps_v2 0.409 REGRESSED (stuck 0.2245 vs it1 0.090). P-B1 REFUTED (cps<0.73, stuck>0.045); component decomposition attributes the loss to stuck/reach, not the safety channel (coll 0.028 unchanged). Single-seed. See phase_i2_deconfound_report.md A5.

<a id="v2-5-1--double-integrator--unlabeled--10"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--10` · ledger line 92 at the time of writing · cps `0.6560` · date `2026-07-12`

P1 distribution-matched evaluation: pi_3 warm-started from R1 (it2_ws), channel V_hat_2dm (labels augmented with 200k on-policy states; on-policy eps_q(0.10)=0.0, P-P1a PASS). condition-A cps_v2 0.656 REGRESSED vs it2_ws 0.738; collision 0.059 UP (vs 0.024). P-P1b REFUTED (coll>0.015, cps<0.76). Single-seed. See phase_i2_deconfound_report.md P1.4.

<a id="v2-5-1--double-integrator--unlabeled--11"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--11` · ledger line 93 at the time of writing · cps `0.7853` · date `2026-07-12`

S' shield on pi_3 filter condition: collision 0.059->0.000, verified_start 1.000, 0 verified-start collisions. P-I1-2 PASS. condition-B cps_v2 0.785 < 0.80 -> P-P1c REFUTED. Single-seed. See phase_i2_deconfound_report.md P1.4.

<a id="v2-5-1--double-integrator--unlabeled--12"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--12` · ledger line 94 at the time of writing · cps `0.8226` · date `2026-07-13`

E axis: exact single-backup certificate V_m0 as the deployed filter (no library, no clip; warm-start from pathfinder it1 = SAME init as R1, so the delta isolates the channel). condition-A cps_v2 0.823 [0.797,0.847] vs R1 learned-V_hat_1 0.738 (+0.085); collision 0.024->0.0015 (16x). P-E1 PASS (coll<=.010 AND cps>=.80); GATE-E PASSED. Single-seed. See phase_i3_perf_report.md E2.

<a id="v2-5-1--double-integrator--unlabeled--13"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--13` · ledger line 95 at the time of writing · cps `0.8367` · date `2026-07-13`

S' shield on the E2 exact_m0 filter condition: collision 0.0015->0.000, verified_start 1.000, 0 verified-start collisions. P-I1-2 PASS. Single-seed.

<a id="v2-5-1--double-integrator--unlabeled--14"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--14` · ledger line 96 at the time of writing · cps `0.8280` · date `2026-07-13`

E replication seed 99 (warm-start R3). condition-A cps_v2 0.828 vs R5 (V_hat_1) 0.7373 (+0.091). P-I1-2 PASS. Single-seed.

<a id="v2-5-1--double-integrator--unlabeled--15"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--15` · ledger line 97 at the time of writing · cps `0.8422` · date `2026-07-13`

S' shield on the E_s99 condition: collision 0.0015->0.000, 0 verified-start collisions. P-I1-2 PASS. Selection: best.pt (step 1500) confirmed on a fresh n1000 train-mode pool (0.8152 > 3000/4500). Single-seed.

<a id="v2-5-1--double-integrator--unlabeled--16"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--16` · ledger line 98 at the time of writing · cps `0.7974` · date `2026-07-13`

E replication seed 12345 (warm-start R4). condition-A cps_v2 0.797 vs R6 (V_hat_1) 0.7482 (+0.049). P-I1-2 PASS. Single-seed.

<a id="v2-5-1--double-integrator--unlabeled--17"></a>
## v2.5.1 · double_integrator · -

`anchor: v2-5-1--double-integrator--unlabeled--17` · ledger line 99 at the time of writing · cps `0.8147` · date `2026-07-13`

S' shield on the E_s12345 condition: collision 0.0025->0.000, 0 verified-start collisions. P-I1-2 PASS. Single-seed.

<a id="v2-5-2--unicycle--unlabeled"></a>
## v2.5.2 · unicycle · -

`anchor: v2-5-2--unicycle--unlabeled` · ledger line 100 at the time of writing · cps `-0.0115` · date `2026-07-14`

Stage-1 comparator for P-U3. Nominal LQR, no filter -> collision 0.337 (LQR does not avoid obstacles); cps_v2==cps_legacy (no filter, infeasibility 0); CI [-0.071, 0.051]. Flagged for Researcher (version,system)-SOTA classification; unicycle rows are never SOTA-bolded against DI rows (06_workflow §2.4 patched). See phase_s1_report.md.

<a id="v2-5-2--double-integrator--unlabeled--1"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--1` · ledger line 101 at the time of writing · cps `0.8755` · date `2026-07-14`

J-arc condition A (full_n2000 eval_full_di_n2000_seed23456, adopted step_040500.pt). P-J0 per-seed cps_legacy 0.8755~0.870. cps_v2 CI [0.8713,0.9109]. Flag Researcher; NOT SOTA-classed. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--2"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--2` · ledger line 102 at the time of writing · cps `0.8651` · date `2026-07-14`

J-arc condition A (adopted step_045000.pt). P-J0 per-seed cps_legacy 0.8651~0.8628. cps_v2 CI [0.8768,0.9181]. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--3"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--3` · ledger line 103 at the time of writing · cps `0.8811` · date `2026-07-14`

J-arc condition A (adopted step_049500.pt). P-J0 per-seed cps_legacy 0.8811~0.8766. cps_v2 CI [0.8894,0.9250]. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--4"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--4` · ledger line 104 at the time of writing · cps `0.8739` · date `2026-07-14`

J-arc condition A POOLED. **P-J0 GATE PASS**: per-seed cps_legacy within ±0.010; pooled 0.8739~0.8698 (Δ+0.0041); cps_v2 0.8992~0.9005 (Δ−0.0013) corroborates -> value channel intact, comparison valid. NOT SOTA-bold (Researcher). See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--5"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--5` · ledger line 105 at the time of writing · cps `0.9090` · date `2026-07-14`

J-arc condition B. verified_start_coll 0, vstart_frac 1.0, ovr/ep 0.530, checks/ep 200, shield× 6.55. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--6"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--6` · ledger line 106 at the time of writing · cps `0.9175` · date `2026-07-14`

J-arc condition B. verified_start_coll 0, vstart_frac 1.0, ovr/ep 0.466, checks/ep 200, shield× 7.80. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--7"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--7` · ledger line 107 at the time of writing · cps `0.9247` · date `2026-07-14`

J-arc condition B. verified_start_coll 0, vstart_frac 1.0, ovr/ep 0.472, checks/ep 200, shield× 7.37. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--8"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--8` · ledger line 108 at the time of writing · cps `0.9171` · date `2026-07-14`

J-arc condition B POOLED. **P-J1 HALT-gate PASS** (verified_start_coll 0, vstart_frac 1.0, ALL seeds). **P-J2 PASS** (0.9171>0.8422). Shield transfers to the learned filter at NO cps cost — a gain: condition A 0.8992->0.9171, coll 0.0060->0.0000, reach 0.9533->0.9548. ovr/ep 0.489 -> **P-J4 FALSIFIED** (>1.0 predicted). Different deployment class (learned filter + S' shield); NOT SOTA-bold; flag Researcher. See phase_j_report.md.

### relocated from the `verdict` column (pre-cap text, verbatim)

J-arc condition B, pooled: the S' shield deployed on top of the learned filter. **P-J1 HALT-gate PASS** and **P-J2 PASS** (0.9171 > 0.8422), the shield transferring at NO cps cost and in fact a gain over condition A (0.8992 -> 0.9171, coll 0.0060 -> 0.0000), while **P-J4 is FALSIFIED** at ovr/ep 0.489 against the >1.0 predicted. A different deployment class (learned filter plus S' shield), so not comparable to unshielded rows. NOT SOTA-bold; flag Researcher. Detail phase_j_report.md and docs/ledger_verdicts.md#v2-5-2--double-integrator--unlabeled--8.

<a id="v2-5-2--double-integrator--unlabeled--9"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--9` · ledger line 109 at the time of writing · cps `0.9073` · date `2026-07-14`

J-arc condition B' (ladder 0.75/0.5/0.25). verified_start_coll 0, ovr/ep 0.688, checks/ep 437.8, shield× 14.31. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--10"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--10` · ledger line 110 at the time of writing · cps `0.9197` · date `2026-07-14`

J-arc condition B'. verified_start_coll 0, ovr/ep 0.518, checks/ep 417.3, shield× 16.56. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--11"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--11` · ledger line 111 at the time of writing · cps `0.9253` · date `2026-07-14`

J-arc condition B'. verified_start_coll 0, ovr/ep 0.512, checks/ep 396.1, shield× 14.38. Flag Researcher. See phase_j_report.md.

<a id="v2-5-2--double-integrator--unlabeled--12"></a>
## v2.5.2 · double_integrator · -

`anchor: v2-5-2--double-integrator--unlabeled--12` · ledger line 112 at the time of writing · cps `0.9174` · date `2026-07-14`

J-arc condition B' POOLED. verified_start_coll 0 all seeds. **P-J3 PASS** (reach 0.9548 >= condition A 0.9533−0.05; no cps drop to attribute). cps_v2 0.9174 ≈ condition B 0.9171 at ~2× checks/ep (417 vs 200) and ovr/ep 0.573 vs 0.489 -> ladder buys no cps here. NOT SOTA-bold; flag Researcher. See phase_j_report.md.

### relocated from the `verdict` column (pre-cap text, verbatim)

J-arc condition B', pooled: the same shield as condition B at roughly twice the checks per episode (417 against 200). **P-J3 PASS** (reach 0.9548 >= condition A's 0.9533 - 0.05), with cps_v2 0.9174 essentially equal to condition B's 0.9171, so the ladder buys no cps here. A different deployment class, so not comparable to unshielded rows. NOT SOTA-bold; flag Researcher. Detail phase_j_report.md and docs/ledger_verdicts.md#v2-5-2--double-integrator--unlabeled--12.

<a id="v2-6-0--quadrotor-planar--unlabeled--1"></a>
## v2.6.0 · quadrotor_planar · -

`anchor: v2-6-0--quadrotor-planar--unlabeled--1` · ledger line 113 at the time of writing · cps `-0.2270` · date `2026-07-15`

[P1] eval_only nominal baseline; not SOTA-eligible.

<a id="v2-6-0--quadrotor-planar--unlabeled--2"></a>
## v2.6.0 · quadrotor_planar · -

`anchor: v2-6-0--quadrotor-planar--unlabeled--2` · ledger line 114 at the time of writing · cps `0.1679` · date `2026-07-15`

[P1] eval_only learned-filter pre-JT; not SOTA-eligible.

<a id="v2-6-0--quadrotor-planar--unlabeled--3"></a>
## v2.6.0 · quadrotor_planar · -

`anchor: v2-6-0--quadrotor-planar--unlabeled--3` · ledger line 115 at the time of writing · cps `0.2920` · date `2026-07-15`

[P1] first navigable JT policy; P3 confirmed.

<a id="v2-6-0--quadrotor-planar--unlabeled--4"></a>
## v2.6.0 · quadrotor_planar · -

`anchor: v2-6-0--quadrotor-planar--unlabeled--4` · ledger line 116 at the time of writing · cps `0.3213` · date `2026-07-15`

[P1] eval_only timeout re-eval; not SOTA-eligible.

<a id="v2-6-1--quadrotor-planar--unlabeled--1"></a>
## v2.6.1 · quadrotor_planar · -

`anchor: v2-6-1--quadrotor-planar--unlabeled--1` · ledger line 117 at the time of writing · cps `0.1769` · date `2026-07-15`

[P1] eval_only corrected-plant pre-JT; not SOTA-eligible.

<a id="v2-6-1--quadrotor-planar--unlabeled--2"></a>
## v2.6.1 · quadrotor_planar · -

`anchor: v2-6-1--quadrotor-planar--unlabeled--2` · ledger line 118 at the time of writing · cps `0.5753` · date `2026-07-16`

[P1] corrected plant; H1/H2 confirmed.

<a id="v2-6-2--quadrotor-planar--unlabeled--1"></a>
## v2.6.2 · quadrotor_planar · -

`anchor: v2-6-2--quadrotor-planar--unlabeled--1` · ledger line 119 at the time of writing · cps `0.6097` · date `2026-07-16`

[P1] situational objective; cps up, collision up.

<a id="v2-6-2--quadrotor-planar--unlabeled--2"></a>
## v2.6.2 · quadrotor_planar · -

`anchor: v2-6-2--quadrotor-planar--unlabeled--2` · ledger line 120 at the time of writing · cps `0.7311` · date `2026-07-16`

[P1] H-appr confirmed; superseded by v2.7.0.

<a id="v2-6-2--quadrotor-planar--unlabeled--3"></a>
## v2.6.2 · quadrotor_planar · -

`anchor: v2-6-2--quadrotor-planar--unlabeled--3` · ledger line 121 at the time of writing · cps `0.7365` · date `2026-07-16`

[P1] deflation NOT a collision lever; Δcoll≤0.0035 all cells; NULL.

<a id="v2-7-0--quadrotor-planar--unlabeled--1"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--1` · ledger line 122 at the time of writing · cps `0.6228` · date `2026-07-17`

[P1] LQR+HardNet(new V_hat); pre-JT.

<a id="v2-7-0--quadrotor-planar--unlabeled--2"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--2` · ledger line 123 at the time of writing · cps `0.8232` · date `2026-07-17`

[P1] obs theta-fix; SOTA single-seed; supersedes v2.6.2; superseded by v2.7.1.

<a id="v2-7-0--quadrotor-planar--unlabeled--3"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--3` · ledger line 124 at the time of writing · cps `0.4729` · date `2026-07-17`

[P1] inject 0.10 value; pessimism datum.

<a id="v2-7-0--quadrotor-planar--unlabeled--4"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--4` · ledger line 125 at the time of writing · cps `0.8327` · date `2026-07-17`

[P1] REFUTED: IC oversampling (nat.~0.5); CI-overlap iter-1; not headline.

<a id="v2-7-0--quadrotor-planar--unlabeled--5"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--5` · ledger line 126 at the time of writing · cps `-` · date `2026-07-17`

[P1] bptt_T=60 training collapse; refuted (unstable).

<a id="v2-7-0--quadrotor-planar--unlabeled--6"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--6` · ledger line 127 at the time of writing · cps `0.5191` · date `2026-07-18`

[P1] continuing-collector value; CI-sep WEAKER pre-JT vs iter-1 0.6490; datum.

<a id="v2-7-0--quadrotor-planar--unlabeled--7"></a>
## v2.7.0 · quadrotor_planar · -

`anchor: v2-7-0--quadrotor-planar--unlabeled--7` · ledger line 128 at the time of writing · cps `0.8343` · date `2026-07-18`

[P1] continuing collector (Track A); H-nondegradation ON PAR w/ iter-1 (cps 0.8343 [0.804,0.863] vs 0.8232 [0.791,0.852] CI-overlap; coll 0.041 [0.033,0.050] vs 0.0455 CI-overlap); H-throughput +2% pipeline (value 1.13x, JT neutral); NOT SOTA-separated; H-thr NOT MET 0.94x eps/hr (P-D).

<a id="v2-7-1--quadrotor-planar--unlabeled--1"></a>
## v2.7.1 · quadrotor_planar · -

`anchor: v2-7-1--quadrotor-planar--unlabeled--1` · ledger line 129 at the time of writing · cps `0.5118` · date `2026-07-18`

[P1] corridor-cell value; pre-JT on par w/ iter-5 0.5191 (CI-overlap); M2-gate PASS; corridor probe gap p50 +0.024.

<a id="v2-7-1--quadrotor-planar--unlabeled--2"></a>
## v2.7.1 · quadrotor_planar · -

`anchor: v2-7-1--quadrotor-planar--unlabeled--2` · ledger line 130 at the time of writing · cps `0.8407` · date `2026-07-18`

[P1] corridor-cell state injection; cps 0.8407 [0.8104,0.8690] highest point but CI-overlaps iter-5 0.8343 & L115 0.8232 (NOT SOTA-separated); coll 0.0410 = iter-5 (not below). corridor start-injection: H-perf not met; ~3x L_g at collisions (1.4481 vs iter-5 0.4915); P-B' k=10 improvable 0.958.

<a id="v2-7-1--quadrotor-planar--unlabeled--3"></a>
## v2.7.1 · quadrotor_planar · -

`anchor: v2-7-1--quadrotor-planar--unlabeled--3` · ledger line 131 at the time of writing · cps `0.9152` · date `2026-07-18`

[P1] k5 empty-fallback eval; CI-sep vs none.

<a id="v2-7-1--quadrotor-planar--unlabeled--4"></a>
## v2.7.1 · quadrotor_planar · -

`anchor: v2-7-1--quadrotor-planar--unlabeled--4` · ledger line 132 at the time of writing · cps `0.9036` · date `2026-07-18`

[P1] k5 on no-inject ckpt; CI-sep. SOTA; adopted config + k5 fallback. SUPERSEDED at the v2.9.1 close (2026-08-14): bold moves to L313, the current-basis re-score of this row's own checkpoint (3b27d691). This row keeps its content; its 0.9036 was scored on a superseded basis (k5 fallback cell, two-leg reach predicate, value ceiling 1.0).

<a id="v2-7-1--quadrotor-planar--unlabeled--5"></a>
## v2.7.1 · quadrotor_planar · -

`anchor: v2-7-1--quadrotor-planar--unlabeled--5` · ledger line 133 at the time of writing · cps `0.9009` · date `2026-07-18`

[P1] k5 on SOTA ckpt; CI-sep.

<a id="v2-7-1--unicycle--unlabeled"></a>
## v2.7.1 · unicycle · -

`anchor: v2-7-1--unicycle--unlabeled` · ledger line 134 at the time of writing · cps `0.6588` · date `2026-07-18`

[P1] k5 eval; point-improved, CI-overlap.

<a id="v2-7-1--double-integrator--unlabeled"></a>
## v2.7.1 · double_integrator · -

`anchor: v2-7-1--double-integrator--unlabeled` · ledger line 135 at the time of writing · cps `0.8580` · date `2026-07-18`

[P1] k5 eval; null as predicted.

<a id="v2-7-2--quadrotor-3d--unlabeled--1"></a>
## v2.7.2 · quadrotor_3d · -

`anchor: v2-7-2--quadrotor-3d--unlabeled--1` · ledger line 136 at the time of writing · cps `0.8481` · date `2026-07-18`

[M4] first quadrotor_3d row; pre-JT LQR+HardNet(V_hat). M3 authority gate PASS (‖L_gV̂‖ med 9.32, degen_frac 0); M4 triviality gate PASS (coll 0.020 ≥ 0.005). cps CI [0.824,0.872]. Not SOTA (no cross-dim claim).

<a id="v2-7-2--quadrotor-3d--unlabeled--2"></a>
## v2.7.2 · quadrotor_3d · -

`anchor: v2-7-2--quadrotor-3d--unlabeled--2` · ledger line 137 at the time of writing · cps `0.9329` · date `2026-07-19`

[M6] JT co-trained policy; H1 CONFIRMED — cps 0.9329 [0.918,0.949] CI-sep above pre-JT 0.8481 [0.824,0.872], collision halved 0.020→0.0095. P3 insertion: live-filter coll 0.032 vs nominal-only 0.414. Not SOTA (no cross-dim claim). lineage headline (bring-up). SUPERSEDED (v2.8.0, §2.4/§6.3): trained on the pre-per-rotor wrench action box that the v2.7.3 per-rotor plant replaced; the checkpoint does not load on the current plant, so these numbers describe a vehicle that cannot be built and are not comparable to any current-plant result. Released from bold; row retained for provenance.

### relocated from the `verdict` column (pre-cap text, verbatim)

[M6] JT co-trained policy on the pre-per-rotor wrench action box -- the lineage bring-up headline. H1 CONFIRMED: cps 0.9329 [0.918, 0.949], CI-separated above the pre-JT 0.8481, with collision halved 0.020 -> 0.0095. NOT comparable to any current-plant result, since the v2.7.3 per-rotor plant replaced that action box and the checkpoint does not load on it, so these numbers describe a vehicle that cannot be built; no cross-dimension claim was made. SUPERSEDED by v2.8.0 (2.4 / 6.3) and RELEASED FROM BOLD, the row retained for provenance. Detail docs/ledger_verdicts.md#v2-7-2--quadrotor-3d--unlabeled--2.

<a id="v2-7-2--quadrotor-3d--unlabeled--3"></a>
## v2.7.2 · quadrotor_3d · -

`anchor: v2-7-2--quadrotor-3d--unlabeled--3` · ledger line 138 at the time of writing · cps `0.9586` · date `2026-07-19`

[Stage-3D] k=5 fallback SCALING MEASUREMENT — deployed default REMAINS mode=none (NOT adoption). collision 0.0095→0.0025 [0.0005,0.005], within-seed CI marginally sep vs none [0.0055,0.0135] (one-scene gap); cps CI-OVERLAP (0.9586 [0.946,0.970] vs M6 0.9329 [0.918,0.949]). 15 fixed/1 new coll (all on empty-branch episodes); wall 21.2x. Report-only; no verdict.

<a id="v2-7-3--quadrotor-3d--unlabeled--1"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--1` · ledger line 139 at the time of writing · cps `0.7369` · date `2026-07-19`

[M3 value — coverage row] oc V̂ value-only run, d2 pool; in-loop best_cps 0.7369 @31500; inloop_n500 ineligible for bold; not cross-pool comparable.

<a id="v2-7-3--quadrotor-3d--unlabeled--2"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--2` · ledger line 140 at the time of writing · cps `0.6794` · date `2026-07-20`

[M4] per-rotor plant + d2 pool pre-JT (LQR+HardNet); triviality gate PASS; not SOTA; not comparable to v2.7.2.

<a id="v2-7-3--quadrotor-3d--unlabeled--3"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--3` · ledger line 141 at the time of writing · cps `0.8573` · date `2026-07-20`

[M6] d2-pool JT canonical, cps CI-separated above M4 pre-JT; not SOTA (single-seed); not comparable to v2.7.2.

<a id="v2-7-3--quadrotor-3d--unlabeled--4"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--4` · ledger line 142 at the time of writing · cps `0.8573` · date `2026-07-20`

[M7 condition none] fallback-trial baseline, reproduces M6 canonical bit-identically; not SOTA; not comparable to v2.7.2.

<a id="v2-7-3--quadrotor-3d--unlabeled--5"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--5` · ledger line 143 at the time of writing · cps `0.9126` · date `2026-07-20`

[M7 condition kstep k5] eval-only k-step empty-branch fallback, adoption rule met (3 clauses); ADOPT deferred to Researcher; not comparable to v2.7.2.

<a id="v2-7-4--quadrotor-3d--unlabeled--1"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--1` · ledger line 144 at the time of writing · cps `0.7549` · date `2026-07-20`

[M3 value — coverage row, EXPLORATORY] oc V̂ value-only run, d2r/SO(3); in-loop best_cps 0.7549 @25500; inloop_n500 ineligible for bold; not cross-pool comparable.

<a id="v2-7-4--quadrotor-3d--unlabeled--2"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--2` · ledger line 145 at the time of writing · cps `0.6478` · date `2026-07-20`

[M4 EXPLORATORY] full-SO(3) IC axis pre-JT (item d, G1 fallback), d2r pool; not SOTA; not comparable to d2/v2.7.3.

<a id="v2-7-4--quadrotor-3d--unlabeled--3"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--3` · ledger line 146 at the time of writing · cps `0.8508` · date `2026-07-20`

[M6 EXPLORATORY] full-SO(3) IC axis JT canonical (B0), cps CI-separated above M4 pre-JT; not SOTA (single-seed); not comparable to d2/v2.7.3.

<a id="v2-7-4--quadrotor-3d--unlabeled--4"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--4` · ledger line 147 at the time of writing · cps `0.8508` · date `2026-07-21`

[M7 condition none EXPLORATORY] SO(3) fallback-trial baseline, reproduces M6 headline bit-identically; not comparable to d2/v2.7.3.

<a id="v2-7-4--quadrotor-3d--unlabeled--5"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--5` · ledger line 148 at the time of writing · cps `0.8940` · date `2026-07-21`

[M7 condition kstep k5 EXPLORATORY] SO(3) k-step empty-branch fallback, adoption rule met (collision separation marginal); ADOPT deferred to Strategist; not comparable to d2/v2.7.3.

<a id="v2-7-5--quadrotor-3d--unlabeled--1"></a>
## v2.7.5 · quadrotor_3d · -

`anchor: v2-7-5--quadrotor-3d--unlabeled--1` · ledger line 149 at the time of writing · cps `0.8508` · date `2026-07-23`

[v2.7.5 M1 condition A — deploy-axis, EXPLORATORY] 20 Hz coarse baseline; reproduces the registered v2.7.4 cps bit-identically (delta 0.0), the harness gate for conditions B/C; eval_only, NOT SOTA-bolded (06_workflow §2.4) — Researcher classifies.

<a id="v2-7-5--quadrotor-3d--unlabeled--2"></a>
## v2.7.5 · quadrotor_3d · -

`anchor: v2-7-5--quadrotor-3d--unlabeled--2` · ledger line 150 at the time of writing · cps `0.8477` · date `2026-07-23`

[v2.7.5 M2 condition B — deploy-axis, EXPLORATORY] 20 Hz control on the fine dt=0.01 grid, the integration-grid comparator for C; cps CI-overlaps condition A (grid change alone is within noise); eval_only, NOT SOTA-bolded — Researcher classifies.

<a id="v2-7-5--quadrotor-3d--unlabeled--3"></a>
## v2.7.5 · quadrotor_3d · -

`anchor: v2-7-5--quadrotor-3d--unlabeled--3` · ledger line 151 at the time of writing · cps `0.8801` · date `2026-07-23`

[v2.7.5 M2 condition C — deploy-axis, EXPLORATORY] 100 Hz control; cps +0.0324 over condition B but CI-OVERLAPPING (H1 not met); collision 0.025->0.016; command TV/s +40% median (filter-induced, u_nom TV/s fell) so H3 fails -> reported as a trade; latency p95 6.23ms<10ms (H4 met); eval_only, NOT SOTA-bolded — Researcher classifies.

<a id="v2-7-6--quadrotor-3d--unlabeled--1"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--1` · ledger line 152 at the time of writing · cps `0.6502` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--2"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--2` · ledger line 153 at the time of writing · cps `0.6449` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--3"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--3` · ledger line 154 at the time of writing · cps `0.8603` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--4"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--4` · ledger line 155 at the time of writing · cps `0.8548` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--5"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--5` · ledger line 156 at the time of writing · cps `0.7986` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--6"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--6` · ledger line 157 at the time of writing · cps `0.6810` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--7"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--7` · ledger line 158 at the time of writing · cps `0.8596` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--8"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--8` · ledger line 159 at the time of writing · cps `0.6283` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--9"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--9` · ledger line 160 at the time of writing · cps `0.8580` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--10"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--10` · ledger line 161 at the time of writing · cps `0.4850` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--11"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--11` · ledger line 162 at the time of writing · cps `0.4053` · date `2026-07-25`

[v2.7.6 SUPERSEDED legacy] SUPERSEDED: dim-32 obs made the band branch unobservable; a JT run was spent before the obs fix; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--12"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--12` · ledger line 163 at the time of writing · cps `0.0638` · date `2026-07-25`

[v2.7.6 SUPERSEDED banded] SUPERSEDED: dim-32 obs made the band branch unobservable; a JT run was spent before the obs fix; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--13"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--13` · ledger line 164 at the time of writing · cps `0.3537` · date `2026-07-25`

[v2.7.6 SUPERSEDED legacy] SUPERSEDED: dim-32 obs made the band branch unobservable; a JT run was spent before the obs fix; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--14"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--14` · ledger line 165 at the time of writing · cps `-0.1758` · date `2026-07-25`

[v2.7.6 SUPERSEDED banded] SUPERSEDED: dim-32 obs made the band branch unobservable; a JT run was spent before the obs fix; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--15"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--15` · ledger line 166 at the time of writing · cps `0.8131` · date `2026-07-25`

[v2.7.6 M7 condition jt_kstep banded EXPLORATORY] eval_only; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--16"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--16` · ledger line 167 at the time of writing · cps `0.9166` · date `2026-07-25`

[v2.7.6 M7 condition jt_kstep legacy EXPLORATORY] eval_only; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--17"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--17` · ledger line 168 at the time of writing · cps `0.9085` · date `2026-07-25`

[v2.7.6 M7 condition jt_kstep legacy EXPLORATORY] eval_only; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--18"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--18` · ledger line 169 at the time of writing · cps `0.5257` · date `2026-07-25`

[v2.7.6 M7 condition v274_kstep banded EXPLORATORY] eval_only; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--19"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--19` · ledger line 170 at the time of writing · cps `0.8990` · date `2026-07-25`

[v2.7.6 M7 condition v274_kstep legacy EXPLORATORY] eval_only; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--20"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--20` · ledger line 171 at the time of writing · cps `0.5165` · date `2026-07-25`

[v2.7.6 eval_only legacy] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--21"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--21` · ledger line 172 at the time of writing · cps `0.4227` · date `2026-07-25`

[v2.7.6 eval_only banded] pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--22"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--22` · ledger line 173 at the time of writing · cps `0.9166` · date `2026-07-25`

[v2.7.6 M7 condition jt_kstep banded EXPLORATORY] eval_only; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--23"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--23` · ledger line 174 at the time of writing · cps `0.7271` · date `2026-07-25`

[v2.7.6 M7 condition v274_kstep banded EXPLORATORY] eval_only; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--24"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--24` · ledger line 175 at the time of writing · cps `0.8957` · date `2026-07-25`

[v2.7.6 M7 condition v274_kstep legacy EXPLORATORY] eval_only; pool eval_bandfeasible_d2r_n2000_seed42 (185659a7); empty_fallback=kstep k=5; non-canonical pool, no cross-pool comparison; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--25"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--25` · ledger line 176 at the time of writing · cps `0.6874` · date `2026-07-26`

[v2.7.6 M8 sweep none banded EXPLORATORY] eval_only; cand_evals 0; pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--26"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--26` · ledger line 177 at the time of writing · cps `0.7771` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k1 banded SELECTED EXPLORATORY] eval_only; cand_evals 25; empty-p95 42.3162ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--27"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--27` · ledger line 178 at the time of writing · cps `0.7934` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k2 banded EXPLORATORY] eval_only; cand_evals 50; empty-p95 42.4097ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--28"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--28` · ledger line 179 at the time of writing · cps `0.8027` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k3 banded EXPLORATORY] eval_only; cand_evals 75; empty-p95 44.1891ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--29"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--29` · ledger line 180 at the time of writing · cps `0.7817` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k4 banded EXPLORATORY] eval_only; cand_evals 100; empty-p95 45.2268ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--30"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--30` · ledger line 181 at the time of writing · cps `0.7756` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k5 banded EXPLORATORY] eval_only; cand_evals 125; empty-p95 43.1477ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--31"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--31` · ledger line 182 at the time of writing · cps `0.7771` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k1 banded EXPLORATORY] eval_only; cand_evals 50; empty-p95 55.8463ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--32"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--32` · ledger line 183 at the time of writing · cps `0.7999` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k2 banded EXPLORATORY] eval_only; cand_evals 100; empty-p95 60.6568ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--33"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--33` · ledger line 184 at the time of writing · cps `0.8014` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k3 banded EXPLORATORY] eval_only; cand_evals 150; empty-p95 76.5575ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--34"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--34` · ledger line 185 at the time of writing · cps `0.8010` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k4 banded EXPLORATORY] eval_only; cand_evals 200; empty-p95 59.3796ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--35"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--35` · ledger line 186 at the time of writing · cps `0.8064` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k5 banded EXPLORATORY] eval_only; cand_evals 250; empty-p95 62.3847ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--36"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--36` · ledger line 187 at the time of writing · cps `0.8525` · date `2026-07-26`

[v2.7.6 M8 sweep p1_k1 legacy EXPLORATORY] eval_only; cand_evals 25; empty-p95 42.3162ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--37"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--37` · ledger line 188 at the time of writing · cps `0.9017` · date `2026-07-26`

[v2.7.6 M8 sweep p2_k5 legacy EXPLORATORY] eval_only; cand_evals 250; empty-p95 62.3847ms (H4 FAIL) pool eval_fullrange_d2r_n2000_seed42 (db0b9eb5); GPU eval (device shift ~0.006 vs CPU M5/M7, within CI); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--38"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--38` · ledger line 189 at the time of writing · cps `0.8029` · date `2026-07-26`

[v2.7.6 CANONICAL jt42000 legacy none EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=legacy; empty_fallback=none; device GPU; parent JT step 42000 (dim-34 value-init OC(a) b2cdaddd -> JT best@42000); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--39"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--39` · ledger line 190 at the time of writing · cps `0.9078` · date `2026-07-26`

[v2.7.6 CANONICAL jt42000 legacy kstep k=5 EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=legacy; empty_fallback=kstep k=5; device GPU; parent JT step 42000 (dim-34 value-init OC(a) b2cdaddd -> JT best@42000); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--40"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--40` · ledger line 191 at the time of writing · cps `0.8486` · date `2026-07-26`

[v2.7.6 CANONICAL v274 legacy none EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=legacy; empty_fallback=none; device GPU; parent v2.7.4 comparator best.pt@39000 band-blind parent; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--41"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--41` · ledger line 192 at the time of writing · cps `0.8967` · date `2026-07-26`

[v2.7.6 CANONICAL v274 legacy kstep k=5 EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=legacy; empty_fallback=kstep k=5; device GPU; parent v2.7.4 comparator best.pt@39000 band-blind parent; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--42"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--42` · ledger line 193 at the time of writing · cps `0.6929` · date `2026-07-26`

[v2.7.6 CANONICAL jt42000 banded none EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=banded; empty_fallback=none; device GPU; parent JT step 42000 (dim-34 value-init OC(a) b2cdaddd -> JT best@42000); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--43"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--43` · ledger line 194 at the time of writing · cps `0.8051` · date `2026-07-26`

[v2.7.6 CANONICAL jt42000 banded kstep k=5 EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=banded; empty_fallback=kstep k=5; device GPU; parent JT step 42000 (dim-34 value-init OC(a) b2cdaddd -> JT best@42000); unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--44"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--44` · ledger line 195 at the time of writing · cps `0.4596` · date `2026-07-26`

[v2.7.6 CANONICAL v274 banded none EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=banded; empty_fallback=none; device GPU; parent v2.7.4 comparator best.pt@39000 band-blind parent; unbolded.

<a id="v2-7-6--quadrotor-3d--unlabeled--45"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--45` · ledger line 196 at the time of writing · cps `0.5445` · date `2026-07-26`

[v2.7.6 CANONICAL v274 banded kstep k=5 EXPLORATORY] eval_only; pool eval_full_quadrotor-3d-d2r_n2000_seed23456 (0ef3751b); scoring=banded; empty_fallback=kstep k=5; device GPU; parent v2.7.4 comparator best.pt@39000 band-blind parent; unbolded.

<a id="v2-7-7--quadrotor-3d--unlabeled--1"></a>
## v2.7.7 · quadrotor_3d · -

`anchor: v2-7-7--quadrotor-3d--unlabeled--1` · ledger line 197 at the time of writing · cps `` · date `2026-07-29`

EXPLORATORY; eval-only re-scoring of the v2.7.6 checkpoint; no new policy; unbolded

<a id="v2-7-7--quadrotor-3d--unlabeled--2"></a>
## v2.7.7 · quadrotor_3d · -

`anchor: v2-7-7--quadrotor-3d--unlabeled--2` · ledger line 198 at the time of writing · cps `0.8051` · date `2026-07-29`

EXPLORATORY; eval-only re-scoring of the v2.7.6 checkpoint; no new policy; unbolded

<a id="v2-7-7--quadrotor-3d--unlabeled--3"></a>
## v2.7.7 · quadrotor_3d · -

`anchor: v2-7-7--quadrotor-3d--unlabeled--3` · ledger line 199 at the time of writing · cps `0.8793` · date `2026-07-29`

EXPLORATORY; eval-only re-scoring of the v2.7.6 checkpoint; no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--1"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--1` · ledger line 200 at the time of writing · cps `0.8051` · date `2026-07-30`

EXPLORATORY; Stage-0 B0 rollout on the v2.7.6 headline checkpoint (reproduces recorded canonical banded kstep-k5 row); B0 falsifier did not fire (median ‖omega‖ at reach 1.34 rad/s); no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--2"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--2` · ledger line 201 at the time of writing · cps `0.8051` · date `2026-07-30`

EXPLORATORY; S1 projection axis, enumerate condition (legacy finite-candidate selection); reproduces recorded v2.7.6 row; no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--3"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--3` · ledger line 202 at the time of writing · cps `0.8069` · date `2026-07-30`

EXPLORATORY; S1 projection axis, dual_solve condition (exact prop:lambda-solve); infeasibility CI-overlaps enumerate (A3 null holds); no cps prediction registered; no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--4"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--4` · ledger line 203 at the time of writing · cps `-0.6670` · date `2026-07-30`

EXPLORATORY; S2 T1 comparator, NOT a performance statement — cps dominated by a reclassification artifact (old-terminal checkpoint re-scored under the new terminal). T1 stuck prediction MET: stuck 0.0000->0.7295 (>0.10); reach collapses 0.9330->0.1935 (arrivals with residual spin hold position). no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--5"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--5` · ledger line 204 at the time of writing · cps `-0.6872` · date `2026-07-30`

EXPLORATORY; S2 C2 shipped-fallback comparison — {phases2,k5} vs the shipped {phases1,k3} (row above): cps -0.6872 vs -0.6670 under the new terminal. reclassification artifact, not a performance statement. no new policy; unbolded

<a id="v2-8-0--quadrotor-planar--unlabeled"></a>
## v2.8.0 · quadrotor_planar · -

`anchor: v2-8-0--quadrotor-planar--unlabeled` · ledger line 205 at the time of writing · cps `0.9020` · date `2026-07-30`

EXPLORATORY; S2 T1 planar — angular condition nearly inert here (planar arrivals already settled, A2 median ‖omega‖@reach 0.28): reach 0.9665->0.9650, stuck delta ~0. No planar threshold was registered. no new policy; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--6"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--6` · ledger line 206 at the time of writing · cps `0.7919` · date `2026-07-31`

EXPLORATORY; S3 deliverable. T2 MET (reach 0.9295 >= 0.8875, within 0.05 of v2.7.6 old-terminal 0.9375). collision decomposed obstacle 0.0095/floor 0.0335/ceiling 0.0010 (floor-dominated; 0 recoverable-class floor collisions per floor_feasibility). single seed; new terminal => no cross-pool/old-row comparison; unbolded. Detail: docs/versions/v2.8.0/s3_retrain.md

<a id="v2-8-0--quadrotor-3d--unlabeled--7"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--7` · ledger line 207 at the time of writing · cps `0.7778` · date `2026-07-31`

EXPLORATORY; S4 enumerate condition vs S3. P3 PAIRED cps Delta(dual-enum) +0.0141 full / -0.0102 hold-feasible (both CI-cross-0, sign flips; no resolvable axis effect); collision_obstacle paired +0.0042 hold-feasible [0.000,0.010]. CONFOUNDED: enum best is pre-crash (cold-buffer resume never regained 0.6794), less-trained than dual step42000. floor 0.0345 (1 recoverable-class). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--8"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--8` · ledger line 208 at the time of writing · cps `0.7906` · date `2026-07-31`

EXPLORATORY; M5 clean deployment-channel axis (unconfounded by training duration): swapping deployed projection dual_solve->enumerate moves cps -0.0013 vs M4 dual 0.7919, reach/collision identical -- matches S1 ~0.002 deployment effect. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--9"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--9` · ledger line 209 at the time of writing · cps `0.7745` · date `2026-07-31`

EXPLORATORY; P3 condition comparison at the LAST COMMON pre-crash checkpoint (step 34500; the resume began here) -- matched budget/conditions/provenance, neither condition resumed. The 50000-step comparison is comparison-unavailable (06_workflow §5; enum endpoint is a post-crash cold-buffer resume). NOT the dual deliverable (that is step42000 M4 0.7919, where T2 is scored). obstacle 0.0095/floor 0.0335/ceiling 0.0005. single seed; unbolded. Detail: docs/versions/v2.8.0/p3_common_step.md

### relocated from the `verdict` column (pre-cap text, verbatim)

EXPLORATORY; P3 condition comparison at the LAST COMMON pre-crash checkpoint (step 34500; the resume began here) -- matched budget/conditions/provenance, neither condition resumed. The 50000-step comparison is comparison-unavailable (06_workflow §5; enum endpoint is a post-crash cold-buffer resume). NOT the dual deliverable (that is step42000 M4 0.7919, where T2 is scored). obstacle 0.0095/floor 0.0335/ceiling 0.0005. single seed; unbolded. Detail: docs/versions/v2.8.0/p3_common_step.md

<a id="v2-8-0--quadrotor-3d--unlabeled--10"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--10` · ledger line 210 at the time of writing · cps `0.7810` · date `2026-07-31`

EXPLORATORY; P3 condition comparison at the common step 34500. PAIRED cps Delta(dual-enum) hold-feasible -0.0321 [-0.0645,-0.0028] CI-SEPARATED (enumerate condition marginally AHEAD on hold-feasible cps at matched budget); full-pool -0.0065 (CI crosses 0). No cps prediction was registered for the axis; the exact projection did not out-train the enumeration here. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--11"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--11` · ledger line 211 at the time of writing · cps `0.7910` · date `2026-07-31`

EXPLORATORY; T2 matched-fallback 2x2 (rate_and_branch.md C). reach 0.9305; at matched p2k5 the settled terminal costs -0.0070 reach vs 09c33bf4 old 0.9375. T2 registered score unchanged. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--12"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--12` · ledger line 212 at the time of writing · cps `0.8032` · date `2026-07-31`

EXPLORATORY; T2 matched-fallback 2x2 (rate_and_branch.md C). reach 0.9350; at matched p1k3 the settled terminal costs -0.0055 vs dual new 0.9295. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--13"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--13` · ledger line 213 at the time of writing · cps `0.7906` · date `2026-07-31`

EXPLORATORY; rate x projection (rate_and_branch.md B). condition A enum = 20Hz reproduction gate = M5. TV/s med 32.55. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--14"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--14` · ledger line 214 at the time of writing · cps `0.7919` · date `2026-07-31`

EXPLORATORY; rate x projection. condition A dual = 20Hz reproduction gate = M4. TV/s med 25.15. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--15"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--15` · ledger line 215 at the time of writing · cps `0.7883` · date `2026-07-31`

EXPLORATORY; rate x projection (integration-grid control). TV/s med 32.67 (~=condition A). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--16"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--16` · ledger line 216 at the time of writing · cps `0.7952` · date `2026-07-31`

EXPLORATORY; rate x projection (integration-grid control). TV/s med 25.05 (~=condition A). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--17"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--17` · ledger line 217 at the time of writing · cps `0.7860` · date `2026-07-31`

EXPLORATORY; rate x projection. TV/s med 63.17 (enum grows with rate). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--18"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--18` · ledger line 218 at the time of writing · cps `0.7830` · date `2026-07-31`

EXPLORATORY; rate x projection. TV/s med 27.98 (dual saturates). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--19"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--19` · ledger line 219 at the time of writing · cps `0.7842` · date `2026-07-31`

EXPLORATORY; rate x projection. TV/s med 179.77 (enum chatters; B1 slope 0.531). single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--20"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--20` · ledger line 220 at the time of writing · cps `0.7750` · date `2026-07-31`

EXPLORATORY; rate x projection. TV/s med 37.43 (dual saturates; B1 slope 0.124<0.25 MET). TV empty-branch frac 0.51 -> empty branch is the residual high-rate obstacle. single seed; unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--21"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--21` · ledger line 221 at the time of writing · cps `0.5409` · date `2026-08-01`

comparable (own/old terminal). gate collision 0.1425 terminal-independent (new-terminal gate 0.1430, <=1 ep). No recorded old-terminal canonical row at shipped fallback; gate fresh. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--22"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--22` · ledger line 222 at the time of writing · cps `-0.9298` · date `2026-08-01`

terminal-confounded for cross-checkpoint comparison — angular settling untrained; comparable row above. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--23"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--23` · ledger line 223 at the time of writing · cps `0.1393` · date `2026-08-01`

comparable (own/old terminal). old==new (all reaches already satisfy omega<=0.48, angular condition inert). gate collision 0.2345 terminal-independent. No recorded canonical row; fresh. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--24"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--24` · ledger line 224 at the time of writing · cps `0.1393` · date `2026-08-01`

terminal-confounded for cross-checkpoint comparison — angular settling untrained; comparable row above. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--25"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--25` · ledger line 225 at the time of writing · cps `0.5418` · date `2026-08-01`

comparable (own/old terminal). gate collision 0.1375 terminal-independent (new 0.1380, <=1 ep). No recorded old-terminal canonical row; gate fresh. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--26"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--26` · ledger line 226 at the time of writing · cps `-0.9737` · date `2026-08-01`

terminal-confounded for cross-checkpoint comparison — angular settling untrained; comparable row above. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--27"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--27` · ledger line 227 at the time of writing · cps `0.7987` · date `2026-08-01`

comparable (own/old terminal). gate collision 0.0435 is terminal-independent (matches new-terminal gate). No old-terminal canonical row at shipped fallback phases1 k3; recorded old-terminal canonical (banded kstep k5, coll 0.0425) reproduced bit-for-bit by the M1 gate (docs/ledger.md L186, delta 0.0). cps_v2 here=gate; CIs in dual_scoring.md. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--28"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--28` · ledger line 228 at the time of writing · cps `-0.6670` · date `2026-08-01`

terminal-confounded for cross-checkpoint comparison — angular settling untrained; comparable row above. unbolded

<a id="v2-8-0--quadrotor-3d--unlabeled--29"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--29` · ledger line 229 at the time of writing · cps `0.7919` · date `2026-08-01`

quadrotor_3d SOTA — first eligible per-rotor-plant baseline (no prior 3-D bold; v2.7.2 released as plant-incompatible). Dual condition best.pt@42000, secured cf948104 (data/secured_data/v2.8.0/seed42/). cps_v2=gate; cps_tilt60/cps_bandopen the two definitions; new terminal is the training terminal.

<a id="v2-8-0--quadrotor-3d--unlabeled--30"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--30` · ledger line 230 at the time of writing · cps `0.7778` · date `2026-08-01`

new terminal is the training terminal; no old-terminal reading exists for this condition by construction. cps_v2=gate; cps_tilt60/cps_bandopen the two definitions. unbolded

<a id="v2-8-1--quadrotor-3d--unlabeled--1"></a>
## v2.8.1 · quadrotor_3d · -

`anchor: v2-8-1--quadrotor-3d--unlabeled--1` · ledger line 231 at the time of writing · cps `0.6496` · date `2026-08-02`

LINEAGE BREAK — soft_topk encoder beta=30 (obs channel changed) + terminal 0.30, value-init from v2.8.0 with policy fresh, n_steps 50000; NOT SOTA-comparable to hard-encoder v2.8.0 (0.7919). **T2 FAILED: gate reach 0.8620 < 0.8875** (terminal NOT relaxed per dispatch; pipeline STOPPED; nothing secured). gate coll 0.0660 (obstacle 0.0320 / band_lower 0.0335 / band_upper 0.0005; cf. v2.8.0 gate 0.0440). D3 ‖Δu‖ crossing ratio: 20Hz deployment 2.34x >= hard 1.84x (NO gain at shipped rate) but 500Hz fine-step 1.54x vs hard 32.27x (21x reduction — encoder mechanism confirmed where isolable). Detail: docs/versions/v2.8.1/s1_retrain.md. single seed; unbolded

### relocated from the `verdict` column (pre-cap text, verbatim)

LINEAGE BREAK — soft_topk encoder beta=30 (obs channel changed) + terminal 0.30, value-init from v2.8.0 with policy fresh, n_steps 50000; NOT SOTA-comparable to hard-encoder v2.8.0 (0.7919). **T2 FAILED: gate reach 0.8620 < 0.8875** (terminal NOT relaxed per dispatch; pipeline STOPPED; nothing secured). gate coll 0.0660 (obstacle 0.0320 / band_lower 0.0335 / band_upper 0.0005; cf. v2.8.0 gate 0.0440). D3 ‖Δu‖ crossing ratio: 20Hz deployment 2.34x >= hard 1.84x (NO gain at shipped rate) but 500Hz fine-step 1.54x vs hard 32.27x (21x reduction — encoder mechanism confirmed where isolable). Detail: docs/versions/v2.8.1/s1_retrain.md. single seed; unbolded

<a id="v2-8-2--quadrotor-3d--unlabeled--1"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--1` · ledger line 232 at the time of writing · cps `0.7785` · date `2026-08-03`

CONTROL (v2.8.2), UNBOLDED — v2.8.0 JT recipe, hard_topk, terminal omega_G=0.30, shared v2.7.6 value-init, policy fresh, n_steps 30000, best.pt@step24000; secured data/secured_data/v2.8.2/seed42/ (best.pt SHA c89f9aef…8e285) as a control/attribution snapshot, NOT a SOTA claim. NOT bold: gate 0.7785 < standing 3-D bold 0.7919, and neither step-matched (30000 vs 50000) nor terminal-matched (0.30 vs 0.48) to it. **T2 PASSES: gate reach 0.9205 >= 0.8875** — v2.8.1 failed T2 at reach 0.8620 under the SAME terminal (0.30) with the soft encoder, so that failure attributes to the ENCODER not the terminal; closes open discrepancy (2) of v2.8.1_results.md. gate collision 0.0455 (obstacle 0.0115 / band_lower 0.0335 / band_upper 0.0005). The run's auto-final eval read cps 0.6667 / reach 0.8925 because it used the training-time fallback {none,k10}; the dual three-cell at the shipped {kstep,phases1,k3} is authoritative. The omega_G=0.48 re-score (DIAGNOSTIC, NOT bold-eligible — CTRL trained at 0.30, bold judged at the training terminal): cps 0.7795 [0.747,0.812], reach 0.9210, coll 0.0455 — the terminal contributes ~0 (0.7785@0.30 -> 0.7795@0.48, +0.0010), so the -0.0124 gap to the standing bold 0.7919 (point-value below the +/-0.005 band; CI overlaps 0.7919) is attributable to the reduced budget (30000 vs 50000, batch 4096) and single seed, NOT the terminal; does not displace 0.7785 in this row. Detail: docs/versions/v2.8.2/s2_conditions.md (S2 conditions, dual three-cell, T2, E1/E2, α_unsafe axis). single seed

### relocated from the `verdict` column (pre-cap text, verbatim)

CONTROL for v2.8.2: the v2.8.0 JT recipe with hard_topk at terminal omega_G 0.30, shared v2.7.6 value-init and a fresh policy, best.pt at step 24000, secured as a control and attribution snapshot. **T2 PASSES at gate reach 0.9205 >= 0.8875**, which attributes v2.8.1's T2 failure to the ENCODER and not the terminal, closing open discrepancy (2) of v2.8.1_results.md. NOT comparable to the standing 3-D bold 0.7919: neither step-matched (30000 against 50000) nor terminal-matched (0.30 against 0.48), and the run's own auto-final eval used the training-time fallback rather than the shipped one, so the dual three-cell is authoritative; single seed. UNBOLDED, no promotion, NOT a SOTA claim. Detail docs/versions/v2.8.2/s2_conditions.md and docs/ledger_verdicts.md#v2-8-2--quadrotor-3d--unlabeled--1.

<a id="v2-8-2--quadrotor-3d--unlabeled--2"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--2` · ledger line 233 at the time of writing · cps `0.7653` · date `2026-08-04`

α_unsafe axis (E1/E2), UNBOLDED — CTRL recipe, only delta filter.alpha_unsafe 100→30. gate cps 0.7653 [0.732,0.798] ≈ CTRL 0.7785 (CI overlaps); infeas 0.1106 ≈ CTRL 0.1107 — the α=30 rung is INERT on infeasibility (reduction appears only at 10/3). collision 0.0460 ≈ CTRL 0.0455 (obstacle 0.0135/band_lower 0.0320/band_upper 0.0005). best.pt@22500, n_steps 30000; NOT secured (axis snapshot). single seed. Detail: docs/versions/v2.8.2/s2_conditions.md

### relocated from the `verdict` column (pre-cap text, verbatim)

α_unsafe axis (E1/E2), UNBOLDED — CTRL recipe, only delta filter.alpha_unsafe 100→30. gate cps 0.7653 [0.732,0.798] ≈ CTRL 0.7785 (CI overlaps); infeas 0.1106 ≈ CTRL 0.1107 — the α=30 rung is INERT on infeasibility (reduction appears only at 10/3). collision 0.0460 ≈ CTRL 0.0455 (obstacle 0.0135/band_lower 0.0320/band_upper 0.0005). best.pt@22500, n_steps 30000; NOT secured (axis snapshot). single seed. Detail: docs/versions/v2.8.2/s2_conditions.md

<a id="v2-8-2--quadrotor-3d--unlabeled--3"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--3` · ledger line 234 at the time of writing · cps `0.7728` · date `2026-08-04`

α_unsafe axis (E1/E2), UNBOLDED — CTRL recipe, only delta filter.alpha_unsafe 100→10. gate infeas 0.0980 [0.089,0.107] < CTRL 0.1107 (−0.0127) at near-CTRL cps 0.7728 (CI overlaps CTRL) and small collision cost 0.0490 vs 0.0455 — the FAVORABLE rung (mild infeasibility reduction, cps ~preserved). best.pt@27000, n_steps 30000. single seed. Detail: docs/versions/v2.8.2/s2_conditions.md

<a id="v2-8-2--quadrotor-3d--unlabeled--4"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--4` · ledger line 235 at the time of writing · cps `0.7609` · date `2026-08-04`

α_unsafe axis (E1/E2), UNBOLDED — CTRL recipe, only delta filter.alpha_unsafe 100→3. gate infeas 0.0802 [0.072,0.089] < CTRL 0.1107 (−0.0305, largest; monotonic α 100/30/10/3 → infeas 0.111/0.111/0.098/0.080) BUT collision RISES 0.0570 vs 0.0455 (+0.0115; obstacle 0.0220 vs 0.0115) and cps 0.7609 < CTRL 0.7785. E1/E2 VERDICT: relaxing α_unsafe TRADES infeasibility for collision — not a free win; CTRL(α=100) retains best gate cps/reach. best.pt@21000, n_steps 30000. single seed. Detail: docs/versions/v2.8.2/s2_conditions.md

### relocated from the `verdict` column (pre-cap text, verbatim)

α_unsafe axis (E1/E2), UNBOLDED — CTRL recipe, only delta filter.alpha_unsafe 100→3. gate infeas 0.0802 [0.072,0.089] < CTRL 0.1107 (−0.0305, largest; monotonic α 100/30/10/3 → infeas 0.111/0.111/0.098/0.080) BUT collision RISES 0.0570 vs 0.0455 (+0.0115; obstacle 0.0220 vs 0.0115) and cps 0.7609 < CTRL 0.7785. E1/E2 VERDICT: relaxing α_unsafe TRADES infeasibility for collision — not a free win; CTRL(α=100) retains best gate cps/reach. best.pt@21000, n_steps 30000. single seed. Detail: docs/versions/v2.8.2/s2_conditions.md

<a id="v2-8-2--quadrotor-3d--unlabeled--5"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--5` · ledger line 236 at the time of writing · cps `0.7289` · date `2026-08-04`

STEP-MISMATCHED, NOT converged — best.pt@9000 / n_steps 10499 (stopped early) vs CTRL best@24000 / 30000, so gate cps 0.7289 [0.696,0.763] is an EARLY-checkpoint score NOT comparable to CTRL 0.7785 (and floor 0.0340 ≈ CTRL 0.0335 is the step-9000 policy, not a floor gain). FLOOR axis (FB deployed-fallback co-adaptation), UNBOLDED, STOPPED-ON-DECISION (Researcher: remaining ~20000 steps buy nothing the eval already answers), halt_reason null, dir intact. CONFOUND BROKEN (data/runs/v2.8.2/floor_axis/fb_confound.json — fallback-matched 2×2 on the in-loop pool seed45678, steps 4500/6000/7500/9000): FB's E0 "pass" was the EVAL FALLBACK, not co-adaptation. Controls: at matched fallback FB floor 0.0145 ≈ CTRL@p1k3 0.0150 ≈ CTRL@p2k5 0.0140 (floor co-adapt ≈ 0); CTRL@p2k5 cps 0.789 LANDS ON FB 0.787 by step 9000. Clean training effect (FB@p1k3 − CTRL@p1k3) = +0.018 cps @9000, narrowing from +0.097 @4500 → training-speed, not quality. FB never fires the fallback (FB@p1k3 ≡ FB@p2k5 byte-identical); the fallback is a lever only for CTRL (CTRL floor {none}→p1k3 0.0250→0.0150). FB TRAINED under {kstep,phases2,k5}, SCORED here at shipped {kstep,phases1,k3}. T2 gate reach 0.8980 ≥ 0.8875. single seed. Detail: docs/versions/v2.8.2/fb_floor.md, unattended_report.md §5

### relocated from the `verdict` column (pre-cap text, verbatim)

FLOOR axis (FB deployed-fallback co-adaptation), stopped on Researcher decision at best.pt step 9000 of 10499. CONFOUND BROKEN: at matched fallback the floor gain vanishes (FB 0.0145 against CTRL 0.0150), so FB's E0 pass was the EVAL FALLBACK and not co-adaptation, leaving a clean training effect of +0.018 cps at step 9000 that narrows from +0.097 at 4500 -- training speed, not quality. STEP-MISMATCHED and NOT converged, so cps 0.7289 [0.696, 0.763] is an EARLY-checkpoint score NOT comparable to CTRL's 0.7785; FB trained under {kstep, phases 2, k 5} and is scored here at the shipped {kstep, phases 1, k 3}; single seed. UNBOLDED, no promotion. Detail docs/versions/v2.8.2/fb_floor.md and docs/ledger_verdicts.md#v2-8-2--quadrotor-3d--unlabeled--5.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(3 cells: gate/cps_tilt60/cps_bandopen; v2.8.2 terminal goal_angrate_radius=0.30, shipped fallback kstep phases1 k3, dual_solve; encoder hard_topk; FLOOR condition FB = deployed-fallback co-adaptation, TRAINED empty_fallback quad {kstep,phases2,k5}; best.pt@9000; n_steps 10499 (STOPPED-ON-DECISION); pools canonical 0ef3751b (band-terminate) / navcone e54d8cd6 (tilt<=60) / canonical 0ef3751b (band-open))

<a id="v2-7-3--quadrotor-3d--unlabeled--6"></a>
## v2.7.3 · quadrotor_3d · -

`anchor: v2-7-3--quadrotor-3d--unlabeled--6` · ledger line 237 at the time of writing · cps `0.5933` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=none(old). BAND-BLIND parent scored band-active.

<a id="v2-7-4--quadrotor-3d--unlabeled--6"></a>
## v2.7.4 · quadrotor_3d · -

`anchor: v2-7-4--quadrotor-3d--unlabeled--6` · ledger line 238 at the time of writing · cps `0.5921` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=none(old). BAND-BLIND parent scored band-active.

<a id="v2-7-6--quadrotor-3d--unlabeled--46"></a>
## v2.7.6 · quadrotor_3d · -

`anchor: v2-7-6--quadrotor-3d--unlabeled--46` · ledger line 239 at the time of writing · cps `0.8704` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=none(old).

<a id="v2-8-0--quadrotor-3d--unlabeled--31"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--31` · ledger line 240 at the time of writing · cps `0.8550` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.48.

<a id="v2-8-0--quadrotor-3d--unlabeled--32"></a>
## v2.8.0 · quadrotor_3d · -

`anchor: v2-8-0--quadrotor-3d--unlabeled--32` · ledger line 241 at the time of writing · cps `0.8401` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.48.

<a id="v2-8-1--quadrotor-3d--unlabeled--2"></a>
## v2.8.1 · quadrotor_3d · -

`anchor: v2-8-1--quadrotor-3d--unlabeled--2` · ledger line 242 at the time of writing · cps `0.7084` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--6"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--6` · ledger line 243 at the time of writing · cps `0.8409` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--7"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--7` · ledger line 244 at the time of writing · cps `0.8270` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--8"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--8` · ledger line 245 at the time of writing · cps `0.8347` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--9"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--9` · ledger line 246 at the time of writing · cps `0.8226` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--10"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--10` · ledger line 247 at the time of writing · cps `0.7897` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY (screened pools; not bold-eligible). terminal=0.3.

<a id="v2-8-2--quadrotor-3d--unlabeled--11"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--11` · ledger line 248 at the time of writing · cps `0.8348` · date `2026-08-04`

unbolded, pool-rebuild EXPLORATORY; M1 prox dual-score (single stopped run @27879, not a controlled comparison)

<a id="v2-8-2--quadrotor-3d--unlabeled--12"></a>
## v2.8.2 · quadrotor_3d · -

`anchor: v2-8-2--quadrotor-3d--unlabeled--12` · ledger line 249 at the time of writing · cps `-0.6148` · date `2026-08-04`

PPO BASELINE (no certificate, no filter), UNBOLDED, conditional on shaping. Learns near-complete collision-avoidance (obstacle 0.0455/floor 0.0275/ceil 0.0035) and BEATS the LQR nominal on cps (-0.6148 vs -0.845; converged same-family run -0.5985) but reach~0 (timeout-dominated 0.9225): cannot achieve the tight settling terminal. REWARD (Researcher-registered + 6 addenda): reach+5/coll-5(obstacle+band)/oob,stuck,timeout-1/per-step-1/max_steps; potential k*(d_t-gamma*d_{t+1}) Ng-form FULL 3-D, k=5/E[‖p0-g‖]=0.9264; attitude deadband theta_hold=60deg(cos=1/TWR,TWR=2.0); SMOOTH-distance-weighted settling regulation sigma(d)=exp(-(d/1.0)^2)*(1-exp(-relu(‖.‖/bound-1))) both channels v_G=omega_G=0.30 w_reg=5.0 (accum ~-10 held-in-violation); smoothness w_du=w_d2u=0.5 on sampled u; NO APF/band term. PPO gamma0.99 lambda0.95 clip0.2 lr3e-4 4ep mb16384 2048env approx_kl~0.008. DI gate PASSED (reach 0.826 w/ settling). born-terminal dropped=0. Caveat: smoothness penalized here but JT w_du OFF (asymmetry favors JT); reach terminal heavily gamma-discounted. Detail: docs/versions/v2.8.2/ppo_baseline.md. single seed

### relocated from the `verdict` column (pre-cap text, verbatim)

PPO BASELINE on quadrotor_3d with no certificate and no filter, under a Researcher-registered shaped reward. It learns near-complete collision avoidance (obstacle 0.0455 / floor 0.0275 / ceiling 0.0035) and beats the LQR nominal at cps -0.6148 against -0.845, but reach is ~0 and the run is timeout-dominated at 0.9225, so it cannot achieve the tight settling terminal. Conditional on that shaping and NOT comparable to the certificate rows on equal terms -- smoothness is penalized here while JT's w_du is off, and the reach terminal is heavily gamma-discounted; single seed. UNBOLDED, no promotion. Detail docs/versions/v2.8.2/ppo_baseline.md and docs/ledger_verdicts.md#v2-8-2--quadrotor-3d--unlabeled--12.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(FILTER-FREE 3 cells: gate/cps_tilt60/cps_bandopen; PPO BASELINE — NO certificate, NO filter: policy -> action box -> plant -> outcomes, infeasibility=0 by construction; pools fullscr40 588a2724/navconescr40 ecdf2d36; terminal goal_angrate_radius=0.30; best.pt@upd50516; budget 213M env-interactions / 54k policy-updates / 2048 envs — STOPPED iter~640 to free GPU for the JT axis, in-loop cps still improving to -0.574; NOT a 30000-step symmetry with JT)

<a id="v2-8-3--quadrotor-3d--r1"></a>
## v2.8.3 · quadrotor_3d · R1

`anchor: v2-8-3--quadrotor-3d--r1` · ledger line 250 at the time of writing · cps `0.8291` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--r2"></a>
## v2.8.3 · quadrotor_3d · R2

`anchor: v2-8-3--quadrotor-3d--r2` · ledger line 251 at the time of writing · cps `0.7545` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--r3"></a>
## v2.8.3 · quadrotor_3d · R3

`anchor: v2-8-3--quadrotor-3d--r3` · ledger line 252 at the time of writing · cps `0.1017` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--r4"></a>
## v2.8.3 · quadrotor_3d · R4

`anchor: v2-8-3--quadrotor-3d--r4` · ledger line 253 at the time of writing · cps `0.5609` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--r5"></a>
## v2.8.3 · quadrotor_3d · R5

`anchor: v2-8-3--quadrotor-3d--r5` · ledger line 254 at the time of writing · cps `0.2096` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--sigma"></a>
## v2.8.3 · quadrotor_3d · SIGMA

`anchor: v2-8-3--quadrotor-3d--sigma` · ledger line 255 at the time of writing · cps `0.8186` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--uprev"></a>
## v2.8.3 · quadrotor_3d · UPREV

`anchor: v2-8-3--quadrotor-3d--uprev` · ledger line 256 at the time of writing · cps `0.8166` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--d2"></a>
## v2.8.3 · quadrotor_3d · D2

`anchor: v2-8-3--quadrotor-3d--d2` · ledger line 257 at the time of writing · cps `0.8194` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--d3"></a>
## v2.8.3 · quadrotor_3d · D3

`anchor: v2-8-3--quadrotor-3d--d3` · ledger line 258 at the time of writing · cps `0.8253` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--trackp"></a>
## v2.8.3 · quadrotor_3d · TrackP

`anchor: v2-8-3--quadrotor-3d--trackp` · ledger line 259 at the time of writing · cps `-0.0036` · date `2026-08-08`

diagnostic; no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(uniform re-score, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, qp_penalty 1e6 on QP rows) [backup Track P + PD]; DIVERGENCE: v2 scored Track P on the v2.7.6 leader at ebs 400 without the angrate terminal — not comparable to the v2 row; artifact data/runs/v2.8.3/pool_v2_rescore/row__backup_trackP_pd.json

<a id="v2-8-3--quadrotor-3d--tracks"></a>
## v2.8.3 · quadrotor_3d · TrackS

`anchor: v2-8-3--quadrotor-3d--tracks` · ledger line 260 at the time of writing · cps `0.1926` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--d1"></a>
## v2.8.3 · quadrotor_3d · D1

`anchor: v2-8-3--quadrotor-3d--d1` · ledger line 261 at the time of writing · cps `0.8291` · date `2026-08-08`

diagnostic; no bold change

<a id="v2-8-3--quadrotor-3d--ppo"></a>
## v2.8.3 · quadrotor_3d · PPO

`anchor: v2-8-3--quadrotor-3d--ppo` · ledger line 262 at the time of writing · cps `0.6770` · date `2026-08-08`

diagnostic; no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(uniform re-score, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, qp_penalty 1e6 on QP rows) [PPO baseline, FILTER-FREE]; DIVERGENCE: filter-free — projection/empty_fallback/alpha/gamma/QP inapplicable, EMPTY+SINGULAR structurally 0; terminal read back and matches; artifact data/runs/v2.8.3/pool_v2_rescore/row__ppo_baseline.json

<a id="v2-8-4--quadrotor-3d--k2--1"></a>
## v2.8.4 · quadrotor_3d · K2

`anchor: v2-8-4--quadrotor-3d--k2--1` · ledger line 263 at the time of writing · cps `0.7301` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K2_soft, pool eval_fullscr41_quadrotor-3d-d2r_n2000_seed723456 9a016919, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode soft beta 20.0, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K2_soft__fullscr41.json

<a id="v2-8-4--quadrotor-3d--k3--1"></a>
## v2.8.4 · quadrotor_3d · K3

`anchor: v2-8-4--quadrotor-3d--k3--1` · ledger line 264 at the time of writing · cps `0.7277` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K3_hard_nofallback, pool eval_fullscr41_quadrotor-3d-d2r_n2000_seed723456 9a016919, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode hard, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K3_hard_nofallback__fullscr41.json

<a id="v2-8-4--quadrotor-3d--k2--2"></a>
## v2.8.4 · quadrotor_3d · K2

`anchor: v2-8-4--quadrotor-3d--k2--2` · ledger line 265 at the time of writing · cps `0.7812` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K2_soft, pool eval_inloopv2_quadrotor-3d-d2r-mixed_n2000_seed145678 50bc2060, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode soft beta 20.0, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K2_soft__inloopv2.json

<a id="v2-8-4--quadrotor-3d--k3--2"></a>
## v2.8.4 · quadrotor_3d · K3

`anchor: v2-8-4--quadrotor-3d--k3--2` · ledger line 266 at the time of writing · cps `0.7731` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K3_hard_nofallback, pool eval_inloopv2_quadrotor-3d-d2r-mixed_n2000_seed145678 50bc2060, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode hard, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K3_hard_nofallback__inloopv2.json

<a id="v2-8-4--quadrotor-3d--k0--1"></a>
## v2.8.4 · quadrotor_3d · K0

`anchor: v2-8-4--quadrotor-3d--k0--1` · ledger line 267 at the time of writing · cps `0.8424` · date `2026-08-08`

eval-only deployment cell, clamp OFF; K1-K3 on this pool flag on clamp-bind, so their infeas and cps are not comparable to this row; see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K0_shipped, pool eval_fullscr41_quadrotor-3d-d2r_n2000_seed723456 9a016919, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp OFF, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K0_shipped__fullscr41.json

<a id="v2-8-4--quadrotor-3d--k0--2"></a>
## v2.8.4 · quadrotor_3d · K0

`anchor: v2-8-4--quadrotor-3d--k0--2` · ledger line 268 at the time of writing · cps `0.8557` · date `2026-08-08`

eval-only deployment cell, clamp OFF; K1-K3 on this pool flag on clamp-bind, so their infeas and cps are not comparable to this row; see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K0_shipped, pool eval_inloopv2_quadrotor-3d-d2r-mixed_n2000_seed145678 50bc2060, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp OFF, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K0_shipped__inloopv2.json

<a id="v2-8-4--quadrotor-3d--k1--1"></a>
## v2.8.4 · quadrotor_3d · K1

`anchor: v2-8-4--quadrotor-3d--k1--1` · ledger line 269 at the time of writing · cps `0.7447` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K1_hard, pool eval_fullscr41_quadrotor-3d-d2r_n2000_seed723456 9a016919, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode hard, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K1_hard__fullscr41.json

<a id="v2-8-4--quadrotor-3d--k1--2"></a>
## v2.8.4 · quadrotor_3d · K1

`anchor: v2-8-4--quadrotor-3d--k1--2` · ledger line 270 at the time of writing · cps `0.7729` · date `2026-08-08`

eval-only deployment cell; infeas and cps not comparable to K0 on this pool (flag on clamp-bind, a superset of shipped-empty, so infeas over-counted and cps under-stated); see docs/versions/v2.8.4/ledger_box_klamp.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(box_klamp deployment cell K1_hard, pool eval_inloopv2_quadrotor-3d-d2r-mixed_n2000_seed145678 50bc2060, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, box_klamp ON kappa 0.8 mode hard, ckpt v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/box_klamp/K1_hard__inloopv2.json

<a id="v2-8-4--quadrotor-3d--ref-c-1-3--1"></a>
## v2.8.4 · quadrotor_3d · REF (C=1.3)

`anchor: v2-8-4--quadrotor-3d--ref-c-1-3--1` · ledger line 271 at the time of writing · cps `0.8308` · date `2026-08-08`

ceiling axis C=1.3, eval-only vs CTRL at matched step 24000; single seed, delta cps below D0's 0.0083 floor so no cps claim; see docs/versions/v2.8.4/ceiling_axis.md; diagnostic, no bold change. SUPERSEDED 2026-08-12 by the v2.8.5 exp0.125 step-45000 row (cps_v2 0.8701, same registered cell, adopted checkpoint step_045000.pt); bold released, and it stays released - the earlier supersession by the v2.8.5 exp0.175 step-33000 row is itself withdrawn, that row being registered but not bold.

### relocated from the `verdict` column (pre-cap text, verbatim)

ceiling axis C=1.3, eval-only vs CTRL at matched step 24000; single seed, delta cps below D0's 0.0083 floor so no cps claim; see docs/versions/v2.8.4/ceiling_axis.md; diagnostic, no bold change. SUPERSEDED 2026-08-12 by the v2.8.5 exp0.125 step-45000 row (cps_v2 0.8701, same registered cell, adopted checkpoint step_045000.pt); bold released, and it stays released - the earlier supersession by the v2.8.5 exp0.175 step-33000 row is itself withdrawn, that row being registered but not bold.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(ceiling condition final score, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, network.value.ceiling 1.3 read back off the run config.yaml, ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000); MATCHED-STEP against CTRL R1 (v2.8.2__jt__20260803-063606__seed42 best.pt @ step 24000), scored in the same session and reproducing R1 to full precision;  artifact data/runs/v2.8.4/ceiling/stageC3_scores.json

<a id="v2-8-4--quadrotor-3d--unlabeled--1"></a>
## v2.8.4 · quadrotor_3d · -

`anchor: v2-8-4--quadrotor-3d--unlabeled--1` · ledger line 272 at the time of writing · cps `0.8149` · date `2026-08-08`

eval-only T2 deployment sweep cell 2_fallback_k5, which changes empty_fallback k 3 -> 5 on the REF checkpoint; see docs/versions/v2.8.4/arm_sweep.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(T2 deployment sweep cell 2_fallback_k5, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 5}, alpha (2.0,100.0), gamma_margin 0.0, value_ceiling 1.3, certificate V_hat (the learned value), ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000, artifact data/runs/v2.8.4/arm_sweep/cell__2_fallback_k5.json)

<a id="v2-8-4--quadrotor-3d--unlabeled--2"></a>
## v2.8.4 · quadrotor_3d · -

`anchor: v2-8-4--quadrotor-3d--unlabeled--2` · ledger line 273 at the time of writing · cps `0.6410` · date `2026-08-08`

eval-only T2 deployment sweep cell 3_fallback_k8, which changes empty_fallback k 3 -> 8 on the REF checkpoint; see docs/versions/v2.8.4/arm_sweep.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(T2 deployment sweep cell 3_fallback_k8, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 8}, alpha (2.0,100.0), gamma_margin 0.0, value_ceiling 1.3, certificate V_hat (the learned value), ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000, artifact data/runs/v2.8.4/arm_sweep/cell__3_fallback_k8.json)

<a id="v2-8-4--quadrotor-3d--unlabeled--3"></a>
## v2.8.4 · quadrotor_3d · -

`anchor: v2-8-4--quadrotor-3d--unlabeled--3` · ledger line 274 at the time of writing · cps `0.8147` · date `2026-08-08`

eval-only T2 deployment sweep cell 4_vdep_hard_k3, which changes the deployed certificate to V_dep hard-min at empty_fallback k 3; see docs/versions/v2.8.4/arm_sweep.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(T2 deployment sweep cell 4_vdep_hard_k3, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value_ceiling 1.3, certificate V_dep hard-min, ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000, artifact data/runs/v2.8.4/arm_sweep/cell__4_vdep_hard_k3.json)

<a id="v2-8-4--quadrotor-3d--unlabeled--4"></a>
## v2.8.4 · quadrotor_3d · -

`anchor: v2-8-4--quadrotor-3d--unlabeled--4` · ledger line 275 at the time of writing · cps `0.6268` · date `2026-08-08`

eval-only T2 deployment sweep cell 5_vdep_hard_k8, which changes the deployed certificate to V_dep hard-min and empty_fallback k 3 -> 8; see docs/versions/v2.8.4/arm_sweep.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(T2 deployment sweep cell 5_vdep_hard_k8, pool fullcb 3682a4e3, n 2000, eval_batch_size 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 8}, alpha (2.0,100.0), gamma_margin 0.0, value_ceiling 1.3, certificate V_dep hard-min, ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000, artifact data/runs/v2.8.4/arm_sweep/cell__5_vdep_hard_k8.json)

<a id="v2-8-4--quadrotor-3d--unlabeled--5"></a>
## v2.8.4 · quadrotor_3d · -

`anchor: v2-8-4--quadrotor-3d--unlabeled--5` · ledger line 276 at the time of writing · cps `0.6291` · date `2026-08-08`

T3 reference rollout, nominal system.lqr_action filtered by CBFQPFilter on the same V_hat the REF deploys; the artifact records only cps, reach and collision, so every other outcome cell is blank; see docs/versions/v2.8.4/arm_failure_class.md; diagnostic, no bold change

<a id="v2-8-4--quadrotor-3d--gamma--1"></a>
## v2.8.4 · quadrotor_3d · GAMMA

`anchor: v2-8-4--quadrotor-3d--gamma--1` · ledger line 277 at the time of writing · cps `0.8243` · date `2026-08-09`

gamma axis horizon_final 33.0 against the REF's 100.0; PRIMARY coll_band_lower 0.0165 vs the REF row's 0.0175, delta -0.0010 or 2 episodes of 2000, and the gamma band_lower scene set is a strict subset of the REF's (fixes scenes 15 and 1245, introduces none); cps 0.8243 CI [0.798275, 0.849950] vs the REF's 0.8308 CI [0.804321, 0.855404], delta -0.006576 with CIs overlapping and the magnitude below D0's 0.0083 floor, so no cps claim in either direction; the control differs in training.jt.schedule_n_steps (50000 vs 30000) as well as in horizon_final, so this is not a one-variable contrast; single seed; see docs/versions/v2.8.4/gamma_axis.md; diagnostic, no bold change

### relocated from the `verdict` column (pre-cap text, verbatim)

gamma axis horizon_final 33.0 against the REF's 100.0; PRIMARY coll_band_lower 0.0165 vs the REF row's 0.0175, delta -0.0010 or 2 episodes of 2000, and the gamma band_lower scene set is a strict subset of the REF's (fixes scenes 15 and 1245, introduces none); cps 0.8243 CI [0.798275, 0.849950] vs the REF's 0.8308 CI [0.804321, 0.855404], delta -0.006576 with CIs overlapping and the magnitude below D0's 0.0083 floor, so no cps claim in either direction; the control differs in training.jt.schedule_n_steps (50000 vs 30000) as well as in horizon_final, so this is not a one-variable contrast; single seed; see docs/versions/v2.8.4/gamma_axis.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(gamma condition final score, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, network.value.ceiling 1.3, gamma_final 0.96969697 from schedules.gamma_disc.horizon_final 33.0 and 0.96919127 at the scored step, ckpt v2.8.4__jt__20260809-002018__seed42/checkpoints/best.pt @ step 39000); scored in the same session as the REF row above, which reproduced its registered artifact on 13 fields with 0 differing; artifact data/runs/v2.8.4/gamma_axis/s3_scores.json key rows.GAMMA_best

<a id="v2-8-4--quadrotor-3d--ref-c-1-3--2"></a>
## v2.8.4 · quadrotor_3d · REF (C=1.3)

`anchor: v2-8-4--quadrotor-3d--ref-c-1-3--2` · ledger line 278 at the time of writing · cps `0.8441` · date `2026-08-09`

eval-only comparator, scored in the same session as the gamma fullscr41 row because no REF fullscr41 number existed on disk — the ledger's other fullscr41 rows are the box_klamp deployment screen whose parent is v2.8.2__jt__20260803-063606__seed42, the CTRL and not this condition; gamma-minus-REF on this pool is cps -0.004335, collision -0.0025, coll_band_lower -0.0010, coll_obstacle -0.0020, reach -0.0095, carrying the same signs as on fullcb with the three collision-family deltas identical to four decimals across the two pools, and both cps magnitudes below D0's 0.0083 floor so no cps claim; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

### relocated from the `verdict` column (pre-cap text, verbatim)

eval-only comparator, scored in the same session as the gamma fullscr41 row because no REF fullscr41 number existed on disk — the ledger's other fullscr41 rows are the box_klamp deployment screen whose parent is v2.8.2__jt__20260803-063606__seed42, the CTRL and not this condition; gamma-minus-REF on this pool is cps -0.004335, collision -0.0025, coll_band_lower -0.0010, coll_obstacle -0.0020, reach -0.0095, carrying the same signs as on fullcb with the three collision-family deltas identical to four decimals across the two pools, and both cps magnitudes below D0's 0.0083 floor so no cps claim; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(comparator scored for the gamma ITEM 2 second-pool read, pool fullscr41 9a016919, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, network.value.ceiling 1.3, ckpt v2.8.4__jt__20260808-133610__seed42/checkpoints/best.pt @ step 24000); artifact data/runs/v2.8.4/gamma_rescore/rescore.json key rows.ARM_fullscr41

<a id="v2-8-4--quadrotor-3d--gamma--2"></a>
## v2.8.4 · quadrotor_3d · GAMMA

`anchor: v2-8-4--quadrotor-3d--gamma--2` · ledger line 279 at the time of writing · cps `0.8534` · date `2026-08-09`

ITEM 1, the in-loop-vs-final gap decomposed on one checkpoint: the recorded in-loop row at step 39000 is collision 0.0205 and cps 0.8319, this cell is 0.0130 and 0.8534, the fullcb final is 0.0250 and 0.8243, so of the collision movement the filter cell accounts for -0.0075 and the pool composition for +0.0120 and the two sum exactly to the +0.0045 total; selection bias is NOT separable from these three numbers because all three score the same checkpoint; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

### relocated from the `verdict` column (pre-cap text, verbatim)

ITEM 1, the in-loop-vs-final gap decomposed on one checkpoint: the recorded in-loop row at step 39000 is collision 0.0205 and cps 0.8319, this cell is 0.0130 and 0.8534, the fullcb final is 0.0250 and 0.8243, so of the collision movement the filter cell accounts for -0.0075 and the pool composition for +0.0120 and the two sum exactly to the +0.0045 total; selection bias is NOT separable from these three numbers because all three score the same checkpoint; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

<a id="v2-8-4--quadrotor-3d--gamma--3"></a>
## v2.8.4 · quadrotor_3d · GAMMA

`anchor: v2-8-4--quadrotor-3d--gamma--3` · ledger line 280 at the time of writing · cps `0.8398` · date `2026-08-09`

ITEM 2, second-pool reproduction: the gamma-minus-REF deltas carry the SAME SIGN on both full pools for all five reported fields, with collision -0.0025, coll_band_lower -0.0010 and coll_obstacle -0.0020 identical on fullcb and fullscr41, reach -0.0095 here vs -0.0110, and cps -0.004335 here vs -0.006576, both magnitudes below D0's 0.0083 floor so no cps claim on either pool; the REF fullscr41 comparator was scored in this same session because the ledger's fullscr41 rows are the box_klamp screen on the v2.8.2 CTRL and not on the REF; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

### relocated from the `verdict` column (pre-cap text, verbatim)

ITEM 2, second-pool reproduction: the gamma-minus-REF deltas carry the SAME SIGN on both full pools for all five reported fields, with collision -0.0025, coll_band_lower -0.0010 and coll_obstacle -0.0020 identical on fullcb and fullscr41, reach -0.0095 here vs -0.0110, and cps -0.004335 here vs -0.006576, both magnitudes below D0's 0.0083 floor so no cps claim on either pool; the REF fullscr41 comparator was scored in this same session because the ledger's fullscr41 rows are the box_klamp screen on the v2.8.2 CTRL and not on the REF; see docs/versions/v2.8.4/gamma_rescore.md; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(gamma checkpoint re-scored on a second full pool, pool fullscr41 9a016919, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, network.value.ceiling 1.3, ckpt v2.8.4__jt__20260809-002018__seed42/checkpoints/best.pt @ step 39000); artifact data/runs/v2.8.4/gamma_rescore/rescore.json key rows.GAMMA_fullscr41

<a id="v2-8-5--quadrotor-3d--own-final-exp0-25"></a>
## v2.8.5 · quadrotor_3d · own-final exp0.25

`anchor: v2-8-5--quadrotor-3d--own-final-exp0-25` · ledger line 281 at the time of writing · cps `0.7336` · date `2026-08-11`

own deploy cell on pool full 0ef3751b; never read in one column with the registered-cell rows; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(the run's OWN final eval on its OWN deploy cell, pool full 0ef3751b, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,k 10}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.25, ckpt v2.8.5__jt__20260810-033431__seed42/checkpoints/best.pt @ step 40500); artifact data/runs/v2.8.5/set__20260810-033431__seed42/v2.8.5__jt__20260810-033431__seed42/eval_metrics.csv row mode=final

<a id="v2-8-5--quadrotor-3d--own-final-exp0-35"></a>
## v2.8.5 · quadrotor_3d · own-final exp0.35

`anchor: v2-8-5--quadrotor-3d--own-final-exp0-35` · ledger line 282 at the time of writing · cps `0.7008` · date `2026-08-11`

own deploy cell on pool full 0ef3751b; never read in one column with the registered-cell rows; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(the run's OWN final eval on its OWN deploy cell, pool full 0ef3751b, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,k 10}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.35, ckpt v2.8.5__jt__20260810-033451__seed42/checkpoints/best.pt @ step 42000); artifact data/runs/v2.8.5/set__20260810-033451__seed42/v2.8.5__jt__20260810-033451__seed42/eval_metrics.csv row mode=final

<a id="v2-8-5--quadrotor-3d--own-final-clip"></a>
## v2.8.5 · quadrotor_3d · own-final clip

`anchor: v2-8-5--quadrotor-3d--own-final-clip` · ledger line 283 at the time of writing · cps `0.7181` · date `2026-08-11`

own deploy cell on pool full 0ef3751b; never read in one column with the registered-cell rows; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(the run's OWN final eval on its OWN deploy cell, pool full 0ef3751b, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,k 10}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard block absent (clip geometry), ckpt v2.8.5__jt__20260810-072139__seed42/checkpoints/best.pt @ step 42000); artifact data/runs/v2.8.5/set__20260810-072139__seed42/v2.8.5__jt__20260810-072139__seed42/eval_metrics.csv row mode=final

<a id="v2-8-5--quadrotor-3d--1a-exp0-175"></a>
## v2.8.5 · quadrotor_3d · 1A exp0.175

`anchor: v2-8-5--quadrotor-3d--1a-exp0-175` · ledger line 284 at the time of writing · cps `0.8465` · date `2026-08-11`

registered-cell re-score, hazard exp ell 0.175, single-seed; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item 1A registered-cell re-score of the run's best.pt, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.175, ckpt v2.8.5__jt__20260810-033417__seed42/checkpoints/best.pt @ step 28500); artifact data/runs/v2.8.5/rescore/row__1A__exp0.175__step28500.json

<a id="v2-8-5--quadrotor-3d--1a-exp0-25"></a>
## v2.8.5 · quadrotor_3d · 1A exp0.25

`anchor: v2-8-5--quadrotor-3d--1a-exp0-25` · ledger line 285 at the time of writing · cps `0.8437` · date `2026-08-11`

registered-cell re-score, hazard exp ell 0.25, single-seed; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item 1A registered-cell re-score of the run's best.pt, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.25, ckpt v2.8.5__jt__20260810-033431__seed42/checkpoints/best.pt @ step 40500); artifact data/runs/v2.8.5/rescore/row__1A__exp0.25__step40500.json

<a id="v2-8-5--quadrotor-3d--1a-exp0-35"></a>
## v2.8.5 · quadrotor_3d · 1A exp0.35

`anchor: v2-8-5--quadrotor-3d--1a-exp0-35` · ledger line 286 at the time of writing · cps `0.7977` · date `2026-08-11`

registered-cell re-score, hazard exp ell 0.35, single-seed; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item 1A registered-cell re-score of the run's best.pt, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.35, ckpt v2.8.5__jt__20260810-033451__seed42/checkpoints/best.pt @ step 42000); artifact data/runs/v2.8.5/rescore/row__1A__exp0.35__step42000.json

<a id="v2-8-5--quadrotor-3d--1a-clip"></a>
## v2.8.5 · quadrotor_3d · 1A clip

`anchor: v2-8-5--quadrotor-3d--1a-clip` · ledger line 287 at the time of writing · cps `0.8317` · date `2026-08-11`

registered-cell re-score, clip geometry (v2.8.5 control), single-seed; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item 1A registered-cell re-score of the run's best.pt, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard block absent (clip geometry), ckpt v2.8.5__jt__20260810-072139__seed42/checkpoints/best.pt @ step 42000); artifact data/runs/v2.8.5/rescore/row__1A__clip__step42000.json

<a id="v2-8-5--quadrotor-3d--1d-exp0-175"></a>
## v2.8.5 · quadrotor_3d · 1D exp0.175

`anchor: v2-8-5--quadrotor-3d--1d-exp0-175` · ledger line 288 at the time of writing · cps `0.8520` · date `2026-08-11`

quadrotor_3d registered-cell row, hazard exp ell 0.175, single-seed 42; its series is TRUNCATED - the run died of a CUDA OOM at step 37199 and has no checkpoint beyond round 360, so 0.8520 is the maximum of a truncated series; registered but NOT bold. SUPERSEDED 2026-08-12: its promotion is withdrawn per the v2.8.5 close, and the bold quadrotor_3d row is the exp0.125 step-45000 row below (cps_v2 0.8701, same registered cell); bold released.

### relocated from the `verdict` column (pre-cap text, verbatim)

quadrotor_3d registered-cell row, hazard exp ell 0.175, single-seed 42; its series is TRUNCATED - the run died of a CUDA OOM at step 37199 and has no checkpoint beyond round 360, so 0.8520 is the maximum of a truncated series; registered but NOT bold. SUPERSEDED 2026-08-12: its promotion is withdrawn per the v2.8.5 close, and the bold quadrotor_3d row is the exp0.125 step-45000 row below (cps_v2 0.8701, same registered cell); bold released.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item 1D registered-cell step sweep, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.175, ckpt v2.8.5__jt__20260810-033417__seed42/checkpoints/step_033000.pt @ step 33000); artifact data/runs/v2.8.5/rescore/row__1D__exp0.175__step33000.json

<a id="v2-8-5--quadrotor-3d--b1-v2-8-4-c-1-3"></a>
## v2.8.5 · quadrotor_3d · B1 v2.8.4 C=1.3

`anchor: v2-8-5--quadrotor-3d--b1-v2-8-4-c-1-3` · ledger line 289 at the time of writing · cps `0.8344` · date `2026-08-11`

B1 matched-step 30000 comparator, v2.8.4 ceiling C=1.3 lineage, single-seed; diagnostic, no bold change

<a id="v2-8-5--quadrotor-3d--b1-v2-8-5-clip"></a>
## v2.8.5 · quadrotor_3d · B1 v2.8.5 clip

`anchor: v2-8-5--quadrotor-3d--b1-v2-8-5-clip` · ledger line 290 at the time of writing · cps `0.8315` · date `2026-08-11`

B1 matched-step 30000, clip geometry vs the v2.8.4 comparator 0.8344, CIs overlap; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(item B1 matched-step cell on the registered cell, pool fullcb 3682a4e3, n 2000, ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard block absent (clip geometry), ckpt v2.8.5__jt__20260810-072139__seed42/checkpoints/step_030000.pt @ step 30000); artifact data/runs/v2.8.5/rescore/row__B1__v2.8.5_CLIP50__step30000.json

<a id="v2-8-5--quadrotor-3d--own-final-exp0-125"></a>
## v2.8.5 · quadrotor_3d · own-final exp0.125

`anchor: v2-8-5--quadrotor-3d--own-final-exp0-125` · ledger line 291 at the time of writing · cps `0.7093` · date `2026-08-12`

the run's own deploy cell on pool full seed 23456, NOT the registered cell and never read in one column with a registered row; single-seed 42; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(the run's OWN final eval on its OWN deploy cell, pool full seed 23456, n 2000, eval_batch_size absent as a column of this cell, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,k 10}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, ckpt v2.8.5__jt__20260811-185046__seed42/checkpoints/best.pt @ step 50000); artifact data/runs/v2.8.5/set__20260811-185046__seed42/v2.8.5__jt__20260811-185046__seed42/eval_metrics.csv row mode=final

<a id="v2-8-5--quadrotor-3d--own-final-exp0-10"></a>
## v2.8.5 · quadrotor_3d · own-final exp0.10

`anchor: v2-8-5--quadrotor-3d--own-final-exp0-10` · ledger line 292 at the time of writing · cps `0.7199` · date `2026-08-12`

the run's own deploy cell on pool full seed 23456, NOT the registered cell and never read in one column with a registered row; single-seed 42; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

final(the run's OWN final eval on its OWN deploy cell, pool full seed 23456, n 2000, eval_batch_size absent as a column of this cell, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {none,k 10}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.10, ckpt v2.8.5__jt__20260811-185049__seed42/checkpoints/best.pt @ step 43500); artifact data/runs/v2.8.5/set__20260811-185049__seed42/v2.8.5__jt__20260811-185049__seed42/eval_metrics.csv row mode=final

<a id="v2-8-5--quadrotor-3d--best-pt-compress500"></a>
## v2.8.5 · quadrotor_3d · best.pt COMPRESS500

`anchor: v2-8-5--quadrotor-3d--best-pt-compress500` · ledger line 293 at the time of writing · cps `0.8113` · date `2026-08-12`

registered-cell row at the step status.json records as best_step, so best.pt carries the same weights; the run was SIGTERM-terminated at step 4779 (round 477.9) with 22 rounds of its budget unrun, so its series is truncated and its maximum is not comparable to one taken over 500 rounds; single-seed 42; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered-cell re-score at the run's best.pt step, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, n 2000, hazard block absent (clip geometry), compressed-update condition: one tenth of the gradient updates per round at the same collection schedule in round terms, ckpt v2.8.5__jt__20260811-235957__seed42/checkpoints/step_004350.pt @ step 4350); artifact data/runs/v2.8.5/rescore/row__COMPRESS500__step4350.json

<a id="v2-8-5--quadrotor-3d--exp0-125-45000-adopted"></a>
## v2.8.5 · quadrotor_3d · exp0.125 @45000 ADOPTED

`anchor: v2-8-5--quadrotor-3d--exp0-125-45000-adopted` · ledger line 294 at the time of writing · cps `0.8701` · date `2026-08-12`

quadrotor_3d SOTA-of-record on the registered cell; a 06_workflow 2.5 BASIS classification (the highest usable evaluation on the current basis) and NOT a 04_eval 5 CI-separated beat -- single-seed 42, and this row's 95% CI [0.8472, 0.8919] overlaps exp0.25, exp0.10 and CLIP50 at every round, so no separation is claimed in either direction; supersedes the v2.8.5 exp0.175 step-33000 row and the v2.8.4 REF row, both of which are released from bold; SUPERSEDED 2026-08-13: the bold quadrotor_3d row is the v2.9.1 SCENELAW_C1000_SIG0 round-930 row below (cps_v2 0.8722, same registered cell); bold released.

### relocated from the `verdict` column (pre-cap text, verbatim)

quadrotor_3d SOTA-of-record on the registered cell; a 06_workflow 2.5 BASIS classification (the highest usable evaluation on the current basis) and NOT a 04_eval 5 CI-separated beat -- single-seed 42, and this row's 95% CI [0.8472, 0.8919] overlaps exp0.25, exp0.10 and CLIP50 at every round, so no separation is claimed in either direction; supersedes the v2.8.5 exp0.175 step-33000 row and the v2.8.4 REF row, both of which are released from bold; SUPERSEDED 2026-08-13: the bold quadrotor_3d row is the v2.9.1 SCENELAW_C1000_SIG0 round-930 row below (cps_v2 0.8722, same registered cell); bold released.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered-cell argmax over this condition's scored steps, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, n 2000, hazard geom_form exp ell 0.125, ckpt v2.8.5__jt__20260811-185046__seed42/checkpoints/step_045000.pt @ step 45000 -- a FIXED-STEP checkpoint, NOT this run's best.pt, which is at step 50000; the adopted checkpoint's digest is pinned in data/secured_data/v2.8.5/seed42/ADOPTED.md and appears in no tracked prose); artifact data/runs/v2.8.5/rescore/row__exp0.125__step45000.json

<a id="v2-8-5--quadrotor-3d--best-pt-exp0-125"></a>
## v2.8.5 · quadrotor_3d · best.pt exp0.125

`anchor: v2-8-5--quadrotor-3d--best-pt-exp0-125` · ledger line 295 at the time of writing · cps `0.8614` · date `2026-08-12`

registered-cell row at the step status.json records as best_step, so best.pt carries the same weights; this is the run's own in-loop selection and is NOT the adopted checkpoint, which is the registered-cell argmax at step 45000; single-seed 42; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered-cell re-score at the run's best.pt step, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, n 2000, hazard geom_form exp ell 0.125, ckpt v2.8.5__jt__20260811-185046__seed42/checkpoints/step_050000.pt @ step 50000); artifact data/runs/v2.8.5/rescore/row__exp0.125__step50000.json

<a id="v2-8-5--quadrotor-3d--best-pt-exp0-10"></a>
## v2.8.5 · quadrotor_3d · best.pt exp0.10

`anchor: v2-8-5--quadrotor-3d--best-pt-exp0-10` · ledger line 296 at the time of writing · cps `0.8515` · date `2026-08-12`

registered-cell row at the step status.json records as best_step, so best.pt carries the same weights; single-seed 42; diagnostic, no bold change

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered-cell re-score at the run's best.pt step, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, n 2000, hazard geom_form exp ell 0.10, ckpt v2.8.5__jt__20260811-185049__seed42/checkpoints/step_043500.pt @ step 43500); artifact data/runs/v2.8.5/rescore/row__exp0.10__step43500.json

<a id="v2-9-0--quadrotor-3d--best-pt-bufcap2k"></a>
## v2.9.0 · quadrotor_3d · best.pt BUFCAP2K

`anchor: v2-9-0--quadrotor-3d--best-pt-bufcap2k` · ledger line 297 at the time of writing · cps `0.7679` · date `2026-08-13`

buffer_cap 2000 against COMPRESS1000's 1000000, differing in exactly that one config key; STOPPED BY RESEARCHER INSTRUCTION at step 5721 = round 572.1 of a 1000-round budget (status.json still reads phase training with halt_reason null; no final.pt, no report.md, zero final-eval rows), so it has NO end-of-run full evaluation and its in-loop maximum is taken over 570 rounds rather than 1000 and is not comparable to one taken over 1000; row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.700592; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

buffer_cap 2000 against COMPRESS1000's 1000000, differing in exactly that one config key; STOPPED BY RESEARCHER INSTRUCTION at step 5721 = round 572.1 of a 1000-round budget (status.json still reads phase training with halt_reason null; no final.pt, no report.md, zero final-eval rows), so it has NO end-of-run full evaluation and its in-loop maximum is taken over 570 rounds rather than 1000 and is not comparable to one taken over 1000; row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.700592; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.0__jt__20260812-173416__seed42/checkpoints/step_004050.pt @ step 4050 = round 405); artifact data/runs/v2.9.0/rescore/row__v290__BUFCAP2K__step4050.json

<a id="v2-9-0--quadrotor-3d--best-pt-bufcap10k"></a>
## v2.9.0 · quadrotor_3d · best.pt BUFCAP10K

`anchor: v2-9-0--quadrotor-3d--best-pt-bufcap10k` · ledger line 298 at the time of writing · cps `0.8401` · date `2026-08-13`

buffer_cap 10000 against COMPRESS1000's 1000000, differing in exactly that one config key; ran to its own terminal path (phase done, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.790540; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

buffer_cap 10000 against COMPRESS1000's 1000000, differing in exactly that one config key; ran to its own terminal path (phase done, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.790540; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.0__jt__20260812-173221__seed42/checkpoints/step_005850.pt @ step 5850 = round 585); artifact data/runs/v2.9.0/rescore/row__v290__BUFCAP10K__step5850.json

<a id="v2-9-0--quadrotor-3d--best-pt-compress1000"></a>
## v2.9.0 · quadrotor_3d · best.pt COMPRESS1000

`anchor: v2-9-0--quadrotor-3d--best-pt-compress1000` · ledger line 299 at the time of writing · cps `0.8607` · date `2026-08-13`

the compressed-update condition (collect_every 10) on the exp0.125 geometry, 1000 rounds, ran to its own terminal path (phase done, final.pt and report.md present); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.847767; the REGISTERED axis A1 of changes.md 2.2 is FALSIFIED for this condition -- below its control exp0.125 at 13 of 14 matched rounds >=300, mean paired difference -0.034378 (4.14x the 0.0083 admissibility floor, negative); per-round CIs overlap at 13 of the 14 and are disjoint at round 300 alone, so NO separation is claimed in either direction -- one round carries nothing, the verdict rests on the sign count -- and this row is not a comparison; A2 (eval share below 59.11%) holds at 7.26% whole-run and 4.16-8.32% per residency segment; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the compressed-update condition (collect_every 10) on the exp0.125 geometry, 1000 rounds, ran to its own terminal path (phase done, final.pt and report.md present); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.847767; the REGISTERED axis A1 of changes.md 2.2 is FALSIFIED for this condition -- below its control exp0.125 at 13 of 14 matched rounds >=300, mean paired difference -0.034378 (4.14x the 0.0083 admissibility floor, negative); per-round CIs overlap at 13 of the 14 and are disjoint at round 300 alone, so NO separation is claimed in either direction -- one round carries nothing, the verdict rests on the sign count -- and this row is not a comparison; A2 (eval share below 59.11%) holds at 7.26% whole-run and 4.16-8.32% per residency segment; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.0__jt__20260812-165849__seed42/checkpoints/step_007500.pt @ step 7500 = round 750); artifact data/runs/v2.9.0/rescore/row__v290__COMPRESS1000__step7500.json

<a id="v2-9-0--quadrotor-3d--best-pt-bufcap5k"></a>
## v2.9.0 · quadrotor_3d · best.pt BUFCAP5K

`anchor: v2-9-0--quadrotor-3d--best-pt-bufcap5k` · ledger line 300 at the time of writing · cps `0.8164` · date `2026-08-13`

buffer_cap 5000 against COMPRESS1000's 1000000, differing in exactly that one config key; ran to its own terminal path (phase done, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.745422; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

buffer_cap 5000 against COMPRESS1000's 1000000, differing in exactly that one config key; ran to its own terminal path (phase done, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.745422; buffer_cap was NOT a registered axis of v2.9.0 -- changes.md 3 declares one substantive axis (gradient updates per collection round) -- and NO verdict is drawn on it here; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.0/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.0__jt__20260812-173227__seed42/checkpoints/step_008100.pt @ step 8100 = round 810); artifact data/runs/v2.9.0/rescore/row__v290__BUFCAP5K__step8100.json

<a id="v2-9-1--quadrotor-3d--best-pt-scenelaw-b10k"></a>
## v2.9.1 · quadrotor_3d · best.pt SCENELAW_B10K

`anchor: v2-9-1--quadrotor-3d--best-pt-scenelaw-b10k` · ledger line 301 at the time of writing · cps `0.8281` · date `2026-08-13`

the v2.9.1 training scene law -- five clearance floors moved together, start obstacle clearance 0.10 -> 0.00, goal obstacle clearance 0.10 -> 0.05, vertical band margin 0.10 -> 0.00, training arena margin 0.30 -> 0.00, init_feasibility_margin 0.05 -> 0.00 -- against BUFCAP10K, differing in exactly those five config keys and nothing else, with the evaluation path untouched; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.797601; this run's in-loop argmax is step 9300 at 0.798977, 0.001375 above the selected step and inside the 0.002 early_stop_min_delta hysteresis, so it never displaced the selection; over 48 matched rounds >= 300 on the registered cell against BUFCAP10K, changes.md 2's registered observables read A1 FALSIFIED, A2 HOLDS, A3 FALSIFIED; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 training scene law -- five clearance floors moved together, start obstacle clearance 0.10 -> 0.00, goal obstacle clearance 0.10 -> 0.05, vertical band margin 0.10 -> 0.00, training arena margin 0.30 -> 0.00, init_feasibility_margin 0.05 -> 0.00 -- against BUFCAP10K, differing in exactly those five config keys and nothing else, with the evaluation path untouched; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.797601; this run's in-loop argmax is step 9300 at 0.798977, 0.001375 above the selected step and inside the 0.002 early_stop_min_delta hysteresis, so it never displaced the selection; over 48 matched rounds >= 300 on the registered cell against BUFCAP10K, changes.md 2's registered observables read A1 FALSIFIED, A2 HOLDS, A3 FALSIFIED; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.1__jt__20260813-021017__seed42/checkpoints/step_005400.pt @ step 5400 = round 540); artifact data/runs/v2.9.1/rescore/row__v291__SCENELAW_B10K__step5400.json

<a id="v2-9-1--quadrotor-3d--best-pt-scenelaw-b10k-sig0"></a>
## v2.9.1 · quadrotor_3d · best.pt SCENELAW_B10K_SIG0

`anchor: v2-9-1--quadrotor-3d--best-pt-scenelaw-b10k-sig0` · ledger line 302 at the time of writing · cps `0.8278` · date `2026-08-13`

the sigma-floor condition, schedules.sigma.sigma_min 0.3 -> 0.0 against SCENELAW_B10K, differing in exactly that one config key; sigma_min was NOT a registered axis of v2.9.1 -- changes.md 3 declares one substantive axis, the clearance floors on the training scene law -- and NO verdict is drawn on it here; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.793990; the three registered observables are scored against SCENELAW_B10K in docs/versions/v2.9.1/final_scoring.md 5.3, a pairing that is NOT one of changes.md 2's two: A1 FALSIFIED, A2 HOLDS, A3 HOLDS; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the sigma-floor condition, schedules.sigma.sigma_min 0.3 -> 0.0 against SCENELAW_B10K, differing in exactly that one config key; sigma_min was NOT a registered axis of v2.9.1 -- changes.md 3 declares one substantive axis, the clearance floors on the training scene law -- and NO verdict is drawn on it here; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.793990; the three registered observables are scored against SCENELAW_B10K in docs/versions/v2.9.1/final_scoring.md 5.3, a pairing that is NOT one of changes.md 2's two: A1 FALSIFIED, A2 HOLDS, A3 HOLDS; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.1__jt__20260813-022406__seed42/checkpoints/step_005850.pt @ step 5850 = round 585); artifact data/runs/v2.9.1/rescore/row__v291__SCENELAW_B10K_SIG0__step5850.json

<a id="v2-9-1--quadrotor-3d--best-pt-scenelaw-c1000"></a>
## v2.9.1 · quadrotor_3d · best.pt SCENELAW_C1000

`anchor: v2-9-1--quadrotor-3d--best-pt-scenelaw-c1000` · ledger line 303 at the time of writing · cps `0.8637` · date `2026-08-13`

the v2.9.1 training scene law -- five clearance floors moved together, start obstacle clearance 0.10 -> 0.00, goal obstacle clearance 0.10 -> 0.05, vertical band margin 0.10 -> 0.00, training arena margin 0.30 -> 0.00, init_feasibility_margin 0.05 -> 0.00 -- against COMPRESS1000, differing in exactly those five config keys and nothing else, with the evaluation path untouched; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.862418; over 48 matched rounds >= 300 on the registered cell against COMPRESS1000, changes.md 2's registered observables read A1 FALSIFIED, A2 HOLDS, A3 HOLDS; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 training scene law -- five clearance floors moved together, start obstacle clearance 0.10 -> 0.00, goal obstacle clearance 0.10 -> 0.05, vertical band margin 0.10 -> 0.00, training arena margin 0.30 -> 0.00, init_feasibility_margin 0.05 -> 0.00 -- against COMPRESS1000, differing in exactly those five config keys and nothing else, with the evaluation path untouched; ran to its own terminal path (phase done, halt_reason null, final.pt and report.md present, 1000 rounds); row is the registered-cell re-score at the step status.json records as best_step, i.e. its in-loop best on pool inloopv2 seed 145678 n 2000 where it scored 0.862418; over 48 matched rounds >= 300 on the registered cell against COMPRESS1000, changes.md 2's registered observables read A1 FALSIFIED, A2 HOLDS, A3 HOLDS; no CI separation is claimed in either direction on any channel; single-seed 42; diagnostic, no bold change; detail docs/versions/v2.9.1/final_scoring.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.1__jt__20260813-021011__seed42/checkpoints/step_008850.pt @ step 8850 = round 885); artifact data/runs/v2.9.1/rescore/row__v291__SCENELAW_C1000__step8850.json

<a id="v2-9-1--quadrotor-3d--best-pt-scenelaw-c1000-sig0"></a>
## v2.9.1 · quadrotor_3d · best.pt SCENELAW_C1000_SIG0

`anchor: v2-9-1--quadrotor-3d--best-pt-scenelaw-c1000-sig0` · ledger line 304 at the time of writing · cps `**0.8722**` · date `**2026-08-13**`

**quadrotor_3d SOTA-of-record on the registered cell; a 06_workflow 2.5 BASIS classification (the highest usable evaluation on the current basis) and NOT a 04_eval 5 CI-separated beat -- against the superseded v2.8.5 exp0.125 step-45000 row the difference is +0.002098 read on the two rows' own artifacts (+0.002123 read against that row's 4-dp ledger cell 0.8701), i.e. 0.25x the 0.0083 admissibility floor, this row's 95% CI [0.8471, 0.8962] overlaps that row's [0.8472, 0.8919], and both rows are single seed 42, so no separation is claimed in either direction; what is NEW relative to that row is the v2.9.1 training scene law (five clearance floors moved together), schedules.sigma.sigma_min 0.3 -> 0.0, and the compressed schedule at one fifth the update total (10000 updates over 1000 rounds against 50000 over 500); ran to its own terminal path (phase done, halt_reason null, 1000 rounds); sigma_min was NOT a registered axis of v2.9.1 and no verdict on it is drawn here -- the three registered observables against SCENELAW_C1000 read A1 HOLDS, A2 FALSIFIED, A3 HOLDS in final_scoring.md 12.5, a pairing that is NOT one of changes.md 2's two; supersedes the v2.8.5 exp0.125 step-45000 row, which is released from bold in this edit; detail docs/versions/v2.9.1/final_scoring.md**

### relocated from the `verdict` column (pre-cap text, verbatim)

**quadrotor_3d SOTA-of-record on the registered cell; a 06_workflow 2.5 BASIS classification (the highest usable evaluation on the current basis) and NOT a 04_eval 5 CI-separated beat -- against the superseded v2.8.5 exp0.125 step-45000 row the difference is +0.002098 read on the two rows' own artifacts (+0.002123 read against that row's 4-dp ledger cell 0.8701), i.e. 0.25x the 0.0083 admissibility floor, this row's 95% CI [0.8471, 0.8962] overlaps that row's [0.8472, 0.8919], and both rows are single seed 42, so no separation is claimed in either direction; what is NEW relative to that row is the v2.9.1 training scene law (five clearance floors moved together), schedules.sigma.sigma_min 0.3 -> 0.0, and the compressed schedule at one fifth the update total (10000 updates over 1000 rounds against 50000 over 500); ran to its own terminal path (phase done, halt_reason null, 1000 rounds); sigma_min was NOT a registered axis of v2.9.1 and no verdict on it is drawn here -- the three registered observables against SCENELAW_C1000 read A1 HOLDS, A2 FALSIFIED, A3 HOLDS in final_scoring.md 12.5, a pairing that is NOT one of changes.md 2's two; supersedes the v2.8.5 exp0.125 step-45000 row, which is released from bold in this edit; detail docs/versions/v2.9.1/final_scoring.md**

### relocated from the `eval_source` column (pre-cap text, verbatim)

**eval_only(registered cell, pool fullcb (the registered pool of record, n 2000), ebs 2000, terminal (0.15,0.3,0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0,100.0), gamma_margin 0.0, value ceiling 1.3, hazard geom_form exp ell 0.125, n 2000, re-score at the run's in-loop best step, ckpt v2.9.1__jt__20260813-121049__seed42/checkpoints/step_009300.pt @ step 9300 = round 930, which is the step status.json records as best_step so best.pt carries the same weights; this is the ADOPTED checkpoint and its digest is pinned in data/secured_data/v2.9.1/seed42/ADOPTED.md and appears in no tracked prose); artifact data/runs/v2.9.1/rescore/row__v291__SCENELAW_C1000_SIG0__step9300.json**

<a id="v2-9-1--quadrotor-3d--best-pt-oc3d-v291"></a>

**BOLD REMOVED 2026-08-20.** Bold moved to the 400-step cold-start cells on Researcher
approval (L359 takes quadrotor_3d). This row is retained for provenance and is not un-registered; per
`06_workflow` 2.5 historical rows never retain bold. It was scored at `eval.max_steps` **200**,
off the 400-step basis the paper reports, which is the condition STATUS 5 recorded as blocking
any bold change and which this move resolves.
## v2.9.1 · quadrotor_3d · best.pt OC3D_V291

`anchor: v2-9-1--quadrotor-3d--best-pt-oc3d-v291` · ledger line 305 at the time of writing · cps `0.6818` · date `2026-08-13 17:48:32`

the v2.9.1 OC-PNCBF condition on quadrotor_3d at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.647000 (reach 0.4510, collision 0.5490), so this row is what the certificate ADDS: +1.328835. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). quadrotor_3d is the ONLY system on which the `ell` axis was ever screened (v2.8.5: 0.10/0.125/0.175/0.25/0.35), so on this row `exp`/0.125 is a SELECTED value, not an inheritance. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10}. EMPTY 0.4845 and SINGULAR 0.1675 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 OC-PNCBF condition on quadrotor_3d at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.647000 (reach 0.4510, collision 0.5490), so this row is what the certificate ADDS: +1.328835. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). quadrotor_3d is the ONLY system on which the `ell` axis was ever screened (v2.8.5: 0.10/0.125/0.175/0.25/0.35), so on this row `exp`/0.125 is a SELECTED value, not an inheritance. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10}. EMPTY 0.4845 and SINGULAR 0.1675 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000), ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 27450 = round 915; cps recomputed from its own components with residual +1.11e-16 against the evaluator); collision decomposed by terminating surface (collision_cause, priority obstacle > band_lower > band_upper): obstacle 0.0310 (62 ep), band_lower 0.0265 (53 ep), band_upper 0.0000 (0 ep), unattributed 0 ep; the three sum to this row's collision with residual +6.94e-18; every outcome field of this row was reproduced before the split was taken; artifact data/runs/v2.9.1/launch/ocrow__quadrotor_3d.json, split artifact data/runs/v2.9.1/launch/collsplit__L305.json

<a id="v2-9-1--quadrotor-planar--best-pt-ocplanar-v291-r3"></a>
## v2.9.1 · quadrotor_planar · best.pt OCPLANAR_V291_R3

`anchor: v2-9-1--quadrotor-planar--best-pt-ocplanar-v291-r3` · ledger line 306 at the time of writing · cps `0.6591` · date `2026-08-13 17:48:32`

the v2.9.1 OC-PNCBF condition on quadrotor_planar at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.206000 (reach 0.5980, collision 0.4020), so this row is what the certificate ADDS: +0.865150. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}, 'quadrotor_planar': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.3885 and SINGULAR 0.2305 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 OC-PNCBF condition on quadrotor_planar at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.206000 (reach 0.5980, collision 0.4020), so this row is what the certificate ADDS: +0.865150. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}, 'quadrotor_planar': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.3885 and SINGULAR 0.2305 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool eval_full_quadrotor-planar_n2000_seed23456 (this system's n2000 full pool of record; no `fullcb` successor exists for it, the cb variant having been built for the 3D band hazard), ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 28350 = round 945; cps recomputed from its own components with residual +0.00e+00 against the evaluator); artifact data/runs/v2.9.1/launch/ocrow__quadrotor_planar.json

<a id="v2-9-1--double-integrator--best-pt-ocdi-v291"></a>
## v2.9.1 · double_integrator · best.pt OCDI_V291

`anchor: v2-9-1--double-integrator--best-pt-ocdi-v291` · ledger line 307 at the time of writing · cps `0.8805` · date `2026-08-13 17:48:32`

the v2.9.1 OC-PNCBF condition on double_integrator at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.009500 (reach 0.6635, collision 0.3365), so this row is what the certificate ADDS: +0.889959. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.2365 and SINGULAR 0.1965 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 OC-PNCBF condition on double_integrator at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.009500 (reach 0.6635, collision 0.3365), so this row is what the certificate ADDS: +0.889959. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.2365 and SINGULAR 0.1965 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

<a id="v2-9-1--unicycle--best-pt-ocuni-v291"></a>
## v2.9.1 · unicycle · best.pt OCUNI_V291

`anchor: v2-9-1--unicycle--best-pt-ocuni-v291` · ledger line 308 at the time of writing · cps `0.6705` · date `2026-08-13 17:48:32`

the v2.9.1 OC-PNCBF condition on unicycle at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps 0.002000 (reach 0.6670, collision 0.3320), so this row is what the certificate ADDS: +0.668486. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.2215 and SINGULAR 0.3180 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 OC-PNCBF condition on unicycle at its own best_step; a diagnostic baseline for the JT condition that warm-starts from this very checkpoint, NOT a rank claim -- the JT condition holds the record and nothing here is bolded or promoted. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps 0.002000 (reach 0.6670, collision 0.3320), so this row is what the certificate ADDS: +0.668486. SCHEDULE IS FROZEN (`schedules.gamma_disc.total_epochs` 10000 against a 1000-epoch run, so `target_rhs` is identically 0.0, `gamma` constant 0.95 and `lambda` never leaves warm-up -- the tail-bootstrap ratchet that collapsed the first 3D and planar OC runs to V == ceiling is unreachable) and the BUFFER IS CAPPED at `collection.oc_pncbf.buffer_capacity` 10000 trajectory records (the OOM fix; the uncapped runs died in `_cat_optional`). `hazard.geom_form exp` / `hazard.ell 0.125` is an UNSCREENED INHERITANCE on this system -- the `ell` axis was screened on quadrotor_3d only (v2.8.5) and this system has never been screened on either key; both DO have a reader here (oc_pncbf/collection.py:599 -> value_target_barrier -> quadrotor_barrier.py:111-112, no system gate), and the form moves the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, i.e. 0.0884 m INWARD and inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer the JT rows use, which REPLACES `filter.empty_fallback` with a flat {kstep, phases 1, k 3} so every system scores on the v2.9.1 four-system cell; this run TRAINED under {'mode': 'none', 'k': 10, 'quadrotor_3d': {'mode': 'kstep', 'phases': 1, 'k': 3}}. EMPTY 0.2215 and SINGULAR 0.3180 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's physically-active steps) and are never summed over steps; detail docs/versions/v2.9.1/oc_arm_rows.md

<a id="v2-9-1--double-integrator--best-pt-jtdi-v291-r2"></a>
## v2.9.1 · double_integrator · best.pt JTDI_V291_R2

`anchor: v2-9-1--double-integrator--best-pt-jtdi-v291-r2` · ledger line 309 at the time of writing · cps `0.9048` · date `2026-08-13 21:52:00`

the v2.9.1 JT-PNCBF condition on double_integrator, warm-started from this system's OWN OC best.pt (training.jt.value_init_ckpt; value only -- v_s_state and v_s_target_state, no pi_state, so the policy trains from scratch); ran to its own terminal path (phase done, halt_reason null, 10000 steps = 1000 rounds, final.pt and report.md present). Against this system's OC row on the SAME pool and the SAME cell, OC3D-style OCDI_V291 row L307 at cps 0.880459 (step 25200 = round 840), the difference is +0.024298; single-seed 42 on both sides and NO CI separation is claimed in either direction. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.009500 (reach 0.6635, collision 0.3365), so the certificate-plus-policy adds +0.914257. TRAINED with the empty-branch fallback OFF -- filter.empty_fallback resolves to {mode none, k 10} in this run's own config, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config, so the k-step branch at filter_hardnet.py:192 was unreachable in training; this is the same training-side resolution the bold quadrotor_3d row L304 had, and it is the eval-only separation the design intends. The predecessor run at this system carried a per-system {kstep, phases 1, k 3} sub-block and was STOPPED and relaunched for that reason (detail docs/versions/v2.9.1/jt_planar.md 5). hazard.geom_form exp / hazard.ell 0.125 remains an UNSCREENED INHERITANCE on this system -- the ell axis was screened on quadrotor_3d only (v2.8.5) -- moving the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer as L304 and the four OC rows. EMPTY 0.2570 and SINGULAR 0.2305 are EPISODE-axis fractions over each episode's physically-active steps and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 JT-PNCBF condition on double_integrator, warm-started value-only from this system's OWN OC best.pt so the policy trains from scratch, run to its own terminal path. Against this system's OC row L307 on the SAME pool and cell the difference is +0.024298, and against the unfiltered LQR nominal (cps -0.009500) the certificate-plus-policy adds +0.914257. Single seed 42 on both sides with NO CI separation claimed in either direction; trained with the empty-branch fallback OFF, and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this system. EMPTY 0.2570 and SINGULAR 0.2305 are EPISODE-axis fractions and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md and docs/ledger_verdicts.md#v2-9-1--double-integrator--best-pt-jtdi-v291-r2.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool eval_full_di_n2000_seed123456 (the v2.9.1 REGENERATED pool of record, n 2000) -- the SAME pool and cell as this system's OC row, ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 9300 = round 930, the step status.json records as best_step; cps recomputed from its own components with residual +0.00e+00 against the evaluator); artifact data/runs/v2.9.1/launch/jtrow__double_integrator.json

<a id="v2-9-1--unicycle--best-pt-jtuni-v291-r2"></a>
## v2.9.1 · unicycle · best.pt JTUNI_V291_R2

`anchor: v2-9-1--unicycle--best-pt-jtuni-v291-r2` · ledger line 310 at the time of writing · cps `0.8731` · date `2026-08-13 21:52:00`

the v2.9.1 JT-PNCBF condition on unicycle, warm-started from this system's OWN OC best.pt (training.jt.value_init_ckpt; value only -- v_s_state and v_s_target_state, no pi_state, so the policy trains from scratch); ran to its own terminal path (phase done, halt_reason null, 10000 steps = 1000 rounds, final.pt and report.md present). Against this system's OC row on the SAME pool and the SAME cell, OCUNI_V291 row L308 at cps 0.670486 (step 14400 = round 480), the difference is +0.202640; single-seed 42 on both sides and NO CI separation is claimed in either direction. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps 0.002000 (reach 0.6670, collision 0.3320), so the certificate-plus-policy adds +0.871126. TRAINED with the empty-branch fallback OFF -- filter.empty_fallback resolves to {mode none, k 10} in this run's own config, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config, so the k-step branch at filter_hardnet.py:192 was unreachable in training; this is the same training-side resolution the bold quadrotor_3d row L304 had, and it is the eval-only separation the design intends. The predecessor run at this system carried a per-system {kstep, phases 1, k 3} sub-block and was STOPPED and relaunched for that reason (detail docs/versions/v2.9.1/jt_planar.md 5). hazard.geom_form exp / hazard.ell 0.125 remains an UNSCREENED INHERITANCE on this system -- the ell axis was screened on quadrotor_3d only (v2.8.5) -- moving the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer as L304 and the four OC rows. EMPTY 0.1830 and SINGULAR 0.1150 are EPISODE-axis fractions over each episode's physically-active steps and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 JT-PNCBF condition on unicycle, warm-started value-only from this system's OWN OC best.pt so the policy trains from scratch, run to its own terminal path. Against this system's OC row L308 on the SAME pool and cell the difference is +0.202640, and against the unfiltered LQR nominal (cps 0.002000) the certificate-plus-policy adds +0.871126. Single seed 42 on both sides with NO CI separation claimed in either direction; trained with the empty-branch fallback OFF, and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this system. EMPTY 0.1830 and SINGULAR 0.1150 are EPISODE-axis fractions and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md and docs/ledger_verdicts.md#v2-9-1--unicycle--best-pt-jtuni-v291-r2.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool eval_full_unicycle_n2000_seed123456 (the v2.9.1 REGENERATED pool of record, n 2000) -- the SAME pool and cell as this system's OC row, ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 9450 = round 945, the step status.json records as best_step; cps recomputed from its own components with residual +0.00e+00 against the evaluator); artifact data/runs/v2.9.1/launch/jtrow__unicycle.json

<a id="v2-9-1--quadrotor-planar--best-pt-jtplanar-v291-r4"></a>
## v2.9.1 · quadrotor_planar · best.pt JTPLANAR_V291_R4

`anchor: v2-9-1--quadrotor-planar--best-pt-jtplanar-v291-r4` · ledger line 311 at the time of writing · cps `0.7757` · date `2026-08-13 23:36:00`

the v2.9.1 JT-PNCBF condition on quadrotor_planar, warm-started from this system's OWN OC best.pt (value only -- v_s_state and v_s_target_state, no pi_state, so the policy trains from scratch); ran to its own terminal path (phase done, halt_reason null, 10000 steps = 1000 rounds, final.pt and report.md present). Against this system's OC row L306 on the SAME pool and the SAME cell, cps 0.659150 at step 28350 = round 945, the difference is +0.116590; single-seed 42 on both sides and NO CI separation is claimed in either direction. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.206000 (reach 0.5980, collision 0.4020), so the certificate-plus-policy adds +0.981740. TRAINED with the empty-branch fallback OFF -- filter.empty_fallback resolves to {mode none, k 10}, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config -- the same training-side resolution the bold quadrotor_3d row L304 had. This run is a RELAUNCH of JTPLANAR_V291_R3, which was byte-identical in config and seed and died of CUDA OOM at round 768 while SHARING the card with two other JT runs; R4 ran ALONE and its cuda_max_mem_mb matched R3's to the decimal at every round through 768 (3076.1 MB at the fatal round), peaking at 3944.8 MB with 7638 MiB still free, so the OOM is attributed to contention with a matched control and NOT to collection.jt.buffer_cap, which is 1e6 by the deliberate D_V=1M setting (exp_config.yaml:183) and was NOT changed. R3's best was 0.588629; the rounds it never ran are worth +0.156597 of in-loop cps. hazard.geom_form exp / hazard.ell 0.125 remains an UNSCREENED INHERITANCE on this system -- the ell axis was screened on quadrotor_3d only (v2.8.5) -- moving the alpha switch from h_scale/2 = 0.1750 m to ell*ln2 = 0.0866 m, inside the 0.15 m eval clearance floor. Cell = v282_agree_gate.gate_overrides, the same producer as L304 and the four OC rows. EMPTY 0.3780 and SINGULAR 0.1085 are EPISODE-axis fractions over each episode's physically-active steps and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 JT-PNCBF condition on quadrotor_planar, warm-started value-only from this system's OWN OC best.pt, and a RELAUNCH of the byte-identical JTPLANAR_V291_R3 that died of CUDA OOM while sharing the card. Against this system's OC row L306 on the SAME pool and cell the difference is +0.116590, and against the unfiltered LQR nominal (cps -0.206000) the certificate-plus-policy adds +0.981740. Single seed 42 with NO CI separation claimed in either direction; trained with the empty-branch fallback OFF, and hazard.ell 0.125 is an UNSCREENED INHERITANCE on this system. EMPTY 0.3780 and SINGULAR 0.1085 are EPISODE-axis fractions and are never summed. No bold change, no promotion, no SOTA claim; detail docs/versions/v2.9.1/jt_planar.md and docs/ledger_verdicts.md#v2-9-1--quadrotor-planar--best-pt-jtplanar-v291-r4.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool eval_full_quadrotor-planar_n2000_seed23456 (this system's n2000 full pool of record, n 2000) -- the SAME pool and cell as its OC row L306, ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 9000 = round 900, the step status.json records as best_step; cps recomputed from its own components with residual +0.00e+00 against the evaluator); artifact data/runs/v2.9.1/launch/jtrow__quadrotor_planar.json

<a id="v2-9-1--quadrotor-3d--best-pt-jt3d-v291-ocr4"></a>
## v2.9.1 · quadrotor_3d · best.pt JT3D_V291_OCR4

`anchor: v2-9-1--quadrotor-3d--best-pt-jt3d-v291-ocr4` · ledger line 312 at the time of writing · cps `0.8680` · date `2026-08-14 08:52:00`

the v2.9.1 JT-PNCBF condition on quadrotor_3d, warm-started from this system's OWN OC best.pt (OC3D_V291_R4, step 27450 = round 915, its own frac(V<=0) 0.7205 on the 3D mixed in-loop pool); value only -- v_s_state and v_s_target_state, no pi_state, so the policy trains from scratch; ran to its own terminal path (phase done, halt_reason null, 10000 steps = 1000 rounds, final.pt and report.md present). WHY THIS RUN EXISTS: the bold row L304 warm-started from a v2.7.6 OC checkpoint that predates all three OC repairs -- it was selected on the pre-d2 n500 pool, deployed under CBFQPFilter, and trained against a label clamped at 1.0 -- while the other three Table I columns start from OC conditions carrying all three repairs; this run makes the quadrotor_3d column STRUCTURALLY IDENTICAL to them. Its config is L304's own persisted config with EXACTLY ONE key changed, training.jt.value_init_ckpt. Against this system's OC row L305 on the SAME pool and cell (cps 0.681835), the difference is +0.186152. Against L304 on the SAME pool and cell (cps 0.872223), the difference is -0.004236, i.e. 0.51x the 0.0083 admissibility floor and BELOW it; this row's point estimate lies INSIDE L304's own 95% CI [0.8471, 0.8962]; no CI was computed for this row by its producer, so NO separation is claimed in either direction. Per the Researcher's disposition fixed in advance of this number: Table I takes THIS row for the quadrotor_3d JT cell and the BOLD STAYS AT L304, which remains this system's record best; the two are different objects and this row is not a rank claim. NOMINAL on this same cell and pool, unfiltered LQR with no certificate and no filter, scores cps -0.647000 (reach 0.4510, collision 0.5490), so the certificate-plus-policy adds +1.514987. TRAINED with the empty-branch fallback OFF -- filter.empty_fallback resolves to {mode none, k 10}, verified before launch by read-back from the CONSTRUCTED collection filter rather than from the config; the same training-side resolution L304 had. hazard.geom_form exp / hazard.ell 0.125; on quadrotor_3d the ell axis WAS screened (v2.8.5: 0.10/0.125/0.175/0.25/0.35) so this is a SELECTED value here, unlike the other three systems. Cell = v282_agree_gate.gate_overrides, the same producer as L304 and L305-L311. EMPTY 0.5830 and SINGULAR 0.0955 are EPISODE-axis fractions over each episode's physically-active steps and are never summed. Single seed 42. No bold change, no promotion in the registering item; detail docs/versions/v2.9.1/final_scoring.md 15

### relocated from the `verdict` column (pre-cap text, verbatim)

the v2.9.1 JT-PNCBF condition on quadrotor_3d, warm-started value-only from this system's OWN OC best.pt (L305) so the quadrotor_3d column is STRUCTURALLY IDENTICAL to the other three, its config being L304's with EXACTLY ONE key changed. Against L305 on the SAME pool and cell the difference is +0.186152; against L304 it is -0.004236, i.e. 0.51x the 0.0083 admissibility floor and inside L304's own 95% CI [0.8471, 0.8962], so NO separation is claimed in either direction. Single seed 42 and no CI computed by this row's producer; EMPTY 0.5830 and SINGULAR 0.0955 are EPISODE-axis fractions and are never summed. Per the Researcher's disposition fixed in advance of this number, Table I takes THIS row for the quadrotor_3d JT cell and the BOLD STAYS AT L304; no promotion in the registering item. Detail docs/versions/v2.9.1/final_scoring.md 15 and docs/ledger_verdicts.md#v2-9-1--quadrotor-3d--best-pt-jt3d-v291-ocr4.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000) -- the SAME pool and cell as L304 and L305, ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 9450 = round 945, the step status.json records as best_step; cps recomputed from its own components with residual +1.11e-16 against the evaluator); collision decomposed by terminating surface (collision_cause, priority obstacle > band_lower > band_upper): obstacle 0.0075 (15 ep), band_lower 0.0180 (36 ep), band_upper 0.0000 (0 ep), unattributed 0 ep; the three sum to this row's collision with residual +0.00e+00; every outcome field of this row was reproduced before the split was taken; artifact data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json, split artifact data/runs/v2.9.1/launch/collsplit__L312.json; tilt<=60deg (n486): reach 0.9753, collision 0.0123, band_lower 0.0021; tilt>60deg (n1514): reach 0.9465, collision 0.0297, band_lower 0.0231; partition from per-episode dump, artifact data/runs/v2.9.2/mppi/perepisode__L312_PARTITION_V292.npz, partition from reproduction-gated re-score (aggregates reproduced L312's artifact at 0.000e+00)

<a id="v2-9-1--quadrotor-planar--l132-current-basis-re-score"></a>
## v2.9.1 · quadrotor_planar · L132 current-basis re-score

`anchor: v2-9-1--quadrotor-planar--l132-current-basis-re-score` · ledger line 313 at the time of writing · cps `**0.8519**` · date `**2026-08-14 09:41:00**`

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L132, which 06_workflow 2.5 holds flagged until re-scored. The checkpoint L132 names by digest -- 'v2.7.0 iter-5 secured best.pt (3b27d691)' -- was LOCATED at data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt, sha256 prefix 3b27d691 matching the row exactly, 5310629 bytes, step 48000; it strict-loads under current code at obs_dim 22. (check_ledger's warning 9 fires because that check matches run-id patterns, not digests; the checkpoint IS on disk.) BASIS THE CHECKPOINT WAS TRAINED UNDER, from its own embedded config: goal_angrate_radius ABSENT, band_hazard/band_collision_limit ABSENT, filter.projection ABSENT, hazard ABSENT, network.value.ceiling ABSENT -> 1.0. Its scoring PREDATES the angular-rate terminal leg and the cps_v2 transition; the vertical-surface transition is 3D-only and structurally irrelevant here. ONE DOCUMENTED ADDITION beyond gate_overrides: env.goal_angrate_radius = 0.3, because the checkpoint's config lacks it and outcomes.py:119 would default it to +inf, giving a TWO-leg reach predicate instead of the three-leg predicate L305-L312 run. On quadrotor_planar this leg BINDS (angular_rate is real, quadrotor_planar.py:110), so the predicate here is strictly tighter than the one that produced L132's 0.9036. RESULT: 0.851927 against L132's 0.9036, i.e. -0.051673 -- the current-basis figure is BELOW the bold, by 6.2x the 0.0083 admissibility floor. Single seed 42; the checkpoint set supports no other seed for this row. No CI computed by this producer. BOLD MOVED HERE AT THE v2.9.1 CLOSE (Researcher decision, 2026-08-14): this row carries the quadrotor_planar bold and supersedes L132, which keeps its content and loses bold. The move is a BASIS re-score of the SAME checkpoint (3b27d691) onto the current cell and pool -- NOT a 04_eval 5 CI-separated beat and not an improvement claim, since 0.851927 is BELOW L132's 0.9036. The natural successors L311 (0.775740) and L306 (0.659150) sit in secured_data/v2.9.1/experiments/, which 06_workflow 6.3 makes bold-INELIGIBLE. Same pool and cell as L306/L311, so the three are directly comparable; single seed on all three. Detail docs/versions/v2.9.1/bold_rescore.md**

### relocated from the `verdict` column (pre-cap text, verbatim)

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L132: the same checkpoint L132 names by digest (3b27d691, located at data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt and strict-loading under current code at obs_dim 22), scored on the current cell and pool. RESULT 0.851927 against L132's 0.9036, i.e. -0.051673, BELOW the bold by 6.2x the 0.0083 admissibility floor. One documented addition beyond gate_overrides, env.goal_angrate_radius 0.3, makes the reach predicate strictly TIGHTER than the one that produced 0.9036, so this row is comparable to L306 and L311 on the same pool and cell but NOT to L132's own figure; single seed 42, no CI. BOLD MOVED HERE at the v2.9.1 close (Researcher decision, 2026-08-14), superseding L132, which keeps its content and loses bold -- a BASIS re-score of the same checkpoint, NOT a 04_eval 5 CI-separated beat and not an improvement claim, the natural successors L311 and L306 being bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md and docs/ledger_verdicts.md#v2-9-1--quadrotor-planar--l132-current-basis-re-score**

### relocated from the `eval_source` column (pre-cap text, verbatim)

**eval_only(current-basis re-score of the L132 checkpoint, registered cell + explicit env.goal_angrate_radius 0.3, pool eval_full_quadrotor-planar_n2000_seed23456 (n 2000) -- the SAME pool and cell as L306/L311, ebs 2000, terminal (0.15, 0.3, 0.3), the checkpoint L132 names by digest 3b27d691 @ step 48000 (path in the verdict); cps recomputed from its own components with residual +2.22e-16); artifact data/runs/v2.9.1/launch/boldrescore__planar_iter5.json**

<a id="v2-9-1--double-integrator--l62-current-basis-re-score"></a>

**BOLD REMOVED 2026-08-20.** Bold moved to the 400-step cold-start cells on Researcher
approval (L358 takes quadrotor_planar). This row is retained for provenance and is not un-registered; per
`06_workflow` 2.5 historical rows never retain bold. It was scored at `eval.max_steps` **200**,
off the 400-step basis the paper reports, which is the condition STATUS 5 recorded as blocking
any bold change and which this move resolves.
## v2.9.1 · double_integrator · L62 current-basis re-score

`anchor: v2-9-1--double-integrator--l62-current-basis-re-score` · ledger line 314 at the time of writing · cps `**0.8682**` · date `**2026-08-14 09:41:00**`

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62, whose own verdict records 'cps_v2 reads -, so this row is not rankable under the current definition -- bold retained but pending re-scoring'. L62 NAMES NO CHECKPOINT: its cells name a 3-SEED AGGREGATE (seeds 42, 12345, 99; D_pi=2k + D_V=1M @50k) and a parent of '-'. What identifies one is the per-seed secured snapshot, and all three were LOCATED: data/secured_data/v2.3.0/seed{42,12345,99}/checkpoints/best.pt, 5273637 bytes each, best_step 43500 / 49500 / 46500 respectively; all three strict-load under current code at obs_dim 19. BASIS, from their own embedded configs: goal_angrate_radius ABSENT, band keys ABSENT, filter.projection ABSENT, hazard ABSENT, ceiling ABSENT -> 1.0; scoring PREDATES the angular-rate terminal and the cps_v2 transition. env.goal_angrate_radius = 0.3 was added explicitly to match the L305-L312 cell; on double_integrator this is STRUCTURALLY INERT, since angular_rate is a hard zero (double_integrator.py:69-71), so the third leg is satisfied identically either way. POOL CAVEAT, load-bearing: scored on eval_full_di_n2000_seed123456, the v2.9.1 REGENERATED pool, which is what L307/L309 use and is therefore what makes this row comparable to them -- but it is NOT the DI pool of record, because the 04_eval 6.2 registration delta is drafted and uninstalled. This row is therefore comparable to L307/L309 and NOT to L62 itself or to any historical DI row. RESULT: 3-seed mean 0.868247 (per seed 0.867112 / 0.876933 / 0.860695; sd 0.008178) against L62's 0.8698, i.e. -0.001553 -- BELOW the bold, by 0.19x the 0.0083 admissibility floor, which is well inside the seed spread and is NOT a separation. BOLD MOVED HERE AT THE v2.9.1 CLOSE (Researcher decision, 2026-08-14): this row carries the double_integrator bold and supersedes L62, which keeps its content and loses bold. The move is a BASIS re-score of the SAME 3-seed checkpoint set onto the current cell -- NOT a 04_eval 5 CI-separated beat and not an improvement claim, since 0.868247 is BELOW L62's 0.8698 by 0.19x the admissibility floor and inside the seed spread. The natural successor L309 (0.904757) sits in secured_data/v2.9.1/experiments/, bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md**

### relocated from the `verdict` column (pre-cap text, verbatim)

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62, which names no checkpoint but a 3-SEED AGGREGATE; all three secured snapshots (seeds 42, 12345, 99) were located and strict-load under current code at obs_dim 19. RESULT 3-seed mean 0.868247 (sd 0.008178) against L62's 0.8698, i.e. -0.001553, BELOW the bold by 0.19x the 0.0083 admissibility floor, well inside the seed spread and NOT a separation. Scored on the v2.9.1 REGENERATED DI pool, which makes this row comparable to L307 and L309 and NOT to L62 itself or to any historical DI row, since that pool is not yet the pool of record. BOLD MOVED HERE at the v2.9.1 close (Researcher decision, 2026-08-14), superseding L62, which keeps its content and loses bold -- a BASIS re-score, not an improvement claim, the natural successor L309 being bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md and docs/ledger_verdicts.md#v2-9-1--double-integrator--l62-current-basis-re-score**

### relocated from the `eval_source` column (pre-cap text, verbatim)

**eval_only(current-basis re-score of the L62 3-seed checkpoint set, registered cell + explicit env.goal_angrate_radius 0.3, pool eval_full_di_n2000_seed123456 (n 2000, the REGENERATED pool -- same as L307/L309, NOT the pool of record) -- ebs 2000, terminal (0.15, 0.3, 0.3), the three secured seed snapshots @ steps 43500/49500/46500 (paths in the verdict); 3-seed mean of per-seed cps, each recomputed from its own components, max residual 1.1e-16); artifacts data/runs/v2.9.1/launch/boldrescore__di_seed42.json, data/runs/v2.9.1/launch/boldrescore__di_seed12345.json, data/runs/v2.9.1/launch/boldrescore__di_seed99.json; pool registration delta pending (Researcher-approved, deferred)**

<a id="v2-9-1--quadrotor-3d--best-pt-valonly3d-v291"></a>

**BOLD REMOVED 2026-08-20.** Bold moved to the 400-step cold-start cells on Researcher
approval (L356 takes double_integrator). This row is retained for provenance and is not un-registered; per
`06_workflow` 2.5 historical rows never retain bold. It was scored at `eval.max_steps` **200**,
off the 400-step basis the paper reports, which is the condition STATUS 5 recorded as blocking
any bold change and which this move resolves.
## v2.9.1 · quadrotor_3d · best.pt VALONLY3D_V291

`anchor: v2-9-1--quadrotor-3d--best-pt-valonly3d-v291` · ledger line 315 at the time of writing · cps `0.8490` · date `2026-08-14 16:26:00`

VALUE-ONLY CONTINUATION of L312: 3000 further macro steps (9450 -> 12450) with the POLICY FROZEN and the value still learning, to ask whether the certificate at step 9450 had converged to that policy or was still lagging it -- the one question the joint runs cannot answer about themselves. Freeze mechanism: training.jt.K_pi 1 -> 0, which the guard at jt_pncbf/train.py:742 (`step > vs_warmup_steps and k_pi > 0`) turns into a total suppression of the policy block WITHOUT touching vs_warmup_steps, which would also move the schedule clock (train.py:669); verified from the artifact -- L_pi_total and grad_norm_pi are identically 0.0 on every one of the 151 metric rows. Config LOADED from L312's own persisted config with three keys changed: K_pi (behaviour), n_steps 10000 -> 12450 (the loop bound expressing 3000 steps), and value_init_ckpt -> None, which train.py:523 REQUIRES because value_init_ckpt and resume_ckpt are mutually exclusive and the resume supplies both value states from L312's own checkpoint anyway (train.py:490-491). SCHEDULE VERIFIED FROZEN before launch: effective_steps 4800 and the run resumes at schedule_step 9250, already 1.9x past the end, so lambda_disc holds 0.050000 and target_rhs holds 0.900000 at BOTH 9450 and 12450 -- the OC-condition curriculum is not re-run. NOT CARRIED across the resume and named before launch: the replay buffers (make_replay_buffers is called fresh, train.py:604) and ContinuingRolloutState, so the first rounds refill both from empty. With no policy step the V_S gradient-leak halt is DEAD (it sits inside the k_pi>0 block, train.py:761-765), leaving nan_or_inf_L_V the only substantive live halt; cps_floor has no JT reader at all. Ran to term, phase done, halt_reason null, 3000/3000 steps. REGISTERED HYPOTHESIS FALSIFIED, on both named columns and in the wrong direction: infeasibility ROSE 0.079209 -> 0.081508 (+0.002299, 0.28x the 0.0083 floor) and the EMPTY episode share ROSE 0.5830 -> 0.5995 (+0.0165, 1.99x the floor); the falsifier reads 'neither moves beyond the floor, or either rises' and both rose. So freezing the policy did NOT make the filter's infeasible states rarer. cps is reported and is NOT the falsifier (it mixes channels this axis does not target): 0.849048 against L312's 0.867987, -0.018939; against the bold L304 0.872223 it is -0.023175 and does NOT exceed it. Selection note: best_step is 12450, the FINAL step, and the last 7 in-loop evals oscillate in a 0.028-wide band (0.798-0.834) after a near-monotone rise over the first ten, so the selected point is the maximum of a plateau, not of a still-rising curve. Single seed 42, no CI. No bold change, no promotion; detail docs/versions/v2.9.1/value_only_continuation.md

### relocated from the `verdict` column (pre-cap text, verbatim)

VALUE-ONLY CONTINUATION of L312: 3000 further macro steps (9450 -> 12450) with the POLICY FROZEN via training.jt.K_pi 1 -> 0 and the value still learning, asking whether the certificate at step 9450 had converged to that policy or was still lagging it. REGISTERED HYPOTHESIS FALSIFIED on both named columns and in the wrong direction -- infeasibility ROSE 0.079209 -> 0.081508 and the EMPTY episode share ROSE 0.5830 -> 0.5995 -- so freezing the policy did NOT make the filter's infeasible states rarer. cps 0.849048 is reported and is NOT the falsifier, since it mixes channels this axis does not target; against L312 it is -0.018939 and against the bold L304 -0.023175, exceeding neither. Single seed 42, no CI, and best_step is the FINAL step, the maximum of a 0.028-wide plateau rather than of a still-rising curve. No bold change, no promotion; detail docs/versions/v2.9.1/value_only_continuation.md and docs/ledger_verdicts.md#v2-9-1--quadrotor-3d--best-pt-valonly3d-v291.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000) -- the SAME pool and cell as L304, L305 and L312, ebs 2000, terminal (0.15, 0.3, 0.3), best.pt @ step 12450 = round 1245; value-only continuation of L312 with the policy frozen (K_pi 0) over steps 9450 -> 12450; cps recomputed from its own components with residual -1.11e-16); artifact data/runs/v2.9.1/launch/valonlyrow__quadrotor_3d.json

<a id="v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292"></a>
## v2.9.2 · quadrotor_3d · best.pt OC3D_CGAIN0_V292

`anchor: v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292` · ledger line 316 at the time of writing · cps `0.6194` · date `2026-08-15 01:28:32`

v2.9.2 REGISTERED AXIS, OC condition: env.quadrotor_3d.c_gain 0.3 -> 0.0, the horizontal approach-term ablation, derived from L305's OWN persisted config with EXACTLY ONE flattened key changed of 329 (gate 1) and the term's absence proven functionally on 12 probe states with nonzero closing speed (gate 2: value_target_barrier - signed_h = 0.0 BIT-EXACT here, 0.5262 on the parent). A1 (coll_obstacle must exceed the parent's) HOLDS on this condition: 0.0310 -> 0.0500, +0.0190, 62 -> 100 episodes of 2000. The vertical band channels are UNCHANGED to the episode: coll_band_lower 53 episodes on both rows, coll_band_upper 0 on both -- the ablation moved the horizontal channel and left the vertical terms alone, which is what the registered hypothesis predicted; band channels carry NO registered verdict. cps 0.6194 against the parent's 0.6818 is reported, not claimed: cps is not the registered observable and no CI exists on either row. Single seed 42. Ran to budget (30000 steps, phase done, halt_reason null); best_step is 30000, the FINAL eval, so this checkpoint was still improving when the budget ended. NOT bold, NOT promoted as SOTA -- a 06_workflow 6.3 experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md

### relocated from the `verdict` column (pre-cap text, verbatim)

v2.9.2 REGISTERED AXIS, OC condition: env.quadrotor_3d.c_gain 0.3 -> 0.0, the horizontal approach-term ablation, derived from L305's OWN persisted config with EXACTLY ONE flattened key changed of 329 and the term's absence proven functionally on 12 probe states. A1 HOLDS on this condition -- coll_obstacle 0.0310 -> 0.0500, 62 -> 100 episodes of 2000 -- while the vertical band channels are UNCHANGED to the episode, which is what the registered hypothesis predicted. cps 0.6194 against the parent L305's 0.6818 is reported and NOT claimed, since cps is not the registered observable and no CI exists on either row; single seed 42, and best_step is the final eval, so this checkpoint was still improving when the budget ended. NOT bold, NOT promoted as SOTA -- a 06_workflow 6.3 experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000), ebs 2000, terminal (0.15, 0.3, 0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0, 100.0), gamma_margin 0.0, value ceiling 1.3, best.pt @ step 30000 = round 1000; c_gain of the SCORED CHECKPOINT read back off its own embedded config = 0.0; scored TWICE by independent framework loads and all nine fields reproduce to abs(A-B) = 0.000e+00 (reproduction gate, 06_workflow 2.5) before the surface split; the three surface components sum to collision with residual +0.000e+00 and 0 unattributed episodes); artifact data/runs/v2.9.2/launch/row__OC3D_CGAIN0_V292__step30000.json

<a id="v2-9-2--quadrotor-3d--mppi-cascrate-n2000-v292-cascaded-rate-privileged-baseline"></a>
## v2.9.2 · quadrotor_3d · MPPI_CASCRATE_N2000_V292 (cascaded_rate, PRIVILEGED baseline)

`anchor: v2-9-2--quadrotor-3d--mppi-cascrate-n2000-v292-cascaded-rate-privileged-baseline` · ledger line 317 at the time of writing · cps `n/a` · date `2026-08-15 01:42:03`

PRIVILEGED MODEL-BASED REFERENCE POINT, NOT A PEER OF A LEARNED-CERTIFICATE ROW. MPPI reads the full 13-D state and the exact obstacle field and rolls the true plant inside its own planner; the learned controllers see a 34-D observation and carry no plant model. It runs an identity filter, so infeasibility and mean_proj_mag are STRUCTURALLY INAPPLICABLE rather than zero. This is the FIRST MPPI row in the ledger and the first MPPI evaluation at n 2000; every prior MPPI figure on record was 400 scenes. Reach 0.2980, collision 0.6525 of which obstacle 0.3280 and band_lower 0.3235 -- the arena floor accounts for essentially half the collisions. TILT PARTITION at 60 deg (the altitude-holding limit at TWR 2, not a tuned threshold), arithmetic on the per-episode dump and not on aggregates: tilt<=60 n 486 reach 0.4198 collision 0.5144 (obstacle 0.4074, band_lower 0.1029); tilt>60 n 1514 reach 0.2589 collision 0.6968 (obstacle 0.3025, band_lower 0.3943). One run, single seed 42, no CI, no selection. Bears on docs/versions/v2.9.2/mppi_root_cause.md 4-iv and on nothing else. NOT bold, NOT promoted. Detail docs/versions/v2.9.2/mppi_registered.md

### relocated from the `verdict` column (pre-cap text, verbatim)

PRIVILEGED MODEL-BASED REFERENCE POINT, NOT A PEER of a learned-certificate row: MPPI reads the full 13-D state and the exact obstacle field and rolls the true plant inside its own planner, where the learned controllers see a 34-D observation and carry no plant model. Reach 0.2980 and collision 0.6525, of which obstacle 0.3280 and band_lower 0.3235, so the arena floor accounts for essentially half the collisions. NOT comparable on infeasibility or mean_proj_mag at all, which are STRUCTURALLY INAPPLICABLE rather than zero because it runs an identity filter; this is the FIRST MPPI row in the ledger and the first at n 2000, every prior figure on record being 400 scenes; one run, single seed 42, no CI, no selection. NOT bold, NOT promoted; it bears on docs/versions/v2.9.2/mppi_root_cause.md 4-iv and on nothing else. Detail docs/versions/v2.9.2/mppi_registered.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--mppi-cascrate-n2000-v292-cascaded-rate-privileged-baseline.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(MPPI baseline at the REGISTERED scale, variant cascaded_rate, pool fullcb (sha8 3682a4e3, n 2000 -- the FULL pool, not a prefix), n_scenes 2000, ebs 200 -- A CELL DIFFERENCE against the learned rows L305-L312, which score at ebs 2000; same pool, same n, same seed 42, different eval batch. Settings verbatim from cell__3C1_control: sigma_channel [0.25,0.15,0.15,0.05] read back off the constructed controller, lam 1.0 absolute, c_crash 1e5, N 1024, H 40, all seven switches false, the scalar sigma INERT under the per-channel override. ANCHOR: the 400-scene prefix reproduces cell__3C1_control's outcome counts EXACTLY, collision 263 / goal 123 / timeout 14. infeasibility and mean_proj_mag are STRUCTURALLY INAPPLICABLE -- MPPI runs an identity filter, forms no half-space and runs no projection, so there is no quantity to report; they are NOT zero-by-measurement. cps and cps_v2 are n/a: filter semantics are out of scope for this condition (06_workflow 2.5) and NO composite headline is claimed. Five outcome shares sum to 1.000000000000; the three surface components sum to collision with residual +0e+00 and 0 unattributed episodes); artifact data/runs/v2.9.2/mppi/row__MPPI_CASCRATE_N2000_V292__n2000.json; tilt<=60deg (n486): reach 0.4198, collision 0.5144, band_lower 0.1029; tilt>60deg (n1514): reach 0.2589, collision 0.6968, band_lower 0.3943; partition from per-episode dump, artifact data/runs/v2.9.2/mppi/perepisode__MPPI_CASCRATE_N2000_V292.npz

<a id="v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292"></a>
## v2.9.2 · quadrotor_3d · best.pt JT3D_CGAIN0_V292

`anchor: v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292` · ledger line 318 at the time of writing · cps `0.8495` · date `2026-08-15 04:28:14`

v2.9.2 REGISTERED AXIS, JT condition: env.quadrotor_3d.c_gain 0.3 -> 0.0, derived from L312's OWN persisted config with EXACTLY TWO flattened keys changed of 328 (gate 1) -- c_gain and training.jt.value_init_ckpt, repointed to this version's OC c_gain-0 cell so the condition is not warm-started from a c_gain-0.3 certificate; the term's absence proven functionally on 12 probe states (gate 2: value_target_barrier - signed_h = 0.0 BIT-EXACT here, 0.5262 on L312). A1 (coll_obstacle must exceed the parent's) HOLDS on this condition: 0.0075 -> 0.0090, +0.0015, 15 -> 18 episodes of 2000. WITH THE OC CONDITION (L316: 0.0310 -> 0.0500) A1 HOLDS ON BOTH CONDITIONS. coll_band_lower 0.0180 -> 0.0160 and coll_band_upper 0.0000 on both rows carry NO registered verdict. cps 0.8495 against L312's 0.8680 is reported, not claimed: cps is not the registered observable and no CI exists on either row. INITIALIZER ASYMMETRY, fact not defect: this condition warm-starts from a BUDGET-TRUNCATED checkpoint (the OC c_gain-0 cell's best_step is 30000, its final eval, still rising), where L312 warm-started from a CONVERGED peak (27450, six evals past it); the budget was held equal and no extension was taken. Single seed 42. Ran to budget (10000 steps, phase done, halt_reason null), best_step 9300 with six evals after it. NOT bold, NOT promoted as SOTA -- a 06_workflow 6.3 experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md

### relocated from the `verdict` column (pre-cap text, verbatim)

v2.9.2 REGISTERED AXIS, JT condition: env.quadrotor_3d.c_gain 0.3 -> 0.0, derived from L312's OWN persisted config with EXACTLY TWO flattened keys changed of 328, the second repointing value_init_ckpt to this version's OC c_gain-0 cell so the condition is not warm-started from a c_gain-0.3 certificate. A1 HOLDS on this condition -- coll_obstacle 0.0075 -> 0.0090, 15 -> 18 episodes of 2000 -- and WITH THE OC CONDITION L316 IT HOLDS ON BOTH CONDITIONS. cps 0.8495 against L312's 0.8680 is reported and NOT claimed, since cps is not the registered observable and no CI exists on either row; INITIALIZER ASYMMETRY, a fact and not a defect, is that this condition warm-starts from a BUDGET-TRUNCATED checkpoint where L312 warm-started from a converged peak, the budget having been held equal; single seed 42. NOT bold, NOT promoted as SOTA -- a 06_workflow 6.3 experiments-class ablation cell. Detail docs/versions/v2.9.2/cgain_cells.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(registered cell, pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000), ebs 2000, terminal (0.15, 0.3, 0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0, 100.0), gamma_margin 0.0, value ceiling 1.3, best.pt @ step 9300 = round 930; c_gain of the SCORED CHECKPOINT read back off its own embedded config = 0.0; scored TWICE by independent framework loads and all nine fields reproduce to abs(A-B) = 0.000e+00 (reproduction gate, 06_workflow 2.5) before the surface split; the three surface components sum to collision with residual +0.000e+00 and 0 unattributed episodes); artifact data/runs/v2.9.2/launch/row__JT3D_CGAIN0_V292__step9300.json

<a id="v2-9-2--quadrotor-3d--m1-mismatch-pi-l312-v-l305"></a>
## v2.9.2 · quadrotor_3d · M1 MISMATCH pi(L312) + V(L305)

`anchor: v2-9-2--quadrotor-3d--m1-mismatch-pi-l312-v-l305` · ledger line 319 at the time of writing · cps `0.6925` · date `2026-08-16 02:35:55`

v2.9.2 MISMATCH PREDICTION TEST. V is V^pi, so this pairing is a WRONG CERTIFICATE, not a controlled variation, and NOT A VALID DEPLOYMENT -- no configuration here is proposed, adopted or recommended. PREDICTION HOLDS on this cell: cps 0.6925 against its own policy's matched cell MATCHED_JT 0.8680, i.e. -0.1755, a degradation far beyond the 0.0083 floor. coll_obstacle 0.0075 -> 0.0555 (15 -> 111 episodes of 2000) is where it goes; coll_band_lower is UNCHANGED at 0.0265. Single seed 42, no CI on any of the four cells. NOT bold, NOT promoted. Detail docs/versions/v2.9.2/decomp.md

### relocated from the `verdict` column (pre-cap text, verbatim)

v2.9.2 MISMATCH PREDICTION TEST, pairing L312's policy with L305's certificate. PREDICTION HOLDS on this cell: cps 0.6925 against its own policy's matched cell MATCHED_JT 0.8680, i.e. -0.1755, a degradation far beyond the 0.0083 floor, and coll_obstacle 0.0075 -> 0.0555 (15 -> 111 episodes of 2000) is where it goes while coll_band_lower is UNCHANGED. V is V^pi, so this is a WRONG CERTIFICATE rather than a controlled variation and NOT A VALID DEPLOYMENT, comparable only to the other three mismatch cells; single seed 42, no CI on any of the four. NOT bold, NOT promoted, and no configuration here is proposed, adopted or recommended. Detail docs/versions/v2.9.2/decomp.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--m1-mismatch-pi-l312-v-l305.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(MISMATCHED nominal-certificate pairing, PREDICTION TEST NOT A DEPLOYMENT; POLICY from L312's pi_state (the JT policy); VALUE from L305's v_s_state (the OC certificate); state provenance asserted by parameter fingerprint off the CONSTRUCTED framework before the cell ran. Registered cell, pool fullcb (sha8 3682a4e3, n 2000), ebs 2000, terminal (0.15, 0.3, 0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0, 100.0), gamma_margin 0.0, ceiling 1.3, hazard exp/0.125 -- the same producer that made L305 and L312, whose matched re-scores in this same run reproduced EVERY outcome field of both rows to <1e-12 before this cell ran. saturation_rate 0.3643 is recorded here; it is BLANK on L305 and L312 because their producers never copied it from eval_row. EMPTY 0.5610 and SINGULAR 0.1100 are EPISODE-axis fractions and are never summed with the step-axis infeasibility. mean live steps 70.71); artifact data/runs/v2.9.2/mismatch/row__M1.json; per-step dump data/runs/v2.9.2/mismatch/perstep__M1.npz

<a id="v2-9-2--quadrotor-3d--m2-mismatch-shipped-lqr-nominal-v-l312"></a>
## v2.9.2 · quadrotor_3d · M2 MISMATCH shipped LQR nominal + V(L312)

`anchor: v2-9-2--quadrotor-3d--m2-mismatch-shipped-lqr-nominal-v-l312` · ledger line 320 at the time of writing · cps `0.8136` · date `2026-08-16 02:36:00`

v2.9.2 MISMATCH PREDICTION TEST. V is V^pi, so this pairing is a WRONG CERTIFICATE, not a controlled variation, and NOT A VALID DEPLOYMENT -- no configuration here is proposed, adopted or recommended. PREDICTION FALSIFIED on this cell, and this is the finding: cps 0.8136 against its own policy's matched cell MATCHED_OC 0.6818, i.e. +0.1317 -- the mismatched certificate EXCEEDS the matched one by 15.9x the 0.0083 floor. The same LQR nominal scores higher under L312's certificate than under the L305 certificate trained against it. On this system and this pool the policy-dependence premise does NOT hold in this direction. Single seed 42, no CI on any of the four cells. NOT bold, NOT promoted. Detail docs/versions/v2.9.2/decomp.md

### relocated from the `verdict` column (pre-cap text, verbatim)

v2.9.2 MISMATCH PREDICTION TEST, pairing the shipped LQR nominal with L312's certificate. PREDICTION FALSIFIED on this cell, and this is the finding: cps 0.8136 against its own policy's matched cell MATCHED_OC 0.6818, i.e. +0.1317, so the mismatched certificate EXCEEDS the matched one by 15.9x the 0.0083 floor and on this system and pool the policy-dependence premise does NOT hold in this direction. V is V^pi, so this is a WRONG CERTIFICATE rather than a controlled variation and NOT A VALID DEPLOYMENT, comparable only to the other three mismatch cells; single seed 42, no CI on any of the four. NOT bold, NOT promoted, and no configuration here is proposed, adopted or recommended. Detail docs/versions/v2.9.2/decomp.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--m2-mismatch-shipped-lqr-nominal-v-l312.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(MISMATCHED nominal-certificate pairing, PREDICTION TEST NOT A DEPLOYMENT; POLICY from the shipped LQR nominal system.lqr_action (L305 carries NO pi_state, so this is the policy its own OC deployment uses); VALUE from L312's v_s_state (the JT certificate); state provenance asserted by parameter fingerprint off the CONSTRUCTED framework before the cell ran. Registered cell, pool fullcb (sha8 3682a4e3, n 2000), ebs 2000, terminal (0.15, 0.3, 0.3), projection dual_solve, empty_fallback {kstep,phases 1,k 3}, alpha (2.0, 100.0), gamma_margin 0.0, ceiling 1.3, hazard exp/0.125 -- the same producer that made L305 and L312, whose matched re-scores in this same run reproduced EVERY outcome field of both rows to <1e-12 before this cell ran. saturation_rate 0.0942 is recorded here; it is BLANK on L305 and L312 because their producers never copied it from eval_row. EMPTY 0.4915 and SINGULAR 0.1260 are EPISODE-axis fractions and are never summed with the step-axis infeasibility. mean live steps 128.84); artifact data/runs/v2.9.2/mismatch/row__M2.json; per-step dump data/runs/v2.9.2/mismatch/perstep__M2.npz

<a id="v2-9-2--quadrotor-3d--best-pt-oc3d-v291-at-h400"></a>
## v2.9.2 · quadrotor_3d · best.pt OC3D_V291 at h400

`anchor: v2-9-2--quadrotor-3d--best-pt-oc3d-v291-at-h400` · ledger line 321 at the time of writing · cps `0.7330` · date `2026-08-18 00:17:18`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-164148__seed42/v2.9.1__oc__20260813-164148__seed42/checkpoints/best.pt @ step 27450, the same checkpoint L305 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L305: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L305 in all 20 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions, plus the collision split and its per-surface counts; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.051139, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.038552, +0.063224], EXCLUDES zero, 6.16x the 0.0083 admissibility floor, 140 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 72 ep = 0.0360 here against 149 ep = 0.0745 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 324.7 while their mean infeasible steps rise 14.07 to 19.36, i.e. per-step infeasible rate 0.0703 over the first 200 steps against 0.0425 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.059326, singular as coded 0.042844, of which singular-and-violated 0.000417 and singular-and-satisfied 0.042427; the union 0.101753 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.059326 and its cps 0.745702, a shift of +0.012728. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.4855 and SINGULAR 0.1680 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.4845 and 0.1675. Collision decomposed by terminating surface (collision_cause, priority obstacle > band_lower > band_upper): obstacle 0.0325 (65 ep), band_lower 0.0265 (53 ep), band_upper 0.0000 (0 ep), unattributed 0 ep; the three sum to this row's collision with residual +0.00e+00. sat_rate 0.1093 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.647000 (reach 0.4510, collision 0.5490) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L305's, on L305's own pool and checkpoint (quadrotor_3d OC). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.051139 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.038552, +0.063224] at the shipped defaults, 140 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L305, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L305 in all 20 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-oc3d-v291-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 27450, the checkpoint L305 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L305_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L305_h400.json, per-episode dumps perepisode__L305_h200.npz and perepisode__L305_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--quadrotor-planar--best-pt-ocplanar-v291-r3-at-h400"></a>
## v2.9.2 · quadrotor_planar · best.pt OCPLANAR_V291_R3 at h400

`anchor: v2-9-2--quadrotor-planar--best-pt-ocplanar-v291-r3-at-h400` · ledger line 322 at the time of writing · cps `0.6963` · date `2026-08-18 00:17:24`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-164154__seed42/v2.9.1__oc__20260813-164154__seed42/checkpoints/best.pt @ step 28350, the same checkpoint L306 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L306: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L306 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.037161, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.025654, +0.049372], EXCLUDES zero, 4.48x the 0.0083 admissibility floor, 145 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 128 ep = 0.0640 here against 193 ep = 0.0965 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 350.5 while their mean infeasible steps rise 7.46 to 10.01, i.e. per-step infeasible rate 0.0373 over the first 200 steps against 0.0170 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.036129, singular as coded 0.042835, of which singular-and-violated 0.000000 and singular-and-satisfied 0.042835; the union 0.078964 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.036129 and its cps 0.709161, a shift of +0.012850. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.3885 and SINGULAR 0.2315 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.3885 and 0.2305. This system has no vertical band, and every collision on it is attributed to the obstacle channel (104 of 104 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.0443 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.206000 (reach 0.5980, collision 0.4020) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L306's, on L306's own pool and checkpoint (quadrotor_planar OC). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.037161 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.025654, +0.049372] at the shipped defaults, 145 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L306, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L306 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--quadrotor-planar--best-pt-ocplanar-v291-r3-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_quadrotor-planar_n2000_seed23456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 28350, the checkpoint L306 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L306_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L306_h400.json, per-episode dumps perepisode__L306_h200.npz and perepisode__L306_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--double-integrator--best-pt-ocdi-v291-at-h400"></a>
## v2.9.2 · double_integrator · best.pt OCDI_V291 at h400

`anchor: v2-9-2--double-integrator--best-pt-ocdi-v291-at-h400` · ledger line 323 at the time of writing · cps `0.8822` · date `2026-08-18 00:17:29`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-170950__seed42/v2.9.1__oc__20260813-170950__seed42/checkpoints/best.pt @ step 25200, the same checkpoint L307 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L307: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L307 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.001764, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.000214, +0.004028], EXCLUDES zero, 0.21x the 0.0083 admissibility floor, 81 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 90 ep = 0.0450 here against 92 ep = 0.0460 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 396.9 while their mean infeasible steps rise 30.60 to 52.24, i.e. per-step infeasible rate 0.1530 over the first 200 steps against 0.1099 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.017500, singular as coded 0.020861, of which singular-and-violated 0.000000 and singular-and-satisfied 0.020861; the union 0.038361 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.017500 and its cps 0.888500, a shift of +0.006277. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.2365 and SINGULAR 0.1965 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.2365 and 0.1965. This system has no vertical band, and every collision on it is attributed to the obstacle channel (11 of 11 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.1426 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.009500 (reach 0.6635, collision 0.3365) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L307's, on L307's own pool and checkpoint (double_integrator OC). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.001764 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.000214, +0.004028] at the shipped defaults, 81 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L307, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L307 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--double-integrator--best-pt-ocdi-v291-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_di_n2000_seed123456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 25200, the checkpoint L307 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L307_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L307_h400.json, per-episode dumps perepisode__L307_h200.npz and perepisode__L307_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--unicycle--best-pt-ocuni-v291-at-h400"></a>
## v2.9.2 · unicycle · best.pt OCUNI_V291 at h400

`anchor: v2-9-2--unicycle--best-pt-ocuni-v291-at-h400` · ledger line 324 at the time of writing · cps `0.6753` · date `2026-08-18 00:17:34`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-170955__seed42/v2.9.1__oc__20260813-170955__seed42/checkpoints/best.pt @ step 14400, the same checkpoint L308 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L308: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L308 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.004828, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.000181, +0.009899], EXCLUDES zero, 0.58x the 0.0083 admissibility floor, 189 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 299 ep = 0.1495 here against 309 ep = 0.1545 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 394.5 while their mean infeasible steps rise 10.02 to 19.59, i.e. per-step infeasible rate 0.0501 over the first 200 steps against 0.0492 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.010785, singular as coded 0.055089, of which singular-and-violated 0.000000 and singular-and-satisfied 0.055089; the union 0.065874 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.010785 and its cps 0.691765, a shift of +0.016451. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.2295 and SINGULAR 0.3240 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.2215 and 0.3180. This system has no vertical band, and every collision on it is attributed to the obstacle channel (4 of 4 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.1627 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps 0.002000 (reach 0.6670, collision 0.3320) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L308's, on L308's own pool and checkpoint (unicycle OC). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.004828 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.000181, +0.009899] at the shipped defaults, 189 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L308, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L308 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--unicycle--best-pt-ocuni-v291-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_unicycle_n2000_seed123456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 14400, the checkpoint L308 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L308_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L308_h400.json, per-episode dumps perepisode__L308_h200.npz and perepisode__L308_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--double-integrator--best-pt-jtdi-v291-r2-at-h400"></a>
## v2.9.2 · double_integrator · best.pt JTDI_V291_R2 at h400

`anchor: v2-9-2--double-integrator--best-pt-jtdi-v291-r2-at-h400` · ledger line 325 at the time of writing · cps `0.9055` · date `2026-08-18 00:17:39`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-182126__seed42/v2.9.1__jt__20260813-182126__seed42/checkpoints/best.pt @ step 9300, the same checkpoint L309 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L309: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L309 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.000745, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [-0.000053, +0.002285], INCLUDES zero, so no separation is claimed, 0.09x the 0.0083 admissibility floor, 47 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 51 ep = 0.0255 here against 52 ep = 0.0260 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 396.3 while their mean infeasible steps rise 33.33 to 66.69, i.e. per-step infeasible rate 0.1666 over the first 200 steps against 0.1699 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.020726, singular as coded 0.034266, of which singular-and-violated 0.000000 and singular-and-satisfied 0.034266; the union 0.054991 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.020726 and its cps 0.915782, a shift of +0.010280. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.2570 and SINGULAR 0.2305 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.2570 and 0.2305. This system has no vertical band, and every collision on it is attributed to the obstacle channel (18 of 18 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.7524 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.009500 (reach 0.6635, collision 0.3365) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L309's, on L309's own pool and checkpoint (double_integrator JT). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.000745 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [-0.000053, +0.002285] at the shipped defaults, 47 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L309, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L309 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--double-integrator--best-pt-jtdi-v291-r2-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_di_n2000_seed123456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 9300, the checkpoint L309 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L309_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L309_h400.json, per-episode dumps perepisode__L309_h200.npz and perepisode__L309_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--unicycle--best-pt-jtuni-v291-r2-at-h400"></a>
## v2.9.2 · unicycle · best.pt JTUNI_V291_R2 at h400

`anchor: v2-9-2--unicycle--best-pt-jtuni-v291-r2-at-h400` · ledger line 326 at the time of writing · cps `0.8732` · date `2026-08-18 00:17:45`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-182144__seed42/v2.9.1__jt__20260813-182144__seed42/checkpoints/best.pt @ step 9450, the same checkpoint L310 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L310: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L310 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.000085, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.000024, +0.000156], EXCLUDES zero, 0.01x the 0.0083 admissibility floor, 41 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 115 ep = 0.0575 here against 115 ep = 0.0575 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 400.0 while their mean infeasible steps rise 5.50 to 8.89, i.e. per-step infeasible rate 0.0275 over the first 200 steps against 0.0170 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.009498, singular as coded 0.009765, of which singular-and-violated 0.000000 and singular-and-satisfied 0.009765; the union 0.019263 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.009498 and its cps 0.876151, a shift of +0.002939. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.1830 and SINGULAR 0.1150 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.1830 and 0.1150. This system has no vertical band, and every collision on it is attributed to the obstacle channel (4 of 4 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.7350 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps 0.002000 (reach 0.6670, collision 0.3320) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L310's, on L310's own pool and checkpoint (unicycle JT). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.000085 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.000024, +0.000156] at the shipped defaults, 41 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L310, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L310 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--unicycle--best-pt-jtuni-v291-r2-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_unicycle_n2000_seed123456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 9450, the checkpoint L310 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L310_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L310_h400.json, per-episode dumps perepisode__L310_h200.npz and perepisode__L310_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--quadrotor-planar--best-pt-jtplanar-v291-r4-at-h400"></a>
## v2.9.2 · quadrotor_planar · best.pt JTPLANAR_V291_R4 at h400

`anchor: v2-9-2--quadrotor-planar--best-pt-jtplanar-v291-r4-at-h400` · ledger line 327 at the time of writing · cps `0.7894` · date `2026-08-18 00:17:51`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260813-215041__seed42/v2.9.1__jt__20260813-215041__seed42/checkpoints/best.pt @ step 9000, the same checkpoint L311 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L311: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L311 in all 14 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.013638, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.006640, +0.021657], EXCLUDES zero, 1.64x the 0.0083 admissibility floor, 100 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 135 ep = 0.0675 here against 160 ep = 0.0800 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 377.8 while their mean infeasible steps rise 8.42 to 12.44, i.e. per-step infeasible rate 0.0421 over the first 200 steps against 0.0226 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.037976, singular as coded 0.011742, of which singular-and-violated 0.000167 and singular-and-satisfied 0.011575; the union 0.049551 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.037976 and its cps 0.792857, a shift of +0.003480. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.3780 and SINGULAR 0.1085 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.3780 and 0.1085. This system has no vertical band, and every collision on it is attributed to the obstacle channel (43 of 43 ep, 0 unattributed), so the coll_ columns are left blank as on its 200-step row. sat_rate 0.1638 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.206000 (reach 0.5980, collision 0.4020) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L311's, on L311's own pool and checkpoint (quadrotor_planar JT). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.013638 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.006640, +0.021657] at the shipped defaults, 100 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L311, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L311 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--quadrotor-planar--best-pt-jtplanar-v291-r4-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_full_quadrotor-planar_n2000_seed23456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 9000, the checkpoint L311 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L311_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L311_h400.json, per-episode dumps perepisode__L311_h200.npz and perepisode__L311_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--quadrotor-3d--best-pt-jt3d-v291-ocr4-at-h400"></a>
## v2.9.2 · quadrotor_3d · best.pt JT3D_V291_OCR4 at h400

`anchor: v2-9-2--quadrotor-3d--best-pt-jt3d-v291-ocr4-at-h400` · ledger line 328 at the time of writing · cps `0.8853` · date `2026-08-18 00:18:03`

NEW CELL at eval.max_steps 400 -- the reporting basis this dispatch adopts. Checkpoint data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt @ step 9450, the same checkpoint L312 names; no checkpoint was re-selected and no config key on disk was changed. THIS ROW IS NOT COMPARABLE TO ANY 200-STEP ROW, including its own 200-step counterpart L312: the horizon moves both the outcome resolution and the infeasibility denominator (the 04_eval Active window paragraph, docs/protocol/04_eval.md:80, makes every step of a stuck or timeout episode active), so no difference taken against a 200-step row is a result. CONTROL: the same instrument scored this checkpoint at max_steps 200 first and reproduced L312 in all 20 compared fields with residual within 1e-12 -- the five outcome counts and fractions, infeasibility, cps, and the EMPTY and SINGULAR episode-axis fractions, plus the collision split and its per-surface counts; key lookups in that gate are unguarded, so a field it could not find would fail rather than be skipped. PAIRED against that same-instrument 200 control on the identical scenes: per-episode cps difference +0.017325, 04_eval section 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF at the shipped defaults (n_resample 1000, seed 20260508) [+0.010542, +0.024130], EXCLUDES zero, 2.09x the 0.0083 admissibility floor, 41 of 2000 episodes changed. INCOMPLETE (stuck plus timeout) 19 ep = 0.0095 here against 42 ep = 0.0210 at the 200 control. DENOMINATOR: mean active steps over the episodes the control left incomplete rise 200.0 to 328.3 while their mean infeasible steps rise 40.45 to 65.10, i.e. per-step infeasible rate 0.2023 over the first 200 steps against 0.1921 over the extension, so any movement in this row's infeasibility is separable into numerator and denominator and is NOT a claim that the filter found more room. INFEASIBILITY CLAUSES, recorded so this row is readable under either definition and never summed with each other: empty 0.067530, singular as coded 0.012441, of which singular-and-violated 0.001012 and singular-and-satisfied 0.011429; the union 0.078960 is what enters this row's cps and equals the shipped filter's returned flag on every step of the rollout (identity checked, 0 mismatching steps). 04_eval section 1 defines the second clause as singular AND violated and calls a singular-and-satisfied row feasible; singular-and-violated is subsumed entirely by empty on this cell (elementwise, every episode), so under that definition this row's infeasibility would be 0.067530 and its cps 0.888741, a shift of +0.003429. The shipped filter returns singular OR empty (src/common/filter_hardnet.py:144,208), the v2.4.x legacy formula. That divergence is RECORDED, NOT RESOLVED: nothing was re-scored under the other definition, no ledger row was rewritten, and src/ was not edited. EMPTY 0.5830 and SINGULAR 0.0960 are EPISODE-axis fractions (share of episodes with at least one such step over that episode's active steps) and are never summed over steps; at the 200 control they read 0.5830 and 0.0955. Collision decomposed by terminating surface (collision_cause, priority obstacle > band_lower > band_upper): obstacle 0.0075 (15 ep), band_lower 0.0180 (36 ep), band_upper 0.0000 (0 ep), unattributed 0 ep; the three sum to this row's collision with residual +0.00e+00. sat_rate 0.3804 is a recorded diagnostic per 04_eval section 1 and enters no score; the 200-step row leaves this column blank, so it reproduces nothing. NOMINAL on this same system and cell at 400, unfiltered LQR with no certificate and no filter, scores cps -0.647000 (reach 0.4510, collision 0.5490) and is reported in the deliverable but DELIBERATELY NOT REGISTERED -- that is a separate decision. Single seed 42; the interval is within-seed and scene-only, so it supports a statement about this cell's scenes and not about seeds. NO BOLD CHANGE AND NO PROMOTION: adopting the 400-step basis leaves the three bold rows on the 200-step basis and therefore off-basis, and what happens to them is the Researcher's decision, not this dispatch's. detail docs/versions/v2.9.2/horizon400_tableI.md

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400 -- the registered v282_agree_gate.gate_overrides cell with every other key identical to L312's, on L312's own pool and checkpoint (quadrotor_3d JT). Paired against a same-instrument 200-step control on identical scenes, the per-episode cps difference is +0.017325 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the DIFFERENCE ITSELF of [+0.010542, +0.024130] at the shipped defaults, 41 of 2000 episodes changed. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L312, because the horizon moves both the outcome resolution and the infeasibility denominator; that 200 control reproduced L312 in all 14 compared fields within 1e-12; single seed 42, the interval within-seed and scene-only. NOT bold and NOT promoted -- adopting this basis leaves the three bold rows off-basis, which is the Researcher's decision and not this dispatch's. Detail docs/versions/v2.9.2/horizon400_tableI.md, and the full clause record and cell read-back in docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-jt3d-v291-ocr4-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl, n 2000, ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 9450, the checkpoint L312 names -- its path is carried in the verdict, NOT here, because check_ledger rule 5 keys this block on the earliest mtime among the paths eval_source names and a checkpoint path would key the row to its training date rather than its scoring date; cps composed by the shipped evaluator src.eval.evaluate.evaluate, not recomposed downstream; artifacts data/runs/v2.9.2/horizon400_tableI/score__L312_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tableI/score__L312_h400.json, per-episode dumps perepisode__L312_h200.npz and perepisode__L312_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tableI/gate_paired.json)

<a id="v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292-at-h400"></a>
## v2.9.2 · quadrotor_3d · best.pt OC3D_CGAIN0_V292 at h400

`anchor: v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292-at-h400` · registered after the five-sentence cap, so this section carries the **full record** rather than a pre-cap verdict · cps `0.6692` · date `2026-08-18 00:54:16`

**Checkpoint** `data/runs/v2.9.2/set__20260815-011200__seed42/v2.9.2__oc__20260815-011200__seed42/checkpoints/best.pt` @ step 30000, the checkpoint L316 names; no checkpoint was re-selected and no config key on disk was changed. The horizon is a run-time override on the in-memory cell dict.

**200 control.** Reproduced L316 in all 20 compared fields with residual within 1e-12: the five outcome counts and fractions, infeasibility, cps, the EMPTY and SINGULAR episode-axis fractions, and the collision split with its per-surface counts. Key lookups in that gate are unguarded.

**Outcomes at 400** — goal 1794, collision 158, oob 0, stuck 14, timeout 34; incomplete (stuck + timeout) 48 = 0.0240, against 127 = 0.0635 at the 200 control.

**Infeasibility clauses at 400** — empty 0.065602; singular as coded 0.115737, of which singular-and-violated 0.000217 and singular-and-satisfied 0.115520; union 0.181122, which is what enters cps and equals the shipped filter's returned flag on every step (0 mismatching steps). Singular-and-violated is subsumed entirely by empty, so under the 04_eval section 1 definition this row's infeasibility would be 0.065602 and its cps 0.703819, a shift of +0.034656. At the 200 control the same clauses read empty 0.066035 and singular 0.115307.

**Episode-axis, never summed** — EMPTY 0.4805 and SINGULAR 0.3990 at 400, against 0.4805 and 0.3955 at the 200 control.

**Denominator** — over the episodes the control left incomplete (n 127), mean active steps rise 200.0 to 303.8 while mean infeasible steps rise 14.04 to 19.91: per-step infeasible rate 0.0702 over the first 200 steps against 0.0565 over the extension. Any movement in this row's infeasibility is therefore separable into numerator and denominator and is not a claim that the filter found more room.

**Against the 400-step parent L321** (artifact `data/runs/v2.9.2/horizon400_tableI/score__L305_h400.json`) — obstacle collisions 65 to 105 episodes (+40), share 0.0325 to 0.0525 (+0.0200); SINGULAR episode-axis 0.1680 to 0.3990 (+0.2310); cps 0.732974 to 0.669163 (-0.063811).

**A1 is not re-opened.** A1 was registered and scored at 200 and its verdict stands as scored. These 400-step figures are a new cell and carry no verdict on A1; L316's verdict was not edited. Single seed 42; the interval is within-seed and scene-only. saturation_rate 0.1186 is a recorded diagnostic and enters no score. NOT bold, NOT promoted.

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400: the v2.9.2 registered-axis OC cell (env.quadrotor_3d.c_gain 0.3 -> 0.0) re-scored on the registered cell at 400 with every other key identical to L316's. cps 0.669163 with reach 0.8970 and collision 0.0790 decomposed obstacle 0.0525 (105 ep) and band_lower 0.0265 (53 ep); against the 400-step parent L321 the obstacle channel is +40 episodes (+0.0200 share) and the SINGULAR episode-axis fraction +0.2310. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L316, because the horizon moves both the outcome resolution and the infeasibility denominator; the 200 control reproduced L316 in all 20 compared fields within 1e-12, and the paired per-episode cps difference against it is +0.049751 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the difference of [+0.037713, +0.062960]; single seed 42. THE REGISTERED OBSERVABLE A1 IS NOT RE-OPENED -- A1 was registered and scored at 200 and its verdict stands as scored, this cell carries NO verdict on it, and L316's verdict is unedited; NOT bold, NOT promoted, no checkpoint re-selected. Detail docs/versions/v2.9.2/horizon400_tables.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-oc3d-cgain0-v292-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000), ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 30000, the checkpoint L316 names, its path carried in the companion and NOT here so rule 5 keys this row on its scoring time; cps composed by the shipped evaluator src.eval.evaluate.evaluate; artifacts data/runs/v2.9.2/horizon400_tables/score__L316_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tables/score__L316_h400.json, per-episode dumps perepisode__L316_h200.npz and perepisode__L316_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tables/gate_paired.json)

<a id="v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292-at-h400"></a>
## v2.9.2 · quadrotor_3d · best.pt JT3D_CGAIN0_V292 at h400

`anchor: v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292-at-h400` · registered after the five-sentence cap, so this section carries the **full record** rather than a pre-cap verdict · cps `0.8669` · date `2026-08-18 00:54:29`

**Checkpoint** `data/runs/v2.9.2/set__20260815-014509__seed42/v2.9.2__jt__20260815-014509__seed42/checkpoints/best.pt` @ step 9300, the checkpoint L318 names; no checkpoint was re-selected and no config key on disk was changed. The horizon is a run-time override on the in-memory cell dict.

**200 control.** Reproduced L318 in all 20 compared fields with residual within 1e-12: the five outcome counts and fractions, infeasibility, cps, the EMPTY and SINGULAR episode-axis fractions, and the collision split with its per-surface counts. Key lookups in that gate are unguarded.

**Outcomes at 400** — goal 1917, collision 50, oob 0, stuck 4, timeout 29; incomplete (stuck + timeout) 33 = 0.0165, against 56 = 0.0280 at the 200 control.

**Infeasibility clauses at 400** — empty 0.067459; singular as coded 0.041395, of which singular-and-violated 0.000824 and singular-and-satisfied 0.040571; union 0.108030, which is what enters cps and equals the shipped filter's returned flag on every step (0 mismatching steps). Singular-and-violated is subsumed entirely by empty, so under the 04_eval section 1 definition this row's infeasibility would be 0.067459 and its cps 0.879012, a shift of +0.012120. At the 200 control the same clauses read empty 0.067800 and singular 0.041241.

**Episode-axis, never summed** — EMPTY 0.5910 and SINGULAR 0.2500 at 400, against 0.5910 and 0.2485 at the 200 control.

**Denominator** — over the episodes the control left incomplete (n 56), mean active steps rise 200.0 to 346.4 while mean infeasible steps rise 38.16 to 64.21: per-step infeasible rate 0.1908 over the first 200 steps against 0.1779 over the extension. Any movement in this row's infeasibility is therefore separable into numerator and denominator and is not a claim that the filter found more room.

**Against the 400-step parent L328** (artifact `data/runs/v2.9.2/horizon400_tableI/score__L312_h400.json`) — obstacle collisions 15 to 18 episodes (+3), share 0.0075 to 0.0090 (+0.0015); SINGULAR episode-axis 0.0960 to 0.2500 (+0.1540); cps 0.885312 to 0.866893 (-0.018419).

**A1 is not re-opened.** A1 was registered and scored at 200 and its verdict stands as scored. These 400-step figures are a new cell and carry no verdict on A1; L318's verdict was not edited. Single seed 42; the interval is within-seed and scene-only. saturation_rate 0.3670 is a recorded diagnostic and enters no score. NOT bold, NOT promoted.

### relocated from the `verdict` column (pre-cap text, verbatim)

NEW CELL at eval.max_steps 400: the v2.9.2 registered-axis JT cell (env.quadrotor_3d.c_gain 0.3 -> 0.0) re-scored on the registered cell at 400 with every other key identical to L318's. cps 0.866893 with reach 0.9585 and collision 0.0250 decomposed obstacle 0.0090 (18 ep) and band_lower 0.0160 (32 ep); against the 400-step parent L328 the obstacle channel is +3 episodes (+0.0015 share) and the SINGULAR episode-axis fraction +0.1540. NOT COMPARABLE TO ANY 200-STEP ROW including its own counterpart L318, because the horizon moves both the outcome resolution and the infeasibility denominator; the 200 control reproduced L318 in all 20 compared fields within 1e-12, and the paired per-episode cps difference against it is +0.017344 with a 04_eval 5 within-seed scene bootstrap 95 pct CI on the difference of [+0.010636, +0.024834]; single seed 42. THE REGISTERED OBSERVABLE A1 IS NOT RE-OPENED -- A1 was registered and scored at 200 and its verdict stands as scored, this cell carries NO verdict on it, and L318's verdict is unedited; NOT bold, NOT promoted, no checkpoint re-selected. Detail docs/versions/v2.9.2/horizon400_tables.md and docs/ledger_verdicts.md#v2-9-2--quadrotor-3d--best-pt-jt3d-cgain0-v292-at-h400.

### relocated from the `eval_source` column (pre-cap text, verbatim)

eval_only(NEW CELL, the registered v282_agree_gate.gate_overrides cell with eval.max_steps 400 in place of 200 and every other key identical: pool fullcb (the registered pool of record, sha8 3682a4e3, n 2000), ebs 2000, projection dual_solve, empty_fallback kstep phases 1 k 3, alpha (2.0, 100.0), terminal (0.15, 0.3, 0.3), dt 0.05, dt_ctrl 0.05; best.pt @ step 9300, the checkpoint L318 names, its path carried in the companion and NOT here so rule 5 keys this row on its scoring time; cps composed by the shipped evaluator src.eval.evaluate.evaluate; artifacts data/runs/v2.9.2/horizon400_tables/score__L318_h200.json (the 200 control) and data/runs/v2.9.2/horizon400_tables/score__L318_h400.json, per-episode dumps perepisode__L318_h200.npz and perepisode__L318_h400.npz, gate and paired statistic data/runs/v2.9.2/horizon400_tables/gate_paired.json)

<a id="v2-9-2--quadrotor-3d--r1-joint-ours-at-h200"></a>
## v2.9.2 · quadrotor_3d · R1 joint (ours) at h200

`anchor: v2-9-2--quadrotor-3d--r1-joint-ours-at-h200` · registered after the character cap, so this section carries the **full record** · cps `0.8680` · date `2026-08-18 01:55:04`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 200`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L250 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1907, collision 51, oob 0, stuck 1, timeout 41; incomplete (stuck + timeout) 42. reach 0.9535, collision 0.0255, infeasibility 0.079209, saturation_rate 0.3812.

**Collision by surface** — obstacle 0.0075, band_lower 0.018, band_upper 0.0.

**Wall** 9.38 s, **peak CUDA** 476.7 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r1-joint-ours-at-h400"></a>
## v2.9.2 · quadrotor_3d · R1 joint (ours) at h400

`anchor: v2-9-2--quadrotor-3d--r1-joint-ours-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.8853` · date `2026-08-18 01:55:13`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 400`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L250 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1930, collision 51, oob 0, stuck 1, timeout 18; incomplete (stuck + timeout) 19. reach 0.9650, collision 0.0255, infeasibility 0.078960, saturation_rate 0.3804.

**Collision by surface** — obstacle 0.0075, band_lower 0.018, band_upper 0.0.

**Wall** 12.57 s, **peak CUDA** 938.1 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r2-learned-pi-backup-qp-at-h200"></a>
## v2.9.2 · quadrotor_3d · R2 learned pi + backup-QP at h200

`anchor: v2-9-2--quadrotor-3d--r2-learned-pi-backup-qp-at-h200` · registered after the character cap, so this section carries the **full record** · cps `0.7693` · date `2026-08-18 01:57:50`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 200`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L251 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1836, collision 102, oob 0, stuck 3, timeout 59; incomplete (stuck + timeout) 62. reach 0.9180, collision 0.0510, infeasibility 0.101473, saturation_rate 0.3464.

**Collision by surface** — obstacle 0.025, band_lower 0.026, band_upper 0.0.

**Wall** 157.28 s, **peak CUDA** 424.4 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r3-learned-pi-backup-switch-at-h400"></a>
## v2.9.2 · quadrotor_3d · R3 learned pi + backup-switch at h400

`anchor: v2-9-2--quadrotor-3d--r3-learned-pi-backup-switch-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.0853` · date `2026-08-18 01:59:29`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 400`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L252 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1228, collision 220, oob 0, stuck 552, timeout 0; incomplete (stuck + timeout) 552. reach 0.6140, collision 0.1100, infeasibility 0.109151, saturation_rate 0.3651.

**Collision by surface** — obstacle 0.0435, band_lower 0.0665, band_upper 0.0.

**Wall** 68.87 s, **peak CUDA** 895.8 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r2-learned-pi-backup-qp-at-h400"></a>
## v2.9.2 · quadrotor_3d · R2 learned pi + backup-QP at h400

`anchor: v2-9-2--quadrotor-3d--r2-learned-pi-backup-qp-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.7865` · date `2026-08-18 02:02:45`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 400`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L251 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1867, collision 109, oob 0, stuck 6, timeout 18; incomplete (stuck + timeout) 24. reach 0.9335, collision 0.0545, infeasibility 0.101652, saturation_rate 0.3454.

**Collision by surface** — obstacle 0.0285, band_lower 0.026, band_upper 0.0.

**Wall** 300.13 s, **peak CUDA** 885.9 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r3-learned-pi-backup-switch-at-h200"></a>
## v2.9.2 · quadrotor_3d · R3 learned pi + backup-switch at h200

`anchor: v2-9-2--quadrotor-3d--r3-learned-pi-backup-switch-at-h200` · registered after the character cap, so this section carries the **full record** · cps `0.0860` · date `2026-08-18 02:03:23`

**Checkpoint** `data/runs/v2.9.1/set__20260814-030107__seed42/v2.9.1__jt__20260814-030107__seed42/checkpoints/best.pt`, at `eval.max_steps 200`. Repointed to the checkpoint L328 names (step 9450) — this cell is learning-dependent.

**Control.** Its original-checkpoint 200 scoring reproduced L252 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1228, collision 220, oob 0, stuck 549, timeout 3; incomplete (stuck + timeout) 552. reach 0.6140, collision 0.1100, infeasibility 0.109236, saturation_rate 0.3890.

**Collision by surface** — obstacle 0.0435, band_lower 0.0665, band_upper 0.0.

**Wall** 38.44 s, **peak CUDA** 424.4 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r4-pd-backup-qp-at-h400"></a>
## v2.9.2 · quadrotor_3d · R4 PD + backup-QP at h400

`anchor: v2-9-2--quadrotor-3d--r4-pd-backup-qp-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.6753` · date `2026-08-18 02:04:25`

**Checkpoint** `data/runs/v2.8.2/set__20260803-063606__seed42/v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt`, at `eval.max_steps 400`. Not repointed: this cell carries no learned component the repointing would touch, or carries its own network.

**Control.** Its original-checkpoint 200 scoring reproduced L253 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1740, collision 134, oob 1, stuck 0, timeout 125; incomplete (stuck + timeout) 125. reach 0.8700, collision 0.0670, infeasibility 0.097441, saturation_rate 0.2811.

**Collision by surface** — obstacle 0.0355, band_lower 0.0315, band_upper 0.0.

**Wall** 295.92 s, **peak CUDA** 877.4 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--r5-pd-backup-switch-at-h400"></a>
## v2.9.2 · quadrotor_3d · R5 PD + backup-switch at h400

`anchor: v2-9-2--quadrotor-3d--r5-pd-backup-switch-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.2209` · date `2026-08-18 02:05:52`

**Checkpoint** `data/runs/v2.8.2/set__20260803-063606__seed42/v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt`, at `eval.max_steps 400`. Not repointed: this cell carries no learned component the repointing would touch, or carries its own network.

**Control.** Its original-checkpoint 200 scoring reproduced L254 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1380, collision 247, oob 0, stuck 367, timeout 6; incomplete (stuck + timeout) 373. reach 0.6900, collision 0.1235, infeasibility 0.123684, saturation_rate 0.3105.

**Collision by surface** — obstacle 0.053, band_lower 0.0705, band_upper 0.0.

**Wall** 87.23 s, **peak CUDA** 877.4 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--ppo-certificate-free-at-h400"></a>
## v2.9.2 · quadrotor_3d · PPO certificate-free at h400

`anchor: v2-9-2--quadrotor-3d--ppo-certificate-free-at-h400` · registered after the character cap, so this section carries the **full record** · cps `0.7077` · date `2026-08-18 02:14:46`

**Checkpoint** `data/baselines/ppo/v2.8.2__ppo__20260805-114505__seed42/checkpoints/step_166896.pt`, at `eval.max_steps 400`. Not repointed: this cell carries no learned component the repointing would touch, or carries its own network.

**Control.** Its original-checkpoint 200 scoring reproduced L262 at |Δcps| = 0.00e+00, so this row's lineage is gated.

**Outcomes** — goal 1795, collision 180, oob 0, stuck 14, timeout 11; incomplete (stuck + timeout) 25. reach 0.8975, collision 0.0900, infeasibility 0.000000, saturation_rate 0.0069.

**Collision by surface** — not decomposed by this producer for this cell.

**Wall** 2.17 s, **peak CUDA** 842.9 MB. Single seed 42. NOT bold, NOT promoted; no checkpoint was re-selected beyond the repointing the rebase fixes.

<a id="v2-9-2--quadrotor-3d--mppi-h400-ebs2000"></a>
## v2.9.2 · quadrotor_3d · MPPI_H400_EBS2000

`anchor: v2-9-2--quadrotor-3d--mppi-h400-ebs2000` · cps `n/a` · date `2026-08-18 02:53:12`

**Cell.** L317's MPPI cell verbatim with eval.max_steps 200 -> 400 and ebs 200 -> 2000 (the registered evaluation batch). Wall 790.6 s, peak CUDA 6251.0 MB; the wall is an UPPER BOUND, inflated by contention with two concurrent figure jobs on the same card.

**Control.** The ebs 200 / 200-step scoring reproduced L317 on every field that row carries: reach 0.2980, collision 0.6525, coll_obstacle 0.3280, coll_band_lower 0.3235, tilt<=60 n 486 reach 0.4198.

**Outcomes** — goal 699, collision 1301, oob 0, stuck 0, timeout 0; incomplete 0. All 99 timeouts of the 200-step cell resolve at 400. Collision split: obstacle 0.3275, band_lower 0.3220, band_upper 0.0010.

**Tilt partition at 60 deg** — upright n 486 reach 0.5082 collision 0.4918; the jointly trained pair on the same subset at 400 reaches 0.9815, and both sides are now on the same horizon AND the same batch, so the batch caveat is closed.

**Inapplicable, not zero:** infeasibility and mean_proj_mag — MPPI runs an identity filter, forms no half-space and runs no projection. cps is n/a: filter semantics are out of scope here and no composite headline is claimed.

Single seed 42, one run, no CI. NOT bold, NOT promoted. Detail docs/versions/v2.9.2/mppi_h400.md.

## v2.9.2 · quadrotor_3d · OBSFREEZE CONTROL live JT (L328)
<a id="v2-9-2--quadrotor-3d--obsfreeze-control-live-jt-l328"></a>
`anchor: v2-9-2--quadrotor-3d--obsfreeze-control-live-jt-l328` · cps `0.8853` · date `2026-08-19 00:46`
**Cell.** The registered cell verbatim at eval.max_steps 400 with `eval.obs_freeze_obstacles` FALSE. It exists as the reproduction control for the frozen row of the same checkpoint.
**Gate.** ALL_MATCH over 18 fields against registered row L312_h400, every residual 0: the five outcome counts and their shares, infeasibility, cps, saturation_rate, the three collision-cause counts, and the EMPTY and SINGULAR episode fractions.
**Not bold, not promoted.** Reported only. docs/versions/v2.9.2/obs_conditioning.md


## v2.9.2 · quadrotor_3d · OBSFREEZE CONTROL live OC (L321)
<a id="v2-9-2--quadrotor-3d--obsfreeze-control-live-oc-l321"></a>
`anchor: v2-9-2--quadrotor-3d--obsfreeze-control-live-oc-l321` · cps `0.7330` · date `2026-08-19 00:46`
**Cell.** The registered cell verbatim at eval.max_steps 400 with `eval.obs_freeze_obstacles` FALSE. It exists as the reproduction control for the frozen row of the same checkpoint.
**Gate.** ALL_MATCH over 18 fields against registered row L305_h400, every residual 0: the five outcome counts and their shares, infeasibility, cps, saturation_rate, the three collision-cause counts, and the EMPTY and SINGULAR episode fractions.
**Not bold, not promoted.** Reported only. docs/versions/v2.9.2/obs_conditioning.md


## v2.9.2 · quadrotor_3d · OBSFREEZE frozen-obstacle JT (L328)
<a id="v2-9-2--quadrotor-3d--obsfreeze-frozen-obstacle-jt-l328"></a>
`anchor: v2-9-2--quadrotor-3d--obsfreeze-frozen-obstacle-jt-l328` · cps `-0.9578` · date `2026-08-19 00:46`
**Cell and departure.** The registered cell with ONE key changed: `eval.obs_freeze_obstacles` TRUE. The 4*k_obs obstacle columns of the observation -- the top-K cylinders' body-frame offsets and radii -- are replaced by their step-zero value for the whole episode; body velocity, body rates, body-frame goal, body-frame gravity and the band channels p_z and v_z stay live. The certificate and the policy read the same frozen block. Eval only; no training key touched.
**Paired against the live control** (scene bootstrap, 10000 resamples, seed 20292): reach -0.7115 CI [-0.7310, -0.6915]; collision +0.4515; coll_obstacle +0.4490; coll_band_lower +0.0000 CI [-0.0030, +0.0030]; cps -1.8431. 1447 of 2000 episodes changed outcome.
**Reach cost by obstacle count** (frozen minus live): 2-6 n 887 -0.6776 CI [-0.7080, -0.6471]; 7-12 n 613 -0.7520 CI [-0.7847, -0.7178]; 13-18 n 500 -0.7220 CI [-0.7600, -0.6820].
**The population the ablation acts on** (A2, live trajectories): the top-K identity set differs from its step-zero value on 0.5118 of live steps; 0.5960 of episodes see any change; median first change at step 13.
**Not bold, not promoted.** Reported only. docs/versions/v2.9.2/obs_conditioning.md


## v2.9.2 · quadrotor_3d · OBSFREEZE frozen-obstacle OC (L321)
<a id="v2-9-2--quadrotor-3d--obsfreeze-frozen-obstacle-oc-l321"></a>
`anchor: v2-9-2--quadrotor-3d--obsfreeze-frozen-obstacle-oc-l321` · cps `-0.4835` · date `2026-08-19 00:46`
**Cell and departure.** The registered cell with ONE key changed: `eval.obs_freeze_obstacles` TRUE. The 4*k_obs obstacle columns of the observation -- the top-K cylinders' body-frame offsets and radii -- are replaced by their step-zero value for the whole episode; body velocity, body rates, body-frame goal, body-frame gravity and the band channels p_z and v_z stay live. The certificate and the policy read the same frozen block. Eval only; no training key touched.
**Paired against the live control** (scene bootstrap, 10000 resamples, seed 20292): reach -0.3990 CI [-0.4205, -0.3775]; collision +0.4155; coll_obstacle +0.4160; coll_band_lower -0.0005 CI [-0.0040, +0.0030]; cps -1.2165. 884 of 2000 episodes changed outcome.
**Reach cost by obstacle count** (frozen minus live): 2-6 n 887 -0.2864 CI [-0.3168, -0.2559]; 7-12 n 613 -0.4568 CI [-0.4976, -0.4160]; 13-18 n 500 -0.5280 CI [-0.5720, -0.4840].
**The population the ablation acts on** (A2, live trajectories): the top-K identity set differs from its step-zero value on 0.5204 of live steps; 0.5890 of episodes see any change; median first change at step 19.
**Not bold, not promoted.** Reported only. docs/versions/v2.9.2/obs_conditioning.md

## v2.9.3 · quadrotor_3d · FIXEDLAYOUT cert on randomized pool
<a id="v2-9-3--quadrotor-3d--fixedlayout-cert-on-randomized-pool"></a>
`anchor: v2-9-3--quadrotor-3d--fixedlayout-cert-on-randomized-pool` · cps `-0.0934` · date `2026-08-19 01:30`
**Cell.** fixed-layout certificate on the full randomized registered pool. n 2000, ebs 2000, eval.max_steps 400, seed 42; the filter and projection are the registered cell's, unchanged.
**Outcomes.** reach 0.6350, collision 0.3330 (obstacle 0.3015, band_lower 0.0315, band_upper 0.0000), incomplete 0.0320, infeasibility 0.1064, EMPTY 0.5070, SINGULAR 0.0965, cps -0.0934 CI [-0.1566, -0.0300].
**The held layout.** 3 active cylinders, radii 0.297 to 0.585. REPRODUCED from the run seed and the run's own config, not read from a persisted artifact -- the trainer does not persist it. Recorded as a gap in the instrumentation rather than hidden.
**Not bold, not promoted.** docs/versions/v2.9.2/obs_conditioning.md


## v2.9.3 · quadrotor_3d · FIXEDLAYOUT cert on its trained layout
<a id="v2-9-3--quadrotor-3d--fixedlayout-cert-on-its-trained-layout"></a>
`anchor: v2-9-3--quadrotor-3d--fixedlayout-cert-on-its-trained-layout` · cps `0.8216` · date `2026-08-19 01:30`
**Cell.** fixed-layout certificate on the layout it trained on. n 1755, ebs 2000, eval.max_steps 400, seed 42; the filter and projection are the registered cell's, unchanged.
**Outcomes.** reach 0.9624, collision 0.0365 (obstacle 0.0074, band_lower 0.0291, band_upper 0.0000), incomplete 0.0011, infeasibility 0.2224, EMPTY 0.2741, SINGULAR 0.4308, cps 0.8216 CI [0.7965, 0.8493].
**The held layout.** 3 active cylinders, radii 0.297 to 0.585. REPRODUCED from the run seed and the run's own config, not read from a persisted artifact -- the trainer does not persist it. Recorded as a gap in the instrumentation rather than hidden.
**Not bold, not promoted.** docs/versions/v2.9.2/obs_conditioning.md


## v2.9.3 · quadrotor_3d · L321 control on that layout
<a id="v2-9-3--quadrotor-3d--l321-control-on-that-layout"></a>
`anchor: v2-9-3--quadrotor-3d--l321-control-on-that-layout` · cps `0.8602` · date `2026-08-19 01:30`
**Cell.** L321 on the same held layout, as the control for the layout cell. n 1755, ebs 2000, eval.max_steps 400, seed 42; the filter and projection are the registered cell's, unchanged.
**Outcomes.** reach 0.9652, collision 0.0325 (obstacle 0.0068, band_lower 0.0256, band_upper 0.0000), incomplete 0.0023, infeasibility 0.1262, EMPTY 0.2769, SINGULAR 0.2866, cps 0.8602 CI [0.8348, 0.8864].
**The held layout.** 3 active cylinders, radii 0.297 to 0.585. REPRODUCED from the run seed and the run's own config, not read from a persisted artifact -- the trainer does not persist it. Recorded as a gap in the instrumentation rather than hidden.
**Not bold, not promoted.** docs/versions/v2.9.2/obs_conditioning.md


## v2.9.3 · double_integrator · HOCBF analytic baseline (a1 1, a2 4)
<a id="v2-9-3--double-integrator--hocbf-analytic-baseline-a1-1-a2-4"></a>
`anchor: v2-9-3--double-integrator--hocbf-analytic-baseline-a1-1-a2-4` · cps `0.6540` · date `2026-08-19 01:37`
**Cell and departure.** The registered double-integrator cell -- producer `v282_agree_gate.gate_overrides`, pool `eval_full_di_n2000_seed123456.pkl`, n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, dt 0.05, terminal (0.15, 0.30, 0.30), seed 42, the OC row's own nominal `system.lqr_action(x, scene.goal)` and the shipped actuator box -- with ONE thing replaced: the certificate. In place of a learned value network the row is a hand-built two-level HIGH-ORDER CBF on the RAW clearance to the nearest active cylinder, `h = r_i* - ||p - c_i*||` in the repository's positive-unsafe convention. There is no checkpoint, no network and no learned quantity anywhere in this row. `src/common/filter_hocbf.py`, scored by `scripts/analysis/v293_hocbf_di.py`.
**The cascade.** `L_g h == 0` identically (relative degree 2, verified against autograd at residual exactly 0), so the first-order row the deployed filter uses is inapplicable and a cascade is required: `psi_0 = h`, `psi_1 = L_f h + a1 h`, enforce `psi_1dot + a2 psi_1 <= 0`, i.e. `A.u <= b` with `A = L_g L_f h = -n^T` and `b = -L_f^2 h - (a1+a2) L_f h - a1 a2 h`, where `L_f h = -n.v` and `L_f^2 h = -(||v||^2 - (n.v)^2)/rho`. AFFINE IN THE INPUT, confirmed numerically before scoring on 1024 real pool states in float64: `psi_1dot(x,u)` matches `A.u + c` to 5.3e-15 and its second difference in `u` is 3.6e-15, i.e. zero. `||A|| == 1` at every row, so the deployed filter's singular clause is structurally vacuous here (measured `mass_singular` exactly 0.0).
**A SECOND DEPARTURE, named.** The registered cell's `empty_fallback = {kstep, phases 1, k 3}` action substitution is NOT applied. On an infeasible row this baseline deploys the least-violating admissible command (`filter_hardnet._box_aware_projection`) and counts the step, per the dispatch; no slack variable was added. Grafting a rollout search onto an analytic filter would make it a hybrid and would stop the infeasibility figure measuring the analytic row.
**Gains, screened.** Nine `(a1, a2)` pairs on the first 400 scenes of the registered pool, full grid in the build-log: cps 0.4646 (1,1), 0.5593 (1,2)=(2,1), **0.6470 (1,4)=(4,1)**, 0.5737 (2,2), 0.6183 (2,4)=(4,2), 0.5958 (4,4). With LINEAR class-K functions the row depends on the gains only through `a1+a2` and `a1*a2`, so the grid is SYMMETRIC -- the three transposed pairs were verified bit-identical in every field, and nine grid points are six distinct filters. The best cell leads the next distinct cell by 0.0287, well outside the 0.0083 admissibility floor, so no tie-break was needed. The maximum sits at a CORNER of the {1,2,4}^2 box; the nine-pair cap was fully consumed, so a wider-spread pair cannot be excluded as better.
**Scored cell.** reach 0.8870, collision 0.1120 (obstacle 0.1120, band 0.0000/0.0000), oob 0.0000, stuck 0.0010, timeout 0.0000, infeasibility 0.0267, saturation 0.1574, cps 0.6540 CI [0.6114, 0.6979]. Beside the registered double-integrator rows on the identical cell: L323 (OC) cps 0.8822, collision 0.0055; L325 (JT) cps 0.9055, collision 0.0090. Paired scene bootstrap, 10 000 resamples: vs L323 dcps -0.2282 CI [-0.2651, -0.1913], dcollision +0.1065 CI [+0.0925, +0.1205]; vs L325 dcps -0.2515 CI [-0.2907, -0.2127], dcollision +0.1030 CI [+0.0895, +0.1165]. Against the unfiltered nominal at the same cell (reach 0.6635, collision 0.3365, cps -0.0095) it is a real filter; against the learned rows it is not close.
**The QP's infeasibility rate, as its own figure.** The row and the box fail to intersect on 0.026697 of active steps, CI [0.024220, 0.029210], and 0.318 of episodes see at least one such step. Because the singular clause cannot fire here, that IS the reported column; the like-for-like EMPTY clause of the learned rows is 0.0175 (L323) and 0.0207 (L325), so the analytic program is infeasible 1.53x and 1.29x more often than theirs, and the reported column being lower than theirs is an artefact of their singular clause, not an advantage. Mean empty-step mass is 0.1491 on collided episodes against 0.0113 on reached episodes.
**Why it is bad.** Of 224 collisions, 205 (91.5%) had an empty row within the 20 steps before impact, 19 (8.5%) are the single-nearest-obstacle row switching cylinders, and 0 are attributable to sampling with a feasible, correctly-locked row. The cascade demands a deceleration the box cannot deliver, and where it does, it hits. 3.15% of pool scenes start outside the level-1 set the cascade certifies (`psi_1(x_0) > 0`) and collide at 0.2222 against 0.1084 for compatible starts -- 14 of the 224 collisions.
**Pre-cap eval_source, verbatim.** eval_only; NO checkpoint -- analytic certificate; registered cell v282_agree_gate.gate_overrides; pool eval_full_di_n2000_seed123456.pkl, n 2000, ebs 2000, max_steps 400; gains screened on the first 400 scenes of that pool; artifacts data/runs/v2.9.3/hocbf_di/score__final_a1-1_a2-4.json, data/runs/v2.9.3/hocbf_di/perepisode__final_a1-1_a2-4.npz, data/runs/v2.9.3/hocbf_di/grid.json, data/runs/v2.9.3/hocbf_di/affinity_check.json, data/runs/v2.9.3/hocbf_di/diag.json, data/runs/v2.9.3/hocbf_di/paired.json.
**Not bold, not promoted.** No training was run, no existing module was edited, no config key on disk was changed. docs/versions/v2.9.3/hocbf_di.md


## v2.9.3 · double_integrator · HOCBF one-parameter gain (a 3.0171)
<a id="v2-9-3--double-integrator--hocbf-one-parameter-gain-a-3-0171"></a>
`anchor: v2-9-3--double-integrator--hocbf-one-parameter-gain-a-3-0171` · cps `0.6130` · date `2026-08-19 03:05`
**Cell and departure.** The registered double-integrator cell -- producer `v282_agree_gate.gate_overrides`, pool `eval_full_di_n2000_seed123456.pkl`, n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, dt 0.05, terminal (0.15, 0.30, 0.30), seed 42, the OC row's own nominal `system.lqr_action(x, scene.goal)` and the shipped actuator box `[-2,2]^2` -- with the certificate replaced by a hand-built two-level HIGH-ORDER CBF on the RAW clearance to the nearest active cylinder. No checkpoint, no network, no learned quantity. Same construction and same code path as L348; the ONLY thing that differs from L348 is the gain. `src/common/filter_hocbf.py`, scored by `scripts/analysis/v293_hocbf_extend.py`.
**The gain, and why it is one number.** With LINEAR class-K functions the row depends on `(a1, a2)` only through `a1+a2` and `a1*a2`, so the two gains are the two ROOTS of one characteristic polynomial and any transposed pair is bit-identical -- L348's nine-point grid was six filters, and its maximum sat at a corner of its `{1,2,4}^2` box. This row replaces that screen with `a1 = a2 = a`, which places the closed loop's double root at `-a` (CRITICALLY DAMPED) and reduces the row to `b = -L_f^2 h - 2a L_f h - a^2 h`. `a` was swept LOGARITHMICALLY over [0.1, 500] at 16 points on the first 400 scenes of the registered pool, every other key at its registered value. Screen cps: -0.2175 (a 0.10), +0.0550, +0.3025, +0.4000, +0.4647, +0.5666, **+0.6037 (a 3.0171)**, +0.4712, +0.1985, -0.0468, -0.1244, -0.1293, -0.1277, -0.1261, -0.1256, -0.1253 (a 500). The runner-up is +0.5666 at `a` 1.7100, a gap of 0.0371 -- outside the 0.0083 admissibility band -- so no tie-break was needed. Sanity: the sweep point `a` 0.9692 reproduces L348's grid cell `(1,1)` (cps 0.4647 vs 0.4646, reach and collision identical).
**Q1 -- is small `a` infeasible from the first step? NO, and the refutation is exact.** At the four smallest gains (`a` 0.10, 0.18, 0.31, 0.55) the empty-clause step mass is EXACTLY 0.000000 and the fraction of scenes whose FIRST row is empty is 0.0000, while 36.00%, 25.50%, 17.00% and 8.25% of those same scenes start OUTSIDE the level-1 set `{psi_1 <= 0}`. L348's most conservative cell was not infeasible -- it was UNDER-DEMANDING: the correction the cascade requires of a violating state is `psi_1dot <= -a psi_1`, which shrinks with `a` at the same rate the level-1 set does, and as `a -> 0` the RHS tends to `-L_f^2 h = +||v_perp||^2/rho >= 0`, so `u = 0` always satisfies the row. Infeasibility here is a LARGE-`a`, interior phenomenon: 0 at `a <= 0.55`, peaking at 0.046774 (`a` 9.3926), decaying again as the row goes vacuous.
**Q2 -- does any `a` trade collisions for incomplete episodes? NO on this plant.** Rows that STALL do exist below the old grid's floor (incomplete 0.2425 at `a` 0.10, 0.1100 at `a` 0.18; the grid never saw them because its smallest gain, `a` 1, is the 5th of 16 sweep points). But collision is MONOTONE DECREASING in `a` across the whole conservative half -- 0.2550, 0.2425, 0.2325, 0.2000, 0.1775, 0.1425, 0.1300 -- and the sweep minimum 0.1300 is attained where incomplete is 0.0000. Every gain that stalls also collides MORE. On the double integrator, under this box, the two-level analytic cascade has no conservative regime: it cannot buy safety with time.
**Scored cell.** reach 0.8730, collision 0.1265 (obstacle 0.1265, band 0.0000/0.0000), oob 0.0000, stuck 0.0005, timeout 0.0000, infeasibility 0.0216, saturation 0.1768, cps 0.6130 CI [0.5665, 0.6558]. Counts goal 1746, collision 253, stuck 1. Beside the registered rows on the identical cell: L323 (OC) cps 0.8822 collision 0.0055; L325 (JT) cps 0.9055 collision 0.0090; unfiltered nominal cps -0.0095 collision 0.3365. Paired scene bootstrap, 10 000 resamples, percentile: vs L323 dcps -0.2692 CI [-0.3082, -0.2304], dcollision +0.1210 CI [+0.1070, +0.1355]; vs L325 dcps -0.2925 CI [-0.3347, -0.2508], dcollision +0.1175 CI [+0.1030, +0.1320].
**Against L348, and the price of the reparameterization.** dcps -0.0410 CI [-0.0656, -0.0173], dcollision +0.0145 CI [+0.0065, +0.0230], 69 of 2000 episodes changed outcome. `a1 = a2` forces `a1*a2 = ((a1+a2)/2)^2`, so the critically-damped family is a ONE-DIMENSIONAL SLICE of the two-dimensional `(sum, product)` plane the row actually depends on, and L348's `(1, 4)` (sum 5, product 4) lies OFF it (`a` 3.0171 has sum 6.03, product 9.10). Tying the gains costs this plant 0.0410 cps with a CI clear of zero. That is reported, not absorbed. **L348 is KEPT, unamended.**
**Mechanism at the selected gain.** Empty-row rate 0.014838 of pooled active steps (per-episode mean 0.021591, the reported column); singular clause exactly 0.000000, as `||L_g L_f h|| == 1` identically on this plant. Of 253 collisions, 230 (90.91%) had an empty row within the 20 steps before impact, 23 (9.09%) are the single-nearest-obstacle row switching cylinders, and 0 are attributable to sampling with a feasible, correctly-locked row. At those empty steps the row demands a mean 6.075 m/s^2 of deceleration along the outward normal (median 5.889, p90 9.045) against a box that can supply at most 2.576 there -- a deficit of 3.499, i.e. the cascade asks 2.36x what the actuator can give -- and the deployed command is a BOX CORNER, mean |u| = (2.000, 2.000), saturated in both channels. Mean empty-step mass 0.1343 on collided episodes against 0.0053 on reached ones.
**A SECOND DEPARTURE, named.** The registered cell's `empty_fallback = {kstep, phases 1, k 3}` action substitution is NOT applied; on an infeasible row this baseline deploys the least-violating admissible command and counts the step, and no slack variable was added. Grafting a rollout search onto an analytic filter would make it a hybrid and would stop the infeasibility figure measuring the analytic row.
**Pre-cap eval_source, verbatim.** eval_only; NO checkpoint -- analytic certificate; registered cell v282_agree_gate.gate_overrides; pool eval_full_di_n2000_seed123456.pkl, n 2000, ebs 2000, max_steps 400; gain from a 16-point log sweep of a1=a2 over [0.1,500] on the first 400 scenes of that pool; artifacts data/runs/v2.9.3/hocbf_extend/double_integrator/score__final_a-3.01709.json, perepisode__final_a-3.01709.npz, diag_a-3.01709.json, paired_a-3.01709.json, paired_vs_L348.json, di_parity.json, affinity_check.json, relative_degree.json; full sweep data/runs/v2.9.3/hocbf_sweep/double_integrator/sweep.json and selection.json.
**Not bold, not promoted.** No training was run, no config key on disk was changed, no git operation was performed, single seed 42. The double-integrator row at L348's gain was re-scored through the extended module and matched it in EVERY field with delta exactly 0 (`di_parity.json`, `identical: true`). docs/versions/v2.9.3/hocbf_di.md §8


## v2.9.3 · unicycle · HOCBF analytic (a 5.3234)
<a id="v2-9-3--unicycle--hocbf-analytic-a-5-3234"></a>
`anchor: v2-9-3--unicycle--hocbf-analytic-a-5-3234` · cps `0.5812` · date `2026-08-19 03:06`
**Cell and departure.** The registered unicycle cell -- producer `v282_agree_gate.gate_overrides`, pool `eval_full_unicycle_n2000_seed123456.pkl`, n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, dt 0.05, terminal (0.15, 0.30, 0.30), seed 42, the OC row's own nominal `system.lqr_action(x, scene.goal)` and the shipped actuator box `[-2,2] x [-3,3]` -- with the certificate replaced by a hand-built two-level HIGH-ORDER CBF on the RAW clearance to the nearest active cylinder, `h = r_i* - ||p - c_i*||` in the repository's positive-unsafe convention. No checkpoint, no network, no learned quantity. `src/common/filter_hocbf.py` (extended additively), scored by `scripts/analysis/v293_hocbf_extend.py`.
**Relative degree, ESTABLISHED not assumed.** `x = [px, py, theta, v]`, `u = [a, omega]`. `L_g h == 0` identically, so the deployed first-order row is inapplicable and a cascade is required. Writing `e = (cos th, sin th)`, `e_perp = (-sin th, cos th)`, `s = n.e`, `t = n.e_perp` (so `s^2 + t^2 = 1` exactly): `L_f h = -v s`, `L_f^2 h = -v^2 t^2 / rho` (the same centripetal term as the double integrator), and `L_g L_f h = [-s, -v t]`. BOTH input channels appear at the second differentiation -- the acceleration through `dot v`, the turn rate through the heading -- so the relative degree is EXACTLY 2 IN BOTH INPUTS and a two-level cascade is the correct depth. Verified against autograd through the true plant on 1024 real pool states in float64: `L_g h` residual 0.000e+00, `L_f h` 4.441e-16, `L_g L_f h` columns 0.000e+00 / 4.441e-16, `psi_1dot(x,u)` vs `A.u + c` 7.105e-15, second difference in `u` 3.553e-15. The drift Lie chain `max |L_gj L_f^k h|` is (0, 0) at k=0 and (1.000, 2.470) at k=1 for `(a, omega)` -- the measurement that fixes the degree at 2.
**The one degeneracy, structurally live and empirically empty.** `||L_g L_f h||^2 = s^2 + v^2 t^2` is NOT identically 1 as on the double integrator; it vanishes on `{v = 0 and s = 0}` -- standing still, heading tangential -- so the deployed filter's `||L_g h|| < 5e-4` singular clause is structurally live here. Measured over all 800 000 scored steps it fires on EXACTLY 0.
**The gain.** `a1 = a2 = a` (critically damped, double root at `-a`), swept LOGARITHMICALLY over [0.1, 500] at 16 points on the first 400 scenes of the registered pool before this cell was scored. Screen cps: -0.1863 (a 0.10), +0.0625, +0.1550, +0.1924, +0.3247, +0.4415, +0.5157, **+0.5599 (a 5.3234)**, +0.4936, +0.1587, +0.0145, -0.0609, -0.0894, -0.1035, -0.1030, -0.1029 (a 500). Runner-up +0.5157 at `a` 3.0171, a gap of 0.0442, outside the 0.0083 band; no tie-break needed.
**What the sweep shows on this plant.** Small `a` is NOT infeasible from the first step: at `a` 0.10, 0.18 and 0.31 the empty-clause mass is EXACTLY 0.000000 and no scene opens with an empty row, while 26.50%, 16.25% and 9.75% of scenes start outside the level-1 set. Infeasibility is instead an interior, large-`a` phenomenon, peaking at 0.029192 (`a` 16.572). And unlike the double integrator, this plant DOES trade collisions for incomplete episodes over an interior interval: as `a` falls 16.572 -> 9.3926 -> 5.3234 -> 3.0171, collision falls 0.2675 -> 0.1250 -> 0.0775 -> 0.0625 while incomplete rises 0.0150 -> 0.0625 -> 0.1025 -> 0.1475. Below `a` 3.0171 the trade reverses and both worsen. Incomplete reaches 0.3300 at `a` 0.10.
**Scored cell.** reach 0.8210, collision 0.0590 (obstacle 0.0590, band 0.0000/0.0000), oob 0.0000, stuck 0.1200, timeout 0.0000, infeasibility 0.0062, saturation 0.1957, cps 0.5812 CI [0.5431, 0.6208]. Counts goal 1642, collision 118, stuck 240. Beside the registered unicycle rows on the identical cell: L324 (OC) cps 0.6753 collision 0.0020 incomplete 0.1495; L326 (JT) cps 0.8732 collision 0.0020 incomplete 0.0575; unfiltered nominal cps 0.0020 collision 0.3320. Paired scene bootstrap, 10 000 resamples, percentile: vs L324 dcps -0.0942 CI [-0.1283, -0.0605], dcollision +0.0570 CI [+0.0470, +0.0675], dincomplete -0.0295, 314 episodes changed; vs L326 dcps -0.2921 CI [-0.3303, -0.2550], dcollision +0.0570 CI [+0.0470, +0.0675], dincomplete +0.0625, 322 episodes changed. This is the analytic row's smallest OC gap of the three systems, but not because it is good: its collision rate is 29x either learned row's, and the OC row it nearly matches on cps is itself losing 0.1495 of the pool to stalls.
**Mechanism.** Empty-row rate 0.003208 of pooled active steps (per-episode mean 0.006016); singular clause 0.000000. Of 118 collisions, 116 (98.31%) had an empty row within the 20 steps before impact, 2 (1.69%) are the nearest-obstacle argmax switching, and 0 are attributable to sampling with a feasible, correctly-locked row. At those empty steps the row demands a mean 10.683 in row units against a box supply of 3.965 (normalized by `||A||`: demand 9.106, supply 3.245, deficit 5.861), and the deployed command is a BOX CORNER, mean |u| = (2.000, 3.000), saturated in both channels. No pool scene starts outside the level-1 set at this gain.
**Two further departures, named.** (1) The registered cell's `empty_fallback = {kstep, phases 1, k 3}` is NOT applied; the least-violating admissible command is deployed and the step is counted, with no slack variable. (2) `h_scale` is not read -- the hazard is the raw signed distance, not `signed_h`'s clipped ramp -- and the row is built from the single nearest active obstacle, which costs 2 of the 118 collisions. One named interaction: this system's third reach leg reads the COMMANDED turn rate off the executed action (`|u[1]| <= 0.30`), and on an empty row this baseline emits `|omega| = 3.0`, so no reach can register at such a step; empty rows are 0.0062 of steps.
**Pre-cap eval_source, verbatim.** eval_only; NO checkpoint -- analytic certificate; registered cell v282_agree_gate.gate_overrides; pool eval_full_unicycle_n2000_seed123456.pkl, n 2000, ebs 2000, max_steps 400; gain from a 16-point log sweep of a1=a2 over [0.1,500] on the first 400 scenes of that pool; artifacts data/runs/v2.9.3/hocbf_extend/unicycle/score__final_a-5.32336.json, perepisode__final_a-5.32336.npz, diag_a-5.32336.json, paired_a-5.32336.json, affinity_check.json, relative_degree.json; full sweep data/runs/v2.9.3/hocbf_sweep/unicycle/sweep.json.
**Not bold, not promoted.** No training was run, no config key on disk was changed, no git operation was performed, single seed 42. docs/versions/v2.9.3/hocbf_extend.md


## v2.9.3 · quadrotor_planar · HOCBF analytic (a 3.0171)
<a id="v2-9-3--quadrotor-planar--hocbf-analytic-a-3-0171"></a>
`anchor: v2-9-3--quadrotor-planar--hocbf-analytic-a-3-0171` · cps `0.4171` · date `2026-08-19 03:07`
**Cell and departure.** The registered planar-quadrotor cell -- producer `v282_agree_gate.gate_overrides`, pool `eval_full_quadrotor-planar_n2000_seed23456.pkl`, n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, dt 0.05, terminal (0.15, 0.30, 0.30), seed 42, the OC row's own cascaded-PD nominal `system.lqr_action(x, scene.goal)` and the shipped actuator box `[0, 19.62] x [-1, 1]` -- with the certificate replaced by a hand-built two-level HIGH-ORDER CBF on the RAW clearance to the nearest active cylinder. No checkpoint, no network, no learned quantity. `src/common/filter_hocbf.py` (extended additively), scored by `scripts/analysis/v293_hocbf_extend.py`.
**Relative degree, and WHERE THE CONSTRUCTION STOPS.** `x = [px, py, theta, vx, vy, omega]`, `u = [f_thr, tau]`. `L_g h == 0` identically. At the second level, with `Re = (-sin th, cos th)` the body thrust axis: `L_f h = -n.v`, `L_f^2 h = -(||v||^2 - (n.v)^2)/rho + g n_y`, and `L_g L_f h = [-(n.Re)/m, 0]`. So the relative degree is EXACTLY 2 -- but only with respect to the THRUST. THE TORQUE COLUMN IS IDENTICALLY ZERO, and not merely at this depth: every drift Lie derivative `L_f^k h` of this plant depends on the state only through `(p, v)` (the drift restricted to `(p, v)` is `(v, -g e_y)`, which is attitude-free), so `L_tau L_f^k h == 0` for EVERY k. Measured: the torque column of `max |L_tau L_f^k h|` over 1024 real pool states in float64 is 0.000e+00 at k = 0, 1, 2 and 3, while the thrust column is 0, 1.000, 9.738, 1.010e+02. Torque reaches the clearance only through the BILINEAR product `tau * f_thr` along the trajectory, which is not a relative degree at all; under the standard dynamic extension (thrust promoted to a state) the vector relative degree is (3, 4), so a FOUR-LEVEL cascade would be required to put torque in the row. It was not built. The depth-2 row therefore constrains the thrust MAGNITUDE only: it can push harder or ease off along the current thrust axis, and cannot re-point it.
**Two further structural facts, both measured.** (1) `L_f^2 h` is NOT sign-definite here -- unlike the double integrator's pure centripetal term it carries `+g n_y`, so free fall closes clearance on anything below the vehicle. (2) `||L_g L_f h|| = |n.Re|/m` vanishes when the thrust axis is tangential to the cylinder, so the `||L_g h|| < 5e-4` singular clause is structurally live AND fires: step mass 0.000373, 2.80% of episodes. Affinity verified against autograd before scoring: `L_g h` 0.000e+00, `L_g L_f h` columns 8.882e-16 / 0.000e+00, `psi_1dot(x,u)` vs `A.u + c` 1.421e-14, second difference in `u` 1.066e-14.
**The gain.** `a1 = a2 = a` (critically damped), swept LOGARITHMICALLY over [0.1, 500] at 16 points on the first 400 scenes of the registered pool before this cell was scored. Screen cps: -0.4899 (a 0.10), -0.0951, +0.0077, +0.1532, +0.1862, +0.3975, **+0.4586 (a 3.0171)**, +0.4583, +0.3581, +0.1334, -0.0368, -0.1222, -0.1429, -0.1416, -0.1483, -0.1478 (a 500). This is the only genuine tie of the three systems: `a` 5.3234 scores +0.4583, within 0.000254 of the maximum and inside the 0.0083 admissibility band, so the SMALLER gain is taken.
**What the sweep shows on this plant, and it is the exception.** Here small `a` IS infeasible from the first step. At `a` 0.10, 43.75% of scenes start outside the level-1 set AND 27.00% already have an EMPTY row at step 0; the empty-clause mass is at its sweep maximum at the conservative end (0.057800 at `a` 0.10, 0.058446 at `a` 0.18) and decays monotonically to 0.000802 at `a` 500 -- the exact opposite of the double integrator and the unicycle, where it is 0.000000 at the conservative end. The cause is the plant, not the gain: as `a -> 0`, `b -> ||v_perp||^2/rho - g n_y`, which is NEGATIVE whenever the vehicle sits above the obstacle centre, and the thrust box is ONE-SIGNED (`f in [0, 19.62]`), so wherever the thrust axis leans toward the cylinder (`n.Re <= 0`) the box's best possible row value is exactly 0 and the row is empty. This plant also trades collisions for incomplete episodes over an interior interval (`a` 9.3926 -> 1.7100: collision 0.2100 -> 0.1700 -> 0.1575 -> 0.1575, incomplete 0.0000 -> 0.0100 -> 0.0275 -> 0.0575), weakly, before the trade reverses below `a` 1.7100. Incomplete reaches 0.2300 at `a` 0.10.
**Scored cell.** reach 0.7990, collision 0.1650 (obstacle 0.1650, band 0.0000/0.0000), oob 0.0000, stuck 0.0360, timeout 0.0000, infeasibility 0.0531, saturation 0.0565, cps 0.4171 CI [0.3645, 0.4682]. Counts goal 1598, collision 330, stuck 72. Beside the registered planar rows on the identical cell: L322 (OC) cps 0.6963 collision 0.0520; L327 (JT) cps 0.7894 collision 0.0215; unfiltered nominal cps -0.2060 collision 0.4020. Paired scene bootstrap, 10 000 resamples, percentile: vs L322 dcps -0.2792 CI [-0.3322, -0.2283], dcollision +0.1130 CI [+0.0970, +0.1295], 455 episodes changed; vs L327 dcps -0.3723 CI [-0.4243, -0.3221], dcollision +0.1435 CI [+0.1285, +0.1595], 452 episodes changed. Against L327 the REPORTED infeasibility column is indistinguishable (dinfeas +0.0035 CI [-0.0024, +0.0097]), but the like-for-like EMPTY clause is higher by +0.0148 CI [+0.0093, +0.0206]: the reported column understates this baseline's feasibility problem, because the learned rows carry a large singular clause and this one carries almost none.
**Mechanism, and it differs from the other two systems.** Empty-row rate 0.012429 of pooled active steps (per-episode mean 0.052810); singular-row rate 0.000246. Of 330 collisions, 190 (57.58%) had an empty row within the 20 steps before impact, 53 (16.06%) are the nearest-obstacle argmax switching, and 87 (26.36%) are sampling with a feasible, correctly-locked row -- the double integrator and the unicycle both record ZERO in that last class. At the empty pre-impact steps the row demands a mean 10.612 in row units against a box supply of 1.797, and on 68.44% OF THOSE STEPS THE SUPPLY IS <= 0: no admissible thrust produces any decay in the row direction at all, because the deficit is not a magnitude the thrust cannot reach but a DIRECTION the thrust axis cannot point in -- and pointing it is a torque action, which is not in the row. Where the supply is positive the row still asks a median 26.97 N against a 19.62 N ceiling. On those steps the filter saturates the thrust (mean 6.19 N, i.e. `f_max` on the 31.6% of steps where thrust helps and `f_min` elsewhere) and passes the nominal torque through UNTOUCHED, because its row has nothing to say about torque. 2.30% of scenes start outside the level-1 set and collide at 0.4348 against 0.1586 for compatible starts.
**Two further departures, named.** (1) The registered cell's `empty_fallback = {kstep, phases 1, k 3}` is NOT applied; the least-violating admissible command is deployed and the step is counted, with no slack variable. (2) `h_scale` is not read, and the row is built from the single nearest active obstacle, which costs 53 of the 330 collisions.
**Pre-cap eval_source, verbatim.** eval_only; NO checkpoint -- analytic certificate; registered cell v282_agree_gate.gate_overrides; pool eval_full_quadrotor-planar_n2000_seed23456.pkl, n 2000, ebs 2000, max_steps 400; gain from a 16-point log sweep of a1=a2 over [0.1,500] on the first 400 scenes of that pool; artifacts data/runs/v2.9.3/hocbf_extend/quadrotor_planar/score__final_a-3.01709.json, perepisode__final_a-3.01709.npz, diag_a-3.01709.json, paired_a-3.01709.json, affinity_check.json, relative_degree.json; full sweep data/runs/v2.9.3/hocbf_sweep/quadrotor_planar/sweep.json; the quadrotor_3d obstruction probe at data/runs/v2.9.3/hocbf_extend/quadrotor_3d_obstruction.json.
**Not bold, not promoted.** No training was run, no config key on disk was changed, no git operation was performed, single seed 42. The 3-D quadrotor was NOT attempted and `HOCBFFilter` raises on it: its hazard is the clearance to a VERTICAL cylinder so `n` is horizontal, the only depth-2 channel is the collective thrust with coefficient `-(n . R e3)/m <= sin(tilt)` which vanishes identically on the hover manifold, the input is four rotor forces of which the row would constrain only the sum, and the torques are four levels away under dynamic extension. docs/versions/v2.9.3/hocbf_extend.md

## v2.9.3 · quadrotor_planar · COLDSTART value_init null (quadrotor_planar)
<a id="v2-9-3--quadrotor-planar--coldstart-value-init-null-quadrotor-planar"></a>
`anchor: v2-9-3--quadrotor-planar--coldstart-value-init-null-quadrotor-planar` · cps `0.6807` · date `2026-08-19 12:40`
**Cell and departure.** The registered cell unchanged at eval.max_steps 400; the ONE training key changed is `training.jt.value_init_ckpt` -> null, so the certificate starts from a random initialization instead of the observation-conditioned stage. Every other key is taken from L327's own persisted config, and the budget is that run's own.
**Against its warm counterpart L327.** cps 0.6807 vs 0.7894, paired difference **-0.1087** CI [-0.1459, -0.0717], **13.09x the 0.0083 admissibility floor**; 1131 of 2000 episodes changed outcome.
**Outcomes.** reach 0.8640, collision 0.0390 (obstacle 0.0390, band_lower 0.0000, band_upper 0.0000), incomplete 0.0970, infeasibility 0.0427, saturation 0.1808.
**Registered prediction FALSIFIED on this system.** It predicted the cold-started pair would land within the floor of its warm counterpart; it does not.
**Not bold, not promoted.** docs/versions/v2.9.3/cold_start.md

## v2.9.3 · double_integrator · COLDSTART value_init null (double_integrator)
<a id="v2-9-3--double-integrator--coldstart-value-init-null-double-integrator"></a>
`anchor: v2-9-3--double-integrator--coldstart-value-init-null-double-integrator` · cps `0.8845` · date `2026-08-19 12:40`
**Cell and departure.** The registered cell unchanged at eval.max_steps 400; the ONE training key changed is `training.jt.value_init_ckpt` -> null, so the certificate starts from a random initialization instead of the observation-conditioned stage. Every other key is taken from L325's own persisted config, and the budget is that run's own.
**Against its warm counterpart L325.** cps 0.8845 vs 0.9055, paired difference **-0.0210** CI [-0.0420, +0.0004], **2.53x the 0.0083 admissibility floor**; 1241 of 2000 episodes changed outcome.
**Outcomes.** reach 0.9580, collision 0.0140 (obstacle 0.0140, band_lower 0.0000, band_upper 0.0000), incomplete 0.0280, infeasibility 0.0583, saturation 0.7190.
**Registered prediction FALSIFIED on this system.** It predicted the cold-started pair would land within the floor of its warm counterpart; it does not.
**Not bold, not promoted.** docs/versions/v2.9.3/cold_start.md

## v2.9.3 · unicycle · COLDSTART value_init null (unicycle)
<a id="v2-9-3--unicycle--coldstart-value-init-null-unicycle"></a>
`anchor: v2-9-3--unicycle--coldstart-value-init-null-unicycle` · cps `0.8129` · date `2026-08-19 12:40`
**Cell and departure.** The registered cell unchanged at eval.max_steps 400; the ONE training key changed is `training.jt.value_init_ckpt` -> null, so the certificate starts from a random initialization instead of the observation-conditioned stage. Every other key is taken from L326's own persisted config, and the budget is that run's own.
**Against its warm counterpart L326.** cps 0.8129 vs 0.8732, paired difference **-0.0603** CI [-0.0823, -0.0382], **7.27x the 0.0083 admissibility floor**; 993 of 2000 episodes changed outcome.
**Outcomes.** reach 0.9150, collision 0.0055 (obstacle 0.0055, band_lower 0.0000, band_upper 0.0000), incomplete 0.0795, infeasibility 0.0387, saturation 0.7027.
**Registered prediction FALSIFIED on this system.** It predicted the cold-started pair would land within the floor of its warm counterpart; it does not.
**Not bold, not promoted.** docs/versions/v2.9.3/cold_start.md

## v2.9.3 · quadrotor_3d · COLDSTART value_init null (quadrotor_3d)
<a id="v2-9-3--quadrotor-3d--coldstart-value-init-null-quadrotor-3d"></a>
`anchor: v2-9-3--quadrotor-3d--coldstart-value-init-null-quadrotor-3d` · cps `0.8686` · date `2026-08-19 12:40`
**Cell and departure.** The registered cell unchanged at eval.max_steps 400; the ONE training key changed is `training.jt.value_init_ckpt` -> null, so the certificate starts from a random initialization instead of the observation-conditioned stage. Every other key is taken from L328's own persisted config, and the budget is that run's own.
**Against its warm counterpart L328.** cps 0.8686 vs 0.8853, paired difference **-0.0167** CI [-0.0316, -0.0023], **2.02x the 0.0083 admissibility floor**; 1555 of 2000 episodes changed outcome.
**Outcomes.** reach 0.9640, collision 0.0280 (obstacle 0.0095, band_lower 0.0180, band_upper 0.0005), incomplete 0.0080, infeasibility 0.1172, saturation 0.3555.
**Registered prediction half held.** The 'not within the floor' half is right; the 'degrades rather than converging' half is wrong -- the run completed its full budget with no halt and posts the smallest cold-warm gap of the four systems. Its certificate matches **neither** the flat nor the saturated attractor: certified fraction 0.63, probe range 2.10, empty 0.322, mean projection 1.87. Signature: data/runs/v2.9.3/cold_start/signature__quadrotor_3d.json.
**Not bold, not promoted.** docs/versions/v2.9.3/cold_start.md

## v2.9.3 / double_integrator / COLDSTART40K bounded D_V (double_integrator)

**The departure is three keys off L325's own registered joint config, all in training, none in the
eval cell:** `training.jt.value_init_ckpt` null (certificate from scratch), `training.jt.n_steps`
40000, `collection.jt.buffer_cap` 200000. The gate in `scripts/analysis/v293_cold40k_launch.py`
refuses any launch whose flattened config diff is not exactly those three keys; this run passed it,
reached 40000/40000 and halted with `halt_reason` null.

**Scored** on the registered cell `v282_agree_gate.gate_overrides` at the h400 basis, this system's
registered pool at n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, seed 42, from the run's own
`best.pt` @ step 26700 -- the rule the warm rows were scored under. **cps 0.893488.**

**Both paired contrasts**, same pool, same order, scene bootstrap, 10000 resamples, percentile,
generator seed 20292:

| contrast | delta | CI | inside 0.0083 floor | CI excludes 0 |
|---|---:|---|---|---|
| vs **L325** (warm, 10000 steps, `value_init_ckpt` SET) | -0.012015 | [-0.031011, +0.007268] | False | False |
| vs **L353** (cold@10000, unbounded cap 1000000) | +0.008964 | [-0.012596, +0.030642] | False | False |

The second contrast is **budget plus cap, not budget alone**: at 10000 steps the 1000000-trajectory
cap never bound, so L353 is an unbounded run and this one is not.

**Buffer cap actually applied: 200000, UNIFORM across all four systems -- a deviation from the
dispatched rule.** The rule was a per-system cap equal to the trajectory count D_V held at the end of
that system's warm comparator's own 10000 steps. That count is logged nowhere and is not derivable
from config arithmetic (the collector appends segments, not episodes), so it was measured by
`scripts/analysis/v293_dv_trajectory_count.py` -- a stratified replay of the real `collect_jt` against
the comparator's own checkpoints -- and for this system is **183691 +/- 4327**
(`data/runs/v2.9.3/bufcap/dv_count__double_integrator.json`). The uniform 200000 exceeds it by **+8.9%**, so this
cold run was permitted that much more history than its warm comparator actually held. That is a
residual difference between the conditions, far smaller than the ~4x the unbounded cap allowed but
not zero, and it is recorded here rather than left to be inferred from a config key.

**Initial certificate state, established AT THE DEPLOYMENT ENTRY POINT** -- not inferred from the null
`value_init_ckpt` key. `jt_pncbf.train.load_framework_from_checkpoint` (the function
`src/eval/run_full._load_framework` calls) on this run's earliest checkpoint, over the registered
pool's initial states: **ORDINARY** at step 150 -- V-hat in [-1.0000, +0.6827], range
1.6827e+00, gradient norm max 1.5996e+00, fraction V-hat > 0 = 0.0320. Neither **flat**
(constant, vanishing gradient, row identically zero, filter inert, every state certified) nor
**saturated** (at the upper clamp, nothing certified, filter refuses everything). The trainer writes
no step-0 checkpoint, so step 150 is the earliest DEPLOYABLE state of this run; that limit is
stated rather than smoothed over. The same probe on the scored `best.pt` @ step 26700: **ORDINARY**,
range 1.1862e+00, gradient norm max 7.0855e+00, fraction V-hat > 0 = 0.0050.

**Registered predictions, scored on this system.**
Prediction 1 -- the cold-warm gap shrinks by more than the floor when the budget goes 10000 -> 40000:
|cold40k - warm| 0.012015 against |cold10k - warm| 0.020979, a shrink of +0.008964 > 0.0083 --
**HELD**. **This leg is INFORMED, not blind:** the unbounded double_integrator run's in-loop series was seen before this cell was scored, so its prediction-1 result is recorded as informed rather than scored as a blind pre-registration.
Prediction 2 concerns quadrotor_3d only and is not scored here.
Prediction 3 -- no run halts and the degenerate verdict is NEITHER: reached 40000/40000, `halt_reason`
null, certificate ORDINARY at both the earliest and the scored checkpoint -- **HELD** on this system.

**`best.pt` discrepancy pair: NONE.** `status.json`/`best.pt` and the eval-record maximum agree
exactly, delta 0.000000. `halt.early_stop_min_delta` 0.002 requires an eval to beat the incumbent by
more than 0.002 to replace it, so `best.pt` CAN lag the best eval -- it did on the unbounded
double_integrator run (+0.90485 @ 26400 against eval-max +0.90520 @ 29550). It did not here.

**Not bold, not promoted.** Single seed 42. Which rows the paper's tables cite is a Researcher
decision taken after all four cells exist; quadrotor_planar is running and quadrotor_3d is queued.



**BOLD 2026-08-20 — a `06_workflow` 2.5 BASIS CLASSIFICATION, not a measured beat.**
The Researcher's approval selects bold on the 400-step basis and fixes the standing comparison
basis to the cold-start condition. Warm-start rows are therefore a **different training class**
and are not ranked against this one -- the clause at `06_workflow` 2.5 that says such rows are
"never SOTA-bolded on `cps` alone" and are "flagged for Researcher classification instead".
That classification has now been made.

**Nothing here is claimed as a beat.** Against the warm h400 runner-up L325 (0.905503): paired delta **-0.012015**, CI [-0.031011, +0.007268], which covers zero.
The bolded row is 1.45x the 0.0083 floor BELOW its warm runner-up; it is bolded because the
basis is the cold-start condition, not because it scores higher.
Single seed 42. Supersedes L314, whose bold is removed in this edit.
## v2.9.3 / unicycle / COLDSTART40K bounded D_V (unicycle)

**The departure is three keys off L326's own registered joint config, all in training, none in the
eval cell:** `training.jt.value_init_ckpt` null (certificate from scratch), `training.jt.n_steps`
40000, `collection.jt.buffer_cap` 200000. The gate in `scripts/analysis/v293_cold40k_launch.py`
refuses any launch whose flattened config diff is not exactly those three keys; this run passed it,
reached 40000/40000 and halted with `halt_reason` null.

**Scored** on the registered cell `v282_agree_gate.gate_overrides` at the h400 basis, this system's
registered pool at n 2000, `eval_batch_size` 2000, `eval.max_steps` 400, seed 42, from the run's own
`best.pt` @ step 31050 -- the rule the warm rows were scored under. **cps 0.874232.**

**Both paired contrasts**, same pool, same order, scene bootstrap, 10000 resamples, percentile,
generator seed 20292:

| contrast | delta | CI | inside 0.0083 floor | CI excludes 0 |
|---|---:|---|---|---|
| vs **L326** (warm, 10000 steps, `value_init_ckpt` SET) | +0.001021 | [-0.018817, +0.021132] | True | False |
| vs **L354** (cold@10000, unbounded cap 1000000) | +0.061328 | [+0.037306, +0.085375] | False | True |

The second contrast is **budget plus cap, not budget alone**: at 10000 steps the 1000000-trajectory
cap never bound, so L354 is an unbounded run and this one is not.

**Buffer cap actually applied: 200000, UNIFORM across all four systems -- a deviation from the
dispatched rule.** The rule was a per-system cap equal to the trajectory count D_V held at the end of
that system's warm comparator's own 10000 steps. That count is logged nowhere and is not derivable
from config arithmetic (the collector appends segments, not episodes), so it was measured by
`scripts/analysis/v293_dv_trajectory_count.py` -- a stratified replay of the real `collect_jt` against
the comparator's own checkpoints -- and for this system is **190396 +/- 2150**
(`data/runs/v2.9.3/bufcap/dv_count__unicycle.json`). The uniform 200000 exceeds it by **+5.0%**, so this
cold run was permitted that much more history than its warm comparator actually held. That is a
residual difference between the conditions, far smaller than the ~4x the unbounded cap allowed but
not zero, and it is recorded here rather than left to be inferred from a config key.

**Initial certificate state, established AT THE DEPLOYMENT ENTRY POINT** -- not inferred from the null
`value_init_ckpt` key. `jt_pncbf.train.load_framework_from_checkpoint` (the function
`src/eval/run_full._load_framework` calls) on this run's earliest checkpoint, over the registered
pool's initial states: **ORDINARY** at step 150 -- V-hat in [-1.0000, +0.4635], range
1.4635e+00, gradient norm max 1.8007e+00, fraction V-hat > 0 = 0.0155. Neither **flat**
(constant, vanishing gradient, row identically zero, filter inert, every state certified) nor
**saturated** (at the upper clamp, nothing certified, filter refuses everything). The trainer writes
no step-0 checkpoint, so step 150 is the earliest DEPLOYABLE state of this run; that limit is
stated rather than smoothed over. The same probe on the scored `best.pt` @ step 31050: **ORDINARY**,
range 1.0871e+00, gradient norm max 4.8933e+00, fraction V-hat > 0 = 0.0020.

**Registered predictions, scored on this system.**
Prediction 1 -- the cold-warm gap shrinks by more than the floor when the budget goes 10000 -> 40000:
|cold40k - warm| 0.001021 against |cold10k - warm| 0.060307, a shrink of +0.059286 > 0.0083 --
**HELD**.
Prediction 2 concerns quadrotor_3d only and is not scored here.
Prediction 3 -- no run halts and the degenerate verdict is NEITHER: reached 40000/40000, `halt_reason`
null, certificate ORDINARY at both the earliest and the scored checkpoint -- **HELD** on this system.

**`best.pt` discrepancy pair: NONE.** `status.json`/`best.pt` and the eval-record maximum agree
exactly, delta 0.000000. `halt.early_stop_min_delta` 0.002 requires an eval to beat the incumbent by
more than 0.002 to replace it, so `best.pt` CAN lag the best eval -- it did on the unbounded
double_integrator run (+0.90485 @ 26400 against eval-max +0.90520 @ 29550). It did not here.

**Not bold, not promoted.** Single seed 42. Which rows the paper's tables cite is a Researcher
decision taken after all four cells exist; quadrotor_planar is running and quadrotor_3d is queued.


**BOLD 2026-08-20 — a `06_workflow` 2.5 BASIS CLASSIFICATION, not a measured beat.**
The Researcher's approval selects bold on the 400-step basis and fixes the standing comparison
basis to the cold-start condition. Warm-start rows are therefore a **different training class**
and are not ranked against this one -- the clause at `06_workflow` 2.5 that says such rows are
"never SOTA-bolded on `cps` alone" and are "flagged for Researcher classification instead".
That classification has now been made.

**Nothing here is claimed as a beat.** Against the warm h400 runner-up L326 (0.873211): paired delta **+0.001021**, CI [-0.018817, +0.021132], which covers zero.
The gap is **INSIDE the 0.0083 floor** at 0.12x. This is also **unicycle's first bold row** --
the lineage had never carried one -- so it supersedes nothing.
Single seed 42. Supersedes nothing.
## v2.9.3 / quadrotor_planar / COLDSTART40K bounded D_V (quadrotor_planar)

**Three keys off L327's own registered joint config, all in training, none in the eval cell:**
`training.jt.value_init_ckpt` null, `training.jt.n_steps` 40000, `collection.jt.buffer_cap` 200000.
Gate passed at exactly those three; the run reached 40000/40000 with `halt_reason` null.

**Scored** on `v282_agree_gate.gate_overrides` at h400, pool `eval_full_quadrotor-planar_n2000_seed23456.pkl` n 2000, ebs 2000, max_steps 400,
seed 42, from `best.pt` @ step 37050. **cps 0.763455.**

| contrast | delta | CI | inside 0.0083 floor | CI excludes 0 |
|---|---:|---|---|---|
| vs **L327** (warm, 10000 steps, `value_init_ckpt` SET) | -0.025922 | [-0.061990, +0.009940] | False | False |
| vs **L352** (cold@10000, unbounded cap) | +0.082763 | [+0.044579, +0.120830] | False | **True** |

**This is the largest established effect in the axis.** The cold@10k contrast is 10.0x the floor with an
interval that excludes zero: four times the budget with a bounded D_V moves this system by +0.083
against the same cell at 10000 steps. The warm contrast remains 3.1x the floor but its interval
contains zero, so cold-at-40000 is not separated from warm here either way.

**Buffer cap 200000 uniform, a deviation from the dispatched per-system rule.** This system's measured
comparator count is **171366 +/- 4237** (`data/runs/v2.9.3/bufcap/dv_count__quadrotor_planar.json`), the LOWEST of
the four and the widest band; the uniform cap exceeds it by **+16.7%**, the largest excess in the
chain. This run was therefore permitted a sixth more history than its warm comparator held.

**The cap nonetheless bound, and memory saturated.** Eviction began near step 11000; the HWM settled
at **4818 MiB** and held flat for the last 18000 steps (slope 0.0 MiB/1000 steps over the final
blocks), against **307 MiB/1000 steps** pre-eviction. The dispatch flagged this system as the one where
the trajectory-count cap was most likely to be in the wrong unit -- it is not. Zero-eviction check, weak
form against its own comparator L327: cold **307** vs warm **307** MiB/1000 steps over steps 1000-10000,
an exact match while the cap is still inert. The strong differencing form is unavailable here: the
unbounded planar run died at step 13059, too early to difference against at depth.

**Initial certificate state at the deployment entry point**, via
`load_framework_from_checkpoint` over the registered pool's initial states: **ORDINARY** at step
150 -- V-hat [-0.7015, +0.8514], range 1.5529e+00, gradient norm max 9.1146e-01,
fraction V-hat > 0 = 0.2190. Neither **flat** nor **saturated**. Same probe on the scored `best.pt`
@ step 37050: **ORDINARY**, range 1.8492e+00, gradient norm max 4.9577e+00.

**Predictions.** 1: |cold40k - warm| 0.025922 against |cold10k - warm| 0.108685, shrink
**+0.082763** > 0.0083 -- **HELD**, and by the widest margin of the three systems scored so far.
This leg is **blind**. 2 concerns quadrotor_3d only. 3: 40000/40000, `halt_reason` null, certificate
ORDINARY at both ends -- **HELD**.

**`best.pt` discrepancy pair: NONE** (+0.77501 @ 37050 both ways, delta 0.000000) -- the third
consecutive run with no discrepancy.

**Still climbing at the budget.** Four new bests after step 25350, the last at 37050 (92.6% of budget),
+0.078 across the second half. Unlike the other two completed systems, this run had not flattened when
its budget ran out. **Not bold, not promoted.** Single seed 42.


**BOLD 2026-08-20 — a `06_workflow` 2.5 BASIS CLASSIFICATION, not a measured beat.**
The Researcher's approval selects bold on the 400-step basis and fixes the standing comparison
basis to the cold-start condition. Warm-start rows are therefore a **different training class**
and are not ranked against this one -- the clause at `06_workflow` 2.5 that says such rows are
"never SOTA-bolded on `cps` alone" and are "flagged for Researcher classification instead".
That classification has now been made.

**Nothing here is claimed as a beat.** Against the warm h400 runner-up L327 (0.789377): paired delta **-0.025922**, CI [-0.061990, +0.009940], which covers zero.
The bolded row is 3.12x the 0.0083 floor BELOW its warm runner-up; it is bolded because the
basis is the cold-start condition, not because it scores higher.
Single seed 42. Supersedes L313, whose bold is removed in this edit.
## v2.9.3 / quadrotor_3d / COLDSTART40K bounded D_V (quadrotor_3d)

**Three keys off L328's own registered joint config, all in training, none in the eval cell:**
`training.jt.value_init_ckpt` null, `training.jt.n_steps` 40000, `collection.jt.buffer_cap` 200000.
Reached 40000/40000 with `halt_reason` null.

**Scored** on `v282_agree_gate.gate_overrides` at h400, pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl`, n 2000, ebs 2000, max_steps 400,
seed 42, from `best.pt` @ step 32550. **cps 0.883110.**

| contrast | delta | CI | inside 0.0083 floor | CI excludes 0 |
|---|---:|---|---|---|
| vs **L328** (warm, 10000 steps, `value_init_ckpt` SET) | -0.002202 | [-0.015168, +0.010789] | **True** | False |
| vs **L355** (cold@10000, unbounded cap) | +0.014533 | [+0.000993, +0.028614] | False | **True** |

**This is the only system in the axis where cold-at-40000 lands INSIDE the admissibility floor of its
warm comparator** -- 0.002202, 0.27x the floor. And its cold@10k contrast has an interval
excluding zero, so the budget increase is an established gain on this system.

**Buffer cap 200000 uniform, deviation from the dispatched per-system rule.** Measured comparator count
**187436 +/- 2921** (`data/runs/v2.9.3/bufcap/dv_count__quadrotor_3d.json`); the uniform cap exceeds it by
**+6.7%**, so this run held that much more history than its warm comparator did.

**Memory.** Eviction began near step 11500; the HWM settled at **6780 MiB** and held flat for the final
~28000 steps (0 MiB/1000 steps across four consecutive blocks -- the cleanest saturation of the four),
against **~600 MiB/1000 steps** pre-eviction, nearly double any other system's pre-cap rate. Zero-eviction
check in the weak form (no unbounded 40000-step twin exists for this system): the HWM tracked its
pre-cap line with no early departure, and the cap first bound after step 10000 as required.

**DEGENERATE SIGNATURE: NEITHER.** Taken on the thresholds registered in `degenerate_probe.md` section 2,
read off the LAST `metrics.csv` row (step 40000): `probe_h_min` -0.7937, `probe_h_max`
+0.8610, `label_mean` -0.4290, `rho_unsafe_label` 0.0922.
Saturated needs `probe_h_min>0` OR `label_mean>0.5` OR `rho_unsafe_label>0.90` -- none hold. Flat needs
`probe_h_max<0` OR `label_mean<-0.90` OR `rho_unsafe_label<0.02` -- none hold. Artifact
`degenerate_signature__quadrotor_3d.json`.

**Initial certificate state at the deployment entry point**, via
`load_framework_from_checkpoint` over the registered pool's initial states: **ORDINARY** at step
150 -- V-hat [-1.0000, +1.0411], range 2.0411e+00, gradient norm max
2.7724e+00, fraction V-hat > 0 = 0.5550. Same probe on `best.pt` @ step
32550: **ORDINARY**, range 2.2155e+00, gradient max 6.9080e+00, frac V-hat > 0
0.1515. Neither flat nor saturated at either end.

**Predictions -- all three scored on this system, all HELD.**
1: |cold40k - warm| 0.002202 against |cold10k - warm| 0.016735, shrink **+0.014533** > 0.0083 --
**HELD**, blind.
2 (**scored here and nowhere else**): cps(cold@40k) within 0.0083 of its warm comparator --
0.002202 -- **HELD**.
3: 40000/40000, `halt_reason` null, degenerate verdict NEITHER, certificate ORDINARY at both ends --
**HELD**.

**`best.pt` discrepancy pair: NONE** (+0.87716 @ 32550 both ways) -- the fourth consecutive run with none.

**Not bold, not promoted.** Single seed 42.


**BOLD 2026-08-20 — a `06_workflow` 2.5 BASIS CLASSIFICATION, not a measured beat.**
The Researcher's approval selects bold on the 400-step basis and fixes the standing comparison
basis to the cold-start condition. Warm-start rows are therefore a **different training class**
and are not ranked against this one -- the clause at `06_workflow` 2.5 that says such rows are
"never SOTA-bolded on `cps` alone" and are "flagged for Researcher classification instead".
That classification has now been made.

**Nothing here is claimed as a beat.** Against the warm h400 runner-up L328 == L332 == L341 (0.8853120544848011, one checkpoint recorded three times): paired delta **-0.002202**, CI [-0.015168, +0.010789], which covers zero.
The gap is **INSIDE the 0.0083 floor** at 0.27x.
Single seed 42. Supersedes L304, whose bold is removed in this edit.
## v2.9.3 / quadrotor_3d / COLD40K h200 instrument-compat (quadrotor_3d)

**What this row is, and what it is not.** It is the L359 checkpoint --
`data/runs/v2.9.3/set__20260820-063427__seed42/v2.9.3__jt__20260820-063427__seed42/checkpoints/best.pt`
@ step 32550 -- scored on the registered cell `v282_agree_gate.gate_overrides` at
`eval.max_steps` **200**, on the registered pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl`,
n 2000, ebs 2000, seed 42. It is **not** a basis row. The paper's basis is the 400-step one and
stays there; L359 (h400, cps 0.8831) is this checkpoint's basis row.

**Why it was taken.** Every figure producer and the policy-alone diagnostic resolve the JT
checkpoint through one artifact, `data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json`, whose
`cell_read_back.max_steps` is 200. `scripts/analysis/v292_pi_only.py:82` runs at `MAX_STEPS = 200`
behind a nine-field reproduction gate against that artifact's stored fields. A cold pointer
carrying h400 numbers would fail that gate on all nine. The tool's horizon was **not** changed:
the warm comparison it reproduces is h200, so moving the horizon would break the comparison
rather than fix the pointer.

**Instrument.** `v293_coldstart_score.run_learned`, imported verbatim -- the same function that
produced L356-L359 -- called at `max_steps=200` by
`scripts/analysis/v293_jt_rebase_pointer.py`. Two-clause infeasibility identity
`identity_two_clauses_compose_flag` = True, mismatch steps 0. Collision split residual 0.0.

**Result, beside the warm h200 comparator** (L312's own artifact,
`data/runs/v2.9.1/launch/jtrow__quadrotor_3d.json`):

| field | cold h200 | warm h200 (L312) | delta |
|---|---|---|---|
| cps | 0.864601 | 0.867987 | **-0.003387** |
| reach | 0.9465 | 0.9535 | -0.0070 |
| collision | 0.0230 | 0.0255 | -0.0025 |
| oob | 0.0000 | 0.0000 | 0.0000 |
| stuck | 0.0000 | 0.0005 | -0.0005 |
| timeout | 0.0305 | 0.0205 | +0.0100 |
| infeasibility | 0.068832 | 0.079209 | -0.010377 |
| EMPTY_episode_frac | 0.5095 | 0.5830 | -0.0735 |
| SINGULAR_episode_frac | 0.1115 | 0.0955 | +0.0160 |

The cps gap is **0.41x the 0.0083 admissibility floor** -- no separation. cps CI
[0.8400, 0.8872]. Saturation rate 0.4205. Incomplete 61/2000 = 0.0305, all timeout, zero stuck.
Collision decomposes 0.0060 obstacle / 0.0170 band_lower / 0.0000 band_upper.

**Cross-horizon note, reported not claimed.** The same checkpoint reads 0.8831 at h400 and
0.8646 at h200; the warm checkpoint reads 0.8853 at h400 and 0.8680 at h200. Both move the same
direction by a similar amount, so the horizon effect is not what separates them -- nothing
separates them at either horizon.

**Artifacts.** `data/runs/v2.9.3/jt_rebase/score__COLD40K_quadrotor_3d_h200.json`,
`perepisode__COLD40K_quadrotor_3d_h200.npz`, `jtrow__quadrotor_3d__COLD40K.json`.

**Not bold, not promoted.** Single seed 42.

## v2.9.3 / quadrotor_3d / COLDABL approach-term ablation (quadrotor_3d)

**The one item in the jt_rebase programme that evaluation could not reach.**
`env.quadrotor_3d.c_gain` has a reader on the TRAINING path (`src/common/quadrotor_barrier.py:115`,
inside `value_target_barrier`, reached from the value-target labelling sites in
`src/frameworks/jt_pncbf/collection.py`) and **none on the scoring path**, so the ablated arm had to
be trained.

**ONE key of 328.** Parent is L359's own persisted launch config
`data/runs/v2.9.3/cold_start_40k/config__quadrotor_3d.yaml`; the flattened diff is exactly
`env.quadrotor_3d.c_gain` 0.3 -> 0.0. The cold parent already carries
`training.jt.value_init_ckpt: null`, so there is nothing to repoint. The **warm** ablation could not
say this: `data/runs/v2.9.2/launch/launch__JT3D_CGAIN0_V292.json` gates `c_gain` **and**
`training.jt.value_init_ckpt` (repointed to a budget-truncated OC certificate), and
`docs/versions/v2.9.2/cgain_cells.md:188` states the design "cannot separate the ablated term from
the truncated warm start".

**Run.** `v293_cold_ablation_launch.py`, a third launcher; `v293_cold40k_launch.py` and
`v293_coldstart_launch.py` are unmodified. Seed 42, 40000/40000 steps, `halt_reason` null,
`phase` done, wall **11.91 h**, `cuda_max_mem_mb` 6730.68 against the L359 twin's 6780 — the two
memory curves agree to within 5 MiB at every matched step. `best.pt` @ step 31500.

**Scored** by `v293_coldstart_score.run_learned`, the same function that produced L356-L359, on the
registered cell at `eval.max_steps` 400, pool fullcb n 2000 ebs 2000. Two-clause infeasibility
identity True, mismatch steps 0; collision split residual 0.0.

| field | L359 (c_gain 0.3) | ablated (c_gain 0.0) | delta |
|---|---:|---:|---:|
| cps | 0.883110 | **0.874313** | −0.008797 |
| reach | 0.9595 | 0.9555 | −0.0040 |
| collision | 0.0235 | 0.0260 | +0.0025 |
| coll_obstacle | 0.0065 | 0.0100 | +0.0035 |
| coll_band_lower | 0.0170 | 0.0160 | −0.0010 |
| stuck | 0.0005 | 0.0000 | −0.0005 |
| timeout | 0.0165 | 0.0185 | +0.0020 |
| infeasibility | 0.068799 | 0.066456 | −0.002343 |
| saturation_rate | 0.4199 | 0.3948 | −0.0251 |

**Paired against L359** on the identical pool in identical order — scene bootstrap, 10000
resamples, percentile, generator seed 20292: **delta −0.008797, CI [−0.021869, +0.004337]**,
**1.06× the 0.0083 floor**, `inside_floor` false, **CI covers zero**, 1374 of 2000 episodes changed.

**The two infeasibility legs, reported separately and never summed:**

| leg | L359 | ablated | delta |
|---|---:|---:|---:|
| **SINGULAR_episode_frac** | 0.1115 | **0.1100** | **−0.0015** |
| EMPTY_episode_frac | 0.5095 | 0.5075 | −0.0020 |
| SINGVIOL_episode_frac | 0.0025 | 0.0020 | −0.0005 |

**PREDICTION 1 — FALSIFIED.** Registered before the run: *removing the approach term raises the
SINGULAR episode fraction above L359, as it did on both warm arms*; falsified if the ablated
fraction is at or below 0.1115. It is **0.1100** — it did not rise. On the warm arms the same
removal multiplied the SINGULAR leg by 2.60 (JT, 0.0960 -> 0.2500) and 2.38 (OC, 0.1680 -> 0.3990).
On the cold basis the effect is absent.

**PREDICTION 2 — HELD, and narrowly.** Registered: *cps falls below L359 by more than the floor
0.0083*; falsified if inside the floor or opposite in sign. The drop is −0.008797, sign correct and
**1.06×** the floor, so the registered criterion is met — **but the interval covers zero**, so the
drop is not separated from no-effect. The prediction as written did not require CI exclusion, and
it is scored as written; the qualifier is recorded here rather than folded into the verdict.

**What this changes.** The warm ablation's headline — a 2.6× jump in the SINGULAR leg — **does not
reproduce once the initializer confound is removed**. The warm arm moved `c_gain` and
`value_init_ckpt` together; this arm moves only `c_gain`, and the SINGULAR leg does not move. That
is evidence the warm jump was carried by the repointed, budget-truncated certificate rather than by
the approach term, which is precisely the separation `cgain_cells.md:188` said its design could not
make.

**Not bold, not promoted.** Single seed 42.

---

## v2.9.3 / quadrotor_3d / R1 cold

*Section added 2026-08-23; `state_retrieve_20260822.md` heading 3 recorded its absence.*

**What the row is.** The factorial table's cell R1 — the joint pair, HardNet enforcement, on the
**cold** basis — re-scored so the manuscript's factorial figure does not quote a warm number beside
cold ones. Artifact `data/runs/v2.9.3/jt_rebase/factorial_cold/score__R1_new400.json`
(mtime 2026-08-20 19:14, 2 695 B).

**The cell, read back from the artifact's own `effective_cell_read_back`.** `HardNetFilter`,
`projection dual_solve`, `empty_fallback` mode `kstep` k 3 phases 1, `alpha` (2.0, 100.0),
`gamma_margin` 0.0, `box_klamp_enabled` false — i.e. the registered
`v282_agree_gate.gate_overrides` cell. Pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456`
sha8 `3682a4e3`, n 2000, ebs 2000, `eval_max_steps` 400, dt 0.05, `band_collision_limit` 4.0.
Nominal source: the checkpoint's own `policy_net`, repointed to L359's `best.pt` @ step 32550.

| field | value |
|---|---:|
| cps | **0.8831103827473097** |
| cps CI | [0.858312, 0.904848] |
| reach | 0.9595 |
| collision | 0.0235 (obstacle 0.0065, band_lower 0.0170, band_upper 0.0000) |
| stuck / timeout / oob | 0.0005 / 0.0165 / 0.0000 |
| infeasibility | 0.068799 (empty share 0.054760, singular share 0.014989) |
| saturation_rate | 0.419916 |
| outcome counts | goal 1919, collision 47, stuck 1, timeout 33 |
| wall | 8.39 s |

**The finding.** `cps` is **bit-identical to L359's** 0.8831103827473097 in every compared field.
That is the cold analogue of the gate L332 passed against L328: the factorial cell **is** the
registered cell at h400, so R1 needs no separate measurement and the manuscript's factorial R1 entry
is vacated in favour of L359's own number.

**Interval convention differs from the registered rows, and the difference is recorded rather than
reconciled.** This artifact's `ci_method` is *"percentile bootstrap over scenes, 1000 resamples,
FIXED seed 20260808"*. The registered cold and alternation rows use 10000 resamples at generator
seed 20292. The point estimate is unaffected; only the interval is drawn under a different
convention, so R1's CI must not be quoted beside L359's as if the two were produced identically.

**Not bold, not promoted.** Single seed 42. Eval-only; no training run belongs to this row.

---

## v2.9.3 / quadrotor_3d / R2 cold

*Section added 2026-08-23; `state_retrieve_20260822.md` heading 3 recorded its absence.*

**What the row is.** The factorial table's cell R2 — the **learned policy with backup-certificate QP
enforcement in place of HardNet** — on the cold basis. It is the one vacated manuscript cell that had
no cold equivalent on disk. Artifact
`data/runs/v2.9.3/jt_rebase/factorial_cold/score__R2_new400.json` (mtime 2026-08-20 19:20, 3 805 B).

**The cell, read back from the artifact.** `filter_class` **`BackupQPFilter`** — not HardNet — with
`T_b` 20, `eps` 0.0, `k_d` 8.0, `kp_att` 320.0, `kd_att` 16.0, `c_v` 0.3, `form` `brake`,
`setpoint` `hold`, `alpha` 1.0, `qp_penalty_read_back` 1.0e6, `terminal` `none`. The HardNet fields
(`projection`, `empty_fallback*`, `alpha_safe/unsafe`, `gamma_margin`, `box_klamp_enabled`) are all
null because that filter is not constructed. Same pool, n, ebs, cap and dt as R1; nominal is again
the checkpoint's own `policy_net` at L359's `best.pt` @ step 32550.

| field | value |
|---|---:|
| cps | **0.7985571637745791** |
| cps CI | [0.764279, 0.830546] |
| reach | 0.9395 |
| collision | 0.0535 (obstacle 0.0295, band_lower 0.0240, band_upper 0.0000) |
| stuck / timeout / oob | 0.0000 / 0.0070 / 0.0000 |
| infeasibility | 0.101476 (empty share 0.000000, singular share 0.000000) |
| saturation_rate | 0.376425 |
| outcome counts | goal 1879, collision 107, timeout 14 |

**The finding.** Against the warm counterpart L335 the cold R2 is **+0.0121**, above the 0.0083
floor, but the two intervals overlap ([0.7643, 0.8305] against [0.7537, 0.8188]), so there is **no
separation**. The substantive reading is the one the row already carries: swapping HardNet for the
backup QP on the identical learned policy costs **0.0846 cps** against R1 on the same basis
(0.883110 → 0.798557) and more than doubles collision (0.0235 → 0.0535, obstacle 0.0065 → 0.0295).

**Why `empty_share` and `singular_share` are exactly zero.** They are HardNet diagnostics —
`last_empty` and `last_singular` on the projection row — and `BackupQPFilter` publishes neither, so
zero here means *not measured on this filter*, not *no infeasible steps*. The `infeasibility` column
0.101476 is the evaluator's own, and it is the number to read.

**Interval convention.** Same as R1: 1000 resamples at fixed seed 20260808, not the registered
10000 at generator seed 20292.

**Not bold, not promoted.** Single seed 42. Eval-only.

---

## v2.9.3 / quadrotor_3d / ALTSEP separated collection (quadrotor_3d)

**The third rung of the alternation ladder, and the one that removes the last shared quantity.**
L359 is the interleaved control (`K_V` 3, `K_pi` 1 every macro step). ALTBLK alternates in
500-macro-step blocks but still collects into **both** buffers every round. ALTSEP alternates in the
same blocks and collects into the **active buffer only**, doubling `n_episodes` on the active pass so
the per-buffer totals and the per-macro-step rollout count equal L359's.

**NINE keys of 328**, all alternation, off L359's own persisted config
(`data/runs/v2.9.3/alternation_sep/launch__ALTSEP_QUADROTOR_3D_V293.json`, mtime 2026-08-22 21:00):
`run.framework` jt_pncbf → alt_pncbf; `training.alt.enabled` absent → true;
`training.alt.value_block` 500; `training.alt.policy_block` 500; `training.alt.first` value;
`training.alt.collect` absent → `active_only`; `training.alt.n_episodes_active_scale` absent → 2;
`training.jt.K_V` 3 → 6; `training.jt.K_pi` 1 → 2. **No eval key moves.**

**Run.** Seed 42, 40000/40000 steps, `halt_reason` null, `phase` done, wall **36 796.720 s**,
`cuda_max_mem_mb` peak 6964.96 against ALTBLK's 6865.71 at matched steps (+99.3 MiB — the cost of the
separated collection). 40 value blocks and 40 policy blocks. `best.pt` @ step **33750**, taken inside
a **policy** block (`block_index` 33). In-loop peak cps 0.892962 — **a reading**, not the registered
cell.

**Scored** by `v293_coldstart_score.run_learned` — the same function that produced L356–L359 and
ALTBLK, imported and not modified (`scripts/analysis/v293_altsep_score.py`, a minimal copy of
`v293_alt_blocks_score.py` that differs only in label, output directory, launch record and the number
of comparators). Registered cell, `eval.max_steps` 400, pool fullcb n 2000 ebs 2000. Two-clause
infeasibility identity True, mismatch steps 0; union-vs-evaluator residual −6.9e−18; collision split
residual 0.0.

| field | L359 | ALTBLK | **ALTSEP** |
|---|---:|---:|---:|
| cps | 0.883110 | 0.886728 | **0.889918** |
| cps CI | [0.858859, 0.905181] | [0.865187, 0.909454] | **[0.865196, 0.912096]** |
| reach | 0.9595 | 0.9605 | **0.9625** |
| collision | 0.0235 | 0.0240 | **0.0235** |
| coll_obstacle | 0.0065 | 0.0070 | **0.0070** |
| coll_band_lower | 0.0170 | 0.0170 | **0.0165** |
| stuck | 0.0005 | 0.0005 | **0.0000** |
| timeout | 0.0165 | 0.0150 | **0.0140** |
| incomplete (stuck+timeout) | 0.0170 | 0.0155 | **0.0140** |
| infeasibility | 0.068799 | 0.059241 | **0.061939** |
| SINGULAR_episode_frac | 0.1115 | 0.0795 | **0.0870** |
| EMPTY_episode_frac | 0.5095 | 0.4875 | **0.4875** |
| saturation_rate | 0.419916 | 0.389479 | **0.393078** |

**Paired**, identical pool in identical order, scene bootstrap, 10000 resamples, percentile,
generator seed 20292, floor 0.0083:

| contrast | delta | CI | floor units | inside floor | CI excludes 0 | episodes changed |
|---|---:|---|---:|---|---|---:|
| ALTSEP − **L359** | **+0.006808** | [−0.005259, +0.018825] | 0.82× | **yes** | **no** | 1279 |
| ALTSEP − **ALTBLK** | **+0.003191** | [−0.008837, +0.015361] | 0.38× | **yes** | **no** | 1233 |

**Freezes and the collection gate — 0 violations, in every direction checked.**
`freeze_verify.json` (first cycle) and `freeze_verify_last_cycle.json` (last cycle) both return
**BOTH FREEZES HOLD**, `total_violations` 0. Method 2 over the last-cycle run covers **all 1995
metrics rows**: `grad_leak_pi_from_LV` nonzero rows 0, `grad_leak_VS_from_Lpi` nonzero rows 0,
`held_param_drift` nonzero rows 0, `k_v_active` {6.0} in value rows and {0.0} in policy rows,
`k_pi_active` {0.0} and {2.0} respectively. `freeze_verify_boundaries.json` adds the direction the
script's own non-vacuity block does not cover: across the **policy→value** boundary (900 → 1050) the
value net moves again — 16 tensors differing, max abs 0.11457, and its Polyak target likewise
(16 tensors, 0.05060) — so the policy-block value freeze is non-vacuous; across the **value→policy**
boundary (450 → 600) the policy moves, 6 tensors, max abs 0.02956. The collection gate over all
**4002** `alt_collect.csv` rows: 2001 value rows and 2001 policy rows, **0** D_π episodes in value
rows and **0** D_V episodes in policy rows against 400 200 into each buffer on its own side, 0 rows
with the wrong `*_ran` flag, `collect_mode` `active_only` on every row, `sigma_v_held` violated on 0
policy rows, `sigma_pi_source_is_schedule` violated on 0 rows.

**Signature series — nothing fires.** `signature_series.json`, 40 value-block ends screened on the
thresholds registered before launch (`degenerate_probe.md` §2): **0 saturated, 0 flat**, first firing
`null`, `blocks_csv_vs_metrics_block_column_mismatches` empty. Ranges over the 40 ends:
`probe_h_min` [−0.844759, −0.426361], `probe_h_max` [0.658388, 1.120270], `label_mean`
[−0.437622, +0.452601], `rho_unsafe_label` [0.084106, 0.610311]. The interleaved control screened on
the identical step grid fires **once** (step 500, `label_mean > 0.5`), so on this axis ALTSEP is
cleaner than its own control.

**The three registered predictions — all FALSIFIED**
(`predictions_registered.json`, `registered_at_utc` 2026-08-22T14:46:41Z,
`registered_before_launch: true`; verdicts in `verdicts__ALTSEP.json`):

- **P-C1** *"cps falls below L359 (0.883110) by more than 0.0083."* → **FALSIFIED**, sign opposite:
  +0.006808.
- **P-C2** *"a flat or saturated signature fires at some value-block end, OR the run halts."* →
  **FALSIFIED**: 40 ends, neither signature, no halt.
- **P-C3** *"cps falls below ALTBLK's (0.886728) by more than 0.0083."* → **FALSIFIED**, sign
  opposite: +0.003191.

**The reading: a MATCH, not a beat, against both comparators.** ALTSEP carries the highest point
estimate of the three, and on every outcome share it is at least as good — reach 0.9625 the highest,
incomplete 0.0140 the lowest, collision tied with L359 at 0.0235. But both paired gaps are **inside**
the 0.0083 floor (0.82× and 0.38×) with intervals covering zero, on a **single seed**. Nothing here
separates separated collection from shared collection, or either from the interleaved control. What
the three rows do establish jointly is negative and useful: neither coarsening the alternation to
500-step blocks nor cutting the inactive buffer's supply moves the registered cps off the interleaved
control by a detectable amount.

**Not bold, not promoted.** Single seed 42.

---

## v2.9.3 / quadrotor_3d / TWRX2 doubled per-rotor thrust (quadrotor_3d)

**The first CI-separated beat in the v2.9.3 quadrotor_3d line, and it lands on the failure mode the
atlas identified.** `docs/versions/v2.9.3/failure_atlas.md` found that **72 %** of L359's registered-pool
collisions are floor contacts, from starts within 0.686 m of the floor at a median tilt of 132°,
terminating at a median of step 5 with `contact_vz` negative on all 45 sampled. Doubling the actuator
doubles the authority to arrest exactly that descent. P-X2 predicted floor contacts would fall; they
fell by **71 %**.

**TWO keys of 328**, off L359's persisted config, per the amendment of 2026-08-23 (option A):

```
env.bounds.quadrotor_3d.f_rotor_max   4.905      -> 9.81        (TWR 2.0 -> 4.0)
loss.policy.sat_excess_threshold      [4.905]x4  -> "u_bounds"
```

The second key is not an extra: `sat_excess_threshold` is the actuator box written a second time by
hand (`losses.py:42-51`). Left at 4.905 under a 9.81 box it sits at the box midpoint and penalises,
quadratically at `lambda_sat` 1.0, exactly the authority the axis grants — turning a term identically
zero across all 2000 of L359's metrics rows into the dominant one. The sentinel makes the threshold
track the box. **Verified over the whole run: `L_satex` is 0.0 on all 2000 metrics rows**, and
`abs_action_max` sits at 9.81 throughout, so the policy used the full new box and never paid a
saturation penalty. The contrast isolates the actuator limit.

**Run.** Seed 42, 40000/40000 steps, `halt_reason` null, `phase` done, wall **11.91 h**
(42 960.801 s). `best.pt` @ step **24750**. Four pre-launch gates passed, including a check that the
sentinel *resolves* to `[9.81, 9.81, 9.81, 9.81]` rather than merely that the config string is right.

**Scored** by `v293_coldstart_score.run_learned`, the instrument that produced L356–L359 and every
alternation row, on the registered cell: pool `eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl`,
n 2000, ebs 2000, `eval.max_steps` 400. The doubled actuator travels with the checkpoint —
`gate_overrides` replaces the filter block only — so the scored plant is the trained plant, asserted
at scoring time rather than assumed.

| field | L359 | **TWR×2** | delta |
|---|---:|---:|---:|
| cps | 0.883110 | **0.926611** | **+0.043501** |
| reach | 0.9595 | **0.9695** | +0.0100 |
| collision | 0.0235 | **0.0090** | −0.0145 |
| coll_obstacle | 0.0065 | 0.0040 | −0.0025 |
| **coll_band_lower** | **0.0170** | **0.0050** | **−0.0120** |
| coll_band_upper | 0.0000 | 0.0000 | 0.0000 |
| stuck / oob | 0.0005 / 0.0000 | 0.0005 / 0.0000 | — |
| timeout | 0.0165 | 0.0210 | +0.0045 |
| incomplete | 0.0170 | 0.0215 | +0.0045 |
| infeasibility | 0.068799 | **0.046297** | −0.022502 |
| saturation_rate | 0.419916 | 0.292100 | −0.127816 |
| collision counts | obstacle 13, floor 34, total 47 | obstacle 8, floor **10**, total **18** | floor −24 |

**Paired against L359** on the identical pool in identical order — scene bootstrap, 10000 resamples,
percentile, generator seed 20292, floor 0.0083:

| quantity | delta | CI | floor units | CI excludes 0 |
|---|---:|---|---:|---|
| **cps** | **+0.043501** | [+0.023292, +0.064763] | **5.24×** | **YES** |
| **reach** | **+0.010000** | [+0.001500, +0.018500] | 1.20× | **YES** |
| **floor contact** | **−0.012000** | [−0.017000, −0.007500] | −1.45× | **YES** |

**Both registered predictions HELD** (`registration.json`, `registered_before_launch: true`,
`registered_at_utc` 2026-08-22T08:40:00Z, amended 08:48 to option A with both predictions untouched):

- **P-X1** — *"Doubled thrust raises reach against L359 beyond the admissibility floor."* Falsifier:
  the paired reach gap's CI covers zero, or the gap is inside the floor. **HELD**: +0.0100, CI
  [+0.0015, +0.0185], 1.20× the floor, interval excludes zero.
- **P-X2** — *"Floor-contact collisions fall against L359."* Falsifier: the share is not lower.
  **HELD**: 0.0170 → 0.0050, 34 episodes → 10.

**The reading: a BEAT on this cell, and the only one in this line whose interval clears zero.** Every
alternation row (L364, L365) and the ablation (L363) sat inside the 0.0083 floor with intervals
covering zero. This does not, at 5.24× the floor on cps. **Single seed 42.**

**Two things this is not.** It is **not a like-for-like comparison of methods**: the actuator is part
of the plant, so TWR×2 is a different, easier problem — a quadrotor with twice the thrust-to-weight
is more recoverable, and the result says the paper's failure mode is actuation-limited, not that the
method improved. And **timeout rose** 0.0165 → 0.0210, so a fifth of the collision reduction is paid
back as episodes that no longer crash but also no longer arrive.

**SOTA classification is FLAGGED and NOT TAKEN.** cps 0.9266 exceeds every quadrotor_3d row on this
cell, but the row sits on a different plant from the rows it would displace, and whether a
different-plant row may hold a system's bold is a Researcher decision. **Not bold, not promoted.**

---

## v2.9.3 / quadrotor_3d / fullvia re-scoring (quadrotor_3d)

**Nine rows, one cell, one axis.** Every quadrotor_3d cell this section covers is re-scored on
`eval_fullvia_quadrotor-3d-d2r_n2000_seed823456` — the registered cell in every key except the pool
— with **the checkpoint its existing ledger row names**. No cold checkpoint is substituted for a
warm one and no row acquires a counterpart it lacks. Existing rows are untouched.

**The pool.** Built by `scripts/analysis/v293_build_fullvia.py`, predicate
`src/eval/viability_screen.py`, sha256 `d64212c5bf558cf5…`. The predicate is the doom certificate of
`docs/versions/v2.9.3/doom_certificate.md` — floor, ceiling and horizontal legs — reused rather than
reimplemented, and **proved bit-identical to the scored certificate on all 2000 registered scenes at
both ω settings before any scene was drawn**. Rejection setting `omega_reject` = 8.005974 rad/s, read
from `measured_rate.json`. **45 of 2045 attempts rejected, 2.200 %**, worst case 3 attempts against
the 1000 cap. Gate: 2000 scenes, 0 flagged at both ω, second generation byte-identical by sha256.

**These numbers are NOT paired against the fullcb rows.** The two pools share **no scene at all** —
first divergence at index **0**, because `fullcb` is its predecessor screened and topped up rather
than the raw seed-823456 sequence. The differences below are between-cell differences of two
populations, not paired per-episode statistics, and no interval is quoted for them.

| row re-scored | basis | ckpt | reach | collision | incomplete | cps | Δcps vs its fullcb row |
|---|---|---:|---:|---:|---:|---:|---:|
| nominal | warm | — | 0.4510 | 0.5490 | 0.0000 | −0.647000 | +0.000000 |
| L321 OC-PNCBF | warm | 27450 | 0.8975 | 0.0560 | 0.0465 | 0.725676 | −0.007298 |
| L328 warm JT | warm | 9450 | 0.9620 | 0.0245 | 0.0135 | 0.882989 | −0.002323 |
| L359 COLD40K | cold | 32550 | 0.9605 | 0.0230 | 0.0165 | 0.885990 | +0.002879 |
| L355 cold@10k | cold | 9600 | 0.9630 | 0.0270 | 0.0100 | 0.870485 | +0.001908 |
| L363 COLDABL | cold | 31500 | 0.9620 | 0.0225 | 0.0155 | 0.890252 | +0.015939 |
| L364 ALTBLK | cold | 32400 | 0.9600 | 0.0230 | 0.0170 | 0.888091 | +0.001364 |
| L365 ALTSEP | cold | 33750 | 0.9620 | 0.0235 | 0.0145 | 0.888961 | −0.000958 |
| **L366 TWRX2** | cold | 24750 | **0.9745** | **0.0075** | 0.0180 | **0.937603** | +0.010992 |

**The screen moves almost nothing, and that is the finding.** Seven of the nine cps differences are
inside the 0.0083 admissibility floor; only L363 (+0.0159) and L366 (+0.0110) exceed it, and both in
the direction the screen predicts — removing certainly-lost scenes can only help a controller that
was losing them. Floor-contact share falls on every learned row (L359 0.0170 → 0.0150, L364 0.0170 →
0.0145, L365 0.0165 → 0.0140, L366 0.0050 → **0.0010**), which is the screen doing exactly what it
was built to do. **The ordering of the learned rows is unchanged**: TWRX2 > COLDABL > ALTSEP > ALTBLK
> COLD40K > cold@10k > warm JT > OC on both pools.

**One coincidence, checked rather than reported as a bug.** The nominal is bit-identical across the
two pools on every aggregate — reach 0.4510, collision 0.5490, cps −0.647000. That looked like the
patched pointer not taking effect. It did: the artifact records
`pool = eval_fullvia_…`, and **970 of 2000 nominal episodes differ episode-for-episode**. The goal
count lands on exactly 902 on both populations by coincidence.

**SOTA classification across the two pools is FLAGGED for the Researcher and NOT taken.** Nothing is
bolded and no existing row is edited or un-bolded. Single seed 42 throughout.

---

## v2.9.3 / quadrotor_3d / PPOLAG fixed-penalty frontier (quadrotor_3d)

**Seven rows, L390 to L396**, one per collision-penalty multiplier λ ∈ {0, 0.3, 0.5, 1, 2, 3, 10}.
Registered **2026-08-30 by Researcher decision**, from evaluations **scored 2026-08-26**; the rows are
diagnostic frontier points, not an axis result, and nothing is bolded or promoted.

**What the cell is.** A certificate-free PPO agent trained with a **fixed** collision penalty — no
dual variable, no certificate, and an identity filter, so `infeasibility` is **0 by construction** on
every one of the seven rows and is not comparable with a filtered row's. Scored through the shipped
PPO scoring path on pool `eval_fullvia_quadrotor-3d-d2r_n2000_seed823456.manifest.json` under
`data/secured_data/pools/`, n 2000, `ebs` 2000, `eval.max_steps` 400, seed 42.

**Which build-log documents this sweep.** `docs/versions/v2.9.3/ppo_dense_cost.md` §§1–2, which
assembles the frontier and uses λ = 2 as the sparse incumbent its dense costs are measured against.
**Not** `docs/versions/v2.9.3/ppo_lagrangian.md` — that file is the *adaptive-dual* run, which halted
under its own stop condition 3 with the dual pinned at its bound for 938 iterations, reports no `cps`
and registers no cell. The two must not be conflated: one is a λ sweep with λ held fixed per run, the
other is a single run with λ adapted and abandoned.

| ledger row | λ | `cps` | `cps_ci` | reach | collision | oob | stuck | timeout | ckpt step |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| L390 | 0 | −0.37500 | [−0.4375, −0.3070] | 0.5385 | 0.4520 | 0.0000 | 0.0095 | 0.0000 | 72204 |
| L391 | 0.3 | −0.25200 | [−0.3153, −0.1907] | 0.5780 | 0.4085 | 0.0000 | 0.0125 | 0.0010 | 83836 |
| L392 | 0.5 | 0.62500 | [+0.5800, +0.6690] | 0.8725 | 0.1200 | 0.0000 | 0.0075 | 0.0000 | 120740 |
| L393 | 1 | 0.66750 | [+0.6245, +0.7078] | 0.8845 | 0.1030 | 0.0000 | 0.0095 | 0.0030 | 137040 |
| **L394** | **2** | **0.71425** | [+0.6728, +0.7513] | **0.8955** | **0.0825** | 0.0000 | 0.0105 | 0.0115 | 158508 |
| L395 | 3 | 0.61825 | [+0.5748, +0.6593] | 0.8520 | 0.1020 | 0.0010 | 0.0135 | 0.0315 | 192668 |
| L396 | 10 | −0.63350 | [−0.6533, −0.6160] | 0.0000 | 0.0800 | 0.1195 | 0.0270 | 0.7735 | 271216 |

**The frontier is single-peaked at λ = 2 and both ends fail differently.** Below λ = 0.5 the penalty
is too weak to buy safety at all — collision 0.4520 and 0.4085, worse than the unfiltered nominal's
0.5490 only because reach is also collapsing. Above λ = 3 the penalty dominates the return: at λ = 10
the agent **never reaches a goal** — reach exactly 0.0000, timeout 0.7735, `oob` 0.1195 — while its
collision, 0.0800, is the lowest of the seven. **A controller that avoids every obstacle by refusing
to move is the degenerate solution the penalty route admits and the certificate route does not**, and
λ = 10 is that solution reached.

**The best point of the sweep is L394 at `cps` 0.71425**, against the joint pair's 0.89025 on the same
pool and cap (L372). The gap is **0.176**, far outside the 0.0083 admissibility floor, and L394's
interval [+0.6728, +0.7513] is disjoint from L372's [+0.86799, +0.91274]. **The manuscript's
PPO-Lagrangian sentence rests on this sweep**, and L394 is the row it should cite.

**No comparison against a filtered row may use `infeasibility`**, which is structurally 0 here rather
than measured, and no `saturation_rate` on these rows is comparable with a filtered row's either.

**Provenance note carried from the artifacts.** All seven score jsons were written without a `pool`
field and had it **back-filled 2026-08-26** by the dense-cost dispatch, which re-scored each named
checkpoint on this pool through the same producer and reproduced every count, every component and
`cps` to the digit before writing the three pool fields. **No number in any of the seven files was
changed.**

Single seed 42 throughout. **NOT bold, nothing promoted.**
