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
## v2.9.1 · double_integrator · L62 current-basis re-score

`anchor: v2-9-1--double-integrator--l62-current-basis-re-score` · ledger line 314 at the time of writing · cps `**0.8682**` · date `**2026-08-14 09:41:00**`

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62, whose own verdict records 'cps_v2 reads -, so this row is not rankable under the current definition -- bold retained but pending re-scoring'. L62 NAMES NO CHECKPOINT: its cells name a 3-SEED AGGREGATE (seeds 42, 12345, 99; D_pi=2k + D_V=1M @50k) and a parent of '-'. What identifies one is the per-seed secured snapshot, and all three were LOCATED: data/secured_data/v2.3.0/seed{42,12345,99}/checkpoints/best.pt, 5273637 bytes each, best_step 43500 / 49500 / 46500 respectively; all three strict-load under current code at obs_dim 19. BASIS, from their own embedded configs: goal_angrate_radius ABSENT, band keys ABSENT, filter.projection ABSENT, hazard ABSENT, ceiling ABSENT -> 1.0; scoring PREDATES the angular-rate terminal and the cps_v2 transition. env.goal_angrate_radius = 0.3 was added explicitly to match the L305-L312 cell; on double_integrator this is STRUCTURALLY INERT, since angular_rate is a hard zero (double_integrator.py:69-71), so the third leg is satisfied identically either way. POOL CAVEAT, load-bearing: scored on eval_full_di_n2000_seed123456, the v2.9.1 REGENERATED pool, which is what L307/L309 use and is therefore what makes this row comparable to them -- but it is NOT the DI pool of record, because the 04_eval 6.2 registration delta is drafted and uninstalled. This row is therefore comparable to L307/L309 and NOT to L62 itself or to any historical DI row. RESULT: 3-seed mean 0.868247 (per seed 0.867112 / 0.876933 / 0.860695; sd 0.008178) against L62's 0.8698, i.e. -0.001553 -- BELOW the bold, by 0.19x the 0.0083 admissibility floor, which is well inside the seed spread and is NOT a separation. BOLD MOVED HERE AT THE v2.9.1 CLOSE (Researcher decision, 2026-08-14): this row carries the double_integrator bold and supersedes L62, which keeps its content and loses bold. The move is a BASIS re-score of the SAME 3-seed checkpoint set onto the current cell -- NOT a 04_eval 5 CI-separated beat and not an improvement claim, since 0.868247 is BELOW L62's 0.8698 by 0.19x the admissibility floor and inside the seed spread. The natural successor L309 (0.904757) sits in secured_data/v2.9.1/experiments/, bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md**

### relocated from the `verdict` column (pre-cap text, verbatim)

**CURRENT-BASIS RE-SCORE of the FLAGGED bold row L62, which names no checkpoint but a 3-SEED AGGREGATE; all three secured snapshots (seeds 42, 12345, 99) were located and strict-load under current code at obs_dim 19. RESULT 3-seed mean 0.868247 (sd 0.008178) against L62's 0.8698, i.e. -0.001553, BELOW the bold by 0.19x the 0.0083 admissibility floor, well inside the seed spread and NOT a separation. Scored on the v2.9.1 REGENERATED DI pool, which makes this row comparable to L307 and L309 and NOT to L62 itself or to any historical DI row, since that pool is not yet the pool of record. BOLD MOVED HERE at the v2.9.1 close (Researcher decision, 2026-08-14), superseding L62, which keeps its content and loses bold -- a BASIS re-score, not an improvement claim, the natural successor L309 being bold-INELIGIBLE per 06_workflow 6.3. Detail docs/versions/v2.9.1/bold_rescore.md and docs/ledger_verdicts.md#v2-9-1--double-integrator--l62-current-basis-re-score**

### relocated from the `eval_source` column (pre-cap text, verbatim)

**eval_only(current-basis re-score of the L62 3-seed checkpoint set, registered cell + explicit env.goal_angrate_radius 0.3, pool eval_full_di_n2000_seed123456 (n 2000, the REGENERATED pool -- same as L307/L309, NOT the pool of record) -- ebs 2000, terminal (0.15, 0.3, 0.3), the three secured seed snapshots @ steps 43500/49500/46500 (paths in the verdict); 3-seed mean of per-seed cps, each recomputed from its own components, max residual 1.1e-16); artifacts data/runs/v2.9.1/launch/boldrescore__di_seed42.json, data/runs/v2.9.1/launch/boldrescore__di_seed12345.json, data/runs/v2.9.1/launch/boldrescore__di_seed99.json; pool registration delta pending (Researcher-approved, deferred)**

<a id="v2-9-1--quadrotor-3d--best-pt-valonly3d-v291"></a>
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

