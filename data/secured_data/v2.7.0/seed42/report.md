# v2.7.0 — results.md (close verdict)

Axis: quadrotor observation gravity-direction restoration (dim 20 -> 22, `01_env` §3.3, ratified
pre-implementation) + four Researcher-directed within-version iterations. Seed 42 throughout;
SINGLE-SEED close per the seed-economy decision (2026-07-14); scene-bootstrap 95% CIs; canonical
headline path = run_full `evaluate()` batch=2000. Multi-seed escalation deliberately deferred to
the post-corridor-axis configuration (Researcher decision at close). All numbers below are
disk-verified in `docs/versions/v2.7.0/retrieve_close_facts.md` (RCF), which records artifact
paths and the discrepancy adjudications cited here.

## 1. Headline verdict (SOTA, single-seed)

**Iteration 1 (obs fix) is the (v2.7.0, quadrotor_planar) full-range baseline and SOTA-bold,
superseding v2.6.2 (0.7311).**
- cps **0.8232** [0.7910, 0.8523]; reach 0.9400; collision **0.0455** [0.0370, 0.0550];
  oob 0.0000; stuck 0.0015; timeout 0.0130; infeasibility 0.0593. Independent recompute matches
  the stored column to <1e-3 and the stored CIs within bootstrap-RNG noise (RCF §1, §7.1).
- Run `data/v2.7.0__20260717-025050__seed42`, best.pt @46500 (sha8 22b902a6). Value chain: M2
  oc V_hat @45000, obs-22, run `…021858` (sha8 4660ca90), in-loop best 0.5581; pre-JT (M3) cps
  0.6490 [0.6124, 0.6823] / collision 0.0495 (CI cited from report §M3; raw json
  non-persistent — RCF §7.3). Insertion cps: lqr -0.3920 / frozen 0.7841 / live 0.8043.
- vs v2.6.2: cps +0.0921 CI-separated (CI-lo 0.7910 > 0.7648); collision -32% CI-separated
  (CI-hi 0.0550 < 0.0565).

## 2. Registered-hypothesis outcomes (obs axis)

- **S1 CONFIRMED** — theta-decodability overall 1.1 deg (was 12.8); failure cell 2.28 deg (was
  98.6). Iter-5 spot-check 2.345 deg (obs unchanged). **S2 CONFIRMED** — aliasing eliminated
  (obs_gap p50 1.44 from ~1e-6; pi-gap p50 2.31). **P2 CONFIRMED (CI-separated)**; the
  registered falsifier branch did NOT trigger (collision moved).
- **S3 NOT confirmed** — infeasible-at-birth by tilt band 5.6% / 9.4% / 18.2% (ref 16.8% at
  high tilt); 64.5% of high-tilt doom-labeled starts still reach (ref 61.5%).
- **P1 NOT met** — tilted-band share concentrated: 76/94 = 0.809 (v2.6.2: 84/123 = 0.683);
  absolute tilted count fell 84 -> 76 while the total fell more.
- **P3 CONFIRMED** — 94 residual (manual-loop count; canonical 91 — §6.2), 0/94 collinear,
  ||L_g V_hat|| median 0.9496 (ref phi 0.453), degen 9.6%; S_h did not transfer.

## 3. Iteration 5 — continuing-batch collector (Track A): ADOPTED

- Final: cps 0.8343 [0.8038, 0.8632]; reach 0.9425; collision 0.0410 [0.0325, 0.0500]; oob
  0.0000; stuck 0.0010; timeout 0.0155; infeasibility 0.0583. Run
  `data/v2.7.0__20260718-000752__seed42`, best.pt @48000 (sha8 3b27d691). Value chain: G2
  continuing, best @39000 (sha8 bacdc3ae), in-loop best 0.5174; pre-JT 0.5191 [0.4735, 0.5633]
  / 0.0950 — CI-separably weaker than iter-1's pre-JT, fully recovered by JT (resilience
  datum 2, after iter-2's). Insertion cps: lqr -0.4055 / frozen 0.7955 / live 0.8008.
- **H-nondegradation CONFIRMED** (both metrics CI-overlap iter-1; NOT SOTA-separated — iter-1
  keeps bold). **H-throughput NOT MET** (offline episodes/hour 948,101 vs 1,004,025 = 0.94x;
  wall-clock value 29.7 vs 26.4 min = 1.13x, JT 230.0 vs 227.6 min = 1.01x).
- Adoption grounds are label provenance at ~equal cost, not throughput: hover fraction 0.145 vs
  ~0.70; per-episode horizon 200 real labels replace the 100-step bootstrap chain; time-unbiased
  sampling via round carry-over; all semantics gates green (bit-parity, segment isolation, label
  recompute 0.00e+00; collector tests 5/5; full suite 139). Distribution side-effects recorded:
  rho_unsafe 0.169/0.180 vs 0.236/0.249; collect_infeasible 0.000 vs 0.0395-0.0880.
  Trained-policy mix R1:R2:R3 = 0.942:0.049:0.008; effective episode length 78.2 steps.
- **collector=continuing is the default configuration going forward** (legacy retained for
  ablation/parity).

## 4. Refutations registered this version

1. **IC-level tilted-band oversampling (iter-2)** — natural band mass ~0.5 makes
   inject_frac=0.10 a null top-up; final 0.8327 [0.8015, 0.8618] / 0.0430, CI-overlap with
   iter-1; mechanism unchanged. Pre-JT eval datum 0.4901 / 0.1025 (RCF §7.4): the value
   pessimism JT recovered — resilience datum 1.
2. **bptt_T = 60 (iter-3)** — training collapse: in-loop peak 0.7244 @21000; first collapse
   @28500 (cps -0.5595), terminal regime from 34500 (-0.6753); grad_norm_pi (pre-clip) 113.3 ->
   311.9 -> 4807.8 -> 14829.9 over 27k-36k; process-terminated @46459 (halt_reason None — no
   hard halt tripped). Refuted (unstable) at this config.
3. **Bootstrap-chain hypothesis (iter-5)** — horizon-200 real labels left the monotone-descent
   mode unchanged: 64/65 tilted (0.985) vs iter-1 62/67, iter-2 59/64; class-a late-rise = 0.
4. **BPTT-window axis grounds dissolved** — iter-5 tau_rec p50 15 / p90 37 / max 74 (baselines
   35-36 / 57-59): successful recovery sits inside the window-30 credit.

## 5. Mechanism state at close (carries to v2.8.0)

The corridor defect survives the obs fix, IC oversampling, and the label-provenance repair —
the sole surviving certificate defect: monotone-V_hat-descent-into-collision 64/65 tilted
(+13/13 non-tilt, +1 descended-never-safe = 78 agreed); infeasible@t0 0.872 (58/65 tilted
0.8923); born-inside 68 / fast-approach 10; ||L_g V_hat|| at collisions median 0.4915 vs
population 7.26 (authority collapse at failure cells; degen 4.8%). P-B authority 2x2 at t0
(n=78): {cont-infeasible x disc-improvable} 28 / {cont-infeasible x both-dead} 40 / feasible
10 (5+5); tilted 20/38/3/4; m_c-vs-deployed-empty agreement 78/78. Two recorded caveats bind
its reading: the 1-step m_d underestimates torque-mediated O(Dt^2) authority, and both margins
are computed on the certificate known wrong in the corridor. Registered next axis:
**corridor-cell state injection** (joint cell near-surface x high-tilt x inward-speed;
natural-mass pre-gate; distinct from the refuted IC-band lever). Sequenced after it: k-step
authority remeasure (P-B'), discrete-row filter (theory T3), unsafe-start policy credit. 3D
extension reviewed and deferred to post-corridor (theory largely dimension-free; new item =
sec:obs 3D corollary, quotient by yaw only).

## 6. Caveats and bookkeeping (binding)

1. **CI provenance**: headline CIs are the stored values; independent recompute agrees within
   bootstrap-RNG noise (<=0.001) — either citation is valid (RCF §7.1).
2. **Residual-count denominators are code-path-dependent** (RCF §7.2): collision RATES use the
   canonical `evaluate()` batch-2000 counts (iter-1 91; iter-5 82); P1/P3 SHARES use the
   manual-loop counts (94; 83); anatomy/authority probes use the roll-vs-eval AGREED sets
   (78/78) with boundary episodes excluded and reported. Three paths, three counts — each
   stated with its denominator wherever cited.
3. Pools unchanged and reused throughout (`eval_full_quadrotor-planar_n2000_seed23456`;
   in-loop n500 seed 12345); POOL_MANIFEST_ERRATA.md caveat unchanged.
4. Ledger: seven v2.7.0 rows (L114-L120); exactly one bold (L115, iter-1 headline); v2.6.2
   rows plain. Iter-5 note carries the H-throughput NOT-MET correction.
5. verify.sh at close: 139 passed. Version strings consistent at v2.7.0.
6. Theory note: the 10-section canonical tex (incl. sec:labeling) accompanies this close and
   supersedes the 8-section v2.6.0 note in the repo.