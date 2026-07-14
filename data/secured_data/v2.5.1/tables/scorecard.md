## FULL VERSION SCORECARD (every registered prediction; verdict + disk citation)

| prediction | threshold | result | verdict | source |
|------------|-----------|--------|---------|--------|
| P-PI1a frac(G⊨C) | ≥0.70 | 0.7767 | CONFIRMED | phase_i0 §scorecard |
| P-PI1b IoU / coverage | ≥0.95 / ≥0.90 | 0.9599 / 0.9096 | CONFIRMED | phase_i0 |
| P-PI1c eps_q(0.10) | ≤0.05 | 0.0189 (all seeds ≤0.047) | CONFIRMED | phase_i0 |
| P-I1-1 it1 arm-A cps_v2 | ≥0.85 | 0.678 | REFUTED | phase_i1 / ledger |
| P-I1-2 verified-start collisions | =0 | 0 (every arm, all phases) | CONFIRMED | ledger / all dual-evals |
| P-I1-3 k=3 fixed-set monotonicity | 0 viol | 53,610 (single-step) | HALT → SPEC ERROR (cumulative fix) | phase_i1 / a1_result.json |
| P-PI2 loop gate (arm-A≥0.83 monotone) | — | 0.678/0.721/0.660 | REJECTED | phase_i1 / ledger |
| P-A1 cumulative monotonicity | 0 viol | 0 (A1, A6 V₂′≤V₁, P1.5 V₃≤V₂′) | CONFIRMED | a1/a6/p1_5_result.json |
| P-A2(i) per-seed Δ(it2_ws−it1)>0 | CI>0 | +0.060/+0.078/+0.063 | CONFIRMED | phase_i2 A5 |
| P-A2(ii) pooled it2_ws | ≥0.75 | 0.7411 [0.724,0.760] | REFUTED (disjoint-CI improvement) | phase_i2 A5 |
| P-B1 horizon critic | cps≥0.73,stuck≤0.045,coll∈[.015,.045] | 0.409 / 0.2245 / 0.028 | REFUTED | phase_i2 A5 / ledger |
| P-P1a on-policy eps_q(0.10) | ≤0.05 | 0.0 | CONFIRMED | p1_2_result.json |
| P-P1b pi_3 coll & cps | ≤0.015 & ≥0.76 | 0.059 / 0.656 | REFUTED | phase_i2 P1.4 / ledger |
| P-P1c pi_3 arm-B | ≥0.80 | 0.785 | REFUTED | phase_i2 P1.4 / ledger |
| P-E1 exact_m0 s42 coll & cps | ≤0.010 & ≥0.80 | 0.0015 / 0.8226 | CONFIRMED | phase_i3 E2 / ledger |
| P-E2 pooled exact_m0 arm-A | ≥0.79 | 0.8160 [0.801,0.829] | CONFIRMED | phase_i3 E2 / ledger |
| P-D1 pi_bc arm-A reach | ≥0.90 | 0.0 (oob) | FALSIFIED | phase_i3 D1 |
| P-D2 fine-tuned cps & stuck | ≥0.83 & ≤0.035 | — (D2 abandoned by Researcher) | NOT SCORED | phase_i3 D2 |
| P-S1 arm-B′ stuck reduction | ≤ armB−0.015 | Δ0.0 (E2, E_s99, D2) | REFUTED | phase_i3 S / s_*.json |
| W (park-targeting terminal) | — | not run (park 0/104; chatter 101) | NOT RUN | phase_i3 CLOSE-PREP |
