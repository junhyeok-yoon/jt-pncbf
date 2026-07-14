## FINAL COMPARISON TABLE (full n2000; arm-B' = minimal-intervention ladder)

| config | arm-A cps_v2 [CI] | reach | coll | stuck | inf_v2 | arm-B | arm-B' | vstart-coll |
|--------|-------------------|-------|------|-------|--------|-------|--------|-------------|
| v2.3.0 DI SOTA (value CBF, 3-seed) | **0.9005** | — | — | — | — | n/a¹ | n/a | n/a |
| v2.5.0 B-2 (maneuver, s42) | 0.8327 [0.805,0.856] | 0.9250 | 0.0055 | 0.0430 | (empty-only) | n/a¹ | n/a | n/a |
| R1 it2_ws (learned V̂₁, s42) | 0.7378 [0.705,0.771] | 0.8815 | 0.0240 | 0.0635 | 0.0557 | 0.7985 | — | 0 |
| **E2 exact_m0 (s42)** | 0.8226 [0.797,0.847] | 0.9090 | 0.0015 | 0.0520 | 0.0423 | 0.8367 | 0.8367 | 0 |
| **E pooled (3-seed)** | 0.8160 [0.801,0.829] | ~0.907 | 0.0018 | ~0.058 | ~0.044 | ~0.831 | — | 0 |
| **best = E_s99 + S (arm-B′)** | **0.8280** | 0.9135 | 0.0015 | 0.0550 | 0.0417 | **0.8422** | 0.8422 (Δ0) | 0 |

¹ value-CBF (v2.3.0) and analytic-maneuver (v2.5.0) finals report a single filtered arm; the S′ successor-
verification shield (arm B/B′) is a CPI-line construct. arm-B′ Δstuck = 0.0 everywhere (P-S1 false). D2 is
ABANDONED (Researcher, P-D1 falsified) and excluded. Best of the version = **E_s99 arm-B 0.8422 / arm-A
0.8280**; **still below the v2.3.0 DI SOTA 0.9005.**
