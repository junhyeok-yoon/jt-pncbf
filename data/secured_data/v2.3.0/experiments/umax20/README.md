# u_max=20 diagnostic (base v2.3.0)

Diagnostic reference run, NOT a SOTA snapshot.

- Base: v2.3.0 (commit 8dbda0c), per-component-uniform DI start sampler, seed 42.
- Single global change: actuator/control bound `u_max` 2 -> 20 (policy output range, LQR clamp,
  RK4 dynamics, and HardNet projection box all become +/-20; `v_max` stays 2.5). Box stays active;
  `disable_box=false`. See `source_changes.patch` for the exact delta.
- Evaluated checkpoint: `checkpoints/best.pt` @ step 72000 (in-loop best, NOT n2000-reselected).
  `step_072000.pt` and `final.pt` (@100000) are included; `best.pt` is the checkpoint the eval used.
- Result (frozen n2000 / seed 23456): cps 0.9419 [0.9290, 0.9551], reach 0.9805, collision 0.0065,
  oob 0.0, stuck 0.0010, timeout 0.0120, infeasibility 0.0620.
- Verdict: SOTA UNCHANGED (different actuator spec). The gain over v2.3.0 (cps 0.8700) is a
  different, easier problem (~10x acceleration authority) plus the +/-20 box rarely binding; it is
  not a +/-2-deployable result. Full interpretation: `docs/versions/v2.3.0_results.md`, section
  "Experiment: u_max widening (2 -> 20)".
- Source run dir (in-flight, git-ignored): `data/previous_runs/v2.3.0_umax20__20260628-212951__seed42`.
  Excluded here as scratch: `tensorboard/`, per-step `metrics.csv`, `*.log`, `figures/inloop/`.
