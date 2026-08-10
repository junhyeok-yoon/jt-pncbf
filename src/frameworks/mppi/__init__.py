"""v2.8.4 MPPI baseline — information-theoretic sampling MPC (Williams et al.) for quadrotor_3d.

A PRIVILEGED, model-based, training-free reference arm. It is NOT a learned-certificate arm and is not
comparable to one without stating the asymmetry:

    MPPI reads the FULL 13-D state and the obstacle field (centers / radii / active) directly, and it
    rolls the TRUE deployed plant forward internally. Every learned arm (JT / OC / CPI / PPO) sees only
    the 34-D observation and has no plant model at deployment.

Additive package. It imports src.common (plant map, outcome predicates), src.envs and src.eval
(pools, scoring); it never imports jt_pncbf / oc_pncbf / cpi / ppo and it edits nothing outside
`src/frameworks/mppi/` (05_code §2).

Modules
-------
cost.py             the five cost terms; every coefficient read from `config.yaml`, none hard-coded.
mppi_controller.py  the sampler / exponential weighting / receding-horizon loop + the eval adapter.
evaluate_mppi.py    scores one (sigma, lambda, H, N, C_crash, m) cell on a pool; per-episode arrays,
                    ESS, and the spawn-tilt reporting split.
screen_grid.py      the screens, run serially, with the VRAM and gamma-arm guards.
repro_gate.py       the backward-reproducibility gate: the superseded sampler still reproduces exactly.
diagnose.py         screen diagnostics (measurement only).
cpu_smoke.py        the CPU correctness checks (tiny settings, seconds).
config.yaml         every coefficient, the screening grids, and the eval-cell overrides.
recovery.py         the charter-"v3" components B1 / B2 / B3, all config-switched and all default OFF.
screen_recovery.py  the charter-"v3" screen (grid_v3 with B1+B2+B3 on) and the two ablation tables.
smoke_recovery.py   the charter-"v3" smoke, including the whole-cell reproduction of an S3 cell.
rounds_goal_v4.py   the charter-"v4" goal-attraction rounds and the reporting table format.
stages_v5.py        the charter-"v5" three-stage driver (sample budget / cascade / relaxed companion).
cascade.py          the charter-"v5" Stage-2 CASCADED planner — A SEPARATE BASELINE, see below.
relaxed_v5.py       the charter-"v5" Stage-3 re-scoring of stored rollouts under a second terminal.

================================================================================================
THE CHARTER-"v5" CASCADED VARIANT IS A SEPARATE BASELINE ROW — READ THIS BEFORE REPORTING IT
================================================================================================

`cascade.py` (charter "v5" Stage 2; artifacts `data/runs/v2.8.4/mppi_v5/`, build log
`docs/versions/v2.8.4/mppi_v5.md`) is CASCADED MPPI and LEAVES THE VANILLA MPPI CLASS. Vanilla MPPI
plans in the plant's own input space and maps the sampled sequence to rotor forces by a
STATE-INDEPENDENT allocation. The cascaded planner plans an OUTER-LOOP command — the desired world
acceleration — and closes the shipped inner attitude PD (`System.lqr_action`, the controller
`src/common/filter_backup.py` names, reached through `filter_backup.override_gains`) underneath it, so
what the plant receives at rollout step k depends on that sample's own state at step k.

    IT IS FILED AND MUST BE REPORTED AS ITS OWN BASELINE ROW. Its numbers must NEVER be blended into,
    averaged with, or substituted for vanilla MPPI numbers, and a cascaded row must NEVER sit in a
    table with a vanilla row unless an explicit VARIANT column is present.

`evaluate_mppi.run_cell` stamps `cell.variant = "cascaded"` and rewrites `arm` on every cascaded
record, and `stages_v5.STAGE2_COLUMNS` puts `variant` first in the Stage-2 table, so the separation
lives in the artifacts and the tables and not only in this docstring.

The charter-"v5" BASE CONFIGURATION, and the naming resolution it needed. The charter names "the v3
timeout-dominant cell" but also fixes B1-B3 off and G1-G4 off. With the switches OFF that is the
charter-"v2" controller, whose retained row is
`data/runs/v2.8.4/mppi_screen_v3/cell__N1024_lam0.05_C100000_m1_H40_sig1.json` (reach 0.0000,
collision 0.2400, timeout 0.7600 at n = 400) — NOT the same-named row in `data/runs/v2.8.4/mppi_v3/`
(collision 0.2225), which has B1 = B2 = B3 = true. The switches are decisive; see `mppi.v5.base_cell`.

A FOURTH SCREEN, the charter's "v3". The charter's v1/v2/v3 labels do NOT match the directory names:
its "v1" is S1 (`mppi_screen/`), its "v2" is S3 (`mppi_screen_v3/`), and its "v3" is the B1/B2/B3 work in
`data/runs/v2.8.4/mppi_v3/` with the build-log at `docs/versions/v2.8.4/mppi_v3.md`. S1, S2 and S3 are
immutable and untouched. All three recovery switches default off, so this package's shipped config still
resolves to the S3 controller; `smoke_recovery.py` check (f) re-runs an S3 cell with them off and
compares the whole cell, per-episode array by per-episode array.

THREE SCREENS, three names — do not conflate them.

    S1  the ORIGINAL 16-cell grid (N x H x lam x C_crash). 8 of 16 cells ran before a false-positive
        liveness stop. Artifacts `data/runs/v2.8.4/mppi_screen/`. The S3 dispatch calls it "v1".
    S2  the R1-R4 amendment's 12-cell grid (sigma x lam_rel x H).
        Artifacts `data/runs/v2.8.4/mppi_screen_v2/`.
    S3  the current 16-cell grid (N x lam_rel x C_crash x control-hold m, H = 40, sigma = 1.0).
        Artifacts `data/runs/v2.8.4/mppi_screen_v3/`. The dispatch calls it "v2" — S3 is NOT the
        `mppi_screen_v2` directory.

v2.8.4 AMENDMENT (R1-R4). The sampler carries the plan in BODY-WRENCH space and allocates through the
system's own relation (R1), perturbs it with stationary OU noise in time (R2), and sets lambda RELATIVE
to the running std of the sample costs (R3); the screen's liveness poll tolerates the training arm's
in-loop-eval pause (R4).

S3 CHANGES. (1) The sampling centre is the EXPLICIT decomposition u = u_hover + u_plan + eps, u_hover a
fixed anchor read from the system object. (2) The terminal cost is the deployed terminal predicate's own
excess — relu'd position, speed and angular-rate legs with all three radii read from the deployed config
— so it is zero exactly where the reach predicate closes. (3) A control hold m applies each decision
entry for m physical steps, giving an H*m*dt lookahead and re-planning every m steps. (4) Every cell also
reports the full decomposition over spawn tilt < 90 deg and >= 90 deg; born-inverted episodes are never
filtered out of a headline number.

BACKWARD REPRODUCIBILITY. S1 and S2 stay exactly reproducible through the legacy switches
`space=rotor, noise=iid, lam_mode=absolute` (S1's sampler) and `center=none, control_hold=1,
terminal=distance` (the pre-S3 controller); the gate re-runs an S1 cell and compares every metric.
"""
