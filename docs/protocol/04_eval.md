# 04 — Evaluation

This document defines **what we measure, on what, and how we report it**. It does not redefine the environment, agent, or learning loop — those live in `01_env`, `02_control`, `03_train`. It only defines the evaluation harness.

The evaluation harness is **framework-agnostic**: it runs on any policy + filter combination (LQR-only, learned no-CBF, CBF-QP filtered, HardNet filtered) and produces a fixed set of artifacts. Frameworks plug into it; the harness does not branch on which framework.

## 0.1 Evaluation invariants

The harness enforces five invariants by construction, each closing a class of measurement error:

- **One metric name, one formula.** A single name (`cps`) and a single formula (§1) carry $(reach - 2 \cdot collision - \text{stuck} - 0.5(\text{oob} + \text{timeout}) - 0.3 \cdot infeasibility)$. No alternate field name and no simpler $(reach - collision)$ variant exists anywhere.
- **One outcome-priority rule.** Outcomes are resolved by the `01_env` §1.6 predicates verbatim (collision before goal; exact contact, no radius inflation). No evaluation path re-derives them.
- **One infeasibility definition.** Infeasibility is the mean over active filtered steps of the per-step infeasible flag (`02_control` §4: empty half-space–box intersection, or singular-and-violated row), then the mean over episodes (§1). No other aggregation is used.
- **Reproducible pools.** Evaluation pools are generated once, serialized, and committed to git (§6); nothing is sampled in memory at evaluation time, so any two reports are on bit-identical scenes.
- **Variance always reported.** Multi-seed bootstrap CIs are required for any aggregated number (§5). A single-seed result may be reported as a version's headline only when explicitly marked single-seed and supplemented by a scene-bootstrap CI; multi-seed is strongly recommended and is the requirement once additional seeds are available.

---

## 1. Metric (canonical)

The single headline metric for every comparison in this project is `cps` ("composite per-scenario"):

$$
\text{cps} = \text{reach}
\;-\; 2 \cdot \text{collision}
\;-\; \text{stuck}
\;-\; 0.5 \cdot (\text{oob} + \text{timeout})
\;-\; 0.3 \cdot \text{infeasibility}.
$$

Defined per scenario; the reported value is the mean across the scenarios in a pool. Components:

- **`reach`** — fraction of episodes whose outcome is goal-reached (per `01_env` §1.6 priority).
- **`collision`** — fraction of episodes whose outcome is collision.
- **`stuck`** — fraction of episodes whose outcome is stuck (per `01_env` §1.6: $\mathrm{disp}_t \le r_{\text{stuck}}$ over a window of $W_{\text{stuck}}$ steps; defaults $0.10$ m / $60$ steps / $3$ s). Penalized at weight 1.
- **`oob`** — fraction of episodes whose outcome is out-of-bounds. Penalized at weight 0.5 (jointly with timeout).
- **`timeout`** — fraction of episodes whose outcome is timeout. Penalized at weight 0.5 (jointly with oob).
- **`infeasibility`** — defined in `02_control` §4: a filtered step is infeasible iff the half-space–box intersection is **empty**, or the row is **singular and violated** ($\|L_g h\| < 5 \times 10^{-4}$ and $L_f h + \alpha\, h_{\text{eff}} > 0$, a $u$-independent test). A singular row that is satisfied is feasible — every $u$ satisfies it and the filter is simply inactive. Aggregation: mean over active filtered steps of the per-step flag, then mean over episodes; episodes with no active filtered steps contribute zero.

  **History note — recorded by explicit Researcher decision as a one-time exception to `00_constitution` §3 Prohibition 3, to prevent mis-scoring across the definition transition; not a precedent for narrative in protocol documents.** Through v2.4.x the per-step flag was `singular OR empty` (legacy). v2.5.0 introduced an exact analytic maneuver-family barrier (`02_control` §8) that by design saturates at the clip floor across the safe interior; its zero gradient raised the legacy flag on nearly every active step (raw infeasibility 0.87–0.91) even though those rows are automatically satisfied and the filter is inactive — a benign state, not a failure. The legacy flag therefore penalized exactness, while the genuinely pathological states — empty intersections, and singular-**and**-violated rows (the authority-loss failure of hazard-blind conditioning) — are exactly what the current definition counts. The current definition was introduced as the `cps_v2` field during the v2.5.0 Stage-B-2 closeout and adopted here. **Application is prospective**: ledger rows written before the adoption were scored with the legacy flag and are annotated as such when cited; verdict-grade comparisons across the boundary use re-scored comparators or dual reporting (`legacy cps | cps_v2`) until the standing comparators are re-scored, after which the v2 flag is the single canonical definition and the `_v2` suffix is retired.
- **`saturation_rate`** — defined in `02_control` §4.1: mean over active filtered steps of the per-step saturation flag (any $u^{\text{safe}}$ component within $10^{-3}$ of its bound), then mean over episodes. A recorded diagnostic only — **not** a term in `cps`.
- Outcomes are mutually exclusive via the `01_env` §1.6 priority (collision → goal → OOB → stuck → timeout). All five outcome fractions sum to 1; all five are reported alongside `cps`.

The harness reads outcomes exactly from `01_env` predicates. No path may re-derive them.

`cps` is the **selection metric** for `best.pt` during training (`03_train` §9) and the **headline metric** in every report. Component fractions and their bootstrap CIs are always reported alongside it.

### 1.1 Stuck-displacement diagnostic (for threshold tuning)

To separate "genuinely stuck" from "moved barely above the threshold", each episode also reports `min_window_displacement` — the smallest window-displacement encountered (`01_env` §1.6). Aggregated per eval, the **stuck-displacement histogram** buckets episodes whose `min_window_displacement` $\le 0.30$ m into six bins of width $0.05$ m:

- bin 0: $[0.00, 0.05)$ m — severely stuck
- bin 1: $[0.05, 0.10)$ m — stuck (would trigger the predicate)
- bin 2: $[0.10, 0.15)$ m — borderline
- bin 3: $[0.15, 0.20)$ m
- bin 4: $[0.20, 0.25)$ m
- bin 5: $[0.25, 0.30)$ m

Episodes with $\min_t \mathrm{disp}_t > 0.30$ m are not bucketed (free-moving). Per-bin counts and fractions are recorded on every eval. This is a pure diagnostic — it is not part of `cps` and does not affect outcome resolution.

Reference on a boundary case: an episode that comes to rest just outside the goal-reach tolerance is resolved as `stuck` even though it is near-success (a "parked" trajectory rather than an obstacle-blocked one). A v2.4.0 study observed a policy-collapse mode dominated by this near-goal parking. When stuck is elevated, reading the stuck-displacement histogram together with the terminal goal-distance separates genuine obstacle-blocked stalls from near-goal parking; the reach tolerance vs parking interaction is noted here so it is considered when interpreting stuck rather than left implicit.

---

## 2. Two evaluation modes

The harness exposes exactly two modes. There is no "quick" mode; if cheaper evaluation is needed, lower `--n` on a one-off run.

**Eval rollout termination (both modes).** Per `01_env` §1.6, the eval rollout terminates at the first step that fires collision, goal, or out-of-bounds, and naturally at `eval.max_steps` for timeout. **Stuck does NOT terminate the eval rollout**: when the stuck predicate fires the episode outcome is locked to `stuck`, but the rollout continues to `max_steps` so the trajectory plot shows whether the system recovers or remains oscillating. This mirrors deployment, where physical outcomes terminate the controller but oscillation does not auto-shut-down.

### 2.1 In-loop evaluation (during training)

Triggered by the trainer every `exp_config.eval.cadence` macro steps. Runs against the **in-loop pool** (`base_config.eval.in_loop`, $N = 500$). One row appended to `data/<run_id>/eval_metrics.csv`; per-episode rows to `data/<run_id>/eval_episodes.csv`; scalars also written to TensorBoard. The trainer updates `data/<run_id>/checkpoints/best.pt` by `cps`. In-loop **also renders the trajectory+control figures** (§3) and the **CBF contour figure** (§3b) to `data/<run_id>/figures/inloop/step_<NNNNNN>_grid_{A,B}.png` and `step_<NNNNNN>_cbf_contour.png` (batched framework-native filter evaluation — CBF-QP for OC-PNCBF, HardNet for JT — makes this cheap). No online insertion in-loop (that is final-only, §4).

### 2.2 Final evaluation (after training)

Triggered automatically when training terminates (completion or halt), and runnable on demand against any saved checkpoint. Runs against the **full pool** (`base_config.eval.full`, $N = 2000$, disjoint from the in-loop pool). Produces:

- One row appended to `data/<run_id>/eval_metrics.csv` (with `mode = "final"`) and per-episode rows to `eval_episodes.csv`.
- The framework's trajectory variants per episode (OC-PNCBF: LQR-only + filtered; Joint Training: LQR-only + learned no-CBF + filtered).
- Two trajectory+control figures (§3) — 16 intervention episodes total.
- One CBF contour figure (§3b).
- Three online-insertion sub-runs (§4): LQR-baseline, frozen-obstacle, live-obstacle. Each produces additional rows in `eval_metrics.csv` with distinct `mode` values.
- Bootstrap statistics over episodes (§5).
- `data/<run_id>/report.md` (auto-generated summary, §7.2).

A separate evaluation against a non-default checkpoint (e.g. `final.pt` instead of `best.pt`) is invoked by `python -m src.eval.run_full --ckpt PATH`. It writes to a new run directory.

**Canonical evaluation path.** Headline metrics are defined by the `run_full` `evaluate()`
batch-2000 path. Alternate rollout implementations (manual loops, different batch sizes) may
flip individual episode outcomes at the few-per-2000 level; their counts are never headline
numbers.

**Probe protocol (re-rolled analyses).** Mechanism probes that re-roll trajectories operate
on the AGREED set — episodes whose outcome matches between the probe roll and the canonical
eval; boundary episodes are excluded from the analysis and reported with their IDs. Every
reported share cites its denominator explicitly (canonical count, probe-path count, or
agreed count).

---

## 3. Trajectory + control plots — fixed format

A single fixed format eliminates per-report customization. All plots in the project route through the one plotting module `src/eval/plotting.py`, the sole definer of colors, markers, fonts, DPI, and layout. The format is identical for in-loop and final eval; only the output path differs (§7.1).

**Purpose.** Each figure shows, for selected episodes, both the spatial trajectory and the control-input time series, so that filter behavior — how often the action saturates the control bounds and how bang-bang it is — is directly readable alongside the path.

**Episode = a panel pair.** Each episode occupies two adjacent panels: a **left trajectory panel** and a **right control-vs-time panel**. The 4×4 grid therefore holds **8 episodes per figure** (8 left + 8 right panels). **Two figures** (`A`, `B`) are produced, for **16 episodes total**.

**Selection — intervention episodes only, deterministic, filled to 16.** Only episodes in which the CBF intervened on at least one step (some step with $\|u^{\text{safe}}_t - u^{\text{nom}}_t\| > 10^{-3}$) are plotted; episodes the filter never touched are skipped (they carry no filter behavior to inspect). Selection is deterministic: scan the pool in ascending index order and take the first 16 intervention episodes (the first 8 to figure `A`, the next 8 to figure `B`). The pool's intervention episodes normally far exceed 16, so the 16 panel-pairs are filled. In the rare case the pool yields fewer than 16, plot all available and leave the trailing panel pairs empty; report the shortfall.

**Left panel — trajectory.** As before: arena boundary (thin black square at $[-\text{world\_lim}, \text{world\_lim}]^2$), obstacles as filled gray circles (true radii), start (green circle), goal (red star). The framework's nominal-only rollout is drawn dotted at reduced alpha (LQR-only for OC-PNCBF; for Joint Training the learned no-CBF rollout is the dotted baseline). The filtered trajectory is colored **per step by intervention**: black where the filter was inactive, gray where it intervened. A collision is a red `x` at the first-collision step. Panel title: pool index + resolved outcome word (`Reach` / `Collision` / `OOB` / `Stuck` / `Timeout`).

**Right panel — control vs time.** The executed control components are plotted against time step for the filtered run, overlaying pre- and post-projection actions so the projection's effect is visible:

- $u^{\text{safe}}$ (post-projection) components: one line per action component, in **distinct, well-separated colors** (e.g. component 1 vs component 2 clearly different hues).
- $u^{\text{nom}}$ (pre-projection: the framework's nominal — LQR for OC-PNCBF, the learned policy for JT) components: the **same hue as the corresponding $u^{\text{safe}}$ component but lighter / more transparent**, so each component's nominal-vs-safe pair reads as one color family.
- **Control bounds:** black dashed horizontal lines at each component's $\pm$bound (`system.u_bounds`; per-component, e.g. Unicycle $a \in [-2,2]$, $\omega \in [-3,3]$). Saturation is visible as $u^{\text{safe}}$ riding the dashed line; bang-bang as rapid swings between bounds.

The overlay of $u^{\text{nom}}$ is kept unless it makes a panel unreadable, in which case it may be dropped (the $u^{\text{safe}}$ lines and bounds are the essential content).

**Titles and legend.** The figure suptitle is version-stamped (begins with the project version, e.g. `vX.Y.Z · …`). The global legend sits at the figure bottom in **two rows**: row 1 describes the trajectory panel (`Start`, `Goal`, `Trajectory (CBF inactive)`, `Trajectory (CBF active)`, `Nominal-only (dotted)`, `Collision`); row 2 describes the control panel (`u₁ safe`, `u₂ safe`, `u₁ nominal`, `u₂ nominal`, `Control bound`). The full set is shown even when an entry is unused in a given figure. No per-panel legend.

**Files.** Final eval: `data/<run_id>/figures/trajectory_grid_A.png` and `trajectory_grid_B.png`. In-loop eval: `data/<run_id>/figures/inloop/step_<NNNNNN>_grid_{A,B}.png` (§7.1). PNG only.

**No PDF.** PNG is the universal format for both the MkDocs site and external use.

---

## 3b. CBF contour plots — fixed format

A second fixed figure visualizing the learned CBF landscape $h(x, \text{obs})$ directly in
position space. Because the deployment CBF is relative degree 1 (it depends on velocity,
`02_control` §5.1), its zero-levelset moves with velocity; this figure makes that
dependence visible and provides a sanity check (at zero velocity the $h = 0$ boundary
should sit near the obstacle perimeters). Routed through the same plotting module
(`src/eval/plotting.py`).

**Grid.** 2×3 panels. **Rows = two different scenes** (row 1 and row 2 are distinct
obstacle layouts, taken deterministically as the first two scenes of the pool). **Columns
= three velocity settings**, held constant across a panel while position is swept. Six
panels total = two scenes × three velocities.

**Velocity columns (per system).**
- Double Integrator $(vx, vy)$: left $(-1.5,\ 0.5)$, center $(0,\ 0)$, right $(1.5,\ -0.5)$.
- Unicycle $(v,\ \theta)$: left $(1,\ 0)$, center $(0,\ 0)$, right $(1,\ \pi)$.

**Per panel.** Fix the panel's scene and velocity. Sweep position $(px, py)$ over a regular
grid covering the arena $[-\text{world\_lim}, \text{world\_lim}]^2$ (a resolution that
renders cleanly, e.g. 100–200 per axis). At each grid point build the full state (swept
position + the panel's fixed velocity), construct the observation, and evaluate the
**deployed** $h$ (`make_h_fn(value_net, system)` deployed mean-ensemble, `02_control`
§3.4/§3.5). Render $h$ as a filled color map over the position grid:

- **Colormap:** diverging, centered at $0$, spanning the clamped range $h \in [-1, 1]$
  (e.g. blue negative/safe → white zero → red positive/unsafe). A shared colorbar.
- **Zero-levelset:** the $h = 0$ contour drawn as a solid **black** line, emphasized.
- **Overlays:** obstacles as outlined circles (true radii) and the goal marker, so the
  $h = 0$ boundary can be compared against the true obstacle perimeters. At the zero-velocity
  center column the black contour should track the obstacle perimeters closely; the left
  and right columns show how velocity inflates/shifts the boundary.
- Panel title: scene index + the velocity setting.

**Titles and files.** Version-stamped suptitle. Final eval:
`data/<run_id>/figures/cbf_contour.png`. In-loop eval:
`data/<run_id>/figures/inloop/step_<NNNNNN>_cbf_contour.png`. Produced in BOTH in-loop and
final eval, alongside the trajectory+control grids (§3). PNG only.

---

## 4. Online insertion

A stress test for the filter's reactivity to a scene change mid-episode. An obstacle is inserted on the filtered run only.

- **`t_insert`** — step at which the new obstacle becomes active. Default 50, in `base_config.eval.insertion.t_insert`.
- **Position** — midpoint of the trajectory segment between $t = 0$ and $t = t_{\text{insert}}$ on the filtered run (so it is plausibly in the path).
- **Radius** — $r = 0.45$ (`base_config.eval.insertion.radius`).
- **Three variants:**
  1. `lqr_baseline` — the same insertion applied to the LQR-only path (no filter); records how a non-CBF baseline reacts.
  2. `frozen_obstacle` — the inserted obstacle's geometric properties are known to the filter (it sees the new $h$), but the observation passed to the value network still reflects the original obstacle set.
  3. `live_obstacle` — the observation is also updated so the value network sees the new obstacle.
- **Same pool.** All three variants run on the full pool (§6). The seed used to pick the insertion location is `eval.full.seed + insertion.seed_offset` (default offset 77), held constant.

Each variant appends one row to `eval_metrics.csv` and per-episode rows to `eval_episodes.csv`, with `mode` $\in$ {`final_insertion_lqr`, `final_insertion_frozen`, `final_insertion_live`}.

---

## 5. Multi-seed reporting and bootstrap

Multi-seed aggregates with bootstrap CIs are the strongly recommended form for any reported result. A single-seed run may be reported as the headline of a version under the following conditions; once additional seeds are run, the multi-seed form takes over and becomes the requirement at version close.

**Single-seed headline (permitted when marked).** A version may report a single-seed result as its headline iff:
- the `seeds` column in `docs/ledger.md` lists the one seed (the row is explicitly "single-seed");
- the `docs/versions/vX.Y.Z/results.md` states "single-seed" prominently in the headline;
- a **scene-bootstrap 95% CI** on the single seed's episodes (resampling episodes within the seed) is reported alongside the point estimate.

This option exists so that a clearly-marked single-seed result is not categorically barred; it is not a substitute for multi-seed validation and must not be referred to as the version's reproducibility evidence.

**Multi-seed (strongly recommended).** Minimum 3 seeds; recommend 5+. Three seeds is the threshold required for any version to claim improvement over the previous version under the comparison rule below.

**Canonical seed set (mandated).** Multi-seed runs use the training seeds $\{42, 99, 12345\}$, in that order: seed 42 is also the single-seed and canonical-chain seed, then 99, then 12345. Additional seeds extend this set; they never replace it. Any departure requires the Researcher's explicit approval and must be stated in the `seeds` column of `docs/ledger.md` and in the version's `results.md`. Training seeds are independent of the fixed pool seeds of §6 (in-loop 12345, full 23456) and of `eval.bootstrap.seed`; the shared value 12345 carries no coupling between a training run and the in-loop pool.

**Within-seed bootstrap.** For each seed, sample episodes with replacement `eval.bootstrap.n_resample` times (default 1000) using `eval.bootstrap.seed` (default 20260508). For each resample, compute `cps` and components. The percentile CI is the 2.5/97.5 quantiles of the resample distribution, using NumPy's default linear interpolation method (`numpy.percentile(..., method="linear")`); the method is pinned so CIs are reproducible. The canonical input is the per-episode records of `eval_episodes.csv` (§7.3); the bootstrap consumes already-resolved per-episode outcomes and does not re-derive them.

**Across-seed aggregation.** Report:

- Mean across seeds.
- Standard deviation across seeds (sample SD, not SE).
- The pooled within-seed 95% CI (concatenated resamples across seeds, then 2.5/97.5 quantiles).

**Comparison rule.** Version B improves over Version A iff B's pooled mean `cps` is above A's pooled mean `cps` **and** the 95% CIs do not overlap. Overlapping CIs are reported as "no improvement detected"; the absence of a positive result is itself a result. The comparison rule applies once both versions have multi-seed aggregates available; a single-seed-vs-multi-seed comparison is not run under this rule.

The aggregated multi-seed table lives in `data/secured_data/<version>/aggregate/multi_seed_metrics.json` and is summarized in `multi_seed_report.md`.

**Seed economy.** New configurations screen on a single seed (42). Escalation to the
canonical multi-seed set `{42, 99, 12345}` occurs only for verdict-grade results — a
CI-separated improvement claim, or a registered multi-seed commitment — and the seed count
is decided at registration time, never after seeing the data. Eval-only measurements
costing minutes run all three seeds. Ad-hoc seed sets are prohibited.

---

## 6. Evaluation pool — generation and storage

### 6.1 Sampler parameters (canonical)

The eval sampler reuses the training mechanism (`03_train` §1) with these overrides:

- $d_{\min}^{\text{eval}} = \texttt{eval.scene.min\_start\_goal\_dist}$ — strictly larger than the train value (looser).
- $\delta^{\text{eval}} = \texttt{eval.scene.start\_goal\_clearance}$ — strictly larger than the train value (looser).
- Obstacle-field parameters identical to training (`base_config.obstacle.*`).
- Initial velocity: same uniform $[-v_{\text{init,max}}, v_{\text{init,max}}]$ as training (per `03_train` §1.2). The unavoidable-collision rejection (`03_train` §1.3) is applied identically.

Eval scenes are **looser** than training (larger clearance, larger min start-goal distance) so we never evaluate on scenes that training would have rejected.

### 6.2 Pools (committed)

Two pools are pre-generated, serialized, and committed to git under `data/secured_data/pools/`, both built by the same `build_pools.py` machinery. Every run reads them; nothing is generated in memory at evaluation time.

| pool | $N$ | seed | role | path |
|---|---|---|---|---|
| in-loop | 500 | 12345 | selection (best.pt by `cps`) | `data/secured_data/pools/eval_inloop_<system>_n500_seed12345.{pkl,manifest.json}` |
| full | 2000 | 23456 | final eval / headline reporting | `data/secured_data/pools/eval_full_<system>_n2000_seed23456.{pkl,manifest.json}` |

`<system>` is a short tag (`di` for Double Integrator, `uni` for Unicycle). The tag is part of the filename because the project has multiple systems and pools are not interchangeable across them; e.g. `eval_inloop_di_n500_seed12345.pkl`. The in-loop and full pools are **disjoint** (different seeds, different sizes), separating selection (in-loop) from reporting (full), analogous to validation vs test in standard ML.

The in-loop pool selects `best.pt` during training; the full pool is what the trainer's automated final eval runs and is the headline-reporting standard. Because pool scenes are drawn by a single seeded RNG in a fixed order (§6.3, `03_train` §1.1), any smaller-$N$ pool at a given seed is a byte-identical prefix of a larger one at the same seed — so checkpoint **re-selection** (re-evaluating a run's saved checkpoints on the full $N=2000$ pool to pick the true best, since the in-loop pool is smaller and noisier) is well-defined and reproducible.

### 6.3 Pool manifest

Each pool ships with a manifest JSON containing:

- `n_scenes`, `system`, `seed`, sampler parameter snapshot (`obstacle.*`, `eval.scene.*`, `v_init_max`), and a `pool_format_version` integer.
- SHA-256 of the pool `.pkl` file. The manifest is the only place SHA-256 appears; the headline reports (`report.md`, `eval_metrics.csv`) do not display it. The eval harness verifies the SHA on load and emits a warning **only on mismatch**, otherwise silent.
- ISO timestamp of generation, git commit at generation (or `"unknown"` with a warning if git is unavailable — this does not fail generation).

**Pool payload (pinned for SHA reproducibility).** The `.pkl` is a single dict with a fixed key order — `pool_format_version`, `system`, `seed`, `n_scenes`, then the stacked scene tensors (`obstacle_centers`, `obstacle_radii`, `obstacle_active`, `start`, `goal`, `init_velocity`) — serialized with `pickle.HIGHEST_PROTOCOL`. The byte-level reproducibility of the file (hence the SHA) depends on this fixed key order and on the deterministic RNG draw order of `03_train` §1.1. The manifest's timestamp and git fields are excluded from the pool file, so manifests are not byte-identical across regenerations even though the pool `.pkl` files are.

The pool generation script is `src/eval/build_pools.py`, which accepts `system` so per-system pools are added without code change. Re-running it must produce a byte-identical pool `.pkl` given the same seed and sampler params (verified by SHA).

---

## 7. Output schema

This is the **only** evaluation output specification. Every framework produces the same files at the same paths within `data/<run_id>/`.

### 7.1 File layout (consolidated)

The run directory is flat. There is no `eval/in_loop/`, `eval/full_<timestamp>/`, or per-eval sub-tree. All evaluation runs append to the same files.

```text
data/<run_id>/
├── config.yaml                # effective config (base + exp + CLI), saved once at start
├── git_commit.txt             # full hash + dirty flag
├── tensorboard/               # event files for live monitoring
├── metrics.csv                # per-macro-step training metrics (one row per logged step)
├── eval_metrics.csv           # per-eval rows: in-loop, final, final_insertion_*
├── eval_episodes.csv          # per-(eval, episode) rows
├── status.json                # current state + halt reason if any
├── pool_manifest.json         # copies of the in-loop and full pool manifests used
├── checkpoints/
│   ├── step_NNNNNN.pt         # optional cadence checkpoints
│   ├── best.pt                # best-by-cps in-loop checkpoint
│   └── final.pt               # last checkpoint at training end
├── figures/
│   ├── trajectory_grid_A.png        # final-eval, 8 intervention episodes
│   ├── trajectory_grid_B.png        # final-eval, next 8 intervention episodes
│   ├── cbf_contour.png              # final-eval, 2 scenes x 3 velocities
│   └── inloop/
│       ├── step_NNNNNN_grid_{A,B}.png    # one pair per in-loop eval
│       └── step_NNNNNN_cbf_contour.png   # one per in-loop eval
└── report.md                  # auto-generated summary at training end
```

Checkpoint contents follow `03_train` §5 (step, pi_state, v_s_state, v_s_target_state, args).

**Run-id formats.**

- **Training runs:** `vX.Y.Z__YYYYMMDD-HHMMSS__seed<N>`. This is the canonical form.
- **Eval-only diagnostic runs** (a run that does no training, only re-evaluates an existing checkpoint under an ablation): may use a descriptive suffix
  `vX.Y.Z__YYYYMMDD-HHMMSS__<descriptive>_seed<N>`, where `<descriptive>` is a short snake_case tag (e.g. `hardnet_oc`, `cbfqp_oc`, `slowmpc`) that identifies the ablation. The descriptive form must never collide with a training run id; a training run is always identified by the canonical form.

### 7.2 CSV schemas

**`metrics.csv`** — one row per logged macro step. The canonical column **set** is listed
below; the listing is the required-membership specification, not an ordering
specification. The **trainer's actual CSV header is the authoritative order**: any new
run's header is the ground truth for column order, and downstream tooling reads by
column name, not by position. Adding a column is allowed; removing a column listed
below requires a protocol patch.

```text
step, wallclock_s, schedule_step,
lambda_disc_active, gamma_disc_active, target_rhs_active, sigma, sigma_pi,
L_R, L_V_total,
L_in_task, L_anorm, L_smooth, L_satex, L_pretanh, L_out, L_pi_total,
grad_norm_VS, grad_norm_pi, grad_leak_VS_from_Lpi,
rho_unsafe_v, rho_unsafe_pi, collect_proj_mag, collect_infeasible,
probe_h_min, probe_h_max, probe_h_mean,
abs_action_mean, abs_action_max, satfrac_a_phi
```

The anchor columns `L_A`, `L_C` are written only when the optional anchor mechanism
(`03_train` §4.3) is active (nonzero $\lambda_A$ or $\lambda_C$); otherwise they are
omitted. Columns that earlier drafts named differently — `n_sched`, `gamma_disc`,
`target_rhs`, `proj_mag_ema`, `sigma_pin_counter` — have been replaced by the columns
above to match the active trainer; their semantics are no longer part of the schema.
Run artifacts written before the v2.1.0 column rename retain the old names (e.g. the
v2.0.1 SOTA artifact still carries `target_rhs`); downstream analysis tooling should
accept both as an alias of `target_rhs_active` when reading historical runs.

**`eval_metrics.csv`** — one row per evaluation. Columns:

```text
mode, step, ckpt_name, pool_name, pool_seed, n_scenes,
cps, reach, collision, oob, stuck, timeout, infeasibility, saturation_rate,
stuck_bin_00_05, stuck_bin_05_10, stuck_bin_10_15,
stuck_bin_15_20, stuck_bin_20_25, stuck_bin_25_30,
cps_ci_lo, cps_ci_hi, reach_ci_lo, reach_ci_hi,
collision_ci_lo, collision_ci_hi, stuck_ci_lo, stuck_ci_hi,
infeasibility_ci_lo, infeasibility_ci_hi
```

`mode` $\in$ {`in_loop`, `final`, `final_insertion_lqr`, `final_insertion_frozen`, `final_insertion_live`, `final_alpha_sweep`}. `saturation_rate` is the episode-mean of the per-step saturation flag (`02_control` §4.1), a diagnostic outside `cps`. The six `stuck_bin_<lo>_<hi>` columns are fractions of episodes whose `min_window_displacement` falls in each $0.05$ m bin from $[0.00, 0.05)$ to $[0.25, 0.30)$ — diagnostic per §1.1, not part of `cps`. Episodes with `min_window_displacement > 0.30` m are not counted in any bin.

During the infeasibility-definition transition (§1 History note), evaluation outputs additionally carry `cps_v2` / `infeasibility_v2` fields alongside the legacy columns; when the transition closes, the v2 semantics are carried by the canonical `cps` / `infeasibility` columns themselves and the `_v2` fields are retired.

**`eval_episodes.csv`** — one row per (eval, episode). Columns:

```text
mode, step, ckpt_name, episode_idx,
outcome, n_steps, cps_episode, reach, collision, oob, stuck, timeout,
infeasible_step_frac, saturation_step_frac, min_window_displacement,
mean_proj_mag, max_h, traj_path_len
```

Outcomes use the `01_env` §1.6 string set: `goal`, `collision`, `oob`, `stuck`, `timeout`. `min_window_displacement` is defined in `01_env` §1.6 as the smallest window-displacement over the episode (or recorded as `NaN` if the episode is shorter than $W_{\text{stuck}}$ steps).

### 7.3 Status and report

**`status.json`** — minimal current-state record. Updated at each eval and at halt. Schema:

```json
{
  "stage": "full",
  "phase": "training" | "eval" | "done" | "halted",
  "current_step": 5500,
  "best_step": 4500,
  "best_cps": 0.84,
  "halt_reason": null,
  "updated_at": "2026-05-27T18:20:00"
}
```

**`report.md`** — generated by the trainer at the end of training (and re-generated on a stand-alone final eval). Sections:

1. Run identity (run_id, version, git commit, config hash, wall time).
2. Final eval results: `cps`, components, with 95% CIs.
3. In-loop training-curve summary: peak `cps`, step of peak, area-under-curve.
4. Halt status and reason if applicable.
5. File index (paths to CSVs, figures, checkpoints).
6. Pool identities (names, seeds, sizes — but no SHA in the body).

No prose conclusions or qualitative assessments are inserted by the trainer. Interpretation belongs to `docs/versions/vX.Y.Z/results.md`.

### 7.4 TensorBoard layout

Live monitoring runs against `data/<run_id>/tensorboard/`. The trainer writes:

- **Scalars (training, per macro step):** every column in `metrics.csv` is also a TB scalar under `train/<col>`.
- **Scalars (eval, per eval step):** every numeric column in `eval_metrics.csv` is a TB scalar under `eval/<mode>/<col>`.
- **Histograms (optional, every `optim.tb_histogram_every` steps, default 200):** `V_S` output distribution; `h` distribution per batch; per-component pre-projection action distribution.
- **Figures:** the trajectory+control grids and the CBF contour are written as TB images — final under `eval/final/trajectory_grid_{A,B}` and `eval/final/cbf_contour`, in-loop under `eval/in_loop/step_<NNNNNN>_grid_{A,B}` and `eval/in_loop/step_<NNNNNN>_cbf_contour`.

The user runs `tensorboard --logdir data/` (which surfaces every active `<run_id>` automatically) to monitor live.

### 7.5 Secured snapshot

When a version is closed (`06_workflow` §2.5), the chosen runs are copied into `data/secured_data/<version>/seed<N>/`. The secured snapshot includes (no TensorBoard event files, which are large and redundant with CSVs):

```text
data/secured_data/<version>/seed<N>/
├── config.yaml
├── git_commit.txt
├── metrics.csv
├── eval_metrics.csv
├── eval_episodes.csv
├── pool_manifest.json
├── checkpoints/{best.pt, final.pt}
├── figures/{trajectory_grid_A.png, trajectory_grid_B.png, cbf_contour.png}
└── report.md
```

Per-version aggregate lives at `data/secured_data/<version>/aggregate/` (only when multi-seed aggregation is performed):

```text
data/secured_data/<version>/aggregate/
├── multi_seed_metrics.json    # cross-seed bootstrap, mean, sd, CIs
└── multi_seed_report.md       # one-page summary with per-seed table
```

This directory **is committed to git** so cloning the repository gives anyone the exact numbers that anchor each version.

A version may additionally carry an optional `data/secured_data/<version>/experiments/<name>/` sub-tree — a secured diagnostic kept for the record that is not the version's SOTA snapshot. Its layout, required `README.md`, and Researcher-gated promotion are defined in `06_workflow` §6.3; it is never SOTA-bolded in the ledger (§2.4).

---

## 8. Comparisons across frameworks

Two frameworks (OC-PNCBF and JT) are compared on **identical pools** (§6.2). The comparison rule is identical to the version-comparison rule (§5): non-overlapping 95% CIs on pooled `cps` is the criterion. Component breakdowns (reach, collision, infeasibility) are reported alongside `cps` so that an improvement driven by one component at the cost of another is visible.

A framework comparison run produces `data/secured_data/comparisons/<name>/multi_seed_report.md` and reuses the same CSV schemas.