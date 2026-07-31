# 05 — Code

This document defines the **code-side standards** that keep the codebase correct and maintainable: a single shared environment/eval core that cannot drift between frameworks, vectorized rollouts that use the GPU, a verification harness that makes QP/HardNet behavior checkable, and a consolidated evaluation-output layout that keeps results comparable.

It does **not** re-state task semantics, training algorithms, or evaluation metrics — those belong to `01_env`, `02_control`, `03_train`, `04_eval`. It defines structure (directories), boundaries (what touches what), performance requirements, logging conventions, and a mandatory verification harness.

## 0.1 Frozen-core principle

`src/common`, `src/envs`, `src/eval` are **frozen** across frameworks: every framework imports from them; no framework owns its own copy. Frameworks differ only inside `src/frameworks/<name>/`. A framework that needs an env or eval change first proposes the change to common (Researcher approval, `06_workflow`), so the change applies to every framework consistently.

There is **one** RK4 step, **one** signed-h calculator, **one** Top-K obstacle encoder, **one** LQR (parameterized per system), **one** pool builder, **one** trajectory plotter, **one** bootstrap routine. Any duplicate is removed.

---

## 1. Repository layout

```text
jt-pncbf/
├── README.md
├── LICENSE
├── pyproject.toml
├── mkdocs.yml                              # MkDocs root (the only mkdocs.yml)
├── data/                                   # see §3
├── docs/
│   ├── javascripts/mathjax.js
│   ├── index.md                            # state dashboard
│   ├── ledger.md                           # one row per run
│   ├── protocol/{00..06}.md
│   └── versions/vX.Y.Z/{changes.md, <task>.md, results.md}    # local-only
├── scripts/                                # entry-point scripts (CLI launchers, pool builders)
│   └── analysis/                           # sanctioned analysis-only drivers (deploy_rate_eval,
│                                           # rescore_cps_v2, promoted diagnostics; never imported by src/)
├── src/
│   ├── common/                             # frozen utilities: rk4, signed_h, top_k_obs, lqr,
│   │                                       # buffers, polyak, logging, bootstrap,
│   │                                       # maneuver_value (analytic V_M: reference + compiled fast path)
│   ├── configs/
│   │   ├── base_config.yaml                # task + structure (rarely changed)
│   │   └── exp_config.yaml                 # per-experiment hyperparameters
│   ├── envs/
│   │   ├── double_integrator.py
│   │   └── unicycle.py
│   ├── eval/
│   │   ├── build_pools.py                  # generates and secures eval pools
│   │   ├── rollout.py                      # framework-agnostic batched rollout
│   │   ├── evaluate.py                     # shared evaluate(): pool -> cps + per-episode rows
│   │   ├── bootstrap.py                    # within-seed and across-seed CIs
│   │   ├── plotting.py                     # the single trajectory+control plotter
│   │   ├── run_full.py                     # final-eval entry point
│   │   └── report.py                       # report.md auto-generator
│   └── frameworks/
│       ├── jt_pncbf/                       # Joint Training: policy + V_S co-training
│       │   ├── train.py
│       │   ├── losses.py
│       │   ├── policy.py
│       │   ├── value.py
│       │   ├── filter.py                   # HardNet wrapper
│       │   └── collection.py               # two-buffer collection
│       └── oc_pncbf/                       # OC-PNCBF: V-only with CBF-QP filter
│           ├── train.py
│           ├── value.py
│           ├── filter.py                   # CBF-QP wrapper (proxsuite)
│           └── collection.py
└── tests/                                  # see §5
    ├── test_qp_correctness.py
    ├── test_hardnet_l2qp_agreement.py
    ├── test_rk4.py
    ├── test_observation_invariance.py
    ├── test_bptt_detach.py
    └── test_pool_reproducibility.py
```

Key constraints:

- **No cross-framework imports.** `src/frameworks/jt_pncbf/` must not import from `src/frameworks/oc_pncbf/` and vice versa. Common needs live in `src/common`.
- **One mkdocs.yml.** It lives at the repo root and is the only one. (A `docs/mkdocs.yml` is forbidden; if found, delete.)
- **No scattered docs.** All durable documents live under `docs/`. Per-subproject `docs/` directories are not created.
- **Scripts are launchers.** Anything more than argument parsing and a single function call belongs in `src/`.

---

## 2. Module boundaries (what may import what)

```text
src/configs/   ← read by everything (via a single loader in src/common/config_io.py)
src/common/    ← may import from configs and stdlib/torch only
src/envs/      ← may import from common
src/eval/      ← may import from common, envs
src/frameworks/<name>/  ← may import from common, envs, eval — but not from each other
```

Every system implements a uniform **System protocol** so that the integrator, rollout,
outcome resolver, and every framework consume `System` and never a concrete class or a
`system.name` branch:

```python
class System(Protocol):
    state_dim: int
    action_dim: int
    obs_dim: int
    name: str
    u_bounds: Tensor                                  # [action_dim, 2]

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor: ...      # x_dot [B, state_dim]
    def observation(self, x: Tensor, scene: Scene) -> Tensor: ...# [B, obs_dim]
    def position(self, x: Tensor) -> Tensor: ...                 # [B, 2]
    def speed(self, x: Tensor) -> Tensor: ...                    # [B], for goal/outcome
    def lqr_action(self, x: Tensor, goal: Tensor) -> Tensor: ... # [B, action_dim]
    def wrap_state(self, x: Tensor) -> Tensor: ...               # angle wrap / normalize
```

Two requirements follow from this protocol and are enforced project-wide:

- **No system-identity branching in shared code.** `src/common/*` and `src/eval/*` must
  not contain `if system.name == ...`. Anything system-specific (e.g. how speed is read,
  how `[a, omega]` is mapped) lives behind a protocol method on the concrete system. The
  outcome resolver uses `system.speed` and `system.position`, never a per-system formula
  inline.
- **Control-affine requirement for filtered systems.** Any system used with the CBF-QP or
  HardNet filter must be control-affine in the action over the action bounds
  ($\dot x = f(x) + g(x)u$). The filters reconstruct $f$ and $g$ from `system.dynamics`
  (evaluating at $u = 0$ for $f$ and against action basis vectors for $g$); a system that
  is not affine must instead implement a common `affine_dynamics(x) -> (f, g)` method that
  the filters consume. Adding a new system means adding `src/envs/<system>.py` implementing
  this protocol — no change to `src/common/`, `src/eval/`, or the protocol itself.

Frameworks expose a uniform interface that `src/eval/rollout.py` can call:

```python
class Framework(Protocol):
    def policy(self) -> Callable[[Tensor], Tensor]: ...
    def filter(self) -> Callable[[Tensor, Tensor], Tensor]: ...  # (state, u_nom) -> u_safe
    def value(self) -> Callable[[Tensor], Tensor]: ...           # for h, infeasibility
```

`src/eval/rollout.py` does not know whether it is rolling out JT or OC-PNCBF; it consumes whichever object satisfies the protocol.

---

## 3. Output layout: data/

The `data/` tree has only **two** layers: in-flight `<run_id>/` directories at the top, and a `secured_data/` subtree that is committed to git.

```text
data/
├── <run_id>/                              # in-flight, git-ignored (see §4)
│   ├── config.yaml
│   ├── git_commit.txt
│   ├── tensorboard/
│   ├── metrics.csv
│   ├── eval_metrics.csv
│   ├── eval_episodes.csv
│   ├── status.json
│   ├── pool_manifest.json
│   ├── checkpoints/{step_NNNNNN.pt, best.pt, final.pt}
│   ├── figures/{trajectory_grid_A.png, trajectory_grid_B.png, cbf_contour.png, inloop/step_*_{grid_*,cbf_contour}.png}
│   └── report.md
└── secured_data/                           # committed to git
    ├── pools/
    │   ├── eval_inloop_di_n200_seed12345.pkl
    │   ├── eval_inloop_di_n200_seed12345.manifest.json
    │   ├── eval_full_di_n500_seed23456.pkl
    │   └── eval_full_di_n500_seed23456.manifest.json
    └── <version>/
        ├── seed<N>/                        # final snapshot per seed (see 04_eval §7.5)
        ├── experiments/<name>/             # optional secured diagnostic, Researcher-gated (06_workflow §6.3)
        └── aggregate/
            ├── multi_seed_metrics.json
            └── multi_seed_report.md
```

**`<run_id>` formats.** Every artifact directory under `data/` — without exception — carries the version and a timestamp:

- **Training runs:** `vX.Y.Z__YYYYMMDD-HHMMSS__seed<N>` (e.g., `v2.0.1__20260529-171057__seed42`). This is the canonical form.
- **Eval-only diagnostic runs** (a run that does no training, only re-evaluates an existing checkpoint under an ablation): `vX.Y.Z__YYYYMMDD-HHMMSS__<descriptive>_seed<N>`, where `<descriptive>` is a short snake_case tag (e.g. `hardnet_oc`, `cbfqp_oc`, `slowmpc`). The descriptive form must never collide with a training run id (see `04_eval` §7.1).
- **Non-training artifact runs** — dataset or label generation, supervised regression, demonstration collection, registries, and any other produced artifact set: `vX.Y.Z__YYYYMMDD-HHMMSS__<descriptive>[_seed<N>]`, under the same rules; `_seed<N>` is appended only when a seed identifies the artifact.

Two rules follow, and they admit no exception:

- **The timestamp is never omitted.** A directory named `vX.Y.Z__<descriptive>` is malformed regardless of what it holds.
- **No loose files at the top of `data/`.** Every produced file lives inside a `<run_id>/` directory; a registry or manifest that spans several runs gets its own `<run_id>/` directory.

The Researcher / Strategist sees the `<run_id>` format consistently across `metrics.csv`, `ledger.md`, and TensorBoard.

**`data/<run_id>/config.yaml`** — the **effective** configuration: deep-merge of `base_config.yaml`, `exp_config.yaml`, and any CLI overrides. Written **once** at run start; never re-read at runtime (the trainer reads the merged in-memory config). This file is the audit record.

**`data/<run_id>/git_commit.txt`** — full hash and a dirty flag (e.g. `5f3a9e... DIRTY` if the working tree is unclean at run start). A dirty start emits a warning to stdout but does not block the run.

**File schemas** — defined in `04_eval` §7.2 / §7.3. `05_code` does not duplicate.

---

## 4. `.gitignore` policy

`data/` is by default git-ignored, with `secured_data/` carved out. `docs/versions/` is git-ignored entirely (per-version `changes.md`, build-logs, and `results.md` are local working artifacts). Exact rules:

```text
# .gitignore (relevant portion)
data/*
!data/secured_data/
!data/secured_data/**
docs/versions/
```

This pattern:

- ignores every in-flight `data/<run_id>/`;
- tracks **everything** inside `data/secured_data/` (pools and version snapshots);
- keeps `docs/versions/` local-only;
- requires no manual exception per file.

Large secured artifacts (checkpoint `.pt` files) are committed plain (no LFS). If the secured tree later grows beyond a few hundred megabytes, switching to git-LFS is a one-line `.gitattributes` change and is reserved as a one-axis future task.

---

## 5. Verification harness (required before any framework code is allowed to run)

The harness lives in `tests/` and must pass before any training. The Executor refuses to start training if any test in §5.1–§5.6 fails. This is a lightweight pre-commit gate implemented as a pytest invocation in `scripts/verify.sh`, which runs `pytest -q -rs` so that the identity of any skipped test is captured and not only its count.

### 5.1 QP correctness (CBF-QP)

For a battery of $(h, L_f h, L_g h, u^{\text{nom}}, \text{bounds})$ cases:

- Solver-reported primal and dual residuals below tolerance ($10^{-6}$). This is the accepted operational definition of "KKT residual" used by this spec (per `02_control` §5.2); the test is not required to independently assemble stationarity / complementarity / active-set residuals.
- Returned $u$ satisfies CBF row and box constraints within tolerance (slack accounted for).
- Solution matches an independent reference (OSQP or cvxpy) within tolerance.

### 5.2 HardNet ≡ L2-QP agreement

For $\varepsilon = 0$, single half-space, box-inactive, feasible cases, the HardNet projection equals the L2-QP solution within $10^{-6}$.

### 5.3 RK4 step

The RK4 step matches an analytic solution for: linear dynamics (Double Integrator under zero input) and pendulum-style small-angle linearization (sanity check on the unicycle's nonlinearity). Tolerance $10^{-5}$ over 100 steps.

### 5.4 Observation invariance

Translating start, goal, and obstacles by a random vector **drawn from the system's retained symmetry group** leaves the observation unchanged (Double Integrator: translation-invariant; Unicycle: translation- and rotation-invariant under matching rotation of $\theta$; 3-D quadrotor: horizontal translation and yaw only, since the vertical coordinate is carried explicitly — `01_env` §3.4). The group the test quantifies over is read from the system spec rather than assumed: narrowing it when a hazard term breaks a symmetry is a correction, not a weakening, and a test that keeps quantifying over a lost symmetry passes only because the observation is still discarding something the hazard needs. A companion assertion pins the positive direction — two states differing only in a carried coordinate must produce different observations.

### 5.5 BPTT detach

A single fake training iteration that runs the policy update under JT must produce $\|\nabla_{V_S}\mathcal{L}_\pi\| < 10^{-12}$. Above the gradient-leak threshold halts the trainer at runtime; the test catches it before training starts.

### 5.6 Pool reproducibility

`src/eval/build_pools.py` run twice with the same seed must produce a byte-identical pool tensor (SHA-256 match). The harness regenerates each committed pool and checks against the manifest SHA.

A green `scripts/verify.sh` is a prerequisite for any `python -m src.frameworks.<name>.train ...` invocation. Smoke-stage gradient routing (`03_train` §6) is a separate runtime check.

---

## 6. Logging and live monitoring

### 6.1 TensorBoard is the live channel

The trainer writes scalars, histograms, and (at final eval) figures to `data/<run_id>/tensorboard/`. The Researcher runs:

```bash
tensorboard --logdir data/
```

and every active and past in-flight run appears automatically. Switching between runs and overlaying scalar curves is a TB-side concern; the trainer's only responsibility is to write event files at a sensible cadence.

The scalar and histogram inventory is in `04_eval` §7.4.

### 6.2 CSV is the durable channel

Every TensorBoard scalar is **also** written to one of `metrics.csv` / `eval_metrics.csv` / `eval_episodes.csv` (`04_eval` §7.2 / §7.3). CSV is the source of truth for any post-hoc analysis, paper figure, or cross-run comparison; TensorBoard is for live, in-flight inspection. The two are kept in sync at the same cadence.

### 6.3 What is **not** logged

Per-step trajectories, per-step actions, and per-step h-values for every episode are **not** logged (volume would dwarf the actual metrics). They are reconstructable from a checkpoint plus the pool, when needed.

---

## 7. Performance requirements (GPU / CPU utilization)

A neural CBF trainer that under-uses the GPU is shipping the wrong product. The hard requirement is **vectorized / batched rollouts; no Python-level loops over the batch dimension**.

Concrete requirements:

- **One-shot RK4 over the batch.** RK4 advances `[B, state_dim]` per step; never a Python loop over `B`.
- **Filter is batched.** HardNet projection operates on `[B, action_dim]`. CBF-QP is batched at the call site by stacking $(h, L_f h, L_g h, u_{\text{nom}})$ rows into one solver call where the solver supports it; where it does not, the per-row solve is parallelized over at most `base_config.filter.cbf_qp.max_workers` CPU threads (default 32) and the host transfer happens once per macro step, not per row.
- **Deploy-time paths run on the same device as training.** A checkpoint loaded for evaluation is moved to the run device before any rollout; loading with `map_location="cpu"` and never moving leaves every downstream path — the filter, its fallback, any lookahead — on the CPU with the accelerator idle, and the cost is then read as the algorithm's rather than the loading path's. Candidate families inside a filter fallback are one batched forward, never a Python loop over candidates: at deployment the scene batch is 1, so a loop that vanished behind batching during evaluation reappears at full length in the per-step latency that decides deployability.
- **No `.item()` inside the hot loop.** Scalars for logging are deferred to a single tensor-to-host transfer at the end of each macro step.
- **One device transfer per macro step.** Collection and the V minibatch live on the GPU; the only host transfer per macro step is the logging payload.
- **`torch.compile` — channel-scoped.** The analytic maneuver barrier ships a compiled fast path as the production default (`VM_FAST=1`), adopted under the `02_control` §8 equivalence standard (function parity $|V| \le 10^{-6}$ / $|\nabla V| \le 10^{-5}$ plus safety equivalence; trajectory bit-parity is unattainable for the branch-discrete projection and is not a gate). The uncompiled reference path is retained for tests/audit (`VM_FAST=0`), and the function-parity suite (`tests/test_vm_fastpath.py`) must be re-run and reported at every PyTorch/compiler version change (current pin: `torch 2.12.0.dev20260404+cu128`). Outside this channel `torch.compile` remains an opt-in, one-axis future addition.
- **Throughput is judged against the class baseline, not raw GPU %.** Kernel-launch-bound certificate rollouts leave the GPU partially idle by construction (compiled $V_{\mathcal M}$ runs ≈ 55% utilization at full speed), so the binding metric is steps/s against the class baseline: learned-filter JT ≈ 4.6 steps/s (v2.4.x class); maneuver-$V_{\mathcal M}$ JT reference ≈ 1.0 s/step; maneuver-$V_{\mathcal M}$ JT compiled ≈ 0.204 s/step (≈ 4.9×; one $V_{\mathcal M}$ update ≈ 5.1× a learned-filter update). A run regressing > 15% below its class baseline triggers the throughput-incident diagnosis (environment snapshot → code inspection → offline cost ratio), as exercised in v2.5.0 Stage B-2. The historical "> 80% sustained GPU" figure remains the aspiration for non-launch-bound workloads.
- **Verdict-grade evaluations pin the eval batch size.** The branch-discrete projection makes aggregate scores sensitive to floating-point reduction order; a batch-size change alone has flipped a re-score by 0.010 (v2.5.0 ARM-C gate, batch 200 vs 250). Re-scores and comparators state and reuse the eval batch size as part of the eval conditions.

The smoke stage (`03_train` §6) implicitly exercises the batched code paths at small batch size; it does not measure utilization.

---

## 8. Reproducibility

Every run records:

- `git_commit.txt` (commit + dirty flag at start);
- `config.yaml` (the effective merged config, see §3);
- the pool manifest with seeds and sampler params (`04_eval` §6.3).

The checkpoint contains `step`, `pi_state`, `v_s_state`, `v_s_target_state`, `args` (`03_train` §5). Optimizer state, RNG state, and schedule state are **not** stored: exact resume-from-step is therefore not supported. Model warm-starts from a checkpoint are supported (load `pi_state` and `v_s_state`). Exact step-level reproducibility (including optimizer state, RNG state, dataloader state) is reserved as a one-axis future addition.

---

## 9. Coding conventions

- **Comments in English; concise; no decorative dividers** (`#####`, `# ---`). One blank line between logical blocks.
- **Imports** ordered: stdlib, third-party, project. One blank line between groups.
- **Type hints** on public functions. `typing.Protocol` for the framework interface (§2).
- **No global mutable state** — `src/configs/` is read by `src/common/config_io.py:load_configs()` exactly once per run; the result is passed in.
- **Determinism flag** `torch.use_deterministic_algorithms(True)` is **not** set by default (it disables some fast paths). Determinism is recovered via seed control; bit-exact determinism is reserved as a future need.
- **Naming.** `cps` everywhere for the headline metric. `V_S` for the value network, `pi` for the policy network, `u_nom` for pre-projection action, `u_safe` for post-projection action.
- **Lint and format.** `ruff` for lint; `black` for format. Both are run by `scripts/format.sh`.
- **Plain text in reports.** No emoji or decorative glyphs in any report, table, log line, commit-adjacent file, or figure. Status is carried by words.
- **Registrations live in markdown.** Hypotheses, falsifiers, and adjudication records are written in the version's markdown documents, which are the record (`00_constitution` §6). A machine-readable copy may be derived from them for scripts to consume, kept with the run artifacts and never under `docs/`; two files that both claim to hold a registration will disagree eventually, and the markdown wins by construction.

---

## 10. Dependencies

Pinned in `pyproject.toml`. Top-level project deps:

```toml
[project]
requires-python = ">=3.11"
dependencies = [
  "torch>=2.12,<2.13",
  "numpy>=1.26",
  "pyyaml>=6.0",
  "proxsuite>=0.6",
  "osqp>=0.6",      # used only by tests/test_qp_correctness.py
  "cvxpy>=1.4",     # used only by tests/test_qp_correctness.py
  "matplotlib>=3.8",
  "pandas>=2.2",    # CSV reading for post-hoc analysis
  "tensorboard>=2.16",
  "pytest>=8.0",
  "ruff>=0.4",
  "black>=24.0",
]
```

The CUDA pin lives in environment management (conda env `pncbf` carries the PyTorch + CUDA combination); it is not a Python-package pin.
