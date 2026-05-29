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
│   └── versions/{v1.md, vX.Y.Z/{changes.md, <task>.md, results.md}, ...}
├── scripts/                                # entry-point scripts (CLI launchers, pool builders)
├── src/
│   ├── common/                             # frozen utilities: rk4, signed_h, top_k_obs, lqr,
│   │                                       # buffers, polyak, logging, bootstrap
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
        └── aggregate/
            ├── multi_seed_metrics.json
            └── multi_seed_report.md
```

**`<run_id>` format:** `vX.Y.Z__YYYYMMDD-HHMMSS__seed<N>` (e.g., `v2.0.0__20260527-182000__seed42`). The Researcher / Strategist sees this format consistently across `metrics.csv`, `ledger.md`, and TensorBoard.

**`data/<run_id>/config.yaml`** — the **effective** configuration: deep-merge of `base_config.yaml`, `exp_config.yaml`, and any CLI overrides. Written **once** at run start; never re-read at runtime (the trainer reads the merged in-memory config). This file is the audit record.

**`data/<run_id>/git_commit.txt`** — full hash and a dirty flag (e.g. `5f3a9e... DIRTY` if the working tree is unclean at run start). A dirty start emits a warning to stdout but does not block the run.

**File schemas** — defined in `04_eval` §7.2 / §7.3. `05_code` does not duplicate.

---

## 4. `.gitignore` policy

`data/` is by default git-ignored, with `secured_data/` carved out. Exact rules:

```text
# .gitignore (relevant portion)
data/*
!data/secured_data/
!data/secured_data/**
```

This pattern:

- ignores every in-flight `data/<run_id>/`;
- tracks **everything** inside `data/secured_data/` (pools and version snapshots);
- requires no manual exception per file.

Large secured artifacts (checkpoint `.pt` files) are committed plain (no LFS in v2.0.0). If the secured tree later grows beyond a few hundred megabytes, switching to git-LFS is a one-line `.gitattributes` change and is reserved as a one-axis future task.

---

## 5. Verification harness (required before any framework code is allowed to run)

The harness lives in `tests/` and must pass before any training. The Executor refuses to start training if any test in §5.1–§5.6 fails. This is the v2 equivalent of a pre-commit hook, implemented as a pytest gate in `scripts/verify.sh`.

### 5.1 QP correctness (CBF-QP)

For a battery of $(h, L_f h, L_g h, u^{\text{nom}}, \text{bounds})$ cases:

- Solver-reported primal and dual residuals below tolerance ($10^{-6}$). This is the accepted operational definition of "KKT residual" for v2.0.0 (per `02_control` §5.2); the test is not required to independently assemble stationarity / complementarity / active-set residuals.
- Returned $u$ satisfies CBF row and box constraints within tolerance (slack accounted for).
- Solution matches an independent reference (OSQP or cvxpy) within tolerance.

### 5.2 HardNet ≡ L2-QP agreement

For $\varepsilon = 0$, single half-space, box-inactive, feasible cases, the HardNet projection equals the L2-QP solution within $10^{-6}$.

### 5.3 RK4 step

The RK4 step matches an analytic solution for: linear dynamics (Double Integrator under zero input) and pendulum-style small-angle linearization (sanity check on the unicycle's nonlinearity). Tolerance $10^{-5}$ over 100 steps.

### 5.4 Observation invariance

For both systems: translating start, goal, and obstacles by a random vector leaves the observation unchanged (Double Integrator: translation-invariant; Unicycle: translation- and rotation-invariant under matching rotation of $\theta$).

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
- **No `.item()` inside the hot loop.** Scalars for logging are deferred to a single tensor-to-host transfer at the end of each macro step.
- **One device transfer per macro step.** Collection and the V minibatch live on the GPU; the only host transfer per macro step is the logging payload.
- **No `torch.compile` in v2.0.0.** It is an optional one-axis addition once the baseline is stable.
- **GPU utilization target.** Sustained > 80% on the training device for the V/policy update windows (verified by a one-time `nvidia-smi`-based sampling during the first 100 macro steps). Below this number indicates a Python-loop regression and triggers an investigation, not a halt.

The smoke stage (`03_train` §6) implicitly exercises the batched code paths at small batch size; it does not measure utilization.

---

## 8. Reproducibility

Every run records:

- `git_commit.txt` (commit + dirty flag at start);
- `config.yaml` (the effective merged config, see §3);
- the pool manifest with seeds and sampler params (`04_eval` §6.3).

In v2.0.0 the checkpoint contains `step`, `pi_state`, `v_s_state`, `v_s_target_state`, `args` (`03_train` §5). Optimizer state, RNG state, and schedule state are **not** stored: exact resume-from-step is therefore not supported in v2.0.0. Model warm-starts from a checkpoint are supported (load `pi_state` and `v_s_state`). Exact step-level reproducibility (including optimizer state, RNG state, dataloader state) is reserved as a one-axis addition once the baseline is in place.

---

## 9. Coding conventions

- **Comments in English; concise; no decorative dividers** (`#####`, `# ---`). One blank line between logical blocks.
- **Imports** ordered: stdlib, third-party, project. One blank line between groups.
- **Type hints** on public functions. `typing.Protocol` for the framework interface (§2).
- **No global mutable state** — `src/configs/` is read by `src/common/config_io.py:load_configs()` exactly once per run; the result is passed in.
- **Determinism flag** `torch.use_deterministic_algorithms(True)` is **not** set by default (it disables some fast paths). Determinism is recovered via seed control; bit-exact determinism is reserved as a future need.
- **Naming.** `cps` everywhere for the headline metric. `V_S` for the value network, `pi` for the policy network, `u_nom` for pre-projection action, `u_safe` for post-projection action.
- **Lint and format.** `ruff` for lint; `black` for format. Both are run by `scripts/format.sh`.

---

## 10. Dependencies

Pinned in `pyproject.toml`. Top-level v2.0.0 deps:

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
