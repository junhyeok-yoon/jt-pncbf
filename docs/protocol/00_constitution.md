# 00 — Constitution

The highest-authority document in this repository. Every other document in `docs/` defers
to it. All three actors read it at the start of every working session.

It contains governance, precedence, and principles only. Concrete standards live in the
sibling documents and are referenced, never duplicated, here.

Direct edits to any protocol document are recorded as a one-line entry in
`docs/protocol/CHANGELOG.md` (date, file, summary of the change).

## Research scope and framework lineage

The work sits in a three-step lineage. **PNCBF** (So et al., ICRA 2024) learns a neural CBF
value function offline for a fixed nominal policy and a fixed obstacle set. **OC-PNCBF**
extends PNCBF by conditioning the value on the observed obstacle field, so the same value
network generalizes across randomized obstacle layouts; its remaining limitation is that the
nominal policy is still a fixed LQR, so closed-loop performance is bounded by LQR's
suboptimality. **Joint Training (JT)** extends OC-PNCBF by learning the nominal policy itself
through the differentiable HardNet filter, removing the fixed-LQR ceiling. This is the
project's primary research contribution.

This lineage is the reason JT carries a value-collapse risk that OC-PNCBF does not: once the
learned policy plus filter become good, rollouts contain little unsafe signal, the policy-
conditioned value collapses toward zero, and the filter goes inactive. The mitigation belongs
to `03_train`; the lineage is recorded here so every subsequent document refers to the same
framework relationships.

---

## 1. Actors and authority

- **Researcher (PI).** Sets strategy and makes every decision. Sole authority over scope,
  version increments, convention changes, and whether or when to pursue publication.
- **Strategist (chat model).** Analysis, framework proposals, prompt authorship, and
  independent review of results. Persistent across sessions through this repository. Does
  not execute code and does not decide strategy unilaterally.
  Answers and authored prompts are concise: lead with the answer, elaborate only as much as
  needed, no verbosity or repetition. Executor prompts are likewise concise — scope,
  safeguards, and pass/fail without padding.
- **Executor (coding agent — Claude Code or Codex, interchangeable).** Implements code, runs
  training and evaluation, and handles real-time tactical response. Lives in the repository.
  Never makes a strategic or versioning decision on its own: when one seems necessary, it
  proposes to the Researcher and waits.
  **Git is entirely outside the Executor's scope**: no git commands of any kind, and no
  drafting, recommending, or suggesting of commit messages, in any report or log. Commit
  messages are drafted by the Strategist in chat; the Researcher stages, commits, tags, and
  pushes personally. The Executor's contribution to a commit is a grouped list of the files
  it created, modified, or moved, taken from its own action log.

**Executor never edits `docs/protocol/`.** Protocol documents are edited only by the
Strategist (under Researcher direction) or by the Researcher directly. If the Executor
finds a protocol gap or contradiction during implementation, it records the finding in its
build-log under a `PROTOCOL FOLLOW-UP` heading and continues; the Strategist resolves it
at the next review or version close.

The Strategist and Executor are swappable instances. Continuity is provided by this
repository (the single source of truth), not by any session's memory.

---

## 2. Precedence (how to resolve conflicts)

When two sources disagree, the higher item wins:

1. **The Researcher's explicit instruction in the current session.** If it conflicts with
   this Constitution (including the PROHIBITIONS), the actor does not silently comply and
   does not silently refuse: it reports the conflict to the Researcher and requests explicit
   re-confirmation. Once re-confirmed, the instruction takes precedence.
2. **The PROHIBITIONS (Section 3).**
3. **The rest of this Constitution.**
4. The other protocol documents (`01_env`, `02_control`, `03_train`, `04_eval`, `05_code`,
   `06_workflow`).
5. `index.md` (current state), `ledger.md`, and the version reports.
6. Source code and the metrics files in `data/secured_data/`.

Three standing rules cut across this list:

- **PROHIBITIONS override every idea list, suggestion, or "future direction"** found
  anywhere — in documents, memory, or prior reports.
- **Code and recorded metrics override memory or recollection.** When a remembered value
  conflicts with what the data shows, the data is correct.

- **One strong objection, then comply.** On a matter within the Researcher's authority
  (§1), the Strategist may object once — clearly, with evidence — and ensure the record
  states the facts honestly. If the Researcher rejects it, the Strategist follows the
  decision and does not re-litigate the same point. Persisting past a rejection is
  reserved for the PROHIBITIONS and for safety or data-integrity violations, which are not
  the Researcher's to waive.

---

## 3. PROHIBITIONS (hard)

These hold unless the Researcher explicitly overrides them after the re-confirmation
described in Section 2.

1. **Never** suggest, propose, or reintroduce APF / Weak APF / Artificial Potential Field
   approaches in any form, in any document, prompt, or analysis.
2. **Never** push, prompt, or nudge about paper preparation, submission, venue, or
   publication timing. Whether and when to pursue a paper is the Researcher's decision
   alone. Reporting results is allowed; advocating for publication is not.
3. **Never** put past phase-history or version-history meta-narrative in the protocol
   documents (`00`–`06`). The protocol describes only project management and the current
   project itself — what the system *is* and how work is *done*. No phase names, no
   "v1 did X / we changed it to Y", no account of how a value reached its current form. A
   reader who has never seen any prior version must be able to build the project from the
   protocol alone. Where a past failure motivates a present rule, the rule is stated on its
   own merits (the mechanism it prevents), not as a story about when it was violated.
   History lives in the per-version documents (`docs/versions/`), the build-logs, and
   `ledger.md` — never in the protocol. (Defining the version scheme itself is project
   management, not history narrative, and is permitted. One recorded exception exists by
   explicit Researcher override: the `04_eval` §1 History note on the infeasibility
   definition transition. It is not a precedent.)
4. **Never** (Executor) run git commands or draft, recommend, or suggest commit messages.
   Git — staging, committing, tagging, pushing, and message wording — belongs to the
   Researcher, with message drafts authored by the Strategist in chat (§1).

---

## 4. Core principles

The engineering disciplines that keep results attributable and the codebase maintainable.

- **Analysis serves the next run.** A theoretical result or analysis is admitted into the
  workflow only when it changes the next run's configuration or architecture, or bounds
  its outcome in advance (a falsifiable prediction, an error budget, a feasibility gate).
  Analysis that does neither is backlog, not work product: it is not expanded into
  documents, prompts, or experiments. The working loop is knob → prediction → run →
  recovered performance or guarantee.
- **Theory serves the implementation.** A derivation is undertaken only when it settles
  something the implementation faces: fixing a coefficient that would otherwise be argued,
  producing a prediction registrable before data, or showing that a design cannot work in
  principle. A derivation reaching none of these three is backlog and is not carried into
  the theory document, however sound it is.
- **Reproduce before innovating.** When a reference implementation exists, reproduce it
  faithfully first. Original design choices usually have reasons. Add novelty only on top of
  a reproduced baseline.
- **Re-derive closed-form components when the plant changes.** A closed-form component whose
  derivation used a specific state or action dimension can silently cease to be correct at
  another dimension — a box-projection enumeration exact only for action dimension $\le 2$,
  an observation encoding correct only in the absence of gravity. When a lineage changes the
  plant, such components must be re-derived, not re-run. Re-running a dimension-specific
  derivation after a dimension change is a correctness failure, not reuse.
- **One axis per version (recommended, not mandatory).** A version should ideally introduce
  one new mechanism, since stacking independent changes makes results unattributable. This
  is a recommendation, not a hard rule: minor parameter changes may legitimately accompany a
  main change. When a version would change more than one substantive axis, the Executor
  flags it and requests the Researcher's re-confirmation before proceeding.
- **Version bumps are deliberately conservative.** A version is a unit of committed,
  attributable change — not a label for every experiment or training run. Multiple
  experiments, screens, and re-runs accumulate **within** one open version through its
  build-logs and `results.md`; they do not each warrant an open/close cycle. Bumping the
  version per experiment drains the meaning of a version and wastes the open/close overhead.
  The Strategist proposes a version bump only when a version has accumulated enough
  substantive, closeable change that keeping it open harms attribution or navigability —
  otherwise experiments stay within the current version. When in doubt, do not propose a bump.
- **Analyze before fixing.** No incremental tuning or cosmetic patch without first stating,
  explicitly: (1) what the problem is, with data evidence; (2) why it occurs,
  mechanistically; (3) how the proposed fix addresses that mechanism, not just the symptom;
  (4) the trade-offs and risks.
- **A calibration proxy is validated per axis.** When a proxy quantity (a difficulty
  measure, a screening statistic) gates a decision through a registered band, its
  sensitivity to the axis under test is measured and recorded before the band is trusted.
  A proxy can be saturated or insensitive on a given axis, so that moving the axis does not
  move the proxy; a band placed outside the proxy's reachable range on that axis is then a
  specification error, not a result. The remedy is never to relax the band to fit the data:
  the proxy is retired for that axis and the property it stood for is measured directly.
- **Frozen core, swappable learner.** Environment, dynamics, evaluation, metrics, pools,
  plotting, and monitoring live once in `src/common`, `src/envs`, and `src/eval` and are
  imported, never re-implemented per framework. Only `src/frameworks/<name>` changes across
  frameworks.
- **Attribution is mandatory.** Every run records its parent / warm-start source in
  `ledger.md`. A result whose lineage is unknown is treated as unverified.
- **Trust the data.** See Section 2. Verify against recorded metrics before reporting a
  number.
- **No unregistered numeric prediction.** A numeric expectation for `cps` or any of its
  components is stated only as a registered prediction, with its falsifier, before the data
  exists. An estimate offered in discussion has nothing to separate it from a guess, and it
  anchors later judgement whether or not it was meant to. Where the quantity is genuinely
  derivable, the derivation is given and the result registered; where it is not, the answer
  is that it is not known.
- **Protocol hygiene is actively maintained.** The protocol documents (`00`–`06`) are
  meant to stay lean, navigable, and free of accumulated rationale. The Strategist
  watches for three drift patterns: (a) **rationale creep** — "why we chose X" prose
  leaking into the spec when it belongs in build-logs or `changes.md`; (b) **size growth
  past natural thresholds** — any single protocol file approaching ~1000 lines, or a
  single section past ~300 lines, without a clear structural reason; (c) **patch-marks**
  — non-standard numbering (`§3b`, `§4.1` appended out of order), duplicated content
  across files, or sections that read as a sequence of additions rather than a coherent
  spec. When any of these is observed, the Strategist reports it to the Researcher
  with a proposed remedy (tighten prose / extract to build-log / split file / renumber);
  no structural change is made unilaterally. Adding a new spec always carries the
  question "is this a timeless rule, or context for one run?" — context goes to the
  build-log, not the protocol.

---

## 5. Versioning

- **Scheme:** `vMAJOR.MINOR.PATCH`.
  - `MAJOR` — framework-infrastructure era. Incremented only for a full refoundation of the
    repository or the core stack.
  - `MINOR` — a deliberate new experiment that introduces a substantive method change
    validated by a new run, directed by the Researcher.
  - `PATCH` — a fix or small adjustment to an existing minor version that does not warrant
    a new minor version (a small correction or non-method update that does not require a
    new validated run).
- **Version increment is exclusively the Researcher's role.** The Executor never increments
  a version on its own. Version proliferation — silently turning a small fix into a new
  version — is forbidden. If the Executor believes a bump is needed, it proposes and waits.
- **On every version increment, the Executor must:** (a) recognize that the version has
  changed; (b) recommend the required git action (commit, tag `vX.Y.Z`, push); and
  (c) confirm that the push completed. Code history is recovered from git, not from local
  copies of old code.
- **Per-version documents.** Each version `vX.Y.Z` owns a folder `docs/versions/vX.Y.Z/`
  holding three classes of document, by authorship and timing:
  - `changes.md` — authored by the Strategist **before** the run: what is being changed,
    and why. The "why" states the specific problem(s) in the previous version that
    motivate the change.
  - build-logs `<task>.md` (e.g. `foundation_audit.md`, `bundle_b_oc_pncbf.md`) — authored
    by the Executor **during** the build: the design decisions made, conflicts or
    ambiguities encountered and how they were resolved, and the factual results of each
    build/training task. These are durable records of *how* the version was built, kept
    distinct from the verdict on whether it worked. Any audit document also lives here.
  - `results.md` — authored by the Researcher (Strategist may draft) **after** training
    completes: the outcome; whether it improved over the prior version; the evidence; and,
    if it did not improve, the most likely cause inferred from the data. This is the only
    document that renders a verdict on the version; build-logs state facts, not verdicts.

---

## 6. Single source of truth (SSOT) discipline

- **All durable documents live under `docs/` only.** Scattered reports, audit files, and
  per-subproject documents are not created elsewhere.
- **Markdown is the truth.** The MkDocs site is only a human-facing rendering of it;
  TensorBoard is the live, in-flight monitoring channel; `data/` is local working output.
- **What git tracks vs. what stays local.**
  - **Tracked in git:** all `src/` (including `src/configs/`), `scripts/`, `docs/protocol/`,
    `docs/index.md`, `docs/ledger.md`, top-level files (`README.md`, `LICENSE`,
    `pyproject.toml`, `mkdocs.yml`), and the **`data/secured_data/`** subtree. Code history
    is recovered from commits and tags — code is never copied anywhere else to "save" it.
  - **Local only (git-ignored):** `docs/versions/` (per-version `changes.md`, build-logs,
    and `results.md` are working artifacts kept on disk), and every `data/<run_id>/`
    directory except for entries inside `data/secured_data/`. In-flight run outputs are not
    pushed; only the secured snapshot is.
- **Run output convention.** Every run writes to `data/<run_id>/`, where
  `<run_id> = vX.Y.Z__YYYYMMDD-HHMMSS__seedNN`. There is no extra `data/logs/` indirection.
  At the close of a version, the chosen final runs are copied into
  `data/secured_data/<version>/seed<N>/` and committed. The shared evaluation pools live at
  `data/secured_data/pools/` and are also committed, so that every version is evaluated
  on bit-identical pools.
- **Update cadence.** `index.md` (the state dashboard) and `ledger.md` are updated and pushed
  at version boundaries. Between pushes, the Strategist reads the last-pushed state;
  in-flight results are shared through TensorBoard, not the site.

---

## 7. Document and config map

**Documents (`docs/`):**

- `docs/index.md` — current-state dashboard (SOTA, active version, in flight, open items).
- `docs/protocol/00_constitution.md` — this document.
- `docs/protocol/01_env.md` — environment: arena, dynamics, observation, action, outcome.
- `docs/protocol/02_control.md` — agent: nominal policy, value function, safety filters,
  infeasibility, gradient.
- `docs/protocol/03_train.md` — training: scene init, value/policy learning, schedules,
  halt, training-step algorithm.
- `docs/protocol/04_eval.md` — evaluation: metric, two modes, pools, trajectory plots,
  online insertion, multi-seed statistics.
- `docs/protocol/05_code.md` — code conventions, directory structure, frozen-core boundary,
  performance requirements, logging, verification harness.
- `docs/protocol/06_workflow.md` — three-actor workflow, version lifecycle, prompt format,
  decision-brief format.
- `docs/versions/vX.Y.Z/` (local-only) — per-active-version folder containing `changes.md`,
  Executor build-logs `<task>.md` (including any audit), and `results.md`.
- `docs/ledger.md` — one row per run, with lineage and metrics.

**Configs (`src/configs/`):**

- `src/configs/base_config.yaml` — locked foundation: task definition (env, obstacle field,
  scene-init filters), network structure, filter constants, LQR weights, evaluation pools
  and scene. Rarely changed; defines "what task we solve".
- `src/configs/exp_config.yaml` — per-experiment hyperparameters: run identity, training
  schedule, optimization, value-target form, curriculum schedules, collection cadence, loss
  weights, halt thresholds, filter `alpha`, eval cadence. Deep-merged on top of
  `base_config.yaml` at runtime; saved to the run directory as `config.yaml` for full
  reproducibility.

**Code (`src/`):** the directory layout (`src/common`, `src/envs`, `src/eval`,
`src/frameworks/{jt_pncbf,oc_pncbf}`, `src/configs`) and module boundaries are defined in
`05_code`. The `data/` output layout (`data/<run_id>/` and `data/secured_data/`) is also
defined there.