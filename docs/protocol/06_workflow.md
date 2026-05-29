# 06 — Workflow

This document defines **how the three actors collaborate**, day to day. The other protocol documents define *what* (env, control, train, eval, code); this one defines *how to make progress*.

`00_constitution` declares the three actors and their authority. This document operationalizes that declaration: how a version is opened and closed, what a prompt looks like, what a decision brief looks like, how conflicts are resolved, what is committed and when, and what a daily working session checks at start and end.

---

## 1. Three actors — operating mode

### 1.1 Researcher (PI)

Sets strategy. Makes every decision: version increments, scope changes, hyperparameter targets, whether a result counts as improvement, whether to push a paper (the Researcher's decision alone — `00_constitution` §3). Reads the Strategist's analyses, decides; reads the Executor's results, decides.

### 1.2 Strategist (chat model)

Analyzes, proposes, and writes prompts. Does **not** execute code. Does **not** decide unilaterally. Persistent across sessions only through this repository — a new Strategist instance is brought up to speed by reading `docs/index.md`, `docs/ledger.md`, the current version's `docs/versions/vX.Y.Z/changes.md` and its build-logs, and the protocol documents.

The Strategist's deliverables are: framework proposals (`docs/proposals/`), retrieve and execute prompts (transient, not committed), decision briefs (delivered in chat), and version `changes.md` / `results.md` drafts.

### 1.3 Executor (Claude Code or Codex — interchangeable)

Implements code, runs training, runs evaluation, handles tactical decisions during a run (e.g. retry on transient OOM). Refuses to make a strategic decision on its own: if the prompt is ambiguous in a way that affects results, it stops and asks. Refuses to silently extend scope: if a task requires changes beyond the prompt's stated scope, it stops and asks.

The Executor's deliverables are: code commits, run outputs (`data/<run_id>/`), per-version build-logs (`docs/versions/vX.Y.Z/<task>.md` — design decisions, conflicts and their resolution, factual task/training results; facts not verdicts), and concise status updates in chat.

### 1.4 Session boundary

A "session" is a chat conversation. Continuity across sessions is provided by the **repository**, not by any session's memory. A new Strategist instance must be able to read the repository and resume work without further context. Therefore every decision and every result lands in a tracked file before the session ends.

---

## 2. Version lifecycle — the spine of the workflow

Every increment of `vX.Y.Z` follows the same six steps. The Researcher initiates; the Strategist drafts; the Executor implements and runs.

### 2.1 Open

The Researcher decides to start a new version (`vX.Y.Z`). The Strategist drafts `docs/versions/vX.Y.Z/changes.md` (creating the version folder) with six headings:

1. **Motivation.** The specific problem(s) in the previous version that motivate this change. Cite data: which numbers, from which run, suggest the problem.
2. **Hypothesis.** What this change is expected to do, and what observable in `eval_metrics.csv` or TensorBoard should move and in which direction.
3. **One-axis check.** Per the `00_constitution` §4 recommendation: is this a single new mechanism? If multiple substantive axes change, flag explicitly and request Researcher re-confirmation before proceeding.
4. **Config delta.** Exactly which keys in `exp_config.yaml` (and, rarely, `base_config.yaml`) change. A change to `base_config.yaml` requires Researcher re-confirmation.
5. **Eval plan.** Number of seeds (≥ 3, recommend ≥ 5). Comparison target: the previous secured version. Comparison rule: non-overlapping 95% CIs on pooled `cps` (`04_eval` §5).
6. **Risks.** What could go wrong; what the halt protocol (`03_train` §4.7) would catch; what to watch in TensorBoard during the first hours.

The document is committed before any code change. The Executor's first action is to read it.

### 2.2 Implement

The Executor modifies `src/configs/exp_config.yaml` per §2.1's config delta. Source code changes accompany the config change only when the new mechanism requires them (most versions are config-only).

The verification harness (`05_code` §5) must remain green. If a source change breaks any test, the Executor halts and reports.

### 2.3 Run

The Executor runs a **smoke** stage first (`03_train` §6) on one seed. The smoke must pass — gradient-routing assertions in §6, no NaN/Inf, the loop completes — before any full run.

The Executor then launches the **full** stage for each seed in the eval plan (≥ 3, ideally in parallel if hardware allows). Each run writes to `data/v{X.Y.Z}__{TIMESTAMP}__seed{N}/`. The Researcher monitors via `tensorboard --logdir data/`.

If a halt triggers (`03_train` §4.7), the Executor reports the halt reason and the step at which it occurred, and waits for the Researcher's instruction. It does **not** auto-restart with adjusted hyperparameters.

As the build/run proceeds, the Executor records a build-log `docs/versions/vX.Y.Z/<task>.md` per substantial build or training task: the design decisions taken, any conflict or ambiguity in the protocol and how it was resolved (flagged under a `PROTOCOL FOLLOW-UP` sub-heading when it implies a protocol amendment), and the factual results. Build-logs hold facts, not the version verdict.

### 2.4 Eval

Final evaluation (`04_eval` §2.2) runs automatically when training completes. The Executor then aggregates the per-seed results into a multi-seed table (`04_eval` §5) — pooled bootstrap CI, mean, sample SD — and writes `data/secured_data/<version>/aggregate/multi_seed_metrics.json` and `multi_seed_report.md` (these paths are written to even before §2.5 close, because the aggregate is the input to the close decision).

### 2.5 Close

The Researcher authors `docs/versions/vX.Y.Z/results.md` (the Strategist may draft) with four headings. Unlike the Executor's build-logs in the same folder, this is the document that renders the version verdict:

1. **Result.** Pooled `cps` (mean and 95% CI), with per-seed values and the component breakdown.
2. **Versus motivation.** Did the change do what §2.1 hypothesized? Cite the numbers.
3. **Improvement verdict.** Improved / no improvement detected / regressed, per the `04_eval` §5 comparison rule (non-overlapping CIs over the previous secured version).
4. **Next axis.** What the data suggests for the next version. This is a suggestion to the Researcher, not a commitment.

The Executor copies the chosen final runs from `data/<run_id>/` into `data/secured_data/<version>/seed<N>/` (`04_eval` §7.5). "Chosen" usually means all seeds; if any run is unrepresentative (e.g. halted early for an unrelated reason), the Researcher decides whether to include it.

### 2.6 Push

The Executor recommends — and, after Researcher confirmation, performs — the git operation:

```bash
git add docs/versions/vX.Y.Z/ \
        docs/index.md docs/ledger.md \
        src/configs/exp_config.yaml \
        data/secured_data/<version>/
git commit -m "vX.Y.Z: <one-line summary>"
git tag vX.Y.Z
git push && git push --tags
```

`docs/index.md` (the state dashboard) and `docs/ledger.md` (one row per run, with lineage and `cps`) are updated as part of the same commit. The tag `vX.Y.Z` is the authoritative reference to this version's code state; secured artifacts in `data/secured_data/<version>/` are the authoritative reference to its results.

Between version pushes, the Strategist reads the **last pushed state** (commits, secured data); in-flight, unsecured work is not the Strategist's source of truth.

---

## 3. Prompt regime — Strategist → Executor

Two kinds of prompt, and only two.

### 3.1 Retrieve prompt — read-only information gathering

Used to extract facts from the repository without altering it. Format:

- **Task** (one sentence).
- **Goal** (concrete deliverable: a single Markdown report at a stated path).
- **Hard constraints (safety).** Read-only on every file. No installs, no network, no state-changing commands. Output exactly one file; do not edit anything else.
- **Scope.** Which subtrees to inspect, ordered by priority.
- **Headings.** The exact section structure of the report (so multiple retrieves are comparable).
- **Done when.** A precise termination condition, ending with "print the report path and its line count, and nothing else."

Quotes from code must be ≤ ~25 lines and always carry a source path. Anything the retrieve cannot determine is reported as an explicit uncertainty, not guessed.

### 3.2 Execute prompt — code or training

Used to write code, modify configs, run training, or run evaluation. Format:

- **Task** (one sentence).
- **Strategic rationale** (one short paragraph: why this matters in the version's context).
- **Scope constraints** (what may be changed; what must remain untouched).
- **Safeguards.** Smoke gate, the FAIL scenarios with the response to each, abort conditions, file read-only flags, checkpoint cadence.
- **Pass / fail criteria.** Objective, observable.
- **Reference patterns.** Pointers to prior code in the repo to follow; **no boilerplate inlined in the prompt** (the Executor reads its environment).

Time budgets are **not** included in execute prompts. The Researcher's mentions of time mean "proceed autonomously," not "finish within X minutes." Quality over speed.

### 3.3 Prompt authoring authority

The Strategist authors every prompt. The Researcher reviews and approves before the Executor sees it. The Executor must not invent strategy from an under-specified prompt; if the scope is ambiguous in a way that affects results, the Executor stops and asks.

---

## 4. Decision brief format — Strategist → Researcher

When a decision is required, the Strategist delivers a four-part brief, never longer than one screen:

1. **Problem.** What the decision is about, with the evidence (data / quote / observation).
2. **Options.** Two or three concrete alternatives. Each option is fully specified — config keys, file changes, expected effect.
3. **Trade-offs.** For each option, what is gained and what is lost. Side effects, risks, computational cost.
4. **Recommendation.** One option, with a one-sentence justification.

The Strategist's `00_constitution` §4 problem-analysis discipline (problem / mechanism / fix / trade-off) applies inside §1 and §3. Long prose analyses without this structure are not decision briefs — they are background documents.

---

## 5. Conflict resolution

When sources disagree, follow `00_constitution` §2. Specifically:

- **Researcher instruction vs. Constitution / PROHIBITIONS.** The actor does not silently comply or refuse; it reports the conflict and requests re-confirmation. Once re-confirmed, the instruction takes precedence.
- **Strategist memory vs. code or recorded data.** The code and the data win. The Strategist verifies before reporting a number.
- **Two retrieve reports conflict.** The more recent source wins; if uncertainty remains, a new retrieve is dispatched.
- **Eval result vs. hypothesis.** The hypothesis is rejected. A prompt does **not** re-interpret the result to preserve the hypothesis; the result is reported as-is and the next version's hypothesis is revised accordingly.
- **Across-version comparison ambiguity.** If a comparison cannot be made on bit-identical pools, it is reported as "comparison unavailable" rather than reported with a caveat. The fix is to re-evaluate on the shared pool, not to caveat.

---

## 6. SSOT operations — git, data, MkDocs, TensorBoard

### 6.1 What git tracks

- All of `src/`, `scripts/`, `docs/`, `tests/`, and the top-level files (`README.md`, `LICENSE`, `pyproject.toml`, `mkdocs.yml`, `.gitignore`). This includes every per-version folder `docs/versions/vX.Y.Z/` (its `changes.md`, build-logs, and `results.md`); audit and build-log files live only there, never at the repo root (SSOT, `00_constitution` §6).
- `data/secured_data/` — both the pools (`data/secured_data/pools/`) and every closed version's snapshot (`data/secured_data/<version>/`).

### 6.2 What is local-only

- Every `data/<run_id>/` directory (in-flight runs). The `.gitignore` pattern in `05_code` §4 enforces this.
- TensorBoard event files inside `data/<run_id>/tensorboard/` — large and redundant with the CSVs that are secured at close.

### 6.3 secured_data conventions

- **Pools.** Generated once via `src/eval/build_pools.py` and committed at the start of v2.0.0. They are immutable thereafter: regenerating them is a major version event because `cps` comparisons across versions assume identical pools.
- **Version snapshots.** Copied at §2.5 close. Contents listed in `04_eval` §7.5. Checkpoint files are committed plain (no LFS) in v2.0.0.
- **Aggregate.** `data/secured_data/<version>/aggregate/` carries the cross-seed bootstrap and the one-page summary. This is the canonical reference for any future comparison.

### 6.4 MkDocs and TensorBoard

- **MkDocs site** is built from `docs/` at every push. It is a rendering of the committed truth; it does not show in-flight work.
- **TensorBoard** is the in-flight monitoring channel, served from `data/`. It is the Researcher's primary view during a run.
- The two never disagree: anything in TensorBoard that matters becomes a row in a secured CSV at version close.

---

## 7. Daily-operations checklist

### 7.1 Starting a session

1. Pull from git.
2. Read `docs/index.md` (current state) and the last row of `docs/ledger.md`.
3. If a version is in flight, read its `docs/versions/vX.Y.Z/changes.md` and any build-logs in that folder.
4. Read `docs/protocol/00_constitution.md` for the precedence rules and PROHIBITIONS.
5. Check `tensorboard --logdir data/` if any run is active.

### 7.2 Before sending a prompt

Confirm: which type (retrieve vs execute) — scope — safeguards — done-when (retrieve) or pass/fail (execute) — reference patterns. If any are missing, fill before sending.

### 7.3 Before closing a version

Confirm: §2.5 `results.md` drafted — multi-seed aggregate written — secured snapshot copied — `index.md` and `ledger.md` updated — git status reviewed. Then §2.6 push.

### 7.4 Ending a session

Confirm: everything decided is in a tracked file (prompts, briefs, results), and any code in flight is either committed or explicitly noted as work-in-progress in `docs/index.md`. Anything not in a tracked file does not exist for the next session.
