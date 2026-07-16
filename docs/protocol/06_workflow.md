# 06 — Workflow

This document defines **how the three actors collaborate**, day to day. The other protocol documents define *what* (env, control, train, eval, code); this one defines *how to make progress*.

`00_constitution` declares the three actors and their authority. This document operationalizes that declaration: how a version is opened and closed, what a prompt looks like, what a decision brief looks like, how conflicts are resolved, what is committed and when, and what a daily working session checks at start and end.

---

## 1. Three actors — operating mode

### 1.1 Researcher (PI)

Sets strategy. Makes every decision: version increments, scope changes, hyperparameter targets, whether a result counts as improvement, whether to push a paper (the Researcher's decision alone — `00_constitution` §3). Reads the Strategist's analyses, decides; reads the Executor's results, decides.

### 1.2 Strategist (chat model)

Analyzes, proposes, and writes prompts. Does **not** execute code. Does **not** decide unilaterally. Persistent across sessions only through this repository — a new Strategist instance is brought up to speed by reading `docs/index.md`, `docs/ledger.md`, the current version's `docs/versions/vX.Y.Z/changes.md` and its build-logs, and the protocol documents.

The Strategist's deliverables are: framework proposals (`docs/proposals/`), retrieve and execute prompts (transient, not committed), decision briefs (delivered in chat), and version `changes.md` / `results.md` drafts. **Protocol edits** (`docs/protocol/`) are also the Strategist's responsibility under Researcher direction; the Executor never edits protocol files (`00_constitution` §1).

### 1.3 Executor (Claude Code or Codex — interchangeable)

Implements code, runs training, runs evaluation, handles tactical decisions during a run (e.g. retry on transient OOM). Refuses to make a strategic decision on its own: if the prompt is ambiguous in a way that affects results, it stops and asks. Refuses to silently extend scope: if a task requires changes beyond the prompt's stated scope, it stops and asks.

The Executor's deliverables are: code commits, run outputs (`data/<run_id>/`), per-version build-logs (`docs/versions/vX.Y.Z/<task>.md` — design decisions, conflicts and their resolution, factual task/training results; facts not verdicts), and concise status updates in chat.

### 1.4 Session boundary

A "session" is a chat conversation. Continuity across sessions is provided by the **repository**, not by any session's memory. A new Strategist instance must be able to read the repository and resume work without further context. Therefore every decision and every result lands in a tracked file (or, for local-only working artifacts such as `docs/versions/`, in the corresponding local file) before the session ends.

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

`docs/versions/` is local-only (§6); `changes.md` is the Strategist's working spec for the version, not a tracked git artifact. It must exist on disk before code changes begin, and the Executor's first action is to read it.

**Version-string bump checklist.** When the version number itself changes, the version
string is encoded in several places and must be updated together so that new run-ids,
package metadata, and config all agree. `src/_version.py` is the single source of truth;
everything else should derive from it rather than hardcode a literal. The checklist:

1. **`src/_version.py`** — the canonical `__version__` string. Update this first; all other
   references should read from it.
2. **`pyproject.toml`** — the package `version` field.
3. **`src/configs/exp_config.yaml`** — `run.version`, if present.
4. **Run-id stamping** — confirm the trainer derives the run-id version segment from
   `__version__` (e.g. `config["run"]["version"] = __version__`) so a new run is stamped
   `vX.Y.Z__<timestamp>__seed<n>`. If any code hardcodes the version literal, fix it to read
   from `_version.py` so future bumps are single-source.
5. **`grep -rn` the old version** across `src/ docs/ scripts/ tests/ *.toml *.yaml *.yml`
   to catch any remaining literal, and update or justify each. Comments and historical
   prose that legitimately reference a past version (e.g. build-logs, results docs of a
   prior version) are left as-is.
6. **`docs/versions/vX.Y.Z/`** — create the new version's working directory (for `changes.md`
   and build-logs).
7. **`docs/index.md` STATUS** — the Strategist updates the dashboard (active version, in-flight
   work); not the Executor.
8. **Existing artifacts are never renamed.** Prior-version run directories, secured snapshots,
   ledger rows, and adopted checkpoints keep their original version names; only NEW runs use
   the bumped version.

Verify after bumping: the printed `__version__` reads the new value, a dry run-id check
produces the new version segment, the post-bump grep finds no stray old-version literal in
code, and the test harness stays green (update any test that asserts the version string).

### 2.2 Implement

The Executor modifies `src/configs/exp_config.yaml` per §2.1's config delta. Source code changes accompany the config change only when the new mechanism requires them (most versions are config-only).

The verification harness (`05_code` §5) must remain green. If a source change breaks any test, the Executor halts and reports.

### 2.3 Run

The Executor runs a **smoke** stage first (`03_train` §6) on one seed. The smoke must pass — gradient-routing assertions in §6, no NaN/Inf, the loop completes — before any full run.

The Executor then launches the **full** stage for each seed in the eval plan — the canonical seed set $\{42, 99, 12345\}$ of `04_eval` §5 (≥ 3, ideally in parallel if hardware allows). Each run writes to `data/v{X.Y.Z}__{TIMESTAMP}__seed{N}/`. Every other artifact the version produces — eval-only diagnostics, label or dataset generation, supervised regression, demonstration sets, registries — is written to a directory following the `<run_id>` formats of `05_code` §3: the timestamp is never omitted, and no produced file is left loose at the top of `data/`. The Researcher monitors via `tensorboard --logdir data/`.

If a halt triggers (`03_train` §4.7), the Executor reports the halt reason and the step at which it occurred, and waits for the Researcher's instruction. It does **not** auto-restart with adjusted hyperparameters.

As the build/run proceeds, the Executor records a build-log `docs/versions/vX.Y.Z/<task>.md` per substantial build or training task: the design decisions taken, any conflict or ambiguity in the protocol and how it was resolved (flagged under a `PROTOCOL FOLLOW-UP` sub-heading when it implies a protocol amendment), and the factual results. Build-logs hold facts, not the version verdict.

### 2.4 Eval

Final evaluation (`04_eval` §2.2) runs automatically when training completes. When a version
is evaluated across multiple seeds, the Executor aggregates the per-seed results into a
multi-seed table (`04_eval` §5) — pooled bootstrap CI, mean, sample SD — and writes
`data/secured_data/<version>/aggregate/multi_seed_metrics.json` and `multi_seed_report.md`.
A single-seed version reports the single run's full-pool metrics with scene-bootstrap 95%
CIs instead; the single-seed status is stated explicitly so it is not mistaken for a
cross-seed result, and multi-seed remains the recommended target (`00_constitution`,
`04_eval` §5).

**Ledger registration (automatic after every eval).** Whenever a run produces a usable
evaluation — a full-pool final, an in-loop best, or an eval-only re-evaluation of an
existing checkpoint under a changed deploy axis — the Executor appends one row to
`docs/ledger.md` as part of completing the eval task, without waiting for a separate
instruction. The row uses the current schema
(`version | system | date | parent | seeds | cps_v2 | eval_source | reach | collision | oob | stuck | timeout | infeas | sat_rate | cps | verdict`;
`system` is the evaluated system's `System.name` (`05_code` §2 — e.g. `double_integrator`,
`unicycle`), read from the run's `config.yaml` and never inferred from the run-id or the
verdict text;
`cps_v2` is scored per `04_eval` §1's current definition — `n/a` = filter semantics out of
scope (e.g. CBF-QP slack rows), `-` = artifact unrecoverable),
filled from the run's real eval artifacts (never approximated; blank any field the eval did
not record). `eval_source` states provenance: `full_n2000`, `inloop_n500@<step>`, or
`eval_only(<note>)` for a re-evaluation that is not a new training run (e.g. a filter swap on
an adopted checkpoint). When one checkpoint is reported under more than one deployment arm,
each arm gets its own row and the arm is carried in `eval_source` (e.g. `full_n2000 (arm A)`);
an arm that did not exist for a given run is recorded `n/a`, which is not a regression.
`parent` records the source checkpoint for an eval-only or
warm-started row. The ledger is a docs file the Executor may edit; protocol files are never
edited by the Executor.

**`cps` is not commensurable across systems.** Systems differ in dynamics, control bounds, and
evaluation pools, so a `cps` measured on one system is never compared with, ranked against, or
substituted for a `cps` measured on another. The comparison rule of `04_eval` §5 and the SOTA
rule below are both scoped to a single `system`. A cross-system comparison is reported as
"comparison unavailable" (§5), never as a caveated number.

**Per-version, per-system SOTA marking (bold).** Within each version block in
`docs/ledger.md`, and separately for each `system` appearing in that block, the single row with
the highest `cps` among rows with `eval_source = full_n2000` is marked as that (version, system)
SOTA by bolding every cell in the row (markdown `**...**`).
At most one row per (version, system) pair is bolded. Rows with `eval_source` of `inloop_n500@<step>` or
`eval_only(<note>)` are not eligible for SOTA bolding. Rows from a **different deployment
or training class** than the standing comparison basis — e.g. training-free
analytic-barrier arms, or evaluations at non-default deploy rates ($dt_{\text{ctrl}}$,
$dt_{V_{\mathcal M}}$) — are likewise never SOTA-bolded on `cps` alone; the Executor
flags such rows for Researcher classification instead. When a new full-pool result
supersedes the previous SOTA **of the same system**, the previous bold is removed and the new
row is bolded in the same edit that registers the new row. A system's first full-pool row
establishes that system's baseline and supersedes nothing.

**Ledger inclusion and formatting.** Smoke runs and runs that produced no usable
evaluation row are not registered. Numeric outcome fields (`reach`, `collision`, `oob`,
`stuck`, `timeout`, `infeasibility`, `saturation_rate`, `cps`) are recorded to 4 decimal
places; CIs use the same precision when included. Run-ids are recorded verbatim from the
`<run_id>` directory. The `parent` column may be `-` if the run is not a child of another
registered run.

During the infeasibility-definition transition (`04_eval` §1 History note), each new row's
`verdict`/note field carries both scores (`legacy cps X / cps-v2 Y`); pre-transition rows
are annotated `(legacy infeasibility flag)` when cited in verdict-grade comparisons. Once
the standing comparators are re-scored under the v2 flag, single-value rows resume.

**Verdict is one line.** The `verdict` cell is a single short line (target <= ~25 words):
the core finding only — the result, SOTA status, and any honest qualifier that changes its
meaning (e.g. "CI overlaps prior -> point-estimate, not a confirmed beat", "tie",
"no collapse"). Mechanism explanations, per-checkpoint detail, and extended context belong
in the build-log and `results.md`, not the ledger; a short "see <build-log>" pointer is
allowed. Long multi-sentence verdicts are not kept in the ledger.

### 2.5 Close

The Researcher authors the results document `docs/versions/v<X>_results.md` (at the `docs/versions/` main level, alongside `v2.0.1_results.md` etc.; the Strategist may draft) — the document that renders the version verdict. Unlike the Executor's build-logs (`docs/versions/vX.Y.Z/<task>.md`), the results document interprets. Both live under the gitignored `docs/versions/` as the local SSOT. It covers at least:

1. **Result.** `cps` and the component breakdown with 95% CIs — pooled across seeds for a multi-seed version, or single-run scene-bootstrap CIs for a single-seed version (stated as such).
2. **Versus motivation.** Did the change do what §2.1 hypothesized? Cite the numbers.
3. **Improvement verdict.** Improved / no improvement detected / regressed. For multi-seed versions, by the `04_eval` §5 rule (non-overlapping CIs over the previous secured version); for single-seed, by the headline comparison against the previous secured baseline, flagged as single-seed.
4. **Next axis.** What the data suggests for the next version. A suggestion, not a commitment.

The Executor copies the chosen final run(s) from `data/<run_id>/` into the secured layout (`04_eval` §7.5), excluding the bulky per-step `metrics.csv`, and writes the `ADOPTED.md` identity record with pinned SHA-256 hashes.

**Close checklist (run at every version close).**

1. `v<X>_results.md` authored/finalized at the `docs/versions/` main level.
2. `docs/ledger.md` row updated. On SOTA change: bold the new row and add the standing
   line; mark the prior SOTA row superseded (keep history).
3. Run data moved: all of the version's artifact directories (`05_code` §3 — training runs,
   eval-only diagnostics, dataset/regression/demonstration/registry runs) from `data/` to
   `data/previous_runs/` (whole directory, as-is; nothing is renamed — §2.1 item 8). The
   SOTA run is additionally saved to
   `data/secured_data/<version>/seed<N>/` as the standard file set (`checkpoints/`,
   `figures/`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
   `git_commit.txt`, `report.md`, `status.json`, `ADOPTED.md`).
4. Confirmed changes reflected in `docs/protocol/`; all `PROTOCOL FOLLOW-UP` items from
   build-logs resolved.
5. Close preparation complete (items 1-4 done and reported). The Researcher then performs
   `git add` / `commit` / `push` directly (§2.6); the Executor does NOT run git.

**Close trigger — the Strategist close sequence (automated on "close").** When the Researcher
directs a close (says "close" or equivalent), the Strategist runs the following in order,
**without re-requesting permission at each step**, pausing only at the marked decision points:

1. **Fact-gather retrieve.** Author and issue a read-only retrieve collecting the final
   metrics, artifact inventory, gate outcomes, and open `PROTOCOL FOLLOW-UP` items the
   documents below need.
2. **`results.md`.** Draft `docs/versions/v<X>_results.md` (the four parts above), stating the
   seed basis explicitly (single-seed vs pooled).
3. **Close execute.** Author and issue the Executor close prompt — checklist items 2–3 (ledger
   row(s), secured snapshot + `ADOPTED.md`, run move) plus the `docs/index.md` dashboard update
   and tagging of any untagged `PROTOCOL FOLLOW-UP`. **No git.**
4. **Protocol delta.** For each `PROTOCOL FOLLOW-UP` item that describes the current system or
   method (not a future axis), author the before→after edit as a diff (Prohibition 3: state the
   rule on its own merits, no version narrative), then apply approved edits to `docs/protocol/`
   directly (the Strategist edits protocol; §1).
5. **Git commands.** Present the step-by-step `git` sequence (§2.6) for the Researcher to run.

**Decision points (the sequence pauses and asks; nothing else is re-asked):** (a) **multi-seed
escalation** — whether to promote a verdict-grade single-seed result to `{42, 99, 12345}` before
the verdict; (b) **protocol-delta approval** (step 4, before applying); (c) **git execution**
(step 5, always the Researcher). Absent a decision-point answer, the Strategist proceeds with the
other steps and surfaces the pending decision, rather than blocking the whole close.

Version string bumps are done AFTER push, not during close.

### 2.6 Push

The Strategist presents the git command sequence, including the commit message; the Executor
prepares the close (checklist items 2–3) but does NOT run git and does NOT draft or recommend
git commands or commit messages (`00_constitution` §1, §3 Prohibition 4). The Executor's only
git-adjacent contribution is the grouped file list from its action log. The Researcher performs
`git add` / `commit` / `push` / `tag` directly via bash once close preparation is complete.
Since `docs/versions/` is local-only (§6.2), the version's `changes.md`, build-logs, and
results are not staged; what is committed is the code/config change, the updated dashboard
and ledger, and the secured snapshot:

```bash
git add src/ tests/ \
        docs/protocol/ docs/index.md docs/ledger.md \
        src/configs/exp_config.yaml \
        data/secured_data/
git commit -m "vX.Y.Z: <one-line summary>"
git tag vX.Y.Z
git push && git push --tags
```

`docs/index.md` (the state dashboard) and `docs/ledger.md` (one row per run, with lineage and `cps`) are updated as part of the same commit. Protocol edits, if any, are committed in the same step. The commit message is **one line, prose, no metrics or data** (`vX.Y.Z: <one-line summary>`) — the version's *what*, not its numbers; results, deltas, and CIs live in `results.md` and the ledger, never in the commit message. The tag `vX.Y.Z` is the authoritative reference to this version's code state; secured artifacts in `data/secured_data/` are the authoritative reference to its results. The version's process documents remain in the local-only `docs/versions/` and are not part of the tagged commit.

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

- `src/`, `scripts/`, `tests/`, and the top-level files (`README.md`, `LICENSE`,
  `pyproject.toml`, `mkdocs.yml`, `.gitignore`).
- Within `docs/`: the protocol (`docs/protocol/`), the state dashboard (`docs/index.md`),
  the run ledger (`docs/ledger.md`), the MkDocs assets (`docs/javascripts/`), and any
  top-level docs pages. The committed repository carries the result — code, spec, ledger,
  dashboard, and the secured baseline — not the development-process documents.
- `data/secured_data/` — both the pools (`data/secured_data/pools/`) and every closed
  version's snapshot.

### 6.2 What is local-only

- **`docs/versions/`** — the per-version working directory (`changes.md`, build-logs,
  `results.md`). These are the development-process record, kept locally as the SSOT
  the Strategist and Executor read, but excluded from the committed repository. The MkDocs
  site (built locally) renders them; the git history does not carry them.
- Every `data/<run_id>/` directory (in-flight and superseded runs) except what is promoted
  to `data/secured_data/`. The `.gitignore` pattern enforces this (`data/*` ignored,
  `data/secured_data/` excepted).
- TensorBoard event files inside `data/<run_id>/tensorboard/` — large and redundant with
  the CSVs that are secured at close.

### 6.3 secured_data conventions

- **Pools.** Generated once via `src/eval/build_pools.py` and committed; they are
  immutable thereafter. Regenerating them is a major-version event because `cps`
  comparisons across versions assume identical pools.
- **Version snapshots.** Copied at §2.5 close into `data/secured_data/<version>/seed<N>/`,
  containing the adopted checkpoint, config, eval metrics/episodes, pinned SHA-256 hashes,
  and figures, with an `ADOPTED.md` identity record. Bulky per-step training logs
  (`metrics.csv`) are excluded from the secured set; the full curve stays in the
  gitignored original run directory. Checkpoint files are committed plain (no LFS) while
  they remain small.
- **Aggregate.** When a version is evaluated across multiple seeds,
  `data/secured_data/<version>/aggregate/` carries the cross-seed bootstrap and the
  one-page summary. A single-seed version has no aggregate; its secured snapshot and
  `ADOPTED.md` are the canonical reference.
- **Experiments (secured diagnostics).** A diagnostic run kept for the record but that is
  **not** the version's SOTA snapshot — a Researcher-directed ablation or what-if on a
  version's base (e.g. an alternate actuator or filter spec) — is stored under
  `data/secured_data/<version>/experiments/<name>/`, `<name>` a short snake_case tag. It
  holds the diagnostic's own eval artifacts (the `04_eval` §7.5 file set, or the subset the
  run produced) plus a `README.md` stating the base version, the exact change from that
  base, the verdict, and a pointer to the results-doc section that interprets it. It never
  carries a SOTA claim: the version's SOTA remains its `seed<N>/` snapshot, and an
  experiments entry is never eligible for ledger SOTA bolding (§2.4). **Promotion is
  Researcher-gated.** Copying anything into `data/secured_data/<version>/experiments/` is,
  like every other secured promotion (§2.5), an explicit Researcher decision; the Executor
  proposes and waits and never moves a diagnostic into `secured_data/` on its own.

### 6.4 MkDocs and TensorBoard

- **MkDocs site** is built locally from `docs/` (`mkdocs serve` / `mkdocs build`); it renders the local SSOT, including the gitignored `docs/versions/`. It is not deployed at push by default.
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

Confirm: §2.5 `results.md` drafted — multi-seed aggregate written (or single-seed status stated) — secured snapshot copied with `ADOPTED.md` — `index.md` and `ledger.md` updated (including SOTA bold per §2.4) — git status reviewed. Then §2.6 push.

### 7.4 Ending a session

Confirm: everything decided is in a tracked file (prompts, briefs, results), and any code in flight is either committed or explicitly noted as work-in-progress in `docs/index.md`. Anything not in a tracked file does not exist for the next session.

---

## 8. Reporting and recovery discipline

**Disk-verified reporting (mandatory).** Every number the Executor reports about a run —
monitoring updates, termination facts, peaks, current steps — is read from the run's
on-disk artifacts at reporting time and cited as such: the run directory, the verbatim
`eval_metrics.csv` / `metrics.csv` row, the process start time, and an arithmetic
consistency check (elapsed $\times$ measured rate $\approx$ reported step). A number
that does not appear on disk is not reported. Recovery and resume prompts re-verify the
run state against disk before recording any termination fact; framing supplied from
outside the disk (a step count, a peak, a "terminated at") is treated as unverified until
the artifacts confirm it.

**External-interruption handling (standing decision, "Option A").** When a training run
is killed by an external event (session teardown, machine contention) rather than by an
abort rule: the latest cadence checkpoint stands in for `final.pt`; the build-log and the
ledger row's `eval_source` note record `terminated @<disk-verified step> (external
interrupt)`; and the standard close-out (re-selection over all cadence checkpoints,
full-pool eval of the selected best) proceeds on the surviving artifacts. Optimizer state
is never spliced to resume a partially lost run; if the full budget is wanted, the run is
relaunched fresh and the killed run is archived to `previous_runs` with its termination
note. A `status.json` left stale by a hard kill is recorded as stale, never edited.

**Parallel low-priority work.** Analysis or evaluation may run alongside a live training
run only under all of: `nice -n 19` with CPU-core pinning away from the trainer's cores;
GPU batch sizes capped to leave $\ge 2$ GB VRAM headroom (halve-and-retry on OOM — never
touch the training process); read-only access to the live run directory, with all outputs
written elsewhere; and a measured throughput guard — if the trainer's step rate degrades
more than 10% from its pre-launch window, the parallel work stops and reports.