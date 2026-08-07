# 06 — Workflow

This document defines **how the three actors collaborate**, day to day. The other protocol documents define *what* (env, control, train, eval, code); this one defines *how to make progress*.

`00_constitution` declares the three actors and their authority. This document operationalizes that declaration: how a version is opened and closed, what a prompt looks like, what a decision brief looks like, how conflicts are resolved, what is committed and when, and what a daily working session checks at start and end.

---

## 1. Three actors — operating mode

### 1.1 Researcher (PI)

Sets strategy. Makes every decision: version increments, scope changes, hyperparameter targets, whether a result counts as improvement, whether to push a paper (the Researcher's decision alone — `00_constitution` §3). Reads the Strategist's analyses, decides; reads the Executor's results, decides.

### 1.2 Strategist (chat model)

Analyzes, proposes, and writes prompts. Does **not** execute code. Does **not** decide unilaterally. Persistent across sessions only through this repository — a new Strategist instance is brought up to speed by reading `docs/index.md`, `docs/ledger.md`, the current version's `docs/versions/vX.Y.Z/changes.md` and its build-logs, and the protocol documents.

The Strategist's deliverables are: framework proposals (`docs/proposals/`), retrieve and execute prompts (transient, not committed), decision briefs (delivered in chat), version `changes.md` / `results.md` drafts, and the theory document (`docs/tex/theory.tex`) — the Strategist's own research: it derives, states, and proves the mathematics and delivers the compiled `.tex` and PDF; the Executor installs verbatim and reports build errors without repairing the mathematics. Falsified statements and the version's measured results are reflected in `theory.tex` before that version's close. **Protocol edits** (`docs/protocol/`) are also the Strategist's responsibility under Researcher direction; the Executor never edits protocol files (`00_constitution` §1).

### 1.3 Executor (Claude Code or Codex — interchangeable)

Implements code, runs training, runs evaluation, handles tactical decisions during a run (e.g. retry on transient OOM). Refuses to make a strategic decision on its own: if the prompt is ambiguous in a way that affects results, it stops and asks. Refuses to silently extend scope: if a task requires changes beyond the prompt's stated scope, it stops and asks.

The Executor's deliverables are: code commits, run outputs (`data/<run_id>/`), per-version build-logs (`docs/versions/vX.Y.Z/<task>.md` — design decisions, conflicts and their resolution, factual task/training results; facts not verdicts), and concise status updates in chat.

**Subagents.** A subagent holds only the authority of the prompt that delegates to it, never more. Its scope is additive: creating new files, run directories, and analyses, and reading anything. Modifying or removing what another run or record depends on is outside it — the test is whether the prior state can be reconstructed exactly afterwards. Launching training, registering a condition, lifting a guard, and restarting a run stay with the Executor. A subagent's output enters a report only after the Executor confirms it on disk. A launch must be stoppable by a single `kill` of a PID recorded in the build-log at launch time; under that condition a session-detached launch form is permitted. A launch whose stop procedure is not recorded is not permitted in any form.

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

**When the bump happens.** The bump to the next version string is performed immediately after
the preceding version's push, not at the next version's close. The close of `vX.Y.Z` therefore
asserts that `src/_version.py`, `pyproject.toml`, and `exp_config.yaml` `run.version` all read
`vX.Y.Z`; a mismatch is reported as a discrepancy and resolved before the close commit, because
an artifact produced under a stale string is indistinguishable at run level from the previous
version's.

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

The Executor then launches the **full** stage for each seed in the eval plan — the canonical seed set $\{42, 99, 12345\}$ of `04_eval` §5 (≥ 3, ideally in parallel if hardware allows). Each run writes to `data/runs/v{X.Y.Z}/v{X.Y.Z}__{TIMESTAMP}__seed{N}/`. Every other artifact the version produces — eval-only diagnostics, label or dataset generation, supervised regression, demonstration sets, registries, analysis output — is written under the same version directory, as one of the three entry kinds of `05_code` §3: a `<run_id>/` whose timestamp is never omitted, a `set__…/` wrapper for a multi-phase execution, or a named `<group>/`. No produced file is left loose at the top of `data/`, `data/runs/`, or the version directory. The Researcher monitors via `tensorboard --logdir data/runs/`.

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

**A run trained under a defective configuration is not ledger material.** When a model was
trained with a plant or observation configuration since found to be wrong — a missing state
component, a mis-specified action space — its numbers describe a system that is not the one
under study. Such a run may be deleted outright; it is not registered, and an already-registered
row for it is removed rather than annotated.

**Uninformative run directories are deleted without approval.** A data directory whose run
terminated before producing any information — a smoke or launch attempt killed by a defect
before its first scored eval, a crashed segment superseded by a clean relaunch, a partial
collection with no conclusion, lesson, or registered reading attached — is deleted by the
Executor directly, recording only the run-id and one line of reason in the active version's
build-log. This keeps `data/` from accumulating meaningless trees. The rule is one-sided:
any run that produced information stays — a completed run, a run stopped by a registered
falsifier or halt, a failed run whose failure carries a recorded conclusion or lesson, and
anything a document cites, are records and are never deleted under this clause. When in
doubt, keep.

**`cps` is not commensurable across systems.** Systems differ in dynamics, control bounds, and
evaluation pools, so a `cps` measured on one system is never compared with, ranked against, or
substituted for a `cps` measured on another. The comparison rule of `04_eval` §5 and the SOTA
rule below are both scoped to a single `system`. A cross-system comparison is reported as
"comparison unavailable" (§5), never as a caveated number.

**SOTA eligibility is a pool-size condition, not a row-type condition.** A row qualifies for
the bold marking below whenever its evaluation used `n >= 2000` episodes. Whether the row was
produced by a full-pool final, an in-loop best, or an eval-only re-evaluation does not enter the
question; the pool size does.

**SOTA marking (bold).** Bold marks the current headline of a lineage: **exactly one bold row
per `system`, across the whole ledger, at any time.** The bold row is that system's highest-scoring
eligible row; every cell of it is bolded (markdown `**...**`).
Eligibility is the pool-size condition stated above and nothing else: `n >= 2000`, whatever the
row type — an eval-only re-evaluation on a 2000-episode pool is eligible, an in-loop best or a
full-pool final on 500 is not. Ranking is on `cps_v2`, the `04_eval` §1 current definition; a row
whose `cps_v2` reads `-` is not rankable and is flagged for Researcher classification rather than
ranked on the legacy `cps` column. A standing bold row that predates the current definition is
flagged, not un-bolded, until it is re-scored. Rows from a **different deployment or training
class** than the standing comparison basis — e.g. training-free analytic-barrier arms, or
evaluations at non-default deploy rates ($dt_{\text{ctrl}}$, $dt_{V_{\mathcal M}}$) — are
likewise never SOTA-bolded on `cps` alone; the
Executor flags such rows for Researcher classification instead. When a new full-pool result
supersedes the previous SOTA **of the same system**, the previous bold is removed in the same
edit that registers the new row, and the superseded row carries a standing supersession line;
historical rows never retain bold. A system's first full-pool row establishes that system's
baseline and supersedes nothing. A bold row is a claim the repository must be able to back:
its adopted checkpoint is in `data/secured_data/` and committed (§6.3).

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

**Within-version iterations.** When the Researcher directs further mechanism iterations
inside an open version: each iteration registers its hypothesis and falsifier in the
transmitted prompt (`changes.md` remains an open-once document and is not amended per
iteration); each completed iteration appends its ledger row(s) immediately (interim
registration — the ledger never waits for close; bold only on a CI-separated improvement
claim); and the phase report carries one clearly-titled section per iteration.

**A prediction is registered against a measurement that can resolve it.** Before a hypothesis is
registered, three things are checked and recorded with it: that the metric responds to the axis
under test, that the pool it is read on exposes that axis rather than suppressing it, and that the
threshold sits inside the range the axis can reach. Any of the three failing produces a prediction
that scores without informing — satisfied before the run starts, saturated on the chosen pool, or
placed where nothing can move it — and that is a specification error rather than a result
(`00_constitution` §4). It is scored as registered, said to be uninformative and why, and the
informative form registered before the next data. A threshold is never re-fitted after seeing the
data it was meant to test.

Where a version changes an outcome predicate, a metric definition, or an initial-condition
distribution, the comparator is re-measured under the new conditions rather than carried over from a
prior report: two numbers that were not produced under the same predicates and deploy settings do
not belong in one column, whatever their names.

### 2.5 Close

The Researcher authors the results document `docs/versions/v<X>_results.md` (at the `docs/versions/` main level, alongside `v2.0.1_results.md` etc.; the Strategist may draft) — the document that renders the version verdict. Unlike the Executor's build-logs (`docs/versions/vX.Y.Z/<task>.md`), the results document interprets. Both live under the gitignored `docs/versions/` as the local SSOT. It covers at least:

1. **Result.** `cps` and the component breakdown with 95% CIs — pooled across seeds for a multi-seed version, or single-run scene-bootstrap CIs for a single-seed version (stated as such).
2. **Versus motivation.** Did the change do what §2.1 hypothesized? Cite the numbers.
3. **Improvement verdict.** Improved / no improvement detected / regressed. For multi-seed versions, by the `04_eval` §5 rule (non-overlapping CIs over the previous secured version); for single-seed, by the headline comparison against the previous secured baseline, flagged as single-seed.
4. **Next axis.** What the data suggests for the next version. A suggestion, not a commitment.

The Executor copies the chosen final run(s) from `data/runs/vX.Y.Z/` into the secured layout (`04_eval` §7.5), excluding the bulky per-step `metrics.csv`, and writes the `ADOPTED.md` identity record with pinned SHA-256 hashes. The originals are not moved.

**Close checklist (run at every version close).**

1. `v<X>_results.md` authored/finalized at the `docs/versions/` main level.
2. `docs/ledger.md` row updated. On SOTA change: bold the new row and add the standing
   line; mark the prior SOTA row superseded (keep history).
3. Run data left in place. The version's artifact directories were written under
   `data/runs/vX.Y.Z/` at creation time (`05_code` §3) and are not moved, renamed, or
   archived at close; a closed version's artifacts stay at the path every report already
   cites. The close's only data operation is promotion: the SOTA run is **copied** to
   `data/secured_data/<version>/seed<N>/` as the standard file set (`checkpoints/`,
   `figures/`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`, `pool_manifest.json`,
   `git_commit.txt`, `report.md`, `status.json`, `ADOPTED.md`), and the copy is confirmed
   staged (`05_code` §4). The §6.3 presence check then runs over **every** system's bold row,
   not only the closing version's, and any bold row whose checkpoint is absent from
   `secured_data/` is reported before the close proceeds.
4. Confirmed changes reflected in `docs/protocol/`; all `PROTOCOL FOLLOW-UP` items from
   build-logs resolved.
5. Close preparation complete (items 1-4 done and reported). The Researcher then performs
   `git add` / `commit` / `push` directly (§2.6); the Executor does NOT run git.

**Close trigger — the Strategist close sequence (automated on "close").** When the Researcher
directs a close (says "close" or equivalent), the Strategist runs the following STRICTLY IN
ORDER, **without re-requesting permission at each step**; each step begins only after the
previous step's artifact exists on disk. Steps marked [PAUSE] require a Researcher decision;
a paused step suspends only its dependents, not the other steps.

1. **Fact-gather retrieve** (Strategist authors, Executor runs). Read-only, with exactly two
   write exceptions: (a) create the EMPTY results placeholder
   `docs/versions/v<X>_results.md` containing only the single line
   `# v<X> results — CLOSE IN PROGRESS (<date>)`, and (b) the retrieve report itself.
   **Work that is already done is not done again.** Before each step of this sequence begins,
   its artifact is checked for existence; where the artifact already exists it is updated, not
   rewritten, and existing content is never overwritten. Write exception (a) is therefore
   performed only when `v<X>_results.md` is absent or empty. The
   report must cover: headline metrics recomputed from artifacts with CIs; provenance (run
   dirs, checkpoint SHA-256 prefixes); gate/test outcomes; ledger rows verbatim with bold
   state; version strings; git status; open `PROTOCOL FOLLOW-UP` items; and an explicit
   discrepancy list. No results content is drafted before this report exists.
2. **`results.md`** (Strategist authors). Drafted EXCLUSIVELY from the step-1 report — every
   number beside its artifact path; the four parts above; the seed basis stated explicitly
   (single-seed vs pooled). [PAUSE] only if a multi-seed escalation decision is open.
3. **Close execute** (Strategist authors, Executor runs). Fills the placeholder with the
   final results text verbatim, then checklist items 2–3: ledger row(s); no run is moved
   (§2.5 item 3); [PAUSE] secured snapshot(s) + `ADOPTED.md` (the promotion scope is
   an explicit Researcher approval carried in the transmittal, never assumed);
   `docs/index.md` dashboard update; tagging of any untagged `PROTOCOL FOLLOW-UP`. **No git.**
   The STATUS update is the last write of the close, performed after the secured promotion so
   that it describes the finished state, and it is carried out by the Executor from the close
   prompt rather than left to a separate hand edit.
4. **Protocol delta** [PAUSE]. For each `PROTOCOL FOLLOW-UP` item that describes the current
   system or method (not a future axis), author the before→after edit as a diff (Prohibition
   3: state the rule on its own merits, no version narrative), then apply approved edits to
   `docs/protocol/` directly (the Strategist edits protocol; §1).
5. **Git commands.** Present the step-by-step `git` sequence (§2.6) for the Researcher to
   run, only after steps 3–4 are complete, so the close commit includes the protocol edits.

**Decision points (the sequence pauses and asks; nothing else is re-asked):** (a) **multi-seed
escalation** — whether to promote a verdict-grade single-seed result to `{42, 99, 12345}` before
the verdict; (b) **secured promotion scope** (step 3 — which run(s) are snapshotted);
(c) **protocol-delta approval** (step 4, before applying); (d) **git execution**
(step 5, always the Researcher). Absent a decision-point answer, the Strategist proceeds with the
other steps and surfaces the pending decision, rather than blocking the whole close.

Version string bumps are done AFTER push, not during close.

### 2.6 Push

The Strategist presents the git command sequence, including the commit message; the Executor
prepares the close (checklist items 2–3) but does NOT run git and does NOT draft or recommend
git commands or commit messages (`00_constitution` §1, §3 Prohibition 4). The Executor's only
git-adjacent contribution is the grouped file list from its action log. The Researcher performs
`git add` / `commit` / `push` / `tag` directly via bash once close preparation is complete.
The version's `changes.md` and build-logs stay local (§6.2) and are not staged; the results
document is tracked (`00_constitution` §6) and is staged with the close. What is committed is
the code/config change, the updated dashboard and ledger, the results document, and the secured
snapshot:

```bash
git add src/ tests/ \
        docs/protocol/ docs/index.md docs/ledger.md \
        docs/versions/*_results.md docs/tex/ \
        src/configs/exp_config.yaml \
        data/secured_data/
git commit -m "vX.Y.Z: <one-line summary>"
git tag vX.Y.Z
git push && git push --tags
```

`docs/index.md` (the state dashboard) and `docs/ledger.md` (one row per run, with lineage and `cps`) are updated as part of the same commit. Protocol edits, if any, are committed in the same step. The commit message is **one line, prose, no metrics or data** (`vX.Y.Z: <one-line summary>`) — the version's *what*, not its numbers; results, deltas, and CIs live in `results.md` and the ledger, never in the commit message. The tag `vX.Y.Z` is the authoritative reference to this version's code state; secured artifacts in `data/secured_data/` are the authoritative reference to its results. The version's working documents — `changes.md` and the build-logs — remain in the local-only `docs/versions/vX.Y.Z/` and are not part of the tagged commit; the results document is.

Between version pushes, the Strategist reads the **last pushed state** (commits, secured data); in-flight, unsecured work is not the Strategist's source of truth.

---

## 3. Prompt regime — Strategist → Executor

Three kinds of prompt, and only three.

**Carry-forward.** Every prompt opens with the items from earlier dispatches that remain unanswered. An item is closed only by an answer, by a retrieve that obtains it, or by a stated reason it is no longer needed — never by omission. Where an item cannot be answered as asked, the Executor reports it as unanswered and says what would answer it.

**Version literals.** A prompt title carries a version literal only if that version is open on disk (`changes.md` exists); before dispatch the Strategist greps the prompt for `v[0-9]+\.[0-9]+\.[0-9]+` and every hit must name the open version. Chat descriptions of a prompt are grounded in its literal text.

### 3.1 Retrieve prompt — read-only information gathering

Used to extract facts from the repository without altering it. Format:

- **Task** (one sentence).
- **Goal** (concrete deliverable: a single Markdown report at a stated path).
- **Hard constraints (safety).** Read-only on every file. No installs, no network, no state-changing commands. Output exactly one file; do not edit anything else.
- **Scope.** Which subtrees to inspect, ordered by priority.
- **Headings.** The exact section structure of the report (so multiple retrieves are comparable).
- **Downstream artifact.** The document this retrieve feeds, and that document's required
  fields, so completeness can be checked mechanically rather than by judgement.
- **Done when.** A precise termination condition, ending with "print the report path and its line count, and nothing else."

A retrieve aims to be a single pass: where a further pass is genuinely required it is run, but
repeated or oversized dispatch for facts an earlier report already contains is waste and is not
issued. The Strategist likewise does not ask the Researcher for a fact a retrieve can return.
None of this narrows the Executor: it is a research actor, and observations, warnings,
disagreements, and proposals outside the prompt's scope are reported rather than suppressed.

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

**Specify decisions, delegate methods.** What the prompt fixes exactly is what was decided in
discussion and cannot be re-derived from the repository: the coordinate frame, the coefficient, the
comparator, which quantity is compared against which, the pass/fail band. What it delegates is how
to achieve that: implementation structure, tooling, the shape of the code. The Executor reasons and
reads its own environment, so a delegated item is stated as the property the result must have plus
what must be reported about it — not as a procedure. Prescribing a procedure invites two failures at
once: the prompt specifies something already in place, and it forecloses the better implementation
the Executor would have found. When a prompt is about to say *how*, it should be saying *what must
be true* and *what must be reported*.

**Cost is part of the design.** Before a prompt is sent, its cells are multiplied out and the known
per-cell cost applied. A battery that a prior version measured at a large wall-clock multiple does
not become cheaper by being requested in more cells. Cells that are genuinely optional are marked
optional and are not simultaneously required by a pass/fail clause elsewhere in the same prompt.

**Amend only for new facts.** Before an amendment is sent, it is checked against a single question:
does this exist because something new was learned, or because something that should have been
decided up front was not? The second case is not amended — it is folded into the next natural
request. A stream of corrective amendments is a symptom of thin design, and each one costs the
Executor a context switch mid-task.

**Standing prompts.** A prompt may define work that begins on a stated trigger. Autonomy is
session-bound: if the Executor session ends at a checkpoint, resumption requires a
Researcher-relayed resume amendment; continuation across sessions is never assumed. Long
autonomous chains schedule their own completion monitors and state the next check time in
every checkpoint message. Autonomy always ends at the prompt's final milestone; no runs
beyond the prompt.

**Performance criteria.** Any performance pass/fail in a prompt is denominated end-to-end
(stage wall-clock, or stage-share-weighted), never as a component-isolation ratio;
micro-benchmark ratios are diagnostic evidence, not gates.

**Measurement persistence.** Every number reported by a milestone, gate, or probe must be
backed by a persisted artifact (a file); stdout-only measurements are not citable at
close. Probe and gate scripts write their results to an artifact (scratchpad JSON at
minimum) as part of passing.

**Invasive code + long GPU in one dispatch.** Splitting them across a session boundary is the default and needs no re-approval; the ordering constraint (code green before measurement) is kept, and the split is reported, not escalated. The split manages session risk, not parallelism: independent light work runs alongside either half whenever resources allow.

### 3.3 Mission prompt — objective and invariants, method delegated

Used when the objective is settled but the method is not. Format:

- **Objective** (what must be determined or achieved, not how).
- **Invariants.** The properties any admissible method must satisfy. These are the mission.
- **Stop-and-report conditions.** Including the case where the objective cannot be met soundly.
- **Governance boundary.** What may be created, what may not be touched.
- **Candidates (optional), marked as such.** A Strategist suggestion is a starting point, never an instruction.

A recorded departure from a candidate is a better outcome than following it; an unrecorded departure is not, because it makes the result unattributable. The build-log therefore carries the approach taken **and the approaches rejected, with the reason for each**.

A mission prompt is chosen over an execute prompt when the Strategist's method rests on a premise the repository can refute — the Executor reads the code and the data, and a fixed procedure forecloses that check.

### 3.4 Prompt authoring authority

The Strategist authors every prompt. The Researcher reviews and approves before the Executor sees it. The Executor must not invent strategy from an under-specified prompt; if the scope is ambiguous in a way that affects results, the Executor stops and asks.

**Prompts are written on request.** The Strategist produces a prompt when the Researcher asks for one. Analysis, review, and reporting are answered as such; an unrequested prompt appended to them pre-empts a decision that is the Researcher's.

**A prompt is presented with its intent.** Every prompt is preceded by a short statement of what it will cause to happen. The prompt addresses the Executor; what the Researcher approves is the effect, and an effect stated only inside the prompt body cannot be reviewed.

**Corrections are amendments.** A change to a dispatched prompt is issued as an amendment stating what is added or withdrawn — never as a rewritten prompt the Executor must diff against the one already in hand.

---

## 4. Decision brief format — Strategist → Researcher

When a decision is required, the Strategist delivers a four-part brief, never longer than one screen:

1. **Problem.** What the decision is about, with the evidence (data / quote / observation).
2. **Options.** Two or three concrete alternatives. Each option is fully specified — config keys, file changes, expected effect.
3. **Trade-offs.** For each option, what is gained and what is lost. Side effects, risks, computational cost.
4. **Recommendation.** One option, with a one-sentence justification.

The Strategist's `00_constitution` §4 problem-analysis discipline (problem / mechanism / fix / trade-off) applies inside §1 and §3. Long prose analyses without this structure are not decision briefs — they are background documents.

---

**Answer-first (all Strategist → Researcher communication).** A yes/no question is answered yes/no first, unconditionally; then the direct answer as asked; then only as needed: result (numbers, CIs) → cause → solution → interpretation, under explicit headings. Chronological narration is banned from Researcher-facing reports (it belongs in build-logs); key numbers appear first; implications are stated outright; multi-aspect versions are reported per aspect.

## 5. Conflict resolution

When sources disagree, follow `00_constitution` §2. Specifically:

- **Researcher instruction vs. Constitution / PROHIBITIONS.** The actor does not silently comply or refuse; it reports the conflict and requests re-confirmation. Once re-confirmed, the instruction takes precedence.
- **Strategist memory vs. code or recorded data.** The code and the data win. The Strategist verifies before reporting a number.
- **Two retrieve reports conflict.** The more recent source wins; if uncertainty remains, a new retrieve is dispatched.
- **Eval result vs. hypothesis.** The hypothesis is rejected. A prompt does **not** re-interpret the result to preserve the hypothesis; the result is reported as-is and the next version's hypothesis is revised accordingly.
- **A number is quoted with its pool.** Two figures measured on different pools are not
  compared, averaged, or substituted for one another, and a figure from a non-canonical pool
  should name that pool where it is quoted. Pools do change over a lineage, so this is a
  labelling obligation rather than a prohibition: the requirement is that the reader can tell
  which pool a number came from, not that the pool never moves.
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
- The whole `data/runs/` subtree, in-flight and closed alike; only what is copied into
  `data/secured_data/` is committed. The `.gitignore` pattern ignores `data/*`, so a new
  file under `data/secured_data/` must be force-added and the staging confirmed
  (`05_code` §4).
- TensorBoard event files inside `data/<run_id>/tensorboard/` — large and redundant with
  the CSVs that are secured at close.

### 6.3 secured_data conventions

**What `secured_data/` is for.** It holds the artifacts behind SOTA claims, so that a claim in
`ledger.md` can be checked against the checkpoint that produced it. Two rules follow, and the
second is the one that is checked:

- A snapshot is created when a version's run is that system's SOTA at close. A later version may
  supersede it; the snapshot stays. `secured_data/` is therefore a record of what was SOTA when,
  not only of what is SOTA now, and a version holding no current bold row may legitimately hold a
  snapshot.
- **Every current bold row's adopted checkpoint is present in `secured_data/` and committed.**
  This direction admits no exception: a bold row whose checkpoint is not in the repository is a
  claim the repository cannot back. The presence check runs at every close (§2.5 item 3) and
  covers every system's bold row, not only the closing version's.

- **Pools.** Generated once via `src/eval/build_pools.py` and committed; they are
  immutable thereafter. Regenerating them is a major-version event because `cps`
  comparisons across versions assume identical pools. **A pool is committed before any number
  measured on it is reported**: the pool is what makes two numbers comparable, so an
  uncommitted pool makes every comparison drawn on it unverifiable. Pools are small; the
  size argument that gates checkpoints does not apply to them.
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

Confirm: it was requested — its intent is stated — unanswered items are carried forward — which type (retrieve / execute / mission) — scope — safeguards — done-when (retrieve), pass/fail (execute), or invariants and stop conditions (mission) — reference patterns. If any are missing, fill before sending.

### 7.3 Before closing a version

Confirm: §2.5 `results.md` drafted — multi-seed aggregate written (or single-seed status stated) — secured snapshot copied with `ADOPTED.md` — `index.md` and `ledger.md` updated (including SOTA bold per §2.4) — git status reviewed. Then §2.6 push.

### 7.4 Ending a session

Confirm: everything decided is in a tracked file (prompts, briefs, results), and any code in flight is either committed or explicitly noted as work-in-progress in `docs/index.md`. Anything not in a tracked file does not exist for the next session.

---

## 8. Reporting and recovery discipline

**No mechanism before the record.** A discrepancy gets no proposed mechanism until the recorded values are re-read from their sources (ledger row, results document, persisted artifact) and compared: read-back → agree/disagree → the differing per-episode set → stop. Mechanisms come only after that set is seen.

**Disk-verified reporting (mandatory).** Every number the Executor reports about a run —
monitoring updates, termination facts, peaks, current steps — is read from the run's
on-disk artifacts at reporting time and cited as such: the run directory, the verbatim
`eval_metrics.csv` / `metrics.csv` row, the process start time, and an arithmetic
consistency check (elapsed $\times$ measured rate $\approx$ reported step). A number
that does not appear on disk is not reported. Recovery and resume prompts re-verify the
run state against disk before recording any termination fact; framing supplied from
outside the disk (a step count, a peak, a "terminated at") is treated as unverified until
the artifacts confirm it.

**Completion is evidenced by its artifact.** "Fixed", "running", "written" is recorded only with
the artifact's path and mtime. A repair proves its artifact on disk before the run that depends on
it starts. Process liveness is judged by the actual child PID, never by a wrapper.

**A causal claim carries its control.** A statement that one change produced one outcome is
reported with the comparison that isolates it, and the conclusion sits beside whatever in the same
data would contradict it. Where no control is available the claim is recorded as unattributed —
this is a labelling obligation, not a bar on reporting: an observation without a control is still
reported, as an observation.

**External-interruption handling (standing decision, "Option A").** When a training run
is killed by an external event (session teardown, machine contention) rather than by an
abort rule: the latest cadence checkpoint stands in for `final.pt`; the build-log and the
ledger row's `eval_source` note record `terminated @<disk-verified step> (external
interrupt)`; and the standard close-out (re-selection over all cadence checkpoints,
full-pool eval of the selected best) proceeds on the surviving artifacts. Optimizer state
is never spliced to resume a partially lost run; if the full budget is wanted, the run is
relaunched fresh and the killed run is kept where it was written, with its termination
note in the build-log and the ledger row; nothing is relocated. A `status.json` left stale
by a hard kill is recorded as stale, never edited.

**Parallel low-priority work.** Analysis or evaluation may run alongside a live training
run only under all of: `nice -n 19` with CPU-core pinning away from the trainer's cores;
GPU batch sizes capped to leave $\ge 2$ GB VRAM headroom (halve-and-retry on OOM — never
touch the training process); read-only access to the live run directory, with all outputs
written elsewhere; and a measured throughput guard — if the trainer's step rate degrades
more than 10% from its pre-launch window, the parallel work stops and reports.