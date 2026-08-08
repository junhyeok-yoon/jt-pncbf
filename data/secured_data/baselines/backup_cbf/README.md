# backup-CBF baseline — scaffold (awaiting B1/B2 artifacts)

Empty by design. **B1/B2 final artifacts file here BY RULE**; nothing is moved into it retroactively.

- Design + results build-log: `docs/versions/v2.8.3/backup_cbf_b1.md` (and `_b2.md` if B2 launches).
- The B1 subagent's own working directories are NOT to be relocated — this scaffold receives FINAL
  artifacts only, on completion.

**Status line that must accompany every table produced from this baseline, verbatim:**

> hand-designed policy + online rollout certificate (implicit/backup CBF)

It is NOT an analytic closed-form CBF. The analytic column stays empty
(`prop:relh` / `cor:degen-pos` / `prop:box-feas`).

---

## v2.8.3 close — what this baseline actually is (appended 2026-08-08; the scaffold note above stands)

**This baseline has NO checkpoint of its own.** It is a pair of *filter cells* — a hand-designed
nominal plus an online rollout certificate — evaluated over an existing checkpoint. There is no
`set__*/` run directory for either track, and none was ever produced. `backup_cbf`,
`backup_cbf_std` and `backup_cbf_v2` under `data/runs/v2.8.3/` are flat artifact directories.

**STATUS (verbatim, required on every citation):**
hand-designed policy + online rollout certificate (implicit/backup CBF); NOT an analytic closed-form CBF.

**Checkpoint the cells run over** — supplies system, config and (for the `leader` arm) the policy:
`data/runs/v2.8.2/set__20260803-063606__seed42/v2.8.2__jt__20260803-063606__seed42/checkpoints/best.pt`,
sha256[:16] = `c89f9aef0cdb5499`.
(The v2 rows in `artifacts/backup_cbf_v2/` were scored over the **v2.7.6 leader** instead:
`data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/best.pt`,
sha256[:16] = `a5c1e55674ea7b23`.)

**The two cells, verbatim.** Both: `eps = 0.0`, `terminal = rest_penalty`, nominal =
`lqr_action` PD with `kp_pos = 4.0, kd_pos = 3.0`.

| track | T_b | k_d | kp_att | kd_att |
|---|---|---|---|---|
| **Track P** (selection of record, pre-declared max-cps rule) | 30 | 8.0 | 320.0 | 64.0 |
| **Track S** (secondary, post-hoc, disclosed) | 50 | 8.0 | 320.0 | 16.0 |

**Scoring artifacts and their cells.**
- `artifacts/backup_cbf_v2/full_P.json`, `full_S.json` — the v2 rows: n 2000, **eval_batch_size 400**,
  v2.7.6 leader, **no angular-rate terminal**. Track P: nominal_pd cps +0.0284 / coll 0.2060;
  leader cps +0.1156 / coll 0.1890. Track S: nominal_pd cps +0.2070 / coll 0.1215;
  leader cps +0.1097 / coll 0.1045.
- `data/secured_data/v2.8.3/experiments/fullcb_rescore/row__backup_trackP_pd.json` and
  `row__backup_trackS_pd.json` — the v2.8.3 uniform re-score: pool `fullcb` sha8 `3682a4e3`,
  n 2000, **eval_batch_size 2000**, terminal (0.15, 0.3, 0.3), projection dual_solve,
  empty_fallback {kstep, phases 1, k 3}, alpha (2.0, 100.0), gamma_margin 0.0.
  Track P cps −0.0036 / coll 0.2100; Track S cps +0.1926 / coll 0.1245.
  **These are NOT comparable to the v2 rows** — different checkpoint, batch size and terminal; the
  divergence is recorded inside each row as `cell_divergence`.

**Implementation** the cells run through: `impl/filter_backup.py` (copy of `src/common/filter_backup.py`).

**Copies only.** Every original stays in place and every existing citation keeps resolving.
