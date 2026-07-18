# v2.7.0 — ADOPTED-COLLECTOR REFERENCE (seed 42, quadrotor_planar) — NOT the SOTA headline

Secured snapshot of the **v2.7.0 iteration-5 continuing-batch collector (Track A)** result. This is the
**adopted default-collector reference**, kept for the record and for going-forward reproducibility. **It is NOT
the v2.7.0 SOTA headline** — that is iteration-1 (`../seed42/`, cps 0.8232, SOTA-bold in the ledger). Prepared
per `06_workflow §2.5`.

## Run identity (value → JT chain, continuing collector)
- **JT run** (this snapshot): `v2.7.0__20260718-000752__seed42`, best.pt step **48000** (sha8 `3b27d691`),
  collector=continuing, bptt_T=30, value-init = the continuing G2 value.
- **value (G2) run** (`value_run/`): `v2.7.0__20260717-232552__seed42`, oc V̂ best.pt step **39000**
  (sha8 `bacdc3ae`), in-loop best cps 0.5174, collector=continuing. Full artifacts also in `data/previous_runs/`.

## Result (full pool n2000 seed23456, canonical `evaluate()` batch-2000)
- cps **0.8343** [0.8038, 0.8632]; reach 0.9425; collision **0.0410** [0.0325, 0.0500]; oob 0.0000;
  stuck 0.0010; timeout 0.0155; infeasibility 0.0583. Insertion cps: lqr −0.4055 / frozen 0.7955 / live 0.8008.
- **H-nondegradation CONFIRMED** — both metrics CI-overlap iter-1; **NOT SOTA-separated** (iter-1 keeps bold).
- **H-throughput NOT MET** — offline episodes/hour 948,101 vs legacy 1,004,025 = **0.94×** (P-D); wall-clock
  value 29.7 vs 26.4 min (1.13×), JT 230.0 vs 227.6 min (1.01×) = +2% pipeline.
- Pre-JT (G3): cps 0.5191 [0.4735, 0.5633] / collision 0.0950 — CI-separably weaker than iter-1's pre-JT,
  fully recovered by JT co-training.
- **Adoption grounds** = label provenance at ~equal cost, not throughput: hover fraction 0.145 vs ~0.70;
  horizon-200 real per-episode labels replace the 100-step bootstrap chain; all semantics gates green
  (bit-parity, segment isolation, label recompute 0.00e+00; collector tests 5/5; suite 139).
  **`collector=continuing` is the default configuration going forward** (legacy retained for ablation/parity).

## Pinned SHA-256
| artifact | sha256 |
|---|---|
| `checkpoints/best.pt` (JT @48000) | `3b27d691bb0ac8bb927846283b59bd7b35a687ca03bd5f784781c72ec5a815ec` |
| `value_run/checkpoints/best.pt` (G2 value @39000) | `bacdc3aed0efa136247af21975f9a16c3c6c6f842c4df38e011cd21e434a1260` |
| eval pool full `eval_full_quadrotor-planar_n2000_seed23456.pkl` | `92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531` |
| eval pool inloop `eval_inloop_quadrotor-planar_n500_seed12345.pkl` | `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19` |

## Snapshot contents (this directory)
- JT run: `checkpoints/{best.pt, final.pt}`, `config.yaml`, `eval_metrics.csv`, `eval_episodes.csv`,
  `pool_manifest.json`, `status.json`, `git_commit.txt`, `figures/*` (incl. `residual_mechanism/` authority +
  entry-anatomy CSVs).
- `value_run/`: `checkpoints/best.pt`, `config.yaml`, `status.json`, `eval_metrics.csv`, `eval_episodes.csv`,
  `pool_manifest.json`.
- `report.md` = the authored close verdict `docs/versions/v2.7.0_results.md`.
- **Excluded** (per instruction / §7.5): `metrics.csv`, TensorBoard event files.

## Notes
- **NOT SOTA-bolded in the ledger** (ledger L120 plain; only L115 iter-1 is bold).
- Residual-collision counts code-path-dependent: RATE 0.0410 = 82/2000 (canonical); P3 manual-loop 83; agreed 78.
- Full mechanism diagnostics: `docs/versions/v2.7.0/diag_iter5_mechanism.md`, `diag_iter5_health.md`,
  `retrieve_close_facts.md`.
