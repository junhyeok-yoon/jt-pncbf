# secured_data/baselines — version-independent baseline artifacts

This subtree holds **baselines**: runs or evaluation cells whose role is cross-version comparison,
not any single version's axis (`05_code` §3). A baseline is trained or evaluated once, cited by many
versions, and is never filed under a version directory.

Each subdirectory is a **copy**. The originals stay where they were produced so every existing
citation keeps resolving; nothing is moved into here retroactively.

- `ppo/` — the PPO filter-free baseline (has its own checkpoint).
- `backup_cbf/` — the backup-CBF baseline: **filter cells, no checkpoint of its own**. See its
  README for the checkpoint the cells run over, both tracks' cells, and the STATUS line that must
  accompany any citation.
