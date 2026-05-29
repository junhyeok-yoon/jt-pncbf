# JT-PNCBF

Self-built research infrastructure for Observation-Conditioned PNCBF and Joint Training
for safe navigation in randomized obstacle environments.

## Documentation

The single source of truth is the documentation site (protocol, version reports, eval
reports, ledger). Build/preview locally:

```bash
pip install mkdocs-material
mkdocs serve            # local preview at http://127.0.0.1:8000
```

On push to `main` (changes under `docs/` or `mkdocs.yml`), GitHub Actions builds and
deploys the site automatically.

## Layout

```
docs/        documentation site source (MkDocs); the SSOT
src/         code (self-built): frozen core + swappable framework learners
scripts/     run entrypoints (train, eval)
data/        run outputs and version snapshots (local; not tracked by git)
```
