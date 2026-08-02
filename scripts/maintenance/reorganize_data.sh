#!/usr/bin/env bash
# data/ reorganization — PROPOSAL script. DRY-RUN BY DEFAULT (prints commands, moves nothing).
#
#   bash scripts/maintenance/reorganize_data.sh                 # dry-run (echo only) — DEFAULT
#   bash scripts/maintenance/reorganize_data.sh --apply         # execute the default SAFE move
#   bash scripts/maintenance/reorganize_data.sh --apply --with-optional   # + archive the unref'd dissection set
#
# Audit + rationale: docs/maintenance/data_reorg_plan.md
#
# WHY plain `mv` (not `git mv`): .gitignore is `data/*` + `!data/secured_data/`, so the lookahead
# runs / diagnostics are git-IGNORED (untracked). `git mv` would fail; plain `mv`
# is correct. Only `data/secured_data/` is tracked, and this script NEVER touches it.
#
# Reference-safety (see plan Part 1): the 20 `data/v2.1.0__*lookahead*` dirs have ZERO references
# in src/ scripts/ docs/. secured_data/ and diagnostics/ are heavily referenced and are NOT moved.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

APPLY=0
WITH_OPTIONAL=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --with-optional) WITH_OPTIONAL=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run() {  # echo in dry-run; execute under --apply
  if [[ "$APPLY" -eq 1 ]]; then echo "+ $*"; "$@"; else echo "[dry-run] $*"; fi
}

guard_not_secured() {  # refuse to ever touch the canonical tracked tree
  case "$1" in
    *data/secured_data/*|data/secured_data*) echo "REFUSING to touch secured_data: $1" >&2; exit 3 ;;
  esac
}

echo "=== data/ reorganization ($([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)) ==="

# ---- Move 1 (DEFAULT, reference-safe): archive the 20 v2.1.0 lookahead eval-only runs ----
# Reference-safe: 0 references in src/ scripts/ docs/ (plan Part 1.3). 29M total.
DEST="data/archive/v2.1.0_lookahead"
run mkdir -p "$DEST"
shopt -s nullglob
moved=0
for d in data/v2.1.0__*lookahead*; do
  [[ -d "$d" ]] || continue
  guard_not_secured "$d"
  base="$(basename "$d")"
  if [[ -e "$DEST/$base" ]]; then
    echo "[skip] already archived: $DEST/$base"   # idempotent
    continue
  fi
  run mv "$d" "$DEST/"     # plain mv: git-ignored path
  moved=$((moved+1))
done
echo "Move 1: $moved lookahead dir(s) -> $DEST  [SAFE: 0 refs]"

# ---- Move 2 (OPTIONAL, --with-optional): archive the UNREFERENCED older dissection set ----
# diagnostics/...failure_dissect...194620 has 0 refs (the cited set is ...194915, 13 refs).
if [[ "$WITH_OPTIONAL" -eq 1 ]]; then
  OLD_DISSECT="data/diagnostics/v2.0.1__20260529-171057__failure_dissect_n2500_seed45678_20260529-194620"
  if [[ -d "$OLD_DISSECT" ]]; then
    guard_not_secured "$OLD_DISSECT"
    run mkdir -p "data/archive/old_dissection"
    run mv "$OLD_DISSECT" "data/archive/old_dissection/"
    echo "Move 2 (optional): old dissection set -> data/archive/old_dissection/  [SAFE: 0 refs]"
  else
    echo "Move 2 (optional): $OLD_DISSECT not present (skip)"
  fi
fi

# ---- NOT performed by this script (documented manual options) ----
echo
echo "NOT moved by this script (see plan Part 2):"
echo "  - data/secured_data/   : canonical + git-tracked + heavily referenced -> NEVER move."
echo "  - data/diagnostics/    : hard-coded by all v2.2.0 scripts + cited in v2.2.0 docs -> keep."
echo "  - (retired) the former data/previous-runs archive was absorbed into data/runs/<version>/ (v2.8.0)"
echo "        and no longer exists; this script no longer references it."
echo
echo "=== done ($([[ $APPLY -eq 1 ]] && echo applied || echo dry-run; )) ==="
