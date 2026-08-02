"""v2.7.4 run-directory migration: reorganize training runs into data/runs/<version>/<run_id>/.

--plan    : write docs/versions/v2.7.4/run_migration_plan.tsv (old\tnew\tdisposition), MOVE NOTHING, and
            report line count, per-version counts, unversioned list, same-filesystem check, collision check.
--execute : move one directory at a time with rename, verifying each; SKIP active (held-open) runs and
            anything under data/secured_data/. Re-derives the plan (does not trust a stale tsv).

Move set: top-level data/ dirs except {eval, eval_pools, secured_data, runs}. (The former previous-runs
archive was absorbed into data/runs/<version>/ and retired in v2.8.0; it no longer exists.)
Routing: a name matching the canonical run-id structure -> data/runs/<version>/; otherwise ->
data/runs/_unversioned/ with its name unchanged. Never guesses a version from mtime or content.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from src.common.run_layout import parse_version

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RUNS = DATA / "runs"
STAY = {"eval", "eval_pools", "secured_data", "runs"}
PLAN = REPO / "docs" / "versions" / "v2.7.4" / "run_migration_plan.tsv"


def _move_candidates() -> list[Path]:
    out = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or d.name in STAY:
            continue
        out.append(d)
    return out


_ACTIVE_WINDOW_S = 600   # a live trainer updates status.json well within this; interrupted historical runs do not


def _is_active(run: Path) -> bool:
    """A run is HELD OPEN by a live trainer iff its status.json phase is 'training' AND it was written very
    recently. Historical runs that were interrupted (Researcher-terminated) also carry phase='training' but
    have ancient status.json mtimes, so the recency test excludes them; only the live run qualifies."""
    sj = run / "status.json"
    if not sj.exists():
        return False
    try:
        if str(json.loads(sj.read_text()).get("phase", "")) != "training":
            return False
        return (time.time() - sj.stat().st_mtime) < _ACTIVE_WINDOW_S
    except Exception:
        return False


def _dest(run: Path) -> Path:
    v = parse_version(run.name)
    return RUNS / (v if v else "_unversioned") / run.name


def build_plan():
    rows = []
    for run in _move_candidates():
        dest = _dest(run)
        if str(run).startswith(str(DATA / "secured_data")):
            disp = "SKIP-secured"
        elif _is_active(run):
            disp = "SKIP-active(step6)"
        else:
            disp = "move"
        rows.append((run.resolve(), dest.resolve(), disp))
    return rows


def cmd_plan():
    rows = build_plan()
    # same-filesystem check (rename vs copy): compare st_dev of data/ and data/runs parent
    RUNS.mkdir(parents=True, exist_ok=True)
    same_fs = os.stat(DATA).st_dev == os.stat(RUNS).st_dev
    # collision check
    dests = [str(n) for _, n, _ in rows]
    collisions = sorted({d for d in dests if dests.count(d) > 1})
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open("w") as f:
        f.write("old_abs_path\tnew_abs_path\tdisposition\n")
        for old, new, disp in rows:
            f.write(f"{old}\t{new}\t{disp}\n")
    per_version, unversioned = {}, []
    for old, new, disp in rows:
        v = parse_version(old.name) or "_unversioned"
        per_version[v] = per_version.get(v, 0) + 1
        if v == "_unversioned":
            unversioned.append(old.name)
    print(f"PLAN written: {PLAN}  ({len(rows)} directories)")
    print(f"same_filesystem (rename not copy): {same_fs}  (data st_dev == runs st_dev)")
    print(f"destination collisions: {collisions if collisions else 'NONE'}")
    print("per-version counts:")
    for v, n in sorted(per_version.items()):
        print(f"  {v}: {n}")
    print(f"unversioned ({len(unversioned)}):")
    for u in sorted(unversioned):
        print(f"  {u}")
    active = [str(o) for o, _, d in rows if d.startswith("SKIP-active")]
    print(f"active-skip (step 6): {active}")
    assert not collisions, f"destination collisions: {collisions}"


def cmd_execute():
    rows = build_plan()
    moved, skipped, failed = [], [], []
    for old, new, disp in rows:
        if disp != "move":
            skipped.append((str(old), disp)); continue
        if not old.exists():
            skipped.append((str(old), "already-gone")); continue
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            failed.append((str(old), f"dest exists {new}")); continue
        shutil.move(str(old), str(new))
        if new.exists() and not old.exists():
            moved.append((str(old), str(new)))
        else:
            failed.append((str(old), "post-move verify failed")); break
    print(f"moved {len(moved)}, skipped {len(skipped)}, failed {len(failed)}")
    for o, d in failed:
        print("  FAILED", o, d)
    for o, why in skipped:
        print("  skip", o, why)
    return 0 if not failed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if a.execute:
        raise SystemExit(cmd_execute())
    cmd_plan()
