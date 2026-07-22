"""v2.7.4 Section B — retroactive value/JT set migration for data/runs/v2.6.0 .. v2.7.4 (B1 plan; B2 execute).

Groups each top-level run under the in-scope version folders into its value/JT set folder (one level down),
pairing STRICTLY from run.value_init_run_id or training.jt.value_init_ckpt — never timestamp adjacency,
never pi_init_ckpt. A value run (and any run that does not resolve a same-version value-init) anchors its
own set set__<its ts>__seed<N>; a run whose value-init resolves to a SAME-VERSION value run joins that
value run's set; a run whose value-init is a DIFFERENT version stays in its own version, in its own set
(separated from the cross-version parent, which is recorded). No run_id changes; rename within one filesystem.

--plan     writes docs/set_migration_plan.tsv and prints the per-version + per-set summary; MOVES NOTHING.
--execute  (B2, only after approval) moves one dir at a time, skipping any held open by a live process.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RUNS = DATA / "runs"
PLAN = REPO / "docs" / "set_migration_plan.tsv"
SCOPE_LO, SCOPE_HI = (2, 6, 0), (2, 7, 4)


def _vt(v: str):
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


def _rid(p: str) -> str | None:
    m = re.search(r"([^/]+__\d{8}-\d{6}__seed\d+)", str(p))
    return m.group(1) if m else None


def _ts_seed(run_id: str):
    ts = re.search(r"(\d{8}-\d{6})", run_id)
    sd = re.search(r"seed(\d+)", run_id)
    return (ts.group(1) if ts else None), (sd.group(1) if sd else None)


def _scope_runs():
    """top-level run dirs under data/runs/<version>/ for in-scope versions (not nested, not _unversioned)."""
    out = []
    for vdir in sorted(RUNS.iterdir()):
        vt = _vt(vdir.name)
        if not vdir.is_dir() or vt is None or not (SCOPE_LO <= vt <= SCOPE_HI):
            continue
        for run in sorted(vdir.iterdir()):
            if run.is_dir() and (run / "config.yaml").exists():
                out.append(run)
    return out


def _value_init(cfg) -> tuple[str | None, str | None]:
    r = cfg.get("run", {}) or {}
    j = (cfg.get("training", {}) or {}).get("jt", {}) or {}
    if r.get("value_init_run_id"):
        return r["value_init_run_id"], "run.value_init_run_id"
    if j.get("value_init_ckpt"):
        return _rid(j["value_init_ckpt"]), "training.jt.value_init_ckpt"
    return None, None


def build_plan():
    runs = {}                        # run_id -> (version, path, cfg)
    for run in _scope_runs():
        try:
            cfg = yaml.safe_load((run / "config.yaml").read_text()) or {}
        except Exception:
            cfg = {}
        runs[run.name] = ((cfg.get("run", {}) or {}).get("version", run.parent.name), run, cfg)
    rows = []                        # (old, new, field, set_name, version, cross_parent_or_None)
    for rid, (ver, path, cfg) in runs.items():
        ts, seed = _ts_seed(rid)
        vi, field = _value_init(cfg)
        cross = None
        if vi and vi in runs:
            pver, ppath, _ = runs[vi]
            if pver == ver:
                pts, pseed = _ts_seed(vi)
                set_name = f"set__{pts}__seed{pseed}"; justify = f"{field} -> {vi} (same-version)"
            else:
                set_name = f"set__{ts}__seed{seed}"; justify = f"{field} -> {vi} (CROSS-VERSION {pver}); own set"
                cross = vi
        else:
            set_name = f"set__{ts}__seed{seed}"
            justify = "value run / own single-member set" if not vi else f"{field} -> {vi} (unresolved); own set"
        new = RUNS / ver / set_name / rid
        rows.append((path.resolve(), new.resolve(), justify, set_name, ver, cross))
    return rows


def _held_open(run: Path) -> bool:
    sj = run / "status.json"
    if not sj.exists():
        return False
    try:
        return (json.loads(sj.read_text()).get("phase") == "training"
                and (time.time() - sj.stat().st_mtime) < 600)
    except Exception:
        return False


def cmd_plan():
    import os
    from collections import defaultdict
    rows = build_plan()
    RUNS.mkdir(parents=True, exist_ok=True)
    same_fs = os.stat(DATA).st_dev == os.stat(RUNS).st_dev
    dests = [str(n) for _, n, *_ in rows]
    collisions = sorted({d for d in dests if dests.count(d) > 1})
    with PLAN.open("w") as f:
        f.write("old_abs_path\tnew_abs_path\tset\tversion\tjustification\n")
        for old, new, justify, set_name, ver, cross in rows:
            f.write(f"{old}\t{new}\t{set_name}\t{ver}\t{justify}\n")
    per_ver = defaultdict(lambda: [0, 0])
    sets = defaultdict(list)
    crosses = []
    skipped = 0
    for old, new, justify, set_name, ver, cross in rows:
        kb = 0
        import subprocess
        try:
            kb = int(subprocess.check_output(["du", "-sk", str(old)]).split()[0])
        except Exception:
            pass
        per_ver[ver][0] += 1; per_ver[ver][1] += kb
        sets[(ver, set_name)].append(old.name)
        if cross:
            crosses.append((old.name, cross, ver))
        if _held_open(old):
            skipped += 1
    print(f"PLAN {PLAN}  ({len(rows)} dirs)  same_fs={same_fs}  collisions={collisions or 'NONE'}  held_open_skipped={skipped}")
    for ver in sorted(per_ver, key=_vt):
        n, kb = per_ver[ver]
        print(f"  {ver}: {n} dirs, {kb/1024:.0f} MB")
    print("SETS:")
    for (ver, sn), members in sorted(sets.items()):
        tag = "" if len(members) > 1 else " [single-member]"
        print(f"  {ver}/{sn} ({len(members)}){tag}: {', '.join(members)}")
    print("CROSS-VERSION separations:")
    for child, parent, ver in crosses:
        print(f"  {child} (in {ver}) had value-init parent {parent} in an earlier version -> own set (separated)")
    assert not collisions, collisions


def cmd_execute():
    import os
    rows = build_plan()                                   # rebuilt from CURRENT disk (the authority)
    # optional cross-check against the earlier TSV (same old->new set); disk wins but a mismatch is reported.
    if PLAN.exists():
        tsv = {ln.split("\t")[0]: ln.split("\t")[1] for ln in PLAN.read_text().splitlines()[1:] if "\t" in ln}
        disk = {str(o): str(n) for o, n, *_ in rows}
        if tsv != disk:
            only_tsv = set(tsv) - set(disk); only_disk = set(disk) - set(tsv)
            print(f"NOTE: plan rebuilt from disk differs from TSV — tsv_only={len(only_tsv)} disk_only={len(only_disk)} "
                  f"(disk is authority; proceeding on disk)")
    same_fs = os.stat(DATA).st_dev == os.stat(RUNS).st_dev
    dests = [str(n) for _, n, *_ in rows]
    collisions = sorted({d for d in dests if dests.count(d) > 1})
    if not same_fs or collisions:
        print(f"ABORT pre-flight: same_fs={same_fs} collisions={collisions}"); return 1
    # PRE-FLIGHT: stop-and-report if ANY source is gone or ANY destination already exists, BEFORE any move.
    for old, new, *_ in rows:
        if not old.exists():
            print(f"STOP: source no longer exists: {old}"); return 1
        if new.exists():
            print(f"STOP: destination already exists: {new}"); return 1
    print(f"pre-flight OK: {len(rows)} sources present, {len(rows)} destinations absent, same_fs, no collisions")
    moved, failed = [], []
    for old, new, *_ in rows:
        if _held_open(old):                                # not expected in scope; belt-and-suspenders
            print(f"STOP: source held open by a live process: {old}"); break
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        if new.exists() and not old.exists():
            moved.append((str(old), str(new)))
        else:
            failed.append((str(old), "post-move verify failed")); break
    print(f"moved {len(moved)}, failed {len(failed)}")
    for o, n in moved: print(f"  {o}  ->  {n}")
    for o, w in failed: print("  FAIL", o, w)
    return 0 if not failed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    raise SystemExit(cmd_execute() if a.execute else (cmd_plan() or 0))
