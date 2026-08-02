"""v2.7.4 run index — walk the run directories (new data/runs/ AND the old locations until the migration
lands), read run.framework / run.system / seed from each run's config.yaml, the final step from
eval_metrics.csv (NOT status.json, which under-reports), the best.pt sha8, and the value-init set link, and
emit docs/run_index.md. Reports per-framework counts and names every run whose framework is undetermined.
Disk/CPU only; reads nothing under a lock and moves nothing.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LEDGER = REPO / "docs" / "ledger.md"
OUT = REPO / "docs" / "run_index.md"


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.yaml").exists()


def _run_dirs():
    seen = []
    # new layout: data/runs/<version>/[<set>/]<run_id> — recurse so runs inside set folders are found
    runs_root = DATA / "runs"
    if runs_root.exists():
        seen += sorted({cfg.parent for cfg in runs_root.glob("**/config.yaml") if _is_run_dir(cfg.parent)},
                       key=lambda p: p.name)
    # old top-level: data/<run_id> (the active run not yet migrated)
    seen += [d for d in sorted(DATA.glob("*__*seed*")) if _is_run_dir(d)]
    return seen


def _final_step(run: Path) -> int | None:
    em = run / "eval_metrics.csv"
    if not em.exists():
        return None
    mx = None
    with em.open() as f:
        for row in csv.DictReader(f):
            try:
                s = int(row["step"])
            except (KeyError, ValueError):
                continue
            mx = s if mx is None else max(mx, s)
    return mx


def _sha8(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8] if p.exists() else None


def _new_path(run_id: str) -> Path:
    # actual current location if it exists (may be inside a set folder post-migration), else the canonical
    # version path
    from src.common.run_layout import locate_run, parse_version
    loc = locate_run(DATA, run_id)
    if loc is not None:
        return loc
    v = parse_version(run_id)
    return DATA / "runs" / (v if v else "_unversioned") / run_id


def main() -> int:
    ledger_text = LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""
    rows, per_fw, undetermined = [], {}, []
    for run in _run_dirs():
        run_id = run.name
        try:
            cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        rcfg = cfg.get("run", {}) if isinstance(cfg, dict) else {}
        fw = rcfg.get("framework")
        system = rcfg.get("system")
        seed = rcfg.get("seed")
        vi = rcfg.get("value_init_run_id")
        best_sha = _sha8(run / "checkpoints" / "best.pt")
        in_ledger = run_id in ledger_text
        if not fw:
            undetermined.append(str(run))
        per_fw[fw or "UNDETERMINED"] = per_fw.get(fw or "UNDETERMINED", 0) + 1
        rows.append(dict(old=str(run.relative_to(REPO)), new=str(_new_path(run_id).relative_to(REPO)),
                         framework=fw or "?", system=system or "?", seed=seed if seed is not None else "?",
                         final_step=_final_step(run), best_sha8=best_sha or "?",
                         value_init=vi or "", in_ledger="yes" if in_ledger else "NO"))

    lines = ["# Run index (v2.7.4 migration)\n",
             f"Generated over {len(rows)} run directories (data/runs/ + old data/*__*seed*).",
             "final_step is the max step in eval_metrics.csv (not status.json). in_ledger = run_id appears in docs/ledger.md.\n",
             "| old path | new path | fw | system | seed | final_step | best sha8 | value_init_run_id | in_ledger |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["old"]):
        lines.append(f"| {r['old']} | {r['new']} | {r['framework']} | {r['system']} | {r['seed']} | "
                     f"{r['final_step']} | {r['best_sha8']} | {r['value_init']} | {r['in_ledger']} |")
    lines.append("\n## Counts per framework")
    for fw, n in sorted(per_fw.items()):
        lines.append(f"- {fw}: {n}")
    lines.append(f"\n## Undetermined framework ({len(undetermined)})")
    lines += [f"- {u}" for u in undetermined] or ["- (none)"]
    # ADOPTED.md old->new mapping note (secured, write-forbidden -> PROTOCOL FOLLOW-UP)
    lines.append("\n## PROTOCOL FOLLOW-UP — data/secured_data/v2.7.2/ADOPTED.md run_id mapping")
    adopted = DATA / "secured_data" / "v2.7.2" / "ADOPTED.md"
    if adopted.exists():
        txt = adopted.read_text(encoding="utf-8")
        import re
        ids = sorted(set(re.findall(r"v\d+\.\d+\.\d+__[0-9]{8}-[0-9]{6}__seed\d+", txt)))
        for rid in ids:
            lines.append(f"- {rid}  ->  {_new_path(rid).relative_to(REPO)}  (ADOPTED.md is write-forbidden; Researcher applies at close)")
        if not ids:
            lines.append("- (no run_ids matched in ADOPTED.md)")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} : {len(rows)} runs")
    print("per framework:", per_fw)
    print(f"undetermined framework: {len(undetermined)}")
    for u in undetermined:
        print("  ", u)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
