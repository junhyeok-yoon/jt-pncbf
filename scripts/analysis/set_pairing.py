"""v2.7.4 step 3 — value/JT set-pairing reconstructability report (NO moves, read-only).

Walks every JT run under data/runs/ and determines whether its saved config records the value-init source,
which field carried it (the name differs across versions), and resolves the value run. Writes
docs/versions/v2.7.4/set_pairing.md with one line per JT run and the totals, and states whether a migration
of existing runs into set folders is worth doing given the coverage.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs"
OUT = REPO / "docs" / "versions" / "v2.7.4" / "set_pairing.md"
LEDGER = REPO / "docs" / "ledger.md"

# candidate config locations that have carried the value-init link, newest field first
FIELDS = [
    ("run.value_init_run_id", lambda c: (c.get("run", {}) or {}).get("value_init_run_id")),
    ("training.jt.value_init_ckpt", lambda c: ((c.get("training", {}) or {}).get("jt", {}) or {}).get("value_init_ckpt")),
    ("training.jt.pi_init_ckpt", lambda c: ((c.get("training", {}) or {}).get("jt", {}) or {}).get("pi_init_ckpt")),
]


def _run_id_from_ckpt_path(p: str) -> str | None:
    # .../<run_id>/checkpoints/best.pt  -> run_id
    m = re.search(r"([^/]+__\d{8}-\d{6}__seed\d+)", str(p))
    return m.group(1) if m else None


def _jt_runs():
    out = []
    for cfgp in RUNS.glob("**/config.yaml"):
        try:
            c = yaml.safe_load(cfgp.read_text()) or {}
        except Exception:
            c = {}
        if (c.get("run", {}) or {}).get("framework") == "jt_pncbf":
            out.append((cfgp.parent, c))
    return sorted(out, key=lambda t: t[0].name)


def main() -> int:
    ledger = LEDGER.read_text() if LEDGER.exists() else ""
    rows, disk, phase_only, none = [], 0, 0, 0
    field_first_version: dict[str, str] = {}
    for run, cfg in _jt_runs():
        run_id = run.name
        ver = (cfg.get("run", {}) or {}).get("version", "?")
        carried, value_id, raw = None, None, None
        for name, get in FIELDS:
            v = get(cfg)
            if v:
                carried, raw = name, v
                value_id = v if name.endswith("run_id") else _run_id_from_ckpt_path(v)
                break
        if value_id:
            # is that value run present on disk (resolvable) or only citable from a phase report / ledger?
            from src.common.run_layout import locate_run
            on_disk = locate_run(REPO / "data", value_id) is not None
            if on_disk:
                disk += 1; status = f"RESOLVED -> {value_id} (disk; field {carried})"
            else:
                cited = value_id in ledger
                phase_only += 1; status = f"CITED-ONLY -> {value_id} ({'ledger/report' if cited else 'name only'}; field {carried}, not on disk)"
            field_first_version.setdefault(carried, ver)
        else:
            none += 1; status = "UNPAIRED (no value-init field in config — standalone or pre-link run)"
        rows.append((run_id, ver, status, str(run.relative_to(REPO))))

    total = len(rows)
    lines = ["# v2.7.4 — value/JT set-pairing reconstructability (read-only; NO runs moved)\n",
             f"{total} JT runs under data/runs/. Each line: run_id, version, resolution, path.",
             "Value-init field precedence checked: run.value_init_run_id (new, migration step 2) -> "
             "training.jt.value_init_ckpt (path, v2.6.0+) -> training.jt.pi_init_ckpt.\n",
             "| jt run_id | ver | resolution | path |", "|---|---|---|---|"]
    for rid, ver, status, path in rows:
        lines.append(f"| {rid} | {ver} | {status} | {path} |")
    lines += ["\n## Totals",
              f"- JT runs total: {total}",
              f"- pair recoverable FROM DISK (value run present): {disk}",
              f"- pair CITED-ONLY (value_init recorded but value run not on disk — recoverable only from a "
              f"phase report/ledger): {phase_only}",
              f"- NOT recoverable at all (no value-init field): {none}"]
    if field_first_version:
        lines.append("\n## Which field carried the link, earliest version seen")
        for f, v in sorted(field_first_version.items()):
            lines.append(f"- {f}: first seen {v}")
    lines += ["\n## Is a migration of existing runs worth doing?",
              f"Coverage: {disk}/{total} pairs are reconstructable directly from disk (config field + the value "
              f"run present), {phase_only} are cited-only, {none} unpaired. A structural migration into set "
              f"folders is {'LOW value' if disk <= 1 else 'worth considering'} for the existing corpus: the "
              f"pairing is already machine-readable in config for the runs that have it, so the folder move "
              f"would add legibility but recovers no information not already on disk. (Step 4 remains "
              f"UNAUTHORIZED; this is a recommendation, not an action.)"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {total} JT runs; disk={disk} cited-only={phase_only} none={none}")
    print("fields seen:", field_first_version)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
