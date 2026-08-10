#!/usr/bin/env python3
"""Shape check for docs/ledger.md.

The ledger is one Markdown table plus trailing Note paragraphs. Three shape defects have
been repaired by hand so far (orphaned rows, blank cps_v2, row order inside a version
block); this script turns the shape into a machine-verified invariant so the fourth one
is caught by a command instead of by reading.

It reports; it never edits. Exit 0 = clean, exit 1 = at least one violation.

Checks, in order:
  1. exactly one Markdown table; every data row inside it; every Note paragraph after it
  2. every row parses to the header's column count
  3. no cell holds a literal `|`; verdict cells are single-line
  4. each version's rows form one contiguous block
  5. within a block, rows ascend by the execution-time key
       (a) parent run-id timestamp `YYYYMMDD-HHMMSS`, else
       (b) earliest mtime among artifact files the eval_source cell names, else
       (c) the date cell -- a bare date sorts last within its day
     mtime lookups that fail on a given machine degrade to (c) rather than erroring
  6. cps and cps_v2 are both present or both blank ('-' = explicitly unavailable and
     pairs with anything; a number opposite a blank is the defect)
  7. where all three coll_* cells are present, they sum to `collision` within 1e-9
  8. parent resolves on disk or is `-` (unresolved -> warning, so the script is usable
     on a partial checkout)
  9. at most one bold row per system, and every bold row carries a resolving parent

Legacy baselines: rules 4 and 5 were installed over a table that already carried
violations in closed version blocks that a later dispatch may not rewrite. Those exact
blocks are listed below and report as warnings; every other block is checked hard, so a
newly introduced defect fails.

Usage:  python scripts/check_ledger.py [path/to/ledger.md]
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO_ROOT, "docs", "ledger.md")
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")

HEADER_PREFIX = "| version |"
# Column indices are resolved FROM THE HEADER by name (see resolve_columns), not hard-coded, so
# inserting a column cannot silently shift every downstream read. These are the names required.
REQUIRED_COLUMNS = (
    "version", "system", "date", "parent", "alias", "cps_v2", "eval_source",
    "collision", "coll_obstacle", "coll_band_lower", "coll_band_upper", "cps", "verdict",
)
COL: dict[str, int] = {}

# Pool stems a row's eval_source may name. Used by rule 10: an alias may repeat within a version
# block when the SAME run/cell was scored on DIFFERENT pools -- that is one run owning several rows,
# not a duplicate -- so uniqueness is checked on (alias, pool stem).
POOL_STEMS = ("fullcb", "fullscr41", "fullscr40", "inloopv2", "navconescr40", "navcone", "mixed")

TS_RE = re.compile(r"(\d{8})-(\d{6})")
ARTIFACT_RE = re.compile(r"[A-Za-z0-9_.\-/]+\.(?:json|npz|npy|pt)")
RUNID_RE = re.compile(r"[A-Za-z0-9_.\-]*__[A-Za-z0-9_.\-]+")

# Version blocks whose rows were already out of contiguity / order when the check was
# installed (2026-08-08). They are out of the installing dispatch's write scope; a later
# dispatch that repairs one should delete it from the list.
LEGACY_SPLIT_VERSIONS = {
    "v2.0.1", "v2.3.0", "v2.7.3", "v2.7.4", "v2.7.6", "v2.8.0", "v2.8.1", "v2.8.2",
}
LEGACY_UNORDERED_VERSIONS = {"v2.0.1", "v2.8.0", "v2.8.2"}


def unbold(text: str) -> str:
    """Strip the `**` a bolded cell wears, plus surrounding space."""
    out = text.strip()
    while out.startswith("**") and out.endswith("**") and len(out) >= 4:
        out = out[2:-2].strip()
    return out.strip("*").strip() if out.startswith("*") or out.endswith("*") else out


def split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def is_bold_row(cells: list[str]) -> bool:
    ver = cells[COL["version"]].strip()
    return ver.startswith("**") and ver.endswith("**")


def parse_date_cell(raw: str) -> dt.datetime | None:
    """Full timestamps keep their time; a bare date sorts last within its day."""
    text = unbold(raw)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        day = dt.datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return day + dt.timedelta(hours=23, minutes=59, seconds=59)


def row_key(cells: list[str]) -> tuple[dt.datetime | None, str]:
    """SCORING-TIME key: the earliest mtime among the artifact files the row's eval_source names.

    THE STANDING RULE (2026-08-09): the ordering key is SCORING time; `parent` is LINEAGE ONLY.
    The previous key preferred the parent run-id timestamp, which conflated two different times --
    an eval-only row inherits the SCORED CHECKPOINT's run timestamp, so a cell scored today sorted
    among rows from the day the checkpoint was trained. Artifact mtime is therefore consulted FIRST.
    The parent timestamp survives only as a fallback for rows that name no resolvable artifact
    (chiefly the legacy blocks), and the date cell as a last resort.
    """
    mtimes = []
    for token in ARTIFACT_RE.findall(cells[COL["eval_source"]]):
        if "/" not in token:
            continue
        path = token if os.path.isabs(token) else os.path.join(REPO_ROOT, token)
        try:
            mtimes.append(dt.datetime.fromtimestamp(os.path.getmtime(path)))
        except OSError:
            continue  # absent on this machine -> fall through to lineage/date
    if mtimes:
        return min(mtimes), "artifact-mtime"

    parent = unbold(cells[COL["parent"]])
    match = TS_RE.search(parent)
    if match:
        try:
            return dt.datetime.strptime(match.group(0), "%Y%m%d-%H%M%S"), "parent-ts(fallback)"
        except ValueError:
            pass

    return parse_date_cell(cells[COL["date"]]), "date"


def pool_stem(cells: list[str]) -> str:
    """The pool a row was scored on, read off eval_source; '?' when none is named."""
    src = unbold(cells[COL["eval_source"]])
    for stem in POOL_STEMS:
        if stem in src:
            return stem
    return "?"


def resolve_columns(header_cells: list[str]) -> list[str]:
    """Fill COL from the header by NAME. Returns a list of problems (empty when clean)."""
    COL.clear()
    names = [unbold(c).strip() for c in header_cells]
    problems = []
    for want in REQUIRED_COLUMNS:
        if want in names:
            COL[want] = names.index(want)
        else:
            problems.append(want)
    return problems


def run_index() -> set[str]:
    """Directory names under data/runs, three levels deep -- run-ids live at that depth."""
    names: set[str] = set()
    if not os.path.isdir(RUNS_DIR):
        return names
    for current, dirs, _files in os.walk(RUNS_DIR):
        names.update(dirs)
        if current[len(RUNS_DIR):].count(os.sep) >= 3:
            dirs[:] = []
    return names


def parent_resolves(parent_cell: str, index: set[str]) -> bool:
    """`-` passes; otherwise a run-id or path inside the cell must exist on disk."""
    parent = unbold(parent_cell).rstrip(".")
    if parent in ("", "-"):
        return True
    candidates = [parent]
    candidates += RUNID_RE.findall(parent)
    candidates += [t for t in re.split(r"[\s,;()·]+", parent) if "/" in t]
    for cand in candidates:
        cand = cand.strip().strip(".,;")
        if not cand:
            continue
        if cand in index:
            return True
        if os.path.exists(os.path.join(REPO_ROOT, cand)):
            return True
        if os.path.exists(os.path.join(RUNS_DIR, cand)):
            return True
    return False


def cell_state(raw: str) -> str:
    text = unbold(raw)
    if text == "":
        return "BLANK"
    if text == "-":
        return "NA"
    return "VALUE"


def as_float(raw: str) -> float | None:
    text = unbold(raw)
    try:
        return float(text)
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT_LEDGER
    violations: list[str] = []
    warnings: list[str] = []

    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    # ---- rule 1: exactly one table, rows inside it, Notes after it -------------------
    pipe_lines = [i for i, l in enumerate(lines) if l.lstrip().startswith("|")]
    if not pipe_lines:
        print("VIOLATION [1] no Markdown table found")
        print("summary: rows=0 blocks=0 bolds=0 violations=1 warnings=0")
        return 1

    header_idx = next(
        (i for i, l in enumerate(lines) if l.lstrip().startswith(HEADER_PREFIX)), None
    )
    if header_idx is None:
        print("VIOLATION [1] no header row starting %r" % HEADER_PREFIX)
        print("summary: rows=0 blocks=0 bolds=0 violations=1 warnings=0")
        return 1

    start, end = pipe_lines[0], pipe_lines[-1]
    gaps = sorted(set(range(start, end + 1)) - set(pipe_lines))
    if gaps:
        violations.append(
            "[1] table is not contiguous: %d non-table line(s) inside it, first at L%d"
            % (len(gaps), gaps[0] + 1)
        )
    if header_idx != start:
        violations.append(
            "[1] more than one table: rows at L%d precede the header at L%d"
            % (start + 1, header_idx + 1)
        )
    for i, line in enumerate(lines):
        if re.match(r"\s*Note\b", line) and i < end:
            violations.append("[1] Note paragraph at L%d sits before the table end" % (i + 1))

    header_cells = split_row(lines[header_idx])
    ncols = len(header_cells)
    missing = resolve_columns(header_cells)
    if missing:
        print("VIOLATION [2] header is missing required column(s): %s" % ", ".join(missing))
        print("summary: rows=0 blocks=0 bolds=0 violations=1 warnings=0")
        return 1
    sep_idx = header_idx + 1

    rows = []  # (line_no, cells)
    for i in pipe_lines:
        if i in (header_idx, sep_idx):
            continue
        rows.append((i + 1, split_row(lines[i])))

    # ---- rules 2 and 3: column count, no literal pipes, single-line verdicts ---------
    for line_no, cells in rows:
        if len(cells) != ncols:
            violations.append(
                "[2] L%d parses to %d columns, header has %d" % (line_no, len(cells), ncols)
            )
        raw = lines[line_no - 1]
        if "\\|" in raw:
            violations.append("[3] L%d contains an escaped literal pipe inside a cell" % line_no)
        if len(cells) == ncols:
            verdict = cells[COL["verdict"]]
            if "<br" in verdict or "\\n" in verdict:
                violations.append("[3] L%d verdict cell is not single-line" % line_no)

    usable = [(ln, c) for ln, c in rows if len(c) == ncols]

    # ---- rule 4: one contiguous block per version -----------------------------------
    blocks: list[tuple[str, list[tuple[int, list[str]]]]] = []
    for line_no, cells in usable:
        version = unbold(cells[COL["version"]])
        if not blocks or blocks[-1][0] != version:
            blocks.append((version, []))
        blocks[-1][1].append((line_no, cells))

    seen: dict[str, list[tuple[int, int]]] = {}
    for version, items in blocks:
        seen.setdefault(version, []).append((items[0][0], items[-1][0]))
    for version, spans in sorted(seen.items()):
        if len(spans) > 1:
            msg = "[4] version %s is split across %d blocks: %s" % (
                version, len(spans), ", ".join("L%d-L%d" % s for s in spans)
            )
            (warnings if version in LEGACY_SPLIT_VERSIONS else violations).append(
                msg + (" (legacy baseline)" if version in LEGACY_SPLIT_VERSIONS else "")
            )

    # ---- rule 5: ascending execution-time key within each block ----------------------
    for version, items in blocks:
        previous = None
        for line_no, cells in items:
            key, source = row_key(cells)
            if key is None:
                violations.append(
                    "[5] L%d has no derivable order key (date cell %r unparseable)"
                    % (line_no, unbold(cells[COL["date"]]))
                )
                continue
            if previous is not None and key < previous[0]:
                msg = "[5] L%d out of order in block %s: key %s (%s) < previous L%d key %s" % (
                    line_no, version, key, source, previous[1], previous[0]
                )
                if version in LEGACY_UNORDERED_VERSIONS:
                    warnings.append(msg + " (legacy baseline)")
                else:
                    violations.append(msg)
            previous = (key, line_no)

    # ---- rule 10: alias uniqueness on (alias, pool stem) within a version block -------
    # A run scored on several pools LEGITIMATELY owns several rows, so a bare alias repeats by
    # design; what must not repeat is the same alias on the same pool.
    for version, items in blocks:
        seen: dict[tuple[str, str], int] = {}
        for line_no, cells in items:
            alias = unbold(cells[COL["alias"]]).strip()
            if alias in ("", "-"):
                continue
            key = (alias, pool_stem(cells))
            if key in seen:
                violations.append(
                    "[10] L%d duplicate alias %r on pool %r in block %s (also L%d)"
                    % (line_no, key[0], key[1], version, seen[key])
                )
            else:
                seen[key] = line_no

    # ---- rules 6 and 7: cps pairing, collision decomposition ------------------------
    for line_no, cells in usable:
        states = {cell_state(cells[COL["cps"]]), cell_state(cells[COL["cps_v2"]])}
        if states == {"BLANK", "VALUE"}:
            violations.append(
                "[6] L%d cps=%r and cps_v2=%r: one carries a value, the other is blank"
                % (line_no, unbold(cells[COL["cps"]]), unbold(cells[COL["cps_v2"]]))
            )
        parts = [as_float(cells[COL[k]]) for k in
                 ("coll_obstacle", "coll_band_lower", "coll_band_upper")]
        total = as_float(cells[COL["collision"]])
        if all(p is not None for p in parts) and total is not None:
            if abs(sum(parts) - total) > 1e-9:
                violations.append(
                    "[7] L%d coll_* sum %.10g != collision %.10g" % (line_no, sum(parts), total)
                )

    # ---- rules 8 and 9: parent resolution, bold rows --------------------------------
    index = run_index()
    unresolved: dict[str, list[int]] = {}
    for line_no, cells in usable:
        if not parent_resolves(cells[COL["parent"]], index):
            unresolved.setdefault(unbold(cells[COL["parent"]]), []).append(line_no)
    for parent, line_nos in sorted(unresolved.items(), key=lambda kv: kv[1][0]):
        warnings.append(
            "[8] parent %r does not resolve on disk (%d row(s): %s)"
            % (parent, len(line_nos), ", ".join("L%d" % n for n in line_nos))
        )

    bolds = [(ln, c) for ln, c in usable if is_bold_row(c)]
    per_system: dict[str, list[int]] = {}
    for line_no, cells in bolds:
        per_system.setdefault(unbold(cells[COL["system"]]), []).append(line_no)
    for system, line_nos in sorted(per_system.items()):
        if len(line_nos) > 1:
            violations.append(
                "[9] system %s carries %d bold rows: %s"
                % (system, len(line_nos), ", ".join("L%d" % n for n in line_nos))
            )
    for line_no, cells in bolds:
        parent = unbold(cells[COL["parent"]])
        if parent == "":
            violations.append("[9] bold row L%d has an empty parent cell" % line_no)
        elif not parent_resolves(cells[COL["parent"]], index):
            warnings.append(
                "[9] bold row L%d parent %r does not resolve on disk" % (line_no, parent)
            )

    for msg in violations:
        print("VIOLATION " + msg)
    for msg in warnings:
        print("WARN " + msg)
    print(
        "summary: rows=%d blocks=%d bolds=%d violations=%d warnings=%d"
        % (len(rows), len(blocks), len(bolds), len(violations), len(warnings))
    )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
