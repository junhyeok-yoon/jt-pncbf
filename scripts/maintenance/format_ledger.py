#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "docs/ledger.md"
NUMERIC_COLUMNS = {
    "reach",
    "collision",
    "oob",
    "stuck",
    "timeout",
    "infeas",
    "infeasibility",
    "sat_rate",
    "saturation_rate",
    "cps",
    "cps_ci_lo",
    "cps_ci_hi",
    "reach_ci_lo",
    "reach_ci_hi",
    "collision_ci_lo",
    "collision_ci_hi",
    "stuck_ci_lo",
    "stuck_ci_hi",
    "infeasibility_ci_lo",
    "infeasibility_ci_hi",
}


def main() -> int:
    original = LEDGER_PATH.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    header: list[str] | None = None
    formatted: list[str] = []
    for line in lines:
        body, ending = _split_ending(line)
        if body.startswith("|") and body.endswith("|"):
            cells = [cell.strip() for cell in body.strip("|").split("|")]
            if header is None and cells and cells[0] == "version":
                header = cells
            elif header is not None and not _is_separator(cells) and len(cells) == len(header):
                cells = [_format_cell(name, cell) for name, cell in zip(header, cells)]
                body = "| " + " | ".join(cells) + " |"
        formatted.append(body + ending)
    updated = "".join(formatted)
    LEDGER_PATH.write_text(updated, encoding="utf-8")
    return 0


def _format_cell(column: str, cell: str) -> str:
    if column not in NUMERIC_COLUMNS or not cell:
        return cell
    bold = cell.startswith("**") and cell.endswith("**") and len(cell) >= 4
    inner = cell[2:-2] if bold else cell
    if not inner:
        return cell
    try:
        formatted = f"{float(inner):.4f}"
    except ValueError:
        return cell
    return f"**{formatted}**" if bold else formatted


def _is_separator(cells: list[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _split_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


if __name__ == "__main__":
    raise SystemExit(main())
