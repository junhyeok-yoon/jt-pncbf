"""Verdict-cell sentence counter for docs/ledger.md, and the shared parser the repair uses.

THE COUNTING RULE (rule 11 of scripts/check_ledger.py; stated here once so the two agree).
A verdict's sentence count is the number of SENTENCE TERMINATORS it contains. A terminator is a
run of `.`, `!` or `?` that is either at end-of-cell or followed by whitespace, AFTER the following
five classes have been masked out so they can never terminate a sentence:

  1. decimals and numeric ranges          1.53, 0.0083, +1.11e-16, 2.5e-04
  2. dotted version / step / id tokens    v2.9.1, 09c33bf4 has none, sha8 3682a4e3, 1e-12
  3. file names, paths and extensions     docs/ledger.md, best.pt, foo.json, a.npz, b.yaml, c.py
  4. section and figure references        §1.6, Fig. 3, No. 2, Eq. 4
  5. a closed list of abbreviations       e.g. i.e. cf. vs. etc. approx. incl. resp. Amdt. St.
                                          Dr. Mr. Ms. Prof. Jr. Sr. et al. viz. NB. pct.

A cell with no terminator at all counts as ONE sentence when it is non-empty (the ledger's own
one-clause style, e.g. "superseded (first full run)"), and ZERO when it is empty.
Trailing terminators do not create an extra empty sentence.

Read-only. Reads docs/ledger.md and nothing else.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "docs" / "ledger.md"
MAX_SENTENCES = 5

_ABBREV = (
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "approx.", "incl.", "resp.", "viz.", "et al.",
    "Amdt.", "St.", "Dr.", "Mr.", "Ms.", "Prof.", "Jr.", "Sr.", "No.", "Fig.", "Eq.", "NB.",
    "pct.", "sec.", "ca.",
)
# order matters: longer patterns first so a shorter one cannot bite into them
_MASKS = (
    re.compile(r"\d+\s*[eE][+-]?\d+"),                    # 1e-12, 2.5e-04 exponent tail
    re.compile(r"\d+\.\d+"),                              # decimals
    re.compile(r"[A-Za-z0-9_~@%+\-]+(?:/[A-Za-z0-9_.~@%+\-]+)+"),   # paths (contain a slash)
    re.compile(r"\b[A-Za-z0-9_\-]+\.(?:md|py|json|npz|pkl|yaml|yml|pt|csv|txt|png|pdf|tex|sh|log)\b"),
    re.compile(r"\bv\d+(?:\.\d+)+[A-Za-z0-9_\-]*"),       # v2.9.1, v2.2.2-uni
    re.compile(r"§\s*\d+(?:\.\d+)*"),                     # §1.6
)


def _mask(text: str) -> str:
    """Replace every protected class with same-length filler containing no terminator."""
    out = text
    for ab in _ABBREV:                                    # abbreviations, case as written
        out = out.replace(ab, "\x00" * len(ab))
    for rx in _MASKS:
        out = rx.sub(lambda m: "\x00" * len(m.group(0)), out)
    return out


def count_sentences(verdict: str) -> int:
    v = verdict.strip()
    if not v:
        return 0
    masked = _mask(v)
    n = len(re.findall(r"[.!?]+(?=\s|$)", masked))
    return n if n else 1


def parse(path: Path = LEDGER):
    """-> (lines, header_index, columns, [(line_no_1based, cells)])"""
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("| version |"))
    cols = [c.strip() for c in lines[hdr].strip().strip("|").split("|")]
    rows = []
    for i, l in enumerate(lines):
        if i <= hdr + 1 or not l.strip().startswith("|") or l.startswith("|---"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) == len(cols):
            rows.append((i + 1, cells))
    return lines, hdr, cols, rows


def main() -> int:
    _, _, cols, rows = parse()
    vi = cols.index("verdict")
    counts = sorted(count_sentences(c[vi]) for _, c in rows)
    n = len(counts)
    pct = lambda p: counts[min(n - 1, int(round((p / 100) * (n - 1))))]
    over = [(ln, count_sentences(c[vi])) for ln, c in rows if count_sentences(c[vi]) > MAX_SENTENCES]
    print(f"rows {n}  min {counts[0]}  median {pct(50)}  p95 {pct(95)}  max {counts[-1]}")
    print(f"already <= {MAX_SENTENCES}: {n - len(over)}   over: {len(over)}")
    if "-v" in sys.argv:
        for ln, k in over:
            print(f"  L{ln}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
