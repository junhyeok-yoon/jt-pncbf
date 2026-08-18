"""Apply the 400-character cap to docs/ledger.md's `verdict` and `eval_source` columns.

eval_source is rebuilt MECHANICALLY by ledger_length_repair.compress_eval_source -- selection of
substrings only. verdict is hand-authored per row below, and every authored cell is checked against
the original by NUMBER SET INCLUSION: every numeric literal in the new text must already occur in
the old text, so no figure can have been altered or invented.

Relocated text is appended to each row's EXISTING companion section under two subheadings naming
the column it came from.

Reads and writes docs/ only. Nothing under data/ is written.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/maintenance"))
from ledger_verdict_lint import parse  # noqa: E402
from ledger_verdict_repair import slug  # noqa: E402
from ledger_length_repair import (  # noqa: E402
    MAX_CELL_CHARS, compress_eval_source, required_tokens, pool_stem, strip_bold)

LEDGER = REPO / "docs" / "ledger.md"
COMPANION = REPO / "docs" / "ledger_verdicts.md"
NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# paths, filenames and companion anchors carry digits of their own that say nothing about a
# measurement; they are masked out before the number-set comparison on BOTH sides.
_PATHISH = re.compile(r"docs/ledger_verdicts\.md#[A-Za-z0-9-]+|[A-Za-z_][A-Za-z0-9_~@%+\-]*(?:/[A-Za-z0-9_.~@%+\-]+)+"
                      r"|\b[A-Za-z0-9_\-]+\.(?:md|py|json|npz|pkl|yaml|yml|pt|csv|txt|log)\b")


def measurement_numbers(text: str) -> set[str]:
    return set(NUM_RE.findall(_PATHISH.sub(" ", text)))

# ---------------------------------------------------------------------------------------------
# Hand-authored verdicts, slot order preserved: 1 what the row is | 2 headline | 3 comparability
# | 4 bold/promotion/supersession | 5 pointer. `{A}` -> that row's companion anchor.
# ---------------------------------------------------------------------------------------------
from ledger_verdicts_authored import V  # noqa: E402


def anchors(rows, cols):
    ai = cols.index("alias")
    base = {}
    for ln, c in rows:
        base[ln] = f"{slug(c[0])}--{slug(c[1])}--{slug(c[ai])}"
    counts = {}
    for b in base.values():
        counts[b] = counts.get(b, 0) + 1
    seen, out = {}, {}
    for ln, c in rows:
        b = base[ln]
        seen[b] = seen.get(b, 0) + 1
        out[ln] = b if counts[b] == 1 else f"{b}--{seen[b]}"
    assert len(set(out.values())) == len(out)
    return out


def main() -> int:
    lines, hdr, cols, rows = parse()
    vi, ei = cols.index("verdict"), cols.index("eval_source")
    ANCH = anchors(rows, cols)
    pre = {ln: {"verdict": c[vi], "eval_source": c[ei]} for ln, c in rows}

    relocated: dict[int, dict[str, str]] = {}
    for ln, c in rows:
        cells = list(c)
        changed = {}
        for name, j in (("verdict", vi), ("eval_source", ei)):
            old = cells[j]
            if len(old) <= MAX_CELL_CHARS:
                continue
            if name == "eval_source":
                new = compress_eval_source(old, ANCH[ln])
            else:
                assert ln in V, f"L{ln} verdict is {len(old)} chars and has no authored replacement"
                new = V[ln].replace("{A}", ANCH[ln])
                extra = measurement_numbers(new) - measurement_numbers(old)
                assert not extra, f"L{ln} verdict introduces numbers absent from the original: {extra}"
                ob, wb = strip_bold(old)
                nb, nwb = strip_bold(new)
                assert wb == nwb, f"L{ln} bold markup changed"
            assert len(new) <= MAX_CELL_CHARS, f"L{ln} {name} is {len(new)} chars"
            assert "|" not in new and "\n" not in new, f"L{ln} {name} illegal char"
            if name == "eval_source":
                for t in required_tokens(old):
                    assert t in new, f"L{ln} lost rule-5 artifact token {t}"
                st = pool_stem(old)
                assert st is None or st in new, f"L{ln} lost rule-10 pool stem {st}"
            cells[j] = new
            changed[name] = old
        if changed:
            relocated[ln] = changed
            lines[ln - 1] = "| " + " | ".join(cells) + " |"

    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"compressed cells on {len(relocated)} rows "
          f"(verdict {sum('verdict' in v for v in relocated.values())}, "
          f"eval_source {sum('eval_source' in v for v in relocated.values())})")

    # --- relocate the pre-edit text into each row's EXISTING companion section ------------------
    comp = COMPANION.read_text()
    for ln, ch in sorted(relocated.items()):
        a = ANCH[ln]
        marker = f'<a id="{a}"></a>'
        i = comp.find(marker)
        assert i >= 0, f"L{ln}: companion section {a} absent"
        j = comp.find('<a id="', i + len(marker))
        j = len(comp) if j < 0 else j
        add = []
        for name in ("verdict", "eval_source"):
            if name in ch:
                add += [f"### relocated from the `{name}` column (pre-cap text, verbatim)", "",
                        ch[name], ""]
        comp = comp[:j] + "\n".join(add) + "\n" + comp[j:]
    COMPANION.write_text(comp)
    print(f"appended relocated text to {len(relocated)} companion sections")

    # --- verification --------------------------------------------------------------------------
    _, _, cols2, rows2 = parse()
    comp = COMPANION.read_text()
    for name in ("verdict", "eval_source"):
        miss = [ln for ln in pre if pre[ln][name] not in comp and len(pre[ln][name]) > MAX_CELL_CHARS]
        print(f"losslessness[{name}]: over-cap cells recoverable verbatim -- failures {miss or 'NONE'}")
    diffs = [(ln, cols[j]) for (ln, a), (ln2, b) in zip(rows, rows2)
             for j in range(len(cols)) if j not in (vi, ei) and a[j] != b[j]]
    print(f"cells outside the two columns differing: {diffs or 'NONE'}")
    import numpy as np
    for name, j in (("verdict", vi), ("eval_source", ei)):
        L = np.array([len(c[j]) for _, c in rows2])
        print(f"post-edit {name:12s} min {L.min()} median {int(np.percentile(L,50))} "
              f"p95 {int(np.percentile(L,95))} max {L.max()} over {MAX_CELL_CHARS}: {int((L>MAX_CELL_CHARS).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
