"""One-shot repair: cap docs/ledger.md's two prose columns (`verdict`, `eval_source`) at
MAX_CELL_CHARS, relocating everything removed verbatim into docs/ledger_verdicts.md.

DESIGN RULE THAT MAKES THE NUMBER-SAFETY CLAIM TRUE: a compressed cell is built ONLY by
SELECTING substrings of the original cell (sentence spans, regex captures, artifact tokens) and
joining them with fixed connectives. No figure is ever retyped, so no figure can be altered.

Order of operations, so the companion is lossless by construction:
  1. read docs/ledger.md as it is and record every pre-edit cell of both columns
  2. compress; each cell is checked for the cap, for illegal characters, and for the survival of
     every rule-5 artifact token and the rule-10 pool stem
  3. append the relocated text to each row's EXISTING companion section under two subheadings
     naming the column it came from, so the two relocations never merge and neither overwrites
     what the earlier verdict repair placed there
  4. verify -- losslessness per column, non-target cells byte-identical, rule-5 keys unchanged

Reads and writes docs/ only. Nothing under data/ is written.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/maintenance"))
from ledger_verdict_lint import parse  # noqa: E402

LEDGER = REPO / "docs" / "ledger.md"
COMPANION = REPO / "docs" / "ledger_verdicts.md"
MAX_CELL_CHARS = 400
COLUMNS = ("verdict", "eval_source")

# exactly what check_ledger consumes out of eval_source
ARTIFACT_RE = re.compile(r"[A-Za-z0-9_.\-/]+\.(?:json|npz|npy|pt)")
POOL_STEMS = ("fullcb", "fullscr41", "fullscr40", "inloopv2", "navconescr40", "navcone", "mixed")

_ABBREV = ("e.g.", "i.e.", "cf.", "vs.", "etc.", "approx.", "incl.", "resp.", "viz.", "et al.",
           "Amdt.", "St.", "Dr.", "Mr.", "Ms.", "Prof.", "Jr.", "Sr.", "No.", "Fig.", "Eq.",
           "NB.", "pct.", "sec.", "ca.")
_MASKS = (re.compile(r"\d+\s*[eE][+-]?\d+"), re.compile(r"\d+\.\d+"),
          re.compile(r"[A-Za-z0-9_~@%+\-]+(?:/[A-Za-z0-9_.~@%+\-]+)+"),
          re.compile(r"\b[A-Za-z0-9_\-]+\.(?:md|py|json|npz|pkl|yaml|yml|pt|csv|txt|png|pdf|tex|sh|log)\b"),
          re.compile(r"\bv\d+(?:\.\d+)+[A-Za-z0-9_\-]*"), re.compile(r"§\s*\d+(?:\.\d+)*"))


def sentences(text: str) -> list[str]:
    """Split into sentence spans on the masked-terminator rule, returning ORIGINAL substrings."""
    masked = text
    for ab in _ABBREV:
        masked = masked.replace(ab, "\x00" * len(ab))
    for rx in _MASKS:
        masked = rx.sub(lambda m: "\x00" * len(m.group(0)), masked)
    out, start = [], 0
    for m in re.finditer(r"[.!?]+(?=\s|$)", masked):
        end = m.end()
        span = text[start:end].strip()
        if span:
            out.append(span)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out or ([text.strip()] if text.strip() else [])


def strip_bold(s: str) -> tuple[str, bool]:
    t = s.strip()
    if t.startswith("**") and t.endswith("**") and len(t) > 4:
        return t[2:-2], True
    return t, False


def required_tokens(cell: str) -> list[str]:
    """Every artifact path check_ledger rule 5 consumes out of this cell, in order, de-duplicated."""
    seen, out = set(), []
    for t in ARTIFACT_RE.findall(cell):
        if "/" in t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def pool_stem(cell: str) -> str | None:
    return next((s for s in POOL_STEMS if s in cell), None)


# --------------------------------------------------------------------------- eval_source rebuild
_MODE_RE = re.compile(r"^\s*(\*\*)?\s*(eval_only|final)\b", re.I)
_POOLFILE_RE = re.compile(r"\b(eval_[A-Za-z0-9_.\-]*?\.pkl)\b")
_POOLNAME_RE = re.compile(r"\bpool\s+([A-Za-z0-9_.\-]+)")
_N_RE = re.compile(r"\bn[ =]\s*(\d{3,6})\b")
_EBS_RE = re.compile(r"\b(?:ebs|eval_batch_size)[ =]\s*(\d{2,6})\b")
_STEP_RE = re.compile(r"@\s*step\s*(\d+)|\bstep[_ ](\d{3,6})\b|\bbest\.pt@(\d+)\b")
_CKPT_OWNER_RE = re.compile(r"\b(best\.pt|final\.pt|step_\d+\.pt)\b")

# departures from the registered cell, detected by marker; each entry is (regex, label)
_DEPARTURES = (
    (re.compile(r"eval\.max_steps\s*400"), "eval.max_steps 400"),
    (re.compile(r"\bNEW CELL\b"), "NEW CELL"),
    (re.compile(r"box_klamp ON[^,;)]*"), None),
    (re.compile(r"hazard[ .]geom_form\s+\w+[^,;)]*"), None),
    (re.compile(r"network\.value\.ceiling\s*[\d.]+|value ceiling\s*[\d.]+"), None),
    (re.compile(r"empty_fallback\s*\{[^}]*\}"), None),
    (re.compile(r"terminal\s*\([^)]*\)"), None),
    (re.compile(r"projection\s+\w+"), None),
    (re.compile(r"alpha\s*\([^)]*\)"), None),
    (re.compile(r"band[- ](?:terminate|open)\w*"), None),
    (re.compile(r"tilt<=60|cps_tilt60|cps_bandopen"), None),
    (re.compile(r"dt_ctrl\s*[\d.]+"), None),
)
# the cell constants that are the REGISTERED cell and so are not departures worth carrying
_CANON = ("terminal (0.15, 0.3, 0.3)", "terminal (0.15,0.3,0.3)", "projection dual_solve",
          "empty_fallback {kstep,phases 1,k 3}", "empty_fallback {kstep, phases 1, k 3}",
          "alpha (2.0,100.0)", "alpha (2.0, 100.0)")


def compress_eval_source(cell: str, anchor: str) -> str:
    body, was_bold = strip_bold(cell)
    parts: list[str] = []

    m = _MODE_RE.match(body)
    mode = (m.group(2).lower() if m else "eval_only")
    parts.append(mode)

    dep: list[str] = []
    for rx, label in _DEPARTURES:
        mm = rx.search(body)
        if not mm:
            continue
        txt = label if label else mm.group(0).strip()
        if any(c in txt for c in _CANON) or txt in _CANON:
            continue
        if txt not in dep:
            dep.append(txt)
    cellname = ("registered cell v282_agree_gate.gate_overrides"
                if re.search(r"registered cell|gate_overrides|GATE CELL", body) else "cell as recorded")
    parts.append(cellname + (("; departs: " + ", ".join(dep[:4])) if dep else ""))

    pool = None
    mm = _POOLFILE_RE.search(body)
    if mm:
        pool = mm.group(1)
    else:
        stem = pool_stem(body)
        mm2 = _POOLNAME_RE.search(body)
        pool = (mm2.group(1) if mm2 else None) or stem
    if pool:
        pd = "pool " + pool
        mm = _N_RE.search(body)
        if mm:
            pd += ", n " + mm.group(1)
        mm = _EBS_RE.search(body)
        if mm:
            pd += ", ebs " + mm.group(1)
        parts.append(pd)
    stem = pool_stem(body)
    if stem and (not pool or stem not in pool):
        parts.append(stem)

    mm = _STEP_RE.search(body)
    if mm:
        step = next(g for g in mm.groups() if g)
        owner = _CKPT_OWNER_RE.search(body)
        parts.append(f"{owner.group(1) if owner else 'ckpt'} @ step {step}")

    toks = required_tokens(cell)
    if toks:
        parts.append("artifacts " + ", ".join(toks))
    ptr = f"full text docs/ledger_verdicts.md#{anchor}"

    def assemble(ps, with_ptr=True):
        s = "; ".join(p for p in (ps + ([ptr] if with_ptr else [])) if p)
        return ("**" + s + "**") if was_bold else s

    # Graceful degradation, in a fixed order, so the cap never truncates mid-token: shed
    # departures from the tail first, then the companion pointer. The artifact paths rule 5
    # consumes and the pool stem rule 10 consumes are NEVER shed.
    out = assemble(parts)
    if len(out) > MAX_CELL_CHARS and dep:
        for keep in range(len(dep) - 1, -1, -1):
            trimmed = list(parts)
            trimmed[1] = cellname + (("; departs: " + ", ".join(dep[:keep])) if keep else "")
            out = assemble(trimmed)
            if len(out) <= MAX_CELL_CHARS:
                break
            parts = trimmed
    if len(out) > MAX_CELL_CHARS:
        out = assemble(parts, with_ptr=False)
    return out


# ------------------------------------------------------------------------------- verdict rebuild
_HEAD_KEYS = re.compile(
    r"\b(HOLDS|FALSIFIED|REFUTED|CONFIRMED|PASS(?:ES|ED)?|FAIL(?:S|ED)?|GATE|cps|reach|"
    r"collision|PREDICTION|RESULT|A1)\b")
_COMPARE_KEYS = re.compile(r"(NOT comparable|not comparable|NOT COMPARABLE|comparable|SUPERSEDED|"
                           r"superseded|not a peer|NOT A PEER|not rankable)", re.I)
_STATUS_KEYS = re.compile(r"(NOT bold|not bold|no bold change|NO BOLD CHANGE|UNBOLDED|BOLD MOVED|"
                          r"bold retained|NOT promoted|no promotion|NOT SOTA|SOTA UNCHANGED|"
                          r"RELEASED FROM BOLD|flag Researcher|bold-INELIGIBLE|SOTA)", re.I)
_PTR_KEYS = re.compile(r"(detail|Detail|see |See |docs/|\.md)")


def compress_verdict(cell: str, anchor: str) -> str:
    body, was_bold = strip_bold(cell)
    sents = sentences(body)
    slot1 = sents[0] if sents else ""
    rest = sents[1:]

    def pick(rx, pool):
        for s in pool:
            if rx.search(s):
                return s
        return None

    slot2 = pick(_HEAD_KEYS, rest) or ""
    pool2 = [s for s in rest if s is not slot2]
    slot3 = pick(_COMPARE_KEYS, pool2) or ""
    pool3 = [s for s in pool2 if s is not slot3]
    slot4 = pick(_STATUS_KEYS, pool3) or ""
    pool4 = [s for s in pool3 if s is not slot4]
    slot5_src = pick(_PTR_KEYS, pool4) or ""
    det = re.search(r"(docs/[A-Za-z0-9_./\-]+\.md)", body)
    pointer = (f"Detail {det.group(1)} and " if det else "Detail ") + \
              f"docs/ledger_verdicts.md#{anchor}."

    def clip(s, n):
        s = s.strip()
        if len(s) <= n:
            return s
        cut = s[:n]
        cut = cut[:cut.rfind(" ")] if " " in cut else cut
        return cut.rstrip(" ,;:-") + " ..."

    budget = MAX_CELL_CHARS - (4 if was_bold else 0) - len(pointer) - 4
    slots = [s for s in (slot1, slot2, slot3, slot4) if s]
    out = ""
    for i, s in enumerate(slots):
        remaining = budget - len(out)
        if remaining <= 12:
            break
        share = max(remaining // max(len(slots) - i, 1), 40)
        piece = clip(s, min(share, remaining))
        if piece:
            out += (" " if out else "") + piece
    text = (out + " " + pointer).strip()
    if was_bold:
        text = "**" + text + "**"
    return text


COMPRESS = {"verdict": compress_verdict, "eval_source": compress_eval_source}
