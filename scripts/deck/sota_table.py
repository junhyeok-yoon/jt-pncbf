"""v2.7.7 M19 (Amdt 8) — SOTA status table, one row per system. Values are parsed DIRECTLY from the recorded
ledger rows (docs/ledger.md), never recomputed; the ledger column order is
  version | system | date | parent | seeds | cps_v2 | eval_source | reach | collision | oob | stuck | timeout | infeas | sat_rate | cps | verdict
and the emitted columns follow the amendment: cps, reach, collision, stuck, oob, timeout, infeasibility, seed
basis, source pointer. Cross-check targets are asserted against the parsed values (DI cps 0.8698 3-seed; planar
cps 0.9036 / collision 0.0215; 3-D legacy 0.9078 / banded 0.8051, reach 0.9375). Emits deck_assets/sota_table.md
and .json. Eval-only (reads recorded docs; no compute)."""
from __future__ import annotations
import json
from pathlib import Path
from scripts.deck.deck_style import OUT

LEDGER = Path("docs/ledger.md")
rows_txt = LEDGER.read_text().splitlines()

# (key label, distinctive substring that uniquely identifies the recorded SOTA row, seed basis, source pointer)
SPEC = [
    ("double_integrator", "DI SOTA OF RECORD", "42, 12345, 99 (3-seed mean)",
     "docs/ledger.md v2.3.0 (bold, DI SOTA of record) · docs/versions/v2.3.0_results.md"),
    ("unicycle", "Secured v2.2.2 unicycle SOTA", "42 (single-seed)",
     "docs/ledger.md v2.2.2-uni-inj (secured unicycle SOTA, best.pt@28000) · docs/versions/unicycle/injection_build.md"),
    ("quadrotor_planar", "k5 on no-inject ckpt; CI-sep. SOTA", "42 (single-seed)",
     "docs/ledger.md v2.7.1 (bold, planar SOTA; k5 on secured 3b27d691) · docs/versions/v2.7.1_results.md"),
    ("quadrotor_3d (legacy-scored)", "CANONICAL jt42000 legacy kstep", "42 (single-seed)",
     "docs/ledger.md v2.7.6 (JT step 42000, 09c33bf4, legacy, kstep k=5) · data/previous_runs/v2.7.6/stage2_eval/canonical_eval.json"),
    ("quadrotor_3d (band-scored)", "CANONICAL jt42000 banded kstep", "42 (single-seed)",
     "docs/ledger.md v2.7.6 (JT step 42000, 09c33bf4, banded, kstep k=5) · data/previous_runs/v2.7.6/stage2_eval/canonical_eval.json"),
]


def _num(s):
    s = s.replace("**", "").strip()
    if not s or s == "-":
        return None
    return float(s.split()[0].split("[")[0])          # first token, drop CI bracket


def parse_row(substr):
    line = next(l for l in rows_txt if substr in l and l.lstrip().startswith("|"))
    f = [c.replace("**", "").strip() for c in line.split("|")]
    # f[1]=version f[2]=system f[5]=seeds f[8]=reach f[9]=collision f[10]=oob f[11]=stuck f[12]=timeout f[13]=infeas f[15]=cps
    return {"version": f[1], "system": f[2], "cps": _num(f[15]), "reach": _num(f[8]), "collision": _num(f[9]),
            "stuck": _num(f[11]), "oob": _num(f[10]), "timeout": _num(f[12]), "infeasibility": _num(f[13])}


table = []
for label, substr, seeds, source in SPEC:
    r = parse_row(substr); r["label"] = label; r["seed_basis"] = seeds; r["source"] = source
    table.append(r)

# --- cross-check targets (must match the records) ---
by = {r["label"]: r for r in table}
def chk(cond, msg):
    assert cond, f"CROSS-CHECK FAILED: {msg}"
chk(by["double_integrator"]["cps"] == 0.8698, f"DI cps {by['double_integrator']['cps']} != 0.8698")
chk(by["quadrotor_planar"]["cps"] == 0.9036 and by["quadrotor_planar"]["collision"] == 0.0215,
    f"planar cps/collision {by['quadrotor_planar']['cps']}/{by['quadrotor_planar']['collision']}")
chk(by["quadrotor_3d (legacy-scored)"]["cps"] == 0.9078, "3-D legacy cps")
chk(by["quadrotor_3d (band-scored)"]["cps"] == 0.8051 and by["quadrotor_3d (band-scored)"]["reach"] == 0.9375,
    "3-D banded cps/reach")

# --- emit .json ---
OUT.mkdir(parents=True, exist_ok=True)
cols = ["cps", "reach", "collision", "stuck", "oob", "timeout", "infeasibility"]
js = {"note": "SOTA-of-record per system; values parsed verbatim from recorded ledger rows (no recompute). "
              "cps = composite; higher is better. Cross-check targets asserted.",
      "columns": cols + ["seed_basis", "source"],
      "rows": [{"system": r["label"], **{c: r[c] for c in cols}, "seed_basis": r["seed_basis"], "source": r["source"]}
               for r in table]}
(OUT / "sota_table.json").write_text(json.dumps(js, indent=2) + "\n")


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


# --- emit .md ---
lines = ["# JT-PNCBF — SOTA status by system", "",
         "One row per system, SOTA-of-record. **All values are parsed verbatim from the recorded ledger rows "
         "(`docs/ledger.md`); nothing is recomputed.** cps = composite performance score (higher better). "
         "Seed basis and the record source are given per row; the full metric split is the recorded eval.", "",
         "| system | cps | reach | collision | stuck | oob | timeout | infeasibility | seed basis | source |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in table:
    lines.append("| " + " | ".join([r["label"], fmt(r["cps"]), fmt(r["reach"]), fmt(r["collision"]), fmt(r["stuck"]),
                                     fmt(r["oob"]), fmt(r["timeout"]), fmt(r["infeasibility"]), r["seed_basis"],
                                     r["source"]]) + " |")
lines += ["", "Notes:",
          "- double_integrator is the only multi-seed row (3-seed mean; CI [0.8527, 0.8869]).",
          "- unicycle is never SOTA-bolded against DI (06_workflow §2.4); the secured v2.2.2-uni-inj checkpoint is "
          "its SOTA of record (a later v2.7.1 k5 eval reached cps 0.6588, CI-overlapping — not a CI-separated beat).",
          "- quadrotor_3d is single-seed (v2.7.6 headline); legacy vs band-scored are the two recorded scoring "
          "predicates of the SAME JT step-42000 checkpoint (09c33bf4) on the canonical pool (0ef3751b), kstep k=5.",
          "- No value was reconstructed; every component was present in the records, so no 'n/a' was needed."]
(OUT / "sota_table.md").write_text("\n".join(lines) + "\n")
print("M19 -> sota_table.md + sota_table.json ; cross-checks PASS")
for r in table:
    print(f"  {r['label']:32s} cps {fmt(r['cps'])} reach {fmt(r['reach'])} coll {fmt(r['collision'])} "
          f"stuck {fmt(r['stuck'])} oob {fmt(r['oob'])} timeout {fmt(r['timeout'])} infeas {fmt(r['infeasibility'])} [{r['seed_basis']}]")
