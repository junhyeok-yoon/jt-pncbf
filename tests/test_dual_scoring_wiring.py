"""v2.8.0 dual-scoring standard — wiring tests (W).

Guards the invariant that quadrotor_3d in-loop evaluation resolves to the MIXED pool while every other
system keeps its legacy in-loop pool, and that the resolution is opt-out-able via config."""
from __future__ import annotations
import copy

from src.frameworks.jt_pncbf.train import _pool_stem_for, load_effective_config


def _cfg(system):
    c = copy.deepcopy(load_effective_config())
    c.setdefault("run", {})["system"] = system
    return c


def test_inloop_pool_resolves_to_mixed_for_quadrotor_3d():
    stem = _pool_stem_for("inloop", _cfg("quadrotor_3d"), "quadrotor_3d")
    assert stem == "eval_inloop_quadrotor-3d-d2r-mixed_n2000_seed45678", stem


def test_inloop_pool_unchanged_for_non_3d_system():
    # a non-3D system must NOT resolve to the mixed pool
    stem = _pool_stem_for("inloop", _cfg("double_integrator"), "double_integrator")
    assert "mixed" not in stem, stem
    assert stem.startswith("eval_inloop_"), stem


def test_inloop_mixed_is_opt_outable():
    c = _cfg("quadrotor_3d")
    c["eval"]["in_loop"]["mixed"] = False
    stem = _pool_stem_for("inloop", c, "quadrotor_3d")
    assert "mixed" not in stem, stem


def test_full_pool_never_mixed_for_quadrotor_3d():
    stem = _pool_stem_for("full", _cfg("quadrotor_3d"), "quadrotor_3d")
    assert "mixed" not in stem, stem


# ---- W1: in-loop sub-score split ----
from pathlib import Path as _Path
from src.eval.evaluate import provenance_half_scores, EVAL_METRIC_COLUMNS


def test_w1_blended_equals_episode_weighted_mean_of_halves():
    # 3 full + 2 tilt60; blended mean must equal the episode-weighted mean of the two halves
    eps = [{"cps_episode": 0.2}, {"cps_episode": 0.8}, {"cps_episode": 0.5},
           {"cps_episode": 0.9}, {"cps_episode": 0.7}]
    prov = ["full", "tilt60", "full", "tilt60", "full"]
    h = provenance_half_scores(eps, prov)
    full = [0.2, 0.5, 0.7]; tilt = [0.8, 0.9]
    assert abs(h["cps_full_half"] - sum(full) / 3) < 1e-12
    assert abs(h["cps_tilt60_half"] - sum(tilt) / 2) < 1e-12
    blended = sum(e["cps_episode"] for e in eps) / len(eps)
    weighted = (3 * h["cps_full_half"] + 2 * h["cps_tilt60_half"]) / 5
    assert abs(blended - weighted) < 1e-12


def test_w1_provenance_free_schema_byte_identical():
    # the base metric schema must NOT carry the half columns; a provenance-free pool therefore keeps
    # exactly today's columns (the _record_eval column list = EVAL_METRIC_COLUMNS + halves-only-if-present).
    assert "cps_full_half" not in EVAL_METRIC_COLUMNS
    assert "cps_tilt60_half" not in EVAL_METRIC_COLUMNS
    row_without = {"cps": 0.5}  # no half keys
    cols = EVAL_METRIC_COLUMNS + [c for c in ("cps_full_half", "cps_tilt60_half") if c in row_without]
    assert cols == EVAL_METRIC_COLUMNS


def test_w1_best_pt_selection_reads_blended_cps_only():
    src = _Path("src/frameworks/jt_pncbf/train.py").read_text()
    assert 'cps = float(eval_result.eval_row["cps"])' in src            # selection reads the blended cps
    # the selection comparison must not key on a half sub-score
    for i, line in enumerate(src.splitlines()):
        if "best_cps" in line and ("if cps >" in line or "= cps" in line):
            assert "cps_full_half" not in line and "cps_tilt60_half" not in line


# ---- W2: three-cell final driver ----
from src.eval.run_full import final_cell_specs


def test_w2_quadrotor_3d_default_three_cells():
    specs = final_cell_specs(_cfg("quadrotor_3d"), "quadrotor_3d")
    names = [s[0] for s in specs]
    assert names == ["mixed", "tilt60", "bandopen"], names
    by = {s[0]: s for s in specs}
    assert "mixed" in by[0 if False else "mixed"][1] and by["mixed"][2] is True
    assert "navcone" in by["tilt60"][1] and by["tilt60"][2] is True
    assert "eval_full_quadrotor-3d-d2r_n2000_seed23456" == by["bandopen"][1] and by["bandopen"][2] is False


def test_w2_non_3d_single_main_cell():
    specs = final_cell_specs(_cfg("double_integrator"), "double_integrator")
    assert specs == [("main", None, True)], specs


def test_w2_eval_final_cells_restricts():
    c = _cfg("quadrotor_3d")
    c.setdefault("eval", {}).setdefault("final", {})["cells"] = ["tilt60"]
    specs = final_cell_specs(c, "quadrotor_3d")
    assert [s[0] for s in specs] == ["tilt60"], specs


# ---- W3: two-row ledger template (parsed by the independent R-audit parser) ----
import re as _re
from src.eval.run_full import dual_scoring_ledger_rows


def _audit_cells(line):  # the R-audit parser: split on UNescaped pipes, drop leading/trailing empties
    return [c.strip() for c in _re.split(r'(?<!\\)\|', line)[1:-1]]


def test_w3_two_row_pair_matches_header_cell_count():
    header = ("| version | system | date | parent | seeds | cps_v2 | cps_tilt60 | cps_bandopen | eval_source "
              "| reach | collision | oob | stuck | timeout | infeas | sat_rate | cps | verdict |")
    ncol = len(_audit_cells(header))
    tilt60 = {"reach": 0.9685, "collision": 0.0055, "oob": 0.0, "stuck": 0.0005, "timeout": 0.0255,
              "infeasibility": 0.0789, "saturation_rate": 0.4395, "cps": 0.9206}
    bandopen = {"reach": 0.9550, "collision": 0.0135, "oob": 0.005, "stuck": 0.0, "timeout": 0.0265,
                "infeasibility": 0.1016, "saturation_rate": 0.4693, "cps": 0.8818, "crossing_rate": 0.0345}
    rows = dual_scoring_ledger_rows(date="2026-08-01", parent="run42", run_id="run42", seeds="42",
                                    blended_cps=0.90, tilt60=tilt60, bandopen=bandopen)
    assert len(rows) == 2
    for r in rows:
        assert len(_audit_cells(r)) == ncol, (len(_audit_cells(r)), r)
    # one carries cps_tilt60 (col 6), the other cps_bandopen (col 7) as "value (rate)"; they share parent/date
    a, b = _audit_cells(rows[0]), _audit_cells(rows[1])
    assert a[6] == "0.9206" and a[7] == ""
    assert b[6] == "" and b[7] == "0.8818 (0.0345)"
    assert a[2] == b[2] and a[3] == b[3]   # shared date, parent
