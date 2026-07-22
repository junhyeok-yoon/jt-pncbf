"""v2.7.4 run-layout: framework-tagged run ids under data/runs/<version>/ (NEW file)."""
from pathlib import Path

from src.common.run_layout import (
    framework_tag,
    make_run_id,
    run_dir_for,
    parse_version,
    resolve_set,
    set_folder_name,
)

TS = "20260720-091830"
VTS = "20260720-083533"     # a value run's timestamp


def test_tag_off_reproduces_todays_run_id():
    # tag disabled -> the historical version__timestamp__seedN form, byte-for-byte
    assert make_run_id("v2.7.4", "jt_pncbf", TS, 42, tag=False) == "v2.7.4__20260720-091830__seed42"
    assert make_run_id("v2.7.3", "oc_pncbf", TS, 42, tag=False) == "v2.7.3__20260720-091830__seed42"


def test_tag_on_inserts_for_both_frameworks():
    assert make_run_id("v2.7.4", "jt_pncbf", TS, 42) == "v2.7.4__jt__20260720-091830__seed42"
    assert make_run_id("v2.7.3", "oc_pncbf", TS, 42) == "v2.7.3__oc__20260720-091830__seed42"
    assert framework_tag("jt_pncbf") == "jt" and framework_tag("oc_pncbf") == "oc"


def test_unknown_framework_degrades_to_no_tag():
    # absent / unrecognized framework -> NO tag (never a guess), reproducing the untagged form
    for fw in (None, "", "cpi", "something_else"):
        assert make_run_id("v2.7.4", fw, TS, 42) == "v2.7.4__20260720-091830__seed42"
        assert framework_tag(fw) == ""


def test_run_dir_resolves_under_data_runs_version():
    rid = make_run_id("v2.7.4", "jt_pncbf", TS, 42)
    d = run_dir_for(Path("/repo/data"), "v2.7.4", rid)
    assert d == Path("/repo/data/runs/v2.7.4/v2.7.4__jt__20260720-091830__seed42")
    assert d.parent == Path("/repo/data/runs/v2.7.4")


def test_parse_version_strict_canonical_only():
    # canonical (untagged and tagged) parse to the version; non-canonical names do NOT parse (-> _unversioned)
    assert parse_version("v2.7.4__20260720-091830__seed42") == "v2.7.4"
    assert parse_version("v2.7.4__jt__20260720-091830__seed42") == "v2.7.4"
    assert parse_version("v2.7.4__20260720-091830__seed42_1") == "v2.7.4"      # collision suffix
    for bad in ("v2.5.1__d1_demos", "v2.5.1__pi_bc", "v2.1.0_lookahead", "v2.7.4_m1__seed12345",
                "verification", "gpu_fix_smoke", "v2.5.1__it2vhat__seed99"):
        assert parse_version(bad) is None, bad


# ---- v2.7.4 value/JT set folders ------------------------------------------------------------------

def _mk_run(tmp, version, set_name, run_id):
    d = tmp / "runs" / version / (set_name or "") / run_id if set_name else tmp / "runs" / version / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_value_run_creates_its_set_folder(tmp_path):
    # rule 1: an oc (value) run creates set__<its ts>__seedN and lives inside it
    set_name, reason = resolve_set(tmp_path, "oc_pncbf", VTS, 42, value_init_run_id=None, version="v2.7.4")
    assert set_name == set_folder_name(VTS, 42) == "set__20260720-083533__seed42"
    assert reason is None
    d = run_dir_for(tmp_path, "v2.7.4", make_run_id("v2.7.4", "oc_pncbf", VTS, 42), set_name=set_name)
    assert d.parent.name == "set__20260720-083533__seed42" and d.parent.parent.name == "v2.7.4"


def test_jt_run_with_resolvable_value_init_joins_set(tmp_path):
    # rule 2: same-version value run already in its set; the JT (value-init) joins the SAME set
    value_rid = make_run_id("v2.7.4", "oc_pncbf", VTS, 42)
    _mk_run(tmp_path, "v2.7.4", set_folder_name(VTS, 42), value_rid)
    set_name, reason = resolve_set(tmp_path, "jt_pncbf", TS, 42, value_init_run_id=value_rid, version="v2.7.4")
    assert set_name == "set__20260720-083533__seed42" and reason is None
    d = run_dir_for(tmp_path, "v2.7.4", make_run_id("v2.7.4", "jt_pncbf", TS, 42), set_name=set_name)
    assert d.parent.name == "set__20260720-083533__seed42"    # same folder as the value run


def test_smoke_joins_parents_set(tmp_path):
    # rule 2 now INCLUDES smoke runs — a smoke of a lineage is part of that lineage's work
    value_rid = make_run_id("v2.7.4", "oc_pncbf", VTS, 42)
    _mk_run(tmp_path, "v2.7.4", set_folder_name(VTS, 42), value_rid)
    # (smoke is not a distinct argument any more; a jt smoke resolves its value-init exactly like a full run)
    set_name, reason = resolve_set(tmp_path, "jt_pncbf", TS, 42, value_init_run_id=value_rid, version="v2.7.4")
    assert set_name == "set__20260720-083533__seed42" and reason is None


def test_cross_version_parent_creates_own_set_in_own_version(tmp_path):
    # rule 3: value-init parent is a DIFFERENT version -> own set in own version folder + reason
    parent = make_run_id("v2.6.1", "oc_pncbf", "20260715-170907", 42)
    _mk_run(tmp_path, "v2.6.1", set_folder_name("20260715-170907", 42), parent)
    sn, reason = resolve_set(tmp_path, "jt_pncbf", "20260716-111140", 42, value_init_run_id=parent, version="v2.6.2")
    assert sn == set_folder_name("20260716-111140", 42) and reason and "v2.6.1" in reason
    d = run_dir_for(tmp_path, "v2.6.2", make_run_id("v2.6.2", "jt_pncbf", "20260716-111140", 42), set_name=sn)
    assert d.parent.parent.name == "v2.6.2"                    # own version, own set


def test_unresolved_creates_own_single_member_set(tmp_path):
    # rule 4: a value-init that cannot be located, and a standalone run, each create their OWN set (never
    # a version-folder fallback, never a timestamp-inferred parent)
    sn, reason = resolve_set(tmp_path, "jt_pncbf", TS, 42,
                             value_init_run_id="v2.7.4__oc__20260720-999999__seed42", version="v2.7.4")
    assert sn == set_folder_name(TS, 42) and reason and "not found" in reason
    sn2, reason2 = resolve_set(tmp_path, "jt_pncbf", TS, 42, value_init_run_id=None, version="v2.7.4")
    assert sn2 == set_folder_name(TS, 42) and reason2 is None


def test_invariant_no_run_ever_directly_under_version_folder(tmp_path):
    # THE correction: resolve_set ALWAYS yields a set; run_dir_for then always nests under set__..., so the
    # run's parent is a set folder and its grandparent is the version — never a run directly under version.
    cases = [
        ("oc_pncbf", VTS, None, "v2.7.4"),
        ("jt_pncbf", TS, None, "v2.7.4"),                                   # standalone
        ("jt_pncbf", TS, "v2.7.4__oc__20260720-999999__seed42", "v2.7.4"),  # unresolvable
    ]
    # a resolvable same-version parent
    value_rid = make_run_id("v2.7.4", "oc_pncbf", VTS, 42)
    _mk_run(tmp_path, "v2.7.4", set_folder_name(VTS, 42), value_rid)
    cases.append(("jt_pncbf", TS, value_rid, "v2.7.4"))
    for fw, ts, vi, ver in cases:
        sn, _ = resolve_set(tmp_path, fw, ts, 42, value_init_run_id=vi, version=ver)
        assert sn is not None and sn.startswith("set__"), (fw, ts, vi)
        d = run_dir_for(tmp_path, ver, make_run_id(ver, fw, ts, 42), set_name=sn)
        assert d.parent.name.startswith("set__"), f"{fw} run placed directly under version folder!"
        assert d.parent.parent.name == ver
