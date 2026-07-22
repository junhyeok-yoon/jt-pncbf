"""v2.7.4 run-directory layout: framework-tagged run ids under data/runs/<version>/ (disk-org only).

The run id keeps its full self-describing form and gains a framework tag inserted directly AFTER the
version, preserving timestamp and seed exactly:

    v2.7.4__20260720-091830__seed42   (tag off — today's form, still produced when framework is unknown)
    v2.7.4__jt__20260720-091830__seed42   (jt_pncbf)
    v2.7.3__oc__20260719-201644__seed42   (oc_pncbf)

The tag is derived ONLY from the existing `run.framework` config field (never a new key) so it cannot
disagree with the framework the checkpoint records. An absent/unrecognized framework yields NO tag rather
than a guess. The version appears in both the folder and the id on purpose: the folder groups, the id stays
globally unique so it stays greppable and citable.
"""
from __future__ import annotations

from pathlib import Path

FRAMEWORK_TAGS = {"oc_pncbf": "oc", "jt_pncbf": "jt"}


def framework_tag(framework: str | None) -> str:
    """Map a run.framework value to its short tag; '' for absent/unrecognized (no guess)."""
    return FRAMEWORK_TAGS.get(str(framework or ""), "")


def make_run_id(version: str, framework: str | None, timestamp: str, seed: int, *, tag: bool = True) -> str:
    """Build the run id. With tag=True and a known framework, insert the tag after the version; otherwise
    (tag=False or unknown framework) reproduce today's `version__timestamp__seedN` form exactly."""
    t = framework_tag(framework) if tag else ""
    segments = [version] + ([t] if t else []) + [timestamp, f"seed{seed}"]
    return "__".join(segments)


def run_dir_for(output_root: Path | str, version: str, run_id: str, *, set_name: str | None = None) -> Path:
    """Resolve the run directory under data/runs/<version>/[<set>/]<run_id> (output_root is the data root).
    With set_name, the run lives inside its value/JT set folder; without it, directly under the version."""
    base = Path(output_root) / "runs" / version
    if set_name:
        base = base / set_name
    return base / run_id


def _timestamp_of(run_id: str) -> str | None:
    m = re.search(r"(\d{8}-\d{6})", str(run_id))
    return m.group(1) if m else None


def set_folder_name(value_run_timestamp: str, seed: int) -> str:
    """A value/JT set folder is named after the VALUE run (it comes first): set__<value_ts>__seed<N>."""
    return f"set__{value_run_timestamp}__seed{seed}"


def locate_run(output_root: Path | str, run_id: str) -> Path | None:
    """Find an existing run directory by id under data/runs/ (directly under a version, or inside a set)."""
    runs = Path(output_root) / "runs"
    if not runs.exists():
        return None
    for cand in list(runs.glob(f"*/{run_id}")) + list(runs.glob(f"*/set__*/{run_id}")):
        if cand.is_dir():
            return cand
    return None


def resolve_set(output_root: Path | str, framework: str | None, timestamp: str, seed: int, *,
                value_init_run_id: str | None, version: str) -> tuple[str, str | None]:
    """Decide the set folder a run being created lives in. EVERY run lives inside a set — this function
    ALWAYS returns a non-empty set name, never None, so no run is ever placed directly under a version
    folder (asserted by test). Returns (set_name, reason | None). Rules, in order:

    1. a run with no resolvable value-init (a value run, or an unresolved/standalone run) creates its own
       set set__<its timestamp>__seed<N> and lives in it (a single-member set is still a set);
    2. a run whose run.value_init_run_id resolves to a run that is already inside a set WITHIN THE SAME
       VERSION joins that set (smoke runs included — a smoke is part of its lineage's work);
    3. a run whose value-init parent resolves to a DIFFERENT version creates its own set in its own
       version folder and records the cross-version parent in the reason (version is the stronger key; the
       dependency stays machine-readable through the config link);
    4. a value-init that cannot be located on disk creates its own single-member set; never infer a parent
       from timestamp adjacency.
    """
    own_set = set_folder_name(timestamp, seed)
    if not value_init_run_id:
        return own_set, None                                          # rule 1
    located = locate_run(output_root, value_init_run_id)
    if located is None:
        return own_set, f"value_init parent '{value_init_run_id}' not found on disk; own single-member set"  # rule 4
    parent_version = located.relative_to(Path(output_root) / "runs").parts[0]
    if located.parent.name.startswith("set__") and parent_version == version:
        return located.parent.name, None                             # rule 2: join same-version parent's set
    return own_set, (f"value_init parent '{value_init_run_id}' is in version {parent_version} "
                     f"(!= {version}) or not yet in a set; own set in own version")  # rule 3


import re

# A canonical run id is v<major>.<minor>.<patch>[__<tag>]__<YYYYMMDD-HHMMSS>__seed<N>[_<collision-suffix>].
# The version routes ONLY when this full structure matches; a name with a version-looking head but no valid
# timestamp segment (e.g. v2.5.1__d1_demos, v2.7.4_m1__seed12345) does NOT parse and goes to _unversioned.
_RUN_ID_RE = re.compile(r"^(v\d+\.\d+\.\d+)__(?:oc__|jt__)?\d{8}-\d{6}__seed\d+(?:_\d+)?$")


def parse_version(run_id_or_name: str) -> str | None:
    """Return the v<major>.<minor>.<patch> version of a canonical run id/dir name, or None if the name does
    not match the full canonical run-id structure (used by the migration to route unparseable names)."""
    m = _RUN_ID_RE.match(str(run_id_or_name))
    return m.group(1) if m else None
