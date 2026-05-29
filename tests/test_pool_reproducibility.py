from __future__ import annotations

import json
from pathlib import Path
import pickle

from src.eval.build_pools import (
    build_pool,
    default_pool_specs,
    load_base_config,
    sha256_file,
    write_pool,
)


def test_default_pool_builds_are_byte_identical(tmp_path: Path) -> None:
    config = load_base_config()

    for spec in default_pool_specs(config):
        first_pool = build_pool(config, "double_integrator", spec)
        second_pool = build_pool(config, "double_integrator", spec)

        first = write_pool(
            first_pool,
            config,
            output_dir=tmp_path / f"{spec.name}_first",
            created_at="2026-05-27T00:00:00+00:00",
            git_commit="test",
        )
        second = write_pool(
            second_pool,
            config,
            output_dir=tmp_path / f"{spec.name}_second",
            created_at="2026-05-27T00:00:00+00:00",
            git_commit="test",
        )

        assert first.pool_path.read_bytes() == second.pool_path.read_bytes()
        assert first.sha256 == second.sha256
        _assert_payload_key_order(first.pool_path)
        _assert_manifest_sha(first.manifest_path, first.pool_path)
        _assert_manifest_sha(second.manifest_path, second.pool_path)


def _assert_manifest_sha(manifest_path: Path, pool_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pool_sha256"] == sha256_file(pool_path)
    assert manifest["system"] == "double_integrator"
    assert manifest["pool_format_version"] == 1


def _assert_payload_key_order(pool_path: Path) -> None:
    with pool_path.open("rb") as pool_file:
        payload = pickle.load(pool_file)
    assert list(payload.keys()) == [
        "pool_format_version",
        "system",
        "seed",
        "n_scenes",
        "obstacle_centers",
        "obstacle_radii",
        "obstacle_active",
        "start",
        "goal",
        "init_velocity",
    ]
