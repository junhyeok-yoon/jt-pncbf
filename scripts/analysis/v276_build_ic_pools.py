"""v2.7.6 M1/M2 — build the eval-IC z-split pools and prove the sampler change inert on the default path.

Three builds, all quadrotor_3d / d2r / n2000, via the canonical build_pool + write_pool path (mode='eval'):
  (gate)      seed 23456, NO ic_eval_z block  -> pool_sha256 MUST equal the canonical 0ef3751b... (M1 gate).
              If it does not, the sampler edit perturbed the default eval path -> HALT.
  (baseline)  seed 42,    NO ic_eval_z block  -> full-range eval pool (R6.1 baseline).
  (treatment) seed 42,    ic_eval_z present   -> recoverable-z eval pool (R6.1 treatment).

The committed configs are NOT changed: the ic_eval_z block is injected into an in-memory config copy only
(no config promotion). Bounds are read from data/runs/v2.7.6/registered_params.json (registered before data).
Outputs under data/runs/v2.7.6/pools/. No git, no securing.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Mapping

import yaml

from src.eval.build_pools import (
    PoolSpec, build_pool, write_pool, load_pool, pool_variant,
)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SYSTEM = "quadrotor_3d"
OUT = REPO / "data/runs/v2.7.6/pools"
OUT.mkdir(parents=True, exist_ok=True)
CANONICAL_SHA_PREFIX = "0ef3751b"
REG = json.loads((REPO / "data/runs/v2.7.6/registered_params.json").read_text())
IC_EVAL_Z = REG["ic_eval_z_bounds"]


def _merged_config() -> dict:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d

    return m(b, e)


def _with_ic_eval_z(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["quadrotor_3d"]["ic_eval_z"] = dict(IC_EVAL_Z)
    return cfg


def build(cfg: dict, name: str, seed: int) -> dict:
    spec = PoolSpec(name=name, n_scenes=2000, seed=seed)
    variant = pool_variant(cfg, SYSTEM)                       # "d2r"
    pool = build_pool(cfg, SYSTEM, spec)
    art = write_pool(pool, cfg, output_dir=OUT, git_commit="uncommitted-v2.7.6", variant=variant)
    reloaded = load_pool(art.pool_path)
    assert reloaded.n_scenes == 2000 == len(reloaded.scenes) and reloaded.system == SYSTEM
    rec = {"name": name, "seed": seed, "stem": art.pool_path.stem, "sha256": art.sha256,
           "pool_path": str(art.pool_path), "manifest_path": str(art.manifest_path),
           "ic_eval_z": cfg["env"]["quadrotor_3d"].get("ic_eval_z")}
    print(f"[{name} seed={seed}] {art.pool_path.name}\n    sha256={art.sha256}")
    return rec


def main() -> None:
    base = _merged_config()
    results = {}

    # (gate) default path, seed 23456 -> must reproduce the canonical pool SHA bit-identically
    gate = build(base, "full", 23456)
    ok = gate["sha256"].startswith(CANONICAL_SHA_PREFIX)
    gate["inertness_gate_pass"] = ok
    results["gate_seed23456_fullrange"] = gate
    print(f"[M1 INERTNESS GATE] rebuilt-seed23456 sha {gate['sha256'][:16]} vs canonical "
          f"{CANONICAL_SHA_PREFIX}...  {'PASS' if ok else 'FAIL'}")
    if not ok:
        (OUT / "build_summary.json").write_text(json.dumps(results, indent=2) + "\n")
        raise SystemExit("HALT M1: default eval path is NOT byte-identical — sampler edit perturbed it.")

    # (baseline / treatment) matched screen seed 42
    results["baseline_seed42_fullrange"] = build(base, "fullrange", 42)
    results["treatment_seed42_evalicz"] = build(_with_ic_eval_z(base), "evalicz", 42)

    (OUT / "build_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("M2 DONE — pools written to", OUT)


if __name__ == "__main__":
    main()
