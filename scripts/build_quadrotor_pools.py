"""v2.6.0 Stage 0 M4 — build the committed planar-quadrotor eval pools (in-loop n500 + full n2000)
with manifest + SHA-256. New system => this is the one legitimate pool-generation event (changes.md §5).
Writes to data/secured_data/pools/ under the canonical stems
`eval_{inloop,full}_quadrotor-planar_n{500,2000}_seed{12345,23456}`. HEAD sha is read from .git files
(no git subprocess; honors the Stage-0 'No git' constraint)."""
from pathlib import Path
from typing import Mapping

import yaml

from src.eval.build_pools import build_pool, default_pool_specs, load_pool, write_pool
from src.eval.build_pools import sampler_param_snapshot  # noqa: F401 (documented in manifest)

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SYSTEM = "quadrotor_planar"


def _merged_config() -> dict:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d

    return m(b, e)


def _head_sha() -> str:
    head = (REPO / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        return (REPO / ".git" / ref).read_text().strip()
    return head


def main() -> None:
    config = _merged_config()
    sha = _head_sha()
    out_dir = REPO / "data/secured_data/pools"
    for spec in default_pool_specs(config):
        pool = build_pool(config, SYSTEM, spec)
        art = write_pool(pool, config, output_dir=out_dir, git_commit=sha)
        # self-check: reload, SHA matches manifest, sampler snapshot present
        reloaded = load_pool(art.pool_path)
        manifest = yaml.safe_load(art.manifest_path.read_text())
        ok_sha = manifest["pool_sha256"] == art.sha256
        ok_n = reloaded.n_scenes == spec.n_scenes == len(reloaded.scenes)
        ok_sys = reloaded.system == SYSTEM
        ok_snap = "sampler_params" in manifest and "env" in manifest["sampler_params"]
        print(f"[{spec.name}] {art.pool_path.name}  n={spec.n_scenes} seed={spec.seed}")
        print(f"    sha256={art.sha256}")
        print(f"    reload_ok={ok_n} system_ok={ok_sys} sha_match={ok_sha} snapshot_ok={ok_snap}")
        assert ok_sha and ok_n and ok_sys and ok_snap, "M4 pool self-check FAILED"
    print("M4 DONE: both quadrotor pools written + self-checked")


if __name__ == "__main__":
    main()
