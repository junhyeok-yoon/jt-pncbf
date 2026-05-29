from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.build_pools import load_base_config, load_pool
from src.eval.plotting import plot_scene_grid


POOL_DIR = REPO_ROOT / "data/secured_data/pools"
OUTPUT_DIR = REPO_ROOT / "data/verification"
POOL_PREVIEWS = [
    (
        POOL_DIR / "eval_inloop_di_n200_seed12345.pkl",
        OUTPUT_DIR / "pool_inloop_di_A.png",
    ),
    (
        POOL_DIR / "eval_full_di_n500_seed23456.pkl",
        OUTPUT_DIR / "pool_full_di_A.png",
    ),
]


def main() -> int:
    config = load_base_config()
    outputs = []

    for pool_path, output_path in POOL_PREVIEWS:
        if not pool_path.exists():
            raise FileNotFoundError(f"Missing pool file: {pool_path}")
        pool = load_pool(pool_path)
        if len(pool.scenes) < 16:
            raise ValueError(f"Pool {pool_path} has fewer than 16 scenes.")
        plot_scene_grid(
            scenes=pool.scenes[:16],
            output_path=output_path,
            config=config,
            role="Eval pool preview",
            system_name=pool.system,
            letter="A",
            start_index=0,
        )
        outputs.append(output_path)

    for output in outputs:
        if not output.exists():
            raise FileNotFoundError(f"Missing output PNG: {output}")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
