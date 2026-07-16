"""Backfill per-eval CBF contour PNGs for runs whose in-loop evals predate the auto-contour hook.
For each step_XXXXXX.pt in <run_dir>/checkpoints that has a matching figures/inloop/step_XXXXXX_grid_A.png
but NO step_XXXXXX_cbf_contour.png, regenerate the contour from that checkpoint's value net using the SAME
inloop pool scenes the live hook uses (scenes[:2]). CPU by default (does not contend with a running GPU
train). Read-only on checkpoints; only writes the missing contour PNGs."""
import sys
from pathlib import Path

import torch

from src.common.value_net import ValueNetEnsemble
from src.eval.build_pools import load_pool
from src.eval.plotting import plot_quadrotor_cbf_contour
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")


def _inloop_pool_path(cfg):
    name = cfg["run"]["system"].replace("_", "-")
    il = cfg["eval"]["in_loop"]
    n = il.get("n", il.get("n_scenes", 500))
    seed = il.get("seed", il.get("pool_seed", 12345))
    return REPO / f"data/secured_data/pools/eval_inloop_{name}_n{n}_seed{seed}.pkl"


def main():
    run_dir = Path(sys.argv[1])
    dev = torch.device("cpu")
    ckdir = run_dir / "checkpoints"
    figdir = run_dir / "figures/inloop"
    steps = sorted(int(p.stem.split("_")[1]) for p in ckdir.glob("step_*.pt"))
    if not steps:
        print("no step checkpoints; nothing to backfill"); return
    ck0 = torch.load(ckdir / f"step_{steps[0]:06d}.pt", map_location="cpu", weights_only=False)
    cfg = ck0["config"]
    system = make_system(cfg)
    try:
        pool_path = _inloop_pool_path(cfg)
        scenes = load_pool(pool_path).scenes
    except Exception as e:
        # fall back to the run's pool_manifest inloop pool name if the derived path misses
        import json
        man = json.load(open(run_dir / "pool_manifest.json"))
        pool_path = REPO / "data/secured_data/pools" / (Path(man["inloop"]["path"]).name if "path" in man.get("inloop", {}) else "")
        scenes = load_pool(pool_path).scenes
    print(f"backfill {run_dir.name}: {len(steps)} steps, inloop pool {pool_path.name} ({len(scenes)} scenes)", flush=True)
    made = skipped = 0
    for s in steps:
        grid = figdir / f"step_{s:06d}_grid_A.png"
        out = figdir / f"step_{s:06d}_cbf_contour.png"
        if not grid.exists() or out.exists():
            skipped += 1; continue
        ck = torch.load(ckdir / f"step_{s:06d}.pt", map_location="cpu", weights_only=False)
        vnet = ValueNetEnsemble(system.obs_dim, ck["config"]).to(dev, torch.float32)
        vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
        for p in vnet.parameters():
            p.requires_grad_(False)
        plot_quadrotor_cbf_contour(scenes, out, ck["config"], system, vnet,
                                   role=f"backfill in-loop CBF contour @ step {s}")
        made += 1
        print(f"  step {s:06d}: contour written", flush=True)
    print(f"DONE {run_dir.name}: made={made} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
