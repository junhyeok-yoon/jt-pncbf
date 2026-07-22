"""v2.7.3 amendment — render the fixed 3-panel quadrotor_3d V_hat CBF contour (plotting.py recipe) for a run's
checkpoints, post-hoc (pure function of a checkpoint; never touches a running job or n_steps).

best.pt   -> figures/cbf_contour.png            (the 04_eval §7.5 name for the secured snapshot)
final.pt  -> figures/final_cbf_contour.png
step_*.pt -> figures/inloop/step_<NNNNNN>_cbf_contour.png   (the 2-D cadence naming) unless --best-final-only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from src.eval.build_pools import DEFAULT_OUTPUT_DIR, load_pool, pool_stem, pool_variant
from src.eval.plotting import plot_quadrotor3d_cbf_contour
from src.common.value_net import ValueNetEnsemble
from src.frameworks.jt_pncbf.train import make_system

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--best-final-only", action="store_true")
a = ap.parse_args()
run_dir = Path(a.run_dir)
ckdir = run_dir / "checkpoints"

# scene 0 of the in-loop d2 pool (fixed recipe), resolved from the run's own config
cfg0 = torch.load(ckdir / "best.pt", map_location="cpu", weights_only=False)["config"]
stem = pool_stem("inloop", "quadrotor_3d", int(cfg0["eval"]["in_loop"]["n"]),
                 int(cfg0["eval"]["in_loop"]["seed"]), "random", pool_variant(cfg0, "quadrotor_3d"))
scene0 = load_pool(DEFAULT_OUTPUT_DIR / f"{stem}.pkl").scenes[0]


def render(ckpt_path: Path, out_path: Path, role: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]; system = make_system(cfg)
    vnet = ValueNetEnsemble(system.obs_dim, cfg); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    plot_quadrotor3d_cbf_contour(scene0, out_path, cfg, system, vnet, role)
    print("rendered", out_path)


targets = [(ckdir / "best.pt", run_dir / "figures" / "cbf_contour.png", "best.pt"),
           (ckdir / "final.pt", run_dir / "figures" / "final_cbf_contour.png", "final.pt")]
if not a.best_final_only:
    for c in sorted(ckdir.glob("step_*.pt")):
        step = int(re.search(r"step_(\d+)", c.name).group(1))
        targets.append((c, run_dir / "figures" / "inloop" / f"step_{step:06d}_cbf_contour.png", f"step {step}"))
for ckpt_path, out_path, role in targets:
    if ckpt_path.exists():
        render(ckpt_path, out_path, f"{role} contour")
