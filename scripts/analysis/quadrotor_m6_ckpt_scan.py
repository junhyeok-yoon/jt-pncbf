"""v2.6.0 M6 — D5 full-pool checkpoint re-selection (04_eval S6.2). Score every retained step_*.pt of the
M6 run on the full pool n2000/seed23456 (= the M6/M3 eval pool) via the canonical dual_arm filtered rollout
(cps_v2 + scene-bootstrap CI), and report any checkpoint above best.pt@46500 cps 0.292. Read-only; loads the
LEARNED policy (pi_state) + value (v_s_state) from each ckpt. Facts only."""
import json
import sys
import time
from pathlib import Path

import torch

from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import _rollout, _arm_summary
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
BEST_STEP = 46500
BEST_CPS = 0.2920


def score(ckpt_path, scenes, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system)
    un_fn = lambda x, bs: policy(system.observation(x, bs))
    ep = _rollout(scenes, un_fn, cfg, h_fn, system, dev, filtered=True, chunk=250)
    return _arm_summary(ep, 0.0), int(ck.get("step", -1))


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = load_pool(POOL).scenes
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt")) + [run_dir / "checkpoints" / "best.pt"]
    rows = []; t0 = time.time()
    for cp in ckpts:
        res, step = score(cp, scenes, dev)
        rows.append(dict(name=cp.name, step=step, cps_v2=res["cps_v2"], ci=res["cps_v2_ci"],
                         reach=res["reach"], collision=res["collision"], timeout=res["timeout"]))
        print(f"  {cp.name:>16} step={step:>6} cps_v2={res['cps_v2']:+.4f} [{res['cps_v2_ci'][0]:+.4f},{res['cps_v2_ci'][1]:+.4f}] "
              f"reach={res['reach']:.4f} coll={res['collision']:.4f} timeout={res['timeout']:.4f}", flush=True)
    step_rows = [r for r in rows if r["name"].startswith("step_")]
    top = max(step_rows, key=lambda r: r["cps_v2"])
    above = [r for r in step_rows if r["cps_v2"] > BEST_CPS]
    print(f"\nD5: {len(ckpts)} checkpoints scored on full n2000 in {time.time()-t0:.0f}s")
    print(f"  top step_*: {top['name']} step {top['step']} cps_v2={top['cps_v2']:.4f} (CI [{top['ci'][0]:.4f},{top['ci'][1]:.4f}])")
    print(f"  above best.pt@{BEST_STEP} cps {BEST_CPS}: {len(above)} " +
          (", ".join(f"{r['name']}({r['cps_v2']:.4f})" for r in above) if above else "NONE"))
    json.dump(dict(best_step=BEST_STEP, best_cps_ref=BEST_CPS, rows=rows, top_step=top,
                   n_above_best=len(above)), open(SP / "quadrotor_m6_ckpt_scan.json", "w"), indent=2)
    print(f"saved {SP / 'quadrotor_m6_ckpt_scan.json'}")


if __name__ == "__main__":
    main()
