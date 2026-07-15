"""v2.6.0 Stage 0 M5 — LQR/PD nominal baseline (filter-off) on the full quadrotor pool.
Reports cps + outcome fractions + scene-bootstrap CI (04_eval Sec 5) via the committed
src/eval/dual_arm.nominal_eval. Comparator baseline for the quadrotor-direct system (changes.md Sec 5)."""
import json
from pathlib import Path
from typing import Mapping

import torch
import yaml

from src.common.observation import scene_goal_tensor
from src.eval.build_pools import load_pool
from src.eval.dual_arm import nominal_eval
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"


def _cfg() -> dict:
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d

    return m(b, e)


def main() -> None:
    cfg = _cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = make_system(cfg)
    system.u_bounds = system.u_bounds.to(device=dev, dtype=torch.float32)
    scenes = load_pool(POOL).scenes
    action_fn = lambda x, sc: system.lqr_action(x, scene_goal_tensor(sc, x))
    res = nominal_eval(scenes, action_fn, cfg, system, dev, chunk=250)
    print(f"n={res['n']} wall={res['wall_s']}s")
    print(
        "cps_v2=%.4f cps_legacy=%.4f  reach=%.4f coll=%.4f oob=%.4f stuck=%.4f timeout=%.4f  "
        "inf_v2=%.4f  CI=[%.4f, %.4f]"
        % (res["cps_v2"], res["cps_legacy"], res["reach"], res["collision"], res["oob"],
           res["stuck"], res["timeout"], res["inf_v2"], res["cps_v2_ci"][0], res["cps_v2_ci"][1])
    )
    json.dump(res, open(SP / "quadrotor_baseline.json", "w"), indent=2)
    print("saved:", SP / "quadrotor_baseline.json")


if __name__ == "__main__":
    main()
