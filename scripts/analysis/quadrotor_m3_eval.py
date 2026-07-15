"""v2.6.0 Stage 1 M3 — pre-JT learned-filter eval. HardNet on the M1-learned V_hat, filter-ON, NOMINAL
policy (lqr_action), full quadrotor pool. Isolates the learned filter's contribution before policy
co-training. Reports cps + outcomes + scene-bootstrap CI vs the M5 nominal baseline (coll 0.409, cps -0.227)."""
import json
import sys
from pathlib import Path

import torch

from src.common.observation import scene_goal_tensor
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import _rollout, _arm_summary
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system = make_system(cfg)
system.u_bounds = system.u_bounds.to(device=dev, dtype=torch.float32)
vnet = ValueNetEnsemble(system.obs_dim, cfg).to(device=dev, dtype=torch.float32)
vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
h_fn = make_h_fn(vnet, system)
scenes = load_pool(POOL).scenes

un_fn = lambda x, bs: system.lqr_action(x, scene_goal_tensor(bs, x))   # NOMINAL policy
import time
t0 = time.time()
ep = _rollout(scenes, un_fn, cfg, h_fn, system, dev, filtered=True, chunk=250)
res = _arm_summary(ep, time.time() - t0)
print(f"M3 pre-JT learned filter (nominal + HardNet on V_hat @ {run_dir.name}/best.pt step {ck.get('step')}), n={res['n']}")
print("cps_v2=%.4f cps_legacy=%.4f  reach=%.4f coll=%.4f oob=%.4f stuck=%.4f timeout=%.4f  inf_v2=%.4f "
      "inf_empty=%.4f inf_sing_viol=%.4f  CI=[%.4f,%.4f]"
      % (res["cps_v2"], res["cps_legacy"], res["reach"], res["collision"], res["oob"], res["stuck"],
         res["timeout"], res["inf_v2"], res["inf_empty"], res["inf_singular_violated"],
         res["cps_v2_ci"][0], res["cps_v2_ci"][1]))
print("  vs M5 nominal (filter-off): cps_v2 -0.2270, collision 0.4090")
json.dump(res, open(SP / "quadrotor_m3_eval.json", "w"), indent=2)
print("saved", SP / "quadrotor_m3_eval.json")
