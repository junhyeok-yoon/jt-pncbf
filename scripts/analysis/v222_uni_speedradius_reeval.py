import sys, json, copy
from pathlib import Path
import torch
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as jt_load
REPO=Path("/home/junhyeok/MIT/jt-pncbf")
POOL=REPO/"data/eval_pools/eval_full_unicycle_n2000_seed23456.pkl"
ckpt=Path(sys.argv[1]); label=sys.argv[2]
DEV="cuda" if torch.cuda.is_available() else "cpu"
pool=load_pool(POOL)
fw,fcfg,fck=jt_load(ckpt); fw.value_net.to(DEV); fw.policy_net.to(DEV)
for sr in (0.30,0.50,1e9):
    c=copy.deepcopy(fcfg); c["env"]["goal_speed_radius"]=float(sr)
    c.setdefault("filter",{})["lg_reg_eps"]=0.0
    r=evaluate(fw,pool,c,mode="reeval",step=int(fck.get("step",0)),ckpt_name=label,include_lqr_baseline=False,eval_batch_size=500).eval_row
    print(f"RESULT {label} sr={sr} "+json.dumps({k:r.get(k) for k in ("cps","cps_ci_lo","cps_ci_hi","reach","collision","stuck","timeout","infeasibility")}))
