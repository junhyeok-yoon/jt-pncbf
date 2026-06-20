"""Focused chattering check: at genuinely-singular (low ||L_g h||) states, does eps=0.005 reduce the
filter correction blow-up vs eps=0? Roll the policy+filter to collect real states, then compare."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np, torch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as jt_load, make_system, load_effective_config
from src.common.value_net import make_h_fn
from src.common.filter_hardnet import _cbf_terms, _base_alpha, _base_projection, _box_aware_projection, _hardnet_params
from src.common.rk4 import rk4_step
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/v2.2.2__20260619-083424__seed42/checkpoints/step_028000.pt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(0)
fw,_,_ = jt_load(CKPT); cfg = load_effective_config(); system = make_system(cfg)
fw.value_net.to(DEV).eval(); fw.policy_net.to(DEV).eval(); h_fn = make_h_fn(fw.value_net, system)
dt = float(cfg["env"]["dt"]); bounds = system.u_bounds.to(DEV, torch.float32)
p0 = _hardnet_params({**cfg, "filter": {**cfg["filter"], "lg_reg_eps": 0.0}})
p5 = _hardnet_params({**cfg, "filter": {**cfg["filter"], "lg_reg_eps": 0.005}})
pool = load_pool(REPO/"data/secured_data/pools/eval_inloop_unicycle_n200_seed12345.pkl")
bs = batch_scenes(pool.scenes[:200], device=DEV, dtype=torch.float32)

# collect real states across a rollout (no goal-termination here; just dynamics) to sample the state dist
xs=[]; x = system.wrap_state(initial_states_from_batch(bs))
with torch.no_grad():
    for t in range(60):
        xs.append(x); x = system.wrap_state(rk4_step(system, x, fw.policy(x, bs), dt))
X = torch.cat(xs,0); SC = batch_scenes([s for s in pool.scenes[:200]]*60, device=DEV, dtype=torch.float32)
u_nom = fw.policy(X.detach(), SC)
h, lf, lg = _cbf_terms(system, h_fn, X.detach(), SC, u_nom, create_graph=False)
lgn = lg.norm(dim=1); speed = X[:,3].abs()
row_upper = -lf - _base_alpha(h, p0)*h
def proj(p):
    base = _base_projection(u_nom, lg, row_upper, bounds, p)
    box,_ = _box_aware_projection(u_nom, base, lg, row_upper, bounds)
    return box
u0 = proj(p0); u5 = proj(p5)
c0 = (u0-u_nom).norm(dim=1); c5 = (u5-u_nom).norm(dim=1)                 # correction magnitudes
sat0 = (u0.abs() >= (bounds[:,1]-1e-3)).any(1).float(); sat5 = (u5.abs() >= (bounds[:,1]-1e-3)).any(1).float()
active = (h > -0.5)                                                       # filter-relevant (near boundary) states
for name,msk in [("low-||Lgh||(<0.2) & active", (lgn<0.2)&active), ("low-|v|(<0.3) & active", (speed<0.3)&active), ("all active", active)]:
    n=int(msk.sum())
    if n==0: print(f"  {name}: n=0"); continue
    print(f"  {name}: n={n} | mean correction |u_safe-u_nom|: eps0 {float(c0[msk].mean()):.3f} -> eps0.005 {float(c5[msk].mean()):.3f}"
          f" | saturation: eps0 {float(sat0[msk].mean()):.3f} -> eps0.005 {float(sat5[msk].mean()):.3f}")
print(f"  finite: {bool(torch.isfinite(u5).all())} | ||Lgh|| min {float(lgn.min()):.4f} | frac states ||Lgh||<0.2: {float((lgn<0.2).float().mean()):.3f}")
print("CHATTER_DONE")
