"""Smoke for the two changes (READ-ONLY ckpt): (a) lg_reg_eps=0 parity, (b) chattering reduction at
low-|v| near-goal states (eps=0 vs 0.005), (c) goal_speed_radius re-scoring 0.30/0.50/inf."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np, torch
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint as jt_load, make_system, load_effective_config
from src.common.value_net import make_h_fn
from src.common.filter_hardnet import HardNetFilter, _cbf_terms, _base_projection, _base_alpha, _hardnet_params
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.eval.rollout import rollout_eval
from src.envs.scene_batch import batch_scenes

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CKPT = REPO / "data/v2.2.2__20260619-083424__seed42/checkpoints/step_028000.pt"   # unicycle injection best
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
fw, _, _ = jt_load(CKPT)
cfg = load_effective_config()                       # unicycle, lg_reg_eps 0.005, goal_speed_radius 0.50
system = make_system(cfg)
fw.value_net.to(DEV).eval(); fw.policy_net.to(DEV).eval()
h_fn = make_h_fn(fw.value_net, system)
dt = float(cfg["env"]["dt"])
cfg0 = copy.deepcopy(cfg); cfg0["filter"]["lg_reg_eps"] = 0.0
cfg5 = copy.deepcopy(cfg); cfg5["filter"]["lg_reg_eps"] = 0.005
pool = load_pool(REPO / "data/secured_data/pools/eval_inloop_unicycle_n200_seed12345.pkl")

# ---- (a) parity: _base_projection with lg_reg_eps=0 == original (denom = ||n||^2 + epsilon^2) ----
bs = batch_scenes(pool.scenes[:64], device=DEV, dtype=torch.float32)
from src.envs.scene_batch import initial_states_from_batch
x = system.wrap_state(initial_states_from_batch(bs))
# push to some near-obstacle states
with torch.no_grad():
    from src.common.rk4 import rk4_step
    for _ in range(8): x = rk4_step(system, x, fw.policy(x, bs), dt)
u_nom = fw.policy(x.detach(), bs)
h, lf, lg = _cbf_terms(system, h_fn, x.detach(), bs, u_nom, create_graph=False)
p0 = _hardnet_params(cfg0); p5 = _hardnet_params(cfg5)
row_upper = -lf - _base_alpha(h, p0) * h
bounds = system.u_bounds.to(DEV, torch.float32)
proj0 = _base_projection(u_nom, lg, row_upper, bounds, p0)
proj5 = _base_projection(u_nom, lg, row_upper, bounds, p5)
# manual original denom (||lg||^2 + epsilon^2), no lg_reg
lhs = torch.sum(lg*u_nom,1); viol = torch.relu(lhs-row_upper); den_orig = torch.sum(lg*lg,1)+p0.epsilon**2
manual = torch.clamp(u_nom - lg*(viol/den_orig).unsqueeze(1), bounds[:,0], bounds[:,1])
print(f"(a) PARITY eps=0 vs original-denom: max|diff| = {float((proj0-manual).abs().max()):.3e}  (==0 -> bit-identical)")
print(f"    eps=0 vs eps=0.005 differ: max|diff| = {float((proj0-proj5).abs().max()):.4f}  (>0 -> eps active)")
print(f"    ||L_g h|| range on these states: [{float(lg.norm(dim=1).min()):.4f}, {float(lg.norm(dim=1).max()):.4f}]")

# ---- (b) chattering at low-|v| near-goal states: roll filter eps=0 vs 0.005, compare |du| + saturation ----
g = torch.Generator(device=DEV).manual_seed(1)
B = 128
sc = batch_scenes(pool.scenes[:B], device=DEV, dtype=torch.float32)
goal = torch.as_tensor(np.stack([s.goal for s in pool.scenes[:B]]), device=DEV, dtype=torch.float32)
off = 0.10*(2*torch.rand(B,2,generator=g,device=DEV)-1)                      # within goal_radius-ish
theta = (2*torch.rand(B,1,generator=g,device=DEV)-1)*np.pi
spd = 0.05+0.20*torch.rand(B,1,generator=g,device=DEV)                       # low |v| 0.05-0.25 (singular region)
x0 = system.wrap_state(torch.cat([goal+off, theta, spd], dim=1))
umax = system.u_bounds.to(DEV,torch.float32)
def roll(cfgx):
    filt = HardNetFilter(system, h_fn, cfgx)
    with torch.no_grad():
        r = rollout_eval(system, lambda xx,s: fw.policy(xx,s), lambda xx,u,s: filt(xx,s,u), sc, x0, 40, dt, cfgx)
    us = r.u_safe                                                            # [T,B,2]
    du = (us[1:]-us[:-1]).norm(dim=2).mean().item()                         # mean step-to-step control change
    sat = ((us.abs() >= (umax[:,1]-1e-3)).any(dim=2).float().mean()).item() # frac steps at action bound
    return du, sat
du0,sat0 = roll(cfg0); du5,sat5 = roll(cfg5)
print(f"(b) CHATTERING near-goal low-|v| (T=40, B={B}):")
print(f"    mean |du|:        eps=0 {du0:.4f}  ->  eps=0.005 {du5:.4f}  ({100*(du5-du0)/max(du0,1e-9):+.1f}%)")
print(f"    saturation frac:  eps=0 {sat0:.4f}  ->  eps=0.005 {sat5:.4f}")

# ---- (c) goal_speed_radius re-scoring 0.30 / 0.50 / inf on a smoke eval (n=64) ----
print("(c) RE-SCORING (n=64 inloop, eps=0.005 filter):")
fcfg = copy.deepcopy(cfg5)
from src.eval.build_pools import EvaluationPool
sub = EvaluationPool(name="smoke", system=pool.system, n_scenes=64, seed=pool.seed, scenes=pool.scenes[:64])
for sr in (0.30, 0.50, 1e9):
    fcfg["env"]["goal_speed_radius"] = sr
    r = evaluate(fw, sub, fcfg, mode="smoke", step=28000, ckpt_name="s", include_lqr_baseline=False, eval_batch_size=64).eval_row
    print(f"    speed_radius {sr:>6}: reach {float(r['reach']):.3f}  cps {float(r['cps']):+.3f}  stuck {float(r['stuck']):.3f}  timeout {float(r['timeout']):.3f}  coll {float(r['collision']):.3f}")
print("SMOKE_DONE")
