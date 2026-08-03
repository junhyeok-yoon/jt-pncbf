"""v2.8.1 S1 beta-screen — phantom-obstacle hazard measurement (on the beta=2.0 artifacts already on disk).

The candidate left after V3 (offset) and V4 (0% far-blind): does soft-rank MIXING dilute the nearest obstacle so
the certificate under-weights the thing it is about to hit? For the beta=2.0 value (022629 step_050000), roll out
the deployed policy+filter on the in-loop pool, find cylinder-collision episodes (V4 definition), and at the
10-step-lookback state report the NEAREST obstacle's soft-rank weight w_0 (slot-0 top-mass share) — vs the same
w_0 distribution over non-collision states. Concentration at LOW w_0 in the collision lookbacks implicates mixing
dilution; absence drops the hypothesis. Report, do not act."""
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.observation import scene_obstacle_tensors, soft_topk_obstacles, SOFT_DC, SOFT_INNER

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.1/set__20260802-022629__seed42/v2.8.1__oc__20260802-022629__seed42/checkpoints/step_050000.pt"
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NS, MAXS, LB, DC_SURF = 500, 200, 10, 0.0


def dist(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "mean": float(v.mean()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "frac_below_0.5": float((v < 0.5).mean())}


fw, cfg, ck = _load_framework(str(CK))
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None:
        m.to(DEV)
sysm = fw.system; k = int(sysm.k_obs); beta = float(sysm.soft_beta)
scenes = load_pool(INLOOP).scenes[:NS]
bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bs)
c = bs.obstacle_centers[..., :2]; r = bs.obstacle_radii; a = bs.obstacle_active
gr = float(cfg["env"]["goal_radius"])

surf_h, goal_h, w0_h = [], [], []
for _ in range(MAXS):
    x = x.detach()
    u_nom = fw.policy(x, bs); _res = fw.filter(x, u_nom, bs)
    u_safe = _res[0] if isinstance(_res, tuple) else _res
    p = x[:, :3]
    d = torch.norm(p[:, None, :2] - c, dim=2) - r
    d = torch.where(a, d, torch.full_like(d, float("inf")))
    surf_h.append(d.min(dim=1).values.detach().cpu().numpy())
    goal_h.append(torch.norm(p - bs.goal, dim=1).detach().cpu().numpy())
    cc, rr, aa = scene_obstacle_tensors(bs, x.device, x.dtype)
    _, _, tm = soft_topk_obstacles(p[:, :2], cc[..., :2], rr, aa, k, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER, return_indices=True)
    w0_h.append(tm[:, 0].detach().cpu().numpy())          # nearest-slot top-mass (w_0)
    x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), 0.05))
S = np.stack(surf_h, 0); G = np.stack(goal_h, 0); W0 = np.stack(w0_h, 0)   # [T,B]
T, B = S.shape

def first(mask):
    out = np.full(B, -1)
    for b in range(B):
        w = np.nonzero(mask[:, b])[0]
        out[b] = w[0] if w.size else -1
    return out
t_coll = first(S <= DC_SURF); t_reach = first(G < gr)
is_coll = (t_coll >= 1) & ((t_reach < 0) | (t_coll < t_reach))
coll_lb_w0, noncoll_w0 = [], []
for b in range(B):
    if is_coll[b] and t_coll[b] >= LB:
        coll_lb_w0.append(float(W0[t_coll[b] - LB, b]))
# non-collision reference: w_0 over all steps of non-collision episodes
for b in np.nonzero(~is_coll)[0]:
    noncoll_w0.extend(W0[:, b].tolist())

rec = {"measurement": "phantom_obstacle_w0", "beta": beta, "ckpt": str(CK),
       "n_collision_episodes": int(is_coll.sum()), "n_collision_lookback": len(coll_lb_w0),
       "w0_collision_lookback": dist(coll_lb_w0), "w0_noncollision_allsteps": dist(noncoll_w0),
       "interpretation_note": "concentration at low w_0 (frac_below_0.5, low median) in collision lookbacks vs "
                              "non-collision implicates mixing dilution; parity drops the hypothesis"}
(OUT / "phantom_w0_beta2.json").write_text(json.dumps(rec, indent=2) + "\n")
cl, nc = rec["w0_collision_lookback"], rec["w0_noncollision_allsteps"]
print(f"beta={beta} coll_eps={int(is_coll.sum())} lookback={len(coll_lb_w0)}")
print(f"  w_0 @ collision lookback: median={cl.get('median')} frac<0.5={cl.get('frac_below_0.5')} (n={cl['n']})")
print(f"  w_0 non-collision       : median={nc.get('median')} frac<0.5={nc.get('frac_below_0.5')} (n={nc['n']})")
print(f"wrote {OUT/'phantom_w0_beta2.json'}")
