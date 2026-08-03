"""v2.8.1 S1 G6 step 2 — offline second-difference probe of the soft encoder at the sigma knots (d=2.5, 3.0 m)
and rank-crossing loci, vs a generic (non-knot, non-crossing) baseline. Three closed-form soft-encoder obs
evaluations per probe (x-eps, x, x+eps); central 2nd difference ||obs(x+e) - 2 obs(x) + obs(x-e)||. No Hessian,
no second-order autograd, no filter recompute. A KNOT-LOCALIZED JUMP would show as a diverging ratio; a C1
smoothstep (bounded, discontinuous-only 2nd derivative) shows a bounded ratio. Off the training loop."""
import json
from pathlib import Path
import numpy as np, torch
from src.frameworks.jt_pncbf.train import load_effective_config, make_system
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_gates"; OUT.mkdir(parents=True, exist_ok=True)
DT = torch.float64
EPS = 1e-3

c = load_effective_config(); c["run"]["system"] = "quadrotor_3d"
c.setdefault("obs", {}).setdefault("quadrotor_3d", {})["encoder"] = "soft_topk"
sysm = make_system(c)
scenes = load_pool(POOL).scenes[:200]
bscene = batch_scenes(scenes, device="cpu", dtype=DT)
x0 = initial_states_from_batch(bscene).to(DT)
from src.common.observation import scene_obstacle_tensors
centers, radii, active = scene_obstacle_tensors(bscene, x0.device, DT)  # [B,K,2 or 3],[B,K],[B,K]
B = x0.shape[0]

def obs(x):
    return sysm.observation(x, bscene)

def sec_diff(x, direction, eps=EPS):
    o_p = obs(x + eps * direction); o_0 = obs(x); o_m = obs(x - eps * direction)
    return torch.linalg.norm((o_p - 2.0 * o_0 + o_m), dim=1) / (eps * eps)   # [B] curvature magnitude

bidx = torch.arange(B)
# nearest active obstacle per scene
p_xy = x0[:, :2]; c_xy = centers[..., :2]
rel = c_xy - p_xy.unsqueeze(1); sd = torch.linalg.norm(rel, dim=-1) - radii
sd = sd.masked_fill(~active, float("inf"))
j = sd.argmin(dim=1)                                                    # nearest active
r_hat = torch.zeros(B, x0.shape[1], dtype=DT)
near_rel = rel[bidx, j]; near_dir = near_rel / near_rel.norm(dim=1, keepdim=True).clamp_min(1e-9)
r_hat[:, :2] = near_dir                                                 # radial toward nearest obstacle (xy)

def place_at_surface(target_surf):
    # move the drone radially so the nearest obstacle's SURFACE distance == target_surf
    cur_surf = sd[bidx, j]
    shift = (cur_surf - target_surf).unsqueeze(1) * near_dir            # move drone toward/away
    x = x0.clone(); x[:, :2] = x0[:, :2] + shift
    return x
finite = torch.isfinite(sd[bidx, j])
res = {}
for name, surf in [("knot_2.5", 2.5), ("knot_3.0", 3.0)]:
    x = place_at_surface(surf)
    d = sec_diff(x, r_hat)[finite]
    res[name] = {"median": float(d.median()), "p90": float(d.quantile(0.90)), "max": float(d.max())}
# generic (non-knot): obstacle at surface 1.5 (well inside, sigma==1 flat), radial perturb
x_gen = place_at_surface(1.5); dg = sec_diff(x_gen, r_hat)[finite]
res["generic_nonknot"] = {"median": float(dg.median()), "p90": float(dg.quantile(0.90)), "max": float(dg.max())}
# rank-crossing: place drone equidistant (surface) to the two nearest obstacles, perturb along their difference
sd2 = sd.clone(); sd2[bidx, j] = float("inf"); j2 = sd2.argmin(dim=1)
has2 = torch.isfinite(sd2[bidx, j2]) & finite
rel2 = rel[bidx, j2]
cross_dir = torch.zeros(B, x0.shape[1], dtype=DT)
cd = (near_rel - rel2); cd = cd / cd.norm(dim=1, keepdim=True).clamp_min(1e-9)
cross_dir[:, :2] = cd
dc = sec_diff(x0, cross_dir)[has2]
res["rank_crossing"] = {"median": float(dc.median()), "p90": float(dc.quantile(0.90)), "max": float(dc.max()), "n": int(has2.sum())}
base = res["generic_nonknot"]["median"]
rec = {"gate": "G6_step2_second_difference", "eps": EPS, "n_scenes": int(finite.sum().item()),
       "second_difference_norm": res,
       "ratios_to_generic_median": {k: (res[k]["median"] / base if base > 0 else float("inf"))
                                    for k in ("knot_2.5", "knot_3.0", "rank_crossing")},
       "knot_localized_jump": bool(max(res["knot_2.5"]["p90"], res["knot_3.0"]["p90"]) > 1e3 * max(base, 1e-9))}
(OUT / "g6_second_difference.json").write_text(json.dumps(rec, indent=2) + "\n")
print("G6 step2 (2nd-diff of obs, curvature magnitude):")
for k, v in res.items(): print(f"  {k}: median {v['median']:.3e} p90 {v['p90']:.3e} max {v['max']:.3e}")
print("ratios to generic median:", {k: round(x, 2) for k, x in rec["ratios_to_generic_median"].items()})
print("knot_localized_jump:", rec["knot_localized_jump"])
