"""v2.6.0 Stage 1+2 M6 — P3 collinear-residual gate (theory note E2 / changes.md P3).

Claim under test: collision drops WITHOUT the collinear brake-only failure transferring to the learned
V^{h,pi} — i.e. the position-only degeneracy set S_h (rem:collinear / Prop 3.2: thrust axis Re orthogonal
to obstacle-normal grad phi, so a position-only filter has ZERO lateral first-order authority, ||L_g phi||~0)
does NOT carry over to V_hat. Operational test at the RESIDUAL COLLISION STATES of the filtered learned
policy on the full pool:
  - S_h membership (geometric): position-only ||L_g phi|| < eps_g  (the collinear brake-only set).
  - Transfer test: learned ||L_g V_hat|| at those same states. If ||L_g V_hat|| >> eps_g where ||L_g phi||~0,
    S_h did NOT transfer -> the learned certificate keeps lateral authority; residual collisions are NOT the
    brake-only trap. P3 CONFIRMED iff (a) collision dropped vs M3 pre-JT AND (b) S_h not transferred.

Rolls the filtered learned best.pt policy on the frozen full pool (n2000/seed23456, same as M3/M6 final),
captures the first-collision state per collision episode, and measures ||L_g phi|| (position-only),
||L_g h_star|| (c=0.3), ||L_g V_hat|| (learned) via _cbf_terms (the exact HardNet primitive). Facts only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.quadrotor_barrier import make_barrier_fn
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
EPS_G = 0.05   # committed lg_authority eps_g (exp_config loss.value.lg_authority)


def lg_norm(system, h_fn, x, bs):
    """||L_g h(x)|| = ||grad_h(x)^T g(x)|| via the HardNet primitive (create_graph=False)."""
    _, _, lg = _cbf_terms(system, h_fn, x, bs, torch.zeros(x.shape[0], system.action_dim,
                                                            device=x.device, dtype=x.dtype),
                          create_graph=False)
    return torch.linalg.norm(lg.detach(), dim=1)


def collinear_cos(system, phi_fn, x, bs):
    """|cos(angle(grad_p phi, Re))| — the DIRECT geometric S_h test (rem:collinear): the collinear
    brake-only set is grad_p phi PARALLEL to Re (obstacle-normal aligned with the thrust axis), so the
    only first-order-admissible thrust action is braking along the normal, no lateral component. Returns
    |cos| in [0,1]; ~1 = collinear (S_h), ~0 = thrust axis orthogonal to the normal (full lateral freedom).
    grad_p phi via autograd on the position channels; Re = (-sin th, cos th)."""
    from src.common.quadrotor_barrier import thrust_axis
    xr = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        phi = phi_fn(xr, bs).reshape(-1)
        gp = torch.autograd.grad(phi.sum(), xr, create_graph=False)[0][:, :2]   # d phi / d p
    n = torch.linalg.norm(gp, dim=1, keepdim=True).clamp_min(1e-12)
    gp_hat = gp / n
    Re = thrust_axis(x)                                                          # [B,2]
    return torch.abs(torch.sum(gp_hat * Re, dim=1)).detach()                     # |cos|


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else max(REPO.glob("data/v2.6.0__*seed42"))
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev)
    policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system)                                          # learned V_hat -> h
    h_scale = float(cfg["env"].get("h_scale", 1.0))
    c_gain = float(cfg["env"]["quadrotor_planar"]["c_gain"])
    phi_fn = make_barrier_fn(c_gain, h_scale, position_only=True)           # phi(p,o) position-only
    hstar_fn = make_barrier_fn(c_gain, h_scale, position_only=False)        # h_star = phi + c v^T Re

    scenes = load_pool(POOL).scenes
    params = _hardnet_params(cfg); bounds = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"]); chunk = 250

    coll_states = []      # first-collision state per collision episode
    n_coll = 0; n_total = 0
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float(); B = x.shape[0]; n_total += B
        states = [x]
        with torch.no_grad():
            for _ in range(max_steps):
                un = policy(system.observation(x, bs))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                alpha = _base_alpha(h, params); row = -lf - alpha * h
                proj = _base_projection(un, lg, row, bounds, params)
                u, _ = _box_aware_projection(un, proj, lg, row, bounds)
                x = rk4_step(system, x, u, dt); states.append(x)
        S = torch.stack(states, 0)                                          # [T+1, B, 6]
        masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        collided = masks.collided                                           # [T+1, B] bool
        for i in range(B):
            if res.outcome[i] == "collision":
                n_coll += 1
                t_hit = int(torch.argmax(collided[:, i].to(torch.int8)))     # first True step
                coll_states.append((S[t_hit, i].clone(), s0 + i))

    if not coll_states:
        print("no residual collisions on the full pool — P3 vacuously holds (0 collinear residuals).")
        json.dump({"n_total": n_total, "n_collision": 0}, open(SP / "quadrotor_p3.json", "w"), indent=2)
        return

    # rebuild a batched scene aligned to the collision states (each from its own scene) and measure ||L_g .||
    X = torch.stack([cs for cs, _ in coll_states]).to(dev)
    idxs = [j for _, j in coll_states]
    bs_c = batch_scenes([scenes[j] for j in idxs], device=dev, dtype=torch.float32)
    lg_phi = lg_norm(system, phi_fn, X, bs_c).cpu().numpy()
    lg_hstar = lg_norm(system, hstar_fn, X, bs_c).cpu().numpy()
    lg_vhat = lg_norm(system, h_fn, X, bs_c).cpu().numpy()
    cos_ang = collinear_cos(system, phi_fn, X, bs_c).cpu().numpy()          # DIRECT collinear geometry |cos(grad_p phi, Re)|

    # S_h (collinear) membership by the DIRECT geometry: |cos(grad_p phi, Re)| > 0.9 (thrust axis within
    # ~26deg of the obstacle normal -> brake-dominated, minimal lateral first-order authority). (Note:
    # ||L_g phi|| is structurally ~0 at ALL states on this thrust-underactuated system, Thm 5.2 — thrust
    # moves velocity not position at first order — so it CANNOT discriminate the collinear subset; the
    # angle test does.)
    Sh = cos_ang > 0.9
    def q(a): return dict(min=float(np.min(a)), p5=float(np.percentile(a, 5)),
                          median=float(np.median(a)), max=float(np.max(a)))
    # transfer test: on the S_h (collinear) collision states, does learned V_hat keep lateral authority?
    vhat_on_Sh = lg_vhat[Sh] if Sh.any() else np.array([])
    vhat_degen_on_Sh = float((vhat_on_Sh < EPS_G).mean()) if vhat_on_Sh.size else 0.0

    out = {
        "run": run_dir.name, "ckpt": "best.pt", "step": int(ck.get("step", -1)),
        "n_total": n_total, "n_collision": n_coll, "collision_rate": n_coll / n_total,
        "eps_g": EPS_G,
        "lg_phi_at_collisions": q(lg_phi),
        "lg_hstar_at_collisions": q(lg_hstar),
        "lg_vhat_at_collisions": q(lg_vhat),
        "collinear_cos_at_collisions": q(cos_ang),
        "Sh_membership_frac": float(Sh.mean()),                            # frac of collisions collinear (|cos(grad_p phi,Re)|>0.9)
        "n_Sh": int(Sh.sum()),
        "vhat_degen_frac_on_Sh": vhat_degen_on_Sh,                         # frac of S_h collisions where learned V_hat ALSO degenerate
        "vhat_median_on_Sh": float(np.median(vhat_on_Sh)) if vhat_on_Sh.size else None,
        "vhat_degen_frac_all_collisions": float((lg_vhat < EPS_G).mean()),  # overall: learned degeneracy at ALL residual collisions
    }
    print(f"P3 collinear-residual gate @ {run_dir.name}/best.pt step {out['step']}  (pool n={n_total})")
    print(f"  residual collisions: {n_coll} ({100*n_coll/n_total:.2f}%)")
    print(f"  ||L_g phi|| (position-only) at collisions: median={out['lg_phi_at_collisions']['median']:.4f} "
          f"(structurally ~0 everywhere on the underactuated system, Thm 5.2 — non-discriminating)")
    print(f"  collinear geometry |cos(grad_p phi, Re)| at collisions: median={out['collinear_cos_at_collisions']['median']:.4f} "
          f"-> S_h (|cos|>0.9) frac = {out['Sh_membership_frac']:.3f} (n={out['n_Sh']})")
    print(f"  ||L_g h_star|| at collisions: median={out['lg_hstar_at_collisions']['median']:.4f}")
    print(f"  ||L_g V_hat|| (learned) at collisions: median={out['lg_vhat_at_collisions']['median']:.4f} "
          f"min={out['lg_vhat_at_collisions']['min']:.4f} p5={out['lg_vhat_at_collisions']['p5']:.4f} "
          f"| degenerate(<{EPS_G}) over ALL collisions = {out['vhat_degen_frac_all_collisions']:.3f}")
    print(f"  TRANSFER TEST — on the {out['n_Sh']} S_h (collinear) collision states: learned V_hat degenerate "
          f"(< {EPS_G}) frac = {out['vhat_degen_frac_on_Sh']:.3f}, median ||L_g V_hat|| = {out['vhat_median_on_Sh']}")
    verdict = ("S_h did NOT transfer (learned V_hat keeps lateral authority at collinear collision states)"
               if out["vhat_degen_frac_on_Sh"] < 0.10 else
               "S_h DID transfer (learned V_hat also degenerate at collinear collisions)")
    print(f"  => {verdict}")
    json.dump(out, open(SP / "quadrotor_p3.json", "w"), indent=2)
    print("saved", SP / "quadrotor_p3.json")


if __name__ == "__main__":
    main()
