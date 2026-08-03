"""v2.8.1 S1 beta-screen Step 1 — decide the crash cause in seconds, before touching code.

Part A: call soft_topk_obstacles directly at beta in {2,6,12,50}, float32 (the TRAINING dtype; the tests use
float64 and beta=inf delegates to top_k, so neither exercised this), on a battery spanning well-separated
obstacles / near-ties / fewer-than-k active / all-inactive. Report per (beta,scenario): any non-finite FORWARD
entry + first slot, any non-finite GRADIENT (d output / d position) + first slot, and per-slot weight sums.
Part B: at a crashed cell's init condition (fresh value net, beta=6), compute the filter's h / L_f h / L_g h via
_cbf_terms on the in-loop init states and report their finiteness (L_g V_hat is the QP row's normal)."""
import numpy as np
import torch

from src.common.observation import soft_topk_obstacles

DT = torch.float32
BETAS = [2.0, 6.0, 12.0, 50.0]
k = 5


def scene(name):
    if name == "well_separated":
        c = torch.tensor([[[2.0, 0.0], [-3.0, 1.0], [0.0, 3.5], [3.2, -2.0], [-2.5, -3.0]]], dtype=DT)
        a = torch.ones(1, 5, dtype=torch.bool)
    elif name == "near_ties":
        c = torch.tensor([[[1.0, 0.0], [1.0001, 0.0], [-1.0, 0.02], [2.0, 2.0], [-2.0, -2.0]]], dtype=DT)
        a = torch.ones(1, 5, dtype=torch.bool)
    elif name == "fewer_than_k":
        c = torch.tensor([[[1.5, 0.0], [-2.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=DT)
        a = torch.tensor([[True, True, False, False, False]])
    else:  # all_inactive
        c = torch.tensor([[[1.5, 0.0], [-2.0, 1.0], [0.0, 3.0], [2.0, 2.0], [-2.0, -2.0]]], dtype=DT)
        a = torch.zeros(1, 5, dtype=torch.bool)
    r = torch.full((1, 5), 0.4, dtype=DT)
    pos = torch.zeros(1, 2, dtype=DT)
    return pos, c, r, a


print("=== Part A: soft_topk forward + gradient finiteness (float32) ===")
for beta in BETAS:
    for name in ("well_separated", "near_ties", "fewer_than_k", "all_inactive"):
        pos, c, r, a = scene(name)
        posg = pos.clone().requires_grad_(True)
        rel, rad, topmass = soft_topk_obstacles(posg, c, r, a, k, beta=beta, d_c=3.0, inner=2.5, return_indices=True)
        fwd_finite = bool(torch.isfinite(rel).all() and torch.isfinite(rad).all())
        first_bad_fwd = None
        if not fwd_finite:
            bad = (~torch.isfinite(rel).all(dim=-1))[0]
            first_bad_fwd = int(torch.nonzero(bad)[0]) if bad.any() else "radii"
        # gradient
        g = torch.autograd.grad(rel.sum() + rad.sum(), posg, retain_graph=False, allow_unused=True)[0]
        grad_finite = bool(g is not None and torch.isfinite(g).all())
        tm = topmass[0].detach().numpy()
        print(f"beta={beta:>4} {name:<14} fwd_finite={fwd_finite} grad_finite={grad_finite} "
              f"first_bad_fwd_slot={first_bad_fwd} | top_mass_per_slot={np.round(tm,4)}")

print("\n=== Part B: crashed-cell init L_f/L_g/row finiteness (beta=6, fresh value net) ===")
try:
    import yaml
    from src.frameworks.jt_pncbf.train import make_system
    from src.frameworks.oc_pncbf.train import load_effective_config
    from src.common.value_net import make_h_fn, ValueNetEnsemble
    from src.common.filter_hardnet import _cbf_terms
    from src.eval.build_pools import load_pool
    from src.envs.scene_batch import batch_scenes, initial_states_from_batch
    from pathlib import Path
    cfg = load_effective_config()
    cfg["run"]["system"] = "quadrotor_3d"
    cfg.setdefault("obs", {}).setdefault("quadrotor_3d", {})["encoder"] = "soft_topk"
    cfg["obs"]["quadrotor_3d"]["beta"] = 6.0
    cfg["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    sysm = make_system(cfg)
    print("system soft_beta =", sysm.soft_beta)
    net = cfg["network"]["value"]
    vnet = ValueNetEnsemble(obs_dim=sysm.obs_dim, hidden=net["hidden"], n_layers=net["n_layers"],
                            beta=net["softplus_beta"], n_vs=net["n_vs"], head=net["head"])
    h_fn = make_h_fn(vnet, sysm)
    pool = load_pool(Path("data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"))
    bs = batch_scenes(pool.scenes[:64], device=torch.device("cpu"), dtype=torch.float32)
    x = initial_states_from_batch(bs)
    u0 = torch.zeros(x.shape[0], int(getattr(sysm, "action_dim", 4)))
    h, lf, lg = _cbf_terms(sysm, h_fn, x, bs, u0, create_graph=False)
    print(f"init states n={x.shape[0]}: h_finite={bool(torch.isfinite(h).all())} "
          f"L_f_finite={bool(torch.isfinite(lf).all())} L_g_finite={bool(torch.isfinite(lg).all())}")
    print(f"  n_nonfinite: h={int((~torch.isfinite(h)).sum())} L_f={int((~torch.isfinite(lf)).sum())} "
          f"L_g={int((~torch.isfinite(lg)).sum())} (L_g is the CBF-QP row normal)")
except Exception as e:
    print("Part B error:", repr(e))
