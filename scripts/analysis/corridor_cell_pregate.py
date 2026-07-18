"""v2.7.1 M0(b) NATURAL-MASS PRE-GATE (no training). Measures the corridor-cell state-visitation fraction in
offline CONTINUING collections under (a) nominal LQR and (b) the secured v2.7.0 iter-5 policy (sha8 3b27d691).
Cell (changes.md §4): surface-dist d in [0.02,0.35] to the nearest ACTIVE obstacle, |theta| in [pi/2,pi],
||v|| in [0.5,1.5], velocity within +-60 deg of the bearing TOWARD that obstacle. If either path >= 5% -> HALT.
"""
import sys
from pathlib import Path
import numpy as np
import torch

from src.common.filter_hardnet import _base_alpha, _base_projection, _box_aware_projection, _cbf_terms, _hardnet_params
from src.common.quadrotor_barrier import value_target_barrier
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import batch_scenes
from src.envs.scene_init import sample_train_scene
from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
from src.frameworks.jt_pncbf.train import make_system, _build_control_net
from src.frameworks.oc_pncbf.collection import OCReplayBuffer

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
ITER5 = REPO / "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt"
DEG60 = np.cos(np.deg2rad(60.0))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def in_cell(states, bs, system):
    """Boolean [B] cell membership per changes.md §4. states [B,D], bs = BatchedScene."""
    p = system.position(states)                                       # [B,2]
    theta = states[:, 2]
    v = states[:, 3:5]
    spd = torch.linalg.norm(v, dim=1)                                 # [B]
    cen = bs.obstacle_centers.to(states.dtype)                        # [B,K,2]
    rad = bs.obstacle_radii.to(states.dtype)                          # [B,K]
    act = bs.obstacle_active.to(torch.bool)                           # [B,K]
    dcen = torch.linalg.norm(p.unsqueeze(1) - cen, dim=2)            # [B,K] center distance
    surf = dcen - rad                                                 # [B,K] surface distance
    surf_m = torch.where(act, surf, torch.full_like(surf, float("inf")))
    d_surf, idx = surf_m.min(dim=1)                                   # nearest active obstacle
    near_c = cen[torch.arange(cen.shape[0]), idx]                     # [B,2]
    in_d = (d_surf >= 0.02) & (d_surf <= 0.35)
    ath = torch.abs(((theta + np.pi) % (2 * np.pi)) - np.pi)
    in_th = ath >= (np.pi / 2)                                        # |theta| in [pi/2, pi]
    in_sp = (spd >= 0.5) & (spd <= 1.5)
    u = near_c - p                                                    # bearing toward obstacle
    un = u / (torch.linalg.norm(u, dim=1, keepdim=True) + 1e-9)
    vhat = v / (spd.unsqueeze(1) + 1e-9)
    cos = (vhat * un).sum(dim=1)
    in_cone = cos >= DEG60
    return in_d & in_th & in_sp & in_cone & torch.isfinite(d_surf)


def collect_fraction(system, cfg, step_fn, dev, label, B=256, rounds=12, seed=1234):
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")
    hbatch = lambda sg, bsc: value_target_barrier(system, sg, bsc, cfg)
    rng = np.random.default_rng(seed)
    buf = OCReplayBuffer(capacity=50_000_000)
    stt = ContinuingState.create(system, sampler, rng, B, cfg, dev, torch.float32, system_name="quadrotor_planar")
    for _ in range(rounds):
        advance_round(stt, round_length=int(cfg["training"]["oc_pncbf"]["horizon"]), step_fn=step_fn,
                      h_batch_fn=hbatch, scene_sampler=sampler, rng=rng, config=cfg, buffer=buf,
                      dt=float(cfg["env"]["dt"]), system_name="quadrotor_planar")
    n_in = 0; n_tot = 0
    for tr in buf._trajectories:
        S = tr.states.to(dev)                                        # [L+1, D]
        bs = batch_scenes([tr.scene] * S.shape[0], device=dev, dtype=torch.float32)
        m = in_cell(S, bs, system)
        n_in += int(m.sum().item()); n_tot += S.shape[0]
    frac = n_in / max(n_tot, 1)
    print(f"[{label}] states={n_tot}  in_cell={n_in}  fraction={frac:.4f} ({frac*100:.2f}%)", flush=True)
    return frac, n_in, n_tot


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import yaml
    def m(b, o):
        d = dict(b)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
        return d
    cfg = m(yaml.safe_load(open("src/configs/base_config.yaml")), yaml.safe_load(open("src/configs/exp_config.yaml")))
    cfg["collection"]["collector"] = "continuing"
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)

    def lqr_step(x, bs):
        g = torch.as_tensor(bs.goal, dtype=x.dtype, device=x.device)
        if g.ndim == 1:
            g = g.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, g)

    # (b) iter-5 policy+HardNet
    ck = torch.load(ITER5, map_location="cpu", weights_only=False); c5 = ck["config"]
    vnet = ValueNetEnsemble(system.obs_dim, c5).to(dev, torch.float32); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for pp in vnet.parameters():
        pp.requires_grad_(False)
    policy = _build_control_net(system, c5).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(c5); bounds = system.u_bounds

    def iter5_step(x, bs):
        un = policy(system.observation(x, bs))
        h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
        alpha = _base_alpha(h, params); row = -lf - alpha * h
        proj = _base_projection(un, lg, row, bounds, params)
        u, _ = _box_aware_projection(un, proj, lg, row, bounds)
        return u.detach()

    print("=== v2.7.1 M0(b) natural-mass pre-gate (offline continuing, no training) ===", flush=True)
    fa, ia, ta = collect_fraction(system, cfg, lqr_step, dev, "nominal-LQR", seed=1234)
    fb, ib, tb = collect_fraction(system, cfg, iter5_step, dev, "iter5-policy(3b27d691)", seed=5678)
    thr = 0.05
    halt = (fa >= thr) or (fb >= thr)
    print(f"\nPRE-GATE threshold {thr:.0%}: nominal={fa*100:.2f}%  iter5={fb*100:.2f}%  -> {'HALT (axis withdrawn)' if halt else 'PASS (cell is rare; proceed)'}", flush=True)
    import json
    json.dump(dict(nominal_frac=fa, nominal_in=ia, nominal_tot=ta, iter5_frac=fb, iter5_in=ib, iter5_tot=tb,
                   threshold=thr, halt=bool(halt)), open(SP / "corridor_cell_pregate.json", "w"), indent=2)


if __name__ == "__main__":
    main()
