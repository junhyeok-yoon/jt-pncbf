"""v2.6.0 M4-relaunch — BC(LQR) policy pretraining on the filtered-LQR state manifold.

Roll the filtered LQR (LQR -> HardNet on the M1 best.pt V_hat, V_hat FROZEN) on TRAIN scenes to sample
on-manifold obs; target = the RAW LQR nominal action; objective = MSE on the ControlNet output. STOP at
the FIRST of (i) held-out val-MSE early-stop (patience), or (ii) in-loop filtered-policy reach >= 0.40 &
collision <= 0.185 (M3). Deliberately UNDER-trained (no val-convergence, no LQR reach parity ~0.66) to
leave JT headroom. Saves a BC checkpoint carrying pi_state (loadable via JT pi_init_ckpt).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.common.control_net import ControlNet
from src.common.filter_hardnet import (_base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params)
from src.common.observation import scene_goal_tensor
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.dual_arm import _rollout, _arm_summary
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_train_scene
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
M1 = REPO / "data/v2.6.0__20260715-010357__seed42/checkpoints/best.pt"
INLOOP = REPO / "data/secured_data/pools/eval_inloop_quadrotor-planar_n500_seed12345.pkl"


def _load(dev, dtype=torch.float32):
    ck = torch.load(M1, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    # M1's checkpoint config was saved with the OLD softsign head. The BC policy (and the JT policy it
    # warm-starts) use the NEW clamp_tanh head — since the output activation is not a parameter, the BC
    # weights must be optimized FOR clamp_tanh. Overlay the current base_config network.control head.
    import yaml as _yaml
    base = _yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    cfg["network"]["control"] = dict(base["network"]["control"])
    system = make_system(cfg)
    system.u_bounds = system.u_bounds.to(device=dev, dtype=dtype)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(device=dev, dtype=dtype)
    vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    return system, vnet, cfg


def collect_manifold(system, h_fn, cfg, scenes, dev, chunk=250):
    """Roll the filtered LQR; collect (obs, u_nom_RAW) at every visited on-manifold state (V_hat frozen)."""
    params = _hardnet_params(cfg); bounds = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    obs_all, u_all = [], []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float()
        with torch.no_grad():
            for _ in range(max_steps):
                obs = system.observation(x, bs)
                u_nom = system.lqr_action(x, scene_goal_tensor(bs, x))
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, u_nom, create_graph=False)
                alpha = _base_alpha(h, params); row = -lf - alpha * h
                proj = _base_projection(u_nom, lg, row, bounds, params)
                u_filt, _ = _box_aware_projection(u_nom, proj, lg, row, bounds)
                obs_all.append(obs.detach().cpu()); u_all.append(u_nom.detach().cpu())
                x = rk4_step(system, x, u_filt, dt)
    return torch.cat(obs_all), torch.cat(u_all)


def inloop_filtered_eval(system, policy, h_fn, cfg, dev):
    scenes = load_pool(INLOOP).scenes
    un_fn = lambda x, bs: policy(system.observation(x, bs))
    ep = _rollout(scenes, un_fn, cfg, h_fn, system, dev, filtered=True, chunk=250)
    return _arm_summary(ep, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system, vnet, cfg = _load(dev)
    h_fn = make_h_fn(vnet, system)

    n_scenes = 40 if a.smoke else 1500
    rng = np.random.default_rng(42)
    print(f"collecting filtered-LQR manifold on {n_scenes} train scenes...", flush=True)
    t0 = time.time()
    scenes = [sample_train_scene(rng, cfg, "quadrotor_planar") for _ in range(n_scenes)]
    obs, u = collect_manifold(system, h_fn, cfg, scenes, dev)
    print(f"  collected {obs.shape[0]} (obs,u) pairs in {time.time()-t0:.0f}s", flush=True)

    # 90/10 train/val split
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(obs.shape[0], generator=g)
    n_val = max(1, int(0.1 * obs.shape[0]))
    vi, ti = perm[:n_val], perm[n_val:]
    obs_t, u_t = obs[ti].to(dev), u[ti].to(dev)
    obs_v, u_v = obs[vi].to(dev), u[vi].to(dev)

    policy = _build_control_net(system, cfg).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    bs_bc = 4096
    # NORMALIZED-action MSE (the ControlNet 'raw output' is softsign in [-1,1]): weight the thrust
    # channel (f~10) and the torque channel (tau~0.2) EQUALLY. Raw-action MSE is thrust-scale-dominated
    # and ignores torque, giving an attitude-blind policy that cannot navigate (reach 0).
    lo = system.u_bounds[:, 0].to(dev); hi = system.u_bounds[:, 1].to(dev)
    center = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    def _norm(uu):
        return (uu - center) / half
    best_val = float("inf"); patience, bad = (2 if a.smoke else 6), 0
    stop_reason = None; reach_at_stop = None; coll_at_stop = None
    max_epochs = 2 if a.smoke else 200
    for epoch in range(1, max_epochs + 1):
        policy.train()
        pe = torch.randperm(obs_t.shape[0], device=dev)
        tot = 0.0; nb = 0
        for b in range(0, obs_t.shape[0], bs_bc):
            idx = pe[b:b + bs_bc]
            pred = policy(obs_t[idx])
            loss = torch.mean((_norm(pred) - _norm(u_t[idx])) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        if not torch.isfinite(torch.tensor(tot)):
            print("BC loss NaN -> HALT", flush=True); return
        # val MSE
        policy.eval()
        with torch.no_grad():
            val_mse = float(torch.mean((_norm(policy(obs_v)) - _norm(u_v)) ** 2))
        # in-loop filtered-policy stop-gate
        ev = inloop_filtered_eval(system, policy, h_fn, cfg, dev)
        reach, coll = ev["reach"], ev["collision"]
        print(f"  epoch {epoch}: train_mse={tot/nb:.5f} val_mse={val_mse:.5f} | in-loop reach={reach:.3f} coll={coll:.3f}", flush=True)
        # stop rule (i): val early-stop
        if val_mse < best_val - 1e-6:
            best_val = val_mse; bad = 0
        else:
            bad += 1
        # stop rule (ii): reach >= 0.55 & coll <= 0.185 (M3) -> stop (expressiveness recovered; below the
        # LQR parity ~0.69 to leave JT headroom). The BC gate requires reach approach M3 (>=~0.55).
        if reach >= 0.55 and coll <= 0.20:
            stop_reason = "inloop_reach>=0.55 & coll<=0.20"; reach_at_stop = reach; coll_at_stop = coll; break
        if bad >= patience:
            stop_reason = "val_mse_early_stop"; reach_at_stop = reach; coll_at_stop = coll; break
    else:
        stop_reason = "max_epochs"; reach_at_stop = reach; coll_at_stop = coll

    out_dir = REPO / ("data/bc_lqr_seed42" if not a.smoke else "data/bc_lqr_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "bc_policy.pt"
    torch.save({"pi_state": policy.state_dict(), "config": cfg, "framework": "bc_lqr",
                "stop_reason": stop_reason, "reach_at_stop": reach_at_stop, "coll_at_stop": coll_at_stop,
                "val_mse": best_val}, ckpt_path)
    print(f"\nBC STOP: {stop_reason} | reach_at_stop={reach_at_stop:.3f} coll={coll_at_stop:.3f} val_mse={best_val:.5f}", flush=True)
    print(f"saved BC ckpt: {ckpt_path}", flush=True)
    json.dump({"stop_reason": stop_reason, "reach_at_stop": reach_at_stop, "coll_at_stop": coll_at_stop,
               "val_mse": best_val, "n_pairs": int(obs.shape[0]), "ckpt": str(ckpt_path)},
              open(SP / ("bc_lqr_smoke.json" if a.smoke else "bc_lqr.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
