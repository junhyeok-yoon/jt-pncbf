"""v2.6.0 M4-relaunch — DAgger-robustified BC(LQR) policy init (fixes covariate-shift goal-stopping).

Round 0 = the committed single-pass BC dataset (filtered-LQR manifold) + its checkpoint. Each round r>=1:
roll the CURRENT filtered policy (pi -> HardNet on M1 best.pt V_hat, V_hat frozen) on TRAIN scenes; query
the RAW LQR at every VISITED state (expert label); AGGREGATE (D += new); retrain the ControlNet (normalized
MSE, val early-stop within round); eval filtered-policy in-loop reach. STOP at the first round with reach
>= 0.55 (do NOT chase LQR parity 0.69). FALLBACK: cap (4) reached with reach < 0.55 -> save + report, do
NOT signal a JT launch. Head/L_pre/L_sat already committed; this only builds the init.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

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
BC0 = REPO / "data/bc_lqr_seed42/bc_policy.pt"
INLOOP = REPO / "data/secured_data/pools/eval_inloop_quadrotor-planar_n500_seed12345.pkl"
REACH_GATE = 0.55
ROUND_CAP = 4
SCENES_PER_ROUND = 1200


def _load(dev, dtype=torch.float32):
    ck = torch.load(M1, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    base = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    cfg["network"]["control"] = dict(base["network"]["control"])   # committed clamp_tanh head
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, dtype)
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, dtype); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    return system, vnet, cfg


def collect(system, h_fn, cfg, scenes, dev, behavior):
    """Roll the filtered `behavior(x,bs)`; at every visited state collect (obs, RAW LQR action = expert)."""
    params = _hardnet_params(cfg); bnd = system.u_bounds
    max_steps = int(cfg["eval"]["max_steps"]); dt = float(cfg["env"]["dt"])
    obs_all, u_all = [], []
    for s0 in range(0, len(scenes), 250):
        bs = batch_scenes(scenes[s0:s0 + 250], device=dev, dtype=torch.float32)
        x = initial_states_from_batch(bs).float()
        with torch.no_grad():
            for _ in range(max_steps):
                obs = system.observation(x, bs)
                u_lqr = system.lqr_action(x, scene_goal_tensor(bs, x))    # expert LABEL (always LQR)
                u_beh = behavior(x, bs)                                   # rollout BEHAVIOR (LQR r0, policy r>=1)
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, u_beh, create_graph=False)
                alpha = _base_alpha(h, params); row = -lf - alpha * h
                proj = _base_projection(u_beh, lg, row, bnd, params)
                u_filt, _ = _box_aware_projection(u_beh, proj, lg, row, bnd)
                obs_all.append(obs.detach().cpu()); u_all.append(u_lqr.detach().cpu())
                x = rk4_step(system, x, u_filt, dt)
    return torch.cat(obs_all), torch.cat(u_all)


def train_on(policy, obs, u, dev, u_bounds, patience=6, max_epochs=120):
    lo = u_bounds[:, 0].to(dev); hi = u_bounds[:, 1].to(dev); center = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    norm = lambda uu: (uu - center) / half
    g = torch.Generator().manual_seed(0); perm = torch.randperm(obs.shape[0], generator=g)
    nv = max(1, int(0.1 * obs.shape[0])); vi, ti = perm[:nv], perm[nv:]
    ot, ut = obs[ti].to(dev), u[ti].to(dev); ov, uv = obs[vi].to(dev), u[vi].to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=1.0e-3); best = float("inf"); bad = 0
    for ep in range(max_epochs):
        policy.train(); pe = torch.randperm(ot.shape[0], device=dev)
        for b in range(0, ot.shape[0], 4096):
            idx = pe[b:b + 4096]; loss = torch.mean((norm(policy(ot[idx])) - norm(ut[idx])) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        policy.eval()
        with torch.no_grad():
            vm = float(torch.mean((norm(policy(ov)) - norm(uv)) ** 2))
        if vm < best - 1e-6:
            best = vm; bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    return best


def inloop(system, policy, h_fn, cfg, dev):
    scenes = load_pool(INLOOP).scenes
    un = lambda x, bs: policy(system.observation(x, bs))
    return _arm_summary(_rollout(scenes, un, cfg, h_fn, system, dev, filtered=True, chunk=250), 0.0)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system, vnet, cfg = _load(dev); h_fn = make_h_fn(vnet, system)
    policy = _build_control_net(system, cfg).to(dev)
    policy.load_state_dict(torch.load(BC0, map_location="cpu", weights_only=False)["pi_state"])   # round-0 seed
    rng = np.random.default_rng(4242)
    # round 0 dataset = filtered-LQR manifold (the BC data)
    print("round 0: collecting LQR manifold...", flush=True); t0 = time.time()
    scenes0 = [sample_train_scene(rng, cfg, "quadrotor_planar") for _ in range(SCENES_PER_ROUND)]
    OBS, U = collect(system, h_fn, cfg, scenes0, dev, lambda x, bs: system.lqr_action(x, scene_goal_tensor(bs, x)))
    ev = inloop(system, policy, h_fn, cfg, dev)
    print(f"  round 0 (BC init): D={OBS.shape[0]} reach={ev['reach']:.3f} coll={ev['collision']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    history = [{"round": 0, "reach": ev["reach"], "coll": ev["collision"], "D": int(OBS.shape[0])}]
    reached = ev["reach"] >= REACH_GATE
    for r in range(1, ROUND_CAP + 1):
        if reached:
            break
        t0 = time.time()
        scenes_r = [sample_train_scene(rng, cfg, "quadrotor_planar") for _ in range(SCENES_PER_ROUND)]
        obs_r, u_r = collect(system, h_fn, cfg, scenes_r, dev, lambda x, bs: policy(system.observation(x, bs)))
        OBS = torch.cat([OBS, obs_r]); U = torch.cat([U, u_r])           # AGGREGATE
        vm = train_on(policy, OBS, U, dev, system.u_bounds)
        ev = inloop(system, policy, h_fn, cfg, dev)
        history.append({"round": r, "reach": ev["reach"], "coll": ev["collision"], "val_mse": vm, "D": int(OBS.shape[0])})
        print(f"  round {r}: D={OBS.shape[0]} val_mse={vm:.5f} reach={ev['reach']:.3f} coll={ev['collision']:.3f} timeout={ev['timeout']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        reached = ev["reach"] >= REACH_GATE

    out_dir = REPO / "data/dagger_lqr_seed42"; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "dagger_policy.pt"
    torch.save({"pi_state": policy.state_dict(), "config": cfg, "framework": "dagger_lqr",
                "reach": ev["reach"], "coll": ev["collision"], "history": history, "gate_passed": bool(reached)}, ckpt)
    verdict = "GATE PASS -> launch JT" if reached else "FALLBACK: reach<0.55 -> DO NOT launch JT"
    print(f"\nDAgger DONE: final reach={ev['reach']:.3f} coll={ev['collision']:.3f} | {verdict}", flush=True)
    print(f"saved: {ckpt}", flush=True)
    json.dump({"history": history, "final_reach": ev["reach"], "gate_passed": bool(reached), "ckpt": str(ckpt)},
              open(SP / "dagger_lqr.json", "w"), indent=2)


if __name__ == "__main__":
    main()
