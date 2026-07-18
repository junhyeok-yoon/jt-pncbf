"""v2.7.1 corridor probe (changes.md §5, run at M3 pre-JT and M6). For n fresh CELL draws x0, compare the
network prediction V_hat(x0) to the REALIZED discounted-avoid value A0 (03_train §2.1.1 backward recurrence)
along the DEPLOYED-LOOP rollout from x0 (nominal LQR + HardNet if value-only ckpt; trained policy + HardNet if
pi_state present). Report the under-prediction gap = realized - predicted (positive = under-prediction) p50/p90.
Usage: corridor_probe.py <run_dir> [ckpt=best.pt] [n=500]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

from src.common.filter_hardnet import _base_alpha, _base_projection, _box_aware_projection, _cbf_terms, _hardnet_params
from src.common.quadrotor_barrier import value_target_barrier
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_cell_state_scene, sample_train_scene
from src.frameworks.jt_pncbf.train import make_system, _build_control_net

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")


def main():
    run = Path(sys.argv[1]); ckpt = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / "checkpoints" / ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    dt = float(cfg["env"]["dt"]); gamma = 0.95; T = int(cfg["training"]["oc_pncbf"]["horizon"])
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    tnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32)
    tnet.load_state_dict(ck.get("v_s_target_state", ck["v_s_state"])); tnet.eval()
    for p in list(vnet.parameters()) + list(tnet.parameters()):
        p.requires_grad_(False)
    h_fn = make_h_fn(vnet, system); h_tgt = make_h_fn(tnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    has_pi = "pi_state" in ck
    if has_pi:
        policy = _build_control_net(system, cfg).to(dev); policy.load_state_dict(ck["pi_state"]); policy.eval()

    def deployed(x, bs):
        if has_pi:
            un = policy(system.observation(x, bs))
        else:
            g = torch.as_tensor(bs.goal, dtype=x.dtype, device=x.device)
            un = system.lqr_action(x, g if g.ndim > 1 else g.unsqueeze(0).expand(x.shape[0], -1))
        h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
        alpha = _base_alpha(h, params); row = -lf - alpha * h
        proj = _base_projection(un, lg, row, bounds, params)
        u, _ = _box_aware_projection(un, proj, lg, row, bounds)
        return u.detach()

    rng = np.random.default_rng(20260718)
    scenes = [sample_cell_state_scene(sample_train_scene(rng, cfg, "quadrotor_planar"), rng, cfg) for _ in range(n)]
    bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x = system.wrap_state(initial_states_from_batch(bs).float())
    states = [x.clone()]
    with torch.no_grad():
        for _ in range(T):
            x = rk4_step(system, x, deployed(x, bs), dt); states.append(x.clone())
        S = torch.stack(states, 0)                                               # [T+1, n, D]
        c = value_target_barrier(system, S, bs, cfg).clamp(-1, 1)                 # [T+1, n] per-step cost h
        v_pred = h_fn(S[0], bs).reshape(-1)                                       # V_hat(x0)
        A = h_tgt(S[-1], bs).reshape(-1).clamp(-1, 1)                             # tail bootstrap A_{T+1}
        for t in range(T, -1, -1):                                               # backward §2.1.1 recurrence
            A = torch.maximum(c[t], (1 - gamma) * c[t] + gamma * A).clamp(-1, 1)
        gap = (A - v_pred).cpu().numpy()                                          # realized - predicted
    p50, p90 = float(np.percentile(gap, 50)), float(np.percentile(gap, 90))
    out = dict(run=run.name, ckpt=ckpt, step=int(ck.get("step", -1)), n=n, gamma=gamma, T=T, has_policy=has_pi,
               gap_p50=p50, gap_p90=p90, gap_mean=float(gap.mean()), gap_max=float(gap.max()),
               frac_underpredict=float((gap > 0).mean()), v_pred_median=float(np.median(v_pred.cpu().numpy())),
               A0_median=float(np.median(A.cpu().numpy())))
    print(f"[corridor-probe {run.name}/{ckpt} step {out['step']} n={n} loop={'policy' if has_pi else 'nominal-LQR'}]", flush=True)
    print(f"  under-prediction gap (realized A0 - V_hat): p50={p50:+.4f} p90={p90:+.4f} mean={out['gap_mean']:+.4f} "
          f"frac(gap>0)={out['frac_underpredict']:.3f}  V_hat p50={out['v_pred_median']:.3f} A0 p50={out['A0_median']:.3f}", flush=True)
    json.dump(out, open(SP / f"corridor_probe_{run.name}_{ckpt.replace('.pt','')}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
