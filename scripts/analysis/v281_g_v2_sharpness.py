"""v2.8.1 S1 V2 — certificate sharpness / authority, measured (not eyeballed).

For a value checkpoint, roll out the deployed policy+filter on the in-loop pool (the on-policy state
distribution the certificate actually sees), then on those states measure, for the LEARNED certificate V_hat the
filter consumes (make_h_fn -> deployed_h) and the geometric target h_star (value_target_barrier, band-aware):
  - ||grad_x V_hat|| distribution (all states, and near the h_star=0 set)   [sharpness]
  - ||L_g V_hat|| distribution (all, and near h_star=0)                     [AUTHORITY — the decisive one]
  - signed alignment of the V_hat=0 and h_star=0 sets: E[V_hat | |h_star|<band] and E[h_star | |V_hat|<band]
  - singular-row share (||L_g V_hat|| < 5e-4, the filter's own threshold) and zero-Jacobian share
  - filter intervention rate (||u_safe - u_nom|| > tol over the rollout)
Runs read-only on any live run dir; outputs go to data/runs/v2.8.1/s1_diagnostics/. CPU by default (item 7)."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.value_net import make_h_fn
from src.common.quadrotor_barrier import value_target_barrier
from src.common.filter_hardnet import _SINGULAR_LG_THRESHOLD

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
BANDS = (0.05, 0.10, 0.20)          # |h_star| < band = "near the boundary"
INTERV_TOL = 1e-6


def _scene_flat(bscene, T):
    """Repeat the batched scene T times so flattened states [T*B,...] (order t*B+b) map to scene b == row%B."""
    return SimpleNamespace(
        goal=bscene.goal.repeat(T, 1),
        obstacle_centers=bscene.obstacle_centers.repeat(T, 1, 1),
        obstacle_radii=bscene.obstacle_radii.repeat(T, 1),
        obstacle_active=bscene.obstacle_active.repeat(T, 1),
    )


def _dist(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def analyze(ckpt, label, n_scenes, max_steps, sub, device):
    dev = torch.device(device)
    fw, cfg, ck = _load_framework(ckpt)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None)
        if m is not None:
            m.to(dev)
    sysm = fw.system
    h_fn = make_h_fn(fw.value_net, sysm)
    dt = float(cfg["env"]["dt"])
    scenes = load_pool(INLOOP).scenes[:n_scenes]
    bscene = batch_scenes(scenes, device=dev, dtype=torch.float32)
    x = initial_states_from_batch(bscene)
    B = x.shape[0]

    # ---- rollout: collect states + filter intervention on-policy ----
    # NOT wrapped in no_grad: the deployed filter's _cbf_terms needs autograd internally (grad_h). We detach x
    # at each step boundary and detach u_safe before rk4 so no graph accumulates across the rollout.
    states, interv = [], []
    for _ in range(max_steps):
        x = x.detach()
        u_nom = fw.policy(x, bscene)
        _res = fw.filter(x, u_nom, bscene)
        u_safe = _res[0] if isinstance(_res, tuple) else _res
        states.append(x.detach().cpu())
        interv.append((torch.norm(u_safe - u_nom, dim=1) > INTERV_TOL).float().detach().cpu())
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), dt))
    # numpy round-trip strips any inference-mode/no-grad taint from the rolled-out states so xr can carry autograd
    X = torch.stack(states, 0).reshape(-1, states[0].shape[1]).detach().cpu().numpy()   # [T*B, state_dim]
    interv_rate = float(torch.stack(interv, 0).mean().item())
    scene_all = _scene_flat(bscene, max_steps)

    # ---- subsample (deterministic) for the autograd/L_g pass ----
    M = X.shape[0]
    idx_np = np.random.default_rng(20260802).permutation(M)[:min(sub, M)]
    idx = torch.as_tensor(idx_np, device=scene_all.goal.device)
    Xs = torch.tensor(X[idx_np], device=dev, dtype=torch.float32)
    scene_s = SimpleNamespace(
        goal=scene_all.goal[idx], obstacle_centers=scene_all.obstacle_centers[idx],
        obstacle_radii=scene_all.obstacle_radii[idx], obstacle_active=scene_all.obstacle_active[idx])
    adim = int(getattr(sysm, "action_dim", 4))

    gradV, LgV, Vhat, Hstar = [], [], [], []
    CH = 4096
    for s in range(0, Xs.shape[0], CH):
        xb = Xs[s:s + CH]
        sc = SimpleNamespace(goal=scene_s.goal[s:s + CH], obstacle_centers=scene_s.obstacle_centers[s:s + CH],
                             obstacle_radii=scene_s.obstacle_radii[s:s + CH], obstacle_active=scene_s.obstacle_active[s:s + CH])
        xr = xb.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = h_fn(xr, sc).reshape(-1)
            gV = torch.autograd.grad(V.sum(), xr)[0]
            zero = torch.zeros(xr.shape[0], adim, device=dev, dtype=xr.dtype)
            f0 = sysm.dynamics(xr, zero)
            cols = []
            for a in range(adim):
                e = torch.zeros_like(zero); e[:, a] = 1.0
                cols.append(sysm.dynamics(xr, e) - f0)
            gx = torch.stack(cols, 2)
            lg = torch.einsum("bs,bsa->ba", gV, gx)
        gradV.append(gV.norm(dim=1).detach().cpu())
        LgV.append(lg.norm(dim=1).detach().cpu())
        Vhat.append(V.detach().cpu())
        Hstar.append(value_target_barrier(sysm, xb, sc, cfg).reshape(-1).detach().cpu())
    gradV = torch.cat(gradV).numpy(); LgV = torch.cat(LgV).numpy()
    Vhat = torch.cat(Vhat).numpy(); Hstar = torch.cat(Hstar).numpy()

    rec = {"label": label, "ckpt": str(ckpt), "encoder": getattr(sysm, "encoder", "?"),
           "n_scenes": n_scenes, "max_steps": max_steps, "n_states_total": int(M), "n_subsampled": int(Xs.shape[0]),
           "filter_intervention_rate": interv_rate,
           "grad_V_all": _dist(gradV), "Lg_V_all": _dist(LgV),
           "singular_row_share_LgV<5e-4": float((LgV < _SINGULAR_LG_THRESHOLD).mean()),
           "zero_jacobian_share_gradV<5e-4": float((gradV < _SINGULAR_LG_THRESHOLD).mean()),
           "near_boundary": {}, "alignment": {}}
    for band in BANDS:
        near = np.abs(Hstar) < band
        rec["near_boundary"][f"|h*|<{band}"] = {
            "n": int(near.sum()),
            "grad_V": _dist(gradV[near]), "Lg_V": _dist(LgV[near]),
            "singular_share": float((LgV[near] < _SINGULAR_LG_THRESHOLD).mean()) if near.any() else None}
        vnear = np.abs(Vhat) < band
        rec["alignment"][f"band={band}"] = {
            "E[V_hat | |h*|<band]": float(Vhat[near].mean()) if near.any() else None,   # signed offset proxy
            "E[h* | |V_hat|<band]": float(Hstar[vnear].mean()) if vnear.any() else None,
            "n_h*_near": int(near.sum()), "n_V_near": int(vnear.sum())}
    (OUT / f"v2_sharpness_{label}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[{label}] enc={rec['encoder']} states={M} sub={Xs.shape[0]} | "
          f"||L_gV|| med={rec['Lg_V_all']['median']:.4f} p05={rec['Lg_V_all']['p05']:.4f} "
          f"singular={rec['singular_row_share_LgV<5e-4']:.4f} | "
          f"||gradV|| med={rec['grad_V_all']['median']:.3f} | interv={interv_rate:.3f}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-scenes", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--sub", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    analyze(a.ckpt, a.label, a.n_scenes, a.max_steps, a.sub, a.device)
