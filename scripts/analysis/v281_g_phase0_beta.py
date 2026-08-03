"""v2.8.1 S1 beta-screen Phase 0 (free — pure function of beta, NO training).

Reuses the G2 protocol: roll out the v2.8.0 deployed checkpoint (hard) on a canonical-pool subset, capture states
+ the hard top-k order (crossing detection). Then, at fixed d_c=3.0 m, sweep beta and for each report:
  - the crossing/non-crossing ||delta obs|| ratio on the obstacle block (obs[:,12:12+4k]) — the G2 continuity number
  - the soft-rank top-slot-0 mass distribution (mean, fraction of slots with top-mass < 0.9)
The pure-hard top_k ratio (beta=inf, d_c=inf) is the marked reference (G2: 211.31). beta=2.0 is the registered
Stage-1 value. Continuity holds at every finite beta; this shows where the crossing ratio sits vs the hard 211
across the grid, so a training cell whose ratio would exceed 211 (STOP-b) can be seen before it is trained."""
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.observation import scene_obstacle_tensors, top_k_obstacles, soft_topk_obstacles, SOFT_DC, SOFT_INNER

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NSUB, MAXS = 256, 200
BETAS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, float("inf")]

over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},
        "eval": {"max_steps": 200, "dt_ctrl": 0.05},
        "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
fw, cfg, ck = _load_framework(str(CK), config_overrides=over)   # hard (v2.8.0 config has no obs key)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None:
        m.to(DEV)
sysm = fw.system; k = int(sysm.k_obs)
scenes = load_pool(POOL).scenes[:NSUB]
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene)

# rollout under the deployed (hard) policy+filter; capture states + hard top-k order
states, topk = [x], []
with torch.no_grad():
    for _ in range(MAXS):
        u_nom = fw.policy(x, bscene); u_safe, _ = fw.filter(x, u_nom, bscene)
        p_xy = x[:, :2]; c, r, a = scene_obstacle_tensors(bscene, x.device, x.dtype)
        _, _, tk = top_k_obstacles(p_xy, c[..., :2], r, a, k, return_indices=True)
        topk.append(tk.cpu().numpy())
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), 0.05)); states.append(x)
TK = np.stack(topk, 0)                     # [T,B,k]
cross = ~(TK[1:] == TK[:-1]).all(axis=2)   # [T-1,B] rank crossings (beta-independent)

# per-beta obstacle-block obs on the captured states, at fixed d_c=3.0
X = torch.stack(states[:-1], 0)            # align with topk length T
bs, be = 12, 12 + 4 * k
rows = []
with torch.no_grad():
    for beta in BETAS:
        obl = []
        for t in range(X.shape[0]):
            old = sysm.soft_beta; sysm.encoder = "soft_topk"; sysm.soft_beta = beta
            try:
                o = sysm.observation(X[t], bscene).detach()
            finally:
                sysm.soft_beta = old
            obl.append(o[:, bs:be].cpu().numpy())
        OB = np.stack(obl, 0)              # [T,B,4k]
        d = np.linalg.norm(OB[1:] - OB[:-1], axis=2)
        mc = float(np.median(d[cross])); mnc = float(np.median(d[~cross]))
        ratio = mc / mnc if mnc > 0 else float("inf")
        # top-slot mass at this beta (share of slot-0's top obstacle)
        c2, r2, a2 = scene_obstacle_tensors(bscene, X[0].device, X[0].dtype)
        tm_all = []
        for t in range(0, X.shape[0], 5):   # subsample time for the mass distribution
            _, _, tm = soft_topk_obstacles(X[t][:, :2], c2[..., :2], r2, a2, k, beta=beta, d_c=SOFT_DC,
                                           inner=SOFT_INNER, return_indices=True)
            tm_all.append(tm.cpu().numpy())
        TM = np.concatenate(tm_all, 0)
        rows.append({"beta": (None if beta == float("inf") else beta), "beta_str": ("inf" if beta == float("inf") else beta),
                     "crossing_ratio": ratio, "dblk_cross_med": mc, "dblk_noncross_med": mnc,
                     "top_mass_mean": float(TM.mean()), "top_mass_frac_below_0.9": float((TM < 0.9).mean()),
                     "delta_d_ln9_over_beta_m": (None if beta == float("inf") else float(np.log(9) / beta))})
        print(f"beta={rows[-1]['beta_str']:>4} | crossing_ratio={ratio:8.2f} | top_mass_mean={TM.mean():.3f} "
              f"frac<0.9={float((TM<0.9).mean()):.3f} | Delta_d={rows[-1]['delta_d_ln9_over_beta_m']}", flush=True)

# pure-hard reference (top_k, d_c=inf): the G2 211.31 marker
rec = {"gate": "phase0_beta_sweep", "n_scenes": NSUB, "n_crossing_steps": int(cross.sum()),
       "hard_reference_ratio_G2": 211.31, "d_c_fixed_m": SOFT_DC, "rows": rows}
(OUT / "phase0_beta_sweep.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"\ncrossing steps {int(cross.sum())}/{int(cross.size)} | hard top_k reference ratio (G2) = 211.31")
print(f"wrote {OUT/'phase0_beta_sweep.json'}")
