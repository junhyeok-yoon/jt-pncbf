"""v2.8.1 S1 G2 — encoder continuity on D3 crossing steps (observation-side only) + informative price metrics.

Roll out cf948104 (v2.8.0 deployed, hard encoder) on a canonical-pool subset, capturing states + the hard
top-k order per step. Locate crossing steps (top-k order change, the v2.8.0 D3 protocol). On the SAME states,
compute ||delta obs|| (consecutive control steps, within episode) under BOTH encoders and report the ratio of
the crossing-step median to the non-crossing median. hard_topk reproduces the jump; soft_topk must be O(1).
Also (informative, not a gate) under soft_topk: fraction of steps whose obstacle block is all-zero (every
active obstacle beyond d_c), and fraction of slots whose top obstacle holds <0.9 of the soft-rank mass."""
import copy, json
from pathlib import Path
import numpy as np, torch
from src.eval.run_full import _load_framework
from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.common.observation import scene_obstacle_tensors, top_k_obstacles, soft_topk_obstacles

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_gates"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NSUB = 256

over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},
        "eval": {"max_steps": 200, "dt_ctrl": 0.05},
        "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
fw, cfg, ck = _load_framework(str(CK), config_overrides=over)   # hard encoder (cf948104 config has no obs key)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None: m.to(DEV)
sysm = fw.system; k_obs = int(sysm.k_obs)
scenes = load_pool(POOL).scenes[:NSUB]
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene); B = x.shape[0]
max_steps = 200

# helper: full obs under a given encoder (build a system view by toggling sysm.encoder)
def obs_under(encoder, x):
    old = sysm.encoder; sysm.encoder = encoder
    try:
        return sysm.observation(x, bscene).detach()
    finally:
        sysm.encoder = old

states=[x]; topk_hist=[]; obs_hard=[]; obs_soft=[]; allzero=[]; topmass=[]
with torch.no_grad():
    for i in range(max_steps):
        u_nom = fw.policy(x, bscene); u_safe, _ = fw.filter(x, u_nom, bscene)
        # hard top-k order (D3 crossing detection) + full obs under both encoders on this state
        p_xy = x[:, :2]; c, r, a = scene_obstacle_tensors(bscene, x.device, x.dtype)
        _, _, tk = top_k_obstacles(p_xy, c[..., :2], r, a, k_obs, return_indices=True)
        topk_hist.append(tk.cpu().numpy())
        oh = obs_under("hard_topk", x); os_ = obs_under("soft_topk", x)
        obs_hard.append(oh.cpu().numpy()); obs_soft.append(os_.cpu().numpy())
        # informative (soft): obstacle block = obs[:,12:12+4k]; all-zero if every active obstacle beyond d_c
        blk = os_[:, 12:12 + 4 * k_obs]
        allzero.append((blk.abs().sum(1) < 1e-8).float().cpu().numpy())
        _, _, tm = soft_topk_obstacles(p_xy, c[..., :2], r, a, k_obs, return_indices=True)  # per-slot top mass
        topmass.append(tm.cpu().numpy())
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), 0.05)); states.append(x)

TK = np.stack(topk_hist, 0)      # [T,B,k]
OH = np.stack(obs_hard, 0); OS = np.stack(obs_soft, 0)   # [T,B,obs_dim]
bs, be = 12, 12 + 4 * k_obs      # obstacle block slice (the ONLY part the encoder controls; [0:12] is state-only)
cross = ~(TK[1:] == TK[:-1]).all(axis=2)                  # [T-1,B]  (v2.8.0 D3 crossing protocol)
def ratio(OB):
    d = np.linalg.norm(OB[1:] - OB[:-1], axis=2)
    c = d[cross]; nc = d[~cross]
    mc = float(np.median(c)) if c.size else float("nan"); mnc = float(np.median(nc)) if nc.size else float("nan")
    return {"dblk_median_crossing": mc, "dblk_median_noncrossing": mnc, "ratio": (mc / mnc if mnc > 0 else float("inf"))}
# obstacle-block-only (encoder-isolating) + full-obs (state-motion-confounded, reported for context)
h_blk = ratio(OH[:, :, bs:be]); s_blk = ratio(OS[:, :, bs:be])
h_full = ratio(OH); s_full = ratio(OS)
AZ = np.stack(allzero, 0); TM = np.stack(topmass, 0)
rec = {"gate": "G2_continuity", "n_scenes": NSUB, "n_crossing_steps": int(cross.sum()), "n_total_pairs": int(cross.size),
       "obstacle_block": {"hard": h_blk, "soft": s_blk},
       "full_obs_confounded_by_state_motion": {"hard": h_full, "soft": s_full},
       "informative_soft": {"all_zero_obstacle_block_frac": float(AZ.mean()),
                            "slots_top_mass_below_0.9_frac": float((TM < 0.9).mean())}}
(OUT / "g2_continuity.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"G2 crossing steps {int(cross.sum())}/{int(cross.size)}", flush=True)
print(f"  OBSTACLE-BLOCK (encoder-isolated): HARD ratio {h_blk['ratio']:.2f} "
      f"(cross {h_blk['dblk_median_crossing']:.3f}/noncross {h_blk['dblk_median_noncrossing']:.4f}) | "
      f"SOFT ratio {s_blk['ratio']:.2f} (cross {s_blk['dblk_median_crossing']:.4f}/noncross {s_blk['dblk_median_noncrossing']:.4f})", flush=True)
print(f"  full-obs (state-motion-confounded): HARD {h_full['ratio']:.2f} | SOFT {s_full['ratio']:.2f}", flush=True)
print(f"  informative(soft): all-zero block frac {rec['informative_soft']['all_zero_obstacle_block_frac']:.4f} | "
      f"slot top-mass<0.9 frac {rec['informative_soft']['slots_top_mass_below_0.9_frac']:.4f}", flush=True)
