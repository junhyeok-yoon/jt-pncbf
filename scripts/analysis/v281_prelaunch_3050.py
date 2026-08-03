"""v2.8.1 S1 beta-screen {30,50} pre-launch disqualifiers (report survivors BEFORE training).

Part A: Phase-0 crossing ‖Δobs‖ ratio at beta in {2,6,12,30,50} (fixed encoder) — the continuity axis of the
        frontier + adoption criterion (ii) (<=121).
Part B: float32 FD agreement at beta 30,50 on the tie battery — float32 autograd vs float64 autograd (the true
        gradient) and vs float32 central-difference; a beta whose float32 gradient is not accurate cannot be
        trained (the screen would measure numerical error)."""
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
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BETAS = [2.0, 6.0, 12.0, 30.0, 50.0]

# ---- Part A: crossing ratio (fixed encoder) ----
over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48}, "eval": {"max_steps": 200, "dt_ctrl": 0.05},
        "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
fw, cfg, ck = _load_framework(str(CK), config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None:
        m.to(DEV)
sysm = fw.system; k = int(sysm.k_obs)
scenes = load_pool(POOL).scenes[:256]
bscene = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x = initial_states_from_batch(bscene)
states, topk = [x], []
with torch.no_grad():
    for _ in range(200):
        u_nom = fw.policy(x, bscene); u_safe, _ = fw.filter(x, u_nom, bscene)
        p_xy = x[:, :2]; c, r, a = scene_obstacle_tensors(bscene, x.device, x.dtype)
        _, _, tk = top_k_obstacles(p_xy, c[..., :2], r, a, k, return_indices=True)
        topk.append(tk.cpu().numpy())
        x = sysm.wrap_state(rk4_step(sysm, x, u_safe.detach(), 0.05)); states.append(x)
TK = np.stack(topk, 0); cross = ~(TK[1:] == TK[:-1]).all(axis=2)
X = torch.stack(states[:-1], 0); bs_, be_ = 12, 12 + 4 * k
print("=== Part A: crossing ratio (fixed encoder), criterion (ii) <= 121 ===")
cross_ratios = {}
with torch.no_grad():
    for beta in BETAS:
        obl = []
        for t in range(X.shape[0]):
            old = sysm.soft_beta; sysm.encoder = "soft_topk"; sysm.soft_beta = beta
            try:
                o = sysm.observation(X[t], bscene).detach()
            finally:
                sysm.soft_beta = old
            obl.append(o[:, bs_:be_].cpu().numpy())
        OB = np.stack(obl, 0); d = np.linalg.norm(OB[1:] - OB[:-1], axis=2)
        ratio = float(np.median(d[cross]) / np.median(d[~cross]))
        cross_ratios[beta] = ratio
        print(f"  beta={beta:>4}: crossing_ratio={ratio:7.2f}  -> criterion(ii) {'SURVIVES' if ratio <= 121 else 'DISQUALIFIED (>121)'}")

# ---- Part B: float32 FD agreement at 30,50 (tie battery) ----
print("\n=== Part B: float32 gradient accuracy at beta 30,50 (tie battery) ===")
def battery(DT):
    ang = torch.linspace(0, 2 * np.pi, 6, dtype=DT)[:5]
    t4 = torch.cat([torch.tensor([[[2.39, 0.0]]], dtype=DT),
                    torch.stack([3.318 * torch.cos(ang[:4]), 3.318 * torch.sin(ang[:4])], 1).unsqueeze(0)], 1)
    return {"dom_near_far": (torch.tensor([[[2.39, 0], [5.2, .5], [0, 0], [0, 0], [0, 0]]], dtype=DT), torch.full((1, 5), .4, dtype=DT), torch.tensor([[True, True, False, False, False]])),
            "near_tie4": (t4, torch.full((1, 5), .4, dtype=DT), torch.ones(1, 5, dtype=torch.bool)),
            "near_ties": (torch.tensor([[[1., 0], [1.0001, 0], [-1., .02], [2, 2], [-2, -2]]], dtype=DT), torch.full((1, 5), .4, dtype=DT), torch.ones(1, 5, dtype=torch.bool))}
def grad_at(c, r, a, beta, DT):
    pos = torch.zeros(1, 2, dtype=DT, requires_grad=True)
    rel, rad = soft_topk_obstacles(pos, c, r, a, 5, beta=beta, d_c=SOFT_DC, inner=SOFT_INNER)
    return torch.autograd.grad((rel.sum() + rad.sum()), pos)[0]
for beta in (30.0, 50.0):
    worst_ag, worst_fd = 0.0, 0.0
    for name, (c, r, a) in battery(torch.float32).items():
        c64, r64 = c.double(), r.double()
        g32 = grad_at(c, r, a, beta, torch.float32)
        g64 = grad_at(c64, r64, a, beta, torch.float64)                      # true gradient
        worst_ag = max(worst_ag, float((g32.double() - g64).abs().max()))    # float32 autograd error vs truth
        pos = torch.zeros(1, 2, dtype=torch.float32); fd = torch.zeros(1, 2)
        for j in range(2):
            e = torch.zeros(1, 2); e[0, j] = 1e-3
            fp = soft_topk_obstacles(pos + e, c, r, a, 5, beta=beta); fm = soft_topk_obstacles(pos - e, c, r, a, 5, beta=beta)
            fd[0, j] = ((fp[0].sum() + fp[1].sum()) - (fm[0].sum() + fm[1].sum())) / 2e-3
        worst_fd = max(worst_fd, float((g32 - fd).abs().max()))
    verdict = "SURVIVES (float32 gradient accurate)" if worst_ag < 1e-2 else "DISQUALIFIED (float32 gradient inaccurate)"
    print(f"  beta={beta:>4}: max|f32_autograd - f64_truth|={worst_ag:.2e}  max|f32_autograd - f32_FD|={worst_fd:.2e}  -> {verdict}")
