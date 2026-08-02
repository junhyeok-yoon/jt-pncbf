"""v2.7.7 M24 (Amdt 12) — simulator-consistent dz_min. Replace the analytic ω-capped righting model with a rollout
of the RECORDED attitude dynamics (same rk4_step + wrap_state clamping as deployment; no src edits). Per IC, apply
the best-effort recovery control the plant allows — MAX righting torque toward upright (saturating PD, box-clipped
through the recorded mixer) with the OPTIMAL collective program (full thrust where cos θ>0, zero when inverted) —
integrate, and flag ICs whose minimal achievable |p_z| still reaches the band (min p_z ≤ −z_lim OR max p_z ≥ +z_lim).
Because the control is granted every advantage the plant allows (all authority to righting+arrest, ignoring the
navigation goal), a flagged IC is genuinely unrecoverable -> the flag is CONSERVATIVE (precision → 1). Reports
precision/recall against the deployed rollout. Eval-only; reads the deployed dump for ICs + actual band outcomes."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.eval.run_full import _load_framework
from src.common.rk4 import rk4_step

JT42 = Path("data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt")
SCR = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
DT, T, ZLIM, KP, KD = 0.05, 200, 4.0, 400.0, 40.0

fw, cfg, _ = _load_framework(JT42, config_overrides={"env": {"dt": DT}, "eval": {"max_steps": T}})
sys_ = fw.system; dev = next(fw.value_net.parameters()).device

z = np.load(SCR / "deployed_dump.npz")
IC = torch.tensor(z["IC"], dtype=torch.float32, device=dev)
pz0 = z["IC"][:, 2]
# deployed FLOOR violations (dz_min is a drop -> the floor); ceiling handled separately below
floor_viol_deployed = z["min_pz"] <= -ZLIM
band_viol_deployed = (z["min_pz"] <= -ZLIM) | (z["max_pz"] >= ZLIM)

x = sys_.wrap_state(IC.clone())
# max-advantage plant-feasible recovery: the plant's cascaded PD (same math as lqr_action, recorded mixer + box)
# but with BOOSTED position gains so the collective SATURATES at f_max (max arrest) and the max attitude gain
# (kp_att=40 already saturates the torque box), aimed to HOLD xy and climb toward the ceiling. Grants every
# advantage the plant's actuation allows.
mixer_inv = sys_.mixer_inv.to(dev, torch.float32); J = sys_.inertia.to(dev, torch.float32)
KP_POS, KD_POS, KP_ATT, KD_ATT, GVAL, MASS, F_RMAX = 12.0, 6.0, 40.0, 8.0, sys_.gravity, sys_.mass, 4.905
gxy = torch.stack([x[:, 0], x[:, 1]], dim=1)


def _R(q):
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9); w, a, b, c = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b),
                        2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a),
                        2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b)], dim=1).reshape(-1, 3, 3)


def control(x):
    p = x[:, :3]; q = x[:, 3:7]; v = x[:, 7:10]; omega = x[:, 10:13]
    goal = torch.cat([gxy, torch.full_like(p[:, :1], ZLIM)], dim=1)   # hold xy, aim ceiling
    a_des = -KP_POS * (p - goal); a_des = a_des - KD_POS * v
    e_up = torch.zeros_like(a_des); e_up[:, 2] = GVAL
    f_des = MASS * (a_des + e_up)
    R = _R(q); b3 = R[:, :, 2]
    f_thr = (f_des * b3).sum(dim=1).clamp(min=0.0)
    b3_des = f_des / f_des.norm(dim=1, keepdim=True).clamp_min(1e-9)
    e_att_world = torch.cross(b3, b3_des, dim=1)
    e_att_body = torch.einsum("bji,bj->bi", R, e_att_world)
    tau = J * (KP_ATT * e_att_body - KD_ATT * omega)
    wrench = torch.cat([f_thr.unsqueeze(1), tau], dim=1)
    return (wrench @ mixer_inv.t()).clamp(0.0, F_RMAX)


min_pz = x[:, 2].clone()
with torch.no_grad():
    for _ in range(T):
        u = control(x)
        x = rk4_step(sys_, x, u, DT)
        min_pz = torch.minimum(min_pz, x[:, 2])
min_pz = min_pz.cpu().numpy()
dz_min = pz0 - min_pz
flag = min_pz <= -ZLIM                                        # best-effort recovery still crosses the floor -> doomed
np.savez_compressed(SCR / "dz_min_sim.npz", flag=flag, dz_min=dz_min, min_pz_sim=min_pz)

nfl = int(flag.sum()); fv = floor_viol_deployed
tp = int((flag & fv).sum()); prec = tp / nfl if nfl else 0.0; rec = tp / int(fv.sum()) if fv.sum() else 0.0
print(f"M24 sim-rollout dz_min (plant lqr recovery, aim ceiling): flagged {nfl}/{len(flag)} ({100*nfl/len(flag):.2f}%)")
print(f"  vs deployed FLOOR violations {int(fv.sum())}: precision {prec:.3f}, recall {rec:.3f}")
print(f"  false positives (flagged, floor not violated): {int((flag & ~fv).sum())}; false negatives (floor violated, not flagged): {int((~flag & fv).sum())}")
fp = flag & ~fv
if fp.sum():
    import scripts.deck.deck_scene3d as S3
    ICn = z["IC"]
    for i in np.nonzero(fp)[0][:12]:
        th = np.degrees(np.arccos(np.clip(S3.quat_to_R(ICn[i, 3:7])[2, 2], -1, 1)))
        wxy = float(np.hypot(ICn[i, 10], ICn[i, 11]))
        print(f"    FP idx {i}: min_pz_sim {min_pz[i]:.2f} (deployed min_pz {z['min_pz'][i]:.2f}) tilt0 {th:.0f} |w0| {wxy:.2f} vz0 {ICn[i,9]:.2f}")
