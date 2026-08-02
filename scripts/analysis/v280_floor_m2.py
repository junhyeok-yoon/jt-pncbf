"""v2.8.0 floor-feasibility M2 — the UPPER bound (demonstrated recoverable), GPU rollouts.

A one-parameter family of arrest controllers under the DEPLOYED integrator+dynamics. The parameter
kappa in [0,1] splits the per-rotor box between collective and differential-torque authority:
  f_base = (1-kappa)*f_rotor_max  (kappa=0 -> all rotors at f_rotor_max: max collective, zero torque),
  torque command tau_des levels the attitude (attitude PD to world-up), mapped through mixer_inv to a
  per-rotor differential df, scaled so max|df| = kappa*f_rotor_max (the reserved headroom),
  f_i = clip(f_base + df_i, 0, f_rotor_max).
Every member is admissible, so the best min-z over the sweep is a valid UPPER bound on achievable
altitude loss. Records best min_z per IC and the achieving kappa. No obstacle-avoid/goal-seek: the
altitude subproblem only. Appends to data/runs/v2.8.0/floor_feasibility.json (M1 must have run)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch

from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.frameworks.jt_pncbf.train import make_system, load_effective_config
from src.envs.quadrotor_3d import _quat_to_R, _rot

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/floor_feasibility.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAND, DT, STEPS = 4.0, 0.05, 200
KAPPAS = torch.linspace(0.0, 1.0, 11)


def arrest_control(system, x, kappa, F, kp, kd):
    q = x[:, 3:7]; omega = x[:, 10:13]
    Rm = _quat_to_R(q)
    b3 = Rm[:, :, 2]                                             # current body-up (world)
    e3 = torch.zeros_like(b3); e3[:, 2] = 1.0
    e_att_world = torch.cross(b3, e3, dim=1)                     # small-angle attitude error toward level
    e_att_body = _rot(Rm.transpose(1, 2), e_att_world)
    J = system.inertia.to(x.device, x.dtype)
    tau_des = J * (kp * e_att_body - kd * omega)                # attitude PD to level
    wrench = torch.cat([torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype), tau_des], dim=1)
    df = wrench @ system.mixer_inv.to(x.device, x.dtype).t()    # per-rotor differential for tau_des
    hp = kappa * F                                              # reserved headroom
    mx = df.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    df = df * torch.clamp(hp / mx, max=1.0)                     # scale so max|df| <= kappa*F
    f_base = (1.0 - kappa) * F
    return torch.clamp(f_base + df, 0.0, F)


cfg = load_effective_config(); cfg["run"]["system"] = "quadrotor_3d"
system = make_system(cfg)
F = float(system.f_rotor_max); kp = float(system.kp_att); kd = float(system.kd_att)
scenes = load_pool(POOL).scenes
bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)
x0 = initial_states_from_batch(bs).to(DEV, torch.float32)
N = x0.shape[0]
best_minz = torch.full((N,), -1e9, device=DEV)
best_kappa = torch.zeros(N, device=DEV)
per_kappa_minz = {}
for kap in KAPPAS:
    k = float(kap)
    x = x0.clone()
    minz = x[:, 2].clone()
    with torch.no_grad():
        for _ in range(STEPS):
            u = arrest_control(system, x, k, F, kp, kd)
            x = system.wrap_state(rk4_step(system, x, u, DT))
            minz = torch.minimum(minz, x[:, 2])
    per_kappa_minz[f"{k:.1f}"] = float(minz.mean())
    better = minz > best_minz
    best_kappa = torch.where(better, torch.full_like(best_kappa, k), best_kappa)
    best_minz = torch.maximum(best_minz, minz)
    print(f"kappa={k:.1f}: mean min_z {float(minz.mean()):.3f}  n_below_floor {int((minz < -BAND).sum())}", flush=True)

recoverable = best_minz >= -BAND                                # best controller keeps it above the floor
rep = json.loads(OUT.read_text())
bk = best_kappa.cpu().numpy(); bm = best_minz.cpu().numpy()
rep["M2_upper_bound"] = {
    "kappa_grid": [float(x) for x in KAPPAS], "steps": STEPS, "dt": DT,
    "n_recoverable": int(recoverable.sum()), "frac_recoverable": float(recoverable.double().mean()),
    "per_kappa_mean_minz": per_kappa_minz,
    "best_kappa_distribution": {f"{float(k):.1f}": int((bk == float(k)).sum()) for k in KAPPAS},
    "best_kappa_median": float(np.median(bk)),
    "controller": "attitude-PD to world-up; kappa reserves kappa*f_rotor_max headroom for torque, "
                  "(1-kappa)*f_rotor_max collective baseline; per-rotor clipped to box",
}
for i in range(N):
    rep["ic"][i]["best_min_z"] = float(bm[i]); rep["ic"][i]["best_kappa"] = float(bk[i])
    rep["ic"][i]["recoverable"] = bool(recoverable[i].item())
# HALT check: M1-unrecoverable must NOT be M2-recoverable (contradiction)
unrec = np.array([d["unrecoverable"] for d in rep["ic"]])
rec = np.array([d["recoverable"] for d in rep["ic"]])
contradiction = int((unrec & rec).sum())
rep["M2_upper_bound"]["contradiction_unrec_and_recoverable"] = contradiction
OUT.write_text(json.dumps(rep, indent=2) + "\n")
print(f"M2 RECOVERABLE: {int(recoverable.sum())}/{N} = {float(recoverable.double().mean()):.4f}; "
      f"best-kappa median {float(np.median(bk)):.2f}; CONTRADICTION(M1unrec&M2rec)={contradiction}")
if contradiction:
    raise SystemExit(f"HALT: {contradiction} ICs are both M1-unrecoverable and M2-recoverable")
