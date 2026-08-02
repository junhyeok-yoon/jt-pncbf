"""v2.8.0 floor-feasibility M1 — the LOWER bound (proved unrecoverable), analytic, no GPU.

Instant-levelling relaxation: the vehicle is level at t=0 and holds maximum collective thrust.
Max upward accel a_max = 4*f_rotor_max/m - g. From descent rate vz0<0 the altitude lost before arrest
is vz0^2/(2*a_max), so the relaxed minimum altitude is z0 - min(vz0,0)^2/(2*a_max). No real trajectory
beats this (no attitude gives more upward accel than level-at-max-thrust; no rotation is faster than
instantaneous), so an IC whose relaxed minimum is below the floor (-band) is UNRECOVERABLE by any
controller. Assumption: floor is the only vertical constraint (obstacles ignored) — a bound on the
altitude subproblem alone. Sidecar: data/runs/v2.8.0/floor_feasibility.json (this writes m1 + states)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch, yaml

from src.eval.build_pools import load_pool
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.quadrotor_3d import _quat_to_R

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.0/floor_feasibility.json"; OUT.parent.mkdir(parents=True, exist_ok=True)

c = yaml.safe_load(open(REPO / "src/configs/exp_config.yaml"))
b = c["env"]["bounds"]["quadrotor_3d"]; p = c["env"]["quadrotor_3d"]
F = float(b["f_rotor_max"]); M = float(p["mass"]); G = float(p["gravity"]); BAND = 4.0
A_MAX = 4.0 * F / M - G

scenes = load_pool(POOL).scenes
bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float64)
x0 = initial_states_from_batch(bs).double()                     # [N,13]
z0 = x0[:, 2]
vz0 = x0[:, 9]                                                   # world vertical velocity
R = _quat_to_R(x0[:, 3:7])
cos_tilt = R[:, 2, 2].clamp(-1, 1)
tilt_deg = torch.rad2deg(torch.arccos(cos_tilt))
omega = torch.linalg.norm(x0[:, 10:13], dim=1)
clearance = z0 - (-BAND)                                        # distance above floor at t=0
relaxed_min_z = z0 - torch.clamp(-vz0, min=0.0) ** 2 / (2.0 * A_MAX)   # min(vz0,0)^2 = clamp(-vz0,0)^2
unrecoverable = relaxed_min_z < -BAND

N = x0.shape[0]
rep = {
    "pool": "eval_full_quadrotor-3d-d2r_n2000_seed23456", "pool_sha8": "0ef3751b", "n": int(N),
    "constants": {"f_rotor_max": F, "mass": M, "g": G, "band": BAND,
                  "max_collective_4frmax": 4 * F, "a_max_4frmax_over_m_minus_g": A_MAX},
    "M1_lower_bound": {
        "n_unrecoverable": int(unrecoverable.sum()), "frac_unrecoverable": float(unrecoverable.double().mean()),
        "assumption": "floor is the only vertical constraint; obstacles ignored (altitude subproblem only)",
    },
    # per-IC state + M1 result (M2 will append min_z_best + kappa; M3 classifies)
    "ic": [{"i": i, "z0": float(z0[i]), "vz0": float(vz0[i]), "tilt_deg": float(tilt_deg[i]),
            "omega": float(omega[i]), "clearance": float(clearance[i]),
            "relaxed_min_z": float(relaxed_min_z[i]), "unrecoverable": bool(unrecoverable[i])}
           for i in range(N)],
}
# distributions of the unrecoverable set
u = unrecoverable.numpy()
def dist(v):
    v = v.numpy()
    return {"all_median": float(np.median(v)), "unrec_median": float(np.median(v[u])) if u.any() else None,
            "unrec_p90": float(np.quantile(v[u], 0.9)) if u.any() else None}
rep["M1_lower_bound"]["unrec_distributions"] = {
    "tilt_deg": dist(tilt_deg), "omega": dist(omega), "vz0": dist(vz0), "clearance": dist(clearance)}
OUT.write_text(json.dumps(rep, indent=2) + "\n")
print(f"a_max={A_MAX:.4f}  N={N}")
print(f"M1 UNRECOVERABLE: {int(unrecoverable.sum())} / {N} = {float(unrecoverable.double().mean()):.4f}")
print(f"  unrec median tilt {float(np.median(tilt_deg.numpy()[u])) if u.any() else float('nan'):.1f}deg "
      f"vz0 {float(np.median(vz0.numpy()[u])) if u.any() else float('nan'):.2f} "
      f"clearance {float(np.median(clearance.numpy()[u])) if u.any() else float('nan'):.2f}")
print("wrote", OUT)
