"""v2.7.6 — diagnose the hover-offset drift, INSIDE the sampled support. The readout column must lie in
|p_y| <= world_lim - 0.3 (03_train 1.2 / _START_GOAL_ARENA_MARGIN); a column at the arena edge reads V_hat as
extrapolation. Reports, per row: the in-support free-space column p_y (NaN if none), the hover-descending
V_hat profile with EVERY sign change and which the detector (min|z-analytic|) selected, and the CLOSED-LOOP
vertical acceleration a_z the policy+filter actually apply along the column. Whether sup_t h is at t=0 depends
on the applied a_z, not the available thrust: at descent w=1.5 and c_z=pi/omega_max the offset sits near zero
only while a_z > w/c_z; an interior offset with a_z below that is an informative reading of the policy's
altitude effort, not a defect. Eval-only, forward passes + one-step dynamics, read-only on the live run dir."""
from __future__ import annotations

import math as _m, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.eval.run_full import _load_framework as L
from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
INLOOP = REPO / "data/eval_pools/eval_inloop_quadrotor-3d-d2r_n500_seed12345.pkl"


def _az(fw, sys_, scene_t, py, pz, vz, dt, dev):
    """closed-loop world-vertical acceleration at (px=0, py, pz, hover, v_z=vz): policy -> filter -> dynamics[9]."""
    x = torch.zeros(1, 13, dtype=dt, device=dev)
    x[0, 1] = py; x[0, 2] = pz; x[0, 3] = 1.0; x[0, 9] = vz
    if hasattr(fw, "reset_deficit_state"):
        fw.reset_deficit_state()
    with torch.no_grad():
        u_nom = fw.policy(x, scene_t)
        u_safe, _ = fw.filter(x, u_nom, scene_t)
        xdot = sys_.dynamics(x, u_safe)
    return float(xdot[0, 9])


def probe(ckpt: Path, res: int = 61):
    fw, cfg, ck = L(ckpt)
    cfg = dict(cfg); cfg["env"] = dict(cfg["env"]); cfg["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    sys_, vn = fw.system, fw.value_net
    world = float(cfg["env"]["world_lim"]); dz = 2 * world / (res - 1); support = world - 0.3
    c_z = _m.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"]); w = 1.5
    az_thresh = w / c_z
    param = next(vn.parameters()); dt, dev = param.dtype, param.device
    axis = np.linspace(-world, world, res)
    scenes = load_pool(INLOOP).scenes[:2]
    vz = -1.5; analytic = -4.0 - c_z * vz
    print(f"=== {ckpt.name} (step {ck.get('step')}) hover descending vz={vz}, c_z={c_z:.4f}, analytic pz={analytic:.3f}, "
          f"dz={dz:.3f}, support|p_y|<={support}, a_z threshold w/c_z={az_thresh:.3f} m/s2 ===")
    for ri, scene in enumerate(scenes):
        C = np.asarray(scene.obstacle_centers, np.float64)[:, :2]; R = np.asarray(scene.obstacle_radii, np.float64)
        Aa = np.asarray(scene.obstacle_active, bool)
        st = SimpleNamespace(obstacle_centers=torch.tensor(C, dtype=dt, device=dev),
            obstacle_radii=torch.tensor(R, dtype=dt, device=dev), obstacle_active=torch.tensor(Aa, dtype=torch.bool, device=dev),
            goal=torch.tensor(np.asarray(scene.goal, np.float64), dtype=dt, device=dev))
        clr = np.full(res, np.inf)
        for j in np.nonzero(Aa)[0]:
            clr = np.minimum(clr, np.abs(axis - C[j, 1]) - (_m.sqrt(max(R[j]**2 - C[j, 0]**2, 0.0)) if abs(C[j, 0]) < R[j] else -abs(C[j, 0])))
        insup = np.where(np.abs(axis) <= support)[0]
        fp = int(insup[np.argmax(clr[insup])]); py = float(axis[fp])
        if not clr[fp] > 0.0:
            print(f"row{ri}: NO in-support free-space column (best in-support clr={clr[fp]:.3f}) -> offset NaN"); continue
        x = torch.zeros(res, 13, dtype=dt, device=dev)
        x[:, 1] = py; x[:, 2] = torch.tensor(axis, dtype=dt, device=dev); x[:, 3] = 1.0; x[:, 9] = vz
        with torch.no_grad():
            v = vn.deployed_h(sys_.observation(x, st)).reshape(-1).cpu().numpy()
        cross = [float(axis[k] + (axis[k+1]-axis[k]) * (0-v[k]) / (v[k+1]-v[k]))
                 for k in range(res-1) if (v[k] <= 0 <= v[k+1]) or (v[k] >= 0 >= v[k+1])]
        sel = min(cross, key=lambda z: abs(z - analytic)) if cross else None
        print(f"row{ri}: in-support free_py={py:.3f} ({abs(py)<=support and 'INSIDE' or 'OUTSIDE'} support)")
        print(f"   ALL sign changes (pz): {[round(z,3) for z in cross]}"
              + (f"  -> detector selected pz={sel:+.3f}, offset={sel-analytic:+.3f} ({abs(sel-analytic)/dz:.1f} cells)" if sel is not None else "  -> NaN"))
        # closed-loop a_z at the readout (crossing) state, and along the descent for context
        probe_pz = [(-4.0, "floor"), (analytic, "analytic"), (-2.0, "-2.0"), (-1.0, "-1.0"), (0.0, "0.0")]
        if sel is not None:
            probe_pz.insert(0, (sel, "CROSSING"))
        azs = [(lab, round(pz, 2), round(_az(fw, sys_, st, py, pz, vz, dt, dev), 3)) for pz, lab in probe_pz]
        print(f"   closed-loop a_z (m/s2) [threshold {az_thresh:.2f}]: " +
              "  ".join(f"{lab}@{pz}: {a:+.2f}{'*' if a < az_thresh else ''}" for lab, pz, a in azs))


if __name__ == "__main__":
    JT = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints"
    steps = sys.argv[1].split(",") if len(sys.argv) > 1 else ["42000"]
    for s in steps:
        probe(JT / f"step_{int(s):06d}.pt")
