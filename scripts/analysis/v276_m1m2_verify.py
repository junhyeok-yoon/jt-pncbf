"""v2.7.6 Stage-2 M1/M2 verification. Confirms: (M2) d h_star/d v_z = +/- c_z != 0 including inside the psi
cap, relative degree 1; the vertical branch is off when disabled (legacy bit-parity); the psi cap bounds the
position term at r_max. (M1) the band collision predicate fires on |p_z|>=4 and is off when the limit is 0."""
from __future__ import annotations

import copy, json, math
from pathlib import Path
from typing import Mapping

import numpy as np, torch, yaml

from src.common.quadrotor_barrier import value_target_barrier
from src.common.outcomes import step_outcomes
from src.envs.quadrotor_3d import QuadrotorQuad3D as Quadrotor3D
from src.envs.scene_batch import batch_scenes
from src.eval.build_pools import load_pool

REPO = Path("/home/junhyeok/MIT/jt-pncbf")


def merged():
    b = yaml.safe_load((REPO / "src/configs/base_config.yaml").read_text())
    e = yaml.safe_load((REPO / "src/configs/exp_config.yaml").read_text())
    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d
    return m(b, e)


def main():
    cfg = merged()
    cfg["run"]["system"] = "quadrotor_3d"
    cfg["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    cfg["env"]["band_collision_limit"] = 4.0
    sys = Quadrotor3D(cfg)
    r_max = float(cfg["obstacle"]["per_system"]["quadrotor_3d"]["r_max"])
    c_z = math.pi / float(cfg["env"]["bounds"]["quadrotor_3d"]["omega_max"])
    sc = load_pool(REPO / "data/runs/v2.7.6/pools/eval_fullrange_quadrotor-3d-d2r_n2000_seed42.pkl").scenes[0]
    bs = batch_scenes([sc], device=torch.device("cpu"), dtype=torch.float64)

    def state(z, vz):
        x = torch.zeros(1, 13, dtype=torch.float64)
        x[0, 0] = 6.0; x[0, 1] = 6.0                 # xy far from obstacles (phi small/negative)
        x[0, 2] = z; x[0, 3] = 1.0; x[0, 9] = vz     # z, quat=identity, v_z
        return x

    out = {"c_z": round(c_z, 6), "psi_cap_r_max": r_max}
    # M2a: ceiling branch active (z inside cap, ascending) -> d h/d v_z = +c_z
    x = state(4.5, 1.0).requires_grad_(True)         # z-limit = 0.5 < r_max (inside cap), ascending
    h = value_target_barrier(sys, x, bs, cfg); g = torch.autograd.grad(h.sum(), x)[0][0, 9].item()
    out["ceiling_dh_dvz"] = round(g, 6); out["ceiling_matches_+c_z"] = abs(g - c_z) < 1e-9
    # M2b: floor branch active (descending) -> d h/d v_z = -c_z
    x = state(-4.5, -1.0).requires_grad_(True)
    h = value_target_barrier(sys, x, bs, cfg); g = torch.autograd.grad(h.sum(), x)[0][0, 9].item()
    out["floor_dh_dvz"] = round(g, 6); out["floor_matches_-c_z"] = abs(g + c_z) < 1e-9
    # M2c: INSIDE the cap (deep excursion, z-limit >> r_max) -> still d h/d v_z = +c_z (clamp on z, not v_z)
    x = state(10.0, 1.0).requires_grad_(True)        # z-4 = 6 >> r_max=0.8 -> capped
    h0 = float(value_target_barrier(sys, x, bs, cfg))
    g = torch.autograd.grad(value_target_barrier(sys, x, bs, cfg).sum(), x)[0][0, 9].item()
    out["deep_cap_dh_dvz"] = round(g, 6); out["deep_cap_matches_+c_z"] = abs(g - c_z) < 1e-9
    out["psi_at_z10_capped"] = round(h0 - c_z * 1.0, 6); out["cap_holds"] = abs((h0 - c_z * 1.0) - r_max) < 1e-6
    # M2d: disabled -> vertical branch off (legacy). Same state, band_hazard off.
    cfg_off = copy.deepcopy(cfg); cfg_off["env"]["band_hazard"]["enabled"] = False
    x = state(10.0, 1.0)
    h_on = float(value_target_barrier(sys, x, bs, cfg)); h_off = float(value_target_barrier(sys, x, bs, cfg_off))
    out["disabled_drops_vertical"] = (h_off < h_on) and (h_off < r_max)
    # M1: band collision predicate
    traj = torch.zeros(3, 1, 13, dtype=torch.float64)
    traj[:, 0, 0] = 6.0; traj[:, 0, 1] = 6.0; traj[:, 0, 3] = 1.0
    traj[0, 0, 2] = 0.0; traj[1, 0, 2] = 4.1; traj[2, 0, 2] = -4.2   # in-band, ceiling, floor
    m_on = step_outcomes(traj, sc, sys, cfg).collided.reshape(-1).tolist()
    m_off = step_outcomes(traj, sc, sys, cfg_off | {"env": {**cfg_off["env"], "band_collision_limit": 0.0}}).collided.reshape(-1).tolist()
    out["M1_collided_banded_[inband,ceil,floor]"] = m_on
    out["M1_collided_legacy_[inband,ceil,floor]"] = m_off
    out["M1_banded_fires_on_band"] = (m_on == [False, True, True])
    out["M1_legacy_no_band_collision"] = (m_off == [False, False, False])

    OUT = REPO / "data/runs/v2.7.6/stage2_m0"; OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m1m2_verify.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    allok = all([out["ceiling_matches_+c_z"], out["floor_matches_-c_z"], out["deep_cap_matches_+c_z"],
                 out["cap_holds"], out["disabled_drops_vertical"], out["M1_banded_fires_on_band"],
                 out["M1_legacy_no_band_collision"]])
    print("ALL M1/M2 CHECKS PASS:", allok)


if __name__ == "__main__":
    main()
