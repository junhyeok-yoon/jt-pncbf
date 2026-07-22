"""v2.7.1 Stage-1 — k-step empty-branch fallback: mode=none bit-parity (f1), flag-invariance + empty-only
action change (f2), kstep_select determinism + grid membership (f3). Uses the M5 value net (skip if absent) to
produce real empty rows."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.filter_hardnet import HardNetFilter
from src.common.kstep_fallback import grid_controls, kstep_select
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_cell_state_scene, sample_train_scene
from src.frameworks.jt_pncbf.train import make_system

REPO = Path(__file__).resolve().parents[1]
# secured iter-5 checkpoint (sha8 3b27d691); the v2.7.1 run dir was archived to previous_runs at close.
M5 = REPO / "data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt"


def _load():
    if not M5.exists():
        pytest.skip("M5 checkpoint absent")
    ck = torch.load(M5, map_location="cpu", weights_only=False); cfg = ck["config"]
    system = make_system(cfg)
    vnet = ValueNetEnsemble(system.obs_dim, cfg); vnet.load_state_dict(ck["v_s_state"]); vnet.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    return system, make_h_fn(vnet, system), cfg


def _cell_batch(system, cfg, n=300):
    rng = np.random.default_rng(0)
    scenes = [sample_cell_state_scene(sample_train_scene(rng, cfg, "quadrotor_planar"), rng, cfg) for _ in range(n)]
    bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
    x = system.wrap_state(initial_states_from_batch(bs).float())
    return x, bs


def _filt(cfg, fb, system, h_fn):
    c = copy.deepcopy(cfg)
    if fb is None:
        c["filter"].pop("empty_fallback", None)
    else:
        c["filter"]["empty_fallback"] = fb
    return HardNetFilter(system, h_fn, c)


def _unom(system, x, bs):
    g = torch.as_tensor(bs.goal, dtype=x.dtype)
    return system.lqr_action(x, g if g.ndim > 1 else g.unsqueeze(0).expand(x.shape[0], -1))


def test_f1_mode_none_bit_parity() -> None:
    system, h_fn, cfg = _load(); x, bs = _cell_batch(system, cfg); un = _unom(system, x, bs)
    f_absent = _filt(cfg, None, system, h_fn)                 # no empty_fallback key
    f_none = _filt(cfg, {"mode": "none", "k": 10}, system, h_fn)
    ua, fa = f_absent(x, bs, un); unn, fn = f_none(x, bs, un)
    assert torch.equal(ua, unn), "mode=none action differs from absent-block"
    assert torch.equal(fa, fn), "mode=none flag differs from absent-block"
    assert torch.equal(f_absent.last_empty, f_none.last_empty)
    assert torch.equal(f_absent.last_singular, f_none.last_singular)


def test_f2_flag_invariance_empty_only_action_change() -> None:
    system, h_fn, cfg = _load(); x, bs = _cell_batch(system, cfg); un = _unom(system, x, bs)
    f_none = _filt(cfg, {"mode": "none", "k": 10}, system, h_fn)
    f_k = _filt(cfg, {"mode": "kstep", "k": 10}, system, h_fn)
    u0, fl0 = f_none(x, bs, un); uk, flk = f_k(x, bs, un)
    em = f_none.last_empty
    assert bool(em.any()), "batch produced no empty rows — test would be vacuous"
    # flags identical (combined AND split): the fallback never touches a flag
    assert torch.equal(fl0, flk)
    assert torch.equal(f_none.last_empty, f_k.last_empty)
    assert torch.equal(f_none.last_singular, f_k.last_singular)
    # actions identical on NON-empty rows; changed on (at least some) empty rows
    assert torch.equal(u0[~em], uk[~em]), "kstep changed a non-empty row"
    assert not torch.equal(u0[em], uk[em]), "kstep changed no empty row"


def test_f3_kstep_select_determinism_and_grid() -> None:
    system, h_fn, cfg = _load(); x, bs = _cell_batch(system, cfg, 80)
    G = grid_controls(system, torch.device("cpu"))
    assert G.ndim == 2 and G.shape[1] == 2 and 2 <= G.shape[0] <= 16
    u1a, ma = kstep_select(x, bs, h_fn, system, G, 10, float(cfg["env"]["dt"]))
    u1b, mb = kstep_select(x, bs, h_fn, system, G, 10, float(cfg["env"]["dt"]))
    assert torch.equal(u1a, u1b) and torch.equal(ma, mb), "kstep_select nondeterministic"
    # every returned first-phase control is a grid member (fixed-order argmin)
    match = (u1a.unsqueeze(1) == G.unsqueeze(0)).all(dim=2).any(dim=1)
    assert bool(match.all()), "u1_star not a grid control"


# ---- Stage-1c: DI + unicycle f1 (bit-parity) + f2 (flag-invariance) on their SOTA checkpoints ----
import pytest as _pytest
from src.eval.build_pools import load_pool
from src.common.rk4 import rk4_step

_SYS = {
    "double_integrator": (REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt",
                          REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"),
    "unicycle": (REPO / "data/runs/v2.2.2/v2.2.2__20260619-083424__seed42/checkpoints/best.pt",   # v2.7.4 migration
                 REPO / "data/secured_data/pools/eval_full_unicycle_n2000_seed23456.pkl"),
}


def _rolled_batch_with_empty(ck_path, pool_path, fb, n=250, roll=25):
    if not ck_path.exists():
        _pytest.skip(f"{ck_path} absent")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False); cfg = copy.deepcopy(ck["config"])
    if fb is None:
        cfg["filter"].pop("empty_fallback", None)
    else:
        cfg["filter"]["empty_fallback"] = fb
    from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint
    fw, cfg2, _ = load_framework_from_checkpoint(ck_path, config_overrides={"filter": cfg["filter"]})
    scenes = load_pool(pool_path).scenes[:n]
    bs = batch_scenes(scenes, device=torch.device("cpu"), dtype=torch.float32)
    x = fw.system.wrap_state(initial_states_from_batch(bs).float())
    dt = float(cfg2["env"]["dt"])
    with torch.no_grad():                                    # roll to reach states where the empty branch fires
        for _ in range(roll):
            un = fw.policy(x, bs); u, _ = fw.filter(x, un, bs); x = rk4_step(fw.system, x, u, dt)
    return fw, x, bs


@_pytest.mark.parametrize("sysname", ["double_integrator", "unicycle"])
def test_f1f2_cross_system(sysname):
    ck_path, pool_path = _SYS[sysname]
    fw_a, x, bs = _rolled_batch_with_empty(ck_path, pool_path, None)
    un = fw_a.policy(x, bs)
    ua, fa = fw_a.filter(x, un, bs)                          # absent-block
    fw_n, _, _ = _rolled_batch_with_empty(ck_path, pool_path, {"mode": "none", "k": 5})
    u0, f0 = fw_n.filter(x, un, bs)                          # mode=none
    # f1 bit-parity: none == absent-block (action + flag)
    assert torch.equal(ua, u0) and torch.equal(fa, f0)
    fw_k, _, _ = _rolled_batch_with_empty(ck_path, pool_path, {"mode": "kstep", "k": 5})
    uk, fk = fw_k.filter(x, un, bs)
    em = fw_n._filter.last_empty
    assert bool(em.any()), f"{sysname}: no empty rows in the rolled batch (test vacuous)"
    # f2 flag-invariance + empty-only action change
    assert torch.equal(f0, fk)
    assert torch.equal(fw_n._filter.last_empty, fw_k._filter.last_empty)
    assert torch.equal(u0[~em], uk[~em]), f"{sysname}: kstep changed a non-empty row"
    assert not torch.equal(u0[em], uk[em]), f"{sysname}: kstep changed no empty row"


# ---- Stage-3D: quadrotor_3d f1 (mode=none bit-parity) + f2 (flag-invariance under mode=kstep) ----
# Exercises the shared 4D grid (16 corners + center + zero + per-axis extremes, deduped -> |G|=25) and
# kstep_select on the 4-input box. M6 JT checkpoint; skip if absent (measurement run dir, not secured).
_M6_3D = REPO / "data/runs/v2.7.2/set__20260718-204313__seed42/v2.7.2__20260718-212348__seed42/checkpoints/best.pt"   # v2.7.4 migration
_POOL_3D = REPO / "data/secured_data/pools/eval_full_quadrotor-3d_n2000_seed23456.pkl"


def test_f1f2_quadrotor_3d():
    # v2.7.3 M0b: the per-rotor actuator set makes the v2.7.2 (wrench-plant) checkpoint plant-incompatible
    # (its config lacks the per-rotor bound/mixer keys). Skip until a v2.7.3 3D checkpoint exists (Stage B).
    if _M6_3D.exists():
        _ck = torch.load(_M6_3D, map_location="cpu", weights_only=False)
        if "f_rotor_max" not in _ck["config"]["env"]["bounds"].get("quadrotor_3d", {}):
            _pytest.skip("v2.7.2 quadrotor_3d checkpoint is plant-incompatible under the v2.7.3 per-rotor set")
    fw_a, x, bs = _rolled_batch_with_empty(_M6_3D, _POOL_3D, None, n=400, roll=40)
    un = fw_a.policy(x, bs)
    ua, fa = fw_a.filter(x, un, bs)                           # absent-block
    fw_n, _, _ = _rolled_batch_with_empty(_M6_3D, _POOL_3D, {"mode": "none", "k": 5}, n=400, roll=40)
    u0, f0 = fw_n.filter(x, un, bs)                           # mode=none
    assert torch.equal(ua, u0) and torch.equal(fa, f0)       # f1 bit-parity: none == absent-block
    fw_k, _, _ = _rolled_batch_with_empty(_M6_3D, _POOL_3D, {"mode": "kstep", "k": 5}, n=400, roll=40)
    uk, fk = fw_k.filter(x, un, bs)
    em = fw_n._filter.last_empty
    assert bool(em.any()), "quadrotor_3d: no empty rows in the rolled batch (test vacuous)"
    # f2 flag-invariance + empty-only action change (the fallback never touches a flag; non-empty rows fixed)
    assert torch.equal(f0, fk)
    assert torch.equal(fw_n._filter.last_empty, fw_k._filter.last_empty)
    assert torch.equal(fw_n._filter.last_singular, fw_k._filter.last_singular)
    assert torch.equal(u0[~em], uk[~em]), "quadrotor_3d: kstep changed a non-empty row"
    assert not torch.equal(u0[em], uk[em]), "quadrotor_3d: kstep changed no empty row"
