"""v2.4.2 decline forensics C — direct eps0 probe vs brake comparator (read-only, throwaway).

Compare deployed h-hat of two checkpoints (early peak vs late) against the brake-rollout
ground-truth V_brake on the SAME states — states sampled from the LATE policy's deployment tube
(active rollout states on the frozen in-loop pool, natural velocity spread). Reports signed error
(h-hat - V_brake) on {V_brake<=0} and the band {|V_brake|<=0.2}. V_brake = max-over-time of the
braked signed-h sequence (a conservative certificate value); use the peak->late TREND, not levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.brake_rollout import brake_h_rollout
from src.common.outcomes import step_outcomes
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options, first_physical_event_step
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_inloop_di_n500_seed12345.pkl"


def _gather_states(framework, config, scenes, n, seed):
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework),
                           batched, x0, max_steps, dt, config=config)
    masks = step_outcomes(res.states, batched, system, config)
    event = first_physical_event_step(masks)
    active_n = torch.where(event >= 0, event, torch.full_like(event, max_steps))
    T = res.states.shape[0]
    t_idx = torch.arange(T, device=device).unsqueeze(1)
    active = t_idx < active_n.unsqueeze(0)                    # [T,B]
    ti, bi = torch.where(active)
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(ti.numel(), generator=g)[: min(n, ti.numel())]
    ti, bi = ti[sel], bi[sel]
    states = res.states[ti, bi].detach().cpu()
    return states, bi.detach().cpu().numpy()


def _h_hat(framework, states, scenes_sel):
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    batched = batch_scenes(scenes_sel, device=device, dtype=dtype)
    x = states.to(device=device, dtype=dtype)
    with torch.no_grad():
        return framework._filter.h_fn(x, batched).reshape(-1).detach().cpu().numpy()


def _v_brake(framework, config, states, scenes_sel):
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    batched = batch_scenes(scenes_sel, device=device, dtype=dtype)
    x = states.to(device=device, dtype=dtype)
    u_max = float(config["env"]["bounds"][system.name]["u_max"])
    bcfg = config["value_target"].get("brake", {})   # v2.3.0 config predates the brake block
    eps_v = float(bcfg.get("eps_v", 0.05))
    T_b = max(30, int(bcfg.get("T_b", 30)))
    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])
    h_seq, _ = brake_h_rollout(x, batched, system, system.observation,
                               T_b, u_max, eps_v, dt, h_scale)
    return torch.clamp(h_seq, -1.0, 1.0).max(dim=1).values.detach().cpu().numpy()


def _q(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}


def main() -> int:
    ap = argparse.ArgumentParser(description="eps0 probe vs brake comparator (read-only).")
    ap.add_argument("--early", type=Path, required=True)
    ap.add_argument("--late", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", type=str, default="eps0_probe")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.2")
    a = ap.parse_args()

    scenes = load_pool(a.pool).scenes
    fw_late, cfg_late, _ = _load_framework(a.late)
    states, bi = _gather_states(fw_late, cfg_late, scenes, a.n, a.seed)
    scenes_sel = [scenes[b] for b in bi]

    vb = _v_brake(fw_late, cfg_late, states, scenes_sel)
    hl = _h_hat(fw_late, states, scenes_sel)
    fw_early, _, _ = _load_framework(a.early)
    he = _h_hat(fw_early, states, scenes_sel)

    err_e, err_l = he - vb, hl - vb
    safe = vb <= 0.0
    band = np.abs(vb) <= 0.2
    out = {
        "early": str(a.early), "late": str(a.late), "pool": str(a.pool),
        "n": int(states.shape[0]), "n_safe": int(safe.sum()), "n_band": int(band.sum()),
        "v_brake": _q(vb),
        "err_early_safe": _q(err_e[safe]), "err_late_safe": _q(err_l[safe]),
        "err_early_band": _q(err_e[band]), "err_late_band": _q(err_l[band]),
        "hhat_early_band": _q(he[band]), "hhat_late_band": _q(hl[band]),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
