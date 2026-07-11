"""Durable cps-v2 re-scorer (v2.5.0 metrics amendment). Loads a JT checkpoint (v_s_state + pi_state),
rolls the framework-native learned HardNet deploy filter on the n2000 full pool (seed23456, dt=0.05,
deterministic), and emits legacy cps | cps_v2 | empty-only + components + per-episode infeasible flags.

cps-v2 per-step flag: infeasible_v2 = empty OR (singular AND row_upper<0), row_upper = -L_f V - alpha*V.
At singular steps L_g V~0 so the CBF row is u-independent and holds iff row_upper>=0; a far-field-flat
step with the row already satisfied is FEASIBLE, not infeasible (removes the far-field-singular inflation).

Usage: python -m scripts.analysis.rescore_cps_v2 <ckpt_path> [<label>]
Output: <ckpt_dir>/cps_v2_rescore.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.common.control_net import ControlNet
from src.common.filter_hardnet import (
    _SINGULAR_LG_THRESHOLD, _base_alpha, _base_projection, _box_aware_projection, _cbf_terms,
    _hardnet_params,
)
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.eval.build_pools import load_pool
from src.eval.evaluate import first_physical_event_step
from src.envs.double_integrator import DoubleIntegrator
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.frameworks.jt_pncbf.train import JTPNCBFFramework
from scripts.analysis.deploy_rate_eval import infeasible_v2

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"


def rescore(ckpt_path: Path, label: str = "") -> dict:
    dev = torch.device("cuda"); DT = torch.float32
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    system = DoubleIntegrator(cfg); system.u_bounds = system.u_bounds.to(device=dev, dtype=DT)
    vn = ValueNetEnsemble(system.obs_dim, cfg).to(device=dev, dtype=DT); vn.load_state_dict(ck["v_s_state"]); vn.eval()
    odim = system.obs_dim + (system.action_dim if cfg.get("loss", {}).get("policy", {}).get("obs_deficit_feedback") else 0)
    pn = ControlNet(odim, system, cfg).to(device=dev, dtype=DT); pn.load_state_dict(ck["pi_state"]); pn.eval()
    fw = JTPNCBFFramework(system, vn, pn, cfg)
    params = _hardnet_params(cfg); dt = float(cfg["env"]["dt"]); max_steps = int(cfg["eval"]["max_steps"])
    bounds = system.u_bounds; h_fn = make_h_fn(vn, system)
    scenes = load_pool(POOL).scenes; ep = []
    for s0 in range(0, len(scenes), 250):
        bs = batch_scenes(scenes[s0:s0 + 250], device=dev, dtype=DT)
        x = initial_states_from_batch(bs).to(DT); B = x.shape[0]
        infc = torch.zeros(max_steps, B, dtype=torch.bool, device=dev); empc = torch.zeros_like(infc); inf2c = torch.zeros_like(infc)
        fw.reset_deficit_state(); states = [x]
        with torch.no_grad():
            for t in range(max_steps):
                un = fw.policy(x, bs)
                us, _ = fw.filter(x, un, bs)                                   # framework-native deploy filter
                h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                h, lf, lg = h.detach(), lf.detach(), lg.detach()
                al = _base_alpha(h, params); row = -lf - al * h
                proj = _base_projection(un, lg, row, bounds, params)
                sg = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
                _, em = _box_aware_projection(un, proj, lg, row, bounds)
                infc[t] = sg | em; empc[t] = em; inf2c[t] = infeasible_v2(sg, em, row)
                x = rk4_step(system, x, us, dt); states.append(x)
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, cfg); res = resolve_outcome(masks)
        ev = first_physical_event_step(masks); an = torch.where(ev >= 0, ev, torch.full_like(ev, max_steps))
        act = torch.arange(max_steps, device=dev).unsqueeze(1) < an.unsqueeze(0)
        for i in range(B):
            am = act[:, i]; na = int(am.sum()); r = lambda f: float((f[:, i] & am).sum()) / na if na else 0.0
            ep.append(dict(outc=res.outcome[i], canon=r(infc), empty=r(empc), v2=r(inf2c), sing_only=r(infc & ~empc)))
    orate = lambda o: float(np.mean([e["outc"] == o for e in ep]))
    reach, coll, oob, stuck, to = (orate(o) for o in ("goal", "collision", "oob", "stuck", "timeout"))
    cps = lambda k: reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * float(np.mean([e[k] for e in ep]))
    out = dict(label=label, ckpt=str(ckpt_path), n=len(ep), reach=round(reach, 4), coll=round(coll, 4),
               oob=round(oob, 4), stuck=round(stuck, 4), timeout=round(to, 4),
               cps_legacy=round(cps("canon"), 4), cps_v2=round(cps("v2"), 4), cps_empty=round(cps("empty"), 4),
               inf_canon=round(float(np.mean([e["canon"] for e in ep])), 4),
               inf_v2=round(float(np.mean([e["v2"] for e in ep])), 4),
               inf_empty=round(float(np.mean([e["empty"] for e in ep])), 4),
               sing_only=round(float(np.mean([e["sing_only"] for e in ep])), 4))
    # NOTE: secured_data/ is read-only -> outputs go to a writable analysis dir keyed by label, NOT next
    # to the checkpoint (would violate read-only-on-secured for the v2.3.0/v2.2.2 secured checkpoints).
    outdir = REPO / "docs/versions/v2.5.0/rescore"; outdir.mkdir(parents=True, exist_ok=True)
    tag = (label or ckpt_path.parent.parent.name).replace("/", "_")
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    r = rescore(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "")
    print(json.dumps(r))
