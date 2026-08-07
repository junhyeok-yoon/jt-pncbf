"""Dual-arm (three-arm) full-pool evaluation of the exact-backup safety chain — the committed replacement
for the v2.5.1 scratchpad driver (audit §6/§12). SYSTEM-GENERIC: it takes a `System` and never branches on
`system.name` (the DI-vs-unicycle difference lives entirely in `system.dynamics`/`observation`/`u_bounds`,
the `build_safety_h_fn` channel, and the shield's `backup.py` primitive).

Arms (same checkpoint, same pool, same pinned batch):
  A  — policy + `exact_m0` HardNet filter (no shield).
  B  — arm A + S' successor-verification shield (`shield_eval`, `ladder=None`).
  B' — arm B with the minimal-intervention ladder `[0.75, 0.5, 0.25]`.
  nominal — a state-feedback action (LQR or a policy) with NO filter (the Stage-1 LQR-only baseline).

Metrics reproduce `data/secured_data/v2.5.1/exact_backup/eval/E42_dualeval.json`:
  `cps_v2` uses the 04_eval §1 infeasibility `empty OR (singular AND row<0)`; `cps_legacy` uses
  `singular OR empty`. Both are reported, and the two infeasibility CAUSES (empty; singular-and-violated)
  are recorded separately per step (needed for P-U5). Within-seed percentile bootstrap CI (1000 resamples,
  seed 20260508, `numpy.percentile(..., method="linear")`), per 04_eval §5.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from src.common.filter_hardnet import (_base_alpha, _row_upper, _base_projection, _box_aware_projection, _cbf_terms,
                                        _hardnet_params, _SINGULAR_LG_THRESHOLD)
from src.common.maneuver_value import build_safety_h_fn
from src.common.outcomes import resolve_outcome, step_outcomes
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.evaluate import first_physical_event_step

Tensor = torch.Tensor
_BOOT_SEED = 20260508
_LADDER = [0.75, 0.5, 0.25]


def cps_v2(o: np.ndarray, inf: np.ndarray) -> float:
    reach = (o == "goal").mean(); coll = (o == "collision").mean(); stuck = (o == "stuck").mean()
    oob = (o == "oob").mean(); to = (o == "timeout").mean()
    return float(reach - 2 * coll - stuck - 0.5 * (oob + to) - 0.3 * np.asarray(inf).mean())


def _boot_ci(o: np.ndarray, inf: np.ndarray, n_resample: int = 1000) -> list[float]:
    rng = np.random.default_rng(_BOOT_SEED); N = len(o); idx = rng.integers(0, N, size=(n_resample, N))
    vals = [cps_v2(o[r], inf[r]) for r in idx]
    return [float(np.percentile(vals, 2.5, method="linear")), float(np.percentile(vals, 97.5, method="linear"))]


def _boot_rate_ci(o: np.ndarray, key: str, n_resample: int = 1000) -> list[float]:
    """v2.8.0 B6.3: within-seed bootstrap CI for a single outcome-component RATE (e.g. oob, timeout)."""
    rng = np.random.default_rng(_BOOT_SEED); N = len(o); idx = rng.integers(0, N, size=(n_resample, N))
    vals = [float((o[r] == key).mean()) for r in idx]
    return [float(np.percentile(vals, 2.5, method="linear")), float(np.percentile(vals, 97.5, method="linear"))]


def _rollout(scenes, un_fn: Callable[[Tensor, Any], Tensor], config, h_fn, system, device,
             *, filtered: bool, chunk: int, dtype=torch.float32):
    """Roll every scene; return per-episode (outcome, inf_v2, inf_canonical, empty, sing_viol) active-step
    rates. `un_fn(x, scene) -> nominal action`. filtered=False rolls the nominal directly (nominal arm)."""
    max_steps = int(config["eval"]["max_steps"]); dt = float(config["env"]["dt"])
    params = _hardnet_params(config); bounds = system.u_bounds
    ep = []
    for s0 in range(0, len(scenes), chunk):
        bs = batch_scenes(scenes[s0:s0 + chunk], device=device, dtype=dtype)
        x = initial_states_from_batch(bs).to(dtype); B = x.shape[0]
        empt = torch.zeros(max_steps, B, dtype=torch.bool, device=device)
        sviol = torch.zeros_like(empt); infc = torch.zeros_like(empt); inf2 = torch.zeros_like(empt)
        sat = torch.zeros_like(empt)                                 # v2.8.0 B6.3: per-step saturation flag
        states = [x]
        with torch.no_grad():
            for t in range(max_steps):
                un = un_fn(x, bs)
                if filtered:
                    h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
                    h, lf, lg = h.detach(), lf.detach(), lg.detach()
                    alpha = _base_alpha(h, params); row = _row_upper(lf, alpha, h, params)
                    proj = _base_projection(un, lg, row, bounds, params)
                    sing = torch.linalg.norm(lg, dim=1) < _SINGULAR_LG_THRESHOLD
                    u, empty = _box_aware_projection(un, proj, lg, row, bounds)
                    empt[t] = empty; sviol[t] = sing & (row < 0.0)
                    infc[t] = sing | empty; inf2[t] = empty | (sing & (row < 0.0))
                else:
                    u = torch.clamp(un, min=bounds[:, 0], max=bounds[:, 1])
                sat[t] = (torch.minimum((u - bounds[:, 0]).abs(), (u - bounds[:, 1]).abs()) <= 1.0e-3).any(dim=1)
                x = rk4_step(system, x, u, dt); states.append(x)
        S = torch.stack(states, 0); masks = step_outcomes(S, bs, system, config); res = resolve_outcome(masks)
        an = first_physical_event_step(masks)
        an = torch.where(an >= 0, an, torch.full_like(an, max_steps))
        act = torch.arange(max_steps, device=device).unsqueeze(1) < an.unsqueeze(0)
        for i in range(B):
            am = act[:, i]; na = int(am.sum())
            rate = lambda f: float((f[:, i] & am).sum()) / na if na else 0.0
            ep.append((res.outcome[i], rate(inf2), rate(infc), rate(empt), rate(sviol), rate(sat),
                       res.collision_cause[i] if res.collision_cause else "not_recorded"))
    return ep


def _arm_summary(ep, wall_s):
    o = np.array([e[0] for e in ep]); i2 = np.array([e[1] for e in ep]); ic = np.array([e[2] for e in ep])
    em = np.array([e[3] for e in ep]); sv = np.array([e[4] for e in ep]); st = np.array([e[5] for e in ep])
    cc = np.array([e[6] for e in ep]) if ep and len(ep[0]) > 6 else np.array([""] * len(ep))
    orate = lambda k: float((o == k).mean())
    # v2.8.0: collision-cause decomposition as episode fractions (of the whole pool, so they sum to `collision`)
    ccrate = lambda k: float((cc == k).mean()) if len(cc) else 0.0
    return {"reach": orate("goal"), "collision": orate("collision"), "oob": orate("oob"),
            "stuck": orate("stuck"), "timeout": orate("timeout"),
            # v2.8.0 B6.3: bootstrap CIs for oob and timeout alongside the other outcome components
            "reach_ci": _boot_rate_ci(o, "goal"), "collision_ci": _boot_rate_ci(o, "collision"),
            "stuck_ci": _boot_rate_ci(o, "stuck"), "oob_ci": _boot_rate_ci(o, "oob"),
            "timeout_ci": _boot_rate_ci(o, "timeout"),
            # v2.8.0 M1: collision cause split (obstacle/band_lower/band_upper; sum == collision)
            "collision_obstacle": ccrate("obstacle"), "collision_band_lower": ccrate("band_lower"),
            "collision_band_upper": ccrate("band_upper"),
            "inf_canonical": float(ic.mean()), "inf_v2": float(i2.mean()),
            "inf_empty": float(em.mean()), "inf_singular_violated": float(sv.mean()),
            "saturation_rate": float(st.mean()),        # v2.8.0 B6.3: now carried (was omitted)
            "cps_legacy": cps_v2(o, ic), "cps_v2": cps_v2(o, i2), "cps_v2_ci": _boot_ci(o, i2),
            "wall_s": round(wall_s, 1), "n": len(ep)}


def _shield_summary(r, wall_s, wall_a):
    outc = r["outcome"]; N = len(outc); orate = lambda k: float(np.mean([x == k for x in outc]))
    oarr = np.array(outc)
    reach = orate("goal"); coll = orate("collision"); oob = orate("oob"); stuck = orate("stuck"); to = orate("timeout")
    vs = r["verified_start"].numpy(); cm = np.array([o == "collision" for o in outc])
    cps = reach - 2 * coll - stuck - 0.5 * (oob + to)
    return {"reach": reach, "collision": coll, "oob": oob, "stuck": stuck, "timeout": to,
            "oob_ci": _boot_rate_ci(oarr, "oob"), "timeout_ci": _boot_rate_ci(oarr, "timeout"),
            "saturation_rate": "not recorded",       # v2.8.0 B6.3: shield rollout does not track saturation
            "cps_legacy": cps, "cps_v2": cps, "verified_start_frac": float(vs.mean()),
            "overrides_per_ep": float(r["n_overrides"].float().mean()),
            "checks_per_ep": float(r["n_checks"].float().mean()),
            "shield_overhead_x": round(wall_s / max(wall_a, 1e-9), 2),
            "verified_start_collisions": int((cm & vs).sum()), "wall_s": round(wall_s, 1), "n": N}


def dual_arm_eval(scenes, policy_net, config, system, device, *, arms=("A", "B", "Bprime"), chunk=250,
                  h_fn=None, dtype=torch.float32):
    """Arms A / B / B' for a ControlNet `policy_net` (obs -> action). `h_fn` defaults to
    `build_safety_h_fn(system, config)` (exact_m0 this version). Returns a dict of arm summaries."""
    if h_fn is None:
        h_fn = build_safety_h_fn(system, config, None)
    from src.frameworks.cpi.shield import shield_eval
    un_fn = lambda x, bs: policy_net(system.observation(x, bs))
    out: dict[str, Any] = {"channel_type": str(config.get("safety_channel", {}).get("type", "value")),
                           "n_scenes": len(scenes), "batch_chunk": chunk}
    wall_a = None
    if "A" in arms:
        t0 = time.time(); ep = _rollout(scenes, un_fn, config, h_fn, system, device, filtered=True, chunk=chunk)
        out["arm_A"] = _arm_summary(ep, time.time() - t0); wall_a = out["arm_A"]["wall_s"]
    if "B" in arms:
        t0 = time.time(); r = shield_eval(scenes, policy_net, config, h_fn, system, device, chunk=chunk, ladder=None)
        out["arm_B"] = _shield_summary(r, time.time() - t0, wall_a if wall_a is not None else 1.0)
    if "Bprime" in arms:
        t0 = time.time(); r = shield_eval(scenes, policy_net, config, h_fn, system, device, chunk=chunk, ladder=_LADDER)
        out["arm_Bprime"] = _shield_summary(r, time.time() - t0, wall_a if wall_a is not None else 1.0)
    return out


def nominal_eval(scenes, action_fn, config, system, device, *, chunk=250, dtype=torch.float32):
    """Nominal-only (no filter, no shield) full-pool eval of a state-feedback `action_fn(x, scene) -> u`
    (e.g. `lambda x, sc: system.lqr_action(x, scene_goal_tensor(sc, x))`). Same summary schema as arm A."""
    t0 = time.time()
    ep = _rollout(scenes, action_fn, config, None, system, device, filtered=False, chunk=chunk)
    return _arm_summary(ep, time.time() - t0)
