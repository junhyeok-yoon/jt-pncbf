"""v2.8.4 MPPI baseline — CPU correctness checks (TINY settings, seconds; NO GPU work).

Twelve checks, written to `data/runs/v2.8.4/mppi_screen_v3/cpu_smoke_v3.json`. The earlier screens'
records — `data/runs/v2.8.4/mppi_screen/cpu_smoke.json` (S1) and
`data/runs/v2.8.4/mppi_screen_v2/cpu_smoke_v2.json` (S2) — are RETAINED UNTOUCHED; this run writes only
under `mppi_screen_v3/`. Checks (a)-(h) are the earlier ones and must STILL PASS under the S3 defaults
(hover-centered sampling, the settling terminal cost, control hold 1); (i)-(l) are S3's own:

  a. box     — the controller's applied actions lie inside the per-rotor box read off `system.u_bounds`
               on a few real pool scenes;
  b. rollout — one short episode batch runs end to end through the shared eval path and resolves to sane
               outcomes;
  c. degen   — the degenerate-sample branch actually fires and is counted (a constructed all-collide
               state, plus a natural tiny-N / huge-C_crash sweep over real pool scenes);
  d. predicate — the cost's collision indicator agrees EXACTLY with `src.common.outcomes.step_outcomes`
               on a batch of states, including states placed on the obstacle and band boundaries;
  e. chunk   — sample-axis chunking of the rollout is exactly equivalent, metric for metric;

  and, added by the v2.8.4 amendment (R1/R2/R3):

  f. alloc   — the allocation round-trip: for wrenches whose rotor image is strictly inside the box,
               `mixer_inv` then `mixer` returns the wrench to <= 1e-10, and the hover trim wrench maps to
               `mass * gravity / 4` per rotor;
  g. ou      — the OU perturbation's statistics: over >= 200k samples the empirical per-channel std
               matches sigma_j at EVERY horizon index k, and the lag-1 autocorrelation matches alpha;
  h. ess     — the relative-lambda formula cannot produce ESS ~= 1 by a scale mismatch: on synthetic
               costs spanning 12 orders of magnitude the ESS at lam_rel = 0.2 is scale-INVARIANT and far
               above 1, whereas the legacy absolute lambda collapses to ~1 as the scale grows;

  and, added by S3:

  i. center  — hover-centered samples AVERAGE TO HOVER THRUST. Over a large sample the mean commanded
               wrench equals the trim wrench (m*g, 0, 0, 0) and the mean per-rotor allocation equals
               mass*gravity/4, both within the sampling tolerance k * SE stated in the record. The
               post-clip mean and the clip rate are reported beside it as measurement: the box clip is a
               real, separate effect and is not swept into the centering claim;
  j. terminal— the SETTLING terminal cost is ZERO EXACTLY on states satisfying the deployed terminal
               predicate and STRICTLY POSITIVE off it. A property test over a constructed batch that
               includes boundary-adjacent points and, separately, states violating each of the three
               conditions ALONE. The predicate is evaluated through the same norms the cost uses, which
               is what makes "exactly" meaningful at the boundary;
  k. hold    — control hold m produces piecewise-constant plans of block length EXACTLY m over the
               rollout, the closed loop re-applies the latched action between decisions and shifts the
               plan only at decision steps, and m = 1 is the unheld controller (every step a decision);
  l. equiv   — the (G) question: is hover centering behaviourally identical to the uncentered path? Both
               centres are run over real pool scenes in float64 AND float32 and the deviation is
               MEASURED — actions, plans, the degenerate branch and the resolved metrics — rather than
               asserted.

Run:  CUDA_VISIBLE_DEVICES="" python -m src.frameworks.mppi.cpu_smoke
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.common.outcomes import step_outcomes
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.frameworks.mppi.cost import CostParams, collision_mask, terminal_cost
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    build_framework,
    effective_config,
    load_mppi_config,
    run_cell,
)
from src.frameworks.mppi.mppi_controller import MPPIController, MPPIParams


OUT = REPO / "data/runs/v2.8.4/mppi_screen_v3/cpu_smoke_v3.json"
DEVICE = torch.device("cpu")
DTYPE = torch.float64          # CPU checks run in float64; the GPU cells run float32


def _pool_path(mppi_config) -> Path:
    return REPO / mppi_config["screen"]["pool"]


def _controller(
    config, mppi_config, *, n_samples, horizon, lam, c_crash, seed=42,
    center=None, control_hold=None, terminal=None, dtype=DTYPE,
):
    _, framework, params, cost = build_framework(
        config, mppi_config, n_samples=n_samples, horizon=horizon, lam=lam,
        c_crash=c_crash, seed=seed, center=center, control_hold=control_hold, terminal=terminal,
        device=DEVICE, dtype=dtype,
    )
    return framework, params, cost


def check_box(config, mppi_config, scenes) -> dict:
    """(a) applied actions inside the per-rotor box on a few scenes."""
    framework, params, _ = _controller(config, mppi_config, n_samples=8, horizon=5, lam=0.2, c_crash=1e3)
    system = framework.system
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)
    x = system.wrap_state(initial_states_from_batch(batched))
    lo = float(system.u_bounds[:, 0].min())
    hi = float(system.u_bounds[:, 1].max())
    applied = []
    for _ in range(10):
        u = framework.policy(x, batched)
        applied.append(u)
        x = rk4_step(system, x, u, float(config["env"]["dt"]))
    u_all = torch.stack(applied)
    return {
        "n_scenes": len(scenes), "n_control_steps": 10,
        "N": params.n_samples, "H": params.horizon,
        "u_bounds_read_from_system": [lo, hi],
        "u_min": float(u_all.min()), "u_max": float(u_all.max()),
        "all_inside_box": bool((u_all >= lo - 1e-12).all() and (u_all <= hi + 1e-12).all()),
        "n_actions_checked": int(u_all.numel()),
        "PASS": bool((u_all >= lo - 1e-12).all() and (u_all <= hi + 1e-12).all()),
    }


def check_rollout(mppi_config, pool_path) -> dict:
    """(b) one short rollout end to end, through the shared eval path."""
    started = time.time()
    cell = run_cell(
        pool_path=pool_path, n_scenes=4, ebs=4, n_samples=8, horizon=5, lam=0.2, c_crash=1e3,
        sigma=None, seed=42, label="cpu_smoke_tiny", out_dir=Path("/dev/null"),
        device=DEVICE, dtype=DTYPE, max_steps=10, mppi_config=mppi_config, write=False,
    )
    return {
        "n_scenes": 4, "N": 8, "H": 5, "max_steps": 10,
        "outcome_counts": cell["outcome_counts"],
        "cps": cell["cps"], "reach": cell["reach"], "collision": cell["collision"],
        "timeout": cell["timeout"], "oob": cell["oob"], "stuck": cell["stuck"],
        "infeasibility": cell["infeasibility"],
        "degenerate": cell["degenerate"],
        "wall_s": round(time.time() - started, 2),
        "PASS": bool(
            sum(cell["outcome_counts"].values()) == 4
            and cell["infeasibility"] == 0.0
            and np.isfinite(cell["cps"])
        ),
    }


def check_degenerate(config, mppi_config, scenes) -> dict:
    """(c) the degenerate branch fires and is counted."""
    framework, _, _ = _controller(config, mppi_config, n_samples=4, horizon=5, lam=0.05, c_crash=1e5)
    system = framework.system
    controller = framework.controller
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)
    x = system.wrap_state(initial_states_from_batch(batched))

    # CONSTRUCTED: park every row above the |p_z| = band_collision_limit surface, so every sampled
    # rollout collides on its first step no matter what the controller commands.
    band = float(config["env"]["band_collision_limit"])
    x_band = x.clone()
    x_band[:, 2] = band + 0.5
    # ... and invert the attitude (q = [0,1,0,0] is a 180 deg roll), the Haar-uniform IC case named in
    # the dispatch: thrust points down, so nothing can be recovered inside the horizon either.
    x_band[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=DTYPE)
    controller.reset(x_band.shape[0])
    u_band = controller.act(x_band, batched)
    fired = controller.last_degenerate
    counted = controller.degenerate_steps
    # the APPLIED action is always per-rotor: in wrench space the plan head is allocated through the
    # system's own relation first, and the braking tail is the hover trim wrench, whose rotor image is
    # exactly mass*gravity/4 per rotor.
    hover_rotor = controller.trim_rotor
    constructed = {
        "case": "p_z = band_collision_limit + 0.5 (all N rollouts collide), inverted attitude",
        "plan_space": controller.params.space,
        "rows": int(x_band.shape[0]),
        "degenerate_rows": int(fired.sum()),
        "all_rows_degenerate": bool(fired.all()),
        "counter_after_one_step": counted.tolist(),
        "action_equals_braking_tail": bool(
            torch.allclose(u_band, hover_rotor.expand_as(u_band), atol=1e-12, rtol=0.0)
        ),
        "applied_action_row0": [round(v, 6) for v in u_band[0].tolist()],
        "hover_trim_per_rotor": round(float(hover_rotor[0]), 6),
        "plan_trim": [round(v, 6) for v in controller.hover_trim.tolist()],
    }

    # CONSTRUCTED 2: park every row at the centre of its first active cylinder.
    centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=DTYPE)
    active = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
    first = active.float().argmax(dim=1)
    x_obs = x.clone()
    x_obs[:, :2] = centers[torch.arange(len(scenes)), first]
    controller.reset(x_obs.shape[0])
    controller.act(x_obs, batched)
    inside_obstacle = {
        "case": "p_xy = centre of the first active cylinder (all N rollouts collide)",
        "degenerate_rows": int(controller.last_degenerate.sum()),
        "all_rows_degenerate": bool(controller.last_degenerate.all()),
    }

    # NATURAL: real pool ICs, tiny N and huge C_crash, 20 control steps.
    framework2, _, _ = _controller(config, mppi_config, n_samples=2, horizon=10, lam=0.05, c_crash=1e5)
    x2 = system.wrap_state(initial_states_from_batch(batched))
    for _ in range(20):
        u = framework2.policy(x2, batched)
        x2 = rk4_step(system, x2, u, float(config["env"]["dt"]))
    natural_counts = framework2.controller.degenerate_steps.tolist()
    natural = {
        "case": "real pool ICs, N=2, H=10, C_crash=1e5, 20 control steps",
        "n_scenes": len(scenes),
        "per_episode_degenerate_steps": natural_counts,
        "episodes_with_any": int(sum(1 for c in natural_counts if c > 0)),
        "total_degenerate_steps": int(sum(natural_counts)),
    }
    return {
        "constructed_band": constructed,
        "constructed_obstacle": inside_obstacle,
        "natural": natural,
        "PASS": bool(
            constructed["all_rows_degenerate"]
            and constructed["action_equals_braking_tail"]
            and all(c == 1 for c in constructed["counter_after_one_step"])
            and inside_obstacle["all_rows_degenerate"]
        ),
    }


def check_predicate(config, mppi_config, scenes) -> dict:
    """(d) the cost's collision indicator vs src.common.outcomes.step_outcomes, exact agreement."""
    framework, _, cost = _controller(config, mppi_config, n_samples=4, horizon=5, lam=0.2, c_crash=1e3)
    system = framework.system
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)
    x = system.wrap_state(initial_states_from_batch(batched))
    dt = float(config["env"]["dt"])

    # (i) a real rollout under random in-box controls
    lo = system.u_bounds[:, 0].to(DTYPE)
    hi = system.u_bounds[:, 1].to(DTYPE)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    states = [x]
    for _ in range(30):
        u = lo + (hi - lo) * torch.rand(
            (x.shape[0], system.action_dim), generator=generator, dtype=DTYPE
        )
        x = rk4_step(system, x, u, dt)
        states.append(x)
    rollout_states = torch.stack(states, dim=0)                       # [T,B,13]

    # (ii) synthetic states ON the obstacle and band boundaries, so the check is not vacuous and the
    #      strict-vs-nonstrict conventions are exercised
    band = float(config["env"]["band_collision_limit"])
    centers = torch.as_tensor(np.stack([s.obstacle_centers for s in scenes]), dtype=DTYPE)
    radii = torch.as_tensor(np.stack([s.obstacle_radii for s in scenes]), dtype=DTYPE)
    active = torch.as_tensor(np.stack([s.obstacle_active for s in scenes]), dtype=torch.bool)
    first = active.float().argmax(dim=1)
    rows = torch.arange(len(scenes))
    c0 = centers[rows, first]
    r0 = radii[rows, first]
    template = rollout_states[0].clone()
    probes = []
    for offset in (-0.05, -1e-9, 0.0, 1e-9, 0.05):                    # across the cylinder surface
        probe = template.clone()
        probe[:, 0] = c0[:, 0] + r0 + offset
        probe[:, 1] = c0[:, 1]
        probes.append(probe)
    for z in (-band - 0.05, -band, -band + 1e-9, 0.0, band - 1e-9, band, band + 0.05):
        probe = template.clone()
        probe[:, 2] = z
        probes.append(probe)
    synthetic = torch.stack(probes, dim=0)                            # [P,B,13]

    all_states = torch.cat([rollout_states, synthetic], dim=0)        # [T+P,B,13]
    reference = step_outcomes(all_states, batched, system, config).collided        # [T+P,B]
    mine = collision_mask(
        system.position(all_states).permute(1, 0, 2),                 # [B, T+P, 3]
        *[t for t in (batched.obstacle_centers, batched.obstacle_radii, batched.obstacle_active)],
        cost,
    ).permute(1, 0)                                                   # [T+P,B]
    agree = (mine == reference)
    return {
        "n_states_compared": int(agree.numel()),
        "n_rollout_steps": int(rollout_states.shape[0]), "n_boundary_probes": int(synthetic.shape[0]),
        "n_scenes": len(scenes),
        "reference_positive_rate": float(reference.double().mean()),
        "mine_positive_rate": float(mine.double().mean()),
        "agreement_rate": float(agree.double().mean()),
        "n_disagreements": int((~agree).sum()),
        "band_collision_limit": band,
        "PASS": bool(agree.all() and reference.any() and (~reference).any()),
    }


def check_chunk_equivalence(mppi_config, pool_path) -> dict:
    """(e) sample-axis chunking of the rollout is exactly equivalent, not an approximation.

    Required before the GPU screen: the largest cell is chunked to stay inside the VRAM cap, and that is
    only legitimate if chunking moves wall time and nothing else. Run at control_hold 1 AND 4, because
    S3's heaviest corner (N = 1024, m = 4) is the one that might need chunking."""
    keys = ("cps", "reach", "collision", "oob", "stuck", "timeout", "saturation_rate",
            "coll_obstacle", "coll_band_lower", "coll_band_upper")
    legs = {}
    for hold in (1, 4):
        common = dict(
            pool_path=pool_path, n_scenes=8, ebs=8, n_samples=32, horizon=10, lam=0.2, c_crash=1e3,
            sigma=None, seed=42, out_dir=Path("/dev/null"), device=DEVICE, dtype=DTYPE,
            max_steps=80, mppi_config=mppi_config, write=False, control_hold=hold,
        )
        whole = run_cell(label="chunk_off", sample_chunk=0, **common)
        chunked = run_cell(label="chunk_on", sample_chunk=8, **common)
        identical = {k: (whole[k] == chunked[k]) for k in keys}
        identical["outcome_counts"] = whole["outcome_counts"] == chunked["outcome_counts"]
        identical["degenerate"] = whole["degenerate"] == chunked["degenerate"]
        legs[f"m{hold}"] = {
            "case": f"n_scenes 8, N 32, H 10, control_hold {hold}, 80 control steps; "
                    "sample_chunk 0 (whole) vs 8 (4 chunks)",
            "outcome_counts": whole["outcome_counts"],
            "degenerate": whole["degenerate"],
            "metrics_whole": {k: whole[k] for k in keys},
            "metrics_chunked": {k: chunked[k] for k in keys},
            "identical": identical,
            "PASS": bool(all(identical.values())),
        }
    legs["PASS"] = bool(legs["m1"]["PASS"] and legs["m4"]["PASS"])
    return legs


def check_allocation_roundtrip(config, mppi_config) -> dict:
    """(f) the allocation round-trip, R1.

    The mixer is invertible, so for any wrench whose rotor image lies STRICTLY inside the per-rotor box
    (no clipping) `mixer_inv` then `mixer` must return the wrench exactly. Sampling is done in the rotor
    box and mapped forward, which is precisely the set on which the identity is claimed. The second leg
    pins the trim: the hover trim wrench (m*g, 0, 0, 0) must allocate to mass*gravity/4 per rotor."""
    framework, params, _ = _controller(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3
    )
    controller = framework.controller
    lo = controller.u_lo
    hi = controller.u_hi
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    # strictly inside the box: a 2% inset on both ends, so no sample can touch a face
    inset = 0.02 * (hi - lo)
    rotor = (lo + inset) + ((hi - inset) - (lo + inset)) * torch.rand(
        (20000, controller.action_dim), generator=generator, dtype=DTYPE
    )
    wrench = controller.project(rotor)                       # rotor -> wrench  (system.mixer)
    back_rotor = controller.allocate(wrench)                 # wrench -> rotor  (system.mixer_inv + clip)
    back_wrench = controller.project(back_rotor)
    wrench_err = float((back_wrench - wrench).abs().max())
    rotor_err = float((back_rotor - rotor).abs().max())
    clipped = int(((back_rotor <= lo).any(dim=1) | (back_rotor >= hi).any(dim=1)).sum())

    trim_rotor = controller.allocate(controller.trim_wrench)
    expected = controller.mass * controller.gravity / controller.action_dim
    trim_err = float((trim_rotor - expected).abs().max())
    return {
        "n_wrenches": int(rotor.shape[0]),
        "construction": "rotor forces drawn strictly inside system.u_bounds (2% inset), mapped to "
                        "wrench by system.mixer; the round-trip is mixer_inv then mixer",
        "max_wrench_roundtrip_error": wrench_err,
        "max_rotor_roundtrip_error": rotor_err,
        "n_samples_that_hit_a_box_face": clipped,
        "trim_wrench": [float(v) for v in controller.trim_wrench.tolist()],
        "trim_rotor_from_allocation": [float(v) for v in trim_rotor.tolist()],
        "expected_trim_per_rotor_mass_gravity_over_4": expected,
        "max_trim_error": trim_err,
        "tolerance": 1e-10,
        "PASS": bool(wrench_err <= 1e-10 and rotor_err <= 1e-10 and trim_err <= 1e-10 and clipped == 0),
    }


def check_ou_statistics(config, mppi_config) -> dict:
    """(g) the OU perturbation is stationary with the declared marginals and the declared pole.

    Marginal std sigma_j at EVERY horizon index k (the stationary parameterisation, not a ramp), and
    lag-1 autocorrelation alpha. >= 200k samples per (k, channel)."""
    horizon = 8
    n_samples = 320
    batch = 80                                                # 80 * 320 = 25600 sequences
    framework, params, _ = _controller(
        config, mppi_config, n_samples=n_samples, horizon=horizon, lam=0.2, c_crash=1e3
    )
    controller = framework.controller
    eps = controller._draw_noise(batch)                       # [B, N, H, 4]
    flat = eps.reshape(-1, horizon, controller.action_dim)    # [S, H, 4]
    n_per_cell = int(flat.shape[0])

    sigma_j = controller.sigma_channel                        # [4]
    std_k = flat.std(dim=0, unbiased=True)                    # [H, 4] empirical marginal std at each k
    std_rel_err = (std_k / sigma_j - 1.0).abs()
    mean_k = flat.mean(dim=0)                                 # [H, 4] should be ~0

    # lag-1 autocorrelation, pooled over k: corr(eps_k, eps_{k+1}) per channel
    a = flat[:, :-1, :].reshape(-1, controller.action_dim)
    b = flat[:, 1:, :].reshape(-1, controller.action_dim)
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    corr = (a_c * b_c).mean(dim=0) / (a_c.std(dim=0, unbiased=False) * b_c.std(dim=0, unbiased=False))
    alpha = float(params.ou_alpha)
    corr_rel_err = (corr / alpha - 1.0).abs()
    return {
        "space": params.space, "noise": params.noise,
        "n_sequences": n_per_cell, "horizon": horizon,
        "n_samples_per_(k,channel)": n_per_cell,
        "sigma_per_rotor_equivalent_N": float(params.sigma),
        "channel_scale": [float(v) for v in controller.channel_scale.tolist()],
        "sigma_per_channel": [float(v) for v in sigma_j.tolist()],
        "empirical_std_by_k": [[float(v) for v in row] for row in std_k.tolist()],
        "max_std_relative_error_over_all_k": float(std_rel_err.max()),
        "max_abs_mean_over_all_k": float(mean_k.abs().max()),
        "alpha_declared": alpha,
        "correlation_steps": float(params.ou_correlation_steps),
        "empirical_lag1_autocorr_per_channel": [float(v) for v in corr.tolist()],
        "max_lag1_relative_error": float(corr_rel_err.max()),
        "tolerance_relative": 0.02,
        "PASS": bool(std_rel_err.max() <= 0.02 and corr_rel_err.max() <= 0.02),
    }


def check_ess_relative_lambda(config, mppi_config) -> dict:
    """(h) the relative-lambda weighting cannot collapse to ESS ~= 1 by a scale mismatch.

    Synthetic sample costs S ~ |N(0,1)| * scale, for scales spanning 1e-3 .. 1e9. Under lam_rel = 0.2 the
    exponent (S - min S) / (lam_rel * std S) is scale-INVARIANT, so the ESS is the same at every scale and
    sits far above 1. The legacy absolute lambda (lam = 0.2, as the superseded screen ran) is shown beside
    it and collapses to ~1 as soon as the cost scale exceeds lambda — which is what the first screen
    measured (median ESS 1.0001 against a cost std of ~267)."""
    n = 256
    lam_rel = 0.2
    lam_abs = 0.2
    generator = torch.Generator(device=DEVICE).manual_seed(23)
    base = torch.randn((4096, n), generator=generator, dtype=DTYPE).abs()
    rows = []
    for scale in (1e-3, 1e0, 1e3, 1e6, 1e9):
        cost = base * scale
        shifted = cost - cost.min(dim=1, keepdim=True).values
        lam_eff = torch.clamp(lam_rel * cost.std(dim=1, keepdim=True), min=1e-9)
        w = torch.exp(-shifted / lam_eff)
        w = w / w.sum(dim=1, keepdim=True)
        ess_rel = 1.0 / w.square().sum(dim=1)
        w_abs = torch.exp(-shifted / lam_abs)
        w_abs = w_abs / w_abs.sum(dim=1, keepdim=True)
        ess_abs = 1.0 / w_abs.square().sum(dim=1)
        rows.append({
            "cost_scale": scale,
            "cost_std_median": float(cost.std(dim=1).median()),
            "lam_eff_median_relative": float(lam_eff.squeeze(1).median()),
            "ESS_relative_mean": float(ess_rel.mean()),
            "ESS_relative_median": float(ess_rel.median()),
            "ESS_relative_min": float(ess_rel.min()),
            "ESS_absolute_lambda_0p2_median": float(ess_abs.median()),
        })
    ess_medians = [r["ESS_relative_median"] for r in rows]
    ess_mins = [r["ESS_relative_min"] for r in rows]
    spread = (max(ess_medians) - min(ess_medians)) / max(ess_medians)
    return {
        "N": n, "lam_rel": lam_rel, "legacy_lam_absolute": lam_abs,
        "synthetic_costs": "S = |N(0,1)| * scale, 4096 independent rows of N=256 samples",
        "rows": rows,
        "ESS_relative_median_across_scales": ess_medians,
        "ESS_relative_min_across_all_scales": float(min(ess_mins)),
        "scale_invariance_relative_spread_of_medians": float(spread),
        "note": "the relative-lambda exponent (S - min S)/(lam_rel * std S) is invariant to any rescaling "
                "of S, so ESS is identical at every scale; the absolute-lambda column collapses to ~1.",
        "PASS": bool(min(ess_mins) > 2.0 and spread < 1e-6),
    }


def check_hover_center(config, mppi_config) -> dict:
    """(i) S3 CHANGE 1 — hover-centered samples AVERAGE TO HOVER THRUST.

    With the plan at its reset value the commanded control is `u_hover + 0 + eps`, so the sample mean of
    the commanded wrench must be the trim wrench (m*g, 0, 0, 0) and the sample mean of its allocation
    must be `mass*gravity/4` per rotor, up to sampling error.

    THE TOLERANCE IS DERIVED, NOT TYPED. Along the horizon the perturbation is a stationary AR(1) with
    pole alpha, so the variance of the mean of H consecutive draws is
    `sigma_j^2 / H * (1 + 2 * sum_{j=1..H-1} (1 - j/H) alpha^j)`, and the S = B*N sequences are
    independent; the standard error is the square root of that over S. The check is |deviation| <= 5 SE
    per channel. The POST-CLIP mean is reported beside it as a measurement, not folded into the claim:
    the per-rotor box clip is a separate, real effect and at sigma = 1 N it biases the realised thrust.
    """
    horizon, n_samples, batch = 16, 512, 64                      # 32768 sequences, 524288 draws/channel
    framework, params, _ = _controller(
        config, mppi_config, n_samples=n_samples, horizon=horizon, lam=0.2, c_crash=1e3, center="hover",
    )
    controller = framework.controller
    controller.reset(batch)
    base = controller.absolute(controller.plan)                  # [B,H,4] = anchor + 0 deviation
    noise = controller._draw_noise(batch)                        # [B,N,H,4]
    wrench = base.unsqueeze(1) + noise                           # [B,N,H,4] commanded wrench
    flat = wrench.reshape(-1, controller.action_dim)
    mean_wrench = flat.mean(dim=0)
    trim = controller.trim_wrench

    alpha = float(params.ou_alpha)
    lags = torch.arange(1, horizon, dtype=DTYPE)
    factor = float(1.0 + 2.0 * ((1.0 - lags / horizon) * alpha**lags).sum())
    n_sequences = batch * n_samples
    se = controller.sigma_channel * math.sqrt(factor / (n_sequences * horizon))     # [4]
    deviation = (mean_wrench - trim).abs()
    k_se = 5.0
    tolerance = k_se * se

    rotor_preclip = flat @ controller.mixer_inv.t()
    rotor_postclip = controller.allocate(flat)
    expected_rotor = controller.mass * controller.gravity / controller.action_dim
    clipped = ((rotor_postclip <= controller.u_lo) | (rotor_postclip >= controller.u_hi))
    # the plan at reset is the ZERO deviation, i.e. the centre is the anchor and nothing else
    return {
        "N": n_samples, "batch": batch, "horizon": horizon, "n_sequences": n_sequences,
        "n_draws_per_channel": int(flat.shape[0]),
        "center": params.center,
        "plan_at_reset_is_zero_deviation": bool((controller.plan == 0).all()),
        "anchor_wrench": [float(v) for v in controller.anchor.tolist()],
        "trim_wrench_from_system": [float(v) for v in trim.tolist()],
        "mean_commanded_wrench": [float(v) for v in mean_wrench.tolist()],
        "abs_deviation_per_channel": [float(v) for v in deviation.tolist()],
        "standard_error_per_channel": [float(v) for v in se.tolist()],
        "tolerance_per_channel_5se": [float(v) for v in tolerance.tolist()],
        "tolerance_rule": f"|mean - trim| <= {k_se} * SE, SE from the stationary AR(1) mean variance "
                          f"(pole alpha = {alpha}, autocorrelation factor = {factor})",
        "deviation_in_se_units": [float(v) for v in (deviation / se).tolist()],
        "hover_per_rotor_from_system_mass_gravity": float(expected_rotor),
        "mean_preclip_rotor_allocation": [float(v) for v in rotor_preclip.mean(dim=0).tolist()],
        "max_abs_preclip_rotor_deviation": float(
            (rotor_preclip.mean(dim=0) - expected_rotor).abs().max()
        ),
        "measured_not_asserted__mean_postclip_rotor": [
            float(v) for v in rotor_postclip.mean(dim=0).tolist()
        ],
        "measured_not_asserted__frac_rotor_entries_clipped": float(clipped.to(DTYPE).mean()),
        "measured_not_asserted__note": "the per-rotor box clip is applied AFTER allocation and is a "
                                       "separate effect from the sampling centre; it is reported, not "
                                       "folded into the centering claim",
        "PASS": bool((deviation <= tolerance).all()),
    }


def check_settling_terminal(config, mppi_config) -> dict:
    """(j) S3 CHANGE 2 — the settling terminal cost is ZERO EXACTLY on the deployed terminal predicate.

    A PROPERTY TEST over a constructed grid of states, not a single case. The three radii are read from
    the effective deployed config; the predicate is evaluated through the SAME norms the cost uses, which
    is what makes the boundary case meaningful (a state whose computed distance is one ulp above the
    radius fails the deployed predicate too, so the two agree by construction rather than by luck).
    """
    framework, _, cost = _controller(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3, terminal="settling",
    )
    _, _, cost_legacy = _controller(
        config, mppi_config, n_samples=4, horizon=3, lam=0.2, c_crash=1e3, terminal="distance",
    )
    system = framework.system
    r_p, r_v, r_w = cost.goal_radius, cost.goal_speed_radius, cost.goal_angrate_radius

    def _states(distances, speeds, rates) -> torch.Tensor:
        grid = torch.cartesian_prod(
            torch.as_tensor(distances, dtype=DTYPE),
            torch.as_tensor(speeds, dtype=DTYPE),
            torch.as_tensor(rates, dtype=DTYPE),
        )
        x = torch.zeros((grid.shape[0], system.state_dim), dtype=DTYPE)
        x[:, 0] = grid[:, 0]            # p = (d, 0, 0), goal at the origin -> ||p - g|| = d
        x[:, 3] = 1.0                   # q = (1, 0, 0, 0), the identity attitude
        x[:, 7] = grid[:, 1]            # v = (s, 0, 0) -> ||v|| = s
        x[:, 10] = grid[:, 2]           # omega = (w, 0, 0) -> ||omega|| = w
        return x

    # a grid that straddles every radius, boundary-adjacent points included
    scales = [0.0, 0.5, 1.0 - 1e-9, 1.0, 1.0 + 1e-9, 1.5, 3.0]
    x = _states([s * r_p for s in scales], [s * r_v for s in scales], [s * r_w for s in scales])
    goal = torch.zeros((x.shape[0], 3), dtype=DTYPE)
    phi = terminal_cost(system, x, goal, cost)
    phi_legacy = terminal_cost(system, x, goal, cost_legacy)

    distance = torch.linalg.norm(system.position(x) - goal, dim=-1)
    speed = system.speed(x)
    rate = system.angular_rate(x)
    predicate = (distance <= r_p) & (speed <= r_v) & (rate <= r_w)      # step_outcomes' conjunction

    # each condition violated ALONE (the other two strictly satisfied)
    singles = {
        "position_only": _states([1.5 * r_p], [0.5 * r_v], [0.5 * r_w]),
        "speed_only": _states([0.5 * r_p], [1.5 * r_v], [0.5 * r_w]),
        "angrate_only": _states([0.5 * r_p], [0.5 * r_v], [1.5 * r_w]),
    }
    single_phi = {
        name: float(terminal_cost(system, s, torch.zeros((s.shape[0], 3), dtype=DTYPE), cost).min())
        for name, s in singles.items()
    }
    zero_on_predicate = bool((phi[predicate] == 0.0).all())
    positive_off_predicate = bool((phi[~predicate] > 0.0).all())
    return {
        "terminal_mode": cost.terminal_mode,
        "radii_read_from_effective_config": {
            "goal_radius": r_p, "goal_speed_radius": r_v, "goal_angrate_radius": r_w,
        },
        "radii_source": "config['env'] of the effective merged config (base + exp + mppi.eval_cell); "
                        "no radius is typed in the cost, the controller or this check",
        "n_states": int(x.shape[0]),
        "scale_grid_applied_to_each_radius": scales,
        "n_satisfying_predicate": int(predicate.sum()),
        "n_violating_predicate": int((~predicate).sum()),
        "max_phi_on_predicate": float(phi[predicate].max()),
        "min_phi_off_predicate": float(phi[~predicate].min()),
        "zero_exactly_on_predicate": zero_on_predicate,
        "strictly_positive_off_predicate": positive_off_predicate,
        "single_condition_violations_min_phi": single_phi,
        "all_single_violations_positive": bool(all(v > 0.0 for v in single_phi.values())),
        "legacy_distance_terminal_on_the_same_predicate_states": {
            "min": float(phi_legacy[predicate].min()), "max": float(phi_legacy[predicate].max()),
            "n_nonzero": int((phi_legacy[predicate] > 0).sum()),
            "note": "the legacy terminal is positive on states the deployed predicate accepts — this is "
                    "the defect S3 change 2 removes, and it is why the position leg must be relu'd",
        },
        "PASS": bool(
            zero_on_predicate
            and positive_off_predicate
            and all(v > 0.0 for v in single_phi.values())
            and int(predicate.sum()) > 0 and int((~predicate).sum()) > 0
        ),
    }


def check_control_hold(config, mppi_config, scenes) -> dict:
    """(k) S3 CHANGE 3 — block length exactly m in the rollout, and the closed-loop hold schedule.

    Three legs: the ROLLOUT expansion (`held_control`) is piecewise-constant with block length exactly m
    and spans H*m physical steps; the CLOSED LOOP re-plans only at decision steps, re-applies the latched
    action between them and shifts the plan only at decisions; and m = 1 is the unheld controller.
    """
    horizon = 6
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)

    legs = {}
    for hold in (1, 4):
        framework, params, _ = _controller(
            config, mppi_config, n_samples=4, horizon=horizon, lam=0.2, c_crash=1e3, control_hold=hold,
        )
        controller = framework.controller
        system = framework.system
        # --- leg 1: the rollout expansion --------------------------------------------------------
        generator = torch.Generator(device=DEVICE).manual_seed(5)
        sampled = torch.randn((2, 3, horizon, controller.action_dim), generator=generator, dtype=DTYPE)
        expanded = torch.stack(
            [controller.held_control(sampled, k) for k in range(horizon * hold)], dim=2
        )                                                              # [2,3,H*hold,4]
        block_constant = all(
            bool(torch.equal(expanded[:, :, j * hold + i], sampled[:, :, j]))
            for j in range(horizon) for i in range(hold)
        )
        boundaries_change = all(
            not bool(torch.equal(sampled[:, :, j], sampled[:, :, j + 1])) for j in range(horizon - 1)
        )
        # --- leg 2: the closed-loop schedule ------------------------------------------------------
        x = system.wrap_state(initial_states_from_batch(batched))
        n_steps = 3 * hold + 1
        actions, decisions, plan_changed = [], [], []
        previous_plan = None
        for _ in range(n_steps):
            u = framework.policy(x, batched)
            actions.append(u.clone())
            decisions.append(bool(controller.last_decision))
            plan_changed.append(
                previous_plan is None or not bool(torch.equal(controller.plan, previous_plan))
            )
            previous_plan = controller.plan.clone()
            x = rk4_step(system, x, u, float(config["env"]["dt"]))
        stacked = torch.stack(actions, dim=0)                          # [T,B,4]
        expected_decisions = [t % hold == 0 for t in range(n_steps)]
        held_equal = all(
            bool(torch.equal(stacked[t], stacked[t - (t % hold)])) for t in range(n_steps)
        )
        legs[f"m{hold}"] = {
            "control_hold": hold,
            "horizon_decision_entries": horizon,
            "rollout_physical_steps": controller.lookahead_steps,
            "rollout_physical_steps_equals_H_times_m": controller.lookahead_steps == horizon * hold,
            "effective_lookahead_s": controller.lookahead_s,
            "dt": controller.dt,
            "rollout_block_constant_with_block_length_m": block_constant,
            "distinct_entries_so_the_test_is_not_vacuous": boundaries_change,
            "closed_loop_decision_flags": decisions,
            "closed_loop_expected_decision_flags": expected_decisions,
            "decision_schedule_correct": decisions == expected_decisions,
            "n_decisions": int(sum(decisions)),
            "applied_action_constant_within_each_block": held_equal,
            "plan_changed_per_step": plan_changed,
            "plan_changes_only_at_decisions": all(
                changed == decision for changed, decision in zip(plan_changed, decisions)
            ),
            "PASS": bool(
                block_constant and boundaries_change
                and controller.lookahead_steps == horizon * hold
                and decisions == expected_decisions and held_equal
                and all(c == d for c, d in zip(plan_changed, decisions))
            ),
        }
    legs["m1_is_the_unheld_controller"] = bool(
        all(legs["m1"]["closed_loop_decision_flags"])
        and legs["m1"]["rollout_physical_steps"] == horizon
    )
    legs["PASS"] = bool(
        legs["m1"]["PASS"] and legs["m4"]["PASS"] and legs["m1_is_the_unheld_controller"]
    )
    return legs


def check_center_equivalence(config, mppi_config, pool_path, scenes) -> dict:
    """(l) RESOLUTION (G) — is hover centering behaviourally identical to the uncentered path?

    NOT asserted: measured. In exact arithmetic the two parameterisations agree on every branch (see the
    mppi_controller docstring), so any difference is floating-point associativity — `(anchor + plan) +
    eps` against `plan_abs + eps`. Two legs:

      * SAME-STATE leg — both controllers are driven from the SAME state sequence with the same seed, so
        their noise draws are identical and the comparison isolates the parameterisation from trajectory
        divergence. Reported: the max absolute action difference, in float64 and in float32.
      * CLOSED-LOOP leg — both are run independently through the shared eval path and the resolved
        metrics are compared. Divergence here, if any, is the amplification of the above through the
        plant, not a different algorithm.
    """
    dt = float(config["env"]["dt"])
    batched = batch_scenes(scenes, device=DEVICE, dtype=DTYPE)
    same_state = {}
    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        batched_dt = batch_scenes(scenes, device=DEVICE, dtype=dtype)
        pair = {
            centre: _controller(
                config, mppi_config, n_samples=16, horizon=5, lam=0.2, c_crash=1e3,
                center=centre, dtype=dtype,
            )[0]
            for centre in ("hover", "none")
        }
        system = pair["hover"].system
        x = system.wrap_state(initial_states_from_batch(batched_dt)).to(dtype)
        worst_action = 0.0
        worst_plan = 0.0
        degenerate_agree = True
        for _ in range(12):
            u_hover = pair["hover"].policy(x, batched_dt)
            u_none = pair["none"].policy(x, batched_dt)
            worst_action = max(worst_action, float((u_hover - u_none).abs().max()))
            absolute_hover = pair["hover"].controller.absolute(pair["hover"].controller.plan)
            absolute_none = pair["none"].controller.absolute(pair["none"].controller.plan)
            worst_plan = max(worst_plan, float((absolute_hover - absolute_none).abs().max()))
            degenerate_agree = degenerate_agree and bool(
                torch.equal(
                    pair["hover"].controller.last_degenerate,
                    pair["none"].controller.last_degenerate,
                )
            )
            x = rk4_step(system, x, u_hover, dt)          # both driven from the SAME state sequence
        same_state[name] = {
            "n_control_steps": 12, "n_scenes": len(scenes),
            "max_abs_action_difference": worst_action,
            "max_abs_absolute_plan_difference": worst_plan,
            "degenerate_masks_identical_every_step": degenerate_agree,
            "bit_identical": bool(worst_action == 0.0 and worst_plan == 0.0),
        }

    keys = ("cps", "reach", "collision", "oob", "stuck", "timeout", "saturation_rate")
    closed_loop = {}
    # BOTH precisions: the GPU cells run float32, where the associativity difference is ~1e-6 per action
    # and could in principle flip a discrete outcome over 200 steps. Whether it does is measured here.
    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        common = dict(
            pool_path=pool_path, n_scenes=8, ebs=8, n_samples=32, horizon=8, lam=0.2, c_crash=1e3,
            sigma=None, seed=42, out_dir=Path("/dev/null"), device=DEVICE, dtype=dtype,
            max_steps=60, mppi_config=mppi_config, write=False,
        )
        hover = run_cell(label="center_hover", center="hover", **common)
        none = run_cell(label="center_none", center="none", **common)
        identical = {k: (hover[k] == none[k]) for k in keys}
        identical["outcome_counts"] = hover["outcome_counts"] == none["outcome_counts"]
        identical["degenerate"] = hover["degenerate"] == none["degenerate"]
        closed_loop[name] = {
            "case": f"n_scenes 8, N 32, H 8, 60 control steps, {name}",
            "metrics_hover": {k: hover[k] for k in keys},
            "metrics_none": {k: none[k] for k in keys},
            "identical": identical,
            "all_metrics_identical": bool(all(identical.values())),
        }
    return {
        "question": "(G) is the explicit hover anchor behaviourally identical to the existing path?",
        "algebra": "identical on every branch — sampling, the weighted update, the degenerate hold and "
                   "the shift tail; the only possible difference is floating-point associativity",
        "same_state_leg": same_state,
        "closed_loop_leg": closed_loop,
        # This check RECORDS the finding; it does not require bit-identity, because float32 associativity
        # is a legitimate difference and pretending otherwise would be the assertion (G) forbids.
        "PASS": True,
    }


def main() -> int:
    torch.set_num_threads(4)
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    pool_path = _pool_path(mppi_config)
    pool = load_pool(pool_path)
    scenes = pool.scenes[:4]

    started = time.time()
    results = {
        "what": "v2.8.4 MPPI baseline — CPU-only correctness checks (tiny settings). NO GPU work. "
                "Checks (a)-(e) are the original checks and (f)-(h) the R1-R3 amendment's; all eight "
                "must still pass under the S3 defaults. (i)-(l) are S3's own: hover-centered sampling, "
                "the settling terminal cost, the control hold, and the (G) equivalence measurement.",
        "screen": "S3 — data/runs/v2.8.4/mppi_screen_v3/ (the dispatch's 'v2'; NOT the mppi_screen_v2 "
                  "directory, which holds S2). S1 and S2 smoke records are retained untouched.",
        "device": str(DEVICE), "dtype": str(DTYPE),
        "cuda_visible_to_process": torch.cuda.is_available(),
        "pool": str(pool_path.relative_to(REPO)), "pool_scenes_used": len(scenes),
        "sampler_defaults": {
            k: mppi_config["sampling"][k]
            for k in ("space", "noise", "lam_mode", "center", "control_hold", "sigma",
                      "channel_scale", "trim", "ou", "lam_eps_abs")
        },
        "cost_defaults": {k: mppi_config["cost"][k] for k in ("w_terminal", "terminal")},
        "eval_cell": mppi_config["eval_cell"],
        "a_action_box": check_box(config, mppi_config, scenes),
        "b_short_rollout": check_rollout(mppi_config, pool_path),
        "c_degenerate_branch": check_degenerate(config, mppi_config, scenes),
        "d_collision_predicate_vs_step_outcomes": check_predicate(config, mppi_config, scenes),
        "e_chunk_equivalence": check_chunk_equivalence(mppi_config, pool_path),
        "f_allocation_roundtrip": check_allocation_roundtrip(config, mppi_config),
        "g_ou_statistics": check_ou_statistics(config, mppi_config),
        "h_ess_relative_lambda": check_ess_relative_lambda(config, mppi_config),
        "i_hover_center": check_hover_center(config, mppi_config),
        "j_settling_terminal": check_settling_terminal(config, mppi_config),
        "k_control_hold": check_control_hold(config, mppi_config, scenes),
        "l_center_equivalence": check_center_equivalence(config, mppi_config, pool_path, scenes),
    }
    results["ALL_PASS"] = all(
        results[key]["PASS"]
        for key in (
            "a_action_box", "b_short_rollout", "c_degenerate_branch",
            "d_collision_predicate_vs_step_outcomes", "e_chunk_equivalence",
            "f_allocation_roundtrip", "g_ou_statistics", "h_ess_relative_lambda",
            "i_hover_center", "j_settling_terminal", "k_control_hold", "l_center_equivalence",
        )
    )
    results["verdict"] = "ALL_PASS" if results["ALL_PASS"] else "ALL_FAIL"
    results["wall_s"] = round(time.time() - started, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
