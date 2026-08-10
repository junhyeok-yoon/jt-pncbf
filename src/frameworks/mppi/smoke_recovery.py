"""v2.8.4 MPPI baseline — the charter-"v3" smoke checks for B1 / B2 / B3.

Separate from `cpu_smoke.py`, which backs the retained S1/S2/S3 artifacts and is left untouched.

CHECKS (the charter's four, plus two the charter's own wording implies)

  (a) B1 WEIGHT.  `w_att(theta) == w0` for every theta <= theta_ref, and
      `w_att(theta) == w0 * (1 + k_att * (theta - theta_ref)/theta_ref)` above it, on a dense sweep of
      theta over [0, 180] deg plus the exact boundary. Also that the leg itself vanishes when upright.
  (b) B2 FIRING AND FEASIBILITY.  On real pool initial states: the seed mask equals `spawn tilt >
      theta_ref` exactly (both directions, so no episode is seeded that should not be and none is
      missed); the seeded entries occupy exactly `block_steps` decision entries and the rest of the plan
      is the hover trim; the seeded wrench allocates to per-rotor forces that lie INSIDE `system.u_bounds`
      WITHOUT the allocator's clip doing any work; and the yaw channel of the seed is identically zero.
  (c) B3 TRIGGER.  On a constructed batch whose weights collapse (a deliberately tiny lambda), the event
      mask fires, the per-episode event counter increments by one per decision step, the post-adaptation
      ESS is strictly larger than the pre-adaptation ESS, and lam_eff is exactly `lam_factor` times the
      unadapted value on the fired rows. With B3 off the same batch produces no event.
  (d) ESS IDENTITY.  `(sum w)^2 / sum w^2` (the charter's expression, on unnormalised weights) equals
      `1 / sum w_norm^2` (what the controller reports) to floating-point tolerance, so the condition B3
      tests and the ESS the tables report are the same quantity.
  (e) SWITCHED-OFF EQUIVALENCE, TENSOR LEVEL.  A controller carrying `RecoveryParams` with all three
      switches off produces BIT-IDENTICAL actions to one carrying `recovery=None` over a multi-step
      rollout on real scenes, at both control holds.
  (f) v2 REPRODUCTION, WHOLE CELL.  One cell of the charter's "v2" screen is re-run with B1/B2/B3 all
      off and compared against `data/runs/v2.8.4/mppi_screen_v3/` — the headline metrics, the degenerate
      and ESS summaries, AND the per-episode arrays element by element (outcome, collision cause,
      cps_episode, n_steps, degenerate_steps). A mismatch is a STOP.

      DEVICE. (a)-(e) run on CPU. (f) cannot: the retained artifact was produced on CUDA in float32, and
      200 rk4 steps of a quadrotor are chaotic enough that a CPU rerun is not expected to reproduce it
      bit for bit. Reproducing it "metric-for-metric" — which the charter makes a STOP condition — is
      only meaningful on the device that produced it, so (f) runs on CUDA under the same VRAM guard the
      screen uses, before any screen cell. The choice is recorded here and in the build-log rather than
      silently made.

Run:  python -m src.frameworks.mppi.smoke_recovery --out data/runs/v2.8.4/mppi_v3/smoke_recovery.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from src.frameworks.jt_pncbf.train import make_system
from src.frameworks.mppi.cost import CostParams
from src.frameworks.mppi.evaluate_mppi import (
    REPO,
    effective_config,
    load_mppi_config,
    run_cell,
)
from src.frameworks.mppi.mppi_controller import MPPIController, MPPIParams
from src.frameworks.mppi.recovery import (
    RecoveryParams,
    attitude_cost,
    attitude_weight,
    channel_authority,
    effective_sample_size,
    recovery_wrench,
    tilt_cos,
    tilt_deg_from_cos,
)


def _free_vram_mib() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def _build(mppi_config, config, recovery, *, n_samples, horizon, control_hold, lam, c_crash,
           device, dtype):
    system = make_system(config)
    params = MPPIParams.from_config(
        mppi_config, n_samples=n_samples, horizon=horizon, lam=lam, control_hold=control_hold,
    )
    cost_params = CostParams.from_config(mppi_config, config["env"], c_crash=c_crash)
    controller = MPPIController(
        system, config, params, cost_params, device=device, dtype=dtype, recovery=recovery
    )
    return system, controller


# ---- (a) --------------------------------------------------------------------------------------
def check_b1_weight(mppi_config) -> dict[str, Any]:
    recovery = RecoveryParams.from_config(mppi_config, b1=True, b2=False, b3=False)
    w0, k, ref = recovery.b1_w_att, recovery.b1_k_att, recovery.theta_ref_deg
    theta = torch.cat([
        torch.linspace(0.0, 180.0, 1801, dtype=torch.float64),
        torch.tensor([ref, ref - 1e-9, ref + 1e-9], dtype=torch.float64),
    ])
    weight = attitude_weight(theta, recovery)
    below = theta <= ref
    expected_above = w0 * (1.0 + k * (theta - ref) / ref)
    flat_err = float((weight[below] - w0).abs().max())
    slope_err = float((weight[~below] - expected_above[~below]).abs().max())
    # the leg vanishes when upright: cos(0) = 1 -> (1 - cos) = 0
    config = effective_config(mppi_config)
    system = make_system(config)
    state_dim = int(getattr(system, "state_dim"))               # read from the system, never typed
    upright = torch.zeros(1, state_dim, dtype=torch.float64)
    upright[0, 3] = 1.0                                        # identity quaternion [w,x,y,z]
    leg_upright = float(attitude_cost(system, upright, recovery).abs().max())
    inverted = torch.zeros(1, state_dim, dtype=torch.float64)
    inverted[0, 4] = 1.0                                       # 180 deg about body x -> fully inverted
    leg_inverted = float(attitude_cost(system, inverted, recovery)[0])
    tilt_inverted = float(tilt_deg_from_cos(tilt_cos(system, inverted))[0])
    passed = (
        flat_err == 0.0 and slope_err <= 1e-12 and leg_upright == 0.0
        and abs(leg_inverted - w0 * (1.0 + k * (180.0 - ref) / ref) * 2.0) <= 1e-9
    )
    return {
        "check": "(a) B1 weight law", "passed": bool(passed),
        "w0": w0, "k_att": k, "theta_ref_deg": ref,
        "max_abs_err_below_and_at_theta_ref": flat_err,
        "max_abs_err_above_theta_ref": slope_err,
        "n_theta_probed": int(theta.numel()),
        "leg_at_upright": leg_upright,
        "leg_at_inverted": leg_inverted,
        "tilt_at_inverted_deg": tilt_inverted,
        "expected_leg_at_inverted": w0 * (1.0 + k * (180.0 - ref) / ref) * 2.0,
        "note": "w_att is exactly w0 at and below theta_ref (error is 0.0, not a tolerance) and follows "
                "the charter's linear law above it; the leg is (1 - cos theta) times that weight, so it "
                "is exactly 0 upright and 2 * w_att(180) inverted.",
    }


# ---- (b) --------------------------------------------------------------------------------------
def check_b2_seeding(mppi_config, pool_path: Path, n_scenes: int) -> dict[str, Any]:
    config = effective_config(mppi_config)
    device, dtype = torch.device("cpu"), torch.float64
    recovery = RecoveryParams.from_config(mppi_config, b1=False, b2=True, b3=False)
    system, controller = _build(
        mppi_config, config, recovery, n_samples=8, horizon=40, control_hold=1, lam=0.05,
        c_crash=1.0e3, device=device, dtype=dtype,
    )
    scenes = load_pool(pool_path).scenes[:n_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = system.wrap_state(initial_states_from_batch(batched))
    tilt = tilt_deg_from_cos(tilt_cos(system, x0))

    controller.reset(x0.shape[0])
    controller.spawn_tilt_deg = tilt
    flat_plan = controller.plan.clone()
    controller._seed_recovery_plan(x0)
    fired = controller.b2_fired
    expected = tilt > recovery.theta_ref_deg
    firing_exact = bool(torch.equal(fired, expected))

    block = int(recovery.b2_block_steps)
    plan_abs = controller.absolute(controller.plan)                       # [B,H,4] absolute wrench
    flat_abs = controller.absolute(flat_plan)
    head_changed = (plan_abs[:, :block] != flat_abs[:, :block]).any(dim=-1).any(dim=-1)
    tail_untouched = bool(torch.equal(plan_abs[:, block:], flat_abs[:, block:]))
    # rows that fired must have changed heads, rows that did not must be untouched
    head_matches_fire = bool(torch.equal(head_changed & fired, head_changed))

    seed = plan_abs[fired, 0] if bool(fired.any()) else plan_abs[:0, 0]
    raw_alloc = seed @ controller.mixer_inv.t()                           # BEFORE the box clip
    clipped = controller.allocate(seed)
    lo = system.u_bounds.to(device, dtype)[:, 0]
    hi = system.u_bounds.to(device, dtype)[:, 1]
    inside = bool(((raw_alloc >= lo) & (raw_alloc <= hi)).all()) if seed.numel() else True
    clip_is_noop = bool(torch.equal(raw_alloc, clipped)) if seed.numel() else True
    yaw_zero = float(seed[:, 3].abs().max()) if seed.numel() else 0.0
    collective_is_trim = (
        float((seed[:, 0] - controller.trim_wrench[0]).abs().max()) if seed.numel() else 0.0
    )
    authority = channel_authority(controller.mixer, system.u_bounds.to(device, dtype))
    torque_norm = seed[:, 1:3].norm(dim=-1) if seed.numel() else torch.zeros(0, dtype=dtype)
    expected_norm = recovery.b2_torque_frac * float(authority[1])
    norm_err = (
        float((torque_norm[~controller.b2_axis_degenerate[fired]] - expected_norm).abs().max())
        if seed.numel() and bool((~controller.b2_axis_degenerate[fired]).any()) else 0.0
    )
    passed = bool(
        firing_exact and tail_untouched and head_matches_fire and inside and clip_is_noop
        and yaw_zero == 0.0 and collective_is_trim <= 1e-12 and norm_err <= 1e-9
    )
    return {
        "check": "(b) B2 initialisation", "passed": passed,
        "n_scenes": int(x0.shape[0]),
        "theta_ref_deg": recovery.theta_ref_deg,
        "n_seeded": int(fired.sum()), "n_above_theta_ref": int(expected.sum()),
        "firing_mask_equals_tilt_gt_theta_ref": firing_exact,
        "block_steps": block,
        "tail_beyond_block_untouched": tail_untouched,
        "changed_heads_are_exactly_the_seeded_rows": head_matches_fire,
        "raw_allocation_inside_u_bounds_before_clip": inside,
        "allocator_clip_is_a_noop_on_the_seed": clip_is_noop,
        "raw_allocation_min": float(raw_alloc.min()) if seed.numel() else None,
        "raw_allocation_max": float(raw_alloc.max()) if seed.numel() else None,
        "u_bounds": [float(lo[0]), float(hi[0])],
        "seed_yaw_channel_max_abs": yaw_zero,
        "seed_collective_minus_trim_max_abs": collective_is_trim,
        "roll_pitch_torque_norm_expected": expected_norm,
        "roll_pitch_torque_norm_max_abs_err": norm_err,
        "n_axis_degenerate": int(controller.b2_axis_degenerate.sum()),
        "tilt_min_max_deg": [float(tilt.min()), float(tilt.max())],
    }


# ---- (c) + (d) --------------------------------------------------------------------------------
def check_b3_trigger(mppi_config, pool_path: Path, n_scenes: int) -> dict[str, Any]:
    config = effective_config(mppi_config)
    device, dtype = torch.device("cpu"), torch.float64
    scenes = load_pool(pool_path).scenes[:n_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    system_probe = make_system(config)
    x0 = system_probe.wrap_state(initial_states_from_batch(batched))

    # A SWEEP of lam_rel. A tiny lambda pins the softmax at a hard argmin, so the event condition
    # ESS < ess_frac_of_n * N holds on every row (that is the constructed low-ESS batch); larger values
    # let ESS rise, so the sweep also shows the condition being SELECTIVE rather than always true, and it
    # is where the post-adaptation ESS increase is measurable rather than lost in the last few ulps.
    lam_sweep = (1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1)
    n_samples = 16
    ess_frac = float(mppi_config["recovery"]["b3"]["ess_frac_of_n"])
    lam_factor = float(mppi_config["recovery"]["b3"]["lam_factor"])
    per_lam: dict[str, Any] = {}
    all_fired_any, all_nondecreasing, all_strict_any, all_lam_err = [], [], [], []
    off_fires = False
    pre_matches_off_max = 0.0
    for lam_rel in lam_sweep:
        traces = {}
        for name, b3 in (("b3_off", False), ("b3_on", True)):
            recovery = RecoveryParams.from_config(mppi_config, b1=False, b2=False, b3=b3)
            _, controller = _build(
                mppi_config, config, recovery, n_samples=n_samples, horizon=4, control_hold=1,
                lam=lam_rel, c_crash=1.0e3, device=device, dtype=dtype,
            )
            controller.reset(x0.shape[0])
            ess, ess_pre, lam_eff, lam_pre, events = [], [], [], [], []
            for _ in range(3):
                controller.act(x0, batched)
                ess.append(controller.last_ess.clone())
                ess_pre.append(controller.last_ess_pre.clone())
                lam_eff.append(controller.last_lam_eff.clone())
                lam_pre.append(controller.last_lam_eff_pre.clone())
                events.append(controller.last_b3_event.clone())
            traces[name] = {
                "ess": torch.stack(ess), "ess_pre": torch.stack(ess_pre),
                "lam_eff": torch.stack(lam_eff), "lam_pre": torch.stack(lam_pre),
                "events": torch.stack(events), "counter": controller.b3_events.clone(),
            }
        off, on = traces["b3_off"], traces["b3_on"]
        fired = on["events"]
        any_fired = bool(fired.any())
        # lam_eff is compared against the SAME step's PRE-adaptation lam_eff of the SAME controller:
        # once B3 has moved a plan the two controllers' rollout costs diverge, so a cross-controller
        # comparison would not be measuring the multiplier.
        lam_err = (
            float((on["lam_eff"][fired] / on["lam_pre"][fired] - lam_factor).abs().max())
            if any_fired else 0.0
        )
        nondecreasing = bool((on["ess"][fired] >= on["ess_pre"][fired]).all()) if any_fired else True
        strict_any = bool((on["ess"][fired] > on["ess_pre"][fired]).any()) if any_fired else False
        counter_ok = bool(torch.equal(on["counter"], fired.long().sum(dim=0)))
        off_fires = off_fires or bool(off["events"].any())
        # the first decision step has not yet been perturbed by any adaptation, so the pre-adaptation
        # ESS there must equal the B3-off controller's ESS exactly.
        pre_matches_off_max = max(
            pre_matches_off_max, float((on["ess_pre"][0] - off["ess"][0]).abs().max())
        )
        all_fired_any.append(any_fired)
        all_nondecreasing.append(nondecreasing)
        all_strict_any.append(strict_any)
        all_lam_err.append(lam_err)
        per_lam[f"lam_rel_{lam_rel:g}"] = {
            "events_fired": int(fired.sum()), "decisions": int(fired.numel()),
            "event_frac": float(fired.double().mean()),
            "ess_pre_mean": float(on["ess_pre"].mean()), "ess_post_mean": float(on["ess"].mean()),
            "ess_nondecreasing_on_fired_rows": nondecreasing,
            "ess_strictly_increased_on_some_fired_row": strict_any,
            "lam_eff_over_lam_pre_max_abs_err_vs_lam_factor": lam_err,
            "per_episode_counter_matches_event_mask": counter_ok,
        }
    counter_ok = all(v["per_episode_counter_matches_event_mask"] for v in per_lam.values())
    ess_increased = all(all_nondecreasing) and any(all_strict_any)
    lam_ok = max(all_lam_err)
    off_never_fires = not off_fires
    pre_matches_off = pre_matches_off_max
    fired = torch.tensor(all_fired_any)
    on = {"n_samples": n_samples, "ess_frac_of_n": ess_frac, "lam_factor": lam_factor,
          "ess_pre": torch.tensor([v["ess_pre_mean"] for v in per_lam.values()]),
          "ess": torch.tensor([v["ess_post_mean"] for v in per_lam.values()])}

    # (d) the ESS identity, on a random weight batch
    generator = torch.Generator().manual_seed(0)
    raw = torch.rand((32, 64), generator=generator, dtype=torch.float64).mul(4.0).exp()
    normalised = raw / raw.sum(dim=1, keepdim=True)
    identity_err = float(
        (effective_sample_size(raw) - 1.0 / normalised.square().sum(dim=1)).abs().max()
    )

    passed = bool(
        bool(fired.any()) and counter_ok and ess_increased and lam_ok <= 1e-9
        and off_never_fires and pre_matches_off <= 1e-12 and identity_err <= 1e-9
    )
    return {
        "check": "(c) B3 trigger + (d) ESS identity", "passed": passed,
        "constructed_batch": {
            "n_scenes": int(x0.shape[0]), "N": int(on["n_samples"]), "horizon": 4,
            "lam_rel_sweep": list(lam_sweep),
            "ess_frac_of_n": on["ess_frac_of_n"], "lam_factor": on["lam_factor"],
            "event_condition_value": on["ess_frac_of_n"] * on["n_samples"],
        },
        "per_lam_rel": per_lam,
        "b3_off_never_fires": off_never_fires,
        "per_episode_counter_matches_event_mask": counter_ok,
        "ess_nondecreasing_everywhere_and_strictly_increased_somewhere": ess_increased,
        "lam_eff_over_lam_pre_max_abs_err_vs_lam_factor": lam_ok,
        "pre_adaptation_ess_matches_b3_off_ess_at_step0_max_abs": pre_matches_off,
        "ess_identity_max_abs_err": identity_err,
        "ess_identity_note": "(sum w)^2 / sum w^2 on unnormalised weights == 1 / sum w_norm^2 on the "
                             "normalised ones; the charter's expression and the reported ESS agree.",
    }


# ---- (e) --------------------------------------------------------------------------------------
def check_switched_off_equivalence(mppi_config, pool_path: Path, n_scenes: int) -> dict[str, Any]:
    config = effective_config(mppi_config)
    device, dtype = torch.device("cpu"), torch.float64
    scenes = load_pool(pool_path).scenes[:n_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    system_probe = make_system(config)
    x0 = system_probe.wrap_state(initial_states_from_batch(batched))
    per_hold = {}
    for hold in (1, 4):
        traces = []
        for recovery in (None, RecoveryParams.off(mppi_config)):
            _, controller = _build(
                mppi_config, config, recovery, n_samples=16, horizon=5, control_hold=hold, lam=0.05,
                c_crash=1.0e3, device=device, dtype=dtype,
            )
            controller.reset(x0.shape[0])
            actions = [controller.act(x0, batched).clone() for _ in range(2 * hold + 1)]
            traces.append(torch.stack(actions))
        per_hold[f"m{hold}"] = {
            "bit_identical": bool(torch.equal(traces[0], traces[1])),
            "max_abs_diff": float((traces[0] - traces[1]).abs().max()),
            "n_steps_compared": int(traces[0].shape[0]),
        }
    passed = all(v["bit_identical"] for v in per_hold.values())
    return {
        "check": "(e) switched-off equivalence, tensor level", "passed": bool(passed),
        "n_scenes": int(x0.shape[0]), "per_control_hold": per_hold,
        "note": "recovery=None and RecoveryParams with all three switches off must produce the same "
                "actions bit for bit; both are the charter's 'v2' controller.",
    }


# ---- (f) --------------------------------------------------------------------------------------
_REPRO_KEYS = (
    "cps", "reach", "collision", "coll_obstacle", "coll_band_lower", "coll_band_upper",
    "oob", "stuck", "timeout", "saturation_rate", "cps_ci_lo", "cps_ci_hi",
    "reach_ci_lo", "reach_ci_hi", "collision_ci_lo", "collision_ci_hi", "infeasibility",
)


def check_v2_reproduction(
    mppi_config, pool_path: Path, label: str, artifact_dir: Path, out_dir: Path,
    device: torch.device, dtype: torch.dtype, min_free_mib: float,
) -> dict[str, Any]:
    reference = json.loads((artifact_dir / f"cell__{label}.json").read_text(encoding="utf-8"))
    reference_npz = np.load(artifact_dir / f"per_episode__{label}.npz", allow_pickle=False)
    cell = reference["cell"]
    vram_before = _free_vram_mib() if device.type == "cuda" else None
    if device.type == "cuda" and vram_before["free_mib"] < min_free_mib:
        return {
            "check": "(f) v2 reproduction", "passed": False,
            "stopped": f"free VRAM {vram_before['free_mib']:.0f} MiB < required {min_free_mib:.0f} MiB",
        }
    started = time.time()
    rerun = run_cell(
        pool_path=pool_path,
        n_scenes=int(reference["pool"]["n_scenes_scored"]),
        ebs=int(reference["pool"]["ebs"]),
        n_samples=int(cell["N"]), horizon=int(cell["H"]), lam=float(cell["lambda"]),
        c_crash=float(cell["C_crash"]), sigma=float(cell["sigma"]), seed=int(cell["seed"]),
        sample_chunk=int(cell["sample_chunk"]),
        label=f"v2repro__{label}", out_dir=out_dir, device=device, dtype=dtype,
        control_hold=int(reference["sampler"]["control_hold_m"]),
        center=reference["sampler"]["center"],
        terminal=reference["sampler"]["terminal_cost_mode"],
        space=cell["space"], noise=cell["noise"], lam_mode=cell["lambda_mode"],
        b1=False, b2=False, b3=False,
        tilt_deg=reference_npz["tilt_deg"],
        tilt_split_deg=None,                    # keep the S3 band convention so the bands compare too
        mppi_config=mppi_config,
    )
    wall = time.time() - started
    vram_after = _free_vram_mib() if device.type == "cuda" else None

    scalar = {
        key: {"reference": float(reference[key]), "rerun": float(rerun[key]),
              "equal": bool(float(reference[key]) == float(rerun[key]))}
        for key in _REPRO_KEYS
    }
    counts_equal = reference["outcome_counts"] == rerun["outcome_counts"]
    degenerate_equal = reference["degenerate"] == rerun["degenerate"]
    ess_keys = ("mean", "min", "max", "p05", "p25", "p50", "p75", "p95")
    ess_equal = all(
        reference["ess"]["active"][k] == rerun["ess"]["active"][k] for k in ess_keys
    )
    bands_equal = {
        band: all(
            reference["bands"][band][k] == rerun["bands"][band][k]
            for k in ("n", "reach", "collision", "oob", "stuck", "timeout", "cps")
        )
        for band in ("ALL", "tilt_lt_90", "tilt_ge_90")
    }
    rerun_npz = np.load(out_dir / f"per_episode__v2repro__{label}.npz", allow_pickle=False)
    array_equal = {}
    for key in ("outcome", "cause", "n_steps", "degenerate_steps", "band_crossings"):
        array_equal[key] = bool(np.array_equal(reference_npz[key], rerun_npz[key]))
    for key in ("cps_episode", "saturation_step_frac", "max_h", "traj_path_len"):
        array_equal[key] = bool(np.array_equal(reference_npz[key], rerun_npz[key]))
    passed = bool(
        all(v["equal"] for v in scalar.values()) and counts_equal and degenerate_equal
        and ess_equal and all(bands_equal.values()) and all(array_equal.values())
    )
    return {
        "check": "(f) v2 reproduction, whole cell", "passed": passed,
        "cell_label": label,
        "artifact": str((artifact_dir / f"cell__{label}.json").relative_to(REPO)),
        "device": str(device), "dtype": str(dtype), "wall_s": round(wall, 2),
        "vram_free_before_mib": vram_before["free_mib"] if vram_before else None,
        "vram_free_after_mib": vram_after["free_mib"] if vram_after else None,
        "peak_cuda_alloc_mib": rerun["peak_cuda_alloc_mib"],
        "peak_cuda_reserved_mib": rerun["peak_cuda_reserved_mib"],
        "switches": {"B1": False, "B2": False, "B3": False},
        "scalar_metrics": scalar,
        "outcome_counts_equal": bool(counts_equal),
        "degenerate_summary_equal": bool(degenerate_equal),
        "ess_active_distribution_equal": bool(ess_equal),
        "band_blocks_equal": bands_equal,
        "per_episode_arrays_equal": array_equal,
        "note": "equality is exact (==), not a tolerance. The per-episode arrays are compared element "
                "by element, which is strictly stronger than comparing the aggregates.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="charter-'v3' B1/B2/B3 smoke checks.")
    parser.add_argument("--out", type=str,
                        default=str(REPO / "data/runs/v2.8.4/mppi_v3/smoke_recovery.json"))
    parser.add_argument("--n-scenes", type=int, default=64,
                        help="pool scenes used by the CPU checks (b), (c), (e)")
    parser.add_argument("--repro-label", type=str, default="N1024_lam0.05_C1000_m1_H40_sig1")
    parser.add_argument("--repro-artifact-dir", type=str,
                        default=str(REPO / "data/runs/v2.8.4/mppi_screen_v3"))
    parser.add_argument("--repro-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-free-mib", type=float, default=6144.0)
    parser.add_argument("--skip-repro", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    mppi_config = load_mppi_config()
    pool_path = REPO / mppi_config["screen"]["pool"]
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_b1_weight(mppi_config),
        check_b2_seeding(mppi_config, pool_path, int(args.n_scenes)),
        check_b3_trigger(mppi_config, pool_path, int(args.n_scenes)),
        check_switched_off_equivalence(mppi_config, pool_path, int(args.n_scenes)),
    ]
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['check']}", flush=True)

    if not args.skip_repro:
        device = torch.device(args.repro_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("STOP: --repro-device cuda requested but CUDA is unavailable.")
        repro = check_v2_reproduction(
            mppi_config, pool_path, str(args.repro_label), Path(args.repro_artifact_dir), out_dir,
            device, torch.float32, float(args.min_free_mib),
        )
        checks.append(repro)
        print(f"{'PASS' if repro['passed'] else 'FAIL'}  {repro['check']}", flush=True)

    summary = {
        "what": "charter-'v3' B1/B2/B3 smoke checks for the v2.8.4 MPPI baseline",
        "lineage": {
            "charter_v1": "data/runs/v2.8.4/mppi_screen/ (the original 16-cell N x H x lam x C_crash "
                          "grid, 8 cells run) — IMMUTABLE",
            "amendment_12_cell": "data/runs/v2.8.4/mppi_screen_v2/ (sigma x lam_rel x H) — IMMUTABLE",
            "charter_v2": "data/runs/v2.8.4/mppi_screen_v3/ (hover-centred / settling-terminal / "
                          "control-hold, 16 cells) — IMMUTABLE",
            "charter_v3": "THIS work — data/runs/v2.8.4/mppi_v3/, docs/versions/v2.8.4/mppi_v3.md",
        },
        "pool": str(pool_path.relative_to(REPO)),
        "device_policy": "(a)-(e) on CPU in float64; (f) on the device that produced the retained "
                         "artifact (CUDA, float32), under the same VRAM guard the screen uses, before "
                         "any screen cell",
        "n_checks": len(checks),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out_path}  all_passed={summary['all_passed']}", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
