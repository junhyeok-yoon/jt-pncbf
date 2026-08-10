"""v2.8.4 — the reach ladder's runner. MEASUREMENT ONLY.

Runs the mission's STRUCTURAL GATE — the obstacle-free straight-line scene of
`src/frameworks/mppi/smoke_cascade_rate.py:66`, closed loop on the FULL RK4 rotor-thrust plant — for a
named rung of `reach_rungs.ReachFlags`, and writes `data/runs/v2.8.4/mppi_reach/**`.

The episode loop is `smoke_cascade_rate.check_straight_line_planner:199-202` entry for entry:

    controller.reset(1); for each of n_steps: u = controller.act(x, batched); x = rk4_step(system, x, u, dt)

so the ALL-FLAGS-OFF rung must reproduce the recorded smoke number

    data/runs/v2.8.4/mppi_cascaded/smoke.json  checks.b2_straight_line_planner.N1024.closest_approach_m
        = 1.0560340480879828

to the last digit. That equality is the OFF-state bit-parity check, and it is asserted here rather than
described: `--require-parity` makes a mismatch a non-zero exit.

NOTHING here touches a pool, selects a cell, registers a threshold, ranks anything or promotes anything.
No number produced by this file may share a table with a rotor-direct or a reference number without a
variant column.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from src._version import __version__
from src.common.rk4 import rk4_step
from src.envs.scene_batch import batch_scenes
from src.frameworks.mppi import smoke_cascade_rate as smoke
from src.frameworks.mppi.evaluate_mppi import REPO, effective_config, load_mppi_config
from src.frameworks.mppi.reach_r0 import OUT_DIR, dir_state, proc_state, write_report
from src.frameworks.mppi.reach_rungs import ReachFlags, build_reach_controller


Tensor = torch.Tensor

# the recorded OFF-state target: the shipped implementation's own straight-line smoke number
RECORDED_SMOKE = REPO / "data/runs/v2.8.4/mppi_cascaded/smoke.json"


def recorded_closest() -> float:
    payload = json.loads(RECORDED_SMOKE.read_text(encoding="utf-8"))
    return float(payload["checks"]["b2_straight_line_planner"]["N1024"]["closest_approach_m"])


# ---- the ladder. Each entry is ONE rung; every rung is independently ablatable because every flag is
#      independent and the OFF state of all of them together is the shipped code path. -----------------
RUNGS: dict[str, ReachFlags] = {
    "R0_all_flags_off":      ReachFlags(),
    "R1_lambda_quantile":    ReachFlags(r1_lambda="quantile", r1_q=0.5),
    "R2_seed32":             ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2),
    "R2_seed32_nojitter":    ReachFlags(r2_n_seed=32, r2_gain_jitter=0.0),
    "R2_seed1":              ReachFlags(r2_n_seed=1, r2_gain_jitter=0.0),
    "R2_seed64":             ReachFlags(r2_n_seed=64, r2_gain_jitter=0.2),
    "R2_seed64_nojitter":    ReachFlags(r2_n_seed=64, r2_gain_jitter=0.0),
    "R2_seed128":            ReachFlags(r2_n_seed=128, r2_gain_jitter=0.2),
    "R2_seed128_nojitter":   ReachFlags(r2_n_seed=128, r2_gain_jitter=0.0),
    "R2_seed256":            ReachFlags(r2_n_seed=256, r2_gain_jitter=0.2),
    "R3_yaw_damped":         ReachFlags(r3_yaw="damped", r3_k_yaw=1.0),
    "R4_index_noise":        ReachFlags(r4_edge=0.25),
    "R6_dt_growth":          ReachFlags(r6_dt_growth=1.05),
    "R2_seed8":              ReachFlags(r2_n_seed=8, r2_gain_jitter=0.2),
    # R1+R2. R0 measured the shipped relative-lambda rule resolving lambda p50 467 (ESS p50 847), which
    # is flat enough that ONE good sequence out of K cannot move the nominal — the R2 sweep shows the
    # seeded block buying its weight by COUNT (weight mass 0.33 at 64 rows, 0.79 at 128). R1 sharpens the
    # temperature, so these rows test whether the deterministic sequence can dominate ON MERIT at
    # GMPPI's own seed ratio (32 of 768 = 4.2 %) instead of by repetition.
    "R1_R2_seed1":           ReachFlags(r2_n_seed=1, r2_gain_jitter=0.0, r1_lambda="quantile", r1_q=0.5),
    "R1_R2_seed8":           ReachFlags(r2_n_seed=8, r2_gain_jitter=0.2, r1_lambda="quantile", r1_q=0.5),
    "R2_seed32_R1":          ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r1_lambda="quantile", r1_q=0.5),
    "R2_seed32_R3":          ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r3_yaw="damped", r3_k_yaw=1.0),
    "R2_seed32_R4":          ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r4_edge=0.25),
    # BALANCE PROBE. Every configuration that reaches above does so with >= 0.79 of the weight mass on
    # the seeded rows; at r1_q = 0.5 it is 0.9999, i.e. the stochastic search contributes nothing. A
    # LARGER r1_q is a SOFTER temperature (the spread is taken at a higher quantile of the shifted cost),
    # so these rows ask whether the gate can be held while genuinely sampled rollouts keep material
    # weight — which is what makes the row an MPPI baseline rather than the deterministic law routed
    # through MPPI.
    "R2_seed32_R1q75":       ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r1_lambda="quantile", r1_q=0.75),
    "R2_seed32_R1q90":       ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r1_lambda="quantile", r1_q=0.90),
    "R2_seed32_R1q99":       ReachFlags(r2_n_seed=32, r2_gain_jitter=0.2, r1_lambda="quantile", r1_q=0.99),
    "R2_seed128_R4":         ReachFlags(r2_n_seed=128, r2_gain_jitter=0.2, r4_edge=0.25),
}


@torch.no_grad()
def run_rung(
    name: str, flags: ReachFlags, mppi_config: dict[str, Any], config: dict[str, Any],
    *, n_samples: int, offset: list[float] | None = None,
) -> dict[str, Any]:
    """One rung on ONE obstacle-free straight-line scene.

    `offset` defaults to the smoke's own `mppi.cascaded.smoke.straight_line.offset`, which is THE GATE.
    Passing another offset builds another CONSTRUCTED obstacle-free scene through the same
    `smoke_cascade_rate.straight_line_scene` — it is a robustness annex, NOT a pool and NOT a screen.
    """
    device, dtype = torch.device("cpu"), torch.float64
    smoke_cfg = mppi_config["cascaded"]["smoke"]
    seed = int(mppi_config["cascaded"]["scale"]["seed"])
    dt = float(config["env"]["dt"])
    goal_radius = float(config["env"]["goal_radius"])

    system, controller = build_reach_controller(
        mppi_config, config, flags=flags, n_samples=n_samples, noise="ou", lam_mode="relative",
        device=device, dtype=dtype,
    )
    offset = [float(v) for v in (smoke_cfg["straight_line"]["offset"] if offset is None else offset)]
    scene = smoke.straight_line_scene(system, offset, dtype)
    batched = batch_scenes([scene], device=device, dtype=dtype)
    x = smoke.level_state(system, torch.as_tensor(scene.start, dtype=dtype).view(1, 3), dtype)
    goal = torch.as_tensor(scene.goal, dtype=dtype).view(1, 3)
    seconds = float(smoke_cfg["straight_line"]["seconds"])
    n_steps = int(round(seconds / dt))

    controller.reset(1)
    controller.generator.manual_seed(seed)

    t0 = time.time()
    dist, ess, lam, speed, abs_z = [], [], [], [], []
    for _ in range(n_steps):
        u = controller.act(x, batched)
        x = rk4_step(system, x, u, dt)
        dist.append(float(torch.linalg.norm(system.position(x) - goal).item()))
        ess.append(float(controller.last_ess[0].item()))
        lam.append(float(controller.last_lam_eff[0].item()))
        speed.append(float(system.speed(x).item()))
        abs_z.append(float(system.position(x)[0, 2].abs().item()))

    def band(values: list[float]) -> dict[str, float]:
        t = torch.as_tensor(values, dtype=torch.float64)
        return {"min": float(t.min()), "p50": float(t.median()), "p95": float(t.quantile(0.95)),
                "max": float(t.max()), "mean": float(t.mean())}

    closest = min(dist)
    return {
        "rung": name,
        "variant": "cascaded_rate",
        "offset_m": offset,
        "is_the_gate_scene": offset == [float(v) for v in smoke_cfg["straight_line"]["offset"]],
        "flags": flags.record(),
        "reach": controller.reach_record(),
        "N_samples": int(controller.params.n_samples),
        "horizon": int(controller.params.horizon),
        "dt": controller.dt,
        "n_control_steps": n_steps,
        "seconds": seconds,
        "goal_radius_read_from_config": goal_radius,
        "closest_approach_m": closest,
        "final_distance_m": dist[-1],
        "final_speed_m_s": float(system.speed(x).item()),
        "final_angular_rate_rad_s": float(system.angular_rate(x).item()),
        "REACHES": bool(closest <= goal_radius),
        "ess": band(ess),
        "lam": band(lam),
        "speed_m_s": band(speed),
        "abs_z_m": band(abs_z),
        "degenerate_steps": int(controller.degenerate_steps[0].item()),
        # R2 HONESTY: if the seeded rows carried essentially all the weight and were always the argmin,
        # the rung would be the deterministic law in disguise rather than a search improvement.
        "r2_seed_weight_mass": band(controller.seed_weight_mass) if controller.seed_weight_mass else None,
        "r2_frac_steps_argmin_is_seeded": (
            sum(controller.seed_is_argmin) / len(controller.seed_is_argmin)
            if controller.seed_is_argmin else None
        ),
        "preclip_over_this_leg": controller.preclip_record(),
        "wall_s": round(time.time() - t0, 2),
        "per_step_distance_m": dist,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", type=str, default="R0_all_flags_off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument(
        "--offsets", type=str, default="",
        help="semicolon-separated 'x,y,z' goal offsets. Each builds another CONSTRUCTED obstacle-free "
             "straight-line scene through smoke_cascade_rate.straight_line_scene — a robustness annex, "
             "NOT a pool and NOT a screen. Empty = the gate offset only.",
    )
    parser.add_argument("--control-law", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    mppi_config = load_mppi_config()
    config = effective_config(mppi_config)
    n_samples = int(mppi_config["cascaded"]["smoke"]["straight_line"]["mppi_n_samples"][0])
    names = [n.strip() for n in args.rungs.split(",") if n.strip()]
    unknown = [n for n in names if n not in RUNGS]
    if unknown:
        raise SystemExit(f"unknown rung(s) {unknown}; known: {sorted(RUNGS)}")

    before = dir_state()
    target = recorded_closest()
    results = []
    parity_ok: bool | None = None
    offsets: list[list[float] | None] = [None]
    if args.offsets:
        offsets = [[float(v) for v in chunk.split(",")] for chunk in args.offsets.split(";")]

    # THE CONTROL. `smoke_cascade_rate.check_straight_line_outer` — the DETERMINISTIC law, no planner at
    # all — on the same constructed scenes. Any seeded rung is bounded above by what this law can do,
    # because the seeded rows ARE this law rolled in the planner's model; a rung failing where the law
    # also fails is a statement about the law, not about the optimizer.
    if args.control_law:
        from src.frameworks.mppi.reach_rungs import build_reach_controller as _build
        control_rows = []
        for offset in offsets:
            system, controller = _build(
                mppi_config, config, flags=ReachFlags(), n_samples=n_samples, noise="ou",
                lam_mode="relative", device=torch.device("cpu"), dtype=torch.float64,
            )
            off = [float(v) for v in (
                mppi_config["cascaded"]["smoke"]["straight_line"]["offset"] if offset is None else offset
            )]
            scene = smoke.straight_line_scene(system, off, torch.float64)
            row = smoke.check_straight_line_outer(
                system, controller, scene, float(config["env"]["dt"]),
                float(mppi_config["cascaded"]["smoke"]["straight_line"]["seconds"]),
                float(config["env"]["goal_radius"]),
            )
            row["offset_m"] = off
            row["rung"] = "CONTROL_deterministic_law_no_planner"
            print(f"[ladder] CONTROL {off}: closest {row['closest_approach_m']:.6f} m  "
                  f"REACHES {row['reaches_position_radius']}", flush=True)
            control_rows.append(row)
        results.extend(control_rows)

    for name in names:
      for offset in offsets:
        print(f"[ladder] {name} offset={offset} ...", flush=True)
        row = run_rung(name, RUNGS[name], mppi_config, config, n_samples=n_samples, offset=offset)
        if name == "R0_all_flags_off" and offset is None:
            row["off_state_bit_parity"] = {
                "target_source": "data/runs/v2.8.4/mppi_cascaded/smoke.json "
                                 "checks.b2_straight_line_planner.N1024.closest_approach_m",
                "target_closest_approach_m": target,
                "measured_closest_approach_m": row["closest_approach_m"],
                "identical": row["closest_approach_m"] == target,
                "abs_difference": abs(row["closest_approach_m"] - target),
                "what_it_proves": "with every ladder flag off the controller reproduces the shipped "
                                  "implementation's own recorded straight-line number to the last digit, "
                                  "on the same scene, the same seed and the same plant",
            }
            parity_ok = row["off_state_bit_parity"]["identical"]
        print(f"[ladder] {name} {row['offset_m']}: closest {row['closest_approach_m']:.6f} m  "
              f"REACHES {row['REACHES']}  ({row['wall_s']} s)", flush=True)
        results.append(row)

    report = {
        "what": "v2.8.4 reach ladder — the obstacle-free straight-line STRUCTURAL GATE, one row per rung",
        "version": __version__,
        "subject": "src/frameworks/mppi/cascade_rate.py, SUBCLASSED and not edited "
                   "(src/frameworks/mppi/reach_rungs.py). cascade.py is not the subject.",
        "variant": "cascaded_rate",
        "gate": "nothing advances to the obstacle pool until this scene is REACHED "
                "(closest approach <= env.goal_radius, read from the config)",
        "process": proc_state(),
        "device": "cpu", "dtype": "torch.float64", "torch_threads": int(args.threads),
        "readonly_before": before,
        "no_selection": "MEASUREMENT ONLY. No pool, no cell selection, no ledger row, no promotion, no "
                        "final score. Cascaded, rotor-direct and reference numbers never share a table "
                        "without a variant column.",
        "rows": results,
        "any_rung_reaches": any(r.get("REACHES") for r in results),
    }
    report["readonly_after"] = dir_state()
    report["readonly_unchanged"] = report["readonly_before"] == report["readonly_after"]
    report["wall_s"] = round(time.time() - t0, 2)
    suffix = f"__{args.tag}" if args.tag else ""
    write_report(f"ladder{suffix}.json", report)
    print(f"readonly unchanged: {report['readonly_unchanged']}", flush=True)
    if args.require_parity and parity_ok is not True:
        print("OFF-STATE PARITY FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":                                                   # pragma: no cover
    raise SystemExit(main())
