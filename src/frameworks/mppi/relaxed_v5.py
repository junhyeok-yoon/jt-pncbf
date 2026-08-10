"""v2.8.4 charter-"v5" STAGE 3 — the RELAXED-SETTLING COMPANION. Re-scoring only, no new rollouts.

WHAT THIS DOES, AND WHAT IT MAY NEVER DO
========================================

One rollout set, two predicates. Every episode array this module reads was produced by a Stage-1 or
Stage-2 cell (or is a RETAINED, READ-ONLY rollout set from an earlier lineage); nothing here integrates
a plant, draws a random number, or touches a controller. The two predicates are:

  DEPLOYED   the eval harness's own outcome == "goal": the conjunction
             ||p - g|| <= env.goal_radius  AND  ||v|| <= env.goal_speed_radius
             AND  ||omega|| <= env.goal_angrate_radius,
             resolved by `src.common.outcomes.step_outcomes` with collision preempting goal at the
             same step. This is THE metric. It is reported FIRST and is NEVER omitted.

  RELAXED    ||p - g|| <= env.goal_radius at ANY active step. The POSITION radius is UNCHANGED and is
             READ from `env.goal_radius`; the velocity and angular-rate conditions are REMOVED.
             AN ANNOTATED COMPANION ONLY. It is never reported without the deployed number beside it,
             it never substitutes for the deployed number, and no gate, ranking or selection is taken
             against it.

All three radii are READ from the merged effective config. None is typed in this file.

THE WINDOW is the ACTIVE window of each episode — state samples 0..n_steps inclusive. `rollout_eval`
freezes the state after the first physical event, so the frozen tail carries no information; this is
the same convention `evaluate`, the degenerate accounting, the v4 columns and the v3 diagnosis use. An
episode that collided therefore cannot "reach relaxed" on its frozen post-collision states.

WHAT THE CHARTER ASKS FOR BESIDES THE PAIRED COLUMNS: for the episodes that reach under the RELAXED
terminal but NOT under the deployed one, the distribution of ||v|| and ||omega|| at the FIRST
position-satisfying step. That is the direct measurement of what the settling condition costs a
sampling planner, and it is the reason the relaxed column exists at all.

A CROSS-CHECK, RECORDED RATHER THAN ASSUMED. The deployed conjunction is ALSO re-derived here from the
stored states, and its agreement with the harness's own `outcome == "goal"` is reported per cell. The
two can legitimately differ — the harness resolves the FIRST event and lets a collision preempt a goal
at the same step, while the re-derivation is a plain "any active step" test — so a disagreement is
reported as a number, not silently reconciled. The harness value is always the one in the table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.common.system import System


DEPLOYED_STATEMENT = (
    "the eval harness's own outcome == 'goal': ||p-g|| <= env.goal_radius AND ||v|| <= "
    "env.goal_speed_radius AND ||omega|| <= env.goal_angrate_radius at one step, with collision "
    "preempting goal at the same step (src.common.outcomes.step_outcomes)"
)
RELAXED_STATEMENT = (
    "||p-g|| <= env.goal_radius at ANY active step; the POSITION radius is UNCHANGED and read from "
    "env.goal_radius, the velocity and angular-rate conditions are REMOVED. ANNOTATED COMPANION ONLY "
    "— never reported without the deployed number beside it and never a substitute for it."
)


def save_rollouts(
    path: Path,
    system: System,
    trajectories: list[Any],
    episode_rows: list[Mapping[str, Any]],
    *,
    tilt_deg: np.ndarray | None = None,
) -> dict[str, Any]:
    """Store a cell's rollouts so Stage 3 can RE-SCORE them instead of producing a second set.

    Written straight off the trajectories `run_cell` already returned through its `capture` hook, in
    the same layout the retained `mppi_diag` rollout set uses, so the same loader reads both.
    """
    states = np.stack(
        [tr.filtered.states[:, 0, :].detach().cpu().numpy().astype(np.float32) for tr in trajectories],
        axis=0,
    )                                                                    # [E, T+1, state_dim]
    goals = np.stack(
        [np.asarray(tr.scene.goal, dtype=np.float32).reshape(-1)[:3] for tr in trajectories], axis=0
    )                                                                    # [E, 3]
    n_steps = np.array([int(row["n_steps"]) for row in episode_rows], dtype=np.int64)
    outcome = np.array([str(row["outcome"]) for row in episode_rows], dtype="U16")
    arrays: dict[str, np.ndarray] = {
        "states": states, "goals": goals, "n_steps": n_steps, "outcome": outcome,
        "starts": states[:, 0, :3].copy(),
    }
    if tilt_deg is not None:
        arrays["tilt_deg"] = np.asarray(tilt_deg, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "states_shape": list(states.shape),
        "n_episodes": int(states.shape[0]),
        "state_samples_per_episode": int(states.shape[1]),
    }


def load_rollouts(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for required in ("states", "goals", "n_steps", "outcome"):
        if required not in out:
            raise KeyError(f"{path} carries no {required!r} array; it is not a v5-readable rollout set.")
    return out


@torch.no_grad()
def score_two_terminals(
    system: System,
    config: Mapping[str, Any],
    rollouts: Mapping[str, np.ndarray],
    *,
    tilt_split_deg: float | None = None,
) -> dict[str, Any]:
    """Score ONE stored rollout set under BOTH terminals. See the module docstring.

    Returns a record whose every band carries the deployed column FIRST, the relaxed column beside it,
    and — for the relaxed-only reachers — the ||v|| and ||omega|| distributions at the first
    position-satisfying step.
    """
    goal_radius = float(config["env"]["goal_radius"])
    speed_radius = float(config["env"]["goal_speed_radius"])
    angrate_radius = float(config["env"]["goal_angrate_radius"])

    states = torch.as_tensor(np.asarray(rollouts["states"]), dtype=torch.float64)   # [E, T+1, S]
    goals = torch.as_tensor(np.asarray(rollouts["goals"]), dtype=torch.float64)     # [E, 3]
    n_steps = np.asarray(rollouts["n_steps"]).astype(np.int64)                      # [E]
    outcome = np.asarray(rollouts["outcome"]).astype("U16")                         # [E]
    n_episodes, n_samples_max = int(states.shape[0]), int(states.shape[1])

    flat = states.reshape(-1, states.shape[-1])
    distance = torch.linalg.norm(
        system.position(flat) - goals.repeat_interleave(n_samples_max, dim=0), dim=-1
    ).reshape(n_episodes, n_samples_max).numpy()
    speed = system.speed(flat).reshape(n_episodes, n_samples_max).numpy()
    angrate = system.angular_rate(flat).reshape(n_episodes, n_samples_max).numpy()

    # ACTIVE window: state samples 0..n_steps inclusive.
    index = np.arange(n_samples_max)[None, :]
    active = index <= np.clip(n_steps, 0, n_samples_max - 1)[:, None]               # [E, T+1]

    position_leg = (distance <= goal_radius) & active
    speed_leg = speed <= speed_radius
    angrate_leg = angrate <= angrate_radius

    deployed_harness = outcome == "goal"
    deployed_rederived = (position_leg & speed_leg & angrate_leg).any(axis=1)
    relaxed = position_leg.any(axis=1)

    # FIRST position-satisfying step of every episode that has one.
    has_position = position_leg.any(axis=1)
    first_position = np.where(has_position, position_leg.argmax(axis=1), -1)

    def band(mask: np.ndarray, name: str) -> dict[str, Any]:
        n = int(mask.sum())
        block: dict[str, Any] = {"band": name, "n": n}
        if n == 0:
            return block
        # DEPLOYED FIRST, ALWAYS.
        block["reach_deployed"] = float(deployed_harness[mask].mean())
        block["reach_deployed_count"] = int(deployed_harness[mask].sum())
        block["reach_relaxed"] = float(relaxed[mask].mean())
        block["reach_relaxed_count"] = int(relaxed[mask].sum())
        block["reach_deployed_rederived_from_states"] = float(deployed_rederived[mask].mean())
        block["deployed_rederivation_agrees_with_harness"] = bool(
            np.array_equal(deployed_rederived[mask], deployed_harness[mask])
        )
        block["deployed_rederivation_disagreements"] = int(
            (deployed_rederived[mask] != deployed_harness[mask]).sum()
        )
        block["outcome_counts"] = {
            o: int((outcome[mask] == o).sum()) for o in sorted(set(outcome[mask].tolist()))
        }
        block["d_min_p50"] = float(np.median(np.where(active[mask], distance[mask], np.inf).min(axis=1)))
        block["d_min_min"] = float(np.where(active[mask], distance[mask], np.inf).min())

        # ---- the relaxed-only reachers -----------------------------------------------------------
        only = mask & relaxed & ~deployed_harness
        block["relaxed_only_n"] = int(only.sum())
        if int(only.sum()) == 0:
            block["relaxed_only_note"] = (
                "no episode in this band reaches under the relaxed terminal but not the deployed one, "
                "so there is no |v| / |omega| distribution to report"
            )
            return block
        rows = np.nonzero(only)[0]
        step = first_position[rows]
        v_at = speed[rows, step]
        w_at = angrate[rows, step]

        def distribution(values: np.ndarray, radius: float) -> dict[str, Any]:
            pct = np.percentile(values, [5, 25, 50, 75, 95])
            return {
                "n": int(values.size), "mean": float(values.mean()),
                "min": float(values.min()), "max": float(values.max()),
                "p05": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
                "p75": float(pct[3]), "p95": float(pct[4]),
                "deployed_radius_read_from_config": radius,
                "frac_inside_deployed_radius": float((values <= radius).mean()),
            }

        block["relaxed_only_at_first_position_satisfying_step"] = {
            "window": "the FIRST active step at which ||p-g|| <= env.goal_radius",
            "first_step_index": {
                "min": int(step.min()), "median": float(np.median(step)), "max": int(step.max()),
            },
            "speed_abs_v": distribution(v_at, speed_radius),
            "angular_rate_abs_omega": distribution(w_at, angrate_radius),
            "frac_both_legs_held": float(
                ((v_at <= speed_radius) & (w_at <= angrate_radius)).mean()
            ),
            "per_episode": {
                "episode_index": [int(i) for i in rows],
                "first_step": [int(s) for s in step],
                "abs_v": [float(v) for v in v_at],
                "abs_omega": [float(w) for w in w_at],
            },
        }
        return block

    record: dict[str, Any] = {
        "what": "the SAME rollout arrays scored under two terminals; one rollout, two predicates. "
                "NO new rollouts were produced.",
        "deployed_terminal": DEPLOYED_STATEMENT,
        "relaxed_terminal": RELAXED_STATEMENT,
        "radii_read_from_config": {
            "goal_radius": goal_radius,
            "goal_speed_radius": speed_radius,
            "goal_angrate_radius": angrate_radius,
            "source": "the merged effective config (env.goal_radius / env.goal_speed_radius / "
                      "env.goal_angrate_radius); no radius is typed in code",
        },
        "window": "active window, state samples 0..n_steps inclusive (states freeze after the event)",
        "n_episodes": n_episodes,
        "ALL": band(np.ones(n_episodes, dtype=bool), "ALL"),
    }
    tilt = rollouts.get("tilt_deg")
    if tilt is not None and tilt_split_deg is not None:
        tilt = np.asarray(tilt, dtype=np.float64)
        split = float(tilt_split_deg)
        record["split_deg"] = split
        record["tilt_le_ref"] = band(tilt <= split, f"tilt<={split:g}")
        record["tilt_gt_ref"] = band(tilt > split, f"tilt>{split:g}")
    return record
