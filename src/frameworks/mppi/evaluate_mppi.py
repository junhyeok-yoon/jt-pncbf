"""v2.8.4 MPPI baseline — score ONE (sigma, lambda, H, N, C_crash, control-hold) cell on a pool.

Runs the shared eval path (`src.eval.evaluate.evaluate`) so every number is produced by exactly the same
scorer, on exactly the same pool, terminal predicate and outcome resolution as the JT / OC / PPO rows.
MPPI enters as a filter-free framework (identity filter), so `infeasibility == 0` and
`mean_proj_mag == 0` by construction and `cps = reach - 2*collision - stuck - 0.5*(oob + timeout)`.

Writes, under --out-dir:
    cell__<label>.json          the eval row, the cell read-back, the sampler record, the degenerate-branch
                                summary, the ESS distribution, wall clock, and — when a tilt array is
                                supplied — the S3 reporting split `bands` (ALL / tilt<90 / tilt>=90)
    per_episode__<label>.npz    per-episode arrays (outcome, cause, cps, n_steps, ..., degenerate_steps,
                                tilt_deg) plus the full per-(step, episode) ESS, lam_eff and DECISION
                                matrices

S3 (the screen in data/runs/v2.8.4/mppi_screen_v3/) adds three controller switches, all config-read:
`--center hover|none` (the explicit hover anchor u = u_hover + u_plan + eps), `--control-hold m` (each
DECISION entry applied for m physical steps; the rollout spans H*m steps and re-planning happens every m
steps) and `--terminal settling|distance` (the deployed terminal predicate's own excess vs the legacy
distance terminal). `--center none --control-hold 1 --terminal distance` restores the S1/S2 controller.

Everything is a command-line argument, and the defaults come from `src/frameworks/mppi/config.yaml`, so
this file needs NO edit to run the screening grid or the final score on a GPU later:

    CUDA_VISIBLE_DEVICES=0 python -m src.frameworks.mppi.evaluate_mppi \
        --n-samples 256 --horizon 40 --lam 0.2 --sigma 0.3 --c-crash 1e5 \
        --pool data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.pkl \
        --n-scenes 400 --ebs 200 --out-dir data/runs/v2.8.4/mppi_screen_v2

The charter-"v4" goal-attraction redesign adds four more config-read switches, `--g1 --g2 --g3 --g4`
(with `--no-g1 ...` to force them off) plus `--w-p` / `--w-omega` for the two v4 coefficients. All four
default to their config values, which are OFF, so an unqualified invocation is the charter-"v3"
controller and every retained artifact stays reproducible.

`--space rotor --noise iid --lam-mode absolute` reproduces the SUPERSEDED first screen's sampler exactly
(the backward-reproducibility gate re-runs one of its cells that way and compares every metric).

Peak sampler memory is ~ ebs * N * H * action_dim * 2 tensors; lower --ebs if a cell does not fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.common.system import System
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_effective_config, make_system
from src.frameworks.mppi.cost import CostParams
from src.frameworks.mppi.mppi_controller import MPPIController, MPPIFramework, MPPIParams
from src.frameworks.mppi.recovery import RecoveryParams


REPO = Path(__file__).resolve().parents[3]
MPPI_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_mppi_config(path: Path = MPPI_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["mppi"]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def effective_config(mppi_config: Mapping[str, Any], max_steps: int | None = None) -> dict[str, Any]:
    """The shipped config with the declared eval-cell overrides applied (mppi.eval_cell)."""
    config = _deep_merge(load_effective_config(), mppi_config["eval_cell"])
    if max_steps is not None:
        config["eval"]["max_steps"] = int(max_steps)
    return config


def build_framework(
    config: Mapping[str, Any],
    mppi_config: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    horizon: int | None = None,
    lam: float | None = None,
    sigma: float | None = None,
    c_crash: float | None = None,
    seed: int | None = None,
    sample_chunk: int | None = None,
    space: str | None = None,
    noise: str | None = None,
    lam_mode: str | None = None,
    center: str | None = None,
    control_hold: int | None = None,
    terminal: str | None = None,
    b1: bool | None = None,
    b2: bool | None = None,
    b3: bool | None = None,
    g1: bool | None = None,
    g2: bool | None = None,
    g3: bool | None = None,
    g4: bool | None = None,
    w_p: float | None = None,
    w_omega: float | None = None,
    cascade: Mapping[str, Any] | None = None,
    rate_cascade: Mapping[str, Any] | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[System, MPPIFramework, MPPIParams, CostParams]:
    system = make_system(config)
    params = MPPIParams.from_config(
        mppi_config, n_samples=n_samples, horizon=horizon, lam=lam, sigma=sigma, seed=seed,
        sample_chunk=sample_chunk, space=space, noise=noise, lam_mode=lam_mode,
        center=center, control_hold=control_hold,
    )
    # charter "v4" G1-G4. The config ships all four OFF, so an unqualified call resolves to the
    # charter-"v3" cost and none of the v4 branches is reachable.
    cost_params = CostParams.from_config(
        mppi_config, config["env"], c_crash=c_crash, terminal=terminal,
        g1=g1, g2=g2, g3=g3, g4=g4, w_p=w_p, w_omega=w_omega,
    )
    # charter "v3" B1/B2/B3. The config ships all three OFF, so an unqualified call resolves to the
    # charter's "v2" controller and none of the recovery branches is reachable.
    recovery = RecoveryParams.from_config(mppi_config, b1=b1, b2=b2, b3=b3)
    if cascade is not None and rate_cascade is not None:
        raise ValueError(
            "`cascade` (the charter-'v5' a_des cascade) and `rate_cascade` (the (T, omega_des) "
            "literature-standard cascade) are DIFFERENT baselines with different plan variables; "
            "exactly one may be requested."
        )
    if rate_cascade is not None:
        # The literature-standard (collective thrust, desired body rate) cascade — L1-MPPI / PA-MPPI's
        # interface. A SEPARATE baseline row: not vanilla rotor-direct MPPI and not the v5 a_des
        # cascade. Constructed here, like the v5 cascade, so that everything downstream of the
        # controller — the shared eval path, the outcome resolution, the bands, the v4 columns and the
        # artifact schema — is byte-identically the vanilla one and the ONLY difference is the
        # controller. With the argument omitted (every pre-existing caller) nothing below is reachable.
        from src.frameworks.mppi.cascade_rate import RateCascadeController
        controller: MPPIController = RateCascadeController(
            system, config, params, cost_params, device=device, dtype=dtype,
            recovery=recovery, cascade=rate_cascade,
        )
    elif cascade is not None:
        # charter "v5" Stage 2 — the CASCADED variant. A separate baseline, never a vanilla row. It is
        # constructed here, rather than in a driver of its own, so that everything downstream of the
        # controller — the shared eval path, the outcome resolution, the bands, the v4 columns, the
        # endpoint probe and the artifact schema — is byte-identically the vanilla one, and the ONLY
        # difference between a Stage-1 row and a Stage-2 row is the controller itself.
        from src.frameworks.mppi.cascade import CascadeController
        controller = CascadeController(
            system, config, params, cost_params, device=device, dtype=dtype,
            recovery=recovery, cascade=cascade,
        )
    else:
        controller = MPPIController(
            system, config, params, cost_params, device=device, dtype=dtype, recovery=recovery
        )
    framework = MPPIFramework(
        system,
        controller,
        max_steps=int(config["eval"]["max_steps"]),
        device=device,
        dtype=dtype,
    )
    # the resolved RecoveryParams is reachable as `framework.controller.recovery`; the return tuple is
    # left at four entries so the pre-existing callers (diagnose.py, cpu_smoke.py) are untouched.
    return system, framework, params, cost_params


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {name!r}.")


def decision_mask(framework: MPPIFramework, n_logged_steps: int, n_episodes: int) -> np.ndarray:
    """[T, n_episodes] bool: which logged control steps were DECISION steps (S3 control hold).

    Under a hold m > 1 only every m-th step re-plans; the others re-issue the latched action and carry
    no weighting of their own. At m = 1 this is all-True and every window below is the pre-S3 window."""
    if framework.decision_chunks:
        return np.concatenate(framework.decision_chunks, axis=1)
    return np.ones((n_logged_steps, n_episodes), dtype=bool)


def degenerate_per_episode(
    framework: MPPIFramework, episode_rows: list[Mapping[str, Any]], max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode degenerate-DECISION count and the fraction of ACTIVE DECISION steps that were degenerate.

    `rollout_eval` keeps calling the policy after an episode is physically done (it zeroes the action
    instead), so degenerate steps are counted only over the active window, i.e. up to `n_steps` for an
    episode that ended on a physical event and over the whole horizon for a timeout — the same
    active-window convention `evaluate` uses for its own per-step fractions. S3: the window is further
    restricted to DECISION steps, because the degenerate branch is a property of an MPPI update and a
    held step performs none; at control_hold = 1 the restriction is vacuous and the numbers are
    identical to the pre-S3 ones. Returns (counts, frac, window) with window the [T, E] bool mask.
    """
    framework.flush()
    if not framework.degenerate_chunks:
        zeros = np.zeros(len(episode_rows), dtype=np.int64)
        return zeros, zeros.astype(np.float64), np.zeros((0, len(episode_rows)), dtype=bool)
    stacked = np.concatenate(framework.degenerate_chunks, axis=1)        # [T, n_episodes]
    active = _active_mask(episode_rows, stacked.shape[0], max_steps)
    window = active & decision_mask(framework, stacked.shape[0], stacked.shape[1])
    counts = (stacked & window).sum(axis=0).astype(np.int64)
    frac = np.divide(counts, np.maximum(window.sum(axis=0), 1)).astype(np.float64)
    return counts, frac, window


def _active_mask(episode_rows: list[Mapping[str, Any]], n_logged_steps: int, max_steps: int) -> np.ndarray:
    """[T, n_episodes] bool: the ACTIVE window of each episode, the same convention `evaluate` and the
    degenerate accounting use — up to `n_steps` for an episode that ended on a physical event, the whole
    horizon for a timeout."""
    n_steps = np.array(
        [
            int(row["n_steps"]) if row["outcome"] in ("goal", "collision", "oob") else max_steps
            for row in episode_rows
        ]
    )
    active = np.clip(n_steps, 0, n_logged_steps)
    return np.arange(n_logged_steps)[:, None] < active[None, :]


def _distribution(values: np.ndarray, n_samples: int) -> dict[str, Any]:
    """mean / min / max / p05 / p25 / p50 / p75 / p95 of an ESS sample, plus ESS/N and the fraction of
    steps at ESS < 2 (i.e. the softmax carrying essentially no more information than a hard argmin)."""
    if values.size == 0:
        return {"n_steps": 0}
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n_steps": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(percentiles[0]), "p25": float(percentiles[1]), "p50": float(percentiles[2]),
        "p75": float(percentiles[3]), "p95": float(percentiles[4]),
        "mean_over_N": float(values.mean() / max(1, n_samples)),
        "p50_over_N": float(percentiles[2] / max(1, n_samples)),
        "frac_steps_ess_below_2": float((values < 2.0).mean()),
        "frac_steps_ess_below_1p01": float((values < 1.01).mean()),
    }


def ess_summary(
    framework: MPPIFramework,
    episode_rows: list[Mapping[str, Any]],
    max_steps: int,
    n_samples: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """The per-(scene, control step) ESS distribution of a cell, and the raw [T, n_episodes] matrices.

    ESS = 1 / sum_n w_n^2 on the NORMALISED weights, so ESS in [1, N]: 1 is a hard argmin (the collapse
    the superseded screen measured), N is uniform. Reported over the ACTIVE window of every episode, and
    separately over the active NON-DEGENERATE steps (on a degenerate step the update is skipped, so the
    weights there never moved the plan). S3: the window is DECISION steps only — a held step computes no
    weights at all, so counting it would dilute the distribution with repeats of the last decision.
    """
    framework.flush()
    if not framework.ess_chunks:
        empty = np.zeros((0, len(episode_rows)), dtype=np.float32)
        return {"n_steps": 0}, empty, empty
    ess = np.concatenate(framework.ess_chunks, axis=1)            # [T, n_episodes]
    lam_eff = np.concatenate(framework.lam_eff_chunks, axis=1)    # [T, n_episodes]
    degenerate = np.concatenate(framework.degenerate_chunks, axis=1)
    decisions = decision_mask(framework, ess.shape[0], ess.shape[1])
    mask = _active_mask(episode_rows, ess.shape[0], max_steps) & decisions
    summary = {
        "definition": "ESS = 1 / sum_n w_n^2 on the normalised weights; ESS in [1, N].",
        "window": "active DECISION steps of every episode (n_steps for a physical end, max_steps for a "
                  "timeout; with control_hold m > 1 only every m-th step re-plans)",
        "n_active_decision_steps": int(mask.sum()),
        "n_active_steps": int(_active_mask(episode_rows, ess.shape[0], max_steps).sum()),
        "N": int(n_samples),
        "active": _distribution(ess[mask], n_samples),
        "active_nondegenerate": _distribution(ess[mask & ~degenerate], n_samples),
        "lam_eff_active": {
            "mean": float(lam_eff[mask].mean()) if mask.any() else 0.0,
            "p50": float(np.median(lam_eff[mask])) if mask.any() else 0.0,
            "min": float(lam_eff[mask].min()) if mask.any() else 0.0,
            "max": float(lam_eff[mask].max()) if mask.any() else 0.0,
        },
    }
    return summary, ess, lam_eff


def endpoint_probe_summary(
    framework: MPPIFramework,
    episode_rows: list[Mapping[str, Any]],
    max_steps: int,
    radii: "Sequence[float]",
) -> tuple[dict[str, Any], np.ndarray]:
    """charter "v5" Stage 1 — the share of DECISIONS whose best-cost sample ENDS inside r of the goal.

    At every decision step the controller recorded `min_n S_n`'s own sample and the distance from that
    sample's endpoint (the state after the last physical rollout step, frozen if the sample collided) to
    the scene's goal. The share below is pooled over the ACTIVE DECISION steps of every scored episode —
    the same window `ess_summary` and the degenerate accounting use, so a step counted here is a step
    counted there. Episodes whose physical event has already fired contribute no steps past it, and hold
    steps (control_hold m > 1) contribute none at all.

    The radii are REPORTING PROBES read from `mppi.v5.stage1.endpoint_probe_radii`. Nothing is decided,
    ranked or selected against them.

    Returns (summary, the raw [T, n_episodes] best-endpoint-distance matrix).
    """
    framework.controller.flush_endpoint_probe()
    chunks = framework.controller.endpoint_chunks
    if not chunks:
        return {"n_decision_steps": 0, "status": "probe was not enabled"}, np.zeros(
            (0, len(episode_rows)), dtype=np.float32
        )
    endpoint = np.concatenate(chunks, axis=1)                      # [T, n_episodes]
    mask = _active_mask(episode_rows, endpoint.shape[0], max_steps) & decision_mask(
        framework, endpoint.shape[0], endpoint.shape[1]
    )
    values = endpoint[mask]
    values = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "definition": "distance from the ENDPOINT of the argmin-cost sample to the scene goal, at "
                      "every decision step; the endpoint is the state after the last physical rollout "
                      "step (H * m steps), frozen at the collision state for a sample that collided",
        "window": "active DECISION steps of every scored episode",
        "n_decision_steps": int(values.size),
        "radii_status": "REPORTING PROBES read from mppi.v5.stage1.endpoint_probe_radii; no gate, no "
                        "ranking and no selection is taken against them",
        "shares_inside_r": {
            f"{float(r):g}": float((values <= float(r)).mean()) if values.size else float("nan")
            for r in radii
        },
        "counts_inside_r": {
            f"{float(r):g}": int((values <= float(r)).sum()) for r in radii
        },
    }
    if values.size:
        pct = np.percentile(values, [5, 25, 50, 75, 95])
        summary["best_endpoint_distance"] = {
            "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "p05": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
            "p75": float(pct[3]), "p95": float(pct[4]),
        }
    return summary, endpoint


TILT_SPLIT_DEG = 90.0
INFEASIBILITY_STATUS = (
    "STRUCTURALLY INAPPLICABLE — MPPI enters with an identity filter and carries no certificate, "
    "so there is no QP and nothing can be infeasible. The 0 is a property of the wiring, not a "
    "measurement of the controller."
)


def metric_block(
    mask: np.ndarray,
    *,
    band: str,
    outcome: np.ndarray,
    cause: np.ndarray,
    cps_episode: np.ndarray,
    saturation: np.ndarray,
    deg_counts: np.ndarray,
    deg_matrix: np.ndarray,
    deg_window: np.ndarray,
    b3_counts: np.ndarray | None = None,
    b3_matrix: np.ndarray | None = None,
    b2_fired: np.ndarray | None = None,
) -> dict[str, Any]:
    """The full outcome decomposition over a SUBSET of the scored episodes (S3 reporting split).

    Every quantity is the same functional of the per-episode rows the headline uses, restricted to
    `mask`; nothing is re-derived and nothing is dropped. `cps` is the mean of the harness's own
    `cps_episode` over the subset, so the ALL block reproduces the headline `cps` exactly. The bands are
    ADDITIONAL columns: the headline metrics of the cell are always over ALL scored episodes, born
    inverted included.
    """
    n = int(mask.sum())
    block: dict[str, Any] = {"band": band, "n": n, "frac_of_scored": float(mask.mean())}
    if n == 0:
        return block
    collision = outcome[mask] == "collision"
    block.update({
        "reach": float((outcome[mask] == "goal").mean()),
        "collision": float(collision.mean()),
        "coll_obstacle": float((collision & (cause[mask] == "obstacle")).mean()),
        "coll_band_lower": float((collision & (cause[mask] == "band_lower")).mean()),
        "coll_band_upper": float((collision & (cause[mask] == "band_upper")).mean()),
        "oob": float((outcome[mask] == "oob").mean()),
        "stuck": float((outcome[mask] == "stuck").mean()),
        "timeout": float((outcome[mask] == "timeout").mean()),
        "cps": float(cps_episode[mask].mean()),
        "saturation_rate": float(saturation[mask].mean()),
        "infeasibility": 0.0,
        "infeasibility_status": INFEASIBILITY_STATUS,
        "degenerate_episode_frac_with_any": float((deg_counts[mask] > 0).mean()),
        "degenerate_episodes_with_any": int((deg_counts[mask] > 0).sum()),
        "degenerate_total_steps": int(deg_counts[mask].sum()),
        "degenerate_step_frac_over_active_decisions": (
            float(deg_matrix[:, mask][deg_window[:, mask]].mean())
            if deg_window[:, mask].any() else 0.0
        ),
        "outcome_counts": {
            o: int((outcome[mask] == o).sum()) for o in sorted(set(outcome[mask].tolist()))
        },
    })
    # ---- charter "v3": the B2 and B3 event rates, with this block's own denominator ---------------
    if b2_fired is not None:
        block["b2_seed_episode_frac"] = float(b2_fired[mask].mean())
        block["b2_seed_episodes"] = int(b2_fired[mask].sum())
    if b3_counts is not None and b3_matrix is not None:
        block["b3_event_episode_frac"] = float((b3_counts[mask] > 0).mean())
        block["b3_event_episodes"] = int((b3_counts[mask] > 0).sum())
        block["b3_event_total_steps"] = int(b3_counts[mask].sum())
        block["b3_event_step_frac_over_active_decisions"] = (
            float(b3_matrix[:, mask][deg_window[:, mask]].mean()) if deg_window[:, mask].any() else 0.0
        )
    return block


def v4_report_columns(
    system: System,
    mppi_config: Mapping[str, Any],
    trajectories: list[Any],
    episode_rows: list[Mapping[str, Any]],
    *,
    tilt_deg: np.ndarray | None = None,
    tilt_split_deg: float | None = None,
) -> dict[str, Any]:
    """The charter-"v4" round-table reporting columns, read off the rollouts the cell already produced.

    Columns, all over the ACTIVE window (state samples 0..n_steps inclusive, action steps 0..n_steps-1 —
    `rollout_eval` freezes the state and zeroes the action after the first physical event, so the frozen
    tail carries no information; the same convention `evaluate`, the degenerate accounting and the v3
    diagnosis already use):

      d_min            per episode, min_t ||p_t - g||; the table reports its MEDIAN over all scored episodes
      inside_r_share   pooled over all active state samples of all scored episodes, the fraction with
                       ||p_t - g|| <= r, r = `mppi.goal_v4.report.probe_radius` READ FROM CONFIG. A PROBE
                       FOR A REPORTING COLUMN, never a threshold — nothing is decided against it.
      omega_p50        median of ||omega_t|| over the same pooled active state samples
      smooth_mean_du   control smoothness: mean of |u_t - u_{t-1}| over every active action transition and
                       every rotor, pooled across episodes (denominator = sum over episodes of
                       (n_active_transitions) x action_dim). An episode with fewer than two active action
                       steps contributes no transition and is counted in `n_episodes_without_transition`.

    NO THRESHOLD is applied and no episode is filtered: the pooled columns run over every scored episode,
    born inverted included, and the tilt bands are ADDITIONAL blocks with their own denominators.

    RNG. Every call below is post-rollout arithmetic on tensors the rollout already produced, so it draws
    no random numbers; it is nevertheless wrapped in `torch.random.fork_rng` so that instrumentation can
    never shift the stream a later cell in the same process draws from. Recorded, not assumed.
    """
    probe_radius = float(mppi_config["goal_v4"]["report"]["probe_radius"])
    n_episodes = len(episode_rows)
    d_min = np.zeros(n_episodes, dtype=np.float64)
    d_end = np.zeros(n_episodes, dtype=np.float64)
    n_active_samples = np.zeros(n_episodes, dtype=np.int64)
    n_inside = np.zeros(n_episodes, dtype=np.int64)
    du_sum = np.zeros(n_episodes, dtype=np.float64)
    du_count = np.zeros(n_episodes, dtype=np.int64)
    omega_pool: list[np.ndarray] = []
    dist_pool: list[np.ndarray] = []

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        with torch.no_grad():
            for i, (row, tr) in enumerate(zip(episode_rows, trajectories)):
                states = tr.filtered.states[:, 0, :]
                actions = tr.filtered.u_safe[:, 0, :]
                n_steps = int(row["n_steps"])
                active = states[: n_steps + 1]
                goal = torch.as_tensor(tr.scene.goal, dtype=active.dtype, device=active.device)
                dist = torch.linalg.norm(system.position(active) - goal, dim=-1)
                omega = system.angular_rate(active)
                d = dist.detach().cpu().numpy().astype(np.float64)
                w = omega.detach().cpu().numpy().astype(np.float64)
                d_min[i] = float(d.min())
                d_end[i] = float(d[-1])
                n_active_samples[i] = int(d.size)
                n_inside[i] = int((d <= probe_radius).sum())
                dist_pool.append(d)
                omega_pool.append(w)
                u_active = actions[: max(n_steps, 0)]
                if u_active.shape[0] >= 2:
                    du = torch.abs(u_active[1:] - u_active[:-1])
                    du_sum[i] = float(du.sum().item())
                    du_count[i] = int(du.numel())

    dist_all = np.concatenate(dist_pool) if dist_pool else np.zeros(0)
    omega_all = np.concatenate(omega_pool) if omega_pool else np.zeros(0)
    per_episode_du = np.divide(du_sum, np.maximum(du_count, 1))
    per_episode_du[du_count == 0] = np.nan

    def block(mask: np.ndarray, band: str) -> dict[str, Any]:
        n = int(mask.sum())
        out: dict[str, Any] = {"band": band, "n": n}
        if n == 0:
            return out
        samples = int(n_active_samples[mask].sum())
        transitions = int(du_count[mask].sum())
        out.update({
            "d_min_p50": float(np.median(d_min[mask])),
            "d_min_p05": float(np.percentile(d_min[mask], 5)),
            "d_min_p95": float(np.percentile(d_min[mask], 95)),
            "d_min_min": float(d_min[mask].min()),
            "d_end_p50": float(np.median(d_end[mask])),
            "inside_r_share": float(n_inside[mask].sum()) / float(max(samples, 1)),
            "inside_r_episodes_entering": int((n_inside[mask] > 0).sum()),
            "n_active_state_samples": samples,
            "smooth_mean_du": (
                float(du_sum[mask].sum()) / float(transitions) if transitions else float("nan")
            ),
            "n_active_transition_entries": transitions,
            "n_episodes_without_transition": int((du_count[mask] == 0).sum()),
        })
        idx = np.nonzero(mask)[0]
        band_omega = np.concatenate([omega_pool[i] for i in idx]) if idx.size else np.zeros(0)
        out["omega_p50"] = float(np.median(band_omega)) if band_omega.size else float("nan")
        out["omega_p95"] = float(np.percentile(band_omega, 95)) if band_omega.size else float("nan")
        return out

    all_mask = np.ones(n_episodes, dtype=bool)
    columns: dict[str, Any] = {
        "definitions": {
            "window": "active window; state samples 0..n_steps inclusive, action steps 0..n_steps-1",
            "probe_radius": probe_radius,
            "probe_radius_status": "PROBE for the time-inside reporting column, read from "
                                   "mppi.goal_v4.report.probe_radius. NOT a threshold: no gate, no "
                                   "ranking and no selection is taken against it.",
            "inside_r_share": "pooled over all active state samples of all scored episodes",
            "omega_p50": "median of ||omega|| over the same pooled active state samples",
            "smooth_mean_du": "mean |u_t - u_{t-1}| over every active action transition and rotor, "
                              "pooled across episodes",
            "filtering": "none; every scored episode contributes, born inverted included",
        },
        "rng_isolation": "post-rollout arithmetic only, wrapped in torch.random.fork_rng",
        "ALL": block(all_mask, "ALL"),
    }
    if tilt_deg is not None and tilt_split_deg is not None:
        tilt = np.asarray(tilt_deg, dtype=np.float64)
        split = float(tilt_split_deg)
        columns["split_deg"] = split
        columns["tilt_le_ref"] = block(tilt <= split, f"tilt<={split:g}")
        columns["tilt_gt_ref"] = block(tilt > split, f"tilt>{split:g}")
    columns["per_episode"] = {
        "d_min": d_min.tolist(),
        "d_end": d_end.tolist(),
        "n_active_samples": n_active_samples.tolist(),
        "n_inside_probe": n_inside.tolist(),
        "mean_du": [None if np.isnan(v) else float(v) for v in per_episode_du],
    }
    return columns


def run_cell(
    *,
    pool_path: Path,
    n_scenes: int | None,
    ebs: int,
    n_samples: int,
    horizon: int,
    lam: float,
    c_crash: float,
    sigma: float | None,
    seed: int | None,
    label: str,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    sample_chunk: int | None = None,
    space: str | None = None,
    noise: str | None = None,
    lam_mode: str | None = None,
    center: str | None = None,
    control_hold: int | None = None,
    terminal: str | None = None,
    b1: bool | None = None,
    b2: bool | None = None,
    b3: bool | None = None,
    g1: bool | None = None,
    g2: bool | None = None,
    g3: bool | None = None,
    g4: bool | None = None,
    w_p: float | None = None,
    w_omega: float | None = None,
    tilt_deg: np.ndarray | None = None,
    tilt_split_deg: float | None = None,
    max_steps: int | None = None,
    mppi_config: Mapping[str, Any] | None = None,
    v4_columns: bool = False,
    cascade: Mapping[str, Any] | None = None,
    rate_cascade: Mapping[str, Any] | None = None,
    endpoint_probe_radii: Sequence[float] | None = None,
    capture: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Score one cell. `b1`/`b2`/`b3` override the charter-"v3" recovery switches (None = the config
    value, which is False for all three, i.e. the charter's "v2" controller).

    `tilt_split_deg` selects the reporting split. None keeps the S3 convention — bands at
    `tilt < 90` / `tilt >= 90` — so the retained S3 artifacts stay reproducible; a value switches to the
    charter-"v3" convention, `tilt <= theta_ref` / `tilt > theta_ref`, with that value as theta_ref.
    """
    mppi_config = mppi_config or load_mppi_config()
    config = effective_config(mppi_config, max_steps=max_steps)
    system, framework, params, cost_params = build_framework(
        config,
        mppi_config,
        n_samples=n_samples,
        horizon=horizon,
        lam=lam,
        sigma=sigma,
        c_crash=c_crash,
        seed=seed,
        sample_chunk=sample_chunk,
        space=space,
        noise=noise,
        lam_mode=lam_mode,
        center=center,
        control_hold=control_hold,
        terminal=terminal,
        b1=b1,
        b2=b2,
        b3=b3,
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4,
        w_p=w_p,
        w_omega=w_omega,
        cascade=cascade,
        rate_cascade=rate_cascade,
        device=device,
        dtype=dtype,
    )
    # charter "v5" Stage 1: switch the best-sample endpoint probe on BEFORE the rollout — the endpoint
    # exists only inside the rollout and cannot be recovered afterwards. With the argument omitted the
    # flag stays False and none of the probe's branches is reachable.
    if endpoint_probe_radii is not None:
        framework.controller.endpoint_probe = True
    recovery = framework.controller.recovery
    steps = int(config["eval"]["max_steps"])
    # The chunk-boundary reset in MPPIFramework.policy assumes one policy call per integration step.
    assert float(config["eval"]["dt_ctrl"]) == float(config["env"]["dt"]), (
        "MPPIFramework's episode bookkeeping requires eval.dt_ctrl == env.dt (substeps = 1)."
    )

    pool = load_pool(pool_path)
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    result = evaluate(
        framework,
        pool,
        config,
        mode="final",
        step=0,
        ckpt_name=label,
        max_scenes=n_scenes,
        include_lqr_baseline=False,
        eval_batch_size=ebs,
    )
    wall = time.time() - start
    peak_alloc_mib = peak_reserved_mib = 0.0
    if device.type == "cuda":
        peak_alloc_mib = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20

    rows = result.episode_rows
    outcome = np.array([r["outcome"] for r in rows], dtype=object)
    cause = np.array([r.get("collision_cause", "") or "" for r in rows], dtype=object)
    deg_counts, deg_frac, deg_window = degenerate_per_episode(framework, rows, steps)
    ess_dist, ess_matrix, lam_matrix = ess_summary(framework, rows, steps, int(params.n_samples))
    if framework.degenerate_chunks:
        deg_matrix = np.concatenate(framework.degenerate_chunks, axis=1)          # [T, n_episodes]
        deg_step_frac = float(deg_matrix[deg_window].mean()) if deg_window.any() else 0.0
    else:
        deg_matrix = np.zeros((0, len(rows)), dtype=bool)
        deg_step_frac = 0.0

    per_episode = {
        key: np.array([float(r[key]) for r in rows])
        for key in (
            "cps_episode", "reach", "collision", "oob", "stuck", "timeout",
            "saturation_step_frac", "min_window_displacement", "mean_proj_mag", "max_h",
            "traj_path_len", "angrate_at_reach",
        )
    }
    per_episode["outcome"] = outcome.astype("U16")
    per_episode["cause"] = cause.astype("U16")
    per_episode["n_steps"] = np.array([int(r["n_steps"]) for r in rows])
    per_episode["band_crossings"] = np.array([int(r["band_crossings"]) for r in rows])
    per_episode["degenerate_steps"] = deg_counts
    per_episode["degenerate_active_frac"] = deg_frac
    # the full per-(control step, episode) weighting log, so the ESS distribution is auditable and
    # re-derivable from the artifact rather than only from the summary.
    per_episode["ess_steps"] = ess_matrix.astype(np.float32)
    per_episode["lam_eff_steps"] = lam_matrix.astype(np.float32)
    per_episode["decision_steps"] = (
        np.concatenate(framework.decision_chunks, axis=1)
        if framework.decision_chunks else np.zeros((0, len(rows)), dtype=bool)
    )
    if tilt_deg is not None:
        per_episode["tilt_deg"] = np.asarray(tilt_deg, dtype=np.float64)

    # ---- charter "v3": the B2 / B3 per-episode record ---------------------------------------------
    b3_matrix = (
        np.concatenate(framework.b3_chunks, axis=1)
        if framework.b3_chunks else np.zeros((0, len(rows)), dtype=bool)
    )
    ess_pre_matrix = (
        np.concatenate(framework.ess_pre_chunks, axis=1)
        if framework.ess_pre_chunks else np.zeros((0, len(rows)), dtype=np.float32)
    )
    b3_counts = (
        (b3_matrix & deg_window).sum(axis=0).astype(np.int64) if b3_matrix.size
        else np.zeros(len(rows), dtype=np.int64)
    )
    b2_fired = (
        np.concatenate(framework.b2_fired_chunks, axis=0)
        if framework.b2_fired_chunks else np.zeros(len(rows), dtype=bool)
    )
    b2_axis_degenerate = (
        np.concatenate(framework.b2_axis_degenerate_chunks, axis=0)
        if framework.b2_axis_degenerate_chunks else np.zeros(len(rows), dtype=bool)
    )
    controller_spawn_tilt = (
        np.concatenate(framework.spawn_tilt_chunks, axis=0)
        if framework.spawn_tilt_chunks else np.zeros(len(rows), dtype=np.float64)
    )
    per_episode["b3_event_steps"] = b3_matrix
    per_episode["b3_event_count"] = b3_counts
    per_episode["ess_pre_steps"] = ess_pre_matrix.astype(np.float32)
    per_episode["b2_seeded"] = b2_fired
    per_episode["b2_axis_degenerate"] = b2_axis_degenerate
    per_episode["controller_spawn_tilt_deg"] = controller_spawn_tilt

    eval_row = result.eval_row
    cell = {
        "label": label,
        "arm": "MPPI (privileged, model-based, training-free)",
        "cell": {
            "N": int(params.n_samples), "H": int(params.horizon), "lambda": float(params.lam),
            "lambda_mode": params.lam_mode,
            "C_crash": float(cost_params.c_crash), "sigma": float(params.sigma),
            "space": params.space, "noise": params.noise,
            "seed": int(params.seed), "sample_chunk": int(params.sample_chunk),
            "control_hold_m": int(framework.controller.control_hold),
            "terminal": cost_params.terminal_mode,
            "center": params.center,
            "B1": bool(recovery.b1_enabled) if recovery is not None else False,
            "B2": bool(recovery.b2_enabled) if recovery is not None else False,
            "B3": bool(recovery.b3_enabled) if recovery is not None else False,
            "G1": bool(cost_params.g1_enabled),
            "G2": bool(cost_params.g2_enabled),
            "G3": bool(cost_params.g3_enabled),
            "G4": bool(cost_params.g4_enabled),
        },
        "goal_v4": cost_params.goal_v4_record(),
        "sampler": framework.controller.sampler_record(),
        "read_back": {
            "system": system.name,
            "dt": float(config["env"]["dt"]),
            "dt_ctrl": float(config["eval"]["dt_ctrl"]),
            "max_steps": steps,
            "u_bounds": system.u_bounds.tolist(),
            "hover_trim_per_rotor": float(framework.controller.trim_rotor[0].item()),
            "terminal": [
                float(config["env"]["goal_radius"]),
                float(config["env"]["goal_speed_radius"]),
                float(config["env"]["goal_angrate_radius"]),
            ],
            "band_collision_limit": float(config["env"]["band_collision_limit"]),
            "band_terminates": bool(config["env"]["band_terminates"]),
            "cost_weights": {
                "w_goal": cost_params.w_goal, "w_vel": cost_params.w_vel,
                "w_angrate": cost_params.w_angrate, "w_terminal": cost_params.w_terminal,
                "gate_scale": cost_params.gate_scale, "gate_rho": cost_params.gate_rho,
            },
            "device": str(device), "dtype": str(dtype),
            "filter": "identity (MPPI carries no certificate; infeasibility == 0 by construction)",
            "privilege": "full 13-D state + obstacle field + true plant model",
        },
        "pool": {
            "path": str(pool_path), "name": pool.name, "seed": int(pool.seed),
            "sha256": pool_sha, "sha8": pool_sha[:8],
            "n_scenes_scored": len(rows), "n_scenes_in_pool": int(pool.n_scenes), "ebs": int(ebs),
        },
        "wall_s": round(wall, 2),
        # control steps actually executed = (number of scene chunks) * max_steps; the eval rollout calls
        # the policy once per integration step for the whole chunk, done episodes included.
        "n_control_steps": int(np.ceil(len(rows) / ebs)) * steps,
        "wall_per_control_step_s": round(wall / max(1, int(np.ceil(len(rows) / ebs)) * steps), 4),
        "peak_cuda_alloc_mib": round(peak_alloc_mib, 1),
        "peak_cuda_reserved_mib": round(peak_reserved_mib, 1),
        "cps": float(eval_row["cps"]),
        "cps_ci_lo": float(eval_row["cps_ci_lo"]), "cps_ci_hi": float(eval_row["cps_ci_hi"]),
        "reach": float(eval_row["reach"]),
        "reach_ci_lo": float(eval_row["reach_ci_lo"]), "reach_ci_hi": float(eval_row["reach_ci_hi"]),
        "collision": float(eval_row["collision"]),
        "collision_ci_lo": float(eval_row["collision_ci_lo"]),
        "collision_ci_hi": float(eval_row["collision_ci_hi"]),
        "coll_obstacle": float(((outcome == "collision") & (cause == "obstacle")).mean()),
        "coll_band_lower": float(((outcome == "collision") & (cause == "band_lower")).mean()),
        "coll_band_upper": float(((outcome == "collision") & (cause == "band_upper")).mean()),
        "oob": float(eval_row["oob"]), "stuck": float(eval_row["stuck"]),
        "timeout": float(eval_row["timeout"]),
        "infeasibility": float(eval_row["infeasibility"]),
        "infeasibility_status": INFEASIBILITY_STATUS,
        "saturation_rate": float(eval_row["saturation_rate"]),
        "outcome_counts": {o: int((outcome == o).sum()) for o in sorted(set(outcome.tolist()))},
        "degenerate": {
            "episodes_with_any": int((deg_counts > 0).sum()),
            "episode_frac_with_any": float((deg_counts > 0).mean()) if deg_counts.size else 0.0,
            "total_steps": int(deg_counts.sum()),
            "step_frac_over_active_window": deg_step_frac,
            "mean_active_frac": float(deg_frac.mean()) if deg_frac.size else 0.0,
            "max_steps_in_one_episode": int(deg_counts.max()) if deg_counts.size else 0,
        },
        "ess": ess_dist,
        # ---- charter "v3": the component record and the measured event rates ----------------------
        "recovery_components": recovery.record() if recovery is not None else None,
        "recovery_events": {
            "b2_seed_episodes": int(b2_fired.sum()),
            "b2_seed_episode_frac": float(b2_fired.mean()) if b2_fired.size else 0.0,
            "b2_block_decision_entries": int(recovery.b2_block_steps) if recovery is not None else 0,
            "b2_block_physical_steps": (
                int(recovery.b2_block_steps) * int(framework.controller.control_hold)
                if recovery is not None else 0
            ),
            "b2_axis_degenerate_episodes": int(b2_axis_degenerate.sum()),
            "b3_event_episodes": int((b3_counts > 0).sum()),
            "b3_event_episode_frac": float((b3_counts > 0).mean()) if b3_counts.size else 0.0,
            "b3_event_total_decision_steps": int(b3_counts.sum()),
            "b3_event_step_frac_over_active_decisions": (
                float(b3_matrix[deg_window].mean()) if (b3_matrix.size and deg_window.any()) else 0.0
            ),
            "ess_pre_adaptation_active": _distribution(
                ess_pre_matrix[deg_window] if (ess_pre_matrix.size and deg_window.any())
                else np.zeros(0, dtype=np.float32),
                int(params.n_samples),
            ),
            "spawn_tilt_agreement_with_pool": (
                {
                    "max_abs_diff_deg": float(
                        np.abs(controller_spawn_tilt - np.asarray(tilt_deg, dtype=np.float64)).max()
                    ),
                    "note": "the controller reads the spawn tilt off the state it is handed; the "
                            "reporting split reads it off the pool's own initial states. The two are "
                            "recorded independently and their agreement is measured, not assumed.",
                }
                if tilt_deg is not None and controller_spawn_tilt.size == len(rows) else None
            ),
        },
    }

    # ---- S3 CHANGE 4: the reporting split (ALL / tilt < 90 / tilt >= 90) --------------------------
    # The headline numbers above are over ALL scored episodes and are never filtered. The blocks below
    # are ADDITIONAL columns that make the born-inverted failure visible as a class property. Present
    # only when the caller supplies the tilt array (the reproducibility gate does not, so the retained
    # artifacts' schema is untouched).
    if tilt_deg is not None:
        tilt = np.asarray(tilt_deg, dtype=np.float64)
        if tilt.shape[0] != len(rows):
            raise ValueError(
                f"tilt_deg has {tilt.shape[0]} entries but {len(rows)} episodes were scored."
            )
        saturation = per_episode["saturation_step_frac"]
        cps_episode = per_episode["cps_episode"]
        common = dict(
            outcome=outcome, cause=cause, cps_episode=cps_episode, saturation=saturation,
            deg_counts=deg_counts, deg_matrix=deg_matrix, deg_window=deg_window,
            b3_counts=b3_counts, b3_matrix=b3_matrix, b2_fired=b2_fired,
        )
        all_block = metric_block(np.ones(len(rows), dtype=bool), band="ALL", **common)
        definition = (
            "spawn tilt of the pool's initial state: cos_tilt = _quat_to_R(x0[:, 3:7])[:, 2, 2], "
            "tilt_deg = degrees(arccos(clip(cos_tilt, -1, 1))) — the established definition at "
            "scripts/analysis/v282_alpha_tilt_screen.py:93-94, read from the pool's own initial "
            "states, not re-derived"
        )
        policy = (
            "born-inverted episodes are NEVER filtered: the headline metrics of this cell are over "
            "all scored episodes and the two bands are additional columns that partition them"
        )
        tilt_summary = {
            "min": float(tilt.min()), "median": float(np.median(tilt)), "max": float(tilt.max()),
        }
        if tilt_split_deg is None:
            # S3 convention, retained so the S3 artifacts stay reproducible: bands at < 90 / >= 90.
            bands: dict[str, Any] = {
                "definition": definition, "split_deg": TILT_SPLIT_DEG, "policy": policy,
                "tilt_summary": tilt_summary, "ALL": all_block,
                "tilt_lt_90": metric_block(tilt < TILT_SPLIT_DEG, band="tilt<90", **common),
                "tilt_ge_90": metric_block(tilt >= TILT_SPLIT_DEG, band="tilt>=90", **common),
            }
        else:
            # charter "v3" convention: the split is theta_ref, the bands are <= theta_ref and
            # > theta_ref, and they use theta_ref itself so the reporting boundary and the boundary
            # B1/B2 act on are one config field rather than two numbers that could drift apart.
            split = float(tilt_split_deg)
            bands = {
                "definition": definition + (
                    "; the manifest of this pool carries NO per-scene spawn-tilt field, so this is the "
                    "documented fallback and it is flagged as a deviation from the charter"
                ),
                "split_deg": split,
                "split_convention": "<= theta_ref and > theta_ref, theta_ref = mppi.recovery."
                                    "theta_ref_deg. NOTE the S3 screen split at 90 deg with < / >=, so "
                                    "the two screens' bands are NOT directly comparable.",
                "policy": policy,
                "tilt_summary": tilt_summary,
                "ALL": all_block,
                "tilt_le_ref": metric_block(tilt <= split, band=f"tilt<={split:g}", **common),
                "tilt_gt_ref": metric_block(tilt > split, band=f"tilt>{split:g}", **common),
            }
        # the ALL block is recomputed from the per-episode rows; it must reproduce the headline
        # numbers the shared scorer produced, and that identity is recorded rather than assumed.
        bands["all_block_matches_headline"] = {
            key: bool(abs(float(all_block[key]) - float(cell[key])) <= 1e-12)
            for key in ("reach", "collision", "oob", "stuck", "timeout", "cps",
                        "saturation_rate", "coll_obstacle", "coll_band_lower", "coll_band_upper")
        }
        cell["bands"] = bands

    # ---- charter "v4": the round-table reporting columns, off by default -------------------------
    # Read off the rollouts this cell already produced, AFTER the scoring above; nothing in the eval
    # path is touched and the pre-v4 artifact schema is unchanged when the flag is off.
    if v4_columns:
        columns = v4_report_columns(
            system, mppi_config, result.trajectories, rows,
            tilt_deg=tilt_deg, tilt_split_deg=tilt_split_deg,
        )
        per_episode["v4_d_min"] = np.asarray(columns["per_episode"]["d_min"], dtype=np.float64)
        per_episode["v4_d_end"] = np.asarray(columns["per_episode"]["d_end"], dtype=np.float64)
        per_episode["v4_n_active_samples"] = np.asarray(
            columns["per_episode"]["n_active_samples"], dtype=np.int64)
        per_episode["v4_n_inside_probe"] = np.asarray(
            columns["per_episode"]["n_inside_probe"], dtype=np.int64)
        per_episode["v4_mean_du"] = np.asarray(
            [np.nan if v is None else v for v in columns["per_episode"]["mean_du"]], dtype=np.float64)
        columns.pop("per_episode")
        cell["v4_columns"] = columns

    # ---- charter "v5" Stage 1: the best-sample endpoint columns ---------------------------------
    if endpoint_probe_radii is not None:
        endpoint_summary, endpoint_matrix = endpoint_probe_summary(
            framework, rows, steps, endpoint_probe_radii
        )
        cell["endpoint_probe"] = endpoint_summary
        per_episode["best_endpoint_dist_steps"] = endpoint_matrix.astype(np.float32)

    # ---- the literature-standard (T, omega_des) cascade: its own record and its own variant token ---
    if rate_cascade is not None:
        cell["cascade"] = framework.controller.cascade_record()
        cell["cell"]["variant"] = "cascaded_rate"
        cell["arm"] = (
            "CASCADED MPPI over (collective thrust T, desired body rates omega_des) — L1-MPPI / "
            "PA-MPPI's interface (privileged, model-based, training-free). A SEPARATE baseline row: "
            "NOT rotor-direct vanilla MPPI and NOT the charter-'v5' a_des cascade; never to be "
            "tabulated beside either without a variant column"
        )
    # ---- charter "v5" Stage 2: the CASCADE record, present only on a cascaded row ----------------
    elif cascade is not None:
        cell["cascade"] = framework.controller.cascade_record()
        cell["cell"]["variant"] = "cascaded"
        cell["arm"] = (
            "CASCADED MPPI (privileged, model-based, training-free) — a SEPARATE baseline row; "
            "NOT vanilla MPPI and never to be tabulated beside one without a variant column"
        )
    else:
        cell["cell"]["variant"] = "vanilla"

    # A caller that must draw THESE rollouts (rather than score a second set) takes them here. Read-only:
    # nothing below consumes `capture`, so passing it cannot change a single number of the cell.
    if capture is not None:
        capture.update({"system": system, "config": config, "result": result, "framework": framework})

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"per_episode__{label}.npz", **per_episode)
        (out_dir / f"cell__{label}.json").write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")
    return cell


def main() -> int:
    mppi_config = load_mppi_config()
    screen = mppi_config["screen"]
    sampling = mppi_config["sampling"]

    parser = argparse.ArgumentParser(description="Score one MPPI cell on an evaluation pool.")
    parser.add_argument("--n-samples", "-N", type=int, default=int(sampling["n_samples"]))
    parser.add_argument("--horizon", "-H", type=int, default=int(sampling["horizon"]))
    parser.add_argument("--lam", "--lambda", dest="lam", type=float, default=float(sampling["lam"]))
    parser.add_argument("--c-crash", type=float, default=float(mppi_config["cost"]["c_crash"]))
    parser.add_argument("--sigma", type=float, default=float(sampling["sigma"]),
                        help="exploration std in PER-ROTOR-EQUIVALENT newtons; the per-channel wrench std "
                             "is sigma * ||mixer_row_j||")
    parser.add_argument("--space", type=str, default=str(sampling["space"]), choices=["wrench", "rotor"])
    parser.add_argument("--noise", type=str, default=str(sampling["noise"]), choices=["ou", "iid"])
    parser.add_argument("--lam-mode", type=str, default=str(sampling["lam_mode"]),
                        choices=["relative", "absolute"])
    parser.add_argument("--center", type=str, default=str(sampling["center"]),
                        choices=["hover", "none"],
                        help="S3 sampling centre: 'hover' makes the trim an explicit anchor "
                             "(u = u_hover + u_plan + eps); 'none' is the legacy S1/S2 parameterisation")
    parser.add_argument("--control-hold", type=int, default=int(sampling["control_hold"]),
                        help="S3 control hold m: physical steps each DECISION entry is applied for; "
                             "the rollout spans H*m steps and re-planning happens every m steps")
    parser.add_argument("--terminal", type=str, default=str(mppi_config["cost"]["terminal"]),
                        choices=["settling", "distance"],
                        help="S3 terminal cost form; 'distance' is the legacy S1/S2 terminal")
    recovery_cfg = mppi_config["recovery"]
    for name, block in (("b1", "b1"), ("b2", "b2"), ("b3", "b3")):
        parser.add_argument(
            f"--{name}", dest=name, action="store_true", default=bool(recovery_cfg[block]["enabled"]),
            help=f"switch charter-'v3' component {name.upper()} ON (config default: "
                 f"{bool(recovery_cfg[block]['enabled'])})",
        )
        parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    v4_cfg = mppi_config["goal_v4"]
    for name in ("g1", "g2", "g3", "g4"):
        parser.add_argument(
            f"--{name}", dest=name, action="store_true", default=bool(v4_cfg[name]["enabled"]),
            help=f"switch charter-'v4' component {name.upper()} ON (config default: "
                 f"{bool(v4_cfg[name]['enabled'])})",
        )
        parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.add_argument("--w-p", type=float, default=float(v4_cfg["g1"]["w_p"]),
                        help="G1 progress weight w_p (config default shown); G2 multiplies it by "
                             f"{float(v4_cfg['g2']['w_p_scale'])!r}")
    parser.add_argument("--w-omega", type=float, default=float(v4_cfg["g4"]["w_omega"]),
                        help="G4 always-on angular-rate weight w_omega (config default shown)")
    parser.add_argument(
        "--tilt-split", type=float, default=None,
        help="reporting split in degrees, charter-'v3' convention (<= split / > split). Omit to keep "
             "the S3 convention (< 90 / >= 90) so the retained S3 artifacts stay reproducible.",
    )
    parser.add_argument("--seed", type=int, default=int(sampling["seed"]))
    parser.add_argument("--sample-chunk", type=int, default=int(sampling["sample_chunk"]),
                        help="cap on the rollout batch along the sample axis; 0 = all N at once. "
                             "Exactly equivalent (samples are independent); trades wall time for peak VRAM.")
    parser.add_argument("--pool", type=str, default=str(REPO / screen["pool"]))
    parser.add_argument("--n-scenes", type=int, default=int(screen["n_scenes"]),
                        help="scenes scored, taken in pool order; -1 = the whole pool")
    parser.add_argument("--ebs", type=int, default=int(screen["ebs"]))
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override eval.max_steps (default: the eval-cell value, 200)")
    parser.add_argument("--out-dir", type=str, default=str(REPO / "data/runs/v2.8.4/mppi_screen"))
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    label = args.label or (
        f"sig{args.sigma:g}_lam{args.lam:g}_H{args.horizon}_N{args.n_samples}_C{args.c_crash:g}"
    )
    cell = run_cell(
        pool_path=Path(args.pool),
        n_scenes=None if args.n_scenes < 0 else int(args.n_scenes),
        ebs=int(args.ebs),
        n_samples=int(args.n_samples),
        horizon=int(args.horizon),
        lam=float(args.lam),
        c_crash=float(args.c_crash),
        sigma=float(args.sigma),
        seed=int(args.seed),
        sample_chunk=int(args.sample_chunk),
        space=str(args.space),
        noise=str(args.noise),
        lam_mode=str(args.lam_mode),
        center=str(args.center),
        control_hold=int(args.control_hold),
        terminal=str(args.terminal),
        b1=bool(args.b1), b2=bool(args.b2), b3=bool(args.b3),
        g1=bool(args.g1), g2=bool(args.g2), g3=bool(args.g3), g4=bool(args.g4),
        w_p=float(args.w_p), w_omega=float(args.w_omega),
        tilt_split_deg=args.tilt_split,
        label=label,
        out_dir=Path(args.out_dir),
        device=device,
        dtype=dtype,
        max_steps=args.max_steps,
        mppi_config=mppi_config,
    )
    print(
        f"[{label}] cps {cell['cps']:.4f} reach {cell['reach']:.4f} coll {cell['collision']:.4f} "
        f"(obs {cell['coll_obstacle']:.4f} lo {cell['coll_band_lower']:.4f} up {cell['coll_band_upper']:.4f}) "
        f"to {cell['timeout']:.4f} stuck {cell['stuck']:.4f} oob {cell['oob']:.4f} "
        f"degen_eps {cell['degenerate']['episode_frac_with_any']:.4f} "
        f"ESS p50 {cell['ess']['active'].get('p50', float('nan')):.2f} ({cell['wall_s']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
