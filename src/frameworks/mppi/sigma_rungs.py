"""v2.8.4 — the sampling distribution and the budget, on the obstacle pool.

SIBLING MODULE, stated as the scope choice: this file imports `reach2_rungs` and reuses its cell
runner, its three-reach scoring and its table assembly rather than extending `Reach2Flags`. The
reason is structural, not stylistic:

  AXIS A (sigma) and AXIS B (lambda) NEED NO NEW FLAG AT ALL. `sigma`, `lam` and `n_samples` are
  shipped `MPPIParams` fields that `stages_v5.base_kwargs` already carries and
  `evaluate_mppi.run_cell` already forwards (`stages_v5.py:186-204`). Setting them changes no code
  path in any module — exactly the argument `Reach2Flags`'s own docstring makes for `lam_mode` and
  `c_crash` being absent from it. Adding flags for them would have manufactured a code path where
  the shipped one already exists.

  L1 / L5 likewise need no new implementation: they exist in `lit_rungs.LitFlags` and are bound by
  the same `RateCascadeController` monkeypatch `reach2_rungs.bound_controller` uses. They are reused,
  not reimplemented.

`cascade_rate.py`, `cost.py`, `mppi_controller.py`, `lit_rungs.py` and `reach2_rungs.py` are NOT
modified. Artifacts: `data/runs/v2.8.4/mppi_sigma/**`.

NO SELECTION. No cell is selected, ranked, promoted or registered; no table is sorted by result; no
threshold is applied and no `cps` claim is made. Rows are in CONFIG ORDER.

`K_seed` IS STRUCTURALLY ZERO here as in `reach2_rungs`: this module contains no seeded / geometric
rollout code and does not import `reach_rungs.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from src.frameworks.mppi import cascade_rate as _cascade_rate_module
from src.frameworks.mppi import reach2_rungs as R2
from src.frameworks.mppi.evaluate_mppi import REPO, load_mppi_config, run_cell, effective_config
from src.frameworks.mppi.lit_rungs import LitCascadeController, LitFlags
from src.frameworks.mppi.screen_cascaded import cascade_kwargs
from src.frameworks.mppi.stages_v5 import base_kwargs
from src.frameworks.mppi.relaxed_v5 import load_rollouts, save_rollouts, score_two_terminals

OUT_DIR = REPO / "data/runs/v2.8.4/mppi_sigma"

# The table this work reports. Every column the dispatch names, plus the realized-sigma block.
TABLE_COLUMNS = (
    "variant", "cell_id", "sigma_mult", "lam", "lam_mode", "N", "H",
    "reach_deployed", "reach_relaxed", "relaxed_only_n",
    "d_min_p50", "d_min_min", "ess_p50", "ess_over_n",
    "coll_obstacle", "coll_band_lower", "coll_band_upper",
    "oob", "stuck", "timeout", "wall_ms_per_step", "preclip_out_of_box_share",
    "sig_T", "sig_wx", "sig_wy", "sig_wz",
)

PRIVILEGE = (
    "PRIVILEGED: the planner reads the full 13-D state and the obstacle field directly; the per-rotor "
    "box is NOT handled by the planner (the inner loop allocates and clips after the fact). "
    "`infeasibility` and `mean_proj_mag` are STRUCTURALLY INAPPLICABLE — this baseline runs no "
    "projection and forms no half-space, so there is no quantity to report."
)

MINARIK_TABLE_II = (0.60, 0.15, 0.15, 0.05)   # diag, transcribed in mppi_reach2.md Milestone 0


# =================================================================================================
# realized sigma, MEASURED — the mppi_conform ITEM 3 code, reused
# =================================================================================================
def measure_realized_sigma(controller: Any, *, batch: int = 1, seed: int = 42) -> dict[str, Any]:
    """Per-channel realized marginal std and lag-k autocorrelation of the ACTUAL draw.

    This is `mppi_conform` ITEM 3's measurement applied to whatever controller is handed in: the
    generator is snapshotted and restored (`parity_rate.our_noise`'s discipline) so measuring does
    not consume the stream the screen will use."""
    state = controller.generator.get_state()
    controller.generator.manual_seed(int(seed))
    eps = controller._draw_noise(batch)[0]                     # [N,H,m]
    controller.generator.set_state(state)
    n, horizon, m = eps.shape
    names = ("T_collective", "omega_x", "omega_y", "omega_z")
    units = ("N", "rad/s", "rad/s", "rad/s")
    cfg = [float(v) for v in controller.sigma_channel.tolist()]
    alpha = float(controller.params.ou_alpha)
    out = []
    for j in range(m):
        col = eps[:, :, j].to(torch.float64)
        realized = float(col.reshape(-1).std(unbiased=True))
        lags = {}
        for k in range(1, 11):
            if k >= horizon:
                lags[f"lag_{k}"] = None
                continue
            a = col[:, :-k].reshape(-1); b = col[:, k:].reshape(-1)
            a = a - a.mean(); b = b - b.mean()
            den = float(torch.linalg.norm(a) * torch.linalg.norm(b))
            lags[f"lag_{k}"] = float((a * b).sum() / den) if den > 0 else None
        errs = [abs(lags[f"lag_{k}"] - alpha ** k) for k in range(1, 11)
                if lags[f"lag_{k}"] is not None]
        out.append({
            "channel": names[j], "unit": units[j],
            "configured_sigma": cfg[j], "realized_marginal_std": realized,
            "realized_over_configured": realized / cfg[j] if cfg[j] else None,
            "minarik_tableII": MINARIK_TABLE_II[j],
            "realized_over_minarik": realized / MINARIK_TABLE_II[j],
            "autocorr": lags, "ou_alpha": alpha,
            "max_abs_autocorr_error_vs_alpha_pow_k": max(errs) if errs else None,
        })
    return {"n_samples_drawn": n, "horizon": horizon, "per_channel": out}


def _build_probe_controller(mppi_config: dict, *, sigma: float, lam: float, lam_mode: str,
                            n_samples: int, horizon: int,
                            device: torch.device, dtype: torch.dtype):
    """A controller built exactly as the screen builds one, used ONLY to measure the draw and to probe
    ESS. Mirrors `lit_rungs.build_lit_controller`'s construction so no shipped file is edited."""
    from src.frameworks.jt_pncbf.train import make_system
    from src.frameworks.mppi.mppi_controller import MPPIParams
    from src.frameworks.mppi.cost import CostParams
    from src.frameworks.mppi.recovery import RecoveryParams
    from src.frameworks.mppi.cascade_rate import RateCascadeController

    config = effective_config(mppi_config)
    kw = base_kwargs(mppi_config)
    system = make_system(config)
    params = MPPIParams.from_config(
        mppi_config, n_samples=int(n_samples), horizon=int(horizon), lam=float(lam),
        sigma=float(sigma), seed=int(mppi_config["cascaded"]["scale"]["seed"]), sample_chunk=0,
        space=kw["space"], noise=kw["noise"], lam_mode=str(lam_mode),
        center=kw["center"], control_hold=kw["control_hold"],
    )
    cost_params = CostParams.from_config(
        mppi_config, config["env"], c_crash=kw["c_crash"], terminal=kw["terminal"],
        g1=kw["g1"], g2=kw["g2"], g3=kw["g3"], g4=kw["g4"],
    )
    recovery = RecoveryParams.from_config(mppi_config, b1=kw["b1"], b2=kw["b2"], b3=kw["b3"])
    controller = RateCascadeController(
        system, config, params, cost_params, device=device, dtype=dtype, recovery=recovery,
        cascade=cascade_kwargs(
            mppi_config, float(mppi_config["cascaded"]["smoke"]["rate_gain_factor"])
        ),
    )
    return system, controller, config


# =================================================================================================
# L1 / L5 binding — reuse lit_rungs, do not reimplement
# =================================================================================================
@contextlib.contextmanager
def bound_lit(flags: LitFlags) -> Iterator[None]:
    """Bind `LitCascadeController` with `flags` in place of `RateCascadeController`, the same
    monkeypatch `reach2_rungs.bound_controller` uses, so `run_cell` reaches the subclass unedited."""
    original = _cascade_rate_module.RateCascadeController

    class _BoundLit(LitCascadeController):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, flags=flags, reference_cost=None, **k)

    _cascade_rate_module.RateCascadeController = _BoundLit
    try:
        yield
    finally:
        _cascade_rate_module.RateCascadeController = original


# =================================================================================================
# one cell
# =================================================================================================
def run_sigma_cell(
    cell_id: str,
    mppi_config: dict[str, Any],
    *,
    sigma_mult: float,
    lam: float | None,
    lam_mode: str,
    n_samples: int,
    horizon: int,
    tilt: np.ndarray,
    theta_ref: float,
    device: torch.device,
    dtype: torch.dtype,
    lit_flags: LitFlags | None = None,
) -> dict[str, Any]:
    """Score ONE cell. `sigma_mult`, `lam` and `n_samples` are applied on the SHIPPED `MPPIParams`
    fields (`stages_v5.base_kwargs`), so with `lit_flags is None` the shipped `RateCascadeController`
    runs with no patch of any kind."""
    scale = mppi_config["cascaded"]["scale"]
    kw = base_kwargs(mppi_config)
    base_sigma = float(kw["sigma"])
    kw["sigma"] = base_sigma * float(sigma_mult)
    kw["lam"] = float(kw["lam"]) if lam is None else float(lam)
    kw["lam_mode"] = str(lam_mode)
    kw["horizon"] = int(horizon)

    # realized sigma MEASURED on a controller built at this cell's own parameters, before the screen
    _, probe, _ = _build_probe_controller(
        mppi_config, sigma=kw["sigma"], lam=kw["lam"], lam_mode=lam_mode,
        n_samples=int(n_samples), horizon=int(horizon), device=device, dtype=dtype,
    )
    sig = measure_realized_sigma(probe, seed=int(scale["seed"]))
    del probe

    capture: dict[str, Any] = {}
    t0 = time.time()
    ctx = contextlib.nullcontext() if lit_flags is None else bound_lit(lit_flags)
    with ctx:
        cell = run_cell(
            pool_path=REPO / mppi_config["screen"]["pool"],
            n_scenes=int(scale["n_scenes"]), ebs=int(scale["ebs"]), seed=int(scale["seed"]),
            n_samples=int(n_samples), sample_chunk=0,
            tilt_deg=tilt, tilt_split_deg=theta_ref, label=cell_id, out_dir=OUT_DIR,
            device=device, dtype=dtype, mppi_config=mppi_config, v4_columns=True,
            rate_cascade=cascade_kwargs(
                mppi_config, float(mppi_config["cascaded"]["smoke"]["rate_gain_factor"])
            ),
            capture=capture, **kw,
        )
    wall = time.time() - t0

    roll = OUT_DIR / f"rollouts__{cell_id}.npz"
    save_rollouts(roll, capture["system"], capture["result"].trajectories,
                  capture["result"].episode_rows, tilt_deg=tilt)
    two = score_two_terminals(capture["system"], capture["config"], load_rollouts(roll),
                              tilt_split_deg=theta_ref)
    row = R2.assemble_row(cell_id, cell, two["ALL"], wall_s=wall,
                          flags_record=None if lit_flags is None else lit_flags.record(),
                          rollouts=roll)
    row["sigma_mult"] = float(sigma_mult)
    row["lam"] = kw["lam"]
    row["sigma_absolute"] = kw["sigma"]
    per = sig["per_channel"]
    for key, ch in zip(("sig_T", "sig_wx", "sig_wy", "sig_wz"), per):
        row[key] = ch["realized_marginal_std"]
    row["_realized_sigma"] = sig
    row["_privilege"] = PRIVILEGE
    row["_infeasibility"] = "STRUCTURALLY INAPPLICABLE"
    row["_mean_proj_mag"] = "STRUCTURALLY INAPPLICABLE"
    return row


def markdown(rows: list[dict[str, Any]], columns: tuple[str, ...] = TABLE_COLUMNS) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(R2._fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])
