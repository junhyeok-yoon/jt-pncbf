from __future__ import annotations

from typing import Any, Mapping

import torch


Tensor = torch.Tensor

GEOM_FORMS = ("clip", "exp")


def hazard_geom(config: Any) -> tuple[str, float | None]:
    """v2.8.5 — read the geometric-term selector off the config. DEFAULT OFF.

    `hazard.geom_form` in {'clip', 'exp'}; absent or 'clip' -> the deployed clipped ramp and the
    arithmetic of `signed_h` is bit-identical to v2.8.4. `hazard.ell` is the exponential decay
    length and is REQUIRED (and must be positive) when geom_form == 'exp'; it is ignored by the
    clip path, which continues to read `env.h_scale`. Checkpoints written before v2.8.5 carry no
    `hazard` block at all, so the absent case must stay bit-parity — it is the parity gate.
    """
    hz: Mapping[str, Any] = {}
    if config is not None:
        try:
            hz = config.get("hazard") or {}                     # type: ignore[union-attr]
        except AttributeError:
            hz = {}
    geom_form = str(hz.get("geom_form", "clip") or "clip")
    if geom_form not in GEOM_FORMS:
        raise ValueError(f"hazard.geom_form must be one of {GEOM_FORMS}, got {geom_form!r}.")
    ell_raw = hz.get("ell", None)
    if geom_form == "exp":
        if ell_raw is None:
            raise ValueError("hazard.geom_form == 'exp' requires a positive hazard.ell.")
        ell = float(ell_raw)
        if ell <= 0.0:
            raise ValueError(f"hazard.ell must be positive, got {ell}.")
        return geom_form, ell
    return geom_form, (float(ell_raw) if ell_raw is not None else None)


def signed_h(p: Tensor, scene: Any, h_scale: float, *,
             geom_form: str = "clip", ell: float | None = None) -> Tensor:
    """The position-only avoid generator phi(p, o).

    geom_form 'clip' (DEFAULT, the deployed v2.8.4 term):
        phi = 1 - 2*clamp(clearance/h_scale, 0, 1)        -- zero at clearance h_scale/2, flat -1
                                                             beyond h_scale.
    geom_form 'exp'  (v2.8.5, default off):
        phi = clamp(2*exp(-clearance/ell) - 1, max=+1)    -- zero at clearance ell*ln2, never
                                                             exactly -1 in exact arithmetic.
    The +1 saturation inside the cylinder is retained by both forms, so the label clamp
    (jt_pncbf/losses.py) and the value ceiling (common/value_net.py) see the same upper endpoint.
    Only the horizontal (cylinder) geometric term is selected here; the vertical band branches of
    `quadrotor_barrier.value_target_barrier` never call this function and read neither h_scale nor
    ell. The obstacle selection is unchanged: both forms are strictly decreasing in clearance, so
    the max over obstacles picks the same minimum-clearance active cylinder.
    """
    if h_scale <= 0.0:
        raise ValueError(f"h_scale must be positive, got {h_scale}.")
    if geom_form not in GEOM_FORMS:
        raise ValueError(f"geom_form must be one of {GEOM_FORMS}, got {geom_form!r}.")
    if geom_form == "exp" and not (ell is not None and float(ell) > 0.0):
        raise ValueError(f"geom_form 'exp' requires a positive ell, got {ell}.")

    centers = torch.as_tensor(scene.obstacle_centers, dtype=p.dtype, device=p.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=p.dtype, device=p.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=p.device)

    # Obstacles live in the first centers.shape[-1] position coords (xy footprint for infinite vertical
    # cylinders; a no-op when position dim == center dim, e.g. DI/unicycle/planar).
    p = p[..., : centers.shape[-1]]
    rel = p.unsqueeze(-2) - centers
    distance = torch.linalg.norm(rel, dim=-1)
    clearance = distance - radii
    if geom_form == "clip":
        h_all = 1.0 - 2.0 * torch.clamp(clearance / h_scale, min=0.0, max=1.0)
    else:
        # clamp(clearance, min=0) inside the exponent is overflow-safety only: it never binds on the
        # branch `torch.where` keeps, and it makes the d < 0 branch a finite constant so no inf/nan
        # can reach the max. Gradient: (2/ell)*exp(-d/ell) for d >= 0 (the boundary d = 0 carries
        # 2/ell, as the +1 clamp does), 0 for d < 0.
        ex = torch.exp(-torch.clamp(clearance, min=0.0) / float(ell))
        h_all = torch.where(clearance >= 0.0, 2.0 * ex - 1.0, torch.ones_like(ex))
    h_all = torch.where(active, h_all, torch.full_like(h_all, -1.0))
    h_value = torch.max(h_all, dim=-1).values
    return torch.clamp(h_value, min=-1.0, max=1.0)
