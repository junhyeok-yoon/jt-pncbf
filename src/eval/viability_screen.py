"""v2.9.3 — the vertical-channel viability screen for quadrotor_3d evaluation pools.

WHAT. A controller-independent sufficient condition for doom on an initial state: floor, ceiling and
horizontal legs, every uncertain term resolved in favour of survival. A flagged scene is one no
admissible input could have saved, so rejecting it removes a scene the eval could never have scored
as anything but a collision.

WHY IT EXISTS. `03_train` 1.3's unavoidable-collision rejection covers the HORIZONTAL channel only,
while `01_env` 1.6 makes the floor and ceiling physical collision surfaces on this system.
`docs/versions/v2.9.3/doom_certificate.md` measured the consequence on the registered pool. This
module carries the same predicate to pool construction.

THE ARITHMETIC IS NOT REIMPLEMENTED. `constants()` and `floor_doomed()` are imported from
`data/runs/v2.9.3/doom_certificate/make_doom_certificate.py` and called, so the floor leg -- the only
leg with a quadrature and the only one whose soundness argument is delicate -- is literally the code
the certificate was scored with. The ceiling and horizontal legs are two closed forms that live
inline in that script's `main()` rather than in functions; they are written out here and then
PROVED bit-identical to the certificate's own flags on the registered pool by
`assert_matches_certificate()`, which is run by the pool builder before any scene is drawn.

EVAL-ONLY BY CONSTRUCTION. `03_train` 1.2 requires training to experience the region the value is
meant to represent, so no training scene path may reach this predicate. Enforcement is structural:
this module lives under `src/eval/`, is imported by nothing in `src/envs/` or `src/frameworks/`, and
is called only from the pool builder. `scripts/analysis/v293_build_fullvia.py` asserts that at build
time by grepping the training path for an import of this module.

NOT A VIABILITY ORACLE. The condition is SUFFICIENT for doom, never necessary: an unflagged scene is
not thereby survivable. Rejecting flagged scenes removes certainly-lost scenes; it does not make the
remainder winnable.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CERT_DIR = REPO / "data/runs/v2.9.3/doom_certificate"
CERT_SRC = CERT_DIR / "make_doom_certificate.py"
MEASURED_RATE = CERT_DIR / "measured_rate.json"
CERT_FLAGS = CERT_DIR / "doom_flags.npz"


def _certificate_module():
    """The scored certificate, loaded from its own artifact so the arithmetic is the same code."""
    if not CERT_SRC.exists():
        raise FileNotFoundError(f"the doom certificate builder is missing at {CERT_SRC}")
    spec = importlib.util.spec_from_file_location("_doom_cert", CERT_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def omega_reject() -> tuple[float, str]:
    """The rejection setting: the realised maximum tilt rate, READ from its persisted artifact.

    Returned with its provenance string. It is an empirical maximum over one sample and is NOT a
    bound -- see doom_certificate.md section 9.2. It is used here as a REJECTION setting, where a
    looser value rejects fewer scenes and is the conservative direction.
    """
    if not MEASURED_RATE.exists():
        raise FileNotFoundError(f"measured_rate.json is missing at {MEASURED_RATE}")
    d = json.loads(MEASURED_RATE.read_text())
    return float(d["max_realised_dtheta_dt"]), (
        f"{MEASURED_RATE.relative_to(REPO)} -> max_realised_dtheta_dt, over "
        f"{d['n_episodes']} episodes / {d['n_states']} states of {d['source']}")


def omega_sound(C: dict) -> tuple[float, str]:
    """sqrt2 * (omega_max + alpha_max * dt): the bound the clamp's own discipline supports."""
    v = np.sqrt(2.0) * (C["omega_clamp_deployed"] + C["alpha_max"] * C["dt"])
    return float(v), ("sqrt2 * (omega_max + alpha_max * dt), computed from the constants of "
                      "doom_certificate.md section 1; not typed")


def initial_state_arrays(scenes) -> dict[str, np.ndarray]:
    """p, v, q, omega for a list of Scene objects, in the layout the certificate expects."""
    return dict(
        p=np.array([np.asarray(s.start, float) for s in scenes]),
        v=np.array([np.asarray(s.initial_velocity, float) for s in scenes]),
        q=np.array([np.asarray(s.initial_attitude_quat, float) for s in scenes]),
        om=np.array([np.asarray(s.initial_omega_vec, float) for s in scenes]),
    )


def flags(scenes, C: dict, omega: float, cert=None) -> dict[str, np.ndarray]:
    """The three legs and their union, for a list of scenes at one omega setting."""
    import torch
    from src.envs.quadrotor_3d import _quat_to_R
    cert = cert or _certificate_module()
    a = initial_state_arrays(scenes)
    p, v, q, om = a["p"], a["v"], a["q"], a["om"]
    theta0 = np.arccos(np.clip(_quat_to_R(torch.from_numpy(q)).numpy()[:, 2, 2], -1.0, 1.0))
    omega0_norm = np.linalg.norm(om, axis=1)
    vz0, pz = v[:, 2], p[:, 2]
    psi0 = pz + C["band_L"]

    # floor: the certificate's own quadrature, called not copied
    psi_min, _t = cert.floor_doomed(psi0, vz0, theta0, omega0_norm, C, omega, C["dt"] / 50.0)
    floor = psi_min <= 0.0

    # ceiling: closed form, doom_certificate.md section 1
    ceiling = (vz0 > 0.0) & ((C["band_L"] - pz) < vz0 ** 2 / (2.0 * C["g"]))

    # horizontal: closed form, per active cylinder
    horizontal = np.zeros(len(scenes), bool)
    for i, s in enumerate(scenes):
        act = np.asarray(s.obstacle_active, bool)
        if not act.any():
            continue
        cen = np.asarray(s.obstacle_centers, float)[act][:, :2]
        rad = np.asarray(s.obstacle_radii, float)[act]
        rel = cen - p[i, :2]
        d = np.linalg.norm(rel, axis=1)
        d0 = d - rad
        u = rel / np.maximum(d, 1e-12)[:, None]
        v_in = np.maximum(np.einsum("kj,j->k", u, v[i, :2]), 0.0)
        horizontal[i] = bool((d0 < v_in ** 2 / (2.0 * C["g"] * C["TWR"])).any())

    return {"floor": floor, "ceiling": ceiling, "horizontal": horizontal,
            "any": floor | ceiling | horizontal, "psi_0": psi0, "theta_0": theta0, "v_z0": vz0}


def assert_matches_certificate(scenes, C: dict, cert=None) -> dict[str, Any]:
    """Prove this module reproduces the SCORED certificate bit-for-bit before it is trusted.

    Compares all four flag arrays against `doom_flags.npz` on the registered pool at the two settings
    that artifact carries for the values used here. Raises on any difference; a screen that does not
    reproduce the certificate is not the certificate.
    """
    if not CERT_FLAGS.exists():
        raise FileNotFoundError(f"the certificate's flags are missing at {CERT_FLAGS}")
    Z = np.load(CERT_FLAGS)
    cert = cert or _certificate_module()
    report = {}
    for tag, om in (("omega_measured", omega_reject()[0]), ("omega_sound", omega_sound(C)[0])):
        f = flags(scenes, C, om, cert=cert)
        for leg in ("floor", "ceiling", "horizontal", "any"):
            ref = Z[f"{tag}_{leg}"]
            if not np.array_equal(f[leg], ref):
                raise AssertionError(
                    f"viability_screen disagrees with the scored certificate at {tag}/{leg}: "
                    f"{int((f[leg] != ref).sum())} of {ref.size} scenes differ")
        report[tag] = {"omega": om, "n_flagged": int(f["any"].sum()),
                       "matches_certificate": True, "n_compared": int(Z[f"{tag}_any"].size)}
    return report
