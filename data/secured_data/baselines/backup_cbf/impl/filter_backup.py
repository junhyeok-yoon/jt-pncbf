"""Backup-CBF switching filter (v2.8.3 B-axis, ADDITIVE, flag-gated OFF by default).

STATUS LINE (must accompany every table built from this module):

    hand-designed policy + online rollout certificate (implicit/backup CBF)

This is NOT an analytic closed-form CBF. There is no closed-form h, no Lie derivative, no QP, and no
gradient of the certificate. The certificate h_b is DEFINED by an online RK4 rollout of a fixed
hand-designed backup policy pi_b, and enforcement is a one-step-lookahead SWITCH between the nominal
action and pi_b. The analytic-CBF propositions (prop:relh / cor:degen-pos / prop:box-feas) do NOT apply.

--------------------------------------------------------------------------------------------------
BACKUP POLICY pi_b
--------------------------------------------------------------------------------------------------
pi_b is the repo's EXISTING cascaded hover PD, `System.lqr_action` (src/envs/quadrotor_3d.py:173-189:
outer position loop -> desired world force -> projection onto the CURRENT thrust axis b3 -> attitude
error b3 x b3_des mapped to body -> inner attitude PD -> mixer inverse -> per-rotor box clip). It is
reused verbatim; nothing is re-implemented.

The one thing this module supplies is pi_b's SETPOINT. `System.lqr_action(x, goal)` takes the setpoint
as an argument; the deployed nominal passes the SCENE GOAL. A backup policy must instead bring the
vehicle to REST, so the default here (`backup.setpoint = "hold"`) passes the vehicle's OWN position at
the instant the backup engages, frozen for the whole backup rollout:

    pi_b(x) := system.lqr_action(x, p_hold),   p_hold = system.position(x_0) of the backup rollout.

At t = 0 of its own rollout p_hold = position(x), so the DEPLOYED backup action is exactly the first
action of the certified backup trajectory from x -- the standard backup-CBF consistency requirement.
`backup.setpoint = "goal"` selects the goal-tracking variant (pi_b == the deployed nominal PD) and is
retained only so the alternative reading of the specification is reachable and scoreable.

Gains are READ from config, never invented: config["lqr"]["quadrotor_3d"] = {kp_pos 1.0, kd_pos 2.0,
kp_att 40.0, kd_att 8.0} at src/configs/exp_config.yaml:77-81. The per-rotor box [0, 4.905] is
config["env"]["bounds"]["quadrotor_3d"] = {f_rotor_min 0.0, f_rotor_max 4.905} at
src/configs/exp_config.yaml:32-34, applied by lqr_action's own `_clamp_action`
(src/envs/quadrotor_3d.py:189,200-202).

--------------------------------------------------------------------------------------------------
v2.8.3 BASELINE-v2 pi_b -- `filter.backup.form = "brake"`  (ADDITIVE; absent key = B1 behaviour)
--------------------------------------------------------------------------------------------------
B1's pi_b carried a POSITION term (`-kp_pos (p - p_hold)`), so it spent thrust returning to an
altitude instead of ARRESTING A FALL. The standard backup form is LEVEL ATTITUDE AND ZERO VELOCITY
with NO POSITION TERM:

    a_des = -k_d v + g e3 ;  f_thr = m (a_des . b3) clipped at 0 ;  b3_des = a_des / ||a_des|| ;
    tau   = J (kp_att e_att_body - kd_att omega) ;  then the X-mixer and the per-rotor box.

`brake_action` obtains exactly this by WRAPPING `System.lqr_action` with kp_pos := 0 (see its
docstring). It takes no `scene` argument: pi_b is scene-blind, the CERTIFICATE is what is scene-aware.

NOTE (recorded divergence from the task specification): the specification places the PD gains under
`env.quadrotor_3d`. On disk they live under the TOP-LEVEL `lqr:` block, `lqr.quadrotor_3d`
(exp_config.yaml:77-81); `env.quadrotor_3d` (exp_config.yaml:45-60) carries the plant constants
(mass/J/arm/c_gain/IC ranges) and no gains. The values quoted in the specification (kp_att 40.0,
kd_att 8.0) match `lqr.quadrotor_3d` exactly, so only the key path differed.

--------------------------------------------------------------------------------------------------
CLEARANCE MARGIN m  (POSITION ONLY -- deliberately NOT signed_h / h_star)
--------------------------------------------------------------------------------------------------
    m(x) = min( min_{i active} (||p_xy - c_i|| - r_i),  p_z + band_z,  band_z - p_z )

sign > 0 = safe. This is the RAW geometric surface distance in metres. It is NOT
`src/common/signed_h.py::signed_h`, which returns the normalized, h_scale-saturated, SIGN-FLIPPED
1 - 2*clamp(clearance/h_scale, 0, 1) in [-1, 1]; and it is NOT `h_star`, whose velocity augmentation
(`approach_barrier`, quadrotor_3d.py:154-170, and the band's c_z v_z term) belongs to the LABEL, not
to a clearance test. Mixing either into a clearance test would make eps uninterpretable.

m is sign-consistent WITH THE SCORED COLLISION PREDICATE by construction:
  - obstacles: `_collided_exact` (src/common/outcomes.py:98-118) fires iff `distance < radii` on an
    active obstacle, i.e. iff the obstacle term of m is < 0;
  - band: `step_outcomes` (src/common/outcomes.py:54-59) fires iff p_z <= -band_z or p_z >= +band_z,
    i.e. iff the corresponding band term of m is <= 0.
band_z is read from config["env"]["band_collision_limit"] (the SAME key the scorer reads); 0.0 turns
the band terms off exactly as it turns the band predicate off.

--------------------------------------------------------------------------------------------------
CERTIFICATE h_b  (online RK4 rollout at env.dt)
--------------------------------------------------------------------------------------------------
    h_b(x) = min( min_{t=0..T_b} m(x_t^{pi_b}),  m_term(x_{T_b}^{pi_b}) )
    m_term(y) = m(y) - c_v * ||v(y)||,     c_v = 0.3 s   (rest penalty: the terminal must be stopped)

The rollout uses the SAME `rk4_step` (incl. `system.wrap_state`) as every deployed path, at
dt = config["env"]["dt"]. Entirely under `torch.no_grad()`: the certificate is never differentiated.

Structural precedent: `src/common/brake_rollout.py::brake_h_rollout` (line 41) -- same batched
"roll a fixed analytic policy T_b steps and reduce the per-step barrier" machinery. It is NOT reused
as a function because it returns the sign-flipped saturated `signed_h` sequence for the VALUE TARGET
(and takes DI-specific u_max/eps_v brake arguments), whereas this needs the raw geometric margin and
a rest-penalized terminal. The loop shape below is deliberately its mirror image.

--------------------------------------------------------------------------------------------------
ENFORCEMENT -- SWITCHING, one-step lookahead, NO QP, NO gradient of h_b
--------------------------------------------------------------------------------------------------
    u_dep(x) = u_nom          if h_b( f_dt(x, u_nom) ) >= eps
             = pi_b(x)        otherwise

f_dt is one `rk4_step` at env.dt with u_nom held. `torch.where` on the branch mask means
d u_dep / d u_nom = 0 exactly on the pi_b branch (the switching analogue of prop:sel(a)); the mask
itself is computed under no_grad and is therefore a constant in backward, so no gradient of h_b ever
exists.

Returned tuple shape is identical to `HardNetFilter.__call__`: (u_safe, infeasible), or
(u_safe, infeasible, u_cbf_raw, singular) when `return_deficit_aux=True`.

`infeasible` semantics: the switching form has NO QP, so "empty feasible set" does not exist. The
honest analogue is "the backup itself is not certified at this state": a row is flagged iff it took
the pi_b branch AND h_b(x) < eps (one extra rollout, restricted to the branch rows only). When
`backup.certify_backup = false` the flag is all-False and the extra rollout is skipped.
`last_empty` / `last_singular` are published as all-False so the shared eval readout
(src/eval/rollout.py:186,214-216) reports 0 rather than aliasing something meaningless.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

import torch

from src.common.rk4 import rk4_step
from src.common.system import System


Tensor = torch.Tensor

_FAR = 1.0e6          # finite stand-in for "no active obstacle" (avoids inf arithmetic in the min)
_C_V_DEFAULT = 0.3    # s -- terminal rest penalty coefficient (specification constant)


@dataclass(frozen=True)
class _BackupParams:
    T_b: int
    eps: float
    dt: float
    c_v: float
    band_z: float
    setpoint: str          # "hold" (default, brake-to-rest backup) | "goal" (goal-tracking variant)
    certify_backup: bool
    # ---- v2.8.3 baseline-v2 additions (ABSENT keys reproduce B1 exactly) ----
    form: str = "setpoint"      # "setpoint" (B1 legacy) | "brake" (v2: NO position term)
    k_d: float = 2.0            # brake form: velocity gain of a_des = -k_d v + g e3
    kp_att: float | None = None  # brake form attitude gains; None -> the system's shipped value
    kd_att: float | None = None
    # ---- v2.8.3 STANDARD-form addition (ABSENT key reproduces B1 and baseline-v2 exactly) ----
    # "rest_penalty" -> the B1/v2 hybrid  h_b = min(min_t m, m(x_T) - c_v ||v(x_T)||)
    # "none"         -> the STANDARD horizon-length variant  h_b = min_t m  (no terminal term at all)
    terminal: str = "rest_penalty"


def backup_params(config: Mapping[str, Any]) -> _BackupParams:
    cfg = dict(config.get("filter", {}).get("backup", {}) or {})
    T_b = int(cfg["T_b"])
    if T_b < 1:
        raise ValueError(f"filter.backup.T_b must be >= 1, got {T_b}.")
    setpoint = str(cfg.get("setpoint", "hold"))
    if setpoint not in {"hold", "goal"}:
        raise ValueError(f"filter.backup.setpoint must be 'hold' or 'goal', got {setpoint!r}.")
    form = str(cfg.get("form", "setpoint"))
    if form not in {"setpoint", "brake"}:
        raise ValueError(f"filter.backup.form must be 'setpoint' or 'brake', got {form!r}.")
    terminal = str(cfg.get("terminal", "rest_penalty"))
    if terminal not in {"rest_penalty", "none"}:
        raise ValueError(
            f"filter.backup.terminal must be 'rest_penalty' or 'none', got {terminal!r}.")
    kp_att = cfg.get("kp_att", None)
    kd_att = cfg.get("kd_att", None)
    return _BackupParams(
        T_b=T_b,
        eps=float(cfg["eps"]),
        dt=float(config.get("env", {}).get("dt", 0.05)),
        c_v=float(cfg.get("c_v", _C_V_DEFAULT)),
        band_z=float(config.get("env", {}).get("band_collision_limit", 0.0)),
        setpoint=setpoint,
        certify_backup=bool(cfg.get("certify_backup", True)),
        form=form,
        k_d=float(cfg.get("k_d", 2.0)),
        kp_att=(None if kp_att is None else float(kp_att)),
        kd_att=(None if kd_att is None else float(kd_att)),
        terminal=terminal,
    )


_GAIN_ATTRS = ("kp_pos", "kd_pos", "kp_att", "kd_att")


@contextlib.contextmanager
def override_gains(system: System, **gains: float | None) -> Iterator[None]:
    """Temporarily swap `System`'s cascaded-PD gain attributes, then restore them.

    This is the WRAP mechanism: nothing about the mixer, the thrust projection, the attitude loop
    or the per-rotor box clip is re-implemented -- `System.lqr_action` is called with different
    gain attributes and restores them on exit (including on exception)."""
    bad = [k for k in gains if k not in _GAIN_ATTRS]
    if bad:
        raise ValueError(f"override_gains got non-gain attributes {bad!r}; allowed {_GAIN_ATTRS}.")
    saved = {k: getattr(system, k) for k in gains if gains[k] is not None}
    try:
        for k, v in gains.items():
            if v is not None:
                setattr(system, k, float(v))
        yield
    finally:
        for k, v in saved.items():
            setattr(system, k, v)


def brake_action(system: System, x: Tensor, params: _BackupParams) -> Tensor:
    """v2.8.3 baseline-v2 pi_b -- LEVEL ATTITUDE AND ZERO VELOCITY, **NO POSITION TERM**:

        a_des  = -k_d v + g e3
        f_thr  = m (a_des . b3), clipped at 0
        b3_des = a_des / ||a_des||
        tau    = J (kp_att e_att_body - kd_att omega)      -> X-mixer -> per-rotor box [0, 4.905]

    Implemented as a WRAP of `System.lqr_action` (src/envs/quadrotor_3d.py:173-189) with the
    POSITION gain set to ZERO and the setpoint fed the state's own position, so the position term
    `-kp_pos (p - goal)` is identically `-0.0 * (p - p) = 0` and every remaining line -- the
    `m (a_des + g e3)` force, the projection onto b3, the `b3 x b3_des` attitude error, the mixer
    inverse and `_clamp_action` -- is the shipped code, byte for byte.

    pi_b is SCENE-BLIND by construction: its only inputs are `x` and the gains. `scene` is not a
    parameter of this function and `system.position(x)` is a slice of the state.
    """
    with override_gains(system, kp_pos=0.0, kd_pos=params.k_d,
                        kp_att=params.kp_att, kd_att=params.kd_att):
        return system.lqr_action(x, system.position(x))


def clearance_margin(p: Tensor, scene: Any, band_z: float) -> Tensor:
    """Signed geometric clearance in metres; > 0 = safe. See the module docstring for why this is
    NOT signed_h and NOT h_star. `p` is [..., pos_dim]; returns [...]."""
    centers = torch.as_tensor(scene.obstacle_centers, dtype=p.dtype, device=p.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=p.dtype, device=p.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=p.device)
    # Obstacles occupy the first centers.shape[-1] position coords (xy footprint of the infinite
    # vertical cylinders) -- the same slicing rule as _collided_exact / signed_h.
    p_o = p[..., : centers.shape[-1]]
    distance = torch.linalg.norm(p_o.unsqueeze(-2) - centers, dim=-1)
    clearance = distance - radii
    clearance = torch.where(active, clearance, torch.full_like(clearance, _FAR))
    m = clearance.min(dim=-1).values
    if band_z > 0.0 and p.shape[-1] >= 3:
        p_z = p[..., 2]
        m = torch.minimum(m, p_z + band_z)
        m = torch.minimum(m, band_z - p_z)
    return m


def backup_action(system: System, x: Tensor, hold: Tensor | None,
                  params: _BackupParams | None = None) -> Tensor:
    """pi_b at state x. Verbatim reuse of System.lqr_action (per-rotor clipped).

    `params.form == "brake"` (v2.8.3 baseline-v2) selects the no-position-term brake and ignores
    `hold`; anything else is the B1 setpoint form and uses `hold`."""
    if params is not None and params.form == "brake":
        return brake_action(system, x, params)
    return system.lqr_action(x, hold)


def backup_h(
    x: Tensor,
    scene: Any,
    system: System,
    params: _BackupParams,
    *,
    return_traj: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """h_b(x) by RK4 rollout of pi_b over T_b steps. Fully batched, entirely under no_grad.

    Returns h_b [B]; with return_traj=True also (m_seq [B, T_b+1], states [B, T_b+1, state_dim])."""
    with torch.no_grad():
        if params.form == "brake":
            hold = None                                   # brake form has NO setpoint at all
        elif params.setpoint == "goal":
            from src.common.observation import scene_goal_tensor
            hold = scene_goal_tensor(scene, x)
        else:
            hold = system.position(x)                     # frozen hover-hold setpoint
        xt = x
        m_list = [clearance_margin(system.position(xt), scene, params.band_z)]
        x_list = [xt]
        for _ in range(params.T_b):
            u = backup_action(system, xt, hold, params)
            xt = rk4_step(system, xt, u, params.dt)
            m_list.append(clearance_margin(system.position(xt), scene, params.band_z))
            if return_traj:
                x_list.append(xt)
        m_seq = torch.stack(m_list, dim=1)                # [B, T_b+1]
        m_min = m_seq.min(dim=1).values
        if params.terminal == "none":
            # v2.8.3 STANDARD form, ITEM 4(b): no terminal term at all. h_b is the horizon minimum of
            # the clearance along the backup flow and the guarantee rests on the horizon LENGTH. The
            # hybrid `m - c_v ||v||` it replaces was neither a control-invariant terminal set nor an
            # explicit omission, so it is removed rather than reinterpreted.
            h_b = m_min
        else:
            m_term = m_seq[:, -1] - params.c_v * system.speed(xt)
            h_b = torch.minimum(m_min, m_term)
    if return_traj:
        return h_b, m_seq, torch.stack(x_list, dim=1)
    return h_b


class BackupSwitchingFilter:
    """One-step-lookahead switch between u_nom and pi_b, gated on the rollout certificate h_b.

    Constructed at the SAME call site as HardNetFilter and called with the SAME signature
    (x, scene, u_nom, ...), returning the SAME tuple shape."""

    def __init__(
        self,
        system: System,
        h_fn: Callable[[Tensor, Any], Tensor] | None,
        config: Mapping[str, Any],
        *,
        policy_fn: Callable[[Tensor, Any], Tensor] | None = None,
    ) -> None:
        self.system = system
        self.h_fn = h_fn                 # accepted for call-site parity; NEVER used (h_b replaces it)
        self.policy_fn = policy_fn
        self.params = backup_params(config)
        self.last_empty: Tensor | None = None
        self.last_singular: Tensor | None = None
        self.last_backup: Tensor | None = None       # pi_b-branch indicator [B] (the switch mask)
        self.last_hb: Tensor | None = None           # h_b(f_dt(x, u_nom)) [B]

    def __call__(
        self,
        x: Tensor,
        scene: Any,
        u_nom: Tensor,
        detach_coeffs: bool = False,
        return_deficit_aux: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 2 or u_nom.ndim != 2:
            raise ValueError("x and u_nom must be batched rank-2 tensors.")
        if x.shape[0] != u_nom.shape[0]:
            raise ValueError("x and u_nom batch sizes must match.")

        p = self.params
        with torch.no_grad():
            x_next = rk4_step(self.system, x.detach(), u_nom.detach(), p.dt)   # f_dt(x, u_nom)
            h_next = backup_h(x_next, scene, self.system, p)
            use_nom = h_next >= p.eps
            take_backup = ~use_nom

        # pi_b(x) with the hover-hold setpoint = position(x): the first action of the backup
        # trajectory from x. Left LIVE in x so BPTT still sees the state dependence on this branch;
        # d u_dep / d u_nom = 0 on this branch by construction (torch.where on a constant mask).
        if p.form == "brake":
            hold_now = None
        elif p.setpoint == "goal":
            from src.common.observation import scene_goal_tensor
            hold_now = scene_goal_tensor(scene, x)
        else:
            hold_now = self.system.position(x)
        u_backup = backup_action(self.system, x, hold_now, p)
        u_dep = torch.where(use_nom.unsqueeze(1), u_nom, u_backup.to(u_nom.dtype))

        # infeasible := the backup ITSELF is not certified here (the switching analogue of an empty
        # QP feasible set). Restricted to the branch rows, so the extra rollout costs nothing when
        # the switch rate is low.
        infeasible = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if p.certify_backup and bool(take_backup.any()):
            from src.common.kstep_fallback import slice_scene
            with torch.no_grad():
                m = take_backup
                h_self = backup_h(x.detach()[m], slice_scene(scene, m), self.system, p)
                infeasible[m] = h_self < p.eps

        zeros = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        self.last_empty = zeros                      # no QP -> no empty intersection (published as 0)
        self.last_singular = zeros.clone()           # no L_g h -> no singular row (published as 0)
        self.last_backup = take_backup.detach()
        self.last_hb = h_next.detach()

        if return_deficit_aux:
            # deficit channel parity: "the control the certificate asked for" is pi_b itself.
            return u_dep, infeasible, u_backup.detach(), zeros.clone()
        return u_dep, infeasible


def resolve_filter_cls(config):
    """v2.8.3 B2 — the SINGLE resolver for `config.filter.type`, shared by the two training-path
    construction sites (`jt_pncbf/losses.py` policy BPTT, `jt_pncbf/collection.py` collection).

    Logically identical to the branch inlined at `jt_pncbf/train.py:207` (the deployed site), which the
    B2 dispatch holds untouched. Because that copy cannot be refactored away, the two copies are
    reconciled at RUNTIME instead: `assert_filter_sites_agree` compares what all three sites resolve to
    and raises if they disagree, so divergence is caught rather than assumed away.

    Absent or "hardnet" -> HardNetFilter (byte-identical to the pre-gate construction).
    "backup_switching" -> BackupSwitchingFilter.
    """
    from src.common.filter_hardnet import HardNetFilter
    ft = str((config.get("filter") or {}).get("type", "hardnet"))
    if ft == "hardnet":
        return HardNetFilter
    if ft == "backup_switching":
        return BackupSwitchingFilter
    raise ValueError(f"Unknown filter.type {ft!r}; expected 'hardnet' or 'backup_switching'.")


def assert_filter_sites_agree(config, resolved: dict) -> dict:
    """v2.8.3 B2 INVARIANT 1: all filter construction sites must resolve to the SAME class within a run.

    `resolved` maps site-name -> class name as actually constructed. A disagreement means the policy
    trained under one certificate and was scored under another, which voids the run -- so this raises
    rather than warns. Returns the record for persistence.
    """
    expected = resolve_filter_cls(config).__name__
    bad = {k: v for k, v in resolved.items() if v != expected}
    if bad:
        raise RuntimeError(
            f"filter-site disagreement: expected {expected!r} at every site, got {bad!r}. "
            f"The run is void -- a policy trained under one certificate cannot be scored under another."
        )
    return {"expected": expected, "sites": dict(resolved)}
