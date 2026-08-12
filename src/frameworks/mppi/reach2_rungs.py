"""v2.8.4 — the SECOND reach ladder for the (T, omega_des) cascade: three screens, one module.

SUBJECT. `src/frameworks/mppi/cascade_rate.py`. THAT FILE IS NOT MODIFIED BY THIS WORK. Every lever below
lives in the subclass `Reach2Controller`, exactly as `lit_rungs.LitCascadeController` does, so "existing
default behaviour unchanged" is STRUCTURAL rather than asserted: with every flag at its default this
class's `act` and `_rollout_chunk` dispatch to `RateCascadeController`'s through `super()`, and the OFF
state is the shipped code path itself rather than a reproduction of it.

`cost.py`, `mppi_controller.py`, `lit_rungs.py` and every v1-v5 artifact are likewise untouched. The
cost objects (`stage_cost`, `terminal_cost`, `collision_mask`), the inner rate loop, the mixer, the
post-allocation per-rotor clip, the full RK4 rotor-thrust SIMULATION plant, the min-cost baseline
subtraction, the ESS definition, the degenerate rule, the control hold and the hover centring are all
CALLED, never re-implemented.

K_seed IS STRUCTURALLY ZERO. `reach_rungs.py` — which holds the FORBIDDEN seeded/geometric rollout rung —
is NOT imported by this module and cannot be reached from it. There is no seeding code here at all: no
reference trajectory, no global planner, no map, no waypoint sequence, no geometric rollout. The absence
is read back on every cell (`Reach2Flags.record()["K_seed"]`) and is verified at import by
`assert_no_seeding()`, which inspects this module's own globals.

=================================================================================================
THE LEVERS
=================================================================================================

MILESTONE 1 — three levers that ALREADY EXIST in the shipped code and were never run on the pool.
  `lam_mode`    {relative, absolute}. `MPPIParams.lam_mode`, implemented at `cascade_rate.py:693-703`.
                A CONSTRUCTION-TIME selection of a shipped field; it enters no branch written here.
  `c_crash`     {the base cell's 1e5, a reference-scale value}. `CostParams.c_crash`, a shipped field
                passed straight through `evaluate_mppi.run_cell`. No branch written here either.
  `thrust_ref`  {absolute, hover_referenced}. THE FLAG EXISTS (`lit_rungs.LitFlags.l2_thrust_ref`) AND
                ITS ONLY CONSUMER IS A THRUST COST. `cost.stage_cost` — the shipped base cell's running
                cost — carries NO thrust term (its terms are w_goal d^2, the gated speed excess, the
                gated rate excess, and optionally G1/G4/B1, none of which reads the collective). The
                flag is therefore STRUCTURALLY INERT at the shipped base cell, and the two levels of
                this axis are the same configuration. That is recorded here and MEASURED on the pool,
                not asserted: the axis is run at both levels and the rows are compared.

MILESTONE 2 — the PA-MPPI cost form, four ADDITIVE terms, each a flag, each defaulting OFF. With every
flag off not one of them is evaluated and the running cost is `cost.stage_cost` byte for byte.
  (a) `m2_progress`      l_goal   = -c_goal * relu(d_0 - d_k),  d_0 = the distance at HORIZON START
  (b) `m2_endpoint`      the FINAL stage charges its own constant `c_goal_endpoint` in place of `c_goal`
                         (requires (a): it is that term's last-stage weight, not a second term)
  (c) `m2_vel_gate`      l_vel    = w_vel_pa * exp(-c_vel * d_k^2) * ||v_k||^2
  (d) `m2_du`            l_du     = sum_j R_delta_j * (u_k,j - u_{k-1},j)^2 on the SAMPLED command
                         sequence. The k = 0 reference is the REALIZED command of the previous decision
                         — `project(u_exec)`'s collective, i.e. the thrust the motors actually produced
                         after the box clip, paired with the vehicle's OWN MEASURED body rate at that
                         instant — and the hover anchor `(m g, 0, 0, 0)` at the first decision of an
                         episode. It is deliberately NOT called "the previous commanded rate": the
                         parent's `act` is CALLED rather than re-implemented, so the commanded rate is
                         a local of that method and is not recoverable after the receding-horizon
                         shift. The realized pair is what this module can read back, and it is what is
                         used and recorded.

MILESTONE 3 — the two structural levers.
  (e) `dt_pred_ratio`    DECOUPLES the prediction step from the control step. The rollout integrates at
                         `dt_pred = ratio * env.dt` while the plant and the eval harness keep `env.dt`,
                         which is NOT touched. The receding-horizon shift becomes PA-MPPI's
                         INTERPOLATE-AND-SHIFT: advancing the plan by one CONTROL step is advancing it
                         by a FRACTION 1/ratio of a prediction stage, so

                             plan'[k] = (1 - f) * plan[k] + f * plan[k+1],   f = 1 / ratio,

                         with the shipped braking tail `plan_trim` appended as entry H. At ratio 1,
                         f = 1 and this is `MPPIController._shift` exactly; the branch is not even
                         entered — `super()._shift` is CALLED.
  (f) `in_planner_clip`  IN-PLANNER motor-limit clipping, the chain PA-MPPI v3 Sect. IV-A says it
                         follows — Minarik et al. IROS 2024 Eqs. (10)-(15) — with OUR inner loop's
                         torque demand in place of theirs:

                             tau_d        = J k_rate (omega_cmd - omega)   `inner_loop`'s own law
                             u            = clip(Gamma^-1 (T, tau_d))      `allocate`, box = u_bounds
                             (T_c, tau_c) = Gamma u                        `project`, u @ mixer.T
                             omega_c      = omega + dt_pred J^-1 (tau_c - omega x J omega)

                         and the rollout simulates (T_c, omega_c). No bound is invented: the box is
                         `system.u_bounds`, the mixer and the inertia are the system's, and every
                         operation is a SHIPPED method CALLED.
  (g) N                  reported, not switched: every cell records its wall-clock per control step.

Every constant that is not a transcribed literature value is READ from the system object or the config.
The transcribed literature values live in `PaMppiCost` and carry their source on every cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.common.rk4 import rk4_step
from src.frameworks.mppi.cascade_rate import RateCascadeController, _W
from src.frameworks.mppi.cost import collision_mask, stage_cost, terminal_cost


Tensor = torch.Tensor

THRUST_REFS = ("absolute", "hover_referenced")
DT_PRED_RATIOS = (1, 2, 4)


# =================================================================================================
# THE TRANSCRIBED LITERATURE CONSTANTS — Milestone 0. Every field carries its source.
# =================================================================================================
@dataclass(frozen=True)
class PaMppiCost:
    """The PA-MPPI cost constants, TRANSCRIBED. They are DATA, not defaults resolved from our config,
    and the whole object is written verbatim onto every cell that uses them so a reader can check the
    transcription without leaving the artifact.

    A field whose value could NOT be confirmed from the source carries `None` and the cell records it as
    UNCONFIRMED. Nothing here is filled from memory.
    """

    c_goal: float | None = None
    c_goal_endpoint: float | None = None
    c_vel: float | None = None
    w_vel_pa: float | None = None
    r_delta: tuple[float, float, float, float] | None = None
    c_crash_reference_scale: float | None = None
    source: str = ""
    provenance: tuple[tuple[str, str, str], ...] = ()
    caveats: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "c_goal": self.c_goal,
            "c_goal_endpoint": self.c_goal_endpoint,
            "c_vel": self.c_vel,
            "w_vel_pa": self.w_vel_pa,
            "r_delta": list(self.r_delta) if self.r_delta is not None else None,
            "c_crash_reference_scale": self.c_crash_reference_scale,
            "unconfirmed_fields": [
                name for name, value in (
                    ("c_goal", self.c_goal), ("c_goal_endpoint", self.c_goal_endpoint),
                    ("c_vel", self.c_vel), ("w_vel_pa", self.w_vel_pa),
                    ("r_delta", self.r_delta),
                    ("c_crash_reference_scale", self.c_crash_reference_scale),
                ) if value is None
            ],
            "provenance": [
                {"symbol": s, "value": v, "where": w} for s, v, w in self.provenance
            ],
            "caveats": list(self.caveats),
            "unit_warning": (
                "OUR plan channels are (N, rad/s, rad/s, rad/s). A source that states a MASS-NORMALIZED "
                "collective thrust (units m/s^2) carries a weight whose transcription into Newtons is off "
                "by mass^2 on any quadratic thrust term. Every cell records the system's own mass so the "
                "conversion is auditable."
            ),
        }


# ---- the transcription itself. Every value carries its location in the source. -------------------
PA_MPPI_V3 = PaMppiCost(
    # Sect. IV-D body text, exploration phase (the goal not in direct line of sight):
    #   "with c_goal = 0.125 and c_goal,H-1 = 10"
    c_goal=0.125,
    c_goal_endpoint=10.0,
    # Table I ("PA-MPPI parameters"), arXiv:2509.14978v3
    c_vel=5.0,
    # l_vel = exp(-c_vel * d_k^2) * ||v_k||^2 carries NO outer weight in the source; the constant is
    # inside the exponent. The 1.0 below is therefore the SOURCE'S OWN form, not a chosen weight.
    w_vel_pa=1.0,
    # Table I, R_Delta = diag(0.02, 0.05, 0.05, 0.05) on (c, w_x, w_y, w_z)
    r_delta=(0.02, 0.05, 0.05, 0.05),
    # Table I, c_collision = 15.0 — the "reference-scale" crash constant of Milestone 1
    c_crash_reference_scale=15.0,
    source="arXiv:2509.14978v3, Zhai, Reiter, Scaramuzza, 'PA-MPPI: Perception-Aware Model Predictive "
           "Path Integral Control for Quadrotor Navigation in Unknown Environments', RA-L, accepted "
           "January 2026",
    provenance=(
        ("c_goal", "0.125", "v3 Sect. IV-D body text (not in Table I): 'with c_goal = 0.125 and "
                            "c_goal,H-1 = 10' — the exploration phase, goal NOT in line of sight"),
        ("c_goal,H-1", "10", "v3 Sect. IV-D body text, same sentence"),
        ("c_vel", "5.0", "v3 Table I"),
        ("l_vel outer weight", "none in the source", "v3 Sect. IV-D: l_vel = exp(-c_vel d_k^2) ||v_k||^2"),
        ("R_Delta", "diag(0.02, 0.05, 0.05, 0.05)", "v3 Table I"),
        ("c_collision", "15.0", "v3 Table I"),
        ("u", "[c, omega_B]", "v3 Sect. IV-A: 'the zero-order hold PA-MPPI control u_t = [c_t w_B,t]^T'"),
        ("c", "NEWTONS, not mass-normalised",
         "v3 Sect. IV-A: 'c = c_1 + ... + c_Nrot ... c_i is the thrust generated by the i-th of N_rot "
         "motors'; Eq. (2) row 3 divides c_B by m INSIDE the dynamics"),
        ("N", "17500", "v3 Table I"),
        ("H", "15 stages", "v3 Table I"),
        ("dt_pred", "0.1 s", "v3 Table I"),
        ("dt_ctrl", "0.02 s", "v3 Table I; 50 Hz in the abstract"),
        ("lambda", "0.02", "v3 Table I"),
        ("weight rule", "w_j = exp(-(L_j - L_min)/lambda) / sum_j exp(-(L_j - L_min)/lambda)",
         "v3 Sect. IV-B — an ABSOLUTE lambda with a min shift; there is NO division by cost std"),
        ("warm start", "'the control sequence is linearly interpolated and shifted by dt_ctrl, then "
                       "down-sampled at dt_pred'", "v3 Sect. IV-B, verbatim"),
        ("in-planner clip", "'we follow the single motor thrust clipping in [12], using the motor "
                            "thrust limits to acquire clipped control input u_t^clip, which is then "
                            "used by PA-MPPI to simulate the dynamics (2)'", "v3 Sect. IV-A, verbatim"),
        ("[12]", "M. Minarik, R. Penicka, V. Vonasek, M. Saska, 'Model predictive path integral "
                 "control for agile unmanned aerial vehicles', IROS 2024 (arXiv:2407.09812)",
         "v3 bibliography entry [12] — the dispatch's identification is correct"),
        ("[12] clip chain", "T_d = Gamma^-1 [F_t; tau_d]; T_clip = clip(T_min, T_d, T_max); "
                            "[F_clip; tau_clip] = Gamma T_clip; "
                            "omega_dot_clip = J^-1 (tau_clip - omega x J omega); "
                            "omega_clip = omega + omega_dot_clip dt",
         "Minarik et al. Eqs. (10)-(15); 'The clipped values [F_t,clip, tau_clip] from (13) are used "
         "to simulate the dynamics (6)'"),
        ("[12] Sigma", "diag(0.60, 0.15, 0.15, 0.05) on [F_t, w_x, w_y, w_z]", "Minarik et al. Table II"),
        ("[12] R_Delta", "diag(0.05, 0.10, 0.10, 0.30)", "Minarik et al. Table II"),
        ("[12] c_obs", "1e6", "Minarik et al. Sect. IV-D, Eq. (22)"),
        ("[12] T_min/T_max", "0.3 N / 19.0 N (sim), 0.3 N / 8.0 N (real)", "Minarik et al. Table I"),
        ("platform", "0.21 kg, arm 19.4 cm, thrust-to-weight 6.8, NVIDIA A1000 laptop GPU (6 GB), JAX",
         "v3 Sect. V para. 1"),
        ("success", "100% for both SUPER and PA-MPPI on all synthetic scenes",
         "v3 Sect. V"),
        ("Faessler RA-L 2017", "MASS-NORMALISED collective thrust, m/s^2",
         "Faessler, Falanga, Scaramuzza, Sect. II: 'a mass normalized collective thrust c can be "
         "applied on the quadrotor's body'; Eq. (2): 'm c = f_1 + f_2 + f_3 + f_4'"),
    ),
    caveats=(
        "THE DISPATCH'S w_thrust = 0.017 IS NOT A PA-MPPI VALUE. The literal string 0.017 occurs in no "
        "version of arXiv:2509.14978 and in no part of [12]. PA-MPPI's only thrust-channel weight is "
        "R[0,0] = 0.01 in l_act = ||u||^2_R + ||du||^2_{R_Delta} (v3 Table I). The 0.017 in this "
        "repository comes from a DIFFERENT paper — arXiv:2605.02147 Table VII, transcribed in "
        "src/frameworks/mppi/lit_rungs.py — and the two must not be conflated.",
        "THE MASS-NORMALISATION HYPOTHESIS DOES NOT APPLY TO PA-MPPI. Faessler RA-L 2017 IS "
        "mass-normalised (m/s^2), but PA-MPPI's collective is in NEWTONS, the same convention as ours. "
        "So a PA-MPPI thrust weight needs no unit correction for our arm. Whether arXiv:2605.02147's "
        "0.017 is mass-normalised was NOT confirmed and is not asserted here.",
        "PHASE. PA-MPPI gates its cost on line of sight to the goal: the constants above are the "
        "EXPLORATION phase (goal not in line of sight). In the line-of-sight phase it sets c_goal = 5.0 "
        "and DROPS l_goal,H-1. Our arm has no map, no ray cast and no perception term, so no phase "
        "gate exists to transcribe; the exploration-phase constants are taken because they are the "
        "pair the endpoint term belongs to. The phase-2 value 5.0 is recorded and NOT run.",
        "CHARGE-ONCE. Our rollout charges the collision indicator ONCE, at its first occurrence "
        "(cascade_rate.py:620). PA-MPPI's l_collision is a per-stage indicator. The graded-vs-indicator "
        "question is NOT opened here: c_crash simply takes the reference-scale value and keeps the "
        "shipped charge-once mechanics.",
        "NOT TRANSCRIBED, because the source does not state them: PA-MPPI's sampling covariance Sigma "
        "(only 'multivariate Gaussian noise'), c_progress, its motor thrust limits and the terminal "
        "safe-set bounds. None of them is used by any cell of this work.",
    ),
)


# =================================================================================================
# THE FLAGS
# =================================================================================================
@dataclass(frozen=True)
class Reach2Flags:
    """One field per lever. The all-default instance is the shipped implementation and `act_flags_on`
    is False, which is what makes the OFF state the shipped code path rather than a re-derivation.

    `lam_mode` and `c_crash` are absent BY DESIGN: they are shipped `MPPIParams` / `CostParams` fields
    that `evaluate_mppi.run_cell` already carries, so setting them changes no code path in this module.
    """

    # ---- Milestone 1 ---------------------------------------------------------------------------
    thrust_ref: str = "absolute"
    # ---- Milestone 2 (all OFF) -----------------------------------------------------------------
    m2_progress: bool = False
    m2_endpoint: bool = False
    m2_vel_gate: bool = False
    m2_du: bool = False
    # ---- Milestone 3 ---------------------------------------------------------------------------
    dt_pred_ratio: int = 1
    in_planner_clip: bool = False

    def __post_init__(self) -> None:
        if self.thrust_ref not in THRUST_REFS:
            raise ValueError(f"thrust_ref must be one of {THRUST_REFS}, got {self.thrust_ref!r}.")
        if self.m2_endpoint and not self.m2_progress:
            raise ValueError(
                "(b) is the LAST-STAGE WEIGHT of (a)'s progress term, not a second term; it cannot be "
                "switched on with (a) off."
            )
        if int(self.dt_pred_ratio) not in DT_PRED_RATIOS:
            raise ValueError(
                f"dt_pred_ratio must be one of {DT_PRED_RATIOS}, got {self.dt_pred_ratio!r}."
            )

    @property
    def act_flags_on(self) -> bool:
        """Whether any flag changes the CODE PATH. `thrust_ref` is absent: its only consumer is a thrust
        cost term, and no term written in this module reads it (see the module docstring), so with only
        `thrust_ref` set the shipped `act` and the shipped `_rollout_chunk` still run."""
        return bool(
            self.m2_progress or self.m2_vel_gate or self.m2_du
            or int(self.dt_pred_ratio) != 1 or self.in_planner_clip
        )

    @property
    def rollout_flags_on(self) -> bool:
        return bool(
            self.m2_progress or self.m2_vel_gate or self.m2_du
            or int(self.dt_pred_ratio) != 1 or self.in_planner_clip
        )

    def record(self) -> dict[str, Any]:
        return {
            "thrust_ref": self.thrust_ref,
            "thrust_ref_status": (
                "STRUCTURALLY INERT at the shipped base cell: `cost.stage_cost` carries no thrust term, "
                "so no expression in this module or in the shipped cost reads this flag. Both levels of "
                "the axis are the same configuration; the identity is MEASURED on the pool, not asserted."
            ),
            "m2_progress_a": self.m2_progress,
            "m2_endpoint_b": self.m2_endpoint,
            "m2_vel_gate_c": self.m2_vel_gate,
            "m2_du_d": self.m2_du,
            "dt_pred_ratio_e": int(self.dt_pred_ratio),
            "in_planner_clip_f": self.in_planner_clip,
            "act_flags_on": self.act_flags_on,
            "K_seed": 0,
            "K_seed_note": (
                "STRUCTURALLY ZERO. This module contains no seeded / geometric rollout code at all; "
                "`reach_rungs.py` (which does) is not imported and cannot be reached from here."
            ),
            "off_state": (
                "with act_flags_on False, `act` and `_rollout_chunk` dispatch to "
                "RateCascadeController's through super() — the shipped code path itself, not a copy"
            ),
        }


def assert_no_seeding() -> dict[str, Any]:
    """Read back, from this module's OWN globals, that no seeded-rollout machinery is reachable."""
    names = sorted(globals().keys())
    forbidden = [n for n in names if "seed_rollout" in n or "k_seed" in n.lower()]
    imported = sorted(
        m for m in (getattr(v, "__name__", "") for v in globals().values())
        if isinstance(m, str) and "reach_rungs" in m
    )
    if forbidden or imported:                                              # pragma: no cover
        raise AssertionError(f"seeding machinery reachable: {forbidden} {imported}")
    return {
        "K_seed": 0,
        "module_globals_with_seeding_names": forbidden,
        "reach_rungs_imported": imported,
        "statement": "no seeded / geometric rollout code exists in this module and reach_rungs is not "
                     "imported, so K_seed = 0 is structural rather than switched off",
    }


# =================================================================================================
# THE CONTROLLER
# =================================================================================================
class Reach2Controller(RateCascadeController):
    """The (T, omega_des) cascade with the second reach ladder's levers behind flags.

    `RateCascadeController` is SUBCLASSED, never edited. With every flag at its default this class's
    `act` is `RateCascadeController.act` reached through `super()`, so the shipped behaviour is the
    shipped code and not a reproduction of it.
    """

    def __init__(
        self,
        *args: Any,
        flags: Reach2Flags | None = None,
        pa_cost: PaMppiCost | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.flags = flags if flags is not None else Reach2Flags()
        self.pa = pa_cost if pa_cost is not None else PaMppiCost()

        # (e) the PREDICTION step. `self.dt` — the PLANT's step, read from env.dt by the base class —
        # is NOT changed anywhere; only the planner's integration step is.
        self.dt_pred = float(self.dt) * int(self.flags.dt_pred_ratio)
        self.lookahead_pred_s = self.dt_pred * int(self.params.horizon) * int(self.control_hold)

        # (d)'s reference for the k = 0 difference: the previously EXECUTED (T, omega_des) command.
        self.last_command: Tensor | None = None

        # (f)'s evidence, accumulated over the cell: PLANNER-SIDE pre-clip entries outside the box.
        self.planner_clip_entries = 0
        self.planner_clip_out_of_box = 0

        # per-term totals, summed contributions to the sample costs over the whole cell
        self.term_totals: dict[str, float] = {}

        # a flag that is switched on must have a constant to charge; refuse rather than invent one.
        if self.flags.m2_progress and self.pa.c_goal is None:
            raise ValueError("(a) is on but PaMppiCost.c_goal is UNCONFIRMED; refusing to invent it.")
        if self.flags.m2_endpoint and self.pa.c_goal_endpoint is None:
            raise ValueError(
                "(b) is on but PaMppiCost.c_goal_endpoint is UNCONFIRMED; refusing to invent it."
            )
        if self.flags.m2_vel_gate and (self.pa.c_vel is None or self.pa.w_vel_pa is None):
            raise ValueError("(c) is on but its PA-MPPI constants are UNCONFIRMED.")
        if self.flags.m2_du and self.pa.r_delta is None:
            raise ValueError("(d) is on but PaMppiCost.r_delta is UNCONFIRMED.")

    # ---- the record a cell JSON carries ----------------------------------------------------------
    def reach2_record(self) -> dict[str, Any]:
        return {
            "what": "the v2.8.4 SECOND reach ladder — one flag per lever, every OFF state the shipped "
                    "code path",
            "subject_file_modified": False,
            "subject": "src/frameworks/mppi/cascade_rate.py (subclassed, not edited)",
            "variant": "cascaded_rate",
            "flags": self.flags.record(),
            "no_seeding": assert_no_seeding(),
            "pa_mppi_cost": self.pa.record(),
            "prediction_step": {
                "dt_ctrl_s": float(self.dt),
                "dt_ctrl_source": "env.dt, read from the effective config; NOT changed by this work — "
                                  "it is the plant's and the eval harness integrates at it",
                "dt_pred_s": self.dt_pred,
                "dt_pred_ratio": int(self.flags.dt_pred_ratio),
                "horizon_stages": int(self.params.horizon),
                "real_time_horizon_s": self.lookahead_pred_s,
                "warm_start": (
                    "PA-MPPI interpolate-and-shift: plan'[k] = (1-f) plan[k] + f plan[k+1], f = 1/ratio, "
                    "with the shipped braking tail `plan_trim` as entry H. At ratio 1 the shipped "
                    "MPPIController._shift is CALLED instead."
                ),
                "rate_lowpass_caveat": (
                    "the sampled-rate low-pass beta is defined in ROLLOUT STEPS (beta = exp(-1/steps), "
                    "asserted at construction). It is left UNCHANGED, so at ratio r the filter's time "
                    "constant in SECONDS is r times longer inside the planner. Recorded, not corrected: "
                    "correcting it would introduce a constant this work has no source for."
                ),
            },
            "in_planner_clip": {
                "enabled": bool(self.flags.in_planner_clip),
                "law": (
                    "Minarik et al. IROS 2024 Eqs. (10)-(15), the chain PA-MPPI v3 Sect. IV-A says it "
                    "follows, with OUR inner loop's torque demand: tau_d = J k_rate (omega_cmd - "
                    "omega); u = allocate((T, tau_d)) — the SHIPPED MPPIController.allocate, i.e. "
                    "wrench @ system.mixer_inv.T clamped to system.u_bounds; (T_c, tau_c) = project(u) "
                    "— the SHIPPED MPPIController.project, u @ system.mixer.T; omega_c = omega + "
                    "dt_pred * J^-1 (tau_c - omega x J omega). The rollout simulates (T_c, omega_c)."
                ),
                "source": "arXiv:2509.14978v3 Sect. IV-A ('we follow the single motor thrust clipping "
                          "in [12], using the motor thrust limits to acquire clipped control input "
                          "u_t^clip, which is then used by PA-MPPI to simulate the dynamics (2)'); "
                          "Minarik et al. IROS 2024 Eqs. (10)-(15)",
                "box_read_from_system": [float(self.u_lo.min().item()), float(self.u_hi.max().item())],
                "closes": (
                    "asymmetry (2) INSIDE THE PLANNER: with this on, the per-rotor box constrains the "
                    "sampled plan before it is simulated, not only the executed command after "
                    "allocation. The executed-command clip in `inner_loop` is unchanged."
                ),
                "planner_preclip_entries": int(self.planner_clip_entries),
                "planner_preclip_out_of_box": int(self.planner_clip_out_of_box),
                "planner_preclip_frac_out_of_box": (
                    float(self.planner_clip_out_of_box) / float(self.planner_clip_entries)
                    if self.planner_clip_entries else None
                ),
            },
            "unchanged": [
                "the (T, omega_des) interface and the plan variable",
                "the inner rate loop tau = J k_rate (omega_des - omega) — CALLED",
                "the post-allocation per-rotor clip inside MPPIController.allocate — CALLED",
                "the full RK4 rotor-thrust SIMULATION plant and env.dt",
                "the collision predicate cost.collision_mask — CALLED, verdict never altered",
                "the charge-once C_crash mechanics and the freeze-on-collision rule",
                "the min-cost baseline subtraction, the ESS definition, the degenerate rule, the "
                "control hold, hover centring and the OU draw",
                "cost.stage_cost and cost.terminal_cost — CALLED; the (a)-(d) terms are ADDITIVE",
            ],
            "cost_term_totals": dict(self.term_totals) if self.term_totals else None,
        }

    # ---- (e) the interpolate-and-shift warm start -------------------------------------------------
    def _shift(self, plan: Tensor) -> Tensor:
        ratio = int(self.flags.dt_pred_ratio)
        if ratio == 1:
            return super()._shift(plan)                                  # THE SHIPPED SHIFT, CALLED
        tail = self.plan_trim.view(1, 1, -1).expand(plan.shape[0], 1, self.action_dim)
        extended = torch.cat([plan, tail], dim=1)                        # [B, H+1, m]
        f = 1.0 / float(ratio)
        return (1.0 - f) * extended[:, :-1] + f * extended[:, 1:]

    # ---- (f) the in-planner motor-limit clip ------------------------------------------------------
    def _planner_clip(
        self, thrust: Tensor, rate_cmd: Tensor, omega_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(T, omega_cmd) -> the (T, omega) the motors can actually produce from `omega_state`.

        This is Minarik et al. (IROS 2024) Eqs. (10)-(15) — the chain PA-MPPI v3 Sect. IV-A says it
        follows — with OUR inner loop's torque demand in place of theirs:

            tau_d      = J k_rate (omega_cmd - omega)        RateCascadeController.inner_loop's own law
            u          = clip(Gamma^-1 (T, tau_d))           MPPIController.allocate, box = system.u_bounds
            (T_c, tau_c) = Gamma u                           MPPIController.project, u @ system.mixer.T
            omega_c    = omega + dt_pred * J^-1 (tau_c - omega x J omega)

        Every operation is a SHIPPED one and every constant is read off the system object: the mixer,
        its inverse, the box, the inertia and the rate gain. Nothing is invented and no bound is typed.
        """
        gain = self.inertia * self.k_rate                                 # [3]
        tau = gain * (rate_cmd - omega_state)
        wrench = torch.cat([thrust, tau], dim=-1)
        with torch.random.fork_rng(devices=self._fork_devices):
            pre = wrench @ self.mixer_inv.t()
            self.planner_clip_entries += int(pre.numel())
            self.planner_clip_out_of_box += int(
                ((pre < self.u_lo) | (pre > self.u_hi)).sum().item()
            )
        rotor = self.allocate(wrench)                                     # SHIPPED clip to the box
        achieved = self.project(rotor)                                    # SHIPPED u @ mixer.T
        gyroscopic = torch.cross(omega_state, self.inertia * omega_state, dim=-1)
        omega_clip = omega_state + self.dt_pred * (achieved[..., 1:] - gyroscopic) / self.inertia
        return achieved[..., :1], omega_clip

    # ---- the rollout ------------------------------------------------------------------------------
    @torch.no_grad()
    def _rollout_chunk(
        self, x: Tensor, sampled: Tensor, goal: Tensor, centers: Tensor, radii: Tensor, active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """`RateCascadeController._rollout_chunk` with the (a)-(d) additive terms, the decoupled
        prediction step and the in-planner clip. With none of them on the PARENT IS CALLED and the OFF
        path is the shipped one.

        The freeze-on-collision rule, the charge-once C_crash mechanics, the SAME `collision_mask`, the
        SAME `stage_cost` / `terminal_cost` objects and the endpoint probe are unchanged.
        """
        if not self.flags.rollout_flags_on:
            return super()._rollout_chunk(x, sampled, goal, centers, radii, active)

        batch, n, horizon, plan_dim = sampled.shape
        flat = batch * n
        state = x.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        goal_flat = goal.unsqueeze(1).expand(batch, n, -1).reshape(flat, -1)
        cost = torch.zeros(flat, device=self.device, dtype=self.dtype)
        dead = torch.zeros(flat, dtype=torch.bool, device=self.device)
        beta = self.lowpass_beta
        rate_f = state[:, _W].clone()
        totals = {"stage_shipped": 0.0, "collision_shipped": 0.0, "terminal_shipped": 0.0,
                  "a_progress": 0.0, "c_vel_gate": 0.0, "d_du": 0.0}

        # (a)'s d_0: the distance at HORIZON START, i.e. of the state the controller was handed.
        d0 = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)

        # (d)'s k = 0 reference: the previously EXECUTED (T, omega_des) command, or the hover anchor at
        # the first decision of an episode. `anchor` is (m*g, 0, 0, 0), read off the system constants.
        if self.flags.m2_du:
            if self.last_command is None:
                prev_cmd = self.anchor.view(1, -1).expand(flat, plan_dim).clone()
            else:
                prev_cmd = (
                    self.last_command.unsqueeze(1).expand(batch, n, plan_dim).reshape(flat, plan_dim)
                    .clone()
                )
            r_delta = torch.as_tensor(
                self.pa.r_delta, device=self.device, dtype=self.dtype
            ).view(1, -1)
        else:
            prev_cmd = None
            r_delta = None

        n_physical = horizon * self.control_hold
        for k in range(n_physical):
            command = self.held_control(sampled, k).reshape(flat, plan_dim)

            # (d) the action-rate cost, on the SAMPLED command sequence, BEFORE any clipping — it is a
            # cost on what the planner proposes.
            if self.flags.m2_du:
                assert prev_cmd is not None and r_delta is not None
                du_term = (r_delta * (command - prev_cmd).square()).sum(dim=-1)
                cost = cost + du_term
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["d_du"] += float(du_term.sum().item())
                prev_cmd = command

            thrust, rate_cmd = command[:, :1], command[:, 1:]
            # (f) the in-planner motor-limit clip, BEFORE the step is simulated.
            if self.flags.in_planner_clip:
                thrust, rate_cmd = self._planner_clip(thrust, rate_cmd, rate_f)

            rate_f = beta * rate_f + (1.0 - beta) * rate_cmd
            state = torch.cat([state[:, :10], rate_f], dim=1)
            stepped = torch.cat([thrust, rate_cmd], dim=-1)
            nxt = rk4_step(self.planner, state, stepped, self.dt_pred)    # (e) the PREDICTION step
            previous = state
            state = torch.where(dead.unsqueeze(-1), state, nxt)

            collided = collision_mask(
                self.system.position(state).view(batch, n, -1), centers, radii, active, self.cost
            ).reshape(flat)
            newly = collided & ~dead
            crash = newly.to(self.dtype) * self.cost.c_crash
            cost = cost + crash
            dead = dead | collided

            shipped = stage_cost(
                self.system, state, goal_flat, self.cost, self.recovery, x_prev=previous
            )
            cost = cost + shipped

            distance = torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1)
            if self.flags.m2_progress:
                weight = (
                    float(self.pa.c_goal_endpoint)
                    if (self.flags.m2_endpoint and k == n_physical - 1)
                    else float(self.pa.c_goal)
                )
                progress = -weight * torch.clamp(d0 - distance, min=0.0)
                cost = cost + progress
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["a_progress"] += float(progress.sum().item())
            if self.flags.m2_vel_gate:
                gated = (
                    float(self.pa.w_vel_pa)
                    * torch.exp(-float(self.pa.c_vel) * distance.square())
                    * self.system.speed(state).square()
                )
                cost = cost + gated
                with torch.random.fork_rng(devices=self._fork_devices):
                    totals["c_vel_gate"] += float(gated.sum().item())

            with torch.random.fork_rng(devices=self._fork_devices):
                totals["stage_shipped"] += float(shipped.sum().item())
                totals["collision_shipped"] += float(crash.sum().item())

        terminal = terminal_cost(self.system, state, goal_flat, self.cost)
        cost = cost + terminal
        with torch.random.fork_rng(devices=self._fork_devices):
            totals["terminal_shipped"] += float(terminal.sum().item())
            for key, value in totals.items():
                self.term_totals[key] = self.term_totals.get(key, 0.0) + value

        if self.endpoint_probe:
            with torch.random.fork_rng(devices=self._fork_devices):
                self._endpoint_chunk_dists.append(
                    torch.linalg.norm(self.system.position(state) - goal_flat, dim=-1).view(batch, n)
                )
        return cost.view(batch, n), dead.view(batch, n)

    # ---- one control step -------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, x: Tensor, scene: Any) -> Tensor:
        """The shipped `RateCascadeController.act` is CALLED. The only thing added is (d)'s record of
        the REALIZED command of this decision, which feeds the k = 0 difference of the NEXT decision.
        It is pure bookkeeping and changes no control.

        WHAT IS RECORDED, precisely: `project(u_exec)[0]` — the collective the motors actually produced
        after the per-rotor box clip — paired with the vehicle's own MEASURED body rate at this instant.
        The COMMANDED rate is a local of the parent's `act`; because that method is CALLED rather than
        re-implemented, the commanded rate is not recoverable here and no attempt is made to pretend
        otherwise.

        With every flag off this method is `super().act` and the block below is not entered at all, so
        the OFF path is the shipped one exactly.
        """
        action = super().act(x, scene)
        if self.flags.m2_du and self.last_decision:
            with torch.random.fork_rng(devices=self._fork_devices):
                # the REALIZED command: `project(action)`'s collective (post-box-clip) paired with the
                # vehicle's own measured body rate. Read back off shipped state; no control law re-run.
                wrench = self.project(action)
                self.last_command = torch.cat(
                    [wrench[:, :1], x.to(device=self.device, dtype=self.dtype)[:, _W]], dim=-1
                ).detach()
        return action

    def reset(self, batch_size: int) -> None:
        super().reset(batch_size)
        self.last_command = None

    # ---- the sampler record, extended with this ladder's own block --------------------------------
    def cascade_record(self) -> dict[str, Any]:
        record = super().cascade_record()
        record["reach2"] = self.reach2_record()
        return record


assert_no_seeding()


# =================================================================================================
# THE RUNNER — three screens on the pool of record. MEASUREMENT ONLY.
# =================================================================================================
# Everything below is a driver: it constructs cells, calls the SHARED eval path through
# `evaluate_mppi.run_cell`, and writes `data/runs/v2.8.4/mppi_reach2/**` plus the append-only build log
# `docs/versions/v2.8.4/mppi_reach2.md`. It selects nothing, ranks nothing and sorts nothing: every table
# is emitted in CONFIG ORDER.
#
# HOW THE SUBCLASS REACHES THE SHARED EVAL PATH WITHOUT EDITING A FILE. `evaluate_mppi.build_framework`
# resolves `RateCascadeController` by a LOCAL import, i.e. by attribute lookup on
# `src.frameworks.mppi.cascade_rate` at call time. `bound_controller` rebinds that attribute to a
# subclass bound to this cell's flags for the duration of one cell and restores it afterwards. No file is
# modified; the class object `cascade_rate.RateCascadeController` itself is untouched (it remains this
# module's own base class through the MRO), and every metric, band, column and artifact downstream of the
# controller is byte-identically the one the 8-cell `mppi_cascaded` screen produced.
import argparse
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src._version import __version__
from src.frameworks.mppi import cascade_rate as _cascade_rate_module
from src.frameworks.mppi.evaluate_mppi import REPO, effective_config, load_mppi_config, run_cell
from src.frameworks.mppi.relaxed_v5 import load_rollouts, save_rollouts, score_two_terminals
from src.frameworks.mppi.screen_cascaded import cascade_kwargs
from src.frameworks.mppi.screen_recovery import spawn_tilt_deg
from src.frameworks.mppi.stages_v5 import base_kwargs


OUT_DIR = REPO / "data/runs/v2.8.4/mppi_reach2"
BUILD_LOG = REPO / "docs/versions/v2.8.4/mppi_reach2.md"
POOL_SHA8 = "3682a4e3"

# The standing GPU floor of this project, a LAUNCH CONDITION and not a coefficient of any controller.
VRAM_FLOOR_MIB = 6144.0

# The four LIVE v2.8.5 runs. Read-only, and only these three files of each.
V285_RUNS = {
    "A": "data/runs/v2.8.5/set__20260810-033417__seed42/v2.8.5__jt__20260810-033417__seed42",
    "B": "data/runs/v2.8.5/set__20260810-033431__seed42/v2.8.5__jt__20260810-033431__seed42",
    "C": "data/runs/v2.8.5/set__20260810-033451__seed42/v2.8.5__jt__20260810-033451__seed42",
    "CLIP50": "data/runs/v2.8.5/set__20260810-072139__seed42/v2.8.5__jt__20260810-072139__seed42",
}
V285_PIDS = {"A": 1481321, "B": 1481873, "C": 1482690, "CLIP50": 1959204}

PRIVILEGE_NOTE = (
    "PRIVILEGE, once per table. This arm reads the privileged full 13-D state and the scene's obstacle "
    "field directly and rolls the TRUE plant model in its planner; no learned arm does. The per-rotor box "
    "is handled AFTER allocation, inside the inner loop, on the executed command only — except on the "
    "rows where the in-planner clip (f) is ON, which is stated in that row's own column. `infeasibility` "
    "and `mean_proj_mag` are STRUCTURALLY INAPPLICABLE: the arm enters the shared eval path with an "
    "identity filter and carries no certificate, so there is no QP and nothing can be infeasible."
)


# ---- environment probes ------------------------------------------------------------------------
def vram() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, used, free = (float(v) for v in out.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def v285_state() -> dict[str, Any]:
    """The four LIVE v2.8.5 runs: /proc state and the s/step each one MEASURES for itself, read from its
    own metrics.csv. READ-ONLY, and nothing under data/runs/v2.8.5 is touched beyond these files."""
    out: dict[str, Any] = {}
    for arm, pid in V285_PIDS.items():
        state = "gone"
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("State:"):
                    state = line.split(":", 1)[1].strip()
                    break
        except OSError:
            state = "gone"
        entry: dict[str, Any] = {"pid": pid, "proc_state": state}
        path = REPO / V285_RUNS[arm] / "metrics.csv"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",")
            i_step, i_wall = header.index("step"), header.index("wallclock_s")
            rows = [ln.split(",") for ln in lines[-6:] if ln.strip()]
            s0, w0 = float(rows[0][i_step]), float(rows[0][i_wall])
            s1, w1 = float(rows[-1][i_step]), float(rows[-1][i_wall])
            entry.update({
                "last_step": int(s1),
                "s_per_step": round((w1 - w0) / max(1.0, (s1 - s0)), 4),
                "window_steps": int(s1 - s0),
                "metrics_csv": str(path.relative_to(REPO)),
            })
        except (OSError, ValueError, IndexError) as exc:                     # pragma: no cover
            entry["metrics_error"] = str(exc)
        out[arm] = entry
    return out


def proc_state() -> dict[str, Any]:
    pid = os.getpid()
    state = ""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
                break
    except OSError:                                                          # pragma: no cover
        state = "unavailable"
    return {"pid": pid, "ppid": os.getppid(), "proc_state": state,
            "launched_with": "plain subprocess; NO setsid and NO nohup"}


# ---- the binding that lets the subclass reach the shared eval path -------------------------------
@contextlib.contextmanager
def bound_controller(flags: Reach2Flags, pa_cost: PaMppiCost):
    original = _cascade_rate_module.RateCascadeController

    class _BoundReach2(Reach2Controller):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, flags=flags, pa_cost=pa_cost, **kwargs)

    _cascade_rate_module.RateCascadeController = _BoundReach2
    try:
        yield
    finally:
        _cascade_rate_module.RateCascadeController = original


# ---- one cell ------------------------------------------------------------------------------------
def run_reach2_cell(
    cell_id: str,
    mppi_config: dict[str, Any],
    *,
    flags: Reach2Flags | None,
    lam_mode: str,
    c_crash: float,
    horizon: int,
    n_samples: int,
    tilt: np.ndarray,
    theta_ref: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Score ONE cell on the first `n_scenes` scenes of the pool of record and return its table row.

    `flags is None` selects the SHIPPED `RateCascadeController` with no patch at all — that is the
    parity reference, and it proves the OFF state by running the shipped class itself.
    """
    scale = mppi_config["cascaded"]["scale"]
    kwargs = base_kwargs(mppi_config)
    kwargs["lam_mode"] = lam_mode
    kwargs["c_crash"] = float(c_crash)
    kwargs["horizon"] = int(horizon)
    pool_path = REPO / mppi_config["screen"]["pool"]
    capture: dict[str, Any] = {}
    t0 = time.time()
    context = (
        contextlib.nullcontext() if flags is None else bound_controller(flags, PA_MPPI_V3)
    )
    with context:
        cell = run_cell(
            pool_path=pool_path,
            n_scenes=int(scale["n_scenes"]),
            ebs=int(scale["ebs"]),
            seed=int(scale["seed"]),
            n_samples=int(n_samples),
            sample_chunk=0,
            tilt_deg=tilt,
            tilt_split_deg=theta_ref,
            label=cell_id,
            out_dir=OUT_DIR,
            device=device,
            dtype=dtype,
            mppi_config=mppi_config,
            v4_columns=True,
            rate_cascade=cascade_kwargs(
                mppi_config, float(mppi_config["cascaded"]["smoke"]["rate_gain_factor"])
            ),
            capture=capture,
            **kwargs,
        )
    wall = time.time() - t0

    # ---- THE THREE REACH COLUMNS. `score_two_terminals` reads the harness's OWN outcome == "goal" for
    #      the deployed column, so the shipped predicate is CALLED and never replicated.
    roll_path = OUT_DIR / f"rollouts__{cell_id}.npz"
    save_rollouts(
        roll_path, capture["system"], capture["result"].trajectories,
        capture["result"].episode_rows, tilt_deg=tilt,
    )
    two = score_two_terminals(
        capture["system"], capture["config"], load_rollouts(roll_path), tilt_split_deg=theta_ref
    )
    block = two["ALL"]

    return assemble_row(cell_id, cell, two["ALL"], wall_s=wall,
                        flags_record=None if flags is None else flags.record(),
                        rollouts=roll_path)


def assemble_row(
    cell_id: str, cell: dict[str, Any], block: dict[str, Any], *,
    wall_s: float, flags_record: dict[str, Any] | None, rollouts: Path,
) -> dict[str, Any]:
    """The table row. ONE construction, used both by a freshly scored cell and by a cell REBUILT from its
    own artifacts on resume, so a resumed row cannot drift from a scored one."""
    reach2 = cell["cascade"].get("reach2")
    row = {
        "variant": "cascaded_rate",
        "cell_id": cell_id,
        "lam_mode": str(cell["cell"]["lambda_mode"]),
        "lam": float(cell["cell"]["lambda"]),
        "c_crash": float(cell["cell"]["C_crash"]),
        "N": int(cell["cell"]["N"]),
        "H": int(cell["cell"]["H"]),
        "reach_deployed": block["reach_deployed"],
        "reach_relaxed": block["reach_relaxed"],
        "relaxed_only_n": block["relaxed_only_n"],
        "d_min_p50": cell["v4_columns"]["ALL"]["d_min_p50"],
        "ess_p50": cell["ess"]["active"]["p50"],
        "ess_over_n": cell["ess"]["active"]["p50"] / float(cell["cell"]["N"]),
        "coll_obstacle": cell["coll_obstacle"],
        "coll_band_lower": cell["coll_band_lower"],
        "coll_band_upper": cell["coll_band_upper"],
        "coll_all": cell["collision"],
        "oob": cell["oob"],
        "stuck": cell["stuck"],
        "timeout": cell["timeout"],
        "wall_ms_per_step": 1000.0 * float(cell["wall_per_control_step_s"]),
        "preclip_out_of_box_share": cell["cascade"]["preclip_evidence"]["frac_outside_box"],
        "infeasibility": "STRUCTURALLY INAPPLICABLE",
        "mean_proj_mag": "STRUCTURALLY INAPPLICABLE",
        # audit fields, not table columns
        "_wall_s": round(float(wall_s), 1),
        "_n_control_steps": int(cell["n_control_steps"]),
        "_outcome_counts": cell["outcome_counts"],
        "_degen_step_frac": cell["degenerate"]["step_frac_over_active_window"],
        "_omega_p50": cell["v4_columns"]["ALL"]["omega_p50"],
        "_smooth_mean_du": cell["v4_columns"]["ALL"]["smooth_mean_du"],
        "_inside_r05_share": cell["v4_columns"]["ALL"]["inside_r_share"],
        "_d_min_min": block["d_min_min"],
        "_reach_deployed_rederivation_agrees": block["deployed_rederivation_agrees_with_harness"],
        "_flags": flags_record,
        "_dt_pred_s": None if reach2 is None else reach2["prediction_step"]["dt_pred_s"],
        "_real_time_horizon_s": None if reach2 is None else
            reach2["prediction_step"]["real_time_horizon_s"],
        "_planner_preclip_share": None if reach2 is None else
            reach2["in_planner_clip"]["planner_preclip_frac_out_of_box"],
        "_switches": {k: bool(cell["cell"][k]) for k in ("B1", "B2", "B3", "G1", "G2", "G3", "G4")},
        "_rollouts": str(rollouts.relative_to(REPO)),
    }
    print(f"[{cell_id}] " + json.dumps({k: v for k, v in row.items() if not k.startswith("_")}),
          flush=True)
    return row


# ---- table emission --------------------------------------------------------------------------
TABLE_COLUMNS = (
    "variant", "cell_id", "lam_mode", "c_crash", "N", "H",
    "reach_deployed", "reach_relaxed", "relaxed_only_n", "d_min_p50",
    "ess_p50", "ess_over_n",
    "coll_obstacle", "coll_band_lower", "coll_band_upper",
    "oob", "stuck", "timeout", "wall_ms_per_step", "preclip_out_of_box_share",
)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(rows: list[dict[str, Any]], columns: tuple[str, ...] = TABLE_COLUMNS) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def append_log(section: str) -> None:
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open("a", encoding="utf-8") as handle:
        handle.write(section.rstrip() + "\n\n")


def write_report(name: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if path.exists():
        path = path.with_name(f"{path.stem}__{time.strftime('%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}", flush=True)
    return path


def preamble(mppi_config: dict[str, Any], stage: str, *, device: str) -> dict[str, Any]:
    import hashlib
    pool_path = REPO / mppi_config["screen"]["pool"]
    sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if sha[:8] != POOL_SHA8:
        raise SystemExit(f"STOP: pool sha8 {sha[:8]} != {POOL_SHA8}; the pool of record is wrong.")
    free = vram()
    return {
        "version": __version__,
        "stage": stage,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "process": proc_state(),
        "device": device,
        "gpu_decision": {
            "free_mib_measured": free["free_mib"],
            "floor_mib": VRAM_FLOOR_MIB,
            "above_floor": free["free_mib"] >= VRAM_FLOOR_MIB,
            "decision": (
                "CPU ONLY. Measured free VRAM is below this project's standing 6144 MiB floor, so no "
                "pool cell of this work touches the GPU; four v2.8.5 trainings hold it."
                if free["free_mib"] < VRAM_FLOOR_MIB else "GPU permitted by the floor"
            ),
        },
        "v285_before": v285_state(),
        "pool": {"path": str(pool_path.relative_to(REPO)), "sha8": sha[:8], "sha256": sha},
        "scale": dict(mppi_config["cascaded"]["scale"]),
        "privilege_note": PRIVILEGE_NOTE,
        "no_seeding": assert_no_seeding(),
        "pa_mppi_source": PA_MPPI_V3.record(),
        "torch_threads": torch.get_num_threads(),
    }


def finish(report: dict[str, Any], t0: float) -> None:
    report["v285_after"] = v285_state()
    report["wall_s"] = round(time.time() - t0, 1)
    report["stopped"] = (
        "STOP HONOURED. No cell is selected, ranked or registered; no row is sorted by result; the "
        "n = 2000 score was NOT run; no ledger entry, no promotion and no git command."
    )


def _common(mppi_config: dict[str, Any], args: argparse.Namespace):
    torch.set_num_threads(int(args.threads))
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    theta_ref = float(mppi_config["recovery"]["theta_ref_deg"])
    tilt = spawn_tilt_deg(
        REPO / mppi_config["screen"]["pool"],
        int(mppi_config["cascaded"]["scale"]["n_scenes"]), mppi_config,
    )
    return device, dtype, theta_ref, tilt


# =================================================================================================
# STAGE: the flag-off bit-parity measurement. RUNS BEFORE ANY SCREEN.
# =================================================================================================
def stage_parity(args: argparse.Namespace, mppi_config: dict[str, Any]) -> int:
    t0 = time.time()
    report = preamble(mppi_config, "flag-off bit-parity", device=args.device)
    device, dtype, theta_ref, tilt = _common(mppi_config, args)
    base = base_kwargs(mppi_config)
    common = dict(
        mppi_config=mppi_config, lam_mode=str(base["lam_mode"]), c_crash=float(base["c_crash"]),
        horizon=int(base["horizon"]), n_samples=int(mppi_config["v5"]["base_cell"]["n_samples"]),
        tilt=tilt, theta_ref=theta_ref, device=device, dtype=dtype,
    )
    shipped = run_reach2_cell("parity_shipped", flags=None, **common)
    flagoff = run_reach2_cell("parity_flagoff", flags=Reach2Flags(), **common)

    # WALL CLOCK IS NOT A DETERMINISM METRIC and is deliberately excluded from the gate: two runs of the
    # same deterministic computation are required to produce the same NUMBERS, not to take the same time.
    # It is reported beside the gate as its own row so the exclusion is visible rather than silent.
    timing_only = ("wall_ms_per_step", "_wall_s")
    compared = [
        c for c in TABLE_COLUMNS if c not in ("variant", "cell_id") and c not in timing_only
    ] + ["_omega_p50", "_smooth_mean_du", "_inside_r05_share", "_d_min_min", "_degen_step_frac",
         "_outcome_counts", "_reach_deployed_rederivation_agrees"]
    diffs = {}
    for key in compared:
        a, b = shipped.get(key), flagoff.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diffs[key] = {"shipped": a, "flag_off": b, "abs_diff": abs(float(a) - float(b))}
        else:
            diffs[key] = {"shipped": a, "flag_off": b, "equal": a == b}
    exact = all(
        (d.get("abs_diff") == 0.0) if "abs_diff" in d else bool(d.get("equal")) for d in diffs.values()
    )
    report["parity"] = {
        "what": "the SHIPPED RateCascadeController and this module's subclass with EVERY flag at its "
                "default, same device, same dtype, same seed, same pool cell, run one after the other "
                "through the same shared eval path. The difference is MEASURED, not asserted.",
        "shipped_row": shipped, "flag_off_row": flagoff, "per_column": diffs,
        "bit_parity_exact": exact,
        "excluded_from_the_gate": {
            "columns": list(timing_only),
            "why": "wall clock is a TIMING measurement, not a determinism metric: two runs of the same "
                   "deterministic computation must produce the same NUMBERS, not take the same time.",
            "measured_anyway": {
                "wall_ms_per_step": {"shipped": shipped["wall_ms_per_step"],
                                     "flag_off": flagoff["wall_ms_per_step"]},
                "wall_s": {"shipped": shipped["_wall_s"], "flag_off": flagoff["_wall_s"]},
            },
        },
        "recorded_gpu_row": {
            "source": "docs/versions/v2.8.4/mppi_cascaded.md, the 8-cell screen, row "
                      "N1024_lam0.05_kfac1 (GPU, float32)",
            "reach_all": 0.0000, "d_min_p50": 1.8492, "inside_r05_share": 0.0001,
            "coll_all": 0.1725, "timeout_all": 0.8275, "omega_p50": 1.0696,
            "smooth_mean_du": 0.0899, "wall_ms_per_step": 77.2, "degen_step_frac": 0.0086,
            "ess_p50": 816.1,
            "note": "this work runs on the CPU because the GPU floor is not met, so the recorded row "
                    "is a DIFFERENT device and dtype stream. It is reported beside the parity pair as "
                    "a device difference, NOT as a parity target.",
        },
    }
    finish(report, t0)
    path = write_report("parity.json", report)
    print(f"bit_parity_exact = {exact}", flush=True)
    report["_artifact"] = str(path)
    return 0 if exact else 1


# =================================================================================================
# THE THREE SCREENS
# =================================================================================================
def milestone1_cells(mppi_config: dict[str, Any]) -> list[dict[str, Any]]:
    """lam_mode {relative, absolute} x c_crash {base, reference-scale} x thrust_ref {absolute,
    hover_referenced}. CONFIG ORDER, never sorted."""
    base = base_kwargs(mppi_config)
    cells = []
    for lam_mode in ("relative", "absolute"):
        for c_name, c_value in (("C1e5", float(base["c_crash"])),
                                ("Cref", float(PA_MPPI_V3.c_crash_reference_scale))):
            for tref in ("absolute", "hover_referenced"):
                short = "abs" if tref == "absolute" else "hov"
                cells.append({
                    "cell_id": f"m1_{lam_mode}_{c_name}_thr{short}",
                    "flags": Reach2Flags(thrust_ref=tref),
                    "lam_mode": lam_mode, "c_crash": c_value,
                    "horizon": int(base["horizon"]),
                    "n_samples": int(mppi_config["v5"]["base_cell"]["n_samples"]),
                })
    return cells


def milestone2_cells(mppi_config: dict[str, Any]) -> list[dict[str, Any]]:
    """The four PA-MPPI terms CUMULATIVELY, on the two lam_mode extremes. c_crash and thrust_ref stay at
    the shipped base cell because the Researcher has selected nothing."""
    base = base_kwargs(mppi_config)
    ladder = (
        ("a", dict(m2_progress=True)),
        ("ab", dict(m2_progress=True, m2_endpoint=True)),
        ("abc", dict(m2_progress=True, m2_endpoint=True, m2_vel_gate=True)),
        ("abcd", dict(m2_progress=True, m2_endpoint=True, m2_vel_gate=True, m2_du=True)),
    )
    cells = []
    for lam_mode in ("relative", "absolute"):
        for name, kw in ladder:
            cells.append({
                "cell_id": f"m2_{lam_mode}_{name}",
                "flags": Reach2Flags(**kw),
                "lam_mode": lam_mode, "c_crash": float(base["c_crash"]),
                "horizon": int(base["horizon"]),
                "n_samples": int(mppi_config["v5"]["base_cell"]["n_samples"]),
            })
    return cells


def milestone3_cells(mppi_config: dict[str, Any]) -> list[dict[str, Any]]:
    """(e) dt_pred/dt_ctrl in {1, 2, 4} x (f) {off, on}, at the LAST rung of Milestone 2's cumulative
    order (a+b+c+d) — the terminus of the stated sequence, not a selection — on BOTH lam_mode extremes,
    so no lam_mode has to be chosen either. H is set so H * dt_pred is the SAME real-time horizon on
    every row: 40 x 0.05 = 20 x 0.10 = 10 x 0.20 = 2.0 s."""
    base = base_kwargs(mppi_config)
    h0 = int(base["horizon"])
    cells = []
    for lam_mode in ("relative", "absolute"):
        for ratio in DT_PRED_RATIOS:
            for clip in (False, True):
                cells.append({
                    "cell_id": f"m3_{lam_mode}_r{ratio}_clip{'on' if clip else 'off'}",
                    "flags": Reach2Flags(
                        m2_progress=True, m2_endpoint=True, m2_vel_gate=True, m2_du=True,
                        dt_pred_ratio=ratio, in_planner_clip=clip,
                    ),
                    "lam_mode": lam_mode, "c_crash": float(base["c_crash"]),
                    "horizon": h0 // ratio,
                    "n_samples": int(mppi_config["v5"]["base_cell"]["n_samples"]),
                })
    return cells


def rebuild_row(
    cell_id: str, mppi_config: dict[str, Any], *, theta_ref: float
) -> dict[str, Any] | None:
    """Rebuild one cell's row from the artifacts THAT CELL ALREADY WROTE — no rollout, no random number,
    no controller. Returns None when either artifact is missing, in which case the caller scores the cell.

    This exists because an external process reaped the runner mid-screen. Re-scoring a cell that is
    already on disk would burn an hour of CPU beside four live trainings for numbers that are already
    recorded; rebuilding is a pure re-read. The row is assembled by `assemble_row`, THE SAME function a
    freshly scored cell uses, so a resumed row cannot differ in shape or definition from a scored one.
    """
    from src.frameworks.jt_pncbf.train import make_system

    cell_path = OUT_DIR / f"cell__{cell_id}.json"
    roll_path = OUT_DIR / f"rollouts__{cell_id}.npz"
    if not (cell_path.exists() and roll_path.exists()):
        return None
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    config = effective_config(mppi_config)
    system = make_system(config)
    two = score_two_terminals(
        system, config, load_rollouts(roll_path), tilt_split_deg=theta_ref
    )
    flags_record = None
    reach2 = cell["cascade"].get("reach2")
    if reach2 is not None:
        flags_record = reach2.get("flags")
    row = assemble_row(
        cell_id, cell, two["ALL"], wall_s=float(cell["wall_s"]),
        flags_record=flags_record, rollouts=roll_path,
    )
    row["_resumed_from_artifact"] = True
    return row


def run_screen(args: argparse.Namespace, mppi_config: dict[str, Any], stage: str) -> int:
    t0 = time.time()
    report = preamble(mppi_config, stage, device=args.device)
    device, dtype, theta_ref, tilt = _common(mppi_config, args)
    builder = {"m1": milestone1_cells, "m2": milestone2_cells, "m3": milestone3_cells}[stage]
    cells = builder(mppi_config)
    rows: list[dict[str, Any]] = []
    resumed: list[str] = []
    for spec in cells:
        row = rebuild_row(spec["cell_id"], mppi_config, theta_ref=theta_ref) if args.resume else None
        if row is not None:
            resumed.append(spec["cell_id"])
            print(f"[{spec['cell_id']}] RESUMED from its own artifacts (no rollout re-run)", flush=True)
        else:
            row = run_reach2_cell(
                spec["cell_id"], mppi_config, flags=spec["flags"], lam_mode=spec["lam_mode"],
                c_crash=spec["c_crash"], horizon=spec["horizon"], n_samples=spec["n_samples"],
                tilt=tilt, theta_ref=theta_ref, device=device, dtype=dtype,
            )
        rows.append(row)
        report["resumed_cells"] = resumed
        report["rows"] = rows
        write_report(f"{stage}_partial.json", {**report, "rows": rows})
    report["rows"] = rows
    report["columns"] = list(TABLE_COLUMNS)
    report["table_markdown"] = markdown(rows)
    finish(report, t0)
    write_report(f"{stage}.json", report)
    print("\n" + markdown(rows), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v2.8.4 second reach ladder: parity, then the three screens. Measurement only."
    )
    parser.add_argument("--stage", required=True, choices=("parity", "m1", "m2", "m3"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--resume", action="store_true",
        help="rebuild any cell whose own cell__*.json and rollouts__*.npz are already on disk instead of "
             "re-scoring it. A pure re-read: no rollout, no random number, no controller.",
    )
    args = parser.parse_args()
    mppi_config = load_mppi_config()
    if args.stage == "parity":
        return stage_parity(args, mppi_config)
    if not (OUT_DIR / "parity.json").exists():
        raise SystemExit(
            "STOP: the flag-off bit-parity measurement runs BEFORE any screen; "
            f"{OUT_DIR / 'parity.json'} does not exist."
        )
    return run_screen(args, mppi_config, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
