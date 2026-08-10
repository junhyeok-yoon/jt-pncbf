from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from src.common.brake_rollout import brake_h_rollout
from src.common.filter_hardnet import HardNetFilter, _base_alpha, _cbf_terms, _hardnet_params
from src.common.observation import top_k_obstacles
from src.common.rk4 import rk4_step
from src.common.signed_h import signed_h
from src.common.spline_max import cubic_spline_max
from src.common.system import System
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import BatchedScene
from src.frameworks.oc_pncbf.collection import TensorTransitionBatch
from src.frameworks.oc_pncbf.value_target import compute_disc_avoid_terms, pncbf_target


Tensor = torch.Tensor

# v2.8.0 S3 M3 — P2 instrumentation buffer. OFF BY DEFAULT (populated only when
# loss.policy.log_jac_classes is true). It is written under torch.no_grad() from detached tensors and
# never touches the loss, gradient, optimizer, or data path, so it is provably free of any training-path
# effect when off (the append block is skipped) and even when on (pure observation). The S3 launcher
# clears it before a run and persists it to jac_classes.csv after.
_JAC_CLASS_LOG: list[dict[str, Any]] = []


@dataclass(frozen=True)
class ValueLossResult:
    total: Tensor
    reach: Tensor
    targets: Tensor
    tail_push_mean: float = 0.0        # v2.4.2 Exp 2: mean target_rhs*relu(rhs_full-lhs) (raw_lagged only)
    tail_exceed_frac: float = 0.0      # v2.4.2 Exp 2: frac rhs_full>lhs (bootstrap tail exceeds avoid)
    # v2.8.3 D9 ROOT REPAIR. `value_loss` builds an aux dict and passes it into `value_targets`; label-path
    # branches populate it (candidate_set: cs_*; sigma_hazard: sh_*). Before this field existed the dict was
    # dropped on return and only two keys were cherry-picked into named fields above, so EVERY other
    # diagnostic was computed and discarded -- it hid the candidate-set instrumentation for the whole v2.8.2
    # CAND run and the sigma-hazard monitors at v2.8.3 launch. Carrying the whole dict fixes it at the root:
    # any future label branch that writes a key gets it logged with no further plumbing.
    aux: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyLossResult:
    total: Tensor
    task: Tensor
    action_norm: Tensor
    smoothness: Tensor
    saturation_excess: Tensor
    pretanh: Tensor
    outside: Tensor
    grad_leak: float
    action_abs_mean: Tensor
    action_abs_max: Tensor
    satfrac_a_phi: Tensor
    l_rate_raw: Tensor
    l_rate_weighted: Tensor
    mean_abs_du: Tensor
    l_u_raw: Tensor
    l_u_weighted: Tensor
    l_deficit: Tensor
    mean_deficit_active: Tensor
    deficit_active_frac: Tensor
    deficit_clip_frac: Tensor
    mean_abs_deficit_feature: Tensor
    friction_loss: Tensor          # v2.5.0 B-2: weighted filter-friction term w_friction*||u_safe-u_nom||^2
    proj_mag_bptt: Tensor          # mean ||u_safe - u_nom|| over the BPTT window (all-active gauge diagnostic)
    l_du_raw: Tensor               # v2.8.2 M3: raw ||Δu^filt||^2/h^2 (logged even when w_du=0 — a monitor)
    l_du_weighted: Tensor          # v2.8.2 M3: w_du * l_du (0 when off)
    l_infeas_raw: Tensor           # v2.8.2 M2: mean relu(s_inf+delta) (0 when w_infeas=0; computed only when on)
    l_infeas_weighted: Tensor      # v2.8.2 M2: w_infeas * l_infeas (0 when off)
    l_floor_recov_raw: Tensor      # v2.8.2 cond(c): mean proximity-weighted tilt excess (0 when w_floor_recov=0)
    l_floor_recov_weighted: Tensor # v2.8.2 cond(c): w_floor_recov * L_floor_recov (0 when off)
    l_agree_raw: Tensor            # v2.8.2 agree: E[mask_nonempty * ||pi - sg(u_filt)||^2] (0 when w_agree=0)
    l_agree_weighted: Tensor       # v2.8.2 agree: w_agree * l_agree_raw (0 when off)


def value_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
    recovery_policy: Any = None,
    lagged_policy: Any = None,
) -> ValueLossResult:
    obs = system.observation(batch.states, batch.scene)
    aux: dict[str, float] = {}
    targets = value_targets(
        system=system,
        target_value_net=target_value_net,
        batch=batch,
        lambda_disc=lambda_disc,
        target_rhs=target_rhs,
        config=config,
        recovery_policy=recovery_policy,
        lagged_policy=lagged_policy,
        aux=aux,
    )
    prediction = value_net(obs)
    reach = torch.mean((prediction - targets.unsqueeze(1)) ** 2)
    total = float(config["loss"]["value"]["lambda_R"]) * reach
    return ValueLossResult(total=total, reach=reach, targets=targets,
                           tail_push_mean=aux.get("tail_push_mean", 0.0),
                           tail_exceed_frac=aux.get("tail_exceed_frac", 0.0),
                           aux=dict(aux))


def _value_ceiling(config) -> float:
    """v2.8.4 ceiling axis — the SINGLE config key, shared with the read-out (value_net.py).
    Default 1.0 reproduces the shipped label bit for bit. The lower bound stays at -1.0."""
    return float((config.get("network", {}) or {}).get("value", {}).get("ceiling", 1.0))


def value_targets(
    *,
    system: System,
    target_value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    lambda_disc: float,
    target_rhs: float,
    config: Mapping[str, Any],
    recovery_policy: Any = None,
    lagged_policy: Any = None,
    aux: dict[str, float] | None = None,
) -> Tensor:
    conditioning = config["value_target"].get("conditioning", "task_stored")
    if conditioning == "task_raw_lagged":
        # v2.4.2: condition the value label on UNFILTERED, noise-free, deterministic re-rolls of a
        # Polyak-lagged copy pi_b of the task policy. Restores the PNCBF Thm-1 object V^{h,pi_b} (a
        # sound certificate for u = pi_b(x)), unlike task_stored which regresses the residual risk of
        # its own V-filtered system. Same target FORM as brake/learned_recovery; only the conditioning
        # rollout policy differs. NO HardNet filter and NO exploration noise (brake_h_rollout applies
        # neither); pi_b output is bounded by the control-net map and additionally clamped to u_bounds.
        if lagged_policy is None:
            raise ValueError("task_raw_lagged conditioning requires a lagged_policy (pi_b).")
        rl_cfg = config["value_target"]["raw_lagged"]
        ub = system.u_bounds.to(device=batch.states.device, dtype=batch.states.dtype)
        u_lo, u_hi = ub[:, 0], ub[:, 1]

        def _pi_b_rollout(x_state: Tensor, obs_state: Tensor) -> Tensor:
            return torch.clamp(lagged_policy(obs_state), min=u_lo, max=u_hi)

        h_seq_bt, tail_obs = brake_h_rollout(
            batch.states,
            batch.scene,
            system,
            system.observation,
            int(rl_cfg["T_b"]),
            0.0,   # u_max unused when policy_fn is given
            0.0,   # eps_v unused when policy_fn is given
            float(config["env"]["dt"]),
            float(config["env"]["h_scale"]),
            policy_fn=_pi_b_rollout,
        )
        with torch.no_grad():
            bootstrap_tail = target_value_net.target_h(tail_obs)
        raw_h_sequence = h_seq_bt.transpose(0, 1).contiguous()   # [T_b+1, B]; index 0 = minibatch state
        dt = float(config["env"]["dt"])
        if aux is not None:
            # v2.4.2 Exp 2 tail diagnostics at the label origin (index 0): the one-sided tail push
            # target_rhs*relu(rhs_full - lhs) and the exceed fraction, from the same disc-avoid terms
            # pncbf_target uses internally (recomputed here; cheap vs the re-roll). No effect on the label.
            with torch.no_grad():
                costs_d = torch.clamp(raw_h_sequence, -1.0, 1.0)
                lhs_d, int_rhs_d, disc_d = compute_disc_avoid_terms(costs_d, lambda_disc, dt)
                rhs_full0 = int_rhs_d[0] + disc_d[0] * bootstrap_tail
                push0 = float(target_rhs) * torch.relu(rhs_full0 - lhs_d[0])
                aux["tail_push_mean"] = float(push0.mean().item())
                aux["tail_exceed_frac"] = float((rhs_full0 > lhs_d[0]).to(push0.dtype).mean().item())
        return pncbf_target(
            raw_h_sequence,
            lambda_disc,
            dt,
            target_rhs,
            bootstrap_tail,
            ceiling=_value_ceiling(config),
        ).detach()[0]
    if conditioning == "learned_recovery":
        # v2.4.0 Step 2: condition the value target on the LEARNED recovery policy (Polyak copy
        # pi_b_target). Same target form as the brake branch; only the conditioning rollout policy
        # differs. Rollout + result are grad-free (no_grad inside brake_h_rollout, then .detach()).
        if recovery_policy is None:
            raise ValueError("learned_recovery conditioning requires a recovery_policy (pi_b_target).")
        brake_cfg = config["value_target"]["brake"]
        rec_cfg = config["value_target"]["recovery"]
        u_max = float(config["env"]["bounds"][system.name]["u_max"])
        h_seq_bt, tail_obs = brake_h_rollout(
            batch.states,
            batch.scene,
            system,
            system.observation,
            int(rec_cfg["T_b"]),
            u_max,
            float(brake_cfg["eps_v"]),
            float(config["env"]["dt"]),
            float(config["env"]["h_scale"]),
            policy_fn=recovery_policy,
        )
        with torch.no_grad():
            bootstrap_tail = target_value_net.target_h(tail_obs)
        brake_h_sequence = h_seq_bt.transpose(0, 1).contiguous()
        return pncbf_target(
            brake_h_sequence,
            lambda_disc,
            float(config["env"]["dt"]),
            target_rhs,
            bootstrap_tail,
            ceiling=_value_ceiling(config),
        ).detach()[0]
    if conditioning == "brake":
        # v2.4.0: decouple the conditioning policy. Roll a fixed analytic brake from each
        # minibatch state and run the SAME pncbf_target recurrence over that h-sequence.
        # Behavior (where states come from) is unchanged; only the label provenance changes.
        brake_cfg = config["value_target"]["brake"]
        u_max = float(config["env"]["bounds"][system.name]["u_max"])
        h_seq_bt, tail_obs = brake_h_rollout(
            batch.states,
            batch.scene,
            system,
            system.observation,
            int(brake_cfg["T_b"]),
            u_max,
            float(brake_cfg["eps_v"]),
            float(config["env"]["dt"]),
            float(config["env"]["h_scale"]),
        )
        with torch.no_grad():
            bootstrap_tail = target_value_net.target_h(tail_obs)
        # time-major [T_b+1, B] for the shared recurrence; index 0 is the value AT the
        # minibatch state (start of the braking rollout).
        brake_h_sequence = h_seq_bt.transpose(0, 1).contiguous()
        return pncbf_target(
            brake_h_sequence,
            lambda_disc,
            float(config["env"]["dt"]),
            target_rhs,
            bootstrap_tail,
            ceiling=_value_ceiling(config),
        ).detach()[0]
    with torch.no_grad():
        tail_obs = system.observation(batch.tail_states, batch.tail_scene)
        bootstrap_tail = target_value_net.target_h(tail_obs)
    # v2.8.2 CANDIDATE-SET LABEL (theory.tex def:cand), behind value_target.candidate_set.enabled
    # (DEFAULT FALSE => the shipped path below runs untouched and byte-identically).
    _cs = (config["value_target"].get("candidate_set") or {})
    if bool(_cs.get("enabled", False)):
        return _candidate_set_target(
            system=system, target_value_net=target_value_net, batch=batch,
            lambda_disc=lambda_disc, target_rhs=target_rhs, config=config,
            bootstrap_tail=bootstrap_tail, cs_cfg=_cs, aux=aux,
        )
    # v2.8.3 SIGMA-AUGMENTED HAZARD, behind value_target.sigma_hazard.enabled (DEFAULT FALSE => the
    # shipped call below runs untouched and byte-identically; the h_sequence handed to pncbf_target is
    # the same object).
    _sh = (config["value_target"].get("sigma_hazard") or {})
    h_seq = batch.h_sequence
    if bool(_sh.get("enabled", False)):
        h_seq = _sigma_hazard_sequence(
            system=system, target_value_net=target_value_net, batch=batch,
            config=config, sh_cfg=_sh, aux=aux,
        )
    targets = pncbf_target(
        h_seq,
        lambda_disc,
        float(config["env"]["dt"]),
        target_rhs,
        bootstrap_tail,
        ceiling=_value_ceiling(config),
    ).detach()
    return targets.gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)


def _sigma_hazard_sequence(*, system, target_value_net, batch, config, sh_cfg, aux=None):
    """v2.8.3 — h_tilde = max(h, beta * psi(sigma)) on the value target's hazard ONLY.

        sigma(x) = alpha_eff(x)*V_t(x) + Lf_V_t(x) - u_max * ||(Lg V_t(x))^-||_1
        psi(s)   = clip(s / s_scale, -1, 1)          (TWO-SIDED; the one-sided variant is rejected)

    sigma is recomputed FRESH here from batch.state_sequence with the TARGET net -- never from the
    collection stash last_b_live, which is one target-net update stale.

    CLIP MASK (prop:ceiling): on {|V_t| >= 1 - clip_tol} sigma = alpha*V_max is a parameterisation
    artifact, not a feasibility fact, so psi is not applied and h_tilde = h there.

    sigma is ONE backward of V_t per state. Time is FLATTENED into the batch (B-major, matching
    _tile_scene's repeat_interleave) so the whole [T+1, B] block costs one backward per chunk rather
    than one per index.
    """
    from src.common.filter_hardnet import _cbf_terms, _base_alpha, _hardnet_params
    from src.common.kstep_fallback import _tile_scene

    xs = batch.state_sequence
    if xs is None:
        raise ValueError("sigma_hazard requires batch.state_sequence (buffer store_state_seq=True).")
    beta = float(sh_cfg["beta"]); m = float(sh_cfg["m"]); s_scale = float(sh_cfg["s_scale"])
    clip_tol = float(sh_cfg.get("clip_tol", 1.0e-6))
    # v2.8.3 AMENDMENT-4 REPAIR A. `sigma_alpha_rule` selects the alpha used to form the LABEL's sigma.
    #   "deployed"     -- the two-valued filter rule (alpha_safe if V<=0 else alpha_unsafe). This is what
    #                     the first arm ran, and it COLLAPSED: alpha jumps 2 -> 100 the instant V crosses
    #                     0, so the label recursion's uniform-inflation gain is
    #                     beta*alpha_unsafe/s_scale = 1.0*100/2.4284 = 41.2 >> 1 -- a runaway with a
    #                     degenerate fixed point at the label ceiling (E[h~-h] 0.041 -> 1.22, P(sigma>0)
    #                     0.068 -> 0.998, reach 0.0).
    #   "uniform_safe" -- alpha_ref = alpha_safe = 2.0 everywhere. On {V<=0}, the sliding band this axis
    #                     targets, the deployed alpha_eff IS 2.0, so sigma_label equals the deployed
    #                     sigma EXACTLY there; divergence is confined to {V>0}, where the alpha=100
    #                     demand is the cor:alpha-thresh pathology this deliberately declines to import.
    #                     Gain becomes alpha_ref/s_scale = 0.824 < 1: a contraction.
    # The DEPLOYED filter, its row and its empty test are untouched by this switch.
    alpha_rule = str(sh_cfg.get("sigma_alpha_rule", "deployed"))
    if alpha_rule not in ("deployed", "uniform_safe"):
        raise ValueError(f"sigma_hazard.sigma_alpha_rule must be 'deployed' or 'uniform_safe', got {alpha_rule!r}.")
    chunk_rows = int(sh_cfg.get("chunk_rows", 65536))
    if s_scale <= 0.0:
        raise ValueError(f"sigma_hazard.s_scale must be positive, got {s_scale}.")
    params = _hardnet_params(config)
    costs = torch.clamp(batch.h_sequence, min=-1.0, max=_value_ceiling(config))   # [T+1, B]
    Tp1, B = costs.shape
    sdim = xs.shape[2]
    u_hi = system.u_bounds.to(device=xs.device, dtype=xs.dtype)[:, 1]
    adim = int(system.u_bounds.shape[0])

    X = xs.permute(1, 0, 2).reshape(B * Tp1, sdim)                    # B-major, matches _tile_scene
    sc = _tile_scene(batch.scene, Tp1, B)
    sig = torch.empty(B * Tp1, dtype=xs.dtype, device=xs.device)
    vv = torch.empty(B * Tp1, dtype=xs.dtype, device=xs.device)

    def h_t(xx, scene):
        return target_value_net.target_h(system.observation(xx, scene)).reshape(-1)

    for i in range(0, B * Tp1, chunk_rows):
        j = min(i + chunk_rows, B * Tp1)
        sl = slice(i, j)
        sc_i = _slice_scene_rows(sc, sl)
        v, lf_v, lg_v = _cbf_terms(system, h_t, X[sl], sc_i,
                                   torch.zeros((j - i, adim), device=xs.device, dtype=xs.dtype),
                                   create_graph=False)
        supply = torch.sum(torch.clamp(lg_v, max=0.0).abs() * u_hi.unsqueeze(0), dim=1)
        a_eff = (_base_alpha(v, params) if alpha_rule == "deployed"
                 else torch.full_like(v, float(params.alpha_safe)))
        sig[sl] = (a_eff * v + lf_v - supply).detach()
        vv[sl] = v.detach()

    sigma = sig.reshape(B, Tp1).t().contiguous()                      # [T+1, B]
    vval = vv.reshape(B, Tp1).t().contiguous()
    psi = torch.clamp((sigma + m) / s_scale, min=-1.0, max=1.0)
    off_clip = vval.abs() < (1.0 - clip_tol)
    h_tilde = torch.where(off_clip, torch.maximum(costs, beta * psi), costs)
    if aux is not None:
        d = (h_tilde - costs)
        aux["sh_label_mass_shift"] = float(d.mean().item())
        aux["sh_raised_frac"] = float((d > 1e-9).to(costs.dtype).mean().item())
        aux["sh_clip_masked_frac"] = float((~off_clip).to(costs.dtype).mean().item())
        aux["sh_sigma_p50"] = float(sigma.median().item())
        aux["sh_sigma_pos_frac"] = float((sigma > 0).to(costs.dtype).mean().item())
        aux["sh_alpha_rule_uniform"] = 1.0 if alpha_rule == "uniform_safe" else 0.0
    return h_tilde


def _slice_scene_rows(scene, sl):
    import dataclasses
    upd = {}
    for f in dataclasses.fields(scene):
        v = getattr(scene, f.name)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            upd[f.name] = v[sl]
    return dataclasses.replace(scene, **upd)


def _candidate_set_target(*, system, target_value_net, batch, lambda_disc, target_rhs, config,
                          bootstrap_tail, cs_cfg, aux=None):
    """v2.8.2 AMENDMENT-2 — IN-RECURSION def:cand label. The min is installed INSIDE the shipped
    T-step recursion at the SAME depth, with the SAME push and clamp:

        lhs[idx] = max( h_idx, (1-g)*h_idx + g*min( lhs[idx+1], min_{u in grid} V_targ(Phi_dt(x_idx,u)) ) )

    lhs[idx+1] IS the pi_f branch: it is the continuation of the trajectory the DEPLOYED filtered loop
    produced, so pi_f is in Q by construction and needs no separate evaluation (prop:label-order(a)).
    Grid successors are batched over candidates; the loop is over TIME only (memory-bound), never over
    candidates (05_code). With an EMPTY grid this reduces to the shipped recursion exactly."""
    from src.common.kstep_fallback import grid_controls, _tile_scene
    from src.common.rk4 import rk4_step
    dt = float(config["env"]["dt"])
    xs = batch.state_sequence
    if xs is None:
        raise ValueError("candidate_set requires batch.state_sequence (buffer store_state_seq=True).")
    include_grid = bool(cs_cfg.get("include_grid", True))
    with torch.no_grad():
        costs = torch.clamp(batch.h_sequence, min=-1.0, max=_value_ceiling(config))   # [T+1, B]
        Tp1, B = costs.shape
        dev, dty = costs.device, costs.dtype
        gamma = torch.exp(-torch.as_tensor(lambda_disc, dtype=dty, device=dev) * dt)
        one_minus = 1.0 - gamma
        # --- per-index grid minimum, batched over candidates, looped over time ---
        gmin = None
        if include_grid:
            Gc = grid_controls(system, dev, dty)
            ng, adim = int(Gc.shape[0]), int(Gc.shape[1])
            sc = _tile_scene(batch.scene, ng, B)
            Uf = Gc.unsqueeze(0).expand(B, ng, adim).reshape(B * ng, adim)
            gmin = torch.empty((Tp1, B), dtype=dty, device=dev)
            for t in range(Tp1):
                X = xs[t].unsqueeze(1).expand(B, ng, xs.shape[2]).reshape(B * ng, xs.shape[2])
                Xn = system.wrap_state(rk4_step(system, X, Uf, dt))
                gmin[t] = target_value_net.target_h(system.observation(Xn, sc)).reshape(B, ng).min(dim=1).values
        # --- the shipped recursion, with the min folded into the continuation ---
        lhs = torch.zeros_like(costs)
        lhs[-1] = costs[-1]
        n_disp = torch.zeros((), dtype=dty, device=dev); n_tot = 0.0
        gaps = []
        for idx in range(Tp1 - 2, -1, -1):
            cont = lhs[idx + 1]
            if gmin is not None:
                g_t = gmin[idx]
                if aux is not None:
                    n_disp = n_disp + (g_t < cont).to(dty).sum(); n_tot += B
                    gaps.append(g_t - cont)
                cont = torch.minimum(cont, g_t)
            lhs[idx] = torch.maximum(costs[idx], one_minus * costs[idx] + gamma * cont)
        # --- push + clamp, EXACTLY as pncbf_target ships them (I2) ---
        _, int_rhs, disc_rhs = compute_disc_avoid_terms(costs, lambda_disc, dt)
        rhs_full = int_rhs + disc_rhs * bootstrap_tail.unsqueeze(0)
        mixed = lhs + torch.as_tensor(target_rhs, dtype=dty, device=dev) * torch.relu(rhs_full - lhs)
        out = torch.clamp(mixed, min=-1.0, max=_value_ceiling(config))
        if aux is not None and gmin is not None and n_tot > 0:
            aux["cs_argmin_not_pif_frac"] = float((n_disp / n_tot).item())
            gp = torch.cat(gaps)
            for p in (10, 50, 90):
                aux[f"cs_gap_p{p}"] = float(torch.quantile(gp.double(), p / 100.0).item())
            aux["cs_n_candidates"] = float(1 + (0 if gmin is None else int(grid_controls(system, dev, dty).shape[0])))
    return out.detach().gather(0, batch.step_indices.unsqueeze(0)).squeeze(0)


@dataclass(frozen=True)
class CBFDerivLossResult:
    loss_raw: Tensor               # unweighted mean ReLU(m + gamma) over kept gate states
    diagnostics: dict[str, float]  # gate_* + lg_norm_gate_* scalars for logging


def cbf_deriv_feasibility_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    scene: BatchedScene,
    config: Mapping[str, Any],
    generator: torch.Generator | None = None,
    collect_diagnostics: bool = True,
) -> CBFDerivLossResult:
    """v2.2.1 box-feasibility CBF-derivative auxiliary term L_feas (value-side; oracle-free).

    Builds a fresh gate set S_gate of near-boundary fast-approach states on the given (real) scene
    layouts, drops doomed states (unavoidable-collision predicate), then restricts to the recoverable
    boundary via an h-band mask (deployed h in [h_band_lo, h_band_hi]), and penalizes states where no
    in-box action satisfies the descent condition:
        m(x) = L_f h + alpha(h) h - u_max * ||L_g h||_1 ,  L_feas = mean ReLU(m + gamma_strict).
    Reducible only by growing ||L_g h|| = ||dh/dv|| in the braking direction. Uses the DEPLOYED
    (mean-ensemble) h via make_h_fn + _cbf_terms; never the analytic 2(p-c).
    """
    cfg = config["loss"]["value"]["cbf_deriv"]
    gamma_strict = float(cfg["gamma_strict"])
    d_lo, d_gate = float(cfg["d_lo"]), float(cfg["d_gate"])
    s_lo, s_hi = float(cfg["s_lo"]), float(cfg["s_hi"])
    lateral_frac = float(cfg["lateral_frac"])
    exclude_doomed = bool(cfg["exclude_doomed"])
    # v2.2.1: recoverable-boundary h-band. Absent keys -> [-inf, +inf] (no band filter; backward compat).
    h_band_lo = float(cfg.get("h_band_lo", float("-inf")))
    h_band_hi = float(cfg.get("h_band_hi", float("inf")))
    di = config["env"]["bounds"]["double_integrator"]
    u_max = float(di["u_max"])
    v_max = float(di["v_max"])
    delta_feas = float(config["scene_train"]["init_feasibility_margin"])

    centers = scene.obstacle_centers          # [B, nmax, 2]
    radii = scene.obstacle_radii              # [B, nmax]
    active = scene.obstacle_active            # [B, nmax] bool
    B = centers.shape[0]
    device, dtype = centers.device, centers.dtype
    eps = 1.0e-9

    # placement obstacle: one random ACTIVE obstacle per scene (n_min=1 guarantees >=1 active)
    weights = active.to(dtype)
    weights = torch.where(weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights))
    sel = torch.multinomial(weights, 1, generator=generator).squeeze(1)        # [B]
    bidx = torch.arange(B, device=device)
    c0 = centers[bidx, sel]                                                     # [B,2]
    r0 = radii[bidx, sel]                                                       # [B]

    g = torch.randn(B, 2, generator=generator, device=device, dtype=dtype)
    u_rad = g / torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)        # outward unit dir
    d = d_lo + (d_gate - d_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    p = c0 + (r0 + d).unsqueeze(1) * u_rad                                      # [B,2]
    s = s_lo + (s_hi - s_lo) * torch.rand(B, generator=generator, device=device, dtype=dtype)
    v_in = -s.unsqueeze(1) * u_rad
    perp = torch.stack([-u_rad[:, 1], u_rad[:, 0]], dim=1)
    lat = (lateral_frac * s) * (2.0 * torch.rand(B, generator=generator, device=device, dtype=dtype) - 1.0)
    v = v_in + lat.unsqueeze(1) * perp
    vnorm = torch.linalg.norm(v, dim=1, keepdim=True)
    v = torch.where(vnorm > v_max, v * (v_max / vnorm.clamp_min(eps)), v)       # env velocity clamp
    x = torch.cat([p, v], dim=1)                                               # [B,4]

    # realized nearest active obstacle (env helper) for the doomed predicate
    top_rel, top_radii = top_k_obstacles(p, centers, radii, active, 1)
    cp = top_rel[:, 0, :]                                                       # c - p
    rr = top_radii[:, 0]
    dist = torch.linalg.norm(cp, dim=1)
    surf_clr = dist - rr
    v_close = torch.relu(torch.sum(v * cp, dim=1) / dist.clamp_min(eps))
    d_stop = v_close * v_close / (2.0 * u_max)
    doomed = surf_clr < (d_stop + delta_feas)                                   # unavoidable-collision
    keep_recoverable = ~doomed if exclude_doomed else torch.ones_like(doomed)

    # descent-condition margin on the DEPLOYED h; create_graph so gradients reach V_S params.
    # h is computed ONCE here and reused both for m(x) and for the h-band mask (no second forward).
    params = _hardnet_params(config)
    h_fn = make_h_fn(value_net, system)
    u_zero = torch.zeros(B, system.action_dim, device=device, dtype=dtype)
    h, lf_h, lg_h = _cbf_terms(system, h_fn, x, scene, u_zero, create_graph=True)
    alpha = _base_alpha(h, params)
    m = lf_h + alpha * h - u_max * torch.sum(torch.abs(lg_h), dim=1)            # u_max*||L_g h||_1
    viol = torch.relu(m + gamma_strict)

    # v2.2.1 h-band: ADDITIONAL filter (after doomed-exclusion) restricting S_gate to the
    # recoverable boundary, deployed h in [h_band_lo, h_band_hi]. Same deployed h as in m(x).
    h_det = h.detach()
    in_band = (h_det >= h_band_lo) & (h_det <= h_band_hi)
    keep = keep_recoverable & in_band                                          # final S_gate mask
    keepf = keep.to(dtype)
    # Masked mean with a >=1 denominator: numerically identical to sum(viol*keepf)/keepf.sum() when any
    # state is kept (clamp is the identity since keepf.sum() is an integer count >= 1), and 0 when none
    # are kept -- so the loss has NO per-step host sync (no int(keep.sum().item()) control-flow branch).
    loss_raw = torch.sum(viol * keepf) / keepf.sum().clamp_min(1.0)

    if collect_diagnostics:
        # pure-logging scalars (host syncs) -- computed only on metrics-logging-cadence steps.
        n_kept = int(keep.sum().item())
        if n_kept > 0:
            active_frac = float((((m + gamma_strict) > 0) & keep).sum().item()) / n_kept
            lg_sel = torch.linalg.norm(lg_h, dim=1)[keep].detach()
            lg_mean = float(lg_sel.mean().item())
            lg_p50 = float(lg_sel.median().item())
        else:
            active_frac, lg_mean, lg_p50 = 0.0, 0.0, 0.0
        diagnostics = {
            "gate_n_constructed": float(B),
            "gate_n_kept": float(n_kept),                                      # AFTER doomed AND h-band
            "gate_doomed_frac": (float(doomed.sum().item()) / B) if B > 0 else 0.0,  # true doomed frac
            "gate_active_frac": float(active_frac),
            "lg_norm_gate_mean": lg_mean,
            "lg_norm_gate_p50": lg_p50,
        }
    else:
        diagnostics = {}
    return CBFDerivLossResult(loss_raw=loss_raw, diagnostics=diagnostics)


@dataclass(frozen=True)
class SobolevLossResult:
    loss_raw: Tensor               # unweighted mean ||grad_x V_S - grad_x V_spline||^2 over the batch
    diagnostics: dict[str, float]  # sob_* scalars for logging


def _overspeed_gate(
    system: System, x: Tensor, scene: BatchedScene, gate_cfg: Mapping[str, Any], eps: float = 1.0e-9,
) -> tuple[Tensor, Tensor, Tensor]:
    """Smooth over-speed gate g(x) in [0,1], higher the more over-speed (DI). Computed by the caller
    under no_grad (a weighting, not part of the differentiated objective).

      g(x) = sigmoid((inward_speed - s0)/tau_s) * sigmoid((c0 - clearance)/tau_c)

    inward_speed = max(0, v . n_hat), n_hat the unit vector from the agent to the NEAREST active obstacle
    center; clearance = nearest-obstacle surface distance. Nearest-obstacle geometry mirrors signed_h
    (rel = center - p, distance = ||rel||, clearance = distance - radius, inactive -> +inf). Fires only
    when BOTH fast-approaching AND near an obstacle; receding/slow/far -> g ~ 0 (term inactive there)."""
    s0 = float(gate_cfg.get("s0", 1.0)); tau_s = float(gate_cfg.get("tau_s", 0.5))
    c0 = float(gate_cfg.get("c0", 0.3)); tau_c = float(gate_cfg.get("tau_c", 0.15))
    p = system.position(x)                                              # [B,2]
    v = x[:, 2:4]                                                       # velocity [B,2] (double integrator)
    centers = torch.as_tensor(scene.obstacle_centers, dtype=x.dtype, device=x.device)
    radii = torch.as_tensor(scene.obstacle_radii, dtype=x.dtype, device=x.device)
    active = torch.as_tensor(scene.obstacle_active, dtype=torch.bool, device=x.device)
    rel = centers - p.unsqueeze(-2)                                    # agent -> center [B,K,2]
    distance = torch.linalg.norm(rel, dim=-1)                          # [B,K]
    clearance_all = torch.where(active, distance - radii, torch.full_like(distance, float("inf")))
    nearest = torch.argmin(clearance_all, dim=-1)                      # [B]
    bidx = torch.arange(x.shape[0], device=x.device)
    clearance = clearance_all[bidx, nearest]                          # [B]
    rel_n = rel[bidx, nearest]                                         # [B,2] toward nearest center
    n_hat = rel_n / torch.linalg.norm(rel_n, dim=-1, keepdim=True).clamp_min(eps)
    inward = torch.clamp(torch.sum(v * n_hat, dim=-1), min=0.0)        # [B] inward approach speed
    g = torch.sigmoid((inward - s0) / tau_s) * torch.sigmoid((c0 - clearance) / tau_c)
    return g, inward, clearance


def spline_sobolev_loss(
    *,
    system: System,
    value_net: ValueNetEnsemble,
    policy_net: nn.Module | None,
    states: Tensor,
    scene: BatchedScene,
    config: Mapping[str, Any],
    collect_diagnostics: bool = True,
) -> SobolevLossResult:
    """Route D: RPCBF cubic-spline-max Sobolev gradient-matching on V_S (value-side, oracle-free).

    Label: roll the (unfiltered) LQR / detached policy H steps from x0, take the cubic-spline
    max-over-time of the ground-truth signed_h sequence (recovers the continuous-time dV/dv that a
    discrete max staircases away), and read its STATE Jacobian grad_x V_spline by autograd (1st-order,
    detached -- a fixed regression target). Prediction: grad_x of the DEPLOYED (mean-ensemble) h via the
    same make_h_fn path the HardNet filter consumes, with create_graph so the term backprops into V_S
    params (clones the L_feas second-order pattern). L_sobolev = mean ||grad_pred - grad_label||^2 over
    the full state vector (the velocity pair feeds L_g h, the position pair feeds L_f h).
    """
    sob_cfg = config["loss"]["value"]["sobolev"]
    horizon = int(sob_cfg["H"])
    rollout_policy = str(sob_cfg.get("rollout_policy", "lqr"))
    dt = float(config["env"]["dt"])
    h_scale = float(config["env"]["h_scale"])

    x0 = system.wrap_state(states.detach()).detach().requires_grad_(True)
    with torch.enable_grad():
        x = x0
        h_seq = [signed_h(system.position(x), scene, h_scale)]
        for _ in range(horizon):
            if rollout_policy == "lqr":
                u = system.lqr_action(x, _scene_goal(scene, x))
            elif rollout_policy == "policy_detached":
                if policy_net is None:
                    raise ValueError("sobolev rollout_policy='policy_detached' requires a policy_net.")
                # v2.8.3 U-PREV AXIS: this sobolev rollout has no action history and is INACTIVE in the
                # CTRL config (loss.sobolev is null). Rather than invent a u_prev convention for a path
                # the axis was never registered against, refuse the combination loudly.
                from src.common.observation import u_prev_feedback_on as _uprev_on
                if _uprev_on(config):
                    raise NotImplementedError(
                        "loss.sobolev rollout_policy='policy_detached' is not supported with "
                        "obs.u_prev_feedback (the policy is dim obs_dim+action_dim here)."
                    )
                u = policy_net(system.observation(x, scene)).detach()
            else:
                raise ValueError(f"Unknown sobolev rollout_policy {rollout_policy!r}.")
            x = rk4_step(system, x, u, dt)
            h_seq.append(signed_h(system.position(x), scene, h_scale))
        h_stack = torch.stack(h_seq, dim=1)                                  # [B, H+1]
        t_grid = torch.linspace(0.0, 1.0, horizon + 1, device=x0.device, dtype=x0.dtype)
        v_spline, _ = cubic_spline_max(t_grid, h_stack)                      # [B]
        grad_label = torch.autograd.grad(v_spline.sum(), x0)[0].detach()     # [B, state_dim], fixed target

        h_fn = make_h_fn(value_net, system)
        x_pred = x0.detach().requires_grad_(True)
        h_pred = h_fn(x_pred, scene).reshape(-1)
        grad_pred = torch.autograd.grad(h_pred.sum(), x_pred, create_graph=True)[0]

    # gated, velocity-only variant (gate.enabled=false -> original global full-vector Sobolev, bit-identical)
    gate_cfg = sob_cfg.get("gate", {}) or {}
    gate_enabled = bool(gate_cfg.get("enabled", False))
    velocity_only = bool(gate_cfg.get("velocity_only", False))
    if gate_enabled and velocity_only:
        diff = grad_pred[:, 2:] - grad_label[:, 2:]                       # velocity channels only (DI: vx,vy)
    else:
        diff = grad_pred - grad_label                                    # full state vector (plain Sobolev)
    per_state = torch.sum(diff * diff, dim=1)                             # [B]
    if gate_enabled:
        with torch.no_grad():
            g, gate_inward, gate_clear = _overspeed_gate(system, x0.detach(), scene, gate_cfg)
        loss_raw = torch.mean(g * per_state)                             # localized: only over-speed states
    else:
        g = None
        loss_raw = torch.mean(per_state)

    if collect_diagnostics:
        gl_v = torch.linalg.norm(grad_label[:, 2:], dim=1)
        gp_v = torch.linalg.norm(grad_pred[:, 2:].detach(), dim=1)
        if gate_enabled:
            gate_mean = float(g.mean().item())
            gate_frac_active = float((g > 0.5).to(g.dtype).mean().item())
            denom = float(g.sum().clamp_min(1.0e-9).item())
            gate_dhdv = float((g * gp_v).sum().item() / denom)           # dh/dv_pred weighted by the gate
        else:
            gate_mean = 0.0; gate_frac_active = 0.0; gate_dhdv = float(gp_v.mean().item())
        diagnostics = {
            "sob_grad_label_v_frac": float((gl_v > 1.0e-6).to(gl_v.dtype).mean().item()),
            "sob_grad_label_v_mean": float(gl_v.mean().item()),
            "sob_dhdv_pred_mean": float(gp_v.mean().item()),
            "sob_v_spline_mean": float(v_spline.detach().mean().item()),
            "sob_gate_mean": gate_mean,
            "sob_gate_frac_active": gate_frac_active,
            "sob_gate_dhdv_pred": gate_dhdv,
        }
    else:
        diagnostics = {}
    return SobolevLossResult(loss_raw=loss_raw, diagnostics=diagnostics)


def policy_bptt_loss(
    *,
    system: System,
    policy_net: nn.Module,
    value_net: ValueNetEnsemble,
    batch: TensorTransitionBatch,
    config: Mapping[str, Any],
    step: int = 0,
    critic_net: nn.Module | None = None,
) -> PolicyLossResult:
    policy_cfg = config["loss"]["policy"]
    dt = float(config["env"]["dt"])
    bptt_t = int(config["training"]["jt"]["bptt_T"])
    gamma_t = float(policy_cfg["gamma_T"])
    lambda_v = float(policy_cfg["lambda_v"])
    mu_u = float(policy_cfg["mu_u"])
    tau_gate = float(policy_cfg["tau_gate"])
    # v2.6.2: situation-dependent velocity objective (running-cost redesign, changes.md §4). QUADROTOR ONLY
    # — the obstacle-approach term needs the linear-velocity VECTOR (x[:,3:5]), whose layout is
    # quadrotor-specific; DI/unicycle keep the legacy quadratic d2 cost (byte-identical parity). Activated
    # only when a new key is set AND system is quadrotor_planar; each added term is weight-gated (w_settle=0
    # or w_appr=0 -> inert, for the goal-only ablation).
    # goal DISTANCE stays quadratic ||p-g||^2 (the Huber variant was reverted per the build-log amendment);
    # the mechanism is the two gated SPEED terms below.
    # v2.7.2: capability-based (no system-name branch). Quadrotor systems expose horizontal_velocity for the
    # obstacle-plane approach term; DI/unicycle lack it -> legacy quadratic cost (byte-identical parity).
    situational = hasattr(system, "horizontal_velocity")
    w_settle = float(policy_cfg.get("w_settle", 0.0))       # 0 -> no goal-gated settling term
    settle_rho = float(policy_cfg.get("settle_rho", 0.30))
    w_settle_ang = float(policy_cfg.get("w_settle_ang", 0.0))  # v2.8.0: goal-gated ANGULAR settling (0 -> off)
    w_appr = float(policy_cfg.get("w_appr", 0.0))           # 0 -> no obstacle-approach (braking-envelope) term
    tau_brake = float(policy_cfg.get("tau_brake", 0.6))     # braking-envelope lookahead time (amendment 2)
    use_situational = situational and (w_settle > 0.0 or w_appr > 0.0 or w_settle_ang > 0.0)
    # v2.4.0 Step 5 (audit C1 fix): when set, detach the CBF coefficients inside the differentiable
    # policy BPTT rollout so the policy gradient does not flow through the projection's state-dependent
    # coefficient Jacobian (the T=60 gradient-explosion source). Default off => byte-identical.
    detach_coeffs = bool(policy_cfg.get("detach_filter_coeffs", False))
    # v2.4.1: control-deficit policy feedback. w_deficit>0 penalizes the box-induced shortfall of the
    # safety correction (delta_u = box-free P_CBF - box-aware P_boxCBF), a per-step MEAN joining L_reg.
    # w_deficit==0 skips the aux path entirely (byte-identical baseline). delta_u is the raw box-free
    # CBF projection minus the deployed box-aware output; its norm is capped at deficit_cap (= 2*u_max)
    # so the alpha_unsafe=100 term at empty intersection cannot recreate a gradient explosion.
    w_deficit = float(policy_cfg.get("w_deficit", 0.0))
    deficit_cap = float(policy_cfg.get("deficit_cap", 4.0))
    # "sq_cap" (default): squared, norm-capped => ZERO gradient above the cap. "huber": smooth-L1 on
    # ||delta_u|| => bounded (=deficit_cap) but NONZERO gradient above the cap (v2.4.1 Huber arm).
    deficit_form = str(policy_cfg.get("deficit_form", "sq_cap"))
    # v2.4.1 Exp 2: obs_deficit_feedback feeds delta_u_{t-1} (DETACHED) as a policy input (+action_dim);
    # the deficit reaches the policy only through the ordinary task-return BPTT gradient, NOT through the
    # filter coefficient Jacobian. Independent of w_deficit (the loss channel).
    obs_deficit = bool(policy_cfg.get("obs_deficit_feedback", False))
    # v2.8.2 M2 (prop:sinfty-decomp / rem:sinfty-signal): live loss on relu(s_inf + delta), the empty-branch
    # depth made visible to training. DEFAULT OFF (w_infeas=0 -> not added -> byte-identical, G3). delta has NO
    # default: it is the S5-derived headroom (changes.md R6.1) and must never be silently consumed.
    w_infeas = float(policy_cfg.get("w_infeas", 0.0))
    infeas_delta = policy_cfg.get("infeas_delta", None)
    if w_infeas > 0.0:
        if infeas_delta is None:
            raise ValueError("loss.policy.infeas_delta must be set (S5-derived; no default) when w_infeas>0.")
        infeas_delta = float(infeas_delta)
    # v2.8.2 I3 (within-version iter, changes.md NOT amended): clip the term to min(relu(s_inf+delta), delta) so the
    # gradient is nonzero ONLY on the boundary layer {-delta<=s_inf<=0} (where feasible actions still exist,
    # rem:sinfty-signal iii) and flat inside E. The cap = delta is DERIVED (ends the ramp at the feasibility
    # boundary), not tuned. DEFAULT False -> unclipped relu (M2/I2 behavior byte-identical).
    infeas_clip = bool(policy_cfg.get("infeas_clip", False))
    # v2.8.2 M3: rate penalty on the filtered command, w_du * ||u_t^filt - u_{t-1}^filt||^2 / h^2. DEFAULT OFF
    # (w_du=0). Distinct from L_rate (dead-zoned, un-normalized): this is un-dead-zoned and /h^2.
    w_du = float(policy_cfg.get("w_du", 0.0))
    # v2.8.2 agree (prop:pi-filter-agreement): policy-filter agreement term L_agree =
    # w_agree * E[ mask_nonempty * || pi_theta(x_t) - sg(u_filt_t) ||^2 ], sg = stop-grad on the filtered
    # action (certificate untouched; the gradient reaches theta ONLY through u_nom=pi_theta). EMPTY steps
    # (filter box empty; last_empty) are MASKED OUT — no agreement target where no feasible action exists.
    # DEFAULT OFF (w_agree=0 -> the per-step mask collection and the term are both skipped -> `total`
    # byte-identical to baseline; mirrors the w_infeas / w_floor_recov gates). u_safe is already in the BPTT
    # rollout so the marginal forward cost is ~0.
    w_agree = float(policy_cfg.get("w_agree", 0.0))
    # v2.8.2 cond(c) (FLOOR-recovery shaping; coordinator-gated coefficient/launch): a per-rollout-step
    # penalty acting on the recovery DYNAMICS of the floor-failing near-inverted FEASIBLE set —
    # proximity-weighted tilt excess above the recoverable-hold angle. The floor-failing 67-set is
    # near-inverted (tilt ~127 deg, |omega| ~2.35) at ~0.18 m above the floor: feasible actions exist,
    # the policy is just too SLOW to level out before drifting in. L_floor_recov =
    # mean_t [ relu(1 - alt_above_floor_t / z_thr) * relu(cos(theta_hold) - R[2,2]_t) ]. DEFAULT OFF
    # (w_floor_recov=0 -> not added -> byte-identical; mirrors the w_infeas gate). Differentiable in
    # theta_pi through the rollout state (theta_pi -> u -> x), matching the task-cost gradient path. The
    # per-step block below is entered ONLY when w_floor_recov>0, so nothing new runs on the default path.
    w_floor_recov = float(policy_cfg.get("w_floor_recov", 0.0))
    floor_recov_zthr = float(policy_cfg.get("floor_recov_zthr", 0.5))
    if w_floor_recov > 0.0:
        from src.envs.quadrotor_3d import _quat_to_R as _floor_quat_to_R
        # theta_hold = arccos(min(1/TWR, 1)), TWR = 4*f_rotor_max/(m*g). All provenance from config (no
        # hardcoded 60 deg): the shipped quadrotor_3d set gives TWR=2 -> theta_hold=60 deg=1.047 rad.
        _fb = config["env"]["bounds"][system.name]
        _fp = config["env"][system.name]
        _twr = 4.0 * float(_fb["f_rotor_max"]) / (float(_fp["mass"]) * float(_fp["gravity"]))
        # v2.8.2 halt-fix: use cos(theta_hold) directly, not the angle. Since R[2,2]=cos(tilt), the smooth
        # excess relu(cos_theta_hold - R[2,2]) replaces the singular relu(arccos(R[2,2]) - theta_hold)
        # (arccos'(x)=-1/sqrt(1-x^2) -> inf grad at R[2,2]=+/-1, which NaN'd L_pi). cos(arccos(min(1/TWR,1)))
        # = min(1/TWR,1), so cos_theta_hold = min(1/TWR,1) = 0.5 for TWR=2 (no arccos, no clamp needed).
        floor_cos_theta_hold = min(1.0 / _twr, 1.0)
        # floor at p_z = -band_collision_limit (config, not a hardcode); alt_above_floor = p_z + limit.
        floor_band_limit = float(config["env"]["band_collision_limit"])
    # v2.5.0 Stage B: safety channel = analytic V_M (maneuver) or learned make_h_fn (value, default) —
    # the SAME builder as the collection filter (one builder, two call sites).
    from src.common.maneuver_value import build_safety_h_fn
    # v2.8.3 B2 GATE (site 2 of 3: policy BPTT). Same branch as train.py:207 via the single shared
    # resolver; absent/"hardnet" -> HardNetFilter, byte-identical to the pre-gate construction.
    from src.common.filter_backup import resolve_filter_cls
    hardnet = resolve_filter_cls(config)(system, build_safety_h_fn(system, config, value_net), config)

    _zero_grads(value_net.parameters())
    with frozen_params(value_net):
        x = system.wrap_state(batch.states.detach())
        scene = batch.scene
        # v2.6.2 obstacle-approach term: fetch the (static-per-episode) obstacle geometry ONCE; the term is
        # vectorized over the top-K obstacles inside the rollout loop (on-GPU, no Python obstacle loop).
        if use_situational and w_appr > 0.0:
            from src.common.observation import scene_obstacle_tensors
            _obs_c, _obs_r, _obs_a = scene_obstacle_tensors(scene, x.device, x.dtype)   # [B,K,2],[B,K],[B,K]
        v_now = value_net.deployed_h(system.observation(x, scene)).detach()
        gate_in = torch.sigmoid(-v_now / tau_gate)
        gate_out = torch.sigmoid(v_now / tau_gate)
        task_cost = x.new_zeros(x.shape[0])
        discount = 1.0
        nominal_actions: list[Tensor] = []
        safe_actions: list[Tensor] = []
        pretanh_values: list[Tensor] = []
        sat_excess_values: list[Tensor] = []
        pretanh_penalties: list[Tensor] = []
        deficit_terms: list[Tensor] = []        # v2.4.1: per-step masked ||delta_u_used||^2 (mean over batch)
        deficit_active_sum: list[Tensor] = []   # per-step sum of raw ||delta_u|| over active steps
        deficit_active_cnt: list[Tensor] = []   # per-step count of active steps
        deficit_clip_cnt: list[Tensor] = []     # per-step count of active steps hitting the cap
        deficit_feat_terms: list[Tensor] = []   # v2.4.1 Exp 2: per-step mean ||delta_u_{t-1}|| fed to policy
        sinf_terms: list[Tensor] = []           # v2.8.2 M2: per-step relu(s_inf + delta) mean (empty-branch depth)
        floor_recov_terms: list[Tensor] = []    # v2.8.2 cond(c): per-step prox-weighted tilt excess (floor recovery)
        agree_nonempty_masks: list[Tensor] = [] # v2.8.2 agree: per-step ~last_empty (only collected when w_agree>0)
        prev_deficit = x.new_zeros((x.shape[0], system.action_dim)) if obs_deficit else None
        # v2.8.3 U-PREV AXIS. The BPTT window starts from a buffer state that carries NO action history,
        # so step 0 of the window uses the t=0 convention: hover trim m*g/4 per rotor, NEVER zeros —
        # the SAME value the collector and both eval paths use. Thereafter it is the previous step's
        # EXECUTED u_safe, DETACHED: the axis opens an INPUT channel, it does not add a second gradient
        # path through the filter (the deployed/collection paths are no_grad, so a gradient-carrying
        # training path would be a train/deploy semantic mismatch).
        from src.common.observation import append_u_prev, u_prev_feedback_on, u_prev_init
        u_prev_on = u_prev_feedback_on(config)
        prev_u = (u_prev_init(system, config, x.shape[0], device=x.device, dtype=x.dtype)
                  if u_prev_on else None)

        for _ in range(bptt_t):
            obs = system.observation(x, scene)
            if obs_deficit:
                obs = torch.cat([obs, prev_deficit], dim=1)   # prev_deficit already detached
                deficit_feat_terms.append(torch.linalg.norm(prev_deficit, dim=1).mean())
            if u_prev_on:
                obs = append_u_prev(obs, prev_u)              # append_u_prev detaches
            u_nom = policy_net(obs)
            nominal_actions.append(u_nom)
            if getattr(policy_net, "last_pretanh", None) is not None:
                z = policy_net.last_pretanh
                pretanh_values.append(z)
                excess_z = (z.abs() - float(policy_cfg["z_target"])).clamp_min(0.0)
                sq_excess = (excess_z * excess_z).mean(dim=1)
                if bool(policy_cfg["vs_gated_pretanh"]):
                    v_gate = value_net.deployed_h(system.observation(x, scene)).detach()
                    gate = torch.sigmoid((-v_gate - 0.02) / float(policy_cfg["vs_gate_tau"]))
                    sq_excess = sq_excess * gate
                pretanh_penalties.append(sq_excess.mean())

            # v2.6.0: sat_excess_threshold may be per-channel [thrust, torque] (quadrotor box bounds) or a
            # scalar (DI/unicycle). Broadcast a per-channel tensor over the action dim.
            _sat_thr = policy_cfg["sat_excess_threshold"]
            sat_thr = (torch.tensor(_sat_thr, device=u_nom.device, dtype=u_nom.dtype)
                       if isinstance(_sat_thr, (list, tuple)) else float(_sat_thr))
            sat_excess = (u_nom.abs() - sat_thr).clamp_min(0.0)
            sat_excess_values.append((sat_excess * sat_excess).sum(dim=1).mean())

            if w_deficit > 0.0 or obs_deficit:
                u_safe, _, u_cbf_only, singular = hardnet(
                    x, scene, u_nom, detach_coeffs=detach_coeffs, return_deficit_aux=True)
                delta_u = u_cbf_only - u_safe                                  # raw CBF ask - box-aware output
                if w_deficit > 0.0:
                    mag = torch.linalg.norm(delta_u, dim=1, keepdim=True)          # [B,1]
                    if deficit_form == "huber":
                        # Huber (smooth-L1) on r=||delta_u||: bounded GRADIENT (=deficit_cap) but NONZERO
                        # above the cap, unlike sq_cap whose ||delta_u_used||^2 is constant (zero grad) there.
                        r = mag.squeeze(1)                                         # [B]
                        deficit_val = torch.where(
                            r <= deficit_cap, 0.5 * r * r, deficit_cap * (r - 0.5 * deficit_cap))
                    else:                                                          # "sq_cap" (default, byte-identical)
                        scale = (deficit_cap / (mag + 1.0e-9)).clamp(max=1.0)      # norm cap at deficit_cap
                        delta_u_used = delta_u * scale
                        deficit_val = torch.sum(delta_u_used * delta_u_used, dim=1)  # [B]
                    nonsingular = (~singular).to(deficit_val.dtype)
                    deficit_terms.append((nonsingular * deficit_val).mean())        # mean over batch (per step)
                    mag_d = mag.detach().squeeze(1)
                    active = (~singular) & (mag_d > 1.0e-6)
                    deficit_active_sum.append(mag_d[active].sum())                 # raw ||delta_u|| diagnostic
                    deficit_active_cnt.append(active.sum().to(deficit_val.dtype))
                    deficit_clip_cnt.append((active & (mag_d > deficit_cap)).sum().to(deficit_val.dtype))
                if obs_deficit:
                    prev_deficit = delta_u.detach()                            # fed to policy next step
            else:
                u_safe, _ = hardnet(x, scene, u_nom, detach_coeffs=detach_coeffs)
            safe_actions.append(u_safe)
            if u_prev_on:
                prev_u = u_safe.detach()          # EXECUTED control, fed to the policy at the next step
            if w_agree > 0.0 and getattr(hardnet, "last_empty", None) is not None:
                # v2.8.2 agree: capture the filter's OWN empty indicator at this step (box_aware path sets it).
                # Detached: the mask gates the term but carries no gradient. Only runs when w_agree>0.
                agree_nonempty_masks.append(~hardnet.last_empty.detach())
            if w_infeas > 0.0 and getattr(hardnet, "last_a_live", None) is not None:
                # v2.8.2 M2: empty-branch depth from the LIVE filter row (a=L_g h, b=row_upper). theta_V frozen
                # => zero theta_V grad (G4); s_inf linear in a,b => grad carries no 1/||a||; delta pre-relu.
                a_live = hardnet.last_a_live
                b_live = hardnet.last_b_live
                _bnd = system.u_bounds.to(a_live.device, a_live.dtype)
                min_corner = torch.where(a_live >= 0.0, _bnd[:, 0], _bnd[:, 1])
                s_inf = (a_live * min_corner).sum(dim=1) - b_live               # [B] empty-branch depth
                _pen = torch.relu(s_inf + infeas_delta)                         # ramp 0 at s_inf=-delta
                if infeas_clip:                                                 # I3: cap at delta -> flat inside E,
                    _pen = torch.clamp(_pen, max=infeas_delta)                  # gradient only on {-delta<=s_inf<=0}
                sinf_terms.append(_pen.mean())
            x = rk4_step(system, x, u_safe, dt)
            if w_floor_recov > 0.0:
                # v2.8.2 cond(c): proximity-weighted tilt-excess on the REACHED rollout state x_{t+1}
                # (same states the task cost scores). alt_above_floor = p_z + band_collision_limit.
                # SMOOTH tilt excess (v2.8.2 halt-fix): R[2,2] = cos(tilt), so relu(cos_theta_hold - R[2,2])
                # is > 0 exactly when cos(tilt) < cos(theta_hold), i.e. tilt > theta_hold; it is monotone
                # increasing in tilt and SINGULARITY-FREE (no arccos => no 1/sqrt(1-R22^2) inf grad at
                # R[2,2]=+/-1, no clamp). prox weight is 0 at/above z_thr and ramps to 1 at the floor.
                _alt = x[:, 2] + floor_band_limit
                _R22 = _floor_quat_to_R(x[:, 3:7])[:, 2, 2]
                _prox = torch.relu(1.0 - _alt / floor_recov_zthr)
                _excess = torch.relu(floor_cos_theta_hold - _R22)
                floor_recov_terms.append((_prox * _excess).mean())
            goal = _scene_goal(scene, x)
            pos_error = system.position(x) - goal
            v2 = system.speed(x) * system.speed(x)
            u2 = torch.sum(u_safe * u_safe, dim=1)
            if use_situational:
                # v2.6.2 situation-dependent velocity objective (changes.md §4 + build-log amendment). The
                # goal-DISTANCE term stays the original quadratic ||p-g||^2 (v2.6.1); the Huber variant was
                # REVERTED (audit: no smooth cost has a nonvanishing gradient at its minimizer, and the
                # specified Huber weakened the far-field goal gradient ~40x — a confound both ablation arms
                # would carry). The v2.6.2 MECHANISM is the two gated SPEED terms: dense goal-gated settling
                # (penalize speed ONLY near the goal) + dense obstacle-gated approach (penalize INWARD speed
                # ONLY near obstacles). lambda_v*v2 (global) and mu_u*u2 retained.
                r = torch.linalg.norm(pos_error, dim=1)                     # ||p - g|| (for the settling gate)
                d2 = torch.sum(pos_error * pos_error, dim=1)                # original quadratic goal distance
                step_cost = d2 + lambda_v * v2 + mu_u * u2
                if w_settle > 0.0:
                    step_cost = step_cost + w_settle * torch.exp(-(r * r) / (settle_rho * settle_rho)) * v2
                if w_settle_ang > 0.0:
                    # v2.8.0: angular settling under the SAME goal gate — supplies the gradient toward
                    # low angular rate that the (predicate-only) reach condition does not. Structurally
                    # zero on DI/unicycle (angular_rate == 0), so inert there by construction.
                    w2 = system.angular_rate(x) * system.angular_rate(x)
                    step_cost = step_cost + w_settle_ang * torch.exp(-(r * r) / (settle_rho * settle_rho)) * w2
                if w_appr > 0.0:
                    # v2.6.2 AMENDMENT 2: BRAKING-ENVELOPE obstacle-approach deficit (replaces the fixed-d0
                    # gaussian gate, which was inert — diag Phase-1 B/C: closed 0.018 at 1 m surface, past the
                    # PNR, and ~36x below the goal term). Form: w_appr*sum_k relu(s_k*tau_brake - surf_k)^2,
                    # s_k = inward speed. Zero when receding (s=0) or outside the envelope (surf > s*tau_brake),
                    # C^1, no division, and the envelope RADIUS = s*tau_brake grows with the approach speed, so
                    # it engages EARLIER the FASTER the approach — the property the fixed gate lacked.
                    p = system.position(x)[:, : _obs_c.shape[-1]]         # [B,2] obstacle-plane position (xy)
                    vel = system.horizontal_velocity(x)                   # [B,2] obstacle-plane velocity (xy)
                    rel = p.unsqueeze(1) - _obs_c                          # [B,K,2] obstacle-center -> body
                    dist_c = torch.linalg.norm(rel, dim=2)                 # [B,K]
                    surf = dist_c - _obs_r                                 # [B,K] surface distance
                    normal = rel / dist_c.unsqueeze(2).clamp_min(1.0e-9)   # [B,K,2] outward surface normal
                    v_dot_n = torch.sum(vel.unsqueeze(1) * normal, dim=2)  # [B,K] (>0 = moving away)
                    inward = torch.relu(-v_dot_n)                          # [B,K] inward speed s_k (>=0)
                    deficit = torch.relu(inward * tau_brake - surf)        # [B,K] envelope violation (>0 inside)
                    deficit = deficit * _obs_a.to(surf.dtype)              # mask inactive obstacles
                    step_cost = step_cost + w_appr * torch.sum(deficit * deficit, dim=1)
                task_cost = task_cost + discount * step_cost
            else:
                d2 = torch.sum(pos_error * pos_error, dim=1)               # legacy quadratic (DI/unicycle parity)
                task_cost = task_cost + discount * (d2 + lambda_v * v2 + mu_u * u2)
            discount *= gamma_t

        # v2.6.0 credit-horizon axis: BPTT TERMINAL VALUE at x_T (03_train §4.4 currently has NO terminal
        # and NO in-window termination, so goal-reaching beyond the ~1.5 s window is never credited and
        # myopic hover is the local optimum). Add an analytic goal-progress terminal V_term = -||p_T - g||
        # (a COST +||p_T - g|| here, since task_cost is minimized), weighted by the running discount
        # (gamma_T^T). DIFFERENTIABLE through x_T (it must carry the credit gradient) — NOT the learned V_hat
        # (a hazard sup-h value, not a goal value). Inside the same g_in safe-region gate as task_cost.
        # PROTOCOL FOLLOW-UP: this changes the §4.4 rollout return; 03_train edit deferred until utility
        # confirmed (Researcher-directed).
        # v2.6.1 Axis B: velocity-aware terminal. The v2.6.0 terminal credits goal POSITION (||p_T - g||) but
        # not goal SPEED, so the policy overshoots and loiters near the goal (v2.6.0 D1: 57% of timeouts reach
        # within goal_radius but miss the goal-speed criterion). Add discount*w_terminal_v*||v_T|| so slowing
        # at the horizon is credited. Differentiable through x_T; same gate_in; rides detach_filter_coeffs.
        w_term = float(policy_cfg.get("w_terminal", 0.0))
        w_term_v = float(policy_cfg.get("w_terminal_v", 0.0))
        if w_term > 0.0 or w_term_v > 0.0:
            goal_T = _scene_goal(scene, x)
            dist_T = torch.linalg.norm(system.position(x) - goal_T, dim=1)   # ||p_T - g||
            speed_T = system.speed(x)                                       # ||v_T|| (world linear velocity)
            task_cost = task_cost + discount * (w_term * dist_T + w_term_v * speed_T)  # discount == gamma_T^T here

        action_stack = torch.stack(nominal_actions, dim=0)
        safe_stack = torch.stack(safe_actions, dim=0)
        # v2.8.0 S3 M3: P2 instrumentation (gated, detached; no effect on total/backward). Clip-class
        # distribution of the filtered BPTT actions = the Jacobian classes S1 used (no-clip / 0<|A|<m /
        # |A|=m) plus |A|=m-1 (legitimately zero-Jacobian for the dual solve). Appended to _JAC_CLASS_LOG.
        if bool(policy_cfg.get("log_jac_classes", False)):
            with torch.no_grad():
                _b = system.u_bounds.to(device=safe_stack.device, dtype=safe_stack.dtype)
                _ss = safe_stack.detach().reshape(-1, safe_stack.shape[-1])          # [T*B, m]
                _m = _ss.shape[-1]
                _clip = ((_ss - _b[:, 0]).abs() < 1e-6) | ((_ss - _b[:, 1]).abs() < 1e-6)
                _nclip = _clip.sum(dim=1)
                _interv = (torch.linalg.norm(safe_stack.detach() - action_stack.detach(), dim=-1)
                           .reshape(-1) > 1e-3)
                _JAC_CLASS_LOG.append({
                    "step": int(step),
                    "n_rows": int(_ss.shape[0]),
                    "frac_noclip": float((_nclip == 0).float().mean()),
                    "frac_partial": float(((_nclip > 0) & (_nclip < _m)).float().mean()),
                    "frac_vertex": float((_nclip == _m).float().mean()),
                    "frac_free1": float((_nclip == _m - 1).float().mean()),
                    "frac_intervention": float(_interv.float().mean()),
                    "zero_jac_frac_all_real": float((_nclip == _m).float().mean()),
                    "zero_jac_frac_dual": float((_nclip >= _m - 1).float().mean()),
                })
        # input-rate regulation (v2.2.2 add-on): DEAD-ZONE penalty on step-to-step change of the
        # APPLIED (post-HardNet) control u_safe. L_rate = mean_t max(0, ||u_t - u_{t-1}|| - delta)^2
        # (delta dead-zone leaves normal control free, only chattering above delta is penalized).
        # Effective weight follows a linear curriculum: w_eff = weight * min(1, step / ramp_steps).
        # mean_abs_du = mean ||du||_2 stays the RAW chattering diagnostic (NOT dead-zoned), comparable
        # across runs/weights.
        rate_cfg = policy_cfg.get("input_rate", {})
        rate_enabled = bool(rate_cfg.get("enabled", False))
        rate_weight = float(rate_cfg.get("weight", 0.0))
        rate_delta = float(rate_cfg.get("delta", 0.0))
        rate_ramp_steps = float(rate_cfg.get("ramp_steps", 0.0))
        rate_w_eff = rate_weight
        if rate_ramp_steps > 0.0:
            rate_w_eff = rate_weight * min(1.0, float(step) / rate_ramp_steps)
        if safe_stack.shape[0] > 1:
            du = safe_stack[1:] - safe_stack[:-1]                       # [T-1, B, A]
            du_norm = torch.linalg.norm(du, dim=2)                      # [T-1, B] = ||du||_2 per step
            excess = (du_norm - rate_delta).clamp_min(0.0)              # dead-zone: free below delta
            l_rate = torch.mean(excess * excess)
            mean_abs_du = du_norm.mean()
            l_du = torch.mean(torch.sum(du * du, dim=2)) / (dt * dt)    # v2.8.2 M3: ||Δu^filt||^2 / h^2 (un-dead-zoned)
        else:
            l_rate = safe_stack.new_zeros(())
            mean_abs_du = safe_stack.new_zeros(())
            l_du = safe_stack.new_zeros(())
        # input control-magnitude regularization (v2.2.2 add-on): LQR-style R term on the APPLIED
        # control, L_u = mean_t ||u_safe_t||^2 (ungated, undiscounted). u is the control whose integral
        # is speed, so penalizing |u| curbs accumulated approach speed (the over-speed-entry cause).
        ureg_cfg = policy_cfg.get("u_reg", {})
        ureg_enabled = bool(ureg_cfg.get("enabled", False))
        ureg_weight = float(ureg_cfg.get("weight", 0.0))
        l_u = torch.mean(torch.sum(safe_stack * safe_stack, dim=2))
        action_norm = torch.mean(torch.sum(action_stack * action_stack, dim=2))
        if action_stack.shape[0] > 1:
            smoothness = torch.mean(
                torch.sum((action_stack[1:] - action_stack[:-1]) ** 2, dim=2)
            )
        else:
            smoothness = action_stack.new_zeros(())
        saturation_excess = torch.stack(sat_excess_values).mean()
        pretanh = (
            torch.stack(pretanh_penalties).mean()
            if pretanh_penalties
            else action_stack.new_zeros(())
        )
        obs0 = system.observation(batch.states.detach(), batch.scene)
        if obs_deficit:
            obs0 = torch.cat([obs0, obs0.new_zeros((obs0.shape[0], system.action_dim))], dim=1)
        if u_prev_on:
            # single-state site (no action history in the buffer row) -> the t=0 convention: hover trim
            obs0 = append_u_prev(obs0, u_prev_init(system, config, obs0.shape[0],
                                                   device=obs0.device, dtype=obs0.dtype))
        u0 = policy_net(obs0)
        x_next_unfiltered = rk4_step(system, batch.states.detach(), u0, dt)
        v_next = value_net.deployed_h(system.observation(x_next_unfiltered, batch.scene))
        outside = (gate_out * v_next).mean()
        task = (gate_in * task_cost).mean()
        total = (
            task
            + float(policy_cfg["lambda_a"]) * action_norm
            + float(policy_cfg["lambda_s"]) * smoothness
            + float(policy_cfg["lambda_sat"]) * saturation_excess
            + float(policy_cfg["lambda_pretanh"]) * pretanh
            + float(policy_cfg["w_outside"]) * outside
        )
        # flag-off (or w_eff==0, e.g. step 0 of the ramp) leaves `total` byte-identical to baseline.
        if rate_enabled and rate_w_eff > 0.0:
            total = total + rate_w_eff * l_rate
            l_rate_weighted = (rate_w_eff * l_rate).detach()
        else:
            l_rate_weighted = l_rate.new_zeros(())
        # u-reg add (flag-off / weight 0 leaves `total` byte-identical to baseline).
        if ureg_enabled and ureg_weight > 0.0:
            total = total + ureg_weight * l_u
            l_u_weighted = (ureg_weight * l_u).detach()
        else:
            l_u_weighted = l_u.new_zeros(())
        # v2.8.2 M3 filtered-command rate (default off; w_du==0 leaves `total` byte-identical to baseline).
        if w_du > 0.0:
            total = total + w_du * l_du
            l_du_weighted = (w_du * l_du).detach()
        else:
            l_du_weighted = l_du.new_zeros(())
        # v2.8.2 M2 infeasibility-margin cost (default off; w_infeas==0 leaves `total` byte-identical, G3). The
        # policy gradient flows through s_inf via the rollout state; theta_V is frozen -> zero theta_V grad (G4).
        if w_infeas > 0.0 and sinf_terms:
            l_infeas_raw = torch.stack(sinf_terms).mean()
            total = total + w_infeas * l_infeas_raw
            l_infeas_weighted = (w_infeas * l_infeas_raw).detach()
        else:
            l_infeas_raw = total.new_zeros(())
            l_infeas_weighted = total.new_zeros(())
        # v2.8.2 cond(c) floor-recovery shaping (default off; w_floor_recov==0 leaves `total` byte-identical).
        # The policy gradient flows through the tilt/altitude of the rollout state x (theta_pi -> u -> x);
        # no new op runs when off (the per-step block is skipped entirely, mirroring w_infeas).
        if w_floor_recov > 0.0 and floor_recov_terms:
            l_floor_recov_raw = torch.stack(floor_recov_terms).mean()
            total = total + w_floor_recov * l_floor_recov_raw
            l_floor_recov_weighted = (w_floor_recov * l_floor_recov_raw).detach()
        else:
            l_floor_recov_raw = total.new_zeros(())
            l_floor_recov_weighted = total.new_zeros(())
        # v2.4.1 control-deficit term (flag-off / w_deficit==0 leaves `total` byte-identical to baseline).
        if w_deficit > 0.0:
            l_deficit_raw = torch.stack(deficit_terms).mean()                  # mean over batch, t
            total = total + w_deficit * l_deficit_raw
            l_deficit_weighted = (w_deficit * l_deficit_raw).detach()
            active_cnt_total = torch.stack(deficit_active_cnt).sum()
            mean_deficit_active = (torch.stack(deficit_active_sum).sum()
                                   / active_cnt_total.clamp_min(1.0)).detach()
            deficit_active_frac = (active_cnt_total
                                   / float(bptt_t * batch.states.shape[0])).detach()
            deficit_clip_frac = (torch.stack(deficit_clip_cnt).sum()
                                 / active_cnt_total.clamp_min(1.0)).detach()
        else:
            l_deficit_weighted = total.new_zeros(())
            mean_deficit_active = total.new_zeros(())
            deficit_active_frac = total.new_zeros(())
            deficit_clip_frac = total.new_zeros(())
        # v2.4.1 Exp 2: mean ||delta_u_{t-1}|| fed to the policy over the window (obs channel diagnostic).
        mean_abs_deficit_feature = (
            torch.stack(deficit_feat_terms).mean().detach()
            if deficit_feat_terms else total.new_zeros(())
        )
        # v2.5.0 Stage B-2: filter friction. L_friction = mean_t mean_B ||u_safe - u_nom||^2 (NO region
        # gate — applies everywhere the filter acts). Removes the projection gauge: the row-normal
        # component of u_nom has EXACTLY zero task gradient (Pi = I - AA^T/||A||^2 annihilates it), so any
        # positive w_friction dominates the gauge mode without competing on the task-relevant tangential
        # component. Gradient reaches theta through BOTH u_nom (policy) and u_safe (= u_nom - A*viol/denom,
        # which still depends on u_nom under detach_filter_coeffs=true — coeffs A,b detached, not u_nom).
        w_friction = float(policy_cfg.get("w_friction", 0.0))
        diff = safe_stack - action_stack                                   # [T,B,A], both carry theta-grad
        l_friction_raw = torch.mean(torch.sum(diff * diff, dim=2))
        proj_mag_bptt = torch.linalg.norm(diff, dim=2).mean().detach()     # mean ||u_safe-u_nom|| (all-active)
        if w_friction > 0.0:                                               # flag-off => total byte-identical
            total = total + w_friction * l_friction_raw
            friction_weighted = (w_friction * l_friction_raw).detach()
        else:
            friction_weighted = total.new_zeros(())

        # v2.8.2 agree (prop:pi-filter-agreement): policy-filter agreement term. L_agree =
        # w_agree * E[ mask_nonempty * ||pi - sg(u_filt)||^2 ]. pi = action_stack (u_nom, carries theta-grad);
        # sg(u_filt) = safe_stack.detach() (stop-grad -> the certificate/filter is untouched, gradient reaches
        # theta ONLY through the policy). mask_nonempty = ~last_empty per step (empty rows carry no feasible
        # target, so they are excluded; this is the S1-verified TRUE empty flag, not the aliased infeasible).
        # E[] = mean over the [T,B] window (friction convention). Flag-off (w_agree=0) => `total` byte-identical
        # to baseline (the mask list is empty and this block is skipped), mirroring the w_infeas / friction gates.
        if w_agree > 0.0 and agree_nonempty_masks:
            nonempty = torch.stack(agree_nonempty_masks, dim=0).to(safe_stack.dtype)   # [T,B]
            agree_sq = torch.sum((action_stack - safe_stack.detach()) ** 2, dim=2)      # [T,B] ||pi - sg(u_filt)||^2
            l_agree_raw = (nonempty * agree_sq).mean()
            total = total + w_agree * l_agree_raw
            l_agree_weighted = (w_agree * l_agree_raw).detach()
            l_agree_raw = l_agree_raw.detach()
        else:
            l_agree_raw = total.new_zeros(())
            l_agree_weighted = total.new_zeros(())

        # v2.5.1 A2(b): horizon-summary critic tail. total += w_hc * gamma_c^T * mean_B W(obs(x_T)) with W's
        # parameters FROZEN (stop-grad) so dL_pi/d(theta_W)=0; the ONLY policy gradient path is pathwise
        # through x_T (the differentiable rollout endpoint). x still holds x_T here (the endpoint is not
        # reassigned after the bptt loop). critic_net=None (horizon_critic.enabled=false) leaves `total`
        # byte-identical to baseline. W reads the dim-19 base observation (no deficit channel).
        if critic_net is not None:
            hc_cfg = config["training"]["jt"]["horizon_critic"]
            gamma_c = float(hc_cfg["gamma"]); w_hc = float(hc_cfg["weight"])
            obs_tail = system.observation(x, scene)
            with frozen_params(critic_net):
                w_tail = critic_net(obs_tail)
            total = total + w_hc * (gamma_c ** bptt_t) * w_tail.mean()

    return PolicyLossResult(
        total=total,
        task=task.detach(),
        action_norm=action_norm.detach(),
        smoothness=smoothness.detach(),
        saturation_excess=saturation_excess.detach(),
        pretanh=pretanh.detach(),
        outside=outside.detach(),
        grad_leak=grad_norm(value_net.parameters()),
        action_abs_mean=safe_stack.abs().mean().detach(),
        action_abs_max=safe_stack.abs().max().detach(),
        satfrac_a_phi=_satfrac(action_stack, system).detach(),
        l_rate_raw=l_rate.detach(),
        l_rate_weighted=l_rate_weighted,
        mean_abs_du=mean_abs_du.detach(),
        l_u_raw=l_u.detach(),
        l_u_weighted=l_u_weighted,
        l_deficit=l_deficit_weighted,
        mean_deficit_active=mean_deficit_active,
        deficit_active_frac=deficit_active_frac,
        deficit_clip_frac=deficit_clip_frac,
        mean_abs_deficit_feature=mean_abs_deficit_feature,
        friction_loss=friction_weighted,
        proj_mag_bptt=proj_mag_bptt,
        l_du_raw=l_du.detach(),
        l_du_weighted=l_du_weighted,
        l_infeas_raw=l_infeas_raw.detach(),
        l_infeas_weighted=l_infeas_weighted,
        l_floor_recov_raw=l_floor_recov_raw.detach(),
        l_floor_recov_weighted=l_floor_recov_weighted,
        l_agree_raw=l_agree_raw,
        l_agree_weighted=l_agree_weighted,
    )


@contextmanager
def frozen_params(module: nn.Module) -> Iterator[None]:
    states = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, state in zip(module.parameters(), states, strict=True):
            param.requires_grad_(state)


def grad_sq_norm(parameters: Any) -> Tensor | None:
    """Sum of squared grads as an on-device scalar tensor (NO host sync). None if no params have grads.

    Lets callers compare the gradient-leak against a threshold on-device every step and materialize the
    Python float only when logging or when the halt condition triggers.
    """
    total: Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        s = torch.sum(grad * grad)
        total = s if total is None else total + s
    return total


def grad_norm(parameters: Any) -> float:
    # Single host transfer (one .item() on the device-summed norm) instead of one per parameter.
    sq = grad_sq_norm(parameters)
    return 0.0 if sq is None else float(sq.sqrt().item())


def _zero_grads(parameters: Any) -> None:
    for parameter in parameters:
        parameter.grad = None


def _scene_goal(scene: BatchedScene, x: Tensor) -> Tensor:
    goal = scene.goal.to(device=x.device, dtype=x.dtype)
    if goal.ndim == 1:
        return goal.unsqueeze(0).expand(x.shape[0], -1)
    return goal


def _satfrac(actions: Tensor, system: System) -> Tensor:
    bounds = system.u_bounds.to(device=actions.device, dtype=actions.dtype)
    lower_dist = torch.abs(actions - bounds[:, 0])
    upper_dist = torch.abs(actions - bounds[:, 1])
    saturated = torch.any(torch.minimum(lower_dist, upper_dist) <= 1.0e-3, dim=-1)
    return saturated.to(dtype=actions.dtype).mean()
