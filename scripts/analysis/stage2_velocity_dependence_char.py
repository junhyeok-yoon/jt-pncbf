"""v2.2.0 Stage 2 — velocity-dependence characterization of the learned V_S (read-only).

Characterizes, from data, HOW the learned V_S depends on velocity, to locate where a velocity-
sensitivity auxiliary loss should bite. Independent of the training-code audit. V_S is queried by
forward + autograd on a CONSTRUCTED grid of states (no training, no env rollout, no fix).

L_g h chain rule (verified elsewhere): obs = [vx,vy, gx-px,gy-py, 5x(cx-px,cy-py,r)]; only obs[0:2]
depend on v (identity), so L_g h = ∂h/∂v = grad_x(h)[2:4]. Analytic oracle (HOCBF, alpha1=2,
r_margin=0.05): psi1 = 2(p-c)·v + alpha1*(||p-c||^2 - r_safe^2); L_g psi1 = 2(p-c) (bounded away
from 0). Safe sets: learned {h<=0}, HOCBF {psi1>=0} (so "danger" = h>0 resp. psi1<0).

Part A velocity-dependent boundary (learned h=0 vs analytic psi1=0) + boundary shift with velocity.
Part B velocity-resolved ||∂h/∂v|| over (clearance, speed, approach-angle) + approach-aligned
decomposition + per-member. Part C h vs closing speed (the value's velocity response, not just
gradient). Part D verdict.

Run: /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage2_velocity_dependence_char.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.scene_batch import batch_scenes  # noqa: E402
from src.envs.scene_init import Scene  # noqa: E402
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint, make_system  # noqa: E402

CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
OUT = REPO_ROOT / "data/diagnostics/v2.2.0_hocbf"
ALPHA1 = 2.0
R_MARGIN = 0.05
GOAL = np.array([3.5, 3.5])              # fixed far goal (stated); velocity-dependence is the variable
# multi-obstacle layout (deterministic, no RNG) for the visual contour
MULTI = [((0.0, 0.0), 0.6), ((2.0, 1.6), 0.5), ((-2.1, 1.4), 0.5), ((-1.6, -2.0), 0.45), ((1.8, -1.7), 0.5)]
# single isolated obstacle at origin for the clean shift / Part B / Part C
SINGLE_C = np.array([0.0, 0.0]); SINGLE_R = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def make_scene(obstacles):
    centers = np.array([c for c, _ in obstacles], float)
    radii = np.array([r for _, r in obstacles], float)
    active = np.ones(len(obstacles), bool)
    return Scene(obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active,
                 start=np.array([-3.0, -3.0]), goal=GOAL.copy(), system="double_integrator",
                 mode="eval", initial_velocity=np.array([0.0, 0.0]))


def h_on(fw, system, positions, vel, scene):
    """deployed h (mean ensemble) on states (positions, vel-broadcast). positions [N,2], vel [2] or [N,2]."""
    N = positions.shape[0]
    v = np.broadcast_to(np.asarray(vel, float), (N, 2))
    x = torch.tensor(np.concatenate([positions, v], axis=1), device=DEVICE, dtype=DTYPE)
    bscene = batch_scenes([scene] * N, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        obs = system.observation(x, bscene)
        h = fw.value_net.deployed_h(obs)
    return h.cpu().numpy()


def grad_v(fw, system, positions, vel, scene, members=False):
    """∂h/∂v [N,2] (and per-member [N,n_vs,2] if members)."""
    N = positions.shape[0]
    v = np.broadcast_to(np.asarray(vel, float), (N, 2))
    x = torch.tensor(np.concatenate([positions, v], axis=1), device=DEVICE, dtype=DTYPE, requires_grad=True)
    bscene = batch_scenes([scene] * N, device=DEVICE, dtype=DTYPE)
    obs = system.observation(x, bscene)
    h = fw.value_net.deployed_h(obs)
    g = torch.autograd.grad(h.sum(), x, create_graph=False)[0][:, 2:4].detach().cpu().numpy()
    if not members:
        return g, h.detach().cpu().numpy(), None
    per = []
    for m in range(fw.value_net.value_all(obs).shape[1]):
        x2 = torch.tensor(np.concatenate([positions, v], axis=1), device=DEVICE, dtype=DTYPE, requires_grad=True)
        obs2 = system.observation(x2, bscene)
        vm = torch.clamp(fw.value_net.forward_all(obs2)[:, m], -1.0, 1.0)
        gm = torch.autograd.grad(vm.sum(), x2, create_graph=False)[0][:, 2:4].detach().cpu().numpy()
        per.append(gm)
    return g, h.detach().cpu().numpy(), np.stack(per, axis=1)


def analytic_psi1(positions, vel, centers, radii, alpha1=ALPHA1, margin=R_MARGIN):
    """min-over-obstacles psi1 (safe if >=0). positions [N,2], vel [2] or [N,2]."""
    v = np.broadcast_to(np.asarray(vel, float), (positions.shape[0], 2))
    rel = positions[:, None, :] - centers[None, :, :]               # p - c
    rsafe = radii + margin
    h0 = np.sum(rel * rel, axis=2) - rsafe[None, :] ** 2
    relv = np.sum(rel * v[:, None, :], axis=2)
    psi1 = 2.0 * relv + alpha1 * h0
    return psi1.min(axis=1)


# ---- Part A: velocity-dependent boundary ---------------------------------------------

def part_a(fw, system):
    scene = make_scene(MULTI)
    centers = np.array([c for c, _ in MULTI]); radii = np.array([r for _, r in MULTI])
    gx = np.linspace(-3.3, 3.3, 220); gy = np.linspace(-3.3, 3.3, 220)
    GX, GY = np.meshgrid(gx, gy); P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    vels = {"v=(-1.5,0) recede": np.array([-1.5, 0.0]), "v=0": np.array([0.0, 0.0]),
            "v=(+1.5,0) approach": np.array([1.5, 0.0])}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=140)
    H = {}; PSI = {}
    for ax, (name, vv) in zip(axes, vels.items()):
        h = h_on(fw, system, P, vv, scene).reshape(GX.shape)
        psi = analytic_psi1(P, vv, centers, radii).reshape(GX.shape)
        H[name] = h; PSI[name] = psi
        for c, r in zip(centers, radii):
            ax.add_patch(Circle(c, r, color="0.6", alpha=0.5))
        ax.contour(GX, GY, h, levels=[0.0], colors="#1f77b4", linewidths=2.0)
        ax.contour(GX, GY, psi, levels=[0.0], colors="#d62728", linewidths=1.6, linestyles="--")
        ax.set_title(f"{name}\nblue=learned h=0, red dashed=analytic psi1=0", fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3)
    fig.suptitle("Part A — velocity-dependent unsafe boundary (multi-obstacle); goal=(3.5,3.5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / "veldep_contours.png"); plt.close(fig)

    # boundary shift on the origin obstacle along the x-axis (approach axis for v_x)
    scene1 = make_scene([((0.0, 0.0), SINGLE_R)])
    xs = np.linspace(-3.0, -SINGLE_R - 1e-3, 600)                    # left ray (approach side for v=+x)
    pos = np.stack([xs, np.zeros_like(xs)], axis=1)

    rsafe = SINGLE_R + R_MARGIN

    def boundary_x(field):
        # danger field>0 near obstacle, <0 (safe) far; crossing nearest the obstacle.
        # No crossing => unsafe region collapsed to the physical surface => boundary at r_safe.
        cr = np.where(np.diff(np.sign(field)) != 0)[0]
        if cr.size == 0:
            return rsafe
        return float(abs(xs[cr[-1]]))

    shift = {}
    for tag, vv in (("approach", np.array([1.5, 0.0])), ("recede", np.array([-1.5, 0.0]))):
        h = h_on(fw, system, pos, vv, scene1)
        psi = analytic_psi1(pos, vv, SINGLE_C[None, :], np.array([SINGLE_R]))
        shift[tag] = {"learned_boundary_dist": boundary_x(h),
                      "analytic_boundary_dist": boundary_x(-psi)}   # psi1<0 = danger => danger field = -psi1
    lb = [shift["approach"]["learned_boundary_dist"], shift["recede"]["learned_boundary_dist"]]
    ab = [shift["approach"]["analytic_boundary_dist"], shift["recede"]["analytic_boundary_dist"]]
    learned_shift = abs(lb[0] - lb[1]) if None not in lb else None
    analytic_shift = abs(ab[0] - ab[1]) if None not in ab else None
    return {"velocities": list(vels.keys()), "boundary_dist_by_v": shift,
            "learned_boundary_shift_approach_vs_recede": learned_shift,
            "analytic_boundary_shift_approach_vs_recede": analytic_shift,
            "shift_ratio_learned_over_analytic": (learned_shift / analytic_shift)
            if (learned_shift is not None and analytic_shift not in (None, 0)) else None,
            "note": "boundary dist = |x| of the h=0 (resp psi1=0) crossing on the obstacle's left/approach "
                    "ray (y=0); shift = |approach - recede| along the velocity axis."}


# ---- Part B: velocity-resolved gradient ----------------------------------------------

def part_b(fw, system):
    scene = make_scene([((0.0, 0.0), SINGLE_R)])
    clears = np.linspace(0.0, 1.5, 16)
    speeds = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    angles = np.linspace(0.0, np.pi, 13)            # 0 = toward obstacle, pi = away
    # position on +x side; approach direction (toward center) = -x
    approach = np.array([-1.0, 0.0]); perp = np.array([0.0, 1.0])
    heat = np.full((len(clears), len(speeds)), np.nan)     # ||∂h/∂v|| at angle=0 (toward)
    aligned_by_clear = np.full(len(clears), np.nan)        # approach-aligned |∂h/∂v . approach| at speed 1.5, toward
    analytic_aligned_by_clear = np.full(len(clears), np.nan)
    grad_by_angle = np.full(len(angles), np.nan)           # ||∂h/∂v|| vs angle at clearance 0.2, speed 1.5
    for i, cl in enumerate(clears):
        p = (SINGLE_C + (SINGLE_R + cl) * np.array([1.0, 0.0]))[None, :]
        for j, sp in enumerate(speeds):
            v = sp * approach                                # toward obstacle
            g, _, _ = grad_v(fw, system, p, v, scene)
            heat[i, j] = float(np.linalg.norm(g[0]))
        v15 = 1.5 * approach
        g, _, _ = grad_v(fw, system, p, v15, scene)
        aligned_by_clear[i] = float(abs(np.dot(g[0], approach)))     # approach-aligned component
        analytic_aligned_by_clear[i] = float(abs(np.dot(2.0 * (p[0] - SINGLE_C), approach)))  # |2(p-c).approach|
    cl_ref = 0.2; p_ref = (SINGLE_C + (SINGLE_R + cl_ref) * np.array([1.0, 0.0]))[None, :]
    for k, th in enumerate(angles):
        v = 1.5 * (np.cos(th) * approach + np.sin(th) * perp)
        g, _, _ = grad_v(fw, system, p_ref, v, scene)
        grad_by_angle[k] = float(np.linalg.norm(g[0]))
    # per-member at the collision-like regime (clearance 0.1, speed 2.0 toward)
    p_cm = (SINGLE_C + (SINGLE_R + 0.1) * np.array([1.0, 0.0]))[None, :]
    g_mean, _, per = grad_v(fw, system, p_cm, 2.0 * approach, scene, members=True)
    member_norms = [float(np.linalg.norm(per[0, m])) for m in range(per.shape[1])]
    return {
        "grid": {"clearances": clears.tolist(), "speeds": speeds.tolist(), "angles_rad": angles.tolist(),
                 "position_side": "+x", "approach_dir": approach.tolist()},
        "lg_norm_heat_clear_x_speed_toward": heat.tolist(),
        "approach_aligned_lg_by_clear_speed1.5": aligned_by_clear.tolist(),
        "analytic_approach_aligned_by_clear": analytic_aligned_by_clear.tolist(),
        "lg_norm_by_angle_clear0.2_speed1.5": grad_by_angle.tolist(),
        "lg_norm_fast_approach_shortclear(cl0.1,sp2.0)": float(np.linalg.norm(g_mean[0])),
        "lg_norm_slow_far(cl1.5,sp0.5)": float(heat[-1, 0]),
        "member_lg_norms_at_cl0.1_sp2.0": member_norms,
        "mean_lg_norm_at_cl0.1_sp2.0": float(np.linalg.norm(g_mean[0])),
        "note": "approach-aligned component = |∂h/∂v · (toward-obstacle unit)|; analytic aligned = |2(p-c)·approach|."}


# ---- Part C: value response to closing speed -----------------------------------------

def part_c(fw, system):
    scene = make_scene([((0.0, 0.0), SINGLE_R)])
    approach = np.array([-1.0, 0.0])
    speeds = np.linspace(0.0, 2.5, 26)
    out = {}
    for cl in (0.1, 0.3, 0.6):
        p = (SINGLE_C + (SINGLE_R + cl) * np.array([1.0, 0.0]))[None, :]
        hs = np.array([h_on(fw, system, p, s * approach, scene)[0] for s in speeds])
        psis = np.array([analytic_psi1(p, s * approach, SINGLE_C[None, :], np.array([SINGLE_R]))[0] for s in speeds])
        # slope dh/d(closing speed) by least squares; analytic dpsi1/ds = -2||p-c||
        sl_h = float(np.polyfit(speeds, hs, 1)[0])
        sl_psi = float(np.polyfit(speeds, psis, 1)[0])
        out[f"clearance_{cl}"] = {"h_at_speed0": float(hs[0]), "h_at_speed2.5": float(hs[-1]),
                                  "h_total_change": float(hs[-1] - hs[0]),
                                  "dh_d_closing_speed": sl_h, "analytic_dpsi1_d_closing_speed": sl_psi,
                                  "analytic_2norm_pc": float(2.0 * (SINGLE_R + cl))}
    _fig_c(fw, system, scene, approach, speeds)
    return out


def _fig_c(fw, system, scene, approach, speeds):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    for cl in (0.1, 0.3, 0.6):
        p = (SINGLE_C + (SINGLE_R + cl) * np.array([1.0, 0.0]))[None, :]
        hs = np.array([h_on(fw, system, p, s * approach, scene)[0] for s in speeds])
        psis = np.array([analytic_psi1(p, s * approach, SINGLE_C[None, :], np.array([SINGLE_R]))[0] for s in speeds])
        ax[0].plot(speeds, hs, "o-", label=f"clearance {cl}")
        ax[1].plot(speeds, psis, "s--", label=f"clearance {cl}")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("closing speed"); ax[0].set_ylabel("learned h")
    ax[0].set_title("learned h vs closing speed (danger=h>0)"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("closing speed"); ax[1].set_ylabel("analytic psi1")
    ax[1].set_title("analytic psi1 vs closing speed (danger=psi1<0)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "veldep_closing_speed.png"); plt.close(fig)


def _fig_b(rep_b):
    clears = np.array(rep_b["grid"]["clearances"]); speeds = np.array(rep_b["grid"]["speeds"])
    heat = np.array(rep_b["lg_norm_heat_clear_x_speed_toward"])
    angles = np.array(rep_b["grid"]["angles_rad"])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=140)
    im = ax[0].imshow(heat, origin="lower", aspect="auto", cmap="viridis",
                      extent=[speeds[0], speeds[-1], clears[0], clears[-1]])
    ax[0].set_xlabel("speed (toward obstacle)"); ax[0].set_ylabel("clearance")
    ax[0].set_title("||∂h/∂v|| (toward-obstacle velocity)"); fig.colorbar(im, ax=ax[0])
    ax[1].plot(clears, rep_b["approach_aligned_lg_by_clear_speed1.5"], "o-", color="#1f77b4", label="learned approach-aligned")
    ax[1].plot(clears, rep_b["analytic_approach_aligned_by_clear"], "s--", color="#d62728", label="analytic |2(p-c)·a|")
    ax[1].set_xlabel("clearance"); ax[1].set_ylabel("approach-aligned |∂h/∂v|"); ax[1].set_title("safety-relevant component vs clearance")
    ax[1].legend(fontsize=8)
    ax[2].plot(angles, rep_b["lg_norm_by_angle_clear0.2_speed1.5"], "o-", color="#2ca02c")
    ax[2].set_xlabel("angle v vs toward-obstacle (rad)"); ax[2].set_ylabel("||∂h/∂v||")
    ax[2].set_title("gradient vs velocity angle (clr 0.2, sp 1.5)")
    fig.tight_layout(); fig.savefig(OUT / "veldep_gradient.png"); plt.close(fig)


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    fw, config, _ = load_framework_from_checkpoint(CKPT)
    system = make_system(config)
    fw.value_net.to(DEVICE, DTYPE).eval(); fw.policy_net.to(DEVICE, DTYPE).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"alpha1": ALPHA1, "r_margin": R_MARGIN, "goal": GOAL.tolist(),
           "multi_layout": [{"center": list(c), "radius": r} for c, r in MULTI],
           "single_obstacle": {"center": SINGLE_C.tolist(), "radius": SINGLE_R},
           "chain_rule": "L_g h = ∂h/∂v = grad_x(h)[2:4]; analytic L_g psi1 = 2(p-c)."}
    print("[A] velocity-dependent boundary ...")
    rep["partA_boundary"] = part_a(fw, system)
    print("[B] velocity-resolved gradient ...")
    rep["partB_gradient"] = part_b(fw, system); _fig_b(rep["partB_gradient"])
    print("[C] value response to closing speed ...")
    rep["partC_value_response"] = part_c(fw, system)
    (OUT / "velocity_dependence_summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("summary ->", OUT / "velocity_dependence_summary.json")
    a = rep["partA_boundary"]
    print("A boundary shift learned/analytic:", a["learned_boundary_shift_approach_vs_recede"],
          a["analytic_boundary_shift_approach_vs_recede"], "ratio", a["shift_ratio_learned_over_analytic"])
    print("B fast-approach-shortclear lg:", rep["partB_gradient"]["lg_norm_fast_approach_shortclear(cl0.1,sp2.0)"],
          "members:", rep["partB_gradient"]["member_lg_norms_at_cl0.1_sp2.0"], "mean:", rep["partB_gradient"]["mean_lg_norm_at_cl0.1_sp2.0"])
    print("C slopes:", {k: (v["dh_d_closing_speed"], v["analytic_dpsi1_d_closing_speed"]) for k, v in rep["partC_value_response"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
