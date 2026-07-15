"""v2.6.0 Stage 0 M6 — GATE P1: offline theory verification (note Sec 8.5 E1).

With the EXACT numerical V^{h,pi} (sup over the nominal closed-loop rollout; NO learning), sample states
near the active-boundary degeneracy set B_0 = {V=0} INT {h=0} (Lemma 5.1) and compare ||L_g V|| under
(i) position-only h=phi(p)  and  (ii) h_star=phi + c (v^T Re) (Thm 5.2 vs 5.3). Sweep the gain c over
[0.1,1.0] (changes.md Sec 4) for the admissible interval (Sec 8.4: C1 authority on B_0 vs C2 distortion off).

B_0 is defined by the POSITION-ONLY barrier (the theory's degeneracy carrier): near-B_0 := {|V^{phi,pi}|<dV}
INT {|phi(x0)|<dh}. On those SAME states we measure ||L_g V|| for both barriers -> "does h_star restore
authority where phi is degenerate" (E1). Degenerate := ||L_g V|| < tau (the HardNet singular threshold 5e-4).

HALT if h_star does not measurably restore ||L_g V|| near B_0 relative to phi.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from src.common.filter_hardnet import _cbf_terms, _SINGULAR_LG_THRESHOLD
from src.common.quadrotor_barrier import make_exact_value_fn, approach_speed, phi_value
from src.frameworks.jt_pncbf.train import make_system

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
TAU = _SINGULAR_LG_THRESHOLD           # 5e-4 degeneracy threshold (HardNet singular floor)
DV = 0.10                              # |V^{phi}| < DV  (near the active boundary V=0)
DH = 0.10                              # |phi(x0)| < DH  (near h=0)


def _cfg() -> dict:
    b = yaml_safe(REPO / "src/configs/base_config.yaml")
    e = yaml_safe(REPO / "src/configs/exp_config.yaml")

    def m(a, o):
        d = dict(a)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, Mapping) and isinstance(d.get(k), Mapping) else v
        return d

    return m(b, e)


def yaml_safe(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def _sample_candidates(cfg, n, dev, dtype, seed=20260715):
    """Single-obstacle B_0 candidates: agent at ~phi=0 around an obstacle, velocity inward, goal beyond."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h_scale = float(cfg["env"]["h_scale"])
    r = 0.3 + 0.5 * torch.rand(n, generator=g)                     # radius U[0.3,0.8]
    psi = (2 * torch.rand(n, generator=g) - 1) * np.pi             # approach angle
    band = (torch.rand(n, generator=g) - 0.5) * (0.9 * h_scale)    # jitter across the phi ramp (|phi|<~1)
    dist = r + 0.5 * h_scale + band                                # ~ boundary (phi ~ 0)
    dirx = torch.stack([torch.cos(psi), torch.sin(psi)], dim=1)
    p = dist.unsqueeze(1) * dirx
    # velocity direction is INDEPENDENT of the approach direction (uniform angle): B_0's degenerate
    # states are where t*=0 (sup at t=0), i.e. velocity outward/tangential so phi is non-increasing.
    # Forcing inward velocity would sample only t*>0 (non-degenerate). Cover all directions.
    vpsi = (2 * torch.rand(n, generator=g) - 1) * np.pi
    vdir = torch.stack([torch.cos(vpsi), torch.sin(vpsi)], dim=1)
    speed = 0.3 + 1.9 * torch.rand(n, generator=g)                 # speed U[0.3,2.2]
    v = speed.unsqueeze(1) * vdir
    theta = (2 * torch.rand(n, generator=g) - 1) * np.pi
    omega = (2 * torch.rand(n, generator=g) - 1) * 1.0
    x = torch.stack([p[:, 0], p[:, 1], theta, v[:, 0], v[:, 1], omega], dim=1).to(device=dev, dtype=dtype)
    # per-state single-obstacle scene (batched tensors); goal on the far side so the nominal drives inward
    K = 12
    C = torch.zeros(n, K, 2, device=dev, dtype=dtype)
    R = torch.zeros(n, K, device=dev, dtype=dtype)
    A = torch.zeros(n, K, dtype=torch.bool, device=dev)
    R[:, 0] = r.to(device=dev, dtype=dtype); A[:, 0] = True        # obstacle 0 at origin
    goal = (-2.0 * dirx).to(device=dev, dtype=dtype)
    scene = SimpleNamespace(obstacle_centers=C, obstacle_radii=R, obstacle_active=A, goal=goal)
    return x, scene


def _lg_norm(system, value_fn, x, scene):
    _, _, lg = _cbf_terms(system, value_fn, x, scene, torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype),
                          create_graph=False)
    return torch.linalg.norm(lg.detach(), dim=1)


def main():
    cfg = _cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    system = make_system(cfg)
    system.u_bounds = system.u_bounds.to(device=dev, dtype=dtype)
    h_scale = float(cfg["env"]["h_scale"]); dt = float(cfg["env"]["dt"]); H = 25
    c0 = float(cfg["env"]["quadrotor_planar"]["c_gain"])

    x, scene = _sample_candidates(cfg, 40000, dev, dtype)
    phi0 = phi_value(x, scene, h_scale).detach()

    # exact V under position-only phi, select near-B_0
    V_phi_fn = make_exact_value_fn(system, c0, h_scale, dt, H, position_only=True)
    with torch.no_grad():
        V_phi = V_phi_fn(x, scene)
    near = (V_phi.abs() < DV) & (phi0.abs() < DH)
    n_near = int(near.sum())
    print(f"candidates=40000  near-B_0 (|V_phi|<{DV} & |phi|<{DH}) = {n_near}")
    if n_near < 200:
        raise SystemExit(f"HALT: too few near-B_0 samples ({n_near}); widen sampling.")
    xb = x[near]; sb = SimpleNamespace(obstacle_centers=scene.obstacle_centers[near],
                                       obstacle_radii=scene.obstacle_radii[near],
                                       obstacle_active=scene.obstacle_active[near], goal=scene.goal[near])

    # ||L_g V|| on the SAME near-B_0 states: position-only vs h_star(c0)
    lg_phi = _lg_norm(system, make_exact_value_fn(system, c0, h_scale, dt, H, position_only=True), xb, sb)
    lg_hs = _lg_norm(system, make_exact_value_fn(system, c0, h_scale, dt, H, position_only=False), xb, sb)

    def stats(t):
        t = t.cpu().numpy()
        return dict(median=float(np.median(t)), p10=float(np.percentile(t, 10)),
                    p90=float(np.percentile(t, 90)), mean=float(t.mean()),
                    degen_frac=float((t < TAU).mean()))

    s_phi, s_hs = stats(lg_phi), stats(lg_hs)
    print(f"\n||L_g V|| near B_0  (tau={TAU}):")
    print(f"  position-only phi : median={s_phi['median']:.5f} p10={s_phi['p10']:.5f} p90={s_phi['p90']:.5f} "
          f"DEGENERATE_frac={s_phi['degen_frac']:.3f}")
    print(f"  h_star (c={c0})    : median={s_hs['median']:.5f} p10={s_hs['p10']:.5f} p90={s_hs['p90']:.5f} "
          f"DEGENERATE_frac={s_hs['degen_frac']:.3f}")

    # E1 / P1 gate: h_star restores authority where phi is degenerate
    phi_degenerate = lg_phi < TAU
    restored_on_phi_degenerate = float((lg_hs[phi_degenerate] >= TAU).float().mean()) if int(phi_degenerate.sum()) else float("nan")
    median_ratio = s_hs["median"] / max(s_phi["median"], 1e-12)
    P1 = (s_phi["degen_frac"] > 0.30) and (s_hs["degen_frac"] < 0.10) and (s_hs["median"] > 10 * max(s_phi["median"], 1e-9))
    print(f"\nE1: on the {int(phi_degenerate.sum())} phi-degenerate states, h_star restores ||L_g V||>=tau in "
          f"{restored_on_phi_degenerate:.3f}; median ratio h_star/phi = {median_ratio:.1f}x")

    # c-sweep: C1 (authority: low degen frac) vs C2 (distortion: |c*(v^T Re)| additive shift to phi)
    aspeed = approach_speed(xb).detach()
    sweep = []
    for c in [round(0.1 * k, 1) for k in range(1, 11)]:
        lg_c = _lg_norm(system, make_exact_value_fn(system, c, h_scale, dt, H, position_only=False), xb, sb)
        degen = float((lg_c < TAU).float().mean().cpu())
        distortion = float((c * aspeed.abs()).mean().cpu())        # mean additive shift to phi (phi in [-1,1])
        distort_frac = float(((c * aspeed.abs()) > 0.5).float().mean().cpu())  # frac exceeding half the phi range
        sweep.append(dict(c=c, median_lg=float(lg_c.median().cpu()), degen_frac=degen,
                          mean_distortion=distortion, distort_gt_half_frac=distort_frac))
        print(f"  c={c:.1f}: median||L_gV||={sweep[-1]['median_lg']:.4f} degen_frac={degen:.3f} "
              f"mean|c*v.Re|={distortion:.3f} frac(>0.5)={distort_frac:.3f}")

    # Final c (C1 authority INT C2 non-distortion): C1 is met for every c>=0.1 (degen_frac 0, since the
    # thrust-channel floor is c/m=c >> tau). The binding constraint is C2. Rule: LARGEST c whose mean
    # additive shift to phi stays <= 0.25 (a quarter of the phi range [-1,1]) -> maximum authority margin
    # for the learned V_hat (R2/O3) at still-small safe-set distortion.
    admissible = [s for s in sweep if s["degen_frac"] < 0.10 and s["mean_distortion"] <= 0.25]
    chosen = max((s["c"] for s in admissible), default=None)
    print(f"\nadmissible c (degen<0.10 & mean|c*v.Re|<=0.25): {[s['c'] for s in admissible]}")
    print(f"chosen final c = {chosen} (largest admissible: max authority margin at <=0.25 distortion)")
    print(f"\nP1 GATE: {'PASS' if P1 else 'FAIL -> HALT'}")

    out = dict(n_candidates=40000, n_near_B0=n_near, tau=TAU, dV=DV, dh=DH, horizon=H,
               lg_phi=s_phi, lg_hstar_c0=s_hs, c0=c0,
               restored_on_phi_degenerate=restored_on_phi_degenerate, median_ratio=median_ratio,
               sweep=sweep, chosen_c=chosen, P1=bool(P1))
    json.dump(out, open(SP / "quadrotor_p1_gate.json", "w"), indent=2)
    print("saved:", SP / "quadrotor_p1_gate.json")


if __name__ == "__main__":
    main()
