"""Stuck-episode attribution (read-only).

For a checkpoint on a frozen pool, roll episodes as run_full's deployed path and, for every STUCK
episode, record why it stalled: terminal speed ||v||, min surface clearance at termination, whether
the terminal state sits on the analytic brake-envelope boundary (|surf_clr - v_in^2/(2 u_max)| small
OR ||v||~0 with small clearance), a corridor/collinear scene tag (an obstacle blocks the straight
start->goal path), and the filter-active fraction along the episode. Emits JSON + markdown, and a
per-episode outcome vector so two runs can be diffed for reach->stuck flips.

Usage:
  python -m scripts.analysis.stuck_attribution --ckpt PATH [--pool PATH] [--max-scenes N]
         [--tag NAME] [--out DIR] [--baseline-json PATH]
Nothing is trained or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.outcomes import resolve_outcome, step_outcomes
from src.eval.build_pools import load_pool
from src.eval.evaluate import _filter_adapter, _tensor_options
from src.eval.rollout import rollout_eval
from src.eval.run_full import _load_framework
from src.envs.scene_batch import batch_scenes, initial_states_from_batch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POOL = REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"

V_REST_TOL = 0.10        # "at rest" terminal speed
CLR_NEAR_TOL = 0.30      # "near boundary" surface clearance
ENVELOPE_TOL = 0.10      # |surf_clr - v_in^2/(2 u_max)| band for envelope-boundary tag
CORRIDOR_MARGIN = 0.10   # obstacle-to-(start-goal) perpendicular distance margin for corridor tag
FILTER_TOL = 1.0e-3      # ||u_safe - u_nom|| above which the filter is "active" at a step


def _perp_dist_point_segment(c, a, b):
    ab = b - a
    denom = float(torch.dot(ab, ab))
    if denom < 1.0e-12:
        return float(torch.linalg.norm(c - a))
    t = float(torch.clamp(torch.dot(c - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(torch.linalg.norm(c - proj))


def attribute(ckpt: Path, pool_path: Path, max_scenes):
    framework, config, checkpoint = _load_framework(ckpt)
    system = framework.system
    dtype, device = _tensor_options(system, framework)
    max_steps = int(config["eval"]["max_steps"])
    dt = float(config["env"]["dt"])
    u_max = float(config["env"]["bounds"][system.name]["u_max"])

    scenes = load_pool(pool_path).scenes
    if max_scenes is not None:
        scenes = scenes[:max_scenes]
    batched = batch_scenes(scenes, device=device, dtype=dtype)
    x0 = initial_states_from_batch(batched)
    with torch.no_grad():
        res = rollout_eval(system, framework.policy, _filter_adapter(framework),
                           batched, x0, max_steps, dt, config=config)
    masks = step_outcomes(res.states, batched, system, config)
    resolved = resolve_outcome(masks)
    outcomes = resolved.outcome
    event_step = resolved.event_step
    du = torch.linalg.norm(res.u_safe - res.u_nom, dim=-1)     # [T,B] filter modification magnitude

    centers = batched.obstacle_centers
    radii = batched.obstacle_radii
    active = batched.obstacle_active
    eps = 1.0e-9

    details = []
    n_stuck = 0
    for b, o in enumerate(outcomes):
        if o != "stuck":
            continue
        n_stuck += 1
        te = int(event_step[b].item())
        te = max(1, min(te, res.states.shape[0] - 1))
        xT = res.states[te, b]
        p = xT[:2]
        v = xT[2:4]
        vnorm = float(torch.linalg.norm(v))
        act = active[b]
        c = centers[b][act]
        r = radii[b][act]
        dist = torch.linalg.norm(p.unsqueeze(0) - c, dim=1)
        clr = dist - r
        j = int(torch.argmin(clr).item())
        min_clear = float(clr[j])
        cp = c[j] - p
        v_in = float(torch.relu(torch.dot(v, cp) / (torch.linalg.norm(cp) + eps)))
        d_stop = v_in * v_in / (2.0 * u_max)
        envelope = bool(abs(min_clear - d_stop) < ENVELOPE_TOL
                        or (vnorm < V_REST_TOL and min_clear < CLR_NEAR_TOL))
        start = torch.as_tensor(scenes[b].start, dtype=dtype, device=device)
        goal = torch.as_tensor(scenes[b].goal, dtype=dtype, device=device)
        perp = min(_perp_dist_point_segment(c[k], start, goal) - float(r[k]) for k in range(c.shape[0]))
        corridor = bool(perp < CORRIDOR_MARGIN)
        filt_active = float((du[:te, b] > FILTER_TOL).to(dtype).mean())
        details.append({
            "scene": b, "terminal_step": te, "terminal_speed": vnorm,
            "min_clearance": min_clear, "v_in": v_in, "d_stop": d_stop,
            "envelope_boundary": envelope, "corridor": corridor,
            "filter_active_frac": filt_active,
        })

    def mean(key):
        return float(sum(d[key] for d in details) / len(details)) if details else 0.0
    def frac(key):
        return float(sum(1 for d in details if d[key]) / len(details)) if details else 0.0

    stats = {
        "checkpoint": str(ckpt), "checkpoint_step": int(checkpoint.get("step", -1)),
        "pool": str(pool_path), "n_scenes": len(scenes),
        "n_stuck": n_stuck, "stuck_rate": n_stuck / len(scenes) if scenes else 0.0,
        "mean_terminal_speed": mean("terminal_speed"),
        "mean_min_clearance": mean("min_clearance"),
        "frac_envelope_boundary": frac("envelope_boundary"),
        "frac_corridor": frac("corridor"),
        "mean_filter_active_frac": mean("filter_active_frac"),
    }
    return stats, outcomes, details


def main() -> int:
    ap = argparse.ArgumentParser(description="Stuck-episode attribution (read-only).")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--out", type=Path, default=REPO / "docs/versions/v2.4.0/attribution")
    ap.add_argument("--baseline-json", type=Path, default=None,
                    help="a prior run's outcomes JSON to compute reach->stuck flips against")
    args = ap.parse_args()

    stats, outcomes, details = attribute(args.ckpt, args.pool, args.max_scenes)
    flips = None
    if args.baseline_json is not None:
        base = json.loads(args.baseline_json.read_text())["per_episode_outcomes"]
        n = min(len(base), len(outcomes))
        # resolve_outcome labels a goal-reaching episode "goal" (not "reach")
        reach_to_stuck = [i for i in range(n) if base[i] == "goal" and outcomes[i] == "stuck"]
        stuck_to_reach = [i for i in range(n) if base[i] == "stuck" and outcomes[i] == "goal"]
        flips = {"baseline": str(args.baseline_json),
                 "reach_to_stuck_count": len(reach_to_stuck),
                 "stuck_to_reach_count": len(stuck_to_reach),
                 "reach_to_stuck_scenes": reach_to_stuck[:100],
                 "stuck_to_reach_scenes": stuck_to_reach[:100]}
        stats["flips_vs_baseline"] = {k: flips[k] for k in ("reach_to_stuck_count", "stuck_to_reach_count")}

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats, "per_episode_outcomes": outcomes, "stuck_details": details}
    if flips is not None:
        payload["flips"] = flips
    (args.out / f"{args.tag}.stuck.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    keys = ["n_scenes", "n_stuck", "stuck_rate", "mean_terminal_speed", "mean_min_clearance",
            "frac_envelope_boundary", "frac_corridor", "mean_filter_active_frac"]
    md = [f"# stuck attribution — {args.tag}", "",
          f"checkpoint: `{stats['checkpoint']}` (step {stats['checkpoint_step']})", "",
          "| metric | value |", "|---|---:|"]
    for k in keys:
        v = stats[k]
        md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    if flips is not None:
        md += ["", f"reach->stuck vs baseline: {flips['reach_to_stuck_count']} | "
                   f"stuck->reach: {flips['stuck_to_reach_count']}"]
    (args.out / f"{args.tag}.stuck.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"OUT_JSON={args.out / f'{args.tag}.stuck.json'}")
    for k in keys:
        print(f"  {k} = {stats[k]}")
    if flips is not None:
        print(f"  reach_to_stuck = {flips['reach_to_stuck_count']}  stuck_to_reach = {flips['stuck_to_reach_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
