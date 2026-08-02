"""v2.8.0 S2 B8 — inertness gate. Evaluate double_integrator and unicycle on their canonical pools and
persist the full per-episode outcome vector + eval row. Run --tag before (pre Part B) and --tag after
(post Part B); the two must be byte-identical (angular_rate is a structural zero on these systems, so the
angular reach condition is vacuous). Any difference is a HALT.
Artifact -> data/runs/v2.8.0/s2_terminal/b8_<tag>.json."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from src.eval.evaluate import evaluate
from src.eval.run_full import _load_framework as load_fw

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
OUT = REPO / "data/runs/v2.8.0/s2_terminal"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYS = {
    "double_integrator": (REPO / "data/secured_data/v2.3.0/seed42/checkpoints/best.pt",
                          REPO / "data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"),
    "unicycle": (REPO / "data/secured_data/v2.2.2/seed42/checkpoints/best.pt",
                 REPO / "data/secured_data/pools/eval_full_unicycle_n2000_seed23456.pkl"),
}

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
ap.add_argument("--angrate", type=float, default=None,
                help="inject goal_angrate_radius (the 'after' run sets 0.48 to test active-condition inertness)")
a = ap.parse_args()
over = {"env": {"goal_angrate_radius": a.angrate}} if a.angrate is not None else None
rep = {"tag": a.tag, "device": str(DEV), "goal_angrate_radius_injected": a.angrate, "systems": {}}
for name, (ck, pool) in SYS.items():
    fw, cfg, ckpt = load_fw(ck, config_overrides=over)
    for n in ("value_net", "policy_net"):
        m = getattr(fw, n, None); m.to(DEV) if m is not None else None
    res = evaluate(fw, pool, cfg, mode="final", step=int(ckpt["step"]), ckpt_name=ck.name,
                   max_scenes=None, include_lqr_baseline=False)
    r = res.eval_row
    outcomes = [ep["outcome"] for ep in res.episode_rows]
    oh = hashlib.sha256(("|".join(outcomes)).encode()).hexdigest()
    rep["systems"][name] = {
        "checkpoint": ck.name, "n_episodes": len(outcomes),
        "outcome_hash": oh,
        "vector": {k: (float(r[k]) if r.get(k) is not None else None)
                   for k in ("cps", "reach", "collision", "oob", "stuck", "timeout", "infeasibility", "saturation_rate")},
    }
    print(f"{name} [{a.tag}]: cps {r['cps']:.6f} reach {r['reach']:.6f} coll {r['collision']:.6f} "
          f"outcome_hash {oh[:12]}", flush=True)
(OUT / f"b8_{a.tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
print("B8", a.tag, "->", OUT / f"b8_{a.tag}.json")
