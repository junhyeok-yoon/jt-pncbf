"""v2.8.0 S1 — the mixed in-loop pool for quadrotor_3d dual scoring.

1000 initial conditions from the canonical full-attitude distribution + 1000 from the tilt<=60deg
distribution (same generator, same rejection rule as the navcone pool, fresh seed), interleaved, each
carrying a provenance flag (full / tilt60). Registered like any pool (manifest + SHA-256, secured copy,
hash-verified); the manifest carries the per-IC provenance array and flags this as an IN-LOOP pool that is
never a ledger-row source. Keeps in-loop cost at one evaluation per cadence while exposing both regimes as
sub-scores; halves are n=1000 (trend, not verdict); best.pt selection stays on the blended cps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.eval.build_pools import (_eval_sampler, write_pool, load_pool, sha256_file, pool_variant,
                                  EvaluationPool)
from src.frameworks.jt_pncbf.train import load_effective_config

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SECURED = REPO / "data/secured_data/pools"
N_HALF, SEED, TILT_CUT = 1000, 45678, 60.0
OUT = REPO / "data/runs/v2.8.0"; OUT.mkdir(parents=True, exist_ok=True)


def tilt_deg(q):
    q = np.asarray(q); return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))


cfg = load_effective_config(); cfg["run"]["system"] = "quadrotor_3d"
variant = pool_variant(cfg, "quadrotor_3d")
samp = _eval_sampler(cfg)
rng = np.random.default_rng(SEED)

full, tilt60, n_draws_tilt = [], [], 0
while len(full) < N_HALF:                       # full-attitude: keep every draw
    full.append(samp(rng, cfg, "quadrotor_3d"))
while len(tilt60) < N_HALF:                      # tilt<=60: rejection, same rule as navcone
    s = samp(rng, cfg, "quadrotor_3d"); n_draws_tilt += 1
    if tilt_deg(s.initial_attitude_quat) <= TILT_CUT:
        tilt60.append(s)

# interleave full / tilt60, carry provenance
scenes, provenance = [], []
for i in range(N_HALF):
    scenes.append(full[i]); provenance.append("full")
    scenes.append(tilt60[i]); provenance.append("tilt60")
assert len(scenes) == 2 * N_HALF

pool = EvaluationPool(name="inloop-mixed", system="quadrotor_3d", n_scenes=2 * N_HALF, seed=SEED, scenes=scenes)
art = write_pool(pool, cfg, output_dir=SECURED, variant=variant)
# augment the manifest: provenance array + in-loop marker (write_pool already set pool_sha256 on the .pkl)
man = json.loads(art.manifest_path.read_text())
man["provenance"] = provenance
man["pool_role"] = "in_loop"
man["never_ledger_source"] = True
man["mixed_halves"] = {"full": N_HALF, "tilt60": N_HALF, "tilt60_acceptance": N_HALF / n_draws_tilt}
art.manifest_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
disk = sha256_file(art.pool_path)
print(f"wrote {art.pool_path.name} sha {disk[:12]}  (manifest re-written with provenance; pkl unchanged)")
print(f"tilt60 acceptance {N_HALF/n_draws_tilt:.4f}; interleaved {len(scenes)} scenes ({N_HALF} full + {N_HALF} tilt60)")

# hash-verify the secured copy against its manifest's recorded pkl sha
recorded = man["pool_sha256"]
print(f"hash-verify: manifest pool_sha256 {recorded[:12]} == disk {disk[:12]} -> {recorded == disk}")
# provenance-tilt cross-check: full half spans SO(3), tilt60 half all <=60
tf = np.array([tilt_deg(s.initial_attitude_quat) for s in full])
tt = np.array([tilt_deg(s.initial_attitude_quat) for s in tilt60])
print(f"full-half tilt: max {tf.max():.1f} frac>60 {(tf>60).mean():.3f} | tilt60-half tilt: max {tt.max():.1f} (all<=60: {(tt<=60).all()})")
rep = {"pool_stem": art.pool_path.stem, "seed": SEED, "n_scenes": 2 * N_HALF, "sha256": disk,
       "tilt60_acceptance": N_HALF / n_draws_tilt, "hash_verified": recorded == disk,
       "full_half_frac_over_60": float((tf > 60).mean()), "tilt60_half_all_le_60": bool((tt <= 60).all())}
(OUT / "mixed_pool_build.json").write_text(json.dumps(rep, indent=2) + "\n")
print("->", OUT / "mixed_pool_build.json")
