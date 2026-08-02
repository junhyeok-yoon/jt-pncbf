"""v2.8.0 M1 — the tilt-restricted navigation pool.

Rejection-sample from the SAME generator the canonical pool uses (sampler_params verified == the canonical
manifest), keeping draws whose initial tilt arccos(R(q)[2,2]) <= 60 deg. Everything else is the generator's
own, unchanged. Writes eval_navcone_quadrotor-3d-d2r_n2000_seed<N> (pkl + manifest + sha256) into
data/secured_data/pools/ and hash-verifies. Compares the accepted set's distributions to the canonical pool
restricted to the SAME tilt cut (clearance, vertical velocity, angular speed, obstacle count); if any differ
materially the rejection is not clean and that is reported."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.eval.build_pools import (_eval_sampler, write_pool, load_pool, sha256_file, pool_variant,
                                  EvaluationPool)
from src.frameworks.jt_pncbf.train import load_effective_config

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
SECURED = REPO / "data/secured_data/pools"
CANON = SECURED / "eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
N_SCENES, SEED, TILT_CUT = 2000, 34567, 60.0
OUT = REPO / "data/runs/v2.8.0"; OUT.mkdir(parents=True, exist_ok=True)


def tilt_deg(q):
    q = np.asarray(q); return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))


def scene_feats(s):
    q = np.asarray(s.initial_attitude_quat)
    vz = float(np.asarray(s.initial_velocity)[2]) if s.initial_velocity is not None else 0.0
    om = float(np.linalg.norm(s.initial_omega_vec)) if s.initial_omega_vec is not None else 0.0
    return {"tilt": tilt_deg(q), "clearance": float(np.asarray(s.start)[2] + 4.0), "vz": vz,
            "omega": om, "nobs": int(np.sum(s.obstacle_active))}


cfg = load_effective_config(); cfg["run"]["system"] = "quadrotor_3d"
variant = pool_variant(cfg, "quadrotor_3d")
samp = _eval_sampler(cfg)
rng = np.random.default_rng(SEED)
accepted, n_draws = [], 0
while len(accepted) < N_SCENES:
    s = samp(rng, cfg, "quadrotor_3d"); n_draws += 1
    if tilt_deg(s.initial_attitude_quat) <= TILT_CUT:
        accepted.append(s)
acc_rate = N_SCENES / n_draws
print(f"navcone: {N_SCENES} accepted from {n_draws} draws; acceptance {acc_rate:.4f} (SO3 pred 0.25)")

pool = EvaluationPool(name="navcone", system="quadrotor_3d", n_scenes=N_SCENES, seed=SEED, scenes=accepted)
art = write_pool(pool, cfg, output_dir=SECURED, variant=variant)
recorded = json.loads(art.manifest_path.read_text())["pool_sha256"]
disk = sha256_file(art.pool_path)
print(f"wrote {art.pool_path.name} sha {disk[:12]} manifest_match {recorded == disk}")
if recorded != disk:
    raise SystemExit("HALT: navcone pool hash mismatch")

# distribution comparison: navcone vs canonical restricted to tilt<=60
nav = [scene_feats(s) for s in accepted]
canon = [scene_feats(s) for s in load_pool(CANON).scenes]
canon_cut = [f for f in canon if f["tilt"] <= TILT_CUT]
def summ(fs, k):
    v = np.array([f[k] for f in fs]); return [float(v.mean()), float(v.std()), float(np.percentile(v, 10)),
                                              float(np.percentile(v, 50)), float(np.percentile(v, 90))]
keys = ["tilt", "clearance", "vz", "omega", "nobs"]
cmp = {}
print(f"\ncanonical restricted to tilt<=60: n={len(canon_cut)} (of {len(canon)})")
print(f"{'feat':10s} {'navcone[mean,std,p10,p50,p90]':>44s}   {'canon-cut[mean,std,p10,p50,p90]':>44s}   rel|dmean|")
for k in keys:
    a, b = summ(nav, k), summ(canon_cut, k)
    reld = abs(a[0] - b[0]) / (abs(b[0]) + 1e-9)
    cmp[k] = {"navcone": a, "canon_restricted": b, "rel_abs_mean_diff": reld}
    print(f"{k:10s} {str([round(x,3) for x in a]):>44s}   {str([round(x,3) for x in b]):>44s}   {reld:.3f}")
material = {k: cmp[k]["rel_abs_mean_diff"] > 0.10 for k in keys if k != "tilt"}  # >10% mean shift = material
cmp["material_difference_any"] = any(material.values()); cmp["material_by_feat"] = material

rep = {"pool_stem": art.pool_path.stem, "seed": SEED, "n_scenes": N_SCENES, "n_draws": n_draws,
       "acceptance_rate": acc_rate, "so3_prediction": 0.25, "sha256": disk, "variant": variant,
       "tilt_cut_deg": TILT_CUT, "canonical_restricted_n": len(canon_cut),
       "distribution_comparison": cmp,
       "clean_rejection": not cmp["material_difference_any"]}
(OUT / "navpool_build.json").write_text(json.dumps(rep, indent=2) + "\n")
print(f"\nclean_rejection (no >10% mean shift on clr/vz/omega/nobs): {rep['clean_rejection']}")
print("->", OUT / "navpool_build.json")
