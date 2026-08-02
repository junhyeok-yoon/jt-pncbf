"""v2.8.0 R1 — collect the driver/standalone cluster values, characterize the two clusters, and produce
the per-episode flip list (one driver run vs one standalone run, tilt60 cell). Also folds in the two
pre-existing fresh-process runs (w4_proof = driver lineage; s4_proofs = standalone lineage) as confirming
points. Writes data/runs/v2.8.0/r1/cluster_report.json and prints a human summary."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
R1 = REPO / "data/runs/v2.8.0/r1"


def _load(p):
    return json.loads(Path(p).read_text())


def _eps_by_idx(path):
    with open(path) as f:
        return {int(r["episode_idx"]): r for r in csv.DictReader(f)}


def main():
    driver = []   # the 5 R1 driver runs (the registered lineage source)
    standalone = []
    for i in (1, 2, 3, 4, 5):
        r = _load(R1 / f"driver_{i}" / "result.json")
        driver.append({"run": f"driver_{i}", **r})
    for i in (1, 2, 3):
        r = _load(R1 / f"standalone_{i}" / "result.json")
        standalone.append({"run": f"standalone_{i}", **r})

    # confirming pre-existing fresh-process runs
    w4 = _load(REPO / "data/runs/v2.8.0/w4_proof/proof.json")
    confirm_driver = {"run": "w4_proof(prior)", "tilt60_cps": w4["proof2_tilt60"]["cps"],
                      "bandopen_cps": w4["proof3_bandopen"]["cps"]}
    s4 = _load(REPO / "data/runs/v2.8.0/s4_proofs.json")
    confirm_standalone = {"run": "s4_proofs(prior)", "tilt60_cps": s4["proof2_tilt60"]["cps"],
                          "bandopen_cps": s4["proof3_bandopen"]["cps"]}

    def stats(rows, key):
        vals = [r[key] for r in rows]
        return {"min": min(vals), "max": max(vals), "median": statistics.median(vals),
                "spread": max(vals) - min(vals), "n": len(vals)}

    rep = {
        "driver_runs": driver, "standalone_runs": standalone,
        "confirm_driver_prior": confirm_driver, "confirm_standalone_prior": confirm_standalone,
        "driver_tilt60": stats(driver, "tilt60_cps"), "driver_bandopen": stats(driver, "bandopen_cps"),
        "standalone_tilt60": stats(standalone, "tilt60_cps"), "standalone_bandopen": stats(standalone, "bandopen_cps"),
    }
    rep["gap_tilt60"] = rep["driver_tilt60"]["median"] - rep["standalone_tilt60"]["median"]
    rep["gap_bandopen"] = rep["driver_bandopen"]["median"] - rep["standalone_bandopen"]["median"]
    # third-cluster / within-cluster check: every driver value must sit far nearer the driver median than
    # the standalone median (and vice versa), and within-cluster spread must be << the between-cluster gap.
    max_within = max(rep["driver_tilt60"]["spread"], rep["driver_bandopen"]["spread"],
                     rep["standalone_tilt60"]["spread"], rep["standalone_bandopen"]["spread"])
    min_gap = min(abs(rep["gap_tilt60"]), abs(rep["gap_bandopen"]))
    rep["max_within_cluster_spread"] = max_within
    rep["min_between_cluster_gap"] = min_gap
    rep["two_clusters_clean"] = max_within < 0.5 * min_gap

    # per-episode flip list: driver_1 vs standalone_1, tilt60 cell
    d_eps = _eps_by_idx(R1 / "driver_1" / "eval" / "tilt60" / "eval_episodes.csv")
    s_eps = _eps_by_idx(R1 / "standalone_1" / "tilt60_episodes.csv")
    flips = []
    for idx in sorted(set(d_eps) & set(s_eps)):
        d, s = d_eps[idx], s_eps[idx]
        dc, sc = float(d["cps_episode"]), float(s["cps_episode"])
        if abs(dc - sc) > 1e-9 or d["outcome"] != s["outcome"]:
            flips.append({"idx": idx, "driver_outcome": d["outcome"], "standalone_outcome": s["outcome"],
                          "driver_cause": d["collision_cause"], "standalone_cause": s["collision_cause"],
                          "driver_cps": dc, "standalone_cps": sc, "d_cps": dc - sc})
    rep["flip_list_driver1_vs_standalone1_tilt60"] = flips
    rep["n_flips"] = len(flips)
    rep["flip_cps_sum"] = sum(f["d_cps"] for f in flips)

    (R1 / "cluster_report.json").write_text(json.dumps(rep, indent=2) + "\n")

    print("=== DRIVER lineage (5 R1 runs) ===")
    for r in driver:
        print(f"  {r['run']:12s} tilt60={r['tilt60_cps']:.12f}  bandopen={r['bandopen_cps']:.12f}")
    print(f"  confirm {confirm_driver['run']}: tilt60={confirm_driver['tilt60_cps']:.12f} bandopen={confirm_driver['bandopen_cps']:.12f}")
    print("=== STANDALONE lineage (3 R1 runs) ===")
    for r in standalone:
        print(f"  {r['run']:12s} tilt60={r['tilt60_cps']:.12f}  bandopen={r['bandopen_cps']:.12f}")
    print(f"  confirm {confirm_standalone['run']}: tilt60={confirm_standalone['tilt60_cps']:.12f} bandopen={confirm_standalone['bandopen_cps']:.12f}")
    print(f"\ndriver tilt60   median={rep['driver_tilt60']['median']:.12f} spread={rep['driver_tilt60']['spread']:.2e}")
    print(f"driver bandopen median={rep['driver_bandopen']['median']:.12f} spread={rep['driver_bandopen']['spread']:.2e}")
    print(f"standalone tilt60   median={rep['standalone_tilt60']['median']:.12f} spread={rep['standalone_tilt60']['spread']:.2e}")
    print(f"standalone bandopen median={rep['standalone_bandopen']['median']:.12f} spread={rep['standalone_bandopen']['spread']:.2e}")
    print(f"gap tilt60={rep['gap_tilt60']:.6f}  gap bandopen={rep['gap_bandopen']:.6f}")
    print(f"max within-cluster spread={max_within:.2e}  min between-cluster gap={min_gap:.6f}  two_clusters_clean={rep['two_clusters_clean']}")
    print(f"\nflips (driver_1 vs standalone_1, tilt60): n={len(flips)}  sum d_cps={rep['flip_cps_sum']:.6f}")
    for f in flips:
        print(f"  idx {f['idx']:4d}: {f['standalone_outcome']}->{f['driver_outcome']}  "
              f"cause {f['standalone_cause'] or '-'}->{f['driver_cause'] or '-'}  "
              f"cps {f['standalone_cps']:.4f}->{f['driver_cps']:.4f} (d={f['d_cps']:+.4f})")


if __name__ == "__main__":
    main()
