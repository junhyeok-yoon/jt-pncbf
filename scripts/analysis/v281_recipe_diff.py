"""v2.8.1 S1 — recipe-fidelity config-diff (the launch gate).

Replicates EXACTLY what scripts/v276_stage2_oc_run.py (value) and scripts/v276_stage2_jt_run.py (JT) patch on
top of the registered exp_config, then diffs each launch config against its recorded reference
(value <-> v2.7.6 043415, JT <-> v2.8.0 015517) over the recipe subtrees. Every differing key is classified:
  AUTHORIZED  the three S1 keys + log_jac_classes(accepted, logging-only)
  STANDING    permanent Researcher decisions carried across the lineage (filter.projection dual_solve)
  PROVENANCE  version/timestamp/path/git — expected, not a recipe change
  REVIEW      anything else -> printed loudly; a non-empty REVIEW set is an ABORT.
The band flip and the empty_fallback pin are applied by the patch and should MATCH the reference (no diff)."""
import sys
import yaml

import src.frameworks.oc_pncbf.train as OC
import src.frameworks.jt_pncbf.train as JT

REF_VALUE = "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__oc__20260725-043415__seed42/config.yaml"
REF_JT = "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/config.yaml"
SUBTREES = ("env", "obstacle", "scene_train", "obs", "network", "filter", "lqr", "loss", "training", "collection")
AUTHORIZED = {"env.goal_angrate_radius", "loss.policy.w_settle_ang", "loss.policy.log_jac_classes",
              "obs.encoder", "obs.quadrotor_3d.encoder", "obs.quadrotor_3d.beta"}   # beta=30 adopted (S1 key 3)
STANDING = {"filter.projection"}                      # dual_solve — permanent Researcher decision (dispatch item 1)
LAUNCH_ARG = {"training.jt.value_init_ckpt"}           # supplied via --value-init at launch; the v2.8.1 retrain
#                                                        value-inits from the FRESH v2.8.1 value (not 043415), by
#                                                        design — a path diff here is expected, not a recipe change.
PROV = ("version", "timestamp", "run_dir", "git", "commit", "sha", "hostname", "output", "created")


def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, p + k + ".")) if isinstance(v, dict) else o.__setitem__(p + k, v)
    return o


def value_launch():
    c = OC.load_effective_config()
    c["run"]["system"] = "quadrotor_3d"
    c["training"]["oc_pncbf"]["epochs"] = 100
    c["collection"]["inject_frac"] = 0.0
    c["collection"]["collector"] = "continuing"
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 0.0
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    return c


def jt_launch():
    c = JT.load_effective_config()
    c["run"]["system"] = "quadrotor_3d"
    c["collection"]["collector"] = "continuing"
    c["collection"]["inject_frac"] = 0.0
    c["loss"]["policy"]["sat_excess_threshold"] = [4.905, 4.905, 4.905, 4.905]
    c["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}
    c["env"]["band_collision_limit"] = 4.0
    c["filter"]["empty_fallback"] = {"mode": "none", "k": 10}
    c["training"]["jt"]["n_steps"] = 50000        # models --steps 50000 (run_training n_steps_override), = ref
    c.setdefault("obs", {}).setdefault("quadrotor_3d", {})["beta"] = 30.0   # adopted beta=30 (--beta 30)
    return c


def cls(k):
    if k in AUTHORIZED:
        return "AUTHORIZED"
    if k in STANDING:
        return "STANDING"
    if k in LAUNCH_ARG:
        return "LAUNCH_ARG"
    if any(s in k for s in PROV):
        return "PROVENANCE"
    return "REVIEW"


def run(label, launch, refpath):
    ref = yaml.safe_load(open(refpath))
    fl = flat({k: launch.get(k, {}) for k in SUBTREES})
    fr = flat({k: ref.get(k, {}) for k in SUBTREES})
    print(f"\n=== {label}: launch vs {refpath.split('/')[2]} (recipe subtrees {SUBTREES}) ===")
    review = []
    for k in sorted(set(fl) | set(fr)):
        a, b = fr.get(k, "<absent>"), fl.get(k, "<absent>")
        if a != b:
            t = cls(k)
            print(f"  [{t}] {k}: ref={a!r} -> launch={b!r}")
            if t == "REVIEW":
                review.append(k)
    # explicit confirmation the patched keys MATCH the reference (no diff = recipe reproduced)
    for probe in ("env.band_hazard.enabled", "env.band_hazard.limit", "env.band_collision_limit",
                  "filter.empty_fallback.mode", "filter.empty_fallback.k"):
        print(f"  [MATCH?] {probe}: ref={fr.get(probe,'<absent>')!r} launch={fl.get(probe,'<absent>')!r} "
              f"-> {'OK' if fr.get(probe)==fl.get(probe) else 'DIFF!'}")
    print(f"  RESULT: {'CLEAN (only authorized/standing/provenance)' if not review else f'REVIEW/ABORT keys: {review}'}")
    return review


if __name__ == "__main__":
    r1 = run("VALUE (OC)", value_launch(), REF_VALUE)
    r2 = run("JT", jt_launch(), REF_JT)
    print(f"\n===== recipe-fidelity gate: {'PASS' if not (r1 or r2) else 'ABORT'} =====")
    sys.exit(1 if (r1 or r2) else 0)
