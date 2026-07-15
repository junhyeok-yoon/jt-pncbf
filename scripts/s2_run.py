"""v2.5.2 Stage 2 launcher — clone the FROZEN v2.5.1 seed42 config.yaml and apply ONLY the changes.md §4
delta (system->unicycle, version->v2.5.2, seed, e_label->U_exactm0_s<N>, pi_init_ckpt->null both places,
safety_channel.checkpoint->null, n_steps 30000->50000). Non-§4 hyperparameters are pinned to v2.5.1 by
cloning the frozen config, NOT the live base+exp merge. Prints the full clone->final diff at runtime and
ABORTS if any changed key is outside §4.

Args: --seed S --stage {smoke,full} --steps N --tag TAG
Writes run_dir to <scratch>/s2_<tag>_rundir.txt; prints the applied config diff and the result.

Provenance: committed at v2.5.2 close from the scratchpad launcher that produced the wave-1 unicycle runs
data/v2.5.2__20260714-013024__seed{42,99}. Launched via scripts/s2_orchestrate.sh (2-concurrent {42,99}).
The `SP` path below is the original session scratchpad (retained verbatim as historical provenance)."""
import argparse, copy, json, sys, time
from pathlib import Path

import yaml

import src.frameworks.jt_pncbf.train as T
from src.frameworks.jt_pncbf.train import run_training

SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CLONE_SRC = REPO / "data/previous_runs/v2.5.1__20260713-040300__seed42/config.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--stage", default="full", choices=["smoke", "full"])
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

_CLONE = yaml.safe_load(CLONE_SRC.read_text())

# §4 keys allowed to differ from the clone (run_training also re-stamps run.version / run.framework).
ALLOWED = ("run.version", "run.system", "run.seed", "run.e_label", "run.pi_init_ckpt",
           "run.pi_init_ckpt_sha256", "run.framework", "training.jt.pi_init_ckpt",
           "safety_channel.checkpoint", "run.n_steps_override", "training.jt.n_steps")


def _patched():
    c = copy.deepcopy(_CLONE)
    # §4 non-kwarg deltas (kwargs handle version/seed/n_steps_override/channel)
    c["run"]["system"] = "unicycle"
    c["run"]["e_label"] = f"U_exactm0_s{a.seed}"
    c["run"]["pi_init_ckpt"] = None
    c["run"].pop("pi_init_ckpt_sha256", None)
    c["training"]["jt"]["pi_init_ckpt"] = None
    c["safety_channel"]["checkpoint"] = None
    c["training"]["jt"]["n_steps"] = int(a.steps)
    return c


def _flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(_flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o


# Build the final config exactly as run_training will (patch + its kwarg mutations) and diff vs the clone.
def _final_preview():
    c = _patched()
    c["run"]["version"] = "v2.5.2"        # run_training: config["run"]["version"] = __version__
    c["run"]["framework"] = "jt_pncbf"
    c["safety_channel"]["type"] = "exact_m0"
    c["run"]["seed"] = int(a.seed)
    c["run"]["n_steps_override"] = int(a.steps)
    return c


fb = _flat(_CLONE); ff = _flat(_final_preview())
bad = []
print(f"=== v2.5.2 Stage 2 config diff (clone {CLONE_SRC.name} -> unicycle seed {a.seed}) ===", flush=True)
for k in sorted(set(fb) | set(ff)):
    if fb.get(k) != ff.get(k):
        ok = any(k == x for x in ALLOWED)
        print(f"  [{'§4' if ok else 'ERROR'}] {k}: {fb.get(k)!r} -> {ff.get(k)!r}", flush=True)
        if not ok:
            bad.append(k)
if bad:
    print("STOP — out-of-§4 keys:", bad, flush=True)
    sys.exit(2)
print("ALL DIFFS IN §4: True", flush=True)

T.load_effective_config = _patched
t0 = time.time()
try:
    r = run_training(stage=a.stage, seed=a.seed, safety_channel="exact_m0",
                     n_steps_override=a.steps, output_root=Path("data"))
finally:
    T.load_effective_config = T.load_effective_config  # (kept simple; process exits after)

rd = r.run_dir if hasattr(r, "run_dir") else r
(SP / f"s2_{a.tag}_rundir.txt").write_text(str(rd) + "\n")
print(f"S2 {a.stage} seed={a.seed} DONE run_dir={rd} halted={getattr(r,'halted',None)} "
      f"halt_reason={getattr(r,'halt_reason',None)} best_cps={getattr(r,'best_cps',None)} "
      f"wall_s={round(time.time()-t0,0)}", flush=True)
