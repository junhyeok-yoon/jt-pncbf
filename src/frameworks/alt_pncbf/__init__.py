"""v2.9.3 ALT-PNCBF — policy iteration: a certificate and a policy trained in long ALTERNATING BLOCKS.

WHAT THIS FRAMEWORK IS. The fourth training framework in this repository, and the in-repository control
for the claim the paper makes about what alternation admits. A run is a sequence of blocks:

  * a POLICY block improves pi against a certificate V_hat held fixed for the whole block, receiving no
    gradient and taking no optimizer step (its Polyak target is frozen too);
  * a VALUE block re-fits V_hat to rollouts collected under pi held fixed for the whole block.

Both blocks keep collecting at the deployed cadence under the policy in force, so each block's data is
drawn under the policy it is held against. A block that reused a buffer filled under an earlier policy
would not be policy iteration; see `schedule.validate_against_loop`, which refuses a block shorter than
one collection period.

WHAT IS NEW HERE, AND WHAT IS NOT. New: the macro-step loop's block dispatch, the two structural freezes
and their four halt checks, and the block schedule (`schedule.py`, pure). Everything else is IMPORTED and
never reimplemented — environment, dynamics, observation, action, outcome predicates, evaluation harness,
metric, pools, plotting and monitoring from `src/common`, `src/envs`, `src/eval`; the certificate network,
the policy network, the HardNet projection, the value/policy losses, the two-buffer collector, the inner
update functions and the whole artifact contract (metrics row, checkpoints, status, report, in-loop eval)
from `src/frameworks/jt_pncbf`. Nothing under `src/frameworks/jt_pncbf/` is modified by this package.

REPORTED DEVIATION FROM 05_code §2. That protocol section says a framework "may import from common, envs,
eval — but not from each other". This package imports `src.frameworks.jt_pncbf` (losses, collection,
train) and `src.frameworks.oc_pncbf.value_target`. The import is deliberate and is the point: an
alternation control whose losses, collector, update math or artifact writer were a second copy would not
be a control at all — a difference in outcome could then be a difference in the copy. The precedent is
already in the tree: `jt_pncbf/train.py:55-59` imports `oc_pncbf.value_target` and `:445` imports
`cpi.value`. Recorded as a code/protocol disagreement, not silently adopted; see
docs/versions/v2.9.3/alt_framework_build.md.

NOT CARRIED OVER (each exists only to serve joint training or is incoherent with a held certificate):
maneuver / cpi / exact_m0 safety channels (they replace the LEARNED certificate the alternation is about),
the horizon critic, the learned-recovery and raw-lagged value-target conditionings (the lagged copy is
Polyak-stepped after every policy step, which has no meaning across a block boundary), `--resume-ckpt`
(a resume rebuilds both buffers empty and resets the collector's persistent row state, which would break
the block clock), and `--stage value_refine`. Each is REFUSED loudly rather than silently ignored.
"""
