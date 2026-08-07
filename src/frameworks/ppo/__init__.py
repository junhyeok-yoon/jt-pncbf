"""v2.8.2 in-house PPO baseline (policy only; NO certificate, NO filter).

Additive framework: supplies the missing unfiltered-learned reference point on the deployed metric.
Reuses src/common (plant, sampler, rk4, outcome predicates) and src.common.control_net.ControlNet
(the SAME policy trunk the JT lineage uses) so the environment is identical BY CONSTRUCTION. Never bolded,
never secured, not expected to be competitive on collision.
"""
