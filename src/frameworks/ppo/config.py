"""PPO config assembly. Reuses the shared base+exp effective config UNCHANGED (env/network/obstacle/scene/
eval/lqr) so the environment is identical to the JT/OC group by construction, then deep-merges the additive
`ppo:` block. NO shared config file is edited; the environment overrides below are applied in memory only and
exactly mirror the CTRL (Stage-2 JT) launch — see scripts/ppo_run.py for the Gate-2 config diff."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.frameworks.jt_pncbf.train import _deep_merge, load_effective_config


PPO_CONFIG_PATH = Path(__file__).resolve().parent / "ppo_config.yaml"


def load_ppo_config(system: str, *, quadrotor_env_stage2: bool = False,
                    overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the effective PPO config.

    system: run.system.
    quadrotor_env_stage2: apply the Stage-2 ENVIRONMENTAL settings CTRL trained under — band_collision_limit=4.0
        (the collision predicate PPO's reward and the scoring both read) and band_hazard {enabled, limit 4.0}
        (recorded for parity; inert for PPO, which has no certificate to read it). goal_angrate_radius=0.30 and
        obs_band_z already flow from base/exp and are unchanged.
    overrides: deep-merged last (launcher knobs, e.g. ppo.num_envs / ppo.n_iterations / the measured k_shaping).
    """
    config = load_effective_config()
    ppo_block = yaml.safe_load(PPO_CONFIG_PATH.read_text(encoding="utf-8"))
    config = _deep_merge(config, ppo_block)
    config["run"]["framework"] = "ppo"
    config["run"]["system"] = str(system)
    if quadrotor_env_stage2:
        config["env"]["band_collision_limit"] = 4.0                       # ENVIRONMENTAL (collision predicate)
        config["env"]["band_hazard"] = {"enabled": True, "limit": 4.0}    # recorded parity; inert for PPO
    if overrides:
        config = _deep_merge(config, dict(overrides))
    return config
