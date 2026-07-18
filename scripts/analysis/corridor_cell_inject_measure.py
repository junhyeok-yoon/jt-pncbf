"""v2.7.1 M2 — realized injected fraction + buffer cell-visitation under inject_frac=0.10 (offline, nominal LQR;
value-stage regime). Reuses the pre-gate's in_cell membership. Compares to the M0(b) 0.04% natural baseline."""
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

from scripts.analysis.corridor_cell_pregate import in_cell
from src.common.quadrotor_barrier import value_target_barrier
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_scenes, sample_train_scene
from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
from src.frameworks.jt_pncbf.train import make_system
from src.frameworks.oc_pncbf.collection import OCReplayBuffer

SP = Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")


def m(b, o):
    d = dict(b)
    for k, v in o.items():
        d[k] = m(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    return d


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = m(yaml.safe_load(open("src/configs/base_config.yaml")), yaml.safe_load(open("src/configs/exp_config.yaml")))
    cfg["collection"]["collector"] = "continuing"
    system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")

    # (1) realized injected fraction: fraction of sampled-scene ICs that land in the cell at inject_frac=0.10
    rng = np.random.default_rng(31)
    scs = sample_scenes(sampler, rng, 4000, inject_frac=0.10, system_name="quadrotor_planar", config=cfg)
    bs = batch_scenes(scs, device=dev, dtype=torch.float32)
    x0 = system.wrap_state(initial_states_from_batch(bs).float())
    ic_in = in_cell(x0, bs, system)
    realized = float(ic_in.float().mean().item())
    print(f"[realized-injection] IC-in-cell fraction @inject_frac=0.10 (n=4000 scenes) = {realized:.4f} ({realized*100:.2f}%)", flush=True)

    # (2) buffer cell-visitation: all collected states in-cell over a nominal-LQR continuing collection, inject 0.10
    def lqr_step(x, b):
        g = torch.as_tensor(b.goal, dtype=x.dtype, device=x.device)
        if g.ndim == 1:
            g = g.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, g)
    hbatch = lambda sg, bsc: value_target_barrier(system, sg, bsc, cfg)
    rng2 = np.random.default_rng(32)
    buf = OCReplayBuffer(capacity=80_000_000)
    stt = ContinuingState.create(system, sampler, rng2, 256, cfg, dev, torch.float32,
                                 inject_frac=0.10, system_name="quadrotor_planar")
    for _ in range(12):
        advance_round(stt, round_length=int(cfg["training"]["oc_pncbf"]["horizon"]), step_fn=lqr_step,
                      h_batch_fn=hbatch, scene_sampler=sampler, rng=rng2, config=cfg, buffer=buf,
                      dt=float(cfg["env"]["dt"]), inject_frac=0.10, system_name="quadrotor_planar")
    n_in = 0; n_tot = 0
    for tr in buf._trajectories:
        S = tr.states.to(dev)
        bsc = batch_scenes([tr.scene] * S.shape[0], device=dev, dtype=torch.float32)
        mm = in_cell(S, bsc, system)
        n_in += int(mm.sum().item()); n_tot += S.shape[0]
    visit = n_in / max(n_tot, 1)
    print(f"[buffer-visitation] all-states-in-cell @inject_frac=0.10 = {n_in}/{n_tot} = {visit:.4f} ({visit*100:.2f}%)", flush=True)
    print(f"[baseline M0(b) nominal] 0.0004 (0.04%)  ->  x{visit/0.0004:.0f} increase", flush=True)
    import json
    json.dump(dict(realized_injection=realized, buffer_visitation=visit, n_in=n_in, n_tot=n_tot,
                   baseline_nominal=0.0004), open(SP / "corridor_cell_inject_measure.json", "w"), indent=2)


if __name__ == "__main__":
    main()
