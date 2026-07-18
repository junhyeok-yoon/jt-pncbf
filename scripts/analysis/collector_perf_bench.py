"""v2.7.0 iter-5 Phase D — collector per-phase micro-benchmark (no full training).
Times continuing vs legacy collection for OC-like (B=4096, nominal LQR) and JT-like (B=100, policy+HardNet),
per phase: sim / trigger / refill / segment / append(+rebuild) / hlabel. Prints the D1 table + D4 ratio.
GPU util (D2) is sampled by the companion shell loop; host<->device syncs (D3) are documented in the report.
"""
import time
import numpy as np
import torch
import yaml

from src.common.filter_hardnet import _base_alpha, _base_projection, _box_aware_projection, _cbf_terms, _hardnet_params
from src.common.quadrotor_barrier import value_target_barrier
from src.common.value_net import ValueNetEnsemble, make_h_fn
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.envs.scene_init import sample_train_scene
from src.common.rk4 import rk4_step
from src.frameworks.jt_pncbf.train import make_system, _build_control_net
from src.frameworks.jt_pncbf.continuing_collector import ContinuingState, advance_round
from src.frameworks.oc_pncbf.collection import OCReplayBuffer
from src.eval.rollout import rollout_lqr


def _cfg():
    base = yaml.safe_load(open("src/configs/base_config.yaml")); exp = yaml.safe_load(open("src/configs/exp_config.yaml"))
    def m(b, o):
        d = dict(b)
        for k, v in o.items():
            d[k] = m(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
        return d
    c = m(base, exp); c["collection"]["collector"] = "continuing"
    return c


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg(); system = make_system(cfg); system.u_bounds = system.u_bounds.to(dev, torch.float32)
    dt = float(cfg["env"]["dt"]); horizon = int(cfg["training"]["oc_pncbf"]["horizon"])
    vnet = ValueNetEnsemble(system.obs_dim, cfg).to(dev, torch.float32).eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    policy = _build_control_net(system, cfg).to(dev).eval()
    h_fn = make_h_fn(vnet, system); params = _hardnet_params(cfg); bounds = system.u_bounds
    sampler = lambda rng: sample_train_scene(rng, cfg, "quadrotor_planar")

    def h_batch(states_g, bscene):
        return value_target_barrier(system, states_g, bscene, cfg)

    def lqr_step(x, bs):
        goal = torch.as_tensor(bs.goal, dtype=x.dtype, device=x.device)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(x.shape[0], -1)
        return system.lqr_action(x, goal)

    def jt_step(x, bs):
        un = policy(system.observation(x, bs))
        h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
        alpha = _base_alpha(h, params); row = -lf - alpha * h
        proj = _base_projection(un, lg, row, bounds, params)
        u, _ = _box_aware_projection(un, proj, lg, row, bounds)
        return u.detach()

    def bench(label, B, step_fn, n_rounds=3, warmup=1):
        # ---- continuing (per-phase via st.timings, sync=True) ----
        rng = np.random.default_rng(0)
        buf = OCReplayBuffer(capacity=2_000_000)
        stt = ContinuingState.create(system, sampler, rng, B, cfg, dev, torch.float32, system_name="quadrotor_planar")
        agg = {}
        cont_wall = 0.0
        for r in range(n_rounds + warmup):
            t0 = time.perf_counter()
            s = advance_round(stt, round_length=horizon, step_fn=step_fn, h_batch_fn=h_batch,
                              scene_sampler=sampler, rng=rng, config=cfg, buffer=buf, dt=dt,
                              system_name="quadrotor_planar", sync=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            w = time.perf_counter() - t0
            if r >= warmup:
                cont_wall += w
                for k, v in s.timings.items():
                    agg[k] = agg.get(k, 0.0) + v
        cont_per_round = cont_wall / n_rounds
        # ---- legacy (fresh scenes, rollout, barrier, append_batch) ----
        rng2 = np.random.default_rng(0)
        buf2 = OCReplayBuffer(capacity=2_000_000)
        leg = dict(sim=0.0, hlabel=0.0, append=0.0)
        leg_wall = 0.0
        for r in range(n_rounds + warmup):
            scenes = [sampler(rng2) for _ in range(B)]
            bs = batch_scenes(scenes, device=dev, dtype=torch.float32)
            x0 = initial_states_from_batch(bs).float()
            t0 = time.perf_counter()
            if label.startswith("OC"):
                with torch.no_grad():
                    states = rollout_lqr(system, bs, x0, horizon, dt)
            else:
                states = _legacy_jt_roll(system, policy, h_fn, params, bounds, bs, x0, horizon, dt)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            with torch.no_grad():
                h = value_target_barrier(system, states, bs, cfg)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()
            buf2.append_batch(scenes, bs, states.detach(), h.detach())
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()
            if r >= warmup:
                leg["sim"] += t1 - t0; leg["hlabel"] += t2 - t1; leg["append"] += t3 - t2
                leg_wall += t3 - t0
        leg_per_round = leg_wall / n_rounds

        print(f"\n===== {label} (B={B}) — per-round seconds ({n_rounds} rounds, horizon={horizon}) =====")
        print(f"{'phase':<10} {'continuing':>12} {'legacy':>12}")
        for k in ["sim", "trigger", "refill", "segment", "hlabel", "append", "other"]:
            print(f"{k:<10} {agg.get(k,0.0)/n_rounds:>12.4f} {leg.get(k,0.0)/n_rounds:>12.4f}")
        print(f"{'TOTAL':<10} {cont_per_round:>12.4f} {leg_per_round:>12.4f}")
        print(f"steps/round={B*horizon}  cont steps/hr={B*horizon/cont_per_round*3600:,.0f}  "
              f"legacy steps/hr={B*horizon/leg_per_round*3600:,.0f}  ratio(cont/leg)={leg_per_round/cont_per_round:.3f}x_slower="
              f"{cont_per_round/leg_per_round:.2f}")
        return cont_per_round, leg_per_round, agg

    bench("OC-like nominal-LQR", 4096, lqr_step)
    bench("JT-like policy+HardNet", 100, jt_step)


def _legacy_jt_roll(system, policy, h_fn, params, bounds, bs, x0, max_steps, dt):
    x = system.wrap_state(x0); states = [x]
    with torch.no_grad():
        for _ in range(max_steps):
            un = policy(system.observation(x, bs))
            h, lf, lg = _cbf_terms(system, h_fn, x, bs, un, create_graph=False)
            alpha = _base_alpha(h, params); row = -lf - alpha * h
            proj = _base_projection(un, lg, row, bounds, params)
            u, _ = _box_aware_projection(un, proj, lg, row, bounds)
            x = rk4_step(system, x, u.detach(), dt); states.append(x)
    return torch.stack(states, 0)


if __name__ == "__main__":
    main()
