"""v2.7.6 M8.3 — per-control-step u_cmd latency for each fallback arm, at the deploy condition batch=1 on the
default (CPU) eval device. One control step = policy forward + V-hat gradient + HardNet projection (+ the
empty-branch fallback on empty rows), reusing the v2.7.5 dt_ctrl instrument's definition and its 10 ms p95
(100 Hz H4) budget. Reports median and p95 SEPARATELY for empty rows (fallback fires) and non-empty rows
(base filter), plus the empty-row fraction from a rollout. The empty/non-empty classification is arm-invariant
(it is the QP feasibility, not the fallback), so states are collected once; only the empty-row timing varies by
arm. Eval-only. Artifacts under stage2_eval/."""
from __future__ import annotations

import copy, json, statistics, time
from pathlib import Path

import torch

from src.common.rk4 import rk4_step
from src.eval.run_full import _load_framework as load_fw
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.eval.build_pools import load_pool
from scripts.analysis.v276_m7_fallback_trial import POOLS

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
STEM = "eval_fullrange_quadrotor-3d-d2r_n2000_seed42"
JT42 = REPO / "data/runs/v2.7.6/set__20260725-043415__seed42/v2.7.6__jt__20260725-052127__seed42/checkpoints/step_042000.pt"
OUT = REPO / "data/runs/v2.7.6/stage2_eval"
DEV = torch.device("cpu")                 # deploy condition; default eval device (M8.0 finding: eval path is CPU)
MAX_STATES = 500                          # cap per class for the p95


def _fw(mode, phases, k):
    ck = torch.load(JT42, map_location="cpu", weights_only=False)
    filt = copy.deepcopy(ck["config"]["filter"])
    filt["empty_fallback"] = {"mode": "none", "k": 10} if mode == "none" else {"mode": "kstep", "k": k, "phases": phases}
    fw, cfg, _ = load_fw(JT42, config_overrides={"env": {"dt": 0.05}, "eval": {"max_steps": 200}, "filter": filt})
    return fw, cfg


def collect_states(fw, cfg, n_scenes=150):
    """Rollout batch to gather single (1,13) states split by empty flag, plus the empty-row fraction."""
    system = fw.system; dt = float(cfg["env"]["dt"]); T = int(cfg["eval"]["max_steps"])
    scenes = load_pool(POOLS / f"{STEM}.pkl").scenes[:n_scenes]
    bs = batch_scenes(scenes, device=DEV, dtype=torch.float32)
    x = system.wrap_state(initial_states_from_batch(bs).float())
    emp_states, non_states = [], []; n_empty = 0; n_active = 0
    with torch.no_grad():
        for _ in range(T):
            un = fw.policy(x, bs); u, _ = fw.filter(x, un, bs)
            em = fw._filter.last_empty.bool()
            n_empty += int(em.sum()); n_active += int(em.numel())
            if em.any() and len(emp_states) < MAX_STATES:
                for idx in torch.nonzero(em).reshape(-1).tolist():
                    if len(emp_states) < MAX_STATES:
                        emp_states.append((x[idx:idx+1].clone(), _slice1(bs, idx)))
            if (~em).any() and len(non_states) < MAX_STATES:
                for idx in torch.nonzero(~em).reshape(-1).tolist()[:5]:
                    if len(non_states) < MAX_STATES:
                        non_states.append((x[idx:idx+1].clone(), _slice1(bs, idx)))
            x = rk4_step(system, x, u, dt)
            if len(emp_states) >= MAX_STATES and len(non_states) >= MAX_STATES:
                break
    return emp_states, non_states, n_empty / max(1, n_active)


def _slice1(bs, idx):
    import dataclasses
    upd = {}
    for f in dataclasses.fields(bs):
        v = getattr(bs, f.name)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            upd[f.name] = v[idx:idx+1]
    return dataclasses.replace(bs, **upd)


def time_states(fw, states):
    ts = []
    for x1, bs1 in states[:3]:            # warmup
        with torch.no_grad():
            fw.filter(x1, fw.policy(x1, bs1), bs1)
    for x1, bs1 in states:
        t0 = time.perf_counter()
        with torch.no_grad():
            u = fw.policy(x1, bs1); fw.filter(x1, u, bs1)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def p95(v):
    v = sorted(v); i = min(len(v) - 1, int(round(0.95 * (len(v) - 1))))
    return v[i]


if __name__ == "__main__":
    fw0, cfg0 = _fw("kstep", 2, 5)         # collect on a kstep rollout (empty flag is arm-invariant)
    emp, non, efrac = collect_states(fw0, cfg0)
    print(f"collected {len(emp)} empty, {len(non)} non-empty states; empty_row_fraction {efrac:.4f}", flush=True)
    base = time_states(fw0, non)           # non-empty (base filter, arm-invariant)
    cells = [{"label": "non_empty_base", "median_ms": round(statistics.median(base), 4), "p95_ms": round(p95(base), 4),
              "n": len(base), "H4_p95_under_10ms": bool(p95(base) < 10.0)}]
    print(f"[non_empty_base] median {cells[0]['median_ms']} ms p95 {cells[0]['p95_ms']} ms (n={len(base)})", flush=True)
    plan = [("none", 0, 0)] + [(("kstep"), p, k) for p in (1, 2) for k in (1, 2, 3, 4, 5)]
    for mode, phases, k in plan:
        fw, _ = _fw(mode, phases, k)
        et = time_states(fw, emp)
        lab = "none" if mode == "none" else f"p{phases}_k{k}"
        cell = {"label": lab, "mode": mode, "phases": phases, "k": k, "n_candidate_evals": (0 if mode == "none" else phases * 25 * k),
                "empty_median_ms": round(statistics.median(et), 4), "empty_p95_ms": round(p95(et), 4),
                "n_empty_timed": len(et), "empty_H4_p95_under_10ms": bool(p95(et) < 10.0), "device": str(DEV)}
        cells.append(cell)
        print(f"[{lab}] empty-row median {cell['empty_median_ms']} ms p95 {cell['empty_p95_ms']} ms "
              f"cand {cell['n_candidate_evals']} H4 {'PASS' if cell['empty_H4_p95_under_10ms'] else 'FAIL'}", flush=True)
    (OUT / "m8_latency.json").write_text(json.dumps({"budget_ms_100Hz": 10.0, "device": str(DEV),
        "empty_row_fraction": round(efrac, 4), "cells": cells}, indent=2) + "\n")
    print("M8.3 latency done ->", OUT / "m8_latency.json", flush=True)
