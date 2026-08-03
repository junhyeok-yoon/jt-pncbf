"""v2.8.1 S1 G3 — recorded-row reproduction. Re-score the secured v2.8.0 checkpoint cf948104 on the canonical
pool through the encoder code path. Under decision 3 the system resolves the encoder from the checkpoint's OWN
config (no obs key -> hard_topk), which by G1 is bit-wise (beta->inf, sigma==1). goal_angrate PINNED at 0.48
(the checkpoint's value), shipped fallback {kstep,phases1,k3}, projection dual_solve. cps must land within
+/-0.005 of the recorded 0.7919 (fresh process -> lineage equivalence band, not bit-parity)."""
import copy, json
from pathlib import Path
import torch, yaml
from src.eval.run_full import _load_framework
from src.eval.evaluate import evaluate

REPO = Path("/home/junhyeok/MIT/jt-pncbf")
CK = REPO / "data/runs/v2.8.0/set__20260731-015517__seed42/v2.8.0__jt__20260731-015517__seed42/checkpoints/best.pt"
POOL = REPO / "data/secured_data/pools/eval_full_quadrotor-3d-d2r_n2000_seed23456.pkl"
OUT = REPO / "data/runs/v2.8.1/s1_gates"; OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECORDED = 0.7919315702452286   # m4_dual.json (canonical, band-terminating)

over = {"env": {"dt": 0.05, "stuck_window_steps": 60, "stuck_radius": 0.10, "band_collision_limit": 4.0,
                "goal_angrate_radius": 0.48},                       # PINNED at the checkpoint's terminal
        "eval": {"max_steps": 200, "dt_ctrl": 0.05},
        "filter": {"empty_fallback": {"mode": "kstep", "phases": 1, "k": 3}, "projection": "dual_solve"}}
fw, cfg, ck = _load_framework(str(CK), config_overrides=over)
for n in ("value_net", "policy_net"):
    m = getattr(fw, n, None)
    if m is not None:
        m.to(DEV)
encoder_ran = getattr(fw.system, "encoder", "unknown")
obs_key_in_cfg = ("obs" in cfg and "encoder" in (cfg.get("obs") or {})) or \
                 ("obs" in cfg and "quadrotor_3d" in (cfg.get("obs") or {}))
res = evaluate(fw, POOL, cfg, mode="final", step=int(ck["step"]), ckpt_name="g3", max_scenes=None,
               include_lqr_baseline=False)
cps = float(res.eval_row["cps"])
(OUT / "g3_config.yaml").write_text(yaml.safe_dump({"obs": cfg.get("obs", "ABSENT"),
                                                    "env.goal_angrate_radius": cfg["env"]["goal_angrate_radius"],
                                                    "filter.projection": cfg["filter"]["projection"],
                                                    "system.encoder": encoder_ran}, sort_keys=False))
rec = {"gate": "G3_recorded_row", "checkpoint_sha8": "cf948104", "encoder_ran": encoder_ran,
       "obs_key_present_in_cfg": bool(obs_key_in_cfg), "goal_angrate_radius": cfg["env"]["goal_angrate_radius"],
       "cps": cps, "recorded": RECORDED, "delta": cps - RECORDED, "band": 0.005,
       "within_band": abs(cps - RECORDED) <= 0.005}
(OUT / "g3_recorded_row.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"G3: encoder_ran={encoder_ran} goal_angrate={cfg['env']['goal_angrate_radius']} "
      f"cps={cps:.6f} vs {RECORDED:.6f} delta={cps-RECORDED:+.6f} within_band={rec['within_band']}", flush=True)
