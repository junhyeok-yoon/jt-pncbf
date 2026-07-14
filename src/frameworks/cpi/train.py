"""CPI iteration-0 training: pinball regression of V_raw with the dim-19 observation.

Per seed: AdamW (lr, weight_decay from cpi.train), global-L2 grad clip, constant LR, early stop on val
pinball, keep best-val as best.pt (+ final.pt). Metrics logged to metrics.csv and TensorBoard. Run dirs use
the house pattern (config.yaml / git_commit.txt / status.json). The dataset is fixed across seeds; seeds
vary init + minibatch shuffling only.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.frameworks.cpi.calib import pinball_loss
from src.frameworks.cpi.value import CPIValue

REPO = Path(__file__).resolve().parents[3]
Tensor = torch.Tensor


def git_commit_text() -> str:
    """Run-record stamping (the only permitted git read; read-only rev-parse)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO, check=False).returncode != 0
    except Exception:
        return "unknown"
    return f"{commit} DIRTY" if dirty else commit


def init_run_dir(root: Path, config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = root / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit_text() + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"run_id": run_id, "phase": "running"}) + "\n", encoding="utf-8")
    return run_dir


def _iter_minibatches(n, batch, generator):
    perm = torch.randperm(n, generator=generator)
    for s0 in range(0, n, batch):
        yield perm[s0:s0 + batch]


def train_seed(obs_tr, y_tr, obs_val, y_val, config, seed: int, run_dir: Path, device) -> dict:
    """Train one seed. Returns summary dict; writes best.pt/final.pt/metrics.csv/status.json + TB scalars."""
    tc = config["cpi"]["train"]; tau = float(tc["tau_quantile"]); bs = int(tc["batch_size"])
    max_epochs = int(tc["max_epochs"]); patience = int(tc["early_stop_patience"])
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    model = CPIValue(obs_dim=obs_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tensorboard"))
    except Exception:
        tb = None
    rows = []; best_val = float("inf"); best_state = None; epoch1_val = None; bad = 0; halt = None
    n = obs_tr.shape[0]
    for epoch in range(max_epochs):
        model.train(); tr_loss = 0.0; nb = 0
        for idx in _iter_minibatches(n, bs, gen):
            idx = idx.to(device)
            yh = model(obs_tr[idx]); loss = pinball_loss(y_tr[idx], yh, tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            opt.step(); tr_loss += float(loss.detach()); nb += 1
        tr_loss /= max(nb, 1)
        model.eval()
        with torch.no_grad():
            vp = float(pinball_loss(y_val, model(obs_val), tau))
        rows.append({"epoch": epoch, "train_pinball": tr_loss, "val_pinball": vp})
        if tb is not None:
            tb.add_scalar("train/pinball", tr_loss, epoch); tb.add_scalar("val/pinball", vp, epoch)
        if epoch == 0:
            epoch1_val = vp
        if vp < best_val - 1e-9:
            best_val = vp; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if not np.isfinite(tr_loss) or not np.isfinite(vp):
            halt = f"NaN/Inf at epoch {epoch}"; break
        if epoch == 9 and epoch1_val is not None and best_val >= epoch1_val - 1e-9:
            halt = f"val pinball not improved on epoch-1 ({epoch1_val:.6f}) by epoch 10 (best {best_val:.6f})"; break
        if bad >= patience:
            break
    _write_csv(run_dir / "metrics.csv", ["epoch", "train_pinball", "val_pinball"], rows)
    if best_state is not None:
        torch.save({"model_state": best_state, "seed": seed, "config": dict(config), "best_val": best_val},
                   run_dir / "checkpoints/best.pt")
    torch.save({"model_state": model.state_dict(), "seed": seed, "config": dict(config)},
               run_dir / "checkpoints/final.pt")
    status = {"run_id": run_dir.name, "phase": "halted" if halt else "done",
              "epochs": len(rows), "best_val_pinball": best_val, "halt": halt}
    (run_dir / "status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")
    if tb is not None:
        tb.close()
    return status


def _write_csv(path: Path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)


def load_split_tensors(dataset_dir: Path, split: str, device):
    """Load obs [N,19] and V_raw [N] for a split from the npz shards + manifest."""
    man = json.load(open(dataset_dir / "manifest.json"))
    obs, y, sid = [], [], []
    for sh in man["shards"]:
        if sh["split"] != split:
            continue
        d = np.load(dataset_dir / sh["file"])
        obs.append(d["obs"]); y.append(d["vraw"]); sid.append(d["scene_id"])
    obs = torch.as_tensor(np.concatenate(obs), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.concatenate(y), dtype=torch.float32, device=device)
    sid = torch.as_tensor(np.concatenate(sid), dtype=torch.long, device=device)
    return obs, y, sid
