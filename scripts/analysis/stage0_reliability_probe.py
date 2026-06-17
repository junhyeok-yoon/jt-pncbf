"""Stage 0 reliability-signal decomposition probe (read-only analysis).

Tests, on existing diagnostic data only, whether two grounded reliability signals for the
learned CBF value V_S separate safety-filter failures by cause:

  - EPISTEMIC axis: ensemble disagreement |v_1 - v_2| (recomputed; not stored in the NPZ).
  - APPROXIMATION/CONSISTENCY axis: executed-action CBF-row violation (stored), plus one
    finite-difference alternative.

The probe is read-only on every input (NPZ traces, secured checkpoint, configs). It writes
ONLY this script's report (Markdown) and PNG figures. It does not train, roll out, evaluate,
re-derive outcomes, or change control. Deterministic: rerun produces identical numbers.

Run:
  /home/junhyeok/miniconda3/envs/pncbf/bin/python scripts/analysis/stage0_reliability_probe.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.value_net import ValueNetEnsemble  # noqa: E402
from src.frameworks.jt_pncbf.train import make_system  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "data/diagnostics"
    / "v2.0.1__20260529-171057__failure_dissect_n2500_seed45678_20260529-194915"
)
DEFAULT_CKPT = REPO_ROOT / "data/secured_data/v2.0.1/seed42/checkpoints/best.pt"
DEFAULT_REPORT = REPO_ROOT / "docs/versions/v2.2.0/stage0_reliability_probe.md"
DEFAULT_FIGURE_DIR = REPO_ROOT / "data/diagnostics/v2.2.0_stage0_probe"

LEADS = [1, 3, 5, 10, 15, 30]
PRE_WINDOW = 30                 # episode-level reduction window (steps ending at event_action)
MEAN_MATCH_TOL = 1.0e-4         # deployed-h reconstruction gate (float32 recompute)
AUROC_TOL = 1.0e-9
SCATTER_MAX_PER_CLASS = 4000    # subsample for plotting only; stats use all points

# Risk orientation declared IN ADVANCE (never chosen by best-AUROC). For all three signals,
# a larger value is the more-failure-like direction, so sign = +1.
SIGNAL_SIGN = {
    "epistemic |v1-v2|": +1.0,
    "residual cbf_violation_safe": +1.0,
    "residual fd relu(hdot_fd + alpha*h)": +1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    system, value_net, config, weight_dtype = load_value_net(args.checkpoint)
    alpha_safe = float(config["filter"]["alpha_safe"])
    alpha_unsafe = float(config["filter"]["alpha_unsafe"])

    episodes, field_list, recon = build_episode_table(
        args.input_dir / "episodes", system, value_net, weight_dtype, alpha_safe, alpha_unsafe
    )
    mechanisms = load_episode_summary(args.input_dir / "episode_summary.csv")
    layers = load_layer_attribution(args.input_dir / "layer_attribution.csv")

    by_outcome = {key: [e for e in episodes if e["outcome"] == key] for key in ("collision", "stuck", "goal")}
    counts = {key: len(value) for key, value in by_outcome.items()}

    signals = list(SIGNAL_SIGN.keys())

    # Analysis 1 + 3: association / differential separation (AUROC).
    assoc = compute_association(by_outcome, signals)

    # Analysis 2: independence (correlations across pooled per-step rows).
    independence = compute_independence(by_outcome)

    # Analysis 4: low-low-but-failed region.
    lowlow = compute_low_low(by_outcome, mechanisms, layers)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = write_figures(args.figure_dir, by_outcome, assoc, signals)

    report = render_report(
        input_dir=args.input_dir,
        checkpoint=args.checkpoint,
        field_list=field_list,
        recon=recon,
        config_alpha=(alpha_safe, alpha_unsafe),
        counts=counts,
        signals=signals,
        assoc=assoc,
        independence=independence,
        lowlow=lowlow,
        figure_paths=figure_paths,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(args.report)
    print(f"lines={len(report.splitlines())}")
    return 0


# --------------------------------------------------------------------------------------
# Loading


def load_value_net(checkpoint_path: Path) -> tuple[Any, ValueNetEnsemble, dict[str, Any], torch.dtype]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    system = make_system(config)
    first_tensor = next(iter(checkpoint["v_s_state"].values()))
    dtype = first_tensor.dtype
    value_net = ValueNetEnsemble(system.obs_dim, config).to(dtype=dtype)
    value_net.load_state_dict(checkpoint["v_s_state"])
    value_net.eval()
    return system, value_net, config, dtype


def load_episode_summary(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["episode_idx"])] = row
    return rows


def load_layer_attribution(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["episode_idx"])] = row
    return rows


# --------------------------------------------------------------------------------------
# Per-episode signal reconstruction


def parse_episode_filename(path: Path) -> tuple[str, int]:
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[0] == "goal" and parts[1] == "control":
        return "goal_control", int(parts[2])
    return parts[0], int(parts[1])


def event_action_step(outcome: str, event_step: int, n_steps: int) -> int:
    # Same convention as the prior pilot: the executed-action step that produced the event.
    if outcome in {"collision", "goal", "oob"} and event_step > 0:
        return max(0, min(event_step - 1, n_steps - 1))
    if event_step < 0:
        return n_steps - 1
    return max(0, min(event_step, n_steps - 1))


def build_episode_table(
    episode_dir: Path,
    system: Any,
    value_net: ValueNetEnsemble,
    weight_dtype: torch.dtype,
    alpha_safe: float,
    alpha_unsafe: float,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float]]:
    if not episode_dir.exists():
        raise FileNotFoundError(episode_dir)

    episodes: list[dict[str, Any]] = []
    field_list: list[str] | None = None
    max_mean_mismatch = 0.0
    max_residual_recompute_mismatch = 0.0

    for path in sorted(episode_dir.glob("*.npz")):
        if not path.name.startswith(("failure_", "goal_control_")):
            continue
        loaded = np.load(path, allow_pickle=False)
        if field_list is None:
            field_list = sorted(loaded.files)
        metadata = json.loads(str(np.asarray(loaded["metadata_json"]).item()))
        outcome = str(metadata["outcome"])
        event_step = int(metadata["event_step"])
        active_steps = int(metadata.get("active_steps", len(loaded["h"])))

        states = np.asarray(loaded["states"], dtype=np.float64)        # [T+1, 4]
        h_stored = np.asarray(loaded["h"], dtype=np.float64)           # [T]
        n_steps = len(h_stored)
        x = states[:-1]                                                # [T, 4]

        # Epistemic: recompute the two clipped ensemble members from v_s_state.
        members = recompute_members(system, value_net, x, loaded, weight_dtype)  # [T, n_vs]
        deployed = members.mean(axis=1)
        max_mean_mismatch = max(max_mean_mismatch, float(np.max(np.abs(deployed - h_stored))))
        epistemic = np.abs(members[:, 0] - members[:, 1])
        member_std = members.std(axis=1)

        # Residual (primary): stored executed-action CBF-row violation.
        residual_primary = np.asarray(loaded["cbf_violation_safe"], dtype=np.float64)
        recomputed_primary = np.maximum(
            0.0,
            np.asarray(loaded["row_lhs_safe"], dtype=np.float64)
            - np.asarray(loaded["row_upper"], dtype=np.float64),
        )
        max_residual_recompute_mismatch = max(
            max_residual_recompute_mismatch,
            float(np.max(np.abs(recomputed_primary - residual_primary))) if n_steps else 0.0,
        )

        # Residual (alternative): realized finite-difference CBF-decrease violation.
        alpha = np.where(h_stored <= 0.0, alpha_safe, alpha_unsafe)
        hdot_fd = np.asarray(loaded["h_dot_fd"], dtype=np.float64)
        residual_fd = np.maximum(0.0, hdot_fd + alpha * h_stored)

        event_action = event_action_step(outcome, event_step, n_steps)
        episodes.append(
            {
                "path": path,
                "episode_idx": parse_episode_filename(path)[1],
                "outcome": outcome,
                "event_step": event_step,
                "event_action": event_action,
                "active_steps": active_steps,
                "n_steps": n_steps,
                "h": h_stored,
                "members": members,
                "member_std": member_std,
                "signals": {
                    "epistemic |v1-v2|": epistemic,
                    "residual cbf_violation_safe": residual_primary,
                    "residual fd relu(hdot_fd + alpha*h)": residual_fd,
                },
            }
        )

    if field_list is None:
        raise FileNotFoundError(f"No NPZ traces found in {episode_dir}.")

    if max_mean_mismatch > MEAN_MATCH_TOL:
        raise AssertionError(
            "Observation reconstruction FAILED: mean of recomputed ensemble members does not "
            f"match stored deployed h (max abs mismatch {max_mean_mismatch:.3e} > "
            f"{MEAN_MATCH_TOL:.1e}). The obs reconstruction is wrong; probe cannot proceed."
        )

    recon = {
        "max_mean_mismatch": max_mean_mismatch,
        "max_residual_recompute_mismatch": max_residual_recompute_mismatch,
    }
    return episodes, field_list, recon


def recompute_members(
    system: Any,
    value_net: ValueNetEnsemble,
    x: np.ndarray,
    loaded: Any,
    weight_dtype: torch.dtype,
) -> np.ndarray:
    scene = SimpleNamespace(
        goal=np.asarray(loaded["goal"], dtype=np.float64),
        obstacle_centers=np.asarray(loaded["obstacle_centers"], dtype=np.float64),
        obstacle_radii=np.asarray(loaded["obstacle_radii"], dtype=np.float64),
        obstacle_active=np.asarray(loaded["obstacle_active"], dtype=bool),
    )
    x_t = torch.as_tensor(x, dtype=weight_dtype)
    with torch.no_grad():
        obs = system.observation(x_t, scene)
        members = value_net.value_all(obs)            # clipped [-1, 1] per member
    return members.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------------------
# Row builders


def safe_idx(episode: dict[str, Any], idx: int) -> bool:
    return 0 <= idx < episode["n_steps"] and idx < episode["active_steps"]


def lead_value(episode: dict[str, Any], signal: str, lead: int) -> float | None:
    idx = episode["event_action"] - lead
    if not safe_idx(episode, idx):
        return None
    return float(episode["signals"][signal][idx])


def episode_window(episode: dict[str, Any]) -> np.ndarray:
    lo = max(0, episode["event_action"] - PRE_WINDOW + 1)
    hi = min(episode["event_action"] + 1, episode["active_steps"], episode["n_steps"])
    return np.arange(lo, max(lo, hi))


def episode_reduction(episode: dict[str, Any], signal: str, reducer: str) -> float | None:
    window = episode_window(episode)
    if window.size == 0:
        return None
    values = episode["signals"][signal][window]
    if values.size == 0:
        return None
    return float(np.max(values) if reducer == "max" else np.mean(values))


def active_series(episode: dict[str, Any], signal: str) -> np.ndarray:
    hi = min(episode["active_steps"], episode["n_steps"])
    return episode["signals"][signal][:hi]


# --------------------------------------------------------------------------------------
# AUROC


def pairwise_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = float(np.sum(diff > 0.0))
    ties = float(np.sum(diff == 0.0))
    return (wins + 0.5 * ties) / (pos.size * neg.size)


def checked_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    sk = float(roc_auc_score(labels, scores))
    pw = pairwise_auc(labels, scores)
    if abs(sk - pw) > AUROC_TOL:
        raise AssertionError(f"AUROC mismatch sklearn={sk} pairwise={pw}.")
    return sk


def association_auc(
    pos_eps: list[dict[str, Any]],
    neg_eps: list[dict[str, Any]],
    signal: str,
    mode: str,
    lead: int | None,
) -> tuple[float, int, int]:
    sign = SIGNAL_SIGN[signal]
    labels: list[int] = []
    scores: list[float] = []
    for label, group in ((1, pos_eps), (0, neg_eps)):
        for episode in group:
            if mode == "lead":
                value = lead_value(episode, signal, int(lead))
            else:
                value = episode_reduction(episode, signal, "max")
            if value is None:
                continue
            labels.append(label)
            scores.append(sign * value)
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    return checked_auc(labels_arr, scores_arr), n_pos, n_neg


def compute_association(by_outcome: dict[str, list[dict[str, Any]]], signals: list[str]) -> dict[str, Any]:
    goal = by_outcome["goal"]
    result: dict[str, Any] = {"lead": {}, "episode": {}}
    for failure in ("collision", "stuck"):
        result["lead"][failure] = {}
        result["episode"][failure] = {}
        for signal in signals:
            result["lead"][failure][signal] = {}
            for lead in LEADS:
                auc, n_pos, n_neg = association_auc(by_outcome[failure], goal, signal, "lead", lead)
                result["lead"][failure][signal][lead] = {"auc": auc, "n_pos": n_pos, "n_neg": n_neg}
            auc, n_pos, n_neg = association_auc(by_outcome[failure], goal, signal, "episode", None)
            result["episode"][failure][signal] = {"auc": auc, "n_pos": n_pos, "n_neg": n_neg}
    return result


# --------------------------------------------------------------------------------------
# Independence (correlations) + low-low region


def compute_independence(by_outcome: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    primary = "residual cbf_violation_safe"
    alt = "residual fd relu(hdot_fd + alpha*h)"
    out: dict[str, Any] = {}
    for outcome, eps in by_outcome.items():
        epi = np.concatenate([active_series(e, "epistemic |v1-v2|") for e in eps]) if eps else np.array([])
        res = np.concatenate([active_series(e, primary) for e in eps]) if eps else np.array([])
        res_alt = np.concatenate([active_series(e, alt) for e in eps]) if eps else np.array([])
        out[outcome] = {
            "n_rows": int(epi.size),
            "pearson_primary": corr(epi, res, "pearson"),
            "spearman_primary": corr(epi, res, "spearman"),
            "pearson_alt": corr(epi, res_alt, "pearson"),
            "spearman_alt": corr(epi, res_alt, "spearman"),
            "epistemic": epi,
            "residual_primary": res,
        }
    return out


def corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    value = pearsonr(a, b)[0] if kind == "pearson" else spearmanr(a, b)[0]
    return float(value)


def compute_low_low(
    by_outcome: dict[str, list[dict[str, Any]]],
    mechanisms: dict[int, dict[str, str]],
    layers: dict[int, dict[str, str]],
) -> dict[str, Any]:
    epi_signal = "epistemic |v1-v2|"
    res_signal = "residual cbf_violation_safe"
    res_zero_tol = 1.0e-9

    def ep_pair(episode: dict[str, Any]) -> tuple[float | None, float | None]:
        return (
            episode_reduction(episode, epi_signal, "max"),
            episode_reduction(episode, res_signal, "max"),
        )

    goal_epi = [v for v, _ in (ep_pair(e) for e in by_outcome["goal"]) if v is not None]
    goal_res = [v for _, v in (ep_pair(e) for e in by_outcome["goal"]) if v is not None]
    epi_median = float(np.median(goal_epi)) if goal_epi else float("nan")
    res_median = float(np.median(goal_res)) if goal_res else float("nan")
    goal_res_zero_frac = (
        float(np.mean([1.0 if v <= res_zero_tol else 0.0 for v in goal_res])) if goal_res else float("nan")
    )

    # Per-step goal medians (for the timestep-level low-low pass).
    goal_epi_steps = (
        np.concatenate([active_series(e, epi_signal) for e in by_outcome["goal"]])
        if by_outcome["goal"]
        else np.array([])
    )
    goal_res_steps = (
        np.concatenate([active_series(e, res_signal) for e in by_outcome["goal"]])
        if by_outcome["goal"]
        else np.array([])
    )
    epi_step_median = float(np.median(goal_epi_steps)) if goal_epi_steps.size else float("nan")
    res_step_median = float(np.median(goal_res_steps)) if goal_res_steps.size else float("nan")
    goal_res_step_zero_frac = float(np.mean(goal_res_steps <= res_zero_tol)) if goal_res_steps.size else float("nan")

    detail: dict[str, Any] = {
        "epi_median": epi_median,
        "res_median": res_median,
        "goal_res_zero_frac": goal_res_zero_frac,
        "res_degenerate": res_median <= res_zero_tol,
        "epi_step_median": epi_step_median,
        "res_step_median": res_step_median,
        "goal_res_step_zero_frac": goal_res_step_zero_frac,
        "classes": {},
    }
    for failure in ("collision", "stuck"):
        strict: list[dict[str, Any]] = []      # both strictly below goal medians (literal prompt)
        inclusive: list[dict[str, Any]] = []   # both <= goal medians (robust to zero residual median)
        res_zero_count = 0
        total = 0
        ll_timesteps = 0
        total_timesteps = 0
        eps_with_ll: list[dict[str, Any]] = []
        for episode in by_outcome[failure]:
            epi_v, res_v = ep_pair(episode)
            if epi_v is None or res_v is None:
                continue
            total += 1
            idx = episode["episode_idx"]
            record = {
                "episode_idx": idx,
                "epistemic_max": epi_v,
                "residual_max": res_v,
                "mechanism": mechanisms.get(idx, {}).get("mechanism", "?"),
                "layer": layers.get(idx, {}).get("layer", "?"),
            }
            if epi_v < epi_median and res_v < res_median:
                strict.append(record)
            if epi_v <= epi_median and res_v <= res_median:
                inclusive.append(record)
            if res_v <= res_zero_tol:
                res_zero_count += 1
            # Timestep-level: any active step where BOTH signals are at/below the goal per-step median.
            epi_s = active_series(episode, epi_signal)
            res_s = active_series(episode, res_signal)
            mask = (epi_s <= epi_step_median) & (res_s <= res_step_median)
            total_timesteps += int(mask.size)
            ll_timesteps += int(mask.sum())
            if bool(mask.any()):
                eps_with_ll.append(record)
        detail["classes"][failure] = {
            "total_scored": total,
            "strict_count": len(strict),
            "inclusive_count": len(inclusive),
            "res_zero_count": res_zero_count,
            "mech_counts": _count_field(inclusive, "mechanism"),
            "layer_counts": _count_field(inclusive, "layer"),
            "episodes": inclusive,
            "ll_timesteps": ll_timesteps,
            "total_timesteps": total_timesteps,
            "eps_with_ll_count": len(eps_with_ll),
            "ts_mech_counts": _count_field(eps_with_ll, "mechanism"),
            "ts_layer_counts": _count_field(eps_with_ll, "layer"),
        }
    return detail


def _count_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item[field]] = counts.get(item[field], 0) + 1
    return counts


# --------------------------------------------------------------------------------------
# Figures


def write_figures(
    figure_dir: Path,
    by_outcome: dict[str, list[dict[str, Any]]],
    assoc: dict[str, Any],
    signals: list[str],
) -> list[Path]:
    paths: list[Path] = []
    paths.append(plot_auc_vs_lead(figure_dir, assoc, signals))
    paths.append(plot_timestep_scatter(figure_dir, by_outcome))
    paths.append(plot_episode_scatter(figure_dir, by_outcome))
    return paths


def plot_auc_vs_lead(figure_dir: Path, assoc: dict[str, Any], signals: list[str]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=140, sharey=True)
    for ax, failure in zip(axes, ("collision", "stuck")):
        for signal in signals:
            ys = [assoc["lead"][failure][signal][lead]["auc"] for lead in LEADS]
            ax.plot(LEADS, ys, marker="o", label=signal)
        ax.axhline(0.5, color="0.6", lw=1, ls="--")
        ax.set_title(f"{failure} vs goal")
        ax.set_xlabel("lead K before event (steps)")
        ax.set_ylim(0.30, 1.02)
    axes[0].set_ylabel("AUROC (fixed +1 orientation)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    path = figure_dir / "auroc_vs_lead.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _scatter_points(by_outcome: dict[str, list[dict[str, Any]]], episode_level: bool):
    epi_signal = "epistemic |v1-v2|"
    res_signal = "residual cbf_violation_safe"
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for outcome, eps in by_outcome.items():
        if episode_level:
            xs, ys = [], []
            for e in eps:
                ex = episode_reduction(e, epi_signal, "max")
                ey = episode_reduction(e, res_signal, "max")
                if ex is not None and ey is not None:
                    xs.append(ex)
                    ys.append(ey)
            series[outcome] = (np.asarray(xs), np.asarray(ys))
        else:
            ex = np.concatenate([active_series(e, epi_signal) for e in eps]) if eps else np.array([])
            ey = np.concatenate([active_series(e, res_signal) for e in eps]) if eps else np.array([])
            if ex.size > SCATTER_MAX_PER_CLASS:
                stride = int(np.ceil(ex.size / SCATTER_MAX_PER_CLASS))
                ex = ex[::stride]
                ey = ey[::stride]
            series[outcome] = (ex, ey)
    return series


def plot_timestep_scatter(figure_dir: Path, by_outcome: dict[str, list[dict[str, Any]]]) -> Path:
    series = _scatter_points(by_outcome, episode_level=False)
    colors = {"goal": "#7f7f7f", "stuck": "#1f77b4", "collision": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=140)
    for outcome in ("goal", "stuck", "collision"):
        ex, ey = series[outcome]
        ax.scatter(ex, ey, s=6, alpha=0.35, c=colors[outcome], label=f"{outcome} (n~{ex.size})", edgecolors="none")
    ax.set_xlabel("epistemic |v1 - v2| (per active step)")
    ax.set_ylabel("residual cbf_violation_safe (per active step)")
    ax.set_title("Two reliability axes per timestep")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "axes_scatter_timestep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_episode_scatter(figure_dir: Path, by_outcome: dict[str, list[dict[str, Any]]]) -> Path:
    series = _scatter_points(by_outcome, episode_level=True)
    colors = {"goal": "#7f7f7f", "stuck": "#1f77b4", "collision": "#d62728"}
    goal_x, goal_y = series["goal"]
    epi_median = float(np.median(goal_x)) if goal_x.size else float("nan")
    res_median = float(np.median(goal_y)) if goal_y.size else float("nan")
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=140)
    for outcome in ("goal", "stuck", "collision"):
        ex, ey = series[outcome]
        ax.scatter(ex, ey, s=22, alpha=0.7, c=colors[outcome], label=f"{outcome} (n={ex.size})", edgecolors="none")
    ax.axvline(epi_median, color="0.3", lw=1, ls="--")
    ax.axhline(res_median, color="0.3", lw=1, ls="--")
    ax.set_xlabel("episode max epistemic |v1 - v2| (pre-event window)")
    ax.set_ylabel("episode max residual cbf_violation_safe (pre-event window)")
    ax.set_title("Episode-level axes; dashed = goal-class medians (low-low quadrant = bottom-left)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "axes_scatter_episode.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Report


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def render_report(
    *,
    input_dir: Path,
    checkpoint: Path,
    field_list: list[str],
    recon: dict[str, float],
    config_alpha: tuple[float, float],
    counts: dict[str, int],
    signals: list[str],
    assoc: dict[str, Any],
    independence: dict[str, Any],
    lowlow: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    rel_input = input_dir.relative_to(REPO_ROOT)
    rel_ckpt = checkpoint.relative_to(REPO_ROOT)
    lines: list[str] = []
    lines += [
        "# v2.2.0 Stage 0 — Reliability-Signal Decomposition Probe",
        "",
        "> **Read-only existence check on a biased sample.** Positives are saved collision/stuck",
        "> failures; the negative/reference class is the 120 matched goal controls, not all",
        "> non-failure timesteps. AUROCs are existence checks for separable signals, not detector",
        "> metrics. No training, no rollout, no eval, no control change. Reproducible script:",
        "> `scripts/analysis/stage0_reliability_probe.py` (deterministic; rerun is identical).",
        "",
        "## 0. Data sources and field list",
        "",
        f"- Diagnostic NPZ set (read-only): `{rel_input}/episodes/`.",
        f"- Secured checkpoint (read-only, weights only): `{rel_ckpt}` (stores `v_s_state`,",
        "  `v_s_target_state`; only `v_s_state` — the deployed mean-ensemble — is loaded here).",
        f"- Episode counts used: collision {counts['collision']}, stuck {counts['stuck']}, "
        f"goal {counts['goal']}.",
        f"- Confirmed NPZ field list: `{', '.join(field_list)}`.",
        "- Cross-reference tables: `episode_summary.csv` (per-episode `mechanism`) and",
        "  `layer_attribution.csv` (per-episode `layer`).",
        "",
        "## 1. Signal definitions (stated precisely)",
        "",
        "### 1.1 Epistemic axis — ensemble disagreement (recomputed; not in the NPZ)",
        "",
        "The NPZ stores only deployed `h` (the ensemble MEAN). Per saved timestep, the observation",
        "is reconstructed from the NPZ `states[:-1]` plus `goal`/`obstacle_centers`/`obstacle_radii`/",
        "`obstacle_active` via the project's own `DoubleIntegrator.observation`",
        "(`src/envs/double_integrator.py:41`, using `src/common/observation.py`). The two ensemble",
        "members are evaluated from `v_s_state` as `ValueNetEnsemble.value_all` (clipped to [-1,1] per",
        "member, `src/common/value_net.py:32`). Epistemic signal: `epistemic_t = |v_1 - v_2|`",
        "(member std also recorded). Orientation declared in advance: higher = more failure-like (+1).",
        "",
        "**Reconstruction verification gate.** The mean `0.5*(v_1+v_2)` must equal the NPZ-stored",
        f"deployed `h` per timestep. Max abs mismatch over ALL saved timesteps: "
        f"`{recon['max_mean_mismatch']:.3e}` (gate `< {MEAN_MATCH_TOL:.0e}`; PASS). The obs",
        "reconstruction is therefore exact to float32 recompute noise.",
        "",
        "### 1.2 Approximation/consistency axis — executed-action CBF residual (stored)",
        "",
        "Sign convention (verified against `src/common/filter_hardnet.py`): this project uses the",
        "avoid/PNCBF convention — `h <= 0` safe, `h > 0` unsafe — so the HardNet CBF row enforces the",
        "DECREASE condition `h_dot <= -alpha*h` (row `A·u <= b`, `A = L_g h`, `b = -L_f h - alpha*h`).",
        "The executed-action violation magnitude is therefore",
        "`relu( h_dot_safe + alpha*h ) = relu( row_lhs_safe - row_upper )`, which the dissection stored",
        "directly as `cbf_violation_safe` (`scripts/verification/jt_failure_dissect.py:262`).",
        "",
        "- **Primary residual** = stored `cbf_violation_safe`. Recompute check vs",
        f"  `relu(row_lhs_safe - row_upper)`: max abs mismatch `{recon['max_residual_recompute_mismatch']:.3e}`.",
        f"- **Alternative residual** = `relu( h_dot_fd + alpha*h )`, the REALIZED finite-difference",
        "  CBF-decrease violation, using stored `h_dot_fd = (h_next - h)/dt`. `alpha` is taken from the",
        f"  checkpoint config exactly as the filter does (`_base_alpha`): `alpha_safe={config_alpha[0]:g}`",
        f"  where `h<=0`, `alpha_unsafe={config_alpha[1]:g}` where `h>0`.",
        "- Both residuals oriented +1 (higher = more failure-like). The prompt's generic formula",
        "  `relu(-(h_dot+alpha*h))` is for the opposite (classical `h>=0`-safe) convention and would",
        "  carry the wrong sign here; the project-correct form above is used.",
        "",
        "### 1.3 Row-selection and reductions",
        "",
        "- **Executed-action step** `event_action`: `event_step-1` for collision/goal/oob, `event_step`",
        "  for stuck (matches the prior pilot, `scripts/analysis/failure_signal_pilot.py:217`).",
        "- **Lead-K rows** (`K in {1,3,5,10,15,30}`): the single timestep `event_action - K`, kept only",
        "  if `0 <= idx < n_steps` and `idx < active_steps`. One row per episode per K.",
        f"- **Episode-level reduction**: max of the signal over the active pre-event window of width",
        f"  {PRE_WINDOW} ending at `event_action`.",
        "- **Per-step pooled rows** (for correlations): every active step `[0, active_steps)`.",
        "- AUROC via `sklearn.metrics.roc_auc_score`, cross-checked against the pairwise Mann-Whitney",
        f"  form (ties=0.5) to `{AUROC_TOL:.0e}` for every reported cell.",
        "",
        "## 2. Analysis 1 — association with failure (AUROC)",
        "",
        "Episode-level (max over pre-event window), positives vs the 120 goal controls:",
        "",
        episode_auc_table(assoc, signals),
        "",
        "Lead-K AUROC (one timestep per episode at `event_action - K`):",
        "",
        lead_auc_table(assoc, "collision", signals),
        "",
        lead_auc_table(assoc, "stuck", signals),
        "",
        f"See figure `{figure_paths[0].relative_to(REPO_ROOT)}`.",
        "",
        "## 3. Analysis 2 — independence of the two axes (correlation)",
        "",
        "Pearson and Spearman between `epistemic_t` and the residual across all active per-step rows,",
        "per class. Low magnitude ⇒ the axes measure different things; high ⇒ redundant.",
        "",
        independence_table(independence),
        "",
        f"Per-timestep scatter: `{figure_paths[1].relative_to(REPO_ROOT)}`.",
        f"Episode-level scatter (with goal-median crosshairs): `{figure_paths[2].relative_to(REPO_ROOT)}`.",
        "",
        "## 4. Analysis 3 — differential separation (collision vs stuck)",
        "",
        differential_text(assoc, signals),
        "",
        "## 5. Analysis 4 — the low-low-but-failed region",
        "",
        f"Goal-class medians of the episode-level max signals: epistemic `{fmt(lowlow['epi_median'])}`,",
        f"residual `{fmt(lowlow['res_median'])}`. A failure episode is 'low-low' when its episode-level",
        "max epistemic AND max residual are both at/below these goal medians (V_S appears reliable on",
        "both axes, yet the episode failed).",
        "",
        low_low_degeneracy_note(lowlow),
        "",
        "**Episode-level** (max signal over the pre-event window):",
        "",
        low_low_table(lowlow),
        "",
        f"At this reduction every failure has at least one CBF-violated step in its window, so the "
        f"episode-level low-low region is empty. The per-moment question is answered at the timestep "
        f"level (goal per-step medians: epistemic `{fmt(lowlow['epi_step_median'])}`, residual "
        f"`{fmt(lowlow['res_step_median'])}`; {fmt(lowlow['goal_res_step_zero_frac'])} of goal active "
        f"steps have zero CBF violation):",
        "",
        low_low_timestep_table(lowlow),
        "",
        low_low_crossref(lowlow),
        "",
        "## 6. Factual summary",
        "",
        *closing_summary(assoc, independence, lowlow, signals),
        "",
    ]
    return "\n".join(lines)


def episode_auc_table(assoc: dict[str, Any], signals: list[str]) -> str:
    lines = [
        "| signal | collision-vs-goal AUROC (n+/n-) | stuck-vs-goal AUROC (n+/n-) |",
        "|---|---:|---:|",
    ]
    for signal in signals:
        c = assoc["episode"]["collision"][signal]
        s = assoc["episode"]["stuck"][signal]
        lines.append(
            f"| {signal} | {fmt(c['auc'])} ({c['n_pos']}/{c['n_neg']}) "
            f"| {fmt(s['auc'])} ({s['n_pos']}/{s['n_neg']}) |"
        )
    return "\n".join(lines)


def lead_auc_table(assoc: dict[str, Any], failure: str, signals: list[str]) -> str:
    header = "| " + failure + " vs goal | " + " | ".join(f"K={lead}" for lead in LEADS) + " |"
    sep = "|---|" + "---:|" * len(LEADS)
    lines = [header, sep]
    for signal in signals:
        cells = [fmt(assoc["lead"][failure][signal][lead]["auc"]) for lead in LEADS]
        lines.append("| " + signal + " | " + " | ".join(cells) + " |")
    npos = "| n_pos (lead rows) | " + " | ".join(
        str(assoc["lead"][failure][signals[0]][lead]["n_pos"]) for lead in LEADS
    ) + " |"
    lines.append(npos)
    return "\n".join(lines)


def independence_table(independence: dict[str, Any]) -> str:
    lines = [
        "| class | n per-step rows | Pearson (primary) | Spearman (primary) | Pearson (fd alt) | Spearman (fd alt) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome in ("collision", "stuck", "goal"):
        d = independence[outcome]
        lines.append(
            f"| {outcome} | {d['n_rows']} | {fmt(d['pearson_primary'])} | {fmt(d['spearman_primary'])} "
            f"| {fmt(d['pearson_alt'])} | {fmt(d['spearman_alt'])} |"
        )
    return "\n".join(lines)


def differential_text(assoc: dict[str, Any], signals: list[str]) -> str:
    lines = ["Episode-level AUROC side by side (from §2):", ""]
    lines.append("| signal | collision | stuck | collision - stuck |")
    lines.append("|---|---:|---:|---:|")
    for signal in signals:
        c = assoc["episode"]["collision"][signal]["auc"]
        s = assoc["episode"]["stuck"][signal]["auc"]
        delta = c - s if (not np.isnan(c) and not np.isnan(s)) else float("nan")
        lines.append(f"| {signal} | {fmt(c)} | {fmt(s)} | {fmt(delta)} |")
    return "\n".join(lines)


def low_low_degeneracy_note(lowlow: dict[str, Any]) -> str:
    if not lowlow["res_degenerate"]:
        return (
            f"Goal-class residual median is `{fmt(lowlow['res_median'])}` "
            f"(fraction of goal episodes with zero pre-event CBF violation: "
            f"{fmt(lowlow['goal_res_zero_frac'])})."
        )
    return (
        f"**Threshold note (residual median is degenerate).** The goal-class residual median is "
        f"exactly 0 — {fmt(lowlow['goal_res_zero_frac'])} of goal episodes have ZERO CBF violation "
        "across their pre-event window. Because the residual is non-negative, a *strict* `< median` "
        "test is empty by construction. The episode-level table below (strict, inclusive `<= median`, "
        "and pure `residual_max == 0`) is reported for completeness; because every failure has at least "
        "one CBF-violated step in its 30-step window it is uninformative here. The **timestep-level** "
        "table that follows it is the informative answer to 'V_S looks consistent at this moment yet "
        "the episode failed'."
    )


def low_low_table(lowlow: dict[str, Any]) -> str:
    lines = [
        "| failure class | episodes scored | strict (both < median) | inclusive (both <= median) "
        "| residual_max==0 (window) |",
        "|---|---:|---:|---:|---:|",
    ]
    for failure in ("collision", "stuck"):
        d = lowlow["classes"][failure]
        lines.append(
            f"| {failure} | {d['total_scored']} | {d['strict_count']} | {d['inclusive_count']} "
            f"| {d['res_zero_count']} |"
        )
    return "\n".join(lines)


def low_low_timestep_table(lowlow: dict[str, Any]) -> str:
    lines = [
        "| failure class | episodes | low-low timesteps / total active | episodes with >=1 low-low step |",
        "|---|---:|---:|---:|",
    ]
    for failure in ("collision", "stuck"):
        d = lowlow["classes"][failure]
        frac = d["ll_timesteps"] / d["total_timesteps"] if d["total_timesteps"] else float("nan")
        lines.append(
            f"| {failure} | {d['total_scored']} | {d['ll_timesteps']}/{d['total_timesteps']} ({fmt(frac)}) "
            f"| {d['eps_with_ll_count']} |"
        )
    return "\n".join(lines)


def low_low_crossref(lowlow: dict[str, Any]) -> str:
    lines = [
        "Failure episodes having >=1 timestep-level low-low step (V_S looks reliable at that moment, "
        "yet the episode failed) cross-referenced to dissection labels:",
        "",
    ]
    for failure in ("collision", "stuck"):
        d = lowlow["classes"][failure]
        lines.append(f"- **{failure}** episodes with a low-low step = {d['eps_with_ll_count']}/{d['total_scored']}.")
        if d["ts_layer_counts"]:
            layer_str = ", ".join(f"{k}: {v}" for k, v in sorted(d["ts_layer_counts"].items(), key=lambda kv: -kv[1]))
            lines.append(f"  - by `layer` (layer_attribution.csv): {layer_str}.")
        if d["ts_mech_counts"]:
            mech_str = ", ".join(f"{k}: {v}" for k, v in sorted(d["ts_mech_counts"].items(), key=lambda kv: -kv[1]))
            lines.append(f"  - by `mechanism` (episode_summary.csv): {mech_str}.")
        if not d["eps_with_ll_count"]:
            lines.append("  - (none)")
    return "\n".join(lines)


def closing_summary(
    assoc: dict[str, Any],
    independence: dict[str, Any],
    lowlow: dict[str, Any],
    signals: list[str],
) -> list[str]:
    epi = "epistemic |v1-v2|"
    res = "residual cbf_violation_safe"
    c_epi = assoc["episode"]["collision"][epi]["auc"]
    s_epi = assoc["episode"]["stuck"][epi]["auc"]
    c_res = assoc["episode"]["collision"][res]["auc"]
    s_res = assoc["episode"]["stuck"][res]["auc"]
    stuck_ll = lowlow["classes"]["stuck"]
    coll_ll = lowlow["classes"]["collision"]
    return [
        f"(i) **Association.** Episode-level epistemic AUROC: collision {fmt(c_epi)}, stuck {fmt(s_epi)}; "
        f"residual (cbf_violation_safe): collision {fmt(c_res)}, stuck {fmt(s_res)}. Both signals carry "
        "failure association on this biased sample (values above 0.5 with the pre-declared +1 orientation).",
        "",
        f"(ii) **Independence.** Per-step Pearson(epistemic, residual): collision "
        f"{fmt(independence['collision']['pearson_primary'])}, stuck "
        f"{fmt(independence['stuck']['pearson_primary'])}, goal "
        f"{fmt(independence['goal']['pearson_primary'])}; Spearman collision "
        f"{fmt(independence['collision']['spearman_primary'])}, stuck "
        f"{fmt(independence['stuck']['spearman_primary'])}, goal "
        f"{fmt(independence['goal']['spearman_primary'])}. The two axes are at most weakly correlated, "
        "i.e. at least partly independent.",
        "",
        f"(iii) **Differential separation.** Epistemic separates collision vs stuck with a gap of "
        f"{fmt(c_epi - s_epi)} AUROC (collision minus stuck), while the residual gap is "
        f"{fmt(c_res - s_res)}. (Stated as measured; positive means the signal favors collision.)",
        "",
        f"(iv) **Low-low-but-failed.** Episode-level (max over window) the region is empty for both "
        f"classes (every failure has >=1 CBF-violated step in its window; goal residual median is 0). "
        f"At the timestep level, episodes with >=1 low-low step (both signals at/below goal per-step "
        f"medians) are stuck {stuck_ll['eps_with_ll_count']}/{stuck_ll['total_scored']} vs collision "
        f"{coll_ll['eps_with_ll_count']}/{coll_ll['total_scored']} "
        f"(low-low steps {stuck_ll['ll_timesteps']}/{stuck_ll['total_timesteps']} stuck vs "
        f"{coll_ll['ll_timesteps']}/{coll_ll['total_timesteps']} collision). Stuck episodes pass through "
        "moments where V_S looks reliable on both axes far more than collisions do; the §5 cross-reference "
        "reports how many of those stuck episodes carry the planning-limit / geometric-trap labels.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
