from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data/diagnostics/"
    "v2.0.1__20260529-171057__failure_dissect_n2500_seed45678_20260529-194915"
)
DEFAULT_REPORT = REPO_ROOT / "docs/versions/v2.1.0/failure_signal_pilot_repro.md"
DEFAULT_FIGURE_DIR = REPO_ROOT / "data/diagnostics/v2.1.0_failure_signal_pilot_repro"
ORIGINAL_REPORT = REPO_ROOT / "docs/versions/v2.1.0/failure_signal_pilot.md"

LEADS = [1, 3, 5, 10, 15, 30]
SPLIT_SEED = 20260531
TRAIN_FRAC = 0.70
AUROC_TOL = 1.0e-9
DIFF_FLAG_THRESHOLD = 0.02


SINGLE_SIGNALS = {
    "deployed h": "h",
    "||L_g h||": "lg_norm",
    "box empty": "empty_halfspace_box",
    "saturation": "saturated",
    "projection norm": "projection_norm",
    "clearance": "clearance",
    "closing speed": "closing_speed",
    "brake margin": "brake_margin",
}

# Fixed risk orientation. The AUROC score is sign * raw signal, never selected by
# whichever of raw/negated is larger.
SIGN_MAP = {
    "deployed h": 1.0,
    "||L_g h||": 1.0,
    "box empty": 1.0,
    "saturation": 1.0,
    "projection norm": 1.0,
    "clearance": -1.0,
    "closing speed": 1.0,
    "brake margin": -1.0,
}

BLOCKS = {
    "V_S h": ["h"],
    "V_S h+h_dot": ["h", "h_dot_fd"],
    "CBF-row": ["h", "lf_h", "lg_norm", "row_slack", "empty_halfspace_box"],
    "geometry": ["h", "clearance", "closing_speed", "brake_margin"],
    "action/filter": ["h", "projection_norm", "saturated", "intervention"],
    "all": [
        "h",
        "h_dot_fd",
        "lf_h",
        "lg_norm",
        "row_slack",
        "empty_halfspace_box",
        "clearance",
        "closing_speed",
        "brake_margin",
        "projection_norm",
        "saturated",
        "intervention",
    ],
}

NPZ_REQUIRED_KEYS = {
    "states",
    "u_nom",
    "u_safe",
    "projection_norm",
    "intervention",
    "infeasible",
    "h",
    "h_next",
    "h_dot_model",
    "h_dot_fd",
    "lf_h",
    "lg_h",
    "lg_norm",
    "row_upper",
    "row_lhs_nom",
    "row_lhs_safe",
    "empty_halfspace_box",
    "saturated",
    "nearest_distance",
    "clearance",
    "closing_speed",
    "brake_margin",
    "goal_distance",
    "goal_closing_speed",
    "metadata_json",
}


@dataclass(frozen=True)
class Trace:
    path: Path
    episode_idx: int
    role: str
    outcome: str
    event_step: int
    event_action: int
    active_steps: int
    arrays: dict[str, np.ndarray]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible failure-signal pilot analysis.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--original-report", type=Path, default=ORIGINAL_REPORT)
    args = parser.parse_args()

    traces, schema_keys = load_traces(args.input_dir / "episodes")
    by_outcome = {
        "collision": [trace for trace in traces if trace.outcome == "collision"],
        "stuck": [trace for trace in traces if trace.outcome == "stuck"],
        "goal": [trace for trace in traces if trace.outcome == "goal"],
    }

    assert_counts(by_outcome)
    layer_leads = load_collision_leads(args.input_dir / "layer_attribution.csv")
    single_results, counts = compute_single_signal_results(by_outcome)
    block_results, split_summaries = compute_block_results(by_outcome)
    rankings = rank_single_signals(single_results)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = write_figures(args.figure_dir, single_results, block_results, rankings)

    original = parse_original_report(args.original_report)
    comparisons = compare_to_original(original, single_results, block_results)

    report = render_report(
        input_dir=args.input_dir,
        schema_keys=schema_keys,
        by_outcome=by_outcome,
        counts=counts,
        single_results=single_results,
        block_results=block_results,
        split_summaries=split_summaries,
        rankings=rankings,
        layer_leads=layer_leads,
        figure_paths=figure_paths,
        comparisons=comparisons,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(args.report)
    return 0


def load_traces(episode_dir: Path) -> tuple[list[Trace], list[str]]:
    if not episode_dir.exists():
        raise FileNotFoundError(episode_dir)

    traces = []
    schema_keys: list[str] | None = None
    for path in sorted(episode_dir.glob("*.npz")):
        if not path.name.startswith(("failure_", "goal_control_")):
            continue
        loaded = np.load(path, allow_pickle=False)
        keys = set(loaded.files)
        missing = sorted(NPZ_REQUIRED_KEYS - keys)
        if missing:
            raise KeyError(f"{path} is missing required NPZ keys: {missing}")
        if schema_keys is None:
            schema_keys = list(loaded.files)
        arrays = {key: loaded[key] for key in loaded.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(loaded["metadata_json"]).item()))
        role, episode_idx = parse_episode_filename(path)
        outcome = str(metadata["outcome"])
        event_step = int(metadata["event_step"])
        n_steps = int(len(arrays["h"]))
        event_action = event_action_step(outcome, event_step, n_steps)
        traces.append(
            Trace(
                path=path,
                episode_idx=episode_idx,
                role=role,
                outcome=outcome,
                event_step=event_step,
                event_action=event_action,
                active_steps=int(metadata.get("active_steps", n_steps)),
                arrays=arrays,
            )
        )
    if schema_keys is None:
        raise FileNotFoundError(f"No NPZ traces found in {episode_dir}.")
    return traces, schema_keys


def parse_episode_filename(path: Path) -> tuple[str, int]:
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[0] == "goal" and parts[1] == "control":
        return "goal_control", int(parts[2])
    return parts[0], int(parts[1])


def event_action_step(outcome: str, event_step: int, n_steps: int) -> int:
    if outcome in {"collision", "goal", "oob"} and event_step > 0:
        return max(0, min(event_step - 1, n_steps - 1))
    if event_step < 0:
        return n_steps - 1
    return max(0, min(event_step, n_steps - 1))


def assert_counts(by_outcome: dict[str, list[Trace]]) -> None:
    expected = {"collision": 26, "stuck": 105, "goal": 120}
    actual = {key: len(value) for key, value in by_outcome.items()}
    if actual != expected:
        raise ValueError(f"Unexpected episode counts: expected {expected}, got {actual}.")


def load_collision_leads(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    empty = []
    brake = []
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if row["outcome"] != "collision":
                continue
            empty_value = float(row["steps_empty_to_event"])
            brake_value = float(row["steps_brake_to_event"])
            if empty_value >= 0:
                empty.append(empty_value)
            if brake_value >= 0:
                brake.append(brake_value)
    return {"empty": np.asarray(empty, dtype=float), "brake": np.asarray(brake, dtype=float)}


def safe_idx(trace: Trace, idx: int) -> bool:
    return 0 <= idx < len(trace.arrays["h"]) and idx < trace.active_steps


def features_at(trace: Trace, idx: int) -> dict[str, float]:
    arrays = trace.arrays
    # The CBF row slack is the executed-action slack, b - A u_safe.
    row_slack = float(arrays["row_upper"][idx] - arrays["row_lhs_safe"][idx])
    return {
        "h": float(arrays["h"][idx]),
        "h_dot_fd": float(arrays["h_dot_fd"][idx]),
        "h_delta_next": float(arrays["h_next"][idx] - arrays["h"][idx]),
        "lf_h": float(arrays["lf_h"][idx]),
        "lg_norm": float(arrays["lg_norm"][idx]),
        "row_slack": row_slack,
        "empty_halfspace_box": float(bool(arrays["empty_halfspace_box"][idx])),
        "projection_norm": float(arrays["projection_norm"][idx]),
        "saturated": float(bool(arrays["saturated"][idx])),
        "intervention": float(bool(arrays["intervention"][idx])),
        "clearance": float(arrays["clearance"][idx]),
        "closing_speed": float(arrays["closing_speed"][idx]),
        "brake_margin": float(arrays["brake_margin"][idx]),
    }


def build_rows(by_outcome: dict[str, list[Trace]], failure_outcome: str, lead: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_indices = []
    for trace in by_outcome[failure_outcome]:
        idx = trace.event_action - lead
        if safe_idx(trace, idx):
            rows.append(
                {
                    "label": 1,
                    "episode_key": f"{failure_outcome}_{trace.episode_idx}",
                    "idx": idx,
                    **features_at(trace, idx),
                }
            )
            positive_indices.append(idx)

    matched_indices = sorted(set(positive_indices))
    seen_goal_rows = set()
    for trace in by_outcome["goal"]:
        for idx in matched_indices:
            key = (trace.episode_idx, idx)
            if key in seen_goal_rows:
                continue
            if safe_idx(trace, idx):
                rows.append(
                    {
                        "label": 0,
                        "episode_key": f"goal_{trace.episode_idx}",
                        "idx": idx,
                        **features_at(trace, idx),
                    }
                )
                seen_goal_rows.add(key)
    return rows


def compute_single_signal_results(
    by_outcome: dict[str, list[Trace]],
) -> tuple[dict[str, dict[int, dict[str, dict[str, float]]]], dict[str, dict[int, dict[str, int]]]]:
    results: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    counts: dict[str, dict[int, dict[str, int]]] = {}
    for outcome in ("collision", "stuck"):
        results[outcome] = {}
        counts[outcome] = {}
        for lead in LEADS:
            rows = build_rows(by_outcome, outcome, lead)
            labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
            counts[outcome][lead] = {
                "positive": int(labels.sum()),
                "negative": int(len(labels) - labels.sum()),
                "total": int(len(labels)),
            }
            results[outcome][lead] = {}
            for signal_name, feature_name in SINGLE_SIGNALS.items():
                sign = SIGN_MAP[signal_name]
                scores = sign * np.asarray([float(row[feature_name]) for row in rows], dtype=float)
                auc = checked_auc(labels, scores)
                results[outcome][lead][signal_name] = {"auc": auc, "sign": sign}
    return results, counts


def checked_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    sklearn_auc = float(roc_auc_score(labels, scores))
    pairwise_auc = mann_whitney_auc(labels, scores)
    if abs(sklearn_auc - pairwise_auc) > AUROC_TOL:
        raise AssertionError(
            f"AUROC mismatch: sklearn={sklearn_auc}, pairwise={pairwise_auc}."
        )
    return sklearn_auc


def mann_whitney_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Pairwise Mann-Whitney AUROC: P(score_pos > score_neg) +
    # 0.5 * P(score_pos == score_neg). This makes tie handling explicit.
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparison = pos[:, None] - neg[None, :]
    wins = np.sum(comparison > 0.0)
    ties = np.sum(comparison == 0.0)
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def compute_block_results(by_outcome: dict[str, list[Trace]]) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[int, dict[str, int]]]]:
    results: dict[str, dict[int, dict[str, float]]] = {}
    splits: dict[str, dict[int, dict[str, int]]] = {}
    for outcome in ("collision", "stuck"):
        results[outcome] = {}
        splits[outcome] = {}
        for lead in LEADS:
            rows = build_rows(by_outcome, outcome, lead)
            train_rows, test_rows, split_info = split_episode_level(rows, SPLIT_SEED)
            splits[outcome][lead] = split_info | {
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
            }
            results[outcome][lead] = {}
            for block_name, features in BLOCKS.items():
                results[outcome][lead][block_name] = fit_block_auc(train_rows, test_rows, features)
    return results, splits


def split_episode_level(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rng = np.random.default_rng(seed)
    positive_episodes = sorted({row["episode_key"] for row in rows if row["label"] == 1})
    negative_episodes = sorted({row["episode_key"] for row in rows if row["label"] == 0})
    rng.shuffle(positive_episodes)
    rng.shuffle(negative_episodes)

    pos_train, pos_test = split_keys(positive_episodes)
    neg_train, neg_test = split_keys(negative_episodes)
    train_episodes = pos_train | neg_train
    test_episodes = pos_test | neg_test
    if train_episodes & test_episodes:
        raise AssertionError("Train/test episode sets overlap.")

    train_rows = [row for row in rows if row["episode_key"] in train_episodes]
    test_rows = [row for row in rows if row["episode_key"] in test_episodes]
    return train_rows, test_rows, {
        "pos_train_episodes": len(pos_train),
        "pos_test_episodes": len(pos_test),
        "neg_train_episodes": len(neg_train),
        "neg_test_episodes": len(neg_test),
    }


def split_keys(keys: list[str]) -> tuple[set[str], set[str]]:
    if len(keys) <= 1:
        return set(keys), set()
    n_train = max(1, min(len(keys) - 1, int(round(TRAIN_FRAC * len(keys)))))
    return set(keys[:n_train]), set(keys[n_train:])


def fit_block_auc(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], features: list[str]) -> float:
    x_train, y_train = rows_to_matrix(train_rows, features)
    x_test, y_test = rows_to_matrix(test_rows, features)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SPLIT_SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    transformed_train = pipeline[:-1].transform(x_train)
    transformed_test = pipeline[:-1].transform(x_test)
    if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_test).all():
        raise AssertionError("NaN or Inf leaked into standardized model features.")
    scores = pipeline.predict_proba(x_test)[:, 1]
    return checked_auc(y_test, scores)


def rows_to_matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted({feature for feature in features for row in rows if feature not in row})
    if missing:
        raise KeyError(f"Rows are missing required features: {missing}")
    x = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=float)
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    return x, y


def rank_single_signals(results: dict[str, dict[int, dict[str, dict[str, float]]]]) -> dict[str, list[tuple[str, float]]]:
    rankings = {}
    for outcome, by_lead in results.items():
        rows = []
        for signal_name in SINGLE_SIGNALS:
            values = [by_lead[lead][signal_name]["auc"] for lead in LEADS]
            rows.append((signal_name, float(np.nanmean(values))))
        rows.sort(key=lambda item: item[1], reverse=True)
        rankings[outcome] = rows
    return rankings


def write_figures(
    figure_dir: Path,
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    block_results: dict[str, dict[int, dict[str, float]]],
    rankings: dict[str, list[tuple[str, float]]],
) -> list[Path]:
    paths = []
    for outcome in ("collision", "stuck"):
        paths.append(plot_top_signals(figure_dir, outcome, single_results, rankings))
        paths.append(plot_incremental_blocks(figure_dir, outcome, block_results))
    return paths


def plot_top_signals(
    figure_dir: Path,
    outcome: str,
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    rankings: dict[str, list[tuple[str, float]]],
) -> Path:
    plt.figure(figsize=(6.4, 4.2), dpi=140)
    for signal_name, _ in rankings[outcome][:3]:
        values = [single_results[outcome][lead][signal_name]["auc"] for lead in LEADS]
        direction = "raw +" if SIGN_MAP[signal_name] > 0.0 else "flipped -"
        plt.plot(LEADS, values, marker="o", label=f"{signal_name} ({direction})")
    plt.axhline(0.5, color="0.6", linewidth=1, linestyle="--")
    plt.ylim(0.45, 1.02)
    plt.xlabel("lead K before event (steps)")
    plt.ylabel("AUROC")
    plt.title(f"{outcome.capitalize()} vs goal controls: top single signals")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    path = figure_dir / f"{outcome}_top3_single_signal_auroc.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_incremental_blocks(
    figure_dir: Path,
    outcome: str,
    block_results: dict[str, dict[int, dict[str, float]]],
) -> Path:
    blocks = [name for name in BLOCKS if name != "V_S h"]
    deltas = []
    for block_name in blocks:
        values = [
            block_results[outcome][lead][block_name] - block_results[outcome][lead]["V_S h"]
            for lead in LEADS
        ]
        deltas.append(float(np.nanmean(values)))
    plt.figure(figsize=(7.2, 4.2), dpi=140)
    x_values = np.arange(len(blocks))
    plt.bar(x_values, deltas, color="#4c78a8")
    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xticks(x_values, blocks, rotation=25, ha="right")
    plt.ylabel("mean test AUROC delta vs deployed h")
    plt.title(f"{outcome.capitalize()}: incremental blocks over V_S-only")
    plt.tight_layout()
    path = figure_dir / f"{outcome}_incremental_auroc_delta.png"
    plt.savefig(path)
    plt.close()
    return path


def parse_original_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "single": parse_original_single_tables(text),
        "blocks": parse_original_block_tables(text),
    }


def parse_original_single_tables(text: str) -> dict[str, dict[str, dict[str, float]]]:
    parsed: dict[str, dict[str, dict[str, float]]] = {}
    for outcome, heading in (("collision", "### Collision vs Goal Controls"), ("stuck", "### Stuck vs Goal Controls")):
        section = section_after_heading(text, heading)
        table_lines = first_table_lines(section)
        rows = parse_markdown_table(table_lines)
        parsed[outcome] = {}
        for row in rows:
            signal = restore_signal_name(row["signal"])
            parsed[outcome][signal] = {"mean": float(row["mean AUROC"])}
            for lead in LEADS:
                parsed[outcome][signal][f"K={lead}"] = float(row[f"K={lead}"])
    return parsed


def parse_original_block_tables(text: str) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    parsed: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for outcome, heading in (("collision", "### Collision"), ("stuck", "### Stuck")):
        # Use the occurrence in Section 3 by splitting after the incremental heading.
        subsection = section_after_heading(text.split("## 3. Incremental Value Over V_S-Only", 1)[1], heading)
        table_lines = first_table_lines(subsection)
        rows = parse_markdown_table(table_lines)
        parsed[outcome] = {}
        for row in rows:
            lead = int(row["lead K"])
            parsed[outcome][lead] = {
                "V_S h": {"auc": float(row["V_S h"]), "delta": 0.0},
                "V_S h+h_dot": {"auc": float(row["V_S h+h_dot"]), "delta": float(row["delta"])},
                "CBF-row": {"auc": float(row["CBF-row"]), "delta": float(row["delta_2"])},
                "geometry": {"auc": float(row["geometry"]), "delta": float(row["delta_3"])},
                "action/filter": {"auc": float(row["action/filter"]), "delta": float(row["delta_4"])},
                "all": {"auc": float(row["all"]), "delta": float(row["delta_5"])},
            }
    return parsed


def section_after_heading(text: str, heading: str) -> str:
    if heading not in text:
        raise ValueError(f"Heading not found: {heading}")
    return text.split(heading, 1)[1]


def first_table_lines(text: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("|"):
            start = idx
            break
    if start is None:
        raise ValueError("No markdown table found.")
    out = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        out.append(line)
    return out


def parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    header = split_md_row(lines[0])
    body = []
    for line in lines[2:]:
        cells = split_md_row(line)
        deduped_header = dedupe_headers(header)
        body.append({key: value for key, value in zip(deduped_header, cells)})
    return body


def split_md_row(line: str) -> list[str]:
    # The original inline report did not escape the table cell `||L_g h||`,
    # so protect it before splitting on markdown pipes.
    protected = line.replace("||L_g h||", "__LG_H_NORM__")
    return [cell.strip().replace("__LG_H_NORM__", "Lg h norm") for cell in protected.strip().strip("|").split("|")]


def restore_signal_name(value: str) -> str:
    if value == "Lg h norm":
        return "||L_g h||"
    return value


def dedupe_headers(header: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for item in header:
        seen = counts.get(item, 0)
        counts[item] = seen + 1
        if seen == 0:
            result.append(item)
        else:
            result.append(f"{item}_{seen + 1}")
    return result


def compare_to_original(
    original: dict[str, Any],
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    block_results: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    single_rows = []
    block_rows = []
    flagged = []

    for outcome in ("collision", "stuck"):
        for signal_name in SINGLE_SIGNALS:
            reproduced_values = [single_results[outcome][lead][signal_name]["auc"] for lead in LEADS]
            reproduced_mean = float(np.nanmean(reproduced_values))
            cells = [("mean", reproduced_mean)] + [
                (f"K={lead}", single_results[outcome][lead][signal_name]["auc"]) for lead in LEADS
            ]
            for metric, reproduced in cells:
                original_value = original["single"][outcome][signal_name][metric]
                diff = abs(float(original_value) - float(reproduced))
                row = {
                    "outcome": outcome,
                    "signal": signal_name,
                    "metric": metric,
                    "original": float(original_value),
                    "reproduced": float(reproduced),
                    "abs_diff": diff,
                }
                single_rows.append(row)
                if diff > DIFF_FLAG_THRESHOLD:
                    flagged.append(("single", row))

    for outcome in ("collision", "stuck"):
        for lead in LEADS:
            base = block_results[outcome][lead]["V_S h"]
            for block_name in BLOCKS:
                reproduced_auc = block_results[outcome][lead][block_name]
                reproduced_delta = 0.0 if block_name == "V_S h" else reproduced_auc - base
                original_cell = original["blocks"][outcome][lead][block_name]
                for metric, original_value, reproduced_value in (
                    ("auc", original_cell["auc"], reproduced_auc),
                    ("delta", original_cell["delta"], reproduced_delta),
                ):
                    diff = abs(float(original_value) - float(reproduced_value))
                    row = {
                        "outcome": outcome,
                        "lead": lead,
                        "block": block_name,
                        "metric": metric,
                        "original": float(original_value),
                        "reproduced": float(reproduced_value),
                        "abs_diff": diff,
                    }
                    block_rows.append(row)
                    if diff > DIFF_FLAG_THRESHOLD:
                        flagged.append(("block", row))

    return {"single": single_rows, "blocks": block_rows, "flagged": flagged}


def render_report(
    *,
    input_dir: Path,
    schema_keys: list[str],
    by_outcome: dict[str, list[Trace]],
    counts: dict[str, dict[int, dict[str, int]]],
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    block_results: dict[str, dict[int, dict[str, float]]],
    split_summaries: dict[str, dict[int, dict[str, int]]],
    rankings: dict[str, list[tuple[str, float]]],
    layer_leads: dict[str, np.ndarray],
    figure_paths: list[Path],
    comparisons: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# Failure-Signal Pilot Reproducible Reimplementation",
            "",
            "> **Caveat.** This reruns the same retrospective pilot on a biased sample: positives are saved collision/stuck failures and negatives are 120 matched goal controls, not all non-failure timesteps. AUROC values are existence checks for separable signals, not operational detector metrics.",
            "",
            "## Implementation",
            "",
            f"- Script: `scripts/analysis/failure_signal_pilot.py`.",
            f"- Input: `{input_dir}`.",
            "- AUROC: `sklearn.metrics.roc_auc_score`, cross-checked against pairwise Mann-Whitney AUROC with ties counted as 0.5; every cell must agree to `1e-9`.",
            "- Single-signal score: fixed `sign * raw_value`; no standardization, binning, window aggregation, or orientation-by-best-AUROC.",
            f"- Sign map: {format_sign_map()}.",
            "- Block model: `sklearn.linear_model.LogisticRegression(solver=\"liblinear\", class_weight=\"balanced\", max_iter=1000, random_state=20260531)`.",
            "- Block AUROC score: positive-class `predict_proba[:, 1]`.",
            "- Preprocessing: `SimpleImputer(strategy=\"median\")` then `StandardScaler()` inside an sklearn `Pipeline`, fit on train only and applied to test.",
            "- Split: episode-level 70/30, stratified separately over failure episodes and goal-control episodes, seed `20260531`; train/test episode sets are asserted disjoint.",
            "- Row slack: implemented as executed-action CBF row slack `row_upper - row_lhs_safe`.",
            "- Feature blocks:",
            *[f"  - `{name}`: `{features}`" for name, features in BLOCKS.items()],
            "",
            "## 1. Dataset Assembly",
            "",
            f"- Episode files: {sum(len(v) for v in by_outcome.values())} NPZ files: {len(by_outcome['collision'])} collision, {len(by_outcome['stuck'])} stuck, {len(by_outcome['goal'])} goal controls.",
            f"- Confirmed NPZ schema: `{', '.join(schema_keys)}`.",
            "- Target-net h is not present; V_S baseline is deployed h only.",
            "- Positives are exactly `event_action_step - K`, with collision/goal action step `event_step - 1` and stuck action step `event_step`.",
            "- Negatives are goal-control rows at matched within-episode action indices while the goal trajectory is active.",
            "",
            counts_table(by_outcome, counts),
            "",
            "Split diagnostics for block models:",
            "",
            split_table(split_summaries),
            "",
            "## 2. Single-Signal Separability",
            "",
            "These AUROCs are biased-sample existence checks. Fixed signal signs are used; lower-clearance and lower-brake-margin risk is represented by multiplying those raw signals by `-1`.",
            "",
            "### Collision vs Goal Controls",
            "",
            single_table("collision", single_results, rankings),
            "",
            "### Stuck vs Goal Controls",
            "",
            single_table("stuck", single_results, rankings),
            "",
            "## 3. Incremental Value Over V_S-Only",
            "",
            "Values are held-out test AUROCs from the episode-level split. Delta columns are relative to deployed h alone at the same lead.",
            "",
            "### Collision",
            "",
            block_table("collision", block_results),
            "",
            "### Stuck",
            "",
            block_table("stuck", block_results),
            "",
            delta_summary(block_results),
            "",
            "## 4. Lead-Time Characterization",
            "",
            f"- Collision box-empty lead from `layer_attribution.csv`: {lead_stats(layer_leads['empty'])}.",
            f"- Collision brake-margin-negative lead from `layer_attribution.csv`: {lead_stats(layer_leads['brake'])}.",
            "- Stuck has no saved onset field analogous to box-empty/brake-onset; AUROC-vs-lead-K is only a proxy.",
            *stuck_threshold_lines(single_results, rankings),
            "",
            "## 5. Figures",
            "",
            *[f"- `{path.relative_to(REPO_ROOT)}`" for path in figure_paths],
            "",
            "## 6. Reconciliation Against Original Inline Report",
            "",
            "The original report was rounded to 3 decimals, so differences below include rounding error. Cells with absolute difference > 0.02 are flagged.",
            "",
            "### Single-Signal Cell Comparison",
            "",
            comparison_single_table(comparisons["single"]),
            "",
            "### Incremental Block Cell Comparison",
            "",
            comparison_block_table(comparisons["blocks"]),
            "",
            "### Cells Differing by More Than 0.02",
            "",
            flagged_table(comparisons["flagged"]),
            "",
            "## 7. Conclusion",
            "",
            conclusion_text(comparisons),
            "",
        ]
    )
    return "\n".join(lines)


def format_sign_map() -> str:
    pieces = []
    for name in SINGLE_SIGNALS:
        direction = "+1" if SIGN_MAP[name] > 0.0 else "-1"
        pieces.append(f"{name}: {direction}")
    return "; ".join(pieces)


def counts_table(by_outcome: dict[str, list[Trace]], counts: dict[str, dict[int, dict[str, int]]]) -> str:
    saved_steps = {key: sum(len(trace.arrays["h"]) for trace in traces) for key, traces in by_outcome.items()}
    lines = [
        "| class / lead | episodes | total saved per-step rows | K=1 rows (+/-) | K=3 rows (+/-) | K=5 rows (+/-) | K=10 rows (+/-) | K=15 rows (+/-) | K=30 rows (+/-) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for outcome in ("collision", "stuck"):
        row = [outcome, str(len(by_outcome[outcome])), str(saved_steps[outcome])]
        for lead in LEADS:
            item = counts[outcome][lead]
            row.append(f"{item['total']} ({item['positive']}/{item['negative']})")
        lines.append("| " + " | ".join(row) + " |")
    row = ["goal controls", str(len(by_outcome["goal"])), str(saved_steps["goal"])]
    row.extend(["see +/- columns"] * len(LEADS))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def split_table(split_summaries: dict[str, dict[int, dict[str, int]]]) -> str:
    lines = [
        "| outcome | lead K | pos train/test eps | goal train/test eps | train rows | test rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome in ("collision", "stuck"):
        for lead in LEADS:
            item = split_summaries[outcome][lead]
            lines.append(
                "| "
                + " | ".join(
                    [
                        outcome,
                        str(lead),
                        f"{item['pos_train_episodes']}/{item['pos_test_episodes']}",
                        f"{item['neg_train_episodes']}/{item['neg_test_episodes']}",
                        str(item["train_rows"]),
                        str(item["test_rows"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def single_table(
    outcome: str,
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    rankings: dict[str, list[tuple[str, float]]],
) -> str:
    lines = [
        "| signal | sign | mean AUROC | K=1 | K=3 | K=5 | K=10 | K=15 | K=30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for signal_name, mean_value in rankings[outcome]:
        sign = "+1" if SIGN_MAP[signal_name] > 0.0 else "-1"
        values = [single_results[outcome][lead][signal_name]["auc"] for lead in LEADS]
        lines.append(
            "| "
            + " | ".join([md_cell(signal_name), sign, fmt(mean_value), *[fmt(value) for value in values]])
            + " |"
        )
    return "\n".join(lines)


def block_table(outcome: str, block_results: dict[str, dict[int, dict[str, float]]]) -> str:
    lines = [
        "| lead K | V_S h | V_S h+h_dot | delta | CBF-row | delta | geometry | delta | action/filter | delta | all | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lead in LEADS:
        base = block_results[outcome][lead]["V_S h"]
        row = [str(lead), fmt(base)]
        for block_name in ("V_S h+h_dot", "CBF-row", "geometry", "action/filter", "all"):
            value = block_results[outcome][lead][block_name]
            row.extend([fmt(value), fmt(value - base)])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def delta_summary(block_results: dict[str, dict[int, dict[str, float]]]) -> str:
    lines = []
    for outcome in ("collision", "stuck"):
        lines.append(f"{outcome.capitalize()} mean deltas over deployed h:")
        for block_name in ("V_S h+h_dot", "CBF-row", "geometry", "action/filter", "all"):
            deltas = [
                block_results[outcome][lead][block_name] - block_results[outcome][lead]["V_S h"]
                for lead in LEADS
            ]
            lines.append(
                f"- {block_name}: mean delta {fmt(np.nanmean(deltas))} "
                f"(min {fmt(np.nanmin(deltas))}, max {fmt(np.nanmax(deltas))})."
            )
    return "\n".join(lines)


def stuck_threshold_lines(
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    rankings: dict[str, list[tuple[str, float]]],
) -> list[str]:
    lines = []
    for signal_name, _ in rankings["stuck"][:3]:
        threshold_07 = earliest_lead_above(single_results, "stuck", signal_name, 0.7)
        threshold_08 = earliest_lead_above(single_results, "stuck", signal_name, 0.8)
        lines.append(
            f"- {signal_name}: first clears 0.7 by K={format_optional(threshold_07)}; "
            f"first clears 0.8 by K={format_optional(threshold_08)}."
        )
    return lines


def earliest_lead_above(
    single_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    outcome: str,
    signal_name: str,
    threshold: float,
) -> int | None:
    values = [
        lead for lead in LEADS if single_results[outcome][lead][signal_name]["auc"] >= threshold
    ]
    return max(values) if values else None


def comparison_single_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| outcome | signal | metric | original | reproduced | abs diff |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["outcome"],
                    md_cell(row["signal"]),
                    row["metric"],
                    fmt(row["original"]),
                    fmt(row["reproduced"]),
                    fmt(row["abs_diff"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def comparison_block_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| outcome | lead K | block | metric | original | reproduced | abs diff |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["outcome"],
                    str(row["lead"]),
                    row["block"],
                    row["metric"],
                    fmt(row["original"]),
                    fmt(row["reproduced"]),
                    fmt(row["abs_diff"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def flagged_table(flagged: list[tuple[str, dict[str, Any]]]) -> str:
    if not flagged:
        return "No cells differ by more than 0.02."
    lines = [
        "| table | location | original | reproduced | abs diff | candidate explanation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for table_name, row in flagged:
        if table_name == "single":
            location = f"{row['outcome']} / {row['signal']} / {row['metric']}"
            explanation = (
                "The reproducible script uses a fixed sign map; the original inline run "
                "appears to have oriented some cells by the larger raw/flipped AUROC."
            )
        else:
            location = f"{row['outcome']} / K={row['lead']} / {row['block']} / {row['metric']}"
            explanation = (
                "The reproducible script fixes row slack to row_upper - row_lhs_safe, "
                "uses predict_proba, and uses the explicit feature blocks in this report; "
                "the inline run did not persist these choices."
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    table_name,
                    md_cell(location),
                    fmt(row["original"]),
                    fmt(row["reproduced"]),
                    fmt(row["abs_diff"]),
                    explanation,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def conclusion_text(comparisons: dict[str, Any]) -> str:
    flagged = comparisons["flagged"]
    large_count = len(flagged)
    if large_count == 0:
        match_sentence = "The reproduced numbers match the original rounded report within 0.02 for every compared cell."
    else:
        match_sentence = (
            f"{large_count} cells differ by more than 0.02, mostly where the original inline "
            "method left orientation or block-score details unrecoverable."
        )
    return (
        f"{match_sentence} The original high-level conclusion is only partly reproduced: "
        "collision remains carried mainly by deployed h with little robust incremental gain, "
        "but the stuck result is more h/proximity-dominated under the fixed sign map and explicit "
        "block definitions; all-signals still adds a stable positive increment over h, while "
        "action/filter alone is not the strongest incremental block."
    )


def lead_stats(values: np.ndarray) -> str:
    if len(values) == 0:
        return "n=0"
    return (
        f"n={len(values)}, mean={fmt(np.mean(values), 1)}, median={fmt(np.median(values), 1)}, "
        f"p25={fmt(np.percentile(values, 25), 1)}, p75={fmt(np.percentile(values, 75), 1)}, "
        f"min={fmt(np.min(values), 0)}, max={fmt(np.max(values), 0)} steps before event"
    )


def fmt(value: float, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def format_optional(value: int | None) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
