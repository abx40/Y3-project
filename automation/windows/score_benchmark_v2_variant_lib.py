import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from automation.windows import score_benchmark_v2 as sb


@dataclass(frozen=True)
class VariantConfig:
    name: str
    output_dir_default: str
    window_seconds: int
    summary_default: str
    sessions_index_default: str
    eval_root_default: str
    strict_false_alerts_per_min: Optional[float]
    strict_fa_time_ratio: Optional[float]
    relaxed_false_alerts_per_min: Optional[float]
    relaxed_fa_time_ratio: Optional[float]
    selection_rule: str
    validation_ranking_rule: str
    summary_title: str
    summary_gate_text: str
    selection_mode: str


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.summary_title)
    parser.add_argument("--summary", default=config.summary_default)
    parser.add_argument("--sessions-index", default=config.sessions_index_default)
    parser.add_argument("--eval-root", default=config.eval_root_default)
    parser.add_argument("--output-dir", default=config.output_dir_default)
    return parser.parse_args()


def pick_best_nogate_maxf1(candidates: Sequence[dict]) -> dict:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["target_metrics"]["f1"],
            item["target_metrics"]["time_to_first_detection_s"],
            -item["target_metrics"]["recall"],
            item["real_only_metrics"]["false_alerts_per_min"],
            item["real_only_metrics"]["false_alert_time_ratio"],
            item["target_metrics"]["alert_flicker"],
            item["op"].threshold,
            item["op"].persistence_windows,
            item["op"].hysteresis_gap,
        ),
    )
    best = dict(ranked[0])
    best["gate_label"] = "nogate_maxf1"
    return best


def rank_validation_rows(rows: Sequence[dict], config: VariantConfig) -> List[dict]:
    if config.selection_mode == "nogate_maxf1":
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["f1"],
                row["time_to_first_detection_s"],
                row["false_alerts_per_min"],
                row["false_alert_time_ratio"],
                row["backend"],
            ),
        )
    else:
        ranked = sorted(
            rows,
            key=lambda row: (
                row["false_alerts_per_min"],
                row["time_to_first_detection_s"],
                -row["f1"],
                row["backend"],
            ),
        )
    output: List[dict] = []
    for idx, row in enumerate(ranked, start=1):
        updated = dict(row)
        updated["rank"] = idx
        updated["winner"] = idx == 1
        output.append(updated)
    return output


def build_summary_report(
    selected_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
    config: VariantConfig,
) -> str:
    winner = validation_rows[0]
    lines = [
        f"# {config.summary_title}",
        "",
        "## Method",
        "",
        f"- Evaluation unit: {config.window_seconds}-second window.",
        (
            "- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into "
            f"{config.window_seconds}-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed."
        ),
        "- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.",
        "- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.",
        f"- {config.summary_gate_text}",
        f"- Calibration selection: {config.selection_rule}",
        f"- Validation ranking: {config.validation_ranking_rule}",
        "",
        "## Winner",
        "",
        f"- Validation winner: `{winner['backend']}`",
        f"- Validation F1: {winner['f1']:.4f}",
        f"- Validation false alerts/min: {winner['false_alerts_per_min']:.4f}",
        f"- Validation time-to-first-detection: {winner['time_to_first_detection_s']:.4f}s",
        "",
        "## Selected Operating Points",
        "",
        "| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['selected_policy']}` | `{row['polarity']}` | {row['persistence_windows']} | "
            f"{row['t_on']:.2f} | {row['t_off']:.2f} | {row['real_only_false_alerts_per_min']:.4f} | "
            f"{row['real_only_false_alert_time_ratio']:.4f} | {row['target_recall']:.4f} | {row['target_f1']:.4f} | "
            f"{row['target_ttfd_s']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Validation Ranking",
            "",
            "| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validation_rows:
        lines.append(
            f"| {row['rank']} | `{row['backend']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['false_alerts_per_min']:.4f} | {row['false_alert_time_ratio']:.4f} | "
            f"{row['time_to_first_detection_s']:.4f} | {row['alert_flicker']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Test Results",
            "",
            "| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| `{row['backend']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['false_alerts_per_min']:.4f} | {row['false_alert_time_ratio']:.4f} | "
            f"{row['time_to_first_detection_s']:.4f} | {row['alert_flicker']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_variant_with_paths(
    config: VariantConfig,
    summary_path: Path,
    sessions_index_path: Path,
    eval_root: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = sb.load_session_index(sessions_index_path, eval_root)
    runs = sb.load_runs(summary_path, session_index, eval_root, window_seconds=config.window_seconds)
    backend_runs: Dict[str, List[sb.RunData]] = defaultdict(list)
    for run in runs:
        backend_runs[run.backend].append(run)

    selected_rows: List[dict] = []
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    thresholds: Dict[str, dict] = {
        "_meta": {
            "variant": config.name,
            "evaluation_unit": f"{config.window_seconds}s_window",
            "window_seconds": config.window_seconds,
            "base_signal_priority": ["score_raw", "score_smoothed_fallback_if_no_raw"],
            "missing_policy": "hold_last_score_for_1_second_then_missing_no_decision",
            "timeline_alignment": "end_aligned_to_video_end_using_pred_t_max_minus_duration",
            "strict_false_alerts_per_min": config.strict_false_alerts_per_min,
            "relaxed_false_alerts_per_min": config.relaxed_false_alerts_per_min,
            "strict_false_alert_time_ratio": config.strict_fa_time_ratio,
            "relaxed_false_alert_time_ratio": config.relaxed_fa_time_ratio,
            "search_space": {
                "thresholds": [round(step / 100.0, 2) for step in range(1, 100)],
                "hysteresis_gaps": [0.0, 0.05, 0.10],
                "persistence_windows": [1, 2, 3],
            },
            "selection_rule": config.selection_rule,
            "validation_ranking_rule": config.validation_ranking_rule,
        }
    }

    for backend in sorted(backend_runs):
        runs_for_backend = backend_runs[backend]
        calibration_runs = [run for run in runs_for_backend if run.split == "calibration"]
        calibration_real_only = [run for run in calibration_runs if run.kind == "real_only"]
        calibration_target = [run for run in calibration_runs if run.kind != "real_only"]
        validation_runs = [run for run in runs_for_backend if run.split == "validation"]
        test_runs = [run for run in runs_for_backend if run.split == "test"]

        polarity_info = sb.determine_backend_polarity(calibration_runs)
        polarity = polarity_info["polarity"]
        signal_source = ",".join(polarity_info["signal_sources"])

        candidate_rows: List[dict] = []
        for op in sb.search_operating_points():
            real_only_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_real_only))
            target_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_target))
            calibration_all_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_runs))
            candidate_rows.append(
                {
                    "op": op,
                    "real_only_metrics": real_only_metrics,
                    "target_metrics": target_metrics,
                    "calibration_all_metrics": calibration_all_metrics,
                }
            )

        strict_best = None
        relaxed_best = None
        if config.selection_mode == "gate":
            strict_best = sb.pick_best_candidate(
                [
                    row
                    for row in candidate_rows
                    if row["real_only_metrics"]["false_alerts_per_min"] <= float(config.strict_false_alerts_per_min)
                    and row["real_only_metrics"]["false_alert_time_ratio"] <= float(config.strict_fa_time_ratio)
                ],
                f"strict_{config.strict_false_alerts_per_min:.2f}",
            )
            relaxed_best = sb.pick_best_candidate(
                [
                    row
                    for row in candidate_rows
                    if row["real_only_metrics"]["false_alerts_per_min"] <= float(config.relaxed_false_alerts_per_min)
                    and row["real_only_metrics"]["false_alert_time_ratio"] <= float(config.relaxed_fa_time_ratio)
                ],
                f"relaxed_{config.relaxed_false_alerts_per_min:.2f}",
            )
            selected = strict_best or relaxed_best or sb.pick_best_available(candidate_rows)
        elif config.selection_mode == "nogate_maxf1":
            selected = pick_best_nogate_maxf1(candidate_rows)
        else:
            raise ValueError(f"Unsupported selection mode: {config.selection_mode}")

        op = selected["op"]
        validation_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, validation_runs))
        test_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, test_runs))

        selected_rows.append(
            sb.round_metrics(
                {
                    "backend": backend,
                    "signal_source": signal_source,
                    "polarity": polarity,
                    "raw_auc_high_means_fake": polarity_info["raw_auc_high_means_fake"],
                    "effective_auc": polarity_info["effective_auc"],
                    "selected_policy": selected["gate_label"],
                    "strict_gate_met": strict_best is not None,
                    "relaxed_gate_met": relaxed_best is not None,
                    "threshold": op.threshold,
                    "t_on": op.t_on,
                    "t_off": op.t_off,
                    "persistence_windows": op.persistence_windows,
                    "real_only_false_alerts_per_min": selected["real_only_metrics"]["false_alerts_per_min"],
                    "real_only_false_alert_time_ratio": selected["real_only_metrics"]["false_alert_time_ratio"],
                    "target_recall": selected["target_metrics"]["recall"],
                    "target_f1": selected["target_metrics"]["f1"],
                    "target_ttfd_s": selected["target_metrics"]["time_to_first_detection_s"],
                    **selected["calibration_all_metrics"],
                }
            )
        )

        validation_rows.append(
            sb.round_metrics(
                {
                    "backend": backend,
                    "signal_source": signal_source,
                    "polarity": polarity,
                    "selected_policy": selected["gate_label"],
                    "strict_gate_met": strict_best is not None,
                    "relaxed_gate_met": relaxed_best is not None,
                    "persistence_windows": op.persistence_windows,
                    "t_on": op.t_on,
                    "t_off": op.t_off,
                    **validation_metrics,
                }
            )
        )
        test_rows.append(
            sb.round_metrics(
                {
                    "backend": backend,
                    "signal_source": signal_source,
                    "polarity": polarity,
                    "selected_policy": selected["gate_label"],
                    "strict_gate_met": strict_best is not None,
                    "relaxed_gate_met": relaxed_best is not None,
                    "persistence_windows": op.persistence_windows,
                    "t_on": op.t_on,
                    "t_off": op.t_off,
                    **test_metrics,
                }
            )
        )

        thresholds[backend] = {
            "signal_source": signal_source,
            "polarity": polarity,
            "raw_auc_high_means_fake": round(polarity_info["raw_auc_high_means_fake"], 6),
            "effective_auc": round(polarity_info["effective_auc"], 6),
            "strict_candidate": None
            if strict_best is None
            else {
                "threshold": strict_best["op"].threshold,
                "t_on": strict_best["op"].t_on,
                "t_off": strict_best["op"].t_off,
                "persistence_windows": strict_best["op"].persistence_windows,
                "real_only_metrics": sb.round_metrics(strict_best["real_only_metrics"]),
                "target_metrics": sb.round_metrics(strict_best["target_metrics"]),
            },
            "relaxed_candidate": None
            if relaxed_best is None
            else {
                "threshold": relaxed_best["op"].threshold,
                "t_on": relaxed_best["op"].t_on,
                "t_off": relaxed_best["op"].t_off,
                "persistence_windows": relaxed_best["op"].persistence_windows,
                "real_only_metrics": sb.round_metrics(relaxed_best["real_only_metrics"]),
                "target_metrics": sb.round_metrics(relaxed_best["target_metrics"]),
            },
            "selected": {
                "policy": selected["gate_label"],
                "threshold": op.threshold,
                "t_on": op.t_on,
                "t_off": op.t_off,
                "persistence_windows": op.persistence_windows,
                "real_only_metrics": sb.round_metrics(selected["real_only_metrics"]),
                "target_metrics": sb.round_metrics(selected["target_metrics"]),
                "calibration_all_metrics": sb.round_metrics(selected["calibration_all_metrics"]),
            },
        }

    ranked_validation = rank_validation_rows(validation_rows, config)
    selected_rows_sorted = sorted(selected_rows, key=lambda row: row["backend"])
    test_rows_sorted = sorted(test_rows, key=lambda row: row["backend"])

    calibration_fieldnames = [
        "backend", "signal_source", "polarity", "raw_auc_high_means_fake", "effective_auc",
        "selected_policy", "strict_gate_met", "relaxed_gate_met",
        "threshold", "t_on", "t_off", "persistence_windows",
        "real_only_false_alerts_per_min", "real_only_false_alert_time_ratio", "target_recall", "target_f1", "target_ttfd_s",
        "precision", "recall", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s", "alert_flicker",
        "tp_windows", "fp_windows", "fn_windows", "tn_windows",
        "predicted_alert_segments", "false_alert_segments", "fake_segments", "detected_fake_segments",
        "real_minutes", "alert_active_minutes", "alert_transitions",
        "missing_windows", "scored_windows", "held_windows", "missing_rate", "run_count",
    ]
    evaluation_fieldnames = [
        "rank", "winner", "backend", "signal_source", "polarity", "selected_policy",
        "strict_gate_met", "relaxed_gate_met", "persistence_windows", "t_on", "t_off",
        "precision", "recall", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s", "alert_flicker",
        "tp_windows", "fp_windows", "fn_windows", "tn_windows",
        "predicted_alert_segments", "false_alert_segments", "fake_segments", "detected_fake_segments",
        "real_minutes", "alert_active_minutes", "alert_transitions",
        "missing_windows", "scored_windows", "held_windows", "missing_rate", "run_count",
    ]

    sb.write_csv(output_dir / "calibration_metrics.csv", selected_rows_sorted, calibration_fieldnames)
    sb.write_csv(output_dir / "validation_metrics.csv", ranked_validation, evaluation_fieldnames)
    sb.write_csv(
        output_dir / "test_metrics.csv",
        test_rows_sorted,
        [field for field in evaluation_fieldnames if field not in {"rank", "winner"}],
    )
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(
        build_summary_report(selected_rows_sorted, ranked_validation, test_rows_sorted, config),
        encoding="utf-8",
    )
    return output_dir


def run_variant(config: VariantConfig) -> Path:
    args = parse_args(config)
    return run_variant_with_paths(
        config=config,
        summary_path=Path(args.summary).resolve(),
        sessions_index_path=Path(args.sessions_index).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
