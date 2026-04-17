import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.windows import score_benchmark_v2 as sb


STRICT_FPR_REAL = 0.05
RELAXED_FPR_REAL = 0.10
DEFAULT_WINDOW_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuned benchmark rescoring with FPR-based calibration on pooled calibration real windows."
    )
    parser.add_argument(
        "--summary",
        default="results_finetune_clean_progress_2026-04-17/lower_window_sweep_2026-04-17/summary_combined.csv",
    )
    parser.add_argument(
        "--sessions-index",
        default=r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\sessions_index.csv",
    )
    parser.add_argument(
        "--eval-root",
        default=r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4",
    )
    parser.add_argument(
        "--output-dir",
        default="results_finetune_clean_fpr_calibration_15s_2026-04-17",
    )
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--strict-fpr-real", type=float, default=STRICT_FPR_REAL)
    parser.add_argument("--relaxed-fpr-real", type=float, default=RELAXED_FPR_REAL)
    return parser.parse_args()


def load_runs_any(
    summary_path: Path,
    session_index: Dict[Tuple[str, str, int], sb.SessionMeta],
    eval_root: Path,
    window_seconds: int,
) -> List[sb.RunData]:
    rows = [row for row in sb.read_csv_rows(summary_path) if row.get("status", "").strip() == "success"]
    if not rows:
        raise ValueError(f"No successful runs found in {summary_path}")

    runs: List[sb.RunData] = []
    for row in rows:
        split = row["split"].strip()
        kind = row["kind"].strip()
        ordinal = int(row["ordinal"])
        key = (split, kind, ordinal)
        if key not in session_index:
            raise KeyError(f"Missing session metadata for {key} in sessions index")
        meta = session_index[key]
        labels_path = eval_root / "sessions" / split / meta.labels_filename
        pred_rows = sb.load_prediction_rows(Path(row["result_dir"]) / "predictions.csv")
        (
            signal_source,
            signal_windows,
            pred_t_min,
            pred_t_max,
            align_shift_s,
            scored_window_count,
            held_window_count,
            missing_window_count,
        ) = sb.build_signal_windows(pred_rows, meta.duration_s, window_seconds=window_seconds)
        segments = sb.parse_label_segments(labels_path)
        runs.append(
            sb.RunData(
                run_id=row["run_id"],
                backend=row["model"].strip(),
                split=split,
                kind=kind,
                ordinal=ordinal,
                duration_s=meta.duration_s,
                result_dir=Path(row["result_dir"]),
                video_filename=meta.video_filename,
                labels_path=labels_path,
                window_seconds=window_seconds,
                signal_source=signal_source,
                signal_windows=signal_windows,
                gt_windows=sb.build_gt_windows(segments, meta.duration_s, window_seconds=window_seconds),
                fake_segments=sb.build_fake_segments(segments),
                pred_t_min=pred_t_min,
                pred_t_max=pred_t_max,
                align_shift_s=align_shift_s,
                scored_window_count=scored_window_count,
                held_window_count=held_window_count,
                missing_window_count=missing_window_count,
            )
        )
    return runs


def runs_with_real_labels(runs: Sequence[sb.RunData]) -> List[sb.RunData]:
    return [run for run in runs if any(label == 0 for label in run.gt_windows)]


def runs_with_fake_labels(runs: Sequence[sb.RunData]) -> List[sb.RunData]:
    return [run for run in runs if any(label == 1 for label in run.gt_windows)]


def collect_oriented_scores_labels(runs: Sequence[sb.RunData], polarity: str) -> Tuple[List[float], List[int]]:
    scores: List[float] = []
    labels: List[int] = []
    for run in runs:
        for score, label in zip(run.signal_windows, run.gt_windows):
            oriented = sb.orient_score(score, polarity)
            if oriented is None:
                continue
            scores.append(oriented)
            labels.append(label)
    return scores, labels


def threshold_free_auc(runs: Sequence[sb.RunData], polarity: str) -> float:
    scores, labels = collect_oriented_scores_labels(runs, polarity)
    return sb.auroc(scores, labels)


def real_window_fpr(metrics: dict) -> float:
    fp = float(metrics.get("fp_windows", 0))
    tn = float(metrics.get("tn_windows", 0))
    denom = fp + tn
    if denom <= 0:
        return 1.0
    return fp / denom


def candidate_rank(candidate: dict) -> Tuple:
    op = candidate["op"]
    return (
        -candidate["target_metrics"]["recall"],
        candidate["target_metrics"]["time_to_first_detection_s"],
        candidate["target_metrics"]["alert_flicker"],
        candidate["gate_fpr_real"],
        candidate["gate_metrics"]["false_alerts_per_min"],
        op.threshold,
        op.persistence_windows,
        op.hysteresis_gap,
    )


def failed_candidate_rank(candidate: dict) -> Tuple:
    op = candidate["op"]
    return (
        candidate["gate_fpr_real"],
        -candidate["target_metrics"]["recall"],
        candidate["target_metrics"]["time_to_first_detection_s"],
        candidate["target_metrics"]["alert_flicker"],
        candidate["gate_metrics"]["false_alerts_per_min"],
        op.threshold,
        op.persistence_windows,
        op.hysteresis_gap,
    )


def round_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value


def round_row(row: dict) -> dict:
    return {key: round_value(value) for key, value in row.items()}


def build_summary_report(
    selected_rows: Sequence[dict],
    failed_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
    window_seconds: int,
    strict_fpr_real: float,
    relaxed_fpr_real: float,
) -> str:
    lines = [
        "# Fine-Tuned Benchmark Scoring Report V4 FPR Calibration",
        "",
        "## Method",
        "",
        f"- Evaluation unit: fixed {window_seconds}-second window.",
        "- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.",
        "- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.",
        "- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.",
        "- Calibration is FPR-based: the primary calibration constraint is real-window FPR on pooled real-labelled calibration windows after post-processing.",
        f"- Strict FPR target: real-window FPR <= {strict_fpr_real:.2f}.",
        f"- Relaxed fallback FPR target: real-window FPR <= {relaxed_fpr_real:.2f}.",
        "- Candidate selection among FPR-compliant settings: maximize recall on fake-labelled calibration windows, then lower TTFD, then lower flicker.",
        "- FA/min and alert-time ratio are still reported as operational metrics, but they are not the main selection rule.",
        "- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.",
        "",
        "## Calibration Outcome",
        "",
        f"- Gate-compliant backends: {len(selected_rows)} / {len(selected_rows) + len(failed_rows)}",
        "",
        "## Selected Operating Points",
        "",
        "| backend | policy | polarity | t_on | t_off | persistence | calib FPR_real | calib recall | calib TTFD s | calib flicker | calib FA/min | calib alert-time ratio | val AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['selected_policy']}` | `{row['polarity']}` | "
            f"{row['t_on']:.2f} | {row['t_off']:.2f} | {int(row['persistence_windows'])} | "
            f"{row['calibration_fpr_real']:.4f} | {row['calibration_recall']:.4f} | "
            f"{row['calibration_ttfd_s']:.4f} | {row['calibration_flicker']:.4f} | "
            f"{row['calibration_false_alerts_per_min']:.4f} | {row['calibration_false_alert_time_ratio']:.4f} | "
            f"{row['validation_auroc']:.4f} |"
        )

    if failed_rows:
        lines.extend(
            [
                "",
                "## Non-Compliant Backends",
                "",
                "| backend | polarity | closest calib FPR_real | closest calib recall | closest calib TTFD s | val AUROC | test AUROC |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in failed_rows:
            lines.append(
                f"| `{row['backend']}` | `{row['polarity']}` | {row['calibration_fpr_real']:.4f} | "
                f"{row['calibration_recall']:.4f} | {row['calibration_ttfd_s']:.4f} | "
                f"{row['validation_auroc']:.4f} | {row['test_auroc']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| rank | backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validation_rows:
        if row["calibration_status"] == "selected":
            lines.append(
                f"| {row['rank']} | `{row['backend']}` | `{row['calibration_status']}` | `{row['selected_policy']}` | "
                f"{row['threshold_free_auroc']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['false_alerts_per_min']:.4f} | {row['false_alert_time_ratio']:.4f} | "
                f"{row['time_to_first_detection_s']:.4f} | {row['alert_flicker']:.4f} |"
            )
        else:
            lines.append(
                f"|  | `{row['backend']}` | `{row['calibration_status']}` | `` | {row['threshold_free_auroc']:.4f} |  |  |  |  |  |  |"
            )

    lines.extend(
        [
            "",
            "## Test",
            "",
            "| backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        if row["calibration_status"] == "selected":
            lines.append(
                f"| `{row['backend']}` | `{row['calibration_status']}` | `{row['selected_policy']}` | "
                f"{row['threshold_free_auroc']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['false_alerts_per_min']:.4f} | {row['false_alert_time_ratio']:.4f} | "
                f"{row['time_to_first_detection_s']:.4f} | {row['alert_flicker']:.4f} |"
            )
        else:
            lines.append(
                f"| `{row['backend']}` | `{row['calibration_status']}` | `` | {row['threshold_free_auroc']:.4f} |  |  |  |  |  |  |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be >= 1")
    if not (0.0 <= args.strict_fpr_real <= 1.0):
        raise ValueError("--strict-fpr-real must be between 0 and 1")
    if not (0.0 <= args.relaxed_fpr_real <= 1.0):
        raise ValueError("--relaxed-fpr-real must be between 0 and 1")
    if args.relaxed_fpr_real < args.strict_fpr_real:
        raise ValueError("--relaxed-fpr-real must be >= --strict-fpr-real")

    summary_path = Path(args.summary).resolve()
    eval_root = Path(args.eval_root).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = sb.load_session_index(sessions_index_path, eval_root)
    runs = load_runs_any(summary_path, session_index, eval_root, window_seconds=args.window_seconds)
    backend_runs: Dict[str, List[sb.RunData]] = defaultdict(list)
    for run in runs:
        backend_runs[run.backend].append(run)

    calibration_rows: List[dict] = []
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    distribution_rows: List[dict] = []
    alignment_rows: List[dict] = []
    thresholds: Dict[str, dict] = {
        "_meta": {
            "evaluation_unit": f"{args.window_seconds}s_window",
            "window_seconds": args.window_seconds,
            "selection_rule": "FPR-based calibration on pooled calibration real windows, maximize recall on fake calibration windows subject to real-window FPR target.",
            "strict_fpr_real": args.strict_fpr_real,
            "relaxed_fpr_real": args.relaxed_fpr_real,
            "search_space": {
                "thresholds": [round(step / 100.0, 2) for step in range(1, 100)],
                "hysteresis_gaps": [0.0, 0.05, 0.10],
                "persistence_windows": [1, 2, 3],
            },
            "uncertainty_note": "No bootstrap UCB gate is applied in this FPR correction pass; selection uses pooled real-window FPR point estimates on calibration only.",
        }
    }

    for backend in sorted(backend_runs):
        runs_for_backend = backend_runs[backend]
        calibration_runs = [run for run in runs_for_backend if run.split == "calibration"]
        calibration_gate_runs = runs_with_real_labels(calibration_runs)
        calibration_target_runs = runs_with_fake_labels(calibration_runs)
        validation_runs = [run for run in runs_for_backend if run.split == "validation"]
        test_runs = [run for run in runs_for_backend if run.split == "test"]

        polarity_info = sb.determine_backend_polarity(calibration_runs)
        polarity = polarity_info["polarity"]
        signal_source = ",".join(polarity_info["signal_sources"])

        calibration_scores: List[float] = []
        calibration_labels: List[int] = []
        all_scores: List[float] = []
        all_labels: List[int] = []
        for run in calibration_runs:
            for score, label in zip(run.signal_windows, run.gt_windows):
                if score is not None:
                    calibration_scores.append(score)
                    calibration_labels.append(label)
        for run in runs_for_backend:
            for score, label in zip(run.signal_windows, run.gt_windows):
                if score is not None:
                    all_scores.append(score)
                    all_labels.append(label)
        distribution_rows.append(sb.distribution_row(backend, "calibration", polarity, signal_source, calibration_scores, calibration_labels))
        distribution_rows.append(sb.distribution_row(backend, "all", polarity, signal_source, all_scores, all_labels))

        for run in sorted(runs_for_backend, key=lambda item: (item.split, item.kind, item.ordinal)):
            alignment_rows.append(
                {
                    "run_id": run.run_id,
                    "backend": run.backend,
                    "split": run.split,
                    "kind": run.kind,
                    "ordinal": run.ordinal,
                    "video_filename": run.video_filename,
                    "video_duration_s": run.duration_s,
                    "signal_source": run.signal_source,
                    "pred_t_min": round(run.pred_t_min, 6),
                    "pred_t_max": round(run.pred_t_max, 6),
                    "end_align_shift_s": round(run.align_shift_s, 6),
                    "scored_window_count": run.scored_window_count,
                    "held_window_count": run.held_window_count,
                    "missing_window_count": run.missing_window_count,
                }
            )

        calibration_auc = threshold_free_auc(calibration_runs, polarity)
        validation_auc = threshold_free_auc(validation_runs, polarity)
        test_auc = threshold_free_auc(test_runs, polarity)

        candidate_rows: List[dict] = []
        for op in sb.search_operating_points():
            gate_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_gate_runs))
            target_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_target_runs))
            calibration_all_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_runs))
            candidate_rows.append(
                {
                    "op": op,
                    "gate_metrics": gate_metrics,
                    "gate_fpr_real": real_window_fpr(gate_metrics),
                    "target_metrics": target_metrics,
                    "calibration_all_metrics": calibration_all_metrics,
                }
            )

        strict_candidates = [row for row in candidate_rows if row["gate_fpr_real"] <= args.strict_fpr_real]
        relaxed_candidates = [row for row in candidate_rows if row["gate_fpr_real"] <= args.relaxed_fpr_real]

        if strict_candidates:
            selected = sorted(strict_candidates, key=candidate_rank)[0]
            selected_policy = "strict_fpr_gate"
        elif relaxed_candidates:
            selected = sorted(relaxed_candidates, key=candidate_rank)[0]
            selected_policy = "relaxed_fpr_gate"
        else:
            selected = None
            selected_policy = ""

        failed = sorted(candidate_rows, key=failed_candidate_rank)[0] if candidate_rows and selected is None else None
        selected_op = selected["op"] if selected is not None else None
        calibration_status = "selected" if selected is not None else "failed_gate"
        selected_like = selected if selected is not None else failed

        calibration_row = {
            "backend": backend,
            "signal_source": signal_source,
            "polarity": polarity,
            "raw_auc_high_means_fake": polarity_info["raw_auc_high_means_fake"],
            "effective_auc": polarity_info["effective_auc"],
            "calibration_auroc": calibration_auc,
            "validation_auroc": validation_auc,
            "test_auroc": test_auc,
            "calibration_status": calibration_status,
            "selected_policy": selected_policy if selected is not None else "failed",
            "calibration_fpr_real": selected_like["gate_fpr_real"] if selected_like else "",
            "calibration_recall": selected_like["target_metrics"]["recall"] if selected_like else "",
            "calibration_ttfd_s": selected_like["target_metrics"]["time_to_first_detection_s"] if selected_like else "",
            "calibration_flicker": selected_like["target_metrics"]["alert_flicker"] if selected_like else "",
            "calibration_false_alerts_per_min": selected_like["gate_metrics"]["false_alerts_per_min"] if selected_like else "",
            "calibration_false_alert_time_ratio": selected_like["gate_metrics"]["false_alert_time_ratio"] if selected_like else "",
            "threshold": selected_op.threshold if selected_op else "",
            "t_on": selected_op.t_on if selected_op else "",
            "t_off": selected_op.t_off if selected_op else "",
            "persistence_windows": selected_op.persistence_windows if selected_op else "",
        }
        calibration_rows.append(round_row(calibration_row))

        if selected is not None:
            validation_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, selected_op, validation_runs))
            test_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, selected_op, test_runs))
            validation_rows.append(
                round_row(
                    {
                        "backend": backend,
                        "signal_source": signal_source,
                        "polarity": polarity,
                        "calibration_status": calibration_status,
                        "threshold_free_auroc": validation_auc,
                        "selected_policy": selected_policy,
                        "threshold": selected_op.threshold,
                        "t_on": selected_op.t_on,
                        "t_off": selected_op.t_off,
                        "persistence_windows": selected_op.persistence_windows,
                        **validation_metrics,
                    }
                )
            )
            test_rows.append(
                round_row(
                    {
                        "backend": backend,
                        "signal_source": signal_source,
                        "polarity": polarity,
                        "calibration_status": calibration_status,
                        "threshold_free_auroc": test_auc,
                        "selected_policy": selected_policy,
                        "threshold": selected_op.threshold,
                        "t_on": selected_op.t_on,
                        "t_off": selected_op.t_off,
                        "persistence_windows": selected_op.persistence_windows,
                        **test_metrics,
                    }
                )
            )
        else:
            validation_rows.append(
                round_row(
                    {
                        "backend": backend,
                        "signal_source": signal_source,
                        "polarity": polarity,
                        "calibration_status": calibration_status,
                        "threshold_free_auroc": validation_auc,
                        "selected_policy": "",
                    }
                )
            )
            test_rows.append(
                round_row(
                    {
                        "backend": backend,
                        "signal_source": signal_source,
                        "polarity": polarity,
                        "calibration_status": calibration_status,
                        "threshold_free_auroc": test_auc,
                        "selected_policy": "",
                    }
                )
            )

        thresholds[backend] = {
            "signal_source": signal_source,
            "polarity": polarity,
            "raw_auc_high_means_fake": round(polarity_info["raw_auc_high_means_fake"], 6),
            "effective_auc": round(polarity_info["effective_auc"], 6),
            "calibration_auroc": round(calibration_auc, 6),
            "validation_auroc": round(validation_auc, 6),
            "test_auroc": round(test_auc, 6),
            "calibration_status": calibration_status,
            "selected_policy": selected_policy if selected is not None else "failed",
            "selected": None
            if selected is None
            else {
                "policy": selected_policy,
                "threshold": selected_op.threshold,
                "t_on": selected_op.t_on,
                "t_off": selected_op.t_off,
                "persistence_windows": selected_op.persistence_windows,
                "gate_fpr_real": round(selected["gate_fpr_real"], 6),
                "gate_metrics": round_row(selected["gate_metrics"]),
                "target_metrics": round_row(selected["target_metrics"]),
                "calibration_all_metrics": round_row(selected["calibration_all_metrics"]),
            },
            "best_failed_gate_candidate": None
            if failed is None
            else {
                "threshold": failed["op"].threshold,
                "t_on": failed["op"].t_on,
                "t_off": failed["op"].t_off,
                "persistence_windows": failed["op"].persistence_windows,
                "gate_fpr_real": round(failed["gate_fpr_real"], 6),
                "gate_metrics": round_row(failed["gate_metrics"]),
                "target_metrics": round_row(failed["target_metrics"]),
            },
        }

    selected_rows = [row for row in calibration_rows if row["calibration_status"] == "selected"]
    failed_rows = [row for row in calibration_rows if row["calibration_status"] != "selected"]

    compliant_validation_rows = [row for row in validation_rows if row["calibration_status"] == "selected"]
    failed_validation_rows = [row for row in validation_rows if row["calibration_status"] != "selected"]
    compliant_validation_rows = sorted(
        compliant_validation_rows,
        key=lambda row: (
            -float(row["threshold_free_auroc"]),
            float(row["false_alerts_per_min"]),
            float(row["time_to_first_detection_s"]),
            -float(row["recall"]),
            row["backend"],
        ),
    )
    ranked_validation_rows: List[dict] = []
    for idx, row in enumerate(compliant_validation_rows, start=1):
        updated = dict(row)
        updated["rank"] = idx
        updated["winner"] = idx == 1
        ranked_validation_rows.append(updated)
    for row in failed_validation_rows:
        updated = dict(row)
        updated["rank"] = ""
        updated["winner"] = False
        ranked_validation_rows.append(updated)

    selected_rows_sorted = sorted(selected_rows, key=lambda row: row["backend"])
    failed_rows_sorted = sorted(failed_rows, key=lambda row: row["backend"])
    test_rows_sorted = sorted(test_rows, key=lambda row: row["backend"])

    calibration_fieldnames = [
        "backend",
        "signal_source",
        "polarity",
        "raw_auc_high_means_fake",
        "effective_auc",
        "calibration_auroc",
        "validation_auroc",
        "test_auroc",
        "calibration_status",
        "selected_policy",
        "calibration_fpr_real",
        "calibration_recall",
        "calibration_ttfd_s",
        "calibration_flicker",
        "calibration_false_alerts_per_min",
        "calibration_false_alert_time_ratio",
        "threshold",
        "t_on",
        "t_off",
        "persistence_windows",
    ]
    evaluation_fieldnames = [
        "rank", "winner", "backend", "signal_source", "polarity", "calibration_status",
        "threshold_free_auroc", "selected_policy", "threshold", "t_on", "t_off", "persistence_windows",
        "precision", "recall", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s", "alert_flicker",
        "tp_windows", "fp_windows", "fn_windows", "tn_windows",
        "predicted_alert_segments", "false_alert_segments", "fake_segments", "detected_fake_segments",
        "real_minutes", "alert_active_minutes", "alert_transitions",
        "missing_windows", "scored_windows", "held_windows", "missing_rate", "run_count",
    ]
    distribution_fieldnames = [
        "backend", "scope", "signal_source", "polarity", "auc_high_means_fake", "effective_auc",
        "real_count", "fake_count",
        "real_mean", "real_std", "real_p05", "real_p25", "real_p50", "real_p75", "real_p95",
        "fake_mean", "fake_std", "fake_p05", "fake_p25", "fake_p50", "fake_p75", "fake_p95",
        "mean_gap_fake_minus_real",
    ]
    alignment_fieldnames = [
        "run_id", "backend", "split", "kind", "ordinal", "video_filename", "video_duration_s",
        "signal_source", "pred_t_min", "pred_t_max", "end_align_shift_s",
        "scored_window_count", "held_window_count", "missing_window_count",
    ]

    sb.write_csv(output_dir / "calibration_metrics.csv", selected_rows_sorted + failed_rows_sorted, calibration_fieldnames)
    sb.write_csv(output_dir / "validation_metrics.csv", ranked_validation_rows, evaluation_fieldnames)
    sb.write_csv(
        output_dir / "test_metrics.csv",
        test_rows_sorted,
        [field for field in evaluation_fieldnames if field not in {"rank", "winner"}],
    )
    sb.write_csv(output_dir / "score_distribution_summary.csv", distribution_rows, distribution_fieldnames)
    sb.write_csv(output_dir / "alignment_audit.csv", alignment_rows, alignment_fieldnames)
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(
        build_summary_report(
            selected_rows=selected_rows_sorted,
            failed_rows=failed_rows_sorted,
            validation_rows=ranked_validation_rows,
            test_rows=test_rows_sorted,
            window_seconds=args.window_seconds,
            strict_fpr_real=args.strict_fpr_real,
            relaxed_fpr_real=args.relaxed_fpr_real,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
