import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows import score_benchmark_v2 as sb


STRICT_FALSE_ALERTS_PER_MIN = 0.10
STRICT_FA_TIME_RATIO = 0.05
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_UCB_QUANTILE = 0.95
DEFAULT_WINDOW_SECONDS = 5
BOOTSTRAP_SEED = 20260414


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neyman-Pearson style live benchmark rescoring with pooled calibration gating.")
    parser.add_argument("--summary", default="results_rgb_fix_full_2026-04-09/summary_final_clean.csv")
    parser.add_argument("--sessions-index", default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv")
    parser.add_argument("--eval-root", default=r"C:\deepfake_eval\leakfree_eval")
    parser.add_argument("--output-dir", default="results_scored_v4_np_calibration_5s_2026-04-14")
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-ucb-quantile", type=float, default=BOOTSTRAP_UCB_QUANTILE)
    return parser.parse_args()


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


def generate_bootstrap_index_sets(session_count: int, samples: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(session_count) for _ in range(session_count)] for _ in range(samples)]


def bootstrap_gate_metrics(
    gate_run_metrics: Sequence[dict],
    bootstrap_indices: Sequence[Sequence[int]],
    quantile: float,
) -> dict:
    if not gate_run_metrics:
        return {
            "session_count": 0,
            "false_alerts_per_min_point": 0.0,
            "false_alert_time_ratio_point": 1.0,
            "false_alerts_per_min_ucb": float("inf"),
            "false_alert_time_ratio_ucb": float("inf"),
        }

    point_metrics = sb.aggregate_metrics(gate_run_metrics)
    segment_counts = [float(metrics["false_alert_segments"]) for metrics in gate_run_metrics]
    real_minutes = [float(metrics["real_minutes"]) for metrics in gate_run_metrics]
    fp_real_seconds = [float(metrics["fp_real_seconds"]) for metrics in gate_run_metrics]
    tn_real_seconds = [float(metrics["tn_real_seconds"]) for metrics in gate_run_metrics]

    fa_per_min_samples: List[float] = []
    fa_time_ratio_samples: List[float] = []
    for replicate in bootstrap_indices:
        segments_sum = 0.0
        real_minutes_sum = 0.0
        fp_real_sum = 0.0
        tn_real_sum = 0.0
        for idx in replicate:
            segments_sum += segment_counts[idx]
            real_minutes_sum += real_minutes[idx]
            fp_real_sum += fp_real_seconds[idx]
            tn_real_sum += tn_real_seconds[idx]
        fa_per_min_samples.append(segments_sum / real_minutes_sum if real_minutes_sum > 0 else 0.0)
        denom = fp_real_sum + tn_real_sum
        fa_time_ratio_samples.append(fp_real_sum / denom if denom > 0 else 1.0)

    return {
        "session_count": len(gate_run_metrics),
        "false_alerts_per_min_point": point_metrics["false_alerts_per_min"],
        "false_alert_time_ratio_point": point_metrics["false_alert_time_ratio"],
        "false_alerts_per_min_ucb": sb.percentile(fa_per_min_samples, quantile),
        "false_alert_time_ratio_ucb": sb.percentile(fa_time_ratio_samples, quantile),
    }


def rank_candidate(candidate: dict) -> Tuple:
    return (
        -candidate["target_metrics"]["recall"],
        candidate["target_metrics"]["time_to_first_detection_s"],
        candidate["target_metrics"]["alert_flicker"],
        candidate["gate_bootstrap"]["false_alerts_per_min_ucb"],
        candidate["gate_bootstrap"]["false_alert_time_ratio_ucb"],
        candidate["op"].threshold,
        candidate["op"].persistence_windows,
        candidate["op"].hysteresis_gap,
    )


def best_failed_candidate(candidate_rows: Sequence[dict]) -> Optional[dict]:
    if not candidate_rows:
        return None
    ranked = sorted(
        candidate_rows,
        key=lambda item: (
            item["gate_bootstrap"]["false_alerts_per_min_ucb"],
            item["gate_bootstrap"]["false_alert_time_ratio_ucb"],
            -item["target_metrics"]["recall"],
            item["target_metrics"]["time_to_first_detection_s"],
            item["target_metrics"]["alert_flicker"],
            item["op"].threshold,
            item["op"].persistence_windows,
            item["op"].hysteresis_gap,
        ),
    )
    return ranked[0]


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
    bootstrap_samples: int,
    bootstrap_quantile: float,
) -> str:
    lines = [
        "# Benchmark Scoring Report V4 NP Calibration",
        "",
        "## Method",
        "",
        f"- Evaluation unit: fixed {window_seconds}-second window.",
        "- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.",
        "- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.",
        "- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.",
        "- Calibration gate: pooled real-labelled windows from all calibration sessions with real content, not just `calibration_real_only_01`.",
        (
            f"- Gate uncertainty: {bootstrap_samples} session bootstrap replicates; candidates pass only if the "
            f"{int(bootstrap_quantile * 100)}th percentile upper bound stays within <= {STRICT_FALSE_ALERTS_PER_MIN:.2f} FA/min "
            f"and <= {STRICT_FA_TIME_RATIO:.2f} false-alert time ratio."
        ),
        "- Calibration objective: among gate-passing candidates, maximize recall on calibration runs containing fake content, then minimize TTFD, then minimize flicker.",
        "- Calibration does not optimize F1 and does not fall back to a 'best available' threshold when the gate fails.",
        "- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.",
        "",
        "## Calibration Outcome",
        "",
        f"- Gate-compliant backends: {len(selected_rows)} / {len(selected_rows) + len(failed_rows)}",
    ]

    if selected_rows:
        lines.extend(
            [
                "",
                "## Selected Operating Points",
                "",
                "| backend | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected_rows:
            lines.append(
                f"| `{row['backend']}` | `{row['polarity']}` | {row['t_on']:.2f} | {row['t_off']:.2f} | {row['persistence_windows']} | "
                f"{row['gate_false_alerts_per_min']:.4f} | {row['gate_false_alerts_per_min_ucb']:.4f} | "
                f"{row['gate_false_alert_time_ratio']:.4f} | {row['gate_false_alert_time_ratio_ucb']:.4f} | "
                f"{row['target_recall']:.4f} | {row['target_ttfd_s']:.4f} | {row['target_alert_flicker']:.4f} | {row['validation_auroc']:.4f} |"
            )

    if failed_rows:
        lines.extend(
            [
                "",
                "## Non-Compliant Backends",
                "",
                "| backend | polarity | closest gate FA/min UCB | closest gate time ratio UCB | closest target recall | val AUROC | test AUROC |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in failed_rows:
            lines.append(
                f"| `{row['backend']}` | `{row['polarity']}` | {row['gate_false_alerts_per_min_ucb']:.4f} | "
                f"{row['gate_false_alert_time_ratio_ucb']:.4f} | {row['target_recall']:.4f} | {row['validation_auroc']:.4f} | {row['test_auroc']:.4f} |"
            )

    if validation_rows:
        lines.extend(
            [
                "",
                "## Validation",
                "",
                "| rank | backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in validation_rows:
            rank = row.get("rank", "")
            lines.append(
                f"| {rank} | `{row['backend']}` | `{row['calibration_status']}` | {row['threshold_free_auroc']:.4f} | "
                f"{row.get('recall', 0.0) if row.get('recall', '') != '' else ''} | "
                f"{row.get('f1', 0.0) if row.get('f1', '') != '' else ''} | "
                f"{row.get('false_alerts_per_min', 0.0) if row.get('false_alerts_per_min', '') != '' else ''} | "
                f"{row.get('false_alert_time_ratio', 0.0) if row.get('false_alert_time_ratio', '') != '' else ''} | "
                f"{row.get('time_to_first_detection_s', 0.0) if row.get('time_to_first_detection_s', '') != '' else ''} | "
                f"{row.get('alert_flicker', 0.0) if row.get('alert_flicker', '') != '' else ''} |"
            )

    if test_rows:
        lines.extend(
            [
                "",
                "## Test",
                "",
                "| backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in test_rows:
            lines.append(
                f"| `{row['backend']}` | `{row['calibration_status']}` | {row['threshold_free_auroc']:.4f} | "
                f"{row.get('recall', 0.0) if row.get('recall', '') != '' else ''} | "
                f"{row.get('f1', 0.0) if row.get('f1', '') != '' else ''} | "
                f"{row.get('false_alerts_per_min', 0.0) if row.get('false_alerts_per_min', '') != '' else ''} | "
                f"{row.get('false_alert_time_ratio', 0.0) if row.get('false_alert_time_ratio', '') != '' else ''} | "
                f"{row.get('time_to_first_detection_s', 0.0) if row.get('time_to_first_detection_s', '') != '' else ''} | "
                f"{row.get('alert_flicker', 0.0) if row.get('alert_flicker', '') != '' else ''} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be >= 1")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be >= 1")
    if not (0.5 < args.bootstrap_ucb_quantile < 1.0):
        raise ValueError("--bootstrap-ucb-quantile must be between 0.5 and 1.0")

    summary_path = Path(args.summary).resolve()
    eval_root = Path(args.eval_root).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = sb.load_session_index(sessions_index_path, eval_root)
    runs = sb.load_runs(summary_path, session_index, eval_root, window_seconds=args.window_seconds)
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
            "selection_rule": "Neyman-Pearson style calibration: pooled calibration real windows gate with bootstrap UCB, then maximize recall on calibration fake windows, then minimize TTFD, then flicker.",
            "strict_false_alerts_per_min": STRICT_FALSE_ALERTS_PER_MIN,
            "strict_false_alert_time_ratio": STRICT_FA_TIME_RATIO,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_ucb_quantile": args.bootstrap_ucb_quantile,
            "search_space": {
                "thresholds": [round(step / 100.0, 2) for step in range(1, 100)],
                "hysteresis_gaps": [0.0, 0.05, 0.10],
                "persistence_windows": [1, 2, 3],
            },
            "no_fallback_when_gate_fails": True,
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
        bootstrap_indices = generate_bootstrap_index_sets(
            session_count=len(calibration_gate_runs),
            samples=args.bootstrap_samples,
            seed=BOOTSTRAP_SEED + sum(ord(ch) for ch in backend),
        )

        candidate_rows: List[dict] = []
        for op in sb.search_operating_points():
            gate_run_metrics = [metrics for metrics in sb.evaluate_candidate(polarity, op, calibration_gate_runs) if metrics["real_minutes"] > 0]
            target_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_target_runs))
            calibration_all_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, calibration_runs))
            gate_bootstrap = bootstrap_gate_metrics(gate_run_metrics, bootstrap_indices, args.bootstrap_ucb_quantile)
            candidate_rows.append(
                {
                    "op": op,
                    "gate_metrics": sb.aggregate_metrics(gate_run_metrics) if gate_run_metrics else {
                        "false_alerts_per_min": 0.0,
                        "false_alert_time_ratio": 1.0,
                        "real_minutes": 0.0,
                        "run_count": 0,
                    },
                    "gate_bootstrap": gate_bootstrap,
                    "target_metrics": target_metrics,
                    "calibration_all_metrics": calibration_all_metrics,
                }
            )

        feasible_candidates = [
            row
            for row in candidate_rows
            if row["gate_bootstrap"]["false_alerts_per_min_ucb"] <= STRICT_FALSE_ALERTS_PER_MIN
            and row["gate_bootstrap"]["false_alert_time_ratio_ucb"] <= STRICT_FA_TIME_RATIO
        ]
        selected = sorted(feasible_candidates, key=rank_candidate)[0] if feasible_candidates else None
        failed = best_failed_candidate(candidate_rows) if selected is None else None
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
            "gate_real_session_count": selected_like["gate_bootstrap"]["session_count"] if selected_like else 0,
            "gate_false_alerts_per_min": selected_like["gate_bootstrap"]["false_alerts_per_min_point"] if selected_like else "",
            "gate_false_alerts_per_min_ucb": selected_like["gate_bootstrap"]["false_alerts_per_min_ucb"] if selected_like else "",
            "gate_false_alert_time_ratio": selected_like["gate_bootstrap"]["false_alert_time_ratio_point"] if selected_like else "",
            "gate_false_alert_time_ratio_ucb": selected_like["gate_bootstrap"]["false_alert_time_ratio_ucb"] if selected_like else "",
            "target_recall": selected_like["target_metrics"]["recall"] if selected_like else "",
            "target_ttfd_s": selected_like["target_metrics"]["time_to_first_detection_s"] if selected_like else "",
            "target_alert_flicker": selected_like["target_metrics"]["alert_flicker"] if selected_like else "",
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
                        "selected_policy": "bootstrap_ucb_gate",
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
                        "selected_policy": "bootstrap_ucb_gate",
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
            "selected": None
            if selected is None
            else {
                "policy": "bootstrap_ucb_gate",
                "threshold": selected_op.threshold,
                "t_on": selected_op.t_on,
                "t_off": selected_op.t_off,
                "persistence_windows": selected_op.persistence_windows,
                "gate_metrics": round_row(selected["gate_metrics"]),
                "gate_bootstrap": round_row(selected["gate_bootstrap"]),
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
                "gate_metrics": round_row(failed["gate_metrics"]),
                "gate_bootstrap": round_row(failed["gate_bootstrap"]),
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
            float(row["alert_flicker"]),
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
        "backend", "signal_source", "polarity", "raw_auc_high_means_fake", "effective_auc",
        "calibration_auroc", "validation_auroc", "test_auroc", "calibration_status",
        "gate_real_session_count", "gate_false_alerts_per_min", "gate_false_alerts_per_min_ucb",
        "gate_false_alert_time_ratio", "gate_false_alert_time_ratio_ucb",
        "target_recall", "target_ttfd_s", "target_alert_flicker",
        "threshold", "t_on", "t_off", "persistence_windows",
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
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_quantile=args.bootstrap_ucb_quantile,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
