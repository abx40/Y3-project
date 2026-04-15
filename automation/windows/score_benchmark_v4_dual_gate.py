import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows import score_benchmark_v2 as sb
from automation.windows.score_benchmark_v4_np_calibration import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    BOOTSTRAP_UCB_QUANTILE,
    DEFAULT_WINDOW_SECONDS,
    bootstrap_gate_metrics,
    load_runs_any,
    round_row,
    threshold_free_auc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 Neyman-Pearson calibration with configurable strict and relaxed gates.")
    parser.add_argument("--summary", default="results_rgb_fix_full_2026-04-09/summary_final_clean.csv")
    parser.add_argument("--sessions-index", default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv")
    parser.add_argument("--eval-root", default=r"C:\deepfake_eval\leakfree_eval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-ucb-quantile", type=float, default=BOOTSTRAP_UCB_QUANTILE)
    parser.add_argument("--gate-name", required=True)
    parser.add_argument("--strict-fa-per-min", type=float, required=True)
    parser.add_argument("--strict-fa-time-ratio", type=float, required=True)
    parser.add_argument("--relaxed-fa-per-min", type=float, required=True)
    parser.add_argument("--relaxed-fa-time-ratio", type=float, required=True)
    parser.add_argument("--gate-real-ordinal", type=int, default=1)
    return parser.parse_args()


def gate_real_runs(calibration_runs: Sequence[sb.RunData], ordinal: int) -> List[sb.RunData]:
    runs = [run for run in calibration_runs if run.kind == "real_only" and run.ordinal == ordinal]
    if not runs:
        raise ValueError(f"Missing calibration real_only run with ordinal={ordinal}")
    return runs


def calibration_target_runs(calibration_runs: Sequence[sb.RunData]) -> List[sb.RunData]:
    runs = [run for run in calibration_runs if run.kind != "real_only"]
    if not runs:
        raise ValueError("Missing calibration fake/mixed runs")
    return runs


def gate_passes(candidate: dict, fa_limit: float, time_ratio_limit: float) -> bool:
    return (
        candidate["gate_bootstrap"]["false_alerts_per_min_ucb"] <= fa_limit
        and candidate["gate_bootstrap"]["false_alert_time_ratio_ucb"] <= time_ratio_limit
    )


def candidate_rank(candidate: dict) -> Tuple:
    return (
        -candidate["target_metrics"]["recall"],
        candidate["target_metrics"]["time_to_first_detection_s"],
        -candidate["target_metrics"]["f1"],
        candidate["gate_bootstrap"]["false_alerts_per_min_ucb"],
        candidate["gate_bootstrap"]["false_alert_time_ratio_ucb"],
        candidate["op"].threshold,
        candidate["op"].persistence_windows,
        candidate["op"].hysteresis_gap,
    )


def failed_rank(candidate: dict, fa_limit: float, time_ratio_limit: float) -> Tuple:
    return (
        max(0.0, candidate["gate_bootstrap"]["false_alerts_per_min_ucb"] - fa_limit),
        max(0.0, candidate["gate_bootstrap"]["false_alert_time_ratio_ucb"] - time_ratio_limit),
        -candidate["target_metrics"]["recall"],
        candidate["target_metrics"]["time_to_first_detection_s"],
        -candidate["target_metrics"]["f1"],
        candidate["op"].threshold,
        candidate["op"].persistence_windows,
        candidate["op"].hysteresis_gap,
    )


def build_bootstrap_indices(session_count: int, samples: int, seed: int) -> List[List[int]]:
    if session_count <= 0:
        return []
    if session_count == 1:
        return [[0] for _ in range(samples)]
    rng = random.Random(seed)
    return [[rng.randrange(session_count) for _ in range(session_count)] for _ in range(samples)]


def build_summary_report(
    gate_name: str,
    strict_fa_per_min: float,
    strict_fa_time_ratio: float,
    relaxed_fa_per_min: float,
    relaxed_fa_time_ratio: float,
    window_seconds: int,
    gate_real_ordinal: int,
    selected_rows: Sequence[dict],
    failed_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
) -> str:
    lines = [
        f"# V4 Dual Gate Report: {gate_name}",
        "",
        f"- Window: `{window_seconds}s`",
        f"- Gate real stream: `calibration_real_only_{gate_real_ordinal:02d}`",
        f"- Strict gate: FA/min <= `{strict_fa_per_min:.2f}`, FA time ratio <= `{strict_fa_time_ratio:.2f}`",
        f"- Relaxed fallback: FA/min <= `{relaxed_fa_per_min:.2f}`, FA time ratio <= `{relaxed_fa_time_ratio:.2f}`",
        "- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.",
        "- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.",
        "",
        f"- Gate-compliant backends: `{len(selected_rows)}` / `{len(selected_rows) + len(failed_rows)}`",
        "",
        "## Validation",
        "",
        "| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row.get('rank', '')} | `{row['backend']}` | `{row['calibration_status']}` | `{row.get('selected_policy', '')}` | "
            f"{row['threshold_free_auroc']:.4f} | {row.get('precision', '')} | {row.get('recall', '')} | {row.get('f1', '')} | "
            f"{row.get('false_alerts_per_min', '')} | {row.get('false_alert_time_ratio', '')} | {row.get('time_to_first_detection_s', '')} | {row.get('alert_flicker', '')} |"
        )
    lines.extend(
        [
            "",
            "## Test",
            "",
            "| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['calibration_status']}` | `{row.get('selected_policy', '')}` | "
            f"{row['threshold_free_auroc']:.4f} | {row.get('precision', '')} | {row.get('recall', '')} | {row.get('f1', '')} | "
            f"{row.get('false_alerts_per_min', '')} | {row.get('false_alert_time_ratio', '')} | {row.get('time_to_first_detection_s', '')} | {row.get('alert_flicker', '')} |"
        )
    return "\n".join(lines) + "\n"


def run_gate_variant(
    summary_path: Path,
    sessions_index_path: Path,
    eval_root: Path,
    output_dir: Path,
    window_seconds: int,
    bootstrap_samples: int,
    bootstrap_ucb_quantile: float,
    gate_name: str,
    strict_fa_per_min: float,
    strict_fa_time_ratio: float,
    relaxed_fa_per_min: float,
    relaxed_fa_time_ratio: float,
    gate_real_ordinal: int,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = sb.load_session_index(sessions_index_path, eval_root)
    runs = load_runs_any(summary_path, session_index, eval_root, window_seconds=window_seconds)
    backend_runs: Dict[str, List[sb.RunData]] = defaultdict(list)
    for run in runs:
        backend_runs[run.backend].append(run)

    calibration_rows: List[dict] = []
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    thresholds: Dict[str, dict] = {
        "_meta": {
            "gate_name": gate_name,
            "window_seconds": window_seconds,
            "strict_false_alerts_per_min": strict_fa_per_min,
            "strict_false_alert_time_ratio": strict_fa_time_ratio,
            "relaxed_false_alerts_per_min": relaxed_fa_per_min,
            "relaxed_false_alert_time_ratio": relaxed_fa_time_ratio,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_ucb_quantile": bootstrap_ucb_quantile,
            "gate_real_ordinal": gate_real_ordinal,
        }
    }

    for backend in sorted(backend_runs):
        runs_for_backend = backend_runs[backend]
        calibration_runs = [run for run in runs_for_backend if run.split == "calibration"]
        validation_runs = [run for run in runs_for_backend if run.split == "validation"]
        test_runs = [run for run in runs_for_backend if run.split == "test"]
        polarity_info = sb.determine_backend_polarity(calibration_runs)
        polarity = polarity_info["polarity"]
        gate_runs = gate_real_runs(calibration_runs, gate_real_ordinal)
        target_runs = calibration_target_runs(calibration_runs)
        validation_auc = threshold_free_auc(validation_runs, polarity)
        test_auc = threshold_free_auc(test_runs, polarity)
        calibration_auc = threshold_free_auc(calibration_runs, polarity)

        bootstrap_indices = build_bootstrap_indices(
            session_count=len(gate_runs),
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED + sum(ord(ch) for ch in backend) + window_seconds,
        )

        candidate_rows: List[dict] = []
        for op in sb.search_operating_points():
            gate_run_metrics = [m for m in sb.evaluate_candidate(polarity, op, gate_runs) if m["real_minutes"] > 0]
            target_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, op, target_runs))
            gate_bootstrap = bootstrap_gate_metrics(gate_run_metrics, bootstrap_indices, bootstrap_ucb_quantile)
            candidate_rows.append(
                {
                    "op": op,
                    "gate_metrics": sb.aggregate_metrics(gate_run_metrics) if gate_run_metrics else {
                        "false_alerts_per_min": 0.0,
                        "false_alert_time_ratio": 1.0,
                        "run_count": 0,
                    },
                    "gate_bootstrap": gate_bootstrap,
                    "target_metrics": target_metrics,
                }
            )

        strict_candidates = [row for row in candidate_rows if gate_passes(row, strict_fa_per_min, strict_fa_time_ratio)]
        relaxed_candidates = [row for row in candidate_rows if gate_passes(row, relaxed_fa_per_min, relaxed_fa_time_ratio)]
        if strict_candidates:
            selected = sorted(strict_candidates, key=candidate_rank)[0]
            selected_policy = "strict"
        elif relaxed_candidates:
            selected = sorted(relaxed_candidates, key=candidate_rank)[0]
            selected_policy = "relaxed"
        else:
            selected = None
            selected_policy = ""
        failed = sorted(candidate_rows, key=lambda row: failed_rank(row, relaxed_fa_per_min, relaxed_fa_time_ratio))[0] if selected is None else None
        selected_op = selected["op"] if selected is not None else None
        selected_like = selected if selected is not None else failed
        calibration_status = "selected" if selected is not None else "failed_gate"

        calibration_rows.append(
            round_row(
                {
                    "backend": backend,
                    "signal_source": calibration_runs[0].signal_source,
                    "polarity": polarity,
                    "raw_auc_high_means_fake": polarity_info["raw_auc_high_means_fake"],
                    "effective_auc": polarity_info["effective_auc"],
                    "calibration_auroc": calibration_auc,
                    "validation_auroc": validation_auc,
                    "test_auroc": test_auc,
                    "calibration_status": calibration_status,
                    "selected_policy": selected_policy,
                    "gate_real_session_count": selected_like["gate_bootstrap"]["session_count"] if selected_like else 0,
                    "gate_false_alerts_per_min": selected_like["gate_bootstrap"]["false_alerts_per_min_point"] if selected_like else "",
                    "gate_false_alerts_per_min_ucb": selected_like["gate_bootstrap"]["false_alerts_per_min_ucb"] if selected_like else "",
                    "gate_false_alert_time_ratio": selected_like["gate_bootstrap"]["false_alert_time_ratio_point"] if selected_like else "",
                    "gate_false_alert_time_ratio_ucb": selected_like["gate_bootstrap"]["false_alert_time_ratio_ucb"] if selected_like else "",
                    "target_recall": selected_like["target_metrics"]["recall"] if selected_like else "",
                    "target_ttfd_s": selected_like["target_metrics"]["time_to_first_detection_s"] if selected_like else "",
                    "target_f1": selected_like["target_metrics"]["f1"] if selected_like else "",
                    "target_alert_flicker": selected_like["target_metrics"]["alert_flicker"] if selected_like else "",
                    "threshold": selected_op.threshold if selected_op else "",
                    "t_on": selected_op.t_on if selected_op else "",
                    "t_off": selected_op.t_off if selected_op else "",
                    "persistence_windows": selected_op.persistence_windows if selected_op else "",
                }
            )
        )

        if selected is not None:
            validation_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, selected_op, validation_runs))
            test_metrics = sb.aggregate_metrics(sb.evaluate_candidate(polarity, selected_op, test_runs))
            validation_rows.append(round_row({
                "backend": backend,
                "signal_source": calibration_runs[0].signal_source,
                "polarity": polarity,
                "calibration_status": calibration_status,
                "threshold_free_auroc": validation_auc,
                "selected_policy": selected_policy,
                "threshold": selected_op.threshold,
                "t_on": selected_op.t_on,
                "t_off": selected_op.t_off,
                "persistence_windows": selected_op.persistence_windows,
                **validation_metrics,
            }))
            test_rows.append(round_row({
                "backend": backend,
                "signal_source": calibration_runs[0].signal_source,
                "polarity": polarity,
                "calibration_status": calibration_status,
                "threshold_free_auroc": test_auc,
                "selected_policy": selected_policy,
                "threshold": selected_op.threshold,
                "t_on": selected_op.t_on,
                "t_off": selected_op.t_off,
                "persistence_windows": selected_op.persistence_windows,
                **test_metrics,
            }))
        else:
            validation_rows.append(round_row({
                "backend": backend,
                "signal_source": calibration_runs[0].signal_source,
                "polarity": polarity,
                "calibration_status": calibration_status,
                "threshold_free_auroc": validation_auc,
                "selected_policy": "",
            }))
            test_rows.append(round_row({
                "backend": backend,
                "signal_source": calibration_runs[0].signal_source,
                "polarity": polarity,
                "calibration_status": calibration_status,
                "threshold_free_auroc": test_auc,
                "selected_policy": "",
            }))

        thresholds[backend] = {
            "signal_source": calibration_runs[0].signal_source,
            "polarity": polarity,
            "raw_auc_high_means_fake": round(polarity_info["raw_auc_high_means_fake"], 6),
            "effective_auc": round(polarity_info["effective_auc"], 6),
            "calibration_auroc": round(calibration_auc, 6),
            "validation_auroc": round(validation_auc, 6),
            "test_auroc": round(test_auc, 6),
            "calibration_status": calibration_status,
            "selected": None if selected is None else {
                "policy": selected_policy,
                "threshold": selected_op.threshold,
                "t_on": selected_op.t_on,
                "t_off": selected_op.t_off,
                "persistence_windows": selected_op.persistence_windows,
                "gate_metrics": round_row(selected["gate_metrics"]),
                "gate_bootstrap": round_row(selected["gate_bootstrap"]),
                "target_metrics": round_row(selected["target_metrics"]),
            },
            "best_failed_gate_candidate": None if failed is None else {
                "threshold": failed["op"].threshold,
                "t_on": failed["op"].t_on,
                "t_off": failed["op"].t_off,
                "persistence_windows": failed["op"].persistence_windows,
                "gate_metrics": round_row(failed["gate_metrics"]),
                "gate_bootstrap": round_row(failed["gate_bootstrap"]),
                "target_metrics": round_row(failed["target_metrics"]),
            },
        }

    compliant_validation_rows = [row for row in validation_rows if row["calibration_status"] == "selected"]
    failed_validation_rows = [row for row in validation_rows if row["calibration_status"] != "selected"]
    compliant_validation_rows = sorted(
        compliant_validation_rows,
        key=lambda row: (-float(row["threshold_free_auroc"]), float(row["false_alerts_per_min"]), float(row["time_to_first_detection_s"]), row["backend"]),
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

    calibration_fieldnames = [
        "backend", "signal_source", "polarity", "raw_auc_high_means_fake", "effective_auc",
        "calibration_auroc", "validation_auroc", "test_auroc", "calibration_status", "selected_policy",
        "gate_real_session_count", "gate_false_alerts_per_min", "gate_false_alerts_per_min_ucb",
        "gate_false_alert_time_ratio", "gate_false_alert_time_ratio_ucb",
        "target_recall", "target_ttfd_s", "target_f1", "target_alert_flicker",
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

    sb.write_csv(output_dir / "calibration_metrics.csv", sorted(calibration_rows, key=lambda row: row["backend"]), calibration_fieldnames)
    sb.write_csv(output_dir / "validation_metrics.csv", ranked_validation_rows, evaluation_fieldnames)
    sb.write_csv(output_dir / "test_metrics.csv", sorted(test_rows, key=lambda row: row["backend"]), [field for field in evaluation_fieldnames if field not in {"rank", "winner"}])
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(
        build_summary_report(
            gate_name=gate_name,
            strict_fa_per_min=strict_fa_per_min,
            strict_fa_time_ratio=strict_fa_time_ratio,
            relaxed_fa_per_min=relaxed_fa_per_min,
            relaxed_fa_time_ratio=relaxed_fa_time_ratio,
            window_seconds=window_seconds,
            gate_real_ordinal=gate_real_ordinal,
            selected_rows=[row for row in calibration_rows if row["calibration_status"] == "selected"],
            failed_rows=[row for row in calibration_rows if row["calibration_status"] != "selected"],
            validation_rows=ranked_validation_rows,
            test_rows=sorted(test_rows, key=lambda row: row["backend"]),
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_gate_variant(
        summary_path=Path(args.summary).resolve(),
        sessions_index_path=Path(args.sessions_index).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        window_seconds=args.window_seconds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_ucb_quantile=args.bootstrap_ucb_quantile,
        gate_name=args.gate_name,
        strict_fa_per_min=args.strict_fa_per_min,
        strict_fa_time_ratio=args.strict_fa_time_ratio,
        relaxed_fa_per_min=args.relaxed_fa_per_min,
        relaxed_fa_time_ratio=args.relaxed_fa_time_ratio,
        gate_real_ordinal=args.gate_real_ordinal,
    )


if __name__ == "__main__":
    main()
