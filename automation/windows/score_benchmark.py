import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LabelSegment:
    start_s: float
    end_s: float
    label: int
    raw_label: str


@dataclass(frozen=True)
class SessionMeta:
    split: str
    kind: str
    ordinal: int
    video_filename: str
    labels_filename: str
    duration_s: int


@dataclass
class RunData:
    run_id: str
    backend: str
    split: str
    kind: str
    ordinal: int
    duration_s: int
    result_dir: Path
    video_filename: str
    labels_path: Path
    window_scores: List[Optional[float]]
    gt_windows: List[int]
    fake_segments: List[Tuple[float, float]]
    pred_t_min: float
    pred_t_max: float
    align_shift_s: float
    aligned_t_min: Optional[float]
    aligned_t_max: Optional[float]
    scored_window_count: int


@dataclass(frozen=True)
class CandidateConfig:
    direction: str
    smoothing_windows: int
    persistence_windows: int
    threshold: float
    hysteresis_gap: float

    @property
    def t_on(self) -> float:
        return self.threshold

    @property
    def t_off(self) -> float:
        return max(0.0, self.threshold - self.hysteresis_gap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the final benchmark run set.")
    parser.add_argument(
        "--summary",
        default="results/summary_final_clean.csv",
        help="Clean benchmark summary CSV.",
    )
    parser.add_argument(
        "--sessions-index",
        default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv",
        help="Leakfree eval sessions_index.csv path.",
    )
    parser.add_argument(
        "--eval-root",
        default=r"C:\deepfake_eval\leakfree_eval",
        help="Leakfree eval root containing sessions/.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Benchmark results root.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_scored",
        help="Output directory for scored artifacts.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_session_index(index_path: Path, eval_root: Path) -> Dict[Tuple[str, str, int], SessionMeta]:
    rows = read_csv_rows(index_path)
    mapping: Dict[Tuple[str, str, int], SessionMeta] = {}
    for row in rows:
        split = row["split"].strip()
        kind = row["kind"].strip()
        ordinal = int(row["ordinal"])
        video_filename = Path(row["video_path"]).name
        labels_filename = Path(row["labels_path"]).name
        duration_s = int(round(float(row["duration_s"])))
        mapping[(split, kind, ordinal)] = SessionMeta(
            split=split,
            kind=kind,
            ordinal=ordinal,
            video_filename=video_filename,
            labels_filename=labels_filename,
            duration_s=duration_s,
        )
    expected_sessions = eval_root / "sessions"
    if not expected_sessions.exists():
        raise FileNotFoundError(f"Evaluation sessions directory not found: {expected_sessions}")
    return mapping


def parse_label_segments(labels_path: Path) -> List[LabelSegment]:
    rows = read_csv_rows(labels_path)
    segments: List[LabelSegment] = []
    for row in rows:
        raw = row["label"].strip().lower()
        label = 1 if raw == "fake" else 0
        segments.append(
            LabelSegment(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                label=label,
                raw_label=raw,
            )
        )
    if not segments:
        raise ValueError(f"No label segments found in {labels_path}")
    return segments


def build_gt_windows(segments: Sequence[LabelSegment], duration_s: int) -> List[int]:
    windows: List[int] = []
    seg_idx = 0
    for second in range(duration_s):
        center = min(duration_s - 1e-6, second + 0.5)
        while seg_idx + 1 < len(segments) and center >= segments[seg_idx].end_s:
            seg_idx += 1
        windows.append(segments[seg_idx].label)
    return windows


def build_fake_segments(segments: Sequence[LabelSegment]) -> List[Tuple[float, float]]:
    return [(segment.start_s, segment.end_s) for segment in segments if segment.label == 1]


def load_prediction_points(predictions_path: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with predictions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score_str = row.get("score_smoothed", "").strip()
            if not score_str:
                continue
            try:
                t_rel = float(row["t_rel"])
                score = float(score_str)
            except (KeyError, ValueError):
                continue
            if t_rel < 0:
                continue
            points.append((t_rel, score))
    if not points:
        raise ValueError(f"No usable prediction points found in {predictions_path}")
    return points


def build_window_scores(
    prediction_points: Sequence[Tuple[float, float]], duration_s: int
) -> Tuple[List[Optional[float]], float, float, float, Optional[float], Optional[float], int]:
    pred_t_min = min(t_rel for t_rel, _ in prediction_points)
    pred_t_max = max(t_rel for t_rel, _ in prediction_points)
    align_shift_s = pred_t_max - duration_s

    sums = [0.0] * duration_s
    counts = [0] * duration_s
    aligned_t_values: List[float] = []
    for t_rel, score in prediction_points:
        aligned_t = t_rel - align_shift_s
        if aligned_t < 0 or aligned_t >= duration_s:
            continue
        second = int(aligned_t)
        sums[second] += score
        counts[second] += 1
        aligned_t_values.append(aligned_t)

    last_value: Optional[float] = None
    windows: List[Optional[float]] = []
    for second in range(duration_s):
        if counts[second]:
            last_value = sums[second] / counts[second]
        windows.append(last_value)

    aligned_t_min = min(aligned_t_values) if aligned_t_values else None
    aligned_t_max = max(aligned_t_values) if aligned_t_values else None
    scored_window_count = sum(1 for count in counts if count > 0)
    return windows, pred_t_min, pred_t_max, align_shift_s, aligned_t_min, aligned_t_max, scored_window_count


def load_runs(
    summary_path: Path,
    results_root: Path,
    eval_root: Path,
    session_index: Dict[Tuple[str, str, int], SessionMeta],
) -> List[RunData]:
    rows = read_csv_rows(summary_path)
    success_rows = [row for row in rows if row.get("status", "").strip() == "success"]
    if len(success_rows) != 96:
        raise ValueError(f"Expected 96 successful runs in {summary_path}, found {len(success_rows)}")

    runs: List[RunData] = []
    for row in success_rows:
        split = row["split"].strip()
        kind = row["kind"].strip()
        ordinal = int(row["ordinal"])
        key = (split, kind, ordinal)
        session = session_index.get(key)
        if session is None:
            raise KeyError(f"Missing session index row for {key}")

        labels_path = eval_root / "sessions" / split / session.labels_filename
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        result_dir = Path(row["result_dir"])
        if not result_dir.is_absolute():
            result_dir = results_root / row["run_id"]
        predictions_path = result_dir / "predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions log: {predictions_path}")

        duration_s = int(round(float(row["duration_s"])))
        if duration_s != session.duration_s:
            raise ValueError(
                f"Duration mismatch for {row['run_id']}: summary={duration_s}, session_index={session.duration_s}"
            )

        segments = parse_label_segments(labels_path)
        prediction_points = load_prediction_points(predictions_path)
        (
            window_scores,
            pred_t_min,
            pred_t_max,
            align_shift_s,
            aligned_t_min,
            aligned_t_max,
            scored_window_count,
        ) = build_window_scores(prediction_points, duration_s)

        runs.append(
            RunData(
                run_id=row["run_id"],
                backend=row["model"].strip(),
                split=split,
                kind=kind,
                ordinal=ordinal,
                duration_s=duration_s,
                result_dir=result_dir,
                video_filename=session.video_filename,
                labels_path=labels_path,
                window_scores=window_scores,
                gt_windows=build_gt_windows(segments, duration_s),
                fake_segments=build_fake_segments(segments),
                pred_t_min=pred_t_min,
                pred_t_max=pred_t_max,
                align_shift_s=align_shift_s,
                aligned_t_min=aligned_t_min,
                aligned_t_max=aligned_t_max,
                scored_window_count=scored_window_count,
            )
        )
    return runs


def trailing_mean(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return list(values)
    output: List[Optional[float]] = []
    running_sum = 0.0
    history: List[float] = []
    for value in values:
        if value is None:
            output.append((running_sum / len(history)) if history else None)
            continue
        history.append(value)
        running_sum += value
        if len(history) > window:
            running_sum -= history.pop(0)
        output.append(running_sum / len(history))
    return output


def orient_scores(values: Sequence[Optional[float]], direction: str) -> List[Optional[float]]:
    if direction == "high":
        return list(values)
    if direction == "low":
        return [None if value is None else 1.0 - value for value in values]
    raise ValueError(f"Unsupported direction: {direction}")


def apply_alert_logic(scores: Sequence[Optional[float]], config: CandidateConfig) -> List[int]:
    oriented = orient_scores(scores, config.direction)
    smoothed = trailing_mean(oriented, config.smoothing_windows)
    alerts: List[int] = []
    state = 0
    on_count = 0
    off_count = 0
    for score in smoothed:
        if score is None:
            state = 0
            on_count = 0
            off_count = 0
            alerts.append(0)
            continue
        if state == 0:
            if score >= config.t_on:
                on_count += 1
                if on_count >= config.persistence_windows:
                    state = 1
                    on_count = 0
                    off_count = 0
            else:
                on_count = 0
        else:
            if score <= config.t_off:
                off_count += 1
                if off_count >= config.persistence_windows:
                    state = 0
                    on_count = 0
                    off_count = 0
            else:
                off_count = 0
        alerts.append(state)
    return alerts


def extract_segments(binary_windows: Sequence[int]) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for idx, value in enumerate(binary_windows):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((float(start), float(idx)))
            start = None
    if start is not None:
        segments.append((float(start), float(len(binary_windows))))
    return segments


def segments_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def compute_delay_s(predicted_segments: Sequence[Tuple[float, float]], fake_segments: Sequence[Tuple[float, float]]) -> float:
    if not fake_segments:
        return 0.0
    delays: List[float] = []
    for fake_start, fake_end in fake_segments:
        delay = fake_end - fake_start
        for pred_start, pred_end in predicted_segments:
            if pred_end <= fake_start:
                continue
            if pred_start >= fake_end:
                break
            if segments_overlap((pred_start, pred_end), (fake_start, fake_end)):
                if pred_start <= fake_start < pred_end:
                    delay = 0.0
                else:
                    delay = max(0.0, pred_start - fake_start)
                break
        delays.append(delay)
    return statistics.fmean(delays) if delays else 0.0


def compute_delay_stats(
    predicted_segments: Sequence[Tuple[float, float]], fake_segments: Sequence[Tuple[float, float]]
) -> Tuple[float, int]:
    if not fake_segments:
        return 0.0, 0
    delays: List[float] = []
    for fake_start, fake_end in fake_segments:
        delay = fake_end - fake_start
        for pred_start, pred_end in predicted_segments:
            if pred_end <= fake_start:
                continue
            if pred_start >= fake_end:
                break
            if segments_overlap((pred_start, pred_end), (fake_start, fake_end)):
                if pred_start <= fake_start < pred_end:
                    delay = 0.0
                else:
                    delay = max(0.0, pred_start - fake_start)
                break
        delays.append(delay)
    return sum(delays), len(delays)


def compute_run_metrics(run: RunData, config: CandidateConfig) -> dict:
    alerts = apply_alert_logic(run.window_scores, config)
    tp = fp = fn = tn = 0
    for pred, gt in zip(alerts, run.gt_windows):
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    predicted_segments = extract_segments(alerts)
    false_alert_windows = [1 if alert and not gt else 0 for alert, gt in zip(alerts, run.gt_windows)]
    false_alert_segments = len(extract_segments(false_alert_windows))
    real_minutes = sum(1 for gt in run.gt_windows if gt == 0) / 60.0
    false_alerts_per_min = false_alert_segments / real_minutes if real_minutes > 0 else 0.0

    transitions = sum(1 for idx in range(1, len(alerts)) if alerts[idx] != alerts[idx - 1])
    alert_active_minutes = sum(alerts) / 60.0
    alert_flicker = transitions / alert_active_minutes if alert_active_minutes > 0 else 0.0

    detected_fake_segments = sum(
        1 for fake_segment in run.fake_segments if any(segments_overlap(fake_segment, pred_segment) for pred_segment in predicted_segments)
    )
    delay_sum_s, delay_count = compute_delay_stats(predicted_segments, run.fake_segments)

    return {
        "run_id": run.run_id,
        "backend": run.backend,
        "split": run.split,
        "kind": run.kind,
        "ordinal": run.ordinal,
        "video_filename": run.video_filename,
        "duration_s": run.duration_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_min": false_alerts_per_min,
        "time_to_first_detection_s": (delay_sum_s / delay_count) if delay_count else 0.0,
        "alert_flicker": alert_flicker,
        "tp_windows": tp,
        "fp_windows": fp,
        "fn_windows": fn,
        "tn_windows": tn,
        "predicted_alert_segments": len(predicted_segments),
        "false_alert_segments": false_alert_segments,
        "fake_segments": len(run.fake_segments),
        "detected_fake_segments": detected_fake_segments,
        "delay_sum_s": delay_sum_s,
        "delay_count": delay_count,
        "real_minutes": real_minutes,
        "alert_active_minutes": alert_active_minutes,
        "alert_transitions": transitions,
    }


def aggregate_metrics(run_metrics: Sequence[dict]) -> dict:
    totals = defaultdict(float)
    totals["run_count"] = len(run_metrics)
    for metrics in run_metrics:
        for key in (
            "tp_windows",
            "fp_windows",
            "fn_windows",
            "tn_windows",
            "predicted_alert_segments",
            "false_alert_segments",
            "fake_segments",
            "detected_fake_segments",
            "delay_sum_s",
            "delay_count",
            "real_minutes",
            "alert_active_minutes",
            "alert_transitions",
        ):
            totals[key] += metrics[key]

    tp = totals["tp_windows"]
    fp = totals["fp_windows"]
    fn = totals["fn_windows"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    false_alerts_per_min = (
        totals["false_alert_segments"] / totals["real_minutes"] if totals["real_minutes"] > 0 else 0.0
    )
    time_to_first_detection_s = totals["delay_sum_s"] / totals["delay_count"] if totals["delay_count"] > 0 else 0.0
    alert_flicker = (
        totals["alert_transitions"] / totals["alert_active_minutes"] if totals["alert_active_minutes"] > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_min": false_alerts_per_min,
        "time_to_first_detection_s": time_to_first_detection_s,
        "alert_flicker": alert_flicker,
        "tp_windows": int(tp),
        "fp_windows": int(fp),
        "fn_windows": int(fn),
        "tn_windows": int(totals["tn_windows"]),
        "predicted_alert_segments": int(totals["predicted_alert_segments"]),
        "false_alert_segments": int(totals["false_alert_segments"]),
        "fake_segments": int(totals["fake_segments"]),
        "detected_fake_segments": int(totals["detected_fake_segments"]),
        "real_minutes": totals["real_minutes"],
        "alert_active_minutes": totals["alert_active_minutes"],
        "alert_transitions": int(totals["alert_transitions"]),
        "run_count": int(totals["run_count"]),
    }


def calibration_search_space() -> Iterable[CandidateConfig]:
    directions = ("high", "low")
    smoothing_windows = (1, 3, 5)
    persistence_windows = (1, 2, 3)
    thresholds = [round(step * 0.05, 2) for step in range(1, 20)]
    hysteresis_gaps = (0.0, 0.05, 0.10, 0.15)
    for direction in directions:
        for smooth in smoothing_windows:
            for persistence in persistence_windows:
                for threshold in thresholds:
                    for gap in hysteresis_gaps:
                        if gap > threshold:
                            continue
                        yield CandidateConfig(
                            direction=direction,
                            smoothing_windows=smooth,
                            persistence_windows=persistence,
                            threshold=threshold,
                            hysteresis_gap=gap,
                        )


def calibration_sort_key(metrics: dict) -> Tuple[float, float, float, float, float]:
    return (
        metrics["false_alerts_per_min"],
        metrics["time_to_first_detection_s"],
        metrics["alert_flicker"],
        -metrics["f1"],
        -metrics["precision"],
    )


def choose_best_config(calibration_runs: Sequence[RunData]) -> Tuple[CandidateConfig, dict]:
    candidates: List[Tuple[CandidateConfig, dict]] = []
    for config in calibration_search_space():
        run_metrics = [compute_run_metrics(run, config) for run in calibration_runs]
        aggregate = aggregate_metrics(run_metrics)
        aggregate["direction"] = config.direction
        aggregate["smoothing_windows"] = config.smoothing_windows
        aggregate["persistence_windows"] = config.persistence_windows
        aggregate["threshold"] = config.threshold
        aggregate["t_on"] = config.t_on
        aggregate["t_off"] = config.t_off
        candidates.append((config, aggregate))

    best_f1 = max(metrics["f1"] for _, metrics in candidates)
    f1_cutoff = max(best_f1 - 0.02, best_f1 * 0.98)
    viable = [(config, metrics) for config, metrics in candidates if metrics["f1"] >= f1_cutoff]
    viable.sort(key=lambda item: calibration_sort_key(item[1]))
    return viable[0]


def rank_validation_rows(rows: Sequence[dict]) -> List[dict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["false_alerts_per_min"],
            row["time_to_first_detection_s"],
            -row["f1"],
            -row["precision"],
            -row["recall"],
            row["alert_flicker"],
            row["backend"],
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return ranked


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def round_metrics(row: dict) -> dict:
    rounded = dict(row)
    for key in (
        "precision",
        "recall",
        "f1",
        "false_alerts_per_min",
        "time_to_first_detection_s",
        "alert_flicker",
        "real_minutes",
        "alert_active_minutes",
    ):
        if key in rounded:
            rounded[key] = round(float(rounded[key]), 6)
    return rounded


def build_markdown_report(
    evaluation_unit: str,
    base_score: str,
    alignment_summary: str,
    calibration_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
) -> str:
    winner = validation_rows[0]
    lines = [
        "# Benchmark Scoring Report",
        "",
        "## Method",
        "",
        f"- Evaluation unit: {evaluation_unit}",
        f"- Base score: {base_score}",
        f"- Timeline alignment: {alignment_summary}",
        "- Score postprocessing: per-backend score polarity calibration (`high` or `low` means more fake), trailing mean smoothing, hysteresis (`t_on`, `t_off`), and persistence in 1-second windows.",
        "- Calibration selection rule: keep configs within 2% or 0.02 F1 of the backend's best calibration F1, then choose the lowest false alerts/min, lowest time-to-first-detection, lowest flicker, and finally highest F1.",
        "- Validation ranking rule: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.",
        "",
        "## Winner",
        "",
        f"- Validation winner: `{winner['backend']}`",
        f"- Validation false alerts/min: {winner['false_alerts_per_min']:.4f}",
        f"- Validation time-to-first-detection: {winner['time_to_first_detection_s']:.4f}s",
        f"- Validation F1: {winner['f1']:.4f}",
        "",
        "## Calibration Settings",
        "",
        "| backend | direction | smooth_s | persistence_s | t_on | t_off | F1 | false alerts/min | TTFD s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in calibration_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['direction']}` | {row['smoothing_windows']} | {row['persistence_windows']} | "
            f"{row['t_on']:.2f} | {row['t_off']:.2f} | {row['f1']:.4f} | {row['false_alerts_per_min']:.4f} | "
            f"{row['time_to_first_detection_s']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Validation Ranking",
            "",
            "| rank | backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validation_rows:
        lines.append(
            f"| {row['rank']} | `{row['backend']}` | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['false_alerts_per_min']:.4f} | {row['time_to_first_detection_s']:.4f} | "
            f"{row['alert_flicker']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Test Results",
            "",
            "| backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| `{row['backend']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['false_alerts_per_min']:.4f} | {row['time_to_first_detection_s']:.4f} | {row['alert_flicker']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    results_root = Path(args.results_root).resolve()
    eval_root = Path(args.eval_root).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = load_session_index(sessions_index_path, eval_root)
    runs = load_runs(summary_path, results_root, eval_root, session_index)

    backend_runs: Dict[str, List[RunData]] = defaultdict(list)
    for run in runs:
        backend_runs[run.backend].append(run)

    calibration_rows: List[dict] = []
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    alignment_rows: List[dict] = []
    thresholds_json: Dict[str, dict] = {
        "_meta": {
            "evaluation_unit": "1s_window",
            "base_score": "predictions.csv score_smoothed",
            "timeline_alignment": "end_aligned_to_video_end_using_pred_t_max_minus_duration",
            "search_space": {
                "direction": ["high", "low"],
                "smoothing_windows": [1, 3, 5],
                "persistence_windows": [1, 2, 3],
                "thresholds": [round(step * 0.05, 2) for step in range(1, 20)],
                "hysteresis_gaps": [0.0, 0.05, 0.10, 0.15],
            },
            "selection_rule": "Calibration configs within 2% or 0.02 F1 of the backend best are ranked by false_alerts_per_min, time_to_first_detection_s, alert_flicker, then F1 and precision.",
            "ranking_rule": "Validation models ranked by false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc, precision desc, recall desc, alert_flicker asc.",
        }
    }

    for backend in sorted(backend_runs):
        runs_for_backend = backend_runs[backend]
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
                    "pred_t_min": round(run.pred_t_min, 6),
                    "pred_t_max": round(run.pred_t_max, 6),
                    "pred_span_s": round(run.pred_t_max - run.pred_t_min, 6),
                    "end_align_shift_s": round(run.align_shift_s, 6),
                    "implied_head_crop_s": round(max(0.0, run.align_shift_s), 6),
                    "aligned_t_min": "" if run.aligned_t_min is None else round(run.aligned_t_min, 6),
                    "aligned_t_max": "" if run.aligned_t_max is None else round(run.aligned_t_max, 6),
                    "scored_window_count": run.scored_window_count,
                }
            )
        calibration_runs = [run for run in runs_for_backend if run.split == "calibration"]
        validation_runs = [run for run in runs_for_backend if run.split == "validation"]
        test_runs = [run for run in runs_for_backend if run.split == "test"]
        if len(calibration_runs) != 6 or len(validation_runs) != 5 or len(test_runs) != 5:
            raise ValueError(
                f"Unexpected split counts for {backend}: calibration={len(calibration_runs)}, "
                f"validation={len(validation_runs)}, test={len(test_runs)}"
            )

        best_config, calibration_metrics = choose_best_config(calibration_runs)
        calibration_row = round_metrics({"backend": backend, **calibration_metrics})
        calibration_rows.append(calibration_row)

        validation_metrics = aggregate_metrics([compute_run_metrics(run, best_config) for run in validation_runs])
        validation_rows.append(
            round_metrics(
                {
                    "backend": backend,
                    "direction": best_config.direction,
                    "smoothing_windows": best_config.smoothing_windows,
                    "persistence_windows": best_config.persistence_windows,
                    "t_on": best_config.t_on,
                    "t_off": best_config.t_off,
                    **validation_metrics,
                }
            )
        )

        test_metrics = aggregate_metrics([compute_run_metrics(run, best_config) for run in test_runs])
        test_rows.append(
            round_metrics(
                {
                    "backend": backend,
                    "direction": best_config.direction,
                    "smoothing_windows": best_config.smoothing_windows,
                    "persistence_windows": best_config.persistence_windows,
                    "t_on": best_config.t_on,
                    "t_off": best_config.t_off,
                    **test_metrics,
                }
            )
        )

        thresholds_json[backend] = {
            "direction": best_config.direction,
            "smoothing_windows": best_config.smoothing_windows,
            "persistence_windows": best_config.persistence_windows,
            "threshold": best_config.threshold,
            "t_on": best_config.t_on,
            "t_off": best_config.t_off,
            "calibration_metrics": calibration_row,
        }

    ranked_validation_rows = rank_validation_rows(validation_rows)
    test_rows_sorted = sorted(test_rows, key=lambda row: row["backend"])
    calibration_rows_sorted = sorted(calibration_rows, key=lambda row: row["backend"])

    calibration_fieldnames = [
        "backend",
        "direction",
        "smoothing_windows",
        "persistence_windows",
        "threshold",
        "t_on",
        "t_off",
        "precision",
        "recall",
        "f1",
        "false_alerts_per_min",
        "time_to_first_detection_s",
        "alert_flicker",
        "tp_windows",
        "fp_windows",
        "fn_windows",
        "tn_windows",
        "predicted_alert_segments",
        "false_alert_segments",
        "fake_segments",
        "detected_fake_segments",
        "real_minutes",
        "alert_active_minutes",
        "alert_transitions",
        "run_count",
    ]
    validation_fieldnames = [
        "rank",
        "winner",
        "backend",
        "direction",
        "smoothing_windows",
        "persistence_windows",
        "t_on",
        "t_off",
        "precision",
        "recall",
        "f1",
        "false_alerts_per_min",
        "time_to_first_detection_s",
        "alert_flicker",
        "tp_windows",
        "fp_windows",
        "fn_windows",
        "tn_windows",
        "predicted_alert_segments",
        "false_alert_segments",
        "fake_segments",
        "detected_fake_segments",
        "real_minutes",
        "alert_active_minutes",
        "alert_transitions",
        "run_count",
    ]
    test_fieldnames = [
        "backend",
        "direction",
        "smoothing_windows",
        "persistence_windows",
        "t_on",
        "t_off",
        "precision",
        "recall",
        "f1",
        "false_alerts_per_min",
        "time_to_first_detection_s",
        "alert_flicker",
        "tp_windows",
        "fp_windows",
        "fn_windows",
        "tn_windows",
        "predicted_alert_segments",
        "false_alert_segments",
        "fake_segments",
        "detected_fake_segments",
        "real_minutes",
        "alert_active_minutes",
        "alert_transitions",
        "run_count",
    ]
    alignment_fieldnames = [
        "run_id",
        "backend",
        "split",
        "kind",
        "ordinal",
        "video_filename",
        "video_duration_s",
        "pred_t_min",
        "pred_t_max",
        "pred_span_s",
        "end_align_shift_s",
        "implied_head_crop_s",
        "aligned_t_min",
        "aligned_t_max",
        "scored_window_count",
    ]

    write_csv(output_dir / "calibration_metrics.csv", calibration_rows_sorted, calibration_fieldnames)
    write_csv(output_dir / "validation_metrics.csv", ranked_validation_rows, validation_fieldnames)
    write_csv(output_dir / "test_metrics.csv", test_rows_sorted, test_fieldnames)
    write_csv(output_dir / "model_ranking.csv", ranked_validation_rows, validation_fieldnames)
    write_csv(output_dir / "alignment_audit.csv", alignment_rows, alignment_fieldnames)

    mean_shift = statistics.fmean(row["end_align_shift_s"] for row in alignment_rows)
    min_shift = min(row["end_align_shift_s"] for row in alignment_rows)
    max_shift = max(row["end_align_shift_s"] for row in alignment_rows)
    mean_scored_windows = statistics.fmean(row["scored_window_count"] for row in alignment_rows)
    alignment_summary = (
        f"Each run is end-aligned by shifting `t_rel` so the last scored timestamp lands at the 600s video end. "
        f"Across 96 runs, the shift ranges {min_shift:.3f}s to {max_shift:.3f}s (mean {mean_shift:.3f}s). "
        f"This shift is the implied head crop before the bot contributed frames. "
        f"After alignment, runs retain a mean of {mean_scored_windows:.1f} directly scored 1-second windows before any forward-fill."
    )

    report = build_markdown_report(
        evaluation_unit="1-second window",
        base_score="Per-window mean of predictions.csv `score_smoothed`; windows before the first aligned score are treated as missing/no-alert, and later gaps are forward-filled from the last seen score.",
        alignment_summary=alignment_summary,
        calibration_rows=calibration_rows_sorted,
        validation_rows=ranked_validation_rows,
        test_rows=test_rows_sorted,
    )
    (output_dir / "summary_report.md").write_text(report, encoding="utf-8")
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
