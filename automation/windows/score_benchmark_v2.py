import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STRICT_FALSE_ALERTS_PER_MIN = 0.10
RELAXED_FALSE_ALERTS_PER_MIN = 0.25
STRICT_FA_TIME_RATIO = 0.05
RELAXED_FA_TIME_RATIO = 0.15


@dataclass(frozen=True)
class LabelSegment:
    start_s: float
    end_s: float
    label: int


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
    signal_source: str
    signal_windows: List[Optional[float]]
    gt_windows: List[int]
    fake_segments: List[Tuple[float, float]]
    pred_t_min: float
    pred_t_max: float
    align_shift_s: float
    scored_window_count: int
    held_window_count: int
    missing_window_count: int


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    hysteresis_gap: float
    persistence_windows: int

    @property
    def t_on(self) -> float:
        return self.threshold

    @property
    def t_off(self) -> float:
        return max(0.0, self.threshold - self.hysteresis_gap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operational rescoring for the final benchmark run set.")
    parser.add_argument("--summary", default="results/summary_final_clean.csv")
    parser.add_argument("--sessions-index", default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv")
    parser.add_argument("--eval-root", default=r"C:\deepfake_eval\leakfree_eval")
    parser.add_argument("--output-dir", default="results_scored_v2")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_session_index(index_path: Path, eval_root: Path) -> Dict[Tuple[str, str, int], SessionMeta]:
    mapping: Dict[Tuple[str, str, int], SessionMeta] = {}
    for row in read_csv_rows(index_path):
        split = row["split"].strip()
        kind = row["kind"].strip()
        ordinal = int(row["ordinal"])
        mapping[(split, kind, ordinal)] = SessionMeta(
            split=split,
            kind=kind,
            ordinal=ordinal,
            video_filename=Path(row["video_path"]).name,
            labels_filename=Path(row["labels_path"]).name,
            duration_s=int(round(float(row["duration_s"]))),
        )
    sessions_dir = eval_root / "sessions"
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Missing evaluation sessions directory: {sessions_dir}")
    return mapping


def parse_label_segments(labels_path: Path) -> List[LabelSegment]:
    segments: List[LabelSegment] = []
    for row in read_csv_rows(labels_path):
        segments.append(
            LabelSegment(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                label=1 if row["label"].strip().lower() == "fake" else 0,
            )
        )
    if not segments:
        raise ValueError(f"No label segments found in {labels_path}")
    return segments


def build_gt_windows(segments: Sequence[LabelSegment], duration_s: int) -> List[int]:
    gt: List[int] = []
    seg_idx = 0
    for second in range(duration_s):
        center = min(duration_s - 1e-6, second + 0.5)
        while seg_idx + 1 < len(segments) and center >= segments[seg_idx].end_s:
            seg_idx += 1
        gt.append(segments[seg_idx].label)
    return gt


def build_fake_segments(segments: Sequence[LabelSegment]) -> List[Tuple[float, float]]:
    return [(segment.start_s, segment.end_s) for segment in segments if segment.label == 1]


def load_prediction_rows(predictions_path: Path) -> List[dict]:
    return read_csv_rows(predictions_path)


def select_signal_source(rows: Sequence[dict]) -> str:
    has_raw = any(row.get("score_raw", "").strip() for row in rows)
    return "score_raw" if has_raw else "score_smoothed"


def build_signal_windows(
    rows: Sequence[dict], duration_s: int
) -> Tuple[str, List[Optional[float]], float, float, float, int, int, int]:
    if not rows:
        raise ValueError("Prediction rows are empty")

    signal_source = select_signal_source(rows)
    time_and_values: List[Tuple[float, float]] = []
    all_t: List[float] = []
    for row in rows:
        try:
            t_rel = float(row["t_rel"])
        except (KeyError, ValueError):
            continue
        all_t.append(t_rel)
        value_str = row.get(signal_source, "").strip()
        if not value_str:
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        time_and_values.append((t_rel, value))

    if not time_and_values or not all_t:
        raise ValueError("No usable prediction timestamps or scores found")

    pred_t_min = min(all_t)
    pred_t_max = max(all_t)
    align_shift_s = pred_t_max - duration_s

    sums = [0.0] * duration_s
    counts = [0] * duration_s
    for t_rel, value in time_and_values:
        aligned_t = t_rel - align_shift_s
        if aligned_t < 0 or aligned_t >= duration_s:
            continue
        second = int(aligned_t)
        sums[second] += value
        counts[second] += 1

    actual: List[Optional[float]] = []
    for second in range(duration_s):
        actual.append((sums[second] / counts[second]) if counts[second] else None)

    signal: List[Optional[float]] = []
    last_actual_value: Optional[float] = None
    last_actual_second: Optional[int] = None
    held_window_count = 0
    missing_window_count = 0
    scored_window_count = 0
    for second, value in enumerate(actual):
        if value is not None:
            signal.append(value)
            last_actual_value = value
            last_actual_second = second
            scored_window_count += 1
            continue
        if last_actual_value is not None and last_actual_second is not None and second - last_actual_second <= 1:
            signal.append(last_actual_value)
            held_window_count += 1
        else:
            signal.append(None)
            missing_window_count += 1

    return (
        signal_source,
        signal,
        pred_t_min,
        pred_t_max,
        align_shift_s,
        scored_window_count,
        held_window_count,
        missing_window_count,
    )


def load_runs(summary_path: Path, session_index: Dict[Tuple[str, str, int], SessionMeta], eval_root: Path) -> List[RunData]:
    rows = [row for row in read_csv_rows(summary_path) if row.get("status", "").strip() == "success"]
    if len(rows) != 96:
        raise ValueError(f"Expected 96 successful runs in {summary_path}, found {len(rows)}")

    runs: List[RunData] = []
    for row in rows:
        split = row["split"].strip()
        kind = row["kind"].strip()
        ordinal = int(row["ordinal"])
        meta = session_index[(split, kind, ordinal)]
        labels_path = eval_root / "sessions" / split / meta.labels_filename
        pred_rows = load_prediction_rows(Path(row["result_dir"]) / "predictions.csv")
        (
            signal_source,
            signal_windows,
            pred_t_min,
            pred_t_max,
            align_shift_s,
            scored_window_count,
            held_window_count,
            missing_window_count,
        ) = build_signal_windows(pred_rows, meta.duration_s)
        segments = parse_label_segments(labels_path)
        runs.append(
            RunData(
                run_id=row["run_id"],
                backend=row["model"].strip(),
                split=split,
                kind=kind,
                ordinal=ordinal,
                duration_s=meta.duration_s,
                result_dir=Path(row["result_dir"]),
                video_filename=meta.video_filename,
                labels_path=labels_path,
                signal_source=signal_source,
                signal_windows=signal_windows,
                gt_windows=build_gt_windows(segments, meta.duration_s),
                fake_segments=build_fake_segments(segments),
                pred_t_min=pred_t_min,
                pred_t_max=pred_t_max,
                align_shift_s=align_shift_s,
                scored_window_count=scored_window_count,
                held_window_count=held_window_count,
                missing_window_count=missing_window_count,
            )
        )
    return runs


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(score, label) for score, label in zip(scores, labels)]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[pos][1][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[ordered[idx][0]] = avg_rank
        pos = end
    rank_sum = sum(ranks[idx] for idx, (_, label) in enumerate(pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def determine_backend_polarity(calibration_runs: Sequence[RunData]) -> dict:
    scores: List[float] = []
    labels: List[int] = []
    real_scores: List[float] = []
    fake_scores: List[float] = []
    signal_sources = set()
    missing_windows = 0
    for run in calibration_runs:
        signal_sources.add(run.signal_source)
        for score, label in zip(run.signal_windows, run.gt_windows):
            if score is None:
                missing_windows += 1
                continue
            scores.append(score)
            labels.append(label)
            if label == 1:
                fake_scores.append(score)
            else:
                real_scores.append(score)

    raw_auc = auroc(scores, labels)
    polarity = "high" if raw_auc >= 0.5 else "low"
    effective_auc = raw_auc if polarity == "high" else (1.0 - raw_auc)
    return {
        "signal_sources": sorted(signal_sources),
        "polarity": polarity,
        "raw_auc_high_means_fake": raw_auc,
        "effective_auc": effective_auc,
        "real_mean": statistics.fmean(real_scores) if real_scores else float("nan"),
        "fake_mean": statistics.fmean(fake_scores) if fake_scores else float("nan"),
        "real_count": len(real_scores),
        "fake_count": len(fake_scores),
        "missing_window_count": missing_windows,
        "real_scores": real_scores,
        "fake_scores": fake_scores,
    }


def orient_score(score: Optional[float], polarity: str) -> Optional[float]:
    if score is None:
        return None
    return score if polarity == "high" else (1.0 - score)


def apply_operating_point(
    signal_windows: Sequence[Optional[float]], polarity: str, op: OperatingPoint
) -> List[int]:
    alerts: List[int] = []
    state = 0
    on_streak = 0
    off_streak = 0
    for raw_score in signal_windows:
        score = orient_score(raw_score, polarity)
        if score is None:
            state = 0
            on_streak = 0
            off_streak = 0
            alerts.append(0)
            continue
        if state == 0:
            if score >= op.t_on:
                on_streak += 1
                if on_streak >= op.persistence_windows:
                    state = 1
                    on_streak = 0
                    off_streak = 0
            else:
                on_streak = 0
        else:
            if score <= op.t_off:
                off_streak += 1
                if off_streak >= op.persistence_windows:
                    state = 0
                    on_streak = 0
                    off_streak = 0
            else:
                off_streak = 0
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


def compute_delay_stats(predicted_segments: Sequence[Tuple[float, float]], fake_segments: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    if not fake_segments:
        return 0.0, 0
    delay_sum = 0.0
    delay_count = 0
    for fake_start, fake_end in fake_segments:
        delay = fake_end - fake_start
        for pred_start, pred_end in predicted_segments:
            if pred_end <= fake_start:
                continue
            if pred_start >= fake_end:
                break
            if segments_overlap((pred_start, pred_end), (fake_start, fake_end)):
                delay = 0.0 if pred_start <= fake_start < pred_end else max(0.0, pred_start - fake_start)
                break
        delay_sum += delay
        delay_count += 1
    return delay_sum, delay_count


def compute_run_metrics(run: RunData, alerts: Sequence[int]) -> dict:
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
    false_alert_time_ratio = fp / (fp + tn) if (fp + tn) > 0 else 1.0
    transitions = sum(1 for idx in range(1, len(alerts)) if alerts[idx] != alerts[idx - 1])
    alert_active_minutes = sum(alerts) / 60.0
    alert_flicker = transitions / alert_active_minutes if alert_active_minutes > 0 else 0.0
    detected_fake_segments = sum(
        1 for fake_segment in run.fake_segments if any(segments_overlap(fake_segment, pred_segment) for pred_segment in predicted_segments)
    )
    delay_sum_s, delay_count = compute_delay_stats(predicted_segments, run.fake_segments)
    missing_windows = sum(1 for score in run.signal_windows if score is None)

    return {
        "run_id": run.run_id,
        "backend": run.backend,
        "split": run.split,
        "kind": run.kind,
        "ordinal": run.ordinal,
        "video_filename": run.video_filename,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_min": false_alerts_per_min,
        "false_alert_time_ratio": false_alert_time_ratio,
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
        "missing_windows": missing_windows,
        "scored_windows": run.scored_window_count,
        "held_windows": run.held_window_count,
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
            "missing_windows",
            "scored_windows",
            "held_windows",
        ):
            totals[key] += metrics[key]

    tp = totals["tp_windows"]
    fp = totals["fp_windows"]
    fn = totals["fn_windows"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_alerts_per_min = totals["false_alert_segments"] / totals["real_minutes"] if totals["real_minutes"] > 0 else 0.0
    false_alert_time_ratio = fp / (fp + totals["tn_windows"]) if (fp + totals["tn_windows"]) > 0 else 1.0
    ttfd = totals["delay_sum_s"] / totals["delay_count"] if totals["delay_count"] > 0 else 0.0
    flicker = totals["alert_transitions"] / totals["alert_active_minutes"] if totals["alert_active_minutes"] > 0 else 0.0
    total_windows = tp + fp + fn + totals["tn_windows"]
    missing_rate = totals["missing_windows"] / total_windows if total_windows > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_min": false_alerts_per_min,
        "false_alert_time_ratio": false_alert_time_ratio,
        "time_to_first_detection_s": ttfd,
        "alert_flicker": flicker,
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
        "missing_windows": int(totals["missing_windows"]),
        "scored_windows": int(totals["scored_windows"]),
        "held_windows": int(totals["held_windows"]),
        "missing_rate": missing_rate,
        "run_count": int(totals["run_count"]),
    }


def round_metrics(row: dict) -> dict:
    rounded = dict(row)
    for key in (
        "precision",
        "recall",
        "f1",
        "false_alerts_per_min",
        "false_alert_time_ratio",
        "time_to_first_detection_s",
        "alert_flicker",
        "real_minutes",
        "alert_active_minutes",
        "missing_rate",
        "raw_auc_high_means_fake",
        "effective_auc",
        "real_only_false_alerts_per_min",
        "real_only_false_alert_time_ratio",
        "target_recall",
        "target_ttfd_s",
    ):
        if key in rounded:
            rounded[key] = round(float(rounded[key]), 6)
    return rounded


def search_operating_points() -> Iterable[OperatingPoint]:
    thresholds = [round(step / 100.0, 2) for step in range(1, 100)]
    hysteresis_gaps = (0.0, 0.05, 0.10)
    persistence_windows = (1, 2, 3)
    for threshold in thresholds:
        for gap in hysteresis_gaps:
            if gap > threshold:
                continue
            for persistence in persistence_windows:
                yield OperatingPoint(threshold=threshold, hysteresis_gap=gap, persistence_windows=persistence)


def evaluate_candidate(polarity: str, op: OperatingPoint, runs: Sequence[RunData]) -> List[dict]:
    metrics = []
    for run in runs:
        alerts = apply_operating_point(run.signal_windows, polarity, op)
        metrics.append(compute_run_metrics(run, alerts))
    return metrics


def pick_best_candidate(candidates: Sequence[dict], gate_label: str) -> Optional[dict]:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["target_metrics"]["recall"],
            item["target_metrics"]["time_to_first_detection_s"],
            -item["target_metrics"]["f1"],
            item["real_only_metrics"]["false_alerts_per_min"],
            item["real_only_metrics"]["false_alert_time_ratio"],
            item["target_metrics"]["alert_flicker"],
            item["op"].threshold,
            item["op"].persistence_windows,
            item["op"].hysteresis_gap,
        ),
    )
    best = dict(ranked[0])
    best["gate_label"] = gate_label
    return best


def pick_best_available(candidates: Sequence[dict]) -> dict:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["real_only_metrics"]["false_alerts_per_min"],
            item["real_only_metrics"]["false_alert_time_ratio"],
            -item["target_metrics"]["recall"],
            item["target_metrics"]["time_to_first_detection_s"],
            -item["target_metrics"]["f1"],
            item["target_metrics"]["alert_flicker"],
            item["op"].threshold,
            item["op"].persistence_windows,
            item["op"].hysteresis_gap,
        ),
    )
    best = dict(ranked[0])
    best["gate_label"] = "best_available"
    return best


def distribution_row(backend: str, scope: str, polarity: str, signal_source: str, scores: Sequence[float], labels: Sequence[int]) -> dict:
    real_scores = [score for score, label in zip(scores, labels) if label == 0]
    fake_scores = [score for score, label in zip(scores, labels) if label == 1]
    auc_high = auroc(scores, labels)
    return {
        "backend": backend,
        "scope": scope,
        "signal_source": signal_source,
        "polarity": polarity,
        "auc_high_means_fake": round(auc_high, 6),
        "effective_auc": round(auc_high if polarity == "high" else (1.0 - auc_high), 6),
        "real_count": len(real_scores),
        "fake_count": len(fake_scores),
        "real_mean": round(statistics.fmean(real_scores), 6) if real_scores else "",
        "real_std": round(statistics.pstdev(real_scores), 6) if len(real_scores) > 1 else "",
        "real_p05": round(percentile(real_scores, 0.05), 6) if real_scores else "",
        "real_p25": round(percentile(real_scores, 0.25), 6) if real_scores else "",
        "real_p50": round(percentile(real_scores, 0.50), 6) if real_scores else "",
        "real_p75": round(percentile(real_scores, 0.75), 6) if real_scores else "",
        "real_p95": round(percentile(real_scores, 0.95), 6) if real_scores else "",
        "fake_mean": round(statistics.fmean(fake_scores), 6) if fake_scores else "",
        "fake_std": round(statistics.pstdev(fake_scores), 6) if len(fake_scores) > 1 else "",
        "fake_p05": round(percentile(fake_scores, 0.05), 6) if fake_scores else "",
        "fake_p25": round(percentile(fake_scores, 0.25), 6) if fake_scores else "",
        "fake_p50": round(percentile(fake_scores, 0.50), 6) if fake_scores else "",
        "fake_p75": round(percentile(fake_scores, 0.75), 6) if fake_scores else "",
        "fake_p95": round(percentile(fake_scores, 0.95), 6) if fake_scores else "",
        "mean_gap_fake_minus_real": round((statistics.fmean(fake_scores) - statistics.fmean(real_scores)), 6)
        if real_scores and fake_scores
        else "",
    }


def build_summary_report(
    selected_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
    baseline_rows: Sequence[dict],
) -> str:
    winner = validation_rows[0]
    lines = [
        "# Benchmark Scoring Report V2",
        "",
        "## Method",
        "",
        "- Evaluation unit: 1-second window.",
        "- Base signal: per-second mean of `predictions.csv:score_raw` after end-aligning each run to the video end. All runs had `score_raw`, so `score_smoothed` fallback was not needed.",
        "- Missing predictions: hold the last observed score for at most 1 second, then mark the window as missing/no-decision.",
        "- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.",
        (
            f"- False-alert gates: strict <= {STRICT_FALSE_ALERTS_PER_MIN:.2f} FA/min and <= {STRICT_FA_TIME_RATIO:.2f} "
            f"real-time alert ratio on `calibration_real_only_01`; relaxed <= {RELAXED_FALSE_ALERTS_PER_MIN:.2f} "
            f"FA/min and <= {RELAXED_FA_TIME_RATIO:.2f} real-time alert ratio."
        ),
        "- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.",
        "- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.",
        "",
        "## Winner",
        "",
        f"- Validation winner: `{winner['backend']}`",
        f"- Validation false alerts/min: {winner['false_alerts_per_min']:.4f}",
        f"- Validation time-to-first-detection: {winner['time_to_first_detection_s']:.4f}s",
        f"- Validation F1: {winner['f1']:.4f}",
        "",
        "## Selected Operating Points",
        "",
        "| backend | selected policy | polarity | persistence_s | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['selected_policy']}` | `{row['polarity']}` | {row['persistence_windows']} | "
            f"{row['t_on']:.2f} | {row['t_off']:.2f} | {row['real_only_false_alerts_per_min']:.4f} | "
            f"{row['real_only_false_alert_time_ratio']:.4f} | "
            f"{row['target_recall']:.4f} | {row['target_ttfd_s']:.4f} |"
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

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| `{row['backend']}` | `{row['split']}` | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['false_alerts_per_min']:.4f} | {row['false_alert_time_ratio']:.4f} | "
            f"{row['time_to_first_detection_s']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    eval_root = Path(args.eval_root).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_index = load_session_index(sessions_index_path, eval_root)
    runs = load_runs(summary_path, session_index, eval_root)
    backend_runs: Dict[str, List[RunData]] = defaultdict(list)
    for run in runs:
        backend_runs[run.backend].append(run)

    selected_rows: List[dict] = []
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    baseline_rows: List[dict] = []
    distribution_rows: List[dict] = []
    alignment_rows: List[dict] = []
    thresholds: Dict[str, dict] = {
        "_meta": {
            "evaluation_unit": "1s_window",
            "base_signal_priority": ["score_raw", "score_smoothed_fallback_if_no_raw"],
            "missing_policy": "hold_last_score_for_1_second_then_missing_no_decision",
            "timeline_alignment": "end_aligned_to_video_end_using_pred_t_max_minus_duration",
            "strict_false_alerts_per_min": STRICT_FALSE_ALERTS_PER_MIN,
            "relaxed_false_alerts_per_min": RELAXED_FALSE_ALERTS_PER_MIN,
            "strict_false_alert_time_ratio": STRICT_FA_TIME_RATIO,
            "relaxed_false_alert_time_ratio": RELAXED_FA_TIME_RATIO,
            "search_space": {
                "thresholds": [round(step / 100.0, 2) for step in range(1, 100)],
                "hysteresis_gaps": [0.0, 0.05, 0.10],
                "persistence_windows": [1, 2, 3],
            },
            "selection_rule": "Gate on calibration_real_only_01 false_alerts_per_min and false_alert_time_ratio, then choose best recall and lowest TTFD on calibration_fake_only_01 plus calibration mixed sessions.",
            "validation_ranking_rule": "false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc",
        }
    }

    for backend in sorted(backend_runs):
        runs_for_backend = backend_runs[backend]
        calibration_runs = [run for run in runs_for_backend if run.split == "calibration"]
        calibration_real_only = [run for run in calibration_runs if run.kind == "real_only"]
        calibration_target = [run for run in calibration_runs if run.kind != "real_only"]
        validation_runs = [run for run in runs_for_backend if run.split == "validation"]
        test_runs = [run for run in runs_for_backend if run.split == "test"]

        polarity_info = determine_backend_polarity(calibration_runs)
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

        distribution_rows.append(distribution_row(backend, "calibration", polarity, signal_source, calibration_scores, calibration_labels))
        distribution_rows.append(distribution_row(backend, "all", polarity, signal_source, all_scores, all_labels))

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

        candidate_rows: List[dict] = []
        for op in search_operating_points():
            real_only_metrics = aggregate_metrics(evaluate_candidate(polarity, op, calibration_real_only))
            target_metrics = aggregate_metrics(evaluate_candidate(polarity, op, calibration_target))
            calibration_all_metrics = aggregate_metrics(evaluate_candidate(polarity, op, calibration_runs))
            candidate_rows.append(
                {
                    "op": op,
                    "real_only_metrics": real_only_metrics,
                    "target_metrics": target_metrics,
                    "calibration_all_metrics": calibration_all_metrics,
                }
            )

        strict_best = pick_best_candidate(
            [
                row
                for row in candidate_rows
                if row["real_only_metrics"]["false_alerts_per_min"] <= STRICT_FALSE_ALERTS_PER_MIN
                and row["real_only_metrics"]["false_alert_time_ratio"] <= STRICT_FA_TIME_RATIO
            ],
            "strict_0.10",
        )
        relaxed_best = pick_best_candidate(
            [
                row
                for row in candidate_rows
                if row["real_only_metrics"]["false_alerts_per_min"] <= RELAXED_FALSE_ALERTS_PER_MIN
                and row["real_only_metrics"]["false_alert_time_ratio"] <= RELAXED_FA_TIME_RATIO
            ],
            "relaxed_0.25",
        )
        selected = strict_best or relaxed_best or pick_best_available(candidate_rows)

        op = selected["op"]
        validation_metrics = aggregate_metrics(evaluate_candidate(polarity, op, validation_runs))
        test_metrics = aggregate_metrics(evaluate_candidate(polarity, op, test_runs))

        selected_rows.append(
            round_metrics(
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
                    "target_ttfd_s": selected["target_metrics"]["time_to_first_detection_s"],
                    **selected["calibration_all_metrics"],
                }
            )
        )

        validation_rows.append(
            round_metrics(
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
            round_metrics(
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
                "real_only_metrics": round_metrics(strict_best["real_only_metrics"]),
                "target_metrics": round_metrics(strict_best["target_metrics"]),
            },
            "relaxed_candidate": None
            if relaxed_best is None
            else {
                "threshold": relaxed_best["op"].threshold,
                "t_on": relaxed_best["op"].t_on,
                "t_off": relaxed_best["op"].t_off,
                "persistence_windows": relaxed_best["op"].persistence_windows,
                "real_only_metrics": round_metrics(relaxed_best["real_only_metrics"]),
                "target_metrics": round_metrics(relaxed_best["target_metrics"]),
            },
            "selected": {
                "policy": selected["gate_label"],
                "threshold": op.threshold,
                "t_on": op.t_on,
                "t_off": op.t_off,
                "persistence_windows": op.persistence_windows,
                "real_only_metrics": round_metrics(selected["real_only_metrics"]),
                "target_metrics": round_metrics(selected["target_metrics"]),
                "calibration_all_metrics": round_metrics(selected["calibration_all_metrics"]),
            },
        }

    ranked_validation = sorted(
        validation_rows,
        key=lambda row: (
            row["false_alerts_per_min"],
            row["time_to_first_detection_s"],
            -row["f1"],
            row["backend"],
        ),
    )
    for idx, row in enumerate(ranked_validation, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1

    test_rows_sorted = sorted(test_rows, key=lambda row: row["backend"])
    selected_rows_sorted = sorted(selected_rows, key=lambda row: row["backend"])

    split_groups = {
        "validation": [run for run in runs if run.split == "validation"],
        "test": [run for run in runs if run.split == "test"],
    }
    for baseline_name, alert_value in {"always_real": 0, "always_fake": 1}.items():
        for split_name, split_runs in split_groups.items():
            metrics = [compute_run_metrics(run, [alert_value] * run.duration_s) for run in split_runs]
            baseline_rows.append(round_metrics({"backend": baseline_name, "split": split_name, **aggregate_metrics(metrics)}))

    calibration_fieldnames = [
        "backend", "signal_source", "polarity", "raw_auc_high_means_fake", "effective_auc",
        "selected_policy", "strict_gate_met", "relaxed_gate_met",
        "threshold", "t_on", "t_off", "persistence_windows",
        "real_only_false_alerts_per_min", "real_only_false_alert_time_ratio", "target_recall", "target_ttfd_s",
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
    baseline_fieldnames = [
        "backend", "split", "precision", "recall", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s",
        "alert_flicker", "tp_windows", "fp_windows", "fn_windows", "tn_windows",
        "predicted_alert_segments", "false_alert_segments", "fake_segments", "detected_fake_segments",
        "real_minutes", "alert_active_minutes", "alert_transitions",
        "missing_windows", "scored_windows", "held_windows", "missing_rate", "run_count",
    ]
    distribution_fieldnames = [
        "backend", "scope", "signal_source", "polarity", "auc_high_means_fake", "effective_auc",
        "real_count", "fake_count", "real_mean", "real_std", "real_p05", "real_p25", "real_p50", "real_p75", "real_p95",
        "fake_mean", "fake_std", "fake_p05", "fake_p25", "fake_p50", "fake_p75", "fake_p95",
        "mean_gap_fake_minus_real",
    ]
    alignment_fieldnames = [
        "run_id", "backend", "split", "kind", "ordinal", "video_filename", "video_duration_s", "signal_source",
        "pred_t_min", "pred_t_max", "end_align_shift_s",
        "scored_window_count", "held_window_count", "missing_window_count",
    ]

    write_csv(output_dir / "calibration_metrics.csv", selected_rows_sorted, calibration_fieldnames)
    write_csv(output_dir / "validation_metrics.csv", ranked_validation, evaluation_fieldnames)
    write_csv(output_dir / "test_metrics.csv", test_rows_sorted, [field for field in evaluation_fieldnames if field not in {"rank", "winner"}])
    write_csv(output_dir / "model_ranking.csv", ranked_validation, evaluation_fieldnames)
    write_csv(output_dir / "baselines.csv", baseline_rows, baseline_fieldnames)
    write_csv(output_dir / "score_distribution_summary.csv", distribution_rows, distribution_fieldnames)
    write_csv(output_dir / "alignment_audit.csv", alignment_rows, alignment_fieldnames)
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(
        build_summary_report(selected_rows_sorted, ranked_validation, test_rows_sorted, baseline_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
