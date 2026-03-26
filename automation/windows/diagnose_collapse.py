import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_SCORED_V2 = REPO_ROOT / "results_scored_v2"
OUTPUT_DIR = REPO_ROOT / "results_diagnostics"
EVAL_ROOT = Path(r"C:\deepfake_eval\leakfree_eval")
SESSION_INDEX = EVAL_ROOT / "sessions_index.csv"
REPRESENTATIVE_KEYS = [
    ("validation", "real_only", 1),
    ("validation", "fake_only", 1),
    ("validation", "mixed", 1),
]
OFFLINE_MODELS = ["xception_df40", "f3net"]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_threshold_lookup(thresholds_json: dict) -> Dict[str, dict]:
    lookup = {}
    for backend, payload in thresholds_json.items():
        if backend.startswith("_"):
            continue
        selected = payload["selected"]
        lookup[backend] = {
            "polarity": payload["polarity"],
            "t_on": float(selected["t_on"]),
            "t_off": float(selected["t_off"]),
            "persistence_windows": int(selected["persistence_windows"]),
        }
    return lookup


def raw_thresholds_for_plot(polarity: str, t_on: float, t_off: float) -> Tuple[float, float]:
    if polarity == "high":
        return t_on, t_off
    return 1.0 - t_on, 1.0 - t_off


def extract_boundary_deltas(scores: Sequence[Optional[float]], gt: Sequence[int], window: int = 3) -> Tuple[List[float], List[float]]:
    onset_deltas: List[float] = []
    offset_deltas: List[float] = []
    for idx in range(1, len(gt)):
        if gt[idx] == 1 and gt[idx - 1] == 0:
            before = [s for s in scores[max(0, idx - window):idx] if s is not None]
            after = [s for s in scores[idx:min(len(scores), idx + window)] if s is not None]
            if before and after:
                onset_deltas.append(statistics.fmean(after) - statistics.fmean(before))
        if gt[idx] == 0 and gt[idx - 1] == 1:
            before = [s for s in scores[max(0, idx - window):idx] if s is not None]
            after = [s for s in scores[idx:min(len(scores), idx + window)] if s is not None]
            if before and after:
                offset_deltas.append(statistics.fmean(after) - statistics.fmean(before))
    return onset_deltas, offset_deltas


def plot_trace(
    backend: str,
    run,
    threshold_cfg: dict,
    score_v2,
    out_png: Path,
    out_csv: Path,
) -> dict:
    alerts = score_v2.apply_operating_point(
        run.signal_windows,
        threshold_cfg["polarity"],
        score_v2.OperatingPoint(
            threshold=threshold_cfg["t_on"],
            hysteresis_gap=max(0.0, threshold_cfg["t_on"] - threshold_cfg["t_off"]),
            persistence_windows=threshold_cfg["persistence_windows"],
        ),
    )
    seconds = np.arange(run.duration_s)
    raw_scores = [np.nan if value is None else value for value in run.signal_windows]
    oriented_scores = [np.nan if value is None else score_v2.orient_score(value, threshold_cfg["polarity"]) for value in run.signal_windows]
    gt = np.array(run.gt_windows)
    alert_arr = np.array(alerts)
    missing = np.array([1 if value is None else 0 for value in run.signal_windows])

    rows = []
    for second in range(run.duration_s):
        rows.append(
            {
                "backend": backend,
                "run_id": run.run_id,
                "second": second,
                "score_raw": "" if np.isnan(raw_scores[second]) else raw_scores[second],
                "score_oriented": "" if np.isnan(oriented_scores[second]) else oriented_scores[second],
                "gt_label": int(gt[second]),
                "alert_state": int(alert_arr[second]),
                "missing": int(missing[second]),
            }
        )
    write_csv(out_csv, rows, ["backend", "run_id", "second", "score_raw", "score_oriented", "gt_label", "alert_state", "missing"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    raw_t_on, raw_t_off = raw_thresholds_for_plot(threshold_cfg["polarity"], threshold_cfg["t_on"], threshold_cfg["t_off"])
    axes[0].plot(seconds, raw_scores, color="tab:blue", linewidth=1.0, label="score_raw")
    axes[0].plot(seconds, oriented_scores, color="tab:purple", linewidth=0.9, alpha=0.7, label="oriented_score")
    axes[0].axhline(raw_t_on, color="tab:red", linestyle="--", linewidth=0.9, label="raw t_on")
    axes[0].axhline(raw_t_off, color="tab:orange", linestyle=":", linewidth=0.9, label="raw t_off")
    axes[0].fill_between(seconds, 0, 1, where=gt > 0, color="red", alpha=0.08, step="pre", label="fake gt")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("score")
    axes[0].set_title(f"{backend} | {run.run_id}")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    axes[1].step(seconds, gt, where="post", color="tab:red", linewidth=1.2, label="gt_label")
    axes[1].step(seconds, alert_arr, where="post", color="tab:green", linewidth=1.0, label="alert_state")
    axes[1].fill_between(seconds, 0, missing, color="gray", alpha=0.2, step="pre", label="missing")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("state")
    axes[1].set_xlabel("video second")
    axes[1].legend(loc="upper right")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    valid_pairs = [(score, label) for score, label in zip(oriented_scores, gt) if not np.isnan(score)]
    eff_auc = score_v2.auroc([score for score, _ in valid_pairs], [int(label) for _, label in valid_pairs]) if valid_pairs else float("nan")
    real_scores = [score for score, label in valid_pairs if label == 0]
    fake_scores = [score for score, label in valid_pairs if label == 1]
    onset_deltas, offset_deltas = extract_boundary_deltas([None if np.isnan(x) else float(x) for x in oriented_scores], gt.tolist())
    return {
        "backend": backend,
        "run_id": run.run_id,
        "split": run.split,
        "kind": run.kind,
        "ordinal": run.ordinal,
        "polarity": threshold_cfg["polarity"],
        "effective_auc": round(eff_auc, 6) if eff_auc == eff_auc else "",
        "real_mean_oriented": round(statistics.fmean(real_scores), 6) if real_scores else "",
        "fake_mean_oriented": round(statistics.fmean(fake_scores), 6) if fake_scores else "",
        "fake_minus_real_mean": round((statistics.fmean(fake_scores) - statistics.fmean(real_scores)), 6) if real_scores and fake_scores else "",
        "onset_count": len(onset_deltas),
        "onset_delta_mean": round(statistics.fmean(onset_deltas), 6) if onset_deltas else "",
        "onset_positive_frac": round(sum(1 for delta in onset_deltas if delta > 0) / len(onset_deltas), 6) if onset_deltas else "",
        "offset_count": len(offset_deltas),
        "offset_delta_mean": round(statistics.fmean(offset_deltas), 6) if offset_deltas else "",
        "offset_negative_frac": round(sum(1 for delta in offset_deltas if delta < 0) / len(offset_deltas), 6) if offset_deltas else "",
        "missing_window_count": int(missing.sum()),
    }


def plot_offline_vs_pipeline(
    model_key: str,
    run,
    offline_rows: Sequence[dict],
    score_v2,
    threshold_cfg: dict,
    out_png: Path,
) -> dict:
    offline_by_second = {int(row["second"]): float(row["score_raw"]) for row in offline_rows if row["score_raw"] not in (None, "")}
    seconds = np.arange(run.duration_s)
    pipeline = np.array([np.nan if value is None else value for value in run.signal_windows], dtype=float)
    offline = np.array([offline_by_second.get(int(second), np.nan) for second in seconds], dtype=float)
    gt = np.array(run.gt_windows)

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.plot(seconds, pipeline, color="tab:blue", linewidth=1.0, label="pipeline score_raw")
    ax.plot(seconds, offline, color="tab:green", linewidth=1.0, alpha=0.8, label="offline score_raw")
    ax.fill_between(seconds, 0, 1, where=gt > 0, color="red", alpha=0.08, step="pre", label="fake gt")
    ax.set_ylim(0, 1)
    ax.set_xlabel("video second")
    ax.set_ylabel("score")
    ax.set_title(f"{model_key} | offline vs pipeline | {run.run_id}")
    ax.legend(loc="upper right")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    def compute_effective_auc(raw_scores: np.ndarray) -> float:
        pairs = [(float(score_v2.orient_score(score, threshold_cfg["polarity"])), int(label)) for score, label in zip(raw_scores, gt) if not np.isnan(score)]
        return score_v2.auroc([score for score, _ in pairs], [label for _, label in pairs]) if pairs else float("nan")

    def mean_gap(raw_scores: np.ndarray) -> float:
        oriented = [float(score_v2.orient_score(score, threshold_cfg["polarity"])) for score in raw_scores if not np.isnan(score)]
        labels = [label for score, label in zip(raw_scores, gt) if not np.isnan(score)]
        real = [score for score, label in zip(oriented, labels) if label == 0]
        fake = [score for score, label in zip(oriented, labels) if label == 1]
        return (statistics.fmean(fake) - statistics.fmean(real)) if real and fake else float("nan")

    return {
        "backend": model_key,
        "run_id": run.run_id,
        "kind": run.kind,
        "pipeline_effective_auc": round(compute_effective_auc(pipeline), 6),
        "offline_effective_auc": round(compute_effective_auc(offline), 6),
        "pipeline_mean_gap": round(mean_gap(pipeline), 6),
        "offline_mean_gap": round(mean_gap(offline), 6),
        "pipeline_valid_seconds": int(np.sum(~np.isnan(pipeline))),
        "offline_valid_seconds": int(np.sum(~np.isnan(offline))),
    }


def main():
    ensure_dir(OUTPUT_DIR)
    traces_dir = ensure_dir(OUTPUT_DIR / "traces")
    offline_dir = ensure_dir(OUTPUT_DIR / "offline")
    frame_samples_dir = ensure_dir(OUTPUT_DIR / "frame_samples")
    reports_dir = ensure_dir(OUTPUT_DIR / "reports")

    score_v2 = load_module(REPO_ROOT / "automation" / "windows" / "score_benchmark_v2.py", "score_benchmark_v2_diag")
    offline_probe = load_module(REPO_ROOT / "automation" / "windows" / "offline_mp4_sanity.py", "offline_mp4_diag")

    session_index = score_v2.load_session_index(SESSION_INDEX, EVAL_ROOT)
    runs = score_v2.load_runs(RESULTS_DIR / "summary_final_clean.csv", session_index, EVAL_ROOT)
    run_lookup = {(run.split, run.kind, run.ordinal, run.backend): run for run in runs}
    thresholds = json.loads((RESULTS_SCORED_V2 / "thresholds.json").read_text(encoding="utf-8"))
    threshold_lookup = build_threshold_lookup(thresholds)

    representative_rows = []
    representative_videos: List[Path] = []
    for split, kind, ordinal in REPRESENTATIVE_KEYS:
        video_filename = session_index[(split, kind, ordinal)].video_filename
        representative_videos.append(EVAL_ROOT / "sessions" / split / video_filename)

    for backend in sorted({run.backend for run in runs}):
        cfg = threshold_lookup[backend]
        for split, kind, ordinal in REPRESENTATIVE_KEYS:
            run = run_lookup[(split, kind, ordinal, backend)]
            tag = f"{backend}__{split}_{kind}_{ordinal:02d}"
            representative_rows.append(
                plot_trace(
                    backend,
                    run,
                    cfg,
                    score_v2,
                    traces_dir / f"{tag}.png",
                    traces_dir / f"{tag}.csv",
                )
            )

    alignment_fieldnames = [
        "backend", "run_id", "split", "kind", "ordinal", "polarity", "effective_auc",
        "real_mean_oriented", "fake_mean_oriented", "fake_minus_real_mean",
        "onset_count", "onset_delta_mean", "onset_positive_frac",
        "offset_count", "offset_delta_mean", "offset_negative_frac", "missing_window_count",
    ]
    write_csv(reports_dir / "alignment_sanity.csv", representative_rows, alignment_fieldnames)

    frame_server = offline_probe.load_frame_server_module()
    sample_seconds = [120]
    offline_rows_all: List[dict] = []
    offline_comparison_rows: List[dict] = []
    sample_videos = representative_videos[:2]
    for model_key in OFFLINE_MODELS:
        offline_probe.init_runtime(frame_server, model_key)
        model_offline_rows: List[dict] = []
        for video_path in representative_videos:
            model_offline_rows.extend(
                offline_probe.score_video(
                    frame_server,
                    model_key,
                    video_path,
                    sample_every_s=1,
                    sample_seconds=sample_seconds if video_path in sample_videos else [],
                    sample_output_dir=frame_samples_dir,
                )
            )
        offline_rows_all.extend(model_offline_rows)
        for split, kind, ordinal in REPRESENTATIVE_KEYS:
            run = run_lookup[(split, kind, ordinal, model_key)]
            video_path = EVAL_ROOT / "sessions" / split / session_index[(split, kind, ordinal)].video_filename
            video_rows = [row for row in model_offline_rows if row["video_path"] == str(video_path)]
            offline_comparison_rows.append(
                plot_offline_vs_pipeline(
                    model_key,
                    run,
                    video_rows,
                    score_v2,
                    threshold_lookup[model_key],
                    offline_dir / f"{model_key}__{split}_{kind}_{ordinal:02d}__offline_vs_pipeline.png",
                )
            )

    write_csv(
        offline_dir / "offline_scores.csv",
        offline_rows_all,
        ["model", "video_path", "second", "score_raw", "score_smoothed", "face_detected", "num_faces", "bbox_area", "face_crop_size"],
    )
    write_csv(
        offline_dir / "offline_vs_pipeline.csv",
        offline_comparison_rows,
        ["backend", "run_id", "kind", "pipeline_effective_auc", "offline_effective_auc", "pipeline_mean_gap", "offline_mean_gap", "pipeline_valid_seconds", "offline_valid_seconds"],
    )

    aggregate_alignment = {}
    for backend in sorted({row["backend"] for row in representative_rows}):
        rows = [row for row in representative_rows if row["backend"] == backend]
        aggregate_alignment[backend] = {
            "mean_effective_auc": round(statistics.fmean(float(row["effective_auc"]) for row in rows if row["effective_auc"] != ""), 6),
            "mean_fake_minus_real": round(statistics.fmean(float(row["fake_minus_real_mean"]) for row in rows if row["fake_minus_real_mean"] != ""), 6),
            "mean_onset_delta": round(statistics.fmean(float(row["onset_delta_mean"]) for row in rows if row["onset_delta_mean"] != ""), 6)
            if any(row["onset_delta_mean"] != "" for row in rows)
            else "",
            "mean_offset_delta": round(statistics.fmean(float(row["offset_delta_mean"]) for row in rows if row["offset_delta_mean"] != ""), 6)
            if any(row["offset_delta_mean"] != "" for row in rows)
            else "",
        }

    findings_lines = [
        "# Diagnostic Findings",
        "",
        "## Representative Runs",
        "",
        "- validation_real_only_01",
        "- validation_fake_only_01",
        "- validation_mixed_01",
        "",
        "## Alignment Summary",
        "",
    ]
    for backend, payload in aggregate_alignment.items():
        findings_lines.append(
            f"- `{backend}`: mean effective AUC {payload['mean_effective_auc']:.4f}, mean fake-real score gap {payload['mean_fake_minus_real']:.4f}, "
            f"mean onset delta {payload['mean_onset_delta'] if payload['mean_onset_delta'] != '' else 'n/a'}, "
            f"mean offset delta {payload['mean_offset_delta'] if payload['mean_offset_delta'] != '' else 'n/a'}."
        )
    findings_lines.extend(["", "## Offline Vs Pipeline", ""])
    for backend in OFFLINE_MODELS:
        rows = [row for row in offline_comparison_rows if row["backend"] == backend and row["kind"] == "mixed"]
        offline_auc = [float(row["offline_effective_auc"]) for row in rows if not math.isnan(float(row["offline_effective_auc"]))]
        pipeline_auc = [float(row["pipeline_effective_auc"]) for row in rows if not math.isnan(float(row["pipeline_effective_auc"]))]
        offline_gap = [float(row["offline_mean_gap"]) for row in rows if not math.isnan(float(row["offline_mean_gap"]))]
        pipeline_gap = [float(row["pipeline_mean_gap"]) for row in rows if not math.isnan(float(row["pipeline_mean_gap"]))]
        findings_lines.append(
            f"- `{backend}` mixed-run offline AUC {statistics.fmean(offline_auc):.4f} vs pipeline {statistics.fmean(pipeline_auc):.4f}; "
            f"offline fake-real gap {statistics.fmean(offline_gap):.4f} vs pipeline {statistics.fmean(pipeline_gap):.4f}."
        )
    (reports_dir / "findings_report.md").write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
