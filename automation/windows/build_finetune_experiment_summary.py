from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = Path(r"C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_finetune_clean_progress_2026-04-17"
DEFAULT_BASELINE_DIR = REPO_ROOT / "results_scored_v4_np_calibration_15s_rebuilt_calibration_v2_2026-04-15"


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    artifact_dir: Path
    finetuned_bundle: Path
    live_results_dir: Path


BACKENDS: List[BackendSpec] = [
    BackendSpec(
        backend="f3net",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_f3net_ft_clean_v4_15s_2026-04-16",
        live_results_dir=REPO_ROOT / "results_f3net_ft_clean_live_2026-04-16",
    ),
    BackendSpec(
        backend="xception_df40",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "xception_finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_ft_clean_xception_efficientnet_v4_15s_2026-04-16",
        live_results_dir=REPO_ROOT / "results_ft_clean_xception_efficientnet_live_retry_2026-04-16",
    ),
    BackendSpec(
        backend="efficientnet_b4",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "efficientnet_finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_ft_clean_xception_efficientnet_v4_15s_2026-04-16",
        live_results_dir=REPO_ROOT / "results_ft_clean_xception_efficientnet_live_retry_2026-04-16",
    ),
    BackendSpec(
        backend="effort_clip_l14",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "effort_finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_ft_clean_effort_v4_15s_2026-04-17",
        live_results_dir=REPO_ROOT / "results_ft_clean_effort_live_2026-04-17",
    ),
    BackendSpec(
        backend="i3d",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "i3d_finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_ft_clean_i3d_v4_15s_2026-04-17",
        live_results_dir=REPO_ROOT / "results_ft_clean_i3d_live_2026-04-17",
    ),
    BackendSpec(
        backend="videomae",
        artifact_dir=DEFAULT_EXPERIMENT_ROOT / "artifacts" / "videomae_finetune_run_01",
        finetuned_bundle=REPO_ROOT / "results_scored_ft_clean_videomae_v4_15s_2026-04-17",
        live_results_dir=REPO_ROOT / "results_ft_clean_videomae_live_2026-04-17",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the clean fine-tuning study into one summary bundle.")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value: object) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def find_row(rows: List[Dict[str, str]], backend: str) -> Dict[str, str]:
    for row in rows:
        if row.get("backend", "").strip() == backend:
            return row
    raise KeyError(f"Backend {backend} not found")


def load_metrics_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_training_row(spec: BackendSpec) -> Dict[str, object]:
    metrics = load_metrics_json(spec.artifact_dir / "metrics.json")
    history = metrics.get("history", []) if isinstance(metrics, dict) else []
    best_val_acc = None
    best_epoch = None
    if isinstance(history, list):
        for item in history:
            val = item.get("val", {}) if isinstance(item, dict) else {}
            acc = maybe_float(val.get("acc"))
            if acc is None:
                continue
            if best_val_acc is None or acc > best_val_acc:
                best_val_acc = acc
                best_epoch = maybe_int(item.get("epoch"))

    checkpoint = metrics.get("best_checkpoint", "") if isinstance(metrics, dict) else ""
    if not checkpoint:
        if spec.backend == "videomae":
            checkpoint = str(spec.artifact_dir / "videomae_model")
        elif spec.backend == "i3d":
            checkpoint = str(spec.artifact_dir / "i3d_finetuned_best.pth")

    if metrics:
        status = "completed_with_metrics"
    elif Path(checkpoint).exists():
        status = "completed_artifact_only"
    else:
        status = "missing"

    notes = ""
    if spec.backend == "i3d":
        notes = "Best checkpoint exists; long run timed out before metrics.json flush."
    elif spec.backend == "videomae":
        notes = "Local HF model directory exists; long run timed out before metrics.json flush."
    elif spec.backend == "effort_clip_l14":
        notes = "Training converged poorly; validation accuracy plateaued after epoch 1."

    return {
        "backend": spec.backend,
        "status": status,
        "artifact_dir": str(spec.artifact_dir),
        "checkpoint_or_model_dir": checkpoint,
        "epochs": maybe_int(metrics.get("epochs")) if metrics else None,
        "batch_size": maybe_int(metrics.get("batch_size")) if metrics else None,
        "lr": maybe_float(metrics.get("lr")) if metrics else None,
        "weight_decay": maybe_float(metrics.get("weight_decay")) if metrics else None,
        "face_crop": metrics.get("face_crop", "") if metrics else "",
        "train_clip_count": maybe_int(metrics.get("train_clip_count")) if metrics else None,
        "val_clip_count": maybe_int(metrics.get("val_clip_count")) if metrics else None,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "live_results_dir": str(spec.live_results_dir),
        "finetuned_scored_dir": str(spec.finetuned_bundle),
        "notes": notes,
    }


def build_metric_row(row: Dict[str, str], backend: str, split: str, source_dir: Path) -> Dict[str, object]:
    return {
        "backend": backend,
        "split": split,
        "source_dir": str(source_dir),
        "calibration_status": row.get("calibration_status", ""),
        "selected_policy": row.get("selected_policy", ""),
        "polarity": row.get("polarity", ""),
        "threshold_free_auroc": maybe_float(row.get("threshold_free_auroc")),
        "threshold": maybe_float(row.get("threshold")),
        "t_on": maybe_float(row.get("t_on")),
        "t_off": maybe_float(row.get("t_off")),
        "persistence_windows": maybe_int(row.get("persistence_windows")),
        "precision": maybe_float(row.get("precision")),
        "recall": maybe_float(row.get("recall")),
        "f1": maybe_float(row.get("f1")),
        "false_alerts_per_min": maybe_float(row.get("false_alerts_per_min")),
        "false_alert_time_ratio": maybe_float(row.get("false_alert_time_ratio")),
        "time_to_first_detection_s": maybe_float(row.get("time_to_first_detection_s")),
        "alert_flicker": maybe_float(row.get("alert_flicker")),
        "real_minutes": maybe_float(row.get("real_minutes")),
        "run_count": maybe_int(row.get("run_count")),
    }


def compare_change(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "unknown"
    if after > before + 1e-9:
        return "improved"
    if after < before - 1e-9:
        return "worsened"
    return "unchanged"


def build_markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_val_rows = read_csv_rows(baseline_dir / "validation_metrics.csv")
    baseline_test_rows = read_csv_rows(baseline_dir / "test_metrics.csv")

    training_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    notes_rows: List[Dict[str, object]] = []

    for spec in BACKENDS:
        training_rows.append(build_training_row(spec))

        ft_calib_rows = read_csv_rows(spec.finetuned_bundle / "calibration_metrics.csv")
        ft_val_rows = read_csv_rows(spec.finetuned_bundle / "validation_metrics.csv")
        ft_test_rows = read_csv_rows(spec.finetuned_bundle / "test_metrics.csv")
        ft_calib = find_row(ft_calib_rows, spec.backend)
        ft_val = find_row(ft_val_rows, spec.backend)
        ft_test = find_row(ft_test_rows, spec.backend)

        baseline_val = find_row(baseline_val_rows, spec.backend)
        baseline_test = find_row(baseline_test_rows, spec.backend)

        calibration_rows.append(build_metric_row(ft_calib, spec.backend, "calibration", spec.finetuned_bundle))
        validation_rows.append(build_metric_row(ft_val, spec.backend, "validation", spec.finetuned_bundle))
        test_rows.append(build_metric_row(ft_test, spec.backend, "test", spec.finetuned_bundle))

        comparison_rows.append(
            {
                "backend": spec.backend,
                "baseline_dir": str(baseline_dir),
                "finetuned_dir": str(spec.finetuned_bundle),
                "baseline_calibration_status": baseline_val.get("calibration_status", ""),
                "finetuned_calibration_status": ft_val.get("calibration_status", ""),
                "baseline_val_auroc": maybe_float(baseline_val.get("threshold_free_auroc")),
                "finetuned_val_auroc": maybe_float(ft_val.get("threshold_free_auroc")),
                "delta_val_auroc": (
                    maybe_float(ft_val.get("threshold_free_auroc")) - maybe_float(baseline_val.get("threshold_free_auroc"))
                    if maybe_float(ft_val.get("threshold_free_auroc")) is not None
                    and maybe_float(baseline_val.get("threshold_free_auroc")) is not None
                    else None
                ),
                "baseline_val_f1": maybe_float(baseline_val.get("f1")),
                "finetuned_val_f1": maybe_float(ft_val.get("f1")),
                "delta_val_f1": (
                    maybe_float(ft_val.get("f1")) - maybe_float(baseline_val.get("f1"))
                    if maybe_float(ft_val.get("f1")) is not None and maybe_float(baseline_val.get("f1")) is not None
                    else None
                ),
                "baseline_test_auroc": maybe_float(baseline_test.get("threshold_free_auroc")),
                "finetuned_test_auroc": maybe_float(ft_test.get("threshold_free_auroc")),
                "delta_test_auroc": (
                    maybe_float(ft_test.get("threshold_free_auroc")) - maybe_float(baseline_test.get("threshold_free_auroc"))
                    if maybe_float(ft_test.get("threshold_free_auroc")) is not None
                    and maybe_float(baseline_test.get("threshold_free_auroc")) is not None
                    else None
                ),
                "baseline_test_f1": maybe_float(baseline_test.get("f1")),
                "finetuned_test_f1": maybe_float(ft_test.get("f1")),
                "delta_test_f1": (
                    maybe_float(ft_test.get("f1")) - maybe_float(baseline_test.get("f1"))
                    if maybe_float(ft_test.get("f1")) is not None and maybe_float(baseline_test.get("f1")) is not None
                    else None
                ),
                "raw_discrimination_change": compare_change(
                    maybe_float(baseline_test.get("threshold_free_auroc")),
                    maybe_float(ft_test.get("threshold_free_auroc")),
                ),
                "operational_change": compare_change(
                    maybe_float(baseline_test.get("f1")),
                    maybe_float(ft_test.get("f1")),
                ),
            }
        )

        notes = []
        if spec.backend in {"xception_df40", "effort_clip_l14"} and baseline_val.get("calibration_status", "") == "failed_gate":
            notes.append("Baseline failed calibration gate; fine-tuned model became selectable.")
        if spec.backend == "videomae":
            notes.append("Fine-tuned local HF checkpoint completed live replay but collapsed to random/inert behavior at 15s.")
        if spec.backend == "i3d":
            notes.append("Fine-tuned checkpoint came from a long run that timed out before metrics.json flush.")
        if spec.backend == "effort_clip_l14":
            notes.append("AUROC improved, but selected operating point stayed near-inert.")
        notes_rows.append(
            {
                "backend": spec.backend,
                "note": " ".join(notes),
            }
        )

    write_csv(
        output_dir / "training_summary.csv",
        training_rows,
        [
            "backend",
            "status",
            "artifact_dir",
            "checkpoint_or_model_dir",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "face_crop",
            "train_clip_count",
            "val_clip_count",
            "best_epoch",
            "best_val_acc",
            "live_results_dir",
            "finetuned_scored_dir",
            "notes",
        ],
    )
    write_csv(
        output_dir / "calibration_metrics.csv",
        calibration_rows,
        list(calibration_rows[0].keys()),
    )
    write_csv(
        output_dir / "validation_metrics.csv",
        validation_rows,
        list(validation_rows[0].keys()),
    )
    write_csv(
        output_dir / "test_metrics.csv",
        test_rows,
        list(test_rows[0].keys()),
    )
    write_csv(
        output_dir / "baseline_vs_finetuned.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    write_csv(
        output_dir / "per_backend_notes.csv",
        notes_rows,
        ["backend", "note"],
    )

    validation_ranked = sorted(
        validation_rows,
        key=lambda row: (
            -(row["threshold_free_auroc"] or float("-inf")),
            row["false_alerts_per_min"] if row["false_alerts_per_min"] is not None else float("inf"),
            row["time_to_first_detection_s"] if row["time_to_first_detection_s"] is not None else float("inf"),
        ),
    )
    strongest_backend = validation_ranked[0]["backend"] if validation_ranked else "unknown"
    improved_test_auroc = sum(1 for row in comparison_rows if row["raw_discrimination_change"] == "improved")
    improved_test_f1 = sum(1 for row in comparison_rows if row["operational_change"] == "improved")

    summary_lines = [
        "# Fine-Tuning Study Summary",
        "",
        "## Scope",
        "",
        f"- New leak-free experiment root: `{args.experiment_root}`",
        "- Fine-tuned backends: xception_df40, effort_clip_l14, efficientnet_b4, f3net, i3d, videomae.",
        "- Main scoring setting: v4 pooled calibration, fixed 15-second window.",
        "- Baseline comparator: current fixed-backend rebuilt-calibration bundle at 15s.",
        "- Caveat: the baseline comparator uses the current rebuilt-calibration benchmark rather than a fully rerun no-finetune pass on the new training root. Validation/test footage is unchanged, but calibration footage is not identical.",
        "",
        "## Training Status",
        "",
    ]
    summary_lines.extend(
        build_markdown_table(
            training_rows,
            ["backend", "status", "best_val_acc", "epochs", "batch_size", "checkpoint_or_model_dir"],
        )
    )
    summary_lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )
    summary_lines.extend(
        build_markdown_table(
            validation_ranked,
            ["backend", "threshold_free_auroc", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s", "calibration_status"],
        )
    )
    summary_lines.extend(
        [
            "",
            "## Test",
            "",
        ]
    )
    summary_lines.extend(
        build_markdown_table(
            sorted(
                test_rows,
                key=lambda row: (-(row["threshold_free_auroc"] or float("-inf")), row["backend"]),
            ),
            ["backend", "threshold_free_auroc", "f1", "false_alerts_per_min", "false_alert_time_ratio", "time_to_first_detection_s", "calibration_status"],
        )
    )
    summary_lines.extend(
        [
            "",
            "## Baseline vs Fine-Tuned",
            "",
        ]
    )
    summary_lines.extend(
        build_markdown_table(
            comparison_rows,
            ["backend", "baseline_test_auroc", "finetuned_test_auroc", "delta_test_auroc", "baseline_test_f1", "finetuned_test_f1", "delta_test_f1"],
        )
    )
    summary_lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Successfully fine-tuned backends: {', '.join(row['backend'] for row in training_rows if row['status'] != 'missing')}.",
            f"2. Fine-tuning improved held-out test AUROC for {improved_test_auroc} / {len(comparison_rows)} backends.",
            f"3. Fine-tuning improved held-out test F1 for {improved_test_f1} / {len(comparison_rows)} backends.",
            f"4. Strongest backend after fine-tuning on validation AUROC: `{strongest_backend}`.",
            "5. Main remaining failure mode is still live-domain mismatch / transport degradation rather than pure model capacity; several backends improved AUROC without translating that cleanly into a strong operating point.",
            "",
            "## Per-Backend Notes",
            "",
        ]
    )
    for row in notes_rows:
        summary_lines.append(f"- `{row['backend']}`: {row['note']}")

    (output_dir / "experiment_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
