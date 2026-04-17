import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW_BUNDLE = REPO_ROOT / "results_finetune_clean_fpr_calibration_15s_2026-04-17"
DEFAULT_OLD_FINE_TUNED = REPO_ROOT / "results_finetune_clean_progress_2026-04-17"
DEFAULT_BASELINE = REPO_ROOT / "results_scored_v4_np_calibration_15s_rebuilt_calibration_v2_2026-04-15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary tables for the fine-tuned FPR calibration correction pass.")
    parser.add_argument("--new-bundle", default=str(DEFAULT_NEW_BUNDLE))
    parser.add_argument("--old-fine-tuned", default=str(DEFAULT_OLD_FINE_TUNED))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE))
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


def find_by_backend(rows: List[Dict[str, str]], backend: str) -> Dict[str, str]:
    for row in rows:
        if row.get("backend", "").strip() == backend:
            return row
    raise KeyError(f"Backend {backend} not found")


def compare_change(old: Optional[float], new: Optional[float]) -> str:
    if old is None or new is None:
        return "unknown"
    if new > old + 1e-9:
        return "improved"
    if new < old - 1e-9:
        return "worsened"
    return "unchanged"


def same_op(old_row: Dict[str, str], new_row: Dict[str, str]) -> bool:
    return (
        old_row.get("selected_policy", "") == new_row.get("selected_policy", "")
        and old_row.get("polarity", "") == new_row.get("polarity", "")
        and old_row.get("threshold", "") == new_row.get("threshold", "")
        and old_row.get("t_on", "") == new_row.get("t_on", "")
        and old_row.get("t_off", "") == new_row.get("t_off", "")
        and old_row.get("persistence_windows", "") == new_row.get("persistence_windows", "")
    )


def make_markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values: List[str] = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    new_bundle = Path(args.new_bundle).resolve()
    old_fine_tuned = Path(args.old_fine_tuned).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()

    new_calib = read_csv_rows(new_bundle / "calibration_metrics.csv")
    new_val = read_csv_rows(new_bundle / "validation_metrics.csv")
    new_test = read_csv_rows(new_bundle / "test_metrics.csv")

    old_calib = read_csv_rows(old_fine_tuned / "calibration_metrics.csv")
    old_val = read_csv_rows(old_fine_tuned / "validation_metrics.csv")
    old_test = read_csv_rows(old_fine_tuned / "test_metrics.csv")

    baseline_val = read_csv_rows(baseline_dir / "validation_metrics.csv")
    baseline_test = read_csv_rows(baseline_dir / "test_metrics.csv")

    backends = sorted(row["backend"] for row in new_calib)

    baseline_vs_finetuned_rows: List[Dict[str, object]] = []
    fpr_vs_old_rows: List[Dict[str, object]] = []

    for backend in backends:
        new_c = find_by_backend(new_calib, backend)
        new_v = find_by_backend(new_val, backend)
        new_t = find_by_backend(new_test, backend)
        old_c = find_by_backend(old_calib, backend)
        old_v = find_by_backend(old_val, backend)
        old_t = find_by_backend(old_test, backend)
        base_v = find_by_backend(baseline_val, backend)
        base_t = find_by_backend(baseline_test, backend)

        baseline_vs_finetuned_rows.append(
            {
                "backend": backend,
                "baseline_calibration_status": base_v.get("calibration_status", ""),
                "fpr_calibration_status": new_v.get("calibration_status", ""),
                "baseline_val_auroc": maybe_float(base_v.get("threshold_free_auroc")),
                "fpr_val_auroc": maybe_float(new_v.get("threshold_free_auroc")),
                "delta_val_auroc": (
                    maybe_float(new_v.get("threshold_free_auroc")) - maybe_float(base_v.get("threshold_free_auroc"))
                    if maybe_float(new_v.get("threshold_free_auroc")) is not None and maybe_float(base_v.get("threshold_free_auroc")) is not None
                    else None
                ),
                "baseline_val_f1": maybe_float(base_v.get("f1")),
                "fpr_val_f1": maybe_float(new_v.get("f1")),
                "baseline_test_auroc": maybe_float(base_t.get("threshold_free_auroc")),
                "fpr_test_auroc": maybe_float(new_t.get("threshold_free_auroc")),
                "delta_test_auroc": (
                    maybe_float(new_t.get("threshold_free_auroc")) - maybe_float(base_t.get("threshold_free_auroc"))
                    if maybe_float(new_t.get("threshold_free_auroc")) is not None and maybe_float(base_t.get("threshold_free_auroc")) is not None
                    else None
                ),
                "baseline_test_f1": maybe_float(base_t.get("f1")),
                "fpr_test_f1": maybe_float(new_t.get("f1")),
                "delta_test_f1": (
                    maybe_float(new_t.get("f1")) - maybe_float(base_t.get("f1"))
                    if maybe_float(new_t.get("f1")) is not None and maybe_float(base_t.get("f1")) is not None
                    else None
                ),
            }
        )

        fpr_vs_old_rows.append(
            {
                "backend": backend,
                "old_status": old_c.get("calibration_status", ""),
                "new_status": new_c.get("calibration_status", ""),
                "old_policy": old_c.get("selected_policy", ""),
                "new_policy": new_c.get("selected_policy", ""),
                "operating_point_changed": not same_op(old_v, new_v),
                "old_polarity": old_c.get("polarity", ""),
                "new_polarity": new_c.get("polarity", ""),
                "old_threshold": old_v.get("threshold", ""),
                "new_threshold": new_v.get("threshold", ""),
                "old_t_on": old_v.get("t_on", ""),
                "new_t_on": new_v.get("t_on", ""),
                "old_t_off": old_v.get("t_off", ""),
                "new_t_off": new_v.get("t_off", ""),
                "old_persistence_windows": old_v.get("persistence_windows", ""),
                "new_persistence_windows": new_v.get("persistence_windows", ""),
                "old_calibration_fpr_real": maybe_float(old_c.get("calibration_false_alert_time_ratio")),
                "new_calibration_fpr_real": maybe_float(new_c.get("calibration_fpr_real")),
                "old_validation_auroc": maybe_float(old_v.get("threshold_free_auroc")),
                "new_validation_auroc": maybe_float(new_v.get("threshold_free_auroc")),
                "old_validation_f1": maybe_float(old_v.get("f1")),
                "new_validation_f1": maybe_float(new_v.get("f1")),
                "old_validation_recall": maybe_float(old_v.get("recall")),
                "new_validation_recall": maybe_float(new_v.get("recall")),
                "old_validation_false_alerts_per_min": maybe_float(old_v.get("false_alerts_per_min")),
                "new_validation_false_alerts_per_min": maybe_float(new_v.get("false_alerts_per_min")),
                "old_test_auroc": maybe_float(old_t.get("threshold_free_auroc")),
                "new_test_auroc": maybe_float(new_t.get("threshold_free_auroc")),
                "old_test_f1": maybe_float(old_t.get("f1")),
                "new_test_f1": maybe_float(new_t.get("f1")),
                "old_test_recall": maybe_float(old_t.get("recall")),
                "new_test_recall": maybe_float(new_t.get("recall")),
                "old_test_false_alerts_per_min": maybe_float(old_t.get("false_alerts_per_min")),
                "new_test_false_alerts_per_min": maybe_float(new_t.get("false_alerts_per_min")),
                "val_auroc_change": compare_change(
                    maybe_float(old_v.get("threshold_free_auroc")),
                    maybe_float(new_v.get("threshold_free_auroc")),
                ),
                "val_f1_change": compare_change(
                    maybe_float(old_v.get("f1")),
                    maybe_float(new_v.get("f1")),
                ),
                "val_recall_change": compare_change(
                    maybe_float(old_v.get("recall")),
                    maybe_float(new_v.get("recall")),
                ),
                "val_fa_per_min_change": compare_change(
                    maybe_float(new_v.get("false_alerts_per_min")),
                    maybe_float(old_v.get("false_alerts_per_min")),
                ),
                "test_auroc_change": compare_change(
                    maybe_float(old_t.get("threshold_free_auroc")),
                    maybe_float(new_t.get("threshold_free_auroc")),
                ),
                "test_f1_change": compare_change(
                    maybe_float(old_t.get("f1")),
                    maybe_float(new_t.get("f1")),
                ),
                "test_recall_change": compare_change(
                    maybe_float(old_t.get("recall")),
                    maybe_float(new_t.get("recall")),
                ),
                "test_fa_per_min_change": compare_change(
                    maybe_float(new_t.get("false_alerts_per_min")),
                    maybe_float(old_t.get("false_alerts_per_min")),
                ),
            }
        )

    write_csv(
        new_bundle / "baseline_vs_finetuned.csv",
        baseline_vs_finetuned_rows,
        list(baseline_vs_finetuned_rows[0].keys()),
    )
    write_csv(
        new_bundle / "fpr_vs_old_gate_comparison.csv",
        fpr_vs_old_rows,
        list(fpr_vs_old_rows[0].keys()),
    )

    try:
        shutil.copyfile(old_fine_tuned / "training_summary.csv", new_bundle / "training_summary.csv")
    except FileNotFoundError:
        pass

    new_rank = [
        row["backend"]
        for row in sorted(
            new_val,
            key=lambda row: (
                row.get("calibration_status", "") != "selected",
                -float(row.get("threshold_free_auroc") or 0.0),
                float(row.get("false_alerts_per_min") or 0.0) if row.get("false_alerts_per_min", "") != "" else float("inf"),
                float(row.get("time_to_first_detection_s") or 0.0) if row.get("time_to_first_detection_s", "") != "" else float("inf"),
            ),
        )
    ]
    old_rank = [
        row["backend"]
        for row in sorted(
            old_val,
            key=lambda row: (
                row.get("calibration_status", "") != "selected",
                -float(row.get("threshold_free_auroc") or 0.0),
                float(row.get("false_alerts_per_min") or 0.0) if row.get("false_alerts_per_min", "") != "" else float("inf"),
                float(row.get("time_to_first_detection_s") or 0.0) if row.get("time_to_first_detection_s", "") != "" else float("inf"),
            ),
        )
    ]

    changed_ops = [row["backend"] for row in fpr_vs_old_rows if str(row["operating_point_changed"]).lower() == "true"]
    changed_status = [
        row["backend"]
        for row in fpr_vs_old_rows
        if row["old_status"] != row["new_status"] or row["old_policy"] != row["new_policy"]
    ]
    fa_better = [row["backend"] for row in fpr_vs_old_rows if row["test_fa_per_min_change"] == "improved"]
    fa_worse = [row["backend"] for row in fpr_vs_old_rows if row["test_fa_per_min_change"] == "worsened"]
    f1_better = [row["backend"] for row in fpr_vs_old_rows if row["test_f1_change"] == "improved"]
    f1_worse = [row["backend"] for row in fpr_vs_old_rows if row["test_f1_change"] == "worsened"]
    recall_better = [row["backend"] for row in fpr_vs_old_rows if row["test_recall_change"] == "improved"]
    recall_worse = [row["backend"] for row in fpr_vs_old_rows if row["test_recall_change"] == "worsened"]

    summary_lines = [
        "# Fine-Tuned FPR Calibration Summary",
        "",
        "## Method",
        "",
        "- Calibration is FPR-based at a fixed 15-second window.",
        "- Candidate selection maximizes recall on fake-labelled calibration windows subject to real-window FPR on pooled real-labelled calibration windows.",
        "- Strict FPR target: 0.05. Relaxed fallback FPR target: 0.10.",
        "- FA/min is reported as an operational metric, not the main selection rule.",
        "- Validation and test remain fully held out from threshold selection.",
        "",
        "## Validation",
        "",
    ]
    summary_lines.extend(
        make_markdown_table(
            [
                {
                    "backend": row["backend"],
                    "status": row["calibration_status"],
                    "policy": row["selected_policy"],
                    "AUROC": maybe_float(row.get("threshold_free_auroc")),
                    "F1": maybe_float(row.get("f1")),
                    "FA/min": maybe_float(row.get("false_alerts_per_min")),
                    "recall": maybe_float(row.get("recall")),
                }
                for row in new_val
            ],
            ["backend", "status", "policy", "AUROC", "F1", "FA/min", "recall"],
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
        make_markdown_table(
            [
                {
                    "backend": row["backend"],
                    "status": row["calibration_status"],
                    "policy": row["selected_policy"],
                    "AUROC": maybe_float(row.get("threshold_free_auroc")),
                    "F1": maybe_float(row.get("f1")),
                    "FA/min": maybe_float(row.get("false_alerts_per_min")),
                    "recall": maybe_float(row.get("recall")),
                }
                for row in new_test
            ],
            ["backend", "status", "policy", "AUROC", "F1", "FA/min", "recall"],
        )
    )
    summary_lines.extend(
        [
            "",
            "## Comparison vs Previous Fine-Tuned 15s Bundle",
            "",
            f"- Operating point changed for: {', '.join(changed_ops) if changed_ops else 'none'}.",
            f"- Status/policy changed for: {', '.join(changed_status) if changed_status else 'none'}.",
            f"- Validation AUROC ranking changed from `{', '.join(old_rank)}` to `{', '.join(new_rank)}`.",
            f"- Held-out FA/min improved for: {', '.join(fa_better) if fa_better else 'none'}. Worsened for: {', '.join(fa_worse) if fa_worse else 'none'}.",
            f"- Held-out F1 improved for: {', '.join(f1_better) if f1_better else 'none'}. Worsened for: {', '.join(f1_worse) if f1_worse else 'none'}.",
            f"- Held-out recall improved for: {', '.join(recall_better) if recall_better else 'none'}. Worsened for: {', '.join(recall_worse) if recall_worse else 'none'}.",
        ]
    )

    conservative_fix = any(
        row["backend"] in {"f3net", "xception_df40", "effort_clip_l14", "videomae"}
        and row["test_f1_change"] == "improved"
        for row in fpr_vs_old_rows
    )
    summary_lines.append(
        f"- Conservative 15s operating points {'were partially relaxed' if conservative_fix else 'were not meaningfully fixed'} under the FPR-based rule."
    )

    (new_bundle / "experiment_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
