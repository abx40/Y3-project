import argparse
import csv
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows.score_benchmark_v4_dual_gate import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_UCB_QUANTILE,
    DEFAULT_WINDOW_SECONDS,
    run_gate_variant,
)


DEFAULT_WINDOWS = [1, 5, 10, 15, 20, 30, 60]
GATE_CONFIGS = {
    "strict": {
        "gate_name": "gate_strict",
        "strict_fa_per_min": 0.10,
        "strict_fa_time_ratio": 0.05,
        "relaxed_fa_per_min": 0.20,
        "relaxed_fa_time_ratio": 0.10,
    },
    "research": {
        "gate_name": "gate_research",
        "strict_fa_per_min": 0.20,
        "strict_fa_time_ratio": 0.10,
        "relaxed_fa_per_min": 0.30,
        "relaxed_fa_time_ratio": 0.15,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V4 dual-gate scorer across the standard window sweep.")
    parser.add_argument("--summary", default="results_rgb_fix_full_2026-04-09/summary_final_clean.csv")
    parser.add_argument("--sessions-index", default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv")
    parser.add_argument("--eval-root", default=r"C:\deepfake_eval\leakfree_eval")
    parser.add_argument("--output-dir", default="results_scored_v4_dual_gate_5s_2026-04-15")
    parser.add_argument("--windows", default=",".join(str(window) for window in DEFAULT_WINDOWS))
    parser.add_argument("--default-window", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-ucb-quantile", type=float, default=BOOTSTRAP_UCB_QUANTILE)
    parser.add_argument("--gate-real-ordinal", type=int, default=1)
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_windows(raw_value: str) -> List[int]:
    windows: List[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if token:
            windows.append(int(token))
    if not windows:
        raise ValueError("At least one window size is required.")
    return windows


def to_float_or_none(value: str) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def selected_rows(rows: List[dict]) -> List[dict]:
    return [row for row in rows if row.get("calibration_status") == "selected"]


def sort_best_combo(rows: List[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            row["calibration_status"] != "selected",
            -(row["validation_auroc"] if row["validation_auroc"] is not None else -1.0),
            row["validation_false_alerts_per_min"] if row["validation_false_alerts_per_min"] is not None else float("inf"),
            row["validation_ttfd_s"] if row["validation_ttfd_s"] is not None else float("inf"),
            -(row["validation_f1"] if row["validation_f1"] is not None else -1.0),
            row["window_s"],
            row["backend"],
        ),
    )[0]


def build_combined_outputs(output_dir: Path, windows: List[int]) -> None:
    combined_window_rows: List[dict] = []
    combined_backend_rows: List[dict] = []
    best_combo_by_gate: Dict[str, List[dict]] = {"strict": [], "research": []}

    for gate_key in GATE_CONFIGS:
        backend_rows_for_gate: Dict[str, List[dict]] = {}
        for window in windows:
            gate_dir = output_dir / "window_sweep" / f"window_{window}s" / f"gate_{gate_key}"
            validation_rows = load_csv_rows(gate_dir / "validation_metrics.csv")
            test_rows = load_csv_rows(gate_dir / "test_metrics.csv")
            calibration_rows = load_csv_rows(gate_dir / "calibration_metrics.csv")
            test_by_backend = {row["backend"]: row for row in test_rows}
            calibration_by_backend = {row["backend"]: row for row in calibration_rows}
            val_selected = selected_rows(validation_rows)
            test_selected = selected_rows(test_rows)
            winner = next((row for row in validation_rows if str(row.get("winner", "")).lower() == "true"), None)

            combined_window_rows.append(
                {
                    "gate": gate_key,
                    "window_s": window,
                    "compliant_backends": len(val_selected),
                    "validation_winner": winner["backend"] if winner else "",
                    "validation_winner_auroc": to_float_or_none(winner["threshold_free_auroc"]) if winner else None,
                    "validation_winner_f1": to_float_or_none(winner["f1"]) if winner else None,
                    "validation_winner_false_alerts_per_min": to_float_or_none(winner["false_alerts_per_min"]) if winner else None,
                    "validation_winner_false_alert_time_ratio": to_float_or_none(winner["false_alert_time_ratio"]) if winner else None,
                    "validation_winner_ttfd_s": to_float_or_none(winner["time_to_first_detection_s"]) if winner else None,
                    "validation_mean_auroc": mean(float(row["threshold_free_auroc"]) for row in val_selected) if val_selected else None,
                    "validation_mean_f1": mean(float(row["f1"]) for row in val_selected) if val_selected else None,
                    "validation_mean_false_alerts_per_min": mean(float(row["false_alerts_per_min"]) for row in val_selected) if val_selected else None,
                    "validation_mean_false_alert_time_ratio": mean(float(row["false_alert_time_ratio"]) for row in val_selected) if val_selected else None,
                    "validation_mean_ttfd_s": mean(float(row["time_to_first_detection_s"]) for row in val_selected) if val_selected else None,
                    "test_mean_auroc": mean(float(row["threshold_free_auroc"]) for row in test_selected) if test_selected else None,
                    "test_mean_f1": mean(float(row["f1"]) for row in test_selected) if test_selected else None,
                    "test_mean_false_alerts_per_min": mean(float(row["false_alerts_per_min"]) for row in test_selected) if test_selected else None,
                    "test_mean_false_alert_time_ratio": mean(float(row["false_alert_time_ratio"]) for row in test_selected) if test_selected else None,
                    "test_mean_ttfd_s": mean(float(row["time_to_first_detection_s"]) for row in test_selected) if test_selected else None,
                }
            )

            for validation_row in validation_rows:
                backend = validation_row["backend"]
                test_row = test_by_backend[backend]
                calibration_row = calibration_by_backend[backend]
                combined_row = {
                    "gate": gate_key,
                    "window_s": window,
                    "backend": backend,
                    "calibration_status": validation_row["calibration_status"],
                    "selected_policy": validation_row.get("selected_policy", ""),
                    "rank": validation_row.get("rank", ""),
                    "winner": str(validation_row.get("winner", "")).lower() == "true",
                    "polarity": validation_row["polarity"],
                    "validation_auroc": to_float_or_none(validation_row["threshold_free_auroc"]),
                    "validation_f1": to_float_or_none(validation_row.get("f1", "")),
                    "validation_precision": to_float_or_none(validation_row.get("precision", "")),
                    "validation_recall": to_float_or_none(validation_row.get("recall", "")),
                    "validation_false_alerts_per_min": to_float_or_none(validation_row.get("false_alerts_per_min", "")),
                    "validation_false_alert_time_ratio": to_float_or_none(validation_row.get("false_alert_time_ratio", "")),
                    "validation_ttfd_s": to_float_or_none(validation_row.get("time_to_first_detection_s", "")),
                    "validation_flicker": to_float_or_none(validation_row.get("alert_flicker", "")),
                    "test_auroc": to_float_or_none(test_row["threshold_free_auroc"]),
                    "test_f1": to_float_or_none(test_row.get("f1", "")),
                    "test_precision": to_float_or_none(test_row.get("precision", "")),
                    "test_recall": to_float_or_none(test_row.get("recall", "")),
                    "test_false_alerts_per_min": to_float_or_none(test_row.get("false_alerts_per_min", "")),
                    "test_false_alert_time_ratio": to_float_or_none(test_row.get("false_alert_time_ratio", "")),
                    "test_ttfd_s": to_float_or_none(test_row.get("time_to_first_detection_s", "")),
                    "test_flicker": to_float_or_none(test_row.get("alert_flicker", "")),
                    "calibration_auroc": to_float_or_none(calibration_row.get("calibration_auroc", "")),
                    "gate_false_alerts_per_min_ucb": to_float_or_none(calibration_row.get("gate_false_alerts_per_min_ucb", "")),
                    "gate_false_alert_time_ratio_ucb": to_float_or_none(calibration_row.get("gate_false_alert_time_ratio_ucb", "")),
                    "target_recall": to_float_or_none(calibration_row.get("target_recall", "")),
                    "target_ttfd_s": to_float_or_none(calibration_row.get("target_ttfd_s", "")),
                    "target_f1": to_float_or_none(calibration_row.get("target_f1", "")),
                    "target_alert_flicker": to_float_or_none(calibration_row.get("target_alert_flicker", "")),
                    "threshold": to_float_or_none(validation_row.get("threshold", "")),
                    "t_on": to_float_or_none(validation_row.get("t_on", "")),
                    "t_off": to_float_or_none(validation_row.get("t_off", "")),
                    "persistence_windows": int(validation_row["persistence_windows"]) if validation_row.get("persistence_windows") not in {"", None} else None,
                }
                combined_backend_rows.append(combined_row)
                backend_rows_for_gate.setdefault(backend, []).append(combined_row)

        for backend, rows in sorted(backend_rows_for_gate.items()):
            best = sort_best_combo(rows)
            best_combo_by_gate[gate_key].append(best)

    write_csv(
        output_dir / "combined_window_summary.csv",
        combined_window_rows,
        [
            "gate",
            "window_s",
            "compliant_backends",
            "validation_winner",
            "validation_winner_auroc",
            "validation_winner_f1",
            "validation_winner_false_alerts_per_min",
            "validation_winner_false_alert_time_ratio",
            "validation_winner_ttfd_s",
            "validation_mean_auroc",
            "validation_mean_f1",
            "validation_mean_false_alerts_per_min",
            "validation_mean_false_alert_time_ratio",
            "validation_mean_ttfd_s",
            "test_mean_auroc",
            "test_mean_f1",
            "test_mean_false_alerts_per_min",
            "test_mean_false_alert_time_ratio",
            "test_mean_ttfd_s",
        ],
    )

    write_csv(
        output_dir / "combined_backend_window_metrics.csv",
        combined_backend_rows,
        [
            "gate",
            "window_s",
            "backend",
            "calibration_status",
            "selected_policy",
            "rank",
            "winner",
            "polarity",
            "calibration_auroc",
            "gate_false_alerts_per_min_ucb",
            "gate_false_alert_time_ratio_ucb",
            "target_recall",
            "target_ttfd_s",
            "target_f1",
            "target_alert_flicker",
            "threshold",
            "t_on",
            "t_off",
            "persistence_windows",
            "validation_auroc",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_false_alerts_per_min",
            "validation_false_alert_time_ratio",
            "validation_ttfd_s",
            "validation_flicker",
            "test_auroc",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_false_alerts_per_min",
            "test_false_alert_time_ratio",
            "test_ttfd_s",
            "test_flicker",
        ],
    )

    for gate_key, path_name in [("strict", "window_best_combo_by_model_strict.csv"), ("research", "window_best_combo_by_model_research.csv")]:
        write_csv(
            output_dir / path_name,
            best_combo_by_gate[gate_key],
            [
                "gate",
                "window_s",
                "backend",
                "calibration_status",
                "selected_policy",
                "rank",
                "winner",
                "polarity",
                "calibration_auroc",
                "gate_false_alerts_per_min_ucb",
                "gate_false_alert_time_ratio_ucb",
                "target_recall",
                "target_ttfd_s",
                "target_f1",
                "target_alert_flicker",
                "threshold",
                "t_on",
                "t_off",
                "persistence_windows",
                "validation_auroc",
                "validation_precision",
                "validation_recall",
                "validation_f1",
                "validation_false_alerts_per_min",
                "validation_false_alert_time_ratio",
                "validation_ttfd_s",
                "validation_flicker",
                "test_auroc",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_false_alerts_per_min",
                "test_false_alert_time_ratio",
                "test_ttfd_s",
                "test_flicker",
            ],
        )

    strict_rows = {row["window_s"]: row for row in combined_window_rows if row["gate"] == "strict"}
    research_rows = {row["window_s"]: row for row in combined_window_rows if row["gate"] == "research"}
    lines = [
        "# V4 Dual Gate Window Sweep Summary",
        "",
        "- Default reporting window: `5s`.",
        "- Strict gate and research gate are kept fully separate in this bundle.",
        "",
        "| window_s | strict compliant | strict winner | strict winner AUROC | strict mean val AUROC | strict mean val F1 | strict mean val FA/min | research compliant | research winner | research winner AUROC | research mean val AUROC | research mean val F1 | research mean val FA/min |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        strict_row = strict_rows[window]
        research_row = research_rows[window]
        lines.append(
            f"| {window} | {strict_row['compliant_backends']} | `{strict_row['validation_winner']}` | "
            f"{strict_row['validation_winner_auroc'] if strict_row['validation_winner_auroc'] is not None else ''} | "
            f"{strict_row['validation_mean_auroc'] if strict_row['validation_mean_auroc'] is not None else ''} | "
            f"{strict_row['validation_mean_f1'] if strict_row['validation_mean_f1'] is not None else ''} | "
            f"{strict_row['validation_mean_false_alerts_per_min'] if strict_row['validation_mean_false_alerts_per_min'] is not None else ''} | "
            f"{research_row['compliant_backends']} | `{research_row['validation_winner']}` | "
            f"{research_row['validation_winner_auroc'] if research_row['validation_winner_auroc'] is not None else ''} | "
            f"{research_row['validation_mean_auroc'] if research_row['validation_mean_auroc'] is not None else ''} | "
            f"{research_row['validation_mean_f1'] if research_row['validation_mean_f1'] is not None else ''} | "
            f"{research_row['validation_mean_false_alerts_per_min'] if research_row['validation_mean_false_alerts_per_min'] is not None else ''} |"
        )
    lines.extend(
        [
            "",
            "## Best Validation Combo By Model",
            "",
            "### Strict Gate",
            "",
            "| backend | best window_s | status | policy | val AUROC | val F1 | val FA/min | val TTFD s | test AUROC | test F1 |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best_combo_by_gate["strict"]:
        lines.append(
            f"| `{row['backend']}` | {row['window_s']} | `{row['calibration_status']}` | `{row['selected_policy']}` | "
            f"{row['validation_auroc'] if row['validation_auroc'] is not None else ''} | "
            f"{row['validation_f1'] if row['validation_f1'] is not None else ''} | "
            f"{row['validation_false_alerts_per_min'] if row['validation_false_alerts_per_min'] is not None else ''} | "
            f"{row['validation_ttfd_s'] if row['validation_ttfd_s'] is not None else ''} | "
            f"{row['test_auroc'] if row['test_auroc'] is not None else ''} | "
            f"{row['test_f1'] if row['test_f1'] is not None else ''} |"
        )
    lines.extend(
        [
            "",
            "### Research Gate",
            "",
            "| backend | best window_s | status | policy | val AUROC | val F1 | val FA/min | val TTFD s | test AUROC | test F1 |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best_combo_by_gate["research"]:
        lines.append(
            f"| `{row['backend']}` | {row['window_s']} | `{row['calibration_status']}` | `{row['selected_policy']}` | "
            f"{row['validation_auroc'] if row['validation_auroc'] is not None else ''} | "
            f"{row['validation_f1'] if row['validation_f1'] is not None else ''} | "
            f"{row['validation_false_alerts_per_min'] if row['validation_false_alerts_per_min'] is not None else ''} | "
            f"{row['validation_ttfd_s'] if row['validation_ttfd_s'] is not None else ''} | "
            f"{row['test_auroc'] if row['test_auroc'] is not None else ''} | "
            f"{row['test_f1'] if row['test_f1'] is not None else ''} |"
        )
    (output_dir / "window_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    eval_root = Path(args.eval_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    windows = parse_windows(args.windows)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "window_sweep").mkdir(parents=True, exist_ok=True)

    for gate_key, gate_cfg in GATE_CONFIGS.items():
        top_level_dir = output_dir / f"gate_{gate_key}"
        run_gate_variant(
            summary_path=summary_path,
            sessions_index_path=sessions_index_path,
            eval_root=eval_root,
            output_dir=top_level_dir,
            window_seconds=args.default_window,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_ucb_quantile=args.bootstrap_ucb_quantile,
            gate_name=gate_cfg["gate_name"],
            strict_fa_per_min=gate_cfg["strict_fa_per_min"],
            strict_fa_time_ratio=gate_cfg["strict_fa_time_ratio"],
            relaxed_fa_per_min=gate_cfg["relaxed_fa_per_min"],
            relaxed_fa_time_ratio=gate_cfg["relaxed_fa_time_ratio"],
            gate_real_ordinal=args.gate_real_ordinal,
        )
        default_window_dir = output_dir / "window_sweep" / f"window_{args.default_window}s" / f"gate_{gate_key}"
        default_window_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(top_level_dir, default_window_dir)

    for window in windows:
        if window == args.default_window:
            continue
        for gate_key, gate_cfg in GATE_CONFIGS.items():
            gate_dir = output_dir / "window_sweep" / f"window_{window}s" / f"gate_{gate_key}"
            run_gate_variant(
                summary_path=summary_path,
                sessions_index_path=sessions_index_path,
                eval_root=eval_root,
                output_dir=gate_dir,
                window_seconds=window,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_ucb_quantile=args.bootstrap_ucb_quantile,
                gate_name=gate_cfg["gate_name"],
                strict_fa_per_min=gate_cfg["strict_fa_per_min"],
                strict_fa_time_ratio=gate_cfg["strict_fa_time_ratio"],
                relaxed_fa_per_min=gate_cfg["relaxed_fa_per_min"],
                relaxed_fa_time_ratio=gate_cfg["relaxed_fa_time_ratio"],
                gate_real_ordinal=args.gate_real_ordinal,
            )

    build_combined_outputs(output_dir, windows)


if __name__ == "__main__":
    main()
