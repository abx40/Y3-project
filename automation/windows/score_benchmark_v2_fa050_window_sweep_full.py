import argparse
import csv
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows.score_benchmark_v2_fa050_5s import CONFIG as BASE_CONFIG
from automation.windows.score_benchmark_v2_variant_lib import run_variant_with_paths


DEFAULT_WINDOWS = [1, 5, 10, 15, 20, 30, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FA0.50 gate scorer across the standard full window sweep.")
    parser.add_argument(
        "--summary",
        default=BASE_CONFIG.summary_default,
        help="Path to the clean run summary CSV.",
    )
    parser.add_argument(
        "--sessions-index",
        default=BASE_CONFIG.sessions_index_default,
        help="Path to the evaluation sessions index CSV.",
    )
    parser.add_argument(
        "--eval-root",
        default=BASE_CONFIG.eval_root_default,
        help="Root directory of the evaluation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_scored_v2_fa050_window_sweep_full_2026-04-14",
        help="Directory to write the full sweep bundle.",
    )
    parser.add_argument(
        "--windows",
        default=",".join(str(window) for window in DEFAULT_WINDOWS),
        help="Comma-separated list of window sizes in seconds.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_windows(raw_value: str) -> list[int]:
    windows = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        windows.append(int(token))
    if not windows:
        raise ValueError("At least one window size is required.")
    return windows


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def build_combined_outputs(output_dir: Path, windows: list[int]) -> None:
    combined_window_rows: list[dict] = []
    combined_backend_rows: list[dict] = []

    for window in windows:
        window_dir = output_dir / f"window_{window}s"
        validation_rows = load_csv_rows(window_dir / "validation_metrics.csv")
        test_rows = load_csv_rows(window_dir / "test_metrics.csv")
        thresholds = json.loads((window_dir / "thresholds.json").read_text(encoding="utf-8"))
        test_by_backend = {row["backend"]: row for row in test_rows}
        winner = next(row for row in validation_rows if row["winner"] == "True")

        combined_window_rows.append(
            {
                "window_s": window,
                "validation_winner": winner["backend"],
                "validation_winner_f1": to_float(winner, "f1"),
                "validation_winner_false_alerts_per_min": to_float(winner, "false_alerts_per_min"),
                "validation_winner_false_alert_time_ratio": to_float(winner, "false_alert_time_ratio"),
                "validation_winner_ttfd_s": to_float(winner, "time_to_first_detection_s"),
                "validation_mean_f1": mean(to_float(row, "f1") for row in validation_rows),
                "validation_mean_false_alerts_per_min": mean(to_float(row, "false_alerts_per_min") for row in validation_rows),
                "validation_mean_false_alert_time_ratio": mean(to_float(row, "false_alert_time_ratio") for row in validation_rows),
                "validation_mean_ttfd_s": mean(to_float(row, "time_to_first_detection_s") for row in validation_rows),
                "test_mean_f1": mean(to_float(row, "f1") for row in test_rows),
                "test_mean_false_alerts_per_min": mean(to_float(row, "false_alerts_per_min") for row in test_rows),
                "test_mean_false_alert_time_ratio": mean(to_float(row, "false_alert_time_ratio") for row in test_rows),
                "test_mean_ttfd_s": mean(to_float(row, "time_to_first_detection_s") for row in test_rows),
            }
        )

        for validation_row in validation_rows:
            backend = validation_row["backend"]
            test_row = test_by_backend[backend]
            selected = thresholds[backend]["selected"]
            combined_backend_rows.append(
                {
                    "window_s": window,
                    "backend": backend,
                    "rank": int(validation_row["rank"]),
                    "winner": validation_row["winner"] == "True",
                    "polarity": validation_row["polarity"],
                    "selected_policy": validation_row["selected_policy"],
                    "threshold": selected["threshold"],
                    "t_on": selected["t_on"],
                    "t_off": selected["t_off"],
                    "persistence_windows": selected["persistence_windows"],
                    "calibration_real_only_fa_per_min": selected["real_only_metrics"]["false_alerts_per_min"],
                    "calibration_real_only_false_alert_time_ratio": selected["real_only_metrics"]["false_alert_time_ratio"],
                    "validation_f1": to_float(validation_row, "f1"),
                    "validation_false_alerts_per_min": to_float(validation_row, "false_alerts_per_min"),
                    "validation_false_alert_time_ratio": to_float(validation_row, "false_alert_time_ratio"),
                    "validation_ttfd_s": to_float(validation_row, "time_to_first_detection_s"),
                    "test_f1": to_float(test_row, "f1"),
                    "test_false_alerts_per_min": to_float(test_row, "false_alerts_per_min"),
                    "test_false_alert_time_ratio": to_float(test_row, "false_alert_time_ratio"),
                    "test_ttfd_s": to_float(test_row, "time_to_first_detection_s"),
                }
            )

    write_csv(
        output_dir / "combined_window_summary.csv",
        combined_window_rows,
        [
            "window_s",
            "validation_winner",
            "validation_winner_f1",
            "validation_winner_false_alerts_per_min",
            "validation_winner_false_alert_time_ratio",
            "validation_winner_ttfd_s",
            "validation_mean_f1",
            "validation_mean_false_alerts_per_min",
            "validation_mean_false_alert_time_ratio",
            "validation_mean_ttfd_s",
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
            "window_s",
            "backend",
            "rank",
            "winner",
            "polarity",
            "selected_policy",
            "threshold",
            "t_on",
            "t_off",
            "persistence_windows",
            "calibration_real_only_fa_per_min",
            "calibration_real_only_false_alert_time_ratio",
            "validation_f1",
            "validation_false_alerts_per_min",
            "validation_false_alert_time_ratio",
            "validation_ttfd_s",
            "test_f1",
            "test_false_alerts_per_min",
            "test_false_alert_time_ratio",
            "test_ttfd_s",
        ],
    )

    best_by_backend: list[dict] = []
    by_backend: dict[str, list[dict]] = {}
    for row in combined_backend_rows:
        by_backend.setdefault(row["backend"], []).append(row)
    for backend, rows in sorted(by_backend.items()):
        best = max(
            rows,
            key=lambda row: (
                row["validation_f1"],
                -row["validation_false_alerts_per_min"],
                -row["validation_false_alert_time_ratio"],
                -row["validation_ttfd_s"],
                -row["test_f1"],
            ),
        )
        best_by_backend.append(
            {
                "backend": backend,
                "best_window_s": best["window_s"],
                "validation_f1": best["validation_f1"],
                "validation_false_alerts_per_min": best["validation_false_alerts_per_min"],
                "validation_false_alert_time_ratio": best["validation_false_alert_time_ratio"],
                "validation_ttfd_s": best["validation_ttfd_s"],
                "test_f1": best["test_f1"],
            }
        )

    write_csv(
        output_dir / "window_best_combo_by_model_fa050.csv",
        best_by_backend,
        [
            "backend",
            "best_window_s",
            "validation_f1",
            "validation_false_alerts_per_min",
            "validation_false_alert_time_ratio",
            "validation_ttfd_s",
            "test_f1",
        ],
    )

    lines = [
        "# FA0.50 V2 Full Window Sweep Summary",
        "",
        "| window_s | validation winner | winner F1 | winner FA/min | winner FA time ratio | winner TTFD s | mean val F1 | mean val FA/min | mean val FA time ratio | mean val TTFD s | mean test F1 | mean test FA/min | mean test FA time ratio | mean test TTFD s |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in combined_window_rows:
        lines.append(
            f"| {row['window_s']} | `{row['validation_winner']}` | {row['validation_winner_f1']:.4f} | "
            f"{row['validation_winner_false_alerts_per_min']:.4f} | {row['validation_winner_false_alert_time_ratio']:.4f} | "
            f"{row['validation_winner_ttfd_s']:.4f} | {row['validation_mean_f1']:.4f} | "
            f"{row['validation_mean_false_alerts_per_min']:.4f} | {row['validation_mean_false_alert_time_ratio']:.4f} | "
            f"{row['validation_mean_ttfd_s']:.4f} | {row['test_mean_f1']:.4f} | "
            f"{row['test_mean_false_alerts_per_min']:.4f} | {row['test_mean_false_alert_time_ratio']:.4f} | "
            f"{row['test_mean_ttfd_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Validation-F1 Window Per Backend",
            "",
            "| backend | best window_s | val F1 | val FA/min | val FA time ratio | val TTFD s | test F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best_by_backend:
        lines.append(
            f"| `{row['backend']}` | {row['best_window_s']} | {row['validation_f1']:.4f} | "
            f"{row['validation_false_alerts_per_min']:.4f} | {row['validation_false_alert_time_ratio']:.4f} | "
            f"{row['validation_ttfd_s']:.4f} | {row['test_f1']:.4f} |"
        )
    lines.append("")
    (output_dir / "window_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary).resolve()
    sessions_index_path = Path(args.sessions_index).resolve()
    eval_root = Path(args.eval_root).resolve()
    windows = parse_windows(args.windows)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        run_variant_with_paths(
            config=replace(
                BASE_CONFIG,
                name=f"fa050_{window}s",
                output_dir_default=str(output_dir / f"window_{window}s"),
                window_seconds=window,
                summary_title=f"Benchmark Scoring Report V2 FA0.50 Gate {window}s",
            ),
            summary_path=summary_path,
            sessions_index_path=sessions_index_path,
            eval_root=eval_root,
            output_dir=output_dir / f"window_{window}s",
        )

    build_combined_outputs(output_dir, windows)


if __name__ == "__main__":
    main()
