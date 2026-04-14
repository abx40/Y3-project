import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows.score_benchmark_v2_variant_lib import VariantConfig, run_variant


CONFIG = VariantConfig(
    name="fa050_5s",
    output_dir_default="results_scored_v2_fa050_5s_2026-04-14",
    window_seconds=5,
    summary_default="results_rgb_fix_full_2026-04-09/summary_final_clean.csv",
    sessions_index_default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv",
    eval_root_default=r"C:\deepfake_eval\leakfree_eval",
    strict_false_alerts_per_min=0.50,
    strict_fa_time_ratio=0.30,
    relaxed_false_alerts_per_min=0.75,
    relaxed_fa_time_ratio=0.45,
    selection_rule=(
        "Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 "
        "(relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on "
        "calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1."
    ),
    validation_ranking_rule="false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc",
    summary_title="Benchmark Scoring Report V2 FA0.50 Gate 5s",
    summary_gate_text=(
        "False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; "
        "relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio."
    ),
    selection_mode="gate",
)


if __name__ == "__main__":
    run_variant(CONFIG)
