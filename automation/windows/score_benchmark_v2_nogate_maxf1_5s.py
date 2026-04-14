import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from automation.windows.score_benchmark_v2_variant_lib import VariantConfig, run_variant


CONFIG = VariantConfig(
    name="nogate_maxf1_5s",
    output_dir_default="results_scored_v2_nogate_maxf1_5s_2026-04-13",
    window_seconds=5,
    summary_default="results_rgb_fix_full_2026-04-09/summary_final_clean.csv",
    sessions_index_default=r"C:\deepfake_eval\leakfree_eval\sessions_index.csv",
    eval_root_default=r"C:\deepfake_eval\leakfree_eval",
    strict_false_alerts_per_min=None,
    strict_fa_time_ratio=None,
    relaxed_false_alerts_per_min=None,
    relaxed_fa_time_ratio=None,
    selection_rule=(
        "No calibration safety gate. Select the operating point that maximizes F1 on calibration_fake_only_01 plus "
        "calibration mixed sessions, then minimize TTFD, then prefer lower calibration real-only false alerts."
    ),
    validation_ranking_rule="F1 desc, time_to_first_detection_s asc",
    summary_title="Benchmark Scoring Report V2 No Gate Max F1 5s",
    summary_gate_text="No false-alert gate is applied during calibration operating-point selection.",
    selection_mode="nogate_maxf1",
)


if __name__ == "__main__":
    run_variant(CONFIG)
