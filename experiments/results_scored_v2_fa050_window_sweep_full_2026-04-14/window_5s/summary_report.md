# Benchmark Scoring Report V2 FA0.50 Gate 5s

## Method

- Evaluation unit: 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 5-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `efficientnet_b4`
- Validation F1: 0.2952
- Validation false alerts/min: 0.5235
- Validation time-to-first-detection: 13.6750s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 3 | 0.81 | 0.71 | 0.5000 | 0.2417 | 0.1917 | 0.2918 | 13.3467 |
| `effort_clip_l14` | `strict_0.50` | `low` | 3 | 0.38 | 0.28 | 0.5000 | 0.1333 | 0.1722 | 0.2690 | 13.9845 |
| `f3net` | `strict_0.50` | `high` | 3 | 0.54 | 0.49 | 0.4000 | 0.2917 | 0.5556 | 0.5857 | 6.4560 |
| `i3d` | `strict_0.50` | `low` | 1 | 0.37 | 0.27 | 0.4000 | 0.1583 | 0.4611 | 0.5321 | 8.5431 |
| `videomae` | `strict_0.50` | `low` | 3 | 0.70 | 0.60 | 0.4000 | 0.1583 | 0.3528 | 0.4644 | 10.2665 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.36 | 0.26 | 0.5000 | 0.2500 | 0.3417 | 0.4323 | 10.7692 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `efficientnet_b4` | 0.5254 | 0.2053 | 0.2952 | 0.5235 | 0.1879 | 13.6750 | 2.9492 |
| 2 | `effort_clip_l14` | 0.5546 | 0.2185 | 0.3135 | 0.6846 | 0.1779 | 13.4359 | 3.4286 |
| 3 | `xception_df40` | 0.5027 | 0.3046 | 0.3794 | 0.9262 | 0.3054 | 11.8787 | 2.8852 |
| 4 | `videomae` | 0.5886 | 0.3079 | 0.4043 | 0.9664 | 0.2181 | 11.8563 | 3.4177 |
| 5 | `i3d` | 0.5249 | 0.4536 | 0.4867 | 1.0470 | 0.4161 | 9.1746 | 1.2414 |
| 6 | `f3net` | 0.5584 | 0.5861 | 0.5719 | 1.6913 | 0.4698 | 5.9686 | 1.8549 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6389 | 0.4525 | 0.5298 | 1.0576 | 0.2644 | 9.4852 | 2.0000 |
| `effort_clip_l14` | 0.3558 | 0.1213 | 0.1809 | 0.7322 | 0.2271 | 15.8296 | 3.9231 |
| `f3net` | 0.6033 | 0.5934 | 0.5983 | 1.6678 | 0.4034 | 6.7721 | 1.9200 |
| `i3d` | 0.5172 | 0.5902 | 0.5513 | 1.5458 | 0.5695 | 7.2965 | 0.7241 |
| `videomae` | 0.5451 | 0.4557 | 0.4964 | 1.0576 | 0.3932 | 9.4470 | 1.9294 |
| `xception_df40` | 0.4765 | 0.2656 | 0.3411 | 0.9763 | 0.3017 | 12.6977 | 3.1059 |
