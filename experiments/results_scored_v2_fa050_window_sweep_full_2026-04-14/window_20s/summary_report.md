# Benchmark Scoring Report V2 FA0.50 Gate 20s

## Method

- Evaluation unit: 20-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 20-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `videomae`
- Validation F1: 0.4341
- Validation false alerts/min: 0.2368
- Validation time-to-first-detection: 10.7878s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 3 | 0.67 | 0.57 | 0.1000 | 0.2667 | 0.6915 | 0.6341 | 4.5827 |
| `effort_clip_l14` | `strict_0.50` | `high` | 3 | 0.97 | 0.87 | 0.1000 | 0.3000 | 0.3830 | 0.4737 | 9.8195 |
| `f3net` | `strict_0.50` | `high` | 3 | 0.44 | 0.44 | 0.2000 | 0.3000 | 0.6489 | 0.6256 | 5.6090 |
| `i3d` | `strict_0.50` | `low` | 2 | 0.32 | 0.27 | 0.1000 | 0.0667 | 0.7128 | 0.6802 | 4.7333 |
| `videomae` | `strict_0.50` | `low` | 3 | 0.58 | 0.48 | 0.1000 | 0.0333 | 0.4894 | 0.5679 | 8.3040 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.24 | 0.19 | 0.2000 | 0.3000 | 0.5319 | 0.5917 | 7.5382 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.5091 | 0.3784 | 0.4341 | 0.2368 | 0.3553 | 10.7878 | 0.4909 |
| 2 | `i3d` | 0.3676 | 0.3378 | 0.3521 | 0.3553 | 0.5658 | 12.2314 | 0.1765 |
| 3 | `effort_clip_l14` | 0.6471 | 0.4459 | 0.5280 | 0.4737 | 0.2368 | 10.3429 | 0.5294 |
| 4 | `xception_df40` | 0.5345 | 0.4189 | 0.4697 | 0.6316 | 0.3553 | 10.1674 | 0.5172 |
| 5 | `efficientnet_b4` | 0.4819 | 0.5405 | 0.5096 | 0.7500 | 0.5658 | 7.9352 | 0.3253 |
| 6 | `f3net` | 0.5472 | 0.7838 | 0.6444 | 0.8289 | 0.6316 | 3.9747 | 0.3113 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4603 | 0.7945 | 0.5829 | 0.9740 | 0.8831 | 3.2045 | 0.1905 |
| `effort_clip_l14` | 0.6154 | 0.1096 | 0.1860 | 0.1558 | 0.0649 | 16.2327 | 0.9231 |
| `f3net` | 0.5208 | 0.6849 | 0.5917 | 0.7792 | 0.5974 | 5.3128 | 0.2812 |
| `i3d` | 0.4766 | 0.8356 | 0.6070 | 0.9740 | 0.8701 | 3.1077 | 0.1172 |
| `videomae` | 0.4767 | 0.5616 | 0.5157 | 0.5455 | 0.5844 | 7.3827 | 0.4884 |
| `xception_df40` | 0.7500 | 0.3288 | 0.4571 | 0.1948 | 0.1039 | 11.9834 | 0.7500 |
