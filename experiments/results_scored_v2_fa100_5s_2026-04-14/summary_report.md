# Benchmark Scoring Report V2 FA1.00 Gate 5s

## Method

- Evaluation unit: 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 5-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 1.00 FA/min and <= 0.60 real-time alert ratio on `calibration_real_only_01`; relaxed <= 1.50 FA/min and <= 0.90 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 1.00 and FA time ratio <= 0.60 (relaxed: FA/min <= 1.50 and FA time ratio <= 0.90), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `effort_clip_l14`
- Validation F1: 0.4902
- Validation false alerts/min: 1.4094
- Validation time-to-first-detection: 7.5215s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_1.00` | `low` | 3 | 0.66 | 0.56 | 0.5000 | 0.5167 | 0.6861 | 0.6325 | 4.4373 |
| `effort_clip_l14` | `strict_1.00` | `low` | 3 | 0.02 | 0.02 | 0.7000 | 0.5667 | 0.6361 | 0.6034 | 5.9485 |
| `f3net` | `strict_1.00` | `high` | 3 | 0.39 | 0.39 | 0.9000 | 0.6000 | 0.7778 | 0.6931 | 3.1742 |
| `i3d` | `strict_1.00` | `low` | 3 | 0.29 | 0.29 | 0.8000 | 0.6000 | 0.5472 | 0.5846 | 7.1975 |
| `videomae` | `strict_1.00` | `low` | 2 | 0.44 | 0.44 | 1.0000 | 0.6000 | 0.6444 | 0.6356 | 4.5765 |
| `xception_df40` | `strict_1.00` | `low` | 3 | 0.15 | 0.10 | 0.5000 | 0.5833 | 0.6389 | 0.6259 | 5.5180 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.4490 | 0.5397 | 0.4902 | 1.4094 | 0.6711 | 7.5215 | 1.6198 |
| 2 | `i3d` | 0.5000 | 0.4603 | 0.4793 | 1.4094 | 0.4664 | 8.4324 | 2.6763 |
| 3 | `xception_df40` | 0.4805 | 0.4901 | 0.4852 | 1.5705 | 0.5369 | 8.3635 | 2.0260 |
| 4 | `efficientnet_b4` | 0.5012 | 0.7086 | 0.5871 | 1.8121 | 0.7148 | 4.3822 | 1.1803 |
| 5 | `videomae` | 0.4986 | 0.5861 | 0.5388 | 1.8926 | 0.5973 | 5.1834 | 3.2451 |
| 6 | `f3net` | 0.5500 | 0.8013 | 0.6523 | 1.9732 | 0.6644 | 2.1243 | 1.3364 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5086 | 0.6820 | 0.5826 | 2.0339 | 0.6814 | 5.3783 | 1.2323 |
| `effort_clip_l14` | 0.4868 | 0.6033 | 0.5388 | 1.5458 | 0.6576 | 6.8038 | 1.9048 |
| `f3net` | 0.5320 | 0.7902 | 0.6359 | 2.1966 | 0.7186 | 3.4316 | 1.1921 |
| `i3d` | 0.5208 | 0.5344 | 0.5275 | 1.5864 | 0.5085 | 7.5326 | 2.5304 |
| `videomae` | 0.5217 | 0.7082 | 0.6008 | 1.8712 | 0.6712 | 4.1110 | 1.9130 |
| `xception_df40` | 0.5075 | 0.5541 | 0.5298 | 1.4644 | 0.5559 | 7.2958 | 1.5856 |
