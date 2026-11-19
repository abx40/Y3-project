# Benchmark Scoring Report V2

## Method

- Evaluation unit: 14-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 14-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1218
- Validation time-to-first-detection: 16.6756s
- Validation F1: 0.1157

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.75 | 0.65 | 0.0000 | 0.0000 | 0.2946 | 12.2121 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2171 | 13.4196 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.60 | 0.50 | 0.1000 | 0.0467 | 0.4574 | 8.9460 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7132 | 4.9639 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.72 | 0.62 | 0.0000 | 0.0000 | 0.3023 | 11.7065 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.50 | 0.45 | 0.0000 | 0.0000 | 0.1318 | 14.9410 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5833 | 0.0642 | 0.1157 | 0.1218 | 0.0460 | 16.6756 | 1.8072 |
| 2 | `videomae` | 0.5946 | 0.2018 | 0.3014 | 0.4060 | 0.1407 | 13.9277 | 1.9767 |
| 3 | `effort_clip_l14` | 0.7222 | 0.3578 | 0.4785 | 0.4871 | 0.1421 | 10.8816 | 0.9524 |
| 4 | `efficientnet_b4` | 0.5455 | 0.2202 | 0.3137 | 0.4871 | 0.1881 | 13.1020 | 1.0749 |
| 5 | `i3d` | 0.4312 | 0.4312 | 0.4312 | 0.5277 | 0.5832 | 10.2076 | 0.1581 |
| 6 | `f3net` | 0.5435 | 0.4587 | 0.4975 | 0.9743 | 0.3965 | 8.9389 | 1.4463 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5000 | 0.5421 | 0.5202 | 1.2367 | 0.5386 | 7.1654 | 0.5549 |
| `effort_clip_l14` | 0.6176 | 0.3925 | 0.4800 | 0.8777 | 0.2407 | 10.7151 | 1.2658 |
| `f3net` | 0.6533 | 0.4579 | 0.5385 | 0.7181 | 0.2407 | 9.8360 | 1.3767 |
| `i3d` | 0.4643 | 0.7290 | 0.5673 | 1.4362 | 0.8324 | 4.8272 | 0.1281 |
| `videomae` | 0.5588 | 0.3551 | 0.4343 | 0.5585 | 0.2779 | 11.2876 | 1.4526 |
| `xception_df40` | 0.8261 | 0.1776 | 0.2923 | 0.1197 | 0.0372 | 14.6071 | 1.8634 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5070 | 1.0000 | 0.6728 | 1.8268 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4977 | 1.0000 | 0.6646 | 1.9149 | 1.0000 | 0.0000 |
