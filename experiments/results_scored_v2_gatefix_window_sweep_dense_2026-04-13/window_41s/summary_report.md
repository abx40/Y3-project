# Benchmark Scoring Report V2

## Method

- Evaluation unit: 41-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 41-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.0815
- Validation time-to-first-detection: 16.4772s
- Validation F1: 0.1333

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.70 | 0.60 | 0.0000 | 0.0000 | 0.3810 | 9.6137 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.1190 | 14.5317 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.6905 | 5.1885 |
| `i3d` | `strict_0.10` | `high` | 2 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.6429 | 5.4496 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.56 | 0.46 | 0.0000 | 0.0000 | 0.4286 | 10.3583 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.19 | 0.14 | 0.1000 | 0.0433 | 0.5000 | 8.7383 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.4286 | 0.0789 | 0.1333 | 0.0815 | 0.1012 | 16.4772 | 0.6618 |
| 2 | `effort_clip_l14` | 0.6667 | 0.3158 | 0.4286 | 0.1630 | 0.1569 | 11.5960 | 0.9129 |
| 3 | `xception_df40` | 0.4231 | 0.2895 | 0.3438 | 0.2853 | 0.3974 | 11.6963 | 0.2938 |
| 4 | `efficientnet_b4` | 0.2667 | 0.2105 | 0.2353 | 0.2853 | 0.5822 | 12.6101 | 0.3544 |
| 5 | `i3d` | 0.5208 | 0.6579 | 0.5814 | 0.3668 | 0.6101 | 5.3886 | 0.2219 |
| 6 | `f3net` | 0.5455 | 0.7895 | 0.6452 | 0.4076 | 0.6861 | 3.6333 | 0.2443 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6364 | 0.6667 | 0.6512 | 0.3712 | 0.4957 | 5.9465 | 0.3689 |
| `effort_clip_l14` | 0.3333 | 0.0476 | 0.0833 | 0.1392 | 0.1152 | 16.6455 | 2.2222 |
| `f3net` | 0.7250 | 0.6905 | 0.7073 | 0.3712 | 0.3140 | 7.1375 | 0.3038 |
| `i3d` | 0.3939 | 0.3095 | 0.3467 | 0.3248 | 0.5878 | 12.7827 | 0.2347 |
| `videomae` | 0.5769 | 0.3571 | 0.4412 | 0.2320 | 0.3140 | 11.9384 | 0.2938 |
| `xception_df40` | 0.5938 | 0.4524 | 0.5135 | 0.2320 | 0.3774 | 9.3672 | 0.1917 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5067 | 1.0000 | 0.6726 | 0.5299 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5600 | 1.0000 | 0.7179 | 0.5104 | 1.0000 | 0.0000 |
