# Benchmark Scoring Report V2

## Method

- Evaluation unit: 46-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 46-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1053
- Validation time-to-first-detection: 16.0146s
- Validation F1: 0.1026

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.38 | 0.28 | 0.1000 | 0.0033 | 0.4318 | 9.8394 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.88 | 0.83 | 0.0000 | 0.0000 | 0.2500 | 13.4726 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.7273 | 4.9100 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6591 | 5.6584 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.62 | 0.57 | 0.0000 | 0.0000 | 0.4545 | 9.0025 |
| `xception_df40` | `strict_0.10` | `low` | 1 | 0.51 | 0.51 | 0.0000 | 0.0000 | 0.1818 | 14.4933 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.2000 | 0.0690 | 0.1026 | 0.1053 | 0.1895 | 16.0146 | 0.7212 |
| 2 | `xception_df40` | 0.2500 | 0.0690 | 0.1081 | 0.1404 | 0.1099 | 15.4212 | 2.1429 |
| 3 | `i3d` | 0.1600 | 0.1379 | 0.1481 | 0.1754 | 0.4877 | 15.1341 | 0.1768 |
| 4 | `effort_clip_l14` | 0.4348 | 0.3448 | 0.3846 | 0.2105 | 0.2982 | 10.7172 | 0.2474 |
| 5 | `efficientnet_b4` | 0.4857 | 0.5862 | 0.5312 | 0.2456 | 0.4327 | 8.0076 | 0.4465 |
| 6 | `f3net` | 0.4211 | 0.5517 | 0.4776 | 0.3158 | 0.5661 | 7.3619 | 0.2892 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.3913 | 0.2812 | 0.3273 | 0.2290 | 0.3257 | 12.9150 | 0.5442 |
| `effort_clip_l14` | 0.6207 | 0.5625 | 0.5902 | 0.1908 | 0.2659 | 8.4933 | 0.2496 |
| `f3net` | 0.5556 | 0.4688 | 0.5085 | 0.3053 | 0.2952 | 9.5213 | 0.4865 |
| `i3d` | 0.4167 | 0.7812 | 0.5435 | 0.4962 | 0.9122 | 3.5039 | 0.1181 |
| `videomae` | 0.5789 | 0.3438 | 0.4314 | 0.1527 | 0.1781 | 11.8650 | 0.5660 |
| `xception_df40` | 0.3750 | 0.0938 | 0.1500 | 0.1145 | 0.1183 | 15.0701 | 2.0370 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4143 | 1.0000 | 0.5859 | 0.5263 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4571 | 1.0000 | 0.6275 | 0.5344 | 1.0000 | 0.0000 |
