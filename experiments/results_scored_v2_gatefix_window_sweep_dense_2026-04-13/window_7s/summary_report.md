# Benchmark Scoring Report V2

## Method

- Evaluation unit: 7-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 7-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0786
- Validation time-to-first-detection: 17.0032s
- Validation F1: 0.0881

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 1 | 0.88 | 0.83 | 0.1000 | 0.0117 | 0.1236 | 14.4433 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.38 | 0.33 | 0.1000 | 0.0083 | 0.1583 | 14.1336 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.61 | 0.51 | 0.1000 | 0.0467 | 0.4633 | 9.0858 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.33 | 0.33 | 0.1000 | 0.0350 | 0.1815 | 13.6833 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.70 | 0.60 | 0.1000 | 0.0467 | 0.3552 | 10.7460 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.73 | 0.73 | 0.1000 | 0.0350 | 0.0811 | 15.6772 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6250 | 0.0474 | 0.0881 | 0.0786 | 0.0275 | 17.0032 | 4.2857 |
| 2 | `effort_clip_l14` | 0.6061 | 0.1896 | 0.2888 | 0.2750 | 0.1166 | 13.8794 | 2.8821 |
| 3 | `efficientnet_b4` | 0.6429 | 0.0853 | 0.1506 | 0.3143 | 0.0458 | 15.5155 | 8.6598 |
| 4 | `i3d` | 0.5306 | 0.1232 | 0.2000 | 0.4322 | 0.1054 | 15.1488 | 3.1487 |
| 5 | `videomae` | 0.5051 | 0.2370 | 0.3226 | 0.5894 | 0.2233 | 12.7050 | 2.2642 |
| 6 | `f3net` | 0.6048 | 0.4787 | 0.5344 | 0.8251 | 0.3012 | 9.0127 | 1.3882 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6818 | 0.2123 | 0.3237 | 0.4348 | 0.0968 | 12.7683 | 4.8052 |
| `effort_clip_l14` | 0.2200 | 0.0519 | 0.0840 | 0.4348 | 0.1798 | 16.6597 | 4.1143 |
| `f3net` | 0.6081 | 0.4245 | 0.5000 | 0.8300 | 0.2661 | 10.2822 | 1.6279 |
| `i3d` | 0.5098 | 0.1226 | 0.1977 | 0.3953 | 0.1153 | 15.7375 | 3.3613 |
| `videomae` | 0.5183 | 0.4009 | 0.4521 | 0.8300 | 0.3617 | 10.7035 | 1.6287 |
| `xception_df40` | 0.5714 | 0.0566 | 0.1030 | 0.1581 | 0.0402 | 16.8172 | 2.8966 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4907 | 1.0000 | 0.6583 | 1.9646 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4930 | 1.0000 | 0.6604 | 2.0553 | 1.0000 | 0.0000 |
