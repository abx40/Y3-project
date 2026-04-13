# Benchmark Scoring Report V2

## Method

- Evaluation unit: 32-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 32-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1408
- Validation time-to-first-detection: 11.1056s
- Validation F1: 0.3714

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.35 | 0.30 | 0.0000 | 0.0000 | 0.2985 | 12.5413 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2388 | 12.9868 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.51 | 0.41 | 0.0000 | 0.0000 | 0.6716 | 5.2291 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7313 | 4.7231 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.45 | 0.45 | 0.0000 | 0.0000 | 0.5821 | 6.3174 |
| `xception_df40` | `strict_0.10` | `high` | 3 | 0.83 | 0.73 | 0.0000 | 0.0000 | 0.1642 | 15.1031 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.4483 | 0.3171 | 0.3714 | 0.1408 | 0.2911 | 11.1056 | 0.4646 |
| 2 | `effort_clip_l14` | 0.5789 | 0.2683 | 0.3667 | 0.1761 | 0.1455 | 11.9011 | 1.3000 |
| 3 | `efficientnet_b4` | 0.4833 | 0.7073 | 0.5743 | 0.2817 | 0.5775 | 5.6667 | 0.2215 |
| 4 | `i3d` | 0.2632 | 0.3659 | 0.3061 | 0.2817 | 0.7746 | 10.9971 | 0.1339 |
| 5 | `videomae` | 0.4478 | 0.7317 | 0.5556 | 0.3521 | 0.6808 | 5.0281 | 0.2273 |
| 6 | `f3net` | 0.4746 | 0.6829 | 0.5600 | 0.3521 | 0.5728 | 5.9451 | 0.3879 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5263 | 0.2381 | 0.3279 | 0.1082 | 0.1587 | 14.2102 | 0.4167 |
| `effort_clip_l14` | 0.4706 | 0.1905 | 0.2712 | 0.2163 | 0.1683 | 13.3356 | 2.0455 |
| `f3net` | 0.4490 | 0.5238 | 0.4835 | 0.3606 | 0.5048 | 8.2685 | 0.3906 |
| `i3d` | 0.4024 | 0.7857 | 0.5323 | 0.4327 | 0.9231 | 3.1203 | 0.1161 |
| `videomae` | 0.4286 | 0.6429 | 0.5143 | 0.3606 | 0.6779 | 6.3481 | 0.2419 |
| `xception_df40` | 0.4062 | 0.3095 | 0.3514 | 0.2524 | 0.3558 | 11.3567 | 0.1800 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4316 | 1.0000 | 0.6029 | 0.4225 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4421 | 1.0000 | 0.6131 | 0.4688 | 1.0000 | 0.0000 |
