# Benchmark Scoring Report V2

## Method

- Evaluation unit: 58-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 58-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.1069
- Validation time-to-first-detection: 14.3693s
- Validation F1: 0.3111

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.5000 | 9.5565 |
| `effort_clip_l14` | `strict_0.10` | `low` | 1 | 0.25 | 0.15 | 0.1000 | 0.0333 | 0.5312 | 8.7644 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.49 | 0.49 | 0.0000 | 0.0000 | 0.4375 | 10.5087 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7188 | 6.8297 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.55 | 0.45 | 0.1000 | 0.0333 | 0.5938 | 7.4594 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.17 | 0.17 | 0.1000 | 0.0333 | 0.6562 | 5.7035 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.3333 | 0.2917 | 0.3111 | 0.1069 | 0.4371 | 14.3693 | 0.2251 |
| 2 | `videomae` | 0.5000 | 0.5000 | 0.5000 | 0.1425 | 0.3682 | 10.0179 | 0.2347 |
| 3 | `xception_df40` | 0.4348 | 0.4167 | 0.4255 | 0.2138 | 0.4026 | 9.7535 | 0.3443 |
| 4 | `efficientnet_b4` | 0.3333 | 0.3750 | 0.3529 | 0.2138 | 0.5974 | 10.4729 | 0.2893 |
| 5 | `effort_clip_l14` | 0.5263 | 0.4167 | 0.4651 | 0.2494 | 0.2648 | 10.2576 | 0.8772 |
| 6 | `f3net` | 0.4500 | 0.7500 | 0.5625 | 0.3207 | 0.7126 | 4.0987 | 0.1661 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4615 | 0.2400 | 0.3158 | 0.1134 | 0.1839 | 14.2038 | 0.3987 |
| `effort_clip_l14` | 0.3158 | 0.2400 | 0.2727 | 0.2645 | 0.4748 | 13.1292 | 1.1434 |
| `f3net` | 0.4118 | 0.2800 | 0.3333 | 0.1889 | 0.2935 | 12.2147 | 0.2878 |
| `i3d` | 0.4286 | 0.4800 | 0.4528 | 0.1134 | 0.5365 | 9.1590 | 0.1192 |
| `videomae` | 0.4516 | 0.5600 | 0.5000 | 0.1511 | 0.5491 | 8.1216 | 0.2187 |
| `xception_df40` | 0.3871 | 0.4800 | 0.4286 | 0.1511 | 0.6222 | 8.3599 | 0.1458 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4364 | 1.0000 | 0.6076 | 0.3919 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4545 | 1.0000 | 0.6250 | 0.3401 | 1.0000 | 0.0000 |
