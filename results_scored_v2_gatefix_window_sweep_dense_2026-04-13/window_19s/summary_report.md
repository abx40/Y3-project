# Benchmark Scoring Report V2

## Method

- Evaluation unit: 19-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 19-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.3168
- Validation time-to-first-detection: 10.1957s
- Validation F1: 0.4500

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.73 | 0.63 | 0.0000 | 0.0000 | 0.6304 | 6.7452 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.1413 | 14.3805 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.51 | 0.41 | 0.0000 | 0.0000 | 0.6196 | 6.3695 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.33 | 0.23 | 0.0000 | 0.0000 | 0.7391 | 4.8807 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.65 | 0.55 | 0.0000 | 0.0000 | 0.4239 | 9.8743 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.28 | 0.18 | 0.0000 | 0.0000 | 0.5109 | 8.9630 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.4444 | 0.4557 | 0.4500 | 0.3168 | 0.5485 | 10.1957 | 0.1593 |
| 2 | `effort_clip_l14` | 0.7576 | 0.3165 | 0.4464 | 0.3168 | 0.0950 | 12.1408 | 0.8724 |
| 3 | `xception_df40` | 0.4706 | 0.2025 | 0.2832 | 0.3564 | 0.2152 | 13.9426 | 1.0611 |
| 4 | `videomae` | 0.4000 | 0.1772 | 0.2456 | 0.3564 | 0.2581 | 14.3719 | 1.1872 |
| 5 | `efficientnet_b4` | 0.4474 | 0.2152 | 0.2906 | 0.4356 | 0.2581 | 14.1371 | 0.3399 |
| 6 | `f3net` | 0.6222 | 0.7089 | 0.6627 | 0.7129 | 0.4158 | 5.7535 | 0.4291 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.3605 | 0.4306 | 0.3924 | 0.6951 | 0.6226 | 9.1989 | 0.3354 |
| `effort_clip_l14` | 0.4375 | 0.0972 | 0.1591 | 0.2561 | 0.0994 | 16.0150 | 1.4189 |
| `f3net` | 0.4533 | 0.4722 | 0.4626 | 0.5122 | 0.4555 | 9.5953 | 0.2166 |
| `i3d` | 0.4000 | 0.5278 | 0.4551 | 0.5122 | 0.6457 | 8.4589 | 0.1354 |
| `videomae` | 0.4783 | 0.4583 | 0.4681 | 0.6220 | 0.4122 | 9.9592 | 1.0193 |
| `xception_df40` | 0.4912 | 0.3889 | 0.4341 | 0.2927 | 0.3360 | 11.0797 | 0.4432 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4938 | 1.0000 | 0.6611 | 1.2673 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4500 | 1.0000 | 0.6207 | 1.2805 | 1.0000 | 0.0000 |
