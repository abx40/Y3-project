# Benchmark Scoring Report V2

## Method

- Evaluation unit: 42-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 42-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.0820
- Validation time-to-first-detection: 16.3005s
- Validation F1: 0.2083

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.39 | 0.29 | 0.0000 | 0.0000 | 0.4583 | 10.0090 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.89 | 0.84 | 0.0000 | 0.0000 | 0.3125 | 12.0571 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.51 | 0.51 | 0.0000 | 0.0000 | 0.6458 | 7.2434 |
| `i3d` | `strict_0.10` | `high` | 2 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.6458 | 6.4208 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.59 | 0.54 | 0.1000 | 0.0200 | 0.5208 | 7.4747 |
| `xception_df40` | `strict_0.10` | `high` | 3 | 0.75 | 0.65 | 0.0000 | 0.0000 | 0.3542 | 12.0540 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.5000 | 0.1316 | 0.2083 | 0.0820 | 0.1230 | 16.3005 | 0.7692 |
| 2 | `effort_clip_l14` | 0.3704 | 0.2632 | 0.3077 | 0.1639 | 0.4672 | 12.0538 | 0.3804 |
| 3 | `efficientnet_b4` | 0.4375 | 0.1842 | 0.2593 | 0.1639 | 0.2377 | 14.8327 | 0.7843 |
| 4 | `f3net` | 0.5588 | 0.5000 | 0.5278 | 0.2049 | 0.4098 | 10.1355 | 0.4036 |
| 5 | `xception_df40` | 0.4062 | 0.3421 | 0.3714 | 0.2049 | 0.5041 | 11.9461 | 0.3349 |
| 6 | `i3d` | 0.5116 | 0.5789 | 0.5432 | 0.3279 | 0.5410 | 7.4839 | 0.2536 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5833 | 0.3784 | 0.4590 | 0.1626 | 0.2236 | 12.3434 | 0.4054 |
| `effort_clip_l14` | 0.6250 | 0.5405 | 0.5797 | 0.2439 | 0.3008 | 7.9655 | 0.2392 |
| `f3net` | 0.5714 | 0.3243 | 0.4138 | 0.1626 | 0.2154 | 12.0137 | 0.2273 |
| `i3d` | 0.3947 | 0.4054 | 0.4000 | 0.4065 | 0.5732 | 10.9958 | 0.2075 |
| `videomae` | 0.6296 | 0.4595 | 0.5312 | 0.2439 | 0.2439 | 10.2537 | 0.5172 |
| `xception_df40` | 0.3438 | 0.2973 | 0.3188 | 0.4065 | 0.5569 | 12.3995 | 0.3349 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5067 | 1.0000 | 0.6726 | 0.4918 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4933 | 1.0000 | 0.6607 | 0.4878 | 1.0000 | 0.0000 |
