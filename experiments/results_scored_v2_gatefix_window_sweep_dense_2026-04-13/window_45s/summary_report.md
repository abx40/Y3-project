# Benchmark Scoring Report V2

## Method

- Evaluation unit: 45-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 45-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0000
- Validation time-to-first-detection: 17.8571s
- Validation F1: 0.0000

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.35 | 0.30 | 0.0000 | 0.0000 | 0.4667 | 10.0439 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.77 | 0.0000 | 0.0000 | 0.2667 | 13.4438 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.59 | 0.49 | 0.0000 | 0.0000 | 0.5111 | 8.1512 |
| `i3d` | `strict_0.10` | `low` | 1 | 0.36 | 0.26 | 0.0000 | 0.0000 | 0.5333 | 8.7669 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.60 | 0.50 | 0.1000 | 0.0250 | 0.6222 | 7.2017 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.21 | 0.16 | 0.1000 | 0.0250 | 0.4667 | 9.5654 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 2 | `efficientnet_b4` | 0.3077 | 0.1250 | 0.1778 | 0.0741 | 0.2500 | 16.3238 | 0.4103 |
| 3 | `xception_df40` | 0.4500 | 0.2812 | 0.3462 | 0.1111 | 0.2685 | 13.2298 | 0.3704 |
| 4 | `videomae` | 0.2727 | 0.0938 | 0.1395 | 0.1111 | 0.2037 | 15.6846 | 0.6452 |
| 5 | `effort_clip_l14` | 0.4211 | 0.5000 | 0.4571 | 0.1852 | 0.5741 | 9.7896 | 0.1509 |
| 6 | `f3net` | 0.6471 | 0.6875 | 0.6667 | 0.2963 | 0.3333 | 5.7788 | 0.7600 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.3000 | 0.0909 | 0.1395 | 0.1165 | 0.1456 | 16.3954 | 0.8333 |
| `effort_clip_l14` | 0.6250 | 0.6061 | 0.6154 | 0.2718 | 0.2913 | 8.1455 | 0.1818 |
| `f3net` | 0.6000 | 0.5455 | 0.5714 | 0.3107 | 0.3301 | 7.0361 | 0.7442 |
| `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `videomae` | 0.4583 | 0.3333 | 0.3860 | 0.1942 | 0.3398 | 11.9240 | 0.5455 |
| `xception_df40` | 0.4706 | 0.4848 | 0.4776 | 0.3107 | 0.4660 | 8.8667 | 0.2917 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4571 | 1.0000 | 0.6275 | 0.4074 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4714 | 1.0000 | 0.6408 | 0.5825 | 1.0000 | 0.0000 |
