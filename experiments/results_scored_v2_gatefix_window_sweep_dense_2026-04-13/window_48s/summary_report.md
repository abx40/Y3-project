# Benchmark Scoring Report V2

## Method

- Evaluation unit: 48-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 48-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.1056
- Validation time-to-first-detection: 11.2912s
- Validation F1: 0.4364

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 1 | 0.76 | 0.66 | 0.0000 | 0.0000 | 0.3429 | 11.6649 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.87 | 0.0000 | 0.0000 | 0.1429 | 13.5726 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.5143 | 6.2931 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.5714 | 6.9053 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.55 | 0.45 | 0.1000 | 0.0400 | 0.6571 | 5.8496 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.17 | 0.12 | 0.1000 | 0.0400 | 0.5143 | 7.1805 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.4444 | 0.4286 | 0.4364 | 0.1056 | 0.4085 | 11.2912 | 0.3431 |
| 2 | `i3d` | 0.1905 | 0.1429 | 0.1633 | 0.1056 | 0.4507 | 15.7893 | 0.2632 |
| 3 | `videomae` | 0.3571 | 0.3571 | 0.3571 | 0.2113 | 0.4789 | 10.9214 | 0.2778 |
| 4 | `xception_df40` | 0.3500 | 0.2500 | 0.2917 | 0.2113 | 0.3380 | 12.6045 | 0.3378 |
| 5 | `efficientnet_b4` | 0.2000 | 0.1071 | 0.1395 | 0.2465 | 0.3099 | 15.1814 | 1.2963 |
| 6 | `f3net` | 0.4103 | 0.5714 | 0.4776 | 0.3521 | 0.6479 | 6.7448 | 0.3289 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6522 | 0.4545 | 0.5357 | 0.2917 | 0.2667 | 9.2142 | 0.9239 |
| `effort_clip_l14` | 0.5862 | 0.5152 | 0.5484 | 0.2917 | 0.3500 | 8.1084 | 0.1852 |
| `f3net` | 0.6562 | 0.6364 | 0.6462 | 0.2917 | 0.3167 | 7.1729 | 0.3333 |
| `i3d` | 0.3235 | 0.3333 | 0.3284 | 0.3333 | 0.7167 | 12.1355 | 0.1154 |
| `videomae` | 0.4667 | 0.4242 | 0.4444 | 0.2917 | 0.4833 | 10.5514 | 0.4464 |
| `xception_df40` | 0.5476 | 0.6970 | 0.6133 | 0.3750 | 0.5833 | 5.5952 | 0.1875 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4308 | 1.0000 | 0.6022 | 0.4577 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5077 | 1.0000 | 0.6735 | 0.5417 | 1.0000 | 0.0000 |
