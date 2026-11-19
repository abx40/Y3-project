# Benchmark Scoring Report V2

## Method

- Evaluation unit: 47-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 47-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1416
- Validation time-to-first-detection: 10.5735s
- Validation F1: 0.4364

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.68 | 0.58 | 0.0000 | 0.0000 | 0.3077 | 12.0022 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.82 | 0.0000 | 0.0000 | 0.2051 | 13.5245 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.56 | 0.46 | 0.0000 | 0.0000 | 0.6667 | 6.1485 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6410 | 5.5523 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.6154 | 7.0615 |
| `xception_df40` | `strict_0.10` | `high` | 3 | 0.70 | 0.70 | 0.0000 | 0.0000 | 0.3077 | 12.2440 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.4444 | 0.4286 | 0.4364 | 0.1416 | 0.4029 | 10.5735 | 0.3398 |
| 2 | `i3d` | 0.1111 | 0.0714 | 0.0870 | 0.1416 | 0.4242 | 16.7296 | 0.2214 |
| 3 | `xception_df40` | 0.3889 | 0.5000 | 0.4375 | 0.2124 | 0.5906 | 8.2960 | 0.2532 |
| 4 | `effort_clip_l14` | 0.3235 | 0.3929 | 0.3548 | 0.2124 | 0.6183 | 10.0020 | 0.1544 |
| 5 | `efficientnet_b4` | 0.5161 | 0.5714 | 0.5424 | 0.2478 | 0.3900 | 8.2107 | 0.2996 |
| 6 | `f3net` | 0.3571 | 0.3571 | 0.3571 | 0.2478 | 0.4926 | 9.7905 | 0.3709 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5686 | 0.9062 | 0.6988 | 0.5574 | 0.6715 | 1.9859 | 0.1777 |
| `effort_clip_l14` | 0.6296 | 0.5312 | 0.5763 | 0.2389 | 0.2973 | 8.2152 | 0.2427 |
| `f3net` | 0.5926 | 0.5000 | 0.5424 | 0.2389 | 0.3285 | 8.7221 | 0.2427 |
| `i3d` | 0.2941 | 0.3125 | 0.3030 | 0.3583 | 0.7266 | 12.0932 | 0.1150 |
| `videomae` | 0.4118 | 0.4375 | 0.4242 | 0.3981 | 0.6019 | 9.8555 | 0.3089 |
| `xception_df40` | 0.4828 | 0.4375 | 0.4590 | 0.3981 | 0.4386 | 9.8433 | 0.3211 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4308 | 1.0000 | 0.6022 | 0.3894 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4923 | 1.0000 | 0.6598 | 0.5574 | 1.0000 | 0.0000 |
