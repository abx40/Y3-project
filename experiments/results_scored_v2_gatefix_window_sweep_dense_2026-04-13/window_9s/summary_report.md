# Benchmark Scoring Report V2

## Method

- Evaluation unit: 9-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 9-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0409
- Validation time-to-first-detection: 17.5995s
- Validation F1: 0.0343

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.83 | 0.73 | 0.0000 | 0.0000 | 0.1667 | 13.7335 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2451 | 12.6801 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.56 | 0.46 | 0.1000 | 0.0450 | 0.5343 | 7.4731 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.38 | 0.28 | 0.1000 | 0.0450 | 0.1520 | 14.4709 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.64 | 0.54 | 0.1000 | 0.0450 | 0.5000 | 7.9686 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.66 | 0.56 | 0.1000 | 0.0450 | 0.1078 | 14.8955 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.7500 | 0.0175 | 0.0343 | 0.0409 | 0.0061 | 17.5995 | 3.3333 |
| 2 | `xception_df40` | 0.7308 | 0.1111 | 0.1929 | 0.2045 | 0.0429 | 15.4198 | 3.5897 |
| 3 | `efficientnet_b4` | 0.6296 | 0.0994 | 0.1717 | 0.2454 | 0.0614 | 15.8788 | 1.4815 |
| 4 | `videomae` | 0.5408 | 0.3099 | 0.3941 | 0.4908 | 0.2740 | 11.6719 | 1.3699 |
| 5 | `effort_clip_l14` | 0.7000 | 0.4912 | 0.5773 | 0.9407 | 0.2209 | 8.1126 | 0.8357 |
| 6 | `f3net` | 0.5319 | 0.5848 | 0.5571 | 1.2270 | 0.5378 | 6.9517 | 0.8881 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6154 | 0.2424 | 0.3478 | 0.4348 | 0.1482 | 13.2415 | 1.2308 |
| `effort_clip_l14` | 0.6260 | 0.4667 | 0.5347 | 1.0277 | 0.2708 | 8.9165 | 1.0899 |
| `f3net` | 0.6031 | 0.4788 | 0.5338 | 0.8300 | 0.3043 | 8.7963 | 0.9744 |
| `i3d` | 0.4800 | 0.0727 | 0.1263 | 0.3162 | 0.0771 | 16.9463 | 1.0667 |
| `videomae` | 0.4913 | 0.5152 | 0.5030 | 0.9881 | 0.5178 | 8.0254 | 0.9690 |
| `xception_df40` | 0.7333 | 0.1333 | 0.2256 | 0.1581 | 0.0455 | 15.4479 | 2.4719 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5104 | 1.0000 | 0.6759 | 1.9632 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4925 | 1.0000 | 0.6600 | 2.0158 | 1.0000 | 0.0000 |
