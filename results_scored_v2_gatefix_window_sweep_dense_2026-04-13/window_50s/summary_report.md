# Benchmark Scoring Report V2

## Method

- Evaluation unit: 50-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 50-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0828
- Validation time-to-first-detection: 15.4137s
- Validation F1: 0.2564

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.4359 | 9.2433 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.93 | 0.83 | 0.0000 | 0.0000 | 0.2308 | 13.5102 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.6410 | 6.4066 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6410 | 6.4788 |
| `videomae` | `strict_0.10` | `low` | 1 | 0.64 | 0.54 | 0.0000 | 0.0000 | 0.5897 | 7.4346 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.25 | 0.20 | 0.0000 | 0.0000 | 0.3590 | 11.3270 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6250 | 0.1613 | 0.2564 | 0.0828 | 0.1034 | 15.4137 | 0.6000 |
| 2 | `i3d` | 0.2000 | 0.1290 | 0.1569 | 0.1241 | 0.5517 | 15.1312 | 0.1800 |
| 3 | `efficientnet_b4` | 0.5185 | 0.4516 | 0.4828 | 0.1655 | 0.4483 | 10.4006 | 0.2222 |
| 4 | `effort_clip_l14` | 0.6111 | 0.3548 | 0.4490 | 0.1655 | 0.2414 | 10.6730 | 0.4667 |
| 5 | `f3net` | 0.5185 | 0.4516 | 0.4828 | 0.2069 | 0.4483 | 9.6445 | 0.2667 |
| 6 | `videomae` | 0.5333 | 0.2581 | 0.3478 | 0.2069 | 0.2414 | 13.3397 | 1.2800 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.0000 | 0.0000 | 0.0000 | 0.1200 | 0.1333 | 18.2216 | 0.9000 |
| `effort_clip_l14` | 0.5500 | 0.3667 | 0.4400 | 0.2400 | 0.3000 | 10.5214 | 0.3600 |
| `f3net` | 0.4848 | 0.5333 | 0.5079 | 0.3200 | 0.5667 | 7.9168 | 0.2545 |
| `i3d` | 0.3548 | 0.3667 | 0.3607 | 0.2800 | 0.6667 | 12.2331 | 0.1161 |
| `videomae` | 0.5385 | 0.4667 | 0.5000 | 0.2800 | 0.4000 | 9.1776 | 0.7846 |
| `xception_df40` | 0.6071 | 0.5667 | 0.5862 | 0.2400 | 0.3667 | 8.5179 | 0.2143 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5167 | 1.0000 | 0.6813 | 0.4552 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5000 | 1.0000 | 0.6667 | 0.5200 | 1.0000 | 0.0000 |
