# Benchmark Scoring Report V2

## Method

- Evaluation unit: 3-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 3-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0806
- Validation time-to-first-detection: 17.2523s
- Validation F1: 0.0347

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.88 | 0.78 | 0.1000 | 0.0200 | 0.1252 | 14.3538 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.84 | 0.79 | 0.1000 | 0.0250 | 0.0434 | 16.3598 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.75 | 0.65 | 0.1000 | 0.0150 | 0.2387 | 11.2962 |
| `i3d` | `strict_0.10` | `low` | 1 | 0.45 | 0.35 | 0.1000 | 0.0250 | 0.0200 | 16.6062 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.92 | 0.82 | 0.1000 | 0.0200 | 0.1369 | 13.4941 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.92 | 0.92 | 0.1000 | 0.0250 | 0.0167 | 16.8208 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.6000 | 0.0179 | 0.0347 | 0.0806 | 0.0121 | 17.2523 | 12.0000 |
| 2 | `xception_df40` | 0.4348 | 0.0198 | 0.0380 | 0.1210 | 0.0262 | 17.3695 | 6.9565 |
| 3 | `effort_clip_l14` | 0.5172 | 0.0595 | 0.1068 | 0.1613 | 0.0565 | 16.4677 | 4.1379 |
| 4 | `videomae` | 0.5932 | 0.1389 | 0.2251 | 0.4839 | 0.0968 | 14.0110 | 6.4407 |
| 5 | `efficientnet_b4` | 0.6250 | 0.1190 | 0.2000 | 0.5242 | 0.0726 | 14.7072 | 6.4583 |
| 6 | `f3net` | 0.5607 | 0.2381 | 0.3343 | 0.9274 | 0.1895 | 11.7317 | 5.0467 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6298 | 0.2615 | 0.3695 | 0.7615 | 0.1543 | 12.0421 | 4.2308 |
| `effort_clip_l14` | 0.2222 | 0.0319 | 0.0558 | 0.4409 | 0.1122 | 17.3017 | 6.6667 |
| `f3net` | 0.6292 | 0.3014 | 0.4076 | 0.8818 | 0.1784 | 11.1333 | 3.4167 |
| `i3d` | 0.2500 | 0.0040 | 0.0079 | 0.0802 | 0.0120 | 18.2374 | 15.0000 |
| `videomae` | 0.6216 | 0.2295 | 0.3353 | 0.6012 | 0.1403 | 12.6490 | 3.8919 |
| `xception_df40` | 0.2222 | 0.0040 | 0.0078 | 0.0802 | 0.0140 | 18.2130 | 6.6667 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5040 | 1.0000 | 0.6702 | 2.0565 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5010 | 1.0000 | 0.6676 | 2.0842 | 1.0000 | 0.0000 |
