# Benchmark Scoring Report V2

## Method

- Evaluation unit: 4-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 4-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0398
- Validation time-to-first-detection: 17.7550s
- Validation F1: 0.0106

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.88 | 0.78 | 0.1000 | 0.0200 | 0.1086 | 14.5862 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.75 | 0.75 | 0.1000 | 0.0200 | 0.0310 | 16.6808 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.70 | 0.60 | 0.1000 | 0.0267 | 0.3038 | 10.7676 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.41 | 0.31 | 0.1000 | 0.0267 | 0.0931 | 15.5623 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.92 | 0.82 | 0.0000 | 0.0000 | 0.1153 | 14.8754 |
| `xception_df40` | `strict_0.10` | `low` | 1 | 0.97 | 0.92 | 0.1000 | 0.0200 | 0.0067 | 17.0650 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6667 | 0.0054 | 0.0106 | 0.0398 | 0.0027 | 17.7550 | 20.0000 |
| 2 | `i3d` | 0.3333 | 0.0027 | 0.0053 | 0.0398 | 0.0053 | 17.7886 | 10.0000 |
| 3 | `videomae` | 0.7627 | 0.1206 | 0.2083 | 0.1989 | 0.0371 | 15.3003 | 4.0678 |
| 4 | `effort_clip_l14` | 0.5400 | 0.0724 | 0.1277 | 0.2387 | 0.0610 | 16.3428 | 3.9000 |
| 5 | `efficientnet_b4` | 0.6129 | 0.1019 | 0.1747 | 0.3979 | 0.0637 | 15.0348 | 6.5323 |
| 6 | `f3net` | 0.5924 | 0.2922 | 0.3914 | 0.9151 | 0.1989 | 11.2412 | 4.2391 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6047 | 0.2091 | 0.3108 | 0.5172 | 0.1353 | 13.1490 | 3.9535 |
| `effort_clip_l14` | 0.2203 | 0.0349 | 0.0602 | 0.4377 | 0.1220 | 17.3223 | 6.1017 |
| `f3net` | 0.6415 | 0.3646 | 0.4650 | 0.9549 | 0.2016 | 10.1462 | 2.9009 |
| `i3d` | 0.1667 | 0.0054 | 0.0104 | 0.1194 | 0.0265 | 18.1652 | 7.5000 |
| `videomae` | 0.6897 | 0.2145 | 0.3272 | 0.3581 | 0.0955 | 13.8786 | 2.5862 |
| `xception_df40` | 0.8000 | 0.0107 | 0.0212 | 0.0398 | 0.0027 | 18.1155 | 6.0000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4973 | 1.0000 | 0.6643 | 2.0292 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4973 | 1.0000 | 0.6643 | 2.0690 | 1.0000 | 0.0000 |
