# Benchmark Scoring Report V2

## Method

- Evaluation unit: 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 5-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0403
- Validation time-to-first-detection: 17.5995s
- Validation F1: 0.0257

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.85 | 0.75 | 0.1000 | 0.0417 | 0.1028 | 15.2146 |
| `effort_clip_l14` | `strict_0.10` | `low` | 2 | 0.79 | 0.79 | 0.1000 | 0.0250 | 0.0472 | 16.2974 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.68 | 0.63 | 0.1000 | 0.0250 | 0.2833 | 11.3432 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.39 | 0.29 | 0.1000 | 0.0250 | 0.1556 | 14.3681 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.73 | 0.63 | 0.1000 | 0.0417 | 0.3250 | 10.7562 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.74 | 0.74 | 0.1000 | 0.0250 | 0.0889 | 15.3061 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.4444 | 0.0132 | 0.0257 | 0.0403 | 0.0168 | 17.5995 | 5.3333 |
| 2 | `xception_df40` | 0.7500 | 0.0795 | 0.1437 | 0.1208 | 0.0268 | 15.8837 | 6.0000 |
| 3 | `efficientnet_b4` | 0.7955 | 0.1159 | 0.2023 | 0.1611 | 0.0302 | 15.4645 | 4.0909 |
| 4 | `effort_clip_l14` | 0.5263 | 0.0662 | 0.1176 | 0.2416 | 0.0604 | 16.2892 | 4.1053 |
| 5 | `f3net` | 0.5957 | 0.2781 | 0.3792 | 0.7651 | 0.1913 | 11.8392 | 3.9149 |
| 6 | `videomae` | 0.6000 | 0.2980 | 0.3982 | 0.8859 | 0.2013 | 11.9707 | 3.2800 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6613 | 0.2689 | 0.3823 | 0.6102 | 0.1424 | 12.6430 | 3.2903 |
| `effort_clip_l14` | 0.2955 | 0.0426 | 0.0745 | 0.4475 | 0.1051 | 17.2125 | 7.0909 |
| `f3net` | 0.7230 | 0.3508 | 0.4724 | 0.6102 | 0.1390 | 11.1870 | 2.3514 |
| `i3d` | 0.4722 | 0.0557 | 0.0997 | 0.2847 | 0.0644 | 17.1495 | 2.0000 |
| `videomae` | 0.5411 | 0.4098 | 0.4664 | 0.8949 | 0.3593 | 10.3809 | 2.1818 |
| `xception_df40` | 0.5667 | 0.0557 | 0.1015 | 0.1627 | 0.0441 | 17.3131 | 2.8000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5033 | 1.0000 | 0.6696 | 2.0134 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5083 | 1.0000 | 0.6740 | 2.1153 | 1.0000 | 0.0000 |
