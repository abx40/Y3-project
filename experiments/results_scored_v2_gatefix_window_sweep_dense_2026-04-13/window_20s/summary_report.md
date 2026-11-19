# Benchmark Scoring Report V2

## Method

- Evaluation unit: 20-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 20-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0395
- Validation time-to-first-detection: 17.8571s
- Validation F1: 0.0000

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.70 | 0.60 | 0.0000 | 0.0000 | 0.6596 | 5.1490 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.1915 | 13.7700 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.51 | 0.41 | 0.0000 | 0.0000 | 0.5851 | 6.3889 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.4894 | 8.8267 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.58 | 0.48 | 0.1000 | 0.0333 | 0.4894 | 8.3040 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.34 | 0.24 | 0.1000 | 0.0333 | 0.3191 | 11.5452 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0395 | 0.3421 | 17.8571 | 0.1154 |
| 2 | `videomae` | 0.5091 | 0.3784 | 0.4341 | 0.2368 | 0.3553 | 10.7878 | 0.4909 |
| 3 | `xception_df40` | 0.4000 | 0.1351 | 0.2020 | 0.2368 | 0.1974 | 15.2046 | 0.7200 |
| 4 | `effort_clip_l14` | 0.7297 | 0.3649 | 0.4865 | 0.3553 | 0.1316 | 11.5857 | 0.8108 |
| 5 | `f3net` | 0.5495 | 0.6757 | 0.6061 | 0.5921 | 0.5395 | 6.0205 | 0.2967 |
| 6 | `efficientnet_b4` | 0.5342 | 0.5270 | 0.5306 | 0.7105 | 0.4474 | 8.2046 | 0.3699 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4632 | 0.6027 | 0.5238 | 0.7403 | 0.6623 | 7.0545 | 0.2842 |
| `effort_clip_l14` | 0.5500 | 0.1507 | 0.2366 | 0.2338 | 0.1169 | 15.5685 | 0.7500 |
| `f3net` | 0.4744 | 0.5068 | 0.4901 | 0.6234 | 0.5325 | 8.6109 | 0.1923 |
| `i3d` | 0.6364 | 0.4795 | 0.5469 | 0.3896 | 0.2597 | 9.7619 | 0.1636 |
| `videomae` | 0.4767 | 0.5616 | 0.5157 | 0.5455 | 0.5844 | 7.3827 | 0.4884 |
| `xception_df40` | 0.8148 | 0.3014 | 0.4400 | 0.1558 | 0.0649 | 12.5090 | 0.6667 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4933 | 1.0000 | 0.6607 | 1.1053 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4867 | 1.0000 | 0.6547 | 1.0909 | 1.0000 | 0.0000 |
