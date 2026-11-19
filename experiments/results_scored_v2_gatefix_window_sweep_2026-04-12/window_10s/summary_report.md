# Benchmark Scoring Report V2

## Method

- Evaluation unit: 10-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 10-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.3649
- Validation time-to-first-detection: 13.6852s
- Validation F1: 0.3092

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.80 | 0.70 | 0.0000 | 0.0000 | 0.2135 | 13.3482 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2079 | 13.2633 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.61 | 0.51 | 0.1000 | 0.0500 | 0.4888 | 7.7465 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.33 | 0.23 | 0.0000 | 0.0000 | 0.6348 | 6.4670 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.66 | 0.56 | 0.0000 | 0.0000 | 0.5056 | 8.1496 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.44 | 0.44 | 0.1000 | 0.0500 | 0.2303 | 12.6091 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5818 | 0.2105 | 0.3092 | 0.3649 | 0.1554 | 13.6852 | 1.5273 |
| 2 | `videomae` | 0.5600 | 0.1842 | 0.2772 | 0.3649 | 0.1486 | 13.9462 | 1.4400 |
| 3 | `efficientnet_b4` | 0.4474 | 0.1118 | 0.1789 | 0.4054 | 0.1419 | 15.8778 | 1.5789 |
| 4 | `i3d` | 0.4512 | 0.2434 | 0.3162 | 0.7297 | 0.3041 | 13.0947 | 0.2927 |
| 5 | `effort_clip_l14` | 0.6436 | 0.4276 | 0.5138 | 0.8919 | 0.2432 | 9.8094 | 0.5347 |
| 6 | `f3net` | 0.5952 | 0.4934 | 0.5396 | 1.0135 | 0.3446 | 8.0713 | 1.8571 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6515 | 0.2848 | 0.3963 | 0.4832 | 0.1544 | 12.2745 | 1.2727 |
| `effort_clip_l14` | 0.5000 | 0.2318 | 0.3167 | 0.8456 | 0.2349 | 13.8585 | 0.9429 |
| `f3net` | 0.6167 | 0.4901 | 0.5461 | 1.0067 | 0.3087 | 8.6613 | 1.5000 |
| `i3d` | 0.4710 | 0.4305 | 0.4498 | 0.4430 | 0.4899 | 10.5075 | 0.1304 |
| `videomae` | 0.5077 | 0.4371 | 0.4698 | 0.7651 | 0.4295 | 10.0099 | 1.0615 |
| `xception_df40` | 0.6596 | 0.2053 | 0.3131 | 0.2819 | 0.1074 | 14.1153 | 2.1702 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5067 | 1.0000 | 0.6726 | 2.0270 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5033 | 1.0000 | 0.6696 | 2.0134 | 1.0000 | 0.0000 |
