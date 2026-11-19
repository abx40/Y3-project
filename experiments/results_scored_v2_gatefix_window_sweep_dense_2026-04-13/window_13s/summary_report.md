# Benchmark Scoring Report V2

## Method

- Evaluation unit: 13-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 13-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1639
- Validation time-to-first-detection: 15.9201s
- Validation F1: 0.1884

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.79 | 0.69 | 0.0000 | 0.0000 | 0.2086 | 13.6065 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2086 | 13.2808 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.60 | 0.50 | 0.0000 | 0.0000 | 0.3453 | 10.3836 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6763 | 5.5808 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.66 | 0.56 | 0.0000 | 0.0000 | 0.3813 | 10.6462 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.1367 | 14.6026 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6842 | 0.1092 | 0.1884 | 0.1639 | 0.0458 | 15.9201 | 1.0667 |
| 2 | `efficientnet_b4` | 0.5161 | 0.1345 | 0.2133 | 0.2869 | 0.1257 | 15.1947 | 1.3776 |
| 3 | `videomae` | 0.4583 | 0.2773 | 0.3456 | 0.4098 | 0.3388 | 12.0366 | 0.8432 |
| 4 | `effort_clip_l14` | 0.7258 | 0.3782 | 0.4972 | 0.6148 | 0.1359 | 10.7004 | 1.3776 |
| 5 | `i3d` | 0.4254 | 0.4790 | 0.4506 | 0.7787 | 0.6537 | 9.2614 | 0.1778 |
| 6 | `f3net` | 0.5648 | 0.5126 | 0.5374 | 1.1066 | 0.4173 | 7.8970 | 0.7692 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5522 | 0.3162 | 0.4022 | 0.4430 | 0.2617 | 11.9399 | 0.8266 |
| `effort_clip_l14` | 0.6076 | 0.4103 | 0.4898 | 0.9262 | 0.2557 | 10.1313 | 1.3883 |
| `f3net` | 0.5424 | 0.2735 | 0.3636 | 0.6846 | 0.2356 | 12.5721 | 1.1905 |
| `i3d` | 0.6090 | 0.8120 | 0.6960 | 1.5705 | 0.5101 | 3.1714 | 0.1210 |
| `videomae` | 0.5543 | 0.4359 | 0.4880 | 0.6443 | 0.3430 | 10.2309 | 0.7739 |
| `xception_df40` | 0.7500 | 0.1282 | 0.2190 | 0.2013 | 0.0362 | 15.9167 | 1.6867 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5064 | 1.0000 | 0.6723 | 1.9262 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4979 | 1.0000 | 0.6648 | 1.9732 | 1.0000 | 0.0000 |
