# Benchmark Scoring Report V2

## Method

- Evaluation unit: 38-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 38-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1275
- Validation time-to-first-detection: 12.7685s
- Validation F1: 0.3103

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.63 | 0.53 | 0.0000 | 0.0000 | 0.6800 | 6.0498 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2000 | 14.2967 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.6000 | 6.0273 |
| `i3d` | `strict_0.10` | `high` | 2 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.7000 | 6.4085 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.54 | 0.49 | 0.1000 | 0.0500 | 0.7200 | 5.2094 |
| `xception_df40` | `strict_0.10` | `high` | 2 | 0.88 | 0.78 | 0.0000 | 0.0000 | 0.1200 | 15.0039 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5625 | 0.2143 | 0.3103 | 0.1275 | 0.1827 | 12.7685 | 0.5000 |
| 2 | `effort_clip_l14` | 0.7333 | 0.2619 | 0.3860 | 0.1700 | 0.1020 | 12.3915 | 0.9609 |
| 3 | `f3net` | 0.5349 | 0.5476 | 0.5412 | 0.2125 | 0.5326 | 6.9930 | 0.3708 |
| 4 | `videomae` | 0.5714 | 0.4762 | 0.5195 | 0.2125 | 0.3924 | 10.1713 | 0.7306 |
| 5 | `efficientnet_b4` | 0.3947 | 0.3571 | 0.3750 | 0.3399 | 0.5963 | 10.1946 | 0.1700 |
| 6 | `i3d` | 0.5410 | 0.7857 | 0.6408 | 0.3824 | 0.7309 | 2.7213 | 0.1317 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6290 | 0.8298 | 0.7156 | 0.3928 | 0.6956 | 2.8694 | 0.1549 |
| `effort_clip_l14` | 0.4286 | 0.0638 | 0.1111 | 0.1473 | 0.1178 | 16.3277 | 1.9200 |
| `f3net` | 0.7429 | 0.5532 | 0.6341 | 0.2946 | 0.2668 | 7.3328 | 0.5054 |
| `i3d` | 0.5400 | 0.5745 | 0.5567 | 0.3928 | 0.6890 | 8.2861 | 0.1613 |
| `videomae` | 0.5926 | 0.6809 | 0.6337 | 0.4419 | 0.6710 | 4.8106 | 0.3846 |
| `xception_df40` | 0.5714 | 0.3404 | 0.4267 | 0.2946 | 0.3601 | 11.6613 | 0.2290 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5250 | 1.0000 | 0.6885 | 0.4249 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5875 | 1.0000 | 0.7402 | 0.5401 | 1.0000 | 0.0000 |
