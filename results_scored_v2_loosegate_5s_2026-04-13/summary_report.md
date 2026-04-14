# Benchmark Scoring Report V2 Loose Gate 5s

## Method

- Evaluation unit: 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 5-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.25 FA/min and <= 0.15 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.50 FA/min and <= 0.30 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.25 and FA time ratio <= 0.15 (relaxed: FA/min <= 0.50 and FA time ratio <= 0.30), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `xception_df40`
- Validation F1: 0.1437
- Validation false alerts/min: 0.1208
- Validation time-to-first-detection: 15.8837s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.25` | `low` | 3 | 0.84 | 0.74 | 0.2000 | 0.0833 | 0.1417 | 0.2345 | 14.5192 |
| `effort_clip_l14` | `strict_0.25` | `low` | 3 | 0.41 | 0.31 | 0.2000 | 0.0417 | 0.1389 | 0.2252 | 14.5777 |
| `f3net` | `strict_0.25` | `high` | 3 | 0.66 | 0.56 | 0.2000 | 0.1417 | 0.3556 | 0.4613 | 9.8912 |
| `i3d` | `strict_0.25` | `low` | 3 | 0.35 | 0.30 | 0.2000 | 0.0667 | 0.2194 | 0.3413 | 13.0616 |
| `videomae` | `strict_0.25` | `low` | 3 | 0.71 | 0.61 | 0.2000 | 0.0917 | 0.3389 | 0.4527 | 10.5158 |
| `xception_df40` | `strict_0.25` | `low` | 3 | 0.74 | 0.74 | 0.1000 | 0.0250 | 0.0889 | 0.1535 | 15.3061 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.7500 | 0.0795 | 0.1437 | 0.1208 | 0.0268 | 15.8837 | 6.0000 |
| 2 | `efficientnet_b4` | 0.5556 | 0.1656 | 0.2551 | 0.3221 | 0.1342 | 14.5177 | 3.0667 |
| 3 | `i3d` | 0.5385 | 0.0927 | 0.1582 | 0.3624 | 0.0805 | 15.7298 | 4.6154 |
| 4 | `effort_clip_l14` | 0.5752 | 0.2152 | 0.3133 | 0.6040 | 0.1611 | 13.5071 | 3.3982 |
| 5 | `f3net` | 0.5829 | 0.3377 | 0.4277 | 0.9262 | 0.2450 | 10.7024 | 3.1543 |
| 6 | `videomae` | 0.5974 | 0.3046 | 0.4035 | 0.9262 | 0.2081 | 11.9318 | 3.3506 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6714 | 0.3082 | 0.4225 | 0.6915 | 0.1559 | 11.6019 | 2.9143 |
| `effort_clip_l14` | 0.3469 | 0.1115 | 0.1687 | 0.6915 | 0.2169 | 15.9516 | 4.1633 |
| `f3net` | 0.6802 | 0.3836 | 0.4906 | 0.7729 | 0.1864 | 10.6486 | 2.4419 |
| `i3d` | 0.5075 | 0.1115 | 0.1828 | 0.4068 | 0.1119 | 15.9497 | 2.8657 |
| `videomae` | 0.5455 | 0.4525 | 0.4946 | 1.0169 | 0.3898 | 9.4470 | 1.9447 |
| `xception_df40` | 0.5667 | 0.0557 | 0.1015 | 0.1627 | 0.0441 | 17.3131 | 2.8000 |
