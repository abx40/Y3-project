# Benchmark Scoring Report V2

## Method

- Evaluation unit: 37-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 37-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.0547
- Validation time-to-first-detection: 15.7812s
- Validation F1: 0.2500

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.43 | 0.33 | 0.0000 | 0.0000 | 0.3265 | 11.3512 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.1429 | 15.4538 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.5918 | 6.2027 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7347 | 4.6273 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.56 | 0.46 | 0.0000 | 0.0000 | 0.5510 | 9.1299 |
| `xception_df40` | `strict_0.10` | `high` | 3 | 0.79 | 0.79 | 0.0000 | 0.0000 | 0.1224 | 14.8861 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.7273 | 0.1509 | 0.2500 | 0.0547 | 0.0747 | 15.7812 | 0.4762 |
| 2 | `xception_df40` | 0.6071 | 0.3208 | 0.4198 | 0.1094 | 0.3710 | 11.5723 | 0.4171 |
| 3 | `effort_clip_l14` | 0.8333 | 0.1887 | 0.3077 | 0.1094 | 0.0410 | 14.6417 | 1.5544 |
| 4 | `efficientnet_b4` | 0.5882 | 0.1887 | 0.2857 | 0.2188 | 0.2097 | 14.6584 | 0.9963 |
| 5 | `f3net` | 0.6538 | 0.6415 | 0.6476 | 0.4376 | 0.6071 | 4.9185 | 0.3215 |
| 6 | `i3d` | 0.4800 | 0.4528 | 0.4660 | 0.4376 | 0.7976 | 11.0070 | 0.1384 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4375 | 0.1628 | 0.2373 | 0.1669 | 0.1912 | 14.7082 | 0.6742 |
| `effort_clip_l14` | 0.5714 | 0.0930 | 0.1600 | 0.1252 | 0.0369 | 17.0296 | 3.1395 |
| `f3net` | 0.5556 | 0.4651 | 0.5063 | 0.3338 | 0.3713 | 8.3755 | 0.5301 |
| `i3d` | 0.4627 | 0.7209 | 0.5636 | 0.3338 | 0.8456 | 5.1544 | 0.1285 |
| `videomae` | 0.3714 | 0.3023 | 0.3333 | 0.2921 | 0.5056 | 10.3696 | 0.2483 |
| `xception_df40` | 0.5000 | 0.3488 | 0.4110 | 0.2921 | 0.3658 | 12.2043 | 0.4563 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.6235 | 1.0000 | 0.7681 | 0.6016 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5059 | 1.0000 | 0.6719 | 0.4590 | 1.0000 | 0.0000 |
