# Benchmark Scoring Report V2

## Method

- Evaluation unit: 12-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 12-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0820
- Validation time-to-first-detection: 16.3419s
- Validation F1: 0.1408

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.76 | 0.66 | 0.0000 | 0.0000 | 0.2933 | 11.5727 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2000 | 13.4734 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.64 | 0.54 | 0.1000 | 0.0400 | 0.3933 | 9.5867 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.6867 | 5.5057 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.69 | 0.59 | 0.1000 | 0.0400 | 0.3267 | 10.3742 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.0667 | 16.0062 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.7143 | 0.0781 | 0.1408 | 0.0820 | 0.0328 | 16.3419 | 1.0714 |
| 2 | `videomae` | 0.6078 | 0.2422 | 0.3464 | 0.5328 | 0.1639 | 12.9126 | 2.3529 |
| 3 | `i3d` | 0.2600 | 0.2031 | 0.2281 | 0.5328 | 0.6066 | 14.1362 | 0.2000 |
| 4 | `effort_clip_l14` | 0.7419 | 0.3594 | 0.4842 | 0.6557 | 0.1311 | 11.3927 | 0.5645 |
| 5 | `f3net` | 0.6522 | 0.4688 | 0.5455 | 0.7787 | 0.2623 | 8.6668 | 1.9565 |
| 6 | `efficientnet_b4` | 0.5068 | 0.2891 | 0.3682 | 0.7787 | 0.2951 | 12.0816 | 1.0274 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5405 | 0.4959 | 0.5172 | 0.8140 | 0.3953 | 8.8859 | 0.6757 |
| `effort_clip_l14` | 0.4773 | 0.1736 | 0.2545 | 0.6589 | 0.1783 | 14.9520 | 0.7955 |
| `f3net` | 0.6667 | 0.3967 | 0.4974 | 0.5426 | 0.1860 | 10.7556 | 1.4583 |
| `i3d` | 0.4721 | 0.7686 | 0.5849 | 1.4341 | 0.8062 | 4.3388 | 0.1269 |
| `videomae` | 0.5258 | 0.4215 | 0.4679 | 0.6977 | 0.3566 | 10.0518 | 1.7526 |
| `xception_df40` | 0.7000 | 0.1157 | 0.1986 | 0.1550 | 0.0465 | 15.9090 | 1.5000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5120 | 1.0000 | 0.6772 | 1.9262 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4840 | 1.0000 | 0.6523 | 1.8217 | 1.0000 | 0.0000 |
