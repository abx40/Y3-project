# Benchmark Scoring Report V2

## Method

- Evaluation unit: 16-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 16-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0802
- Validation time-to-first-detection: 16.0511s
- Validation F1: 0.1786

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.68 | 0.58 | 0.0000 | 0.0000 | 0.7130 | 5.0043 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2000 | 13.2557 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.6087 | 5.8436 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.5391 | 7.7907 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.62 | 0.57 | 0.0000 | 0.0000 | 0.3739 | 10.7338 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.43 | 0.33 | 0.0000 | 0.0000 | 0.2348 | 13.3218 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5882 | 0.1053 | 0.1786 | 0.0802 | 0.0695 | 16.0511 | 0.9375 |
| 2 | `i3d` | 0.4118 | 0.1474 | 0.2171 | 0.2807 | 0.2032 | 15.2012 | 0.3462 |
| 3 | `videomae` | 0.5156 | 0.3474 | 0.4151 | 0.4813 | 0.3262 | 10.8979 | 0.8858 |
| 4 | `effort_clip_l14` | 0.6981 | 0.3895 | 0.5000 | 0.6016 | 0.1711 | 10.7304 | 0.9286 |
| 5 | `f3net` | 0.5690 | 0.6947 | 0.6256 | 1.0027 | 0.5241 | 5.7498 | 0.3947 |
| 6 | `efficientnet_b4` | 0.5377 | 0.6000 | 0.5672 | 1.0829 | 0.5134 | 6.0418 | 0.5383 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4643 | 0.6842 | 0.5532 | 1.2500 | 0.7849 | 5.1477 | 0.3261 |
| `effort_clip_l14` | 0.4848 | 0.1684 | 0.2500 | 0.5645 | 0.1774 | 14.6269 | 1.1719 |
| `f3net` | 0.5794 | 0.6526 | 0.6139 | 0.7661 | 0.4677 | 6.2970 | 0.4286 |
| `i3d` | 0.6727 | 0.3895 | 0.4933 | 0.3629 | 0.1828 | 11.2165 | 0.2103 |
| `videomae` | 0.5057 | 0.4632 | 0.4835 | 0.7258 | 0.4516 | 9.5414 | 0.7456 |
| `xception_df40` | 0.8077 | 0.2211 | 0.3471 | 0.1613 | 0.0538 | 13.8675 | 1.1538 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5000 | 1.0000 | 0.6667 | 1.5642 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5000 | 1.0000 | 0.6667 | 1.6935 | 1.0000 | 0.0000 |
