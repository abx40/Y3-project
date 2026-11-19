# Benchmark Scoring Report V2

## Method

- Evaluation unit: 36-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 36-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1905
- Validation time-to-first-detection: 12.1539s
- Validation F1: 0.3684

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.41 | 0.31 | 0.0000 | 0.0000 | 0.2778 | 13.2140 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.82 | 0.0000 | 0.0000 | 0.2778 | 11.9556 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.6481 | 6.9180 |
| `i3d` | `strict_0.10` | `high` | 2 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.6296 | 6.0679 |
| `videomae` | `strict_0.10` | `low` | 1 | 0.64 | 0.54 | 0.1000 | 0.0400 | 0.4259 | 8.8367 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.29 | 0.24 | 0.0000 | 0.0000 | 0.2407 | 13.1134 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5185 | 0.2857 | 0.3684 | 0.1905 | 0.3524 | 12.1539 | 0.3205 |
| 2 | `efficientnet_b4` | 0.5500 | 0.2245 | 0.3188 | 0.1905 | 0.2476 | 13.0073 | 0.9649 |
| 3 | `f3net` | 0.6327 | 0.6327 | 0.6327 | 0.2857 | 0.5048 | 6.3034 | 0.3125 |
| 4 | `effort_clip_l14` | 0.6053 | 0.4694 | 0.5287 | 0.2857 | 0.4190 | 9.0337 | 0.4464 |
| 5 | `videomae` | 0.6667 | 0.3265 | 0.4384 | 0.2857 | 0.2190 | 12.2395 | 1.7857 |
| 6 | `i3d` | 0.6061 | 0.8163 | 0.6957 | 0.3810 | 0.7143 | 2.6261 | 0.1295 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5789 | 0.2558 | 0.3548 | 0.1639 | 0.1803 | 12.7416 | 0.4630 |
| `effort_clip_l14` | 0.5556 | 0.4651 | 0.5063 | 0.2459 | 0.3770 | 9.0037 | 0.2830 |
| `f3net` | 0.5152 | 0.3953 | 0.4474 | 0.2869 | 0.3770 | 8.9088 | 0.4688 |
| `i3d` | 0.3953 | 0.3953 | 0.3953 | 0.3279 | 0.6066 | 10.7669 | 0.2016 |
| `videomae` | 0.5366 | 0.5116 | 0.5238 | 0.4508 | 0.4508 | 8.0789 | 1.2083 |
| `xception_df40` | 0.6250 | 0.3488 | 0.4478 | 0.0820 | 0.2131 | 11.6458 | 0.3521 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5765 | 1.0000 | 0.7313 | 0.4286 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5059 | 1.0000 | 0.6719 | 0.4508 | 1.0000 | 0.0000 |
