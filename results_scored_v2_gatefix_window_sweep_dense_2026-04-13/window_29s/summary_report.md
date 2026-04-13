# Benchmark Scoring Report V2

## Method

- Evaluation unit: 29-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 29-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.2108
- Validation time-to-first-detection: 11.1416s
- Validation F1: 0.5116

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.36 | 0.26 | 0.0000 | 0.0000 | 0.3077 | 12.3188 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.98 | 0.88 | 0.0000 | 0.0000 | 0.2154 | 12.8897 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.50 | 0.40 | 0.0000 | 0.0000 | 0.6462 | 6.8478 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.6154 | 5.7177 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.63 | 0.53 | 0.0000 | 0.0000 | 0.4308 | 9.8584 |
| `xception_df40` | `strict_0.10` | `high` | 1 | 0.91 | 0.86 | 0.1000 | 0.0483 | 0.2000 | 13.2080 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.7097 | 0.4000 | 0.5116 | 0.2108 | 0.1708 | 11.1416 | 0.2064 |
| 2 | `effort_clip_l14` | 0.7273 | 0.4364 | 0.5455 | 0.2530 | 0.1834 | 11.0505 | 0.5696 |
| 3 | `videomae` | 0.4839 | 0.2727 | 0.3488 | 0.2952 | 0.3197 | 11.9715 | 1.0112 |
| 4 | `efficientnet_b4` | 0.4375 | 0.3818 | 0.4078 | 0.3795 | 0.5376 | 12.0158 | 0.2198 |
| 5 | `xception_df40` | 0.5806 | 0.3273 | 0.4186 | 0.4216 | 0.2586 | 11.9572 | 1.9069 |
| 6 | `f3net` | 0.5068 | 0.6727 | 0.5781 | 0.5060 | 0.7147 | 6.8579 | 0.1448 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5625 | 0.3396 | 0.4235 | 0.2446 | 0.2575 | 12.3346 | 0.2691 |
| `effort_clip_l14` | 0.5556 | 0.0943 | 0.1613 | 0.0408 | 0.0788 | 16.3209 | 0.9195 |
| `f3net` | 0.4902 | 0.4717 | 0.4808 | 0.3668 | 0.4939 | 8.7708 | 0.2495 |
| `i3d` | 0.4819 | 0.7547 | 0.5882 | 0.4484 | 0.8227 | 5.0059 | 0.1270 |
| `videomae` | 0.5200 | 0.4906 | 0.5049 | 0.2853 | 0.4606 | 9.5954 | 0.6325 |
| `xception_df40` | 0.4865 | 0.3396 | 0.4000 | 0.5299 | 0.3621 | 11.1715 | 1.5488 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5238 | 1.0000 | 0.6875 | 0.6746 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5048 | 1.0000 | 0.6709 | 0.5299 | 1.0000 | 0.0000 |
