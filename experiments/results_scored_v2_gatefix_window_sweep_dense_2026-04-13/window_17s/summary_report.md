# Benchmark Scoring Report V2

## Method

- Evaluation unit: 17-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 17-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1683
- Validation time-to-first-detection: 17.2987s
- Validation F1: 0.0588

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.69 | 0.59 | 0.0000 | 0.0000 | 0.6476 | 5.8262 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.13 | 0.03 | 0.0000 | 0.0000 | 0.5143 | 8.1444 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.61 | 0.51 | 0.0000 | 0.0000 | 0.4571 | 8.8185 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.4571 | 9.3534 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.56 | 0.46 | 0.1000 | 0.0367 | 0.6190 | 6.9665 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.50 | 0.50 | 0.0000 | 0.0000 | 0.1333 | 14.8789 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.3750 | 0.0319 | 0.0588 | 0.1683 | 0.0512 | 17.2987 | 1.4516 |
| 2 | `i3d` | 0.3056 | 0.2340 | 0.2651 | 0.4208 | 0.5708 | 14.1839 | 0.2041 |
| 3 | `effort_clip_l14` | 0.6607 | 0.3936 | 0.4933 | 0.4628 | 0.2013 | 10.8157 | 0.7965 |
| 4 | `f3net` | 0.6515 | 0.4574 | 0.5375 | 0.6732 | 0.2742 | 8.6650 | 1.3904 |
| 5 | `videomae` | 0.4828 | 0.4468 | 0.4641 | 0.7153 | 0.5196 | 9.3849 | 0.5773 |
| 6 | `efficientnet_b4` | 0.6044 | 0.5851 | 0.5946 | 0.8836 | 0.4123 | 7.4334 | 0.5162 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4609 | 0.6782 | 0.5488 | 1.1350 | 0.7339 | 4.9693 | 0.2820 |
| `effort_clip_l14` | 0.4095 | 0.4943 | 0.4479 | 0.7828 | 0.6641 | 9.3598 | 0.2763 |
| `f3net` | 0.6596 | 0.3563 | 0.4627 | 0.3523 | 0.1774 | 11.3122 | 1.4485 |
| `i3d` | 0.6667 | 0.4138 | 0.5106 | 0.3523 | 0.1840 | 10.9183 | 0.2041 |
| `videomae` | 0.5116 | 0.7586 | 0.6111 | 1.0568 | 0.6830 | 4.1252 | 0.3060 |
| `xception_df40` | 0.7333 | 0.1264 | 0.2157 | 0.1566 | 0.0444 | 15.8419 | 1.4118 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5222 | 1.0000 | 0.6861 | 1.6410 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4833 | 1.0000 | 0.6517 | 1.5656 | 1.0000 | 0.0000 |
