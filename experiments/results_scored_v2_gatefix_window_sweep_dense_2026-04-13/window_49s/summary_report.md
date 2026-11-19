# Benchmark Scoring Report V2

## Method

- Evaluation unit: 49-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 49-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1158
- Validation time-to-first-detection: 16.0441s
- Validation F1: 0.1667

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.32 | 0.27 | 0.1000 | 0.0200 | 0.5238 | 9.3370 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.10 | 0.05 | 0.1000 | 0.0200 | 0.6667 | 7.7663 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.5952 | 6.4084 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6429 | 6.5106 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.60 | 0.50 | 0.1000 | 0.0200 | 0.6190 | 7.4647 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.18 | 0.18 | 0.1000 | 0.0200 | 0.4286 | 8.9953 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.2353 | 0.1290 | 0.1667 | 0.1158 | 0.3859 | 16.0441 | 0.3769 |
| 2 | `i3d` | 0.4324 | 0.5161 | 0.4706 | 0.1929 | 0.5904 | 9.5144 | 0.1843 |
| 3 | `xception_df40` | 0.4500 | 0.2903 | 0.3529 | 0.1929 | 0.2990 | 12.8211 | 0.3452 |
| 4 | `efficientnet_b4` | 0.5000 | 0.4839 | 0.4918 | 0.2701 | 0.4489 | 10.2102 | 0.2208 |
| 5 | `effort_clip_l14` | 0.5417 | 0.4194 | 0.4727 | 0.2701 | 0.2990 | 10.3915 | 0.2817 |
| 6 | `f3net` | 0.5000 | 0.5806 | 0.5373 | 0.3473 | 0.5196 | 8.1649 | 0.3713 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.1429 | 0.0357 | 0.0571 | 0.1081 | 0.1099 | 18.1063 | 0.7759 |
| `effort_clip_l14` | 0.2069 | 0.2143 | 0.2105 | 0.1802 | 0.6102 | 13.4572 | 0.2290 |
| `f3net` | 0.5484 | 0.6071 | 0.5763 | 0.2523 | 0.3676 | 7.1141 | 0.3835 |
| `i3d` | 0.3659 | 0.5357 | 0.4348 | 0.1802 | 0.6985 | 8.4526 | 0.1290 |
| `videomae` | 0.3158 | 0.2143 | 0.2553 | 0.2162 | 0.3381 | 13.7904 | 0.7001 |
| `xception_df40` | 0.4194 | 0.4643 | 0.4407 | 0.1802 | 0.4631 | 8.6717 | 0.2983 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4769 | 1.0000 | 0.6458 | 0.5016 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4308 | 1.0000 | 0.6022 | 0.3604 | 1.0000 | 0.0000 |
