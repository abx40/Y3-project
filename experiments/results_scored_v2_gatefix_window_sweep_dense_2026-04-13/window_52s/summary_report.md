# Benchmark Scoring Report V2

## Method

- Evaluation unit: 52-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 52-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.1210
- Validation time-to-first-detection: 12.4283s
- Validation F1: 0.5217

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.33 | 0.28 | 0.0000 | 0.0000 | 0.3421 | 10.6819 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.05 | 0.05 | 0.1000 | 0.0467 | 0.6053 | 5.6498 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.6842 | 5.2414 |
| `i3d` | `strict_0.10` | `high` | 3 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.3947 | 10.5699 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.57 | 0.52 | 0.0000 | 0.0000 | 0.5263 | 8.6106 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.31 | 0.21 | 0.1000 | 0.0467 | 0.6579 | 5.3730 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.7500 | 0.4000 | 0.5217 | 0.1210 | 0.1237 | 12.4283 | 0.4592 |
| 2 | `f3net` | 0.4167 | 0.3333 | 0.3704 | 0.1613 | 0.4731 | 11.3960 | 0.3571 |
| 3 | `videomae` | 0.5789 | 0.3667 | 0.4490 | 0.1613 | 0.2634 | 11.5270 | 0.6383 |
| 4 | `i3d` | 0.3684 | 0.2333 | 0.2857 | 0.1613 | 0.3871 | 12.3776 | 0.1277 |
| 5 | `efficientnet_b4` | 0.4516 | 0.4667 | 0.4590 | 0.2419 | 0.5780 | 8.7641 | 0.2727 |
| 6 | `effort_clip_l14` | 0.4242 | 0.4667 | 0.4444 | 0.2823 | 0.6156 | 8.3116 | 0.2222 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5333 | 0.3077 | 0.3902 | 0.1435 | 0.1746 | 13.2275 | 0.3509 |
| `effort_clip_l14` | 0.4103 | 0.6154 | 0.4923 | 0.1794 | 0.6579 | 7.9213 | 0.1572 |
| `f3net` | 0.5217 | 0.4615 | 0.4898 | 0.2153 | 0.3134 | 9.5810 | 0.3737 |
| `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0359 | 0.0167 | 18.2927 | 2.1429 |
| `videomae` | 0.3939 | 0.5000 | 0.4407 | 0.2153 | 0.5789 | 8.8906 | 0.2963 |
| `xception_df40` | 0.6250 | 0.3846 | 0.4762 | 0.0359 | 0.1722 | 12.3270 | 0.3713 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5000 | 1.0000 | 0.6667 | 0.4032 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4333 | 1.0000 | 0.6047 | 0.3589 | 1.0000 | 0.0000 |
