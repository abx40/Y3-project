# Benchmark Scoring Report V2

## Method

- Evaluation unit: 8-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 8-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0798
- Validation time-to-first-detection: 16.5387s
- Validation F1: 0.0792

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.85 | 0.75 | 0.0000 | 0.0000 | 0.1126 | 14.6925 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.32 | 0.22 | 0.1000 | 0.0133 | 0.1577 | 13.9115 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.57 | 0.47 | 0.1000 | 0.0400 | 0.5450 | 7.3859 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.32 | 0.1000 | 0.0400 | 0.2838 | 11.3118 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.67 | 0.57 | 0.1000 | 0.0400 | 0.4459 | 9.4289 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.69 | 0.59 | 0.1000 | 0.0400 | 0.0676 | 15.5303 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5333 | 0.0428 | 0.0792 | 0.0798 | 0.0372 | 16.5387 | 4.0000 |
| 2 | `efficientnet_b4` | 0.8000 | 0.0856 | 0.1546 | 0.1197 | 0.0213 | 16.2186 | 2.2500 |
| 3 | `effort_clip_l14` | 0.6076 | 0.2567 | 0.3609 | 0.5186 | 0.1649 | 12.6737 | 2.0886 |
| 4 | `i3d` | 0.3881 | 0.1390 | 0.2047 | 0.5585 | 0.2181 | 15.0177 | 2.6866 |
| 5 | `videomae` | 0.4667 | 0.2246 | 0.3032 | 0.5984 | 0.2553 | 12.9090 | 1.9167 |
| 6 | `f3net` | 0.5435 | 0.5348 | 0.5391 | 1.1968 | 0.4468 | 7.5387 | 1.1821 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6667 | 0.1675 | 0.2678 | 0.2446 | 0.0870 | 15.0390 | 1.8750 |
| `effort_clip_l14` | 0.3284 | 0.1152 | 0.1705 | 0.5707 | 0.2446 | 15.9539 | 2.2388 |
| `f3net` | 0.6054 | 0.4660 | 0.5266 | 1.0190 | 0.3152 | 9.4280 | 1.4286 |
| `i3d` | 0.6508 | 0.2147 | 0.3228 | 0.3668 | 0.1196 | 13.4601 | 3.0952 |
| `videomae` | 0.5733 | 0.4503 | 0.5044 | 0.7745 | 0.3478 | 9.7268 | 1.3000 |
| `xception_df40` | 0.6786 | 0.0995 | 0.1735 | 0.2038 | 0.0489 | 16.2557 | 1.8750 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4987 | 1.0000 | 0.6655 | 1.9947 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5093 | 1.0000 | 0.6749 | 2.1196 | 1.0000 | 0.0000 |
