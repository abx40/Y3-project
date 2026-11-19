# Benchmark Scoring Report V2

## Method

- Evaluation unit: 40-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 40-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1184
- Validation time-to-first-detection: 11.1040s
- Validation F1: 0.3824

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.63 | 0.58 | 0.0000 | 0.0000 | 0.5000 | 9.0506 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.86 | 0.76 | 0.0000 | 0.0000 | 0.3636 | 10.7248 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.7727 | 4.5541 |
| `i3d` | `strict_0.10` | `high` | 1 | 0.72 | 0.62 | 0.0000 | 0.0000 | 0.5682 | 7.4152 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.59 | 0.49 | 0.0000 | 0.0000 | 0.2273 | 11.9451 |
| `xception_df40` | `strict_0.10` | `high` | 1 | 0.93 | 0.83 | 0.0000 | 0.0000 | 0.1136 | 15.2195 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.4194 | 0.3514 | 0.3824 | 0.1184 | 0.4737 | 11.1040 | 0.3387 |
| 2 | `efficientnet_b4` | 0.5854 | 0.6486 | 0.6154 | 0.3158 | 0.4474 | 6.2034 | 0.1463 |
| 3 | `effort_clip_l14` | 0.4250 | 0.4595 | 0.4416 | 0.3158 | 0.6053 | 9.0921 | 0.1500 |
| 4 | `xception_df40` | 0.5714 | 0.3243 | 0.4138 | 0.3158 | 0.2368 | 12.1217 | 1.3571 |
| 5 | `f3net` | 0.6000 | 0.8108 | 0.6897 | 0.3947 | 0.5263 | 3.2509 | 0.2700 |
| 6 | `i3d` | 0.5500 | 0.5946 | 0.5714 | 0.3947 | 0.4737 | 6.6088 | 0.2625 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4167 | 0.7143 | 0.5263 | 0.4125 | 0.8750 | 4.4020 | 0.1750 |
| `effort_clip_l14` | 0.5429 | 0.5429 | 0.5429 | 0.3000 | 0.4000 | 7.4400 | 0.1714 |
| `f3net` | 0.5641 | 0.6286 | 0.5946 | 0.3375 | 0.4250 | 6.8684 | 0.3077 |
| `i3d` | 0.2903 | 0.2571 | 0.2727 | 0.2250 | 0.5500 | 12.7072 | 0.1935 |
| `videomae` | 0.3333 | 0.2286 | 0.2712 | 0.1875 | 0.4000 | 13.3513 | 0.5000 |
| `xception_df40` | 0.4000 | 0.1714 | 0.2400 | 0.1875 | 0.2250 | 13.9914 | 1.2000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4933 | 1.0000 | 0.6607 | 0.5132 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4667 | 1.0000 | 0.6364 | 0.4875 | 1.0000 | 0.0000 |
