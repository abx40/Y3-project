# Benchmark Scoring Report V2

## Method

- Evaluation unit: 53-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 53-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.0782
- Validation time-to-first-detection: 12.9182s
- Validation F1: 0.3000

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.3243 | 11.2934 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.03 | 0.03 | 0.1000 | 0.0283 | 0.8108 | 3.8697 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.6486 | 6.3611 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.5946 | 6.1405 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.58 | 0.48 | 0.1000 | 0.0283 | 0.5405 | 7.2044 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.29 | 0.19 | 0.1000 | 0.0283 | 0.6757 | 5.4326 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.5455 | 0.2069 | 0.3000 | 0.0782 | 0.1492 | 12.9182 | 0.5484 |
| 2 | `i3d` | 0.3077 | 0.2759 | 0.2909 | 0.1564 | 0.5511 | 13.8785 | 0.1945 |
| 3 | `efficientnet_b4` | 0.4167 | 0.3448 | 0.3774 | 0.1954 | 0.4834 | 12.2675 | 0.4000 |
| 4 | `f3net` | 0.5333 | 0.5517 | 0.5424 | 0.2345 | 0.4834 | 9.1737 | 0.3162 |
| 5 | `xception_df40` | 0.4762 | 0.3448 | 0.4000 | 0.2345 | 0.3094 | 11.9595 | 0.3715 |
| 6 | `effort_clip_l14` | 0.4222 | 0.6552 | 0.5135 | 0.3127 | 0.8274 | 5.4131 | 0.1905 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5294 | 0.3000 | 0.3830 | 0.1660 | 0.2185 | 12.8265 | 0.3170 |
| `effort_clip_l14` | 0.4524 | 0.6333 | 0.5278 | 0.3320 | 0.7434 | 6.2980 | 0.1466 |
| `f3net` | 0.6087 | 0.4667 | 0.5283 | 0.3320 | 0.2801 | 9.1087 | 0.4860 |
| `i3d` | 0.3030 | 0.3333 | 0.3175 | 0.3320 | 0.7683 | 11.6313 | 0.1097 |
| `videomae` | 0.3793 | 0.3667 | 0.3729 | 0.2075 | 0.5851 | 11.4538 | 0.2584 |
| `xception_df40` | 0.5217 | 0.4000 | 0.4528 | 0.1245 | 0.3534 | 10.4375 | 0.3139 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4833 | 1.0000 | 0.6517 | 0.3909 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5000 | 1.0000 | 0.6667 | 0.4979 | 1.0000 | 0.0000 |
