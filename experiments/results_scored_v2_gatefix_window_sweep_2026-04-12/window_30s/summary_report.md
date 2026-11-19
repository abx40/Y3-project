# Benchmark Scoring Report V2

## Method

- Evaluation unit: 30-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 30-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `efficientnet_b4`
- Validation false alerts/min: 0.2222
- Validation time-to-first-detection: 12.0778s
- Validation F1: 0.3200

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.73 | 0.63 | 0.0000 | 0.0000 | 0.3509 | 10.6344 |
| `effort_clip_l14` | `strict_0.10` | `low` | 2 | 0.19 | 0.09 | 0.1000 | 0.0500 | 0.3333 | 11.1514 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.49 | 0.39 | 0.0000 | 0.0000 | 0.6316 | 6.9221 |
| `i3d` | `strict_0.10` | `high` | 3 | 0.70 | 0.65 | 0.0000 | 0.0000 | 0.3860 | 9.6564 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.54 | 0.44 | 0.1000 | 0.0500 | 0.5439 | 7.7693 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.26 | 0.21 | 0.0000 | 0.0000 | 0.5088 | 7.2939 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `efficientnet_b4` | 0.4138 | 0.2609 | 0.3200 | 0.2222 | 0.3148 | 12.0778 | 0.6897 |
| 2 | `xception_df40` | 0.5769 | 0.3261 | 0.4167 | 0.2222 | 0.2037 | 12.5823 | 0.4615 |
| 3 | `i3d` | 0.4655 | 0.5870 | 0.5192 | 0.2593 | 0.5741 | 6.0687 | 0.2069 |
| 4 | `f3net` | 0.4918 | 0.6522 | 0.5607 | 0.2593 | 0.5741 | 7.3341 | 0.2623 |
| 5 | `effort_clip_l14` | 0.6316 | 0.5217 | 0.5714 | 0.2593 | 0.2593 | 9.4175 | 0.5789 |
| 6 | `videomae` | 0.4651 | 0.4348 | 0.4494 | 0.2593 | 0.4259 | 10.0944 | 0.6512 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5439 | 0.6458 | 0.5905 | 0.3077 | 0.5000 | 6.4575 | 0.5263 |
| `effort_clip_l14` | 0.3556 | 0.3333 | 0.3441 | 0.2692 | 0.5577 | 11.0543 | 0.5778 |
| `f3net` | 0.5000 | 0.4792 | 0.4894 | 0.2308 | 0.4423 | 8.4418 | 0.2609 |
| `i3d` | 0.2368 | 0.1875 | 0.2093 | 0.1923 | 0.5577 | 14.3075 | 0.2105 |
| `videomae` | 0.4932 | 0.7500 | 0.5950 | 0.3462 | 0.7115 | 4.9520 | 0.3288 |
| `xception_df40` | 0.5588 | 0.3958 | 0.4634 | 0.1923 | 0.2885 | 10.9243 | 0.5294 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4600 | 1.0000 | 0.6301 | 0.3704 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4800 | 1.0000 | 0.6486 | 0.4231 | 1.0000 | 0.0000 |
