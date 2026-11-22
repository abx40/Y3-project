# Benchmark Scoring Report V2

## Method

- Evaluation unit: 57-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 57-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1367
- Validation time-to-first-detection: 9.3778s
- Validation F1: 0.5532

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.68 | 0.58 | 0.0000 | 0.0000 | 0.4839 | 7.3204 |
| `effort_clip_l14` | `strict_0.10` | `low` | 1 | 0.26 | 0.16 | 0.1000 | 0.0500 | 0.5161 | 9.3926 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.59 | 0.49 | 0.0000 | 0.0000 | 0.4839 | 8.1431 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7419 | 5.6122 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.56 | 0.46 | 0.0000 | 0.0000 | 0.5806 | 8.0615 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.15 | 0.15 | 0.1000 | 0.0500 | 0.8065 | 4.0893 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.7647 | 0.4333 | 0.5532 | 0.1367 | 0.1526 | 9.3778 | 0.2623 |
| 2 | `i3d` | 0.1765 | 0.1000 | 0.1277 | 0.1822 | 0.5444 | 16.2250 | 0.2027 |
| 3 | `effort_clip_l14` | 0.6316 | 0.4000 | 0.4898 | 0.2733 | 0.2620 | 10.0145 | 0.8746 |
| 4 | `f3net` | 0.6538 | 0.5667 | 0.6071 | 0.3189 | 0.3690 | 6.3353 | 0.5773 |
| 5 | `xception_df40` | 0.6429 | 0.6000 | 0.6207 | 0.3645 | 0.3918 | 6.1917 | 0.1980 |
| 6 | `efficientnet_b4` | 0.4783 | 0.3667 | 0.4151 | 0.3645 | 0.4374 | 11.5246 | 0.1995 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5517 | 0.6400 | 0.5926 | 0.2247 | 0.4457 | 4.9684 | 0.3002 |
| `effort_clip_l14` | 0.3750 | 0.2400 | 0.2927 | 0.2622 | 0.3390 | 13.8407 | 1.2203 |
| `f3net` | 0.5909 | 0.5200 | 0.5532 | 0.1873 | 0.3034 | 7.5571 | 0.7000 |
| `i3d` | 0.5000 | 0.5600 | 0.5283 | 0.1498 | 0.4644 | 9.1224 | 0.1188 |
| `videomae` | 0.4839 | 0.6000 | 0.5357 | 0.1873 | 0.5187 | 8.0708 | 0.2170 |
| `xception_df40` | 0.4516 | 0.5600 | 0.5000 | 0.2247 | 0.5543 | 8.2366 | 0.1447 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5455 | 1.0000 | 0.7059 | 0.5923 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4545 | 1.0000 | 0.6250 | 0.4120 | 1.0000 | 0.0000 |
