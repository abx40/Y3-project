# Benchmark Scoring Report V2

## Method

- Evaluation unit: 44-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 44-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0402
- Validation time-to-first-detection: 17.2258s
- Validation F1: 0.1026

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.40 | 0.30 | 0.0000 | 0.0000 | 0.2750 | 12.1137 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.77 | 0.0000 | 0.0000 | 0.3250 | 10.9125 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.59 | 0.49 | 0.0000 | 0.0000 | 0.5250 | 9.3397 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.5250 | 9.1206 |
| `videomae` | `strict_0.10` | `high` | 3 | 0.42 | 0.42 | 0.0000 | 0.0000 | 0.4500 | 9.5253 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.23 | 0.13 | 0.1000 | 0.0467 | 0.4750 | 9.3333 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.5000 | 0.0571 | 0.1026 | 0.0402 | 0.0483 | 17.2258 | 0.3750 |
| 2 | `xception_df40` | 0.5000 | 0.1429 | 0.2222 | 0.0804 | 0.1367 | 15.4188 | 0.5882 |
| 3 | `efficientnet_b4` | 0.4762 | 0.2857 | 0.3571 | 0.1206 | 0.3244 | 12.7796 | 0.4626 |
| 4 | `effort_clip_l14` | 0.5000 | 0.5429 | 0.5205 | 0.2011 | 0.5389 | 9.6824 | 0.1493 |
| 5 | `videomae` | 0.5455 | 0.5143 | 0.5294 | 0.2011 | 0.4209 | 10.4333 | 0.3458 |
| 6 | `f3net` | 0.5625 | 0.5143 | 0.5373 | 0.3619 | 0.4021 | 7.8272 | 0.9914 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4348 | 0.3448 | 0.3846 | 0.2069 | 0.3011 | 11.7548 | 0.5063 |
| `effort_clip_l14` | 0.5484 | 0.5862 | 0.5667 | 0.2414 | 0.3264 | 8.1390 | 0.1846 |
| `f3net` | 0.6667 | 0.6207 | 0.6429 | 0.2069 | 0.2184 | 7.9990 | 0.9343 |
| `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0345 | 0.2943 | 18.2927 | 0.1172 |
| `videomae` | 0.2727 | 0.1034 | 0.1500 | 0.1379 | 0.1839 | 15.8250 | 0.5310 |
| `xception_df40` | 0.5161 | 0.5517 | 0.5333 | 0.1379 | 0.3609 | 9.5806 | 0.1368 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5000 | 1.0000 | 0.6667 | 0.4424 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4143 | 1.0000 | 0.5859 | 0.4483 | 1.0000 | 0.0000 |
