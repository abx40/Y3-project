# Benchmark Scoring Report V2

## Method

- Evaluation unit: 43-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 43-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0425
- Validation time-to-first-detection: 16.6832s
- Validation F1: 0.1860

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.40 | 0.30 | 0.0000 | 0.0000 | 0.3684 | 11.0868 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.87 | 0.77 | 0.0000 | 0.0000 | 0.2632 | 12.1052 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.57 | 0.47 | 0.0000 | 0.0000 | 0.6053 | 8.7068 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.5526 | 8.0139 |
| `videomae` | `strict_0.10` | `high` | 3 | 0.43 | 0.38 | 0.0000 | 0.0000 | 0.3684 | 10.1425 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.34 | 0.34 | 0.0000 | 0.0000 | 0.1579 | 14.5392 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6667 | 0.1081 | 0.1860 | 0.0425 | 0.0595 | 16.6832 | 0.7031 |
| 2 | `i3d` | 0.5000 | 0.0541 | 0.0976 | 0.0850 | 0.0595 | 17.2258 | 0.3529 |
| 3 | `efficientnet_b4` | 0.2727 | 0.0811 | 0.1250 | 0.1701 | 0.2424 | 15.8581 | 0.8917 |
| 4 | `effort_clip_l14` | 0.5556 | 0.5405 | 0.5479 | 0.2977 | 0.4833 | 9.5753 | 0.1558 |
| 5 | `f3net` | 0.5833 | 0.5676 | 0.5753 | 0.4252 | 0.4557 | 6.5587 | 0.8150 |
| 6 | `videomae` | 0.4516 | 0.3784 | 0.4118 | 0.4252 | 0.5138 | 11.7293 | 0.3165 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5882 | 0.2941 | 0.3922 | 0.1558 | 0.1916 | 13.3137 | 0.4979 |
| `effort_clip_l14` | 0.5806 | 0.5294 | 0.5538 | 0.2727 | 0.3591 | 7.9076 | 0.1811 |
| `f3net` | 0.6364 | 0.6176 | 0.6269 | 0.3506 | 0.3338 | 6.4170 | 0.8481 |
| `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0390 | 0.3338 | 18.2927 | 0.1167 |
| `videomae` | 0.3846 | 0.1471 | 0.2128 | 0.1558 | 0.2208 | 15.6219 | 0.2162 |
| `xception_df40` | 1.0000 | 0.1176 | 0.2105 | 0.0000 | 0.0000 | 16.7948 | 0.6977 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5286 | 1.0000 | 0.6916 | 0.6804 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4857 | 1.0000 | 0.6538 | 0.4675 | 1.0000 | 0.0000 |
