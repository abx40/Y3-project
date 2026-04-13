# Benchmark Scoring Report V2

## Method

- Evaluation unit: 34-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 34-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0803
- Validation time-to-first-detection: 16.1488s
- Validation F1: 0.1111

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.36 | 0.26 | 0.0000 | 0.0000 | 0.4314 | 10.3253 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.0784 | 14.7109 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.5490 | 7.7866 |
| `i3d` | `strict_0.10` | `high` | 1 | 0.72 | 0.62 | 0.1000 | 0.0367 | 0.5686 | 6.5127 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.50 | 0.45 | 0.0000 | 0.0000 | 0.6667 | 6.4380 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.43 | 0.43 | 0.0000 | 0.0000 | 0.1373 | 14.2760 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.3333 | 0.0667 | 0.1111 | 0.0803 | 0.1285 | 16.1488 | 1.0204 |
| 2 | `effort_clip_l14` | 0.7500 | 0.3333 | 0.4615 | 0.1606 | 0.1058 | 11.7598 | 1.3473 |
| 3 | `videomae` | 0.6757 | 0.5556 | 0.6098 | 0.2410 | 0.2570 | 8.3336 | 0.2455 |
| 4 | `efficientnet_b4` | 0.4912 | 0.6222 | 0.5490 | 0.3213 | 0.6439 | 5.7161 | 0.3175 |
| 5 | `f3net` | 0.5208 | 0.5556 | 0.5376 | 0.4016 | 0.5154 | 5.8442 | 0.4887 |
| 6 | `i3d` | 0.6066 | 0.8222 | 0.6981 | 0.4418 | 0.5221 | 3.3157 | 0.1787 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5278 | 0.4318 | 0.4750 | 0.2375 | 0.3575 | 9.5017 | 0.5102 |
| `effort_clip_l14` | 0.3158 | 0.1364 | 0.1905 | 0.3166 | 0.2836 | 13.4204 | 1.5434 |
| `f3net` | 0.5789 | 0.5000 | 0.5366 | 0.3562 | 0.3430 | 8.0027 | 0.5255 |
| `i3d` | 0.3684 | 0.3182 | 0.3415 | 0.2770 | 0.5066 | 12.6565 | 0.3409 |
| `videomae` | 0.4524 | 0.4318 | 0.4419 | 0.2375 | 0.4921 | 10.4466 | 0.2609 |
| `xception_df40` | 0.8889 | 0.1818 | 0.3019 | 0.0396 | 0.0224 | 14.6967 | 1.1765 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5000 | 1.0000 | 0.6667 | 0.5622 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4889 | 1.0000 | 0.6567 | 0.4354 | 1.0000 | 0.0000 |
