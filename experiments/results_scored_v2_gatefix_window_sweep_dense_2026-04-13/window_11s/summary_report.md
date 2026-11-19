# Benchmark Scoring Report V2

## Method

- Evaluation unit: 11-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 11-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0804
- Validation time-to-first-detection: 16.3962s
- Validation F1: 0.1325

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.78 | 0.68 | 0.0000 | 0.0000 | 0.2988 | 11.8190 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2439 | 12.8455 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.59 | 0.49 | 0.0000 | 0.0000 | 0.3902 | 9.9523 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.8049 | 3.3902 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.72 | 0.62 | 0.1000 | 0.0367 | 0.3293 | 10.9053 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.58 | 0.48 | 0.0000 | 0.0000 | 0.0976 | 15.2922 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.7692 | 0.0725 | 0.1325 | 0.0804 | 0.0188 | 16.3962 | 1.8045 |
| 2 | `effort_clip_l14` | 0.6750 | 0.3913 | 0.4954 | 0.6032 | 0.1917 | 10.2620 | 0.5455 |
| 3 | `videomae` | 0.5862 | 0.2464 | 0.3469 | 0.6032 | 0.1736 | 12.9839 | 2.9384 |
| 4 | `efficientnet_b4` | 0.4444 | 0.1739 | 0.2500 | 0.6032 | 0.2178 | 14.4643 | 1.5280 |
| 5 | `i3d` | 0.2880 | 0.2609 | 0.2738 | 0.7641 | 0.6461 | 13.2019 | 0.2222 |
| 6 | `f3net` | 0.5167 | 0.4493 | 0.4806 | 0.9651 | 0.4276 | 9.0763 | 1.0000 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5981 | 0.4571 | 0.5182 | 0.9829 | 0.3229 | 8.8884 | 0.9176 |
| `effort_clip_l14` | 0.5000 | 0.2214 | 0.3069 | 0.7782 | 0.2294 | 14.3644 | 1.0714 |
| `f3net` | 0.5570 | 0.3143 | 0.4018 | 0.7372 | 0.2628 | 11.4011 | 1.1806 |
| `i3d` | 0.4599 | 0.4500 | 0.4549 | 0.5324 | 0.5488 | 9.7753 | 0.2011 |
| `videomae` | 0.5444 | 0.3500 | 0.4261 | 0.6553 | 0.3044 | 10.9458 | 2.0102 |
| `xception_df40` | 0.6957 | 0.1143 | 0.1963 | 0.2048 | 0.0491 | 15.8359 | 1.6935 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5018 | 1.0000 | 0.6683 | 1.9705 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5091 | 1.0000 | 0.6747 | 2.0887 | 1.0000 | 0.0000 |
