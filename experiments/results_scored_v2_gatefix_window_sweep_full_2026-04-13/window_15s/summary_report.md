# Benchmark Scoring Report V2

## Method

- Evaluation unit: 15-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 15-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.2857
- Validation time-to-first-detection: 16.0607s
- Validation F1: 0.1626

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 1 | 0.53 | 0.43 | 0.1000 | 0.0500 | 0.1290 | 14.0700 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.1774 | 14.3823 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.5484 | 7.5046 |
| `i3d` | `strict_0.10` | `high` | 3 | 0.71 | 0.61 | 0.0000 | 0.0000 | 0.6129 | 6.8531 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.69 | 0.59 | 0.0000 | 0.0000 | 0.3790 | 10.5004 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.52 | 0.52 | 0.1000 | 0.0500 | 0.1613 | 14.1346 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.4762 | 0.0980 | 0.1626 | 0.2857 | 0.1122 | 16.0607 | 1.7143 |
| 2 | `videomae` | 0.4865 | 0.1765 | 0.2590 | 0.3673 | 0.1939 | 14.3568 | 1.8378 |
| 3 | `efficientnet_b4` | 0.5143 | 0.1765 | 0.2628 | 0.4082 | 0.1735 | 14.0271 | 3.5429 |
| 4 | `effort_clip_l14` | 0.7069 | 0.4020 | 0.5125 | 0.5306 | 0.1735 | 10.6538 | 0.7586 |
| 5 | `f3net` | 0.6190 | 0.6373 | 0.6280 | 0.9388 | 0.4082 | 6.3103 | 0.4571 |
| 6 | `i3d` | 0.5214 | 0.5980 | 0.5571 | 1.0204 | 0.5714 | 7.2842 | 0.1709 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5185 | 0.1359 | 0.2154 | 0.4124 | 0.1340 | 14.6722 | 4.7407 |
| `effort_clip_l14` | 0.5000 | 0.2136 | 0.2993 | 0.6598 | 0.2268 | 13.5214 | 1.0000 |
| `f3net` | 0.5455 | 0.5243 | 0.5347 | 0.7010 | 0.4639 | 8.1537 | 0.3232 |
| `i3d` | 0.4211 | 0.4660 | 0.4424 | 0.9485 | 0.6804 | 9.8951 | 0.1754 |
| `videomae` | 0.5890 | 0.4175 | 0.4886 | 0.5773 | 0.3093 | 10.2679 | 1.3151 |
| `xception_df40` | 0.6923 | 0.1748 | 0.2791 | 0.2474 | 0.0825 | 14.7473 | 2.0000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5100 | 1.0000 | 0.6755 | 1.7143 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5150 | 1.0000 | 0.6799 | 1.6907 | 1.0000 | 0.0000 |
