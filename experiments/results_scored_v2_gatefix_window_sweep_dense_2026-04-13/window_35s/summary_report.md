# Benchmark Scoring Report V2

## Method

- Evaluation unit: 35-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 35-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0772
- Validation time-to-first-detection: 17.1582s
- Validation F1: 0.0000

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.39 | 0.29 | 0.1000 | 0.0083 | 0.2941 | 11.7139 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.0000 | 0.0000 | 0.2157 | 13.5737 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.53 | 0.48 | 0.0000 | 0.0000 | 0.6078 | 7.4741 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.30 | 0.25 | 0.0000 | 0.0000 | 0.7059 | 5.7996 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.60 | 0.50 | 0.1000 | 0.0083 | 0.5294 | 7.8901 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.45 | 0.45 | 0.0000 | 0.0000 | 0.1373 | 14.8118 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.0000 | 0.0000 | 0.0000 | 0.0772 | 0.1158 | 17.1582 | 1.0000 |
| 2 | `i3d` | 0.3684 | 0.3256 | 0.3457 | 0.1158 | 0.5209 | 11.3538 | 0.1452 |
| 3 | `videomae` | 0.4000 | 0.2326 | 0.2941 | 0.1158 | 0.3183 | 12.2475 | 0.7811 |
| 4 | `efficientnet_b4` | 0.5714 | 0.4651 | 0.5128 | 0.1543 | 0.3183 | 9.0029 | 0.4120 |
| 5 | `effort_clip_l14` | 0.6667 | 0.3256 | 0.4375 | 0.1929 | 0.1383 | 11.3012 | 1.4222 |
| 6 | `f3net` | 0.3617 | 0.3953 | 0.3778 | 0.3087 | 0.6559 | 9.1332 | 0.3028 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5455 | 0.2353 | 0.3288 | 0.1446 | 0.2088 | 13.8282 | 0.3692 |
| `effort_clip_l14` | 0.7500 | 0.2353 | 0.3582 | 0.1446 | 0.0643 | 14.8921 | 1.6596 |
| `f3net` | 0.7333 | 0.4314 | 0.5432 | 0.2410 | 0.1767 | 10.9002 | 0.3125 |
| `i3d` | 0.5161 | 0.6275 | 0.5664 | 0.4819 | 0.7711 | 5.8916 | 0.1171 |
| `videomae` | 0.5455 | 0.4706 | 0.5053 | 0.3855 | 0.5141 | 8.8686 | 0.4552 |
| `xception_df40` | 0.7000 | 0.1373 | 0.2295 | 0.1446 | 0.0602 | 14.8187 | 1.3125 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4778 | 1.0000 | 0.6466 | 0.4244 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5667 | 1.0000 | 0.7234 | 0.6265 | 1.0000 | 0.0000 |
