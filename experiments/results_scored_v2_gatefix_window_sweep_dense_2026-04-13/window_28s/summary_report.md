# Benchmark Scoring Report V2

## Method

- Evaluation unit: 28-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 28-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0381
- Validation time-to-first-detection: 16.2614s
- Validation F1: 0.2333

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.68 | 0.58 | 0.0000 | 0.0000 | 0.5075 | 9.3344 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.99 | 0.89 | 0.1000 | 0.0467 | 0.2687 | 12.6020 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.5821 | 6.6734 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.33 | 0.28 | 0.0000 | 0.0000 | 0.4627 | 9.4383 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.46 | 0.36 | 0.1000 | 0.0200 | 0.6418 | 6.7012 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.31 | 0.21 | 0.1000 | 0.0200 | 0.3433 | 11.4945 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.8750 | 0.1346 | 0.2333 | 0.0381 | 0.0178 | 16.2614 | 0.8654 |
| 2 | `xception_df40` | 0.4615 | 0.1154 | 0.1846 | 0.1523 | 0.1041 | 16.1846 | 0.7229 |
| 3 | `effort_clip_l14` | 0.7083 | 0.3269 | 0.4474 | 0.2284 | 0.1142 | 11.5349 | 1.5000 |
| 4 | `efficientnet_b4` | 0.2353 | 0.1538 | 0.1860 | 0.2665 | 0.4315 | 14.5764 | 0.4054 |
| 5 | `f3net` | 0.6346 | 0.6346 | 0.6346 | 0.3046 | 0.3376 | 7.0301 | 0.4213 |
| 6 | `videomae` | 0.5075 | 0.6538 | 0.5714 | 0.3046 | 0.5558 | 7.3030 | 0.1670 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5244 | 0.8431 | 0.6466 | 0.4534 | 0.6474 | 3.7095 | 0.1354 |
| `effort_clip_l14` | 0.5000 | 0.2745 | 0.3544 | 0.3023 | 0.2267 | 11.2974 | 1.7120 |
| `f3net` | 0.5897 | 0.4510 | 0.5111 | 0.3401 | 0.2620 | 10.2918 | 0.6322 |
| `i3d` | 0.5143 | 0.3529 | 0.4186 | 0.1889 | 0.2796 | 11.5289 | 0.3797 |
| `videomae` | 0.5000 | 0.8039 | 0.6165 | 0.4156 | 0.6927 | 4.0745 | 0.1613 |
| `xception_df40` | 0.5806 | 0.3529 | 0.4390 | 0.1889 | 0.2191 | 11.0798 | 0.3521 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4727 | 1.0000 | 0.6420 | 0.6091 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4636 | 1.0000 | 0.6335 | 0.5668 | 1.0000 | 0.0000 |
