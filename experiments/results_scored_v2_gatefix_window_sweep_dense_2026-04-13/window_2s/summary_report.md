# Benchmark Scoring Report V2

## Method

- Evaluation unit: 2-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 2-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0399
- Validation time-to-first-detection: 17.5331s
- Validation F1: 0.0210

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.89 | 0.79 | 0.1000 | 0.0200 | 0.1293 | 14.3493 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.84 | 0.79 | 0.1000 | 0.0267 | 0.0424 | 16.3406 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.79 | 0.69 | 0.0000 | 0.0000 | 0.1973 | 12.1227 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.44 | 0.34 | 0.1000 | 0.0267 | 0.0212 | 16.8362 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.88 | 0.78 | 0.1000 | 0.0200 | 0.2174 | 11.7109 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.96 | 0.96 | 0.1000 | 0.0233 | 0.0078 | 16.9597 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6667 | 0.0107 | 0.0210 | 0.0399 | 0.0053 | 17.5331 | 10.0000 |
| 2 | `i3d` | 0.5000 | 0.0120 | 0.0235 | 0.1198 | 0.0120 | 17.5107 | 10.0000 |
| 3 | `effort_clip_l14` | 0.5747 | 0.0668 | 0.1196 | 0.1598 | 0.0493 | 16.2535 | 4.1379 |
| 4 | `efficientnet_b4` | 0.6667 | 0.0908 | 0.1598 | 0.3995 | 0.0453 | 15.6024 | 6.1765 |
| 5 | `videomae` | 0.5579 | 0.1736 | 0.2648 | 0.7190 | 0.1371 | 13.1841 | 6.0515 |
| 6 | `f3net` | 0.5521 | 0.2123 | 0.3067 | 0.9987 | 0.1718 | 11.2193 | 7.2917 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6137 | 0.1907 | 0.2909 | 0.5600 | 0.1200 | 13.9148 | 3.8627 |
| `effort_clip_l14` | 0.2437 | 0.0387 | 0.0667 | 0.5200 | 0.1200 | 17.1445 | 6.5546 |
| `f3net` | 0.6287 | 0.2573 | 0.3652 | 0.9200 | 0.1520 | 11.5937 | 5.6678 |
| `i3d` | 0.2273 | 0.0067 | 0.0130 | 0.1600 | 0.0227 | 18.1652 | 10.9091 |
| `videomae` | 0.6062 | 0.2853 | 0.3880 | 0.7200 | 0.1853 | 11.5479 | 3.7394 |
| `xception_df40` | 0.5000 | 0.0067 | 0.0132 | 0.0400 | 0.0067 | 18.1643 | 6.0000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4993 | 1.0000 | 0.6661 | 2.0373 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5000 | 1.0000 | 0.6667 | 2.0800 | 1.0000 | 0.0000 |
