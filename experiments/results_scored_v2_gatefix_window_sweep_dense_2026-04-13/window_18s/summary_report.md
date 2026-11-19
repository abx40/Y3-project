# Benchmark Scoring Report V2

## Method

- Evaluation unit: 18-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 18-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.2326
- Validation time-to-first-detection: 13.5617s
- Validation F1: 0.3273

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.67 | 0.57 | 0.0000 | 0.0000 | 0.7129 | 5.5200 |
| `effort_clip_l14` | `strict_0.10` | `low` | 2 | 0.31 | 0.21 | 0.1000 | 0.0100 | 0.2178 | 13.2464 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.6634 | 5.8398 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.33 | 0.23 | 0.0000 | 0.0000 | 0.7129 | 5.0925 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.58 | 0.48 | 0.0000 | 0.0000 | 0.4752 | 8.3139 |
| `xception_df40` | `strict_0.10` | `high` | 1 | 0.95 | 0.85 | 0.1000 | 0.0300 | 0.2079 | 12.4150 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.6429 | 0.2195 | 0.3273 | 0.2326 | 0.1008 | 13.5617 | 2.0000 |
| 2 | `i3d` | 0.5263 | 0.3659 | 0.4317 | 0.4264 | 0.2984 | 11.2612 | 0.2454 |
| 3 | `videomae` | 0.4286 | 0.3293 | 0.3724 | 0.4264 | 0.4031 | 11.7624 | 0.5405 |
| 4 | `xception_df40` | 0.5094 | 0.3293 | 0.4000 | 0.5814 | 0.2946 | 11.5201 | 1.8710 |
| 5 | `f3net` | 0.5152 | 0.6220 | 0.5635 | 0.6589 | 0.5426 | 6.7910 | 0.3460 |
| 6 | `efficientnet_b4` | 0.5529 | 0.5732 | 0.5629 | 0.6977 | 0.4264 | 6.8100 | 0.4781 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5177 | 0.8391 | 0.6404 | 1.2863 | 0.8133 | 2.1304 | 0.1446 |
| `effort_clip_l14` | 0.3600 | 0.1034 | 0.1607 | 0.4149 | 0.1992 | 15.9526 | 2.4000 |
| `f3net` | 0.5735 | 0.4483 | 0.5032 | 0.4979 | 0.3444 | 10.3794 | 0.3535 |
| `i3d` | 0.6944 | 0.5747 | 0.6289 | 0.5394 | 0.2573 | 8.3248 | 0.1429 |
| `videomae` | 0.5098 | 0.5977 | 0.5503 | 0.8714 | 0.6058 | 7.1768 | 0.5667 |
| `xception_df40` | 0.6042 | 0.3333 | 0.4296 | 0.6639 | 0.2199 | 11.7306 | 2.1014 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4824 | 1.0000 | 0.6508 | 1.2403 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5118 | 1.0000 | 0.6770 | 1.4108 | 1.0000 | 0.0000 |
