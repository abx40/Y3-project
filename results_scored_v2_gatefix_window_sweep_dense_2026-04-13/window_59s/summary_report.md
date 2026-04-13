# Benchmark Scoring Report V2

## Method

- Evaluation unit: 59-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 59-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.1151
- Validation time-to-first-detection: 16.0455s
- Validation F1: 0.1429

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.38 | 0.28 | 0.0000 | 0.0000 | 0.4516 | 9.2781 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.90 | 0.80 | 0.0000 | 0.0000 | 0.3226 | 11.4343 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.58 | 0.48 | 0.0000 | 0.0000 | 0.4516 | 7.9222 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6774 | 7.5190 |
| `videomae` | `strict_0.10` | `high` | 2 | 0.48 | 0.38 | 0.0000 | 0.0000 | 0.4516 | 10.0380 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.17 | 0.17 | 0.1000 | 0.0167 | 0.7097 | 5.2425 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.1875 | 0.1154 | 0.1429 | 0.1151 | 0.4277 | 16.0455 | 0.1418 |
| 2 | `effort_clip_l14` | 0.3636 | 0.3077 | 0.3333 | 0.1918 | 0.4968 | 12.1176 | 0.4000 |
| 3 | `f3net` | 0.5714 | 0.6154 | 0.5926 | 0.2302 | 0.4527 | 5.8633 | 0.4492 |
| 4 | `xception_df40` | 0.5455 | 0.4615 | 0.5000 | 0.2302 | 0.3146 | 9.7616 | 0.3000 |
| 5 | `efficientnet_b4` | 0.4000 | 0.3846 | 0.3922 | 0.2302 | 0.5345 | 10.5097 | 0.4066 |
| 6 | `videomae` | 0.2381 | 0.1923 | 0.2128 | 0.2302 | 0.5409 | 12.9087 | 0.2747 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5625 | 0.3333 | 0.4186 | 0.1236 | 0.1827 | 13.0417 | 0.3209 |
| `effort_clip_l14` | 0.6111 | 0.4074 | 0.4889 | 0.1236 | 0.2500 | 10.0329 | 0.4146 |
| `f3net` | 0.6800 | 0.6296 | 0.6538 | 0.2885 | 0.2905 | 7.2292 | 0.6972 |
| `i3d` | 0.4286 | 0.4444 | 0.4364 | 0.1236 | 0.5810 | 9.1955 | 0.1196 |
| `videomae` | 0.5294 | 0.3333 | 0.4091 | 0.1648 | 0.2569 | 12.6326 | 0.3978 |
| `xception_df40` | 0.3704 | 0.3704 | 0.3704 | 0.1648 | 0.5879 | 9.9587 | 0.1718 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4727 | 1.0000 | 0.6420 | 0.4220 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4909 | 1.0000 | 0.6585 | 0.3297 | 1.0000 | 0.0000 |
