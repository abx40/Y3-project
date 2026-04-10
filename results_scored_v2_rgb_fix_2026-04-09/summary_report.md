# Benchmark Scoring Report V2

## Method

- Evaluation unit: 1-second window.
- Base signal: per-second mean of `predictions.csv:score_raw` after end-aligning each run to the video end. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min on `calibration_real_only_01`, relaxed <= 0.25 FA/min.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 2.0027
- Validation time-to-first-detection: 0.1667s
- Validation F1: 0.6622

## Selected Operating Points

| backend | selected policy | polarity | persistence_s | t_on | t_off | real-only FA/min | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 1 | 0.05 | 0.05 | 0.1000 | 1.0000 | 0.0000 |
| `effort_clip_l14` | `strict_0.10` | `low` | 1 | 0.05 | 0.00 | 0.1000 | 0.9911 | 0.1526 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.0000 |
| `i3d` | `strict_0.10` | `low` | 1 | 0.01 | 0.01 | 0.1000 | 0.9950 | 0.0000 |
| `videomae` | `strict_0.10` | `low` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.0000 |
| `xception_df40` | `strict_0.10` | `low` | 1 | 0.05 | 0.00 | 0.1000 | 0.9839 | 0.2788 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.5002 | 0.9794 | 0.6622 | 2.0027 | 0.1667 | 0.0816 |
| 2 | `f3net` | 0.4985 | 0.9907 | 0.6632 | 2.0427 | 0.0000 | 0.0603 |
| 3 | `effort_clip_l14` | 0.5041 | 0.9887 | 0.6677 | 2.0427 | 0.2024 | 0.0815 |
| 4 | `videomae` | 0.4977 | 0.9880 | 0.6619 | 2.0427 | 0.2143 | 0.0201 |
| 5 | `xception_df40` | 0.5007 | 0.9873 | 0.6644 | 2.0427 | 0.2262 | 0.0810 |
| 6 | `efficientnet_b4` | 0.5000 | 0.9920 | 0.6649 | 2.2029 | 0.0213 | 0.5638 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4993 | 0.9980 | 0.6656 | 2.2355 | 0.0000 | 0.2807 |
| `effort_clip_l14` | 0.4964 | 0.9793 | 0.6589 | 2.0359 | 0.3821 | 0.0610 |
| `f3net` | 0.4990 | 1.0000 | 0.6658 | 2.0758 | 0.0000 | 0.0000 |
| `i3d` | 0.4990 | 1.0000 | 0.6658 | 2.0758 | 0.0000 | 0.0000 |
| `videomae` | 0.4985 | 0.9980 | 0.6649 | 2.0758 | 0.0366 | 0.0400 |
| `xception_df40` | 0.4963 | 0.9893 | 0.6610 | 2.0758 | 0.1951 | 0.0201 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5007 | 1.0000 | 0.6673 | 2.0427 | 0.0000 |
| `always_fake` | `test` | 0.4990 | 1.0000 | 0.6658 | 2.0758 | 0.0000 |
