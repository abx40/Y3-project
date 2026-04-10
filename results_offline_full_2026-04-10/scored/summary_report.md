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
- Validation false alerts/min: 1.9626
- Validation time-to-first-detection: 0.2920s
- Validation F1: 0.6677

## Selected Operating Points

| backend | selected policy | polarity | persistence_s | t_on | t_off | real-only FA/min | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.0000 |
| `effort_clip_l14` | `strict_0.10` | `high` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.0000 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.10 | 0.00 | 0.1000 | 1.0000 | 0.0000 |
| `i3d` | `strict_0.10` | `high` | 1 | 0.01 | 0.01 | 0.1000 | 0.9772 | 0.3958 |
| `videomae` | `strict_0.10` | `low` | 1 | 0.05 | 0.00 | 0.1000 | 0.9761 | 0.4115 |
| `xception_df40` | `strict_0.10` | `high` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.0000 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.5053 | 0.9840 | 0.6677 | 1.9626 | 0.2920 | 0.1026 |
| 2 | `videomae` | 0.5046 | 0.9814 | 0.6665 | 1.9626 | 0.3396 | 0.1027 |
| 3 | `xception_df40` | 0.5030 | 0.9993 | 0.6692 | 2.0027 | 0.0060 | 0.0201 |
| 4 | `f3net` | 0.5010 | 1.0000 | 0.6676 | 2.0427 | 0.0000 | 0.0200 |
| 5 | `efficientnet_b4` | 0.5008 | 1.0000 | 0.6674 | 2.0427 | 0.0000 | 0.0200 |
| 6 | `effort_clip_l14` | 0.5007 | 1.0000 | 0.6673 | 2.0427 | 0.0000 | 0.0000 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4988 | 0.9993 | 0.6655 | 2.0758 | 0.0122 | 0.0200 |
| `effort_clip_l14` | 0.4988 | 0.9993 | 0.6655 | 2.0758 | 0.0122 | 0.0200 |
| `f3net` | 0.4988 | 0.9993 | 0.6655 | 2.0758 | 0.0122 | 0.0200 |
| `i3d` | 0.5014 | 0.9793 | 0.6632 | 2.0359 | 0.3780 | 0.1026 |
| `videomae` | 0.4998 | 0.9699 | 0.6597 | 2.0359 | 0.5431 | 0.1033 |
| `xception_df40` | 0.4988 | 0.9993 | 0.6655 | 2.0758 | 0.0122 | 0.0200 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5007 | 1.0000 | 0.6673 | 2.0427 | 0.0000 |
| `always_fake` | `test` | 0.4990 | 1.0000 | 0.6658 | 2.0758 | 0.0000 |
