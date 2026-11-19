# Benchmark Scoring Report V2

## Method

- Evaluation unit: 25-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 25-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.1627
- Validation time-to-first-detection: 10.0496s
- Validation F1: 0.4602

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.70 | 0.60 | 0.0000 | 0.0000 | 0.5571 | 6.5329 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.12 | 0.02 | 0.0000 | 0.0000 | 0.6286 | 6.3126 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.47 | 0.37 | 0.0000 | 0.0000 | 0.7571 | 4.7529 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.5571 | 7.4629 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.5429 | 8.8272 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.30 | 0.20 | 0.1000 | 0.0417 | 0.5286 | 9.1962 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.5000 | 0.4262 | 0.4602 | 0.1627 | 0.4407 | 10.0496 | 0.2308 |
| 2 | `i3d` | 0.4828 | 0.4590 | 0.4706 | 0.2441 | 0.5085 | 10.7397 | 0.2069 |
| 3 | `effort_clip_l14` | 0.5238 | 0.3607 | 0.4272 | 0.3254 | 0.3390 | 9.8721 | 0.5143 |
| 4 | `xception_df40` | 0.4333 | 0.2131 | 0.2857 | 0.3661 | 0.2881 | 13.8062 | 0.5600 |
| 5 | `efficientnet_b4` | 0.4444 | 0.3279 | 0.3774 | 0.4881 | 0.4237 | 11.9650 | 0.3733 |
| 6 | `f3net` | 0.5250 | 0.6885 | 0.5957 | 0.6508 | 0.6441 | 5.7852 | 0.2400 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5506 | 0.7778 | 0.6447 | 0.6316 | 0.7018 | 3.5336 | 0.1618 |
| `effort_clip_l14` | 0.4615 | 0.3810 | 0.4174 | 0.3789 | 0.4912 | 11.0468 | 0.2308 |
| `f3net` | 0.5441 | 0.5873 | 0.5649 | 0.5474 | 0.5439 | 7.9075 | 0.3176 |
| `i3d` | 0.4952 | 0.8254 | 0.6190 | 0.8421 | 0.9298 | 2.9827 | 0.1143 |
| `videomae` | 0.5244 | 0.6825 | 0.5931 | 0.4632 | 0.6842 | 5.5828 | 0.1756 |
| `xception_df40` | 0.7778 | 0.3333 | 0.4667 | 0.1263 | 0.1053 | 11.9980 | 0.7111 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5083 | 1.0000 | 0.6740 | 0.8136 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5250 | 1.0000 | 0.6885 | 0.8421 | 1.0000 | 0.0000 |
