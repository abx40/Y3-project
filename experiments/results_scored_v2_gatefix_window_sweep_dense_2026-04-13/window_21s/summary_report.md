# Benchmark Scoring Report V2

## Method

- Evaluation unit: 21-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 21-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.1992
- Validation time-to-first-detection: 13.4038s
- Validation F1: 0.3191

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.39 | 0.34 | 0.0000 | 0.0000 | 0.3537 | 10.9257 |
| `effort_clip_l14` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.1000 | 0.0200 | 0.1463 | 14.5894 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.50 | 0.40 | 0.0000 | 0.0000 | 0.6463 | 6.0481 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.6585 | 6.2135 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.52 | 0.42 | 0.1000 | 0.0200 | 0.6220 | 6.3696 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.35 | 0.25 | 0.0000 | 0.0000 | 0.2927 | 12.5145 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.6818 | 0.2083 | 0.3191 | 0.1992 | 0.0857 | 13.4038 | 1.8919 |
| 2 | `efficientnet_b4` | 0.4286 | 0.1250 | 0.1935 | 0.1992 | 0.1614 | 15.7647 | 1.1348 |
| 3 | `xception_df40` | 0.2105 | 0.0556 | 0.0879 | 0.2789 | 0.1972 | 16.5505 | 0.9449 |
| 4 | `i3d` | 0.4054 | 0.4167 | 0.4110 | 0.3187 | 0.5956 | 10.2196 | 0.1581 |
| 5 | `videomae` | 0.4930 | 0.4861 | 0.4895 | 0.5179 | 0.4960 | 9.0007 | 0.4073 |
| 6 | `f3net` | 0.5222 | 0.6528 | 0.5802 | 0.5578 | 0.5936 | 5.9061 | 0.2899 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.7619 | 0.2162 | 0.3368 | 0.1649 | 0.0722 | 13.9474 | 0.6944 |
| `effort_clip_l14` | 0.3333 | 0.1081 | 0.1633 | 0.3299 | 0.2309 | 15.8320 | 1.6667 |
| `f3net` | 0.4938 | 0.5405 | 0.5161 | 0.6598 | 0.5670 | 7.6110 | 0.3261 |
| `i3d` | 0.5122 | 0.8514 | 0.6396 | 0.9072 | 0.8412 | 3.2825 | 0.1182 |
| `videomae` | 0.5300 | 0.7162 | 0.6092 | 0.6186 | 0.6598 | 5.4190 | 0.2326 |
| `xception_df40` | 0.8846 | 0.3108 | 0.4600 | 0.1237 | 0.0433 | 12.4468 | 0.6593 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4966 | 1.0000 | 0.6636 | 1.0757 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5103 | 1.0000 | 0.6758 | 1.0309 | 1.0000 | 0.0000 |
