# Benchmark Scoring Report V2

## Method

- Evaluation unit: 6-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 6-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0395
- Validation time-to-first-detection: 17.5995s
- Validation F1: 0.0239

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.87 | 0.77 | 0.1000 | 0.0300 | 0.0906 | 15.1499 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.68 | 0.68 | 0.1000 | 0.0100 | 0.0604 | 16.2495 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.71 | 0.61 | 0.1000 | 0.0300 | 0.2718 | 11.0523 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.43 | 0.33 | 0.0000 | 0.0000 | 0.0235 | 16.7632 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.71 | 0.61 | 0.1000 | 0.0300 | 0.4128 | 9.9586 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.69 | 0.64 | 0.1000 | 0.0400 | 0.1074 | 15.2419 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.7500 | 0.0121 | 0.0239 | 0.0395 | 0.0040 | 17.5995 | 5.0000 |
| 2 | `xception_df40` | 0.7667 | 0.0931 | 0.1661 | 0.1186 | 0.0277 | 15.7515 | 4.6667 |
| 3 | `effort_clip_l14` | 0.5000 | 0.0648 | 0.1147 | 0.1186 | 0.0632 | 16.5076 | 3.1250 |
| 4 | `efficientnet_b4` | 0.6923 | 0.1093 | 0.1888 | 0.2767 | 0.0474 | 15.3446 | 5.3846 |
| 5 | `videomae` | 0.6270 | 0.3198 | 0.4236 | 0.7115 | 0.1858 | 11.8114 | 2.5397 |
| 6 | `f3net` | 0.5424 | 0.2591 | 0.3507 | 0.9091 | 0.2134 | 11.4386 | 4.4068 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6667 | 0.2311 | 0.3432 | 0.4819 | 0.1165 | 12.8105 | 3.6782 |
| `effort_clip_l14` | 0.1739 | 0.0159 | 0.0292 | 0.2008 | 0.0763 | 18.0064 | 5.2174 |
| `f3net` | 0.6503 | 0.3705 | 0.4721 | 0.9639 | 0.2008 | 9.8029 | 3.2867 |
| `i3d` | 0.2857 | 0.0080 | 0.0155 | 0.1205 | 0.0201 | 18.1652 | 8.5714 |
| `videomae` | 0.5764 | 0.4661 | 0.5154 | 0.9237 | 0.3454 | 8.7174 | 1.8227 |
| `xception_df40` | 0.6364 | 0.0837 | 0.1479 | 0.1606 | 0.0482 | 16.7501 | 2.1212 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4940 | 1.0000 | 0.6613 | 1.9763 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5020 | 1.0000 | 0.6684 | 2.0884 | 1.0000 | 0.0000 |
