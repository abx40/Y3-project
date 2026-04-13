# Benchmark Scoring Report V2

## Method

- Evaluation unit: 31-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 31-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0805
- Validation time-to-first-detection: 15.6226s
- Validation F1: 0.2439

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.45 | 0.35 | 0.0000 | 0.0000 | 0.1667 | 14.5125 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.97 | 0.87 | 0.0000 | 0.0000 | 0.3500 | 11.1041 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.5667 | 7.3622 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.5833 | 6.9522 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.62 | 0.52 | 0.0000 | 0.0000 | 0.4667 | 8.9559 |
| `xception_df40` | `strict_0.10` | `low` | 1 | 0.60 | 0.50 | 0.0000 | 0.0000 | 0.1667 | 14.2999 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.3125 | 0.2000 | 0.2439 | 0.0805 | 0.4309 | 15.6226 | 0.2632 |
| 2 | `effort_clip_l14` | 0.6875 | 0.4400 | 0.5366 | 0.1208 | 0.2081 | 10.9594 | 0.5556 |
| 3 | `efficientnet_b4` | 0.6250 | 0.2000 | 0.3030 | 0.1208 | 0.1114 | 15.1034 | 1.3158 |
| 4 | `videomae` | 0.4783 | 0.2200 | 0.3014 | 0.1611 | 0.2362 | 13.6922 | 0.9524 |
| 5 | `xception_df40` | 0.2500 | 0.0400 | 0.0690 | 0.2013 | 0.0980 | 16.2487 | 2.8846 |
| 6 | `f3net` | 0.6296 | 0.6800 | 0.6538 | 0.2819 | 0.4161 | 6.0574 | 0.3672 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6923 | 0.1667 | 0.2687 | 0.1337 | 0.0921 | 15.5789 | 0.7833 |
| `effort_clip_l14` | 0.8000 | 0.3704 | 0.5063 | 0.1783 | 0.1003 | 11.4881 | 0.7152 |
| `f3net` | 0.6977 | 0.5556 | 0.6186 | 0.3120 | 0.2548 | 8.4281 | 0.4789 |
| `i3d` | 0.4667 | 0.5185 | 0.4912 | 0.3120 | 0.6924 | 8.6260 | 0.1348 |
| `videomae` | 0.5111 | 0.4259 | 0.4646 | 0.3566 | 0.4770 | 9.7074 | 0.5843 |
| `xception_df40` | 0.4444 | 0.0741 | 0.1270 | 0.2229 | 0.1003 | 15.9062 | 2.0849 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5000 | 1.0000 | 0.6667 | 0.4027 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5400 | 1.0000 | 0.7013 | 0.6241 | 1.0000 | 0.0000 |
