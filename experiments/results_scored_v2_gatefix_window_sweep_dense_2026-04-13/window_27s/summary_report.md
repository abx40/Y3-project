# Benchmark Scoring Report V2

## Method

- Evaluation unit: 27-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 27-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.0771
- Validation time-to-first-detection: 15.5726s
- Validation F1: 0.2154

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.75 | 0.65 | 0.0000 | 0.0000 | 0.3906 | 10.7088 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.99 | 0.94 | 0.0000 | 0.0000 | 0.0781 | 15.8961 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.6094 | 5.4678 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6406 | 5.8408 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.55 | 0.45 | 0.1000 | 0.0100 | 0.5312 | 8.2239 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.34 | 0.24 | 0.0000 | 0.0000 | 0.2812 | 12.3335 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.7000 | 0.1273 | 0.2154 | 0.0771 | 0.0520 | 15.5726 | 1.3333 |
| 2 | `videomae` | 0.5208 | 0.4545 | 0.4854 | 0.1156 | 0.3854 | 10.5135 | 0.2871 |
| 3 | `i3d` | 0.3810 | 0.4364 | 0.4068 | 0.1927 | 0.6358 | 10.2904 | 0.1880 |
| 4 | `efficientnet_b4` | 0.4643 | 0.2364 | 0.3133 | 0.1927 | 0.2331 | 14.1067 | 0.6723 |
| 5 | `xception_df40` | 0.2857 | 0.0727 | 0.1159 | 0.1927 | 0.1464 | 16.7148 | 1.0714 |
| 6 | `f3net` | 0.5938 | 0.6909 | 0.6387 | 0.3468 | 0.4509 | 4.9638 | 0.4982 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4038 | 0.4375 | 0.4200 | 0.3826 | 0.4852 | 9.8316 | 0.5983 |
| `effort_clip_l14` | 0.2857 | 0.0417 | 0.0727 | 0.0696 | 0.0661 | 16.9752 | 1.0714 |
| `f3net` | 0.5636 | 0.6458 | 0.6019 | 0.2435 | 0.3513 | 6.7786 | 0.5485 |
| `i3d` | 0.3960 | 0.8333 | 0.5369 | 0.5217 | 0.9061 | 2.9501 | 0.1144 |
| `videomae` | 0.3043 | 0.4375 | 0.3590 | 0.3130 | 0.7148 | 8.9362 | 0.2698 |
| `xception_df40` | 0.6875 | 0.2292 | 0.3438 | 0.1043 | 0.0783 | 13.8962 | 0.8333 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4783 | 1.0000 | 0.6471 | 0.5780 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4174 | 1.0000 | 0.5890 | 0.5217 | 1.0000 | 0.0000 |
