# Benchmark Scoring Report V2

## Method

- Evaluation unit: 23-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 23-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.2711
- Validation time-to-first-detection: 8.3765s
- Validation F1: 0.4962

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.5132 | 8.7150 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.12 | 0.02 | 0.0000 | 0.0000 | 0.5789 | 7.8014 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.49 | 0.44 | 0.0000 | 0.0000 | 0.7500 | 4.7710 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.30 | 0.25 | 0.0000 | 0.0000 | 0.7895 | 4.6438 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.49 | 0.39 | 0.1000 | 0.0417 | 0.7632 | 4.6945 |
| `xception_df40` | `strict_0.10` | `high` | 1 | 0.95 | 0.85 | 0.1000 | 0.0383 | 0.1579 | 14.0223 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.4783 | 0.5156 | 0.4962 | 0.2711 | 0.5074 | 8.3765 | 0.2756 |
| 2 | `effort_clip_l14` | 0.5918 | 0.4531 | 0.5133 | 0.3099 | 0.2699 | 9.8422 | 0.5075 |
| 3 | `efficientnet_b4` | 0.4464 | 0.3906 | 0.4167 | 0.3486 | 0.4332 | 10.9971 | 0.5388 |
| 4 | `xception_df40` | 0.5143 | 0.2812 | 0.3636 | 0.3486 | 0.2253 | 12.2963 | 1.7300 |
| 5 | `i3d` | 0.2877 | 0.3281 | 0.3066 | 0.4648 | 0.7179 | 11.8320 | 0.1906 |
| 6 | `f3net` | 0.5000 | 0.6719 | 0.5733 | 0.5423 | 0.5978 | 5.9451 | 0.3168 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5938 | 0.2754 | 0.3762 | 0.2510 | 0.1792 | 13.6261 | 0.4458 |
| `effort_clip_l14` | 0.4762 | 0.4348 | 0.4545 | 0.4184 | 0.4854 | 10.7938 | 0.2637 |
| `f3net` | 0.6667 | 0.5507 | 0.6032 | 0.5021 | 0.2755 | 8.3645 | 0.5288 |
| `i3d` | 0.4957 | 0.8261 | 0.6196 | 0.8787 | 0.8717 | 3.4131 | 0.1181 |
| `videomae` | 0.5340 | 0.7971 | 0.6395 | 0.7113 | 0.7113 | 4.4020 | 0.1855 |
| `xception_df40` | 0.5278 | 0.2754 | 0.3619 | 0.5439 | 0.2434 | 12.4424 | 1.4902 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4741 | 1.0000 | 0.6432 | 0.8909 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5111 | 1.0000 | 0.6765 | 1.0042 | 1.0000 | 0.0000 |
