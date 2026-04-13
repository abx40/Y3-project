# Benchmark Scoring Report V2

## Method

- Evaluation unit: 24-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 24-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.2119
- Validation time-to-first-detection: 13.9415s
- Validation F1: 0.2824

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.35 | 0.25 | 0.0000 | 0.0000 | 0.5316 | 8.0538 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.97 | 0.92 | 0.0000 | 0.0000 | 0.1646 | 14.7589 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.46 | 0.36 | 0.0000 | 0.0000 | 0.8228 | 3.6514 |
| `i3d` | `strict_0.10` | `high` | 2 | 0.72 | 0.62 | 0.0000 | 0.0000 | 0.4430 | 9.9894 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.62 | 0.52 | 0.0000 | 0.0000 | 0.4810 | 8.7459 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.38 | 0.28 | 0.0000 | 0.0000 | 0.3038 | 11.9780 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.6316 | 0.1818 | 0.2824 | 0.2119 | 0.1186 | 13.9415 | 1.3158 |
| 2 | `effort_clip_l14` | 0.6667 | 0.3636 | 0.4706 | 0.2542 | 0.2034 | 11.4082 | 0.5556 |
| 3 | `i3d` | 0.6190 | 0.3939 | 0.4815 | 0.3390 | 0.2712 | 10.5230 | 0.2381 |
| 4 | `videomae` | 0.4444 | 0.3030 | 0.3604 | 0.4237 | 0.4237 | 12.0350 | 1.0556 |
| 5 | `efficientnet_b4` | 0.4032 | 0.3788 | 0.3906 | 0.5508 | 0.6271 | 11.6505 | 0.2823 |
| 6 | `f3net` | 0.5581 | 0.7273 | 0.6316 | 0.5932 | 0.6441 | 5.3125 | 0.2326 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5897 | 0.3898 | 0.4694 | 0.2652 | 0.2424 | 11.8468 | 0.2564 |
| `effort_clip_l14` | 0.0000 | 0.0000 | 0.0000 | 0.0379 | 0.0455 | 17.8637 | 1.6667 |
| `f3net` | 0.5077 | 0.5593 | 0.5323 | 0.4545 | 0.4848 | 8.2769 | 0.2692 |
| `i3d` | 0.1304 | 0.0508 | 0.0732 | 0.0758 | 0.3030 | 17.3107 | 0.3261 |
| `videomae` | 0.4717 | 0.4237 | 0.4464 | 0.4545 | 0.4242 | 10.4328 | 0.8019 |
| `xception_df40` | 0.8333 | 0.2542 | 0.3896 | 0.1136 | 0.0455 | 13.5675 | 1.1111 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5280 | 1.0000 | 0.6911 | 0.8898 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4720 | 1.0000 | 0.6413 | 0.7955 | 1.0000 | 0.0000 |
