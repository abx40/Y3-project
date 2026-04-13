# Benchmark Scoring Report V2

## Method

- Evaluation unit: 51-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 51-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0451
- Validation time-to-first-detection: 16.6007s
- Validation F1: 0.1622

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.70 | 0.60 | 0.0000 | 0.0000 | 0.3235 | 10.5337 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.91 | 0.86 | 0.0000 | 0.0000 | 0.3235 | 11.4846 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.6765 | 5.1452 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6471 | 6.6296 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.57 | 0.47 | 0.0000 | 0.0000 | 0.5588 | 8.1178 |
| `xception_df40` | `strict_0.10` | `low` | 1 | 0.55 | 0.55 | 0.0000 | 0.0000 | 0.1471 | 14.5412 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.7500 | 0.0909 | 0.1622 | 0.0451 | 0.0293 | 16.6007 | 1.5625 |
| 2 | `videomae` | 0.6500 | 0.3939 | 0.4906 | 0.1354 | 0.2596 | 10.0657 | 0.3614 |
| 3 | `effort_clip_l14` | 0.7778 | 0.4242 | 0.5490 | 0.1806 | 0.1445 | 10.8017 | 0.4636 |
| 4 | `i3d` | 0.2500 | 0.1515 | 0.1887 | 0.1806 | 0.5485 | 15.2964 | 0.1829 |
| 5 | `efficientnet_b4` | 0.4667 | 0.2121 | 0.2917 | 0.2257 | 0.2889 | 14.6787 | 0.4858 |
| 6 | `f3net` | 0.5758 | 0.5758 | 0.5758 | 0.3160 | 0.5102 | 8.4258 | 0.3670 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6389 | 0.7419 | 0.6866 | 0.2516 | 0.4549 | 5.0627 | 0.3311 |
| `effort_clip_l14` | 0.6190 | 0.4194 | 0.5000 | 0.2516 | 0.2683 | 10.4038 | 0.3438 |
| `f3net` | 0.5152 | 0.5484 | 0.5312 | 0.3354 | 0.5367 | 8.2736 | 0.2588 |
| `i3d` | 0.3871 | 0.3871 | 0.3871 | 0.2516 | 0.6520 | 12.2778 | 0.1165 |
| `videomae` | 0.4722 | 0.5484 | 0.5075 | 0.2516 | 0.6520 | 7.8776 | 0.2013 |
| `xception_df40` | 0.6667 | 0.0645 | 0.1176 | 0.0419 | 0.0273 | 17.2583 | 2.1277 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5500 | 1.0000 | 0.7097 | 0.5418 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5167 | 1.0000 | 0.6813 | 0.4612 | 1.0000 | 0.0000 |
