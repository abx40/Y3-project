# Benchmark Scoring Report V2

## Method

- Evaluation unit: 39-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 39-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0901
- Validation time-to-first-detection: 15.6459s
- Validation F1: 0.1132

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.72 | 0.62 | 0.0000 | 0.0000 | 0.3404 | 11.7554 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.85 | 0.75 | 0.0000 | 0.0000 | 0.4468 | 10.7211 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.52 | 0.42 | 0.0000 | 0.0000 | 0.6596 | 6.0067 |
| `i3d` | `strict_0.10` | `high` | 1 | 0.72 | 0.62 | 0.1000 | 0.0250 | 0.6383 | 7.9234 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.5957 | 7.3603 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.41 | 0.31 | 0.0000 | 0.0000 | 0.2553 | 12.7866 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.3333 | 0.0682 | 0.1132 | 0.0901 | 0.1577 | 15.6459 | 0.7921 |
| 2 | `efficientnet_b4` | 0.4118 | 0.1591 | 0.2295 | 0.1802 | 0.2568 | 14.2612 | 0.7805 |
| 3 | `effort_clip_l14` | 0.4773 | 0.4773 | 0.4773 | 0.2252 | 0.6374 | 8.9373 | 0.1481 |
| 4 | `videomae` | 0.2308 | 0.1364 | 0.1714 | 0.2252 | 0.5495 | 15.1268 | 0.1911 |
| 5 | `i3d` | 0.5800 | 0.6591 | 0.6170 | 0.4054 | 0.5608 | 5.5748 | 0.1639 |
| 6 | `f3net` | 0.5806 | 0.8182 | 0.6792 | 0.4505 | 0.7252 | 3.0754 | 0.2584 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5000 | 0.4872 | 0.4935 | 0.2794 | 0.4770 | 8.5153 | 0.4527 |
| `effort_clip_l14` | 0.5143 | 0.4615 | 0.4865 | 0.1996 | 0.4092 | 7.6763 | 0.2320 |
| `f3net` | 0.5000 | 0.5385 | 0.5185 | 0.3593 | 0.5130 | 7.6517 | 0.4215 |
| `i3d` | 0.2059 | 0.1795 | 0.1918 | 0.2395 | 0.6367 | 13.5585 | 0.2488 |
| `videomae` | 0.5098 | 0.6667 | 0.5778 | 0.3992 | 0.5848 | 5.2041 | 0.1605 |
| `xception_df40` | 0.8889 | 0.2051 | 0.3333 | 0.0399 | 0.0259 | 14.5625 | 1.3675 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5500 | 1.0000 | 0.7097 | 0.4505 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4875 | 1.0000 | 0.6555 | 0.5190 | 1.0000 | 0.0000 |
