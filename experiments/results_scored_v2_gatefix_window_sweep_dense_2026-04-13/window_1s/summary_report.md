# Benchmark Scoring Report V2

## Method

- Evaluation unit: 1-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 1-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0401
- Validation time-to-first-detection: 17.4736s
- Validation F1: 0.0210

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.91 | 0.81 | 0.1000 | 0.0083 | 0.0922 | 14.9326 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.90 | 0.80 | 0.1000 | 0.0300 | 0.0394 | 16.4725 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.81 | 0.71 | 0.1000 | 0.0167 | 0.1799 | 12.4714 |
| `i3d` | `strict_0.10` | `low` | 3 | 0.45 | 0.35 | 0.1000 | 0.0250 | 0.0122 | 16.9672 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.98 | 0.88 | 0.1000 | 0.0067 | 0.1083 | 14.5474 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.98 | 0.88 | 0.1000 | 0.0150 | 0.0061 | 17.0650 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.8000 | 0.0107 | 0.0210 | 0.0401 | 0.0027 | 17.4736 | 18.0000 |
| 2 | `i3d` | 0.5366 | 0.0146 | 0.0285 | 0.1202 | 0.0127 | 17.3923 | 11.7073 |
| 3 | `effort_clip_l14` | 0.4050 | 0.0326 | 0.0604 | 0.1602 | 0.0481 | 16.9604 | 4.9587 |
| 4 | `videomae` | 0.6866 | 0.0992 | 0.1734 | 0.2003 | 0.0454 | 15.2794 | 5.5300 |
| 5 | `efficientnet_b4` | 0.6885 | 0.0839 | 0.1496 | 0.3204 | 0.0381 | 15.5452 | 7.2131 |
| 6 | `f3net` | 0.5518 | 0.2057 | 0.2997 | 0.9212 | 0.1676 | 11.7157 | 7.5000 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6450 | 0.1590 | 0.2551 | 0.3593 | 0.0872 | 14.2150 | 4.5528 |
| `effort_clip_l14` | 0.2634 | 0.0394 | 0.0686 | 0.4790 | 0.1098 | 17.0703 | 7.5000 |
| `f3net` | 0.6589 | 0.2452 | 0.3574 | 0.7984 | 0.1264 | 11.9449 | 5.8169 |
| `i3d` | 0.2105 | 0.0027 | 0.0053 | 0.0798 | 0.0100 | 18.2496 | 12.6316 |
| `videomae` | 0.7617 | 0.1196 | 0.2067 | 0.3194 | 0.0373 | 15.4162 | 4.5957 |
| `xception_df40` | 0.7000 | 0.0094 | 0.0185 | 0.0399 | 0.0040 | 18.1277 | 6.0000 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5007 | 1.0000 | 0.6673 | 2.0427 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4990 | 1.0000 | 0.6658 | 2.0758 | 1.0000 | 0.0000 |
