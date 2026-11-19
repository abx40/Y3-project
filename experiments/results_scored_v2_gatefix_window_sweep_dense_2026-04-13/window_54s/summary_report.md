# Benchmark Scoring Report V2

## Method

- Evaluation unit: 54-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 54-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.0000
- Validation time-to-first-detection: 17.8571s
- Validation F1: 0.0000

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.36 | 0.26 | 0.1000 | 0.0100 | 0.5000 | 8.4453 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.86 | 0.86 | 0.0000 | 0.0000 | 0.3333 | 12.7500 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.6389 | 7.1672 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.5833 | 7.7460 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.49 | 0.49 | 0.0000 | 0.0000 | 0.6111 | 7.3111 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.16 | 0.16 | 0.1000 | 0.0100 | 0.6944 | 6.0072 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 2 | `efficientnet_b4` | 0.5909 | 0.4194 | 0.4906 | 0.0844 | 0.3080 | 10.3430 | 0.3297 |
| 3 | `effort_clip_l14` | 0.4286 | 0.2903 | 0.3462 | 0.0844 | 0.4219 | 12.8648 | 0.3030 |
| 4 | `videomae` | 0.6471 | 0.3548 | 0.4583 | 0.1688 | 0.1603 | 11.0584 | 0.3876 |
| 5 | `xception_df40` | 0.6957 | 0.5161 | 0.5926 | 0.2110 | 0.2321 | 7.9929 | 0.2094 |
| 6 | `f3net` | 0.5143 | 0.5806 | 0.5455 | 0.2110 | 0.6118 | 8.1231 | 0.3093 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5333 | 0.2222 | 0.3137 | 0.1630 | 0.2120 | 14.0941 | 0.3883 |
| `effort_clip_l14` | 0.7778 | 0.5833 | 0.6667 | 0.2174 | 0.1630 | 8.1838 | 0.1896 |
| `f3net` | 0.7895 | 0.4167 | 0.5455 | 0.1630 | 0.1087 | 11.1690 | 0.3401 |
| `i3d` | 0.1875 | 0.0833 | 0.1154 | 0.1630 | 0.5489 | 17.0362 | 0.1562 |
| `videomae` | 0.6774 | 0.5833 | 0.6269 | 0.2174 | 0.4022 | 7.6837 | 0.1961 |
| `xception_df40` | 0.5556 | 0.4167 | 0.4762 | 0.2174 | 0.4565 | 9.6843 | 0.1896 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5167 | 1.0000 | 0.6813 | 0.4219 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.6000 | 1.0000 | 0.7500 | 0.4891 | 1.0000 | 0.0000 |
