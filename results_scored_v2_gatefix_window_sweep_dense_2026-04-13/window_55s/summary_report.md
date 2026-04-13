# Benchmark Scoring Report V2

## Method

- Evaluation unit: 55-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 55-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
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
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.5161 | 8.5825 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.78 | 0.78 | 0.0000 | 0.0000 | 0.4516 | 9.7133 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.62 | 0.52 | 0.0000 | 0.0000 | 0.4194 | 9.6596 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.5484 | 7.8229 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.56 | 0.51 | 0.0000 | 0.0000 | 0.6452 | 6.5986 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.18 | 0.18 | 0.0000 | 0.0000 | 0.5806 | 7.8401 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 2 | `videomae` | 0.7500 | 0.3214 | 0.4500 | 0.0819 | 0.1092 | 12.3530 | 0.4580 |
| 3 | `efficientnet_b4` | 0.1000 | 0.0357 | 0.0526 | 0.1229 | 0.3345 | 16.9102 | 0.5505 |
| 4 | `f3net` | 0.6364 | 0.5000 | 0.5600 | 0.1638 | 0.2969 | 8.4976 | 0.5975 |
| 5 | `effort_clip_l14` | 0.3929 | 0.3929 | 0.3929 | 0.2048 | 0.6280 | 10.0121 | 0.1579 |
| 6 | `xception_df40` | 0.6296 | 0.6071 | 0.6182 | 0.2457 | 0.3686 | 6.5774 | 0.2041 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5000 | 0.2069 | 0.2927 | 0.1277 | 0.2234 | 13.9006 | 0.3750 |
| `effort_clip_l14` | 0.4643 | 0.4483 | 0.4561 | 0.2979 | 0.5709 | 7.5661 | 0.1980 |
| `f3net` | 0.6471 | 0.3793 | 0.4783 | 0.2128 | 0.2340 | 9.6885 | 0.5806 |
| `i3d` | 0.2000 | 0.0345 | 0.0588 | 0.0426 | 0.1525 | 17.1093 | 0.2222 |
| `videomae` | 0.3462 | 0.3103 | 0.3273 | 0.2128 | 0.6525 | 11.2019 | 0.2968 |
| `xception_df40` | 0.4762 | 0.3448 | 0.4000 | 0.1277 | 0.4220 | 10.8648 | 0.2096 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5091 | 1.0000 | 0.6747 | 0.4096 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5273 | 1.0000 | 0.6905 | 0.4255 | 1.0000 | 0.0000 |
