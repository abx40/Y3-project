# Benchmark Scoring Report V2

## Method

- Evaluation unit: 60-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 60-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.1250
- Validation time-to-first-detection: 12.2436s
- Validation F1: 0.3913

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 2 | 0.66 | 0.56 | 0.0000 | 0.0000 | 0.5714 | 6.7745 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.90 | 0.80 | 0.0000 | 0.0000 | 0.3571 | 11.1022 |
| `f3net` | `strict_0.10` | `high` | 1 | 0.58 | 0.48 | 0.0000 | 0.0000 | 0.5000 | 7.5023 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6429 | 7.3504 |
| `videomae` | `strict_0.10` | `high` | 2 | 0.47 | 0.37 | 0.0000 | 0.0000 | 0.3214 | 10.5424 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.17 | 0.17 | 0.0000 | 0.0000 | 0.7143 | 5.3163 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.4500 | 0.3462 | 0.3913 | 0.1250 | 0.4583 | 12.2436 | 0.3500 |
| 2 | `i3d` | 0.0714 | 0.0385 | 0.0500 | 0.1250 | 0.5417 | 16.0455 | 0.1429 |
| 3 | `f3net` | 0.7037 | 0.7308 | 0.7170 | 0.2083 | 0.3333 | 5.8533 | 0.4074 |
| 4 | `videomae` | 0.5833 | 0.5385 | 0.5600 | 0.2083 | 0.4167 | 8.7301 | 0.2500 |
| 5 | `efficientnet_b4` | 0.5556 | 0.5769 | 0.5660 | 0.2500 | 0.5000 | 7.0598 | 0.2593 |
| 6 | `xception_df40` | 0.5652 | 0.5000 | 0.5306 | 0.2500 | 0.4167 | 8.2095 | 0.2609 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5385 | 0.8400 | 0.6562 | 0.3600 | 0.7200 | 3.6069 | 0.1538 |
| `effort_clip_l14` | 0.6190 | 0.5200 | 0.5652 | 0.1600 | 0.3200 | 7.5644 | 0.2857 |
| `f3net` | 0.7826 | 0.7200 | 0.7500 | 0.2000 | 0.2000 | 7.0410 | 0.5652 |
| `i3d` | 0.5200 | 0.5200 | 0.5200 | 0.1600 | 0.4800 | 9.2321 | 0.1200 |
| `videomae` | 0.3571 | 0.2000 | 0.2564 | 0.2000 | 0.3600 | 13.5191 | 0.4286 |
| `xception_df40` | 0.4783 | 0.4400 | 0.4583 | 0.1600 | 0.4800 | 10.0197 | 0.1304 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5200 | 1.0000 | 0.6842 | 0.4167 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5000 | 1.0000 | 0.6667 | 0.4400 | 1.0000 | 0.0000 |
