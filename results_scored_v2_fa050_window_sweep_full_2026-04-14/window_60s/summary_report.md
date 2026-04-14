# Benchmark Scoring Report V2 FA0.50 Gate 60s

## Method

- Evaluation unit: 60-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 60-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `i3d`
- Validation F1: 0.0500
- Validation false alerts/min: 0.1250
- Validation time-to-first-detection: 16.0455s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 2 | 0.66 | 0.56 | 0.0000 | 0.0000 | 0.5714 | 0.5424 | 6.7745 |
| `effort_clip_l14` | `strict_0.50` | `high` | 1 | 0.88 | 0.88 | 0.2000 | 0.3000 | 0.4643 | 0.5200 | 8.6376 |
| `f3net` | `strict_0.50` | `high` | 1 | 0.49 | 0.44 | 0.2000 | 0.3000 | 0.6786 | 0.6230 | 4.7747 |
| `i3d` | `strict_0.50` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.6429 | 0.6667 | 7.3504 |
| `videomae` | `strict_0.50` | `high` | 1 | 0.47 | 0.47 | 0.3000 | 0.3000 | 0.4643 | 0.5098 | 9.4303 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.17 | 0.17 | 0.0000 | 0.0000 | 0.7143 | 0.6349 | 5.3163 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.0714 | 0.0385 | 0.0500 | 0.1250 | 0.5417 | 16.0455 | 0.1429 |
| 2 | `efficientnet_b4` | 0.5556 | 0.5769 | 0.5660 | 0.2500 | 0.5000 | 7.0598 | 0.2593 |
| 3 | `xception_df40` | 0.5652 | 0.5000 | 0.5306 | 0.2500 | 0.4167 | 8.2095 | 0.2609 |
| 4 | `effort_clip_l14` | 0.5455 | 0.4615 | 0.5000 | 0.2917 | 0.4167 | 9.5165 | 0.9545 |
| 5 | `f3net` | 0.6053 | 0.8846 | 0.7188 | 0.3333 | 0.6250 | 1.5529 | 0.3158 |
| 6 | `videomae` | 0.6296 | 0.6538 | 0.6415 | 0.3750 | 0.4167 | 6.8627 | 0.9259 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5385 | 0.8400 | 0.6562 | 0.3600 | 0.7200 | 3.6069 | 0.1538 |
| `effort_clip_l14` | 0.5500 | 0.4400 | 0.4889 | 0.2800 | 0.3600 | 7.8147 | 0.7500 |
| `f3net` | 0.5938 | 0.7600 | 0.6667 | 0.4000 | 0.5200 | 4.9202 | 0.5625 |
| `i3d` | 0.5200 | 0.5200 | 0.5200 | 0.1600 | 0.4800 | 9.2321 | 0.1200 |
| `videomae` | 0.4737 | 0.3600 | 0.4091 | 0.3600 | 0.4000 | 11.4897 | 1.0000 |
| `xception_df40` | 0.4783 | 0.4400 | 0.4583 | 0.1600 | 0.4800 | 10.0197 | 0.1304 |
