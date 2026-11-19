# Benchmark Scoring Report V2 FA0.50 Gate 30s

## Method

- Evaluation unit: 30-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 30-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `videomae`
- Validation F1: 0.4494
- Validation false alerts/min: 0.2593
- Validation time-to-first-detection: 10.0944s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 2 | 0.66 | 0.66 | 0.2000 | 0.3000 | 0.5789 | 0.6168 | 7.8085 |
| `effort_clip_l14` | `strict_0.50` | `low` | 2 | 0.09 | 0.04 | 0.3000 | 0.3000 | 0.6140 | 0.5833 | 6.2933 |
| `f3net` | `strict_0.50` | `high` | 3 | 0.41 | 0.41 | 0.1000 | 0.3000 | 0.8772 | 0.7143 | 2.7172 |
| `i3d` | `strict_0.50` | `high` | 2 | 0.70 | 0.65 | 0.1000 | 0.1500 | 0.6316 | 0.5714 | 5.8705 |
| `videomae` | `strict_0.50` | `low` | 2 | 0.54 | 0.44 | 0.1000 | 0.0500 | 0.5439 | 0.5586 | 7.7693 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.17 | 0.17 | 0.2000 | 0.3000 | 0.7193 | 0.6308 | 4.6302 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.4651 | 0.4348 | 0.4494 | 0.2593 | 0.4259 | 10.0944 | 0.6512 |
| 2 | `i3d` | 0.4857 | 0.7391 | 0.5862 | 0.2963 | 0.6667 | 3.8985 | 0.2000 |
| 3 | `xception_df40` | 0.6154 | 0.5217 | 0.5647 | 0.2963 | 0.2778 | 10.4099 | 0.4103 |
| 4 | `f3net` | 0.5200 | 0.8478 | 0.6446 | 0.3333 | 0.6667 | 3.7145 | 0.2133 |
| 5 | `effort_clip_l14` | 0.5625 | 0.5870 | 0.5745 | 0.3333 | 0.3889 | 8.6196 | 0.5833 |
| 6 | `efficientnet_b4` | 0.4630 | 0.5435 | 0.5000 | 0.3704 | 0.5370 | 6.7769 | 0.7037 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5769 | 0.6250 | 0.6000 | 0.3077 | 0.4231 | 7.5767 | 0.7308 |
| `effort_clip_l14` | 0.4032 | 0.5208 | 0.4545 | 0.3077 | 0.7115 | 7.6402 | 0.4194 |
| `f3net` | 0.4769 | 0.6458 | 0.5487 | 0.3462 | 0.6538 | 5.4215 | 0.2769 |
| `i3d` | 0.4714 | 0.6875 | 0.5593 | 0.2692 | 0.7115 | 5.4021 | 0.1429 |
| `videomae` | 0.4932 | 0.7500 | 0.5950 | 0.3462 | 0.7115 | 4.9520 | 0.3288 |
| `xception_df40` | 0.4902 | 0.5208 | 0.5051 | 0.2308 | 0.5000 | 8.5924 | 0.3529 |
