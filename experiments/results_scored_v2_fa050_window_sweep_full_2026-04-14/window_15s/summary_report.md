# Benchmark Scoring Report V2 FA0.50 Gate 15s

## Method

- Evaluation unit: 15-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 15-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `xception_df40`
- Validation F1: 0.3537
- Validation false alerts/min: 0.4082
- Validation time-to-first-detection: 13.0467s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `high` | 2 | 0.39 | 0.34 | 0.5000 | 0.3000 | 0.3387 | 0.4352 | 10.7457 |
| `effort_clip_l14` | `strict_0.50` | `high` | 2 | 0.97 | 0.97 | 0.3000 | 0.2750 | 0.2742 | 0.3820 | 10.7575 |
| `f3net` | `strict_0.50` | `high` | 3 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.5484 | 0.5913 | 7.5046 |
| `i3d` | `strict_0.50` | `high` | 3 | 0.71 | 0.61 | 0.0000 | 0.0000 | 0.6129 | 0.6129 | 6.8531 |
| `videomae` | `strict_0.50` | `low` | 3 | 0.52 | 0.52 | 0.2000 | 0.2500 | 0.5081 | 0.5676 | 8.5444 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.27 | 0.17 | 0.1000 | 0.1500 | 0.6048 | 0.6198 | 6.7003 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5778 | 0.2549 | 0.3537 | 0.4082 | 0.1939 | 13.0467 | 0.8000 |
| 2 | `efficientnet_b4` | 0.5254 | 0.3039 | 0.3851 | 0.4490 | 0.2857 | 11.1510 | 1.6271 |
| 3 | `videomae` | 0.4881 | 0.4020 | 0.4409 | 0.6531 | 0.4388 | 9.4040 | 0.9524 |
| 4 | `effort_clip_l14` | 0.6269 | 0.4118 | 0.4970 | 0.6939 | 0.2551 | 10.4544 | 1.9104 |
| 5 | `f3net` | 0.6190 | 0.6373 | 0.6280 | 0.9388 | 0.4082 | 6.3103 | 0.4571 |
| 6 | `i3d` | 0.5214 | 0.5980 | 0.5571 | 1.0204 | 0.5714 | 7.2842 | 0.1709 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4677 | 0.2816 | 0.3515 | 0.7010 | 0.3402 | 13.0491 | 1.5484 |
| `effort_clip_l14` | 0.5882 | 0.2913 | 0.3896 | 0.7423 | 0.2165 | 10.3082 | 3.1373 |
| `f3net` | 0.5455 | 0.5243 | 0.5347 | 0.7010 | 0.4639 | 8.1537 | 0.3232 |
| `i3d` | 0.4211 | 0.4660 | 0.4424 | 0.9485 | 0.6804 | 9.8951 | 0.1754 |
| `videomae` | 0.5234 | 0.6505 | 0.5801 | 1.0309 | 0.6289 | 5.7152 | 0.6875 |
| `xception_df40` | 0.6557 | 0.3883 | 0.4878 | 0.4536 | 0.2165 | 10.8656 | 0.5246 |
