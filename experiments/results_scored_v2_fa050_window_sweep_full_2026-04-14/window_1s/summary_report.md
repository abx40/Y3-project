# Benchmark Scoring Report V2 FA0.50 Gate 1s

## Method

- Evaluation unit: 1-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 1-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `xception_df40`
- Validation F1: 0.1696
- Validation false alerts/min: 0.6409
- Validation time-to-first-detection: 14.6218s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 3 | 0.86 | 0.76 | 0.5000 | 0.1833 | 0.1710 | 0.2688 | 13.0440 |
| `effort_clip_l14` | `strict_0.50` | `low` | 3 | 0.51 | 0.41 | 0.5000 | 0.1000 | 0.1682 | 0.2635 | 12.8165 |
| `f3net` | `strict_0.50` | `high` | 3 | 0.71 | 0.61 | 0.5000 | 0.1117 | 0.3276 | 0.4225 | 9.8440 |
| `i3d` | `strict_0.50` | `low` | 3 | 0.37 | 0.27 | 0.5000 | 0.2983 | 0.3937 | 0.4968 | 9.5719 |
| `videomae` | `strict_0.50` | `low` | 3 | 0.91 | 0.81 | 0.4000 | 0.0800 | 0.1738 | 0.2718 | 12.2622 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.77 | 0.72 | 0.5000 | 0.1017 | 0.1122 | 0.1860 | 13.6890 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.5412 | 0.1005 | 0.1696 | 0.6409 | 0.0854 | 14.6218 | 8.1720 |
| 2 | `effort_clip_l14` | 0.5293 | 0.1864 | 0.2757 | 0.7610 | 0.1662 | 12.8027 | 6.5785 |
| 3 | `videomae` | 0.5596 | 0.1751 | 0.2667 | 0.8411 | 0.1382 | 12.9536 | 7.4043 |
| 4 | `efficientnet_b4` | 0.5817 | 0.2204 | 0.3197 | 0.9212 | 0.1589 | 12.2595 | 5.4833 |
| 5 | `f3net` | 0.5248 | 0.2956 | 0.3782 | 1.2417 | 0.2684 | 9.8561 | 5.8156 |
| 6 | `i3d` | 0.4668 | 0.3322 | 0.3882 | 1.2417 | 0.3805 | 11.3387 | 1.5716 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6203 | 0.3100 | 0.4134 | 0.9581 | 0.1890 | 10.6379 | 4.4920 |
| `effort_clip_l14` | 0.3634 | 0.1075 | 0.1660 | 0.7984 | 0.1876 | 14.6474 | 7.0429 |
| `f3net` | 0.5791 | 0.3373 | 0.4263 | 1.2774 | 0.2442 | 9.6861 | 4.8853 |
| `i3d` | 0.5261 | 0.4716 | 0.4974 | 1.3174 | 0.4232 | 8.7104 | 1.5201 |
| `videomae` | 0.6375 | 0.2725 | 0.3818 | 0.8383 | 0.1544 | 11.5678 | 5.0625 |
| `xception_df40` | 0.4606 | 0.0782 | 0.1336 | 0.5589 | 0.0912 | 15.3874 | 9.6850 |
