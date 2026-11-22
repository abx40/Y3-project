# Benchmark Scoring Report V2 No Gate Max F1 5s

## Method

- Evaluation unit: 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 5-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- No false-alert gate is applied during calibration operating-point selection.
- Calibration selection: No calibration safety gate. Select the operating point that maximizes F1 on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then prefer lower calibration real-only false alerts.
- Validation ranking: F1 desc, time_to_first_detection_s asc

## Winner

- Validation winner: `effort_clip_l14`
- Validation F1: 0.6727
- Validation false alerts/min: 2.0134
- Validation time-to-first-detection: 0.1786s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `nogate_maxf1` | `low` | 1 | 0.17 | 0.17 | 0.1000 | 1.0000 | 0.9972 | 0.7503 | 0.0000 |
| `effort_clip_l14` | `nogate_maxf1` | `low` | 1 | 0.05 | 0.00 | 0.1000 | 1.0000 | 0.9778 | 0.7434 | 0.3798 |
| `f3net` | `nogate_maxf1` | `high` | 3 | 0.20 | 0.20 | 0.2000 | 0.9583 | 0.9611 | 0.7530 | 0.4613 |
| `i3d` | `nogate_maxf1` | `low` | 1 | 0.01 | 0.01 | 0.1000 | 1.0000 | 0.9972 | 0.7487 | 0.0000 |
| `videomae` | `nogate_maxf1` | `low` | 1 | 0.12 | 0.07 | 0.3000 | 0.9833 | 0.9889 | 0.7550 | 0.1498 |
| `xception_df40` | `nogate_maxf1` | `low` | 2 | 0.05 | 0.00 | 0.1000 | 0.9917 | 0.9694 | 0.7433 | 0.5308 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.5094 | 0.9901 | 0.6727 | 2.0134 | 0.9664 | 0.1786 | 0.0818 |
| 2 | `i3d` | 0.5042 | 0.9868 | 0.6674 | 1.9732 | 0.9832 | 0.1667 | 0.0406 |
| 3 | `xception_df40` | 0.5042 | 0.9834 | 0.6667 | 1.9732 | 0.9799 | 0.2976 | 0.1019 |
| 4 | `efficientnet_b4` | 0.5017 | 0.9735 | 0.6622 | 2.0134 | 0.9799 | 0.3525 | 0.2457 |
| 5 | `f3net` | 0.5036 | 0.9238 | 0.6519 | 2.0537 | 0.9228 | 0.7257 | 0.4765 |
| 6 | `videomae` | 0.4930 | 0.9338 | 0.6453 | 2.0134 | 0.9732 | 0.7275 | 0.5035 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5084 | 0.9934 | 0.6726 | 2.1153 | 0.9932 | 0.0408 | 0.0805 |
| `effort_clip_l14` | 0.5042 | 0.9738 | 0.6644 | 2.0746 | 0.9898 | 0.4797 | 0.0611 |
| `f3net` | 0.5089 | 0.9410 | 0.6605 | 2.1153 | 0.9390 | 1.0071 | 0.3191 |
| `i3d` | 0.5083 | 1.0000 | 0.6740 | 2.1153 | 1.0000 | 0.0000 | 0.0000 |
| `videomae` | 0.5140 | 0.9607 | 0.6697 | 2.2373 | 0.9390 | 0.2345 | 0.6316 |
| `xception_df40` | 0.5077 | 0.9738 | 0.6674 | 2.0746 | 0.9763 | 0.4724 | 0.1026 |
