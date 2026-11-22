# Fine-Tuned Benchmark Scoring Report V4 FPR Calibration

## Method

- Evaluation unit: fixed 10-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.
- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.
- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.
- Calibration is FPR-based: the primary calibration constraint is real-window FPR on pooled real-labelled calibration windows after post-processing.
- Strict FPR target: real-window FPR <= 0.05.
- Relaxed fallback FPR target: real-window FPR <= 0.10.
- Candidate selection among FPR-compliant settings: maximize recall on fake-labelled calibration windows, then lower TTFD, then lower flicker.
- FA/min and alert-time ratio are still reported as operational metrics, but they are not the main selection rule.
- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.

## Calibration Outcome

- Gate-compliant backends: 6 / 6

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | calib FPR_real | calib recall | calib TTFD s | calib flicker | calib FA/min | calib alert-time ratio | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_fpr_gate` | `high` | 0.77 | 0.77 | 3 | 0.0471 | 0.1510 | 11.2777 | 2.2441 | 0.2239 | 0.0475 | 0.5351 |
| `effort_clip_l14` | `strict_fpr_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5089 |
| `f3net` | `strict_fpr_gate` | `high` | 0.82 | 0.77 | 2 | 0.0471 | 0.2408 | 9.8486 | 3.1669 | 0.2239 | 0.0444 | 0.5768 |
| `i3d` | `strict_fpr_gate` | `high` | 0.73 | 0.63 | 2 | 0.0438 | 0.1347 | 11.8475 | 0.2733 | 0.2035 | 0.0410 | 0.5081 |
| `videomae` | `strict_fpr_gate` | `low` | 0.51 | 0.51 | 3 | 0.0370 | 0.0408 | 13.1595 | 2.8000 | 0.1221 | 0.0373 | 0.5313 |
| `xception_df40` | `strict_fpr_gate` | `high` | 0.95 | 0.95 | 2 | 0.0404 | 0.1265 | 11.4046 | 4.0431 | 0.1628 | 0.0377 | 0.5537 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_fpr_gate` | 0.5768 | 0.3355 | 0.4789 | 0.2027 | 0.0676 | 11.0304 | 2.9508 |
| 2 | `xception_df40` | `selected` | `strict_fpr_gate` | 0.5537 | 0.1579 | 0.2682 | 0.0811 | 0.0203 | 14.5244 | 4.2222 |
| 3 | `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5351 | 0.1711 | 0.2842 | 0.1622 | 0.0338 | 14.3142 | 1.7419 |
| 4 | `videomae` | `selected` | `strict_fpr_gate` | 0.5313 | 0.0724 | 0.1310 | 0.1216 | 0.0338 | 16.2924 | 3.3750 |
| 5 | `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5089 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 6 | `i3d` | `selected` | `strict_fpr_gate` | 0.5081 | 0.1316 | 0.2041 | 0.4459 | 0.1622 | 15.3842 | 0.1364 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5622 | 0.1126 | 0.1910 | 0.1208 | 0.0671 | 15.9808 | 2.2222 |
| `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5714 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `f3net` | `selected` | `strict_fpr_gate` | 0.6587 | 0.2649 | 0.4061 | 0.1611 | 0.0403 | 12.7588 | 3.3913 |
| `i3d` | `selected` | `strict_fpr_gate` | 0.5505 | 0.2914 | 0.4190 | 0.4027 | 0.1007 | 13.0870 | 0.2034 |
| `videomae` | `selected` | `strict_fpr_gate` | 0.4375 | 0.0530 | 0.0958 | 0.1611 | 0.0537 | 17.1027 | 3.0000 |
| `xception_df40` | `selected` | `strict_fpr_gate` | 0.6094 | 0.1656 | 0.2793 | 0.0805 | 0.0201 | 14.6458 | 3.0000 |
