# Fine-Tuned Benchmark Scoring Report V4 FPR Calibration

## Method

- Evaluation unit: fixed 5-second window.
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
| `efficientnet_b4` | `strict_fpr_gate` | `high` | 0.85 | 0.75 | 3 | 0.0480 | 0.1444 | 11.3511 | 4.2131 | 0.2593 | 0.0482 | 0.5330 |
| `effort_clip_l14` | `strict_fpr_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5037 |
| `f3net` | `strict_fpr_gate` | `high` | 0.87 | 0.87 | 3 | 0.0447 | 0.1569 | 11.0385 | 4.0359 | 0.2394 | 0.0436 | 0.5579 |
| `i3d` | `strict_fpr_gate` | `high` | 0.73 | 0.68 | 3 | 0.0397 | 0.1213 | 12.0292 | 0.8876 | 0.1396 | 0.0399 | 0.5192 |
| `videomae` | `strict_fpr_gate` | `low` | 0.52 | 0.52 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5199 |
| `xception_df40` | `strict_fpr_gate` | `high` | 0.96 | 0.96 | 3 | 0.0497 | 0.1255 | 11.4686 | 5.2243 | 0.2394 | 0.0485 | 0.5367 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_fpr_gate` | 0.5579 | 0.2914 | 0.4324 | 0.4027 | 0.0570 | 11.2070 | 4.1143 |
| 2 | `xception_df40` | `selected` | `strict_fpr_gate` | 0.5367 | 0.2152 | 0.3476 | 0.1208 | 0.0235 | 13.0853 | 4.6667 |
| 3 | `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5330 | 0.2219 | 0.3481 | 0.4430 | 0.0537 | 13.3489 | 3.7590 |
| 4 | `videomae` | `selected` | `strict_fpr_gate` | 0.5199 | 0.0000 | 0.0000 | 0.0403 | 0.0034 | 17.8571 | 24.0000 |
| 5 | `i3d` | `selected` | `strict_fpr_gate` | 0.5192 | 0.0795 | 0.1408 | 0.2416 | 0.0503 | 16.1950 | 3.0769 |
| 6 | `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5458 | 0.1115 | 0.1921 | 0.1627 | 0.0508 | 15.9397 | 3.1837 |
| `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5950 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `f3net` | `selected` | `strict_fpr_gate` | 0.6456 | 0.2754 | 0.4242 | 0.1220 | 0.0237 | 12.3234 | 3.6923 |
| `i3d` | `selected` | `strict_fpr_gate` | 0.5451 | 0.2393 | 0.3641 | 0.2847 | 0.0780 | 13.7718 | 1.2500 |
| `videomae` | `selected` | `strict_fpr_gate` | 0.4394 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `xception_df40` | `selected` | `strict_fpr_gate` | 0.5999 | 0.1869 | 0.3089 | 0.1220 | 0.0237 | 14.2142 | 3.5625 |
