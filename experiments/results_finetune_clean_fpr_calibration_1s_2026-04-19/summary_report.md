# Fine-Tuned Benchmark Scoring Report V4 FPR Calibration

## Method

- Evaluation unit: fixed 1-second window.
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
| `efficientnet_b4` | `strict_fpr_gate` | `high` | 0.95 | 0.95 | 3 | 0.0432 | 0.0825 | 11.8833 | 11.8367 | 0.3791 | 0.0432 | 0.5234 |
| `effort_clip_l14` | `strict_fpr_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5027 |
| `f3net` | `strict_fpr_gate` | `high` | 0.97 | 0.97 | 3 | 0.0486 | 0.0808 | 11.6177 | 11.8563 | 0.3791 | 0.0486 | 0.5515 |
| `i3d` | `strict_fpr_gate` | `high` | 0.74 | 0.74 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5057 |
| `videomae` | `strict_fpr_gate` | `low` | 0.52 | 0.52 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5245 |
| `xception_df40` | `relaxed_fpr_gate` | `high` | 0.99 | 0.99 | 1 | 0.0858 | 0.1302 | 9.8282 | 35.1592 | 1.6561 | 0.0858 | 0.5359 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_fpr_gate` | 0.5515 | 0.1944 | 0.3046 | 0.6809 | 0.0821 | 11.6909 | 8.9639 |
| 2 | `xception_df40` | `selected` | `relaxed_fpr_gate` | 0.5359 | 0.1485 | 0.2371 | 2.0828 | 0.1041 | 11.4983 | 30.5541 |
| 3 | `videomae` | `selected` | `strict_fpr_gate` | 0.5245 | 0.0000 | 0.0000 | 0.0401 | 0.0067 | 17.8571 | 12.0000 |
| 4 | `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5234 | 0.1152 | 0.1951 | 0.5207 | 0.0654 | 14.0527 | 9.0775 |
| 5 | `i3d` | `selected` | `strict_fpr_gate` | 0.5057 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 6 | `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5027 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5345 | 0.0448 | 0.0843 | 0.1198 | 0.0173 | 16.6544 | 7.7419 |
| `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5765 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `f3net` | `selected` | `strict_fpr_gate` | 0.6155 | 0.1169 | 0.1990 | 0.4790 | 0.0579 | 13.8785 | 10.5344 |
| `i3d` | `selected` | `strict_fpr_gate` | 0.5410 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `videomae` | `selected` | `strict_fpr_gate` | 0.4415 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `xception_df40` | `selected` | `relaxed_fpr_gate` | 0.5884 | 0.1663 | 0.2692 | 1.5968 | 0.0692 | 12.2648 | 27.5354 |
