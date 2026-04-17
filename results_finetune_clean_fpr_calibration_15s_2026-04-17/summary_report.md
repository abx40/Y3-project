# Fine-Tuned Benchmark Scoring Report V4 FPR Calibration

## Method

- Evaluation unit: fixed 15-second window.
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
| `efficientnet_b4` | `strict_fpr_gate` | `high` | 0.75 | 0.65 | 3 | 0.0439 | 0.1911 | 10.9928 | 0.9262 | 0.1577 | 0.0444 | 0.5569 |
| `effort_clip_l14` | `strict_fpr_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.0000 | 0.0000 | 0.5329 |
| `f3net` | `strict_fpr_gate` | `high` | 0.78 | 0.68 | 2 | 0.0439 | 0.3121 | 9.3619 | 1.5457 | 0.1775 | 0.0398 | 0.5985 |
| `i3d` | `strict_fpr_gate` | `low` | 0.35 | 0.35 | 2 | 0.0439 | 0.0382 | 13.0556 | 3.5762 | 0.1183 | 0.0384 | 0.4742 |
| `videomae` | `strict_fpr_gate` | `low` | 0.51 | 0.46 | 3 | 0.0293 | 0.1465 | 11.7560 | 0.4296 | 0.0592 | 0.0250 | 0.5000 |
| `xception_df40` | `strict_fpr_gate` | `high` | 0.83 | 0.78 | 3 | 0.0488 | 0.3758 | 8.4496 | 0.7287 | 0.1380 | 0.0493 | 0.5613 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_fpr_gate` | 0.5985 | 0.3627 | 0.5000 | 0.3265 | 0.0918 | 10.9645 | 1.3913 |
| 2 | `xception_df40` | `selected` | `strict_fpr_gate` | 0.5613 | 0.4412 | 0.5488 | 0.5306 | 0.1735 | 9.5684 | 0.5806 |
| 3 | `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5569 | 0.1078 | 0.1947 | 0.0000 | 0.0000 | 15.8876 | 1.0909 |
| 4 | `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.5329 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | 0.0000 |
| 5 | `videomae` | `selected` | `strict_fpr_gate` | 0.5000 | 0.5490 | 0.6512 | 0.4082 | 0.1429 | 8.3036 | 0.1143 |
| 6 | `i3d` | `selected` | `strict_fpr_gate` | 0.4742 | 0.0686 | 0.1148 | 0.3265 | 0.1327 | 16.6821 | 3.2000 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | alert-time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_fpr_gate` | 0.5808 | 0.1165 | 0.2017 | 0.0825 | 0.0412 | 15.8132 | 1.5000 |
| `effort_clip_l14` | `selected` | `strict_fpr_gate` | 0.6278 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | 0.0000 |
| `f3net` | `selected` | `strict_fpr_gate` | 0.6968 | 0.2621 | 0.3885 | 0.2474 | 0.0928 | 13.1348 | 2.1111 |
| `i3d` | `selected` | `strict_fpr_gate` | 0.4654 | 0.0874 | 0.1525 | 0.2062 | 0.0619 | 16.3560 | 3.2000 |
| `videomae` | `selected` | `strict_fpr_gate` | 0.4201 | 0.0291 | 0.0566 | 0.0000 | 0.0000 | 17.7439 | 1.3333 |
| `xception_df40` | `selected` | `strict_fpr_gate` | 0.6337 | 0.2718 | 0.4088 | 0.1649 | 0.0619 | 12.8228 | 0.8235 |
