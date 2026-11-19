# Fine-Tuned FPR Calibration Summary

## Method

- Calibration is FPR-based at a fixed 5s window.
- Candidate selection maximizes recall on fake-labelled calibration windows subject to real-window FPR on pooled real-labelled calibration windows.
- Strict FPR target: 0.05. Relaxed fallback FPR target: 0.10.
- FA/min is reported as an operational metric, not the main selection rule.
- Validation and test remain fully held out from threshold selection.

## Validation

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | selected | strict_fpr_gate | 0.5579 | 0.4324 | 0.4027 | 0.2914 |
| xception_df40 | selected | strict_fpr_gate | 0.5367 | 0.3476 | 0.1208 | 0.2152 |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5330 | 0.3481 | 0.4430 | 0.2219 |
| videomae | selected | strict_fpr_gate | 0.5199 | 0.0000 | 0.0403 | 0.0000 |
| i3d | selected | strict_fpr_gate | 0.5192 | 0.1408 | 0.2416 | 0.0795 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5037 | 0.0000 | 0.0000 | 0.0000 |

## Test

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5458 | 0.1921 | 0.1627 | 0.1115 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5950 | 0.0000 | 0.0000 | 0.0000 |
| f3net | selected | strict_fpr_gate | 0.6456 | 0.4242 | 0.1220 | 0.2754 |
| i3d | selected | strict_fpr_gate | 0.5451 | 0.3641 | 0.2847 | 0.2393 |
| videomae | selected | strict_fpr_gate | 0.4394 | 0.0000 | 0.0000 | 0.0000 |
| xception_df40 | selected | strict_fpr_gate | 0.5999 | 0.3089 | 0.1220 | 0.1869 |

## Comparison vs Previous Fine-Tuned 5s Bundle

- Operating point changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Status/policy changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Validation AUROC ranking changed from `f3net, efficientnet_b4, videomae, i3d, effort_clip_l14, xception_df40` to `f3net, xception_df40, efficientnet_b4, videomae, i3d, effort_clip_l14`.
- Held-out FA/min improved for: none. Worsened for: efficientnet_b4, f3net, i3d.
- Held-out F1 improved for: efficientnet_b4, f3net, i3d. Worsened for: none.
- Held-out recall improved for: efficientnet_b4, f3net, i3d. Worsened for: none.
- Conservative 5s operating points were partially relaxed under the FPR-based rule.
