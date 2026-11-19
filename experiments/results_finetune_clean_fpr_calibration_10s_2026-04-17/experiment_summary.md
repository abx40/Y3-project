# Fine-Tuned FPR Calibration Summary

## Method

- Calibration is FPR-based at a fixed 10s window.
- Candidate selection maximizes recall on fake-labelled calibration windows subject to real-window FPR on pooled real-labelled calibration windows.
- Strict FPR target: 0.05. Relaxed fallback FPR target: 0.10.
- FA/min is reported as an operational metric, not the main selection rule.
- Validation and test remain fully held out from threshold selection.

## Validation

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | selected | strict_fpr_gate | 0.5768 | 0.4789 | 0.2027 | 0.3355 |
| xception_df40 | selected | strict_fpr_gate | 0.5537 | 0.2682 | 0.0811 | 0.1579 |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5351 | 0.2842 | 0.1622 | 0.1711 |
| videomae | selected | strict_fpr_gate | 0.5313 | 0.1310 | 0.1216 | 0.0724 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5089 | 0.0000 | 0.0000 | 0.0000 |
| i3d | selected | strict_fpr_gate | 0.5081 | 0.2041 | 0.4459 | 0.1316 |

## Test

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5622 | 0.1910 | 0.1208 | 0.1126 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5714 | 0.0000 | 0.0000 | 0.0000 |
| f3net | selected | strict_fpr_gate | 0.6587 | 0.4061 | 0.1611 | 0.2649 |
| i3d | selected | strict_fpr_gate | 0.5505 | 0.4190 | 0.4027 | 0.2914 |
| videomae | selected | strict_fpr_gate | 0.4375 | 0.0958 | 0.1611 | 0.0530 |
| xception_df40 | selected | strict_fpr_gate | 0.6094 | 0.2793 | 0.0805 | 0.1656 |

## Comparison vs Previous Fine-Tuned 10s Bundle

- Operating point changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Status/policy changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Validation AUROC ranking changed from `f3net, xception_df40, efficientnet_b4, videomae, effort_clip_l14, i3d` to `f3net, xception_df40, efficientnet_b4, videomae, effort_clip_l14, i3d`.
- Held-out FA/min improved for: none. Worsened for: efficientnet_b4, f3net, i3d, videomae, xception_df40.
- Held-out F1 improved for: efficientnet_b4, f3net, i3d, videomae, xception_df40. Worsened for: none.
- Held-out recall improved for: efficientnet_b4, f3net, i3d, videomae, xception_df40. Worsened for: none.
- Conservative 10s operating points were partially relaxed under the FPR-based rule.
