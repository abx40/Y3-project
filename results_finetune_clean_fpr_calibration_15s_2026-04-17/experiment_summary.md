# Fine-Tuned FPR Calibration Summary

## Method

- Calibration is FPR-based at a fixed 15-second window.
- Candidate selection maximizes recall on fake-labelled calibration windows subject to real-window FPR on pooled real-labelled calibration windows.
- Strict FPR target: 0.05. Relaxed fallback FPR target: 0.10.
- FA/min is reported as an operational metric, not the main selection rule.
- Validation and test remain fully held out from threshold selection.

## Validation

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | selected | strict_fpr_gate | 0.5985 | 0.5000 | 0.3265 | 0.3627 |
| xception_df40 | selected | strict_fpr_gate | 0.5613 | 0.5488 | 0.5306 | 0.4412 |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5569 | 0.1947 | 0.0000 | 0.1078 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5329 | 0.0000 | 0.0000 | 0.0000 |
| videomae | selected | strict_fpr_gate | 0.5000 | 0.6512 | 0.4082 | 0.5490 |
| i3d | selected | strict_fpr_gate | 0.4742 | 0.1148 | 0.3265 | 0.0686 |

## Test

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5808 | 0.2017 | 0.0825 | 0.1165 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.6278 | 0.0000 | 0.0000 | 0.0000 |
| f3net | selected | strict_fpr_gate | 0.6968 | 0.3885 | 0.2474 | 0.2621 |
| i3d | selected | strict_fpr_gate | 0.4654 | 0.1525 | 0.2062 | 0.0874 |
| videomae | selected | strict_fpr_gate | 0.4201 | 0.0566 | 0.0000 | 0.0291 |
| xception_df40 | selected | strict_fpr_gate | 0.6337 | 0.4088 | 0.1649 | 0.2718 |

## Comparison vs Previous Fine-Tuned 15s Bundle

- Operating point changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Status/policy changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Validation AUROC ranking changed from `f3net, xception_df40, efficientnet_b4, effort_clip_l14, videomae, i3d` to `f3net, xception_df40, efficientnet_b4, effort_clip_l14, videomae, i3d`.
- Held-out FA/min improved for: none. Worsened for: f3net, xception_df40.
- Held-out F1 improved for: efficientnet_b4, f3net, videomae, xception_df40. Worsened for: i3d.
- Held-out recall improved for: efficientnet_b4, f3net, videomae, xception_df40. Worsened for: i3d.
- Conservative 15s operating points were partially relaxed under the FPR-based rule.
