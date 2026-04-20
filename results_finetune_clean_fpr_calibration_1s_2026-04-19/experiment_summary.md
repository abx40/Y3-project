# Fine-Tuned FPR Calibration Summary

## Method

- Calibration is FPR-based at a fixed 1s window.
- Candidate selection maximizes recall on fake-labelled calibration windows subject to real-window FPR on pooled real-labelled calibration windows.
- Strict FPR target: 0.05. Relaxed fallback FPR target: 0.10.
- FA/min is reported as an operational metric, not the main selection rule.
- Validation and test remain fully held out from threshold selection.

## Validation

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | selected | strict_fpr_gate | 0.5515 | 0.3046 | 0.6809 | 0.1944 |
| xception_df40 | selected | relaxed_fpr_gate | 0.5359 | 0.2371 | 2.0828 | 0.1485 |
| videomae | selected | strict_fpr_gate | 0.5245 | 0.0000 | 0.0401 | 0.0000 |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5234 | 0.1951 | 0.5207 | 0.1152 |
| i3d | selected | strict_fpr_gate | 0.5057 | 0.0000 | 0.0000 | 0.0000 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5027 | 0.0000 | 0.0000 | 0.0000 |

## Test

| backend | status | policy | AUROC | F1 | FA/min | recall |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b4 | selected | strict_fpr_gate | 0.5345 | 0.0843 | 0.1198 | 0.0448 |
| effort_clip_l14 | selected | strict_fpr_gate | 0.5765 | 0.0000 | 0.0000 | 0.0000 |
| f3net | selected | strict_fpr_gate | 0.6155 | 0.1990 | 0.4790 | 0.1169 |
| i3d | selected | strict_fpr_gate | 0.5410 | 0.0000 | 0.0000 | 0.0000 |
| videomae | selected | strict_fpr_gate | 0.4415 | 0.0000 | 0.0000 | 0.0000 |
| xception_df40 | selected | relaxed_fpr_gate | 0.5884 | 0.2692 | 1.5968 | 0.1663 |

## Comparison vs Previous Fine-Tuned 1s Bundle

- Operating point changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Status/policy changed for: efficientnet_b4, effort_clip_l14, f3net, i3d, videomae, xception_df40.
- Validation AUROC ranking changed from `videomae, efficientnet_b4, i3d, effort_clip_l14, f3net, xception_df40` to `f3net, xception_df40, videomae, efficientnet_b4, i3d, effort_clip_l14`.
- Held-out FA/min improved for: none. Worsened for: efficientnet_b4.
- Held-out F1 improved for: efficientnet_b4. Worsened for: none.
- Held-out recall improved for: efficientnet_b4. Worsened for: none.
- Conservative 1s operating points were not meaningfully fixed under the FPR-based rule.
