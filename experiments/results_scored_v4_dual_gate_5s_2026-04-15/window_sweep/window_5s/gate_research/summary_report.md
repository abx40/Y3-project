# V4 Dual Gate Report: gate_research

- Window: `5s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.20`, FA time ratio <= `0.10`
- Relaxed fallback: FA/min <= `0.30`, FA time ratio <= `0.15`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5350 | 0.597403 | 0.304636 | 0.403509 | 0.926174 | 0.208054 | 11.931762 | 3.350649 |
| 2 | `f3net` | `selected` | `strict` | 0.5258 | 0.587879 | 0.321192 | 0.415418 | 0.885906 | 0.228188 | 11.031655 | 3.345455 |
| 3 | `efficientnet_b4` | `selected` | `strict` | 0.5204 | 0.555556 | 0.165563 | 0.255102 | 0.322148 | 0.134228 | 14.51775 | 3.066667 |
| 4 | `i3d` | `selected` | `strict` | 0.4940 | 0.538462 | 0.092715 | 0.158192 | 0.362416 | 0.080537 | 15.72981 | 4.615385 |
| 5 | `effort_clip_l14` | `selected` | `strict` | 0.4755 | 0.575221 | 0.215232 | 0.313253 | 0.604027 | 0.161074 | 13.50706 | 3.39823 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4626 | 0.75 | 0.07947 | 0.143713 | 0.120805 | 0.026846 | 15.883679 | 6.0 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5523 | 0.671429 | 0.308197 | 0.422472 | 0.691525 | 0.155932 | 11.601927 | 2.914286 |
| `effort_clip_l14` | `selected` | `strict` | 0.4781 | 0.346939 | 0.111475 | 0.168734 | 0.691525 | 0.216949 | 15.951573 | 4.163265 |
| `f3net` | `selected` | `strict` | 0.5419 | 0.717105 | 0.357377 | 0.477024 | 0.650847 | 0.145763 | 11.187 | 2.289474 |
| `i3d` | `selected` | `strict` | 0.5514 | 0.507463 | 0.111475 | 0.182796 | 0.40678 | 0.111864 | 15.949683 | 2.865672 |
| `videomae` | `selected` | `strict` | 0.5728 | 0.545455 | 0.452459 | 0.494624 | 1.016949 | 0.389831 | 9.447037 | 1.944664 |
| `xception_df40` | `selected` | `strict` | 0.5046 | 0.566667 | 0.055738 | 0.101493 | 0.162712 | 0.044068 | 17.31311 | 2.8 |
