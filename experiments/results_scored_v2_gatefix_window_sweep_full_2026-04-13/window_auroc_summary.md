# Gatefixed Window AUROC Summary

| window_s | mean validation AUROC | mean test AUROC | best validation backend | best validation AUROC | best test backend | best test AUROC |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 0.5039 | 0.5315 | `videomae` | 0.5333 | `videomae` | 0.5696 |
| 5 | 0.5022 | 0.5335 | `videomae` | 0.5350 | `videomae` | 0.5728 |
| 10 | 0.5115 | 0.5389 | `videomae` | 0.5556 | `videomae` | 0.5865 |
| 15 | 0.5146 | 0.5136 | `f3net` | 0.5582 | `videomae` | 0.5729 |
| 20 | 0.4969 | 0.5685 | `f3net` | 0.5910 | `videomae` | 0.6234 |
| 30 | 0.5458 | 0.5072 | `videomae` | 0.6345 | `efficientnet_b4` | 0.6110 |
| 60 | 0.5609 | 0.5403 | `f3net` | 0.6763 | `f3net` | 0.7968 |

## Best Validation-AUROC Window Per Backend

| backend | best window_s | polarity | validation AUROC | test AUROC |
| --- | ---: | --- | ---: | ---: |
| `efficientnet_b4` | 5 | `low` | 0.5204 | 0.5523 |
| `effort_clip_l14` | 60 | `high` | 0.6234 | 0.4928 |
| `f3net` | 60 | `high` | 0.6763 | 0.7968 |
| `i3d` | 15 | `high` | 0.5155 | 0.4719 |
| `videomae` | 60 | `high` | 0.6651 | 0.4128 |
| `xception_df40` | 30 | `low` | 0.5725 | 0.4631 |
