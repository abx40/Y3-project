# V4 Dual Gate Window Sweep Summary

- Default reporting window: `5s`.
- Strict gate and research gate are kept fully separate in this bundle.

| window_s | strict compliant | strict winner | strict winner AUROC | strict mean val AUROC | strict mean val F1 | strict mean val FA/min | research compliant | research winner | research winner AUROC | research mean val AUROC | research mean val F1 | research mean val FA/min |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 6 | `videomae` | 0.533332 | 0.5038856666666667 | 0.12209066666666667 | 0.2937248333333333 | 6 | `videomae` | 0.533332 | 0.5038856666666667 | 0.15979583333333333 | 0.44058749999999997 |
| 5 | 6 | `videomae` | 0.534979 | 0.5022191666666667 | 0.211143 | 0.3691275 | 6 | `videomae` | 0.534979 | 0.5022191666666667 | 0.28153116666666667 | 0.5369126666666667 |
| 10 | 6 | `videomae` | 0.55561 | 0.5114725 | 0.3558325 | 0.6283785 | 6 | `videomae` | 0.55561 | 0.5114725 | 0.3464276666666667 | 0.5743243333333333 |
| 15 | 6 | `f3net` | 0.558223 | 0.5145998333333334 | 0.39699433333333334 | 0.5918366666666667 | 6 | `f3net` | 0.558223 | 0.5145998333333334 | 0.42675383333333333 | 0.5782313333333333 |
| 20 | 6 | `f3net` | 0.591038 | 0.49689116666666666 | 0.376548 | 0.361842 | 6 | `f3net` | 0.591038 | 0.49689116666666666 | 0.4687903333333333 | 0.4539471666666667 |
| 30 | 6 | `videomae` | 0.634461 | 0.545767 | 0.47291883333333334 | 0.24691333333333335 | 6 | `videomae` | 0.634461 | 0.545767 | 0.4730835 | 0.25308616666666667 |
| 60 | 6 | `f3net` | 0.676282 | 0.5608975 | 0.4691558333333333 | 0.19444433333333333 | 6 | `f3net` | 0.676282 | 0.5608975 | 0.4691558333333333 | 0.19444433333333333 |

## Best Validation Combo By Model

### Strict Gate

| backend | best window_s | status | policy | val AUROC | val F1 | val FA/min | val TTFD s | test AUROC | test F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 5 | `selected` | `strict` | 0.520379 | 0.202312 | 0.161074 | 15.4645 | 0.552253 | 0.382284 |
| `effort_clip_l14` | 60 | `selected` | `strict` | 0.623397 | 0.391304 | 0.125 | 12.243643 | 0.4928 | 0.565217 |
| `f3net` | 60 | `selected` | `strict` | 0.676282 | 0.716981 | 0.208333 | 5.853298 | 0.7968 | 0.75 |
| `i3d` | 15 | `selected` | `strict` | 0.51547 | 0.557078 | 1.020408 | 7.284179 | 0.471925 | 0.442396 |
| `videomae` | 60 | `selected` | `strict` | 0.665064 | 0.56 | 0.208333 | 8.730131 | 0.4128 | 0.25641 |
| `xception_df40` | 30 | `selected` | `strict` | 0.572464 | 0.416667 | 0.222222 | 12.582298 | 0.463141 | 0.463415 |

### Research Gate

| backend | best window_s | status | policy | val AUROC | val F1 | val FA/min | val TTFD s | test AUROC | test F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 5 | `selected` | `strict` | 0.520379 | 0.255102 | 0.322148 | 14.51775 | 0.552253 | 0.422472 |
| `effort_clip_l14` | 60 | `selected` | `strict` | 0.623397 | 0.391304 | 0.125 | 12.243643 | 0.4928 | 0.565217 |
| `f3net` | 60 | `selected` | `strict` | 0.676282 | 0.716981 | 0.208333 | 5.853298 | 0.7968 | 0.75 |
| `i3d` | 15 | `selected` | `strict` | 0.51547 | 0.557078 | 1.020408 | 7.284179 | 0.471925 | 0.442396 |
| `videomae` | 60 | `selected` | `strict` | 0.665064 | 0.56 | 0.208333 | 8.730131 | 0.4128 | 0.25641 |
| `xception_df40` | 30 | `selected` | `strict` | 0.572464 | 0.416667 | 0.222222 | 12.582298 | 0.463141 | 0.463415 |
