# V4 Dual Gate Report: gate_research

- Window: `1s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.20`, FA time ratio <= `0.10`
- Relaxed fallback: FA/min <= `0.30`, FA time ratio <= `0.15`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5333 | 0.686636 | 0.099201 | 0.173357 | 0.200267 | 0.045394 | 15.279369 | 5.529954 |
| 2 | `f3net` | `selected` | `strict` | 0.5195 | 0.533502 | 0.280959 | 0.368077 | 1.161549 | 0.246328 | 10.239738 | 5.764855 |
| 3 | `efficientnet_b4` | `selected` | `strict` | 0.5194 | 0.664122 | 0.115846 | 0.197279 | 0.480641 | 0.058745 | 14.907631 | 7.557252 |
| 4 | `i3d` | `selected` | `strict` | 0.5070 | 0.55814 | 0.015979 | 0.031068 | 0.12016 | 0.012684 | 17.368476 | 11.162791 |
| 5 | `effort_clip_l14` | `selected` | `strict` | 0.4765 | 0.419847 | 0.073236 | 0.124717 | 0.400534 | 0.101469 | 15.830369 | 4.122137 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4676 | 0.448276 | 0.034621 | 0.064277 | 0.280374 | 0.042724 | 16.626548 | 7.241379 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5546 | 0.644788 | 0.223113 | 0.331514 | 0.558882 | 0.122422 | 12.464488 | 4.633205 |
| `effort_clip_l14` | `selected` | `strict` | 0.4824 | 0.266187 | 0.049432 | 0.08338 | 0.558882 | 0.135729 | 16.71361 | 6.906475 |
| `f3net` | `selected` | `strict` | 0.5394 | 0.589436 | 0.327989 | 0.421459 | 1.197605 | 0.227545 | 9.815817 | 4.969988 |
| `i3d` | `selected` | `strict` | 0.5498 | 0.311111 | 0.009352 | 0.018158 | 0.159681 | 0.020625 | 18.1005 | 13.333333 |
| `videomae` | `selected` | `strict` | 0.5696 | 0.761702 | 0.119572 | 0.206697 | 0.319361 | 0.037259 | 15.416183 | 4.595745 |
| `xception_df40` | `selected` | `strict` | 0.4932 | 0.692308 | 0.03006 | 0.057618 | 0.07984 | 0.013307 | 17.3375 | 6.461538 |
