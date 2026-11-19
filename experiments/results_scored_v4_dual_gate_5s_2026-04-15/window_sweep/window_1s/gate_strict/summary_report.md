# V4 Dual Gate Report: gate_strict

- Window: `1s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5333 | 0.686636 | 0.099201 | 0.173357 | 0.200267 | 0.045394 | 15.279369 | 5.529954 |
| 2 | `f3net` | `selected` | `strict` | 0.5195 | 0.551786 | 0.205726 | 0.299709 | 0.921228 | 0.167557 | 11.715679 | 7.5 |
| 3 | `efficientnet_b4` | `selected` | `strict` | 0.5194 | 0.688525 | 0.083888 | 0.149555 | 0.320427 | 0.038051 | 15.54525 | 7.213115 |
| 4 | `i3d` | `selected` | `strict` | 0.5070 | 0.536585 | 0.014647 | 0.028516 | 0.12016 | 0.012684 | 17.392286 | 11.707317 |
| 5 | `effort_clip_l14` | `selected` | `strict` | 0.4765 | 0.404959 | 0.032623 | 0.060382 | 0.160214 | 0.048064 | 16.960405 | 4.958678 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4676 | 0.8 | 0.010652 | 0.021025 | 0.040053 | 0.00267 | 17.473595 | 18.0 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5546 | 0.644986 | 0.158985 | 0.255091 | 0.359281 | 0.087159 | 14.215024 | 4.552846 |
| `effort_clip_l14` | `selected` | `strict` | 0.4824 | 0.263393 | 0.039412 | 0.068565 | 0.479042 | 0.10978 | 17.070305 | 7.5 |
| `f3net` | `selected` | `strict` | 0.5394 | 0.658887 | 0.245157 | 0.357352 | 0.798403 | 0.126414 | 11.944878 | 5.816876 |
| `i3d` | `selected` | `strict` | 0.5498 | 0.210526 | 0.002672 | 0.005277 | 0.07984 | 0.00998 | 18.249622 | 12.631579 |
| `videomae` | `selected` | `strict` | 0.5696 | 0.761702 | 0.119572 | 0.206697 | 0.319361 | 0.037259 | 15.416183 | 4.595745 |
| `xception_df40` | `selected` | `strict` | 0.4932 | 0.7 | 0.009352 | 0.018457 | 0.03992 | 0.003992 | 18.127671 | 6.0 |
