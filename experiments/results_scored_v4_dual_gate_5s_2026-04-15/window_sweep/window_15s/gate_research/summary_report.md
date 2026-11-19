# V4 Dual Gate Report: gate_research

- Window: `15s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.20`, FA time ratio <= `0.10`
- Relaxed fallback: FA/min <= `0.30`, FA time ratio <= `0.15`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict` | 0.5582 | 0.619048 | 0.637255 | 0.628019 | 0.938776 | 0.408163 | 6.310286 | 0.457143 |
| 2 | `effort_clip_l14` | `selected` | `strict` | 0.5308 | 0.706897 | 0.401961 | 0.5125 | 0.530612 | 0.173469 | 10.653786 | 0.758621 |
| 3 | `videomae` | `selected` | `strict` | 0.5266 | 0.5 | 0.323529 | 0.392857 | 0.489796 | 0.336735 | 11.227 | 0.848485 |
| 4 | `i3d` | `selected` | `strict` | 0.5155 | 0.521368 | 0.598039 | 0.557078 | 1.020408 | 0.571429 | 7.284179 | 0.17094 |
| 5 | `efficientnet_b4` | `selected` | `strict` | 0.5144 | 0.44186 | 0.186275 | 0.262069 | 0.326531 | 0.244898 | 13.732798 | 1.860465 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4421 | 0.565217 | 0.127451 | 0.208 | 0.163265 | 0.102041 | 15.403762 | 1.043478 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.4483 | 0.452381 | 0.184466 | 0.262069 | 0.371134 | 0.237113 | 14.656146 | 1.428571 |
| `effort_clip_l14` | `selected` | `strict` | 0.5406 | 0.5 | 0.213592 | 0.29932 | 0.659794 | 0.226804 | 13.521402 | 1.0 |
| `f3net` | `selected` | `strict` | 0.5549 | 0.545455 | 0.524272 | 0.534653 | 0.701031 | 0.463918 | 8.153732 | 0.323232 |
| `i3d` | `selected` | `strict` | 0.4719 | 0.421053 | 0.466019 | 0.442396 | 0.948454 | 0.680412 | 9.895073 | 0.175439 |
| `videomae` | `selected` | `strict` | 0.5729 | 0.533333 | 0.543689 | 0.538462 | 0.742268 | 0.505155 | 7.676195 | 0.72381 |
| `xception_df40` | `selected` | `strict` | 0.4929 | 0.741935 | 0.223301 | 0.343284 | 0.247423 | 0.082474 | 13.867451 | 1.290323 |
