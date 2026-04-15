# V4 Dual Gate Report: gate_strict

- Window: `15s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict` | 0.5582 | 0.619048 | 0.637255 | 0.628019 | 0.938776 | 0.408163 | 6.310286 | 0.457143 |
| 2 | `effort_clip_l14` | `selected` | `strict` | 0.5308 | 0.706897 | 0.401961 | 0.5125 | 0.530612 | 0.173469 | 10.653786 | 0.758621 |
| 3 | `videomae` | `selected` | `strict` | 0.5266 | 0.486486 | 0.176471 | 0.258993 | 0.367347 | 0.193878 | 14.356821 | 1.837838 |
| 4 | `i3d` | `selected` | `strict` | 0.5155 | 0.521368 | 0.598039 | 0.557078 | 1.020408 | 0.571429 | 7.284179 | 0.17094 |
| 5 | `efficientnet_b4` | `selected` | `strict` | 0.5144 | 0.514286 | 0.176471 | 0.262774 | 0.408163 | 0.173469 | 14.027119 | 3.542857 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4421 | 0.47619 | 0.098039 | 0.162602 | 0.285714 | 0.112245 | 16.060679 | 1.714286 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.4483 | 0.518519 | 0.135922 | 0.215385 | 0.412371 | 0.134021 | 14.672195 | 4.740741 |
| `effort_clip_l14` | `selected` | `strict` | 0.5406 | 0.5 | 0.213592 | 0.29932 | 0.659794 | 0.226804 | 13.521402 | 1.0 |
| `f3net` | `selected` | `strict` | 0.5549 | 0.545455 | 0.524272 | 0.534653 | 0.701031 | 0.463918 | 8.153732 | 0.323232 |
| `i3d` | `selected` | `strict` | 0.4719 | 0.421053 | 0.466019 | 0.442396 | 0.948454 | 0.680412 | 9.895073 | 0.175439 |
| `videomae` | `selected` | `strict` | 0.5729 | 0.589041 | 0.417476 | 0.488636 | 0.57732 | 0.309278 | 10.267927 | 1.315068 |
| `xception_df40` | `selected` | `strict` | 0.4929 | 0.692308 | 0.174757 | 0.27907 | 0.247423 | 0.082474 | 14.747293 | 2.0 |
