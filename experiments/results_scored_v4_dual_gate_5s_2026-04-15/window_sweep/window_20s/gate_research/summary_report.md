# V4 Dual Gate Report: gate_research

- Window: `20s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.20`, FA time ratio <= `0.10`
- Relaxed fallback: FA/min <= `0.30`, FA time ratio <= `0.15`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict` | 0.5910 | 0.549451 | 0.675676 | 0.606061 | 0.592105 | 0.539474 | 6.020536 | 0.296703 |
| 2 | `effort_clip_l14` | `selected` | `strict` | 0.5381 | 0.72973 | 0.364865 | 0.486486 | 0.355263 | 0.131579 | 11.585714 | 0.810811 |
| 3 | `i3d` | `selected` | `strict` | 0.4881 | 0.367647 | 0.337838 | 0.352113 | 0.355263 | 0.565789 | 12.23144 | 0.176471 |
| 4 | `videomae` | `selected` | `strict` | 0.4648 | 0.509091 | 0.378378 | 0.434109 | 0.236842 | 0.355263 | 10.78781 | 0.490909 |
| 5 | `efficientnet_b4` | `selected` | `strict` | 0.4577 | 0.534247 | 0.527027 | 0.530612 | 0.710526 | 0.447368 | 8.204643 | 0.369863 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4417 | 0.533333 | 0.324324 | 0.403361 | 0.473684 | 0.276316 | 11.966667 | 0.666667 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5707 | 0.463158 | 0.60274 | 0.52381 | 0.74026 | 0.662338 | 7.054476 | 0.284211 |
| `effort_clip_l14` | `selected` | `strict` | 0.4915 | 0.55 | 0.150685 | 0.236559 | 0.233766 | 0.116883 | 15.568512 | 0.75 |
| `f3net` | `selected` | `strict` | 0.5958 | 0.474359 | 0.506849 | 0.490066 | 0.623377 | 0.532468 | 8.610878 | 0.192308 |
| `i3d` | `selected` | `strict` | 0.6001 | 0.476562 | 0.835616 | 0.606965 | 0.974026 | 0.87013 | 3.10772 | 0.117188 |
| `videomae` | `selected` | `strict` | 0.6234 | 0.476744 | 0.561644 | 0.515723 | 0.545455 | 0.584416 | 7.382732 | 0.488372 |
| `xception_df40` | `selected` | `strict` | 0.5296 | 0.821429 | 0.315068 | 0.455446 | 0.155844 | 0.064935 | 12.227305 | 0.642857 |
