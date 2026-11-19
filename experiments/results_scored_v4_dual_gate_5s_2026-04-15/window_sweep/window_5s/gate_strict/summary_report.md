# V4 Dual Gate Report: gate_strict

- Window: `5s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5350 | 0.6 | 0.298013 | 0.39823 | 0.885906 | 0.201342 | 11.97069 | 3.28 |
| 2 | `f3net` | `selected` | `strict` | 0.5258 | 0.595745 | 0.278146 | 0.379233 | 0.765101 | 0.191275 | 11.839179 | 3.914894 |
| 3 | `efficientnet_b4` | `selected` | `strict` | 0.5204 | 0.795455 | 0.115894 | 0.202312 | 0.161074 | 0.030201 | 15.4645 | 4.090909 |
| 4 | `i3d` | `selected` | `strict` | 0.4940 | 0.444444 | 0.013245 | 0.025723 | 0.040268 | 0.016779 | 17.599524 | 5.333333 |
| 5 | `effort_clip_l14` | `selected` | `strict` | 0.4755 | 0.526316 | 0.066225 | 0.117647 | 0.241611 | 0.060403 | 16.289202 | 4.105263 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4626 | 0.75 | 0.07947 | 0.143713 | 0.120805 | 0.026846 | 15.883679 | 6.0 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5523 | 0.66129 | 0.268852 | 0.382284 | 0.610169 | 0.142373 | 12.642988 | 3.290323 |
| `effort_clip_l14` | `selected` | `strict` | 0.4781 | 0.295455 | 0.042623 | 0.074499 | 0.447458 | 0.105085 | 17.212512 | 7.090909 |
| `f3net` | `selected` | `strict` | 0.5419 | 0.722973 | 0.35082 | 0.472406 | 0.610169 | 0.138983 | 11.187 | 2.351351 |
| `i3d` | `selected` | `strict` | 0.5514 | 0.472222 | 0.055738 | 0.099707 | 0.284746 | 0.064407 | 17.149512 | 2.0 |
| `videomae` | `selected` | `strict` | 0.5728 | 0.541126 | 0.409836 | 0.466418 | 0.894915 | 0.359322 | 10.380878 | 2.181818 |
| `xception_df40` | `selected` | `strict` | 0.5046 | 0.566667 | 0.055738 | 0.101493 | 0.162712 | 0.044068 | 17.31311 | 2.8 |
