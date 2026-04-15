# V4 Dual Gate Report: gate_strict

- Window: `60s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict` | 0.6763 | 0.703704 | 0.730769 | 0.716981 | 0.208333 | 0.333333 | 5.853298 | 0.407407 |
| 2 | `videomae` | `selected` | `strict` | 0.6651 | 0.583333 | 0.538462 | 0.56 | 0.208333 | 0.416667 | 8.730131 | 0.25 |
| 3 | `effort_clip_l14` | `selected` | `strict` | 0.6234 | 0.45 | 0.346154 | 0.391304 | 0.125 | 0.458333 | 12.243643 | 0.35 |
| 4 | `efficientnet_b4` | `selected` | `strict` | 0.5128 | 0.555556 | 0.576923 | 0.566038 | 0.25 | 0.5 | 7.059798 | 0.259259 |
| 5 | `i3d` | `selected` | `strict` | 0.4936 | 0.071429 | 0.038462 | 0.05 | 0.125 | 0.541667 | 16.045536 | 0.142857 |
| 6 | `xception_df40` | `selected` | `strict` | 0.3942 | 0.565217 | 0.5 | 0.530612 | 0.25 | 0.416667 | 8.209548 | 0.26087 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.4432 | 0.538462 | 0.84 | 0.65625 | 0.36 | 0.72 | 3.60689 | 0.153846 |
| `effort_clip_l14` | `selected` | `strict` | 0.4928 | 0.619048 | 0.52 | 0.565217 | 0.16 | 0.32 | 7.564415 | 0.285714 |
| `f3net` | `selected` | `strict` | 0.7968 | 0.782609 | 0.72 | 0.75 | 0.2 | 0.2 | 7.041024 | 0.565217 |
| `i3d` | `selected` | `strict` | 0.5616 | 0.52 | 0.52 | 0.52 | 0.16 | 0.48 | 9.232134 | 0.12 |
| `videomae` | `selected` | `strict` | 0.4128 | 0.357143 | 0.2 | 0.25641 | 0.2 | 0.36 | 13.519085 | 0.428571 |
| `xception_df40` | `selected` | `strict` | 0.5344 | 0.478261 | 0.44 | 0.458333 | 0.16 | 0.48 | 10.019659 | 0.130435 |
