# V4 Dual Gate Report: gate_strict

- Window: `30s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.6345 | 0.465116 | 0.434783 | 0.449438 | 0.259259 | 0.425926 | 10.094393 | 0.651163 |
| 2 | `xception_df40` | `selected` | `strict` | 0.5725 | 0.576923 | 0.326087 | 0.416667 | 0.222222 | 0.203704 | 12.582298 | 0.461538 |
| 3 | `f3net` | `selected` | `strict` | 0.5588 | 0.491803 | 0.652174 | 0.560748 | 0.259259 | 0.574074 | 7.334083 | 0.262295 |
| 4 | `effort_clip_l14` | `selected` | `strict` | 0.5386 | 0.631579 | 0.521739 | 0.571429 | 0.259259 | 0.259259 | 9.417512 | 0.578947 |
| 5 | `efficientnet_b4` | `selected` | `strict` | 0.5052 | 0.413793 | 0.26087 | 0.32 | 0.222222 | 0.314815 | 12.077774 | 0.689655 |
| 6 | `i3d` | `selected` | `strict` | 0.4650 | 0.465517 | 0.586957 | 0.519231 | 0.259259 | 0.574074 | 6.068726 | 0.206897 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.6110 | 0.54386 | 0.645833 | 0.590476 | 0.307692 | 0.5 | 6.457463 | 0.526316 |
| `effort_clip_l14` | `selected` | `strict` | 0.3678 | 0.355556 | 0.333333 | 0.344086 | 0.269231 | 0.557692 | 11.054305 | 0.577778 |
| `f3net` | `selected` | `strict` | 0.6054 | 0.5 | 0.479167 | 0.489362 | 0.230769 | 0.442308 | 8.441817 | 0.26087 |
| `i3d` | `selected` | `strict` | 0.4247 | 0.236842 | 0.1875 | 0.209302 | 0.192308 | 0.557692 | 14.307512 | 0.210526 |
| `videomae` | `selected` | `strict` | 0.5713 | 0.493151 | 0.75 | 0.595041 | 0.346154 | 0.711538 | 4.952012 | 0.328767 |
| `xception_df40` | `selected` | `strict` | 0.4631 | 0.558824 | 0.395833 | 0.463415 | 0.192308 | 0.288462 | 10.92428 | 0.529412 |
