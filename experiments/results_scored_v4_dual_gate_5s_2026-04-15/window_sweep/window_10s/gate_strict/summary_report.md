# V4 Dual Gate Report: gate_strict

- Window: `10s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.10`, FA time ratio <= `0.05`
- Relaxed fallback: FA/min <= `0.20`, FA time ratio <= `0.10`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5556 | 0.56 | 0.184211 | 0.277228 | 0.364865 | 0.148649 | 13.946167 | 1.44 |
| 2 | `effort_clip_l14` | `selected` | `strict` | 0.5220 | 0.643564 | 0.427632 | 0.513834 | 0.891892 | 0.243243 | 9.809381 | 0.534653 |
| 3 | `f3net` | `selected` | `strict` | 0.5220 | 0.595238 | 0.493421 | 0.539568 | 1.013514 | 0.344595 | 8.071321 | 1.857143 |
| 4 | `efficientnet_b4` | `selected` | `strict` | 0.5100 | 0.447368 | 0.111842 | 0.178947 | 0.405405 | 0.141892 | 15.877798 | 1.578947 |
| 5 | `i3d` | `selected` | `strict` | 0.4801 | 0.45122 | 0.243421 | 0.316239 | 0.72973 | 0.304054 | 13.09475 | 0.292683 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4792 | 0.581818 | 0.210526 | 0.309179 | 0.364865 | 0.155405 | 13.685226 | 1.527273 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5316 | 0.651515 | 0.284768 | 0.396313 | 0.483221 | 0.154362 | 12.274512 | 1.272727 |
| `effort_clip_l14` | `selected` | `strict` | 0.5193 | 0.5 | 0.231788 | 0.316742 | 0.845638 | 0.234899 | 13.858549 | 0.942857 |
| `f3net` | `selected` | `strict` | 0.5439 | 0.616667 | 0.490066 | 0.546125 | 1.006711 | 0.308725 | 8.661341 | 1.5 |
| `i3d` | `selected` | `strict` | 0.5515 | 0.471014 | 0.430464 | 0.449827 | 0.442953 | 0.489933 | 10.507463 | 0.130435 |
| `videomae` | `selected` | `strict` | 0.5865 | 0.507692 | 0.437086 | 0.469751 | 0.765101 | 0.42953 | 10.00989 | 1.061538 |
| `xception_df40` | `selected` | `strict` | 0.5006 | 0.659574 | 0.205298 | 0.313131 | 0.281879 | 0.107383 | 14.115341 | 2.170213 |
