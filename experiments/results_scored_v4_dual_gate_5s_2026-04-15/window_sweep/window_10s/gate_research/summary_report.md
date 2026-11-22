# V4 Dual Gate Report: gate_research

- Window: `10s`
- Gate real stream: `calibration_real_only_01`
- Strict gate: FA/min <= `0.20`, FA time ratio <= `0.10`
- Relaxed fallback: FA/min <= `0.30`, FA time ratio <= `0.15`
- Calibration selection: maximize recall, then minimize TTFD, then maximize F1.
- Validation/test ranking: AUROC descending, then FA/min ascending, then TTFD ascending.

- Gate-compliant backends: `6` / `6`

## Validation

| rank | backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict` | 0.5556 | 0.477612 | 0.210526 | 0.292237 | 0.405405 | 0.236486 | 13.50369 | 1.432836 |
| 2 | `effort_clip_l14` | `selected` | `strict` | 0.5220 | 0.738462 | 0.315789 | 0.442396 | 0.527027 | 0.114865 | 11.258964 | 2.584615 |
| 3 | `f3net` | `selected` | `strict` | 0.5220 | 0.595238 | 0.493421 | 0.539568 | 1.013514 | 0.344595 | 8.071321 | 1.857143 |
| 4 | `efficientnet_b4` | `selected` | `strict` | 0.5100 | 0.447368 | 0.111842 | 0.178947 | 0.405405 | 0.141892 | 15.877798 | 1.578947 |
| 5 | `i3d` | `selected` | `strict` | 0.4801 | 0.45122 | 0.243421 | 0.316239 | 0.72973 | 0.304054 | 13.09475 | 0.292683 |
| 6 | `xception_df40` | `selected` | `strict` | 0.4792 | 0.581818 | 0.210526 | 0.309179 | 0.364865 | 0.155405 | 13.685226 | 1.527273 |

## Test

| backend | status | gate_used | AUROC | precision | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict` | 0.5316 | 0.651515 | 0.284768 | 0.396313 | 0.483221 | 0.154362 | 12.274512 | 1.272727 |
| `effort_clip_l14` | `selected` | `strict` | 0.5193 | 0.564516 | 0.231788 | 0.328638 | 0.724832 | 0.181208 | 13.148549 | 4.064516 |
| `f3net` | `selected` | `strict` | 0.5439 | 0.616667 | 0.490066 | 0.546125 | 1.006711 | 0.308725 | 8.661341 | 1.5 |
| `i3d` | `selected` | `strict` | 0.5515 | 0.471014 | 0.430464 | 0.449827 | 0.442953 | 0.489933 | 10.507463 | 0.130435 |
| `videomae` | `selected` | `strict` | 0.5865 | 0.52349 | 0.516556 | 0.52 | 0.966443 | 0.47651 | 8.355341 | 1.087248 |
| `xception_df40` | `selected` | `strict` | 0.5006 | 0.692308 | 0.238411 | 0.35468 | 0.281879 | 0.107383 | 13.813476 | 1.730769 |
