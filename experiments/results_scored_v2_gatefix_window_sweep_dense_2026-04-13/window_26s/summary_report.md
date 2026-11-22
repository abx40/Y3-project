# Benchmark Scoring Report V2

## Method

- Evaluation unit: 26-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 26-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `effort_clip_l14`
- Validation false alerts/min: 0.1732
- Validation time-to-first-detection: 12.2973s
- Validation F1: 0.3478

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 3 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.3788 | 11.6454 |
| `effort_clip_l14` | `strict_0.10` | `high` | 3 | 0.97 | 0.87 | 0.0000 | 0.0000 | 0.2273 | 14.2752 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.6364 | 6.5536 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.7424 | 6.0179 |
| `videomae` | `strict_0.10` | `low` | 3 | 0.50 | 0.45 | 0.1000 | 0.0033 | 0.6364 | 6.2726 |
| `xception_df40` | `strict_0.10` | `high` | 2 | 0.89 | 0.79 | 0.0000 | 0.0000 | 0.2879 | 12.1648 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `effort_clip_l14` | 0.5517 | 0.2540 | 0.3478 | 0.1732 | 0.2092 | 12.2973 | 0.5099 |
| 2 | `efficientnet_b4` | 0.3421 | 0.2063 | 0.2574 | 0.1732 | 0.4343 | 14.9670 | 0.3275 |
| 3 | `i3d` | 0.6585 | 0.4286 | 0.5192 | 0.2165 | 0.2280 | 10.1838 | 0.1811 |
| 4 | `xception_df40` | 0.4091 | 0.2857 | 0.3364 | 0.2597 | 0.4531 | 11.9408 | 0.4380 |
| 5 | `videomae` | 0.5079 | 0.5079 | 0.5079 | 0.3030 | 0.5469 | 8.9629 | 0.3448 |
| 6 | `f3net` | 0.5818 | 0.5079 | 0.5424 | 0.3896 | 0.3968 | 8.7154 | 0.4860 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6000 | 0.2812 | 0.3830 | 0.2206 | 0.1765 | 13.1198 | 0.5263 |
| `effort_clip_l14` | 0.6667 | 0.0312 | 0.0597 | 0.0441 | 0.0191 | 17.7730 | 1.5385 |
| `f3net` | 0.6444 | 0.4531 | 0.5321 | 0.3971 | 0.2529 | 9.6611 | 0.4469 |
| `i3d` | 0.5238 | 0.8594 | 0.6509 | 0.6618 | 0.9029 | 2.8038 | 0.1367 |
| `videomae` | 0.5556 | 0.7031 | 0.6207 | 0.4853 | 0.6529 | 5.8011 | 0.2655 |
| `xception_df40` | 0.5000 | 0.2969 | 0.3725 | 0.3971 | 0.3279 | 13.1809 | 0.7205 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5250 | 1.0000 | 0.6885 | 0.6926 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5333 | 1.0000 | 0.6957 | 0.7059 | 1.0000 | 0.0000 |
