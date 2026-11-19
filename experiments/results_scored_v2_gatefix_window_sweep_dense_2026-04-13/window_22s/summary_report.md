# Benchmark Scoring Report V2

## Method

- Evaluation unit: 22-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 22-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `i3d`
- Validation false alerts/min: 0.3125
- Validation time-to-first-detection: 13.9933s
- Validation F1: 0.1905

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.71 | 0.61 | 0.0000 | 0.0000 | 0.7160 | 4.9185 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.10 | 0.00 | 0.1000 | 0.0467 | 0.6420 | 6.1673 |
| `f3net` | `strict_0.10` | `high` | 3 | 0.54 | 0.44 | 0.0000 | 0.0000 | 0.6914 | 6.0269 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.7160 | 4.9673 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.62 | 0.52 | 0.1000 | 0.0467 | 0.4074 | 9.9326 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.32 | 0.22 | 0.1000 | 0.0100 | 0.4568 | 8.8554 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.2069 | 0.1765 | 0.1905 | 0.3125 | 0.6276 | 13.9933 | 0.1980 |
| 2 | `efficientnet_b4` | 0.3721 | 0.2353 | 0.2883 | 0.3516 | 0.3555 | 13.1075 | 0.6013 |
| 3 | `videomae` | 0.3590 | 0.2059 | 0.2617 | 0.4297 | 0.3372 | 13.4994 | 1.0169 |
| 4 | `f3net` | 0.5616 | 0.6029 | 0.5816 | 0.4688 | 0.4479 | 6.9954 | 0.3466 |
| 5 | `xception_df40` | 0.4615 | 0.2647 | 0.3364 | 0.4688 | 0.2799 | 12.4785 | 1.2593 |
| 6 | `effort_clip_l14` | 0.5368 | 0.7500 | 0.6258 | 0.6641 | 0.5990 | 4.1220 | 0.1493 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.4138 | 0.5294 | 0.4645 | 0.7105 | 0.7171 | 8.7062 | 0.1913 |
| `effort_clip_l14` | 0.4268 | 0.5147 | 0.4667 | 0.5526 | 0.6382 | 9.0798 | 0.1740 |
| `f3net` | 0.6279 | 0.3971 | 0.4865 | 0.3158 | 0.2000 | 11.2421 | 0.2721 |
| `i3d` | 0.6000 | 0.7500 | 0.6667 | 0.6711 | 0.4711 | 4.7784 | 0.1647 |
| `videomae` | 0.5190 | 0.6029 | 0.5578 | 0.5132 | 0.5289 | 7.3452 | 0.7456 |
| `xception_df40` | 0.5625 | 0.2647 | 0.3600 | 0.3553 | 0.1816 | 12.8818 | 1.6071 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4857 | 1.0000 | 0.6538 | 0.9766 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.4857 | 1.0000 | 0.6538 | 0.8684 | 1.0000 | 0.0000 |
