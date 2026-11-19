# Benchmark Scoring Report V2

## Method

- Evaluation unit: 33-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 33-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `xception_df40`
- Validation false alerts/min: 0.0875
- Validation time-to-first-detection: 17.0126s
- Validation F1: 0.1034

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `low` | 3 | 0.71 | 0.71 | 0.0000 | 0.0000 | 0.3019 | 12.1544 |
| `effort_clip_l14` | `strict_0.10` | `high` | 2 | 0.98 | 0.98 | 0.0000 | 0.0000 | 0.1887 | 14.0760 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.53 | 0.43 | 0.0000 | 0.0000 | 0.6981 | 5.0286 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.32 | 0.27 | 0.0000 | 0.0000 | 0.6038 | 6.4971 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.59 | 0.49 | 0.1000 | 0.0100 | 0.5283 | 8.7743 |
| `xception_df40` | `strict_0.10` | `low` | 2 | 0.46 | 0.46 | 0.0000 | 0.0000 | 0.0755 | 15.3410 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | 0.4286 | 0.0588 | 0.1034 | 0.0875 | 0.0766 | 17.0126 | 0.8824 |
| 2 | `effort_clip_l14` | 0.7143 | 0.1961 | 0.3077 | 0.1313 | 0.0963 | 13.8786 | 1.0390 |
| 3 | `videomae` | 0.5455 | 0.2353 | 0.3288 | 0.1313 | 0.2210 | 14.0741 | 0.6009 |
| 4 | `efficientnet_b4` | 0.2500 | 0.0588 | 0.0952 | 0.1313 | 0.1772 | 17.0766 | 0.9524 |
| 5 | `i3d` | 0.2857 | 0.2353 | 0.2581 | 0.3063 | 0.6630 | 14.0047 | 0.2398 |
| 6 | `f3net` | 0.5962 | 0.6078 | 0.6019 | 0.5252 | 0.4858 | 6.4073 | 0.4037 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.6364 | 0.2917 | 0.4000 | 0.1247 | 0.1642 | 12.6195 | 0.6009 |
| `effort_clip_l14` | 0.3333 | 0.0208 | 0.0392 | 0.0832 | 0.0270 | 17.8271 | 2.5000 |
| `f3net` | 0.4706 | 0.5000 | 0.4848 | 0.4158 | 0.5613 | 7.9021 | 0.4571 |
| `i3d` | 0.4189 | 0.6458 | 0.5082 | 0.3742 | 0.9085 | 5.0992 | 0.1300 |
| `videomae` | 0.5556 | 0.5208 | 0.5376 | 0.2911 | 0.4200 | 9.7598 | 0.5556 |
| `xception_df40` | 0.7000 | 0.1458 | 0.2414 | 0.0832 | 0.0499 | 15.1812 | 1.7822 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.5368 | 1.0000 | 0.6986 | 0.6565 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5053 | 1.0000 | 0.6713 | 0.4158 | 1.0000 | 0.0000 |
