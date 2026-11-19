# Benchmark Scoring Report V2

## Method

- Evaluation unit: 56-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 56-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.10 FA/min and <= 0.05 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.25 FA/min and <= 0.15 real-time alert ratio.
- Calibration selection: among configs passing the gate, maximize recall on calibration fake+mixed, then minimize time-to-first-detection, then maximize F1.
- Validation ranking: lowest false alerts/min, then lowest time-to-first-detection, then highest F1.

## Winner

- Validation winner: `videomae`
- Validation false alerts/min: 0.0798
- Validation time-to-first-detection: 10.0679s
- Validation F1: 0.5581

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.10` | `high` | 2 | 0.37 | 0.27 | 0.0000 | 0.0000 | 0.5667 | 9.3547 |
| `effort_clip_l14` | `strict_0.10` | `low` | 3 | 0.11 | 0.11 | 0.0000 | 0.0000 | 0.5000 | 9.7987 |
| `f3net` | `strict_0.10` | `high` | 2 | 0.55 | 0.45 | 0.0000 | 0.0000 | 0.5333 | 8.8980 |
| `i3d` | `strict_0.10` | `low` | 2 | 0.31 | 0.26 | 0.0000 | 0.0000 | 0.7333 | 6.0220 |
| `videomae` | `strict_0.10` | `low` | 2 | 0.56 | 0.46 | 0.0000 | 0.0000 | 0.5667 | 8.0437 |
| `xception_df40` | `strict_0.10` | `low` | 3 | 0.30 | 0.30 | 0.0000 | 0.0000 | 0.2000 | 14.2308 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | 0.7500 | 0.4444 | 0.5581 | 0.0798 | 0.1383 | 10.0679 | 0.2778 |
| 2 | `xception_df40` | 0.5000 | 0.1111 | 0.1818 | 0.0798 | 0.1011 | 16.4429 | 0.7895 |
| 3 | `i3d` | 0.1579 | 0.1111 | 0.1304 | 0.1596 | 0.5638 | 15.0546 | 0.1772 |
| 4 | `f3net` | 0.5000 | 0.4815 | 0.4906 | 0.1995 | 0.4734 | 8.6103 | 0.3371 |
| 5 | `effort_clip_l14` | 0.4444 | 0.4444 | 0.4444 | 0.2394 | 0.5160 | 10.7876 | 0.2933 |
| 6 | `efficientnet_b4` | 0.2941 | 0.1852 | 0.2273 | 0.2394 | 0.4255 | 14.5146 | 0.3913 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5556 | 0.3226 | 0.4082 | 0.2344 | 0.3125 | 12.0137 | 0.2542 |
| `effort_clip_l14` | 0.1538 | 0.0645 | 0.0909 | 0.1406 | 0.4562 | 17.0146 | 0.3448 |
| `f3net` | 0.6875 | 0.3548 | 0.4681 | 0.1875 | 0.1938 | 11.5836 | 0.2123 |
| `i3d` | 0.5714 | 0.5161 | 0.5424 | 0.1406 | 0.5000 | 9.0858 | 0.1184 |
| `videomae` | 0.5000 | 0.4839 | 0.4918 | 0.1875 | 0.6188 | 8.7019 | 0.2970 |
| `xception_df40` | 0.7692 | 0.3226 | 0.4545 | 0.0938 | 0.1187 | 12.1272 | 0.2528 |

## Baselines

| baseline | split | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_real` | `validation` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 |
| `always_real` | `test` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 18.2927 |
| `always_fake` | `validation` | 0.4909 | 1.0000 | 0.6585 | 0.4787 | 1.0000 | 0.0000 |
| `always_fake` | `test` | 0.5636 | 1.0000 | 0.7209 | 0.5156 | 1.0000 | 0.0000 |
