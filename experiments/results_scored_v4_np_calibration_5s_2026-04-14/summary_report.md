# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.
- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.
- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.
- Calibration gate: pooled real-labelled windows from all calibration sessions with real content, not just `calibration_real_only_01`.
- Gate uncertainty: 500 session bootstrap replicates; candidates pass only if the 95th percentile upper bound stays within <= 0.10 FA/min and <= 0.05 false-alert time ratio.
- Calibration objective: among gate-passing candidates, maximize recall on calibration runs containing fake content, then minimize TTFD, then minimize flicker.
- Calibration does not optimize F1 and does not fall back to a 'best available' threshold when the gate fails.
- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.

## Calibration Outcome

- Gate-compliant backends: 6 / 6

## Selected Operating Points

| backend | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `low` | 0.90 | 0.80 | 3 | 0.0333 | 0.0984 | 0.0028 | 0.0082 | 0.0361 | 16.5462 | 4.2857 | 0.5204 |
| `effort_clip_l14` | `low` | 0.96 | 0.96 | 1 | 0.0333 | 0.0741 | 0.0111 | 0.0247 | 0.0028 | 17.1002 | 24.0000 | 0.4755 |
| `f3net` | `high` | 0.95 | 0.90 | 3 | 0.0333 | 0.0819 | 0.0028 | 0.0068 | 0.0306 | 16.5924 | 5.0000 | 0.5258 |
| `i3d` | `low` | 0.42 | 0.32 | 3 | 0.0333 | 0.0811 | 0.0139 | 0.0338 | 0.0083 | 17.1366 | 6.0000 | 0.4940 |
| `videomae` | `low` | 0.96 | 0.91 | 3 | 0.0333 | 0.0984 | 0.0028 | 0.0082 | 0.0889 | 15.5501 | 3.6364 | 0.5350 |
| `xception_df40` | `low` | 0.96 | 0.96 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 17.1238 | 8.0000 | 0.4626 |

## Validation

| rank | backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | 0.5350 | 0.086093 | 0.156156 | 0.080537 | 0.016779 | 16.149821 | 4.645161 |
| 2 | `f3net` | `selected` | 0.5258 | 0.086093 | 0.1571 | 0.120805 | 0.010067 | 15.76419 | 4.965517 |
| 3 | `efficientnet_b4` | `selected` | 0.5204 | 0.036424 | 0.06962 | 0.080537 | 0.010067 | 17.04419 | 6.0 |
| 4 | `i3d` | `selected` | 0.4940 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 5 | `effort_clip_l14` | `selected` | 0.4755 | 0.006623 | 0.013072 | 0.040268 | 0.006711 | 17.730952 | 6.0 |
| 6 | `xception_df40` | `selected` | 0.4626 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |

## Test

| backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | 0.5523 | 0.045902 | 0.083582 | 0.081356 | 0.054237 | 17.37489 | 2.4 |
| `effort_clip_l14` | `selected` | 0.4781 | 0.016393 | 0.031447 | 0.244068 | 0.027119 | 17.862415 | 12.923077 |
| `f3net` | `selected` | 0.5419 | 0.101639 | 0.183976 | 0.040678 | 0.00339 | 16.152427 | 3.0 |
| `i3d` | `selected` | 0.5514 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `videomae` | `selected` | 0.5728 | 0.111475 | 0.194286 | 0.162712 | 0.037288 | 15.898841 | 3.2 |
| `xception_df40` | `selected` | 0.5046 | 0.003279 | 0.006494 | 0.040678 | 0.00678 | 18.261817 | 8.0 |
