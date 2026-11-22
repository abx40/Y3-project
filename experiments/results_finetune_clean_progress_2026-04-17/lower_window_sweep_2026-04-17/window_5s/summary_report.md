# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 5-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.
- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.
- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.
- Calibration gate: pooled real-labelled windows from all calibration sessions with real content, not just `calibration_real_only_01`.
- Gate uncertainty: 500 session bootstrap replicates; candidates pass only if the 95th percentile upper bound stays within <= 0.10 FA/min and <= 0.05 false-alert time ratio.
- Calibration objective: among gate-passing candidates, maximize recall on calibration runs containing fake content, then minimize TTFD, then minimize flicker.
- Calibration does not fall back to a 'best available' threshold when the gate fails.
- Calibration does not optimize F1.
- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.

## Calibration Outcome

- Gate-compliant backends: 5 / 6

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_bootstrap_ucb_gate` | `high` | 0.92 | 0.87 | 3 | 0.0598 | 0.0991 | 0.0133 | 0.0217 | 0.0921 | 12.2240 | 5.1121 | 0.5330 |
| `effort_clip_l14` | `strict_bootstrap_ucb_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5037 |
| `f3net` | `strict_bootstrap_ucb_gate` | `high` | 0.96 | 0.86 | 3 | 0.0399 | 0.0902 | 0.0050 | 0.0119 | 0.0816 | 12.2996 | 5.1429 | 0.5579 |
| `i3d` | `strict_bootstrap_ucb_gate` | `high` | 0.74 | 0.74 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5192 |
| `videomae` | `strict_bootstrap_ucb_gate` | `low` | 0.52 | 0.52 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5199 |

## Non-Compliant Backends

| backend | polarity | closest gate FA/min UCB | closest gate time ratio UCB | closest target recall | val AUROC | test AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `xception_df40` | `high` | 0.2191 | 0.0233 | 0.0502 | 0.5367 | 0.5999 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.5579 | 0.248344 | 0.394737 | 0.120805 | 0.010067 | 12.405869 | 3.692308 |
| 2 | `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5330 | 0.165563 | 0.277008 | 0.322148 | 0.030201 | 14.469179 | 3.457627 |
| 3 | `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.5199 | 0.0 | 0.0 | 0.040268 | 0.003356 | 17.857143 | 24.0 |
| 4 | `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5192 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 5 | `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5037 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
|  | `xception_df40` | `failed_gate` | `` | 0.5367 |  |  |  |  |  |  |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5458 | 0.032787 | 0.062696 | 0.040678 | 0.013559 | 17.532244 | 5.142857 |
| `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5950 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.6456 | 0.154098 | 0.263305 | 0.081356 | 0.016949 | 15.103183 | 3.230769 |
| `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5451 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.4394 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `xception_df40` | `failed_gate` | `` | 0.5999 |  |  |  |  |  |  |
