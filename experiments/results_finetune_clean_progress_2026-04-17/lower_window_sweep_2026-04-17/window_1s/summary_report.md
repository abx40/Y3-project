# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 1-second window.
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

- Gate-compliant backends: 4 / 6

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_bootstrap_ucb_gate` | `high` | 0.99 | 0.89 | 2 | 0.0399 | 0.0889 | 0.0043 | 0.0111 | 0.0360 | 12.8027 | 9.6970 | 0.5234 |
| `effort_clip_l14` | `strict_bootstrap_ucb_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5027 |
| `i3d` | `strict_bootstrap_ucb_gate` | `high` | 0.74 | 0.74 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5057 |
| `videomae` | `strict_bootstrap_ucb_gate` | `low` | 0.52 | 0.52 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5245 |

## Non-Compliant Backends

| backend | polarity | closest gate FA/min UCB | closest gate time ratio UCB | closest target recall | val AUROC | test AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `f3net` | `high` | 0.3725 | 0.0481 | 0.0393 | 0.5515 | 0.6155 |
| `xception_df40` | `high` | 0.9048 | 0.1226 | 0.1038 | 0.5359 | 0.5884 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.5245 | 0.0 | 0.0 | 0.040053 | 0.006676 | 17.857143 | 12.0 |
| 2 | `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5234 | 0.078562 | 0.141064 | 0.280374 | 0.035381 | 15.282226 | 5.614035 |
| 3 | `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5057 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 4 | `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5027 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
|  | `f3net` | `failed_gate` | `` | 0.5515 |  |  |  |  |  |  |
|  | `xception_df40` | `failed_gate` | `` | 0.5359 |  |  |  |  |  |  |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5345 | 0.02672 | 0.051713 | 0.03992 | 0.006653 | 17.076195 | 9.6 |
| `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5765 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `f3net` | `failed_gate` | `` | 0.6155 |  |  |  |  |  |  |
| `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5410 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.4415 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `xception_df40` | `failed_gate` | `` | 0.5884 |  |  |  |  |  |  |
