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

- Gate-compliant backends: 4 / 6

## Selected Operating Points

| backend | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `high` | 0.66 | 0.56 | 3 | 0.0402 | 0.0906 | 0.0067 | 0.0169 | 0.0534 | 12.8284 | 6.0811 | 0.4796 |
| `f3net` | `high` | 0.91 | 0.86 | 3 | 0.0201 | 0.0608 | 0.0017 | 0.0051 | 0.0452 | 12.9423 | 6.2609 | 0.5258 |
| `i3d` | `low` | 0.49 | 0.49 | 1 | 0.0201 | 0.0665 | 0.0017 | 0.0055 | 0.0062 | 13.6414 | 24.0000 | 0.4940 |
| `videomae` | `low` | 0.99 | 0.89 | 3 | 0.0201 | 0.0507 | 0.0034 | 0.0085 | 0.0287 | 13.2103 | 4.5000 | 0.5350 |

## Non-Compliant Backends

| backend | polarity | closest gate FA/min UCB | closest gate time ratio UCB | closest target recall | val AUROC | test AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `effort_clip_l14` | `high` | 1.2358 | 0.2534 | 0.3655 | 0.5245 | 0.5219 |
| `xception_df40` | `high` | 0.3788 | 0.0914 | 0.0616 | 0.5374 | 0.4954 |

## Validation

| rank | backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `videomae` | `selected` | 0.5350 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 2 | `f3net` | `selected` | 0.5258 | 0.109272 | 0.19469 | 0.161074 | 0.013423 | 15.258321 | 5.189189 |
| 3 | `i3d` | `selected` | 0.4940 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 4 | `efficientnet_b4` | `selected` | 0.4796 | 0.059603 | 0.107784 | 0.161074 | 0.04698 | 16.509429 | 4.875 |
|  | `effort_clip_l14` | `failed_gate` | 0.5245 |  |  |  |  |  |  |
|  | `xception_df40` | `failed_gate` | 0.5374 |  |  |  |  |  |  |

## Test

| backend | status | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | 0.4477 | 0.02623 | 0.050955 | 0.040678 | 0.00339 | 17.724415 | 5.333333 |
| `effort_clip_l14` | `failed_gate` | 0.5219 |  |  |  |  |  |  |
| `f3net` | `selected` | 0.5419 | 0.131148 | 0.231214 | 0.040678 | 0.00339 | 15.649988 | 2.926829 |
| `i3d` | `selected` | 0.5514 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `videomae` | `selected` | 0.5728 | 0.04918 | 0.092879 | 0.081356 | 0.010169 | 17.409037 | 1.333333 |
| `xception_df40` | `failed_gate` | 0.4954 |  |  |  |  |  |  |
