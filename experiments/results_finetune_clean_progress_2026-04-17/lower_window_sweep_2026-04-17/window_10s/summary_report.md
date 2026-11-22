# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 10-second window.
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

- Gate-compliant backends: 6 / 6

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_bootstrap_ucb_gate` | `high` | 0.83 | 0.78 | 3 | 0.0204 | 0.0611 | 0.0068 | 0.0204 | 0.0898 | 12.2558 | 2.5000 | 0.5351 |
| `effort_clip_l14` | `strict_bootstrap_ucb_gate` | `high` | 0.51 | 0.51 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5089 |
| `f3net` | `strict_bootstrap_ucb_gate` | `high` | 0.94 | 0.94 | 2 | 0.0407 | 0.0901 | 0.0136 | 0.0300 | 0.0367 | 12.9280 | 4.6154 | 0.5768 |
| `i3d` | `strict_bootstrap_ucb_gate` | `high` | 0.73 | 0.73 | 2 | 0.0407 | 0.0790 | 0.0068 | 0.0132 | 0.0122 | 13.5115 | 6.0000 | 0.5081 |
| `videomae` | `strict_bootstrap_ucb_gate` | `low` | 0.52 | 0.52 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.7095 | 0.0000 | 0.5313 |
| `xception_df40` | `strict_bootstrap_ucb_gate` | `high` | 0.98 | 0.98 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0204 | 13.4206 | 2.4000 | 0.5537 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.5768 | 0.177632 | 0.301676 | 0.0 | 0.0 | 13.95869 | 4.0 |
| 2 | `xception_df40` | `selected` | `strict_bootstrap_ucb_gate` | 0.5537 | 0.105263 | 0.189349 | 0.040541 | 0.006757 | 15.869393 | 2.117647 |
| 3 | `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5351 | 0.138158 | 0.237288 | 0.121622 | 0.027027 | 14.95725 | 1.44 |
| 4 | `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.5313 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 5 | `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5089 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |
| 6 | `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5081 | 0.032895 | 0.062893 | 0.040541 | 0.013514 | 17.144012 | 3.428571 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5622 | 0.092715 | 0.169697 | 0.0 | 0.0 | 16.578671 | 2.142857 |
| `effort_clip_l14` | `selected` | `strict_bootstrap_ucb_gate` | 0.5714 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.6587 | 0.10596 | 0.190476 | 0.040268 | 0.006711 | 16.2255 | 2.823529 |
| `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5505 | 0.046358 | 0.08805 | 0.040268 | 0.006711 | 17.230817 | 4.5 |
| `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.4375 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `xception_df40` | `selected` | `strict_bootstrap_ucb_gate` | 0.6094 | 0.07947 | 0.145455 | 0.040268 | 0.013423 | 16.500622 | 3.0 |
