# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 15-second window.
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

- Gate-compliant backends: 2 / 2

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_bootstrap_ucb_gate` | `high` | 0.83 | 0.78 | 2 | 0.0394 | 0.0932 | 0.0099 | 0.0233 | 0.1019 | 12.1800 | 2.9104 | 0.5569 |
| `xception_df40` | `strict_bootstrap_ucb_gate` | `high` | 0.95 | 0.85 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0701 | 12.6467 | 1.4545 | 0.5613 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `xception_df40` | `selected` | `strict_bootstrap_ucb_gate` | 0.5613 | 0.107843 | 0.19469 | 0.0 | 0.0 | 15.766571 | 0.727273 |
| 2 | `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5569 | 0.137255 | 0.235294 | 0.122449 | 0.030612 | 15.362655 | 2.352941 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5808 | 0.087379 | 0.156522 | 0.082474 | 0.030928 | 16.588049 | 3.333333 |
| `xception_df40` | `selected` | `strict_bootstrap_ucb_gate` | 0.6337 | 0.223301 | 0.365079 | 0.0 | 0.0 | 13.805049 | 0.521739 |
