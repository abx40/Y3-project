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

- Gate-compliant backends: 1 / 1

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f3net` | `strict_bootstrap_ucb_gate` | `high` | 0.92 | 0.92 | 2 | 0.0394 | 0.0900 | 0.0099 | 0.0225 | 0.0446 | 13.0128 | 3.5556 | 0.5985 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.5985 | 0.107843 | 0.19469 | 0.0 | 0.0 | 15.694631 | 2.181818 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.6968 | 0.048544 | 0.091743 | 0.041237 | 0.010309 | 17.331878 | 2.666667 |
