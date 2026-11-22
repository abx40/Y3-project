# Benchmark Scoring Report V4 NP Calibration

## Method

- Evaluation unit: fixed 15-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into fixed windows. `score_smoothed` is used only when no raw score exists.
- Score polarity: determined per backend from calibration labels only, using AUROC. Scores are then oriented so higher always means more fake-like.
- Calibration search: threshold `0.01..0.99`, hysteresis gaps `{0.00, 0.05, 0.10}`, persistence windows `{1, 2, 3}`.
- Calibration gate: pooled real-labelled windows from all calibration sessions with real content, not just `calibration_real_only_01`.
- Gate uncertainty: 500 session bootstrap replicates; candidates pass only if the 95th percentile upper bound stays within <= 0.10 FA/min and <= 0.05 false-alert time ratio.
- Calibration objective: among gate-passing candidates, maximize recall on calibration runs containing fake content, then minimize TTFD, then minimize flicker.
- Relaxed fallback: if no candidate passes the strict gate, allow <= 0.20 FA/min and <= 0.05 false-alert time ratio under the same bootstrap-UCB rule.
- Calibration does not optimize F1.
- Validation ranking: compliant backends only, ranked by threshold-free AUROC first, then lower FA/min, then lower TTFD, then higher recall.

## Calibration Outcome

- Gate-compliant backends: 6 / 6

## Selected Operating Points

| backend | policy | polarity | t_on | t_off | persistence | gate FA/min | gate FA/min UCB | gate time ratio | gate time ratio UCB | target recall | target TTFD s | target flicker | val AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_bootstrap_ucb_gate` | `high` | 0.45 | 0.45 | 3 | 0.0202 | 0.0590 | 0.0051 | 0.0148 | 0.0854 | 12.5616 | 1.9718 | 0.5144 |
| `effort_clip_l14` | `relaxed_bootstrap_ucb_gate` | `high` | 0.99 | 0.99 | 3 | 0.0809 | 0.1975 | 0.0202 | 0.0494 | 0.1280 | 11.7235 | 1.6000 | 0.5308 |
| `f3net` | `strict_bootstrap_ucb_gate` | `high` | 0.65 | 0.55 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2683 | 9.9136 | 0.2727 | 0.5582 |
| `i3d` | `strict_bootstrap_ucb_gate` | `low` | 0.44 | 0.44 | 2 | 0.0404 | 0.0945 | 0.0152 | 0.0389 | 0.0061 | 13.4860 | 4.0000 | 0.4845 |
| `videomae` | `strict_bootstrap_ucb_gate` | `low` | 0.93 | 0.83 | 2 | 0.0404 | 0.0883 | 0.0101 | 0.0221 | 0.0427 | 13.1033 | 3.5556 | 0.5266 |
| `xception_df40` | `relaxed_bootstrap_ucb_gate` | `high` | 0.99 | 0.99 | 2 | 0.0607 | 0.1030 | 0.0253 | 0.0464 | 0.0061 | 13.5980 | 2.6667 | 0.5579 |

## Validation

| rank | backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.5582 | 0.352941 | 0.45 | 0.408163 | 0.22449 | 11.215881 | 0.896552 |
| 2 | `xception_df40` | `selected` | `relaxed_bootstrap_ucb_gate` | 0.5579 | 0.078431 | 0.144144 | 0.040816 | 0.010204 | 16.363298 | 3.555556 |
| 3 | `effort_clip_l14` | `selected` | `relaxed_bootstrap_ucb_gate` | 0.5308 | 0.215686 | 0.34375 | 0.163265 | 0.040816 | 13.5705 | 1.230769 |
| 4 | `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.5266 | 0.058824 | 0.108108 | 0.081633 | 0.030612 | 16.724071 | 2.666667 |
| 5 | `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.5144 | 0.088235 | 0.146341 | 0.163265 | 0.122449 | 16.040024 | 1.52381 |
| 6 | `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.4845 | 0.0 | 0.0 | 0.0 | 0.0 | 17.857143 | 0.0 |

## Test

| backend | status | policy | AUROC | recall | F1 | FA/min | FA time ratio | TTFD s | flicker |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `selected` | `strict_bootstrap_ucb_gate` | 0.4483 | 0.038835 | 0.074766 | 0.0 | 0.0 | 17.561 | 1.0 |
| `effort_clip_l14` | `selected` | `relaxed_bootstrap_ucb_gate` | 0.5406 | 0.067961 | 0.122807 | 0.164948 | 0.041237 | 17.01178 | 0.727273 |
| `f3net` | `selected` | `strict_bootstrap_ucb_gate` | 0.5549 | 0.38835 | 0.512821 | 0.371134 | 0.134021 | 11.353939 | 0.830189 |
| `i3d` | `selected` | `strict_bootstrap_ucb_gate` | 0.5281 | 0.0 | 0.0 | 0.0 | 0.0 | 18.292707 | 0.0 |
| `videomae` | `selected` | `strict_bootstrap_ucb_gate` | 0.5729 | 0.116505 | 0.205128 | 0.082474 | 0.020619 | 15.954878 | 2.285714 |
| `xception_df40` | `selected` | `relaxed_bootstrap_ucb_gate` | 0.5071 | 0.009709 | 0.018519 | 0.082474 | 0.041237 | 17.533427 | 3.2 |
