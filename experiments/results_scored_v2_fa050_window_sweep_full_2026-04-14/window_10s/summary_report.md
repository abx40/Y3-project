# Benchmark Scoring Report V2 FA0.50 Gate 10s

## Method

- Evaluation unit: 10-second window.
- Base signal: aligned per-second mean of `predictions.csv:score_raw`, then aggregated into 10-second windows. All runs had `score_raw`, so `score_smoothed` fallback was not needed.
- Missing predictions: hold the last observed score for at most 1 second before window aggregation, then mark the window as missing/no-decision.
- Score polarity: determined per backend from calibration labels using AUROC. Invert only when `AUROC(high means fake) < 0.5`.
- False-alert gates: strict <= 0.50 FA/min and <= 0.30 real-time alert ratio on `calibration_real_only_01`; relaxed <= 0.75 FA/min and <= 0.45 real-time alert ratio.
- Calibration selection: Gate on calibration_real_only_01 with FA/min <= 0.50 and FA time ratio <= 0.30 (relaxed: FA/min <= 0.75 and FA time ratio <= 0.45), then maximize recall on calibration_fake_only_01 plus calibration mixed sessions, then minimize TTFD, then maximize F1.
- Validation ranking: false_alerts_per_min asc, time_to_first_detection_s asc, F1 desc

## Winner

- Validation winner: `i3d`
- Validation F1: 0.3162
- Validation false alerts/min: 0.7297
- Validation time-to-first-detection: 13.0947s

## Selected Operating Points

| backend | selected policy | polarity | persistence_windows | t_on | t_off | real-only FA/min | real-only FA time ratio | target recall | target F1 | target TTFD s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | `strict_0.50` | `low` | 2 | 0.70 | 0.70 | 0.5000 | 0.3000 | 0.4213 | 0.4983 | 9.4414 |
| `effort_clip_l14` | `strict_0.50` | `high` | 2 | 0.98 | 0.98 | 0.4000 | 0.2500 | 0.3427 | 0.4604 | 10.4123 |
| `f3net` | `strict_0.50` | `high` | 2 | 0.58 | 0.48 | 0.4000 | 0.2833 | 0.5393 | 0.5766 | 6.6948 |
| `i3d` | `strict_0.50` | `low` | 3 | 0.33 | 0.23 | 0.0000 | 0.0000 | 0.6348 | 0.6384 | 6.4670 |
| `videomae` | `strict_0.50` | `low` | 3 | 0.53 | 0.48 | 0.5000 | 0.3000 | 0.5843 | 0.6172 | 6.8832 |
| `xception_df40` | `strict_0.50` | `low` | 3 | 0.28 | 0.18 | 0.3000 | 0.2500 | 0.5618 | 0.5865 | 7.0276 |

## Validation Ranking

| rank | backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `i3d` | 0.4512 | 0.2434 | 0.3162 | 0.7297 | 0.3041 | 13.0947 | 0.2927 |
| 2 | `xception_df40` | 0.5490 | 0.3684 | 0.4409 | 0.7703 | 0.3108 | 10.6359 | 1.2941 |
| 3 | `videomae` | 0.4924 | 0.4276 | 0.4577 | 0.8108 | 0.4527 | 9.6866 | 1.1818 |
| 4 | `effort_clip_l14` | 0.6444 | 0.3816 | 0.4793 | 0.8108 | 0.2162 | 10.0274 | 2.4667 |
| 5 | `efficientnet_b4` | 0.5294 | 0.4145 | 0.4649 | 1.0946 | 0.3784 | 9.3371 | 2.6218 |
| 6 | `f3net` | 0.5548 | 0.5329 | 0.5436 | 1.2568 | 0.4392 | 7.1879 | 1.7671 |

## Test Results

| backend | precision | recall | F1 | false alerts/min | false alert time ratio | TTFD s | flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_b4` | 0.5500 | 0.5099 | 0.5292 | 1.3691 | 0.4228 | 7.9314 | 2.1000 |
| `effort_clip_l14` | 0.5682 | 0.3311 | 0.4184 | 1.0067 | 0.2550 | 10.6215 | 3.5455 |
| `f3net` | 0.5865 | 0.5166 | 0.5493 | 1.1678 | 0.3691 | 8.0982 | 1.5338 |
| `i3d` | 0.4710 | 0.4305 | 0.4498 | 0.4430 | 0.4899 | 10.5075 | 0.1304 |
| `videomae` | 0.5108 | 0.6291 | 0.5638 | 1.2886 | 0.6107 | 6.1399 | 0.8387 |
| `xception_df40` | 0.6282 | 0.3245 | 0.4279 | 0.4027 | 0.1946 | 12.3090 | 1.2308 |
