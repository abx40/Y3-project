# Clean Fine-Tuning Progress (2026-04-16)

This bundle records the completed portion of the clean fine-tuning study on branch `codex/f3net-finetune-clean-2026-04-15`.

## Benchmark Root

- External benchmark root: `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4`
- Split roles: `train`, `train_val`, `calibration`, `validation`, `test`
- Leakage checks:
  - train vs calibration identities: `0`
  - train vs validation identities: `0`
  - train vs test identities: `0`
  - train vs calibration sources: `0`
  - train vs validation sources: `0`
  - train vs test sources: `0`

Primary benchmark metadata lives outside the repo in:
- `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\build_summary.md`
- `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\experiment_leakage_report.json`
- `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\finetune_manifest.csv`
- `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\sessions_index.csv`

## Scripts Added

- `automation/windows/build_f3net_clean_experiment.py`
- `automation/windows/train_f3net_finetune.py`
- `automation/windows/train_image_backend_finetune.py`
- `automation/windows/train_temporal_backend_finetune.py`

## Completed End-to-End Result

The only backend fully completed through live replay plus v4 scoring on this branch is `f3net`.

Repo-local outputs:
- Live run summary: `results_f3net_ft_clean_live_2026-04-16/summary.csv`
- 5s scoring: `results_scored_f3net_ft_clean_v4_5s_2026-04-16/`
- 15s scoring: `results_scored_f3net_ft_clean_v4_15s_2026-04-16/`

External fine-tuned checkpoint:
- `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\finetune_run_01\f3net_finetuned_best.pth`

## Additional Fine-Tuning Status

Training-only artefacts were produced for:
- `xception_df40`
- `efficientnet_b4`
- `effort_clip_l14`
- `i3d` (partial checkpoint saved, training wrapper timed out before metrics flush)

Training summary is in:
- `results_finetune_clean_progress_2026-04-16/training_summary.csv`

## Invalid / Partial Live Batch

The first attempted live batch for `xception_df40` and `efficientnet_b4` was **not** promoted as valid experiment data.

Path:
- `results_ft_clean_xception_efficientnet_live_2026-04-16/`

Reason:
- repeated frame-server startup failures (`Timed out waiting for frame server at http://127.0.0.1:8001/info`)
- the bundle is useful only as a failure log, not as scored model data

## Interpretation So Far

- `f3net` clearly benefits from clean leak-free fine-tuning in threshold-free live discrimination.
- `xception_df40` and `efficientnet_b4` trained cleanly offline and have usable fine-tuned checkpoints, but they still need a clean live replay batch on this benchmark root.
- `effort_clip_l14` was feasible to fine-tune on the same split, but training quality was poor and needs live confirmation before any claim is made.
- `i3d` is technically trainable with the new temporal trainer, but runtime cost is much higher than the image backends and the first full run did not finish cleanly enough to count as complete.
- `videomae` has not yet been promoted to a full clean fine-tuned live result on this branch.

## Next Recommended Steps

1. Rerun a clean live batch for `xception_df40` and `efficientnet_b4` on the same benchmark root.
2. If one of those improves materially on validation, run held-out test once with weights and calibration frozen.
3. Only extend to `effort_clip_l14`, `i3d`, and `videomae` after the live runtime path is stable again.
