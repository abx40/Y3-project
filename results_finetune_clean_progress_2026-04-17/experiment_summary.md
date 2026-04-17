# Fine-Tuning Study Summary

## Scope

- New leak-free experiment root: `C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4`
- Fine-tuned backends: xception_df40, effort_clip_l14, efficientnet_b4, f3net, i3d, videomae.
- Main scoring setting: v4 pooled calibration, fixed 15-second window.
- Baseline comparator: current fixed-backend rebuilt-calibration bundle at 15s.
- Caveat: the baseline comparator uses the current rebuilt-calibration benchmark rather than a fully rerun no-finetune pass on the new training root. Validation/test footage is unchanged, but calibration footage is not identical.

## Training Status

| backend | status | best_val_acc | epochs | batch_size | checkpoint_or_model_dir |
| --- | --- | --- | --- | --- | --- |
| f3net | completed_with_metrics | 0.8022 | 2 | 8 | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\finetune_run_01\f3net_finetuned_best.pth |
| xception_df40 | completed_with_metrics | 0.8320 | 2 | 8 | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\xception_finetune_run_01\xception_df40_finetuned_best.pth |
| efficientnet_b4 | completed_with_metrics | 0.7394 | 2 | 8 | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\efficientnet_finetune_run_01\efficientnet_b4_finetuned_best.pth |
| effort_clip_l14 | completed_with_metrics | 0.8006 | 2 | 1 | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\effort_finetune_run_01\effort_clip_l14_finetuned_best.pth |
| i3d | completed_artifact_only |  |  |  | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\i3d_finetune_run_01\i3d_finetuned_best.pth |
| videomae | completed_artifact_only |  |  |  | C:\deepfake_eval\f3net_clean_experiment_2026-04-15_v4\artifacts\videomae_finetune_run_01\videomae_model |

## Validation

| backend | threshold_free_auroc | f1 | false_alerts_per_min | false_alert_time_ratio | time_to_first_detection_s | calibration_status |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | 0.5985 | 0.1947 | 0.0000 | 0.0000 | 15.6946 | selected |
| xception_df40 | 0.5613 | 0.1947 | 0.0000 | 0.0000 | 15.7666 | selected |
| efficientnet_b4 | 0.5569 | 0.2353 | 0.1224 | 0.0306 | 15.3627 | selected |
| effort_clip_l14 | 0.5329 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | selected |
| videomae | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 17.8571 | selected |
| i3d | 0.4742 | 0.0714 | 0.2449 | 0.0612 | 16.5346 | selected |

## Test

| backend | threshold_free_auroc | f1 | false_alerts_per_min | false_alert_time_ratio | time_to_first_detection_s | calibration_status |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | 0.6968 | 0.0917 | 0.0412 | 0.0103 | 17.3319 | selected |
| xception_df40 | 0.6337 | 0.3651 | 0.0000 | 0.0000 | 13.8050 | selected |
| effort_clip_l14 | 0.6278 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | selected |
| efficientnet_b4 | 0.5808 | 0.1565 | 0.0825 | 0.0309 | 16.5880 | selected |
| i3d | 0.4654 | 0.2000 | 0.2062 | 0.0515 | 16.1570 | selected |
| videomae | 0.4201 | 0.0000 | 0.0000 | 0.0000 | 18.2927 | selected |

## Baseline vs Fine-Tuned

| backend | baseline_test_auroc | finetuned_test_auroc | delta_test_auroc | baseline_test_f1 | finetuned_test_f1 | delta_test_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| f3net | 0.5549 | 0.6968 | 0.1419 | 0.5128 | 0.0917 | -0.4211 |
| xception_df40 | 0.5071 | 0.6337 | 0.1266 |  | 0.3651 |  |
| efficientnet_b4 | 0.4483 | 0.5808 | 0.1325 | 0.0748 | 0.1565 | 0.0818 |
| effort_clip_l14 | 0.5406 | 0.6278 | 0.0872 |  | 0.0000 |  |
| i3d | 0.5281 | 0.4654 | -0.0627 | 0.0000 | 0.2000 | 0.2000 |
| videomae | 0.5729 | 0.4201 | -0.1528 | 0.2051 | 0.0000 | -0.2051 |

## Answers

1. Successfully fine-tuned backends: f3net, xception_df40, efficientnet_b4, effort_clip_l14, i3d, videomae.
2. Fine-tuning improved held-out test AUROC for 4 / 6 backends.
3. Fine-tuning improved held-out test F1 for 2 / 6 backends.
4. Strongest backend after fine-tuning on validation AUROC: `f3net`.
5. Main remaining failure mode is still live-domain mismatch / transport degradation rather than pure model capacity; several backends improved AUROC without translating that cleanly into a strong operating point.

## Per-Backend Notes

- `f3net`: 
- `xception_df40`: Baseline failed calibration gate; fine-tuned model became selectable.
- `efficientnet_b4`: 
- `effort_clip_l14`: Baseline failed calibration gate; fine-tuned model became selectable. AUROC improved, but selected operating point stayed near-inert.
- `i3d`: Fine-tuned checkpoint came from a long run that timed out before metrics.json flush.
- `videomae`: Fine-tuned local HF checkpoint completed live replay but collapsed to random/inert behavior at 15s.
