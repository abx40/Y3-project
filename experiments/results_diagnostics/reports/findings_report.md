# Diagnostic Findings

## Representative Runs

- validation_real_only_01
- validation_fake_only_01
- validation_mixed_01

## Alignment Summary

- `efficientnet_b4`: mean effective AUC 0.4786, mean fake-real score gap -0.0309, mean onset delta 0.068032, mean offset delta 0.000407.
- `effort_clip_l14`: mean effective AUC 0.5062, mean fake-real score gap -0.0248, mean onset delta -0.027174, mean offset delta 0.14907.
- `f3net`: mean effective AUC 0.4981, mean fake-real score gap -0.0053, mean onset delta -0.068741, mean offset delta 0.063062.
- `i3d`: mean effective AUC 0.5080, mean fake-real score gap 0.0020, mean onset delta 0.004671, mean offset delta -0.006733.
- `videomae`: mean effective AUC 0.5142, mean fake-real score gap 0.0460, mean onset delta -0.021005, mean offset delta 0.05372.
- `xception_df40`: mean effective AUC 0.4983, mean fake-real score gap -0.0334, mean onset delta 0.039562, mean offset delta 0.017282.

## Offline Vs Pipeline

- `xception_df40` mixed-run offline AUC 0.4611 vs pipeline 0.4950; offline fake-real gap -0.0455 vs pipeline -0.0334.
- `f3net` mixed-run offline AUC 0.7935 vs pipeline 0.4943; offline fake-real gap 0.3212 vs pipeline -0.0053.
