# Rebuilt Calibration V2 Rescoring

This folder documents the calibration-pack rebuild and the rescoring run that used it.

## What Changed

The original live benchmark used the existing leak-free pack under `C:\deepfake_eval\leakfree_eval`.
That pack had only one `calibration_real_only_01` session, which made the benign-side gate brittle.

I rebuilt calibration only, without touching validation or test, into:

- `C:\deepfake_eval\leakfree_eval_calibration_v2_2026-04-14`

The rebuilt calibration pack contains:

- `4` mixed sessions
- `3` real_only sessions
- `2` fake_only sessions

Validation and test remained the original held-out sessions.

## Leak-Free Guarantees

The rebuilt calibration sessions were drawn only from source clips assigned to the original `calibration` split.
Validation and test were reused unchanged from the original leak-free pack.

Snapshot files copied here:

- `rebuilt_pack_build_summary.md`
- `rebuilt_pack_leakage_report.json`

Key leakage checks from the rebuilt pack:

- validation identity overlap: `0`
- test identity overlap: `0`
- validation source overlap: `0`
- test source overlap: `0`

## Live Rerun Workflow

1. Built the new calibration sessions outside the repo with:
   - `automation/windows/rebuild_calibration_pack_v2.py`
2. Ran only the new calibration sessions through the live pipeline into:
   - `results_calibration_v2_live_2026-04-14`
3. Kept the original validation/test result rows from:
   - `results_rgb_fix_full_2026-04-09/summary_final_clean.csv`
4. Replaced only the calibration rows with the new calibration live-run rows.
5. Wrote the combined ledger here as:
   - `summary_combined.csv`
6. Rescored that combined ledger with the v4 NP-style calibration method into:
   - `results_scored_v4_np_calibration_5s_rebuilt_calibration_v2_2026-04-15`

## Files In This Folder

- `summary_combined.csv`
  - final ledger used for rescoring
- `calibration_live_summary.csv`
  - summary of the 54 calibration-only reruns
- `rebuilt_pack_build_summary.md`
  - snapshot of the external rebuilt pack summary
- `rebuilt_pack_leakage_report.json`
  - snapshot of the external rebuilt pack leakage checks

## Important Code Change Included

- `automation/windows/score_benchmark_v4_np_calibration.py`
  - added a variable-run loader so rescoring works with the rebuilt protocol (`114` runs total) instead of assuming exactly `96`

## Operational Issues Encountered

- OBS initially failed because it was launched ad hoc with the wrong working directory and could not find `locale/en-US.ini`.
- Two reruns failed transiently in browser automation with:
  - `Camera is starting,please wait.`
  Those were rerun individually and completed successfully.
- Space pressure also caused one early `videomae` model-load failure before free space was recovered.

## Final Outcome

Rescored output bundle:

- `results_scored_v4_np_calibration_5s_rebuilt_calibration_v2_2026-04-15`

High-level result:

- `f3net` remained the best operational backend after rebuilt calibration
- `videomae` kept the strongest held-out AUROC, but its selected operating point became much more conservative
- `effort_clip_l14` and `xception_df40` failed the rebuilt strict calibration gate at `5s`

## Not Included In Git

These remain local only because of size:

- raw calibration rerun directory:
  - `results_calibration_v2_live_2026-04-14` (~26 GB)
- rebuilt evaluation pack:
  - `C:\deepfake_eval\leakfree_eval_calibration_v2_2026-04-14`
