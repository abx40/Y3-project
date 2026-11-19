This bundle contains the pushed summary outputs from the corrected multi-scope `10s` ROC run.

Source of the local script used to generate these CSVs:
- `C:\Users\weissach\Desktop\Y3\report\current_latex\figures\gen_roc_10s.py`

That report-side path is outside the main git repo on this machine, so the script itself was not committed on this branch. These CSVs were copied into the repo so the scope results are preserved and pushable.

Scopes included:
- `test`
- `heldout` = `validation + test`
- `all` = `calibration + validation + test`

Columns:
- `backend`
- `scope`
- `scope_splits`
- `live_windows`
- `offline_windows`
- `live_polarity`
- `offline_polarity`
- `live_auc`
- `offline_auc`

Backend order is unchanged:
- `f3net`
- `xception_df40`
- `efficientnet_b4`
- `effort_clip_l14`
- `i3d`
- `videomae`
