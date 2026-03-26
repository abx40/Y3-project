# Windows OBS Virtual Camera Evaluation

## Prerequisites

- Windows machine with OBS Studio installed.
- OBS websocket enabled on `127.0.0.1:4455`.
- Docker Desktop running in Linux container mode.
- Python available, or [zoom-frame-server/venv](C:/Users/weissach/Desktop/Y3/Y3-project/zoom-frame-server/venv) present.
- Node.js available for the Vite web client.
- Browser installed: Edge or Chrome.
- Python websocket client dependency installed in the runner environment:

```powershell
pip install websocket-client
```

- Zoom Video SDK credentials available in [zoom-token-server/.env](C:/Users/weissach/Desktop/Y3/Y3-project/zoom-token-server/.env) as `SDK_KEY` and `SDK_SECRET`.
- Evaluation dataset available locally with:
  - `sessions_index.csv`
  - `sessions/`

The runner auto-discovers the evaluation root and currently finds the expected dataset at `C:\deepfake_eval\leakfree_eval`.

## One-Time OBS Setup

1. Install and open OBS Studio.
2. Enable the built-in OBS websocket server on port `4455`.
3. Allow the runner to use scene `Y3 Eval` and media input `Y3 Eval Media`.
   The runner will create them if they do not already exist.
4. Verify that `Start Virtual Camera` works in OBS.

No per-run clicking is required after that.

## Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\windows\run_eval.ps1 -SmokeTest -Models xception_df40 -DurationLimitSeconds 45
```

## Full Benchmark

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\windows\run_eval.ps1 -Models xception_df40,effort_clip_l14,efficientnet_b4,f3net,i3d,videomae
```

## Resume Failed Runs

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\windows\run_eval.ps1 -ResumeFailed -Models xception_df40,effort_clip_l14,efficientnet_b4,f3net,i3d,videomae
```

## Outputs

Each run writes into [results](C:/Users/weissach/Desktop/Y3/Y3-project/results):

- `results/<run_id>/token_server.log`
- `results/<run_id>/frame_server.log`
- `results/<run_id>/web_client.log`
- `results/<run_id>/bot.log`
- `results/<run_id>/metadata.json`
- `results/<run_id>/run_meta.json`
- `results/<run_id>/predictions.csv`
- `results/<run_id>/predictions.jsonl`
- `results/<run_id>/status.json`
- `results/summary.csv`

## Notes

- The runner launches the browser directly on `/video` and auto-selects a camera whose label contains `OBS Virtual Camera`.
- The browser is launched with media auto-grant flags so camera permission prompts do not block runs.
- Model backends are detected from [zoom-frame-server/frame_server.py](C:/Users/weissach/Desktop/Y3/Y3-project/zoom-frame-server/frame_server.py).
- Docker image `vsdk-1.11.2-on-ubuntu` is built automatically if it is missing, unless `-SkipBuildBotImage` is passed.
