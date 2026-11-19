# Real-Time Deepfake Detection for Video Conferencing

An end-to-end system that joins a live Zoom Video SDK session, captures participants' raw video frames,
and scores them with one of six fine-tuned deepfake detectors in real time. It also includes a benchmark
for measuring how well those detectors actually hold up inside a live call, compared with offline tests.

BSc (Hons) Computer Science final-year project, University of Manchester, 2025–2026. Awarded First Class.

**Try it without any setup:** `python3 demo/replay_trace.py` replays recorded detector output from the real
evaluation runs into the live dashboard (standard library only, no weights, no Zoom account).
See [`demo/README.md`](demo/README.md).

---

## Key finding

Detectors that look strong offline lose much of their accuracy once they run inside a live call.

| Backend | Frame-level (offline) | 10 s window (offline) | 10 s window (live) | Change |
|---|---:|---:|---:|---:|
| F3Net | 0.837 | **0.863** | **0.659** | −0.204 |
| Xception | 0.747 | 0.795 | 0.609 | −0.186 |
| EfficientNet-B4 | 0.761 | 0.782 | 0.562 | −0.220 |
| Effort (CLIP ViT-L/14) | 0.582 | 0.439 | 0.571 | +0.133 |
| I3D | 0.547 | 0.557 | 0.551 | −0.006 |
| VideoMAE | 0.482 | 0.517 | 0.438 | −0.080 |

AUROC on the held-out test split. "Offline" scores the session videos directly; "live" scores the same
material after it has passed through the conferencing stack (virtual camera → Zoom → bot → server), with
face cropping on in both. At 1-second granularity every backend is close to chance live (0.44–0.54),
which is why the system scores 10-second windows.

Sources: [`experiments/results_offline_ft_all_full_2026-04-17/offline_results_summary.csv`](experiments/results_offline_ft_all_full_2026-04-17/offline_results_summary.csv)
and [`experiments/results_roc_10s_scope_summaries_2026-04-20/roc_10s_test_summary.csv`](experiments/results_roc_10s_scope_summaries_2026-04-20/roc_10s_test_summary.csv).

## Method

- **Detectors.** Six backends covering spatial, frequency-aware and temporal model families: F3Net,
  Xception, EfficientNet-B4, Effort (CLIP ViT-L/14), I3D and VideoMAE. Each starts from a public
  pretrained checkpoint (mostly DeepfakeBench) and was fine-tuned on this project's benchmark.
- **Benchmark.** 591 unique source clips from FaceForensics++ (C23, six manipulation methods) and
  Celeb-DF, with splits grouped by identity so the same person never appears in both training and test
  data. Fine-tuning used 4,587 training and 637 validation clips.
- **Live evaluation.** Clips were arranged into 19 ten-minute meeting-style sessions (real-only,
  fake-only and mixed) and replayed through OBS Virtual Camera into a live Zoom client. That gave 114
  full-length runs with no failures. Operating points were calibrated on 9 sessions, then locked
  before evaluating 5 validation and 5 test sessions.
- **Metrics.** AUROC plus operational measures: false alerts per minute, F1 and alert flicker.

## Architecture

```
Web client (Vite/React) ──joins──▶ Zoom Video SDK session ◀──joins── Linux bot (C++, Docker)
        ▲                                                                 │ raw I420 frames, HTTP POST
   OBS Virtual Camera                                                     ▼
                                                     Frame server (FastAPI + PyTorch)
Token server (FastAPI) issues session JWTs            face crop → detector → 10 s window score
                                                                          │
                                                                          ▼
                                                             Live dashboard (probability, alerts)
```

## Repository layout

| Path | Contents |
|---|---|
| `zoom-frame-server/` | FastAPI inference server, the six detector wrappers (`models/`) and the dashboard |
| `videosdk-linux-raw-recording-sample/` | C++ bot built on Zoom's Linux SDK sample, extended to stream raw frames; `run_bot.py` launches it in Docker |
| `videosdk-web-sample/` | Vite/React client used to join test sessions |
| `zoom-token-server/` | Issues Zoom Video SDK JWTs |
| `automation/windows/` | Evaluation harness: OBS control, run orchestration, fine-tuning and scoring scripts |
| `experiments/` | Logged results from every evaluation run (summaries, ROC data, per-second traces) |
| `demo/` | Trace replay, local-video pipeline demo and the static site builder |
| `site/` | Static results site, deployed to GitHub Pages |

## Running it

### Demo, no credentials

```bash
python3 demo/replay_trace.py           # trace replay dashboard on http://127.0.0.1:8100
demo/run_demo.sh --video path/to/clip.mp4   # real inference on a local video, no Zoom needed
```

`run_demo.sh` defaults to VideoMAE, whose weights download from Hugging Face. The other five backends
need fine-tuned checkpoints in `zoom-frame-server/models/train_on_df40/`, which are not committed
because of their size.

### Full live stack

Requires Docker, Python 3, Node.js, OBS Studio and Zoom Video SDK credentials
(`SDK_KEY` and `SDK_SECRET` in `zoom-token-server/.env`).

```bash
# 1. Token server (port 8000)
cd zoom-token-server && pip install -r requirements.txt && uvicorn main:app --port 8000

# 2. Frame server and dashboard (port 8001)
cd zoom-frame-server && pip install -r requirements.txt torch torchvision
python3 frame_server.py --video_model f3net

# 3. Web client: set VITE_VIDEO_SDK_* in .env.local, then
cd videosdk-web-sample && npm install && npm start

# 4. Bot
cd videosdk-linux-raw-recording-sample
docker build -t vsdk-1.11.2-on-ubuntu -f Dockerfile-Ubuntu/Dockerfile .
python3 run_bot.py --session test
```

`start_y3_stack.sh` starts all four components (token server, frame server, web client, bot) in one go. The repeatable
evaluation used for the results above is documented in [`automation/windows/README.md`](automation/windows/README.md).

## Limitations

- One monitored participant at a time; faces are cropped per frame but not tracked across frames.
- Live AUROC is well below offline AUROC for every frame-level model (see table), so treat alerts as
  a prompt for human checking, not a verdict.
- Fine-tuned checkpoints are not in the repository.
- `zoom-frame-server/model_inference.py` is a legacy smoke-test stub (untrained ResNet18 head). It is
  not used for any reported result.

## Acknowledgements

Supervised by Prof. Gaven Smith. Detector implementations and pretrained weights come from
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) and the original model authors; datasets are
FaceForensics++ and Celeb-DF, used under their research licences and not redistributed here. The bot and
web client are built on Zoom's Video SDK samples, which are subject to Zoom's SDK terms.
