# Demo

Three ways to see this system work. The first is a static website and needs nothing at all.

---

## 0. Static site — publish it for free, no server

```bash
python3 demo/build_site.py
open site/index.html
```

Produces a single self-contained `site/index.html` (~190 KB) with the traces and the
offline-vs-live summary inlined. No fetch calls, no dependencies, no build tooling — it
works by double-clicking the file and it works on any static host.

Publish it on GitHub Pages: `.github/workflows/pages.yml` deploys `site/` on every push to
`main` that touches it. One-time setup: **Settings → Pages → Build and deployment → Source:
GitHub Actions**. (Branch-based Pages can only serve `/` or `/docs`, which is why this uses a
workflow.) Cloudflare Pages and Netlify also work — point them at `site/` with no build command.

Regenerate with `python3 demo/build_site.py` whenever the traces or summaries change.

---

## 1. Trace replay — 30 seconds, no weights, no credentials

```bash
python3 demo/replay_trace.py
# then open http://127.0.0.1:8100/
```

Replays a recorded per-second detector trace into a live dashboard at capture speed:
detector score, ground-truth bands, alert count and a running AUROC, plus the
offline-vs-live comparison across all six backends.

Standard library only — no pip install, no GPU, no Zoom account, works offline.
Traces come from `experiments/results_diagnostics/traces/` and are the real logged output of
the evaluation runs. `--list` shows what is available; `--port` changes the port.

---

## 2. Live pipeline on a local video — no Zoom session needed

The Zoom Video SDK bot is the only component that needs credentials. `feed_video.py`
replaces it: it decodes a local file with ffmpeg and POSTs frames to `/frame` with the
same headers the C++ bot sends, so the frame server, model and dashboard run unmodified.

```bash
demo/run_demo.sh --video "../test vids/1.mov"
```

That starts the frame server, waits for the model, opens the dashboard and feeds the
video through real inference. Or drive the two halves yourself:

```bash
# terminal 1 — frame server
cd zoom-frame-server && ./venv/bin/python frame_server.py --video_model videomae

# terminal 2 — feed a video in
python3 demo/feed_video.py --video "../test vids/1.mov" --fps 5
# dashboard: http://localhost:8001/
```

Verified working end to end on CPU: 30 frames POSTed, scores returned, e.g.
`[DF] user=demo_participant fake_prob=0.418 smooth=0.457 (N=15)`.

Notes:
- **`videomae` is the default backend because it needs no local checkpoint** — its weights
  pull from Hugging Face (`Ammar2k/videomae-base-finetuned-deepfake-subset`) on first run.
- The other five backends need `models/train_on_df40/*.pth`, which are **not** in this
  repository. Starting with one of those reports `model: failed` and the run
  will warn you; the scores would be meaningless. Use mode 1 for those.
- VideoMAE buffers 16 frames before its first score, so allow a few seconds of feed
  before the dashboard populates. CPU inference is slower than the requested FPS, so the
  feed paces itself down — that is expected, not an error.
- `--webcam` uses the camera. `--no-realtime` decodes as fast as possible.
  `ffmpeg` must be on PATH.
- VideoMAE is the **weakest** of the six backends (0.438 live AUROC). It is here because it
  runs anywhere, not because it is the result. The table below is the result.

---

## 0b. Browser inference — `site/live.html`

Runs detectors **client-side** with ONNX Runtime Web 1.30, on WebGPU when available and WASM
otherwise. Frames never leave the page. Model files come from Hugging Face, which serves them
with `access-control-allow-origin: *`, so a static host costs nothing even for the weights.

What the page does:

- **Model selector.** The project's fine-tuned backends (F3Net, Xception, EfficientNet-B4), listed from
  a `manifest.json` produced by `export_onnx.py`, plus a public third-party ViT for comparison.
  One model is loaded at a time; switching releases the previous session.
- **Precision per model.** fp32 / fp16 / int8 for the project backends; q4f16 / int8 / fp16 / fp32 for
  the ViT. **int8 always runs on WASM**: int8 QDQ graphs return garbage on WebGPU in ORT Web 1.30
  (measured, e.g. logits of 4.6e6).
- **Probe check.** After loading, the page runs a fixed input (`0.8·sin(0.001·i)`) and compares
  the logits with the same file run under onnxruntime CPU at export time. Green means this browser
  runs the file correctly; red names the problem (e.g. fp16 drift on some WebGPU devices).
  Quantisation loss is reported separately, from held-out frames, in the model description.
- **Face crop** mirrors `select_inference_region()` in `frame_server.py`: faces under 3% of the frame
  are ignored, otherwise a square crop with a 25% margin and at least 70% of the short side.
  The detector is different (MediaPipe FaceLandmarker instead of OpenCV Haar), so crops are close
  to the server's, not identical.
- **10-second verdict.** The verdict uses a 10 s rolling mean, because per-second scores were near
  chance for every backend in the evaluation. The per-frame score is still plotted.
- **Face warp (manipulation toggle).** Textures a snapshot of the visitor's own face onto their live
  landmark mesh in WebGL, with a feathered seam and colour matching. It is a **domain-shift exhibit**:
  none of the detectors were trained on this manipulation, so expect the scores to barely move.
  Safeguards: camera input only (not picked files), no target-identity upload, nothing recorded
  or exportable, the canvas context menu is disabled, the effect switches off when the tab is
  hidden, the snapshot texture is cleared on Stop, and a "SYNTHETIC" label is drawn into the
  displayed pixels (the detector sees the unlabelled frame).

Measured in Chrome on this Mac (Apple Silicon), per frame, untrained weights of the real architectures:

| Backend | fp32 WebGPU | fp16 WebGPU | fp32 WASM | int8 WASM | Sizes fp32 / fp16 / int8 |
|---|---:|---:|---:|---:|---|
| F3Net | 35 ms | 34 ms | 313 ms | 385 ms | 86 / 43 / 22 MB |
| Xception | 27 ms | 18 ms | 327 ms | 362 ms | 83 / 42 / 22 MB |
| EfficientNet-B4 | 34 ms | 35 ms | 170 ms | 213 ms | 70 / 35 / 20 MB |
| ViT-base (third-party) | — | — | — | 843 ms | q4f16 47 MB: 206 ms on WebGPU |

int8 does **not** make WASM faster here; its only benefit is a quarter of the download. Face
tracking adds ~20 ms per frame.

### Exporting the project backends

The checkpoints are not in the repository (they are on the training PC). Copy them to the paths
the frame server expects, then export:

```bash
# zoom-frame-server/models/train_on_df40/{f3net_best,xception,efficientnet_b4}.pth
zoom-frame-server/venv/bin/python -m pip install onnx onnxruntime onnxconverter-common efficientnet_pytorch onnxscript
zoom-frame-server/venv/bin/python demo/export_onnx.py --calib-video "../test vids/1.mov" --out onnx_export
```

`export_onnx.py` refuses a checkpoint if fewer than 99% of tensors load (the server loaders use
`strict=False`, so a partial load would otherwise run silently with random layers). int8 is
calibrated on even frames of the calibration video and every variant is scored against PyTorch on
the odd frames; the numbers go into `onnx_export/manifest.json` and are shown on the page. Effort
CLIP-L14 (~300M parameters) and the temporal models (I3D, VideoMAE) are not exported.

Test the export locally before publishing. Serve the repo root so the page and the models share
an origin (`?models=` is honoured only on localhost):

```bash
python3 -m http.server 8200
# open http://127.0.0.1:8200/site/live.html?models=/onnx_export/
```

Do not commit the ONNX files; `onnx_export/` is gitignored.

Publish to Hugging Face, which the page reads from by default (`DF40_MODELS_BASE` in `live.html`):

```bash
huggingface-cli login
huggingface-cli upload abx40/df40-browser-onnx onnx_export . --repo-type model
```

If the Hugging Face username is not `abx40`, change `DF40_MODELS_BASE` in `demo/live.html` and
re-run `python3 demo/build_site.py`. Until the manifest exists, the page lists the project backends
as "weights not published yet" and the third-party ViT still works.

The **ViT** is [`prithivMLmods/Deepfake-Detection-Exp-02-21-ONNX`](https://huggingface.co/prithivMLmods/Deepfake-Detection-Exp-02-21-ONNX)
— ViT-base/16, 224x224, preprocessing `(x/127.5) - 1`, `id2label {0: Deepfake, 1: Real}`
(**index 0 is fake**; `dima806/deepfake_vs_real_image_detection` uses the opposite polarity). It
is a third-party image detector, not one of the project's backends, and the page says so.

---

## What the numbers say

Six detector backends, starting from DeepfakeBench pretrained checkpoints and fine-tuned on FaceForensics++ and Celeb-DF, were evaluated both offline (on the session
packs directly) and live (through the running Zoom pipeline). Test split, 10-second windows:

| Backend | Frame-level (offline) | 10s window (offline) | 10s window (live) | Drop |
|---|---:|---:|---:|---:|
| **F3Net** | 0.837 | **0.863** | **0.659** | −0.204 |
| Xception-DF40 | 0.747 | 0.795 | 0.609 | −0.186 |
| EfficientNet-B4 | 0.761 | 0.782 | 0.562 | −0.220 |
| Effort CLIP-L14 | 0.582 | 0.439 | 0.571 | +0.133 |
| I3D | 0.547 | 0.557 | 0.551 | −0.006 |
| VideoMAE | 0.482 | 0.517 | 0.438 | −0.080 |

AUROC. Sources: `experiments/results_offline_ft_all_full_2026-04-17/offline_results_summary.csv`
and `experiments/results_roc_10s_scope_summaries_2026-04-20/roc_10s_test_summary.csv`.

**The finding is the gap, not the peak.** F3Net reaches 0.863 AUROC on the session packs
and 0.659 on the same material through the live pipeline — a 0.204 drop. The live path adds
re-encoding, frame drops and resolution changes (face cropping was on in both); the
evaluation measures their combined effect, not each factor separately. The exceptions are
Effort CLIP-L14, which rises live from a below-chance offline score, and I3D, which is
roughly indifferent and weak everywhere. At 1-second
granularity the live signal is near chance for every backend (0.44–0.54), which is why
windowing matters: the replay dashboard shows that raw per-second trace directly.

A detector validated in a lab is not a detector that works in a video call, and this
measures how much is lost across six architectures.
