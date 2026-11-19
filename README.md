# Simple demo

A one-page, browser-only demo of the project idea: your webcam is shown twice,
once untouched and once with a face swapped onto it, and a detector scores both
feeds live.

## Run

```bash
python3 -m http.server 8080 --directory demo
```

Then open <http://localhost:8080> in Chrome, Edge or Firefox and allow the camera.
It must be opened via `localhost` (browsers block the camera on `file://` pages).
Internet access is needed on first load to fetch the face tracker.

## Using it

- **Original camera** – the raw feed.
- **Deepfake** – the same feed with another face swapped in.
- **Swap method**
  - **Mesh (instant)** – the default. Warps a still photo of the donor face
    onto yours. Loads instantly but looks like a mask.
  - **AI neural swap** – optional, behind a button. A neural network repaints
    your face as the donor's, keeping your pose, expression and lighting.
    Needs a one-time ~280 MB download (another ~90 MB for "My own face" or an
    uploaded photo; presets don't need it), cached by the browser afterwards.
    It needs WebGPU (recent Chrome, Edge or Safari) for usable speed; without
    it, it falls back to the CPU at well under 1 frame per second.
- **Swap in**
  - **Preset faces** – ten built-in synthetic faces (people who do not exist).
  - **My own face** – a frozen snapshot of your own face.
  - **Upload photo** – the face from any photo (front-facing works best).
- Under each feed is the fake probability; the chart shows both over time, with
  the same 15-frame smoothing and 70% alert threshold as the frame server.

## How it works (and what it is not)

- **Face tracking**: MediaPipe Face Landmarker, 468 points per frame.
- **Mesh swap**: the donor face is warped onto those points as a textured mesh
  with WebGL, colour-matched to the live lighting and blended with soft edges.
- **AI swap**: [InsightFace](https://github.com/deepinsight/insightface)'s
  `inswapper_128` run with onnxruntime-web on WebGPU. Each frame the face is
  cut out and straightened to 128×128 (using eyes, nose and mouth corners), the
  network repaints it with the donor's identity, and the result is pasted back
  with a soft edge. The donor's identity code comes from the ArcFace
  `w600k_r50` recognition model; for the presets it is precomputed in
  `faces/latents.json`. Both models replace the face only, not hair or head shape.
- **Detector**: a stand-in for the project's model. It measures fine
  frame-to-frame flicker (camera sensor noise) on the face relative to the
  background, learns the camera's normal level from the original feed, and
  flags a face that is far cleaner than that. It catches both swaps here, but
  it is not a general-purpose deepfake detector.

Nothing is uploaded; all processing happens in the browser tab.

## AI model files

The model files are too large for the main branch, so they are only published
on the `gh-pages` branch, under `models/`. To rebuild them (for example to run
the AI swap locally):

```bash
pip install onnx onnxruntime numpy pillow
python3 demo/tools/build_models.py
```

This downloads the original models, stores their weights at half size, splits
the swap model into parts under GitHub's 100 MB file limit, and writes them to
`demo/models/` (ignored by git).

## Credits

Preset faces in `faces/` are synthetic images from the
[SFHQ dataset](https://github.com/SelfishGene/SFHQ-dataset) by David Beniaguev.
`faces/faces.json` holds their precomputed face landmarks and
`faces/latents.json` their precomputed identity codes for the AI swap.

The AI swap uses `inswapper_128` and ArcFace `w600k_r50` from InsightFace,
downloaded from FaceFusion's model release. InsightFace releases inswapper for
**non-commercial research use only**.
