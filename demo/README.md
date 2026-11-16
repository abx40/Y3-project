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
- **Deepfake** – the same feed with another face blended over yours. Your real
  eyes and mouth show through, so blinking and talking still look natural.
  - **Preset faces** – ten built-in synthetic faces (people who do not exist).
  - **My own face** – a frozen snapshot of your own face, driven by your live
    movement. It looks almost identical to the original, but is still flagged.
  - **Upload photo** – the face from any photo (front-facing works best).
- Under each feed is the fake probability; the chart shows both over time, with
  the same 15-frame smoothing and 70% alert threshold as the frame server.

## How it works (and what it is not)

These are stand-ins, not the models from the main project:

- **Face swap**: MediaPipe Face Landmarker tracks 468 face points. The donor
  face is warped onto them as a textured mesh with WebGL, its colouring is
  matched to the live lighting and skin tone, and it is blended in with soft
  edges. It replaces the face only (not hair or head shape), and it is a mesh
  swap, not a neural deepfake.
- **Detector**: measures fine frame-to-frame flicker (camera sensor noise) on
  the face relative to the background. It learns the camera's normal level
  from the original feed, then flags a face that is far cleaner than that, as
  a pasted face is. It works best on ordinary, slightly noisy webcams; on very
  clean cameras it gets less reliable while the head is moving. It is not a
  general-purpose deepfake detector.

Nothing is uploaded; all processing happens in the browser tab.

## Credits

Preset faces in `faces/` are synthetic images from the
[SFHQ dataset](https://github.com/SelfishGene/SFHQ-dataset) by David Beniaguev.
`faces/faces.json` holds their precomputed face landmarks.
