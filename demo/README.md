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
- **Deepfake** – the same feed with a face pasted over yours. It starts with a
  frozen snapshot of your own face, driven by your live head movement.
  - **Capture my face** retakes that snapshot.
  - **Use a photo…** swaps in the face from any photo (front-facing works best).
- Under each feed is the fake probability; the chart shows both over time, with
  the same 15-frame smoothing and 70% alert threshold as the frame server.

## How it works (and what it is not)

These are stand-ins, not the models from the main project:

- **Face swap**: MediaPipe Face Landmarker tracks 468 face points; the donor
  face is warped onto them as a textured mesh with WebGL and blended at the
  edges. It is a classic mesh face swap, not a neural deepfake.
- **Detector**: looks at each feed's pixels and compares frame-to-frame flicker
  on the face with flicker in the background. Real footage has camera sensor
  noise everywhere; a pasted face is unnaturally clean. This catches the swap
  in this demo but is not a general-purpose deepfake detector.

Nothing is uploaded; all processing happens in the browser tab.
