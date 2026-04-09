import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ZOOM_ROOT = REPO_ROOT / "zoom-frame-server"


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_frame_server_module():
    if str(ZOOM_ROOT) not in sys.path:
        sys.path.insert(0, str(ZOOM_ROOT))
    spec = importlib.util.spec_from_file_location("frame_server_offline", ZOOM_ROOT / "frame_server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def init_runtime(frame_server, model_key: str):
    frame_server.VIDEO_MODEL = model_key
    frame_server.ACTIVE_VIDEO_MODEL = model_key
    frame_server.PRED_HISTORY = defaultdict(lambda: deque(maxlen=15))
    frame_server.TIME_HISTORY = defaultdict(lambda: deque(maxlen=300))
    frame_server.I3D_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, frame_server.I3D_CLIP_SIZE * 2)))
    frame_server.I3D_FRAME_COUNT = defaultdict(int)
    frame_server.I3D_LAST_SCORE = defaultdict(lambda: None)
    frame_server.VIDEOMAE_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, frame_server.VIDEOMAE_CLIP_SIZE * 2)))
    frame_server.VIDEOMAE_FRAME_COUNT = defaultdict(int)
    frame_server.VIDEOMAE_LAST_SCORE = defaultdict(lambda: None)
    with pushd(ZOOM_ROOT):
        frame_server.init_face_detector()
        frame_server.load_model()
    return frame_server


def preprocess_frame(frame_server, bgr_frame: np.ndarray):
    if bgr_frame.ndim == 2:
        source_frame = bgr_frame
        gray = bgr_frame
    else:
        source_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        gray = frame_server.rgb_to_grayscale(source_frame)
    crop_box, face_detected, num_faces, bbox_area, face_crop_size = frame_server.select_inference_region(gray)
    crop = frame_server.crop_frame(source_frame, crop_box)
    if crop.ndim == 2:
        pil_crop_rgb = Image.fromarray(crop, mode="L").convert("RGB")
    else:
        pil_crop_rgb = Image.fromarray(crop, mode="RGB")
    resized_rgb = pil_crop_rgb
    if hasattr(frame_server, "TRANSFORM") and hasattr(frame_server.TRANSFORM, "transforms"):
        for step in frame_server.TRANSFORM.transforms:
            if step.__class__.__name__ == "Resize":
                size = step.size
                if isinstance(size, int):
                    size = (size, size)
                elif isinstance(size, list):
                    size = tuple(size)
                resized_rgb = pil_crop_rgb.resize((size[1], size[0]) if len(size) == 2 else (size, size))
                break
    return source_frame, crop, resized_rgb, face_detected, num_faces, bbox_area, face_crop_size


def score_video(
    frame_server,
    model_key: str,
    video_path: Path,
    sample_every_s: int = 1,
    sample_seconds: Optional[Sequence[int]] = None,
    sample_output_dir: Optional[Path] = None,
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    duration_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    rows: List[dict] = []
    selected_seconds = set(sample_seconds or [])
    user_id = "offline_probe"

    frame_server.PRED_HISTORY.clear()
    frame_server.TIME_HISTORY.clear()
    frame_server.I3D_FRAME_BUFFER.clear()
    frame_server.I3D_FRAME_COUNT.clear()
    frame_server.I3D_LAST_SCORE.clear()
    frame_server.VIDEOMAE_FRAME_BUFFER.clear()
    frame_server.VIDEOMAE_FRAME_COUNT.clear()
    frame_server.VIDEOMAE_LAST_SCORE.clear()

    for second in range(0, duration_s, sample_every_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(second * 1000))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        source_frame, crop, resized_rgb, face_detected, num_faces, bbox_area, face_crop_size = preprocess_frame(frame_server, frame)
        fake_prob, smooth_prob = frame_server.run_deepfake_inference(user_id, crop)
        rows.append(
            {
                "model": model_key,
                "video_path": str(video_path),
                "second": second,
                "score_raw": fake_prob,
                "score_smoothed": smooth_prob,
                "face_detected": face_detected,
                "num_faces": num_faces,
                "bbox_area": bbox_area,
                "face_crop_size": face_crop_size,
            }
        )
        if sample_output_dir is not None and second in selected_seconds:
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{video_path.stem}__{model_key}__t{second:04d}"
            if source_frame.ndim == 3:
                Image.fromarray(source_frame, mode="RGB").save(sample_output_dir / f"{prefix}__raw_frame.png")
            else:
                Image.fromarray(source_frame, mode="L").save(sample_output_dir / f"{prefix}__raw_frame.png")
            if crop.ndim == 3:
                Image.fromarray(crop, mode="RGB").save(sample_output_dir / f"{prefix}__inference_crop.png")
            else:
                Image.fromarray(crop, mode="L").save(sample_output_dir / f"{prefix}__inference_crop.png")
            resized_rgb.save(sample_output_dir / f"{prefix}__model_input_rgb.png")
    cap.release()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Offline sanity scorer for session mp4 files.")
    parser.add_argument("--models", required=True, help="Comma-separated model keys.")
    parser.add_argument("--videos", required=True, help="Comma-separated absolute video paths.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results_diagnostics" / "offline_probe"))
    parser.add_argument("--sample-seconds", default="", help="Comma-separated integer seconds to export sample frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    frame_server = load_frame_server_module()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    videos = [Path(item.strip()) for item in args.videos.split(",") if item.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_seconds = [int(item.strip()) for item in args.sample_seconds.split(",") if item.strip()]

    all_rows: List[dict] = []
    summary: Dict[str, dict] = {}
    for model_key in models:
        init_runtime(frame_server, model_key)
        model_rows: List[dict] = []
        for video_path in videos:
            model_rows.extend(
                score_video(
                    frame_server,
                    model_key,
                    video_path,
                    sample_every_s=1,
                    sample_seconds=sample_seconds,
                    sample_output_dir=output_dir / "frame_samples",
                )
            )
        all_rows.extend(model_rows)
        valid_scores = [row["score_raw"] for row in model_rows if row["score_raw"] is not None]
        summary[model_key] = {
            "rows": len(model_rows),
            "valid_score_rows": len(valid_scores),
            "mean_score_raw": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
        }

    csv_path = output_dir / "offline_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "video_path",
                "second",
                "score_raw",
                "score_smoothed",
                "face_detected",
                "num_faces",
                "bbox_area",
                "face_crop_size",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (output_dir / "offline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
