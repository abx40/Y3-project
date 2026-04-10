from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import FileResponse, HTMLResponse
import argparse
import csv
import os
import sys
import time
import io
import uuid
import socket
import platform
import subprocess
import threading
from pathlib import Path
import numpy as np
from PIL import Image


# NEW: optional deepfake model imports
import torch
from torchvision import transforms as T
from models.xception_ffpp import Xception as CadeneXception
from models.effort_clip import (
    EFFORT_IMG_SIZE,
    EFFORT_MEAN,
    EFFORT_STD,
    load_effort_clip_l14_model,
)
from models.efficientnet_dfb import (
    EFFNB4_IMG_SIZE,
    EFFNB4_MEAN,
    EFFNB4_STD,
    load_dfb_efficientnet_b4_model,
)
from models.f3net_dfb import (
    F3NET_IMG_SIZE,
    F3NET_MEAN,
    F3NET_STD,
    load_dfb_f3net_model,
)
from models.i3d_dfb import (
    I3D_DEFAULT_CLIP_SIZE,
    I3D_IMG_SIZE,
    I3D_MEAN,
    I3D_STD,
    load_dfb_i3d_model,
)
from models.videomae_dfb import (
    VIDEOMAE_DEFAULT_CLIP_SIZE,
    VIDEOMAE_DEFAULT_MODEL_ID,
    VIDEOMAE_IMG_SIZE,
    VIDEOMAE_MEAN,
    VIDEOMAE_STD,
    load_dfb_videomae_model,
)
import json
# Audio model disabled for video-only mode.
# from models.aasist.AASIST import Model as AASISTModel
from collections import defaultdict, deque

try:
    import cv2
except Exception:
    cv2 = None


# =========================
# CONFIG
# =========================


# Toggle saving of raw and PNG files
SAVE_RAW = os.getenv("SAVE_RAW", "0").strip().lower() in {"1", "true", "yes", "on"}
SAVE_PNG = os.getenv("SAVE_PNG", "0").strip().lower() in {"1", "true", "yes", "on"}

APP_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = APP_DIR / "index.html"


# Toggle deepfake inference
ENABLE_DEEPFAKE = True


# Directory where frames will be stored
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)


# Simple in-memory counter so each frame gets a unique filename
frame_counter = 0


# FPS tracking
fps_last_time = time.time()
fps_count = 0


# =========================
# DEEPFAKE MODEL SETUP
# =========================


MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "unknown"
MODEL_DEVICE = str(DEVICE)
MODEL_STRICT_LOAD = False
MODEL_CHECKPOINT = None
MODEL_FAKE_INDEX = 1
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "xception_df40").strip().lower()
ACTIVE_VIDEO_MODEL = VIDEO_MODEL
AVAILABLE_VIDEO_MODELS = [
    "auto",
    "xception_df40",
    "effort_clip_l14",
    "efficientnet_b4",
    "f3net",
    "i3d",
    "videomae",
]
PRED_HISTORY = defaultdict(lambda: deque(maxlen=15))  # rolling window per user

# Keep a longer, time-stamped history per user for plotting (last ~60s)
TIME_HISTORY = defaultdict(lambda: deque(maxlen=300))  # (timestamp, fake_prob)
HISTORY_WINDOW_SEC = 60.0
# How long to keep a user “active” after last media (seconds)
ACTIVITY_TIMEOUT = float(os.getenv("ACTIVITY_TIMEOUT", "5.0"))
# Audio ingestion/scoring disabled for video-only mode.
# AUDIO_LAST = defaultdict(lambda: 0.0)
# AUDIO_FAKE = defaultdict(lambda: None)
# AUDIO_HISTORY = defaultdict(lambda: deque(maxlen=300))
# AASIST_CKPT = os.getenv("AASIST_CKPT", "./models/aasist/aasist_orig.pth")
# AASIST_CONF = os.getenv("AASIST_CONF", "./models/aasist/AASIST.conf")
# AUDIO_MODEL = None
# AUDIO_CFG = {}
# AUDIO_NB_SAMP = 64600
# AUDIO_SR = 16000

# Track presence events (join/leave) even if no media seen yet
USERS = {}

# Evaluation run logging defaults (overridden by CLI)
RUN_ID = None
SESSION_ID = "default_session"
RUN_SOURCE = "zoom-bot"  # obs | zoom-bot | file
RUN_MODEL_NAME = "auto"
DECISION_THRESHOLD = 0.5
LOG_BASE_DIR = "logs"
LABELS_CSV = None
LABEL_SEGMENTS = []  # list of dict(start, end, gt_label, src, filename)
LABEL_OVERLAP_WARNED_KEYS = set()
FACE_CROP_MODE = os.getenv("FACE_CROP_MODE", "on").lower()  # on | off
FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.25"))
# Guardrails for face-crop stability with DF40.
# - Ignore tiny detections likely to be false positives.
# - Keep enough context around face by enforcing a minimum crop side ratio.
FACE_MIN_BBOX_AREA_RATIO = float(os.getenv("FACE_MIN_BBOX_AREA_RATIO", "0.03"))
FACE_MIN_CROP_SIDE_RATIO = float(os.getenv("FACE_MIN_CROP_SIDE_RATIO", "0.70"))
FACE_CV2_CASCADE = None
FACE_DETECTOR_NAME = "none"

# Runtime evaluation state
RUN_START_MONO = None
RUN_START_WALL = None
RUN_DIR = None
RUN_META_PATH = None
PRED_LOG_PATH = None
PRED_JSONL_PATH = None
AUTOMATION_LOG_PATH = None
FRAME_DECISION_IDX = 0
PARTICIPANT_START_MONO = {}
PRED_LOG_LOCK = threading.Lock()
AUTOMATION_LOCK = threading.Lock()
AUTOMATION_EVENTS = deque(maxlen=200)
AUTOMATION_STATUS = {
    "connected": False,
    "camera_selected": False,
    "video_started": False,
    "last_event": None,
    "last_error": None,
    "recent_events": [],
}


def _skip_user(user_id: str) -> bool:
    return user_id in ("mixed", "unknown")

# Xception (DF40) defaults
XCEPTION_CKPT = os.getenv("XCEPTION_CKPT", "./models/train_on_df40/xception.pth")
XCEPTION_IMG_SIZE = 256
XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD = [0.5, 0.5, 0.5]


def resolve_default_checkpoint(env_name: str, *candidates: str) -> str:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_absolute():
            env_path = APP_DIR / env_path
        return str(env_path.resolve())
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = APP_DIR / candidate_path
        if candidate_path.is_file():
            return str(candidate_path.resolve())
    if not candidates:
        return ""
    fallback = Path(candidates[0])
    if not fallback.is_absolute():
        fallback = APP_DIR / fallback
    return str(fallback.resolve())


EFFORT_CKPT = resolve_default_checkpoint(
    "EFFORT_CKPT",
    "./models/effort/effort_ffpp_clip_l14.pth",
    "./models/train_on_df40/clip_large.pth",
    "./models/train_on_df40/clip.pth",
)
EFFICIENTNET_CKPT = os.getenv("EFFICIENTNET_CKPT", "./models/train_on_df40/efficientnet_b4.pth")
F3NET_CKPT = os.getenv("F3NET_CKPT", "./models/train_on_df40/f3net_best.pth")
I3D_CKPT = os.getenv("I3D_CKPT", "./models/train_on_df40/i3d.pth")
I3D_CLIP_SIZE = int(os.getenv("I3D_CLIP_SIZE", str(I3D_DEFAULT_CLIP_SIZE)))
I3D_INFER_EVERY = int(os.getenv("I3D_INFER_EVERY", "1"))
VIDEOMAE_CKPT = os.getenv("VIDEOMAE_CKPT", "").strip()
VIDEOMAE_MODEL_ID = os.getenv("VIDEOMAE_MODEL_ID", VIDEOMAE_DEFAULT_MODEL_ID)
VIDEOMAE_CLIP_SIZE = int(os.getenv("VIDEOMAE_CLIP_SIZE", str(VIDEOMAE_DEFAULT_CLIP_SIZE)))
VIDEOMAE_INFER_EVERY = int(os.getenv("VIDEOMAE_INFER_EVERY", "1"))

I3D_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, I3D_CLIP_SIZE * 2)))
I3D_FRAME_COUNT = defaultdict(int)
I3D_LAST_SCORE = defaultdict(lambda: None)
VIDEOMAE_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, VIDEOMAE_CLIP_SIZE * 2)))
VIDEOMAE_FRAME_COUNT = defaultdict(int)
VIDEOMAE_LAST_SCORE = defaultdict(lambda: None)

# Will be set inside load_model depending on which model loads
TRANSFORM = T.Compose([
    T.Resize((XCEPTION_IMG_SIZE, XCEPTION_IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.pt")





def load_aasist_model():
    """Audio loading disabled (video-only mode)."""
    # Audio path intentionally disabled.
    return


def _pad_or_repeat(wav: np.ndarray, target_len: int) -> np.ndarray:
    # Audio helper retained but unused in video-only mode.
    if len(wav) >= target_len:
        return wav[:target_len]
    # repeat-pad
    reps = target_len // len(wav) + 1
    wav = np.tile(wav, reps)
    return wav[:target_len]


def _resample_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    # Audio helper retained but unused in video-only mode.
    if sr_in == sr_out:
        return wav
    if len(wav) < 2:
        return np.zeros(int(sr_out * max(len(wav) / max(sr_in,1), 0.001)), dtype=np.float32)
    import numpy as np
    t_old = np.linspace(0, len(wav) / sr_in, num=len(wav), endpoint=False)
    t_new = np.linspace(0, len(wav) / sr_in, num=int(len(wav) * sr_out / sr_in), endpoint=False)
    return np.interp(t_new, t_old, wav).astype(np.float32)


def init_face_detector():
    global FACE_CV2_CASCADE, FACE_CROP_MODE, FACE_DETECTOR_NAME
    FACE_CV2_CASCADE = None
    FACE_DETECTOR_NAME = "none"

    if FACE_CROP_MODE != "on":
        print("[FACE] Face crop disabled (--face_crop off)")
        return

    if cv2 is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError(f"Failed to load cascade at {cascade_path}")
            FACE_CV2_CASCADE = cascade
            FACE_DETECTOR_NAME = "opencv_haar"
            print("[FACE] Face crop enabled (opencv_haar).")
            return
        except Exception as e:
            print(f"[FACE] OpenCV Haar init failed: {e}")

    print("[FACE] No available face detector; disabling face crop.")
    FACE_CROP_MODE = "off"
    FACE_DETECTOR_NAME = "none"


def rgb_to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unsupported frame shape for grayscale conversion: {frame.shape}")
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    r = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 2].astype(np.float32)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def crop_frame(frame: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1]


def select_inference_region(gray_frame: np.ndarray):
    """
    Returns:
        crop_box, face_detected, num_faces, bbox_area, face_crop_size
    """
    h, w = gray_frame.shape[:2]
    num_faces = 0
    bbox_area = None

    if FACE_CROP_MODE == "on" and FACE_CV2_CASCADE is not None and cv2 is not None:
        # Haar works on grayscale; choose the largest face.
        faces = FACE_CV2_CASCADE.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        num_faces = int(len(faces))
        if num_faces > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            bbox_area = int(fw * fh)
            frame_area = max(1, int(w * h))
            bbox_area_ratio = bbox_area / frame_area

            # Reject tiny detections (often unstable/false positives).
            if bbox_area_ratio >= FACE_MIN_BBOX_AREA_RATIO:
                # Build a context-preserving square crop centered on face.
                cx = x + fw // 2
                cy = y + fh // 2
                side_from_bbox = int(max(fw, fh) * (1.0 + 2.0 * FACE_MARGIN))
                min_side = int(min(w, h) * FACE_MIN_CROP_SIDE_RATIO)
                side = max(side_from_bbox, min_side)
                side = min(side, min(w, h))

                x0 = max(0, cx - side // 2)
                y0 = max(0, cy - side // 2)
                x1 = x0 + side
                y1 = y0 + side
                if x1 > w:
                    x1 = w
                    x0 = max(0, x1 - side)
                if y1 > h:
                    y1 = h
                    y0 = max(0, y1 - side)

                crop_box = (x0, y0, x1, y1)
                face_crop = crop_frame(gray_frame, crop_box)
                if face_crop.size > 0:
                    face_crop_size = f"{x1 - x0}x{y1 - y0}"
                    return crop_box, 1, num_faces, bbox_area, face_crop_size

    # Fallback to center square crop
    side = min(h, w)
    start_y = (h - side) // 2
    start_x = (w - side) // 2
    crop_box = (start_x, start_y, start_x + side, start_y + side)
    center_crop = crop_frame(gray_frame, crop_box)
    face_crop_size = f"{side}x{side}"
    return crop_box, 0, num_faces, bbox_area, face_crop_size


def select_inference_crop(gray_frame: np.ndarray):
    """
    Backward-compatible wrapper for grayscale-first callers.

    Returns:
        crop, face_detected, num_faces, bbox_area, face_crop_size
    """
    crop_box, face_detected, num_faces, bbox_area, face_crop_size = select_inference_region(gray_frame)
    crop = crop_frame(gray_frame, crop_box)
    return crop, face_detected, num_faces, bbox_area, face_crop_size


PREDICTION_COLUMNS = [
    "run_id",
    "session_id",
    "participant_id",
    "source",
    "frame_idx",
    "t_rel",
    "t_recv",
    "t_infer_start",
    "t_infer_end",
    "t_decision",
    "score_raw",
    "score_smoothed",
    "pred_label",
    "threshold",
    "face_detected",
    "num_faces",
    "dropped_frame",
    "bbox_area",
    "face_crop_size",
    "gt_label",
]


def _write_jsonl_line(path: str, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def reset_automation_state():
    global AUTOMATION_EVENTS, AUTOMATION_STATUS
    AUTOMATION_EVENTS = deque(maxlen=200)
    AUTOMATION_STATUS = {
        "connected": False,
        "camera_selected": False,
        "video_started": False,
        "last_event": None,
        "last_error": None,
        "recent_events": [],
    }


def _update_automation_status(event_name: str, detail: dict):
    AUTOMATION_STATUS["last_event"] = event_name
    if event_name in {"connected", "join_success"}:
        AUTOMATION_STATUS["connected"] = True
    elif event_name in {"connection_closed", "join_failed"}:
        AUTOMATION_STATUS["connected"] = False
    if event_name == "camera_selected":
        AUTOMATION_STATUS["camera_selected"] = True
    elif event_name == "camera_missing":
        AUTOMATION_STATUS["camera_selected"] = False
    if event_name == "video_started":
        AUTOMATION_STATUS["video_started"] = True
    elif event_name in {"video_stopped", "video_start_failed"}:
        AUTOMATION_STATUS["video_started"] = False
    if event_name.endswith("failed") or event_name.endswith("error") or event_name in {"camera_missing"}:
        AUTOMATION_STATUS["last_error"] = detail.get("error") or detail.get("reason") or event_name


def log_automation_event(event_name: str, detail: dict):
    if not event_name:
        event_name = "unknown"
    record = {
        "event": event_name,
        "detail": detail,
        "server_time_unix": time.time(),
        "server_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": RUN_ID,
        "session_id": SESSION_ID,
    }
    with AUTOMATION_LOCK:
        AUTOMATION_EVENTS.append(record)
        _update_automation_status(event_name, detail)
        AUTOMATION_STATUS["recent_events"] = list(AUTOMATION_EVENTS)
        if AUTOMATION_LOG_PATH:
            _write_jsonl_line(AUTOMATION_LOG_PATH, record)


def _get_git_commit():
    try:
        repo_dir = Path(__file__).resolve().parent
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            text=True,
        ).strip()
        return commit or None
    except Exception:
        return None


def _parse_gt_label(raw_value):
    val = str(raw_value).strip().lower()
    if val in {"fake", "1", "true"}:
        return 1
    if val in {"real", "0", "false"}:
        return 0
    return None


def load_label_segments(labels_csv_path):
    global LABEL_SEGMENTS, LABEL_OVERLAP_WARNED_KEYS
    LABEL_SEGMENTS = []
    LABEL_OVERLAP_WARNED_KEYS = set()

    if not labels_csv_path:
        return
    if not os.path.isfile(labels_csv_path):
        print(f"[EVAL] Labels file not found: {labels_csv_path}")
        return

    parsed = []
    with open(labels_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"start", "end"}.issubset(set(reader.fieldnames)):
            for row in reader:
                try:
                    start = float(row.get("start"))
                    end = float(row.get("end"))
                except Exception:
                    continue
                gt = _parse_gt_label(row.get("real|fake") or row.get("label"))
                parsed.append({
                    "start": start,
                    "end": end,
                    "gt_label": gt,
                    "src": row.get("src"),
                    "filename": row.get("filename"),
                })
        else:
            f.seek(0)
            rows = csv.reader(f)
            for row in rows:
                if len(row) < 3:
                    continue
                try:
                    start = float(row[0])
                    end = float(row[1])
                except Exception:
                    continue
                gt = _parse_gt_label(row[2])
                parsed.append({
                    "start": start,
                    "end": end,
                    "gt_label": gt,
                    "src": row[3] if len(row) > 3 else None,
                    "filename": row[4] if len(row) > 4 else None,
                })

    LABEL_SEGMENTS = sorted(parsed, key=lambda x: x["start"])
    print(f"[EVAL] Loaded {len(LABEL_SEGMENTS)} label segments from {labels_csv_path}")

    # Warn once if labels contain overlaps. Runtime chooses the tightest segment.
    overlap_count = 0
    for i in range(1, len(LABEL_SEGMENTS)):
        prev = LABEL_SEGMENTS[i - 1]
        cur = LABEL_SEGMENTS[i]
        if cur["start"] < prev["end"]:
            overlap_count += 1
    if overlap_count > 0:
        print(f"[EVAL] WARNING: labels contain {overlap_count} overlapping segment pairs; using tightest match at inference time.")


def gt_label_for_time(t_rel):
    matches = [seg for seg in LABEL_SEGMENTS if seg["start"] <= t_rel < seg["end"]]
    if not matches:
        # Explicit unknown when labels exist but this timestamp is outside segments.
        return "unknown"

    if len(matches) > 1:
        # Prefer the tightest segment if labels overlap.
        key = tuple((m["start"], m["end"], m["gt_label"]) for m in matches)
        if key not in LABEL_OVERLAP_WARNED_KEYS:
            LABEL_OVERLAP_WARNED_KEYS.add(key)
            print(
                f"[EVAL] WARNING: overlapping labels at t_rel={t_rel:.3f}s; "
                f"choosing tightest segment among {len(matches)} matches."
            )
        matches = sorted(matches, key=lambda m: ((m["end"] - m["start"]), m["start"]))
    return matches[0]["gt_label"]


def _resolve_checkpoint_name():
    if MODEL_CHECKPOINT:
        return MODEL_CHECKPOINT
    return None


def init_eval_run():
    global RUN_ID, RUN_START_MONO, RUN_START_WALL
    global RUN_DIR, RUN_META_PATH, PRED_LOG_PATH, PRED_JSONL_PATH
    global AUTOMATION_LOG_PATH, FRAME_DECISION_IDX
    global PARTICIPANT_START_MONO

    if not RUN_ID:
        RUN_ID = str(uuid.uuid4())

    RUN_START_MONO = time.monotonic()
    RUN_START_WALL = time.time()
    FRAME_DECISION_IDX = 0
    PARTICIPANT_START_MONO = {}

    run_dir = Path(LOG_BASE_DIR) / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    RUN_DIR = str(run_dir)
    RUN_META_PATH = str(run_dir / "run_meta.json")
    PRED_LOG_PATH = str(run_dir / "predictions.csv")
    PRED_JSONL_PATH = str(run_dir / "predictions.jsonl")
    AUTOMATION_LOG_PATH = str(run_dir / "web_client.log")
    reset_automation_state()

    run_meta = {
        "run_id": RUN_ID,
        "session_id": SESSION_ID,
        "video_model": VIDEO_MODEL,
        "model_name": MODEL_NAME if RUN_MODEL_NAME == "auto" else RUN_MODEL_NAME,
        "resolved_model_name": MODEL_NAME,
        "checkpoint": _resolve_checkpoint_name(),
        "threshold": DECISION_THRESHOLD,
        "source": RUN_SOURCE,
        "labels_csv": LABELS_CSV,
        "host_machine": socket.gethostname(),
        "platform": platform.platform(),
        "device": MODEL_DEVICE,
        "git_commit": _get_git_commit(),
        "run_start_unix": RUN_START_WALL,
        "t_rel_definition": "monotonic seconds since first accepted frame per participant",
        "face_crop_mode": FACE_CROP_MODE,
        "face_detector": FACE_DETECTOR_NAME,
        "face_margin": FACE_MARGIN,
        "face_min_bbox_area_ratio": FACE_MIN_BBOX_AREA_RATIO,
        "face_min_crop_side_ratio": FACE_MIN_CROP_SIDE_RATIO,
        "predictions_csv": PRED_LOG_PATH,
        "predictions_jsonl": PRED_JSONL_PATH,
        "web_client_log": AUTOMATION_LOG_PATH,
    }

    with open(RUN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with open(PRED_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
    with open(PRED_JSONL_PATH, "w", encoding="utf-8") as f:
        f.write("")
    with open(AUTOMATION_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    print(f"[EVAL] run_id={RUN_ID}")
    print(f"[EVAL] prediction_log={PRED_LOG_PATH}")
    print(f"[EVAL] run_meta={RUN_META_PATH}")


def write_prediction_row(row):
    if not PRED_LOG_PATH or not PRED_JSONL_PATH:
        return
    with PRED_LOG_LOCK:
        with open(PRED_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
            writer.writerow(row)
        _write_jsonl_line(PRED_JSONL_PATH, row)


def load_model():
    """
    Initialise deepfake model from modular key:
      - xception_df40
      - effort_clip_l14
      - efficientnet_b4
      - f3net
      - i3d
      - videomae
      - auto (xception_df40 -> effort_clip_l14 -> efficientnet_b4)
    """
    global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
    global ACTIVE_VIDEO_MODEL

    if not ENABLE_DEEPFAKE:
        print("[DF] Deepfake detection disabled (ENABLE_DEEPFAKE = False)")
        MODEL = None
        MODEL_NAME = "disabled"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = False
        MODEL_CHECKPOINT = None
        MODEL_FAKE_INDEX = 1
        ACTIVE_VIDEO_MODEL = "disabled"
        return

    def _load_xception_df40():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        if not XCEPTION_CKPT or not os.path.isfile(XCEPTION_CKPT):
            raise FileNotFoundError(f"Xception checkpoint not found: {XCEPTION_CKPT}")
        print(f"[DF] Initialising Xception (DF40) on {DEVICE} ...")
        model = CadeneXception(num_classes=2, in_chans=3)
        sd = torch.load(XCEPTION_CKPT, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("module.backbone."):
                k = k[len("module.backbone."):]
            elif k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=True)
        MODEL = model.to(DEVICE)
        MODEL.eval()
        MODEL_NAME = "Xception DF40"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = True
        MODEL_CHECKPOINT = XCEPTION_CKPT
        MODEL_FAKE_INDEX = 1
        TRANSFORM = T.Compose([
            T.Resize((XCEPTION_IMG_SIZE, XCEPTION_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
        ])
        print("[DF] Xception model loaded from checkpoint (strict=True).")

    def _load_effort_clip_l14():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        if not EFFORT_CKPT or not os.path.isfile(EFFORT_CKPT):
            raise FileNotFoundError(f"Effort checkpoint not found: {EFFORT_CKPT}")
        print(f"[DF] Initialising Effort CLIP-L14 on {DEVICE} ...")
        model, report = load_effort_clip_l14_model(EFFORT_CKPT, DEVICE)
        MODEL = model
        MODEL_NAME = "Effort CLIP-L14"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = (
            report.loaded_ratio > 0.98
            and report.head_weight_loaded
            and report.head_bias_loaded
        )
        MODEL_CHECKPOINT = EFFORT_CKPT
        MODEL_FAKE_INDEX = 1
        TRANSFORM = T.Compose([
            T.Resize((EFFORT_IMG_SIZE, EFFORT_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=EFFORT_MEAN, std=EFFORT_STD),
        ])
        print(
            "[DF] Effort checkpoint loaded "
            f"(mode={report.mode}, loaded_ratio={report.loaded_ratio:.3f}, "
            f"missing={report.missing_count}, unexpected={report.unexpected_count}, "
            f"head.weight_in_ckpt={report.head_weight_in_checkpoint}, "
            f"head.weight_loaded={report.head_weight_loaded}, "
            f"head.bias_in_ckpt={report.head_bias_in_checkpoint}, "
            f"head.bias_loaded={report.head_bias_loaded})."
        )

    def _load_efficientnet_b4():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        if not EFFICIENTNET_CKPT or not os.path.isfile(EFFICIENTNET_CKPT):
            raise FileNotFoundError(
                f"EfficientNet checkpoint not found: {EFFICIENTNET_CKPT}. "
                "Download effnb4_best.pth from DeepfakeBench and set --efficientnet_ckpt."
            )
        print(f"[DF] Initialising EfficientNet-B4 (DeepfakeBench) on {DEVICE} ...")
        model, report = load_dfb_efficientnet_b4_model(EFFICIENTNET_CKPT, DEVICE)
        MODEL = model
        MODEL_NAME = "EfficientNet-B4 DF"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = (report.loaded_ratio > 0.99)
        MODEL_CHECKPOINT = EFFICIENTNET_CKPT
        MODEL_FAKE_INDEX = 1
        # Match DeepfakeBench effnb4 config.
        TRANSFORM = T.Compose([
            T.Resize((EFFNB4_IMG_SIZE, EFFNB4_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=EFFNB4_MEAN, std=EFFNB4_STD),
        ])
        print(
            "[DF] EfficientNet-B4 checkpoint loaded "
            f"(loaded_ratio={report.loaded_ratio:.3f}, "
            f"missing={report.missing_count}, unexpected={report.unexpected_count})."
        )

    def _load_f3net():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        if not F3NET_CKPT or not os.path.isfile(F3NET_CKPT):
            raise FileNotFoundError(f"F3Net checkpoint not found: {F3NET_CKPT}")
        print(f"[DF] Initialising F3Net (frequency branch) on {DEVICE} ...")
        model, report = load_dfb_f3net_model(F3NET_CKPT, DEVICE)
        MODEL = model
        MODEL_NAME = "F3Net DF"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = (report.loaded_ratio > 0.99)
        MODEL_CHECKPOINT = F3NET_CKPT
        MODEL_FAKE_INDEX = 1
        TRANSFORM = T.Compose([
            T.Resize((F3NET_IMG_SIZE, F3NET_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=F3NET_MEAN, std=F3NET_STD),
        ])
        print(
            "[DF] F3Net checkpoint loaded "
            f"(loaded_ratio={report.loaded_ratio:.3f}, "
            f"missing={report.missing_count}, unexpected={report.unexpected_count})."
        )

    def _load_i3d():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        if not I3D_CKPT or not os.path.isfile(I3D_CKPT):
            raise FileNotFoundError(f"I3D checkpoint not found: {I3D_CKPT}")
        print(f"[DF] Initialising I3D (temporal) on {DEVICE} ...")
        model, report = load_dfb_i3d_model(I3D_CKPT, DEVICE, clip_size=I3D_CLIP_SIZE)
        MODEL = model
        MODEL_NAME = "I3D DF"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = (report.loaded_ratio > 0.99)
        MODEL_CHECKPOINT = I3D_CKPT
        MODEL_FAKE_INDEX = 1
        TRANSFORM = T.Compose([
            T.Resize((I3D_IMG_SIZE, I3D_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=I3D_MEAN, std=I3D_STD),
        ])
        print(
            "[DF] I3D checkpoint loaded "
            f"(loaded_ratio={report.loaded_ratio:.3f}, "
            f"missing={report.missing_count}, unexpected={report.unexpected_count}, "
            f"clip_size={I3D_CLIP_SIZE}, infer_every={I3D_INFER_EVERY})."
        )

    def _load_videomae():
        global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD, MODEL_CHECKPOINT, MODEL_FAKE_INDEX
        ckpt = None
        if VIDEOMAE_CKPT:
            if not os.path.isfile(VIDEOMAE_CKPT):
                raise FileNotFoundError(f"VideoMAE checkpoint not found: {VIDEOMAE_CKPT}")
            ckpt = VIDEOMAE_CKPT
        print(
            f"[DF] Initialising VideoMAE on {DEVICE} "
            f"(model_id={VIDEOMAE_MODEL_ID}, clip_size={VIDEOMAE_CLIP_SIZE}) ..."
        )
        model, report = load_dfb_videomae_model(VIDEOMAE_MODEL_ID, ckpt, DEVICE)
        MODEL = model
        MODEL_NAME = "VideoMAE DF"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = (report.loaded_ratio > 0.99)
        MODEL_CHECKPOINT = ckpt if ckpt else f"hf:{VIDEOMAE_MODEL_ID}"
        MODEL_FAKE_INDEX = int(report.fake_index)
        TRANSFORM = T.Compose([
            T.Resize((int(report.img_size), int(report.img_size))),
            T.ToTensor(),
            T.Normalize(mean=report.mean, std=report.std),
        ])
        print(
            "[DF] VideoMAE loaded "
            f"(mode={report.mode}, loaded_ratio={report.loaded_ratio:.3f}, "
            f"missing={report.missing_count}, unexpected={report.unexpected_count}, "
            f"fake_index={MODEL_FAKE_INDEX}, infer_every={VIDEOMAE_INFER_EVERY})."
        )

    loaders = {
        "xception_df40": _load_xception_df40,
        "xception": _load_xception_df40,
        "effort_clip_l14": _load_effort_clip_l14,
        "effort": _load_effort_clip_l14,
        "efficientnet_b4": _load_efficientnet_b4,
        "efficientnet": _load_efficientnet_b4,
        "effnet": _load_efficientnet_b4,
        "f3net": _load_f3net,
        "i3d": _load_i3d,
        "videomae": _load_videomae,
    }

    requested = VIDEO_MODEL or "xception_df40"
    if requested == "auto":
        try_order = ["xception_df40", "effort_clip_l14", "efficientnet_b4"]
    else:
        try_order = [requested]

    attempted = []
    for key in try_order:
        loader = loaders.get(key)
        if loader is None:
            continue
        try:
            loader()
            ACTIVE_VIDEO_MODEL = key
            print(f"[DF] Active model key: {key}")
            return
        except Exception as e:
            attempted.append(f"{key}: {e}")
            print(f"[DF] Model load failed for '{key}': {e}")

    MODEL = None
    MODEL_NAME = "failed"
    MODEL_DEVICE = str(DEVICE)
    MODEL_STRICT_LOAD = False
    MODEL_CHECKPOINT = None
    MODEL_FAKE_INDEX = 1
    ACTIVE_VIDEO_MODEL = "failed"
    print("[DF] Failed to initialise any model. Attempts:")
    for item in attempted:
        print(f"  - {item}")



def run_deepfake_inference(user_id: str, frame_crop: np.ndarray):
    """
    Run deepfake detection on a single cropped frame.

    Grayscale crops are expanded to RGB for model input.
    RGB crops preserve source colour through transforms.


    Returns (fake_prob, smoothed_fake_prob) or (None, None) if no model.
    """
    if not ENABLE_DEEPFAKE or MODEL is None:
        return None, None


    try:
        if frame_crop.ndim == 2:
            pil_img = Image.fromarray(frame_crop, mode="L").convert("RGB")
        elif frame_crop.ndim == 3 and frame_crop.shape[2] == 3:
            pil_img = Image.fromarray(frame_crop, mode="RGB")
        else:
            raise ValueError(f"Unsupported inference crop shape: {frame_crop.shape}")
        frame_tensor = TRANSFORM(pil_img)

        # Temporal model path (I3D / VideoMAE): score on a clip window per user.
        if ACTIVE_VIDEO_MODEL in {"i3d", "videomae"}:
            if ACTIVE_VIDEO_MODEL == "i3d":
                frame_count_map = I3D_FRAME_COUNT
                frame_buffer_map = I3D_FRAME_BUFFER
                last_score_map = I3D_LAST_SCORE
                clip_size = I3D_CLIP_SIZE
                infer_every = I3D_INFER_EVERY
            else:
                frame_count_map = VIDEOMAE_FRAME_COUNT
                frame_buffer_map = VIDEOMAE_FRAME_BUFFER
                last_score_map = VIDEOMAE_LAST_SCORE
                clip_size = VIDEOMAE_CLIP_SIZE
                infer_every = VIDEOMAE_INFER_EVERY

            frame_count_map[user_id] += 1
            frame_buffer_map[user_id].append(frame_tensor)

            enough_frames = len(frame_buffer_map[user_id]) >= clip_size
            should_infer = enough_frames and (
                frame_count_map[user_id] % max(1, infer_every) == 0
            )

            if should_infer:
                clip = torch.stack(
                    list(frame_buffer_map[user_id])[-clip_size:],
                    dim=0,
                ).unsqueeze(0).to(DEVICE)  # [B, T, C, H, W]
                with torch.no_grad():
                    out = MODEL(clip)
                if out.ndim == 2 and out.shape[1] >= 2:
                    probs = torch.softmax(out, dim=1)
                    fake_idx = int(max(0, min(MODEL_FAKE_INDEX, out.shape[1] - 1)))
                    fake_prob = probs[0, fake_idx].item()
                elif out.ndim == 2 and out.shape[1] == 1:
                    fake_prob = torch.sigmoid(out[0, 0]).item()
                else:
                    fake_prob = float(out.squeeze().item())
                last_score_map[user_id] = fake_prob
            else:
                fake_prob = last_score_map[user_id]
                if fake_prob is None:
                    # Warm-up phase before first full clip.
                    return None, None
        else:
            x = frame_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = MODEL(x)

            # Handle a couple of common head formats:
            if out.ndim == 2 and out.shape[1] >= 2:
                # logits for [real, fake]
                probs = torch.softmax(out, dim=1)
                fake_idx = int(max(0, min(MODEL_FAKE_INDEX, out.shape[1] - 1)))
                fake_prob = probs[0, fake_idx].item()
            elif out.ndim == 2 and out.shape[1] == 1:
                # single logit, apply sigmoid
                fake_prob = torch.sigmoid(out[0, 0]).item()
            else:
                # already probability-ish
                fake_prob = float(out.squeeze().item())


        # short rolling average (per-user) for log noise smoothing
        hist = PRED_HISTORY[user_id]
        hist.append(fake_prob)
        smooth = sum(hist) / len(hist)


        # longer, timestamped history for plotting (last ~60 seconds)
        now_t = time.time()
        thist = TIME_HISTORY[user_id]
        thist.append((now_t, fake_prob))
        # drop points older than HISTORY_WINDOW_SEC
        while thist and (now_t - thist[0][0] > HISTORY_WINDOW_SEC):
            thist.popleft()


        print(f"[DF] user={user_id} fake_prob={fake_prob:.3f} "
              f"smooth={smooth:.3f} (N={len(hist)})")


        return fake_prob, smooth


    except Exception as e:
        print(f"[DF] Inference error for user {user_id}: {e}")
        return None, None



# =========================
# FASTAPI APP
# =========================


app = FastAPI()

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/frame")
async def receive_frame(request: Request):
    # Metadata headers from C++ bot
    width = request.headers.get("X-Width", "0")
    height = request.headers.get("X-Height", "0")
    instance = request.headers.get("X-Instance-Id", "0")
    user_id = request.headers.get("X-User-Id", "unknown")
    pixel_format = request.headers.get("X-Pixel-Format", "").strip().lower()
    bytes_per_pixel_header = request.headers.get("X-Bytes-Per-Pixel", "").strip()
    try:
        bytes_per_pixel = int(bytes_per_pixel_header) if bytes_per_pixel_header else 0
    except ValueError:
        bytes_per_pixel = 0
    participant_id = user_id if user_id not in ("unknown", "mixed") else "single"
    source = request.headers.get("X-Source", RUN_SOURCE)

    t_recv = time.monotonic()
    if RUN_START_MONO is None:
        # Safety fallback if startup hook did not initialize.
        init_eval_run()
    if participant_id not in PARTICIPANT_START_MONO:
        PARTICIPANT_START_MONO[participant_id] = t_recv

    # Raw frame bytes (legacy Y plane, or RGB/BGR24)
    body = await request.body()
    size = len(body)


    global frame_counter
    frame_counter += 1


    # FPS calculation
    global fps_count, fps_last_time
    fps_count += 1
    now = time.time()
    if now - fps_last_time >= 1.0:
        print(f"[FPS] {fps_count} fps")
        fps_count = 0
        fps_last_time = now


    # Build a safe filename for this frame
    safe_user_id = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in user_id
    )
    filename = (
        f"frame_{frame_counter:06d}_{safe_user_id}_inst{instance}_{width}x{height}.raw"
    )
    filepath = os.path.join(FRAME_DIR, filename)


    fake_prob = None
    smooth_fake_prob = None
    t_infer_start = t_recv
    t_infer_end = t_recv
    dropped_frame = 0
    face_detected = 0
    num_faces = 0
    bbox_area = None
    face_crop_size = None
    # Decode raw frame, preserving colour when available.
    try:
        w = int(width)
        h = int(height)
        arr = np.frombuffer(body, dtype=np.uint8)

        gray_formats = {"y", "gray", "gray8", "mono8", "luma", "y8"}
        rgb_formats = {"rgb", "rgb24", "bgr", "bgr24"}
        if bytes_per_pixel <= 0:
            if pixel_format in rgb_formats:
                bytes_per_pixel = 3
            elif pixel_format in gray_formats:
                bytes_per_pixel = 1

        # If dimensions are missing or zero, try to infer common resolutions or square
        if w <= 0 or h <= 0:
            candidates = [
                (640, 360), (1280, 720), (1920, 1080),
                (256, 256), (224, 224), (299, 299),
                (512, 512), (320, 240), (480, 270),
            ]
            bpp_candidates = [bytes_per_pixel] if bytes_per_pixel in (1, 3) else [1, 3]
            match = None
            inferred_bpp = None
            for bpp in bpp_candidates:
                match = next(
                    ((cw, ch) for cw, ch in candidates if cw * ch * bpp == arr.size),
                    None,
                )
                if match:
                    inferred_bpp = bpp
                    break
            if match:
                w, h = match
                if inferred_bpp:
                    bytes_per_pixel = inferred_bpp
                print(
                    f"[WARN] Dimensions missing; inferring w={w}, h={h}, "
                    f"bpp={bytes_per_pixel} from {arr.size} bytes"
                )
            else:
                bpp_for_square = bytes_per_pixel if bytes_per_pixel in (1, 3) else 1
                if arr.size % bpp_for_square != 0:
                    bpp_for_square = 1
                pixels = arr.size // bpp_for_square
                side = int(np.sqrt(pixels))
                if side * side == pixels:
                    w = h = side
                    bytes_per_pixel = bpp_for_square
                    print(
                        f"[WARN] Dimensions missing; falling back to square "
                        f"{side}x{side}, bpp={bytes_per_pixel}"
                    )
                else:
                    print(f"[WARN] Invalid dimensions (w={width}, h={height}); skipping inference for this frame")
                    raise ValueError("Invalid dimensions")

        expected_gray = w * h
        expected_rgb = w * h * 3
        gray_img = None
        color_img = None

        # Legacy grayscale payload
        if arr.size == expected_gray and (bytes_per_pixel in (0, 1)):
            gray_img = arr.reshape((h, w))
        # RGB/BGR payload
        elif arr.size == expected_rgb and (bytes_per_pixel in (0, 3)):
            rgb_img = arr.reshape((h, w, 3))
            if pixel_format in {"bgr", "bgr24"}:
                if cv2 is not None:
                    color_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                else:
                    color_img = rgb_img[:, :, ::-1].copy()
            else:
                color_img = rgb_img
            gray_img = rgb_to_grayscale(color_img)

        if gray_img is not None:

            # Run face detection on grayscale, then crop the original colour frame if available.
            crop_box, face_detected, num_faces, bbox_area, face_crop_size = select_inference_region(gray_img)
            if color_img is not None:
                arr_crop = crop_frame(color_img, crop_box)
                full_frame = color_img
            else:
                arr_crop = crop_frame(gray_img, crop_box)
                full_frame = gray_img

            # PNG saving
            if SAVE_PNG:
                png_full_filename = filename.replace(".raw", "_full.png")
                png_crop_filename = filename.replace(".raw", "_crop.png")
                png_full_path = os.path.join(FRAME_DIR, png_full_filename)
                png_crop_path = os.path.join(FRAME_DIR, png_crop_filename)
                if full_frame.ndim == 3:
                    Image.fromarray(full_frame, mode="RGB").save(png_full_path)
                    Image.fromarray(arr_crop, mode="RGB").save(png_crop_path)
                else:
                    Image.fromarray(full_frame, mode="L").save(png_full_path)
                    Image.fromarray(arr_crop, mode="L").save(png_crop_path)
                print(f"[PNG] Saved full={png_full_path} crop={png_crop_path}")

            # Deepfake inference
            t_infer_start = time.monotonic()
            fake_prob, smooth_fake_prob = run_deepfake_inference(user_id, arr_crop)
            t_infer_end = time.monotonic()
        else:
            print(
                f"[WARN] Size mismatch: expected {expected_gray} (gray) or "
                f"{expected_rgb} (rgb24), got {arr.size}"
            )
            dropped_frame = 1
    except Exception as e:
        print(f"[ERR] Frame processing failed: {e}")
        dropped_frame = 1
        t_infer_end = time.monotonic()


    # Save raw grayscale bytes
    if SAVE_RAW:
        with open(filepath, "wb") as f:
            f.write(body)


    saved_desc = []
    if SAVE_RAW:
        saved_desc.append("RAW")
    if SAVE_PNG:
        saved_desc.append("PNG")


    print(
        f"[FRAME] user={user_id} inst={instance} {width}x{height} "
        f"bytes={size} -> saved: {', '.join(saved_desc) if saved_desc else 'nothing'}"
    )


    # Response now includes optional probabilities
    t_decision = time.monotonic()
    t_rel_zero = PARTICIPANT_START_MONO.get(participant_id)
    if t_rel_zero is None:
        t_rel_zero = RUN_START_MONO if RUN_START_MONO is not None else t_decision
    t_rel = t_decision - t_rel_zero
    score_for_label = smooth_fake_prob if smooth_fake_prob is not None else fake_prob
    pred_label = int(score_for_label >= DECISION_THRESHOLD) if score_for_label is not None else None
    gt_label = gt_label_for_time(t_rel) if LABEL_SEGMENTS else None

    global FRAME_DECISION_IDX
    FRAME_DECISION_IDX += 1
    write_prediction_row({
        "run_id": RUN_ID,
        "session_id": SESSION_ID,
        "participant_id": participant_id,
        "source": source,
        "frame_idx": FRAME_DECISION_IDX,
        "t_rel": round(t_rel, 6),
        "t_recv": round(t_recv, 6),
        "t_infer_start": round(t_infer_start, 6),
        "t_infer_end": round(t_infer_end, 6),
        "t_decision": round(t_decision, 6),
        "score_raw": fake_prob,
        "score_smoothed": smooth_fake_prob,
        "pred_label": pred_label,
        "threshold": DECISION_THRESHOLD,
        "face_detected": face_detected,
        "num_faces": num_faces,
        "dropped_frame": dropped_frame,
        "bbox_area": bbox_area,
        "face_crop_size": face_crop_size,
        "gt_label": gt_label,
    })

    return {
        "status": "ok",
        "received": size,
        "fake_prob": fake_prob,
        "smooth_fake_prob": smooth_fake_prob,
    }


@app.get("/data/{user_id}")
async def get_user_data(user_id: str, window: int = 60):
    """
    Get detection data for a specific user in JSON format (video-only).
    """
    hist = list(TIME_HISTORY.get(user_id, []))

    if not hist:
        return {"raw": [], "smoothed": []}

    # Filter by time window
    now_t = time.time()
    cutoff_time = now_t - window
    filtered = [(t, p) for (t, p) in hist if t >= cutoff_time]

    # Video raw
    raw_data = [
        {"timestamp": int(t * 1000), "probability": round(p, 4)}
        for (t, p) in filtered
    ]

    # Video smoothed
    smoothed_data = []
    smooth_window = 15
    for i, (t, p) in enumerate(filtered):
        start_idx = max(0, i - smooth_window + 1)
        window_probs = [prob for (_, prob) in filtered[start_idx:i+1]]
        smooth_prob = sum(window_probs) / len(window_probs)
        smoothed_data.append({"timestamp": int(t * 1000), "probability": round(smooth_prob, 4)})

    return {
        "raw": raw_data,
        "smoothed": smoothed_data,
    }


@app.get("/users")
async def get_active_users():
    """
    Get list of all users with recent video activity OR presence pings.
    """
    now_t = time.time()
    user_map = {}

    # Seed with presence data (users who joined even if no media yet)
    for user_id, meta in USERS.items():
        if _skip_user(user_id):
            continue
        user_map[user_id] = {
            "user_id": user_id,
            "last_seen": int(meta.get("last_seen", 0) * 1000),
            "last_probability": 0.0,
            "avg_probability": 0.0,
            "frame_count": 0,
            "last_event": meta.get("last_event"),
        }

    # Merge in video activity
    for user_id, hist in TIME_HISTORY.items():
        if _skip_user(user_id):
            continue
        if not hist:
            continue
        last_time, last_prob = hist[-1]
        if now_t - last_time >= ACTIVITY_TIMEOUT:
            continue  # stale

        avg_prob = sum(p for (_, p) in hist) / len(hist)

        entry = user_map.get(user_id, {
            "user_id": user_id,
            "last_seen": int(last_time * 1000),
            "last_probability": 0.0,
            "avg_probability": 0.0,
            "frame_count": 0,
            "last_event": None,
        })

        entry["last_seen"] = int(max(entry.get("last_seen", 0) / 1000.0, last_time) * 1000)
        entry["last_probability"] = round(last_prob, 4)
        entry["avg_probability"] = round(avg_prob, 4)
        entry["frame_count"] = len(hist)
        user_map[user_id] = entry

    # Return as list (most recent first)
    users = sorted(user_map.values(), key=lambda u: u["last_seen"], reverse=True)
    return {"users": users}


@app.get("/plot/{user_id}")
async def plot_user_history(user_id: str):
    """
    Return the last ~60s of fake probability for a user as a simple SVG line chart.
    This is for debugging/visualisation, not production UI.
    """
    hist = list(TIME_HISTORY.get(user_id, []))
    if not hist:
        return HTMLResponse(
            content=f"<html><body><h3>No data yet for user {user_id}</h3></body></html>",
            status_code=200,
        )


    # normalise times to 'seconds ago'
    now_t = time.time()
    xs = [now_t - t for (t, _) in hist]  # seconds ago (0 = now)
    ys = [p for (_, p) in hist]


    # simple SVG dimensions
    width, height = 400, 200
    padding = 20


    # avoid division by zero if flat history
    max_x = max(xs) if max(xs) > 0 else 1.0
    min_y, max_y = 0.0, 1.0


    # map to SVG coordinates (x to the right, y down)
    def to_svg_coords(x, y):
        # x: 0 (now) at right, HISTORY_WINDOW_SEC at left
        x_norm = x / max_x
        svg_x = padding + (1 - x_norm) * (width - 2 * padding)
        # y: 0 at bottom, 1 at top
        y_norm = (y - min_y) / (max_y - min_y + 1e-6)
        svg_y = padding + (1 - y_norm) * (height - 2 * padding)
        return svg_x, svg_y


    # build polyline points string
    points = []
    for x, y in zip(xs, ys):
        sx, sy = to_svg_coords(x, y)
        points.append(f"{sx:.1f},{sy:.1f}")
    points_str = " ".join(points)


    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="black" />
      <polyline fill="none" stroke="red" stroke-width="2" points="{points_str}" />
      <text x="{padding}" y="{padding}" font-size="12">user={user_id} (last ~{HISTORY_WINDOW_SEC:.0f}s)</text>
      <text x="{padding}" y="{height - 5}" font-size="10">time → (most recent on the right)</text>
      <text x="{width - 60}" y="{padding + 10}" font-size="10">fake prob</text>
    </svg>
    """


    return HTMLResponse(content=svg, media_type="image/svg+xml")


@app.get("/live/{user_id}")
async def live_user_history(user_id: str):
    """
    Simple HTML wrapper that auto-refreshes the SVG every 1s.
    Visit: http://localhost:8001/live/{user_id}
    """
    html = f"""
    <html>
      <head>
        <meta http-equiv="refresh" content="1">
        <title>Deepfake probability - {user_id}</title>
      </head>
      <body>
        <h2>Deepfake probability (last {HISTORY_WINDOW_SEC:.0f}s) for user {user_id}</h2>
        <img src="/plot/{user_id}" alt="probability plot"/>
      </body>
    </html>
    """
    return HTMLResponse(content=html, media_type="text/html")


@app.get("/")
async def root():
    """
    Serve the dashboard HTML
    """
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@app.get("/health")
async def health():
    return {
        "message": "Deepfake Detection API",
        "status": "ok",
        "endpoints": {
            "POST /frame": "Receive video frame for analysis",
            "GET /data/{user_id}": "Get detection data for user",
            "GET /users": "Get list of active users",
            "GET /plot/{user_id}": "Simple SVG plot (legacy)",
            "GET /live/{user_id}": "Auto-refresh plot (legacy)",
            "GET /info": "Runtime and model configuration",
        },
    }


# =========================
# AUDIO INGEST (DISABLED)
# =========================
# Audio ingest endpoint intentionally commented out for video-only mode.
# @app.post("/audio")
# async def receive_audio(request: Request):
#     return {"status": "disabled", "reason": "audio path is commented out (video-only mode)"}


@app.post("/presence")
async def presence(payload: dict):
    """
    Record user presence (join/leave) even if no media is sent.
    Expected JSON: {"user_id": "...", "event": "join"|"leave"}
    """
    user_id = str(payload.get("user_id", "unknown"))
    event = payload.get("event", "unknown")
    USERS[user_id] = {
        "last_seen": time.time(),
        "last_event": event,
    }
    return {"status": "ok"}


@app.post("/automation/event")
async def automation_event(payload: dict):
    event_name = str(payload.get("event") or payload.get("type") or "unknown")
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        detail = {
            key: value
            for key, value in payload.items()
            if key not in {"event", "type"}
        }
    log_automation_event(event_name, detail)
    return {"status": "ok", "event": event_name}


@app.get("/automation/status")
async def get_automation_status():
    with AUTOMATION_LOCK:
        return dict(AUTOMATION_STATUS)


@app.get("/info")
async def get_info():
    return {
        "model": MODEL_NAME,
        "video_model": VIDEO_MODEL,
        "active_video_model": ACTIVE_VIDEO_MODEL,
        "device": MODEL_DEVICE,
        "model_strict_load": MODEL_STRICT_LOAD,
        "model_fake_index": MODEL_FAKE_INDEX,
        "python_executable": sys.executable,
        "xception_ckpt": XCEPTION_CKPT,
        "effort_ckpt": EFFORT_CKPT,
        "efficientnet_ckpt": EFFICIENTNET_CKPT,
        "f3net_ckpt": F3NET_CKPT,
        "i3d_ckpt": I3D_CKPT,
        "i3d_clip_size": I3D_CLIP_SIZE,
        "i3d_infer_every": I3D_INFER_EVERY,
        "videomae_ckpt": VIDEOMAE_CKPT,
        "videomae_model_id": VIDEOMAE_MODEL_ID,
        "videomae_clip_size": VIDEOMAE_CLIP_SIZE,
        "videomae_infer_every": VIDEOMAE_INFER_EVERY,
        "model_checkpoint": MODEL_CHECKPOINT,
        "available_video_models": AVAILABLE_VIDEO_MODELS,
        "enable_deepfake": ENABLE_DEEPFAKE,
        "save_raw": SAVE_RAW,
        "save_png": SAVE_PNG,
        "run_id": RUN_ID,
        "session_id": SESSION_ID,
        "threshold": DECISION_THRESHOLD,
        "face_crop_mode": FACE_CROP_MODE,
        "face_detector": FACE_DETECTOR_NAME,
        "face_margin": FACE_MARGIN,
        "face_min_bbox_area_ratio": FACE_MIN_BBOX_AREA_RATIO,
        "face_min_crop_side_ratio": FACE_MIN_CROP_SIDE_RATIO,
    }


@app.on_event("startup")
async def startup_event():
    # Ensure the model is loaded inside the uvicorn worker process
    load_model()
    # load_aasist_model()  # audio path disabled (video-only mode)
    init_face_detector()
    load_label_segments(LABELS_CSV)
    init_eval_run()


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Zoom frame server with evaluation logging")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id; generated if omitted.")
    parser.add_argument("--session_id", type=str, default="default_session", help="Session name, e.g. session2_mix.")
    parser.add_argument("--labels_csv", type=str, default=None, help="Optional labels CSV for GT alignment.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where logs/<run_id>/... are written.")
    parser.add_argument("--model_name", type=str, default="auto", help="Model tag for metadata (e.g. DF40).")
    parser.add_argument(
        "--video_model",
        type=str,
        default=VIDEO_MODEL,
        choices=AVAILABLE_VIDEO_MODELS,
        help="Selectable video model backend.",
    )
    parser.add_argument(
        "--effort_ckpt",
        type=str,
        default=EFFORT_CKPT,
        help="Path to Effort CLIP-L14 checkpoint (.pth).",
    )
    parser.add_argument(
        "--efficientnet_ckpt",
        type=str,
        default=EFFICIENTNET_CKPT,
        help="Path to EfficientNet-B4 deepfake checkpoint (.pth).",
    )
    parser.add_argument(
        "--f3net_ckpt",
        type=str,
        default=F3NET_CKPT,
        help="Path to F3Net deepfake checkpoint (.pth).",
    )
    parser.add_argument(
        "--i3d_ckpt",
        type=str,
        default=I3D_CKPT,
        help="Path to I3D deepfake checkpoint (.pth).",
    )
    parser.add_argument(
        "--i3d_clip_size",
        type=int,
        default=I3D_CLIP_SIZE,
        help="I3D temporal clip size in frames.",
    )
    parser.add_argument(
        "--i3d_infer_every",
        type=int,
        default=I3D_INFER_EVERY,
        help="Run I3D inference every N received frames after warm-up.",
    )
    parser.add_argument(
        "--videomae_ckpt",
        type=str,
        default=VIDEOMAE_CKPT,
        help="Optional path to VideoMAE deepfake checkpoint (.pth). Leave empty to use HF classifier model_id directly.",
    )
    parser.add_argument(
        "--videomae_model_id",
        type=str,
        default=VIDEOMAE_MODEL_ID,
        help="Hugging Face model id for VideoMAE backbone.",
    )
    parser.add_argument(
        "--videomae_clip_size",
        type=int,
        default=VIDEOMAE_CLIP_SIZE,
        help="VideoMAE temporal clip size in frames.",
    )
    parser.add_argument(
        "--videomae_infer_every",
        type=int,
        default=VIDEOMAE_INFER_EVERY,
        help="Run VideoMAE inference every N received frames after warm-up.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for pred_label.")
    parser.add_argument("--source", type=str, default="zoom-bot", choices=["obs", "zoom-bot", "file"], help="Source tag in prediction logs.")
    parser.add_argument("--face_crop", type=str, default="on", choices=["on", "off"], help="Enable face detection crop before inference.")
    parser.add_argument("--face_min_confidence", type=float, default=0.5, help="Deprecated: ignored (detector is OpenCV Haar only).")
    parser.add_argument("--face_margin", type=float, default=FACE_MARGIN, help="Relative margin around detected face bbox when face crop is on.")
    parser.add_argument("--face_min_bbox_area_ratio", type=float, default=FACE_MIN_BBOX_AREA_RATIO, help="Minimum face bbox area ratio (bbox_area/frame_area) to accept detection.")
    parser.add_argument("--face_min_crop_side_ratio", type=float, default=FACE_MIN_CROP_SIDE_RATIO, help="Minimum crop side ratio relative to min(frame_width, frame_height).")
    parser.add_argument("--save_raw", type=str, default=("on" if SAVE_RAW else "off"), choices=["on", "off"], help="Save incoming raw Y frames to frames/.")
    parser.add_argument("--save_png", type=str, default=("on" if SAVE_PNG else "off"), choices=["on", "off"], help="Save full and cropped PNG frames to frames/.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def apply_cli_config(args):
    global RUN_ID, SESSION_ID, LABELS_CSV, LOG_BASE_DIR
    global RUN_MODEL_NAME, DECISION_THRESHOLD, RUN_SOURCE
    global FACE_CROP_MODE, FACE_MARGIN, FACE_MIN_BBOX_AREA_RATIO, FACE_MIN_CROP_SIDE_RATIO
    global VIDEO_MODEL, EFFORT_CKPT, EFFICIENTNET_CKPT, F3NET_CKPT, I3D_CKPT
    global I3D_CLIP_SIZE, I3D_INFER_EVERY
    global VIDEOMAE_CKPT, VIDEOMAE_MODEL_ID, VIDEOMAE_CLIP_SIZE, VIDEOMAE_INFER_EVERY
    global I3D_FRAME_BUFFER, I3D_FRAME_COUNT, I3D_LAST_SCORE
    global VIDEOMAE_FRAME_BUFFER, VIDEOMAE_FRAME_COUNT, VIDEOMAE_LAST_SCORE
    global SAVE_RAW, SAVE_PNG

    RUN_ID = args.run_id
    SESSION_ID = args.session_id
    LABELS_CSV = args.labels_csv
    LOG_BASE_DIR = args.log_dir
    RUN_MODEL_NAME = args.model_name
    VIDEO_MODEL = args.video_model
    EFFORT_CKPT = args.effort_ckpt
    EFFICIENTNET_CKPT = args.efficientnet_ckpt
    F3NET_CKPT = args.f3net_ckpt
    I3D_CKPT = args.i3d_ckpt
    I3D_CLIP_SIZE = max(2, int(args.i3d_clip_size))
    I3D_INFER_EVERY = max(1, int(args.i3d_infer_every))
    VIDEOMAE_CKPT = str(args.videomae_ckpt).strip()
    VIDEOMAE_MODEL_ID = str(args.videomae_model_id).strip()
    VIDEOMAE_CLIP_SIZE = max(2, int(args.videomae_clip_size))
    VIDEOMAE_INFER_EVERY = max(1, int(args.videomae_infer_every))
    I3D_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, I3D_CLIP_SIZE * 2)))
    I3D_FRAME_COUNT = defaultdict(int)
    I3D_LAST_SCORE = defaultdict(lambda: None)
    VIDEOMAE_FRAME_BUFFER = defaultdict(lambda: deque(maxlen=max(64, VIDEOMAE_CLIP_SIZE * 2)))
    VIDEOMAE_FRAME_COUNT = defaultdict(int)
    VIDEOMAE_LAST_SCORE = defaultdict(lambda: None)
    DECISION_THRESHOLD = float(args.threshold)
    RUN_SOURCE = args.source
    FACE_CROP_MODE = args.face_crop
    FACE_MARGIN = max(0.0, float(args.face_margin))
    FACE_MIN_BBOX_AREA_RATIO = min(1.0, max(0.0, float(args.face_min_bbox_area_ratio)))
    FACE_MIN_CROP_SIDE_RATIO = min(1.0, max(0.0, float(args.face_min_crop_side_ratio)))
    SAVE_RAW = (args.save_raw == "on")
    SAVE_PNG = (args.save_png == "on")


if __name__ == "__main__":
    cli_args = parse_cli_args()
    apply_cli_config(cli_args)
    uvicorn.run(app, host=cli_args.host, port=cli_args.port, reload=False)
