from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import HTMLResponse
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
from torchvision.models import resnet18, ResNet18_Weights
from models.xception_ffpp import Xception as CadeneXception
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
FRAME_DECISION_IDX = 0
PARTICIPANT_START_MONO = {}
PRED_LOG_LOCK = threading.Lock()


def _skip_user(user_id: str) -> bool:
    return user_id in ("mixed", "unknown")

# Xception (DF40) defaults
XCEPTION_CKPT = os.getenv("XCEPTION_CKPT", "./models/train_on_df40/xception.pth")
XCEPTION_IMG_SIZE = 256
XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD = [0.5, 0.5, 0.5]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

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


def select_inference_crop(gray_frame: np.ndarray):
    """
    Returns:
        crop, face_detected, num_faces, bbox_area, face_crop_size
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

                face_crop = gray_frame[y0:y1, x0:x1]
                if face_crop.size > 0:
                    face_crop_size = f"{x1 - x0}x{y1 - y0}"
                    return face_crop, 1, num_faces, bbox_area, face_crop_size

    # Fallback to center square crop
    side = min(h, w)
    start_y = (h - side) // 2
    start_x = (w - side) // 2
    center_crop = gray_frame[start_y:start_y + side, start_x:start_x + side]
    face_crop_size = f"{side}x{side}"
    return center_crop, 0, num_faces, bbox_area, face_crop_size


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
    if MODEL_NAME.startswith("Xception"):
        return XCEPTION_CKPT
    if MODEL_NAME.startswith("ResNet18"):
        return "torchvision:ResNet18_Weights.DEFAULT"
    return None


def init_eval_run():
    global RUN_ID, RUN_START_MONO, RUN_START_WALL
    global RUN_DIR, RUN_META_PATH, PRED_LOG_PATH, FRAME_DECISION_IDX
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

    run_meta = {
        "run_id": RUN_ID,
        "session_id": SESSION_ID,
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
    }

    with open(RUN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with open(PRED_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()

    print(f"[EVAL] run_id={RUN_ID}")
    print(f"[EVAL] prediction_log={PRED_LOG_PATH}")
    print(f"[EVAL] run_meta={RUN_META_PATH}")


def write_prediction_row(row):
    if not PRED_LOG_PATH:
        return
    with PRED_LOG_LOCK:
        with open(PRED_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
            writer.writerow(row)


def load_model():
    """ 
    Initialise a simple deepfake model.
    Prefer Xception checkpoint (DF40) if present, otherwise fall back to ResNet18 pretrained.
    """
    global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE, MODEL_STRICT_LOAD


    if not ENABLE_DEEPFAKE:
        print("[DF] Deepfake detection disabled (ENABLE_DEEPFAKE = False)")
        MODEL = None
        MODEL_NAME = "disabled"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = False
        return

    # Try Xception (DF40)
    if XCEPTION_CKPT and os.path.isfile(XCEPTION_CKPT):
        try:
            print(f"[DF] Initialising Xception (DF40) on {DEVICE} ...")
            model = CadeneXception(num_classes=2, in_chans=3)
            sd = torch.load(XCEPTION_CKPT, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Strip common wrappers from checkpoints (e.g., module.backbone.)
            cleaned = {}
            for k, v in sd.items():
                if k.startswith("module.backbone."):
                    k = k[len("module.backbone."):]
                elif k.startswith("module."):
                    k = k[len("module."):]
                cleaned[k] = v
            # For DF40 we want exact loading (source-of-truth for evaluation).
            model.load_state_dict(cleaned, strict=True)
            MODEL = model.to(DEVICE)
            MODEL.eval()
            MODEL_NAME = "Xception DF40"
            MODEL_DEVICE = str(DEVICE)
            MODEL_STRICT_LOAD = True
            TRANSFORM = T.Compose([
                T.Resize((XCEPTION_IMG_SIZE, XCEPTION_IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
            ])
            print("[DF] Xception model loaded from checkpoint (strict=True).")
            return
        except Exception as e:
            print(f"[DF] Failed to load Xception checkpoint '{XCEPTION_CKPT}': {e}. Falling back to ResNet18.")

    # Fallback: ResNet18 pretrained backbone
    try:
        print(f"[DF] Initialising ResNet18 (ImageNet pretrained) on {DEVICE} ...")
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        # 2-class head: [real, fake]
        base.fc = torch.nn.Linear(in_features, 2)
        MODEL = base.to(DEVICE)
        MODEL.eval()
        MODEL_NAME = "ResNet18 ImageNet"
        MODEL_DEVICE = str(DEVICE)
        MODEL_STRICT_LOAD = False
        TRANSFORM = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        print("[DF] ResNet18 model initialised and set to eval()")
    except Exception as e:
        print(f"[DF] Failed to initialise model: {e}")
        MODEL = None



def run_deepfake_inference(user_id: str, y_plane: np.ndarray):
    """
    Run deepfake detection on a single grayscale frame (Y plane).


    Returns (fake_prob, smoothed_fake_prob) or (None, None) if no model.
    """
    if not ENABLE_DEEPFAKE or MODEL is None:
        return None, None


    try:
        # y_plane: HxW uint8
        pil_img = Image.fromarray(y_plane).convert("RGB")
        x = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)


        with torch.no_grad():
            out = MODEL(x)


        # Handle a couple of common head formats:
        if out.ndim == 2 and out.shape[1] == 2:
            # logits for [real, fake]
            probs = torch.softmax(out, dim=1)
            fake_prob = probs[0, 1].item()
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


    # Convert raw frame into grayscale tensor input once.
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

        # Legacy grayscale payload
        if arr.size == expected_gray and (bytes_per_pixel in (0, 1)):
            gray_img = arr.reshape((h, w))
        # RGB/BGR payload
        elif arr.size == expected_rgb and (bytes_per_pixel in (0, 3)):
            rgb_img = arr.reshape((h, w, 3))
            if pixel_format in {"bgr", "bgr24"}:
                if cv2 is not None:
                    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                else:
                    r = rgb_img[:, :, 2].astype(np.float32)
                    g = rgb_img[:, :, 1].astype(np.float32)
                    b = rgb_img[:, :, 0].astype(np.float32)
                    gray_img = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
            else:
                if cv2 is not None:
                    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                else:
                    r = rgb_img[:, :, 0].astype(np.float32)
                    g = rgb_img[:, :, 1].astype(np.float32)
                    b = rgb_img[:, :, 2].astype(np.float32)
                    gray_img = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)

        if gray_img is not None:

            # Face crop (if enabled) with center-crop fallback.
            arr_crop, face_detected, num_faces, bbox_area, face_crop_size = select_inference_crop(gray_img)

            # PNG saving
            if SAVE_PNG:
                png_full_filename = filename.replace(".raw", "_full.png")
                png_crop_filename = filename.replace(".raw", "_crop.png")
                png_full_path = os.path.join(FRAME_DIR, png_full_filename)
                png_crop_path = os.path.join(FRAME_DIR, png_crop_filename)
                Image.fromarray(gray_img, mode="L").save(png_full_path)
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
    # You can serve the dashboard.html file here or redirect to it
    return {"message": "Deepfake Detection API", "endpoints": {
        "POST /frame": "Receive video frame for analysis",
        "GET /data/{user_id}": "Get detection data for user",
        "GET /users": "Get list of active users",
        "GET /plot/{user_id}": "Simple SVG plot (legacy)",
        "GET /live/{user_id}": "Auto-refresh plot (legacy)"
    }}


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


@app.get("/info")
async def get_info():
    return {
        "model": MODEL_NAME,
        "device": MODEL_DEVICE,
        "model_strict_load": MODEL_STRICT_LOAD,
        "python_executable": sys.executable,
        "xception_ckpt": XCEPTION_CKPT,
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
    global SAVE_RAW, SAVE_PNG

    RUN_ID = args.run_id
    SESSION_ID = args.session_id
    LABELS_CSV = args.labels_csv
    LOG_BASE_DIR = args.log_dir
    RUN_MODEL_NAME = args.model_name
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
