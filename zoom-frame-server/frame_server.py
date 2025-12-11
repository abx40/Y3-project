from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import HTMLResponse
import os
import time
import io
import numpy as np
from PIL import Image


# NEW: optional deepfake model imports
import torch
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from models.xception_ffpp import Xception as CadeneXception
import json
from models.aasist.AASIST import Model as AASISTModel
from collections import defaultdict, deque


# =========================
# CONFIG
# =========================


# Toggle saving of raw and PNG files
SAVE_RAW = False
SAVE_PNG = False


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
PRED_HISTORY = defaultdict(lambda: deque(maxlen=15))  # rolling window per user

# Keep a longer, time-stamped history per user for plotting (last ~60s)
TIME_HISTORY = defaultdict(lambda: deque(maxlen=300))  # (timestamp, fake_prob)
HISTORY_WINDOW_SEC = 60.0
# How long to keep a user “active” after last media (seconds)
ACTIVITY_TIMEOUT = 1.0
# Track last audio timestamp per user (seconds)
AUDIO_LAST = defaultdict(lambda: 0.0)
# Track last audio fake probability
AUDIO_FAKE = defaultdict(lambda: None)
# Audio deepfake (AASIST)
AASIST_CKPT = os.getenv("AASIST_CKPT", "./models/aasist/aasist_orig.pth")
AASIST_CONF = os.getenv("AASIST_CONF", "./models/aasist/AASIST.conf")
AUDIO_MODEL = None
AUDIO_CFG = {}
AUDIO_NB_SAMP = 64600
AUDIO_SR = 16000

# Track presence events (join/leave) even if no media seen yet
USERS = {}


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
    """Load AASIST audio anti-spoof model if checkpoint + config are present."""
    global AUDIO_MODEL, AUDIO_CFG, AUDIO_NB_SAMP, AUDIO_SR
    if not os.path.isfile(AASIST_CKPT) or not os.path.isfile(AASIST_CONF):
        print(f"[AUDIO] AASIST assets missing (ckpt={AASIST_CKPT}, conf={AASIST_CONF}); audio scoring disabled")
        AUDIO_MODEL = None
        return
    try:
        cfg = json.load(open(AASIST_CONF, "r"))
        model_cfg = cfg.get('model_config', {})
        AUDIO_CFG = model_cfg
        AUDIO_NB_SAMP = int(model_cfg.get('nb_samp', 64600))
        AUDIO_SR = int(model_cfg.get('sample_rate', 16000)) if 'sample_rate' in model_cfg else 16000
        print(f"[AUDIO] Loading AASIST on {DEVICE} ...")
        model = AASISTModel(model_cfg)
        sd = torch.load(AASIST_CKPT, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[AUDIO] Load warning. Missing: {missing}. Unexpected: {unexpected}.")
        model.eval()
        model.to(DEVICE)
        AUDIO_MODEL = model
        print("[AUDIO] AASIST ready")
    except Exception as e:
        print(f"[AUDIO] Failed to load AASIST: {e}")
        AUDIO_MODEL = None


def _pad_or_repeat(wav: np.ndarray, target_len: int) -> np.ndarray:
    if len(wav) >= target_len:
        return wav[:target_len]
    # repeat-pad
    reps = target_len // len(wav) + 1
    wav = np.tile(wav, reps)
    return wav[:target_len]


def _resample_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    if len(wav) < 2:
        return np.zeros(int(sr_out * max(len(wav) / max(sr_in,1), 0.001)), dtype=np.float32)
    import numpy as np
    t_old = np.linspace(0, len(wav) / sr_in, num=len(wav), endpoint=False)
    t_new = np.linspace(0, len(wav) / sr_in, num=int(len(wav) * sr_out / sr_in), endpoint=False)
    return np.interp(t_new, t_old, wav).astype(np.float32)

def load_model():
    """ 
    Initialise a simple deepfake model.
    Prefer Xception checkpoint (DF40) if present, otherwise fall back to ResNet18 pretrained.
    """
    global MODEL, TRANSFORM, MODEL_NAME, MODEL_DEVICE


    if not ENABLE_DEEPFAKE:
        print("[DF] Deepfake detection disabled (ENABLE_DEEPFAKE = False)")
        MODEL = None
        MODEL_NAME = "disabled"
        MODEL_DEVICE = str(DEVICE)
        return

    # Try Xception (DF40)
    if XCEPTION_CKPT and os.path.isfile(XCEPTION_CKPT):
        try:
            print(f"[DF] Initialising Xception (DF40) on {DEVICE} ...")
            model = CadeneXception(num_classes=2, in_chans=3)
            sd = torch.load(XCEPTION_CKPT, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # strip/rename common prefixes from checkpoints (e.g., module.backbone.)
            cleaned = {}
            for k, v in sd.items():
                orig = k
                if k.startswith("module.backbone."):
                    k = k[len("module.backbone."):]
                elif k.startswith("module."):
                    k = k[len("module."):]
                # map last_linear -> fc to align with our class
                if k.startswith("last_linear."):
                    k = "fc." + k[len("last_linear."):]
                cleaned[k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing or unexpected:
                print(f"[DF] Xception load warning. Missing: {missing}. Unexpected: {unexpected}.")
            MODEL = model.to(DEVICE)
            MODEL.eval()
            MODEL_NAME = "Xception DF40"
            MODEL_DEVICE = str(DEVICE)
            TRANSFORM = T.Compose([
                T.Resize((XCEPTION_IMG_SIZE, XCEPTION_IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
            ])
            print("[DF] Xception model loaded from checkpoint.")
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


    # Raw grayscale frame bytes (Y plane)
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


    # Convert raw Y plane into PNG / tensor once
    try:
        w = int(width)
        h = int(height)
        arr = np.frombuffer(body, dtype=np.uint8)

        # If dimensions are missing or zero, try to infer common resolutions or square
        if w <= 0 or h <= 0:
            candidates = [
                (640, 360), (1280, 720), (1920, 1080),
                (256, 256), (224, 224), (299, 299),
                (512, 512), (320, 240), (480, 270),
            ]
            match = next(((cw, ch) for cw, ch in candidates if cw * ch == arr.size), None)
            if match:
                w, h = match
                print(f"[WARN] Dimensions missing; inferring w={w}, h={h} from {arr.size} bytes")
            else:
                side = int(np.sqrt(arr.size))
                if side * side == arr.size:
                    w = h = side
                    print(f"[WARN] Dimensions missing; falling back to square {side}x{side}")
                else:
                    print(f"[WARN] Invalid dimensions (w={width}, h={height}); skipping inference for this frame")
                    raise ValueError("Invalid dimensions")

        if arr.size == w * h:
            arr = arr.reshape((h, w))


            # Center crop to square then infer
            side = min(h, w)
            start_y = (h - side) // 2
            start_x = (w - side) // 2
            arr_crop = arr[start_y:start_y + side, start_x:start_x + side]

            # PNG saving
            if SAVE_PNG:
                png_filename = filename.replace(".raw", ".png")
                png_path = os.path.join(FRAME_DIR, png_filename)
                Image.fromarray(arr_crop, mode="L").save(png_path)
                print(f"[PNG] Saved {png_path}")

            # Deepfake inference
            fake_prob, smooth_fake_prob = run_deepfake_inference(user_id, arr_crop)
        else:
            print(f"[WARN] Size mismatch: expected {w*h}, got {arr.size}")
    except Exception as e:
        print(f"[ERR] Frame processing failed: {e}")


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
    return {
        "status": "ok",
        "received": size,
        "fake_prob": fake_prob,
        "smooth_fake_prob": smooth_fake_prob,
    }


@app.get("/data/{user_id}")
async def get_user_data(user_id: str, window: int = 60):
    """
    NEW ENDPOINT: Get detection data for a specific user in JSON format
    for the frontend dashboard.
    
    Query params:
        window: time window in seconds (default 60)
    
    Returns:
        {
            "raw": [{"timestamp": ms, "probability": 0.0-1.0}, ...],
            "smoothed": [{"timestamp": ms, "probability": 0.0-1.0}, ...]
        }
    """
    hist = list(TIME_HISTORY.get(user_id, []))
    
    if not hist:
        return {"raw": [], "smoothed": []}
    
    # Filter by time window
    now_t = time.time()
    cutoff_time = now_t - window
    filtered = [(t, p) for (t, p) in hist if t >= cutoff_time]
    
    # Build raw data points
    raw_data = [
        {
            "timestamp": int(t * 1000),  # Convert to milliseconds
            "probability": round(p, 4)
        }
        for (t, p) in filtered
    ]
    
    # Calculate smoothed data (rolling average over last 15 points)
    smoothed_data = []
    smooth_window = 15
    
    for i, (t, p) in enumerate(filtered):
        start_idx = max(0, i - smooth_window + 1)
        window_probs = [prob for (_, prob) in filtered[start_idx:i+1]]
        smooth_prob = sum(window_probs) / len(window_probs)
        
        smoothed_data.append({
            "timestamp": int(t * 1000),
            "probability": round(smooth_prob, 4)
        })
    
    return {
        "raw": raw_data,
        "smoothed": smoothed_data
    }


@app.get("/users")
async def get_active_users():
    """
    NEW ENDPOINT: Get list of all users with recent activity (frames/audio) OR presence pings.
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
            "has_audio": False,
            "last_audio": None,
            "last_event": meta.get("last_event"),
            "last_audio_fake": AUDIO_FAKE.get(user_id),
        }

    # Merge in media activity (frames/audio)
    for user_id, hist in TIME_HISTORY.items():
        if _skip_user(user_id):
            continue
        if not hist:
            continue
        last_time, last_prob = hist[-1]
        if now_t - last_time >= ACTIVITY_TIMEOUT:
            continue  # stale

        avg_prob = sum(p for (_, p) in hist) / len(hist)
        last_audio = AUDIO_LAST.get(user_id, 0.0)
        has_audio = (now_t - last_audio) < ACTIVITY_TIMEOUT if last_audio else False

        entry = user_map.get(user_id, {
            "user_id": user_id,
            "last_seen": int(last_time * 1000),
            "last_probability": 0.0,
            "avg_probability": 0.0,
            "frame_count": 0,
            "has_audio": False,
            "last_audio": None,
            "last_event": None,
            "last_audio_fake": AUDIO_FAKE.get(user_id),
        })

        entry["last_seen"] = int(max(entry.get("last_seen", 0) / 1000.0, last_time) * 1000)
        entry["last_probability"] = round(last_prob, 4)
        entry["avg_probability"] = round(avg_prob, 4)
        entry["frame_count"] = len(hist)
        entry["has_audio"] = has_audio
        entry["last_audio"] = int(last_audio * 1000) if last_audio else None
        entry["last_audio_fake"] = AUDIO_FAKE.get(user_id, entry.get("last_audio_fake"))
        user_map[user_id] = entry

    # Add audio-only users (no frames yet)
    for user_id, last_audio in AUDIO_LAST.items():
        if _skip_user(user_id):
            continue
        if not last_audio:
            continue
        entry = user_map.get(user_id, {
            "user_id": user_id,
            "last_seen": int(last_audio * 1000),
            "last_probability": 0.0,
            "avg_probability": 0.0,
            "frame_count": 0,
            "has_audio": False,
            "last_audio": None,
            "last_event": None,
            "last_audio_fake": AUDIO_FAKE.get(user_id),
        })
        entry["last_seen"] = int(max(entry.get("last_seen", 0) / 1000.0, last_audio) * 1000)
        entry["has_audio"] = (now_t - last_audio) < ACTIVITY_TIMEOUT
        entry["last_audio"] = int(last_audio * 1000)
        entry["last_audio_fake"] = AUDIO_FAKE.get(user_id, entry.get("last_audio_fake"))
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
# AUDIO INGEST
# =========================


@app.post("/audio")
async def receive_audio(request: Request):
    """
    Receive raw PCM audio from the bot.
    Expected headers:
      - X-Sample-Rate
      - X-Channels
      - X-User-Id
    Body is raw PCM bytes (little-endian).
    """
    sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
    channels = int(request.headers.get("X-Channels", "1"))
    user_id = request.headers.get("X-User-Id", "unknown")

    body = await request.body()
    size = len(body)

    fake_prob = None
    if AUDIO_MODEL is not None and size > 0:
        try:
            wav = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1 and len(wav) % channels == 0:
                wav = wav.reshape(-1, channels)[:, 0]
            wav = _resample_linear(wav, sample_rate, AUDIO_SR)
            wav = _pad_or_repeat(wav, AUDIO_NB_SAMP)
            tensor = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                _, out = AUDIO_MODEL(tensor)
                probs = torch.softmax(out, dim=1)
                fake_prob = float(probs[0, 1].item())
        except Exception as e:
            print(f"[AUDIO] inference error for user {user_id}: {e}")
            fake_prob = None

    # No file writing; just log receipt.
    print(f"[AUDIO] user={user_id} {sample_rate}Hz {channels}ch bytes={size} (not saved) fake_prob={fake_prob}")
    AUDIO_LAST[user_id] = time.time()
    AUDIO_FAKE[user_id] = fake_prob

    return {
        "status": "ok",
        "received": size,
        "sample_rate": sample_rate,
        "channels": channels,
        "file": None,
        "fake_prob": fake_prob,
    }


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
        "xception_ckpt": XCEPTION_CKPT,
        "enable_deepfake": ENABLE_DEEPFAKE,
    }


@app.on_event("startup")
async def startup_event():
    # Ensure the model is loaded inside the uvicorn worker process
    load_model()
    load_aasist_model()


if __name__ == "__main__":
    uvicorn.run(
        "frame_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
