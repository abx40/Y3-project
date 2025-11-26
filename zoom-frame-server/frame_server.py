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
from torchvision.models import resnet18
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
DEVICE = torch.device("cpu")
PRED_HISTORY = defaultdict(lambda: deque(maxlen=15))  # rolling window per user


# Keep a longer, time-stamped history per user for plotting (last ~60s)
TIME_HISTORY = defaultdict(lambda: deque(maxlen=300))  # (timestamp, fake_prob)
HISTORY_WINDOW_SEC = 60.0


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # adjust mean/std to match whatever you train with later
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.pt")



def load_model():
    """ 
    Initialise a simple baseline deepfake model so the server works
    even if you don't have a custom `deepfake_model.pt`.


    Right now this uses a torchvision ResNet18 with a 2-class head
    (index 0 = real, index 1 = fake) and random weights.
    You can later replace this with your own trained model by
    loading from disk instead of constructing from scratch.
    """
    global MODEL


    if not ENABLE_DEEPFAKE:
        print("[DF] Deepfake detection disabled (ENABLE_DEEPFAKE = False)")
        MODEL = None
        return


    try:
        print(f"[DF] Initialising baseline ResNet18 model on {DEVICE} ...")
        base = resnet18(weights=None)  # random weights for now
        in_features = base.fc.in_features
        # 2-class head: [real, fake]
        base.fc = torch.nn.Linear(in_features, 2)
        MODEL = base.to(DEVICE)
        MODEL.eval()
        print("[DF] Baseline ResNet18 model initialised and set to eval()")
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
        x = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)  # shape: [1,3,224,224]


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
        if arr.size == w * h:
            arr = arr.reshape((h, w))


            # PNG saving
            if SAVE_PNG:
                png_filename = filename.replace(".raw", ".png")
                png_path = os.path.join(FRAME_DIR, png_filename)
                Image.fromarray(arr, mode="L").save(png_path)
                print(f"[PNG] Saved {png_path}")


            # Deepfake inference
            fake_prob, smooth_fake_prob = run_deepfake_inference(user_id, arr)
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
    NEW ENDPOINT: Get list of all users with recent activity
    """
    now_t = time.time()
    active_users = []
    
    for user_id, hist in TIME_HISTORY.items():
        if hist:
            last_time, last_prob = hist[-1]
            # Consider active if last frame within 10 seconds
            if now_t - last_time < 10:
                avg_prob = sum(p for (_, p) in hist) / len(hist)
                active_users.append({
                    "user_id": user_id,
                    "last_seen": int(last_time * 1000),
                    "last_probability": round(last_prob, 4),
                    "avg_probability": round(avg_prob, 4),
                    "frame_count": len(hist)
                })
    
    return {"users": active_users}


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


@app.on_event("startup")
async def startup_event():
    # Ensure the model is loaded inside the uvicorn worker process
    load_model()


if __name__ == "__main__":
    uvicorn.run(
        "frame_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
