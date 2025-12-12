# 🧠 Deepfake Detection Zoom Bot

A real-time deepfake detection system that uses a **Zoom Video SDK Linux Bot** to capture raw video frames, analyze them with a custom deepfake detection model, and display results on a live web dashboard.

---

## 📋 Overview

This project implements a complete pipeline for deepfake detection in Zoom sessions:

- **Linux Bot** (C++) joins Zoom sessions and extracts raw YUV video frames
- **Frame Analysis Server** (Python FastAPI) performs deepfake detection
- **Token Server** (Python FastAPI) generates Zoom SDK JWT tokens
- **Web Dashboard** displays real-time detection results
- **Web Client** (Vite/React) joins sessions for testing

---

## 📁 Project Structure

```
repo/
│
├── videosdk-linux-raw-recording-sample/  # Linux bot (C++ raw frame extraction)
│   ├── CMakeLists.txt
│   ├── Dockerfile-Ubuntu/
│   ├── bin/
│   ├── config.txt
│   ├── include/
│   ├── lib/
│   └── src/
│
├── zoom-token-server/                     # Token server (Python FastAPI)
│   ├── main.py
│   └── venv/
│
├── zoom-frame-server/                     # Frame analysis + deepfake detection
│   ├── frame_server.py
│   ├── deepfake_model.pt
│   ├── frames/                            # Saved PNG/RAW frames
│   ├── venv/
│   └── index.html                         # Live dashboard
│
├── videosdk-web-sample/                   # Browser-based Zoom client
│   ├── .env.local
│   ├── index.tsx
│   └── package.json
│
├── start_y3_stack.sh                      # Optional automation script
├── run_bot.py                             # Bot launcher/orchestrator
└── README.md
```

---

## 🏗️ System Architecture

```
Web Client → Token Server → run_bot.py → Linux Bot → Frame Server → Dashboard
```

### Component Flow:

1. **Web Client (Vite)** - Joins Video SDK session using OBS Virtual Camera
2. **Token Server** - Creates fresh JWT tokens to avoid expiration issues
3. **run_bot.py** - Requests token, configures bot, starts Docker container
4. **Linux Bot (C++)** - Joins session, subscribes to raw YUV pipes, sends Y-plane frames to Frame Server
5. **Frame Server** - Receives frames, performs deepfake detection, computes metrics
6. **Dashboard** - Displays real-time probability graphs and alerts

---

## 🚀 Quick Start

### Prerequisites

- Docker
- Python 3.8+
- Node.js 16+
- OBS Studio (for testing)
- Zoom Video SDK credentials

### 1. Start Token Server

```bash
cd zoom-token-server
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pyjwt
uvicorn main:app --reload --port 8000
```

**Endpoint:** `POST http://localhost:8000/token`

**Request:**
```json
{
  "tpc": "session_name",
  "role_type": 1,
  "expires_in": 7200
}
```

**Response:**
```json
{
  "token": "eyJhbGc..."
}
```

### 2. Start Frame Analysis Server

```bash
cd zoom-frame-server
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pillow torch torchvision numpy
python3 frame_server.py
```

**Runs on:** `http://localhost:8001`

**Features:**
- Receives raw grayscale frames via `POST /frame`
- Performs deepfake detection using ResNet18
- Saves frames as RAW/PNG (optional)
- Computes real-time FPS
- Applies rolling window smoothing
- Serves metrics at `/metrics`

### 3. Start Web Client

```bash
cd videosdk-web-sample
npm install
npm start
```

**Configure `.env.local`:**
```env
VITE_VIDEO_SDK_JWT=<your_token>
VITE_VIDEO_SDK_TPC=test
VITE_VIDEO_SDK_USERNAME=Safari
VITE_VIDEO_SDK_REGION=eu
```

### 4. Build and Run Linux Bot

```bash
cd videosdk-linux-raw-recording-sample

# Build Docker image
docker build -t vsdk-1.11.2-on-ubuntu -f Dockerfile-Ubuntu/Dockerfile .

# Run bot (automated)
python3 ../run_bot.py --session test
```

**What `run_bot.py` does:**
- Requests token from token server
- Writes `config.txt` with session details
- Starts Docker container with Linux bot
- Ensures clean exit on Ctrl+C

---

## 🔧 Component Details

### Linux Bot (videosdk-linux-raw-recording-sample)

Modified Zoom SDK sample that:

- Extracts raw I420 video frames
- Sends Y-plane (grayscale) as HTTP POST to Frame Server
- Includes metadata headers:
  - `X-Width`
  - `X-Height`
  - `X-Instance-Id`
  - `X-User-Id`
- Implements clean exit logic
- Runs in Docker container

### Frame Analysis Server (zoom-frame-server)

FastAPI server with deepfake detection:

**Endpoints:**
- `POST /frame` - Receive raw video frames
- `GET /metrics` - Get detection metrics (JSON)
- `GET /` - Serve dashboard HTML

**Deepfake Detection:**
- Uses ResNet18 baseline model
- Converts Y-plane grayscale to RGB
- Applies rolling window smoothing
- Logs predictions: `[DF] user=... fake_prob=0.708 smooth=0.731`

**Frame Storage (optional):**
- RAW format (binary Y-plane data)
- PNG format (grayscale image)

### Dashboard (index.html)

Real-time web dashboard featuring:

- Live probability chart (Chart.js)
- Smoothed probability trend line
- High fake probability alerts
- Automatic updates via polling

**Access:** `http://localhost:8001`

---

## 🎥 Testing with OBS

To test deepfake detection with a fake video:

1. Open **OBS Studio**
2. Add **Media Source**
   - Select your deepfake test video
   - Enable **Loop**
   - Fit to screen
3. Start **Virtual Camera**
4. In web client, select **OBS Virtual Camera** as video source
5. Bot will receive and analyze the fake stream

---

## 📊 Monitoring

**Frame Server Logs:**
```
[DF] user=participant_123 fake_prob=0.708 smooth=0.731 (N=15)
FPS: 24.5
Received frame from user participant_123: 640x360
```

**Dashboard Metrics:**
- Current fake probability
- Smoothed probability (rolling average)
- Frame rate
- Alert status

---

## 🔐 Configuration Files

### config.txt (Linux Bot)

```
tpc=test
user_name=ZoomBot
jwt=eyJhbGc...
record_path=/tmp/recordings
```

### .env.local (Web Client)

```
VITE_VIDEO_SDK_JWT=<token>
VITE_VIDEO_SDK_TPC=test
VITE_VIDEO_SDK_USERNAME=Safari
VITE_VIDEO_SDK_REGION=eu
```

---

## 🛠️ Development

### Modifying Deepfake Model

Replace the baseline ResNet18 in `frame_server.py`:

```python
# Load your trained model
model = torch.load('deepfake_model.pt')
model.eval()
```

### Adjusting Detection Parameters

```python
# Smoothing window size
SMOOTHING_WINDOW = 15

# Detection threshold
ALERT_THRESHOLD = 0.7

# Frame saving options
SAVE_RAW = True
SAVE_PNG = False
```

---

## 🚧 Known Limitations

- Baseline ResNet18 uses random weights (placeholder)
- Single-user detection only
- No face tracking or cropping
- Limited to 15-30 FPS depending on hardware
- No GPU acceleration configured

---

## 🎯 Future Improvements

- [ ] Replace baseline model with trained deepfake detector
- [ ] Multi-user simultaneous detection
- [ ] Face detection and tracking
- [ ] Face cropping before inference
- [ ] Cloud deployment for Frame Server
- [ ] GPU acceleration (CUDA/MPS/ROCm)
- [ ] FPS stabilization (30 FPS target)
- [ ] WebSocket-based dashboard updates
- [ ] Recording and playback functionality
- [ ] Alert notifications (email/Slack)

---

## 📝 License

This project uses Zoom Video SDK which requires appropriate licensing. Ensure you have valid Zoom Video SDK credentials before use.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

For issues related to:
- **Zoom SDK**: Check [Zoom Video SDK Documentation](https://developers.zoom.us/docs/video-sdk/)
- **Project Setup**: Open an issue on GitHub

---

**Built with:** Zoom Video SDK, FastAPI, Docker, PyTorch, Chart.js