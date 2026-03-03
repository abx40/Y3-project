#!/usr/bin/env bash
set -e

# === Paths (auto-detected from this script) ===
ROOT="$(cd "$(dirname "$0")" && pwd)"
TOKEN_DIR="$ROOT/zoom-token-server"
BOT_DIR="$ROOT/videosdk-linux-raw-recording-sample"
WEB_DIR="$ROOT/videosdk-web-sample"

SESSION_NAME="test"   # must match the session name used by your web client

echo "[stack] Root:       $ROOT"
echo "[stack] Token dir:  $TOKEN_DIR"
echo "[stack] Bot dir:    $BOT_DIR"
echo "[stack] Web dir:    $WEB_DIR"
echo

# === Start token server (FastAPI) ===
echo "[stack] Starting token server on http://127.0.0.1:8000 ..."
cd "$TOKEN_DIR"

# Activate venv and run uvicorn in the background
source "$TOKEN_DIR/venv/bin/activate"
uvicorn main:app --reload --port 8000 > token-server.log 2>&1 &
TOKEN_PID=$!
deactivate

echo "[stack] Token server PID: $TOKEN_PID"

# === Start web client (Vite) ===
echo "[stack] Starting web client (npm start) ..."
cd "$WEB_DIR"
npm start > web-client.log 2>&1 &
WEB_PID=$!
echo "[stack] Web client PID:   $WEB_PID"

# === Small delay to let servers spin up ===
sleep 3

# === Start Zoom bot (Docker) ===
echo "[stack] Starting Zoom bot for session '$SESSION_NAME' ..."
cd "$BOT_DIR"
# Optionally set ANALYZER_URL before running to point at a remote frame server
# e.g., export ANALYZER_URL=http://192.168.1.134:8001/frame
python3 run_bot.py --session "$SESSION_NAME"

# === Cleanup when bot exits ===
echo "[stack] Bot exited, stopping token server and web client..."
kill "$TOKEN_PID" "$WEB_PID" 2>/dev/null || true
echo "[stack] Done."
