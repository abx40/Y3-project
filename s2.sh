#!/usr/bin/env bash
set -e

# --- Config ---------------------------------------------------------
SESSION_NAME="test"

REPO_ROOT="$HOME/Desktop/Y3 Project/repo"
TOKEN_DIR="$REPO_ROOT/zoom-token-server"
FRAME_DIR="$REPO_ROOT/zoom-frame-server"
WEB_DIR="$REPO_ROOT/videosdk-web-sample"
BOT_DIR="$REPO_ROOT/videosdk-linux-raw-recording-sample"
# -------------------------------------------------------------------

echo "[STACK] Using repo root: $REPO_ROOT"
echo "[STACK] Session name: $SESSION_NAME"

# ---------------- Token server ----------------
echo "[STACK] Starting token server..."
cd "$TOKEN_DIR"
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 > token_server.log 2>&1 &
TOKEN_PID=$!
deactivate
echo "[STACK] Token server PID: $TOKEN_PID (logs: $TOKEN_DIR/token_server.log)"

# ---------------- Frame server ----------------
echo "[STACK] Starting frame server..."
cd "$FRAME_DIR"
source venv/bin/activate
uvicorn frame_server:app --host 0.0.0.0 --port 8001 > frame_server.log 2>&1 &
FRAME_PID=$!
deactivate
echo "[STACK] Frame server PID: $FRAME_PID (logs: $FRAME_DIR/frame_server.log)"

# ---------------- Web client ------------------
echo "[STACK] Starting web client (npm start)..."
cd "$WEB_DIR"
npm start > web_client.log 2>&1 &
WEB_PID=$!
echo "[STACK] Web client PID: $WEB_PID (logs: $WEB_DIR/web_client.log)"

# ---------------- Bot (foreground) -----------
echo "[STACK] Starting bot..."
cd "$BOT_DIR"
python3 run_bot.py --session "$SESSION_NAME"

BOT_EXIT=$?
echo "[STACK] Bot exited with code $BOT_EXIT"

# ---------------- Cleanup ---------------------
echo "[STACK] Stopping servers..."
kill $TOKEN_PID $FRAME_PID $WEB_PID 2>/dev/null || true
echo "[STACK] Done."
exit $BOT_EXIT