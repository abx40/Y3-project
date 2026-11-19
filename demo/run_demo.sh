#!/usr/bin/env bash
# One-command live demo: starts the frame server, waits for it, opens the
# dashboard and feeds a local video through the real inference pipeline.
# No Zoom credentials, no OBS, no Docker.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VIDEO="${VIDEO:-}"
BACKEND="${BACKEND:-videomae}"   # videomae needs no local checkpoint
PORT="${PORT:-8001}"
FPS="${FPS:-5}"
SECONDS_TO_RUN="${SECONDS_TO_RUN:-0}"

usage() {
  cat <<EOF
Usage: demo/run_demo.sh --video PATH [options]

  --video PATH     video file to analyse (required unless --webcam)
  --webcam         use the default camera instead of a file
  --backend NAME   videomae (default, downloads weights) | f3net | xception_df40 |
                   efficientnet_b4 | i3d | effort_clip_l14   (others need local .pth)
  --fps N          frames per second submitted for inference (default 5)
  --seconds N      stop feeding after N seconds (default: whole file)
  --port N         frame server port (default 8001)

Example:
  demo/run_demo.sh --video "../test vids/1.mov"
EOF
}

WEBCAM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)   VIDEO="$2"; shift 2 ;;
    --webcam)  WEBCAM=1; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --fps)     FPS="$2"; shift 2 ;;
    --seconds) SECONDS_TO_RUN="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ $WEBCAM -eq 0 && -z "$VIDEO" ]]; then
  echo "error: pass --video PATH or --webcam" >&2; usage; exit 1
fi

PY="zoom-frame-server/venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found on PATH" >&2; exit 1; }

cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting frame server (backend: $BACKEND) on port $PORT ..."
( cd zoom-frame-server && exec "../$PY" frame_server.py \
    --video_model "$BACKEND" --port "$PORT" --source file ) >/tmp/y3_demo_server.log 2>&1 &
SERVER_PID=$!

echo "Loading model (first run downloads weights; this can take a minute) ..."
for _ in $(seq 1 120); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

if ! curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "error: frame server did not come up. Log:" >&2
  tail -20 /tmp/y3_demo_server.log >&2
  exit 1
fi

MODEL_STATE="$(curl -fsS -m 5 "http://127.0.0.1:$PORT/info" \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("model","unknown"))' 2>/dev/null || echo unknown)"
echo "Frame server ready. Model: $MODEL_STATE"
if [[ "$MODEL_STATE" == "failed" ]]; then
  echo "WARNING: no checkpoint loaded for '$BACKEND' - scores will not be meaningful." >&2
  echo "         Use --backend videomae, or run demo/replay_trace.py instead." >&2
fi

echo "Dashboard: http://127.0.0.1:$PORT/"
( command -v open >/dev/null && open "http://127.0.0.1:$PORT/" ) 2>/dev/null || true

FEED_ARGS=(--server "http://127.0.0.1:$PORT" --fps "$FPS")
[[ "$SECONDS_TO_RUN" != "0" ]] && FEED_ARGS+=(--max-seconds "$SECONDS_TO_RUN")
if [[ $WEBCAM -eq 1 ]]; then FEED_ARGS+=(--webcam); else FEED_ARGS+=(--video "$VIDEO"); fi

python3 demo/feed_video.py "${FEED_ARGS[@]}"
echo
echo "Feed finished. Server still running on port $PORT - press Ctrl+C to stop."
wait "$SERVER_PID"
