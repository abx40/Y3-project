#!/usr/bin/env python3
"""Feed a video file (or webcam) into the frame server exactly as the Zoom bot does.

This replaces the only credential-bound component of the stack. The Zoom Video SDK
Linux bot subscribes to a live session and POSTs raw frames to /frame; this script
decodes a local file with ffmpeg and POSTs the same bytes with the same headers,
so the frame server, the model and the dashboard all run unmodified.

Standard library only. Requires ffmpeg on PATH.
"""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request


def build_ffmpeg_command(source: str, width: int, height: int, fps: int, realtime: bool, webcam: bool):
    cmd = ["ffmpeg", "-loglevel", "error"]
    if realtime and not webcam:
        # Decode at native speed so the dashboard behaves like a live session.
        cmd += ["-re"]
    if webcam:
        cmd += ["-f", "avfoundation", "-framerate", "30", "-i", source]
    else:
        cmd += ["-i", source]
    cmd += [
        "-vf", f"scale={width}:{height}",
        "-r", str(fps),
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]
    return cmd


def post_frame(url: str, payload: bytes, width: int, height: int, user_id: str, instance: str):
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/octet-stream")
    request.add_header("X-Width", str(width))
    request.add_header("X-Height", str(height))
    request.add_header("X-Pixel-Format", "bgr24")
    request.add_header("X-Bytes-Per-Pixel", "3")
    request.add_header("X-User-Id", user_id)
    request.add_header("X-Instance-Id", instance)
    request.add_header("X-Source", "file")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def wait_for_server(base_url: str, attempts: int = 40) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="Path to a video file to analyse.")
    parser.add_argument("--webcam", action="store_true", help="Use the default camera instead of a file.")
    parser.add_argument("--camera-index", default="0", help="avfoundation device index for --webcam (macOS).")
    parser.add_argument("--server", default="http://localhost:8001", help="Frame server base URL.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to submit for inference.")
    parser.add_argument("--user-id", default="demo_participant")
    parser.add_argument("--instance-id", default="demo")
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop after N seconds (0 = whole file).")
    parser.add_argument("--no-realtime", action="store_true", help="Decode as fast as possible instead of at native speed.")
    args = parser.parse_args()

    if not args.video and not args.webcam:
        parser.error("pass --video PATH or --webcam")

    base_url = args.server.rstrip("/")
    print(f"Waiting for frame server at {base_url} ...", flush=True)
    if not wait_for_server(base_url):
        print(f"ERROR: no frame server responding at {base_url}/health", file=sys.stderr)
        print("Start it first, e.g.:  ./demo/run_demo.sh --video 'test vids/1.mov'", file=sys.stderr)
        return 2
    print("Frame server is up.", flush=True)

    source = args.camera_index if args.webcam else args.video
    command = build_ffmpeg_command(source, args.width, args.height, args.fps,
                                   realtime=not args.no_realtime, webcam=args.webcam)
    frame_bytes = args.width * args.height * 3
    print(f"Feeding {'camera ' + source if args.webcam else source} "
          f"at {args.width}x{args.height}, {args.fps} fps", flush=True)

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    sent = 0
    started = time.time()
    try:
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload or len(payload) < frame_bytes:
                break
            try:
                post_frame(f"{base_url}/frame", payload, args.width, args.height,
                           args.user_id, args.instance_id)
            except urllib.error.URLError as error:
                print(f"\nPOST failed: {error}", file=sys.stderr)
                break
            sent += 1
            if sent % args.fps == 0:
                elapsed = time.time() - started
                print(f"  {sent} frames sent ({elapsed:0.1f}s)", end="\r", flush=True)
            if args.max_seconds and (time.time() - started) >= args.max_seconds:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\nDone. {sent} frames analysed.")
    print(f"Dashboard:  {base_url}/")
    print(f"Raw series: {base_url}/data/{args.user_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
