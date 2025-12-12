#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import urllib.request

# URL of your FastAPI token server
TOKEN_URL = "http://localhost:8000/token"


def get_host_token(tpc: str, role_type: int = 1, expires_in: int = 2 * 60 * 60) -> str:
    """
    Ask the token server for a host token for the given session (tpc).
    role_type=1 => host
    """
    payload = {
        "tpc": tpc,
        "role_type": role_type,
        "expires_in": expires_in,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
    except Exception as e:
        print(f"[run_bot] Failed to contact token server at {TOKEN_URL}")
        print(f"[run_bot] Error: {e}")
        sys.exit(1)

    try:
        resp_json = json.loads(body)
    except Exception as e:
        print("[run_bot] Could not parse token server response as JSON")
        print(f"[run_bot] Body: {body!r}")
        print(f"[run_bot] Error: {e}")
        sys.exit(1)

    token = resp_json.get("token")
    if not token:
        print("[run_bot] Token server response did not contain 'token' field")
        print(f"[run_bot] Response JSON: {resp_json}")
        sys.exit(1)

    print("[run_bot] Got host token from token server.")
    return token


def write_config(repo_root: str, session_name: str, token: str) -> str:
    """
    Overwrite config.txt with a fresh config for this session & token.
    IMPORTANT: format must match what the C++ sample expects (plain key: "value" lines).
    """
    cfg_text = f'''session_name: "{session_name}"
session_token: "{token}"
session_psw: ""
GetVideoRawData: "true"
GetAudioRawData: "true"
SendVideoRawData: "false"
SendAudioRawData: "false"
'''

    cfg_path = os.path.join(repo_root, "config.txt")
    with open(cfg_path, "w") as f:
        f.write(cfg_text)

    print(f"[run_bot] Wrote config to {cfg_path}")
    return cfg_path

def run_docker(repo_root: str, image: str = "vsdk-1.11.2-on-ubuntu") -> None:
    """
    Run the Docker bot container with the same command you’ve been using,
    but also inject ANALYZER_URL/ANALYZER_AUDIO_URL so the C++ code knows where to POST frames/audio.
    """
    config_path = os.path.join(repo_root, "config.txt")

    analyzer_url = os.environ.get(
        "ANALYZER_URL",
        "http://host.docker.internal:9001/analyze_frame"
    )
    analyzer_audio_url = os.environ.get(
        "ANALYZER_AUDIO_URL",
        "http://host.docker.internal:8001/audio"
    )

    cmd = [
        "docker", "run",
        "--platform=linux/amd64",
        "--rm", "-it",
        "-v", f"{config_path}:/app/bin/config.txt:ro",
        "-e", f"ANALYZER_URL={analyzer_url}",   # 👈 inject URL
        "-e", f"ANALYZER_AUDIO_URL={analyzer_audio_url}",
        "--entrypoint", "/bin/bash",
        image,
        "-lc",
        "chmod +x /app/src/setup-pulseaudio.sh /app/bin/run.sh; /app/bin/run.sh",
    ]

    print("[run_bot] Running Docker command:")
    print("          " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[run_bot] Docker exited with code {e.returncode}")
        sys.exit(e.returncode)



def main():
    parser = argparse.ArgumentParser(description="Run Zoom Video SDK bot with fresh host token.")
    parser.add_argument(
        "--session", "-s",
        default="test",
        help="Session name (tpc) to join. Must match the web client."
    )
    parser.add_argument(
        "--image",
        default="vsdk-1.11.2-on-ubuntu",
        help="Docker image name for the bot container."
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    session_name = args.session

    print(f"[run_bot] Session name (tpc): {session_name}")
    print(f"[run_bot] Using Docker image: {args.image}")

    # 1) Get fresh host token
    token = get_host_token(session_name, role_type=1)

    # 2) Write config.txt for the bot
    write_config(repo_root, session_name, token)

    # 3) Launch Docker container
    run_docker(repo_root, args.image)


if __name__ == "__main__":
    main()
