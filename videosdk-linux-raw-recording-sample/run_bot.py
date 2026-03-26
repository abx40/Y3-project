#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import urllib.request

# URL of your FastAPI token server
TOKEN_URL = os.environ.get("TOKEN_URL", "http://localhost:8000/token")


def get_host_token(token_url: str, tpc: str, role_type: int = 1, expires_in: int = 2 * 60 * 60) -> str:
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
        token_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
    except Exception as e:
        print(f"[run_bot] Failed to contact token server at {token_url}")
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


def write_config(repo_root: str, session_name: str, token: str, config_out: str | None = None) -> str:
    """
    Overwrite config.txt with a fresh config for this session & token.
    IMPORTANT: format must match what the C++ sample expects (plain key: "value" lines).
    """
    cfg_text = f'''session_name: "{session_name}"
session_token: "{token}"
session_psw: ""
GetVideoRawData: "true"
GetAudioRawData: "false"
SendVideoRawData: "false"
SendAudioRawData: "false"
'''

    cfg_path = config_out or os.path.join(repo_root, "config.txt")
    with open(cfg_path, "w") as f:
        f.write(cfg_text)

    print(f"[run_bot] Wrote config to {cfg_path}")
    return cfg_path

def run_docker(
    config_path: str,
    image: str = "vsdk-1.11.2-on-ubuntu",
    analyzer_url: str | None = None,
    interactive: bool = True,
) -> None:
    """
    Run the Docker bot container with the same command you’ve been using,
    but inject ANALYZER_URL so the C++ code knows where to POST frames (audio disabled).
    """
    analyzer_url = analyzer_url or os.environ.get(
        "ANALYZER_URL",
        "http://host.docker.internal:8001/frame"
    )
    # ANALYZER_AUDIO_URL intentionally disabled while running video-only.

    cmd = [
        "docker", "run",
        "--platform=linux/amd64",
        "--rm",
        "-v", f"{config_path}:/app/bin/config.txt:ro",
        "-e", f"ANALYZER_URL={analyzer_url}",   # 👈 inject URL
        "--entrypoint", "/bin/bash",
        image,
        "-lc",
        "chmod +x /app/src/setup-pulseaudio.sh /app/bin/run.sh; /app/bin/run.sh",
    ]
    if interactive:
        cmd.insert(4, "-it")

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
    parser.add_argument(
        "--token-url",
        default=TOKEN_URL,
        help="Token server URL."
    )
    parser.add_argument(
        "--analyzer-url",
        default=os.environ.get("ANALYZER_URL", "http://host.docker.internal:8001/frame"),
        help="Frame server analyzer URL."
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Optional output path for config.txt."
    )
    parser.add_argument(
        "--role-type",
        type=int,
        default=1,
        help="Zoom Video SDK role type for the bot token (default: host=1)."
    )
    parser.add_argument(
        "--expires-in",
        type=int,
        default=2 * 60 * 60,
        help="Token lifetime in seconds."
    )
    parser.add_argument(
        "--no-tty",
        action="store_true",
        help="Run docker without -it for non-interactive automation."
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    session_name = args.session

    print(f"[run_bot] Session name (tpc): {session_name}")
    print(f"[run_bot] Using Docker image: {args.image}")
    print(f"[run_bot] Token URL: {args.token_url}")
    print(f"[run_bot] Analyzer URL: {args.analyzer_url}")

    # 1) Get fresh host token
    token = get_host_token(args.token_url, session_name, role_type=args.role_type, expires_in=args.expires_in)

    # 2) Write config.txt for the bot
    config_path = write_config(repo_root, session_name, token, config_out=args.config_out)

    # 3) Launch Docker container
    run_docker(
        config_path=config_path,
        image=args.image,
        analyzer_url=args.analyzer_url,
        interactive=not args.no_tty,
    )


if __name__ == "__main__":
    main()
