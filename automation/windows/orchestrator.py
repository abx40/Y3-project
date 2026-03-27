from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obs_controller import OBSController, OBSControllerError


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_DIR = REPO_ROOT / "zoom-token-server"
FRAME_DIR = REPO_ROOT / "zoom-frame-server"
WEB_DIR = REPO_ROOT / "videosdk-web-sample"
BOT_DIR = REPO_ROOT / "videosdk-linux-raw-recording-sample"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

SUMMARY_COLUMNS = [
    "run_id",
    "status",
    "split",
    "kind",
    "ordinal",
    "model",
    "session_name",
    "video_path",
    "labels_path",
    "duration_s",
    "runtime_s",
    "buffer_s",
    "prediction_rows",
    "active_video_model",
    "resolved_model_name",
    "result_dir",
    "started_at_iso",
    "ended_at_iso",
    "error",
]


class OrchestrationError(RuntimeError):
    pass


@dataclass
class SessionRecord:
    split: str
    kind: str
    ordinal: int
    video_path: Path
    labels_path: Path
    duration_s: float
    raw_row: dict[str, str]

    @property
    def session_key(self) -> str:
        return sanitize_identifier(f"{self.split}_{self.kind}_{self.ordinal:02d}")

    @property
    def video_stem(self) -> str:
        return sanitize_identifier(self.video_path.stem)


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.env = env or os.environ.copy()
        self.proc: subprocess.Popen[str] | None = None
        self._log_handle = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env=self.env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def terminate(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self._close_log()
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:  # pragma: no cover
            self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if self.proc.poll() is None:
                self.proc.kill()
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None


def sanitize_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def split_csv_arg(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def probe_http_url(url: str, timeout: float = 10.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
            return {
                "url": url,
                "status": response.status,
                "ok": response.status == 200,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return {
            "url": url,
            "status": exc.code,
            "ok": False,
            "content_type": exc.headers.get("Content-Type", ""),
            "content_length": len(body),
            "sha256": hashlib.sha256(body).hexdigest() if body else "",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "url": url,
            "status": None,
            "ok": False,
            "content_type": "",
            "content_length": 0,
            "sha256": "",
            "error": str(exc),
        }


def wait_for_http_json(
    url: str,
    timeout_seconds: float,
    predicate: callable | None = None,
    description: str = "service",
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            data = http_get_json(url, timeout=5.0)
            if predicate is None or predicate(data):
                return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise OrchestrationError(f"Timed out waiting for {description} at {url}: {last_error}")


def wait_for_http_status(url: str, timeout_seconds: float, description: str) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise OrchestrationError(f"Timed out waiting for {description} at {url}: {last_error}")


def build_web_asset_urls(web_port: int) -> list[str]:
    base_url = f"http://127.0.0.1:{web_port}"
    return [
        f"{base_url}/",
        f"{base_url}/lib/manifest.json",
        f"{base_url}/lib/js_media.min.js",
        f"{base_url}/lib/video.decode.wasm",
    ]


def preflight_web_assets(web_port: int, output_path: Path | None = None) -> list[dict[str, Any]]:
    probes = [probe_http_url(url) for url in build_web_asset_urls(web_port)]
    if output_path is not None:
        write_json_file(output_path, {"probes": probes})
    failures = [probe for probe in probes if probe.get("status") != 200]
    if failures:
        first_failure = failures[0]
        raise OrchestrationError(
            "Web client asset preflight failed: "
            f"{first_failure['url']} -> {first_failure.get('status') or first_failure.get('error')}"
        )
    return probes


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def resolve_local_dataset_path(raw_path: str, eval_root: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    if "/sessions/" in normalized:
        suffix = normalized.split("/sessions/", 1)[1]
        return eval_root / "sessions" / Path(suffix)
    if normalized.startswith("sessions/"):
        return eval_root / Path(normalized)
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return eval_root / candidate


def discover_eval_root(explicit_root: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.extend(
        [
            Path(r"C:\deepfake_eval\leakfree_eval"),
            Path(r"C:\deepfake_eval"),
            Path.home() / "deepfake_eval" / "leakfree_eval",
            Path.home() / "deepfake_eval",
        ]
    )

    for candidate in candidates:
        sessions_index = candidate / "sessions_index.csv"
        sessions_dir = candidate / "sessions"
        if sessions_index.is_file() and sessions_dir.is_dir():
            return candidate

    for drive in ["C:\\", "D:\\", "E:\\"]:
        drive_path = Path(drive)
        if not drive_path.exists():
            continue
        for root, dirs, files in os.walk(drive):
            if "sessions_index.csv" in files and "sessions" in dirs:
                return Path(root)

    raise OrchestrationError("Could not discover evaluation root containing sessions_index.csv and sessions/.")


def load_sessions(eval_root: Path) -> list[SessionRecord]:
    index_path = eval_root / "sessions_index.csv"
    if not index_path.is_file():
        raise OrchestrationError(f"Missing sessions_index.csv at {index_path}")

    sessions: list[SessionRecord] = []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sessions.append(
                SessionRecord(
                    split=row["split"].strip(),
                    kind=row["kind"].strip(),
                    ordinal=int(row["ordinal"]),
                    video_path=resolve_local_dataset_path(row["video_path"], eval_root),
                    labels_path=resolve_local_dataset_path(row["labels_path"], eval_root),
                    duration_s=float(row["duration_s"]),
                    raw_row=row,
                )
            )
    return sessions


def filter_sessions(
    sessions: list[SessionRecord],
    splits: list[str],
    kinds: list[str],
    ordinals: list[int],
    smoke_test: bool,
) -> list[SessionRecord]:
    selected = [
        item
        for item in sessions
        if (not splits or item.split in splits)
        and (not kinds or item.kind in kinds)
        and (not ordinals or item.ordinal in ordinals)
    ]
    if not selected:
        raise OrchestrationError("No sessions matched the requested filters.")
    selected.sort(key=lambda item: (item.split, item.kind, item.ordinal))
    if smoke_test:
        return [selected[0]]
    return selected


def detect_frame_server_backends(frame_server_path: Path) -> list[str]:
    tree = ast.parse(frame_server_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AVAILABLE_VIDEO_MODELS":
                    models = ast.literal_eval(node.value)
                    return [model for model in models if model != "auto"]
    raise OrchestrationError(f"Could not detect AVAILABLE_VIDEO_MODELS from {frame_server_path}")


def choose_models(raw_models: str, available_models: list[str], smoke_test: bool) -> list[str]:
    requested = split_csv_arg(raw_models)
    if not requested or requested == ["all"]:
        return [available_models[0]] if smoke_test else available_models
    unknown = [model for model in requested if model not in available_models and model != "auto"]
    if unknown:
        raise OrchestrationError(f"Unknown model backend(s): {', '.join(unknown)}")
    return [requested[0]] if smoke_test else requested


def detect_browser(browser_arg: str | None) -> Path:
    if browser_arg:
        path = Path(browser_arg)
        if path.is_file():
            return path
        raise OrchestrationError(f"Browser executable not found: {browser_arg}")

    candidates = [
        shutil.which("msedge.exe"),
        shutil.which("chrome.exe"),
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise OrchestrationError("Could not find Edge or Chrome. Pass --browser with an explicit executable path.")


def detect_python() -> Path:
    venv_python = FRAME_DIR / "venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return venv_python
    system_python = shutil.which("python")
    if system_python:
        return Path(system_python)
    raise OrchestrationError("Could not find a Python interpreter.")


def detect_command(command_name: str) -> str:
    found = shutil.which(command_name)
    if found:
        return found
    raise OrchestrationError(f"Required command '{command_name}' was not found on PATH.")


def ensure_web_dependencies(npm_cmd: str, results_dir: Path) -> None:
    if (WEB_DIR / "node_modules").is_dir():
        return
    log_path = results_dir / "bootstrap" / "npm-install.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [npm_cmd, "install"],
            cwd=str(WEB_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise OrchestrationError(f"'npm install' failed. See {log_path}")


def ensure_bot_image(docker_cmd: str, image_name: str, skip_build: bool, results_dir: Path) -> None:
    inspect = subprocess.run(
        [docker_cmd, "image", "inspect", image_name],
        cwd=str(BOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if inspect.returncode == 0:
        return
    if skip_build:
        raise OrchestrationError(
            f"Docker image '{image_name}' is missing. Re-run without --skip-build-bot-image or build it manually."
        )
    log_path = results_dir / "bootstrap" / "docker-build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [docker_cmd, "build", "-t", image_name, "-f", "Dockerfile-Ubuntu/Dockerfile", "."],
            cwd=str(BOT_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise OrchestrationError(f"Docker image build failed. See {log_path}")


def build_browser_url(
    web_port: int,
    session_name: str,
    participant_token: str,
    run_id: str,
    camera_label: str,
    frame_port: int,
    auto_leave_seconds: float,
) -> str:
    params = {
        "topic": session_name,
        "name": "Y3 Eval Browser",
        "signature": participant_token,
        "runId": run_id,
        "autoStartVideo": "1",
        "cameraLabelContains": camera_label,
        "automationEventUrl": f"http://127.0.0.1:{frame_port}/automation/event",
        "autoLeaveSeconds": str(int(auto_leave_seconds)),
    }
    return f"http://127.0.0.1:{web_port}/video?{urllib.parse.urlencode(params)}"


def fetch_participant_token(token_port: int, session_name: str) -> str:
    response = http_post_json(
        f"http://127.0.0.1:{token_port}/token",
        {"tpc": session_name, "role_type": 0, "expires_in": 2 * 60 * 60},
        timeout=10.0,
    )
    token = response.get("token")
    if not token:
        raise OrchestrationError(f"Token server response did not include 'token': {response}")
    return token


def prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def update_summary(summary_path: Path, row: dict[str, Any]) -> None:
    rows_by_run: dict[str, dict[str, Any]] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                rows_by_run[existing["run_id"]] = existing
    if row.get("status") == "skipped_existing" and row["run_id"] in rows_by_run:
        return
    rows_by_run[row["run_id"]] = {column: str(row.get(column, "")) for column in SUMMARY_COLUMNS}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for run_id in sorted(rows_by_run):
            writer.writerow(rows_by_run[run_id])


def wait_for_automation_ready(frame_port: int, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        try:
            last_status = http_get_json(f"http://127.0.0.1:{frame_port}/automation/status", timeout=5.0)
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
            continue

        if last_status.get("last_error"):
            raise OrchestrationError(f"Web client automation failed: {last_status['last_error']}")
        if (
            last_status.get("connected")
            and last_status.get("camera_selected")
            and last_status.get("video_started")
        ):
            return last_status
        time.sleep(1.0)
    raise OrchestrationError(f"Timed out waiting for web client readiness: {last_status}")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_single(
    session: SessionRecord,
    model: str,
    args: argparse.Namespace,
    python_executable: Path,
    browser_path: Path,
    results_dir: Path,
) -> dict[str, Any]:
    run_id = sanitize_identifier(f"{session.video_stem}__{model}")
    run_dir = results_dir / run_id
    status_path = run_dir / "status.json"
    runtime_s = session.duration_s
    if args.duration_limit_seconds:
        runtime_s = min(runtime_s, args.duration_limit_seconds)
    elif args.smoke_test and args.smoke_duration_seconds:
        runtime_s = min(runtime_s, args.smoke_duration_seconds)

    if args.resume_failed:
        existing_status = read_json_file(status_path)
        existing_runtime_s = 0.0
        try:
            existing_runtime_s = float((existing_status or {}).get("runtime_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            existing_runtime_s = 0.0
        if (
            existing_status
            and existing_status.get("status") == "success"
            and existing_runtime_s >= (float(runtime_s) - 1e-6)
        ):
            return {
                "run_id": run_id,
                "status": "skipped_existing",
                "split": session.split,
                "kind": session.kind,
                "ordinal": session.ordinal,
                "model": model,
                "session_name": run_id,
                "video_path": str(session.video_path),
                "labels_path": str(session.labels_path),
                "duration_s": session.duration_s,
                "runtime_s": existing_status.get("runtime_s", ""),
                "buffer_s": args.buffer_seconds,
                "prediction_rows": existing_status.get("prediction_rows", ""),
                "active_video_model": existing_status.get("active_video_model", ""),
                "resolved_model_name": existing_status.get("resolved_model_name", ""),
                "result_dir": str(run_dir),
                "started_at_iso": existing_status.get("started_at_iso", ""),
                "ended_at_iso": existing_status.get("ended_at_iso", ""),
                "error": "",
            }

    prepare_run_dir(run_dir)

    session_name = run_id
    total_wait_s = runtime_s + args.buffer_seconds
    started_at_iso = iso_now()
    browser_url = ""
    write_json_file(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "session_name": session_name,
            "split": session.split,
            "kind": session.kind,
            "ordinal": session.ordinal,
            "video_path": str(session.video_path),
            "labels_path": str(session.labels_path),
            "duration_s": session.duration_s,
            "runtime_s": runtime_s,
            "buffer_s": args.buffer_seconds,
            "model": model,
            "started_at_iso": started_at_iso,
        },
    )

    token_proc = None
    frame_proc = None
    browser_proc = None
    bot_proc = None
    obs = None
    final_status = "failed"
    error_message = ""

    try:
        if not session.video_path.is_file():
            raise OrchestrationError(f"Session video not found: {session.video_path}")
        if not session.labels_path.is_file():
            raise OrchestrationError(f"Session labels not found: {session.labels_path}")

        token_proc = ManagedProcess(
            "token-server",
            [
                str(python_executable),
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.token_port),
            ],
            TOKEN_DIR,
            run_dir / "token_server.log",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        token_proc.start()
        wait_for_http_status(
            f"http://127.0.0.1:{args.token_port}/openapi.json",
            args.service_ready_timeout_seconds,
            "token server",
        )

        frame_proc = ManagedProcess(
            "frame-server",
            [
                str(python_executable),
                "frame_server.py",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.frame_port),
                "--run_id",
                run_id,
                "--session_id",
                session_name,
                "--labels_csv",
                str(session.labels_path),
                "--log_dir",
                str(results_dir),
                "--model_name",
                model,
                "--video_model",
                model,
                "--source",
                "zoom-bot",
                "--save_raw",
                "off",
                "--save_png",
                "off",
            ],
            FRAME_DIR,
            run_dir / "frame_server.log",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        frame_proc.start()
        frame_info = wait_for_http_json(
            f"http://127.0.0.1:{args.frame_port}/info",
            args.service_ready_timeout_seconds,
            description="frame server",
        )
        if frame_info.get("active_video_model") == "failed":
            raise OrchestrationError(f"Frame server failed to initialize model '{model}'.")

        participant_token = fetch_participant_token(args.token_port, session_name)

        obs = OBSController(args.obs_host, args.obs_port, args.obs_password, 10.0)
        obs.connect()
        obs.configure_media_input(args.obs_scene, args.obs_input, session.video_path, loop=False)
        obs.stop_media_input(args.obs_input)
        obs.start_virtual_camera()

        browser_url = build_browser_url(
            args.web_port,
            session_name,
            participant_token,
            run_id,
            args.browser_camera_label,
            args.frame_port,
            total_wait_s + 2,
        )
        (run_dir / "browser_url.txt").write_text(f"{browser_url}\n", encoding="utf-8")
        preflight_web_assets(args.web_port, run_dir / "web_asset_probe.json")

        browser_profile_dir = results_dir / "browser-profile" / run_id
        browser_profile_dir.mkdir(parents=True, exist_ok=True)
        browser_proc = ManagedProcess(
            "browser",
            [
                str(browser_path),
                "--new-window",
                f"--user-data-dir={browser_profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--autoplay-policy=no-user-gesture-required",
                "--use-fake-ui-for-media-stream",
                "--window-size=1280,900",
                browser_url,
            ],
            REPO_ROOT,
            run_dir / "browser_process.log",
        )
        browser_proc.start()
        wait_for_automation_ready(args.frame_port, args.web_ready_timeout_seconds)

        obs.restart_media_input(args.obs_input)
        time.sleep(1.0)

        if args.skip_bot or args.dry_run:
            (run_dir / "bot.log").write_text("bot skipped by runner configuration\n", encoding="utf-8")
        else:
            bot_proc = ManagedProcess(
                "bot",
                [
                    str(python_executable),
                    "run_bot.py",
                    "--session",
                    session_name,
                    "--image",
                    args.bot_image,
                    "--token-url",
                    f"http://127.0.0.1:{args.token_port}/token",
                    "--analyzer-url",
                    f"http://host.docker.internal:{args.frame_port}/frame",
                    "--config-out",
                    str(run_dir / "bot_config.txt"),
                    "--no-tty",
                ],
                BOT_DIR,
                run_dir / "bot.log",
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            bot_proc.start()

        deadline = time.time() + total_wait_s
        while time.time() < deadline:
            if bot_proc is not None and bot_proc.poll() is not None:
                if bot_proc.poll() != 0:
                    raise OrchestrationError(f"Bot exited early with code {bot_proc.poll()}.")
                break
            time.sleep(2.0)

        final_status = "success"
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        final_status = "failed"
    finally:
        if browser_proc is not None:
            browser_proc.terminate()
        if bot_proc is not None:
            bot_proc.terminate()
        if frame_proc is not None:
            frame_proc.terminate()
        if token_proc is not None:
            token_proc.terminate()
        if obs is not None:
            try:
                obs.stop_media_input(args.obs_input)
                obs.stop_virtual_camera()
            except Exception:
                pass
            obs.close()

        frame_meta = read_json_file(run_dir / "run_meta.json") or {}
        status_payload = {
            "status": final_status,
            "run_id": run_id,
            "started_at_iso": started_at_iso,
            "ended_at_iso": iso_now(),
            "runtime_s": runtime_s,
            "buffer_s": args.buffer_seconds,
            "prediction_rows": count_csv_rows(run_dir / "predictions.csv"),
            "active_video_model": frame_meta.get("video_model", ""),
            "resolved_model_name": frame_meta.get("resolved_model_name", ""),
            "error": error_message,
        }
        write_json_file(status_path, status_payload)

    status_data = read_json_file(status_path) or {}
    return {
        "run_id": run_id,
        "status": status_data.get("status", final_status),
        "split": session.split,
        "kind": session.kind,
        "ordinal": session.ordinal,
        "model": model,
        "session_name": session_name,
        "video_path": str(session.video_path),
        "labels_path": str(session.labels_path),
        "duration_s": session.duration_s,
        "runtime_s": runtime_s,
        "buffer_s": args.buffer_seconds,
        "prediction_rows": status_data.get("prediction_rows", 0),
        "active_video_model": status_data.get("active_video_model", ""),
        "resolved_model_name": status_data.get("resolved_model_name", ""),
        "result_dir": str(run_dir),
        "started_at_iso": started_at_iso,
        "ended_at_iso": status_data.get("ended_at_iso", iso_now()),
        "error": status_data.get("error", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows benchmark orchestration for OBS Virtual Camera evaluation.")
    parser.add_argument("--eval-root", default=None)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models", default="all", help="Comma-separated model backends or 'all'.")
    parser.add_argument("--splits", default="", help="Optional comma-separated split filter.")
    parser.add_argument("--kinds", default="", help="Optional comma-separated kind filter.")
    parser.add_argument("--ordinals", default="", help="Optional comma-separated ordinal filter.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--duration-limit-seconds", type=float, default=None)
    parser.add_argument("--smoke-duration-seconds", type=float, default=45.0)
    parser.add_argument("--buffer-seconds", type=float, default=15.0)
    parser.add_argument("--browser", default=None)
    parser.add_argument("--web-port", type=int, default=3000)
    parser.add_argument("--token-port", type=int, default=8000)
    parser.add_argument("--frame-port", type=int, default=8001)
    parser.add_argument("--obs-host", default="127.0.0.1")
    parser.add_argument("--obs-port", type=int, default=4455)
    parser.add_argument("--obs-password", default=os.environ.get("OBS_WEBSOCKET_PASSWORD", ""))
    parser.add_argument("--obs-scene", default="Y3 Eval")
    parser.add_argument("--obs-input", default="Y3 Eval Media")
    parser.add_argument("--browser-camera-label", default="OBS Virtual Camera")
    parser.add_argument("--bot-image", default="vsdk-1.11.2-on-ubuntu")
    parser.add_argument("--skip-build-bot-image", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-bot", action="store_true")
    parser.add_argument("--service-ready-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--web-ready-timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_root = discover_eval_root(args.eval_root)
    sessions = load_sessions(eval_root)
    selected_sessions = filter_sessions(
        sessions=sessions,
        splits=split_csv_arg(args.splits),
        kinds=split_csv_arg(args.kinds),
        ordinals=[int(item) for item in split_csv_arg(args.ordinals)],
        smoke_test=args.smoke_test,
    )
    available_models = detect_frame_server_backends(FRAME_DIR / "frame_server.py")
    selected_models = choose_models(args.models, available_models, args.smoke_test)

    if args.dry_run:
        print(f"[orchestrator] dry_run=true")
        print(f"[orchestrator] eval_root={eval_root}")
        print(f"[orchestrator] models={selected_models}")
        for session in selected_sessions:
            print(
                f"[orchestrator] session split={session.split} kind={session.kind} "
                f"ordinal={session.ordinal} video={session.video_path}"
            )
        return 0

    python_executable = detect_python()
    browser_path = detect_browser(args.browser)
    npm_cmd = detect_command("npm.cmd" if os.name == "nt" else "npm")
    docker_cmd = ""

    ensure_web_dependencies(npm_cmd, results_dir)
    if not args.skip_bot:
        docker_cmd = detect_command("docker")
        ensure_bot_image(docker_cmd, args.bot_image, args.skip_build_bot_image, results_dir)

    web_server_proc = ManagedProcess(
        "web-server",
        [npm_cmd, "start", "--", "--host", "127.0.0.1", "--port", str(args.web_port), "--strictPort"],
        WEB_DIR,
        results_dir / "web_server.log",
        env={**os.environ, "VITE_OPEN_BROWSER": "0"},
    )
    web_server_proc.start()

    try:
        wait_for_http_status(
            f"http://127.0.0.1:{args.web_port}/",
            args.service_ready_timeout_seconds,
            "web client dev server",
        )
        preflight_web_assets(args.web_port, results_dir / "bootstrap" / "web_asset_probe.json")
        print(f"[orchestrator] eval_root={eval_root}")
        print(f"[orchestrator] models={selected_models}")
        print(f"[orchestrator] sessions={len(selected_sessions)}")

        summary_path = results_dir / "summary.csv"
        exit_code = 0
        for session in selected_sessions:
            for model in selected_models:
                row = run_single(session, model, args, python_executable, browser_path, results_dir)
                update_summary(summary_path, row)
                print(f"[orchestrator] {row['status']} {row['run_id']}")
                if row["status"] == "failed":
                    exit_code = 1
        return exit_code
    finally:
        web_server_proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
