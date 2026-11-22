from __future__ import annotations

import base64
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


class OBSControllerError(RuntimeError):
    pass


class OBSController:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4455,
        password: str = "",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.timeout_seconds = timeout_seconds
        self._ws = None

    @property
    def websocket_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def connect(self) -> None:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise OBSControllerError(
                "Missing Python dependency 'websocket-client'. Install it with "
                "'pip install websocket-client'."
            ) from exc

        try:
            self._ws = websocket.create_connection(self.websocket_url, timeout=self.timeout_seconds)
            hello = self._recv_json()
        except Exception as exc:
            raise OBSControllerError(
                f"Failed to connect to OBS websocket at {self.websocket_url}: {exc}"
            ) from exc

        if hello.get("op") != 0:
            raise OBSControllerError(f"Unexpected OBS hello payload: {hello}")

        identify_payload: dict[str, Any] = {"rpcVersion": 1}
        auth = hello.get("d", {}).get("authentication")
        if auth:
            identify_payload["authentication"] = self._build_authentication(
                challenge=auth["challenge"],
                salt=auth["salt"],
            )

        self._send_json({"op": 1, "d": identify_payload})
        identified = self._recv_json()
        if identified.get("op") != 2:
            raise OBSControllerError(f"Unexpected OBS identify response: {identified}")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def ensure_scene(self, scene_name: str) -> None:
        scene_list = self.request("GetSceneList").get("scenes", [])
        if any(scene.get("sceneName") == scene_name for scene in scene_list):
            return
        self.request("CreateScene", {"sceneName": scene_name})

    def ensure_media_input(self, scene_name: str, input_name: str) -> None:
        inputs = self.request("GetInputList").get("inputs", [])
        if any(item.get("inputName") == input_name for item in inputs):
            return
        self.request(
            "CreateInput",
            {
                "sceneName": scene_name,
                "inputName": input_name,
                "inputKind": "ffmpeg_source",
                "inputSettings": {
                    "is_local_file": True,
                    "local_file": "",
                    "looping": False,
                    "restart_on_activate": True,
                    "clear_on_media_end": False,
                },
                "sceneItemEnabled": True,
            },
        )

    def set_current_program_scene(self, scene_name: str) -> None:
        self.request("SetCurrentProgramScene", {"sceneName": scene_name})

    def configure_media_input(self, scene_name: str, input_name: str, video_path: Path, loop: bool = False) -> None:
        self.ensure_scene(scene_name)
        self.ensure_media_input(scene_name, input_name)
        self.set_current_program_scene(scene_name)
        self.request(
            "SetInputSettings",
            {
                "inputName": input_name,
                "inputSettings": {
                    "is_local_file": True,
                    "local_file": str(video_path),
                    "looping": bool(loop),
                    "restart_on_activate": True,
                    "clear_on_media_end": False,
                },
                "overlay": False,
            },
        )

    def restart_media_input(self, input_name: str) -> None:
        self.request(
            "TriggerMediaInputAction",
            {
                "inputName": input_name,
                "mediaAction": "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
            },
        )

    def stop_media_input(self, input_name: str) -> None:
        self.request(
            "TriggerMediaInputAction",
            {
                "inputName": input_name,
                "mediaAction": "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_STOP",
            },
        )

    def start_virtual_camera(self) -> None:
        if not self.get_virtual_camera_active():
            self.request("StartVirtualCam")

    def stop_virtual_camera(self) -> None:
        if self.get_virtual_camera_active():
            self.request("StopVirtualCam")

    def get_virtual_camera_active(self) -> bool:
        data = self.request("GetVirtualCamStatus")
        return bool(data.get("outputActive"))

    def request(self, request_type: str, request_data: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._ws is None:
            raise OBSControllerError("OBS websocket is not connected.")

        request_id = str(uuid.uuid4())
        self._send_json(
            {
                "op": 6,
                "d": {
                    "requestType": request_type,
                    "requestId": request_id,
                    "requestData": request_data or {},
                },
            }
        )

        while True:
            payload = self._recv_json()
            if payload.get("op") != 7:
                continue
            data = payload.get("d", {})
            if data.get("requestId") != request_id:
                continue
            status = data.get("requestStatus", {})
            if not status.get("result"):
                raise OBSControllerError(
                    f"OBS request '{request_type}' failed: "
                    f"{status.get('comment') or status.get('code')}"
                )
            return data.get("responseData", {})

    def _build_authentication(self, challenge: str, salt: str) -> str:
        secret = base64.b64encode(hashlib.sha256(f"{self.password}{salt}".encode("utf-8")).digest()).decode("ascii")
        return base64.b64encode(hashlib.sha256(f"{secret}{challenge}".encode("utf-8")).digest()).decode("ascii")

    def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise OBSControllerError("OBS websocket is not connected.")
        self._ws.send(json.dumps(payload))

    def _recv_json(self) -> dict[str, Any]:
        if self._ws is None:
            raise OBSControllerError("OBS websocket is not connected.")
        raw = self._ws.recv()
        return json.loads(raw)
