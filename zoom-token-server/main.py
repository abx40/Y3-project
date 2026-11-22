import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def encode_hs256_jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_segment = b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_segment}.{payload_segment}.{b64url_encode(signature)}"


load_env_file(Path(__file__).resolve().parent / ".env")

SDK_KEY = os.getenv("SDK_KEY")
SDK_SECRET = os.getenv("SDK_SECRET")
DEFAULT_TPC = os.getenv("DEFAULT_TPC", "test")

if not SDK_KEY or not SDK_SECRET:
    raise RuntimeError("SDK_KEY or SDK_SECRET not set in .env")

app = FastAPI(title="Zoom Video SDK Token Server")


class TokenRequest(BaseModel):
    tpc: Optional[str] = None       # session name
    role_type: int = 0              # 0 = participant, 1 = host
    expires_in: int = 2 * 60 * 60   # seconds (default 2h)


@app.post("/token")
def generate_token(req: TokenRequest):
    tpc = req.tpc or DEFAULT_TPC
    now = int(time.time())

    if req.role_type not in (0, 1):
        raise HTTPException(status_code=400, detail="role_type must be 0 or 1")

    payload = {
        "app_key": SDK_KEY,
        "tpc": tpc,
        "role_type": req.role_type,
        "version": 1,
        "iat": now,
        "exp": now + req.expires_in,
    }

    token = encode_hs256_jwt(payload, SDK_SECRET)
    return {
        "token": token,
        "tpc": tpc,
        "role_type": req.role_type,
        "expires_at": now + req.expires_in,
    }
