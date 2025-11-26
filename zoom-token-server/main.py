import os
import time
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

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

    token = jwt.encode(payload, SDK_SECRET, algorithm="HS256")
    return {
        "token": token,
        "tpc": tpc,
        "role_type": req.role_type,
        "expires_at": now + req.expires_in,
    }
