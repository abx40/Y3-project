# gen_token.py
import os
import time

import jwt

SDK_KEY = os.getenv("SDK_KEY")
SDK_SECRET = os.getenv("SDK_SECRET")
SESSION_NAME = os.getenv("ZOOM_VIDEO_SDK_TPC", "deepmeet-session-001")

if not SDK_KEY or not SDK_SECRET:
    raise RuntimeError("Set SDK_KEY and SDK_SECRET in your environment before generating a token")

payload = {
    "app_key": SDK_KEY,
    "tpc": SESSION_NAME,
    "role_type": 1,
    "version": 1,
    "iat": int(time.time()),
    "exp": int(time.time()) + 60 * 60 * 2,
}

token = jwt.encode(payload, SDK_SECRET, algorithm="HS256")
print(token if isinstance(token, str) else token.decode("utf-8"))
