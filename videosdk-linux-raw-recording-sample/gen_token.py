# gen_token.py
import time, jwt

SDK_KEY    = "dyrLqR9RpazLRNS7J0jtzTCR2GlnGIguDjnU"
SDK_SECRET = "OPzRIAZ38UFIepj0i9aasseTcVNGgye3oI1Z"

payload = {
    "app_key": SDK_KEY,                 # your SDK key
    "tpc": "deepmeet-session-001",      # session name (NOT a Zoom meeting ID)
    "role_type": 1,                     # 1 = host, 0 = participant
    "version": 1,
    "iat": int(time.time()),            # issued at
    "exp": int(time.time()) + 60*60*2   # expires in 2 hours
}

token = jwt.encode(payload, SDK_SECRET, algorithm="HS256")
# PyJWT returns str on recent versions; on older it may return bytes:
print(token if isinstance(token, str) else token.decode("utf-8"))
