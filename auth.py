"""Authentication helpers — password hashing (bcrypt) + JWT tokens (PyJWT).

Kept separate from the web layer so it's easy to read as a reference. The secret
comes from SECRET_KEY in your .env; a dev default is used if it's not set.
"""

import os
import time
import bcrypt
import jwt

SECRET = os.getenv("SECRET_KEY", "dev-secret-change-me-in-production")
ALGO = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 30   # 30 days


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def make_token(user_id: int) -> str:
    now = int(time.time())
    payload = {"sub": str(user_id), "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def decode_token(token: str) -> dict:
    """Return the token payload, or raise jwt.PyJWTError if invalid/expired."""
    return jwt.decode(token, SECRET, algorithms=[ALGO])
