from __future__ import annotations

import hmac
import secrets
from hashlib import sha256


def generate_support_code(length: int = 32) -> str:
    return secrets.token_urlsafe(length)


def hash_support_code(code: str) -> str:
    return sha256(str(code).encode("utf-8")).hexdigest()


def verify_support_code_hash(code: str, code_hash: str) -> bool:
    return hmac.compare_digest(hash_support_code(code), str(code_hash or ""))
