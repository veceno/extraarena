"""Минимальный IO-хелпер: fake-JWT для ``?_auth=`` и audio-query парсер.

Socket.IO-транспорт НЕ нужен — плеер идёт путём (B): bridge зовёт
``window.handleStateChanged`` напрямую, recorder stub'ит ``window.io``.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, Optional


def _b64u(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def make_fake_jwt(*, uid: int = 1001, seed: str = "orchestra") -> str:
    """JWT-подобная строка ``xxx.yyy.zzz`` для ``looksLikeArenaJwtBearer`` (arena.js:2189).

    Сервер оркестра не валидирует содержимое — нужен только вид Bearer'а.
    """
    header = _b64u('{"alg":"HS256","typ":"JWT"}')
    payload = _b64u(f'{{"uid":{uid},"src":"orchestra","seed":"{seed}"}}')
    sig = _b64u("orchestra-sig-" + seed)
    return f"{header}.{payload}.{sig}"


def audio_query(spec: Optional[Dict[str, Any]]) -> str:
    """``?music=0&sfx=1``-суффикс для redirect_url из spec.audio (как rlhf).

    В оркестре sfx всегда включён (демо-ролик со звуком арены), music выключена.
    """
    music = 0
    sfx = 1
    if spec and isinstance(spec.get("audio"), dict):
        a = spec["audio"]
        music = 0 if a.get("music") in (False, 0) else 1
        sfx = 1 if a.get("sfx") in (True, 1, None) else 0
    return f"&music={music}&sfx={sfx}"


__all__ = ["make_fake_jwt", "audio_query"]