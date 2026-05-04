from __future__ import annotations

import json
from pathlib import Path
from typing import Set

BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "data" / "users.json"


def _ensure_storage() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("[]", encoding="utf-8")


def _read_users() -> Set[int]:
    _ensure_storage()
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = []
    return {int(user_id) for user_id in data}


def _write_users(users: Set[int]) -> None:
    USERS_FILE.write_text(json.dumps(sorted(users)), encoding="utf-8")


def add_user_id(user_id: int) -> None:
    users = _read_users()
    if user_id not in users:
        users.add(user_id)
        _write_users(users)


def get_all_user_ids() -> Set[int]:
    return _read_users()

