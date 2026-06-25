#!/usr/bin/env python3
"""Создать (или переиспользовать) синтетический тестовый аккаунт для RLHF-dev.

Синтетический user_id >= SYNTHETIC_USER_ID_MIN → telegram_linked=False (по
числовому правилу, web/server.py:7166), поэтому /api/rlhf/request-code идёт во
внутригровую почту (user_mail), НЕ дёргая Telegram. Это позволяет тестировать
login-флоу на дев-инстансе (8082) без спама реальным бот-аккаунтам Telegram.

Скрипт идемпотентен: переиспользует существующий аккаунт rlhf_dev_*, если он уже
есть в extra_accounts. Печатает user_id и display_id для использования в curl.

Запуск (зеркалит env dev_game_server.py):
    python3 rlhf_env/dev_seed_synthetic.py
"""
from __future__ import annotations

import asyncio
import os
import secrets
import string
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Те же дефолты env, что в dev_game_server.py.
_DEFAULTS = {
    "ENVIRONMENT": "development",
    "DB_HOST": "localhost", "DB_PORT": "5434",
    "DB_USER": "postgres", "DB_PASSWORD": "", "DB_NAME": "laveqox",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)


def _dsn(host, port, user, password, db):
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"


os.environ.setdefault(
    "DATABASE_URL",
    _dsn(_DEFAULTS["DB_HOST"], _DEFAULTS["DB_PORT"],
         _DEFAULTS["DB_USER"], _DEFAULTS["DB_PASSWORD"], _DEFAULTS["DB_NAME"]),
)
os.environ.setdefault(
    "EXTRAID_DATABASE_URL",
    _dsn(_DEFAULTS["DB_HOST"], _DEFAULTS["DB_PORT"],
         _DEFAULTS["DB_USER"], _DEFAULTS["DB_PASSWORD"], "extraid"),
)

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase


async def seed() -> None:
    s = get_settings()
    db = Database(s.database)
    await db.connect()
    extraid_db = ExtraIDDatabase(s.extraid_database.dsn)
    await extraid_db.connect()

    # 1) Переиспользуем существующий rlhf_dev-аккаунт, если уже сеяли.
    existing = await extraid_db.fetchrow(
        "SELECT user_id, display_id FROM extra_accounts "
        "WHERE email LIKE 'rlhf\\_dev\\_%@example.test' AND deleted_at IS NULL "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if existing:
        user_id = int(existing["user_id"])
        display_id = existing["display_id"]
        print(f"REUSED user_id={user_id} display_id={display_id}")
        # подтверждаем, что users-запись тоже на месте
        u = await db.fetchrow(
            "SELECT user_id, username, primary_deck FROM users WHERE user_id=$1",
            user_id,
        )
        print(f"  users row: {dict(u) if u else 'MISSING — перезапустите с --reseed'}")
    else:
        # 2) Выделяем синтетический user_id из seq (>= 9.1e12).
        user_id = await extraid_db.get_synthetic_user_id()
        # 3) users + 2 дефолтных пресета + стартовые карты + легальная стартовая
        #    колода в preset 1 (hero slot_1, 8 warriors slots 2-9, все owned).
        await db.ensure_user(
            user_id=user_id,
            username=f"rlhf_dev_{user_id}",
            first_name="RLHF",
            last_name="Dev",
        )
        # 4) Помечаем preset 1 как primary (is_primary=True в /api/rlhf/decks).
        try:
            await db.set_primary_deck(user_id, 1)
        except Exception as exc:
            print(f"  set_primary_deck warning: {exc}")
        # 5) Создаём ExtraID-аккаунт (display_id + email). Уникальность
        #    display_id/email —loop на коллизию.
        for _ in range(20):
            digits = "".join(secrets.choice(string.digits) for _ in range(4))
            letters = "".join(secrets.choice(string.ascii_uppercase) for _ in range(3))
            display_id = f"{digits}-{letters}"
            email = f"rlhf_dev_{user_id}@example.test"
            try:
                await extraid_db.create_extra_account(
                    user_id=user_id,
                    display_id=display_id,
                    email=email,
                    password_hash="dev_only_no_real_auth",
                    nickname=f"RLHFDev{digits}",
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError("failed_to_create_extra_account (collisions)")
        print(f"SEEDED user_id={user_id} display_id={display_id} email={email}")

    # 6) Проверяем, что колода playable через тот же контракт, что /api/rlhf/decks.
    presets = await db.get_user_deck_presets(user_id)
    playable = [p for p in (presets or []) if p.get("is_playable")]
    print(f"  deck presets: {len(presets or [])} total, {len(playable)} playable")
    for p in (presets or [])[:3]:
        print(f"    preset {p.get('preset_number')} playable={p.get('is_playable')} "
              f"has_hero={p.get('has_hero')} primary={p.get('is_primary')} "
              f"name={p.get('preset_name')!r}")

    await extraid_db.disconnect()
    await db.close()


if __name__ == "__main__":
    asyncio.run(seed())