#!/usr/bin/env python3
"""DEV-only ExtraArena web server для тестирования /api/rlhf/* через rlhf_env.

Поднимает /api/rlhf/* (и остальной web/server.py) на 127.0.0.1:8082 против
локальной БД (как живой 8081), НО:
  - НЕ запускает aiogram polling (нет Telegram 409 ConflictError с живым 8081);
  - НЕ запускает миграции схемы (только Database.connect, без init_schema);
  - НЕ запускает фоновые циклы записи в shared-БД (start_background_tasks
    neutralised через app.on_startup.clear());
  - НЕ запускает support-bot polling.

Запуск (зеркалит env живого 8081, который использует DB_* и trust-auth на 5434):
    python3 rlhf_env/dev_game_server.py
или с переопределениями:
    WEBAPP_PORT=8082 DB_NAME=laveqox python3 rlhf_env/dev_game_server.py

load_dotenv (в get_settings) использует override=False → переменные процесса
приоритетнее .env, поэтому живой 8081 (читающий .env) не трогаем. BOT_TOKEN
наследуется из .env и используется ТОЛЬКО для HTTP-фолбэка _rlhf_send_telegram;
синтетические тестовые аккаунты (user_id >= SYNTHETIC_USER_ID_MIN) имеют
telegram_linked=False → request-code идёт во внутриигровую почту, Telegram не
дёргается (web/server.py:7159-7168).
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

# Дефолты env зеркалируют живой 8081 (DB_* + trust-auth на localhost:5434).
# Не перезаписываем уже заданные переменные (setdefault).
_DEFAULTS = {
    "ENVIRONMENT": "development",
    "DB_HOST": "localhost",
    "DB_PORT": "5434",
    "DB_USER": "postgres",
    "DB_PASSWORD": "",
    "DB_NAME": "laveqox",
    "WEBAPP_HOST": "127.0.0.1",
    "WEBAPP_PORT": "8082",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# .env содержит placeholder DATABASE_URL (postgresql://user:password@...),
# который config.py парсит ПЕРЕД fallback на DB_*. Чтобы обойти placeholder,
# выставляем ЯВНЫЕ DSN через setdefault ДО того, как get_settings() позовёт
# load_dotenv(override=False) — тогда load_dotenv не перетрёт уже заданные
# значения, и config.py разберёт нашу реальную локальную БД. Если в окружении
# уже задан настоящий DATABASE_URL/EXTRAID_DATABASE_URL, он приоритетнее
# (setdefault не перезаписывает существующие ключи).
def _dsn(host: str, port: str, user: str, password: str, db: str) -> str:
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"

os.environ.setdefault(
    "DATABASE_URL",
    _dsn(_DEFAULTS["DB_HOST"], _DEFAULTS["DB_PORT"],
         _DEFAULTS["DB_USER"], _DEFAULTS["DB_PASSWORD"], _DEFAULTS["DB_NAME"]),
)
# extraid БД: отдельная база `extraid` на том же кластере (как у живого 8081).
os.environ.setdefault(
    "EXTRAID_DATABASE_URL",
    _dsn(_DEFAULTS["DB_HOST"], _DEFAULTS["DB_PORT"],
         _DEFAULTS["DB_USER"], _DEFAULTS["DB_PASSWORD"], "extraid"),
)

# sys.path для import core.* / infrastructure.* — repo root.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rlhf_env.dev_game_server")


async def main() -> None:
    settings = get_settings()

    # 1) Базы — БЕЗ миграций (connect не мигрирует; миграции только в
    #    main.py::init_schema, который мы не вызываем).
    db = Database(settings.database)
    await db.connect()
    extraid_db = ExtraIDDatabase(settings.extraid_database.dsn)
    await extraid_db.connect()  # idempotent CREATE TABLE IF NOT EXISTS

    # 2) web-приложение с минимальными аргументами, нужными rlhf-хендлерам:
    #    bot_token (читается require_user_id) + extraid_db (503 без него).
    app = create_web_app(
        db,
        settings.bot_token,
        extraid_db=extraid_db,
        webapp_url="http://127.0.0.1:8082",
        extra_shop_url="http://127.0.0.1:8082",
    )

    # 3) Neutralise фоновые циклы записи в shared-БД (match_timer_checker,
    #    _announcement_expiry_loop, _squad_weekly_cbrp_loop,
    #    _rating_snapshot_refresh_loop — web/server.py:21435-21440).
    app.on_startup.clear()
    app.on_cleanup.clear()

    # 4) Старт на дев-порту. НИКОГДА не зовём create_bot()/dp.start_polling()
    #    (единственный источник getUpdates-конфликта 409 с живым 8081).
    port = int(os.getenv("WEBAPP_PORT", "8082"))
    host = os.getenv("WEBAPP_HOST", "127.0.0.1")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    log.warning(
        "DEV rlhf game server on http://%s:%s (no polling, no migrations, "
        "no background tasks) — game DB=%s extraid DB=%s",
        host, port, settings.database.database, settings.extraid_database.database,
    )
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await extraid_db.disconnect()
        await db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass