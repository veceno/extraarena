from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

SYNTHETIC_USER_ID_MIN = 9_100_000_000_000


def _validate_schema_identifier(identifier: str) -> str:
    identifier = str(identifier or "")
    if (
        not identifier
        or identifier[0].isdigit()
        or not all(char.isascii() and (char.isalnum() or char == "_") for char in identifier)
    ):
        raise ValueError(f"invalid_schema_identifier:{identifier!r}")
    return identifier


def _coerce_uuid(value):
    if value is None or isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


class ExtraIDDatabase:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self, min_size: int = 2, max_size: int = 10) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=min_size, max_size=max_size
        )
        await self._initialize()

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ═══════════════════════════════════════════════════════════════════
    # Core query helpers
    # ═══════════════════════════════════════════════════════════════════

    async def execute(self, query: str, *args) -> str:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Optional[Any]:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # ═══════════════════════════════════════════════════════════════════
    # Schema helpers
    # ═══════════════════════════════════════════════════════════════════

    async def _get_columns(self, table: str) -> set[str]:
        rows = await self.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
            """,
            table,
        )
        return {row["column_name"] for row in rows}

    async def _add_column_if_missing(
        self, table: str, existing_columns: set[str], column_definition: str
    ) -> bool:
        column_name = column_definition.split()[0]
        table = _validate_schema_identifier(table)
        column_name = _validate_schema_identifier(column_name)
        if column_name in existing_columns:
            return False
        await self.execute(f"ALTER TABLE {table} ADD COLUMN {column_definition}")
        return True

    async def _initialize(self) -> None:
        await self._ensure_extra_accounts_table()
        await self._ensure_auth_sessions_table()
        await self._ensure_rate_limits_table()
        await self._ensure_bot_auth_codes_table()
        await self._ensure_device_analytics_table()
        await self._ensure_synthetic_user_id_seq()
        await self._migrate_session_fk_on_delete_set_null()

    # ═══════════════════════════════════════════════════════════════════
    # Schema: tables
    # ═══════════════════════════════════════════════════════════════════

    async def _ensure_extra_accounts_table(self) -> None:
        if not await self.fetchval("SELECT to_regclass('public.extra_accounts')"):
            await self.execute("""
                CREATE TABLE extra_accounts (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         BIGINT,
                    display_id      TEXT UNIQUE NOT NULL,
                    email           TEXT NOT NULL,
                    password_hash   TEXT NOT NULL,
                    nickname        TEXT UNIQUE,
                    is_email_verified       BOOLEAN NOT NULL DEFAULT FALSE,
                    email_verify_token      TEXT,
                    email_verify_expires    TIMESTAMPTZ,
                    reg_bonus_claimed       BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deleted_at      TIMESTAMPTZ
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_extra_accounts_user_id ON extra_accounts(user_id)")
            await self.execute("CREATE INDEX IF NOT EXISTS idx_extra_accounts_email ON extra_accounts(LOWER(email))")

        columns = await self._get_columns("extra_accounts")
        await self._add_column_if_missing("extra_accounts", columns, "id UUID PRIMARY KEY DEFAULT gen_random_uuid()")
        await self._add_column_if_missing("extra_accounts", columns, "user_id BIGINT")
        await self._add_column_if_missing("extra_accounts", columns, "display_id TEXT UNIQUE NOT NULL")
        await self._add_column_if_missing("extra_accounts", columns, "email TEXT NOT NULL")
        await self._add_column_if_missing("extra_accounts", columns, "password_hash TEXT NOT NULL")
        await self._add_column_if_missing("extra_accounts", columns, "nickname TEXT UNIQUE")
        await self._add_column_if_missing("extra_accounts", columns, "is_email_verified BOOLEAN NOT NULL DEFAULT FALSE")
        await self._add_column_if_missing("extra_accounts", columns, "email_verify_token TEXT")
        await self._add_column_if_missing("extra_accounts", columns, "email_verify_expires TIMESTAMPTZ")
        await self._add_column_if_missing("extra_accounts", columns, "reg_bonus_claimed BOOLEAN NOT NULL DEFAULT FALSE")
        await self._add_column_if_missing("extra_accounts", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("extra_accounts", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("extra_accounts", columns, "deleted_at TIMESTAMPTZ")

        # Drop legacy plain UNIQUE constraints on email that block reuse after soft-delete.
        # Replaced by a partial unique index scoped to active (non-deleted) rows.
        await self._drop_legacy_email_unique_constraints()

        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_extra_accounts_active_user_id_unique
            ON extra_accounts(user_id)
            WHERE user_id IS NOT NULL AND deleted_at IS NULL
            """
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_extra_accounts_active_email_unique
            ON extra_accounts(LOWER(email))
            WHERE deleted_at IS NULL
            """
        )

    async def _drop_legacy_email_unique_constraints(self) -> None:
        rows = await self.fetch(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'public.extra_accounts'::regclass
              AND contype = 'u'
              AND conname LIKE '%email%'
            """
        )
        for row in rows:
            try:
                await self.execute(f"ALTER TABLE extra_accounts DROP CONSTRAINT IF EXISTS {_validate_schema_identifier(row['conname'])}")
            except Exception:
                logger.warning("Failed to drop legacy email unique constraint %s", row["conname"], exc_info=True)

    async def _ensure_auth_sessions_table(self) -> None:
        if not await self.fetchval("SELECT to_regclass('public.auth_sessions')"):
            await self.execute("""
                CREATE TABLE auth_sessions (
                    session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         BIGINT NOT NULL,
                    auth_method     TEXT NOT NULL,
                    token_hash      TEXT UNIQUE NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at      TIMESTAMPTZ NOT NULL,
                    last_used_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    revoked         BOOLEAN NOT NULL DEFAULT FALSE,
                    revoked_at      TIMESTAMPTZ,
                    device_label    TEXT
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id)")
            await self.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash ON auth_sessions(token_hash)")

        columns = await self._get_columns("auth_sessions")
        await self._add_column_if_missing("auth_sessions", columns, "session_id UUID PRIMARY KEY DEFAULT gen_random_uuid()")
        await self._add_column_if_missing("auth_sessions", columns, "user_id BIGINT NOT NULL")
        await self._add_column_if_missing("auth_sessions", columns, "auth_method TEXT NOT NULL")
        await self._add_column_if_missing("auth_sessions", columns, "token_hash TEXT UNIQUE NOT NULL")
        await self._add_column_if_missing("auth_sessions", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("auth_sessions", columns, "expires_at TIMESTAMPTZ NOT NULL")
        await self._add_column_if_missing("auth_sessions", columns, "last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("auth_sessions", columns, "revoked BOOLEAN NOT NULL DEFAULT FALSE")
        await self._add_column_if_missing("auth_sessions", columns, "revoked_at TIMESTAMPTZ")
        await self._add_column_if_missing("auth_sessions", columns, "device_label TEXT")

    async def _ensure_rate_limits_table(self) -> None:
        if not await self.fetchval("SELECT to_regclass('public.extraid_rate_limits')"):
            await self.execute("""
                CREATE TABLE extraid_rate_limits (
                    key      TEXT PRIMARY KEY,
                    count    INTEGER NOT NULL DEFAULT 0,
                    reset_at TIMESTAMPTZ NOT NULL
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_extraid_rate_limits_reset_at ON extraid_rate_limits(reset_at)")

    async def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        allowed = await self.fetchval(
            """
            WITH upsert AS (
                INSERT INTO extraid_rate_limits (key, count, reset_at)
                VALUES ($1, 1, NOW() + make_interval(secs => $3::int))
                ON CONFLICT (key) DO UPDATE SET
                    count = CASE
                        WHEN extraid_rate_limits.reset_at <= NOW() THEN 1
                        ELSE extraid_rate_limits.count + 1
                    END,
                    reset_at = CASE
                        WHEN extraid_rate_limits.reset_at <= NOW()
                            THEN NOW() + make_interval(secs => $3::int)
                        ELSE extraid_rate_limits.reset_at
                    END
                RETURNING count
            )
            SELECT count <= $2 FROM upsert
            """,
            key,
            int(max_requests),
            int(window_seconds),
        )
        return bool(allowed)

    async def _ensure_bot_auth_codes_table(self) -> None:
        if not await self.fetchval("SELECT to_regclass('public.bot_auth_codes')"):
            await self.execute("""
                CREATE TABLE bot_auth_codes (
                    code            CHAR(6) PRIMARY KEY,
                    user_id         BIGINT NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at      TIMESTAMPTZ NOT NULL,
                    used_at         TIMESTAMPTZ,
                    session_id      UUID REFERENCES auth_sessions(session_id)
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_bot_auth_codes_user_id ON bot_auth_codes(user_id)")

        columns = await self._get_columns("bot_auth_codes")
        await self._add_column_if_missing("bot_auth_codes", columns, "code CHAR(6) PRIMARY KEY")
        await self._add_column_if_missing("bot_auth_codes", columns, "user_id BIGINT NOT NULL")
        await self._add_column_if_missing("bot_auth_codes", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("bot_auth_codes", columns, "expires_at TIMESTAMPTZ NOT NULL")
        await self._add_column_if_missing("bot_auth_codes", columns, "used_at TIMESTAMPTZ")
        await self._add_column_if_missing("bot_auth_codes", columns, "session_id UUID REFERENCES auth_sessions(session_id)")

    async def _ensure_device_analytics_table(self) -> None:
        if not await self.fetchval("SELECT to_regclass('public.device_analytics')"):
            await self.execute("""
                CREATE TABLE device_analytics (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         BIGINT NOT NULL,
                    session_id      UUID REFERENCES auth_sessions(session_id),
                    platform        TEXT NOT NULL,
                    device_model    TEXT,
                    os_name         TEXT,
                    os_version      TEXT,
                    browser_name    TEXT,
                    browser_version TEXT,
                    app_version     TEXT,
                    screen_width    INTEGER,
                    screen_height   INTEGER,
                    device_pixel_ratio FLOAT,
                    locale_language TEXT,
                    locale_region   TEXT,
                    timezone        TEXT,
                    tg_platform     TEXT,
                    tg_version      TEXT,
                    ip_country      TEXT,
                    ip_hash         TEXT,
                    raw_user_agent  TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_device_analytics_user_id ON device_analytics(user_id)")
            await self.execute("CREATE INDEX IF NOT EXISTS idx_device_analytics_platform ON device_analytics(platform)")
            await self.execute("CREATE INDEX IF NOT EXISTS idx_device_analytics_created_at ON device_analytics(created_at)")

        columns = await self._get_columns("device_analytics")
        await self._add_column_if_missing("device_analytics", columns, "id UUID PRIMARY KEY DEFAULT gen_random_uuid()")
        await self._add_column_if_missing("device_analytics", columns, "user_id BIGINT NOT NULL")
        await self._add_column_if_missing("device_analytics", columns, "session_id UUID REFERENCES auth_sessions(session_id)")
        await self._add_column_if_missing("device_analytics", columns, "platform TEXT NOT NULL")
        await self._add_column_if_missing("device_analytics", columns, "device_model TEXT")
        await self._add_column_if_missing("device_analytics", columns, "os_name TEXT")
        await self._add_column_if_missing("device_analytics", columns, "os_version TEXT")
        await self._add_column_if_missing("device_analytics", columns, "browser_name TEXT")
        await self._add_column_if_missing("device_analytics", columns, "browser_version TEXT")
        await self._add_column_if_missing("device_analytics", columns, "app_version TEXT")
        await self._add_column_if_missing("device_analytics", columns, "screen_width INTEGER")
        await self._add_column_if_missing("device_analytics", columns, "screen_height INTEGER")
        await self._add_column_if_missing("device_analytics", columns, "device_pixel_ratio FLOAT")
        await self._add_column_if_missing("device_analytics", columns, "locale_language TEXT")
        await self._add_column_if_missing("device_analytics", columns, "locale_region TEXT")
        await self._add_column_if_missing("device_analytics", columns, "timezone TEXT")
        await self._add_column_if_missing("device_analytics", columns, "tg_platform TEXT")
        await self._add_column_if_missing("device_analytics", columns, "tg_version TEXT")
        await self._add_column_if_missing("device_analytics", columns, "ip_country TEXT")
        await self._add_column_if_missing("device_analytics", columns, "ip_hash TEXT")
        await self._add_column_if_missing("device_analytics", columns, "raw_user_agent TEXT")
        await self._add_column_if_missing("device_analytics", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

    async def _migrate_session_fk_on_delete_set_null(self) -> None:
        for table in ("bot_auth_codes", "device_analytics"):
            rows = await self.fetch(
                """
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = $1::regclass
                  AND contype = 'f'
                  AND conname LIKE '%session_id%'
                """,
                f"public.{table}",
            )
            for row in rows:
                try:
                    await self.execute(
                        f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {_validate_schema_identifier(row['conname'])}"
                    )
                except Exception:
                    logger.warning("Failed to drop session_id FK on %s", table, exc_info=True)
            try:
                await self.execute(
                    f"ALTER TABLE {table} ADD CONSTRAINT {table}_session_id_fkg_on_delete_set_null "
                    f"FOREIGN KEY (session_id) REFERENCES auth_sessions(session_id) ON DELETE SET NULL"
                )
            except Exception:
                logger.warning("Failed to add ON DELETE SET NULL FK on %s.session_id", table, exc_info=True)

    async def _ensure_synthetic_user_id_seq(self) -> None:
        await self.execute("""
            CREATE SEQUENCE IF NOT EXISTS synthetic_user_id_seq
                START WITH 9100000000000
                INCREMENT BY 1
        """)
        try:
            current = await self.fetchval("SELECT last_value FROM synthetic_user_id_seq")
            if current is None or int(current) < SYNTHETIC_USER_ID_MIN - 1:
                await self.fetchval(
                    "SELECT setval('synthetic_user_id_seq', $1, true)",
                    SYNTHETIC_USER_ID_MIN - 1,
                )
        except Exception:
            logger.exception("Failed to normalize synthetic_user_id_seq")

    # ═══════════════════════════════════════════════════════════════════
    # ExtraID account methods
    # ═══════════════════════════════════════════════════════════════════

    async def get_extra_account_by_user_id(self, user_id: int) -> dict | None:
        row = await self.fetchrow(
            "SELECT * FROM extra_accounts WHERE user_id = $1 AND deleted_at IS NULL", user_id
        )
        return dict(row) if row else None

    async def get_any_extra_account_by_user_id(self, user_id: int) -> dict | None:
        row = await self.fetchrow(
            "SELECT * FROM extra_accounts WHERE user_id = $1 ORDER BY deleted_at IS NULL DESC, created_at DESC LIMIT 1",
            user_id,
        )
        return dict(row) if row else None

    async def get_extra_account_by_email(self, email: str) -> dict | None:
        row = await self.fetchrow(
            "SELECT * FROM extra_accounts WHERE LOWER(email) = LOWER($1) AND deleted_at IS NULL", email
        )
        return dict(row) if row else None

    async def get_any_extra_account_by_email(self, email: str) -> dict | None:
        row = await self.fetchrow(
            """
            SELECT *
            FROM extra_accounts
            WHERE LOWER(email) = LOWER($1) AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            email,
        )
        return dict(row) if row else None

    async def has_user_claimed_reg_bonus(self, user_id: int) -> bool:
        row = await self.fetchrow(
            "SELECT 1 FROM extra_accounts WHERE user_id = $1 AND reg_bonus_claimed = TRUE LIMIT 1",
            user_id,
        )
        return row is not None

    async def create_extra_account(
        self, user_id: int, display_id: str, email: str,
        password_hash: str, nickname: str | None = None
    ) -> dict:
        row = await self.fetchrow(
            """
            INSERT INTO extra_accounts (user_id, display_id, email, password_hash, nickname)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            user_id, display_id, email, password_hash, nickname
        )
        return dict(row)

    async def get_synthetic_user_id(self) -> int:
        for _ in range(100):
            user_id = int(await self.fetchval("SELECT nextval('synthetic_user_id_seq')"))
            if user_id < SYNTHETIC_USER_ID_MIN:
                await self.fetchval(
                    "SELECT setval('synthetic_user_id_seq', $1, true)",
                    SYNTHETIC_USER_ID_MIN - 1,
                )
                continue
            try:
                exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
            except Exception:
                exists = None
            if not exists:
                return user_id
        raise RuntimeError("failed_to_allocate_synthetic_user_id")

    async def mark_reg_bonus_claimed(self, extra_account_id) -> bool:
        result = await self.execute(
            """
            UPDATE extra_accounts SET reg_bonus_claimed = TRUE
            WHERE id = $1 AND reg_bonus_claimed = FALSE
            """,
            _coerce_uuid(extra_account_id)
        )
        return result != "UPDATE 0"

    async def soft_delete_extra_account(self, extra_account_id) -> None:
        await self.execute(
            """
            UPDATE extra_accounts
            SET deleted_at = NOW(),
                email = email || '#deleted-' || id::text,
                updated_at = NOW()
            WHERE id = $1
            """,
            _coerce_uuid(extra_account_id)
        )

    # ═══════════════════════════════════════════════════════════════════
    # Auth sessions
    # ═══════════════════════════════════════════════════════════════════

    async def create_auth_session(
        self, user_id: int, auth_method: str, token_hash: str,
        expires_at, device_label: str | None = None, session_id=None
    ) -> dict:
        # Если session_id передан (напр. RLHF-верификация ссылает на него
        # bot_auth_codes.session_id по FK), вставляем с явным session_id —
        # иначе БД генерирует свой (поведение прежних вызовов не меняется).
        if session_id is not None:
            row = await self.fetchrow(
                """
                INSERT INTO auth_sessions (session_id, user_id, auth_method, token_hash, expires_at, device_label)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING session_id, created_at
                """,
                session_id, user_id, auth_method, token_hash, expires_at, device_label
            )
        else:
            row = await self.fetchrow(
                """
                INSERT INTO auth_sessions (user_id, auth_method, token_hash, expires_at, device_label)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING session_id, created_at
                """,
                user_id, auth_method, token_hash, expires_at, device_label
            )
        return dict(row) if row else {}

    async def verify_session(self, session_id, token: str) -> dict | None:
        try:
            session_id = _coerce_uuid(session_id)
        except (ValueError, TypeError, AttributeError):
            return None
        row = await self.fetchrow(
            """
            SELECT s.*
            FROM auth_sessions s
            LEFT JOIN extra_accounts ea ON ea.user_id = s.user_id
            WHERE s.session_id = $1
              AND s.revoked = FALSE
              AND s.expires_at > NOW()
              AND (ea.id IS NULL OR ea.deleted_at IS NULL)
            """,
            session_id
        )
        if not row:
            return None
        if hashlib.sha256(token.encode()).hexdigest() != row["token_hash"]:
            return None
        await self.execute(
            "UPDATE auth_sessions SET last_used_at = NOW() WHERE session_id = $1",
            session_id
        )
        return dict(row)

    async def revoke_session(self, session_id) -> bool:
        try:
            session_id = _coerce_uuid(session_id)
        except (ValueError, TypeError, AttributeError):
            return False
        result = await self.execute(
            """
            UPDATE auth_sessions SET revoked = TRUE, revoked_at = NOW()
            WHERE session_id = $1 AND revoked = FALSE
            """,
            session_id
        )
        return result != "UPDATE 0"

    async def revoke_all_user_sessions(self, user_id: int) -> None:
        await self.execute(
            """
            UPDATE auth_sessions SET revoked = TRUE, revoked_at = NOW()
            WHERE user_id = $1 AND revoked = FALSE
            """,
            user_id
        )

    async def get_user_sessions(self, user_id: int) -> list[dict]:
        rows = await self.fetch(
            """
            SELECT session_id, auth_method, created_at, last_used_at, device_label
            FROM auth_sessions
            WHERE user_id = $1 AND revoked = FALSE AND expires_at > NOW()
            ORDER BY last_used_at DESC
            """,
            user_id
        )
        return [dict(r) for r in rows]

    async def cleanup_expired_sessions(self) -> int:
        result = await self.execute(
            "DELETE FROM auth_sessions WHERE expires_at < NOW() OR (revoked = TRUE AND revoked_at < NOW() - INTERVAL '7 days')"
        )
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    async def cleanup_old_rate_limits(self) -> int:
        result = await self.execute("DELETE FROM extraid_rate_limits WHERE reset_at < NOW()")
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    async def cleanup_used_bot_codes(self) -> int:
        result = await self.execute("DELETE FROM bot_auth_codes WHERE used_at IS NOT NULL OR expires_at < NOW()")
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    # ═══════════════════════════════════════════════════════════════════
    # Bot auth codes
    # ═══════════════════════════════════════════════════════════════════

    async def create_bot_auth_code(self, code: str, user_id: int) -> dict:
        row = await self.fetchrow(
            """
            INSERT INTO bot_auth_codes (code, user_id, expires_at)
            VALUES ($1, $2, NOW() + INTERVAL '5 minutes')
            RETURNING code, expires_at
            """,
            code, user_id
        )
        return dict(row) if row else {}

    async def verify_bot_auth_code(self, code: str) -> dict | None:
        row = await self.fetchrow(
            """
            SELECT * FROM bot_auth_codes
            WHERE code = $1 AND used_at IS NULL AND expires_at > NOW()
            """,
            code
        )
        return dict(row) if row else None

    async def consume_bot_auth_code(self, code: str) -> dict | None:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE bot_auth_codes
                    SET used_at = NOW()
                    WHERE code = $1 AND used_at IS NULL AND expires_at > NOW()
                    RETURNING *
                    """,
                    code,
                )
                return dict(row) if row else None

    async def mark_bot_code_used(self, code: str, session_id) -> None:
        await self.execute(
            """
            UPDATE bot_auth_codes SET session_id = $1
            WHERE code = $2 AND used_at IS NOT NULL
            """,
            _coerce_uuid(session_id), code
        )

    async def cleanup_old_bot_codes(self, user_id: int) -> None:
        await self.execute(
            "DELETE FROM bot_auth_codes WHERE user_id = $1 AND used_at IS NULL",
            user_id
        )

    # ═══════════════════════════════════════════════════════════════════
    # Device analytics
    # ═══════════════════════════════════════════════════════════════════

    async def log_device_analytics(
        self, user_id: int, platform: str, session_id=None,
        device_model: str | None = None, os_name: str | None = None,
        os_version: str | None = None, browser_name: str | None = None,
        browser_version: str | None = None, app_version: str | None = None,
        screen_width: int | None = None, screen_height: int | None = None,
        device_pixel_ratio: float | None = None, locale_language: str | None = None,
        locale_region: str | None = None, timezone: str | None = None,
        tg_platform: str | None = None, tg_version: str | None = None,
        ip_country: str | None = None, ip_hash: str | None = None,
        raw_user_agent: str | None = None
    ) -> None:
        try:
            session_id = _coerce_uuid(session_id) if session_id else None
            await self.execute(
                """
                INSERT INTO device_analytics (
                    user_id, session_id, platform, device_model, os_name, os_version,
                    browser_name, browser_version, app_version, screen_width, screen_height,
                    device_pixel_ratio, locale_language, locale_region, timezone,
                    tg_platform, tg_version, ip_country, ip_hash, raw_user_agent
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17, $18, $19, $20
                )
                """,
                user_id, session_id, platform, device_model, os_name, os_version,
                browser_name, browser_version, app_version, screen_width, screen_height,
                device_pixel_ratio, locale_language, locale_region, timezone,
                tg_platform, tg_version, ip_country, ip_hash, raw_user_agent
            )
        except Exception:
            logger.warning("log_device_analytics failed", exc_info=True)
