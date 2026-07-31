from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)

SYNTHETIC_USER_ID_MIN = 9_100_000_000_000
BOT_AUTH_CODE_MAX_ATTEMPTS = 5
BOT_AUTH_CODE_DEFAULT_PURPOSE = "telegram_transfer"
ACCOUNT_ACTION_TOKEN_MAX_ATTEMPTS = 5
EMAIL_OUTBOX_MAX_ATTEMPTS = 24
EXTRAID_SCHEMA_VERSION = 5
_TOKEN_PURPOSE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


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


def _normalize_token_purpose(value: str) -> str:
    purpose = str(value or "").strip().lower()
    if not _TOKEN_PURPOSE_RE.fullmatch(purpose):
        raise ValueError("invalid_token_purpose")
    return purpose


def _bot_auth_code_hash(code: str, purpose: str) -> str:
    """Return a keyed lookup digest so a DB leak does not expose six-digit codes."""
    purpose = _normalize_token_purpose(purpose)
    secret = get_settings().jwt_secret.encode("utf-8")
    message = f"extraarena-bot-code:{purpose}:{str(code or '').strip()}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _account_email_code_hash(code: str, email: str, token_id) -> str:
    """Return a keyed digest so a token-table leak cannot reveal six-digit codes."""
    normalized = str(code or "").strip()
    if not re.fullmatch(r"\d{6}", normalized):
        raise ValueError("invalid_email_verification_code")
    normalized_email = str(email or "").strip().lower()
    if not normalized_email:
        raise ValueError("email_required")
    normalized_token_id = str(_coerce_uuid(token_id))
    secret = get_settings().jwt_secret.encode("utf-8")
    message = (
        "extraarena-email-code:verify_email:"
        f"{normalized_email}:{normalized_token_id}:{normalized}"
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


class ExtraIDDatabase:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None
        self._migration_conn: Any | None = None

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
        if self._migration_conn is not None:
            return await self._migration_conn.execute(query, *args)
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        if self._migration_conn is not None:
            return await self._migration_conn.fetch(query, *args)
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        if self._migration_conn is not None:
            return await self._migration_conn.fetchrow(query, *args)
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Optional[Any]:
        if self._migration_conn is not None:
            return await self._migration_conn.fetchval(query, *args)
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
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as migration_conn:
            await migration_conn.execute(
                "SELECT pg_advisory_lock(hashtext('extraid_schema_migration'))"
            )
            try:
                await migration_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS extraid_schema_meta (
                        id         SMALLINT PRIMARY KEY DEFAULT 1,
                        version    INTEGER NOT NULL,
                        state      TEXT NOT NULL,
                        error      TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                stored_version = await migration_conn.fetchval(
                    "SELECT version FROM extraid_schema_meta WHERE id = 1"
                )
                if (
                    stored_version is not None
                    and int(stored_version) > EXTRAID_SCHEMA_VERSION
                ):
                    raise RuntimeError(
                        "schema_newer_than_code:"
                        f"{int(stored_version)}>{EXTRAID_SCHEMA_VERSION}"
                    )
                await migration_conn.execute(
                    """
                    INSERT INTO extraid_schema_meta (id, version, state)
                    VALUES (1, $1, 'migrating')
                    ON CONFLICT (id) DO UPDATE
                    SET state = 'migrating', error = NULL, updated_at = NOW()
                    """,
                    EXTRAID_SCHEMA_VERSION,
                )
                try:
                    self._migration_conn = migration_conn
                    async with migration_conn.transaction():
                        await self._initialize_locked()
                except Exception as exc:
                    await migration_conn.execute(
                        """
                        UPDATE extraid_schema_meta
                        SET state = 'failed', error = $1, updated_at = NOW()
                        WHERE id = 1
                        """,
                        str(exc)[:1000],
                    )
                    raise
                finally:
                    self._migration_conn = None
                await migration_conn.execute(
                    """
                    UPDATE extraid_schema_meta
                    SET version = $1, state = 'ready', error = NULL, updated_at = NOW()
                    WHERE id = 1
                    """,
                    EXTRAID_SCHEMA_VERSION,
                )
            finally:
                await migration_conn.execute(
                    "SELECT pg_advisory_unlock(hashtext('extraid_schema_migration'))"
                )

    async def _initialize_locked(self) -> None:
        await self._ensure_extra_accounts_table()
        await self._ensure_identity_bindings_table()
        await self._ensure_auth_sessions_table()
        await self._ensure_anonymous_auth_bootstraps_table()
        await self._ensure_rate_limits_table()
        await self._ensure_bot_auth_codes_table()
        await self._ensure_account_action_tokens_table()
        await self._ensure_email_outbox_table()
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
                    nickname        TEXT,
                    is_email_verified       BOOLEAN NOT NULL DEFAULT FALSE,
                    email_verification_required BOOLEAN NOT NULL DEFAULT TRUE,
                    verification_source     TEXT NOT NULL DEFAULT 'pending',
                    email_verify_token      TEXT,
                    email_verify_expires    TIMESTAMPTZ,
                    reg_bonus_claimed       BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deleted_at      TIMESTAMPTZ,
                    deletion_state  TEXT,
                    link_state      TEXT,
                    link_previous_user_id BIGINT,
                    registration_origin TEXT NOT NULL DEFAULT 'standalone',
                    pending_expires_at TIMESTAMPTZ
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
        await self._add_column_if_missing(
            "extra_accounts",
            columns,
            "email_verification_required BOOLEAN NOT NULL DEFAULT FALSE",
        )
        await self._add_column_if_missing(
            "extra_accounts",
            columns,
            "verification_source TEXT NOT NULL DEFAULT 'legacy_migration'",
        )
        await self._add_column_if_missing("extra_accounts", columns, "email_verify_token TEXT")
        await self._add_column_if_missing("extra_accounts", columns, "email_verify_expires TIMESTAMPTZ")
        await self._add_column_if_missing("extra_accounts", columns, "reg_bonus_claimed BOOLEAN NOT NULL DEFAULT FALSE")
        await self._add_column_if_missing("extra_accounts", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("extra_accounts", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("extra_accounts", columns, "deleted_at TIMESTAMPTZ")
        await self._add_column_if_missing("extra_accounts", columns, "deletion_state TEXT")
        await self._add_column_if_missing("extra_accounts", columns, "link_state TEXT")
        await self._add_column_if_missing(
            "extra_accounts", columns, "link_previous_user_id BIGINT"
        )
        await self._add_column_if_missing(
            "extra_accounts",
            columns,
            "registration_origin TEXT NOT NULL DEFAULT 'legacy'",
        )
        await self._add_column_if_missing(
            "extra_accounts", columns, "pending_expires_at TIMESTAMPTZ"
        )

        duplicate_user = await self.fetchrow(
            """
            SELECT user_id, COUNT(*) AS row_count
            FROM extra_accounts
            WHERE user_id IS NOT NULL AND deleted_at IS NULL
            GROUP BY user_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
        if duplicate_user:
            raise RuntimeError(
                f"duplicate_active_extraid_user:{duplicate_user['user_id']}"
            )
        duplicate_email = await self.fetchrow(
            """
            SELECT LOWER(email) AS normalized_email, COUNT(*) AS row_count
            FROM extra_accounts
            WHERE deleted_at IS NULL
            GROUP BY LOWER(email)
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
        if duplicate_email:
            raise RuntimeError(
                "duplicate_active_extraid_email:"
                f"{duplicate_email['normalized_email']}"
            )
        duplicate_nickname = await self.fetchrow(
            """
            SELECT LOWER(nickname) AS normalized_nickname, COUNT(*) AS row_count
            FROM extra_accounts
            WHERE deleted_at IS NULL AND nickname IS NOT NULL
            GROUP BY LOWER(nickname)
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
        if duplicate_nickname:
            raise RuntimeError(
                "duplicate_active_extraid_nicknames:"
                f"{duplicate_nickname['normalized_nickname']}"
            )

        # Only after all duplicate preflights pass may migration remove the
        # legacy constraints and replace them with active/case-insensitive ones.
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
        await self._drop_legacy_nickname_unique_constraints()
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_extra_accounts_active_nickname_ci_unique
            ON extra_accounts(LOWER(nickname))
            WHERE nickname IS NOT NULL AND deleted_at IS NULL
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
            await self.execute(
                f"ALTER TABLE extra_accounts DROP CONSTRAINT IF EXISTS "
                f"{_validate_schema_identifier(row['conname'])}"
            )

    async def _drop_legacy_nickname_unique_constraints(self) -> None:
        rows = await self.fetch(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'public.extra_accounts'::regclass
              AND contype = 'u'
              AND conname LIKE '%nickname%'
            """
        )
        for row in rows:
            constraint_name = _validate_schema_identifier(row["conname"])
            await self.execute(
                f"ALTER TABLE extra_accounts DROP CONSTRAINT IF EXISTS {constraint_name}"
            )

    async def get_active_identity_conflict_report(
        self,
        limit: int = 100,
    ) -> dict[str, list[dict]]:
        """Read-only migration report; never chooses a winner or mutates game state."""
        limit = max(1, min(int(limit), 1000))
        duplicate_users = await self.fetch(
            """
            SELECT
                user_id,
                COUNT(*) AS row_count,
                ARRAY_AGG(id ORDER BY created_at, id) AS account_ids
            FROM extra_accounts
            WHERE user_id IS NOT NULL AND deleted_at IS NULL
            GROUP BY user_id
            HAVING COUNT(*) > 1
            ORDER BY user_id
            LIMIT $1
            """,
            limit,
        )
        duplicate_emails = await self.fetch(
            """
            SELECT
                LOWER(email) AS normalized_email,
                COUNT(*) AS row_count,
                ARRAY_AGG(id ORDER BY created_at, id) AS account_ids,
                ARRAY_AGG(DISTINCT user_id) AS user_ids
            FROM extra_accounts
            WHERE deleted_at IS NULL
            GROUP BY LOWER(email)
            HAVING COUNT(*) > 1
            ORDER BY LOWER(email)
            LIMIT $1
            """,
            limit,
        )
        duplicate_nicknames = await self.fetch(
            """
            SELECT
                LOWER(nickname) AS normalized_nickname,
                COUNT(*) AS row_count,
                ARRAY_AGG(id ORDER BY created_at, id) AS account_ids,
                ARRAY_AGG(DISTINCT user_id) AS user_ids
            FROM extra_accounts
            WHERE deleted_at IS NULL AND nickname IS NOT NULL
            GROUP BY LOWER(nickname)
            HAVING COUNT(*) > 1
            ORDER BY LOWER(nickname)
            LIMIT $1
            """,
            limit,
        )
        return {
            "duplicate_users": [dict(row) for row in duplicate_users],
            "duplicate_emails": [dict(row) for row in duplicate_emails],
            "duplicate_nicknames": [dict(row) for row in duplicate_nicknames],
        }

    async def _ensure_identity_bindings_table(self) -> None:
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS extra_account_identity_bindings (
                provider         TEXT NOT NULL,
                subject          TEXT NOT NULL,
                extra_account_id UUID NOT NULL REFERENCES extra_accounts(id) ON DELETE CASCADE,
                user_id          BIGINT NOT NULL,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (provider, subject),
                UNIQUE (provider, extra_account_id)
            )
            """
        )
        # Existing deployments used this numeric split. Backfill it once, then all
        # security decisions use this immutable ledger instead of re-evaluating IDs.
        await self.execute(
            """
            INSERT INTO extra_account_identity_bindings (
                provider, subject, extra_account_id, user_id
            )
            SELECT
                CASE
                    WHEN user_id < $1 THEN 'telegram'
                    ELSE 'synthetic_user'
                END,
                user_id::text,
                id,
                user_id
            FROM extra_accounts
            WHERE user_id IS NOT NULL AND deleted_at IS NULL
            ON CONFLICT DO NOTHING
            """,
            SYNTHETIC_USER_ID_MIN,
        )
        unbound = await self.fetchrow(
            """
            SELECT a.id
            FROM extra_accounts a
            LEFT JOIN extra_account_identity_bindings b
              ON b.extra_account_id = a.id
            WHERE a.deleted_at IS NULL
              AND a.user_id IS NOT NULL
              AND b.extra_account_id IS NULL
            LIMIT 1
            """
        )
        if unbound:
            raise RuntimeError(f"active_extraid_without_identity_binding:{unbound['id']}")

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

    async def _ensure_anonymous_auth_bootstraps_table(self) -> None:
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS anonymous_auth_bootstraps (
                bootstrap_id     TEXT PRIMARY KEY,
                secret_hash      TEXT NOT NULL,
                user_id          BIGINT NOT NULL,
                session_id       UUID REFERENCES auth_sessions(session_id) ON DELETE SET NULL,
                generation       BIGINT NOT NULL DEFAULT 0,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_used_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                disabled_at      TIMESTAMPTZ
            )
            """
        )
        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_anonymous_auth_bootstraps_user_id
            ON anonymous_auth_bootstraps(user_id)
            """
        )

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
                    code_hash       TEXT NOT NULL,
                    user_id         BIGINT NOT NULL,
                    purpose         TEXT NOT NULL DEFAULT 'telegram_transfer',
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at      TIMESTAMPTZ NOT NULL,
                    used_at         TIMESTAMPTZ,
                    session_id      UUID REFERENCES auth_sessions(session_id)
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_bot_auth_codes_user_id ON bot_auth_codes(user_id)")

        columns = await self._get_columns("bot_auth_codes")
        await self._add_column_if_missing("bot_auth_codes", columns, "code CHAR(6) PRIMARY KEY")
        await self._add_column_if_missing("bot_auth_codes", columns, "code_hash TEXT")
        await self._add_column_if_missing("bot_auth_codes", columns, "user_id BIGINT NOT NULL")
        await self._add_column_if_missing(
            "bot_auth_codes", columns, "purpose TEXT NOT NULL DEFAULT 'telegram_transfer'"
        )
        await self._add_column_if_missing(
            "bot_auth_codes", columns, "failed_attempts INTEGER NOT NULL DEFAULT 0"
        )
        await self._add_column_if_missing("bot_auth_codes", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self._add_column_if_missing("bot_auth_codes", columns, "expires_at TIMESTAMPTZ NOT NULL")
        await self._add_column_if_missing("bot_auth_codes", columns, "used_at TIMESTAMPTZ")
        await self._add_column_if_missing("bot_auth_codes", columns, "session_id UUID REFERENCES auth_sessions(session_id)")

        # Legacy rows stored the six-digit secret in plaintext. They are short-lived,
        # so invalidate them during the migration rather than copying weak secrets.
        await self.execute("DELETE FROM bot_auth_codes WHERE code_hash IS NULL")
        await self.execute("ALTER TABLE bot_auth_codes ALTER COLUMN code_hash SET NOT NULL")
        await self.execute(
            "DROP INDEX IF EXISTS idx_bot_auth_codes_purpose_hash_unique"
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_bot_auth_codes_user_purpose_hash_unique
            ON bot_auth_codes(user_id, purpose, code_hash)
            """
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_bot_auth_codes_one_active_per_purpose
            ON bot_auth_codes(user_id, purpose)
            WHERE used_at IS NULL
            """
        )

    async def _ensure_account_action_tokens_table(self) -> None:
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS account_action_tokens (
                id               UUID PRIMARY KEY,
                extra_account_id UUID NOT NULL REFERENCES extra_accounts(id) ON DELETE CASCADE,
                purpose          TEXT NOT NULL,
                token_hash       TEXT NOT NULL,
                email_snapshot   TEXT NOT NULL,
                attempts         INTEGER NOT NULL DEFAULT 0,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at       TIMESTAMPTZ NOT NULL,
                consumed_at      TIMESTAMPTZ
            )
            """
        )
        columns = await self._get_columns("account_action_tokens")
        await self._add_column_if_missing(
            "account_action_tokens",
            columns,
            "email_snapshot TEXT",
        )
        await self.execute(
            """
            UPDATE account_action_tokens t
            SET email_snapshot = a.email
            FROM extra_accounts a
            WHERE t.extra_account_id = a.id AND t.email_snapshot IS NULL
            """
        )
        await self.execute(
            "ALTER TABLE account_action_tokens ALTER COLUMN email_snapshot SET NOT NULL"
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_account_action_tokens_hash_unique
            ON account_action_tokens(purpose, token_hash)
            """
        )
        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_account_action_tokens_account_purpose
            ON account_action_tokens(extra_account_id, purpose, created_at DESC)
            """
        )

    async def _ensure_email_outbox_table(self) -> None:
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS extraid_email_outbox (
                id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                extra_account_id UUID NOT NULL REFERENCES extra_accounts(id) ON DELETE CASCADE,
                purpose          TEXT NOT NULL,
                email_snapshot   TEXT NOT NULL,
                attempts         INTEGER NOT NULL DEFAULT 0,
                available_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                claimed_at       TIMESTAMPTZ,
                expires_at       TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
                last_error       TEXT,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_extraid_email_outbox_one_pending
            ON extraid_email_outbox(extra_account_id, purpose)
            """
        )
        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_extraid_email_outbox_due
            ON extraid_email_outbox(available_at, created_at)
            """
        )

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
                await self.execute(
                    f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS "
                    f"{_validate_schema_identifier(row['conname'])}"
                )
            await self.execute(
                f"ALTER TABLE {table} ADD CONSTRAINT {table}_session_id_fkg_on_delete_set_null "
                f"FOREIGN KEY (session_id) REFERENCES auth_sessions(session_id) ON DELETE SET NULL"
            )

    async def _ensure_synthetic_user_id_seq(self) -> None:
        await self.execute("""
            CREATE SEQUENCE IF NOT EXISTS synthetic_user_id_seq
                START WITH 9100000000000
                INCREMENT BY 1
        """)
        current = await self.fetchval("SELECT last_value FROM synthetic_user_id_seq")
        if current is None or int(current) < SYNTHETIC_USER_ID_MIN - 1:
            await self.fetchval(
                "SELECT setval('synthetic_user_id_seq', $1, true)",
                SYNTHETIC_USER_ID_MIN - 1,
            )

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
        password_hash: str, nickname: str | None = None,
        *,
        identity_provider: str,
        identity_subject: str,
        registration_origin: str = "standalone",
    ) -> dict:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        provider = _normalize_token_purpose(identity_provider)
        subject = str(identity_subject).strip()
        if not subject:
            raise ValueError("identity_subject_required")
        registration_origin = _normalize_token_purpose(registration_origin)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO extra_accounts (
                        user_id, display_id, email, password_hash, nickname,
                        email_verification_required, verification_source,
                        registration_origin, pending_expires_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, TRUE, 'pending',
                        $6, NOW() + INTERVAL '48 hours'
                    )
                    RETURNING *
                    """,
                    user_id,
                    display_id,
                    email,
                    password_hash,
                    nickname,
                    registration_origin,
                )
                await conn.execute(
                    """
                    INSERT INTO extra_account_identity_bindings (
                        provider, subject, extra_account_id, user_id
                    )
                    VALUES ($1, $2, $3, $4)
                    """,
                    provider,
                    subject,
                    row["id"],
                    int(user_id),
                )
                return dict(row)

    async def account_has_identity_provider(
        self,
        extra_account_id,
        provider: str,
    ) -> bool:
        return bool(
            await self.fetchval(
                """
                SELECT 1
                FROM extra_account_identity_bindings
                WHERE extra_account_id = $1 AND provider = $2
                LIMIT 1
                """,
                _coerce_uuid(extra_account_id),
                _normalize_token_purpose(provider),
            )
        )

    async def change_unverified_email(
        self,
        extra_account_id,
        *,
        new_email: str,
        expected_old_email: str,
    ) -> str:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        new_email = str(new_email).strip().lower()
        expected_old_email = str(expected_old_email).strip().lower()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    str(extra_account_id),
                )
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    new_email,
                )
                duplicate = await conn.fetchval(
                    """
                    SELECT 1
                    FROM extra_accounts
                    WHERE LOWER(email) = LOWER($1)
                      AND id <> $2
                      AND deleted_at IS NULL
                    LIMIT 1
                    """,
                    new_email,
                    extra_account_id,
                )
                if duplicate:
                    return "email_taken"
                updated = await conn.fetchrow(
                    """
                    UPDATE extra_accounts
                    SET email = $1,
                        updated_at = NOW(),
                        verification_source = 'pending'
                    WHERE id = $2
                      AND LOWER(email) = LOWER($3)
                      AND is_email_verified = FALSE
                      AND email_verification_required = TRUE
                      AND deleted_at IS NULL
                    RETURNING *
                    """,
                    new_email,
                    extra_account_id,
                    expected_old_email,
                )
                if not updated:
                    return "email_change_not_allowed"
                await conn.execute(
                    """
                    UPDATE account_action_tokens
                    SET consumed_at = COALESCE(consumed_at, NOW())
                    WHERE extra_account_id = $1
                      AND purpose IN ('verify_email', 'password_reset')
                      AND consumed_at IS NULL
                    """,
                    extra_account_id,
                )
                return "changed"

    async def change_unverified_email_and_enqueue_verification(
        self,
        extra_account_id,
        *,
        new_email: str,
        expected_old_email: str,
    ) -> str:
        """Change a pending address and replace its verification work atomically.

        A fresh outbox row gets a new id.  Consequently a worker that leased the
        old row before this transaction cannot later acknowledge/delete the new
        delivery.  All links and codes addressed to the previous email become
        unusable in the same commit as the address change.
        """
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        new_email = str(new_email).strip().lower()
        expected_old_email = str(expected_old_email).strip().lower()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # This is the same per-account lock used by token issuance and
                # the delivery guard.  It serializes a change with an email that
                # is already being prepared or handed to the provider.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    str(extra_account_id),
                )
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    new_email,
                )
                duplicate = await conn.fetchval(
                    """
                    SELECT 1
                    FROM extra_accounts
                    WHERE LOWER(email) = LOWER($1)
                      AND id <> $2
                      AND deleted_at IS NULL
                    LIMIT 1
                    """,
                    new_email,
                    extra_account_id,
                )
                if duplicate:
                    return "email_taken"
                updated = await conn.fetchrow(
                    """
                    UPDATE extra_accounts
                    SET email = $1,
                        updated_at = NOW(),
                        verification_source = 'pending'
                    WHERE id = $2
                      AND LOWER(email) = LOWER($3)
                      AND is_email_verified = FALSE
                      AND email_verification_required = TRUE
                      AND deleted_at IS NULL
                    RETURNING id
                    """,
                    new_email,
                    extra_account_id,
                    expected_old_email,
                )
                if not updated:
                    return "email_change_not_allowed"
                await conn.execute(
                    """
                    UPDATE account_action_tokens
                    SET consumed_at = COALESCE(consumed_at, NOW())
                    WHERE extra_account_id = $1
                      AND purpose IN (
                          'verify_email',
                          'cancel_registration',
                          'password_reset'
                      )
                      AND consumed_at IS NULL
                    """,
                    extra_account_id,
                )
                # Never update the old row in place: a worker may still hold
                # its id and complete it after this transaction commits.
                await conn.execute(
                    """
                    DELETE FROM extraid_email_outbox
                    WHERE extra_account_id = $1
                      AND purpose IN ('verify_email', 'password_reset')
                    """,
                    extra_account_id,
                )
                await conn.execute(
                    """
                    INSERT INTO extraid_email_outbox (
                        extra_account_id, purpose, email_snapshot
                    )
                    VALUES ($1, 'verify_email', $2)
                    """,
                    extra_account_id,
                    new_email,
                )
                return "changed"

    async def get_synthetic_user_id(self) -> int:
        user_id = int(await self.fetchval("SELECT nextval('synthetic_user_id_seq')"))
        if user_id < SYNTHETIC_USER_ID_MIN:
            await self.fetchval(
                "SELECT setval('synthetic_user_id_seq', $1, true)",
                SYNTHETIC_USER_ID_MIN - 1,
            )
            user_id = int(await self.fetchval("SELECT nextval('synthetic_user_id_seq')"))
        return user_id

    async def mark_reg_bonus_claimed(self, extra_account_id) -> bool:
        result = await self.execute(
            """
            UPDATE extra_accounts SET reg_bonus_claimed = TRUE
            WHERE id = $1 AND reg_bonus_claimed = FALSE
            """,
            _coerce_uuid(extra_account_id)
        )
        return result != "UPDATE 0"

    async def link_extra_account_to_user(
        self,
        extra_account_id,
        old_user_id: int,
        new_user_id: int,
    ) -> str:
        """Atomically CAS ownership and revoke every session belonging to the old owner."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Serialize all link attempts targeting the same provider subject.
                await conn.execute("SELECT pg_advisory_xact_lock($1)", int(new_user_id))
                prior = await conn.fetchrow(
                    """
                    SELECT id
                    FROM extra_accounts
                    WHERE user_id = $1 AND id <> $2
                    LIMIT 1
                    """,
                    int(new_user_id),
                    extra_account_id,
                )
                if prior:
                    return "target_already_linked"
                row = await conn.fetchrow(
                    """
                    UPDATE extra_accounts
                    SET user_id = $1,
                        link_state = 'pending_primary',
                        link_previous_user_id = $3,
                        updated_at = NOW()
                    WHERE id = $2
                      AND user_id = $3
                      AND deleted_at IS NULL
                    RETURNING id
                    """,
                    int(new_user_id),
                    extra_account_id,
                    int(old_user_id),
                )
                if not row:
                    return "ownership_changed"
                await conn.execute(
                    """
                    INSERT INTO extra_account_identity_bindings (
                        provider, subject, extra_account_id, user_id
                    )
                    VALUES ('telegram', $1, $2, $3)
                    """,
                    str(int(new_user_id)),
                    extra_account_id,
                    int(new_user_id),
                )
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id = $1 AND revoked = FALSE
                    """,
                    int(old_user_id),
                )
                return "linked"

    async def rollback_extra_account_link(
        self,
        extra_account_id,
        expected_user_id: int,
        restore_user_id: int,
    ) -> bool:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    """
                    UPDATE extra_accounts
                    SET user_id = $1,
                        link_state = NULL,
                        link_previous_user_id = NULL,
                        updated_at = NOW()
                    WHERE id = $2 AND user_id = $3 AND deleted_at IS NULL
                    """,
                    int(restore_user_id),
                    extra_account_id,
                    int(expected_user_id),
                )
                if result == "UPDATE 0":
                    return False
                await conn.execute(
                    """
                    DELETE FROM extra_account_identity_bindings
                    WHERE provider = 'telegram'
                      AND subject = $1
                      AND extra_account_id = $2
                    """,
                    str(int(expected_user_id)),
                    extra_account_id,
                )
                return True

    async def complete_extra_account_link(self, extra_account_id, expected_user_id: int) -> bool:
        result = await self.execute(
            """
            UPDATE extra_accounts
            SET link_state = NULL,
                link_previous_user_id = NULL,
                updated_at = NOW()
            WHERE id = $1
              AND user_id = $2
              AND link_state IN ('pending_primary', 'reconcile_required')
              AND deleted_at IS NULL
            """,
            _coerce_uuid(extra_account_id),
            int(expected_user_id),
        )
        return result != "UPDATE 0"

    async def mark_primary_link_reconcile_required(
        self,
        extra_account_id,
        *,
        user_id: int,
    ) -> None:
        """Fail closed when the primary DB owner conflicts after credential creation."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE extra_accounts
                    SET link_state = 'primary_owner_reconcile',
                        updated_at = NOW()
                    WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL
                    """,
                    extra_account_id,
                    int(user_id),
                )
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id = $1 AND revoked = FALSE
                    """,
                    int(user_id),
                )

    async def complete_primary_link_reconciliation(
        self,
        extra_account_id,
        *,
        user_id: int,
    ) -> bool:
        result = await self.execute(
            """
            UPDATE extra_accounts
            SET link_state = NULL,
                updated_at = NOW()
            WHERE id = $1
              AND user_id = $2
              AND link_state = 'primary_owner_reconcile'
              AND deleted_at IS NULL
            """,
            _coerce_uuid(extra_account_id),
            int(user_id),
        )
        return result != "UPDATE 0"

    async def mark_extra_account_link_reconcile_required(
        self,
        extra_account_id,
        *,
        old_user_id: int,
        new_user_id: int,
    ) -> None:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE extra_accounts
                    SET link_state = 'reconcile_required',
                        link_previous_user_id = $2,
                        updated_at = NOW()
                    WHERE id = $1 AND deleted_at IS NULL
                    """,
                    extra_account_id,
                    int(old_user_id),
                )
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id IN ($1, $2) AND revoked = FALSE
                    """,
                    int(old_user_id),
                    int(new_user_id),
                )

    async def get_pending_identity_reconciliations(self, limit: int = 100) -> list[dict]:
        rows = await self.fetch(
            """
            SELECT id, user_id, link_previous_user_id, link_state
            FROM extra_accounts
            WHERE link_state IN (
                'pending_primary',
                'reconcile_required',
                'primary_owner_reconcile'
            )
              AND deleted_at IS NULL
            ORDER BY updated_at
            LIMIT $1
            """,
            max(1, min(int(limit), 1000)),
        )
        return [dict(row) for row in rows]

    async def soft_delete_extra_account(self, extra_account_id) -> None:
        await self.execute(
            """
            UPDATE extra_accounts
            SET deleted_at = NOW(),
                deletion_state = COALESCE(deletion_state, 'deleted'),
                email = email || '#deleted-' || id::text,
                updated_at = NOW()
            WHERE id = $1
            """,
            _coerce_uuid(extra_account_id)
        )

    async def begin_account_deletion(
        self,
        extra_account_id,
        *,
        user_id: int,
    ) -> bool:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE extra_accounts
                    SET deleted_at = NOW(),
                        deletion_state = 'pending_primary_purge',
                        email = email || '#deleted-' || id::text,
                        updated_at = NOW()
                    WHERE id = $1 AND deleted_at IS NULL AND user_id = $2
                    RETURNING id
                    """,
                    extra_account_id,
                    int(user_id),
                )
                if not row:
                    return False
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id = $1 AND revoked = FALSE
                    """,
                    int(user_id),
                )
                await conn.execute(
                    """
                    UPDATE anonymous_auth_bootstraps
                    SET disabled_at = COALESCE(disabled_at, NOW()),
                        last_used_at = NOW()
                    WHERE user_id = $1
                    """,
                    int(user_id),
                )
                return True

    async def complete_account_deletion(self, extra_account_id) -> None:
        await self.execute(
            """
            UPDATE extra_accounts
            SET deletion_state = 'completed', updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NOT NULL
            """,
            _coerce_uuid(extra_account_id),
        )

    async def get_pending_account_deletions(self, limit: int = 100) -> list[dict]:
        rows = await self.fetch(
            """
            SELECT id, user_id
            FROM extra_accounts
            WHERE deletion_state = 'pending_primary_purge'
              AND deleted_at IS NOT NULL
            ORDER BY updated_at
            LIMIT $1
            """,
            max(1, min(int(limit), 1000)),
        )
        return [dict(row) for row in rows]

    async def restore_soft_deleted_extra_account(
        self,
        extra_account_id,
        *,
        email: str,
    ) -> bool:
        result = await self.execute(
            """
            UPDATE extra_accounts
            SET deleted_at = NULL,
                email = $2,
                deletion_state = NULL,
                updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NOT NULL
            """,
            _coerce_uuid(extra_account_id),
            str(email).strip().lower(),
        )
        return result != "UPDATE 0"

    async def enqueue_account_email_action(self, email: str, purpose: str) -> None:
        """Queue eligible work without exposing whether the address exists."""
        purpose = _normalize_token_purpose(purpose)
        if purpose not in {"verify_email", "password_reset"}:
            raise ValueError("invalid_email_outbox_purpose")
        eligibility = (
            "a.is_email_verified = FALSE"
            if purpose == "verify_email"
            else "a.is_email_verified = TRUE"
        )
        await self.execute(
            f"""
            INSERT INTO extraid_email_outbox (
                extra_account_id, purpose, email_snapshot
            )
            SELECT a.id, $2, LOWER(a.email)
            FROM extra_accounts a
            WHERE LOWER(a.email) = LOWER($1)
              AND a.deleted_at IS NULL
              AND {eligibility}
            ON CONFLICT (extra_account_id, purpose) DO NOTHING
            """,
            str(email).strip().lower(),
            purpose,
        )

    async def claim_account_email_outbox(self, limit: int = 20) -> list[dict]:
        """Lease due jobs; stale leases are recoverable after five minutes."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        limit = max(1, min(int(limit), 100))
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    WITH picked AS (
                        SELECT id
                        FROM extraid_email_outbox
                        WHERE available_at <= NOW()
                          AND expires_at > NOW()
                          AND attempts < $2
                          AND (
                              claimed_at IS NULL
                              OR claimed_at < NOW() - INTERVAL '5 minutes'
                          )
                        ORDER BY available_at, created_at
                        LIMIT $1
                        FOR UPDATE SKIP LOCKED
                    ),
                    leased AS (
                        UPDATE extraid_email_outbox o
                        SET claimed_at = NOW(),
                            attempts = attempts + 1
                        FROM picked
                        WHERE o.id = picked.id
                        RETURNING o.*
                    )
                    SELECT
                        leased.id AS outbox_id,
                        leased.extra_account_id,
                        leased.purpose,
                        leased.email_snapshot,
                        leased.attempts,
                        leased.expires_at AS outbox_expires_at,
                        a.id,
                        a.user_id,
                        a.email,
                        a.is_email_verified,
                        a.email_verification_required,
                        a.deleted_at,
                        a.reg_bonus_claimed
                    FROM leased
                    LEFT JOIN extra_accounts a ON a.id = leased.extra_account_id
                    ORDER BY leased.created_at
                    """,
                    limit,
                    EMAIL_OUTBOX_MAX_ATTEMPTS,
                )
                return [dict(row) for row in rows]

    async def complete_account_email_outbox(self, outbox_id) -> None:
        await self.execute(
            "DELETE FROM extraid_email_outbox WHERE id = $1",
            _coerce_uuid(outbox_id),
        )

    async def retry_account_email_outbox(
        self,
        outbox_id,
        *,
        error: str,
    ) -> bool:
        """Release a lease with capped exponential backoff, then drop exhausted jobs."""
        outbox_id = _coerce_uuid(outbox_id)
        result = await self.execute(
            """
            UPDATE extraid_email_outbox
            SET claimed_at = NULL,
                available_at = NOW() + make_interval(
                    secs => LEAST(
                        3600,
                        (30 * power(2, GREATEST(attempts - 1, 0)))::int
                    )
                ),
                last_error = $2
            WHERE id = $1
              AND attempts < $3
              AND expires_at > NOW()
            """,
            outbox_id,
            str(error)[:500],
            EMAIL_OUTBOX_MAX_ATTEMPTS,
        )
        if result != "UPDATE 0":
            return True
        await self.complete_account_email_outbox(outbox_id)
        return False

    async def cleanup_account_email_outbox(self) -> int:
        result = await self.execute(
            """
            DELETE FROM extraid_email_outbox
            WHERE expires_at <= NOW() OR attempts >= $1
            """,
            EMAIL_OUTBOX_MAX_ATTEMPTS,
        )
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    @asynccontextmanager
    async def account_email_delivery_guard(
        self,
        *,
        extra_account_id,
        email_snapshot: str,
        token_ids: list,
    ):
        """Serialize provider handoff with a pending account email change.

        The transaction intentionally stays open only while the provider call
        is in flight.  This is a low-volume security flow, and holding the
        advisory lock guarantees that an address change either happens before
        delivery (making this guard fail) or after delivery and consumes every
        token sent to the previous address before returning to the client.
        """
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        extra_account_id = _coerce_uuid(extra_account_id)
        email_snapshot = str(email_snapshot).strip().lower()
        normalized_token_ids = [_coerce_uuid(value) for value in token_ids]
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    str(extra_account_id),
                )
                account_matches = await conn.fetchval(
                    """
                    SELECT 1
                    FROM extra_accounts
                    WHERE id = $1
                      AND LOWER(email) = LOWER($2)
                      AND deleted_at IS NULL
                    """,
                    extra_account_id,
                    email_snapshot,
                )
                active_token_count = 0
                if account_matches and normalized_token_ids:
                    active_token_count = int(
                        await conn.fetchval(
                            """
                            SELECT COUNT(*)
                            FROM account_action_tokens
                            WHERE extra_account_id = $1
                              AND id = ANY($2::uuid[])
                              AND LOWER(email_snapshot) = LOWER($3)
                              AND consumed_at IS NULL
                              AND expires_at > NOW()
                            """,
                            extra_account_id,
                            normalized_token_ids,
                            email_snapshot,
                        )
                        or 0
                    )
                yield bool(
                    account_matches
                    and active_token_count == len(normalized_token_ids)
                )

    async def create_account_action_token(
        self,
        *,
        token_id,
        extra_account_id,
        purpose: str,
        token_hash: str,
        email_snapshot: str,
        ttl_seconds: int,
        cooldown_seconds: int = 60,
    ) -> bool:
        """Issue one purpose-bound token, respecting a DB-enforced resend cooldown."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        purpose = _normalize_token_purpose(purpose)
        token_id = _coerce_uuid(token_id)
        extra_account_id = _coerce_uuid(extra_account_id)
        email_snapshot = str(email_snapshot).strip().lower()
        if not email_snapshot:
            raise ValueError("email_snapshot_required")
        ttl_seconds = max(60, min(int(ttl_seconds), 86_400))
        cooldown_seconds = max(0, min(int(cooldown_seconds), 3_600))
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    str(extra_account_id),
                )
                account = await conn.fetchrow(
                    """
                    SELECT email, is_email_verified, deleted_at
                    FROM extra_accounts
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    extra_account_id,
                )
                if (
                    not account
                    or account["deleted_at"] is not None
                    or str(account["email"]).strip().lower() != email_snapshot
                ):
                    return False
                if purpose in {"verify_email", "cancel_registration"} and bool(
                    account["is_email_verified"]
                ):
                    return False
                if purpose == "password_reset" and not bool(
                    account["is_email_verified"]
                ):
                    return False
                recent = await conn.fetchval(
                    """
                    SELECT 1
                    FROM account_action_tokens
                    WHERE extra_account_id = $1
                      AND purpose = $2
                      AND created_at > NOW() - make_interval(secs => $3::int)
                      AND consumed_at IS NULL
                      AND LOWER(email_snapshot) = LOWER($4)
                    LIMIT 1
                    """,
                    extra_account_id,
                    purpose,
                    cooldown_seconds,
                    email_snapshot,
                )
                if recent:
                    return False
                await conn.execute(
                    """
                    UPDATE account_action_tokens
                    SET consumed_at = COALESCE(consumed_at, NOW())
                    WHERE extra_account_id = $1
                      AND purpose = $2
                      AND consumed_at IS NULL
                    """,
                    extra_account_id,
                    purpose,
                )
                await conn.execute(
                    """
                    INSERT INTO account_action_tokens (
                        id, extra_account_id, purpose, token_hash,
                        email_snapshot, expires_at
                    )
                    VALUES ($1, $2, $3, $4, $5, NOW() + make_interval(secs => $6::int))
                    """,
                    token_id,
                    extra_account_id,
                    purpose,
                    token_hash,
                    email_snapshot,
                    ttl_seconds,
                )
                return True

    async def consume_email_verification_token(
        self,
        *,
        token_id,
        token_hash: str,
    ) -> dict | None:
        return await self._consume_account_action_token(
            token_id=token_id,
            token_hash=token_hash,
            purpose="verify_email",
            new_password_hash=None,
        )

    async def consume_email_verification_code(
        self,
        *,
        email: str,
        code: str,
    ) -> dict | None:
        """Consume the latest active verification code for an email address."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        normalized_email = str(email or "").strip().lower()
        if not normalized_email:
            return None
        token_id = await self.fetchval(
            """
            SELECT t.id
            FROM account_action_tokens t
            JOIN extra_accounts a ON a.id = t.extra_account_id
            WHERE LOWER(a.email) = LOWER($1)
              AND t.purpose = 'verify_email'
              AND t.consumed_at IS NULL
              AND a.deleted_at IS NULL
            ORDER BY t.created_at DESC
            LIMIT 1
            """,
            normalized_email,
        )
        if not token_id:
            return None
        return await self._consume_account_action_token(
            token_id=token_id,
            token_hash=_account_email_code_hash(
                code,
                normalized_email,
                token_id,
            ),
            purpose="verify_email",
            new_password_hash=None,
            verification_source="email_code",
        )

    async def revoke_account_action_token(self, token_id) -> None:
        try:
            token_id = _coerce_uuid(token_id)
        except (TypeError, ValueError, AttributeError):
            return
        await self.execute(
            """
            DELETE FROM account_action_tokens
            WHERE id = $1 AND consumed_at IS NULL
            """,
            token_id,
        )

    async def cleanup_account_action_tokens(self, limit: int = 1000) -> int:
        result = await self.execute(
            """
            WITH doomed AS (
                SELECT id
                FROM account_action_tokens
                WHERE (
                    consumed_at IS NOT NULL
                    AND consumed_at < NOW() - INTERVAL '1 day'
                ) OR expires_at < NOW() - INTERVAL '1 day'
                ORDER BY expires_at
                LIMIT $1
            )
            DELETE FROM account_action_tokens t
            USING doomed
            WHERE t.id = doomed.id
            """,
            max(1, min(int(limit), 10_000)),
        )
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    async def consume_password_reset_token(
        self,
        *,
        token_id,
        token_hash: str,
        new_password_hash: str,
    ) -> dict | None:
        return await self._consume_account_action_token(
            token_id=token_id,
            token_hash=token_hash,
            purpose="password_reset",
            new_password_hash=new_password_hash,
        )

    async def consume_registration_cancel_token(
        self,
        *,
        token_id,
        token_hash: str,
    ) -> dict | None:
        return await self._consume_account_action_token(
            token_id=token_id,
            token_hash=token_hash,
            purpose="cancel_registration",
            new_password_hash=None,
        )

    async def _consume_account_action_token(
        self,
        *,
        token_id,
        token_hash: str,
        purpose: str,
        new_password_hash: str | None,
        verification_source: str = "email_token",
    ) -> dict | None:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        purpose = _normalize_token_purpose(purpose)
        try:
            token_id = _coerce_uuid(token_id)
        except (TypeError, ValueError, AttributeError):
            return None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT
                        t.*,
                        a.user_id,
                        a.email,
                        a.deleted_at,
                        a.reg_bonus_claimed,
                        a.registration_origin
                    FROM account_action_tokens t
                    JOIN extra_accounts a ON a.id = t.extra_account_id
                    WHERE t.id = $1 AND t.purpose = $2
                    FOR UPDATE OF t, a
                    """,
                    token_id,
                    purpose,
                )
                if not row:
                    return None
                if (
                    row["consumed_at"] is not None
                    or row["expires_at"] <= datetime.now(timezone.utc)
                    or row["deleted_at"] is not None
                    or int(row["attempts"] or 0) >= ACCOUNT_ACTION_TOKEN_MAX_ATTEMPTS
                ):
                    return None
                if str(row["email"]).strip().lower() != str(
                    row["email_snapshot"]
                ).strip().lower():
                    await conn.execute(
                        """
                        UPDATE account_action_tokens
                        SET consumed_at = NOW()
                        WHERE id = $1
                        """,
                        token_id,
                    )
                    return None
                if not hmac.compare_digest(str(row["token_hash"]), str(token_hash)):
                    await conn.execute(
                        """
                        UPDATE account_action_tokens
                        SET attempts = attempts + 1
                        WHERE id = $1
                        """,
                        token_id,
                    )
                    return None
                await conn.execute(
                    """
                    UPDATE account_action_tokens
                    SET consumed_at = NOW()
                    WHERE id = $1
                    """,
                    token_id,
                )
                if purpose == "verify_email":
                    await conn.execute(
                        """
                        UPDATE extra_accounts
                        SET is_email_verified = TRUE,
                            email_verification_required = FALSE,
                            verification_source = $2,
                            pending_expires_at = NULL,
                            email_verify_token = NULL,
                            email_verify_expires = NULL,
                            updated_at = NOW()
                        WHERE id = $1 AND deleted_at IS NULL
                        """,
                        row["extra_account_id"],
                        verification_source,
                    )
                    # The verification message contains a paired cancellation
                    # token. It must become unusable as soon as verification
                    # succeeds, even if the user later clicks the stale link.
                    await conn.execute(
                        """
                        UPDATE account_action_tokens
                        SET consumed_at = COALESCE(consumed_at, NOW())
                        WHERE extra_account_id = $1
                          AND purpose = 'cancel_registration'
                          AND consumed_at IS NULL
                        """,
                        row["extra_account_id"],
                    )
                elif purpose == "password_reset":
                    if not new_password_hash:
                        raise ValueError("new_password_hash_required")
                    updated_account = await conn.fetchrow(
                        """
                        UPDATE extra_accounts
                        SET password_hash = $1, updated_at = NOW()
                        WHERE id = $2
                          AND is_email_verified = TRUE
                          AND deleted_at IS NULL
                        RETURNING id
                        """,
                        new_password_hash,
                        row["extra_account_id"],
                    )
                    if not updated_account:
                        raise RuntimeError("password_reset_account_not_verified")
                    await conn.execute(
                        """
                        UPDATE auth_sessions
                        SET revoked = TRUE, revoked_at = NOW()
                        WHERE user_id = $1 AND revoked = FALSE
                        """,
                        int(row["user_id"]),
                    )
                    # Password reset is a full credential reset.  Leaving an
                    # installation bootstrap active would let its old secret
                    # mint a fresh anonymous JWT immediately after every
                    # session was revoked.
                    await conn.execute(
                        """
                        UPDATE anonymous_auth_bootstraps
                        SET disabled_at = COALESCE(disabled_at, NOW()),
                            last_used_at = NOW()
                        WHERE user_id = $1
                        """,
                        int(row["user_id"]),
                    )
                elif purpose == "cancel_registration":
                    cancelled = await conn.fetchrow(
                        """
                        UPDATE extra_accounts
                        SET deleted_at = NOW(),
                            deletion_state = 'cancelled_pending_primary_cleanup',
                            email = email || '#cancelled-' || id::text,
                            updated_at = NOW()
                        WHERE id = $1
                          AND is_email_verified = FALSE
                          AND email_verification_required = TRUE
                          AND deleted_at IS NULL
                        RETURNING id
                        """,
                        row["extra_account_id"],
                    )
                    if not cancelled:
                        # A verified or already-deleted account makes this stale
                        # token invalid; consume it without surfacing a 500.
                        return None
                return dict(row)

    async def expire_pending_registrations(self, limit: int = 100) -> list[dict]:
        """Hard-delete expired, never-verified credentials and release their email/nickname."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, registration_origin
                    FROM extra_accounts
                    WHERE is_email_verified = FALSE
                      AND email_verification_required = TRUE
                      AND pending_expires_at IS NOT NULL
                      AND pending_expires_at <= NOW()
                      AND deleted_at IS NULL
                    ORDER BY pending_expires_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    max(1, min(int(limit), 1000)),
                )
                expired: list[dict] = []
                for row in rows:
                    deleted = await conn.fetchrow(
                        """
                        UPDATE extra_accounts
                        SET deleted_at = NOW(),
                            deletion_state = 'expired_pending_primary_cleanup',
                            email = email || '#expired-' || id::text,
                            updated_at = NOW()
                        WHERE id = $1
                          AND is_email_verified = FALSE
                          AND email_verification_required = TRUE
                          AND deleted_at IS NULL
                        RETURNING id
                        """,
                        row["id"],
                    )
                    if deleted:
                        expired.append(dict(row))
                return expired

    async def get_pending_registration_cleanups(self, limit: int = 100) -> list[dict]:
        rows = await self.fetch(
            """
            SELECT id, user_id, registration_origin
            FROM extra_accounts
            WHERE deletion_state IN (
                'cancelled_pending_primary_cleanup',
                'expired_pending_primary_cleanup'
            )
              AND deleted_at IS NOT NULL
            ORDER BY updated_at
            LIMIT $1
            """,
            max(1, min(int(limit), 1000)),
        )
        return [dict(row) for row in rows]

    async def finalize_pending_registration_cleanup(self, extra_account_id) -> bool:
        result = await self.execute(
            """
            DELETE FROM extra_accounts
            WHERE id = $1
              AND deletion_state IN (
                  'cancelled_pending_primary_cleanup',
                  'expired_pending_primary_cleanup'
              )
              AND deleted_at IS NOT NULL
            """,
            _coerce_uuid(extra_account_id),
        )
        return result != "DELETE 0"

    # ═══════════════════════════════════════════════════════════════════
    # Auth sessions
    # ═══════════════════════════════════════════════════════════════════

    async def get_anonymous_auth_bootstrap(self, bootstrap_id: str) -> dict | None:
        row = await self.fetchrow(
            """
            SELECT bootstrap_id, secret_hash, user_id, session_id,
                   generation, disabled_at
            FROM anonymous_auth_bootstraps
            WHERE bootstrap_id = $1
            """,
            str(bootstrap_id),
        )
        return dict(row) if row else None

    async def claim_anonymous_auth_bootstrap(
        self,
        *,
        bootstrap_id: str,
        secret_hash: str,
        proposed_user_id: int,
    ) -> dict:
        """Create an installation-to-user mapping or return the race winner."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        bootstrap_id = str(bootstrap_id)
        secret_hash = str(secret_hash)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"anonymous-bootstrap:{bootstrap_id}",
                )
                existing = await conn.fetchrow(
                    """
                    SELECT bootstrap_id, secret_hash, user_id, session_id,
                           generation, disabled_at
                    FROM anonymous_auth_bootstraps
                    WHERE bootstrap_id = $1
                    FOR UPDATE
                    """,
                    bootstrap_id,
                )
                if existing:
                    return dict(existing)
                created = await conn.fetchrow(
                    """
                    INSERT INTO anonymous_auth_bootstraps (
                        bootstrap_id, secret_hash, user_id
                    )
                    VALUES ($1, $2, $3)
                    RETURNING bootstrap_id, secret_hash, user_id, session_id,
                              generation, disabled_at
                    """,
                    bootstrap_id,
                    secret_hash,
                    int(proposed_user_id),
                )
                return dict(created)

    async def create_anonymous_bootstrap_session(
        self,
        *,
        bootstrap_id: str,
        secret_hash: str,
        session_id,
        token_hash: str,
        expires_at,
        device_label: str | None = None,
    ) -> dict:
        """Rotate the recoverable anonymous session in one transaction.

        The raw installation secret and JWT never enter the database.  A retry
        proves knowledge of the installation secret, gets a new JWT/session,
        and invalidates every older anonymous JWT for the same game user.
        """
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        bootstrap_id = str(bootstrap_id)
        secret_hash = str(secret_hash)
        session_id = _coerce_uuid(session_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"anonymous-bootstrap:{bootstrap_id}",
                )
                bootstrap = await conn.fetchrow(
                    """
                    SELECT bootstrap_id, secret_hash, user_id, generation,
                           disabled_at
                    FROM anonymous_auth_bootstraps
                    WHERE bootstrap_id = $1
                    FOR UPDATE
                    """,
                    bootstrap_id,
                )
                if not bootstrap or not hmac.compare_digest(
                    str(bootstrap["secret_hash"]),
                    secret_hash,
                ):
                    return {"status": "invalid_bootstrap"}
                if bootstrap["disabled_at"] is not None:
                    return {
                        "status": "bootstrap_upgraded",
                        "user_id": int(bootstrap["user_id"]),
                    }
                user_id = int(bootstrap["user_id"])
                await conn.execute(
                    """
                    INSERT INTO auth_sessions (
                        session_id, user_id, auth_method, token_hash,
                        expires_at, device_label
                    )
                    VALUES ($1, $2, 'android_anonymous', $3, $4, $5)
                    """,
                    session_id,
                    user_id,
                    str(token_hash),
                    expires_at,
                    device_label,
                )
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id = $1
                      AND auth_method = 'android_anonymous'
                      AND session_id <> $2
                      AND revoked = FALSE
                    """,
                    user_id,
                    session_id,
                )
                updated = await conn.fetchrow(
                    """
                    UPDATE anonymous_auth_bootstraps
                    SET session_id = $2,
                        generation = generation + 1,
                        last_used_at = NOW()
                    WHERE bootstrap_id = $1
                    RETURNING user_id, generation
                    """,
                    bootstrap_id,
                    session_id,
                )
                return {
                    "status": "created",
                    "user_id": int(updated["user_id"]),
                    "generation": int(updated["generation"]),
                }

    async def create_email_password_session(
        self,
        *,
        session_id,
        user_id: int,
        token_hash: str,
        expires_at,
        device_label: str | None = None,
    ) -> dict:
        """Create a stronger session and retire only anonymous credentials."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        session_id = _coerce_uuid(session_id)
        user_id = int(user_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO auth_sessions (
                        session_id, user_id, auth_method, token_hash,
                        expires_at, device_label
                    )
                    VALUES ($1, $2, 'email_password', $3, $4, $5)
                    """,
                    session_id,
                    user_id,
                    str(token_hash),
                    expires_at,
                    device_label,
                )
                await conn.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE, revoked_at = NOW()
                    WHERE user_id = $1
                      AND auth_method = 'android_anonymous'
                      AND revoked = FALSE
                    """,
                    user_id,
                )
                await conn.execute(
                    """
                    UPDATE anonymous_auth_bootstraps
                    SET disabled_at = COALESCE(disabled_at, NOW()),
                        last_used_at = NOW()
                    WHERE user_id = $1
                    """,
                    user_id,
                )
                return {"session_id": session_id}

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
              AND (
                  ea.id IS NULL
                  OR (ea.deleted_at IS NULL AND ea.link_state IS NULL)
              )
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

    async def create_bot_auth_code(
        self,
        code: str,
        user_id: int,
        purpose: str = BOT_AUTH_CODE_DEFAULT_PURPOSE,
    ) -> dict:
        """Create at most one active code per user/purpose without storing the secret."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    DELETE FROM bot_auth_codes
                    WHERE user_id = $1
                      AND purpose = $2
                      AND used_at IS NULL
                      AND expires_at <= NOW()
                    """,
                    int(user_id),
                    purpose,
                )
                existing = await conn.fetchrow(
                    """
                    SELECT expires_at
                    FROM bot_auth_codes
                    WHERE user_id = $1
                      AND purpose = $2
                      AND used_at IS NULL
                      AND expires_at > NOW()
                    LIMIT 1
                    """,
                    int(user_id),
                    purpose,
                )
                if existing:
                    return {"created": False, "expires_at": existing["expires_at"]}
                for _ in range(8):
                    storage_key = secrets.token_hex(3)
                    row = await conn.fetchrow(
                        """
                        INSERT INTO bot_auth_codes (
                            code, code_hash, user_id, purpose, expires_at
                        )
                        VALUES ($1, $2, $3, $4, NOW() + INTERVAL '5 minutes')
                        ON CONFLICT DO NOTHING
                        RETURNING expires_at
                        """,
                        storage_key,
                        code_hash,
                        int(user_id),
                        purpose,
                    )
                    if row:
                        return {
                            "created": True,
                            "expires_at": row["expires_at"],
                            "purpose": purpose,
                        }
                    # A storage-key collision is harmless; another active code for
                    # this user/purpose means a concurrent request won the race.
                    active = await conn.fetchrow(
                        """
                        SELECT expires_at
                        FROM bot_auth_codes
                        WHERE user_id = $1
                          AND purpose = $2
                          AND used_at IS NULL
                          AND expires_at > NOW()
                        LIMIT 1
                        """,
                        int(user_id),
                        purpose,
                    )
                    if active:
                        return {"created": False, "expires_at": active["expires_at"]}
                raise RuntimeError("failed_to_allocate_bot_auth_code_storage_key")

    async def verify_bot_auth_code(
        self,
        code: str,
        purpose: str = BOT_AUTH_CODE_DEFAULT_PURPOSE,
        user_id: int | None = None,
    ) -> dict | None:
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        params: list[Any] = [code_hash, purpose, BOT_AUTH_CODE_MAX_ATTEMPTS]
        user_clause = ""
        if user_id is not None:
            params.append(int(user_id))
            user_clause = "AND user_id = $4"
        row = await self.fetchrow(
            f"""
            SELECT *
            FROM bot_auth_codes
            WHERE code_hash = $1
              AND purpose = $2
              AND failed_attempts < $3
              AND used_at IS NULL
              AND expires_at > NOW()
              {user_clause}
            """,
            *params,
        )
        if row:
            return dict(row)
        if user_id is not None:
            await self.execute(
                """
                UPDATE bot_auth_codes
                SET failed_attempts = failed_attempts + 1
                WHERE user_id = $1
                  AND purpose = $2
                  AND failed_attempts < $3
                  AND used_at IS NULL
                  AND expires_at > NOW()
                """,
                int(user_id),
                purpose,
                BOT_AUTH_CODE_MAX_ATTEMPTS,
            )
        return None

    async def consume_bot_auth_code(
        self,
        code: str,
        purpose: str = BOT_AUTH_CODE_DEFAULT_PURPOSE,
        user_id: int | None = None,
    ) -> dict | None:
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                params: list[Any] = [
                    code_hash,
                    purpose,
                    BOT_AUTH_CODE_MAX_ATTEMPTS,
                ]
                user_clause = ""
                if user_id is not None:
                    params.append(int(user_id))
                    user_clause = "AND user_id = $4"
                row = await conn.fetchrow(
                    f"""
                    UPDATE bot_auth_codes
                    SET used_at = NOW()
                    WHERE code_hash = $1
                      AND purpose = $2
                      AND failed_attempts < $3
                      AND used_at IS NULL
                      AND expires_at > NOW()
                      {user_clause}
                    RETURNING *
                    """,
                    *params,
                )
                if row:
                    return dict(row)
                if user_id is not None:
                    await conn.execute(
                        """
                        UPDATE bot_auth_codes
                        SET failed_attempts = failed_attempts + 1
                        WHERE user_id = $1
                          AND purpose = $2
                          AND failed_attempts < $3
                          AND used_at IS NULL
                          AND expires_at > NOW()
                        """,
                        int(user_id),
                        purpose,
                        BOT_AUTH_CODE_MAX_ATTEMPTS,
                    )
                return None

    async def consume_bot_auth_code_and_create_session(
        self,
        code: str,
        *,
        purpose: str,
        user_id: int,
        session_id,
        auth_method: str,
        token_hash: str,
        expires_at,
        device_label: str | None = None,
    ) -> dict | None:
        """Consume an OTP and create its session in the same DB transaction."""
        if not self._pool:
            raise RuntimeError("ExtraID DB not connected")
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        session_id = _coerce_uuid(session_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO auth_sessions (
                        session_id, user_id, auth_method, token_hash, expires_at, device_label
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    session_id,
                    int(user_id),
                    str(auth_method),
                    str(token_hash),
                    expires_at,
                    device_label,
                )
                consumed = await conn.fetchrow(
                    """
                    UPDATE bot_auth_codes
                    SET used_at = NOW(), session_id = $1
                    WHERE code_hash = $2
                      AND purpose = $3
                      AND user_id = $4
                      AND failed_attempts < $5
                      AND used_at IS NULL
                      AND expires_at > NOW()
                    RETURNING *
                    """,
                    session_id,
                    code_hash,
                    purpose,
                    int(user_id),
                    BOT_AUTH_CODE_MAX_ATTEMPTS,
                )
                if not consumed:
                    await conn.execute(
                        "DELETE FROM auth_sessions WHERE session_id = $1",
                        session_id,
                    )
                    await conn.execute(
                        """
                        UPDATE bot_auth_codes
                        SET failed_attempts = failed_attempts + 1
                        WHERE user_id = $1
                          AND purpose = $2
                          AND failed_attempts < $3
                          AND used_at IS NULL
                          AND expires_at > NOW()
                        """,
                        int(user_id),
                        purpose,
                        BOT_AUTH_CODE_MAX_ATTEMPTS,
                    )
                    return None
                return dict(consumed)

    async def mark_bot_code_used(
        self,
        code: str,
        session_id,
        purpose: str = BOT_AUTH_CODE_DEFAULT_PURPOSE,
        user_id: int | None = None,
    ) -> None:
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        params: list[Any] = [_coerce_uuid(session_id), code_hash, purpose]
        user_clause = ""
        if user_id is not None:
            params.append(int(user_id))
            user_clause = "AND user_id = $4"
        await self.execute(
            f"""
            UPDATE bot_auth_codes
            SET session_id = $1
            WHERE code_hash = $2
              AND purpose = $3
              AND used_at IS NOT NULL
              {user_clause}
            """,
            *params,
        )

    async def invalidate_bot_auth_code(
        self,
        code: str,
        *,
        purpose: str = BOT_AUTH_CODE_DEFAULT_PURPOSE,
        user_id: int | None = None,
    ) -> None:
        purpose = _normalize_token_purpose(purpose)
        code_hash = _bot_auth_code_hash(code, purpose)
        params: list[Any] = [code_hash, purpose]
        user_clause = ""
        if user_id is not None:
            params.append(int(user_id))
            user_clause = "AND user_id = $3"
        await self.execute(
            f"""
            DELETE FROM bot_auth_codes
            WHERE code_hash = $1
              AND purpose = $2
              AND used_at IS NULL
              {user_clause}
            """,
            *params,
        )

    async def cleanup_old_bot_codes(
        self,
        user_id: int,
        purpose: str | None = None,
    ) -> None:
        if purpose is None:
            await self.execute(
                """
                DELETE FROM bot_auth_codes
                WHERE user_id = $1
                  AND (used_at IS NOT NULL OR expires_at <= NOW())
                """,
                int(user_id),
            )
            return
        purpose = _normalize_token_purpose(purpose)
        await self.execute(
            """
            DELETE FROM bot_auth_codes
            WHERE user_id = $1
              AND purpose = $2
              AND (used_at IS NOT NULL OR expires_at <= NOW())
            """,
            int(user_id),
            purpose,
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
