from __future__ import annotations

from typing import Any, Optional

from infrastructure.config import DatabaseSettings

try:  # pragma: no cover - exercised implicitly when asyncpg is installed.
    import asyncpg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local unit tests can run without asyncpg.
    asyncpg = None  # type: ignore


def _ensure_asyncpg() -> None:
    if asyncpg is None:
        raise RuntimeError(
            "asyncpg is not installed. Install project requirements before connecting to the support database."
        )


class SupportDatabase:
    """Asyncpg wrapper for the support funnel's permanent PostgreSQL schema."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self._pool: Optional[Any] = None

    async def connect(self) -> None:
        _ensure_asyncpg()
        self._pool = await asyncpg.create_pool(
            host=self.settings.host,
            port=self.settings.port,
            user=self.settings.user,
            password=self.settings.password,
            database=self.settings.database,
            min_size=1,
            max_size=10,
        )

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def close(self) -> None:
        await self.disconnect()

    async def execute(self, query: str, *args) -> str:
        if not self._pool:
            raise RuntimeError("Support database is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[Any]:
        if not self._pool:
            raise RuntimeError("Support database is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[Any]:
        if not self._pool:
            raise RuntimeError("Support database is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Optional[Any]:
        if not self._pool:
            raise RuntimeError("Support database is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def init_schema(self) -> None:
        await self.fetchval("SELECT to_regclass('public.support_tickets')")
        for statement in SUPPORT_SCHEMA_SQL:
            await self.execute(statement)


SUPPORT_SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE EXTENSION IF NOT EXISTS pgcrypto
    """,
    """
    CREATE TABLE IF NOT EXISTS support_identities (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        scope TEXT NOT NULL,
        external_user_id TEXT,
        game_user_id BIGINT,
        display_name TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        channel_id TEXT,
        is_verified BOOLEAN NOT NULL DEFAULT FALSE,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (scope, external_user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_tickets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        topic TEXT NOT NULL,
        status TEXT NOT NULL,
        account_scope TEXT NOT NULL,
        requester_identity_id UUID REFERENCES support_identities(id) ON DELETE SET NULL,
        game_user_id BIGINT,
        channel TEXT NOT NULL,
        channel_id TEXT,
        subject TEXT NOT NULL DEFAULT '',
        priority_score INTEGER NOT NULL DEFAULT 0,
        priority_tier TEXT NOT NULL DEFAULT 'guest',
        profile_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_message_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        closed_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ticket_id UUID NOT NULL REFERENCES support_tickets(id),
        identity_id UUID REFERENCES support_identities(id) ON DELETE SET NULL,
        direction TEXT NOT NULL,
        body TEXT NOT NULL DEFAULT '',
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_attachments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ticket_id UUID NOT NULL REFERENCES support_tickets(id),
        message_id UUID REFERENCES support_messages(id) ON DELETE SET NULL,
        uploader_identity_id UUID REFERENCES support_identities(id) ON DELETE SET NULL,
        storage_path TEXT NOT NULL,
        original_filename TEXT NOT NULL DEFAULT '',
        content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        width INTEGER,
        height INTEGER,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_auth_codes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        identity_id UUID REFERENCES support_identities(id) ON DELETE SET NULL,
        game_user_id BIGINT,
        purpose TEXT NOT NULL DEFAULT 'verify_account',
        code_hash TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        used_at TIMESTAMPTZ,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (code_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_admin_login_codes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        admin_channel_id TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        used_at TIMESTAMPTZ,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (code_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_delivery_outbox (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ticket_id UUID REFERENCES support_tickets(id) ON DELETE SET NULL,
        message_id UUID REFERENCES support_messages(id) ON DELETE SET NULL,
        channel TEXT NOT NULL,
        channel_id TEXT,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_error TEXT NOT NULL DEFAULT '',
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS support_audit_events (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        actor_identity_id UUID REFERENCES support_identities(id) ON DELETE SET NULL,
        admin_channel_id TEXT,
        ticket_id UUID REFERENCES support_tickets(id) ON DELETE SET NULL,
        event_type TEXT NOT NULL,
        event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS support_identities_lookup_idx
    ON support_identities (scope, external_user_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_identities_game_user_idx
    ON support_identities (game_user_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_tickets_inbox_idx
    ON support_tickets (status, priority_score DESC, last_message_at ASC, created_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_tickets_game_user_idx
    ON support_tickets (game_user_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_tickets_requester_idx
    ON support_tickets (requester_identity_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_messages_ticket_created_idx
    ON support_messages (ticket_id, created_at ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_attachments_ticket_idx
    ON support_attachments (ticket_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_attachments_sha256_idx
    ON support_attachments (sha256)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_auth_codes_identity_idx
    ON support_auth_codes (identity_id, purpose, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_auth_codes_unused_idx
    ON support_auth_codes (purpose, expires_at)
    WHERE used_at IS NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS support_admin_login_codes_channel_idx
    ON support_admin_login_codes (admin_channel_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_delivery_outbox_ready_idx
    ON support_delivery_outbox (status, next_attempt_at, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_delivery_outbox_channel_idx
    ON support_delivery_outbox (channel, channel_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_audit_events_ticket_created_idx
    ON support_audit_events (ticket_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS support_audit_events_type_created_idx
    ON support_audit_events (event_type, created_at DESC)
    """,
)
