import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.support_database import SupportDatabase


class SchemaIntentDB(SupportDatabase):
    def __init__(self):
        super().__init__(DatabaseSettings("localhost", 5432, "user", "password", "support"))
        self.executed: list[str] = []
        self.fetchvals: list[str] = []

    async def execute(self, query: str, *args):
        self.executed.append(query)
        return "OK"

    async def fetchval(self, query: str, *args):
        self.fetchvals.append(query)
        return None


@pytest.mark.asyncio
async def test_support_database_schema_covers_permanent_support_records():
    db = SchemaIntentDB()

    await db.init_schema()

    sql = "\n".join(db.executed)
    normalized = " ".join(sql.lower().split())

    assert any("to_regclass('public.support_tickets')" in query.lower() for query in db.fetchvals)
    for table_name in (
        "support_tickets",
        "support_messages",
        "support_attachments",
        "support_identities",
        "support_auth_codes",
        "support_admin_login_codes",
        "support_delivery_outbox",
        "support_audit_events",
    ):
        assert f"create table if not exists {table_name}" in normalized

    assert "uuid primary key" in normalized
    assert "metadata jsonb not null default '{}'::jsonb" in normalized
    assert "created_at timestamptz not null default now()" in normalized
    assert "channel_id text" in normalized
    assert "profile_snapshot jsonb not null default '{}'::jsonb" in normalized
    assert "support_tickets_inbox_idx" in normalized
    assert "support_tickets_game_user_idx" in normalized
    assert "support_messages_ticket_created_idx" in normalized
    assert "support_delivery_outbox_ready_idx" in normalized
    assert "support_audit_events_ticket_created_idx" in normalized
    assert "delete from" not in normalized
