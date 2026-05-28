import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database


def _settings() -> DatabaseSettings:
    return DatabaseSettings(
        host="localhost",
        port=5432,
        user="test",
        password="test",
        database="test",
    )


class FakeStarterGrantDB(Database):
    def __init__(self):
        super().__init__(_settings())
        self._pool = object()
        self.fetch_calls = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [{"card_id": 1}, {"card_id": 36}, {"card_id": 37}]


@pytest.mark.asyncio
async def test_grant_start_cards_adds_all_start_rarity_cards():
    db = FakeStarterGrantDB()

    result = await db.grant_start_cards(123)

    assert result == {"success": True, "added": 3}
    query, args = db.fetch_calls[0]
    assert args == (123,)
    assert "FROM cards" in query
    assert "WHERE rarity = 'start'" in query
    assert "ON CONFLICT (user_id, card_id) DO NOTHING" in query


class FakeEnsureUserDB(Database):
    def __init__(self):
        super().__init__(_settings())
        self._pool = object()
        self.executed = []
        self.start_cards_user_id = None

    async def fetchval(self, query, *args):
        if "SELECT 1 FROM users" in query:
            return None
        if "SELECT 1 FROM deck_presets" in query:
            return None
        return None

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "OK"

    async def grant_start_cards(self, user_id):
        self.start_cards_user_id = user_id
        return {"success": True, "added": 8}

    async def add_card_to_user(self, user_id, card_id):
        raise AssertionError("ensure_user must grant start rarity cards, not a hard-coded card")

    async def grant_starter_cosmetics(self, user_id):
        return None


@pytest.mark.asyncio
async def test_new_user_registration_grants_start_rarity_cards():
    db = FakeEnsureUserDB()

    created = await db.ensure_user(
        user_id=987,
        username="player",
        first_name="Player",
        last_name=None,
    )

    assert created is True
    assert db.start_cards_user_id == 987
