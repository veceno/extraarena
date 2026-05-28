import json

import pytest

from ai.bot_factory import BotGenerator
from infrastructure.database import Database


class RuntimeDBHarness(Database):
    def __init__(self, settings=None):
        self.settings = dict(settings or {})

    async def fetch(self, _query, keys):
        return [{"key": key, "value": self.settings[key]} for key in keys if key in self.settings]

    async def fetchval(self, _query, key):
        return self.settings.get(key)

    async def execute(self, _query, key, value, _description=""):
        self.settings[key] = json.loads(value) if isinstance(value, str) else value


@pytest.mark.asyncio
async def test_runtime_config_defaults_and_normalization():
    db = RuntimeDBHarness({
        "feature_availability": {"shop": False, "unknown": False},
        "maintenance_mode": {"enabled": 1},
        "disabled_card_ids": ["2", "bad", 2, 5],
    })

    config = await db.get_runtime_config()

    assert config["maintenance_mode"] == {"enabled": True}
    assert config["feature_availability"]["shop"] is False
    assert config["feature_availability"]["collection"] is True
    assert config["disabled_card_ids"] == [2, 5]


@pytest.mark.asyncio
async def test_runtime_config_set_preserves_defaults():
    db = RuntimeDBHarness()

    config = await db.set_runtime_config(
        maintenance_mode={"enabled": True},
        feature_availability={"training": False},
        disabled_card_ids=["7", "7", None, 8],
    )

    assert config["maintenance_mode"]["enabled"] is True
    assert config["feature_availability"]["training"] is False
    assert config["feature_availability"]["classic"] is True
    assert config["disabled_card_ids"] == [7, 8]


class BotDBHarness:
    def __init__(self, disabled):
        self.disabled = disabled

    async def get_disabled_card_ids(self):
        return list(self.disabled)

    async def get_cards_list(self):
        return [
            {"id": 1, "name": "Hero A", "card_type": "hero"},
            {"id": 2, "name": "Hero B", "card_type": "hero"},
            {"id": 3, "name": "Unit A", "card_type": "unit"},
            {"id": 4, "name": "Unit B", "card_type": "unit"},
            {"id": 5, "name": "Unit C", "card_type": "unit"},
        ]


@pytest.mark.asyncio
async def test_bot_generator_replaces_blacklisted_cards_same_type():
    generator = BotGenerator(BotDBHarness(disabled={1, 3}))

    deck = await generator._sanitize_deck([1, 3, 4])

    assert 1 not in deck
    assert 3 not in deck
    assert 2 in deck
    assert len(deck) == 3


@pytest.mark.asyncio
async def test_bot_generator_fallback_deck_excludes_blacklist_and_keeps_hero():
    generator = BotGenerator(BotDBHarness(disabled={1, 3}))

    deck = await generator._build_bot_deck(100)

    assert 1 not in deck
    assert 3 not in deck
    assert 2 in deck
