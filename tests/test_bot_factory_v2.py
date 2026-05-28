"""
Tests for BotFactory v2: reusable bots, trophy/difficulty, cosmetics, ONNX independence.
v2: per-card levels via _build_bot_card_levels.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.bot_factory import BotGenerator
from infrastructure.config import DECK_SIZE


# ============================================================================
# Difficulty calculation (returns diff only now)
# ============================================================================

class TestDifficultyCalculation:
    @pytest.mark.parametrize("player_trophies, expected_diff", [
        (0, "tier_lite_0000"),
        (99, "tier_lite_0000"),
        (100, "tier_easy_0100"),
        (299, "tier_easy_0100"),
        (300, "tier_easy_plus_0300"),
        (599, "tier_easy_plus_0300"),
        (600, "tier_medium_minus_0600"),
        (1199, "tier_medium_minus_0600"),
        (1200, "tier_medium_1200"),
        (1999, "tier_medium_1200"),
        (2000, "tier_medium_plus_2000"),
        (2999, "tier_medium_plus_2000"),
        (3000, "tier_hard_minus_3000"),
        (4499, "tier_hard_minus_3000"),
        (4500, "tier_hard_4500"),
        (5999, "tier_hard_4500"),
        (6000, "tier_hard_plus_6000"),
        (7499, "tier_hard_plus_6000"),
        (7500, "tier_max_minus_7500"),
        (8999, "tier_max_minus_7500"),
        (9000, "tier_max_9000"),
        (10000, "tier_max_9000"),
    ])
    def test_difficulty_only(self, player_trophies, expected_diff):
        diff = BotGenerator._calc_difficulty(player_trophies)
        assert diff == expected_diff

    def test_never_noob(self):
        for trophies in [0, 100, 500, 2000, 5000, 9000]:
            diff = BotGenerator._calc_difficulty(trophies)
            assert diff != "noob"

    def test_only_valid_difficulties(self):
        import random as rnd
        for _ in range(200):
            diff = BotGenerator._calc_difficulty(rnd.randint(0, 5000))
            assert diff in BotGenerator.DIFFICULTIES
            assert diff != "noob"


# ============================================================================
# Per-card level builder
# ============================================================================

class TestPerCardLevels:
    def test_lite_all_1s(self):
        levels = BotGenerator._build_bot_card_levels("lite", 10, 8)
        assert levels == [1] * 8

    def test_easy_all_1s(self):
        levels = BotGenerator._build_bot_card_levels("easy", 5, 5)
        assert levels == [2] * 5

    def test_medium_max_level_5(self):
        for _ in range(20):
            levels = BotGenerator._build_bot_card_levels("medium", 5, 8)
            assert len(levels) == 8
            for lvl in levels:
                assert lvl == 5

    def test_hard_max_level_5(self):
        for _ in range(20):
            levels = BotGenerator._build_bot_card_levels("hard", 5, 8)
            assert len(levels) == 8
            for lvl in levels:
                assert lvl == 5

    def test_max_max_level_5(self):
        for _ in range(20):
            levels = BotGenerator._build_bot_card_levels("max", 5, 8)
            assert len(levels) == 8
            for lvl in levels:
                assert 5 <= lvl <= 6

    def test_levels_can_differ_within_deck(self):
        """Некоторые tier-ы все еще могут давать slot-level variation."""
        import random as _rnd
        _rnd.seed(42)
        levels = BotGenerator._build_bot_card_levels("tier_medium_plus_2000", 5, 4)
        assert levels == [4, 4, 5, 4], f"got {levels}"

    def test_clamp_to_10(self):
        for _ in range(30):
            levels = BotGenerator._build_bot_card_levels("max", 15, 5)
            for lvl in levels:
                assert lvl <= 10

    def test_clamp_to_1(self):
        for _ in range(30):
            levels = BotGenerator._build_bot_card_levels("medium", 0, 5)
            for lvl in levels:
                assert lvl >= 1

    def test_no_level_advantage_before_6000(self):
        pre_6000_tiers = (
            "tier_lite_0000",
            "tier_easy_0100",
            "tier_easy_plus_0300",
            "tier_medium_minus_0600",
            "tier_medium_1200",
            "tier_medium_plus_2000",
            "tier_hard_minus_3000",
            "tier_hard_4500",
        )
        for tier in pre_6000_tiers:
            for _ in range(20):
                levels = BotGenerator._build_bot_card_levels(tier, 5, 9)
                assert max(levels) <= 5

    def test_late_tiers_have_controlled_partial_boost(self):
        levels = BotGenerator._build_bot_card_levels("tier_max_9000", 5, 10)
        assert max(levels) <= 6
        assert sum(1 for lvl in levels if lvl == 6) <= 4

    def test_max_minus_keeps_some_cards_below_full_cap(self):
        import random
        random.seed(42)
        levels = BotGenerator._build_bot_card_levels("tier_max_minus_7500", 10, 12)
        assert all(9 <= lvl <= 10 for lvl in levels)
        assert any(lvl == 9 for lvl in levels)

    def test_unknown_difficulty_is_rejected(self):
        with pytest.raises(KeyError):
            BotGenerator._build_bot_card_levels("typo-medium", 5, 9)

    @pytest.mark.parametrize(
        "alias, tier_key",
        [
            ("lite", "tier_lite_0000"),
            ("easy", "tier_easy_0100"),
            ("medium", "tier_medium_1200"),
            ("hard", "tier_hard_4500"),
            ("max", "tier_max_9000"),
        ],
    )
    def test_public_aliases_share_tier_metadata(self, alias, tier_key):
        alias_meta = BotGenerator._difficulty_metadata(alias)
        tier_meta = BotGenerator._difficulty_metadata(tier_key)

        assert alias_meta == tier_meta


# ============================================================================
# Trophy calculation (no hard cap)
# ============================================================================

class TestTrophyCalculation:
    def test_no_300_cap(self):
        bg = BotGenerator.__new__(BotGenerator)
        for player_trophies in [500, 1000, 3000, 5000, 9000]:
            n = max(25, min(round(player_trophies * 0.08), 500))
            delta = __import__("random").randint(-n, n)
            bot_trophies = max(0, player_trophies + delta)
            assert bot_trophies >= 0
            if player_trophies >= 3000:
                assert bot_trophies > 300

    def test_minimum_zero(self):
        for player_trophies in [0, 5, 10]:
            n = max(25, min(round(player_trophies * 0.08), 500))
            delta = __import__("random").randint(-n, n)
            bot_trophies = max(0, player_trophies + delta)
            assert bot_trophies >= 0


# ============================================================================
# ExtraPass roll
# ============================================================================

class TestExtraPass:
    def test_valid_values(self):
        for _ in range(100):
            ep = BotGenerator._roll_extra_pass()
            assert ep in ("inactive", "active", "ultra")

    def test_active_percentage(self):
        results = [BotGenerator._roll_extra_pass() for _ in range(5000)]
        active = results.count("active")
        ultra = results.count("ultra")
        assert 600 <= active <= 1000
        assert 80 <= ultra <= 250

    def test_probabilities_are_config_driven(self, monkeypatch):
        monkeypatch.setattr(
            "ai.bot_factory.BOT_EXTRA_PASS_ROLL_PROBABILITIES",
            {"ultra": 1.0, "active": 0.0, "inactive": 0.0},
            raising=False,
        )

        assert BotGenerator._roll_extra_pass() == "ultra"


# ============================================================================
# Cosmetic weight picking
# ============================================================================

class TestCosmeticPicking:
    def test_weighted_pick_returns_item(self):
        bg = BotGenerator.__new__(BotGenerator)
        catalog = {
            "starter": [
                {"id": 1, "slug": "avatar_1", "item_type": "avatar", "class": "starter"},
                {"id": 2, "slug": "avatar_2", "item_type": "avatar", "class": "starter"},
            ],
            "epic": [
                {"id": 3, "slug": "avatar_epic", "item_type": "avatar", "class": "epic"},
            ],
        }
        picked = bg._weighted_pick_by_class(catalog, "avatar")
        assert picked is not None
        assert "id" in picked

    def test_weighted_pick_empty_catalog(self):
        bg = BotGenerator.__new__(BotGenerator)
        assert bg._weighted_pick_by_class({}, "avatar") is None

    def test_weighted_pick_no_matching_type(self):
        bg = BotGenerator.__new__(BotGenerator)
        catalog = {"starter": [{"id": 1, "slug": "t", "item_type": "title", "class": "starter"}]}
        assert bg._weighted_pick_by_class(catalog, "profile_background") is None

    def test_starter_more_frequent_than_epic(self):
        bg = BotGenerator.__new__(BotGenerator)
        catalog = {
            "starter": [{"id": 1, "slug": "s1", "item_type": "avatar", "class": "starter"}],
            "epic": [{"id": 2, "slug": "e1", "item_type": "avatar", "class": "epic"}],
        }
        counts = {"starter": 0, "epic": 0}
        for _ in range(1000):
            picked = bg._weighted_pick_by_class(catalog, "avatar")
            if picked:
                counts[picked["class"]] += 1
        assert counts["starter"] > counts["epic"]


# ============================================================================
# Deck validation
# ============================================================================

class TestDeckValidation:
    def test_valid_deck(self):
        assert BotGenerator._is_valid_deck([1] * DECK_SIZE) is True

    def test_invalid_deck(self):
        assert BotGenerator._is_valid_deck([]) is False
        assert BotGenerator._is_valid_deck([1]) is False
        assert BotGenerator._is_valid_deck([1, 2, 3]) is False


class StrictDeckDB:
    def __init__(self, *, disabled=None, catalog=None):
        self._disabled = disabled or []
        self._catalog = catalog or []

    async def get_disabled_card_ids(self):
        return self._disabled

    async def get_cards_list(self):
        return self._catalog


class TestStrictDeckSanitization:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_disabled_cards_are_replaced_to_exact_deck_shape(self):
        db = StrictDeckDB(
            disabled=[101, 102, 103, 104, 105, 106, 107, 108, 109],
            catalog=[
                {"id": 1, "card_type": "hero"},
                {"id": 2, "card_type": "warrior"},
                {"id": 3, "card_type": "warrior"},
                {"id": 4, "card_type": "warrior"},
                {"id": 5, "card_type": "warrior"},
                {"id": 6, "card_type": "warrior"},
                {"id": 7, "card_type": "warrior"},
                {"id": 8, "card_type": "warrior"},
                {"id": 9, "card_type": "potion"},
            ],
        )
        bg = BotGenerator(db)

        deck = await bg._sanitize_deck([101, 102, 103, 104, 105, 106, 107, 108, 109])

        assert len(deck) == DECK_SIZE
        assert not set(deck) & set(db._disabled)
        hero_count = sum(1 for card_id in deck if card_id == 1)
        assert hero_count == 1

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sanitization_fails_when_catalog_cannot_build_valid_deck(self):
        db = StrictDeckDB(
            disabled=[1, 2, 3],
            catalog=[
                {"id": 10, "card_type": "warrior"},
                {"id": 11, "card_type": "warrior"},
            ],
        )
        bg = BotGenerator(db)

        with pytest.raises(ValueError, match="valid bot deck"):
            await bg._sanitize_deck([1, 2, 3])


class PersistFailureDB(StrictDeckDB):
    def __init__(self):
        super().__init__(
            disabled=[],
            catalog=[
                {"id": 1, "card_type": "hero"},
                {"id": 2, "card_type": "warrior"},
                {"id": 3, "card_type": "warrior"},
                {"id": 4, "card_type": "warrior"},
                {"id": 5, "card_type": "warrior"},
                {"id": 6, "card_type": "warrior"},
                {"id": 7, "card_type": "warrior"},
                {"id": 8, "card_type": "warrior"},
                {"id": 9, "card_type": "potion"},
            ],
        )

    async def get_bot_deck_from_donor(self, *_args, **_kwargs):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]

    async def get_random_users_with_avatars(self, *_args, **_kwargs):
        return []

    async def get_cosmetic_catalog_by_class(self):
        return {}

    async def get_next_bot_id(self):
        return 900000123

    async def create_or_update_bot_profile(self, **_kwargs):
        raise RuntimeError("db down")


class TestPersistenceFallback:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_generate_bot_persistence_failure_returns_unpersisted_temp_payload(self):
        bg = BotGenerator(PersistFailureDB())

        payload = await bg._generate_bot(player_id=42, player_trophies=1200)

        assert payload["persisted"] is False
        assert payload["source_bot_id"] == 900000123
        assert payload["user_id"] < 0
        assert len(payload["deck_ids"]) == DECK_SIZE


# ============================================================================
# Payload building (no more card_level/level)
# ============================================================================

class TestPayloadBuilding:
    def test_payload_has_all_keys(self):
        payload = BotGenerator._build_payload(
            bot_id=900000001,
            deck_ids=[1, 2, 3],
            bot_name="TestBot",
            bot_avatar_url="/test.png",
            bot_trophies=1500,
            difficulty="hard",
            bot_league=4,
            cosmetics={"avatar": {"id": 1, "slug": "a1"}},
            extra_pass="active",
            reused=False,
        )
        for key in ("user_id", "deck_ids", "name", "avatar_url",
                     "difficulty", "difficulty_label", "strength_tier", "brain_profile",
                     "selection", "temperature", "card_level_policy", "deck_policy",
                     "trophies", "league", "extra_pass", "cosmetics", "reused"):
            assert key in payload, f"Missing key: {key}"
        # No more card_level/level
        assert "card_level" not in payload
        assert "level" not in payload

    def test_payload_reused_flag(self):
        p1 = BotGenerator._build_payload(900000001, [1, 2, 3], "B", None, 100, "easy", 1, {}, "inactive", True)
        p2 = BotGenerator._build_payload(900000002, [1, 2, 3], "B", None, 100, "easy", 1, {}, "inactive", False)
        assert p1["reused"] is True
        assert p2["reused"] is False

    def test_fallback_payload(self):
        payload = BotGenerator._build_fallback_payload(
            900000001, [1, 2, 3], "FB", None, 500, "medium", {}, "inactive"
        )
        assert payload["user_id"] < 0
        assert payload["persisted"] is False
        assert payload["source_bot_id"] == 900000001
        assert payload["difficulty"] == "medium"
        assert payload["brain_profile"] == "extra-lr-v4-opti"
        assert payload["league"] > 0
        assert "cosmetics" in payload

    def test_fallback_payload_marks_unpersisted_temp_bot(self):
        payload = BotGenerator._build_fallback_payload(
            900000001, [1] * DECK_SIZE, "FB", None, 500, "medium", {}, "inactive"
        )

        assert payload["persisted"] is False
        assert payload["user_id"] < 0
