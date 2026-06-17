from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from battle_engine import BattleEngine
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from infrastructure.database import Database


def _card(
    *,
    card_id: int,
    card_type: CardType = CardType.WARRIOR,
    mana_cost: int = 1,
    attack: int = 1,
    hp: int = 1,
    mechanics: list[str] | None = None,
    level: int = 1,
) -> CardInstance:
    return CardInstance(
        card_id=card_id,
        name=f"Card {card_id}",
        card_type=card_type,
        mana_cost=mana_cost,
        attack=attack,
        hp=hp,
        max_hp=hp,
        mechanics=mechanics or [],
        level=level,
    )


def _hero(card_id: int) -> CardInstance:
    return _card(
        card_id=card_id,
        card_type=CardType.HERO,
        mana_cost=0,
        attack=0,
        hp=30,
        mechanics=[],
    )


def _engine_with_action_context() -> BattleEngine:
    p1_hand = [
        _card(card_id=37, mana_cost=2, attack=3, hp=4, mechanics=["taunt"], level=5),
        _card(card_id=38, mana_cost=9, attack=9, hp=9, mechanics=["charge"], level=2),
    ]
    state = GameState(
        p1=PlayerState(
            user_id=101,
            hero=_hero(1),
            mana=2,
            max_mana=2,
            hand=p1_hand,
            deck=[_card(card_id=39, mana_cost=3, attack=2, hp=2)],
        ),
        p2=PlayerState(
            user_id=202,
            hero=_hero(2),
            mana=0,
            max_mana=0,
            hand=[_card(card_id=44, mana_cost=1, attack=1, hp=1)],
            deck=[_card(card_id=45, mana_cost=4, attack=4, hp=4)],
        ),
        current_turn_owner_id=101,
        turn_number=1,
        status=GameStatus.ONGOING,
    )
    engine = BattleEngine(match_id="ctx-test", player_ids=[101, 202])
    engine._arena = ArenaEnvironment(state, apply_start_effects=False)
    return engine


def test_battle_action_context_uses_train_v3_card_params_without_id_only_payloads():
    engine = _engine_with_action_context()

    engine.record_analytics_action(101, {"type": "play_card", "card_ref": 0})

    sample = engine._analytics_actions[0]
    context = sample["context_json"]
    acting_hand = context["acting_hand"]
    available = context["available_hand_cards"]

    assert context["schema"] == "train_v3_action_context_v1"
    assert context["card_params_schema"] == "train_v3_card_params_v1"
    assert sample["acting_player"] == 1
    assert sample["state_json"]["p1"]["deck"]
    assert sample["state_json"]["p2"]["deck"]

    first = acting_hand[0]["card"]
    assert first == {
        "schema": "train_v3_card_params_v1",
        "type": "warrior",
        "mana_cost": 2,
        "attack": 3,
        "hp": 4,
        "max_hp": 4,
        "mechanics": ["taunt"],
        "is_ready": False,
        "is_frozen": False,
        "level": 5,
    }
    assert "id" not in first
    assert "card_id" not in first
    assert "mana" not in first
    assert "atk" not in first

    assert [item["hand_index"] for item in acting_hand] == [0, 1]
    assert [item["hand_index"] for item in available] == [0]
    assert context["selected_card"]["hand_index"] == 0
    assert context["selected_card"]["card"] == first


class _FakeDatabase(Database):
    def __init__(self, rows):
        self._pool = object()
        self.rows = rows
        self.last_query = ""

    async def fetch(self, query, *args):
        self.last_query = query
        return self.rows


@pytest.mark.asyncio
async def test_export_battle_dataset_includes_action_context_and_deck_param_snapshots():
    metadata = {
        "deck_param_snapshots": {
            "p1": {"cards": [{"slot": 0, "card": {"type": "hero", "mana_cost": 0}}]},
            "p2": {"cards": [{"slot": 0, "card": {"type": "hero", "mana_cost": 0}}]},
        }
    }
    db = _FakeDatabase(
        [
            {
                "id": 1,
                "battle_id": "m1",
                "turn_number": 1,
                "acting_player": 1,
                "acting_user_id": 101,
                "is_bot": False,
                "state_json": json.dumps({"turn": 1}),
                "action_json": json.dumps({"type": "end_turn"}),
                "context_json": json.dumps({"schema": "train_v3_action_context_v1", "acting_user_id": 101}),
                "quality_score": None,
                "created_at": datetime.now(timezone.utc),
                "game_mode": "classic",
                "match_type": "classic",
                "winner_user_id": 101,
                "p1_user_id": 101,
                "p2_user_id": 202,
                "battle_metadata": json.dumps(metadata),
            }
        ]
    )

    rows = await db.export_train_v2_battle_dataset(days=1, limit=1)

    assert "ba.context_json" in db.last_query
    assert rows[0]["format"] == "train_v2_admin_battle_action_jsonl_v2"
    assert rows[0]["format_version"] == 2
    assert rows[0]["dataset_schema"] == "train_v3_battle_action_context_v1"
    assert rows[0]["acting_user_id"] is None
    assert rows[0]["state_json"] == {"turn": 1}
    assert rows[0]["action_json"] == {"type": "end_turn"}
    assert rows[0]["context_json"] == {"schema": "train_v3_action_context_v1"}
    assert "acting_user_id" not in rows[0]["context_json"]
    assert rows[0]["deck_param_snapshots"] == metadata["deck_param_snapshots"]
    assert rows[0]["battle_metadata"] == metadata
