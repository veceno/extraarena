import math

import pytest

from battle_engine import BattleEngine
from bot_ai import BotAI
from database import Card, RARITY_STATS


def test_card_stat_scaling(mocker):
    """
    Проверяем, что рост статов для Divine-карты с 15% скейлом
    корректно масштабируется к 10 уровню.
    """

    # Гарантируем рост 0.15 для divine, даже если глобальный словарь изменится.
    mocker.patch.dict(RARITY_STATS, {"divine": 0.15}, clear=False)

    base_attack = 100
    base_hp = 200
    level = 10

    # Используем from_row, чтобы Card подтянул множитель роста из редкости.
    card = Card.from_row(
        {
            "id": 999,
            "name": "Divine Test",
            "description": None,
            "rarity": "divine",
            "power": 0,
            "mana_cost": 3,
            "base_attack": base_attack,
            "base_hp": base_hp,
            "mechanics": [],
        }
    )

    stats = card.get_current_stats(level)

    # Формула роста: base * (1 + growth) ** (level - 1)
    growth = RARITY_STATS["divine"]
    expected_multiplier = (1 + growth) ** (level - 1)
    expected_attack = int(round(base_attack * expected_multiplier))
    expected_hp = int(round(base_hp * expected_multiplier))

    # Проверяем, что множитель близок к 3.5x (≈3.518) и числа совпадают.
    assert math.isclose(expected_multiplier, 3.5, rel_tol=0.02)
    assert stats["attack"] == expected_attack
    assert stats["hp"] == expected_hp


def test_deck_cycle_and_mana():
    """
    Имитируем 15 ходов игрока: мана должна остановиться на 10,
    а колода обязана перетянуть карты из сброса хотя бы один раз.
    """

    engine = BattleEngine(p1_deck_ids=["c1", "c2", "c3"], p2_deck_ids=["e1", "e2", "e3"])
    p1 = engine.p1_state

    # Чистим колоду и заранее кладем карты в сброс, чтобы вынудить рецикл.
    p1.draw_pile.clear()
    p1.discard_pile = ["d1", "d2"]

    refilled_from_discard = False
    for _ in range(15):
        pre_empty_draw = not p1.draw_pile
        pre_discard_nonempty = bool(p1.discard_pile)

        engine.start_turn(p1.user_id)

        # Фиксируем момент, когда сброс перенесен в колоду.
        if pre_empty_draw and pre_discard_nonempty and p1.draw_pile:
            refilled_from_discard = True

    assert p1.max_mana == 10
    assert refilled_from_discard


def test_bot_ai_taunt_priority():
    """
    Проверяем, что ИИ всегда выбирает целью таунт, даже если есть более слабая цель.
    """

    taunt_instance_id = 201
    non_taunt_instance_id = 202

    battle_state = {
        "current_player": 1,
        "players": {
            1: {
                # Наши атакующие — один активный юнит.
                "board": [
                    {
                        "instance_id": 101,
                        "attack_current": 5,
                        "hp_current": 5,
                        "mechanics": [],
                        "can_attack": True,
                        "is_asleep": False,
                    }
                ],
            },
            2: {
                # У оппонента есть таунт и обычный юнит; таунт должен стать целью.
                "board": [
                    {
                        "instance_id": taunt_instance_id,
                        "attack_current": 2,
                        "hp_current": 6,
                        "mechanics": ["taunt"],
                        "can_attack": False,
                        "is_asleep": False,
                    },
                    {
                        "instance_id": non_taunt_instance_id,
                        "attack_current": 10,
                        "hp_current": 1,
                        "mechanics": [],
                        "can_attack": False,
                        "is_asleep": False,
                    },
                ]
            },
        },
    }

    action = BotAI.decide_turn(battle_state)

    assert action["action"] == "attack"
    assert action["target_id"] == taunt_instance_id







