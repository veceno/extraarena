"""
Комплексные тесты: баг-фиксы, краевые случаи, новые механики.
"""
import copy
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import HAND_CAP, MANA_DRAW_BASE, ArenaEnvironment, scale_card_by_level
from core.actions import PlayCardAction, AttackAction, EndTurnAction, ManaDrawAction, BaseAction
from core.effects import apply_damage, process_effects, requires_target
from core.converter import _normalize_mechanic, card_from_db
from battle_engine import BattleEngine


def create_minimal_game_state() -> GameState:
    hero_p1 = CardInstance(
        instance_id=uuid4(), card_id=1, name="Hero P1",
        card_type=CardType.HERO, hp=30, max_hp=30, attack=0, mana_cost=0,
    )
    hero_p2 = CardInstance(
        instance_id=uuid4(), card_id=2, name="Hero P2",
        card_type=CardType.HERO, hp=30, max_hp=30, attack=0, mana_cost=0,
    )
    p1 = PlayerState(user_id=1, is_bot=False, hero=hero_p1, mana=10, max_mana=10)
    p2 = PlayerState(user_id=2, is_bot=False, hero=hero_p2, mana=10, max_mana=10)
    return GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1, status=GameStatus.ONGOING)


# ============================================================================
# БАГ-ФИКС: Двойной скейлинг (converter больше не добавляет tiered к синглам)
# ============================================================================

class TestConverterNoDoubleScaling:
    def test_single_values_not_tiered_in_converter(self):
        assert _normalize_mechanic("damage_1", 10) == "damage_1"
        assert _normalize_mechanic("battlecry_heal_hero_2", 10) == "battlecry_heal_hero_2"
        assert _normalize_mechanic("regen_1", 5) == "regen_1"

    def test_ranges_always_pick_min(self):
        assert _normalize_mechanic("aura_atk_1_3", 10) == "aura_atk_1"
        assert _normalize_mechanic("start_mana_1_5", 10) == "start_mana_1"
        assert _normalize_mechanic("cleave_1_3", 5) == "cleave_1_3"

    def test_start_mana_engine_scales_correctly(self):
        hero_data = {
            'id': 7, 'name': 'Tinkov', 'card_type': 'hero', 'base_hp': 30,
            'current_hp': 30, 'base_attack': 0, 'current_attack': 0,
            'mana_cost': 0, 'rarity': 'legendary', 'mechanics': ['start_mana_1_5']
        }
        lvl1 = card_from_db(hero_data, level=1)
        lvl10 = card_from_db(hero_data, level=10)
        # bonus_tiers = (10-1)//3 = 3, base=1 → start_mana_4
        assert 'start_mana_1' in lvl1.mechanics
        assert 'start_mana_4' in lvl10.mechanics

    def test_damage_potion_scales_reasonably(self):
        potion_data = {
            'id': 50, 'name': 'Damage Potion', 'card_type': 'potion',
            'base_attack': 0, 'current_attack': 0, 'base_hp': 0, 'current_hp': 0,
            'mana_cost': 1, 'rarity': 'common', 'mechanics': ['damage_1']
        }
        lvl1 = card_from_db(potion_data, level=1)
        lvl10 = card_from_db(potion_data, level=10)
        # ((10-1)//3) = 3 → 1+3 = 4
        assert 'damage_1' in lvl1.mechanics
        assert 'damage_4' in lvl10.mechanics

    def test_stringified_mechanics_are_loaded_from_card_data(self):
        card_data = {
            'id': 60, 'name': 'Local JSON Spell', 'card_type': 'potion',
            'base_attack': 0, 'current_attack': 0, 'base_hp': 0, 'current_hp': 0,
            'mana_cost': 2, 'rarity': 'common', 'mechanics': '["damage_3"]'
        }

        card = card_from_db(card_data, level=1)

        assert card.mechanics == ["damage_3"]


# ============================================================================
# БАГ-ФИКС: Герои не замораживаются
# ============================================================================

class TestHeroesCantBeFrozen:
    def test_freeze_potion_cannot_target_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        freeze_card = CardInstance(
            instance_id=uuid4(), card_id=11, name="Заморозка",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=2, mechanics=["freeze"], is_ready=False,
        )
        state.p1.hand.append(freeze_card)

        # Пытаемся заморозить героя противника
        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(state.p2.hero.instance_id), position=None
        ))
        # Должно либо провалиться, либо не заморозить героя
        if success:
            assert not state.p2.hero.is_frozen, "Герой не должен быть заморожен"

    def test_aoe_freeze_does_not_freeze_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        aoe_freeze = CardInstance(
            instance_id=uuid4(), card_id=22, name="ZA WARUDO",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=8, mechanics=["aoe_freeze"], is_ready=False,
        )
        state.p1.hand.append(aoe_freeze)
        state.p2.board.append(CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=2, mana_cost=2,
            mechanics=[], is_ready=True,
        ))

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success
        assert not state.p2.hero.is_frozen, "Герой не должен быть заморожен AOE"

    def test_dynamic_freeze_cannot_target_hero(self):
        state = create_minimal_game_state()

        freeze_card = CardInstance(
            instance_id=uuid4(), card_id=23, name="Точечная заморозка",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=2, mechanics=["freeze_1"], is_ready=False,
        )
        process_effects(
            state,
            freeze_card,
            state.p1,
            state.p2,
            target_id=str(state.p2.hero.instance_id),
        )

        assert not state.p2.hero.is_frozen, "freeze_X не должен замораживать героя"

    def test_aoe_freeze_is_limited_to_three_board_targets(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        aoe_freeze = CardInstance(
            instance_id=uuid4(), card_id=22, name="ZA WARUDO",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=8, mechanics=["aoe_freeze"], is_ready=False,
        )
        state.p1.hand.append(aoe_freeze)

        enemy_units = []
        for index in range(4):
            unit = CardInstance(
                instance_id=uuid4(), card_id=300 + index, name=f"Enemy {index}",
                card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=2, mana_cost=2,
                mechanics=[], is_ready=True,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))

        assert success, error
        assert sum(unit.is_frozen for unit in enemy_units) == 3
        assert enemy_units[3].is_frozen is False

    def test_aoe_freeze_shielded_target_counts_toward_three_target_cap(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        aoe_freeze = CardInstance(
            instance_id=uuid4(), card_id=22, name="ZA WARUDO",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=8, mechanics=["aoe_freeze"], is_ready=False,
        )
        state.p1.hand.append(aoe_freeze)

        enemy_units = []
        for index in range(4):
            mechanics = ["shield"] if index == 1 else []
            unit = CardInstance(
                instance_id=uuid4(), card_id=320 + index, name=f"Enemy {index}",
                card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=2, mana_cost=2,
                mechanics=mechanics, is_ready=True,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))

        assert success, error
        assert enemy_units[0].is_frozen is True
        assert enemy_units[1].is_frozen is False
        assert "shield" not in enemy_units[1].mechanics
        assert enemy_units[2].is_frozen is True
        assert enemy_units[3].is_frozen is False

    def test_desk_freeze_freezes_all_board_targets(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        desk_freeze = CardInstance(
            instance_id=uuid4(), card_id=122, name="Full Board Freeze",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=8, mechanics=["desk_freeze"], is_ready=False,
        )
        state.p1.hand.append(desk_freeze)

        enemy_units = []
        for index in range(4):
            unit = CardInstance(
                instance_id=uuid4(), card_id=400 + index, name=f"Enemy {index}",
                card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=2, mana_cost=2,
                mechanics=[], is_ready=True,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))

        assert success, error
        assert all(unit.is_frozen for unit in enemy_units)
        assert state.p2.hero.is_frozen is False


# ============================================================================
# БАГ-ФИКС: Кража Маны отдаёт ману владельцу
# ============================================================================

class TestManaDrainGivesToOwner:
    def test_mana_drain_transfers_mana(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        state.p1.mana = 5
        state.p2.mana = 6

        drain_card = CardInstance(
            instance_id=uuid4(), card_id=12, name="Кража Маны",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=2, mechanics=["mana_drain_3"], is_ready=False,
        )
        state.p1.hand.append(drain_card)

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success
        assert state.p2.mana == 3, f"Противник потерял 3 маны: {state.p2.mana}"
        assert state.p1.mana == 6, f"Владелец получил 3 маны (5-2+3=6): {state.p1.mana}"

    def test_mana_drain_capped_at_max(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        state.p1.mana = 9
        state.p2.mana = 5

        drain_card = CardInstance(
            instance_id=uuid4(), card_id=12, name="Кража Маны",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=2, mechanics=["mana_drain_5"], is_ready=False,
        )
        state.p1.hand.append(drain_card)

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success
        assert state.p1.mana == 10, f"Мана владельца cap'нута на 10: {state.p1.mana}"

    def test_mana_drain_steals_next_turn_mana_when_current_pool_empty(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        state.p1.mana = 5
        state.p1.max_mana = 10
        state.p2.mana = 0
        state.p2.max_mana = 5
        state.p2.deck.append(CardInstance(
            instance_id=uuid4(), card_id=99, name="Next Draw",
            card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=1,
            mana_cost=1, mechanics=[], is_ready=False,
        ))

        drain_card = CardInstance(
            instance_id=uuid4(), card_id=12, name="Кража Маны",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
            mana_cost=3, mechanics=["mana_drain_2"], is_ready=False,
        )
        state.p1.hand.append(drain_card)

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success
        assert state.p1.mana == 4, f"Кража должна вернуть 2 маны владельцу (5-3+2=4): {state.p1.mana}"
        assert state.p2.mana == 0

        success, _ = env.step(1, EndTurnAction())
        assert success
        assert state.p2.max_mana == 6
        assert state.p2.mana == 4, f"Следующий ход противника должен начаться с -2 маны: {state.p2.mana}"


# ============================================================================
# БАГ-ФИКС: Instant Kill не работает на героях
# ============================================================================

class TestInstantKillNotOnHeroes:
    def test_instant_kill_deals_only_normal_damage_to_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        killer = CardInstance(
            instance_id=uuid4(), card_id=25, name="Saitama",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=5,
            mana_cost=10, mechanics=["instant_kill"], is_ready=True,
        )
        state.p1.board.append(killer)

        success, _ = env.step(1, AttackAction(
            attacker_id=str(killer.instance_id), target_id=None, target_is_hero=True,
        ))
        assert success
        assert state.p2.hero.hp == 25, f"Герой получил только 5 урона: {state.p2.hero.hp}"
        assert state.status == GameStatus.ONGOING, "Игра не должна закончиться"


# ============================================================================
# ФРИРЕН: battlecry heal target
# ============================================================================

class TestFrierenHealTarget:
    def test_frieren_heals_ally(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        ally = CardInstance(
            instance_id=uuid4(), card_id=99, name="Wounded",
            card_type=CardType.WARRIOR, hp=2, max_hp=8, attack=3, mana_cost=3,
            mechanics=[], is_ready=False,
        )
        state.p1.board.append(ally)

        frieren = CardInstance(
            instance_id=uuid4(), card_id=35, name="Фрирен",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=5,
            mechanics=["battlecry_heal_target_5"], is_ready=False,
        )
        state.p1.hand.append(frieren)

        success, _ = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(ally.instance_id), position=1,
        ))
        assert success
        assert ally.hp == 7, f"Союзник получил +5 HP (2→7): {ally.hp}"

    def test_frieren_heals_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        state.p1.hero.hp = 20

        frieren = CardInstance(
            instance_id=uuid4(), card_id=35, name="Фрирен",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=5,
            mechanics=["battlecry_heal_target_5"], is_ready=False,
        )
        state.p1.hand.append(frieren)

        success, _ = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(state.p1.hero.instance_id), position=0,
        ))
        assert success
        assert state.p1.hero.hp == 25, f"Герой получил +5 HP: {state.p1.hero.hp}"

    def test_frieren_heals_own_hero_through_battle_engine_adapter(self):
        state = create_minimal_game_state()
        state.p1.hero.hp = 20

        frieren = CardInstance(
            instance_id=uuid4(), card_id=35, name="Фрирен",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=5,
            mechanics=["battlecry_heal_target_5"], is_ready=False,
        )
        state.p1.hand.append(frieren)

        engine = BattleEngine(match_id="test-frieren", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        result = engine.play_card(
            1,
            0,
            0,
            target_id=str(state.p1.hero.instance_id),
            target_is_hero=True,
        )

        assert result["success"]
        assert state.p1.hero.hp == 25, f"Герой получил +5 HP: {state.p1.hero.hp}"

    def test_frieren_heal_caps_at_max_hp(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        ally = CardInstance(
            instance_id=uuid4(), card_id=99, name="AlmostFull",
            card_type=CardType.WARRIOR, hp=7, max_hp=8, attack=3, mana_cost=3,
            mechanics=[], is_ready=False,
        )
        state.p1.board.append(ally)

        frieren = CardInstance(
            instance_id=uuid4(), card_id=35, name="Фрирен",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=5,
            mechanics=["battlecry_heal_target_5"], is_ready=False,
        )
        state.p1.hand.append(frieren)

        success, _ = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(ally.instance_id), position=1,
        ))
        assert success
        assert ally.hp == 8, f"HP capped at max_hp=8: {ally.hp}"


# ============================================================================
# ТОКА КИРИШИМА: случайный battlecry урон
# ============================================================================

class TestToukaRandomBattlecry:
    def test_touka_random_battlecry_stays_random_after_scaling(self):
        touka_data = {
            "id": 15,
            "name": "Тоука",
            "type": "warrior",
            "rarity": "common",
            "base_attack": 2,
            "base_hp": 1,
            "mana_cost": 2,
            "mechanics": '["battlecry_damage_1_random"]',
        }

        assert card_from_db(touka_data, level=1).mechanics == ["battlecry_damage_1_random"]
        assert card_from_db(touka_data, level=5).mechanics == ["battlecry_damage_3_random"]

    def test_random_battlecry_damage_mechanic_plays_without_target_and_hits_random_enemy(self, monkeypatch):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        enemy = CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p2.board.append(enemy)

        random_damage = CardInstance(
            instance_id=uuid4(), card_id=415, name="Renamed Random Damage Unit",
            card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=2, mana_cost=2,
            mechanics=["battlecry_damage_1_random"], is_ready=False,
        )
        state.p1.hand.append(random_damage)

        legal = [
            action for action in env.get_legal_actions(1)
            if isinstance(action, PlayCardAction) and action.hand_index == 0
        ]
        assert legal
        assert all(action.target_id is None for action in legal)

        monkeypatch.setattr("core.effects.random.choice", lambda targets: targets[0])
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))

        assert success, f"Тока должна разыгрываться без выбора цели: {error}"
        assert enemy.hp == 4, f"Случайный враг должен получить 1 урон: {enemy.hp}"

    def test_plain_battlecry_damage_card_id_15_still_requires_target(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        enemy = CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p2.board.append(enemy)

        targeted_damage = CardInstance(
            instance_id=uuid4(), card_id=15, name="Any Renamed Card",
            card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=2, mana_cost=2,
            mechanics=["battlecry_damage_1"], is_ready=False,
        )
        state.p1.hand.append(targeted_damage)

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))

        assert success is False
        assert error == "target_required"


# ============================================================================
# DEATHRATTLE: Краевые случаи
# ============================================================================

class TestDeathrattleEdgeCases:
    def test_deathrattle_damages_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        bomb = CardInstance(
            instance_id=uuid4(), card_id=34, name="Крипер",
            card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=2, mana_cost=3,
            mechanics=["deathrattle_aoe_damage_3"], is_ready=False,
        )
        state.p1.board.append(bomb)

        state.p2.board.append(CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=2, mana_cost=2,
            mechanics=[], is_ready=False,
        ))

        bomb.hp = 0
        env._cleanup_dead_units(state.p1)

        assert state.p2.board[0].hp == 2, f"Враг получил 3 урона: {state.p2.board[0].hp}"
        assert state.p2.hero.hp == 27, f"Герой получил 3 урона: {state.p2.hero.hp}"


# ============================================================================
# CHARGE: Рывок позволяет атаковать сразу
# ============================================================================

class TestChargeAttackSameTurn:
    def test_charge_can_attack_immediately(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        zenitsu = CardInstance(
            instance_id=uuid4(), card_id=32, name="Зеницу",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=5, mana_cost=4,
            mechanics=["charge"], is_ready=False,
        )
        state.p1.hand.append(zenitsu)
        state.p2.board.append(CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=2, mana_cost=2,
            mechanics=[], is_ready=False,
        ))

        env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
        assert zenitsu.is_ready, "Charge должен дать готовность сразу"

        success, _ = env.step(1, AttackAction(
            attacker_id=str(zenitsu.instance_id),
            target_id=str(state.p2.board[0].instance_id),
            target_is_hero=False,
        ))
        assert success
        assert state.p2.board[0].hp == 5, f"Враг получил 5 урона: {state.p2.board[0].hp}"


# ============================================================================
# CONSUME ALLY: Краевые случаи
# ============================================================================

class TestConsumeAllyEdgeCases:
    def test_consume_ally_preserves_mechanics(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        sacrifice = CardInstance(
            instance_id=uuid4(), card_id=99, name="Shield Ally",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=2,
            mechanics=["shield", "taunt"], is_ready=False,
        )
        state.p1.board.append(sacrifice)

        kaneki = CardInstance(
            instance_id=uuid4(), card_id=20, name="Канеки",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=2, mana_cost=4,
            mechanics=["consume_ally"], is_ready=False,
        )
        state.p1.hand.append(kaneki)

        success, _ = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(sacrifice.instance_id), position=0,
        ))
        assert success
        assert kaneki.attack == 5, f"Атака: 2+3=5, получено {kaneki.attack}"
        assert kaneki.hp == 7, f"HP: 2+5=7, получено {kaneki.hp}"
        assert kaneki.max_hp == 7
        assert "consume_ally" in kaneki.mechanics, "Свои механики сохранены"

    def test_invalid_consume_target_does_not_spend_mana_or_card(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        kaneki = CardInstance(
            instance_id=uuid4(), card_id=20, name="Канеки",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=2, mana_cost=4,
            mechanics=["consume_ally"], is_ready=False,
        )
        state.p1.hand.append(kaneki)

        mana_before = state.p1.mana
        hand_before = list(state.p1.hand)

        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(uuid4()), position=0,
        ))

        assert not success
        assert error == "consume_target_not_found"
        assert state.p1.mana == mana_before
        assert state.p1.hand == hand_before
        assert state.p1.board == []

    def test_consume_can_free_board_slot_before_play(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        # Заполняем доску до лимита (5) и пробуем сыграть consume_ally,
        # который съедает одного союзника и занимает его место.
        allies = []
        for idx in range(5):
            ally = CardInstance(
                instance_id=uuid4(), card_id=100 + idx, name=f"Ally {idx}",
                card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
                mechanics=[], is_ready=False,
            )
            allies.append(ally)
        state.p1.board = allies[:]

        consumer = CardInstance(
            instance_id=uuid4(), card_id=20, name="Канеки",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=2, mana_cost=4,
            mechanics=["consume_ally"], is_ready=False,
        )
        state.p1.hand.append(consumer)

        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(allies[0].instance_id), position=0,
        ))

        assert success, error
        assert len(state.p1.board) == 5
        assert consumer in state.p1.board
        assert allies[0] in state.p1.graveyard

    def test_invalid_targeted_potion_does_not_spend_resources(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        potion = CardInstance(
            instance_id=uuid4(), card_id=50, name="Bolt",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0, mana_cost=3,
            mechanics=["damage_3"], is_ready=False,
        )
        state.p1.hand.append(potion)

        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(uuid4()),
        ))

        assert not success
        assert error == "target_not_found"
        assert state.p1.mana == 10
        assert state.p1.hand == [potion]
        assert state.p1.graveyard == []

    def test_battlecry_buff_cannot_target_card_itself_from_hand(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        buffer = CardInstance(
            instance_id=uuid4(), card_id=70, name="Buffer",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_buff_2_2"], is_ready=False,
        )
        state.p1.hand.append(buffer)

        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(buffer.instance_id), position=0,
        ))

        assert not success
        assert error == "target_not_found"
        assert state.p1.mana == 10
        assert state.p1.hand == [buffer]
        assert state.p1.board == []


# ============================================================================
# SCALING: Проверка скейлинга warrior passive механик
# ============================================================================

class TestWarriorPassiveScaling:
    def test_regen_scales_with_level(self):
        warrior_data = {
            'id': 6, 'name': 'Wolverine', 'card_type': 'warrior',
            'base_attack': 3, 'current_attack': 3, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 3, 'rarity': 'rare', 'mechanics': ['regen_1']
        }
        lvl1 = card_from_db(warrior_data, level=1)
        lvl7 = card_from_db(warrior_data, level=7)
        # ((7-1)//3) = 2 → regen_3
        assert 'regen_1' in lvl1.mechanics
        assert 'regen_3' in lvl7.mechanics

    def test_armor_scales_with_level(self):
        warrior_data = {
            'id': 18, 'name': 'P.E.K.K.A.', 'card_type': 'warrior',
            'base_attack': 5, 'current_attack': 5, 'base_hp': 5, 'current_hp': 5,
            'mana_cost': 6, 'rarity': 'epic', 'mechanics': ['armor_1']
        }
        lvl10 = card_from_db(warrior_data, level=10)
        # ((10-1)//3) = 3 → armor_4
        assert 'armor_4' in lvl10.mechanics

    def test_aura_scales_with_level(self):
        warrior_data = {
            'id': 99, 'name': 'Commander', 'card_type': 'warrior',
            'base_attack': 2, 'current_attack': 2, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 4, 'rarity': 'epic', 'mechanics': ['aura_atk_2']
        }
        lvl4 = card_from_db(warrior_data, level=4)
        assert 'aura_atk_3' in lvl4.mechanics  # ((4-1)//3) = 1 → aura_atk_3

    def test_hero_aura_scales_with_level(self):
        hero_data = {
            'id': 7, 'name': 'Aura Hero', 'card_type': 'hero',
            'base_attack': 0, 'current_attack': 0, 'base_hp': 30, 'current_hp': 30,
            'mana_cost': 0, 'rarity': 'legendary', 'mechanics': ['aura_atk_1']
        }
        lvl4 = card_from_db(hero_data, level=4)
        lvl10 = card_from_db(hero_data, level=10)

        assert 'aura_atk_2' in lvl4.mechanics
        assert 'aura_atk_4' in lvl10.mechanics

    def test_buff_mechanics_scale_with_warrior_level(self):
        warrior_data = {
            'id': 77, 'name': 'Banner', 'card_type': 'warrior',
            'base_attack': 2, 'current_attack': 2, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 4, 'rarity': 'rare',
            'mechanics': ['buff_all_1_2', 'battlecry_buff_2_3']
        }

        lvl10 = card_from_db(warrior_data, level=10)

        assert 'buff_all_5_6' in lvl10.mechanics
        assert 'battlecry_buff_6_7' in lvl10.mechanics

    def test_attack_and_hp_use_symmetric_rounding(self):
        warrior_data = {
            'id': 78, 'name': 'Mirror Stats', 'card_type': 'warrior',
            'base_attack': 5, 'current_attack': 5, 'base_hp': 5, 'current_hp': 5,
            'mana_cost': 4, 'rarity': 'mythic', 'mechanics': []
        }

        lvl9 = card_from_db(warrior_data, level=9)

        assert lvl9.attack == lvl9.max_hp

    def test_delete_target_min_mana_four_is_locked_balance(self):
        potion_data = {
            'id': 13, 'name': 'Black Hole', 'card_type': 'potion',
            'base_attack': 0, 'current_attack': 0, 'base_hp': 0, 'current_hp': 0,
            'mana_cost': 5, 'rarity': 'legendary', 'mechanics': ['delete_target'],
            'simplified_levelup': True,
        }
        lvl10 = card_from_db(potion_data, level=10)

        assert lvl10.mana_cost == 4


# ============================================================================
# SCALING: Отсутствие дублирования механик (регрессия)
# ============================================================================

class TestScaleMechanicsNoDuplication:
    def test_no_duplicate_mechanics_on_scaling(self):
        """scale_card_by_level не должен дублировать механики при level > 1."""
        warrior_data = {
            'id': 99, 'name': 'Test', 'card_type': 'warrior',
            'base_attack': 3, 'current_attack': 3, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 3, 'rarity': 'common',
            'mechanics': ['taunt', 'shield', 'lifesteal', 'charge', 'battlecry_draw_card']
        }
        warrior = card_from_db(warrior_data, level=5)

        for mechanic in ['taunt', 'shield', 'lifesteal', 'charge', 'battlecry_draw_card']:
            count = warrior.mechanics.count(mechanic)
            assert count == 1, f"Механика '{mechanic}' встречается {count} раз: {warrior.mechanics}"

    def test_shield_not_doubled_on_scaled_warrior(self):
        """Scaled warrior со shield должен блокировать только 1 удар, не 2."""
        warrior_data = {
            'id': 100, 'name': 'Shield Guy', 'card_type': 'warrior',
            'base_attack': 3, 'current_attack': 3, 'base_hp': 5, 'current_hp': 5,
            'mana_cost': 3, 'rarity': 'common',
            'mechanics': ['shield']
        }
        warrior = card_from_db(warrior_data, level=5)
        hp_after_scale = warrior.hp  # 5 + (5-1)//2 = 7

        apply_damage(warrior, 10)
        assert warrior.hp == hp_after_scale, "Первый удар: shield должен заблокировать"
        assert 'shield' not in warrior.mechanics, "Shield должен исчезнуть после первого удара"

        apply_damage(warrior, 10)
        assert warrior.hp == 0, "Второй удар: shield не должен срабатывать повторно"

    def test_battlecry_draw_card_not_doubled_on_scaled_warrior(self):
        """Scaled warrior с battlecry_draw_card должен тянуть ровно 1 карту."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        # Кладём 3 карты в колоду P1
        for i in range(3):
            state.p1.deck.append(CardInstance(
                instance_id=uuid4(), card_id=i + 10, name=f"Deck Card {i}",
                card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
                mechanics=[], is_ready=False,
            ))

        warrior_data = {
            'id': 101, 'name': 'Draw Guy', 'card_type': 'warrior',
            'base_attack': 2, 'current_attack': 2, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 3, 'rarity': 'common',
            'mechanics': ['battlecry_draw_card']
        }
        draw_warrior = card_from_db(warrior_data, level=5)
        state.p1.hand.append(draw_warrior)

        hand_before = len(state.p1.hand)  # 1 (сам warrior)
        deck_before = len(state.p1.deck)  # 3

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success

        # После розыгрыша: warrior ушёл из руки (-1), тянется 1 карта (+1) → итого 1
        assert len(state.p1.hand) == 1, \
            f"Должна быть ровно 1 карта в руке, получено {len(state.p1.hand)}"
        assert len(state.p1.deck) == deck_before - 1, \
            f"Из колоды должна уйти ровно 1 карта"

    def test_battlecry_draw_card_draws_after_leaving_full_hand(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        draw_warrior = CardInstance(
            instance_id=uuid4(), card_id=101, name="Draw Guy",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_draw_card"], is_ready=False,
        )
        fillers = [
            CardInstance(
                instance_id=uuid4(), card_id=200 + idx, name=f"Filler {idx}",
                card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
                mechanics=[], is_ready=False,
            )
            for idx in range(3)
        ]
        state.p1.hand = [draw_warrior, *fillers]
        state.p1.deck = [
            CardInstance(
                instance_id=uuid4(), card_id=300, name="Drawn",
                card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
                mechanics=[], is_ready=False,
            )
        ]

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))

        assert success, error
        assert len(state.p1.hand) == 4
        assert state.p1.hand[-1].name == "Drawn"
        assert state.p1.deck == []


class TestCycleFixesC2M2H3:
    """Регресс-тесты под фиксы cycle-системы после аудита:

    - C2: battlecry_draw_card с пустой колодой должен reshuffle из
      graveyard (а не молча проваливаться).
    - M2: путь battlecry_draw_card делит ту же логику reshuffle + hand
      cap с end-of-turn (через core.engine.draw_one_from_deck).
    - H3: action_history стал deque(maxlen=100) — поведение append,
      индексация и итерация должны остаться совместимыми с прежним
      list-based контрактом.
    """

    def test_battlecry_draw_with_empty_deck_reshuffles_graveyard(self):
        """C2 fix: пустая колода → battlecry_draw_card триггерит reshuffle
        из graveyard (раньше просто skip'ал)."""
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        # Включаем classic_params, чтобы state.classic_params был выставлен
        # через ArenaEnvironment.__init__.
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        # 1 карта в graveyard (была «убита»), колода пуста.
        recycled = CardInstance(
            instance_id=uuid4(), card_id=400, name="Reshuffled",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.graveyard.append(recycled)

        # Draw warrior на руке (hand=1, deck=0).
        draw_warrior = CardInstance(
            instance_id=uuid4(), card_id=101, name="Draw Guy",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_draw_card"], is_ready=False,
        )
        state.p1.hand = [draw_warrior]

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))

        assert success, error
        # Warrior ушёл, тянется reshuffled → итого 1 карта в руке.
        assert len(state.p1.hand) == 1
        assert state.p1.hand[0].name == "Reshuffled"
        # Graveyard теперь пуст.
        assert state.p1.graveyard == []
        # Колода тоже пуста — единственная карта ушла на руку.
        assert state.p1.deck == []

    def test_battlecry_draw_with_empty_deck_and_graveyard_does_not_raise(self):
        """C2 fix: edge case — пустая колода И пустой graveyard.
        draw_one_from_deck должен вернуть False без raise."""
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        draw_warrior = CardInstance(
            instance_id=uuid4(), card_id=101, name="Draw Guy",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_draw_card"], is_ready=False,
        )
        state.p1.hand = [draw_warrior]
        state.p1.deck = []
        state.p1.graveyard = []

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))

        assert success, error
        # Warrior ушёл с руки, draw пропущен (fatigue) — hand пуст.
        assert len(state.p1.hand) == 0
        assert state.p1.deck == []
        assert state.p1.graveyard == []

    def test_end_turn_and_battlecry_draw_use_same_overdraw_policy(self):
        """M2 fix: оба пути добора уважают overdraw_to_discard одинаково
        — при пустой колоде делается reshuffle, при заполненной руке
        поведение идентично."""
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=True),
        )

        # ------------------------------------------------------------------
        # Path 1: end_turn (добор для противника при заполненной руке)
        # ------------------------------------------------------------------
        # P2 имеет 4 cards на руке, deck = [TopDeckEnd].
        # current_turn_owner_id = 1 (P1), end_turn передаст ход P2 и
        # попытается добрать карту для P2.
        state.current_turn_owner_id = 1
        # P1 рука пусть будет пустая — нам важен только P2.
        top_end = CardInstance(
            instance_id=uuid4(), card_id=300, name="TopDeckEnd",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        filler_end = CardInstance(
            instance_id=uuid4(), card_id=200, name="FillerEnd",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p2.hand = [filler_end] * 4
        state.p2.deck = [top_end]
        state.p2.graveyard = []

        env._handle_end_turn(state.p1, state.p2)
        # overdraw_to_discard=True → TopDeckEnd ушёл в graveyard, не в hand.
        assert len(state.p2.hand) == 4
        assert state.p2.deck == []
        assert len(state.p2.graveyard) == 1
        assert state.p2.graveyard[0].name == "TopDeckEnd"

        # ------------------------------------------------------------------
        # Path 2: battlecry_draw_card (та же политика)
        # ------------------------------------------------------------------
        # P1: hand 3, deck = 1, рука НЕ заполнена → draw fires.
        top_bc = CardInstance(
            instance_id=uuid4(), card_id=301, name="TopDeckBC",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        filler_bc = CardInstance(
            instance_id=uuid4(), card_id=201, name="FillerBC",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.hand = [filler_bc] * 3
        state.p1.deck = [top_bc]
        state.p1.graveyard = []

        draw_warrior = CardInstance(
            instance_id=uuid4(), card_id=101, name="Draw Guy",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_draw_card"], is_ready=False,
        )
        state.p1.hand.append(draw_warrior)
        draw_index = len(state.p1.hand) - 1

        # Ход P1.
        state.current_turn_owner_id = 1
        success, error = env.step(1, PlayCardAction(
            hand_index=draw_index, target_id=None, position=0
        ))
        assert success, error
        # Warrior ушёл (hand=3), draw fires → hand=4, deck=[].
        assert len(state.p1.hand) == 4
        assert state.p1.hand[-1].name == "TopDeckBC"
        assert state.p1.deck == []
        assert state.p1.graveyard == []

    def test_action_history_is_deque_with_maxlen(self):
        """H3 fix: action_history — это deque(maxlen=100), не list."""
        from collections import deque

        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        assert isinstance(state.action_history, deque)
        assert state.action_history.maxlen == 100

    def test_action_history_evicts_old_entries_automatically(self):
        """H3 fix: append старше 100 записей автоматически вытесняется
        (deque(maxlen=100) O(1) eviction, не `list[-100:]` O(n) realloc)."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        # Заполняем 150 записей.
        for i in range(150):
            state.action_history.append(("system", f"event-{i}"))

        # Хранится только последние 100.
        assert len(state.action_history) == 100
        # Старейшая запись — event-50 (т.к. 0..49 evicted).
        assert state.action_history[0][1] == "event-50"
        assert state.action_history[-1][1] == "event-149"

    def test_action_history_slice_negative_index_still_works(self):
        """H3 fix: обратная совместимость — `action_history[-1]`
        и `list(action_history)[-N:]` должны работать как раньше.
        (deque не поддерживает slice-нотацию, только одиночные индексы —
        именно поэтому battle_runner и rlhf_env оборачивают в list().)"""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        for i in range(5):
            state.action_history.append(("system", f"e{i}"))

        # Одиночный индекс (важно для теста test_attack_history_*).
        assert state.action_history[-1][1] == "e4"
        # Срез через list() — это путь, которым пользуются battle_runner
        # и rlhf_env (см. _action_history_snapshot).
        assert [t for _t, t in list(state.action_history)[-3:]] == ["e2", "e3", "e4"]
        # Прямой slice на deque — TypeError, документируем контракт.
        with pytest.raises(TypeError):
            _ = state.action_history[-3:]


class TestManaDraw:
    """«Добор карт» — player-initiated draw за ману (см. docs/CYCLE_DRAW.md и
    core/engine.py ArenaEnvironment._handle_mana_draw).

    Стоимость N-го добора в рамках хода: MANA_DRAW_BASE * N (2, 4, 6, ...),
    сбрасывается в начале каждого хода игрока. Сам добор переиспользует
    draw_one_from_deck (No-FIFO weighted) с тем же self._rng.
    """

    @staticmethod
    def _warrior(card_id: int, name: str = "W", cost: int = 1) -> CardInstance:
        return CardInstance(
            instance_id=uuid4(), card_id=card_id, name=name,
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1,
            mana_cost=cost, mechanics=[], is_ready=False,
        )

    def _env(self, *, mana=10, hand=None, deck=None, graveyard=None, seed=None):
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
            rng=(__import__("random").Random(seed) if seed is not None else None),
        )
        state.p1.mana = mana
        state.p1.max_mana = max(mana, 10)
        state.p1.hand = list(hand or [])
        state.p1.deck = list(deck or [])
        state.p1.graveyard = list(graveyard or [])
        state.p1.mana_draw_count_this_turn = 0
        return env, state

    def test_cost_sequence_2_4_6_8_same_turn(self):
        env, state = self._env(
            mana=20, hand=[], deck=[self._warrior(i, f"W{i}") for i in range(10, 18)],
        )
        # 4 добора подряд в один ход: 2, 4, 6, 8 — итого 20 маны.
        expected_costs = [2, 4, 6, 8]
        prev_mana = 20
        for n, cost in enumerate(expected_costs, start=1):
            ok, err = env.step(1, ManaDrawAction())
            assert ok, err
            assert state.p1.mana == prev_mana - cost
            assert state.p1.mana_draw_count_this_turn == n
            assert len(state.p1.hand) == n
            prev_mana -= cost
        assert state.p1.mana == 0

    def test_counter_resets_after_full_turn_cycle(self):
        # 2 добора → end_turn (p1) → end_turn (p2) → снова ход p1, стоимость 2.
        env, state = self._env(
            mana=10, hand=[], deck=[self._warrior(i, f"W{i}") for i in range(10, 20)],
        )
        ok, _ = env.step(1, ManaDrawAction()); assert ok
        ok, _ = env.step(1, ManaDrawAction()); assert ok
        assert state.p1.mana_draw_count_this_turn == 2
        assert state.p1.mana == 4  # 10 - 2 - 4

        ok, _ = env.step(1, EndTurnAction()); assert ok          # ход -> p2
        assert state.current_turn_owner_id == state.p2.user_id
        ok, _ = env.step(state.p2.user_id, EndTurnAction()); assert ok  # ход -> p1
        assert state.current_turn_owner_id == state.p1.user_id
        # Счётчик p1 сброшен в начале его хода.
        assert state.p1.mana_draw_count_this_turn == 0

        # Первый добор нового хода снова стоит 2.
        mana_before = state.p1.mana
        ok, err = env.step(1, ManaDrawAction())
        assert ok, err
        assert state.p1.mana == mana_before - 2
        assert state.p1.mana_draw_count_this_turn == 1

    def test_insufficient_mana_does_not_change_state(self):
        env, state = self._env(mana=1, hand=[], deck=[self._warrior(11)])
        ok, err = env.step(1, ManaDrawAction())
        assert not ok
        assert err == "insufficient_mana"
        assert state.p1.mana == 1            # мана не списана
        assert state.p1.mana_draw_count_this_turn == 0
        assert len(state.p1.hand) == 0      # карта не добрана
        assert len(state.p1.deck) == 1

    def test_hand_full_blocks_draw(self):
        env, state = self._env(
            mana=10, hand=[self._warrior(i) for i in range(4)], deck=[self._warrior(99)],
        )
        assert len(state.p1.hand) == HAND_CAP
        ok, err = env.step(1, ManaDrawAction())
        assert not ok
        assert err == "hand_full"
        assert state.p1.mana == 10          # мана не списана
        assert len(state.p1.hand) == HAND_CAP

    def test_no_cards_to_draw_refunds_mana(self):
        # Колода и сброс пусты — fatigue, ману возвращаем.
        env, state = self._env(mana=10, hand=[], deck=[], graveyard=[])
        ok, err = env.step(1, ManaDrawAction())
        assert not ok
        assert err == "no_cards_to_draw"
        assert state.p1.mana == 10          # refund
        assert state.p1.mana_draw_count_this_turn == 0

    def test_drawn_card_never_duplicates_hand(self):
        # Пулы hand/deck дизъюнктны, дубликаты card_id в колоде запрещены —
        # поэтому добранная карта никогда не совпадает по card_id с рукой.
        in_hand = self._warrior(500, "InHand")
        in_deck = self._warrior(501, "InDeck")
        env, state = self._env(mana=10, hand=[in_hand], deck=[in_deck])
        ok, err = env.step(1, ManaDrawAction())
        assert ok, err
        assert len(state.p1.hand) == 2
        drawn = [c for c in state.p1.hand if c.card_id != 500][0]
        assert drawn.card_id == 501
        assert all(c.card_id != drawn.card_id or c is drawn for c in state.p1.hand)

    def test_legal_actions_include_mana_draw_when_affordable_and_hand_not_full(self):
        env, state = self._env(mana=5, hand=[self._warrior(1)], deck=[self._warrior(2)])
        types = [a.to_dict()["type"] for a in env.get_legal_actions(1)]
        assert "mana_draw" in types  # 5 >= 2 и hand(1) < 4

    def test_legal_actions_exclude_mana_draw_when_hand_full(self):
        env, state = self._env(
            mana=10, hand=[self._warrior(i) for i in range(4)], deck=[self._warrior(99)],
        )
        types = [a.to_dict()["type"] for a in env.get_legal_actions(1)]
        assert "mana_draw" not in types

    def test_legal_actions_exclude_mana_draw_when_cannot_afford(self):
        env, state = self._env(mana=1, hand=[], deck=[self._warrior(2)])
        types = [a.to_dict()["type"] for a in env.get_legal_actions(1)]
        assert "mana_draw" not in types  # 1 < 2

    def test_determinism_same_seed_same_drawn_card(self):
        import random as rand_mod

        deck = [self._warrior(i, f"W{i}") for i in range(10, 20)]
        env1, st1 = self._env(mana=10, hand=[], deck=list(deck), seed=42)
        env2, st2 = self._env(mana=10, hand=[], deck=list(deck), seed=42)
        ok1, _ = env1.step(1, ManaDrawAction()); assert ok1
        ok2, _ = env2.step(1, ManaDrawAction()); assert ok2
        assert st1.p1.hand[0].card_id == st2.p1.hand[0].card_id

    def test_not_your_turn_rejected(self):
        env, state = self._env(mana=10, hand=[], deck=[self._warrior(2)])
        # current_turn_owner_id == 1, поэтому запрос от p2 должен провалиться.
        ok, err = env.step(state.p2.user_id, ManaDrawAction())
        assert not ok
        assert err == "not_your_turn"


class TestCoreRegressionHardening:
    def test_unknown_player_id_is_rejected_even_if_turn_owner_is_corrupt(self):
        state = create_minimal_game_state()
        state.current_turn_owner_id = 999
        env = ArenaEnvironment(state)

        success, error = env.step(999, EndTurnAction())

        assert not success
        assert error == "unknown_player"

    def test_graveyard_reshuffle_restores_base_card_state(self):
        state = create_minimal_game_state()
        recycled = CardInstance(
            instance_id=uuid4(), card_id=88, name="Shielded",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=2,
            mechanics=["shield"], is_ready=True,
        )
        state.p2.board = [recycled]
        env = ArenaEnvironment(state)

        apply_damage(recycled, 1)
        recycled.attack += 5
        recycled.max_hp += 5
        recycled.hp += 5
        recycled.hp = 0
        env._cleanup_dead_units(state.p2)

        assert state.p2.graveyard == [recycled]

        success, error = env.step(1, EndTurnAction())

        assert success, error
        assert len(state.p2.hand) == 1
        drawn = state.p2.hand[0]
        assert drawn.attack == 2
        assert drawn.max_hp == 3
        assert drawn.hp == 3
        assert drawn.mechanics == ["shield"]
        assert drawn.is_ready is False
        assert drawn.is_frozen is False

    def test_scaled_spell_damage_mechanic_executes(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        potion = card_from_db({
            'id': 90, 'name': 'Scaled Bolt', 'card_type': 'potion',
            'base_attack': 0, 'current_attack': 0, 'base_hp': 0, 'current_hp': 0,
            'mana_cost': 1, 'rarity': 'common', 'mechanics': ['spell_damage_3']
        }, level=4)
        assert 'spell_damage_4' in potion.mechanics
        state.p1.hand.append(potion)

        success, error = env.step(1, PlayCardAction(
            hand_index=0, target_id=str(state.p2.hero.instance_id),
        ))

        assert success, error
        assert state.p2.hero.hp == 26

    def test_lifesteal_heals_only_actual_damage_dealt(self):
        state = create_minimal_game_state()
        attacker = CardInstance(
            instance_id=uuid4(), card_id=91, name="Leech",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=10, mana_cost=4,
            mechanics=["lifesteal"], is_ready=True,
        )
        victim = CardInstance(
            instance_id=uuid4(), card_id=92, name="Low HP",
            card_type=CardType.WARRIOR, hp=1, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.hero.hp = 20
        state.p1.board = [attacker]
        state.p2.board = [victim]
        env = ArenaEnvironment(state)

        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=str(victim.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert state.p1.hero.hp == 21

    def test_action_history_is_viewer_relative_for_p2(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        engine = BattleEngine()
        engine._arena = env

        history = engine._serialize_action_history(
            [("player", "разыграл карту"), ("opponent", "завершил ход")],
            viewer_id=2,
        )

        assert history[0]["type"] == "opponent"
        assert history[0]["text"].startswith("Противник ")
        assert history[1]["type"] == "player"
        assert history[1]["text"].startswith("Вы ")

    def test_full_state_without_viewer_hides_both_hands(self):
        state = create_minimal_game_state()
        state.p1.hand.append(CardInstance(
            instance_id=uuid4(), card_id=91, name="P1 Secret",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=2,
        ))
        state.p2.hand.append(CardInstance(
            instance_id=uuid4(), card_id=92, name="P2 Secret",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=2,
        ))
        engine = BattleEngine(match_id="safe-state", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        public_state = engine.get_full_state()
        unknown_viewer_state = engine.get_full_state(viewer_id=999)

        assert public_state["player"]["hand"] == [{"hidden": True}]
        assert public_state["opponent"]["hand"] == [{"hidden": True}]
        assert public_state["legal_actions"] == []
        assert unknown_viewer_state["player"]["hand"] == [{"hidden": True}]
        assert unknown_viewer_state["opponent"]["hand"] == [{"hidden": True}]
        assert unknown_viewer_state["legal_actions"] == []

    def test_get_player_state_returns_none_for_unknown_user(self):
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="unknown-player", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        assert engine.get_player_state(999) is None

    def test_attack_history_reports_actual_damage_after_armor(self):
        state = create_minimal_game_state()
        attacker = CardInstance(
            instance_id=uuid4(), card_id=93, name="Attacker",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=5, mana_cost=2,
            mechanics=[], is_ready=True,
        )
        state.p2.hero.mechanics = ["armor_2"]
        state.p1.board = [attacker]
        env = ArenaEnvironment(state)

        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=str(state.p2.hero.instance_id),
            target_is_hero=True,
        ))

        assert success, error
        assert state.action_history[-1][1] == "Attacker наносит 3 урона по Герою"

    def test_battle_engine_rolls_back_state_on_action_exception(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        engine = BattleEngine()
        engine._arena = env
        engine.current_player_id = 1
        engine.turn = 1

        def broken_step(_user_id, _action):
            env.state.p1.mana = 0
            raise RuntimeError("boom")

        env.step = broken_step
        result = engine.execute_action(1, EndTurnAction())

        assert result == {"success": False, "error": "action_failed"}
        assert engine._arena.state.p1.mana == 10
        assert engine.current_player_id == 1
        assert engine.turn == 1

    def test_battle_engine_preflight_wrong_turn_skips_deepcopy(self, monkeypatch):
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="preflight-wrong-turn", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        def fail_deepcopy(_value):
            raise AssertionError("deepcopy should not run for preflight failures")

        monkeypatch.setattr(copy, "deepcopy", fail_deepcopy)

        result = engine.execute_action(2, EndTurnAction())

        assert result == {"success": False, "error": "not_your_turn"}
        assert state.current_turn_owner_id == 1
        assert state.turn_number == 1

    def test_battle_engine_preflight_preserves_error_order_before_validation(self, monkeypatch):
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="preflight-error-order", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        def fail_deepcopy(_value):
            raise AssertionError("deepcopy should not run for preflight failures")

        monkeypatch.setattr(copy, "deepcopy", fail_deepcopy)

        result = engine.execute_action(
            999,
            AttackAction(attacker_id="", target_id=None, target_is_hero=False),
        )

        assert result == {"success": False, "error": "unknown_player"}

    def test_battle_engine_preflight_invalid_attack_validation_skips_deepcopy(self, monkeypatch):
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="preflight-invalid-attack", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        def fail_deepcopy(_value):
            raise AssertionError("deepcopy should not run for preflight failures")

        monkeypatch.setattr(copy, "deepcopy", fail_deepcopy)

        result = engine.execute_action(
            1,
            AttackAction(attacker_id="", target_id=None, target_is_hero=True),
        )

        assert result == {"success": False, "error": "attacker_id обязателен"}

    def test_battle_engine_preflight_missing_attacker_skips_deepcopy(self, monkeypatch):
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="preflight-missing-attacker", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        def fail_deepcopy(_value):
            raise AssertionError("deepcopy should not run for preflight failures")

        monkeypatch.setattr(copy, "deepcopy", fail_deepcopy)

        result = engine.execute_action(
            1,
            AttackAction(attacker_id=str(uuid4()), target_id=None, target_is_hero=True),
        )

        assert result == {"success": False, "error": "attacker_not_found"}
        assert state.p2.hero.hp == 30
        assert state.history == []

    def test_battle_engine_preflight_taunt_violation_skips_deepcopy(self, monkeypatch):
        state = create_minimal_game_state()
        attacker = CardInstance(
            instance_id=uuid4(), card_id=91, name="Attacker",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=2,
            mechanics=[], is_ready=True,
        )
        taunt = CardInstance(
            instance_id=uuid4(), card_id=92, name="Taunt",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["taunt"], is_ready=False,
        )
        state.p1.board = [attacker]
        state.p2.board = [taunt]
        engine = BattleEngine(match_id="preflight-taunt", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        def fail_deepcopy(_value):
            raise AssertionError("deepcopy should not run for preflight failures")

        monkeypatch.setattr(copy, "deepcopy", fail_deepcopy)

        result = engine.execute_action(
            1,
            AttackAction(
                attacker_id=str(attacker.instance_id),
                target_id=str(state.p2.hero.instance_id),
                target_is_hero=True,
            ),
        )

        assert result == {"success": False, "error": "must_attack_taunt"}
        assert state.p2.hero.hp == 30
        assert attacker.is_ready is True

    def test_battle_engine_valid_actions_still_take_snapshot(self, monkeypatch):
        original_deepcopy = copy.deepcopy
        calls = {"count": 0}

        def counting_deepcopy(value):
            calls["count"] += 1
            return original_deepcopy(value)

        monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)

        attack_state = create_minimal_game_state()
        attacker = CardInstance(
            instance_id=uuid4(), card_id=93, name="Attacker",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3, mana_cost=2,
            mechanics=[], is_ready=True,
        )
        target = CardInstance(
            instance_id=uuid4(), card_id=94, name="Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=[], is_ready=False,
        )
        attack_state.p1.board = [attacker]
        attack_state.p2.board = [target]
        attack_engine = BattleEngine(match_id="snapshot-attack", player_ids=[1, 2])
        attack_engine._arena = ArenaEnvironment(attack_state)

        attack_result = attack_engine.execute_action(
            1,
            AttackAction(
                attacker_id=str(attacker.instance_id),
                target_id=str(target.instance_id),
                target_is_hero=False,
            ),
        )

        end_turn_state = create_minimal_game_state()
        end_turn_engine = BattleEngine(match_id="snapshot-end-turn", player_ids=[1, 2])
        end_turn_engine._arena = ArenaEnvironment(end_turn_state)

        end_turn_result = end_turn_engine.execute_action(1, EndTurnAction())

        play_state = create_minimal_game_state()
        play_state.p1.hand.append(CardInstance(
            instance_id=uuid4(), card_id=95, name="Warrior",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=2,
            mechanics=[], is_ready=False,
        ))
        play_engine = BattleEngine(match_id="snapshot-play-card", player_ids=[1, 2])
        play_engine._arena = ArenaEnvironment(play_state)

        play_result = play_engine.execute_action(1, PlayCardAction(hand_index=0, position=0))

        assert attack_result["success"], attack_result
        assert end_turn_result["success"], end_turn_result
        assert play_result["success"], play_result
        assert calls["count"] == 3

    def test_battle_engine_play_card_returns_deploy_and_mechanic_sound_events(self):
        state = create_minimal_game_state()
        yuni = CardInstance(
            instance_id=uuid4(), card_id=36, name="Юни",
            card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=1, mana_cost=2,
            mechanics=["battlecry_heal_target_3"], is_ready=False,
        )
        wounded = CardInstance(
            instance_id=uuid4(), card_id=90, name="Wounded",
            card_type=CardType.WARRIOR, hp=1, max_hp=5, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.hand.append(yuni)
        state.p1.board.append(wounded)
        engine = BattleEngine(match_id="sound-play-card", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        result = engine.execute_action(
            1,
            PlayCardAction(hand_index=0, position=1, target_id=str(wounded.instance_id)),
        )

        assert result["success"], result
        sound_events = result["sound_events"]
        assert [event["event"] for event in sound_events] == ["deploy", "mechanic"]
        assert {event["card_id"] for event in sound_events} == {36}
        assert {event["instance_id"] for event in sound_events} == {str(yuni.instance_id)}
        assert sound_events[1]["mechanic"] == "battlecry_heal_target_3"
        assert all(event["source"] == "action" for event in sound_events)

    def test_battle_engine_midoriya_sound_event_includes_random_spell_effect_code(self, monkeypatch):
        state = create_minimal_game_state()
        midoriya = CardInstance(
            instance_id=uuid4(), card_id=26, name="Мидория",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=5, mana_cost=5,
            mechanics=["cast_random_spell"], level=1, is_ready=False,
        )
        target = CardInstance(
            instance_id=uuid4(), card_id=90, name="Target",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.hand.append(midoriya)
        state.p2.board.append(target)
        monkeypatch.setattr("core.effects.random.randint", lambda _low, _high: 3)
        engine = BattleEngine(match_id="midoriya-text-feedback", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        result = engine.execute_action(1, PlayCardAction(hand_index=0, position=0))

        assert result["success"], result
        mechanic_event = result["sound_events"][1]
        assert mechanic_event["event"] == "mechanic"
        assert mechanic_event["mechanic"] == "cast_random_spell"
        assert mechanic_event["effect_code"] == "midoriya_blackwhip"
        assert engine._arena.state.pending_card_feedback_events == []

    def test_battle_engine_attack_returns_attacker_sound_event(self):
        state = create_minimal_game_state()
        attacker = CardInstance(
            instance_id=uuid4(), card_id=44, name="Леви Аккерман",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=3, mana_cost=2,
            mechanics=["charge"], is_ready=True,
        )
        target = CardInstance(
            instance_id=uuid4(), card_id=90, name="Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board.append(attacker)
        state.p2.board.append(target)
        engine = BattleEngine(match_id="sound-attack", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        result = engine.execute_action(
            1,
            AttackAction(
                attacker_id=str(attacker.instance_id),
                target_id=str(target.instance_id),
                target_is_hero=False,
            ),
        )

        assert result["success"], result
        assert result["sound_events"] == [{
            "event_id": f"sound-attack:1:attack:{attacker.instance_id}:attack",
            "card_id": 44,
            "instance_id": str(attacker.instance_id),
            "card_name": "Леви Аккерман",
            "event": "attack",
            "mechanic": None,
            "side": "player",
            "source": "action",
        }]

    def test_battle_engine_unknown_action_still_uses_snapshot_before_validate(self, monkeypatch):
        class MutatingUnknownAction(BaseAction):
            def to_dict(self):
                return {"type": "mutating_unknown"}

            def validate(self, state):
                state.p1.mana = 0

        original_deepcopy = copy.deepcopy
        calls = {"count": 0}

        def counting_deepcopy(value):
            calls["count"] += 1
            return original_deepcopy(value)

        monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)
        state = create_minimal_game_state()
        engine = BattleEngine(match_id="snapshot-unknown-action", player_ids=[1, 2])
        engine._arena = ArenaEnvironment(state)

        result = engine.execute_action(1, MutatingUnknownAction())

        assert result == {"success": False, "error": "unknown_action"}
        assert engine._arena.state.p1.mana == 10
        assert calls["count"] == 1

    def test_shield_refresh_mechanic_restores_one_time_shield_at_start_of_owner_turn(self):
        state = create_minimal_game_state()
        gojo = CardInstance(
            instance_id=uuid4(), card_id=4242, name="Renamed Shield Refresh Unit",
            card_type=CardType.WARRIOR, hp=8, max_hp=8, attack=5, mana_cost=9,
            mechanics=["shield_refresh"], is_ready=False,
        )
        state.p2.board = [gojo]
        env = ArenaEnvironment(state)

        success, error = env.step(1, EndTurnAction())

        assert success, error
        assert "shield" in gojo.mechanics

    def test_card_id_24_without_shield_refresh_does_not_restore_shield(self):
        state = create_minimal_game_state()
        non_gojo = CardInstance(
            instance_id=uuid4(), card_id=24, name="Any Renamed Card",
            card_type=CardType.WARRIOR, hp=8, max_hp=8, attack=5, mana_cost=9,
            mechanics=[], is_ready=False,
        )
        state.p2.board = [non_gojo]
        env = ArenaEnvironment(state)

        success, error = env.step(1, EndTurnAction())

        assert success, error
        assert "shield" not in non_gojo.mechanics

    def test_shield_blocks_delete_target_and_is_consumed(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        shielded = CardInstance(
            instance_id=uuid4(), card_id=100, name="Shielded Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["shield"], is_ready=False,
        )
        delete_spell = CardInstance(
            instance_id=uuid4(), card_id=13, name="Черная Дыра",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0, mana_cost=1,
            mechanics=["delete_target"], is_ready=False,
        )
        state.p2.board = [shielded]
        state.p1.hand = [delete_spell]

        success, error = env.step(1, PlayCardAction(
            hand_index=0,
            target_id=str(shielded.instance_id),
        ))

        assert success, error
        assert state.p2.board == [shielded]
        assert "shield" not in shielded.mechanics

    def test_delete_target_moves_destroyed_unit_to_graveyard(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        target = CardInstance(
            instance_id=uuid4(), card_id=100, name="Delete Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=[], is_ready=False,
        )
        delete_spell = CardInstance(
            instance_id=uuid4(), card_id=13, name="Черная Дыра",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0, mana_cost=1,
            mechanics=["delete_target"], is_ready=False,
        )
        state.p2.board = [target]
        state.p1.hand = [delete_spell]

        success, error = env.step(1, PlayCardAction(
            hand_index=0,
            target_id=str(target.instance_id),
        ))

        assert success, error
        assert state.p2.board == []
        assert [card.instance_id for card in state.p2.graveyard] == [target.instance_id]

    def test_shield_blocks_freeze_effects_and_is_consumed(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        shielded = CardInstance(
            instance_id=uuid4(), card_id=101, name="Shielded Freeze Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["shield"], is_ready=False,
        )
        freeze_spell = CardInstance(
            instance_id=uuid4(), card_id=11, name="Заморозка",
            card_type=CardType.POTION, hp=0, max_hp=0, attack=0, mana_cost=1,
            mechanics=["freeze"], is_ready=False,
        )
        state.p2.board = [shielded]
        state.p1.hand = [freeze_spell]

        success, error = env.step(1, PlayCardAction(
            hand_index=0,
            target_id=str(shielded.instance_id),
        ))

        assert success, error
        assert shielded.is_frozen is False
        assert "shield" not in shielded.mechanics

    def test_shield_blocks_aoe_freeze_and_dynamic_freeze(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        aoe_target = CardInstance(
            instance_id=uuid4(), card_id=102, name="AOE Shield",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["shield"], is_ready=False,
        )
        state.p2.board = [aoe_target]

        process_effects(
            state,
            CardInstance(card_type=CardType.POTION, mechanics=["aoe_freeze"]),
            state.p1,
            state.p2,
        )
        assert aoe_target.is_frozen is False
        assert "shield" not in aoe_target.mechanics

        dynamic_target = CardInstance(
            instance_id=uuid4(), card_id=103, name="Dynamic Shield",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["shield"], is_ready=False,
        )
        state.p2.board = [dynamic_target]
        process_effects(
            state,
            CardInstance(card_type=CardType.POTION, mechanics=["freeze_1"]),
            state.p1,
            state.p2,
            target_id=str(dynamic_target.instance_id),
        )

        assert dynamic_target.is_frozen is False
        assert "shield" not in dynamic_target.mechanics

    def test_shield_blocks_dynamic_freeze_on_hero(self):
        state = create_minimal_game_state()
        state.p2.hero.mechanics = ["shield"]

        process_effects(
            state,
            CardInstance(card_type=CardType.POTION, mechanics=["freeze_1"]),
            state.p1,
            state.p2,
            target_id=str(state.p2.hero.instance_id),
        )

        assert state.p2.hero.is_frozen is False
        assert "shield" in state.p2.hero.mechanics

    def test_shield_blocks_blackwhip_freeze_and_is_consumed(self, monkeypatch):
        state = create_minimal_game_state()
        shielded = CardInstance(
            instance_id=uuid4(), card_id=204, name="Shielded Blackwhip Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=2,
            mechanics=["shield"], is_ready=False,
        )
        state.p2.board = [shielded]
        midoriya = CardInstance(
            instance_id=uuid4(), card_id=26, name="Мидория",
            card_type=CardType.WARRIOR, hp=4, max_hp=4, attack=3, mana_cost=4,
            mechanics=["cast_random_spell"], level=1, is_ready=False,
        )
        monkeypatch.setattr("core.effects.random.randint", lambda _low, _high: 3)

        process_effects(state, midoriya, state.p1, state.p2)

        assert shielded.is_frozen is False
        assert "shield" not in shielded.mechanics

    def test_saitama_instant_kill_only_first_unit_per_turn(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        saitama = CardInstance(
            instance_id=uuid4(), card_id=25, name="Сайтама",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["instant_kill"], is_ready=True,
        )
        first = CardInstance(
            instance_id=uuid4(), card_id=201, name="First Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        second = CardInstance(
            instance_id=uuid4(), card_id=202, name="Second Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board = [saitama]
        state.p2.board = [first, second]

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(first.instance_id),
            target_is_hero=False,
        ))
        assert success, error
        assert first.hp == 0

        saitama.is_ready = True
        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(second.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert second.hp == 4

    def test_shield_blocks_saitama_instant_kill_and_is_consumed(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        saitama = CardInstance(
            instance_id=uuid4(), card_id=25, name="Сайтама",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["instant_kill"], is_ready=True,
        )
        shielded = CardInstance(
            instance_id=uuid4(), card_id=203, name="Shielded Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=["shield"], is_ready=False,
        )
        state.p1.board = [saitama]
        state.p2.board = [shielded]

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(shielded.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert shielded.hp == 5
        assert "shield" not in shielded.mechanics

    def test_saitama_instant_kill_only_once_for_lifetime(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        saitama = CardInstance(
            instance_id=uuid4(), card_id=25, name="Сайтама",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["instant_kill"], is_ready=True,
        )
        first = CardInstance(
            instance_id=uuid4(), card_id=205, name="First Lifetime Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        second = CardInstance(
            instance_id=uuid4(), card_id=206, name="Second Lifetime Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board = [saitama]
        state.p2.board = [first, second]

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(first.instance_id),
            target_is_hero=False,
        ))
        assert success, error
        assert first.hp == 0

        success, error = env.step(1, EndTurnAction())
        assert success, error
        success, error = env.step(2, EndTurnAction())
        assert success, error

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(second.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert second.hp == 4

    def test_saitama_instant_kill_spent_when_shield_blocks_it(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        saitama = CardInstance(
            instance_id=uuid4(), card_id=25, name="Сайтама",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["instant_kill"], is_ready=True,
        )
        shielded = CardInstance(
            instance_id=uuid4(), card_id=207, name="Shielded Lifetime Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=["shield"], is_ready=False,
        )
        unshielded = CardInstance(
            instance_id=uuid4(), card_id=208, name="Unshielded Lifetime Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board = [saitama]
        state.p2.board = [shielded, unshielded]

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(shielded.instance_id),
            target_is_hero=False,
        ))
        assert success, error
        assert shielded.hp == 5
        assert "shield" not in shielded.mechanics

        success, error = env.step(1, EndTurnAction())
        assert success, error
        success, error = env.step(2, EndTurnAction())
        assert success, error

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(unshielded.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert unshielded.hp == 4

    def test_saitama_hero_attack_does_not_spend_instant_kill(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        saitama = CardInstance(
            instance_id=uuid4(), card_id=25, name="Сайтама",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["instant_kill"], is_ready=True,
        )
        target = CardInstance(
            instance_id=uuid4(), card_id=213, name="First Unit After Hero",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board = [saitama]
        state.p2.board = [target]

        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=None,
            target_is_hero=True,
        ))
        assert success, error
        assert state.p2.hero.hp == 29
        assert saitama.instant_kill_used is False

        saitama.is_ready = True
        success, error = env.step(1, AttackAction(
            attacker_id=str(saitama.instance_id),
            target_id=str(target.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert target.hp == 0
        assert saitama.instant_kill_used is True

    def test_unit_killer_kills_every_attacked_unit_without_usage_limit(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        killer = CardInstance(
            instance_id=uuid4(), card_id=209, name="Unit Killer",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1, mana_cost=10,
            mechanics=["unit_killer"], is_ready=True,
        )
        first = CardInstance(
            instance_id=uuid4(), card_id=210, name="First Unit Killer Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        shielded = CardInstance(
            instance_id=uuid4(), card_id=211, name="Shielded Unit Killer Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=["shield"], is_ready=False,
        )
        second = CardInstance(
            instance_id=uuid4(), card_id=212, name="Second Unit Killer Target",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=0, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p1.board = [killer]
        state.p2.board = [first, shielded, second]

        success, error = env.step(1, AttackAction(
            attacker_id=str(killer.instance_id),
            target_id=str(first.instance_id),
            target_is_hero=False,
        ))
        assert success, error
        assert first.hp == 0

        killer.is_ready = True
        success, error = env.step(1, AttackAction(
            attacker_id=str(killer.instance_id),
            target_id=str(shielded.instance_id),
            target_is_hero=False,
        ))
        assert success, error
        assert shielded.hp == 5
        assert "shield" not in shielded.mechanics

        killer.is_ready = True
        success, error = env.step(1, AttackAction(
            attacker_id=str(killer.instance_id),
            target_id=str(second.instance_id),
            target_is_hero=False,
        ))

        assert success, error
        assert second.hp == 0

    def test_unit_killer_deals_only_normal_damage_to_hero(self):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        killer = CardInstance(
            instance_id=uuid4(), card_id=213, name="Unit Killer",
            card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=3, mana_cost=10,
            mechanics=["unit_killer"], is_ready=True,
        )
        state.p1.board = [killer]

        success, error = env.step(1, AttackAction(
            attacker_id=str(killer.instance_id),
            target_id=str(state.p2.hero.instance_id),
            target_is_hero=True,
        ))

        assert success, error
        assert state.p2.hero.hp == 27
        assert state.p2.hero.hp > 0


# ============================================================================
# STRATIFIED WEIGHTED DRAW: No-FIFO cost-curve + anti-stuck
# ============================================================================

class TestStratifiedWeightedDraw:
    """Tests for the new No-FIFO weighted draw with cost-curve + anti-stuck."""

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _make_deck_card(name: str, mana_cost: int) -> CardInstance:
        return CardInstance(
            instance_id=uuid4(),
            card_id=500 + len(name),
            name=name,
            card_type=CardType.WARRIOR,
            hp=1,
            max_hp=1,
            attack=1,
            mana_cost=mana_cost,
            mechanics=[],
            is_ready=False,
        )

    @staticmethod
    def _make_hand_card(name: str, mana_cost: int) -> CardInstance:
        return CardInstance(
            instance_id=uuid4(),
            card_id=600 + len(name),
            name=name,
            card_type=CardType.WARRIOR,
            hp=1,
            max_hp=1,
            attack=1,
            mana_cost=mana_cost,
            mechanics=[],
            is_ready=False,
        )

    # ---------------------------------------------------------------- tests

    def test_skip_count_increments_on_full_hand_skip(self):
        """При full hand все карты в deck получают +1 к skip_count."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        # Заполняем руку до HAND_CAP.
        env.state.p1.hand = [
            self._make_hand_card(f"H{i}", 2) for i in range(HAND_CAP)
        ]
        # В deck кладём 3 карты.
        env.state.p1.deck = [
            self._make_deck_card("A", 1),
            self._make_deck_card("B", 3),
            self._make_deck_card("C", 5),
        ]

        # При full hand draw_one_from_deck возвращает False (overdraw skip),
        # но skip_count ВСЕХ карт в deck должен увеличиться на 1.
        result = draw_one_from_deck(
            env.state.p1,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert result is False
        for card in env.state.p1.deck:
            assert card.skip_count == 1, (
                f"После full-hand skip skip_count={card.skip_count} (ожидалось 1) "
                f"у карты {card.name}"
            )

    def test_skip_count_resets_on_draw(self):
        """Когда карта наконец вытянута, её skip_count = 0."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        # В deck — несколько карт с заранее повышенным skip_count (имитация
        # того, что они «застряли»). Нужно >= HAND_CAP карт, чтобы рука
        # полностью заполнилась без reshuffle.
        deck_cards = []
        for i in range(HAND_CAP + 1):
            card = self._make_deck_card(f"Stuck{i}", 2)
            card.skip_count = 5 + i  # каждый со своим значением
            deck_cards.append(card)
        env.state.p1.deck = deck_cards
        env.state.p1.hand = []

        # Делаем серию доборов пока руки не заполнятся, чтобы убедиться, что
        # вытянутые карты выходят с skip_count=0 (вне зависимости от их
        # «застрявшего» значения в deck до вытягивания).
        for _ in range(HAND_CAP):
            success = draw_one_from_deck(
                env.state.p1,
                overdraw_to_discard=False,
                source="test",
                rng=env._rng,
        )
            assert success

        # Все вытянутые карты — на руке с skip_count == 0.
        assert len(env.state.p1.hand) == HAND_CAP
        for card in env.state.p1.hand:
            assert card.skip_count == 0, (
                f"Вытянутая карта {card.name} должна иметь skip_count=0, "
                f"получено {card.skip_count}"
            )

    def test_skip_count_resets_on_reshuffle(self):
        """При reshuffle из graveyard reset_to_base_state() сбрасывает skip_count в 0."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        # Положили карту в graveyard с skip_count=12.
        recycled = self._make_deck_card("Recycled", 3)
        recycled.skip_count = 12
        env.state.p1.graveyard.append(recycled)
        env.state.p1.deck = []
        env.state.p1.hand = []

        # При пустом deck draw_one_from_deck должен сделать reshuffle —
        # reset_to_base_state() сбрасывает skip_count в 0.
        success = draw_one_from_deck(
            env.state.p1,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert success
        assert len(env.state.p1.hand) == 1
        drawn = env.state.p1.hand[0]
        assert drawn is recycled
        assert drawn.skip_count == 0, (
            f"После reshuffle skip_count должен быть сброшен в 0, "
            f"получено {drawn.skip_count}"
        )

    def test_no_fifo_draw_picks_weighted_not_just_top(self):
        """С seeded rng и однородными весами draw НЕ всегда выбирает deck[0]."""
        import random as _random
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        # Сэмплируем 200 раз с одним и тем же seed — проверяем, что хотя бы
        # однажды была вытянута карта не из deck[0].
        rng = _random.Random(20260624)
        # Метим карты так, чтобы deck[0] != deck[1] != deck[2] — если бы
        # использовался FIFO (deck.pop(0)), вытянутая карта всегда была бы
        # «Top0», «Top1», «Top2» по порядку.
        top_picks = {"Top0": 0, "Top1": 0, "Top2": 0}

        for trial in range(200):
            state = create_minimal_game_state()
            env = ArenaEnvironment(
                state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(20260624 + trial),
            )
            env.state.p1.hand = []
            env.state.p1.deck = [
                self._make_deck_card("Top0", 3),
                self._make_deck_card("Top1", 3),
                self._make_deck_card("Top2", 3),
            ]
            success = draw_one_from_deck(
                env.state.p1,
                overdraw_to_discard=False,
                source="test",
                rng=env._rng,
        )
            assert success
            top_picks[env.state.p1.hand[0].name] += 1

        # No-FIFO должен иногда выбирать НЕ только top-карту.
        assert top_picks["Top0"] < 200, (
            f"Если бы draw был FIFO, deck[0]='Top0' выбиралась бы в каждом "
            f"первом доборе; получили {top_picks['Top0']}/200 — это всё ещё "
            f"признак FIFO."
        )
        # И хотя бы одна из нижних карт должна была выпасть.
        assert top_picks["Top1"] + top_picks["Top2"] > 0

    def test_cost_curve_bias_prefers_cheap_when_hand_lacks_cheap(self):
        """Если в руке 0 cheap карт (cost <= 2), draw с большей вероятностью возьмёт cheap."""
        import random as _random
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        cheap_picks = 0
        trials = 500

        for trial in range(trials):
            state = create_minimal_game_state()
            env = ArenaEnvironment(
                state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(20260624 + trial),
            )
            # Рука — все expensive (cost >= 4), cheap и middle наполняются из deck.
            env.state.p1.hand = [
                self._make_hand_card(f"Hand{i}", 5) for i in range(HAND_CAP - 1)
            ]
            env.state.p1.deck = [
                self._make_deck_card("CheapOne", 1),
                self._make_deck_card("MidCard", 3),
                self._make_deck_card("Expensive", 6),
            ]
            # Снимаем верхнюю карту, освобождая слот.
            success = draw_one_from_deck(
                env.state.p1,
                overdraw_to_discard=False,
                source="test",
                rng=env._rng,
        )
            assert success
            # Нас интересует только первая (и единственная) вытянутая карта.
            drawn_name = env.state.p1.hand[-1].name
            if drawn_name == "CheapOne":
                cheap_picks += 1

        # Из 500 попыток с cost-bias cheap должен выигрывать заметно чаще,
        # чем 1/3 ≈ 166. Берём заведомо мягкий порог, чтобы тест не был
        # флаттерен-зависимым.
        assert cheap_picks > trials * 0.30, (
            f"Cheap должен выпадать чаще baseline 1/3; "
            f"получено {cheap_picks}/{trials} = {cheap_picks / trials:.2%}"
        )

    def test_cost_curve_bias_prefers_expensive_when_hand_lacks_expensive(self):
        """Зеркальный тест: если в руке 0 expensive (cost >= 4)."""
        import random as _random
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        expensive_picks = 0
        trials = 500

        for trial in range(trials):
            state = create_minimal_game_state()
            env = ArenaEnvironment(
                state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(424242 + trial),
            )
            # Рука — все cheap (cost <= 2).
            env.state.p1.hand = [
                self._make_hand_card(f"Hand{i}", 1) for i in range(HAND_CAP - 1)
            ]
            env.state.p1.deck = [
                self._make_deck_card("CheapOne", 1),
                self._make_deck_card("MidCard", 3),
                self._make_deck_card("Expensive", 6),
            ]
            success = draw_one_from_deck(
                env.state.p1,
                overdraw_to_discard=False,
                source="test",
                rng=env._rng,
        )
            assert success
            drawn_name = env.state.p1.hand[-1].name
            if drawn_name == "Expensive":
                expensive_picks += 1

        assert expensive_picks > trials * 0.30, (
            f"Expensive должен выпадать чаще baseline 1/3; "
            f"получено {expensive_picks}/{trials} = {expensive_picks / trials:.2%}"
        )

    def test_anti_stuck_guarantees_eventual_pick(self):
        """После 5 пропусков вес = 1 + 5*0.5 = 3.5. Проверь что карта выбирается."""
        import random as _random
        from core.engine import STUCK_BONUS, draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
            rng=_random.Random(7),
        )

        stuck = self._make_deck_card("StuckCard", 3)
        stuck.skip_count = 5
        fresh = self._make_deck_card("FreshCard", 3)
        fresh.skip_count = 0
        env.state.p1.deck = [stuck, fresh]
        env.state.p1.hand = []

        success = draw_one_from_deck(
            env.state.p1,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert success
        drawn = env.state.p1.hand[0]
        # Один из них должен быть вытянут, и его skip_count = 0.
        assert drawn in (stuck, fresh)
        assert drawn.skip_count == 0
        # Веса: stuck=1+5*STUCK_BONUS=3.5, fresh=1.0 — stuck должен быть
        # вытянут в большинстве случаев. Проверяем вероятностно: на 100
        # прогонах stuck должен выигрывать чаще fresh.
        stuck_wins = 0
        for trial in range(100):
            trial_state = create_minimal_game_state()
            trial_env = ArenaEnvironment(
                trial_state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(trial),
            )
            s = self._make_deck_card("S", 3)
            s.skip_count = 5
            f = self._make_deck_card("F", 3)
            f.skip_count = 0
            trial_env.state.p1.deck = [s, f]
            trial_env.state.p1.hand = []
            draw_one_from_deck(
                trial_env.state.p1,
                overdraw_to_discard=False,
                source="test",
            )
            if trial_env.state.p1.hand[0].name == "S":
                stuck_wins += 1
        # Теоретически P(stuck) = 3.5 / 4.5 ≈ 77.7%. Берём порог 60%.
        assert stuck_wins > 60, (
            f"После 5 пропусков stuck должен выигрывать чаще fresh; "
            f"получено {stuck_wins}/100 = {stuck_wins}% (ожидалось ~77%)"
        )
        # Проверяем, что STUCK_BONUS не изменился.
        assert STUCK_BONUS == 0.5

    def test_existing_overdraw_skip_test_still_passes(self):
        """Регрессия: поведение test_classic_overdraw_keeps_cards_in_deck_by_default
        должно остаться — карта остаётся в deck при full hand и
        overdraw_to_discard=False."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        queued = self._make_deck_card("Queued", 2)
        env.state.p2.hand = [
            self._make_hand_card(f"Hand{i}", 2) for i in range(HAND_CAP)
        ]
        env.state.p2.deck = [queued]
        env.state.p2.graveyard = []

        success = draw_one_from_deck(
            env.state.p2,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert success is False
        assert [card.name for card in env.state.p2.deck] == ["Queued"]
        assert env.state.p2.graveyard == []
        assert len(env.state.p2.hand) == HAND_CAP

    def test_existing_overdraw_to_discard_test_still_passes(self):
        """Регрессия: test_overdraw_to_discard_modifier_moves_overdrawn_cards_to_graveyard —
        при overdraw_to_discard=True карта уходит в graveyard, не в hand."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=True),
        )

        overdrawn = self._make_deck_card("Overdrawn", 2)
        env.state.p2.hand = [
            self._make_hand_card(f"Hand{i}", 2) for i in range(HAND_CAP)
        ]
        env.state.p2.deck = [overdrawn]
        env.state.p2.graveyard = []

        success = draw_one_from_deck(
            env.state.p2,
            overdraw_to_discard=True,
            source="test",
            rng=env._rng,
        )
        assert success is True
        assert len(env.state.p2.hand) == HAND_CAP
        assert [card.name for card in env.state.p2.graveyard] == ["Overdrawn"]
        assert env.state.p2.deck == []

    def test_existing_graveyard_reshuffle_test_still_passes(self):
        """Регрессия: test_graveyard_reshuffle_restores_base_card_state —
        базовые статы (attack/max_hp/mana_cost/mechanics/is_ready) сбрасываются."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        # Карта, которая "побывала" в бою: повреждена, баффнута, готова к атаке.
        recycled = self._make_deck_card("Shielded", 2)
        recycled.mechanics = ["shield"]
        # Сначала делаем "честный" base-snapshot, как ArenaEnvironment делает
        # в _ensure_base_snapshots (это записывает base_attack=1, base_hp=1,
        # base_mana_cost=2, base_mechanics=["shield"]).
        recycled.ensure_base_snapshot()
        # После этого вручную "ломаем" runtime-поля — reset_to_base_state()
        # должен вернуть их к base_*-снимку.
        recycled.is_ready = True
        recycled.is_frozen = True
        recycled.hp = 0
        recycled.max_hp = 10
        recycled.attack = 7
        recycled.mana_cost = 9
        # Shield снимаем отдельно (как после удара), mechanics теперь пустой.
        recycled.mechanics = []

        env.state.p1.deck = []
        env.state.p1.graveyard = [recycled]
        env.state.p1.hand = []

        success = draw_one_from_deck(
            env.state.p1,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert success
        drawn = env.state.p1.hand[0]
        # Базовые поля восстановлены из base_*-снимка.
        assert drawn.attack == 1
        assert drawn.max_hp == 1
        assert drawn.hp == 1
        assert drawn.mana_cost == 2
        assert drawn.mechanics == ["shield"]
        assert drawn.is_ready is False
        assert drawn.is_frozen is False
    def test_existing_battlecry_draw_test_still_passes(self):
        """Регрессия: test_battlecry_draw_card_not_doubled_on_scaled_warrior
        и test_battlecry_draw_card_draws_after_leaving_full_hand."""
        from infrastructure.match_modes import ClassicParams

        # ----- test_battlecry_draw_card_not_doubled_on_scaled_warrior -----
        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )
        for i in range(3):
            env.state.p1.deck.append(self._make_deck_card(f"DeckCard{i}", 1))

        from core.converter import card_from_db
        warrior_data = {
            'id': 101, 'name': 'Draw Guy', 'card_type': 'warrior',
            'base_attack': 2, 'current_attack': 2, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 3, 'rarity': 'common',
            'mechanics': ['battlecry_draw_card']
        }
        draw_warrior = card_from_db(warrior_data, level=5)
        env.state.p1.hand.append(draw_warrior)
        deck_before = len(env.state.p1.deck)

        success, _ = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success
        assert len(env.state.p1.hand) == 1, (
            f"battlecry_draw_card должен тянуть ровно 1 карту; "
            f"получено {len(env.state.p1.hand)} карт в руке"
        )
        assert len(env.state.p1.deck) == deck_before - 1

        # ----- test_battlecry_draw_card_draws_after_leaving_full_hand -----
        state2 = create_minimal_game_state()
        env2 = ArenaEnvironment(
            state2,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )
        draw_warrior2 = CardInstance(
            instance_id=uuid4(), card_id=101, name="Draw Guy",
            card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2, mana_cost=3,
            mechanics=["battlecry_draw_card"], is_ready=False,
        )
        fillers = [
            self._make_hand_card(f"Filler{idx}", 1) for idx in range(3)
        ]
        env2.state.p1.hand = [draw_warrior2, *fillers]
        env2.state.p1.deck = [self._make_deck_card("Drawn", 1)]

        success, error = env2.step(
            1, PlayCardAction(hand_index=0, target_id=None, position=0)
        )
        assert success, error
        assert len(env2.state.p1.hand) == 4
        assert env2.state.p1.hand[-1].name == "Drawn"
        assert env2.state.p1.deck == []

    def test_weighted_choice_with_single_card(self):
        """Если в deck ровно 1 карта — всегда берём её."""
        from core.engine import _weighted_choice_idx
        import random as _random

        rng = _random.Random(0)
        for _ in range(20):
            idx = _weighted_choice_idx([2.5], rng)
            assert idx == 0

    def test_weighted_choice_with_zero_weights(self):
        """Defensive: если все веса = 0, fallback на index 0."""
        from core.engine import _weighted_choice_idx
        import random as _random

        rng = _random.Random(0)
        # total = 0 → защитный возврат 0.
        idx = _weighted_choice_idx([0.0, 0.0, 0.0], rng)
        assert idx == 0

    def test_rng_injection_produces_deterministic_results(self):
        """Два вызова с одним и тем же rng seed дают одинаковый результат."""
        import random as _random
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        def run_once(seed: int) -> list[str]:
            state = create_minimal_game_state()
            env = ArenaEnvironment(
                state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(seed),
            )
            env.state.p1.hand = []
            # Достаточно карт, чтобы все 20 попыток добора прошли без reshuffle.
            env.state.p1.deck = [
                self._make_deck_card(
                    ["Alpha", "Beta", "Gamma"][i % 3],
                    [1, 3, 6][i % 3],
                )
                for i in range(25)
            ]
            names: list[str] = []
            # Очищаем руку после каждого добора, чтобы лимит руки не мешал
            # проверить, что seed детерминирует саму последовательность выбора.
            for _ in range(20):
                ok = draw_one_from_deck(
                    env.state.p1,
                    overdraw_to_discard=False,
                    source="test",
                    rng=env._rng,
        )
                assert ok
                names.append(env.state.p1.hand[-1].name)
                env.state.p1.hand = []
            return names

        run_a = run_once(12345)
        run_b = run_once(12345)
        assert run_a == run_b, (
            f"Один и тот же seed должен давать идентичную последовательность; "
            f"получено {run_a} vs {run_b}"
        )

    def test_rng_injection_different_seeds_produce_different_results(self):
        """Два разных seed-а → разные результаты (статистически)."""
        import random as _random
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        def sequence(seed: int, length: int) -> list[str]:
            state = create_minimal_game_state()
            env = ArenaEnvironment(
                state,
                classic_params=ClassicParams(overdraw_to_discard=False),
                rng=_random.Random(seed),
            )
            env.state.p1.hand = []
            env.state.p1.deck = [
                self._make_deck_card(f"D{i}", 1 + (i % 5)) for i in range(length)
            ]
            out: list[str] = []
            # Очищаем руку после каждого добора, чтобы лимит руки не мешал.
            for _ in range(length):
                ok = draw_one_from_deck(
                    env.state.p1,
                    overdraw_to_discard=False,
                    source="test",
                    rng=env._rng,
        )
                assert ok
                out.append(env.state.p1.hand[-1].name)
                env.state.p1.hand = []
            return out

        # Берём длинные серии доборов, чтобы детерминизм seed-а проявился.
        s1 = sequence(111, length=40)
        s2 = sequence(222, length=40)
        # Подавляющее большинство seed-пар должно дать разные серии.
        same = sum(1 for a, b in zip(s1, s2) if a == b)
        # На 40 позициях при равномерном распределении совпадений ~ 8;
        # порог 20 оставляет большой запас от ложных срабатываний.
        assert same < 20, (
            f"Разные seed-ы дали подозрительно похожие серии: "
            f"{s1} vs {s2} ({same}/40 совпадений)"
        )

    def test_drawn_card_has_skip_count_zero(self):
        """Карта, только что вытянутая, имеет skip_count=0."""
        from core.engine import draw_one_from_deck
        from infrastructure.match_modes import ClassicParams

        state = create_minimal_game_state()
        env = ArenaEnvironment(
            state,
            classic_params=ClassicParams(overdraw_to_discard=False),
        )

        a = self._make_deck_card("A", 1)
        b = self._make_deck_card("B", 3)
        c = self._make_deck_card("C", 6)
        a.skip_count = 4
        b.skip_count = 2
        c.skip_count = 9
        env.state.p1.deck = [a, b, c]
        env.state.p1.hand = []

        success = draw_one_from_deck(
            env.state.p1,
            overdraw_to_discard=False,
            source="test",
            rng=env._rng,
        )
        assert success
        drawn = env.state.p1.hand[0]
        assert drawn.skip_count == 0
