"""
Комплексные тесты: баг-фиксы, краевые случаи, новые механики.
"""
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment, scale_card_by_level
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.effects import apply_damage, process_effects, requires_target
from core.converter import _normalize_mechanic, card_from_db


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
