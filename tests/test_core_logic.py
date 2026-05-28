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
    def test_touka_plays_without_target_and_hits_random_enemy(self, monkeypatch):
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)

        enemy = CardInstance(
            instance_id=uuid4(), card_id=99, name="Enemy",
            card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1, mana_cost=1,
            mechanics=[], is_ready=False,
        )
        state.p2.board.append(enemy)

        touka = CardInstance(
            instance_id=uuid4(), card_id=15, name="Тока Киришима",
            card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=2, mana_cost=2,
            mechanics=["battlecry_damage_1"], is_ready=False,
        )
        state.p1.hand.append(touka)

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

        allies = []
        for idx in range(7):
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
        assert len(state.p1.board) == 7
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
            'mana_cost': 5, 'rarity': 'legendary', 'mechanics': ['delete_target']
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
