"""
Регрессионные тесты по ТЗ: Сукуна cleave, Черная Дыра, Крипер deathrattle,
уровни ботов, ONNX-профили, унифицированные статы.
"""
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.effects import apply_damage
from core.converter import card_from_db
from core.card_scaling import scale_base_stats, get_rarity_growth


def create_minimal_game_state(p1_id=1, p2_id=2) -> GameState:
    hero_p1 = CardInstance(
        instance_id=uuid4(), card_id=1, name="Hero P1",
        card_type=CardType.HERO, hp=30, max_hp=30, attack=0, mana_cost=0,
    )
    hero_p2 = CardInstance(
        instance_id=uuid4(), card_id=2, name="Hero P2",
        card_type=CardType.HERO, hp=30, max_hp=30, attack=0, mana_cost=0,
    )
    p1 = PlayerState(user_id=p1_id, is_bot=False, hero=hero_p1, mana=10, max_mana=10)
    p2 = PlayerState(user_id=p2_id, is_bot=False, hero=hero_p2, mana=10, max_mana=10)
    return GameState(p1=p1, p2=p2, current_turn_owner_id=p1_id, turn_number=1, status=GameStatus.ONGOING)


# ============================================================================
# 1. SUKUNA CLEAVE ON ATTACK
# ============================================================================

def test_sukuna_cleave_on_attack_middle():
    """Сукуна атакует средний юнит: левый и правый сосед получают cleave damage."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    sukuna = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1_3"], is_ready=True,
    )
    state.p1.board.append(sukuna)

    left = CardInstance(
        instance_id=uuid4(), card_id=100, name="Left",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    middle = CardInstance(
        instance_id=uuid4(), card_id=101, name="Middle",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    right = CardInstance(
        instance_id=uuid4(), card_id=102, name="Right",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.extend([left, middle, right])

    success, error = env.step(1, AttackAction(
        attacker_id=str(sukuna.instance_id),
        target_id=str(middle.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака Сукуны должна пройти: {error}"

    # Основная цель получает 7 урона
    assert middle.hp == 0, f"Middle должен получить 7 урона (убит), HP={middle.hp}"
    # Соседи получают cleave 1 урон
    assert left.hp == 4, f"Left должен получить 1 cleave урона, HP={left.hp}"
    assert right.hp == 4, f"Right должен получить 1 cleave урона, HP={right.hp}"


def test_sukuna_cleave_on_attack_edge():
    """Сукуна атакует крайний юнит: только существующий сосед получает cleave."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    sukuna = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1_3"], is_ready=True,
    )
    state.p1.board.append(sukuna)

    first = CardInstance(
        instance_id=uuid4(), card_id=100, name="First",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    second = CardInstance(
        instance_id=uuid4(), card_id=101, name="Second",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.extend([first, second])

    success, error = env.step(1, AttackAction(
        attacker_id=str(sukuna.instance_id),
        target_id=str(first.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака должна пройти: {error}"

    assert first.hp == 0, "First должен получить 7 урона"
    assert second.hp == 4, "Second должен получить 1 cleave урона"


def test_sukuna_cleave_legacy_db_format_on_attack():
    """Старый DB-формат cleave_X тоже должен бить соседей при атаке."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    sukuna = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1"], is_ready=True,
    )
    state.p1.board.append(sukuna)

    left = CardInstance(
        instance_id=uuid4(), card_id=100, name="Left",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    middle = CardInstance(
        instance_id=uuid4(), card_id=101, name="Middle",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    right = CardInstance(
        instance_id=uuid4(), card_id=102, name="Right",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.extend([left, middle, right])

    success, error = env.step(1, AttackAction(
        attacker_id=str(sukuna.instance_id),
        target_id=str(middle.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака Сукуны должна пройти: {error}"

    assert left.hp == 4, f"Left должен получить 1 cleave урона, HP={left.hp}"
    assert right.hp == 4, f"Right должен получить 1 cleave урона, HP={right.hp}"


def test_sukuna_cleave_does_not_trigger_on_hero_attack():
    """При атаке по герою cleave не срабатывает."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    sukuna = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1_3"], is_ready=True,
    )
    state.p1.board.append(sukuna)

    enemy_unit = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy_unit)

    hero_hp_before = state.p2.hero.hp
    enemy_hp_before = enemy_unit.hp

    success, error = env.step(1, AttackAction(
        attacker_id=str(sukuna.instance_id),
        target_id=None,
        target_is_hero=True,
    ))
    assert success, f"Атака героя должна пройти: {error}"

    assert state.p2.hero.hp == hero_hp_before - 7, "Герой должен получить 7 урона"
    assert enemy_unit.hp == enemy_hp_before, "Соседние юниты не должны получить cleave при атаке героя"


def test_sukuna_cleave_does_not_hit_allies():
    """Cleave не бьёт союзников атакующего."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    sukuna = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1_3"], is_ready=True,
    )
    state.p1.board.append(sukuna)

    ally = CardInstance(
        instance_id=uuid4(), card_id=99, name="Ally",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p1.board.append(ally)

    enemy = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy)

    success, error = env.step(1, AttackAction(
        attacker_id=str(sukuna.instance_id),
        target_id=str(enemy.instance_id),
        target_is_hero=False,
    ))
    assert success

    assert ally.hp == 3, "Союзник не должен получить cleave-урон"


# ============================================================================
# 2. BLACK HOLE TARGETS
# ============================================================================

def test_black_hole_targets_only_units():
    """Черная Дыра: legal_actions не включает героя."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    enemy_unit = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=2,
        mana_cost=3, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy_unit)

    bh = CardInstance(
        instance_id=uuid4(), card_id=13, name="Черная Дыра",
        card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
        mana_cost=5, mechanics=["delete_target"], is_ready=False,
    )
    state.p1.hand.append(bh)

    actions = env.get_legal_actions(1)
    hero_id_str = str(state.p2.hero.instance_id)
    unit_id_str = str(enemy_unit.instance_id)

    for action in actions:
        if isinstance(action, PlayCardAction) and action.target_id is not None:
            assert action.target_id != hero_id_str, (
                f"Герой {hero_id_str} не должен быть в legal targets для delete_target"
            )
            assert action.target_id == unit_id_str, "Единственная цель — вражеский юнит"


def test_black_hole_rejects_hero_target_without_spending_card():
    """Черная Дыра: play_card по hero id возвращает ошибку и не списывает ману/карту."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    bh = CardInstance(
        instance_id=uuid4(), card_id=13, name="Черная Дыра",
        card_type=CardType.POTION, hp=0, max_hp=0, attack=0,
        mana_cost=5, mechanics=["delete_target"], is_ready=False,
    )
    state.p1.hand.append(bh)
    initial_mana = state.p1.mana
    initial_hand_size = len(state.p1.hand)

    success, error = env.step(1, PlayCardAction(
        hand_index=0,
        target_id=str(state.p2.hero.instance_id),
        position=None,
    ))

    assert not success, "delete_target по герою должно вернуть ошибку"
    assert "delete_target_cannot_target_hero" in error, f"Ожидалась ошибка про героя: {error}"
    assert state.p1.mana == initial_mana, "Мана не должна быть списана"
    assert len(state.p1.hand) == initial_hand_size, "Карта не должна быть потрачена"


# ============================================================================
# 3. CREEPER DEATHRATTLE
# ============================================================================

def test_creeper_deathrattle_hits_enemy_side_only():
    """Крипер: deathrattle наносит урон врагам владельца, не своим."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    # P1 - владелец крипера
    creeper = CardInstance(
        instance_id=uuid4(), card_id=34, name="Крипер",
        card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=2,
        mana_cost=3, mechanics=["deathrattle_aoe_damage_3"], is_ready=True,
    )
    state.p1.board.append(creeper)

    p1_ally = CardInstance(
        instance_id=uuid4(), card_id=99, name="Ally",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=1, mechanics=[], is_ready=False,
    )
    state.p1.board.append(p1_ally)

    # P2 - враги
    p2_enemy = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=10, max_hp=10, attack=3,
        mana_cost=1, mechanics=[], is_ready=True,
    )
    state.p2.board.append(p2_enemy)

    # P2 атакует и убивает крипера
    success, error = env.step(1, EndTurnAction())
    assert success, f"End turn: {error}"

    p2_hero_hp_before = state.p2.hero.hp
    success, error = env.step(2, AttackAction(
        attacker_id=str(p2_enemy.instance_id),
        target_id=str(creeper.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака крипера: {error}"

    assert creeper.hp == 0, "Крипер должен умереть"
    assert creeper not in state.p1.board, "Крипер удалён с доски"
    assert p1_ally.hp == 5, "Союзник владельца не должен получить deathrattle урон"
    # Враг получил ответный удар 2 (creeper counter) + deathrattle 3 = 5 total
    assert p2_enemy.hp == 5, f"Враг должен получить 2+3=5 урона, HP={p2_enemy.hp}"
    assert state.p2.hero.hp == p2_hero_hp_before - 3, "Вражеский герой должен получить deathrattle урон"


def test_creeper_deathrattle_opposite_direction():
    """Крипер P2 умирает: урон получает P1 (opponent)."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)

    creeper = CardInstance(
        instance_id=uuid4(), card_id=34, name="Крипер",
        card_type=CardType.WARRIOR, hp=1, max_hp=1, attack=2,
        mana_cost=3, mechanics=["deathrattle_aoe_damage_3"], is_ready=False,
    )
    state.p2.board.append(creeper)

    p1_attacker = CardInstance(
        instance_id=uuid4(), card_id=99, name="Attacker",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=3,
        mana_cost=2, mechanics=[], is_ready=True,
    )
    state.p1.board.append(p1_attacker)

    p2_ally = CardInstance(
        instance_id=uuid4(), card_id=98, name="Ally P2",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=1, mechanics=[], is_ready=False,
    )
    state.p2.board.append(p2_ally)

    p1_hero_hp_before = state.p1.hero.hp

    success, error = env.step(1, AttackAction(
        attacker_id=str(p1_attacker.instance_id),
        target_id=str(creeper.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака крипера: {error}"

    assert creeper not in state.p2.board
    assert p2_ally.hp == 5, "Союзник P2 не должен получить deathrattle"
    assert state.p1.hero.hp == p1_hero_hp_before - 3, "Герой P1 должен получить deathrattle урон"


# ============================================================================
# 4. BOT DIFFICULTY & CARD LEVELS
# ============================================================================

def test_bot_card_level_by_difficulty_table():
    from ai.bot_factory import BotGenerator

    # trophy road strength tiers
    assert BotGenerator._calc_difficulty(0) == "tier_lite_0000"
    assert BotGenerator._calc_difficulty(100) == "tier_easy_0100"
    assert BotGenerator._calc_difficulty(300) == "tier_easy_plus_0300"
    assert BotGenerator._calc_difficulty(800) == "tier_medium_minus_0600"
    assert BotGenerator._calc_difficulty(2000) == "tier_medium_plus_2000"
    assert BotGenerator._calc_difficulty(9000) == "tier_max_9000"

    assert BotGenerator._build_bot_card_levels("lite", 5, 8) == [1] * 8
    assert BotGenerator._build_bot_card_levels("easy", 5, 8) == [2] * 8

    # medium/hard no longer get early level advantage
    for _ in range(20):
        for lvl in BotGenerator._build_bot_card_levels("medium", 5, 8):
            assert lvl == 5

    for _ in range(20):
        for lvl in BotGenerator._build_bot_card_levels("hard", 5, 8):
            assert lvl == 5

    # max gets a controlled partial +1 boost, not full-deck overlevel.
    for _ in range(20):
        for lvl in BotGenerator._build_bot_card_levels("max", 5, 8):
            assert 5 <= lvl <= 6


# ============================================================================
# 5. ONNX PROFILES
# ============================================================================

def test_ai_profiles_temperature_table():
    from infrastructure.config import BOT_DIFFICULTY_PROFILES

    profiles = BOT_DIFFICULTY_PROFILES
    assert "tier_lite_0000" in profiles
    assert "tier_easy_0100" in profiles
    assert "tier_medium_minus_0600" in profiles
    assert "tier_medium_plus_2000" in profiles
    assert "tier_max_9000" in profiles
    assert "lite" in profiles
    assert "easy" in profiles
    assert "medium" in profiles
    assert "hard" in profiles
    assert "max" in profiles

    assert profiles["lite"]["temperature_range"] == (5.0, 5.0)
    assert profiles["tier_easy_0100"]["temperature_range"] == (3.2, 3.2)
    assert profiles["easy"]["temperature_range"] == (3.2, 3.2)
    assert profiles["tier_easy_plus_0300"]["temperature_range"] == (2.8, 2.8)
    assert profiles["medium"]["temperature_range"] == (1.8, 1.8)
    assert profiles["hard"]["temperature_range"] == (1.6, 1.6)
    assert profiles["tier_max_9000"].get("selection") == "softmax"
    assert profiles["tier_max_minus_7500"]["temperature_range"] == (2.0, 2.0)
    assert profiles["tier_max_9000"]["temperature_range"] == (0.45, 0.45)

    # max: very cold softmax is stronger than deterministic argmax in tournament sweep.
    assert profiles["max"].get("selection") == "softmax"
    assert profiles["max"]["temperature_range"] == (0.45, 0.45)

    # Models
    assert "extra-lr-v4-micro" in profiles["lite"]["model_path"]
    assert "extra-lr-v4-micro" in profiles["easy"]["model_path"]
    assert "extra-lr-v4-lite" in profiles["tier_easy_plus_0300"]["model_path"]
    assert "extra-lr-v4-opti" in profiles["medium"]["model_path"]
    assert "extra-lr-v4-max" in profiles["hard"]["model_path"]
    assert "extra-lr-v4-max" in profiles["max"]["model_path"]
    for difficulty in ("lite", "easy", "medium", "hard", "max"):
        assert profiles[difficulty]["format"] == "train_v2_classic_v1"
        assert profiles[difficulty]["obs_dim"] == 1456
        assert profiles[difficulty]["action_feature_dim"] == 171
        assert profiles[difficulty]["max_candidate_actions"] == 601
        assert profiles[difficulty]["placement_mode"] == "append_only"
        assert profiles[difficulty]["verify_mask"] is False


# ============================================================================
# 6. UNIFIED STAT FORMULA
# ============================================================================

def test_arena_card_stats_match_collection_formula():
    """Одна и та же карта level 1/5/10 имеет одинаковые статы через card_from_db и DB формулу."""
    card_data = {
        'id': 50, 'name': 'Test Card', 'card_type': 'warrior',
        'base_attack': 5, 'current_attack': 5, 'base_hp': 5, 'current_hp': 5,
        'mana_cost': 3, 'rarity': 'common', 'mechanics': []
    }

    from infrastructure.database import calculate_card_stats

    for level in [1, 5, 10]:
        # Arena (engine) path
        arena_card = card_from_db(card_data, level=level)
        # DB path
        db_stats = calculate_card_stats(card_data, level)

        assert arena_card.attack == db_stats["attack"], (
            f"Level {level}: Arena attack={arena_card.attack}, DB attack={db_stats['attack']}"
        )
        assert arena_card.hp == db_stats["hp"], (
            f"Level {level}: Arena hp={arena_card.hp}, DB hp={db_stats['hp']}"
        )
        assert arena_card.level == level


def test_hero_stats_match_arena_and_collection():
    """Hero level 5/10 совпадает между коллекцией и ареной."""
    hero_data = {
        'id': 7, 'name': 'Tinkov', 'card_type': 'hero',
        'base_hp': 30, 'current_hp': 30, 'base_attack': 0, 'current_attack': 0,
        'mana_cost': 0, 'rarity': 'legendary', 'mechanics': []
    }

    from infrastructure.database import calculate_card_stats

    for level in [1, 5, 10]:
        arena_hero = card_from_db(hero_data, level=level)
        db_stats = calculate_card_stats(hero_data, level)

        expected_hp = 30 + (level - 1) * 2
        assert arena_hero.hp == expected_hp, (
            f"Level {level}: Arena hero hp={arena_hero.hp}, expected={expected_hp}"
        )
        assert db_stats["hp"] == expected_hp, (
            f"Level {level}: DB hero hp={db_stats['hp']}, expected={expected_hp}"
        )
        assert arena_hero.attack == 0


def test_simplified_card_stats_match_arena_and_collection():
    """Simplified-карта level 2 в коллекции и арене имеет один и тот же mana/mechanics."""
    card_data = {
        'id': 13, 'name': 'Черная Дыра', 'card_type': 'potion',
        'base_attack': 0, 'current_attack': 0, 'base_hp': 0, 'current_hp': 0,
        'mana_cost': 5, 'rarity': 'legendary', 'mechanics': ['delete_target'],
        'simplified_levelup': True,
    }

    from infrastructure.database import calculate_card_stats

    arena_card = card_from_db(card_data, level=2)
    db_stats = calculate_card_stats(card_data, level=2)

    assert arena_card.level == 2
    assert arena_card.mana_cost == 4
    assert db_stats["mana"] == 4
    assert arena_card.mechanics == db_stats["mechanics"] == ['delete_target']
    assert db_stats["max_level"] == 2
    assert db_stats["is_max_level"] is True


def test_simplified_upgrade_cost_uses_regular_level_9_cost():
    """Переход simplified 1->2 стоит как обычный 9->10 для той же редкости."""
    from infrastructure.config import DatabaseSettings
    from infrastructure.database import Database

    db = Database(DatabaseSettings(host="localhost", port=5432, user="u", password="p", database="d"))

    assert db.get_card_max_level({"simplified_levelup": True}) == 2
    assert db.get_upgrade_cost("legendary", level=1, simplified_levelup=True) == {
        "particles": db.calculate_upgrade_particles("legendary", 9),
        "coins": db.calculate_upgrade_coins("legendary", 9),
    }


# ============================================================================
# 7. CARD LEVEL MODE PRECEDENCE
# ============================================================================

def _make_mode_config(card_level_mode: str):
    from infrastructure.match_modes import ClassicParams, ModeConfig, resolve_mode_config

    cfg = resolve_mode_config("classic")
    return ModeConfig(
        mode_id=cfg.mode_id,
        ruleset=cfg.ruleset,
        label=cfg.label,
        available=cfg.available,
        classic=ClassicParams(
            turn_duration_seconds=25, mana_per_turn=1,
            hero_health_multiplier=1.0, card_level_mode=card_level_mode,
            bot_turn_delay_range=(4.0, 6.0), bot_hard_turn_delay_range=(1.5, 2.5),
            bot_action_gap_range=(0.4, 0.8), bot_emergency_threshold_seconds=5.0,
        ),
    )


def test_card_level_mode_disabled_overrides_bot():
    """disabled режим: все карты level 1, включая бота."""
    from battle_engine import BattleEngine

    engine = BattleEngine.__new__(BattleEngine)
    engine.mode_config = _make_mode_config("disabled")

    levels = {10: 5, 20: 8}
    result = engine._apply_card_level_mode(levels, [10, 20])
    assert result[10] == 1
    assert result[20] == 1


def test_card_level_mode_max_overrides_bot():
    """max режим: все карты level 10, включая бота."""
    from battle_engine import BattleEngine

    engine = BattleEngine.__new__(BattleEngine)
    engine.mode_config = _make_mode_config("max")

    levels = {10: 3, 20: 5}
    result = engine._apply_card_level_mode(levels, [10, 20])
    assert result[10] == 10
    assert result[20] == 10


def test_card_level_mode_normal_allows_bot_override():
    """normal режим пропускает уровни как есть (bot override поверх)."""
    from battle_engine import BattleEngine

    engine = BattleEngine.__new__(BattleEngine)
    engine.mode_config = _make_mode_config("normal")

    levels = {10: 3, 20: 5}
    result = engine._apply_card_level_mode(levels, [10, 20])
    assert result[10] == 3
    assert result[20] == 5


def test_serialize_card_includes_level():
    from battle_engine import BattleEngine
    from core.state import CardInstance, CardType
    from uuid import uuid4

    engine = BattleEngine.__new__(BattleEngine)
    card = CardInstance(
        instance_id=uuid4(), card_id=23, name="Сукуна",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=7,
        mana_cost=7, mechanics=["cleave_1_3"], is_ready=True,
        level=4, rarity="mythic",
    )
    serialized = engine._serialize_card(card)
    assert serialized["level"] == 4
    assert "level" in serialized


def test_serialize_card_includes_mechanics_description():
    from battle_engine import BattleEngine
    from core.state import CardInstance, CardType
    from uuid import uuid4

    engine = BattleEngine.__new__(BattleEngine)
    card = CardInstance(
        instance_id=uuid4(), card_id=12, name="Юни",
        card_type=CardType.WARRIOR, hp=4, max_hp=4, attack=2,
        mana_cost=3, mechanics=["heal_target_2"], is_ready=True,
        mechanics_desc="Лечит дружественную цель.",
    )

    serialized = engine._serialize_card(card)

    assert serialized["mechanics_desc"] == "Лечит дружественную цель."


def test_turn_time_history_records_completed_turns():
    from battle_engine import BattleEngine

    engine = BattleEngine.__new__(BattleEngine)
    engine.turn_duration = 25
    engine.turn_time_history = []

    engine._record_completed_turn(3, 101, 1000.0, now=1012.4)

    assert engine.turn_time_history == [
        {"turn": 3, "player_id": 101, "elapsed_seconds": 12.4}
    ]


# ============================================================================
# 8. PER-CARD BOT LEVELS
# ============================================================================

def test_hard_bot_at_800_trophies():
    """Игрок с 800 trophies получает осмысленный mid-game tier."""
    from ai.bot_factory import BotGenerator
    diff = BotGenerator._calc_difficulty(800)
    assert diff == "tier_medium_minus_0600"


def test_lite_easy_always_level_1():
    """lite remains starter-level; easy is still capped below normal play."""
    from ai.bot_factory import BotGenerator
    assert BotGenerator._build_bot_card_levels("lite", 10, 8) == [1] * 8
    assert BotGenerator._build_bot_card_levels("easy", 10, 8) == [2] * 8


def test_medium_per_card_level_5():
    """medium при player_max_level=5 не получает level advantage."""
    from ai.bot_factory import BotGenerator
    for _ in range(20):
        levels = BotGenerator._build_bot_card_levels("medium", 5, 8)
        for lvl in levels:
            assert lvl == 5


def test_hard_per_card_level_5():
    """hard при player_max_level=5 не получает level advantage."""
    from ai.bot_factory import BotGenerator
    for _ in range(20):
        levels = BotGenerator._build_bot_card_levels("hard", 5, 8)
        for lvl in levels:
            assert lvl == 5


def test_max_per_card_level_5():
    """max при player_max_level=5 получает только частичный +1 boost."""
    from ai.bot_factory import BotGenerator
    for _ in range(20):
        levels = BotGenerator._build_bot_card_levels("max", 5, 8)
        for lvl in levels:
            assert 5 <= lvl <= 6


def test_per_card_levels_can_differ():
    """Уровни в колоде могут отличаться (per-card random)."""
    import random
    random.seed(42)
    from ai.bot_factory import BotGenerator
    levels = BotGenerator._build_bot_card_levels("tier_medium_plus_2000", 5, 4)
    assert levels == [4, 4, 5, 4], f"got {levels}"


def test_clamp_to_10_at_high_max():
    """max при player_max_level=9 не выходит выше 10."""
    from ai.bot_factory import BotGenerator
    for _ in range(30):
        levels = BotGenerator._build_bot_card_levels("max", 9, 8)
        for lvl in levels:
            assert lvl <= 10


def test_fallback_player_max_level_1():
    """При пустой колоде fallback max_level=1."""
    from ai.bot_factory import BotGenerator
    levels = BotGenerator._build_bot_card_levels("medium", 1, 5)
    for lvl in levels:
        assert lvl == 1


class _FakeMaxDB:
    def __init__(self, presets=None, user_cards=None, primary_deck=None):
        self._presets = presets or []
        self._user_cards = user_cards or []
        self._primary_deck = primary_deck
        self._pool = True

    async def get_user_deck_presets(self, user_id):
        return self._presets

    async def fetchval(self, query, *args):
        return self._primary_deck

    async def get_user_cards(self, user_id):
        return self._user_cards


def test_get_player_deck_max_level_no_presets():
    import asyncio
    from infrastructure.database import Database
    db = _FakeMaxDB(presets=[], user_cards=[])
    val = asyncio.run(Database.get_player_deck_max_level(db, 123))
    assert val == 1


def test_get_player_deck_max_level_with_data():
    import asyncio
    from infrastructure.database import Database
    presets = [{"preset_number": 1, "card_ids": [10, 20, 30]}]
    user_cards = [
        {"id": 10, "level": 3},
        {"id": 20, "level": 7},
        {"id": 30, "level": 5},
    ]
    db = _FakeMaxDB(presets=presets, user_cards=user_cards, primary_deck=1)
    val = asyncio.run(Database.get_player_deck_max_level(db, 123))
    assert val == 7


def test_get_player_deck_max_level_missing_card():
    import asyncio
    from infrastructure.database import Database
    presets = [{"preset_number": 1, "card_ids": [10, 999]}]
    user_cards = [{"id": 10, "level": 5}]
    db = _FakeMaxDB(presets=presets, user_cards=user_cards, primary_deck=1)
    val = asyncio.run(Database.get_player_deck_max_level(db, 123))
    assert val == 5


def test_get_player_deck_avg_level_with_data():
    import asyncio
    from infrastructure.database import Database
    presets = [{"preset_number": 1, "card_ids": [10, 20, 30]}]
    user_cards = [
        {"id": 10, "level": 3},
        {"id": 20, "level": 7},
        {"id": 30, "level": 5},
    ]
    db = _FakeMaxDB(presets=presets, user_cards=user_cards, primary_deck=1)
    val = asyncio.run(Database.get_player_deck_avg_level(db, 123))
    assert val == 5


# ============================================================================
# 9. SLOT-BASED DUPES
# ============================================================================

def test_slot_levels_handle_duplicate_cards():
    """deck=[10,10,10], card_levels=[4,6,8] → три CardInstance с уровнями 4,6,8."""
    from core.converter import deck_from_card_ids

    cards_data = {
        10: {
            'id': 10, 'name': 'Duplicate', 'card_type': 'warrior',
            'base_attack': 3, 'current_attack': 3, 'base_hp': 3, 'current_hp': 3,
            'mana_cost': 2, 'rarity': 'common', 'mechanics': [],
        },
    }

    deck = deck_from_card_ids(
        card_ids=[10, 10, 10],
        cards_data=cards_data,
        user_levels={10: 1},
        slot_levels=[4, 6, 8],
    )
    assert len(deck) == 3
    assert deck[0].level == 4
    assert deck[1].level == 6
    assert deck[2].level == 8


# ============================================================================
# 10. HERO AURA (regression: hero aura_atk_* buffs friendly units)
# ============================================================================

def test_hero_aura_atk_buffs_friendly_units():
    """Герой с aura_atk_1 даёт +1 атаки союзному юниту на board."""
    state = create_minimal_game_state()
    state.p1.hero.mechanics = ["aura_atk_1"]

    ally = CardInstance(
        instance_id=uuid4(), card_id=99, name="Ally",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2,
        mana_cost=2, mechanics=[], is_ready=True,
    )
    state.p1.board.append(ally)

    enemy = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy)

    env = ArenaEnvironment(state)
    success, error = env.step(1, AttackAction(
        attacker_id=str(ally.instance_id),
        target_id=str(enemy.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака должна пройти: {error}"
    assert enemy.hp == 2, f"Enemy должен получить 2+1=3 урона, HP={enemy.hp}"


def test_hero_aura_atk_buffs_friendly_unit_attacking_hero():
    """Герой с aura_atk_1: союзный юнит бьёт вражеского героя с бонусом."""
    state = create_minimal_game_state()
    state.p1.hero.mechanics = ["aura_atk_1"]

    ally = CardInstance(
        instance_id=uuid4(), card_id=99, name="Ally",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2,
        mana_cost=2, mechanics=[], is_ready=True,
    )
    state.p1.board.append(ally)

    hero_hp_before = state.p2.hero.hp
    env = ArenaEnvironment(state)
    success, error = env.step(1, AttackAction(
        attacker_id=str(ally.instance_id),
        target_id=None,
        target_is_hero=True,
    ))
    assert success, f"Атака героя должна пройти: {error}"
    assert state.p2.hero.hp == hero_hp_before - 3, (
        f"Герой должен получить 2+1=3 урона, HP={state.p2.hero.hp}"
    )


def test_hero_aura_atk_stacks_with_board_aura():
    """Герой с aura_atk_1 и board-юнит с aura_atk_2 дают суммарно +3 атаки."""
    state = create_minimal_game_state()
    state.p1.hero.mechanics = ["aura_atk_1"]

    aura_unit = CardInstance(
        instance_id=uuid4(), card_id=98, name="AuraUnit",
        card_type=CardType.WARRIOR, hp=2, max_hp=2, attack=0,
        mana_cost=3, mechanics=["aura_atk_2"], is_ready=False,
    )
    state.p1.board.append(aura_unit)

    attacker = CardInstance(
        instance_id=uuid4(), card_id=99, name="Attacker",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=2,
        mana_cost=2, mechanics=[], is_ready=True,
    )
    state.p1.board.append(attacker)

    enemy = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=5, max_hp=5, attack=1,
        mana_cost=2, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy)

    env = ArenaEnvironment(state)
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=str(enemy.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака должна пройти: {error}"
    assert enemy.hp == 0, f"Enemy должен получить 2+1+2=5 урона и умереть, HP={enemy.hp}"


def test_hero_aura_atk_appears_in_legal_actions_effective_attack():
    """Эффективная атака в legal_actions учитывает hero aura, проверяя != 0."""
    state = create_minimal_game_state()
    state.p1.hero.mechanics = ["aura_atk_1"]

    zero_atk_unit = CardInstance(
        instance_id=uuid4(), card_id=99, name="ZeroAtk",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=0,
        mana_cost=1, mechanics=[], is_ready=True,
    )
    state.p1.board.append(zero_atk_unit)

    enemy = CardInstance(
        instance_id=uuid4(), card_id=100, name="Enemy",
        card_type=CardType.WARRIOR, hp=3, max_hp=3, attack=1,
        mana_cost=1, mechanics=[], is_ready=False,
    )
    state.p2.board.append(enemy)

    env = ArenaEnvironment(state)
    actions = env.get_legal_actions(1)
    attack_actions = [a for a in actions if isinstance(a, AttackAction)]
    assert len(attack_actions) >= 1, (
        f"Юнит с базовой атакой 0 должен иметь attack action благодаря hero aura +1, "
        f"получено {len(attack_actions)}"
    )


# ============================================================================
# 11. HERO REGEN (regression: hero regen_* heals at end turn)
# ============================================================================

def test_hero_regen_triggers_on_end_turn():
    """Повреждённый герой с regen_1 лечится на 1 в конце хода противника."""
    state = create_minimal_game_state()
    state.p2.hero.hp = 25
    state.p2.hero.max_hp = 30
    state.p2.hero.mechanics = ["regen_1"]

    env = ArenaEnvironment(state)
    success, error = env.step(1, EndTurnAction())
    assert success, f"End turn должен пройти: {error}"
    assert state.p2.hero.hp == 26, (
        f"Герой P2 должен был регенерировать 1 HP, HP={state.p2.hero.hp}"
    )


def test_hero_regen_does_not_exceed_max_hp():
    """Герой с regen_2 не превышает max_hp при почти полном здоровье."""
    state = create_minimal_game_state()
    state.p2.hero.hp = 29
    state.p2.hero.max_hp = 30
    state.p2.hero.mechanics = ["regen_2"]

    env = ArenaEnvironment(state)
    success, error = env.step(1, EndTurnAction())
    assert success, f"End turn должен пройти: {error}"
    assert state.p2.hero.hp == 30, (
        f"Герой P2 не должен превысить max_hp=30, HP={state.p2.hero.hp}"
    )


def test_hero_regen_does_not_heal_dead_hero():
    """Мёртвый герой (hp <= 0) не лечится регеном."""
    state = create_minimal_game_state()
    state.p2.hero.hp = 0
    state.p2.hero.max_hp = 30
    state.p2.hero.mechanics = ["regen_2"]

    env = ArenaEnvironment(state)
    success, error = env.step(1, EndTurnAction())
    assert success, f"End turn должен пройти: {error}"
    assert state.p2.hero.hp == 0, (
        f"Мёртвый герой не должен лечиться, HP={state.p2.hero.hp}"
    )


def test_hero_regen_does_not_affect_board_units():
    """Hero regen не лечит board-юнитов, а board regen работает как раньше."""
    state = create_minimal_game_state()
    state.p2.hero.mechanics = ["regen_1"]

    board_unit = CardInstance(
        instance_id=uuid4(), card_id=100, name="Unit",
        card_type=CardType.WARRIOR, hp=2, max_hp=5, attack=1,
        mana_cost=2, mechanics=["regen_1"], is_ready=False,
    )
    state.p2.board.append(board_unit)

    env = ArenaEnvironment(state)
    success, error = env.step(1, EndTurnAction())
    assert success, f"End turn должен пройти: {error}"
    assert board_unit.hp == 3, f"Board unit с regen_1 должен был вылечиться на 1, HP={board_unit.hp}"
    assert state.p2.hero.hp == 30, "Герой P2 не был повреждён и не должен менять HP"
