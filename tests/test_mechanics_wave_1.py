"""
Тесты для первой волны механик: щит, броня, заморозка, боевые кличи.
"""
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.effects import apply_damage


def create_minimal_game_state() -> GameState:
    """Создать минимальное игровое состояние для тестов."""
    # Герой игрока 1
    hero_p1 = CardInstance(
        instance_id=uuid4(),
        card_id=1,
        name="Hero P1",
        card_type=CardType.HERO,
        hp=30,
        max_hp=30,
        attack=0,
        mana_cost=0,
    )
    
    # Герой игрока 2
    hero_p2 = CardInstance(
        instance_id=uuid4(),
        card_id=2,
        name="Hero P2",
        card_type=CardType.HERO,
        hp=30,
        max_hp=30,
        attack=0,
        mana_cost=0,
    )
    
    p1 = PlayerState(
        user_id=1,
        is_bot=False,
        hero=hero_p1,
        mana=10,
        max_mana=10,
        hand=[],
        board=[],
        deck=[],
    )
    
    p2 = PlayerState(
        user_id=2,
        is_bot=False,
        hero=hero_p2,
        mana=10,
        max_mana=10,
        hand=[],
        board=[],
        deck=[],
    )
    
    return GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=1,
        turn_number=1,
        status=GameStatus.ONGOING,
    )


def test_shield_blocks_damage():
    """Тест 1: Юнит со щитом получает 10 урона -> HP не меняется, щит исчезает."""
    state = create_minimal_game_state()
    
    # Создаем юнита со щитом
    unit_with_shield = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Shield Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=["shield"],
        is_ready=False,
    )
    
    state.p1.board.append(unit_with_shield)
    
    # Наносим 10 урона
    apply_damage(unit_with_shield, 10)
    
    # Проверяем, что HP не изменилось
    assert unit_with_shield.hp == 5, "Щит должен полностью заблокировать урон"
    
    # Проверяем, что щит исчез
    assert "shield" not in unit_with_shield.mechanics, "Щит должен исчезнуть после блокировки"
    
    # Второй удар должен пройти
    apply_damage(unit_with_shield, 3)
    assert unit_with_shield.hp == 2, "Второй удар должен нанести урон"


def test_armor_reduces_damage():
    """Тест 2: Юнит с броней 2 получает 5 урона -> теряет только 3 HP."""
    state = create_minimal_game_state()
    
    # Создаем юнита с броней
    unit_with_armor = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Armored Warrior",
        card_type=CardType.WARRIOR,
        hp=10,
        max_hp=10,
        attack=3,
        mana_cost=4,
        mechanics=["armor_2"],
        is_ready=False,
    )
    
    state.p1.board.append(unit_with_armor)
    
    # Наносим 5 урона
    apply_damage(unit_with_armor, 5)
    
    # Проверяем, что HP уменьшилось на 3 (5 - 2)
    assert unit_with_armor.hp == 7, f"Юнит с броней 2 должен потерять 3 HP из 5 урона, осталось {unit_with_armor.hp}"


def test_freeze_prevents_attack_on_next_turn():
    """Тест 3: Замороженный юнит после end_turn теряет статус заморозки и не готов атаковать."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем замороженного юнита у игрока 2
    frozen_unit = CardInstance(
        instance_id=uuid4(),
        card_id=102,
        name="Frozen Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=[],
        is_ready=True,
        is_frozen=True,  # Движок использует is_frozen
    )
    
    state.p2.board.append(frozen_unit)
    
    # Игрок 1 завершает ход
    success, error = env.step(1, EndTurnAction())
    assert success, f"Завершение хода должно быть успешным: {error}"
    
    # Проверяем, что заморозка снята
    assert not frozen_unit.is_frozen, "is_frozen должен быть сброшен"
    
    # Проверяем, что юнит НЕ готов атаковать (пропустил активацию)
    assert not frozen_unit.is_ready, "Замороженный юнит не должен быть готов атаковать сразу после разморозки"


def test_battlecry_heal_hero():
    """Тест 4: Разыгрывание карты с battlecry_heal_hero_2 лечит героя."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Уменьшаем HP героя
    state.p1.hero.hp = 20
    
    # Создаем карту с лечением героя
    heal_card = CardInstance(
        instance_id=uuid4(),
        card_id=103,
        name="Healer",
        card_type=CardType.WARRIOR,
        hp=2,
        max_hp=2,
        attack=2,
        mana_cost=2,
        mechanics=["battlecry_heal_hero_2"],
        is_ready=False,
    )
    
    state.p1.hand.append(heal_card)
    
    # Разыгрываем карту
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
    assert success, f"Розыгрыш карты должен быть успешным: {error}"
    
    # Проверяем, что герой получил 2 HP
    assert state.p1.hero.hp == 22, f"Герой должен был получить 2 HP, текущий HP: {state.p1.hero.hp}"


def test_battlecry_damage_to_target():
    """Тест 5: Разыгрывание карты с battlecry_damage_1 наносит урон выбранной цели."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем вражеское существо - цель
    enemy_unit = CardInstance(
        instance_id=uuid4(),
        card_id=104,
        name="Enemy Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=[],
        is_ready=False,
    )
    
    state.p2.board.append(enemy_unit)
    
    # Создаем карту с боевым кличем урона
    damage_card = CardInstance(
        instance_id=uuid4(),
        card_id=105,
        name="Damage Dealer",
        card_type=CardType.WARRIOR,
        hp=2,
        max_hp=2,
        attack=2,
        mana_cost=2,
        mechanics=["battlecry_damage_1"],
        is_ready=False,
    )
    
    state.p1.hand.append(damage_card)
    
    # Разыгрываем карту с указанием цели
    target_id = str(enemy_unit.instance_id)
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=target_id, position=0))
    assert success, f"Розыгрыш карты должен быть успешным: {error}"
    
    # Проверяем, что враг получил 1 урон
    assert enemy_unit.hp == 4, f"Вражеский юнит должен потерять 1 HP, текущий HP: {enemy_unit.hp}"


def test_armor_range():
    """Бонусный тест: Броня в диапазоне armor_1_3 работает корректно."""
    state = create_minimal_game_state()
    
    # Создаем юнита с броней в диапазоне
    unit_with_range_armor = CardInstance(
        instance_id=uuid4(),
        card_id=106,
        name="Flexible Armor Warrior",
        card_type=CardType.WARRIOR,
        hp=10,
        max_hp=10,
        attack=3,
        mana_cost=4,
        mechanics=["armor_1_3"],
        is_ready=False,
    )
    
    state.p1.board.append(unit_with_range_armor)
    
    # Наносим 5 урона несколько раз и проверяем, что урон уменьшается
    results = []
    for _ in range(5):
        unit = CardInstance(
            instance_id=uuid4(),
            card_id=106,
            name="Test",
            card_type=CardType.WARRIOR,
            hp=10,
            max_hp=10,
            attack=3,
            mana_cost=4,
            mechanics=["armor_1_3"],
            is_ready=False,
        )
        apply_damage(unit, 5)
        results.append(unit.hp)
    
    # Проверяем, что HP находится в ожидаемом диапазоне (10 - (5 - броня[1-3]) = от 6 до 8)
    for hp in results:
        assert 6 <= hp <= 8, f"HP должно быть между 6 и 8 (10 HP - (5 урона - броня 1-3)), получено: {hp}"

