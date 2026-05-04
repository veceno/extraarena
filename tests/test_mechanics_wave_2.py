"""
Тесты для второй волны механик: Taunt, Bypass Taunt, AOE Damage/Freeze, Mana Control.
"""
import pytest
from uuid import uuid4

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.effects import apply_damage


def create_minimal_game_state() -> GameState:
    """Создать минимальное игровое состояние для тестов."""
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


def test_taunt_blocks_hero_attack():
    """Тест 1: Taunt блокирует атаку героя."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем атакующего юнита у игрока 1
    attacker = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Attacker",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=[],
        is_ready=True,
    )
    state.p1.board.append(attacker)
    
    # Создаем taunt юнита у игрока 2
    taunt_unit = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Taunt Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=2,
        mana_cost=3,
        mechanics=["taunt"],
        is_ready=False,
    )
    state.p2.board.append(taunt_unit)
    
    # Пытаемся атаковать героя
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=None,
        target_is_hero=True,
    ))
    
    # Атака должна быть заблокирована
    assert not success, "Атака героя должна быть заблокирована при наличии Taunt"
    assert error == "must_attack_taunt", f"Ожидалась ошибка 'must_attack_taunt', получено: {error}"


def test_taunt_forces_attack_on_taunt_unit():
    """Тест 2: Taunt заставляет атаковать только taunt юнитов."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем атакующего юнита
    attacker = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Attacker",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=[],
        is_ready=True,
    )
    state.p1.board.append(attacker)
    
    # Создаем обычного юнита у противника
    normal_unit = CardInstance(
        instance_id=uuid4(),
        card_id=102,
        name="Normal Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=2,
        mana_cost=3,
        mechanics=[],
        is_ready=False,
    )
    state.p2.board.append(normal_unit)
    
    # Создаем taunt юнита
    taunt_unit = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Taunt Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=2,
        mana_cost=3,
        mechanics=["taunt"],
        is_ready=False,
    )
    state.p2.board.append(taunt_unit)
    
    # Пытаемся атаковать обычного юнита (должно быть заблокировано)
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=str(normal_unit.instance_id),
        target_is_hero=False,
    ))
    
    assert not success, "Атака обычного юнита должна быть заблокирована при наличии Taunt"
    assert error == "must_attack_taunt"
    
    # Атакуем taunt юнита (должно пройти)
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=str(taunt_unit.instance_id),
        target_is_hero=False,
    ))
    
    assert success, f"Атака taunt юнита должна быть успешной: {error}"
    assert taunt_unit.hp == 2, "Taunt юнит должен получить урон"


def test_bypass_taunt_ignores_taunt():
    """Тест 3: bypass_taunt игнорирует taunt."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем атакующего с bypass_taunt
    attacker = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Bypass Attacker",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=4,
        mechanics=["bypass_taunt"],
        is_ready=True,
    )
    state.p1.board.append(attacker)
    
    # Создаем taunt юнита у противника
    taunt_unit = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Taunt Warrior",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=2,
        mana_cost=3,
        mechanics=["taunt"],
        is_ready=False,
    )
    state.p2.board.append(taunt_unit)
    
    # Атакуем героя напрямую (должно пройти благодаря bypass_taunt)
    hero_hp_before = state.p2.hero.hp
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=None,
        target_is_hero=True,
    ))
    
    assert success, f"Атака героя с bypass_taunt должна быть успешной: {error}"
    assert state.p2.hero.hp == hero_hp_before - 3, "Герой должен получить урон"


def test_aoe_damage():
    """Тест 4: AOE урон наносится всем вражеским юнитам."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем несколько вражеских юнитов
    enemy_units = []
    for i in range(3):
        unit = CardInstance(
            instance_id=uuid4(),
            card_id=200 + i,
            name=f"Enemy {i}",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=2,
            mechanics=[],
            is_ready=False,
        )
        state.p2.board.append(unit)
        enemy_units.append(unit)
    
    # Создаем карту с AOE уроном
    aoe_card = CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name="AOE Spell",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=3,
        mechanics=["aoe_damage_2"],
        is_ready=False,
    )
    state.p1.hand.append(aoe_card)
    
    # Разыгрываем AOE карту
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш AOE карты должен быть успешным: {error}"
    
    # Проверяем, что все враги получили урон
    for unit in enemy_units:
        assert unit.hp == 3, f"Юнит {unit.name} должен получить 2 урона, текущий HP: {unit.hp}"


def test_aoe_freeze():
    """Тест 5: AOE заморозка замораживает всех вражеских юнитов."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем несколько вражеских юнитов
    enemy_units = []
    for i in range(3):
        unit = CardInstance(
            instance_id=uuid4(),
            card_id=200 + i,
            name=f"Enemy {i}",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=2,
            mechanics=[],
            is_ready=True,  # Готовы к атаке
        )
        state.p2.board.append(unit)
        enemy_units.append(unit)
    
    # Создаем карту с AOE заморозкой
    freeze_card = CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name="Freeze Spell",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=4,
        mechanics=["aoe_freeze"],
        is_ready=False,
    )
    state.p1.hand.append(freeze_card)
    
    # Разыгрываем карту заморозки
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш карты заморозки должен быть успешным: {error}"
    
    # Проверяем, что все враги заморожены
    for unit in enemy_units:
        assert "freeze" in unit.mechanics, f"Юнит {unit.name} должен быть заморожен"


def test_mana_gain():
    """Тест 6: mana_gain восстанавливает ману владельцу."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Уменьшаем ману игрока
    state.p1.mana = 3
    
    # Создаем карту с восстановлением маны
    mana_card = CardInstance(
        instance_id=uuid4(),
        card_id=400,
        name="Mana Potion",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=2,
        mechanics=["mana_gain_3"],
        is_ready=False,
    )
    state.p1.hand.append(mana_card)
    
    # Разыгрываем карту (стоит 2 маны, восстанавливает 3)
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш карты должен быть успешным: {error}"
    
    # Проверяем ману: 3 - 2 (стоимость) + 3 (восстановление) = 4
    assert state.p1.mana == 4, f"Мана должна быть 4, текущая: {state.p1.mana}"


def test_mana_drain():
    """Тест 7: mana_drain отнимает ману у противника."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Устанавливаем ману противника
    state.p2.mana = 5
    
    # Создаем карту с дрейном маны
    drain_card = CardInstance(
        instance_id=uuid4(),
        card_id=500,
        name="Mana Drain",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=2,
        mechanics=["mana_drain_3"],
        is_ready=False,
    )
    state.p1.hand.append(drain_card)
    
    # Разыгрываем карту
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш карты должен быть успешным: {error}"
    
    # Проверяем, что противник потерял 3 маны
    assert state.p2.mana == 2, f"Мана противника должна быть 2, текущая: {state.p2.mana}"


def test_mana_drain_cannot_go_negative():
    """Тест 8: mana_drain не может уйти в минус."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Устанавливаем ману противника
    state.p2.mana = 1
    
    # Создаем карту с дрейном маны
    drain_card = CardInstance(
        instance_id=uuid4(),
        card_id=500,
        name="Mana Drain",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=2,
        mechanics=["mana_drain_5"],
        is_ready=False,
    )
    state.p1.hand.append(drain_card)
    
    # Разыгрываем карту
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш карты должен быть успешным: {error}"
    
    # Проверяем, что мана противника = 0 (не ушла в минус)
    assert state.p2.mana == 0, f"Мана противника должна быть 0, текущая: {state.p2.mana}"


