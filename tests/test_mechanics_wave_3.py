"""
Тесты для третьей волны механик: Instant Kill, Cleave, Consume, Reflect, Permanent Shield.
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


def test_instant_kill_unit():
    """Тест 1: instant_kill мгновенно убивает юнита независимо от HP."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем атакующего с instant_kill
    instant_killer = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Assassin",
        card_type=CardType.WARRIOR,
        hp=3,
        max_hp=3,
        attack=1,  # Всего 1 атака, но убьет любого
        mana_cost=5,
        mechanics=["instant_kill"],
        is_ready=True,
    )
    state.p1.board.append(instant_killer)
    
    # Создаем танка с огромным HP
    tank = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Tank",
        card_type=CardType.WARRIOR,
        hp=100,
        max_hp=100,
        attack=2,
        mana_cost=10,
        mechanics=[],
        is_ready=False,
    )
    state.p2.board.append(tank)
    
    # Атакуем танка
    success, error = env.step(1, AttackAction(
        attacker_id=str(instant_killer.instance_id),
        target_id=str(tank.instance_id),
        target_is_hero=False,
    ))
    
    assert success, f"Атака должна быть успешной: {error}"
    assert tank.hp == 0, f"Танк должен быть мгновенно убит, HP: {tank.hp}"


def test_instant_kill_hero():
    """Тест 2: instant_kill НЕ убивает героя мгновенно — наносит базовый урон."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    instant_killer = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Assassin",
        card_type=CardType.WARRIOR,
        hp=3,
        max_hp=3,
        attack=1,
        mana_cost=5,
        mechanics=["instant_kill"],
        is_ready=True,
    )
    state.p1.board.append(instant_killer)
    
    hero_hp_before = state.p2.hero.hp
    success, error = env.step(1, AttackAction(
        attacker_id=str(instant_killer.instance_id),
        target_id=None,
        target_is_hero=True,
    ))
    
    assert success, f"Атака героя должна быть успешной: {error}"
    assert state.p2.hero.hp == hero_hp_before - 1, f"Герой должен получить только 1 урона, HP: {state.p2.hero.hp}"


def test_cleave_damages_multiple_targets():
    """Тест 3: cleave_X_Y наносит урон Y случайным целям."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем 3 вражеских юнита
    enemy_units = []
    for i in range(3):
        unit = CardInstance(
            instance_id=uuid4(),
            card_id=200 + i,
            name=f"Enemy {i}",
            card_type=CardType.WARRIOR,
            hp=10,
            max_hp=10,
            attack=2,
            mana_cost=3,
            mechanics=[],
            is_ready=False,
        )
        state.p2.board.append(unit)
        enemy_units.append(unit)
    
    # Создаем карту с cleave: 3 урона 2 раза
    cleave_card = CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name="Cleave Spell",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=4,
        mechanics=["cleave_3_2"],  # 3 урона, 2 раза
        is_ready=False,
    )
    state.p1.hand.append(cleave_card)
    
    # Разыгрываем cleave
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Розыгрыш cleave должен быть успешным: {error}"
    
    # Проверяем, что был нанесен урон (как минимум одному юниту)
    total_damage = sum(10 - unit.hp for unit in enemy_units)
    assert total_damage >= 3, f"Должен быть нанесен урон, всего: {total_damage}"


def test_cleave_handles_dying_targets():
    """Тест 4: cleave корректно обрабатывает смерть целей в процессе."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем 2 слабых врага
    enemy_units = []
    for i in range(2):
        unit = CardInstance(
            instance_id=uuid4(),
            card_id=200 + i,
            name=f"Weak Enemy {i}",
            card_type=CardType.WARRIOR,
            hp=2,  # Легко убиваемые
            max_hp=2,
            attack=1,
            mana_cost=1,
            mechanics=[],
            is_ready=False,
        )
        state.p2.board.append(unit)
        enemy_units.append(unit)
    
    # Cleave который может убить всех
    cleave_card = CardInstance(
        instance_id=uuid4(),
        card_id=300,
        name="Mass Cleave",
        card_type=CardType.POTION,
        hp=0,
        max_hp=0,
        attack=0,
        mana_cost=5,
        mechanics=["cleave_5_3"],  # 5 урона, 3 раза (больше чем врагов)
        is_ready=False,
    )
    state.p1.hand.append(cleave_card)
    
    # Разыгрываем
    success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
    assert success, f"Cleave должен быть успешным: {error}"
    
    # Проверяем, что оба врага мертвы
    for unit in enemy_units:
        assert unit.hp == 0, f"Юнит {unit.name} должен быть убит"


def test_consume_ally():
    """Тест 5: consume_ally поглощает союзника и получает его статы."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем жертву
    sacrifice = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Sacrifice",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=2,
        mechanics=[],
        is_ready=False,
    )
    state.p1.board.append(sacrifice)
    
    # Создаем поглотителя
    consumer = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Consumer",
        card_type=CardType.WARRIOR,
        hp=2,
        max_hp=2,
        attack=2,
        mana_cost=3,
        mechanics=["consume_ally"],
        is_ready=False,
    )
    state.p1.hand.append(consumer)
    
    board_size_before = len(state.p1.board)
    
    # Разыгрываем поглотителя, указывая жертву
    success, error = env.step(1, PlayCardAction(
        hand_index=0,
        target_id=str(sacrifice.instance_id),
        position=0
    ))
    assert success, f"Разыгрывание с consume должно быть успешным: {error}"
    
    # Проверяем, что жертва исчезла
    assert len(state.p1.board) == board_size_before, "Количество юнитов не должно измениться"
    
    # Проверяем, что поглотитель получил статы жертвы
    assert consumer.attack == 5, f"Атака должна быть 2+3=5, получено: {consumer.attack}"
    assert consumer.hp == 7, f"HP должно быть 2+5=7, получено: {consumer.hp}"
    assert consumer.max_hp == 7, f"Max HP должно быть 2+5=7, получено: {consumer.max_hp}"


def test_reflect_damages_attacker():
    """Тест 6: reflect_X отражает урон атакующему."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Создаем атакующего
    attacker = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Attacker",
        card_type=CardType.WARRIOR,
        hp=10,
        max_hp=10,
        attack=5,
        mana_cost=4,
        mechanics=[],
        is_ready=True,
    )
    state.p1.board.append(attacker)
    
    # Создаем юнита с reflect
    reflector = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Reflector",
        card_type=CardType.WARRIOR,
        hp=8,
        max_hp=8,
        attack=2,
        mana_cost=4,
        mechanics=["reflect_3"],  # Отражает 3 урона
        is_ready=False,
    )
    state.p2.board.append(reflector)
    
    # Атакуем reflector'а
    success, error = env.step(1, AttackAction(
        attacker_id=str(attacker.instance_id),
        target_id=str(reflector.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака должна быть успешной: {error}"
    
    # Проверяем урон reflector'у: получил 5 атаки
    assert reflector.hp == 3, f"Reflector должен получить 5 урона, HP: {reflector.hp}"
    
    # Проверяем урон атакующему: получил 2 (ответный удар) + 3 (reflect) = 5
    assert attacker.hp == 5, f"Attacker должен получить 2+3=5 урона, HP: {attacker.hp}"


def test_permanent_shield_never_breaks():
    """Тест 7: permanent_shield блокирует урон навсегда."""
    state = create_minimal_game_state()
    
    immortal = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Immortal",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=6,
        mechanics=["permanent_shield"],
        is_ready=False,
    )
    state.p1.board.append(immortal)
    
    # Наносим урон много раз
    for _ in range(5):
        apply_damage(immortal, 10)
        assert immortal.hp == 5, "HP не должно меняться с permanent_shield"
        assert "permanent_shield" in immortal.mechanics, "Permanent shield не должен исчезать"


def test_permanent_shield_vs_normal_shield():
    """Тест 8: permanent_shield отличается от обычного щита."""
    state = create_minimal_game_state()
    
    # Юнит с обычным щитом
    normal = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Normal Shield",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=3,
        mechanics=["shield"],
        is_ready=False,
    )
    
    # Юнит с permanent_shield
    permanent = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Permanent Shield",
        card_type=CardType.WARRIOR,
        hp=5,
        max_hp=5,
        attack=3,
        mana_cost=6,
        mechanics=["permanent_shield"],
        is_ready=False,
    )
    
    # Первый удар
    apply_damage(normal, 10)
    apply_damage(permanent, 10)
    
    assert normal.hp == 5, "Обычный щит должен заблокировать первый удар"
    assert permanent.hp == 5, "Permanent shield должен заблокировать первый удар"
    assert "shield" not in normal.mechanics, "Обычный щит должен исчезнуть"
    assert "permanent_shield" in permanent.mechanics, "Permanent shield должен остаться"
    
    # Второй удар
    apply_damage(normal, 10)
    apply_damage(permanent, 10)
    
    assert normal.hp == 0, "Без щита юнит должен получить урон"
    assert permanent.hp == 5, "Permanent shield должен заблокировать и второй удар"


def test_instant_kill_vs_permanent_shield():
    """Тест 9: instant_kill обходит permanent_shield."""
    state = create_minimal_game_state()
    env = ArenaEnvironment(state)
    
    # Assassin с instant_kill
    assassin = CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name="Assassin",
        card_type=CardType.WARRIOR,
        hp=3,
        max_hp=3,
        attack=1,
        mana_cost=5,
        mechanics=["instant_kill"],
        is_ready=True,
    )
    state.p1.board.append(assassin)
    
    # Immortal с permanent_shield
    immortal = CardInstance(
        instance_id=uuid4(),
        card_id=101,
        name="Immortal",
        card_type=CardType.WARRIOR,
        hp=10,
        max_hp=10,
        attack=2,
        mana_cost=8,
        mechanics=["permanent_shield"],
        is_ready=False,
    )
    state.p2.board.append(immortal)
    
    # Атакуем
    success, error = env.step(1, AttackAction(
        attacker_id=str(assassin.instance_id),
        target_id=str(immortal.instance_id),
        target_is_hero=False,
    ))
    assert success, f"Атака должна быть успешной: {error}"
    
    # instant_kill должен убить даже с permanent_shield
    assert immortal.hp == 0, "instant_kill должен убить через permanent_shield"


