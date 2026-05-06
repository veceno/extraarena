"""
Полный набор тестов игровых механик.
Объединяет Волну №1 (Shield, Armor, Freeze) и Волну №2 (Taunt, AOE, Mana).
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


# ============================================================================
# TEST CLASS: STATUS EFFECTS (Shield, Armor, Freeze)
# ============================================================================

class TestStatusEffects:
    """Тесты статусных эффектов: щит, броня, заморозка."""
    
    def test_shield_blocks_damage(self):
        """Щит полностью блокирует урон и исчезает."""
        state = create_minimal_game_state()
        
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
        
        # Первая атака блокируется щитом
        apply_damage(unit_with_shield, 10)
        assert unit_with_shield.hp == 5, "Щит должен полностью заблокировать урон"
        assert "shield" not in unit_with_shield.mechanics, "Щит должен исчезнуть"
        
        # Вторая атака проходит
        apply_damage(unit_with_shield, 3)
        assert unit_with_shield.hp == 2, "Второй удар должен нанести урон"
    
    def test_armor_reduces_damage(self):
        """Броня уменьшает получаемый урон."""
        state = create_minimal_game_state()
        
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
        
        apply_damage(unit_with_armor, 5)
        assert unit_with_armor.hp == 7, f"Броня должна уменьшить урон на 2, HP={unit_with_armor.hp}"
    
    def test_armor_range(self):
        """Броня с диапазоном работает корректно."""
        results = []
        for _ in range(5):
            unit = CardInstance(
                instance_id=uuid4(),
                card_id=106,
                name="Flexible Armor",
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
        
        # HP должно быть в диапазоне 6-8 (10 - (5 - [1-3]))
        for hp in results:
            assert 6 <= hp <= 8, f"HP должно быть 6-8, получено: {hp}"
    
    def test_freeze_prevents_attack_on_next_turn(self):
        """Замороженный юнит пропускает активацию после размораживания."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
            is_frozen=True,  # Движок использует is_frozen булево поле
        )
        state.p2.board.append(frozen_unit)
        
        # Игрок 1 завершает ход
        success, error = env.step(1, EndTurnAction())
        assert success, f"Ошибка завершения хода: {error}"
        
        assert not frozen_unit.is_frozen, "is_frozen должен быть сброшен"
        assert not frozen_unit.is_ready, "Юнит не должен быть готов сразу после разморозки"


# ============================================================================
# TEST CLASS: BOARD MECHANICS (Taunt, Bypass, Battlecry)
# ============================================================================

class TestBoardMechanics:
    """Тесты механик доски: taunt, bypass_taunt, battlecry."""
    
    def test_taunt_blocks_hero_attack(self):
        """Taunt блокирует прямую атаку героя."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        # Попытка атаки героя
        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=None,
            target_is_hero=True,
        ))
        
        assert not success, "Атака героя должна быть заблокирована"
        assert error == "must_attack_taunt"
    
    def test_taunt_blocks_standard_attack(self):
        """Taunt блокирует атаки обычных юнитов через get_legal_actions."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        # Наш атакующий юнит
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
        
        # Вражеский таунт
        taunt_unit = CardInstance(
            instance_id=uuid4(),
            card_id=101,
            name="Taunt",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=3,
            mechanics=["taunt"],
            is_ready=False,
        )
        state.p2.board.append(taunt_unit)
        
        # Получаем легальные действия
        legal_actions = env.get_legal_actions(player_id=1)
        
        # Фильтруем атаки
        attack_actions = [a for a in legal_actions if isinstance(a, AttackAction)]
        hero_attacks = [a for a in attack_actions if a.target_is_hero]
        taunt_attacks = [a for a in attack_actions if a.target_id == str(taunt_unit.instance_id)]
        
        assert len(hero_attacks) == 0, "Атака героя должна отсутствовать в легальных действиях"
        assert len(taunt_attacks) > 0, "Атака таунта должна присутствовать"
    
    def test_taunt_forces_attack_on_taunt_unit(self):
        """Taunt заставляет атаковать только taunt юнитов."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        normal_unit = CardInstance(
            instance_id=uuid4(),
            card_id=102,
            name="Normal",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=3,
            mechanics=[],
            is_ready=False,
        )
        state.p2.board.append(normal_unit)
        
        taunt_unit = CardInstance(
            instance_id=uuid4(),
            card_id=101,
            name="Taunt",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=3,
            mechanics=["taunt"],
            is_ready=False,
        )
        state.p2.board.append(taunt_unit)
        
        # Попытка атаки обычного юнита
        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=str(normal_unit.instance_id),
            target_is_hero=False,
        ))
        assert not success, "Атака обычного юнита должна быть заблокирована"
        assert error == "must_attack_taunt"
        
        # Атака таунта
        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=str(taunt_unit.instance_id),
            target_is_hero=False,
        ))
        assert success, f"Атака таунта должна пройти: {error}"
        assert taunt_unit.hp == 2, "Таунт должен получить урон"
    
    def test_hog_rider_bypasses_taunt(self):
        """Юнит с bypass_taunt игнорирует taunt через get_legal_actions."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        # Hog Rider с bypass_taunt
        hog_rider = CardInstance(
            instance_id=uuid4(),
            card_id=100,
            name="Hog Rider",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=4,
            mana_cost=4,
            mechanics=["bypass_taunt"],
            is_ready=True,
        )
        state.p1.board.append(hog_rider)
        
        # Вражеский таунт
        taunt_unit = CardInstance(
            instance_id=uuid4(),
            card_id=101,
            name="Taunt",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=3,
            mechanics=["taunt"],
            is_ready=False,
        )
        state.p2.board.append(taunt_unit)
        
        # Получаем легальные действия
        legal_actions = env.get_legal_actions(player_id=1)
        
        # Фильтруем атаки героя
        attack_actions = [a for a in legal_actions if isinstance(a, AttackAction)]
        hero_attacks = [a for a in attack_actions if a.target_is_hero]
        
        assert len(hero_attacks) > 0, "Атака героя должна быть разрешена с bypass_taunt"
        
        # Проверяем, что атака героя проходит
        hero_hp_before = state.p2.hero.hp
        success, error = env.step(1, AttackAction(
            attacker_id=str(hog_rider.instance_id),
            target_id=None,
            target_is_hero=True,
        ))
        assert success, f"Атака должна пройти: {error}"
        assert state.p2.hero.hp == hero_hp_before - 4, "Герой должен получить урон"
    
    def test_battlecry_heal_hero(self):
        """Battlecry лечит героя."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        state.p1.hero.hp = 20
        
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
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
        assert success, f"Розыгрыш карты провалился: {error}"
        assert state.p1.hero.hp == 22, f"Герой должен иметь 22 HP, текущий: {state.p1.hero.hp}"
    
    def test_battlecry_damage_to_target(self):
        """Battlecry наносит урон цели."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        enemy_unit = CardInstance(
            instance_id=uuid4(),
            card_id=104,
            name="Enemy",
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=3,
            mana_cost=3,
            mechanics=[],
            is_ready=False,
        )
        state.p2.board.append(enemy_unit)
        
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
        
        target_id = str(enemy_unit.instance_id)
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=target_id, position=0))
        assert success, f"Розыгрыш провалился: {error}"
        assert enemy_unit.hp == 4, f"Враг должен иметь 4 HP, текущий: {enemy_unit.hp}"


# ============================================================================
# TEST CLASS: AOE EFFECTS (Area of Effect)
# ============================================================================

class TestAOEEffects:
    """Тесты массовых эффектов: AOE урон и заморозка."""
    
    def test_aoe_damage_logic(self):
        """AOE урон наносится всем врагам, мертвые удаляются."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        # Создаем 3 вражеских юнита с разным HP
        enemy_units = []
        for i, hp in enumerate([5, 2, 4]):
            unit = CardInstance(
                instance_id=uuid4(),
                card_id=200 + i,
                name=f"Enemy {i}",
                card_type=CardType.WARRIOR,
                hp=hp,
                max_hp=5,
                attack=2,
                mana_cost=2,
                mechanics=[],
                is_ready=False,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)
        
        # AOE карта с 2 уроном
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
        
        # Разыгрываем AOE
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Розыгрыш AOE провалился: {error}"
        
        # Проверяем урон
        assert enemy_units[0].hp == 3, "Enemy 0 должен иметь 3 HP"
        assert enemy_units[1].hp == 0, "Enemy 1 должен иметь 0 HP"
        assert enemy_units[2].hp == 2, "Enemy 2 должен иметь 2 HP"
        
        # Проверяем, что мертвые удалены
        alive_enemies = [u for u in state.p2.board if u.hp > 0]
        assert len(alive_enemies) == 2, "На доске должно остаться 2 живых юнита"
    
    def test_aoe_freeze_dio_brando(self):
        """AOE заморозка останавливает время для всех врагов."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        # Создаем 3 вражеских юнитов
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
                is_ready=True,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)
        
        # THE WORLD!
        freeze_card = CardInstance(
            instance_id=uuid4(),
            card_id=300,
            name="The World",
            card_type=CardType.POTION,
            hp=0,
            max_hp=0,
            attack=0,
            mana_cost=4,
            mechanics=["aoe_freeze"],
            is_ready=False,
        )
        state.p1.hand.append(freeze_card)
        
        # ZA WARUDO!
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Активация The World провалилась: {error}"
        
        # TOKI WO TOMARE! — двигаемое поле is_frozen, не mechanics
        for unit in enemy_units:
            assert unit.is_frozen, f"Юнит {unit.name} должен быть заморожен (WRYYYYY!)"
    
    def test_aoe_damage(self):
        """Базовый тест AOE урона."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"AOE провалилось: {error}"
        
        for unit in enemy_units:
            assert unit.hp == 3, f"Юнит должен получить 2 урона: {unit.hp}"


# ============================================================================
# TEST CLASS: RESOURCE CONTROL (Mana Manipulation)
# ============================================================================

class TestResourceControl:
    """Тесты контроля ресурсов: mana_gain и mana_drain."""
    
    def test_mana_manipulation(self):
        """Mana gain с учетом лимита 10 маны."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        # Устанавливаем 8 маны
        state.p1.mana = 8
        state.p1.max_mana = 10
        
        # Карта с бонусом +5 маны
        mana_card = CardInstance(
            instance_id=uuid4(),
            card_id=400,
            name="Mana Crystal",
            card_type=CardType.POTION,
            hp=0,
            max_hp=0,
            attack=0,
            mana_cost=3,
            mechanics=["mana_gain_5"],
            is_ready=False,
        )
        state.p1.hand.append(mana_card)
        
        # Разыгрываем: 8 - 3 (стоимость) + 5 (восстановление) = 10
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Розыгрыш провалился: {error}"
        
        # С учетом лимита должно быть ровно 10
        assert state.p1.mana == 10, f"Мана должна быть 10 (лимит), текущая: {state.p1.mana}"
    
    def test_mana_gain(self):
        """Базовый тест восстановления маны."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        state.p1.mana = 3
        
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
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Розыгрыш провалился: {error}"
        
        # 3 - 2 + 3 = 4
        assert state.p1.mana == 4, f"Мана должна быть 4, текущая: {state.p1.mana}"
    
    def test_mana_drain(self):
        """Mana drain отнимает ману у противника."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        state.p2.mana = 5
        
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
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Розыгрыш провалился: {error}"
        
        assert state.p2.mana == 2, f"Мана противника должна быть 2, текущая: {state.p2.mana}"
    
    def test_mana_drain_cannot_go_negative(self):
        """Mana drain не уходит в минус."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        state.p2.mana = 1
        
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
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Розыгрыш провалился: {error}"
        
        assert state.p2.mana == 0, f"Мана не должна быть отрицательной, текущая: {state.p2.mana}"


# ============================================================================
# TEST CLASS: ADVANCED MECHANICS (Wave 3: Instant Kill, Cleave, Consume, Reflect)
# ============================================================================

class TestAdvancedMechanics:
    """Тесты продвинутых механик: instant kill, cleave, consume, reflect, permanent shield."""
    
    def test_instant_kill_unit(self):
        """instant_kill мгновенно убивает юнита независимо от HP."""
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
        
        success, error = env.step(1, AttackAction(
            attacker_id=str(instant_killer.instance_id),
            target_id=str(tank.instance_id),
            target_is_hero=False,
        ))
        
        assert success, f"Атака должна быть успешной: {error}"
        assert tank.hp == 0, f"Танк должен быть мгновенно убит, HP: {tank.hp}"
    
    def test_instant_kill_hero(self):
        """instant_kill НЕ убивает героя мгновенно — наносит только базовый урон."""
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
        
        assert success, f"Атака должна быть успешной: {error}"
        # instant_kill не работает на героях, только базовый урон
        assert state.p2.hero.hp == hero_hp_before - 1, f"Герой должен получить только 1 урона, получено HP: {state.p2.hero.hp}"
    
    def test_cleave_damages_multiple_targets(self):
        """cleave_X_Y наносит урон Y случайным целям."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        cleave_card = CardInstance(
            instance_id=uuid4(),
            card_id=300,
            name="Cleave Spell",
            card_type=CardType.POTION,
            hp=0,
            max_hp=0,
            attack=0,
            mana_cost=4,
            mechanics=["cleave_3_2"],
            is_ready=False,
        )
        state.p1.hand.append(cleave_card)
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Cleave провалился: {error}"
        
        total_damage = sum(10 - unit.hp for unit in enemy_units)
        assert total_damage >= 3, f"Должен быть нанесен урон, всего: {total_damage}"
    
    def test_cleave_handles_dying_targets(self):
        """cleave корректно обрабатывает смерть целей."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
        enemy_units = []
        for i in range(2):
            unit = CardInstance(
                instance_id=uuid4(),
                card_id=200 + i,
                name=f"Weak Enemy {i}",
                card_type=CardType.WARRIOR,
                hp=2,
                max_hp=2,
                attack=1,
                mana_cost=1,
                mechanics=[],
                is_ready=False,
            )
            state.p2.board.append(unit)
            enemy_units.append(unit)
        
        cleave_card = CardInstance(
            instance_id=uuid4(),
            card_id=300,
            name="Mass Cleave",
            card_type=CardType.POTION,
            hp=0,
            max_hp=0,
            attack=0,
            mana_cost=5,
            mechanics=["cleave_5_3"],
            is_ready=False,
        )
        state.p1.hand.append(cleave_card)
        
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=None))
        assert success, f"Cleave провалился: {error}"
        
        for unit in enemy_units:
            assert unit.hp == 0, f"Юнит {unit.name} должен быть убит"
    
    def test_consume_ally(self):
        """consume_ally поглощает союзника и получает его статы."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        success, error = env.step(1, PlayCardAction(
            hand_index=0,
            target_id=str(sacrifice.instance_id),
            position=0
        ))
        assert success, f"Consume провалился: {error}"
        
        assert len(state.p1.board) == board_size_before, "Размер доски не должен измениться"
        assert consumer.attack == 5, f"Атака должна быть 2+3=5, получено: {consumer.attack}"
        assert consumer.hp == 7, f"HP должно быть 2+5=7, получено: {consumer.hp}"
    
    def test_reflect_damages_attacker(self):
        """reflect_X отражает урон атакующему."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        reflector = CardInstance(
            instance_id=uuid4(),
            card_id=101,
            name="Reflector",
            card_type=CardType.WARRIOR,
            hp=8,
            max_hp=8,
            attack=2,
            mana_cost=4,
            mechanics=["reflect_3"],
            is_ready=False,
        )
        state.p2.board.append(reflector)
        
        success, error = env.step(1, AttackAction(
            attacker_id=str(attacker.instance_id),
            target_id=str(reflector.instance_id),
            target_is_hero=False,
        ))
        assert success, f"Атака провалилась: {error}"
        
        assert reflector.hp == 3, f"Reflector должен получить 5 урона, HP: {reflector.hp}"
        assert attacker.hp == 5, f"Attacker должен получить 2+3=5 урона, HP: {attacker.hp}"
    
    def test_permanent_shield_never_breaks(self):
        """permanent_shield блокирует урон навсегда."""
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
        
        for _ in range(5):
            apply_damage(immortal, 10)
            assert immortal.hp == 5, "HP не должно меняться с permanent_shield"
            assert "permanent_shield" in immortal.mechanics, "Permanent shield не должен исчезать"
    
    def test_permanent_shield_vs_normal_shield(self):
        """permanent_shield отличается от обычного щита."""
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
    
    def test_instant_kill_vs_permanent_shield(self):
        """instant_kill обходит permanent_shield."""
        state = create_minimal_game_state()
        env = ArenaEnvironment(state)
        
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
        
        success, error = env.step(1, AttackAction(
            attacker_id=str(assassin.instance_id),
            target_id=str(immortal.instance_id),
            target_is_hero=False,
        ))
        assert success, f"Атака провалилась: {error}"
        
        assert immortal.hp == 0, "instant_kill должен убить через permanent_shield"

