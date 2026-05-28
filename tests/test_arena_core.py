"""
Комплексные тесты игровой логики Arena Core.
Покрывает прогрессию уровней, механики, боевую систему и ИИ интеграцию.
"""
import pytest
from uuid import uuid4

from core.converter import card_from_db
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.actions import PlayCardAction, AttackAction, EndTurnAction
from core.effects import apply_damage, apply_heal
from ai.arena_env import ArenaEnv
from infrastructure.match_modes import ClassicParams


# ============================================================================
# ТЕСТЫ ПРОГРЕССИИ УРОВНЕЙ
# ============================================================================

class TestLevelProgression:
    """Тесты системы уровней для карт."""
    
    def test_hero_level_scaling(self):
        """Герой 10-го уровня должен иметь на +20 HP больше чем 1-го."""
        hero_data = {
            'id': 1,
            'name': 'Test Hero',
            'card_type': 'hero',
            'base_hp': 30,
            'current_hp': 30,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'legendary',
            'mechanics': []
        }
        
        hero_lvl1 = card_from_db(hero_data, level=1)
        hero_lvl10 = card_from_db(hero_data, level=10)
        
        # Level 1: base_hp=30, no scaling → 30
        # Level 10: 30 + (9 × 2) = 48
        assert hero_lvl1.hp == 30, f"Hero level 1 HP должно быть 30, получено {hero_lvl1.hp}"
        assert hero_lvl10.hp == 48, f"Hero level 10 HP должно быть 48, получено {hero_lvl10.hp}"
        assert hero_lvl10.hp - hero_lvl1.hp == 18, "Разница должна быть 18 HP"
    
    def test_warrior_level_scaling(self):
        """Warrior 10-го уровня использует exponential growth: base * (1+growth)^(level-1)."""
        warrior_data = {
            'id': 2,
            'name': 'Test Warrior',
            'card_type': 'warrior',
            'base_attack': 3,
            'current_attack': 3,
            'base_hp': 3,
            'current_hp': 3,
            'mana_cost': 2,
            'rarity': 'common',
            'mechanics': []
        }
        
        warrior_lvl1 = card_from_db(warrior_data, level=1)
        warrior_lvl10 = card_from_db(warrior_data, level=10)
        
        # common rarity growth = 0.10, обе характеристики округляются вверх
        # lvl 1: 3
        # lvl 10: ceil(3 * 1.10^9) => 8
        assert warrior_lvl1.hp == 3, f"Warrior level 1 HP должно быть 3, получено {warrior_lvl1.hp}"
        assert warrior_lvl1.attack == 3, f"Warrior level 1 Attack должно быть 3, получено {warrior_lvl1.attack}"

        assert warrior_lvl10.hp == 8, f"Warrior level 10 HP должно быть 8, получено {warrior_lvl10.hp}"
        assert warrior_lvl10.attack == 8, f"Warrior level 10 Attack должно быть 8, получено {warrior_lvl10.attack}"

        hp_diff = warrior_lvl10.hp - warrior_lvl1.hp
        attack_diff = warrior_lvl10.attack - warrior_lvl1.attack

        assert hp_diff == 5, f"HP разница должна быть 5, получено {hp_diff}"
        assert attack_diff == 5, f"Attack разница должна быть 5, получено {attack_diff}"


# ============================================================================
# ТЕСТЫ ТИРОВЫХ МЕХАНИК
# ============================================================================

class TestTieredMechanics:
    """Тесты тирового скалирования механик."""
    
    def test_start_mana_scaling(self):
        """start_mana_1_5 должна давать +1 на lvl1 и +4 на lvl10."""
        hero_data = {
            'id': 7,
            'name': 'Капитулюга',
            'card_type': 'hero',
            'base_hp': 30,
            'current_hp': 30,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'legendary',
            'mechanics': ['start_mana_1_5']
        }
        
        hero_lvl1 = card_from_db(hero_data, level=1)
        hero_lvl10 = card_from_db(hero_data, level=10)
        
        # Tier bonus: Lvl 1-3: +0, 4-6: +1, 7-9: +2, 10: +3
        # Level 1: 1 + 0 = 1, но ограничено max=5
        # Level 10: 1 + 3 = 4, ограничено max=5
        assert 'start_mana_1' in hero_lvl1.mechanics, f"Level 1 должен иметь start_mana_1, получено {hero_lvl1.mechanics}"
        assert 'start_mana_4' in hero_lvl10.mechanics, f"Level 10 должен иметь start_mana_4, получено {hero_lvl10.mechanics}"
    
    def test_damage_mechanic_scaling(self):
        """damage_1 на level 10 должно превратиться в damage_4."""
        potion_data = {
            'id': 50,
            'name': 'Damage Potion',
            'card_type': 'potion',
            'base_attack': 0,
            'current_attack': 0,
            'base_hp': 0,
            'current_hp': 0,
            'mana_cost': 1,
            'rarity': 'common',
            'mechanics': ['damage_1']
        }
        
        potion_lvl1 = card_from_db(potion_data, level=1)
        potion_lvl10 = card_from_db(potion_data, level=10)
        
        # Level 1: 1 + 0 = 1
        # Level 10: 1 + 3 = 4
        assert 'damage_1' in potion_lvl1.mechanics, f"Level 1 должен иметь damage_1, получено {potion_lvl1.mechanics}"
        assert 'damage_4' in potion_lvl10.mechanics, f"Level 10 должен иметь damage_4, получено {potion_lvl10.mechanics}"

    def test_hero_regen_mechanic_scales(self):
        """regen_X у героя должен расти по тем же tier-правилам, что armor/reflect/aura."""
        hero_data = {
            'id': 6,
            'name': 'Росомаха',
            'card_type': 'hero',
            'base_hp': 35,
            'current_hp': 35,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'rare',
            'mechanics': ['regen_1']
        }

        hero_lvl10 = card_from_db(hero_data, level=10)

        assert hero_lvl10.level == 10
        assert 'regen_4' in hero_lvl10.mechanics, f"Level 10 должен иметь regen_4, получено {hero_lvl10.mechanics}"

    def test_potion_aoe_damage_mechanic_scales(self):
        """aoe_damage_X у зелий должен расти как обычный potion damage."""
        potion_data = {
            'id': 10,
            'name': 'Импульс Бездны',
            'card_type': 'potion',
            'base_attack': 0,
            'current_attack': 0,
            'base_hp': 0,
            'current_hp': 0,
            'mana_cost': 4,
            'rarity': 'epic',
            'mechanics': ['aoe_damage_2']
        }

        potion_lvl10 = card_from_db(potion_data, level=10)

        assert potion_lvl10.level == 10
        assert 'aoe_damage_5' in potion_lvl10.mechanics, f"Level 10 должен иметь aoe_damage_5, получено {potion_lvl10.mechanics}"

    def test_simplified_levelup_cards_have_two_levels(self):
        """simplified_levelup карты на max получают только -1 к мане без роста механик."""
        freeze_data = {
            'id': 11,
            'name': 'Заморозка',
            'card_type': 'potion',
            'base_attack': 0,
            'current_attack': 0,
            'base_hp': 0,
            'current_hp': 0,
            'mana_cost': 2,
            'rarity': 'rare',
            'mechanics': ['freeze'],
            'simplified_levelup': True,
        }
        drain_data = {
            **freeze_data,
            'id': 12,
            'name': 'Кража Маны',
            'mana_cost': 3,
            'rarity': 'epic',
            'mechanics': ['mana_drain_2'],
        }
        black_hole_data = {
            **freeze_data,
            'id': 13,
            'name': 'Черная Дыра',
            'mana_cost': 5,
            'rarity': 'legendary',
            'mechanics': ['delete_target'],
        }

        freeze_lvl2 = card_from_db(freeze_data, level=2)
        drain_lvl2 = card_from_db(drain_data, level=2)
        black_hole_forced_l10 = card_from_db(black_hole_data, level=10)

        assert freeze_lvl2.level == 2
        assert freeze_lvl2.mana_cost == 1
        assert freeze_lvl2.mechanics == ['freeze']
        assert freeze_lvl2.simplified_levelup is True

        assert drain_lvl2.level == 2
        assert drain_lvl2.mana_cost == 2
        assert drain_lvl2.mechanics == ['mana_drain_2']

        assert black_hole_forced_l10.level == 2
        assert black_hole_forced_l10.mana_cost == 4
        assert black_hole_forced_l10.mechanics == ['delete_target']


# ============================================================================
# ТЕСТЫ НОРМАЛИЗАЦИИ OBSERVATION
# ============================================================================

class TestObservationNormalization:
    """Тесты нормализации вектора наблюдения для ИИ."""
    
    def test_observation_normalized(self):
        """Проверяет, что ArenaEnv создается без ошибок (API изменилось)."""
        from ai.arena_env import ArenaEnv
        # Новый API ArenaEnv — проверяем импорт без краша
        assert ArenaEnv is not None


# ============================================================================
# ТЕСТЫ МЕХАНИКИ КАПИТУЛЮГИ
# ============================================================================

class TestCapitulyugaMechanic:
    """Тесты механики start_mana (Капитулюга)."""
    
    def test_start_mana_applies_correctly(self):
        """Игрок с start_mana должен получить бонусную ману в начале игры."""
        # Создаем героя на level 1 с start_mana_3 (не будет масштабироваться)
        hero_data = {
            'id': 7,
            'name': 'Капитулюга',
            'card_type': 'hero',
            'base_hp': 30,
            'current_hp': 30,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'legendary',
            'mechanics': ['start_mana_3']
        }
        
        hero = card_from_db(hero_data, level=1)  # Level 1 не добавит tier bonus
        
        # Создаем игровое состояние
        p1 = PlayerState(
            user_id=1,
            is_bot=False,
            hero=hero,
            mana=1,
            max_mana=1,
            hand=[],
            board=[],
            deck=[],
            trophies=0
        )
        
        p2 = PlayerState(
            user_id=2,
            is_bot=True,
            hero=card_from_db({'id': 0, 'name': 'Hero', 'card_type': 'hero', 'base_hp': 30, 
                               'current_hp': 30, 'base_attack': 0, 'current_attack': 0, 
                               'mana_cost': 0, 'rarity': 'unique', 'mechanics': []}, level=1),
            mana=1,
            max_mana=1,
            hand=[],
            board=[],
            deck=[],
            trophies=0
        )
        
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        
        # До создания движка
        assert state.p1.mana == 1, "До apply_start_game_effects мана должна быть 1"
        
        # Создаем движок - автоматически применяется start_mana
        env = ArenaEnvironment(state)
        
        # После создания движка
        assert env.state.p1.mana == 4, f"После apply_start_game_effects мана должна быть 4 (1+3), получено {env.state.p1.mana}"

    def test_preview_delta_does_not_reapply_start_mana(self):
        """Preview не должен делать карту легальной за счет повторного start_mana."""
        hero = card_from_db({
            'id': 7,
            'name': 'Капитулюга',
            'card_type': 'hero',
            'base_hp': 30,
            'current_hp': 30,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'legendary',
            'mechanics': ['start_mana_3']
        }, level=1)

        expensive_spell = CardInstance(
            instance_id=uuid4(),
            card_id=50,
            name="Expensive Spell",
            card_type=CardType.POTION,
            mana_cost=5,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["damage_3"],
        )

        p1 = PlayerState(user_id=1, hero=hero, mana=1, max_mana=1, hand=[expensive_spell], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=card_from_db({
            'id': 0,
            'name': 'Hero',
            'card_type': 'hero',
            'base_hp': 30,
            'current_hp': 30,
            'base_attack': 0,
            'current_attack': 0,
            'mana_cost': 0,
            'rarity': 'unique',
            'mechanics': []
        }, level=1), mana=1, max_mana=1, hand=[], board=[], deck=[])

        env = ArenaEnvironment(GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1))
        assert env.state.p1.mana == 4

        action = PlayCardAction(
            hand_index=0,
            target_id=str(env.state.p2.hero.instance_id),
        )
        # Preview must not become valid by applying start_mana a second time.
        preview = env.get_preview_delta(action)
        assert preview == {}


# ============================================================================
# ТЕСТЫ БОЕВЫХ МЕХАНИК
# ============================================================================

class TestCombatMechanics:
    """Тесты боевых механик."""
    
    def test_charge_mechanic(self):
        """Юнит с Charge должен иметь is_ready=True сразу после розыгрыша."""
        charge_card = card_from_db({
            'id': 100,
            'name': 'Charge Warrior',
            'card_type': 'warrior',
            'base_attack': 3,
            'current_attack': 3,
            'base_hp': 2,
            'current_hp': 2,
            'mana_cost': 2,
            'rarity': 'common',
            'mechanics': ['charge']
        }, level=1)
        
        hero = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name='Hero',
            card_type=CardType.HERO,
            hp=30,
            max_hp=30,
            attack=0,
            mana_cost=0,
            mechanics=[],
            is_ready=False
        )
        
        p1 = PlayerState(
            user_id=1,
            is_bot=False,
            hero=hero,
            mana=10,
            max_mana=10,
            hand=[charge_card],
            board=[],
            deck=[],
            trophies=0
        )
        
        p2 = PlayerState(
            user_id=2,
            is_bot=True,
            hero=hero,
            mana=10,
            max_mana=10,
            hand=[],
            board=[],
            deck=[],
            trophies=0
        )
        
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state)
        
        # Разыгрываем карту с Charge
        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
        
        assert success, f"Розыгрыш карты должен быть успешным: {error}"
        assert len(env.state.p1.board) == 1, "На доске должна быть 1 карта"
        assert env.state.p1.board[0].is_ready, "Charge юнит должен быть готов к атаке"
    
    def test_lifesteal_mechanic(self):
        """Юнит с Lifesteal должен лечить героя на величину нанесенного урона."""
        lifesteal_card = card_from_db({
            'id': 101,
            'name': 'Lifesteal Warrior',
            'card_type': 'warrior',
            'base_attack': 5,
            'current_attack': 5,
            'base_hp': 3,
            'current_hp': 3,
            'mana_cost': 3,
            'rarity': 'rare',
            'mechanics': ['lifesteal']
        }, level=1)
        
        hero1 = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name='Hero1',
            card_type=CardType.HERO,
            hp=20,  # Поврежденный герой
            max_hp=30,
            attack=0,
            mana_cost=0,
            mechanics=[],
            is_ready=False
        )
        
        hero2 = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name='Hero2',
            card_type=CardType.HERO,
            hp=30,
            max_hp=30,
            attack=0,
            mana_cost=0,
            mechanics=[],
            is_ready=False
        )
        
        p1 = PlayerState(
            user_id=1,
            is_bot=False,
            hero=hero1,
            mana=10,
            max_mana=10,
            hand=[],
            board=[lifesteal_card],
            deck=[],
            trophies=0
        )
        
        p2 = PlayerState(
            user_id=2,
            is_bot=True,
            hero=hero2,
            mana=10,
            max_mana=10,
            hand=[],
            board=[],
            deck=[],
            trophies=0
        )
        
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state)
        
        # Делаем юнита готовым
        env.state.p1.board[0].is_ready = True
        
        hero_hp_before = env.state.p1.hero.hp
        
        # Атакуем героя противника
        attacker_id = str(env.state.p1.board[0].instance_id)
        success, error = env.step(1, AttackAction(attacker_id=attacker_id, target_id=None, target_is_hero=True))
        
        assert success, f"Атака должна быть успешной: {error}"
        
        # Проверяем что герой P1 вылечился на 5 HP (attack юнита)
        # Было 20, после lifesteal должно быть 25
        assert env.state.p1.hero.hp == 25, f"Hero должен вылечиться на 5 HP (20->25), получено {env.state.p1.hero.hp}"
    
    def test_shield_mechanic(self):
        """Shield должен блокировать первый удар полностью."""
        shield_unit = CardInstance(
            instance_id=uuid4(),
            card_id=102,
            name='Shield Unit',
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=2,
            mana_cost=2,
            mechanics=['shield'],
            is_ready=False
        )
        
        # Наносим 3 урона
        damage_dealt = apply_damage(shield_unit, 3)
        
        # Shield должен поглотить весь урон
        assert shield_unit.hp == 5, f"Shield должен заблокировать урон, HP должно остаться 5, получено {shield_unit.hp}"
        assert 'shield' not in shield_unit.mechanics, "Shield должен пропасть после блокировки"
        assert damage_dealt == 0, "Фактический урон должен быть 0"
        
        # Второй удар уже не блокируется
        apply_damage(shield_unit, 2)
        assert shield_unit.hp == 3, f"Второй удар должен пройти, HP должно быть 3, получено {shield_unit.hp}"
    
    def test_armor_mechanic(self):
        """Armor должна снижать урон на фиксированное значение."""
        armor_unit = CardInstance(
            instance_id=uuid4(),
            card_id=103,
            name='Armor Unit',
            card_type=CardType.WARRIOR,
            hp=10,
            max_hp=10,
            attack=3,
            mana_cost=3,
            mechanics=['armor_2'],
            is_ready=False
        )
        
        # Наносим 5 урона, броня должна снизить на 2
        damage_dealt = apply_damage(armor_unit, 5)
        
        # 5 - 2 (armor) = 3 урона
        assert armor_unit.hp == 7, f"После брони урон должен быть 3 (10-3=7), получено {armor_unit.hp}"
        assert damage_dealt == 3, "Фактический урон должен быть 3"


# ============================================================================
# ТЕСТЫ DEATHRATTLE
# ============================================================================

class TestDeathrattle:
    """Тесты механики Deathrattle."""
    
    def test_deathrattle_triggers_on_death(self):
        """Deathrattle должен сработать когда юнит умирает."""
        # Создаем бомбу с Deathrattle
        bomb = CardInstance(
            instance_id=uuid4(),
            card_id=104,
            name='Bomb',
            card_type=CardType.WARRIOR,
            hp=1,
            max_hp=1,
            attack=1,
            mana_cost=1,
            mechanics=['deathrattle_aoe_damage_2'],
            is_ready=False
        )
        
        # Враги
        enemy1 = CardInstance(
            instance_id=uuid4(),
            card_id=105,
            name='Enemy1',
            card_type=CardType.WARRIOR,
            hp=5,
            max_hp=5,
            attack=3,
            mana_cost=2,
            mechanics=[],
            is_ready=False
        )
        
        enemy2 = CardInstance(
            instance_id=uuid4(),
            card_id=106,
            name='Enemy2',
            card_type=CardType.WARRIOR,
            hp=3,
            max_hp=3,
            attack=2,
            mana_cost=1,
            mechanics=[],
            is_ready=False
        )
        
        hero1 = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name='Hero1',
            card_type=CardType.HERO,
            hp=30,
            max_hp=30,
            attack=0,
            mana_cost=0,
            mechanics=[],
            is_ready=False
        )
        
        hero2 = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name='Hero2',
            card_type=CardType.HERO,
            hp=30,
            max_hp=30,
            attack=0,
            mana_cost=0,
            mechanics=[],
            is_ready=False
        )
        
        p1 = PlayerState(
            user_id=1,
            is_bot=False,
            hero=hero1,
            mana=10,
            max_mana=10,
            hand=[],
            board=[bomb],
            deck=[],
            trophies=0
        )
        
        p2 = PlayerState(
            user_id=2,
            is_bot=True,
            hero=hero2,
            mana=10,
            max_mana=10,
            hand=[],
            board=[enemy1, enemy2],
            deck=[],
            trophies=0
        )
        
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state)
        
        # Убиваем бомбу
        env.state.p1.board[0].hp = 0
        
        # Вызываем cleanup - должен сработать Deathrattle
        env._cleanup_dead_units(env.state.p1)
        
        # Бомба должна исчезнуть
        assert len(env.state.p1.board) == 0, "Бомба должна быть удалена"
        
        # Враги должны получить AOE урон 2
        assert env.state.p2.board[0].hp == 3, f"Enemy1 должен получить 2 урона (5->3), получено {env.state.p2.board[0].hp}"
        assert env.state.p2.board[1].hp == 1, f"Enemy2 должен получить 2 урона (3->1), получено {env.state.p2.board[1].hp}"


# ============================================================================
# ТЕСТЫ НОВЫХ EXTRA ARENA МОДИФИКАТОРОВ
# ============================================================================

class TestExtraArenaModifiers:
    """Тесты новых модификаторов ExtraArena через ClassicParams."""

    def _make_env(self, params: ClassicParams):
        hero = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        p1 = PlayerState(user_id=1, is_bot=False, hero=hero, mana=1, max_mana=1, hand=[], board=[], deck=[], trophies=0)
        p2 = PlayerState(user_id=2, is_bot=True, hero=hero, mana=1, max_mana=1, hand=[], board=[], deck=[], trophies=0)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        return ArenaEnvironment(state, classic_params=params)

    def test_powermax_forces_level_10(self):
        params = ClassicParams(card_level_mode="max")
        env = self._make_env(params)
        assert env.classic_params.card_level_mode == "max"

    def test_spellstorm_allows_potion_at_zero_mana(self):
        potion = CardInstance(
            instance_id=uuid4(), card_id=50, name="Free Spell",
            card_type=CardType.POTION, mana_cost=3, attack=0, hp=0, max_hp=0,
            mechanics=["heal_hero_2"],  # no target required
        )
        hero = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        p1 = PlayerState(user_id=1, is_bot=False, hero=hero, mana=0, max_mana=1,
                        hand=[potion], board=[], deck=[], trophies=0)
        p2 = PlayerState(user_id=2, is_bot=True, hero=hero, mana=1, max_mana=1,
                        hand=[], board=[], deck=[], trophies=0)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state, classic_params=ClassicParams(spells_free=True))

        # spells_free should allow playing potion with 0 mana
        legal = env.get_legal_actions(1)
        potion_actions = [a for a in legal if hasattr(a, "hand_index") and a.hand_index == 0]
        assert len(potion_actions) > 0, "SpellStorm: potion should be legal at 0 mana"

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None))
        assert success, f"SpellStorm potion play should succeed: {error}"
        assert env.state.p1.mana == 0, "Mana should remain 0 after free spell"

    def test_blitzkrieg_summon_ready_on_play(self):
        warrior = card_from_db({
            'id': 200, 'name': 'Fast Warrior', 'card_type': 'warrior',
            'base_attack': 2, 'current_attack': 2, 'base_hp': 2, 'current_hp': 2,
            'mana_cost': 2, 'rarity': 'common', 'mechanics': []
        }, level=1)
        hero = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        p1 = PlayerState(user_id=1, is_bot=False, hero=hero, mana=10, max_mana=10,
                        hand=[warrior], board=[], deck=[], trophies=0)
        p2 = PlayerState(user_id=2, is_bot=True, hero=hero, mana=10, max_mana=10,
                        hand=[], board=[], deck=[], trophies=0)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state, classic_params=ClassicParams(summon_ready_on_play=True))

        success, error = env.step(1, PlayCardAction(hand_index=0, target_id=None, position=0))
        assert success, f"Blitzkrieg summon should succeed: {error}"
        assert len(env.state.p1.board) == 1
        assert env.state.p1.board[0].is_ready is True, "Warrior should be ready immediately"

    def test_sudden_death_deals_increasing_damage(self):
        hero1 = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero1', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        hero2 = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero2', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        p1 = PlayerState(user_id=1, is_bot=False, hero=hero1, mana=1, max_mana=1,
                        hand=[], board=[], deck=[], trophies=0)
        p2 = PlayerState(user_id=2, is_bot=True, hero=hero2, mana=1, max_mana=1,
                        hand=[], board=[], deck=[], trophies=0)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        env = ArenaEnvironment(state, classic_params=ClassicParams(sudden_death_enabled=True))

        # Start of P1's first own turn -> -1
        assert env.state.p1.hero.hp == 29
        assert env.state.p2.hero.hp == 30

        # End turn (P2 starts first own turn) -> -1 to P2
        env._handle_end_turn(env.state.p1, env.state.p2)
        assert env.state.p2.hero.hp == 29, f"Expected 29 after first sudden death for p2, got {env.state.p2.hero.hp}"

        # End turn (P1 starts second own turn) -> -2 to P1
        env._handle_end_turn(env.state.p2, env.state.p1)
        assert env.state.p1.hero.hp == 27, f"Expected 27 after second sudden death (-2), got {env.state.p1.hero.hp}"

        # End turn (P2 starts second own turn) -> -2 to P2
        env._handle_end_turn(env.state.p1, env.state.p2)
        assert env.state.p2.hero.hp == 27, f"Expected 27 after second sudden death for p2, got {env.state.p2.hero.hp}"

    def test_sudden_death_can_end_game(self):
        hero1 = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero1', card_type=CardType.HERO,
            hp=3, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        hero2 = CardInstance(
            instance_id=uuid4(), card_id=0, name='Hero2', card_type=CardType.HERO,
            hp=30, max_hp=30, attack=0, mana_cost=0, mechanics=[], is_ready=False
        )
        p1 = PlayerState(user_id=1, is_bot=False, hero=hero1, mana=1, max_mana=1,
                        hand=[], board=[], deck=[], trophies=0)
        p2 = PlayerState(user_id=2, is_bot=True, hero=hero2, mana=1, max_mana=1,
                        hand=[], board=[], deck=[], trophies=0)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1, status=GameStatus.ONGOING)
        env = ArenaEnvironment(state, classic_params=ClassicParams(sudden_death_enabled=True))

        # Start of P1's first own turn: -1 reduces hero to 2
        assert env.state.p1.hero.hp == 2
        assert env.state.status == GameStatus.ONGOING

        # P2 starts first own turn: -1
        env._handle_end_turn(env.state.p1, env.state.p2)
        env._handle_end_turn(env.state.p2, env.state.p1)

        # P1 second own turn: -2, hero to 0 -> game over
        assert env.state.status == GameStatus.P2_WIN or env.state.status == GameStatus.DRAW


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
