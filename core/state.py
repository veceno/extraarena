"""
Структуры данных игрового состояния для детерминированных симуляций.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional
from uuid import UUID, uuid4


# Размер кольцевого буфера истории действий. Соответствует прежнему
# лимиту в 100 записей, который раньше держался через `list[-100:]`
# (realloc O(n) на каждом append). deque(maxlen=N) делает это за O(1).
ACTION_HISTORY_MAXLEN = 100


# Фиксированный список всех механик для ML observation (34 механики)
MECHANICS_LIST = [
    "taunt",
    "shield",
    "permanent_shield",
    "freeze",
    "aoe_freeze",
    "battlecry_freeze",
    "instant_kill",
    "bypass_taunt",
    "consume_ally",
    "reflect",           # reflect_X -> общий флаг
    "armor",             # armor_X -> общий флаг
    "regen",             # regen_X -> общий флаг
    "aura_atk",          # aura_atk_X -> общий флаг
    "cleave",
    "delete_target",
    "cast_random_spell",
    "choose_shield_damage",
    "start_mana",
    "mana_drain",
    "mana_gain",
    "battlecry_damage",  # battlecry_damage_X -> общий флаг
    "battlecry_heal",    # battlecry_heal_X -> общий флаг
    "battlecry_draw",
    "battlecry_buff",
    "damage",            # для зелий damage_X
    "heal",              # для зелий heal_X
    "damage_all",        # AOE урон
    "heal_all",          # AOE лечение
    "buff_all",
    "summon",
    "deathrattle",
    "charge",
    "lifesteal",
    "desk_freeze",
]


class CardType(Enum):
    HERO = "hero"
    WARRIOR = "warrior"
    POTION = "potion"


class GameStatus(Enum):
    ONGOING = "ongoing"
    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    DRAW = "draw"


class ReplacementStatus(Enum):
    """Статус замены игрока ботом."""
    ACTIVE = "active"          # Игрок активен
    AFK = "afk"                # Игрок пропал (2+ таймаута)
    SURRENDERED = "surrendered"  # Игрок сдался


@dataclass
class CardInstance:
    instance_id: UUID = field(default_factory=uuid4)
    card_id: int = 0
    name: str = ""
    card_type: CardType = CardType.WARRIOR
    rarity: str = "common"
    mana_cost: int = 0
    attack: int = 0
    hp: int = 0
    max_hp: int = 0
    mechanics: List[str] = field(default_factory=list)
    is_ready: bool = False
    is_frozen: bool = False
    description: str = ""
    mechanics_desc: str = ""
    level: int = 1  # Уровень карты (1-10)
    simplified_levelup: bool = False
    base_attack: Optional[int] = None
    base_hp: Optional[int] = None
    base_max_hp: Optional[int] = None
    base_mana_cost: Optional[int] = None
    base_mechanics: Optional[List[str]] = None
    instant_kill_used: bool = False
    skip_count: int = 0

    @property
    def is_asleep(self) -> bool:
        """Существо спит (не готово к атаке) если is_ready=False."""
        return not self.is_ready

    def ensure_base_snapshot(self) -> None:
        """Запомнить исходное состояние карты до runtime-модификаций."""
        if self.base_attack is not None:
            return
        self.base_attack = int(self.attack)
        self.base_max_hp = int(self.max_hp)
        self.base_hp = int(self.max_hp if self.max_hp else self.hp)
        self.base_mana_cost = int(self.mana_cost)
        self.base_mechanics = list(self.mechanics)

    def reset_to_base_state(self) -> None:
        """Восстановить карту перед возвратом из сброса в колоду."""
        self.ensure_base_snapshot()
        self.attack = int(self.base_attack or 0)
        self.max_hp = int(self.base_max_hp or 0)
        self.hp = int(self.base_hp if self.base_hp is not None else self.max_hp)
        self.mana_cost = int(self.base_mana_cost or 0)
        self.mechanics = list(self.base_mechanics or [])
        self.is_ready = False
        self.is_frozen = False
        self.instant_kill_used = False
        self.skip_count = 0


@dataclass
class PlayerState:
    user_id: int
    is_bot: bool = False
    replacement_status: "ReplacementStatus" = None  # Статус замены (ACTIVE/AFK/SURRENDERED)
    hero: CardInstance = field(default_factory=CardInstance)
    mana: int = 0
    max_mana: int = 0
    hand: List[CardInstance] = field(default_factory=list)
    board: List[CardInstance] = field(default_factory=list)
    deck: List[CardInstance] = field(default_factory=list)
    graveyard: List[CardInstance] = field(default_factory=list)  # Сброс (кладбище)
    trophies: int = 0
    surrender_processed: bool = False  # Флаг немедленного списания трофеев при сдаче
    
    def __post_init__(self):
        """Инициализация дефолтного статуса."""
        if self.replacement_status is None:
            from core.state import ReplacementStatus
            self.replacement_status = ReplacementStatus.ACTIVE


@dataclass
class GameState:
    p1: PlayerState
    p2: PlayerState
    current_turn_owner_id: int
    turn_number: int = 1
    history: List[Dict] = field(default_factory=list)
    # Опциональные ссылки на параметры режима и движок, чтобы эффекты
    # в core.effects могли учитывать правила арены (overdraw_to_discard
    # и т.п.) без жёсткой связки с ArenaEnvironment. Заполняется
    # BattleEngine после конструирования GameState; None допустим —
    # означает "classic defaults".
    classic_params: Optional["ClassicParams"] = None
    arena_engine: Optional["ArenaEnvironment"] = None
    # Кольцевой буфер последних действий. deque(maxlen=N) автоматически
    # вытесняет старейшие записи при append — никаких ручных `list[-100:]`
    # realloc на каждом `step`. Совместим с прежним API: поддерживает
    # `append`, индексацию `[-1]`, `[-100:]`, `len()`, итерацию и `list()`.
    action_history: Deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=ACTION_HISTORY_MAXLEN)
    )
    status: GameStatus = GameStatus.ONGOING
    sudden_death_turns_by_player: Dict[int, int] = field(default_factory=dict)
    sudden_death_last_applied_turn_by_player: Dict[int, int] = field(default_factory=dict)
    pending_mana_drain_by_player: Dict[int, int] = field(default_factory=dict)
    pending_card_feedback_events: List[Dict] = field(default_factory=list)

    def _card_features(self, prefix: str, card: CardInstance) -> Dict[str, int]:
        """Извлечь фичи карты включая вектор механик."""
        features = {
            f"{prefix}_attack": card.attack,
            f"{prefix}_hp": card.hp,
            f"{prefix}_max_hp": card.max_hp,
            f"{prefix}_mana_cost": card.mana_cost,
            f"{prefix}_is_ready": int(card.is_ready),
        }
        
        # Добавляем бинарный вектор механик
        for mechanic in MECHANICS_LIST:
            has_mechanic = 0
            for card_mechanic in card.mechanics:
                # Проверяем точное совпадение или префикс (для parametric механик)
                if card_mechanic == mechanic or card_mechanic.startswith(mechanic + "_"):
                    has_mechanic = 1
                    break
            features[f"{prefix}_mech_{mechanic}"] = has_mechanic
        
        return features

    def _player_features(self, player: PlayerState, label: str) -> Dict[str, int]:
        data: Dict[str, int] = {
            f"{label}_mana": player.mana,
            f"{label}_max_mana": player.max_mana,
            f"{label}_trophies": player.trophies,
            f"{label}_hand_size": len(player.hand),
            f"{label}_deck_size": len(player.deck),
        }

        data.update(self._card_features(f"{label}_hero", player.hero))

        for idx, unit in enumerate(player.board):
            data.update(self._card_features(f"{label}_board_{idx}", unit))

        return data

    def get_observation(self) -> Dict[str, int]:
        """
        Плоское числовое представление состояния для RL.
        Герой и существа на столе идут в фиксированном порядке по игрокам.
        """
        obs: Dict[str, int] = {
            "turn_number": self.turn_number,
            "current_turn_owner_id": self.current_turn_owner_id,
            "status": {
                GameStatus.ONGOING: 0,
                GameStatus.P1_WIN: 1,
                GameStatus.P2_WIN: 2,
                GameStatus.DRAW: 3,
            }[self.status],
        }
        obs.update(self._player_features(self.p1, "p1"))
        obs.update(self._player_features(self.p2, "p2"))
        return obs
