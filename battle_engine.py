"""
Класс-адаптер BattleEngine для совместимости со старым кодом.
Связывает старый API с новой архитектурой core/.
"""

from __future__ import annotations
from typing import Any, Optional, Dict, List
from uuid import UUID
import logging
import time

from core.actions import AttackAction, BaseAction, EndTurnAction, PlayCardAction
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState as CorePlayerState


# Глобальный словарь активных матчей
ACTIVE_MATCHES: Dict[str, "BattleEngine"] = {}


class BattleEventEmitter:
    """Система событий для рассылки обновлений клиентам."""
    
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Any]] = {}
    
    def on(self, event_type: str, callback: Any) -> None:
        """Регистрирует обработчик события."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def emit(self, event_type: str, match_id: str, data: dict[str, Any]) -> None:
        """Генерирует событие."""
        if event_type not in self._listeners:
            return
        for callback in self._listeners[event_type]:
            try:
                callback(match_id, data)
            except Exception as exc:
                logging.getLogger(__name__).warning("Event callback failed: %s", exc)


class BattleEngine:
    """
    Адаптер между старым API и core/engine.ArenaEnvironment.
    Управляет жизненным циклом матча и сериализацией состояния для фронтенда.
    """
    
    def __init__(
        self,
        db: Any = None,
        match_id: Any = None,
        player_ids: Optional[list[int]] = None,
        is_bot_match: bool = False,
        p1_deck_ids: Optional[List[str]] = None,
        p2_deck_ids: Optional[List[str]] = None,
        active_matches: Optional[Dict[str, "BattleEngine"]] = None,
        p1_hero_hp_override: Optional[int] = None,
        p2_hero_hp_override: Optional[int] = None,
        card_cache: Optional[Dict[str, Any]] = None,
        event_emitter: Optional[BattleEventEmitter] = None,
        p1_name: Optional[str] = None,
        p1_avatar_url: Optional[str] = None,
        p2_name: Optional[str] = None,
        p2_avatar_url: Optional[str] = None,
        game_mode: str = "classic",
    ) -> None:
        """Инициализация боевого движка."""
        self._db = db
        self.match_id = match_id
        self.player_ids = player_ids or []
        self.is_bot_match = is_bot_match
        self.bot_id: Optional[int] = None
        self.bot_difficulty: str = "medium"  # Сложность бота (lite/easy/medium/hard/max)
        self._active_matches = active_matches if active_matches is not None else ACTIVE_MATCHES
        self._event_emitter = event_emitter
        self.current_player_id: Optional[int] = None
        self.turn = 0
        self.is_ended = False
        self.game_over_processed = False
        self.rewards_granted = False
        self.turn_start_time: Optional[float] = None
        self.match_start_time: Optional[float] = None
        self.game_mode = game_mode
        self._is_blitz = (game_mode == "extra_arena:blitz")
        self.turn_duration = 5 if self._is_blitz else 25
        self.client_ready = False
        self._card_cache: Dict[str, Any] = card_cache or {}
        self._logger = logging.getLogger(__name__)
        
        # Счётчики таймаутов для AFK-детекции
        self.p1_consecutive_timeouts: int = 0
        self.p2_consecutive_timeouts: int = 0
        
        # Ядро боя — ArenaEnvironment из core/
        self._arena: Optional[ArenaEnvironment] = None
        
        # Метаданные игроков для UI
        self._p1_name = p1_name or "Игрок 1"
        self._p1_avatar_url = p1_avatar_url
        self._p2_name = p2_name or "Игрок 2"
        self._p2_avatar_url = p2_avatar_url
        self._p1_trophies: int = 0
        self._p2_trophies: int = 0
        self._p1_clan: str = ""
        self._p2_clan: str = ""
        
        # Для совместимости со старым кодом (минимальные заглушки)
        p1_id = player_ids[0] if player_ids and len(player_ids) > 0 else 1
        p2_id = player_ids[1] if player_ids and len(player_ids) > 1 else 2
        self._p1_id = p1_id
        self._p2_id = p2_id

        # Analytics data collection
        self._analytics_actions: list[dict[str, Any]] = []
        self._analytics_flushed: bool = False
        self._p1_initial_deck_ids: list[int] = []
        self._p2_initial_deck_ids: list[int] = []
    
    # =========================================================================
    # СОЗДАНИЕ МАТЧА
    # =========================================================================
    
    async def create_match(
        self,
        match_id: str,
        p1_data: Dict[str, Any],
        p2_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Создать матч с использованием core/engine.
        
        Args:
            match_id: ID матча
            p1_data: {user_id, deck_ids, name, avatar_url, is_bot, trophies}
            p2_data: {user_id, deck_ids, name, avatar_url, is_bot, trophies}
        """
        try:
            from core.converter import deck_from_card_ids
            
            self._logger.info(
                "create_match: match_id=%s p1_id=%s p2_id=%s",
                match_id, p1_data.get("user_id"), p2_data.get("user_id")
            )
            
            self.match_id = match_id
            self.player_ids = [p1_data["user_id"], p2_data["user_id"]]
            self._p1_id = p1_data["user_id"]
            self._p2_id = p2_data["user_id"]
            self.is_bot_match = p1_data.get("is_bot", False) or p2_data.get("is_bot", False)
            
            if p1_data.get("is_bot"):
                self.bot_id = p1_data["user_id"]
                self.bot_difficulty = p1_data.get("difficulty", "medium")
            elif p2_data.get("is_bot"):
                self.bot_id = p2_data["user_id"]
                self.bot_difficulty = p2_data.get("difficulty", "medium")
            
            # Метаданные
            self._p1_name = p1_data.get("name", "Игрок 1")
            self._p1_avatar_url = p1_data.get("avatar_url")
            self._p1_trophies = p1_data.get("trophies", 0)
            self._p1_clan = p1_data.get("clan", "")
            self._p2_name = p2_data.get("name", "Игрок 2")
            self._p2_avatar_url = p2_data.get("avatar_url")
            self._p2_trophies = p2_data.get("trophies", 0)
            self._p2_clan = p2_data.get("clan", "")
            
            # Загружаем данные карт из БД
            all_deck_ids = (p1_data.get("deck_ids") or []) + (p2_data.get("deck_ids") or [])
            cards_data = await self._load_cards_data(all_deck_ids)
            self._logger.info(
                "[ADAPTER] match=%s: загружено %d карточных записей из БД",
                self.match_id,
                len(cards_data),
            )
            
            # Загружаем уровни карт
            p1_levels = await self._load_user_card_levels(p1_data["user_id"], p1_data.get("deck_ids") or [])
            p2_levels = await self._load_user_card_levels(p2_data["user_id"], p2_data.get("deck_ids") or [])
            
            # Создаем колоды CardInstance
            p1_deck = deck_from_card_ids(p1_data.get("deck_ids") or [], cards_data, p1_levels)
            p2_deck = deck_from_card_ids(p2_data.get("deck_ids") or [], cards_data, p2_levels)

            # Store initial deck IDs for analytics
            self._p1_initial_deck_ids = [int(c) for c in (p1_data.get("deck_ids") or [])]
            self._p2_initial_deck_ids = [int(c) for c in (p2_data.get("deck_ids") or [])]

            self._logger.debug(
                "[ADAPTER] match=%s: собраны колоды (p1=%d карт, p2=%d карт)",
                self.match_id,
                len(p1_deck),
                len(p2_deck),
            )
            
            # Извлекаем героев
            p1_hero = self._extract_hero(p1_deck)
            p2_hero = self._extract_hero(p2_deck)

            # Blitz: hero HP -50% for both players
            if self._is_blitz:
                self._logger.info("[BLITZ] Halving hero HP: p1=%d→%d p2=%d→%d", p1_hero.hp, p1_hero.hp // 2, p2_hero.hp, p2_hero.hp // 2)
                p1_hero.hp = p1_hero.hp // 2
                p1_hero.max_hp = p1_hero.hp
                p2_hero.hp = p2_hero.hp // 2
                p2_hero.max_hp = p2_hero.hp
            
            # Создаем состояния игроков
            p1_state = CorePlayerState(
                user_id=p1_data["user_id"],
                is_bot=p1_data.get("is_bot", False),
                hero=p1_hero,
                mana=1,
                max_mana=1,
                hand=[],
                board=[],
                deck=p1_deck,
                trophies=p1_data.get("trophies", 0),
            )
            
            p2_state = CorePlayerState(
                user_id=p2_data["user_id"],
                is_bot=p2_data.get("is_bot", False),
                hero=p2_hero,
                mana=0,
                max_mana=0,
                hand=[],
                board=[],
                deck=p2_deck,
                trophies=p2_data.get("trophies", 0),
            )
            
            # Раздаем стартовые руки (3 карты каждому) - CHEAPEST FIRST
            # КРИТИЧНО: Фильтруем героев из колоды перед раздачей рук
            import random
            
            # P1: выбираем 3 самых дешевых ВОИНА (исключая героев и зелья)
            if p1_state.deck:
                # Фильтруем героев из колоды
                playable_deck = [c for c in p1_state.deck if c.card_type != CardType.HERO]
                p1_state.deck = playable_deck
                
                # Отделяем воинов от зелий
                warriors = [c for c in p1_state.deck if c.card_type == CardType.WARRIOR]
                potions = [c for c in p1_state.deck if c.card_type == CardType.POTION]
                
                # Сортируем воинов по стоимости маны
                warriors.sort(key=lambda c: c.mana_cost)
                
                # Берем 3 самых дешевых воина в руку
                for _ in range(min(3, len(warriors))):
                    p1_state.hand.append(warriors.pop(0))
                
                # Собираем колоду обратно: оставшиеся воины + зелья
                p1_state.deck = warriors + potions
                
                # Перемешиваем оставшуюся колоду
                random.shuffle(p1_state.deck)
                self._logger.info(
                    "[GAME_START] Player %s starting hand (warriors only): %s",
                    p1_state.user_id,
                    [c.name for c in p1_state.hand]
                )
            
            # P2: выбираем 3 самых дешевых ВОИНА (исключая героев и зелья)
            if p2_state.deck:
                # Фильтруем героев из колоды
                playable_deck = [c for c in p2_state.deck if c.card_type != CardType.HERO]
                p2_state.deck = playable_deck
                
                # Отделяем воинов от зелий
                warriors = [c for c in p2_state.deck if c.card_type == CardType.WARRIOR]
                potions = [c for c in p2_state.deck if c.card_type == CardType.POTION]
                
                # Сортируем воинов по стоимости маны
                warriors.sort(key=lambda c: c.mana_cost)
                
                # Берем 3 самых дешевых воина в руку
                for _ in range(min(3, len(warriors))):
                    p2_state.hand.append(warriors.pop(0))
                
                # Собираем колоду обратно: оставшиеся воины + зелья
                p2_state.deck = warriors + potions
                
                # Перемешиваем оставшуюся колоду
                random.shuffle(p2_state.deck)
                self._logger.info(
                    "[GAME_START] Player %s starting hand (warriors only): %s",
                    p2_state.user_id,
                    [c.name for c in p2_state.hand]
                )
            
            # Создаем игровое состояние
            game_state = GameState(
                p1=p1_state,
                p2=p2_state,
                current_turn_owner_id=p1_data["user_id"],
                turn_number=1,
            )
            
            # Инициализируем ArenaEnvironment
            self._arena = ArenaEnvironment(game_state, mana_per_turn=2 if self._is_blitz else 1)
            self.current_player_id = p1_data["user_id"]
            self.turn = 1
            self.turn_start_time = time.time()
            self.match_start_time = time.time()
            
            # Регистрируем в глобальном словаре
            self._active_matches[match_id] = self
            
            self._logger.info("create_match: successfully initialized match_id=%s", match_id)
            
            return {
                "success": True,
                "match_id": match_id,
                "player_ids": self.player_ids,
                "current_player_id": self.current_player_id,
            }
            
        except Exception as exc:
            self._logger.error("create_match failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}
    
    async def _load_cards_data(self, card_ids: List[Any]) -> Dict[int, Dict[str, Any]]:
        """Загрузить данные карт из БД."""
        if not self._db:
            return {}
        
        cards_data = {}
        for card_id in set(card_ids):
            try:
                cid = int(str(card_id).split(":")[0]) if ":" in str(card_id) else int(card_id)
            except (ValueError, TypeError):
                continue
            
            if cid in self._card_cache:
                cards_data[cid] = self._card_cache[cid]
                continue
            
            try:
                card_info = await self._db.get_card_info(cid, level=1)
                if card_info:
                    cards_data[cid] = card_info
                    self._card_cache[cid] = card_info
            except Exception as exc:
                self._logger.warning("Failed to load card %s: %s", cid, exc)
        
        return cards_data
    
    async def _load_user_card_levels(self, user_id: int, deck_ids: List[Any]) -> Dict[int, int]:
        """Загрузить уровни карт пользователя."""
        if not self._db:
            return {}
        
        try:
            user_cards = await self._db.get_user_cards(user_id)
            levels = {}
            for card in user_cards:
                card_id = card.get("id")
                level = card.get("level", 1)
                if card_id:
                    levels[int(card_id)] = level
            return levels
        except Exception as exc:
            self._logger.warning("Failed to load user card levels: %s", exc)
            return {}
    
    def _extract_hero(self, deck: List[CardInstance]) -> CardInstance:
        """Извлечь героя из колоды или создать дефолтного."""
        from uuid import uuid4
        
        for i, card in enumerate(deck):
            if card.card_type == CardType.HERO:
                self._logger.info(
                    "[ADAPTER] match=%s: найден герой card_id=%s",
                    self.match_id,
                    card.card_id,
                )
                return deck.pop(i)
        
        self._logger.warning(
            "[ADAPTER] match=%s: герой не найден в колоде, создан дефолтный",
            self.match_id,
        )
        return CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name="Герой",
            card_type=CardType.HERO,
            rarity="common",
            mana_cost=0,
            attack=0,
            hp=30,
            max_hp=30,
            mechanics=[],
            is_ready=True,
            description="Дефолтный герой",
        )
    
    # =========================================================================
    # ВЫПОЛНЕНИЕ ДЕЙСТВИЙ (ГЛАВНЫЙ МЕТОД ИНТЕГРАЦИИ)
    # =========================================================================
    
    def execute_action(self, user_id: int, action: BaseAction) -> Dict[str, Any]:
        """
        Выполнить действие игрока через core/engine.
        
        Args:
            user_id: ID игрока
            action: Объект действия (PlayCardAction, AttackAction, EndTurnAction)
            
        Returns:
            Результат выполнения с обновленным состоянием
        """
        if not self._arena:
            return {"success": False, "error": "arena_not_initialized"}
        
        # Выполняем действие
        success, error = self._arena.step(user_id, action)
        
        if not success:
            return {"success": False, "error": error}
        
        # Обновляем метаданные
        state = self._arena.state
        self.current_player_id = state.current_turn_owner_id
        self.turn = state.turn_number
        
        # Если был EndTurn, сбрасываем таймер
        if isinstance(action, EndTurnAction):
            self.turn_start_time = time.time()
        
        # Проверяем завершение игры
        result: Dict[str, Any] = {"success": True}
        
        if state.status != GameStatus.ONGOING:
            self.is_ended = True
            winner_id = None
            if state.status == GameStatus.P1_WIN:
                winner_id = state.p1.user_id
            elif state.status == GameStatus.P2_WIN:
                winner_id = state.p2.user_id
            
            result["game_over"] = True
            result["winner"] = winner_id
            result["winner_id"] = winner_id
        
        # Рассылаем событие обновления (если есть эмиттер)
        if self._event_emitter:
            # КРИТИЧНО: emit_to_match ожидает state_p1, а не state
            full_state = self.get_full_state()
            event_data = {
                "event_type": "state_changed",
                "match_id": self.match_id,
                "action": action.to_dict(),
                "state_p1": full_state,  # Исправлено: было "state", теперь "state_p1"
            }
            self._event_emitter.emit("state_changed", self.match_id, event_data)
        
        return result
    
    # =========================================================================
    # МЕТОДЫ СОВМЕСТИМОСТИ (обертки вокруг execute_action)
    # =========================================================================
    
    def play_card(
        self,
        user_id: int,
        card_id_from_hand: Any,
        target_position: int,
        target_id: Optional[Any] = None,
        target_is_hero: bool = False
    ) -> Dict[str, Any]:
        """
        Розыгрыш карты из руки.
        card_id_from_hand может быть индексом в руке или instance_id карты.
        """
        if not self._arena:
            return {"action": "play_card", "error": "arena_not_initialized"}
        
        state = self._arena.state
        player = state.p1 if state.p1.user_id == user_id else state.p2
        
        # Определяем индекс карты в руке
        hand_index = self._resolve_hand_index(player.hand, card_id_from_hand)
        if hand_index < 0:
            return {"action": "play_card", "error": "card_not_found_in_hand"}
        
        # Определяем target_id для core/actions
        resolved_target_id = None
        if target_is_hero:
            opponent = state.p2 if state.p1.user_id == user_id else state.p1
            resolved_target_id = str(opponent.hero.instance_id)
        elif target_id is not None:
            resolved_target_id = str(target_id)
        
        action = PlayCardAction(
            hand_index=hand_index,
            target_id=resolved_target_id,
            position=target_position,
        )
        
        result = self.execute_action(user_id, action)
        result["action"] = "play_card"
        return result
    
    def attack_target(
        self,
        user_id: int,
        attacker_id: Any,
        target_id: Any,
        target_is_hero: bool = False,
    ) -> Dict[str, Any]:
        """Атака существом."""
        if not self._arena:
            return {"action": "attack", "error": "arena_not_initialized"}
        
        action = AttackAction(
            attacker_id=str(attacker_id),
            target_id=str(target_id) if not target_is_hero else None,
            target_is_hero=target_is_hero,
        )
        
        result = self.execute_action(user_id, action)
        result["action"] = "attack"
        return result
    
    def end_turn(self, user_id: int) -> Dict[str, Any]:
        """Завершение хода."""
        if not self._arena:
            return {"action": "end_turn", "error": "arena_not_initialized"}
        
        action = EndTurnAction()
        result = self.execute_action(user_id, action)
        result["action"] = "end_turn"
        return result
    
    def _resolve_hand_index(self, hand: List[CardInstance], card_ref: Any) -> int:
        """
        Определить индекс карты в руке.
        card_ref может быть:
        - int индексом напрямую
        - instance_id (UUID или строка)
        - card_id (int)
        """
        # Прямой индекс
        if isinstance(card_ref, int) and 0 <= card_ref < len(hand):
            return card_ref
        
        # Поиск по instance_id или card_id
        card_ref_str = str(card_ref)
        for i, card in enumerate(hand):
            if str(card.instance_id) == card_ref_str:
                return i
            if str(card.card_id) == card_ref_str:
                return i
        
        return -1
    
    # =========================================================================
    # ПОЛУЧЕНИЕ СОСТОЯНИЯ ДЛЯ ФРОНТЕНДА
    # =========================================================================
    
    def get_full_state(self, viewer_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Сериализация состояния для отправки на фронтенд.
        Включает legal_actions для текущего игрока.
        """
        if not self._arena:
            return {
                "match_id": self.match_id,
                "error": "arena_not_initialized",
            }
        
        state = self._arena.state
        p1 = state.p1
        p2 = state.p2
        
        # Определяем, кто player, кто opponent относительно viewer
        if viewer_id is not None:
            if p1.user_id == viewer_id:
                player_state, opponent_state = p1, p2
                player_name, opponent_name = self._p1_name, self._p2_name
                player_avatar, opponent_avatar = self._p1_avatar_url, self._p2_avatar_url
            else:
                player_state, opponent_state = p2, p1
                player_name, opponent_name = self._p2_name, self._p1_name
                player_avatar, opponent_avatar = self._p2_avatar_url, self._p1_avatar_url
        else:
            # Без viewer_id отдаем p1 как player
            player_state, opponent_state = p1, p2
            player_name, opponent_name = self._p1_name, self._p2_name
            player_avatar, opponent_avatar = self._p1_avatar_url, self._p2_avatar_url
        
        is_my_turn = viewer_id is not None and state.current_turn_owner_id == viewer_id
        
        # Сериализуем legal_actions для viewer (если это его ход)
        legal_actions_json = []
        if viewer_id is not None and is_my_turn:
            legal_actions = self._arena.get_legal_actions(viewer_id)
            legal_actions_json = [self._serialize_action(a) for a in legal_actions]
        
        return {
            "match_id": self.match_id,
            "turn": state.turn_number,
            "current_player_id": state.current_turn_owner_id,
            "is_my_turn": is_my_turn,
            "player_ids": [p1.user_id, p2.user_id],
            "viewer_id": viewer_id,
            "is_ended": self.is_ended,
            "game_over": state.status != GameStatus.ONGOING,
            "winner_id": self._get_winner_id(),
            "turn_time_remaining": self.get_turn_time_remaining(),
            "turn_duration": self.turn_duration,
            
            # HP героев (для логов и быстрой проверки)
            "player1_hp": p1.hero.hp,
            "player2_hp": p2.hero.hp,
            
            # Состояние viewer'а (player)
            "player": self._serialize_player_state(player_state, player_name, player_avatar, show_hand=True),
            
            # Состояние оппонента (скрываем содержимое руки)
            "opponent": self._serialize_player_state(opponent_state, opponent_name, opponent_avatar, show_hand=False),
            
            # КРИТИЧНО: Список доступных действий
            "legal_actions": legal_actions_json,
            
            # История последних 100 действий (преобразуем в формат для клиента)
            "action_history": self._serialize_action_history(state.action_history, is_my_turn),
        }
    
    def _serialize_player_state(
        self, 
        ps: CorePlayerState, 
        name: str, 
        avatar_url: Optional[str],
        show_hand: bool
    ) -> Dict[str, Any]:
        """Сериализация состояния игрока."""
        hand_data = []
        if show_hand:
            hand_data = [self._serialize_card(c) for c in ps.hand]
        else:
            # Для оппонента показываем только количество карт
            hand_data = [{"hidden": True} for _ in ps.hand]
        
        return {
            "user_id": ps.user_id,
            "name": name,
            "avatar_url": avatar_url,
            "is_bot": ps.is_bot,
            "mana": ps.mana,
            "max_mana": ps.max_mana,
            "trophies": ps.trophies,
            "hero": self._serialize_card(ps.hero),
            "hand": hand_data,
            "hand_count": len(ps.hand),
            "deck_count": len(ps.deck),
            "board": [self._serialize_card(c, owner_id=ps.user_id) for c in ps.board],
        }
    
    def _serialize_card(self, card: CardInstance, owner_id: Optional[int] = None) -> Dict[str, Any]:
        """Сериализация карты для JSON."""
        return {
            "instance_id": str(card.instance_id),
            "card_id": card.card_id,
            "name": card.name,
            "description": card.description,
            "card_type": card.card_type.value,
            "rarity": card.rarity,
            "mana": card.mana_cost,
            "mana_cost": card.mana_cost,
            "attack": card.attack,
            "atk": card.attack,
            "hp": card.hp,
            "hp_current": card.hp,
            "max_hp": card.max_hp,
            "mechanics": card.mechanics,
            "is_ready": card.is_ready,
            "can_attack": card.is_ready and card.attack > 0,
            "is_asleep": card.is_asleep,
            "is_frozen": card.is_frozen,
            "owner_id": owner_id,
            "image": f"/DesignAssets/Cards/{card.card_id}.png" if card.card_id else "/DesignAssets/Cards/9.png",
        }
    
    def _serialize_action(self, action: BaseAction) -> Dict[str, Any]:
        """Сериализация действия для JSON."""
        data = action.to_dict()
        # Добавляем дополнительные поля для удобства фронтенда
        if isinstance(action, PlayCardAction):
            data["hand_index"] = action.hand_index
        elif isinstance(action, AttackAction):
            data["attacker_id"] = action.attacker_id
            data["target_id"] = action.target_id
            data["target_is_hero"] = action.target_is_hero
        return data
    
    def _serialize_action_history(
        self, 
        action_history: List[tuple[str, str]], 
        is_my_turn: bool
    ) -> List[Dict[str, str]]:
        """
        Сериализация истории действий для клиента.
        Преобразует (type, text) в формат с правильными метками 'Вы'/'Противник'.
        
        Args:
            action_history: Список (type, text) из GameState
            is_my_turn: Текущий ход игрока (для определения перспективы)
            
        Returns:
            Список словарей {"type": "player"/"opponent"/"system", "text": "описание"}
        """
        result = []
        for log_type, text in action_history:
            # Определяем финальный тип для клиента
            if log_type == "player":
                final_type = "player"
                prefix = "Вы"
            elif log_type == "opponent":
                final_type = "opponent"
                prefix = "Противник"
            else:
                final_type = "system"
                prefix = ""
            
            # Формируем итоговый текст с префиксом
            if prefix:
                # Если текст содержит ":" (формат "Карта: эффект"), добавляем префикс перед двоеточием
                if ":" in text and not text.startswith(prefix):
                    # Разделяем на часть до : и после
                    parts = text.split(":", 1)
                    final_text = f"{prefix} {parts[0]}:{parts[1]}"
                else:
                    # Простой текст - добавляем префикс в начало
                    final_text = f"{prefix} {text}"
            else:
                final_text = text
            
            result.append({
                "type": final_type,
                "text": final_text
            })
        
        return result
    
    def _get_winner_id(self) -> Optional[int]:
        """Определить победителя. Возвращает None при ничьей."""
        if not self._arena:
            return None
        status = self._arena.state.status
        if status == GameStatus.P1_WIN:
            return self._arena.state.p1.user_id
        elif status == GameStatus.P2_WIN:
            return self._arena.state.p2.user_id
        return None
    
    # =========================================================================
    # LEGAL ACTIONS (удобные методы)
    # =========================================================================
    
    def check_game_over(self) -> Dict[str, Any]:
        """Проверить завершение игры. Возвращает {'game_over': bool, 'winner_id': int|None}."""
        if not self._arena:
            return {"game_over": False, "winner_id": None}
        state = self._arena.state
        game_over = state.status != GameStatus.ONGOING
        winner_id = self._get_winner_id() if game_over else None
        if game_over:
            self.is_ended = True
        return {"game_over": game_over, "winner_id": winner_id}

    # =========================================================================
    
    def get_legal_actions(self, user_id: int) -> List[Dict[str, Any]]:
        """Получить список доступных действий для игрока (сериализованный)."""
        if not self._arena:
            return []
        actions = self._arena.get_legal_actions(user_id)
        return [self._serialize_action(a) for a in actions]
    
    def get_preview_delta(self, action: BaseAction) -> Dict[str, int]:
        """
        Получить предпросмотр изменений HP для действия.
        
        Args:
            action: Действие для предпросмотра
            
        Returns:
            Словарь {instance_id: delta_hp}
        """
        if not self._arena:
            return {}
        return self._arena.get_preview_delta(action)
    
    # =========================================================================
    # ТАЙМЕРЫ И СЛУЖЕБНЫЕ МЕТОДЫ
    # =========================================================================
    
    def get_current_player_id(self) -> Optional[int]:
        """Возвращает ID текущего игрока."""
        if self._arena:
            return self._arena.state.current_turn_owner_id
        return self.current_player_id
    
    def get_turn_time_remaining(self) -> float:
        """Возвращает оставшееся время хода."""
        if self.turn_start_time is None:
            return float(self.turn_duration)
        elapsed = time.time() - self.turn_start_time
        remaining = max(0, self.turn_duration - elapsed)
        return remaining
    
    def is_turn_expired(self) -> bool:
        """Проверяет истечение времени хода."""
        return self.get_turn_time_remaining() <= 0
    
    def is_bot(self, player_id: Optional[int]) -> bool:
        """Проверяет, является ли игрок ботом."""
        if not self._arena or player_id is None:
            return False
        state = self._arena.state
        if state.p1.user_id == player_id:
            return state.p1.is_bot
        elif state.p2.user_id == player_id:
            return state.p2.is_bot
        return False
    
    def is_current_player_bot(self) -> bool:
        """Проверяет, является ли текущий игрок ботом."""
        return self.is_bot(self.get_current_player_id())
    
    def mark_timeout(self, player_id: int) -> None:
        """
        Отмечает таймаут игрока. При достижении 2 таймаутов переводит в статус AFK.
        
        Args:
            player_id: ID игрока, у которого произошёл таймаут
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            return
        
        state = self._arena.state
        
        # Определяем игрока и увеличиваем счётчик
        if state.p1.user_id == player_id:
            self.p1_consecutive_timeouts += 1
            player = state.p1
            timeout_count = self.p1_consecutive_timeouts
        elif state.p2.user_id == player_id:
            self.p2_consecutive_timeouts += 1
            player = state.p2
            timeout_count = self.p2_consecutive_timeouts
        else:
            self._logger.warning("mark_timeout: unknown player_id=%s", player_id)
            return
        
        self._logger.info(
            "[TIMEOUT] Match: %s | Player: %s | Consecutive timeouts: %d",
            self.match_id, player_id, timeout_count
        )
        
        # При 2+ таймаутах переводим в AFK
        if timeout_count >= 2 and player.replacement_status == ReplacementStatus.ACTIVE:
            player.replacement_status = ReplacementStatus.AFK
            self._logger.warning(
                "[AFK_DETECTED] Match: %s | Player: %s marked as AFK (2+ timeouts)",
                self.match_id, player_id
            )
            print(f"!!! [BATTLE_ENGINE] Игрок {player_id} помечен как AFK (таймауты: {timeout_count})")
    
    def mark_surrender(self, player_id: int) -> None:
        """
        Отмечает сдачу игрока. Игрок переводится в статус SURRENDERED,
        но бой продолжается (за него играет бот).
        
        Args:
            player_id: ID игрока, который сдался
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            return
        
        state = self._arena.state
        
        # Определяем игрока
        if state.p1.user_id == player_id:
            player = state.p1
        elif state.p2.user_id == player_id:
            player = state.p2
        else:
            self._logger.warning("mark_surrender: unknown player_id=%s", player_id)
            return
        
        # Переводим в статус SURRENDERED
        player.replacement_status = ReplacementStatus.SURRENDERED
        
        self._logger.warning(
            "[SURRENDER] Player %s surrendered. Bot will continue playing.",
            player_id
        )
        print(f"!!! [BATTLE_ENGINE] Игрок {player_id} сдался, за него играет бот")
    
    def mark_client_ready(self) -> None:
        """Помечает клиента готовым."""
        self.client_ready = True
    
    def set_player_replacement_status(self, user_id: int, status: "ReplacementStatus") -> None:
        """
        Устанавливает статус замены для игрока.
        Используется для мгновенного перевода в AFK при разрыве сокета.
        
        Args:
            user_id: ID игрока
            status: Новый статус (ReplacementStatus.AFK, SURRENDERED, ACTIVE)
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            self._logger.warning("set_player_replacement_status: arena not initialized")
            return
        
        state = self._arena.state
        
        # Определяем игрока и меняем статус
        if state.p1.user_id == user_id:
            state.p1.replacement_status = status
            self._logger.info(
                "[STATUS_CHANGE] Match: %s | Player: %s | New status: %s",
                self.match_id, user_id, status.value
            )
        elif state.p2.user_id == user_id:
            state.p2.replacement_status = status
            self._logger.info(
                "[STATUS_CHANGE] Match: %s | Player: %s | New status: %s",
                self.match_id, user_id, status.value
            )
        else:
            self._logger.warning(
                "set_player_replacement_status: unknown player_id=%s in match=%s",
                user_id, self.match_id
            )
    
    # =========================================================================
    # СОВМЕСТИМОСТЬ СО СТАРЫМ API (property-подобные атрибуты)
    # =========================================================================
    
    @property
    def p1_state(self) -> Any:
        """Совместимость: возвращает обертку над p1."""
        if self._arena:
            return _LegacyPlayerStateWrapper(
                self._arena.state.p1, 
                self._p1_name, 
                self._p1_avatar_url
            )
        return _EmptyLegacyState(self._p1_id, self._p1_name)
    
    @property
    def p2_state(self) -> Any:
        """Совместимость: возвращает обертку над p2."""
        if self._arena:
            return _LegacyPlayerStateWrapper(
                self._arena.state.p2,
                self._p2_name,
                self._p2_avatar_url
            )
        return _EmptyLegacyState(self._p2_id, self._p2_name)
    
    def get_player_state(self, user_id: int) -> Any:
        """Возвращает состояние игрока."""
        if self._arena:
            if self._arena.state.p1.user_id == user_id:
                return self.p1_state
            return self.p2_state
        return _EmptyLegacyState(user_id, "Игрок")
    
    def get_opponent_state(self, user_id: int) -> Any:
        """Возвращает состояние противника."""
        if self._arena:
            if self._arena.state.p1.user_id == user_id:
                return self.p2_state
            return self.p1_state
        return _EmptyLegacyState(0, "Противник")
    
    def execute_bot_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действие бота (из словаря)."""
        if not self._arena:
            return {"error": "arena_not_initialized"}
        
        action_type = action_dict.get("type")
        
        if action_type == "play_card":
            action = PlayCardAction(
                hand_index=action_dict.get("hand_index", 0),
                target_id=action_dict.get("target_id"),
                position=action_dict.get("position"),
            )
        elif action_type == "attack":
            action = AttackAction(
                attacker_id=str(action_dict.get("attacker_id", "")),
                target_id=action_dict.get("target_id"),
                target_is_hero=action_dict.get("target_is_hero", False),
            )
        elif action_type == "end_turn":
            action = EndTurnAction()
        else:
            return {"error": f"unknown_action_type: {action_type}"}
        
        return self.execute_action(self.get_current_player_id(), action)

    # ── Analytics helpers ──

    @staticmethod
    def _snapshot_card(card: CardInstance) -> dict:
        return {
            "id": card.card_id,
            "mana": getattr(card, "mana_cost", 0),
            "atk": card.attack,
            "hp": card.hp,
            "max_hp": card.max_hp,
            "is_ready": card.is_ready,
            "mechanics": list(card.mechanics or []),
        }

    def build_analytics_snapshot(self) -> Optional[dict]:
        if not self._arena:
            return None
        st = self._arena.state
        def _player(p):
            return {
                "hp": p.hero.hp,
                "max_hp": p.hero.max_hp,
                "mana": p.mana,
                "max_mana": p.max_mana,
                "hand": [self._snapshot_card(c) for c in p.hand],
                "board": [self._snapshot_card(c) for c in p.board],
                "hero": self._snapshot_card(p.hero),
            }
        return {
            "turn": st.turn_number,
            "current_player": st.current_turn_owner_id,
            "p1": _player(st.p1),
            "p2": _player(st.p2),
        }

    def record_analytics_action(
        self,
        user_id: int,
        action_json: dict[str, Any],
        quality_score: Optional[float] = None,
    ) -> None:
        if not self._arena or self._analytics_flushed:
            return
        st = self._arena.state
        acting_player = 1 if user_id == st.p1.user_id else 2
        snapshot = self.build_analytics_snapshot()
        self._analytics_actions.append({
            "turn_number": st.turn_number,
            "acting_player": acting_player,
            "acting_user_id": user_id,
            "is_bot": self.is_bot(user_id),
            "state_json": snapshot or {},
            "action_json": action_json,
            "quality_score": quality_score,
        })


class _LegacyPlayerStateWrapper:
    """Обертка для совместимости со старым кодом, обращающимся к p1_state/p2_state."""
    
    def __init__(self, core_state: CorePlayerState, name: str, avatar_url: Optional[str]):
        self._core = core_state
        self.name = name
        self.avatar_url = avatar_url
    
    @property
    def user_id(self) -> int:
        return self._core.user_id
    
    @property
    def hero_hp(self) -> int:
        return self._core.hero.hp
    
    @property
    def mana(self) -> int:
        return self._core.mana
    
    @property
    def max_mana(self) -> int:
        return self._core.max_mana
    
    @property
    def hand(self) -> List[CardInstance]:
        return self._core.hand
    
    @property
    def board(self) -> List[CardInstance]:
        return self._core.board
    
    @property
    def deck(self) -> List[CardInstance]:
        return self._core.deck
    
    @property
    def draw_pile(self) -> List[str]:
        # Совместимость: возвращаем ID карт в колоде
        return [str(c.card_id) for c in self._core.deck]
    
    @property
    def hero_attack(self) -> int:
        return self._core.hero.attack
    
    @property
    def hero_mechanics(self) -> List[str]:
        return self._core.hero.mechanics
    
    @property
    def hero_name(self) -> str:
        return self._core.hero.name
    
    @property
    def replacement_status(self) -> Any:
        """Статус замены игрока (ACTIVE/AFK/SURRENDERED)."""
        return self._core.replacement_status
    
    @replacement_status.setter
    def replacement_status(self, value: Any) -> None:
        """Устанавливает статус замены игрока."""
        self._core.replacement_status = value
    
    @property
    def surrender_processed(self) -> bool:
        """Флаг немедленного списания трофеев при сдаче."""
        return getattr(self._core, "surrender_processed", False)
    
    @surrender_processed.setter
    def surrender_processed(self, value: bool) -> None:
        """Устанавливает флаг surrender_processed."""
        self._core.surrender_processed = value


class _EmptyLegacyState:
    """Пустая заглушка для случаев, когда арена не инициализирована."""
    
    def __init__(self, user_id: int, name: str):
        from core.state import ReplacementStatus
        self.user_id = user_id
        self.name = name
        self.avatar_url = None
        self.hero_hp = 30
        self.mana = 0
        self.max_mana = 0
        self.hand = []
        self.board = []
        self.draw_pile = []
        self.hero_attack = 0
        self.hero_mechanics = []
        self.hero_name = "Герой"
        self.replacement_status = ReplacementStatus.ACTIVE
        self.surrender_processed = False
