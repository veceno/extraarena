from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.bot_factory import BotGenerator
from infrastructure.database import Database
from infrastructure.match_config import MM_BOT_TIMEOUT, QUEUE_POLL_INTERVAL, SEARCH_WINDOWS
from infrastructure.config import DECK_SIZE, MM_TROPHY_LIMIT_CLASSIC


@dataclass
class QueueEntry:
    """Запись очереди на подбор соперника."""

    user_id: int
    trophies: int
    avg_level: int
    enqueued_at: float
    selected_deck_id: int | None = None
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    matched: bool = False


class Matchmaker:
    """
    Простая in-memory реализация матчмейкера с приоритетом «мгновенного старта» (Soft Start).

    Допущения:
    - bot_factory предоставляет create_match или BotGenerator для сборки бота.
    - battle_engine (если передан) умеет регистрировать бои через create_match.
    - Хранилище - оперативное: рестарт процесса очистит очередь и кеш.
    """

    def __init__(self, db: Database, bot_factory: Any, battle_engine: Any | None = None) -> None:
        self._db = db
        self._bot_factory = bot_factory
        self._battle_engine = battle_engine

        self._queue: List[QueueEntry] = []
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._logger = logging.getLogger(__name__)

    async def find_match(
        self,
        user_id: int,
        trophies: int,
        user_avg_level: int,
        selected_deck_id: int | None = None,
        game_mode: str = "classic",
    ) -> Dict[str, Any]:
        """
        Главный вход: вернуть соперника или зарегистрировать ожидание.

        1) Soft Start - игроки с трофеями <300 сразу играют с ботом.
        2) Остальные ставятся в очередь и ищут соперника по окнам 50/200/500.
        3) После 15 секунд без соперника игрок гарантированно получает бота.
        
        Args:
            user_id: ID игрока
            trophies: Количество трофеев
            user_avg_level: Средний уровень карт
            selected_deck_id: ID выбранного пресета колоды (опционально)
        """
        # Мгновенный PvE для новичков
        if trophies < MM_TROPHY_LIMIT_CLASSIC:
            # Логируем попадание в PvE-зону, чтобы отладить залипание "Поиск..."
            self._logger.info(
                "PvE soft-start: user_id=%s trophies=%s avg_level=%s deck=%s",
                user_id, trophies, user_avg_level, selected_deck_id,
            )
            try:
                # Агрессивно логируем каждое действие перед вызовом фабрики ботов, чтобы поймать точку зависания.
                self._logger.info(
                    "DEBUG: Calling bot_factory.create_match... user_id=%s trophies=%s avg_level=%s",
                    user_id,
                    trophies,
                    user_avg_level,
                )
                match_payload = await self._create_bot_match(user_id, trophies, user_avg_level, selected_deck_id, game_mode=game_mode)
                self._logger.info("DEBUG: Bot factory returned payload: %s", match_payload)
                return match_payload
            except Exception as exc:  # noqa: BLE001
                # Логируем крит и стек, чтобы увидеть точку падения, и возвращаем ошибку вместо зависания.
                self._logger.error(
                    "CRITICAL ERROR in create_bot_match for user_id=%s: %s", user_id, exc, exc_info=True
                )
                return {"status": "error", "message": str(exc), "user_id": user_id, "is_bot": True}

        seeker = QueueEntry(
            user_id=user_id,
            trophies=trophies,
            avg_level=user_avg_level,
            enqueued_at=time.monotonic(),
            selected_deck_id=selected_deck_id,
        )

        async with self._lock:
            # Если игрок уже в очереди, очищаем старую запись, чтобы не было дублей
            self._drop_existing(user_id)
            # Мгновенно пытаемся найти соперника в уже существующей очереди
            opponent = self._find_candidate(seeker, SEARCH_WINDOWS[0])
            if opponent:
                return await self._pair_players(seeker, opponent, game_mode=game_mode)

            # Никого не нашли - ставим в очередь и создаем статус ожидания
            self._queue.append(seeker)
            self._matches[seeker.match_id] = {
                "status": "waiting",
                "match_id": seeker.match_id,
                "opponent_id": None,
                "is_bot": False,
                "queued_at": seeker.enqueued_at,
                "user_id": seeker.user_id,
                "game_mode": game_mode,
            }

            # Запускаем фоновую корутину поиска с расширяющимися окнами
            task = asyncio.create_task(self._search_loop(seeker, game_mode=game_mode))
            self._tasks[seeker.match_id] = task

            result = {
                "status": "waiting",
                "match_id": seeker.match_id,
                "opponent_id": None,
                "is_bot": False,
                "game_mode": game_mode,
            }
            if selected_deck_id:
                result["selected_deck_id"] = selected_deck_id
            return result

    async def get_status(self, match_id: str) -> Dict[str, Any]:
        """
        Статус матча для клиентского поллинга.
        Возвращает 'waiting' если поиск продолжается, иначе 'found' с деталями.
        """
        async with self._lock:
            match = self._matches.get(match_id)
            if match:
                return match

        # Если матча нет в кеше - считаем, что он не найден/просрочен.
        return {"status": "not_found", "match_id": match_id}

    async def _search_loop(self, seeker: QueueEntry, game_mode: str = "classic") -> None:
        """
        Пошаговый поиск соперника:
        - Проверяем окна 50 / 200 / 500.
        - Между проверками спим по QUEUE_POLL_INTERVAL.
        - По истечении MM_BOT_TIMEOUT выдаем бота.
        """
        start = seeker.enqueued_at

        for window in SEARCH_WINDOWS:
            # Цикл опроса для текущего окна
            while time.monotonic() - start < MM_BOT_TIMEOUT:
                async with self._lock:
                    if seeker.matched:
                        return  # Уже подобран другим запросом

                    opponent = self._find_candidate(seeker, window)
                    if opponent:
                        await self._pair_players(seeker, opponent, game_mode=game_mode)
                        return

                await asyncio.sleep(QUEUE_POLL_INTERVAL)

        # Таймаут - создаем бота и завершаем ожидание
        await self._handle_bot_timeout(seeker, game_mode=game_mode)

    async def _handle_bot_timeout(self, seeker: QueueEntry, game_mode: str = "classic") -> None:
        """Выдача бота по истечении дедлайна."""
        async with self._lock:
            if seeker.matched:
                return
            # Убираем из очереди, чтобы не мешал будущим подборам
            self._queue = [entry for entry in self._queue if entry.match_id != seeker.match_id]
            self._tasks.pop(seeker.match_id, None)

        self._logger.info(
            "_handle_bot_timeout: creating bot for user=%s deck=%s game_mode=%s",
            seeker.user_id, seeker.selected_deck_id, game_mode,
        )
        bot_match = await self._create_bot_match(seeker.user_id, seeker.trophies, seeker.avg_level, selected_deck_id=seeker.selected_deck_id, game_mode=game_mode)
        async with self._lock:
            self._matches[seeker.match_id] = bot_match

    def _find_candidate(self, seeker: QueueEntry, window: int) -> Optional[QueueEntry]:
        """Поиск соперника в очереди по допуску по трофеям."""
        for entry in self._queue:
            if entry.user_id == seeker.user_id or entry.matched:
                continue
            if abs(entry.trophies - seeker.trophies) <= window:
                return entry
        return None

    async def _pair_players(self, seeker: QueueEntry, opponent: QueueEntry, game_mode: str = "classic") -> Dict[str, Any]:
        """Формируем матч между двумя игроками и фиксируем статус для обоих.
        ВАЖНО: вызывающая сторона уже держит self._lock."""
        final_match_id = str(uuid.uuid4())

        seeker.matched = opponent.matched = True
        self._queue = [
            entry for entry in self._queue if not entry.matched and entry.match_id not in {seeker.match_id, opponent.match_id}
        ]

        # Обновляем статусы обоих участников (используем один final_match_id)
        self._matches[seeker.match_id] = {
            "status": "found",
            "match_id": final_match_id,
            "opponent_id": opponent.user_id,
            "is_bot": False,
            "user_id": seeker.user_id,
            "game_mode": game_mode,
        }
        self._matches[opponent.match_id] = {
            "status": "found",
            "match_id": final_match_id,
            "opponent_id": seeker.user_id,
            "is_bot": False,
            "user_id": opponent.user_id,
            "game_mode": game_mode,
        }

        # Дополнительная запись по итоговому match_id.
        #
        # Зачем она нужна:
        # - фронт после редиректа ходит в /api/battle/state?id=<final_match_id>
        # - сервер (battle_state_handler) умеет «лениво» инициализировать движок, если
        #   по этому match_id ещё нет BattleEngine в кеше, но для этого ему нужен matchmaker.get_status().
        # - в PvE (бот) такая запись часто уже есть, а в PvP раньше её не было, из‑за чего
        #   on-demand init мог не срабатывать и приводить к 404 match_not_found.
        #
        # Поэтому фиксируем финальный match_id -> (оба игрока) в совместимом формате.
        self._matches[final_match_id] = {
            "status": "found",
            "match_id": final_match_id,
            "player_ids": [seeker.user_id, opponent.user_id],
            # Сохраняем выбранные пресеты для каждого игрока
            "player_decks": {
                str(seeker.user_id): seeker.selected_deck_id,
                str(opponent.user_id): opponent.selected_deck_id,
            },
            # Для совместимости со старым форматом возвращаем user_id/opponent_id,
            # даже если запрос к этому ключу не привязан к конкретному игроку.
            "user_id": seeker.user_id,
            "opponent_id": opponent.user_id,
            "is_bot": False,
            "game_mode": game_mode,
        }

        # Останавливаем фоновые задачи обоих игроков, если они были
        for match_id in (seeker.match_id, opponent.match_id):
            task = self._tasks.pop(match_id, None)
            if task:
                task.cancel()

        # Пробуем зарегистрировать бой в движке, если он предоставлен
        await self._safe_call_battle_engine(final_match_id, [seeker.user_id, opponent.user_id], is_bot=False)
        return self._matches[seeker.match_id]

    async def _create_bot_match(
        self,
        user_id: int,
        trophies: int,
        user_avg_level: int,
        selected_deck_id: int | None = None,
        game_mode: str = "classic",
    ) -> Dict[str, Any]:
        """
        Генерация боя против бота.
        Приоритет - использовать bot_factory.create_match, иначе падаем на генератор ботов.
        
        Args:
            user_id: ID игрока
            trophies: Трофеи игрока
            user_avg_level: Средний уровень карт
            selected_deck_id: ID выбранного пресета колоды (опционально)
        """
        # Если колода не выбрана, берем случайную из доступных пресетов игрока
        if not selected_deck_id:
            try:
                presets = await self._db.get_user_deck_presets(user_id)
                if presets:
                    import random
                    selected_preset = random.choice(presets)
                    selected_deck_id = selected_preset.get("preset_number", 1)
                    self._logger.info("Auto-selected deck preset %s for user %s", selected_deck_id, user_id)
                else:
                    selected_deck_id = 1  # Дефолтный пресет
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failed to load deck presets: %s", exc)
                selected_deck_id = 1
        # Пытаемся воспользоваться готовым create_match из bot_factory, если он есть
        factory_callable = getattr(self._bot_factory, "create_match", None)
        if callable(factory_callable):
            try:
                maybe_result = factory_callable(user_id=user_id, trophies=trophies, user_avg_level=user_avg_level)
                result = await maybe_result if asyncio.iscoroutine(maybe_result) else maybe_result
                if not isinstance(result, dict) or not result:
                    # Если фабрика вернула пустоту, логируем и переходим на fallback
                    raise ValueError("bot_factory.create_match вернула пустой результат")
                # Гарантируем минимальный набор ключей
                match_id = str(result.get("match_id") or uuid.uuid4())
                opponent_id = result.get("opponent_id", result.get("bot_id"))
                if opponent_id is None:
                    opponent_id = -1
                payload = {
                    "status": "found",
                    "match_id": match_id,
                    "opponent_id": opponent_id,
                    "is_bot": True,
                    "user_id": user_id,
                    "game_mode": game_mode,
                    # Единый список участников - помогает серверу инициализировать бой,
                    # не завися от того, какой именно ключ использован в кеше статусов.
                    "player_ids": [user_id, opponent_id],
                    "selected_deck_id": selected_deck_id,
                    "player_decks": {
                        str(user_id): selected_deck_id,
                    }
                }
                # Сохраняем для поллинга
                async with self._lock:
                    self._matches[match_id] = payload
                return payload
            except Exception as exc:  # noqa: BLE001 - нам нужно логировать любые сбои фабрики
                self._logger.error("bot_factory.create_match упал: %s", exc, exc_info=True)

        # Fallback: строим бота сами через BotGenerator
        opponent_id = -1
        bot_name = None
        bot_avatar_url = None
        bot_difficulty = "lite"  # КРИТИЧНО: Дефолт для новичков
        bot_level = 1
        bot_deck_ids = []
        try:
            if isinstance(self._bot_factory, BotGenerator):
                bot_profile = await self._bot_factory.create_persistent_bot(
                    player_id=user_id,
                    player_trophies=trophies,
                    player_avg_level=user_avg_level,
                )
                opponent_id = bot_profile.get("user_id", -1)
                bot_name = bot_profile.get("name")  # Извлекаем имя бота из профиля
                bot_avatar_url = bot_profile.get("avatar_url")  # Извлекаем аватарку бота из профиля
                bot_difficulty = bot_profile.get("difficulty", "lite")  # КРИТИЧНО: Извлекаем сложность
                bot_level = bot_profile.get("level", 1)  # Извлекаем уровень бота
                bot_deck_ids = bot_profile.get("deck_ids", [])  # Извлекаем колоду бота
        except Exception as exc:  # noqa: BLE001
            self._logger.error("BotGenerator fallback не удался: %s", exc, exc_info=True)

        match_id = str(uuid.uuid4())
        
        # ВАЖНО: ВСЕГДА создаем bot_info, даже если имя None, чтобы сервер понял, что это бот
        # КРИТИЧНО: Передаем difficulty для правильной настройки ИИ
        bot_info_payload = {
            "name": bot_name,
            "avatar_url": bot_avatar_url,
            "user_id": opponent_id,  # Добавляем ID бота для полноты информации
            "difficulty": bot_difficulty,  # КРИТИЧНО: Передаем сложность бота
            "level": bot_level,  # Уровень бота для статов карт
            "deck_ids": bot_deck_ids,  # Колода бота
        }
        
        self._logger.info(
            "DEBUG: _create_bot_match creating payload with bot_info=%s", 
            bot_info_payload
        )
        
        payload = {
            "status": "found",
            "match_id": match_id,
            "opponent_id": opponent_id,
            "is_bot": True,
            "user_id": user_id,
            # Единый список участников для удобства server-side инициализации боя.
            "player_ids": [user_id, opponent_id],
            # ВСЕГДА добавляем bot_info для передачи в server.py
            "bot_info": bot_info_payload,
            "selected_deck_id": selected_deck_id,
            "game_mode": game_mode,
            "player_decks": {
                str(user_id): selected_deck_id,
            }
        }

        async with self._lock:
            self._matches[match_id] = payload

        await self._safe_call_battle_engine(match_id, [user_id, opponent_id], is_bot=True)
        return payload

    async def _safe_call_battle_engine(self, match_id: str, players: List[int], is_bot: bool) -> None:
        """
        Безопасно вызывает battle_engine.create_match, если он предоставлен.
        
        ПРИМЕЧАНИЕ: В новой архитектуре инициализацией занимается server.py при первом
        обращении к состоянию боя. Вызов здесь закомментирован, чтобы избежать 
        ошибок несовпадения аргументов и дублирования логики.
        """
        if not self._battle_engine:
            return

        # engine_callable = getattr(self._battle_engine, "create_match", None)
        # if not callable(engine_callable):
        #     return

        # try:
        #     # В новом формате BattleEngine.create_match ожидает p1_data и p2_data
        #     # maybe_coro = engine_callable(match_id=match_id, players=players, is_bot=is_bot)
        #     # if asyncio.iscoroutine(maybe_coro):
        #     #     await maybe_coro
        #     pass
        # except Exception as exc:  # noqa: BLE001
        #     self._logger.error("Ошибка battle_engine.create_match: %s", exc, exc_info=True)
        return

    def _drop_existing(self, user_id: int) -> None:
        """Удаляет старые записи игрока из очереди/кеша статусов и отменяет фоновые задачи."""
        match_ids_to_drop = [entry.match_id for entry in self._queue if entry.user_id == user_id]
        self._queue = [entry for entry in self._queue if entry.user_id != user_id]

        for match_id in match_ids_to_drop:
            task = self._tasks.pop(match_id, None)
            if task:
                task.cancel()
            self._matches.pop(match_id, None)

        for match_id, data in list(self._matches.items()):
            if data.get("user_id") == user_id:
                self._matches.pop(match_id, None)







