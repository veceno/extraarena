from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.bot_factory import BotGenerator
from infrastructure.database import Database
from infrastructure.match_config import MM_BOT_TIMEOUT, QUEUE_POLL_INTERVAL, SEARCH_WINDOWS
from infrastructure.config import DECK_SIZE, MM_TROPHY_LIMIT_CLASSIC
import infrastructure.match_modes as match_modes


@dataclass
class QueueEntry:
    """Запись очереди на подбор соперника."""

    user_id: int
    trophies: int
    max_level: int
    enqueued_at: float
    selected_deck_id: int | None = None
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    matched: bool = False
    game_mode: str = "classic"
    canonical_mode: str = "classic"


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
        self._match_aliases: Dict[str, set[str]] = {}
        self._match_updated_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._soft_start_tasks: Dict[tuple[int, str], asyncio.Task] = {}
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Matchmaker uses process-local in-memory queue/state; run a single web process "
            "or add shared storage before multi-worker deployment."
        )

    def _register_match_aliases(self, aliases: set[str]) -> None:
        clean_aliases = {str(alias) for alias in aliases if alias}
        now = time.monotonic()
        for alias in clean_aliases:
            self._match_aliases[alias] = set(clean_aliases)
            self._match_updated_at[alias] = now

    def _aliases_for(self, match_id: str) -> set[str]:
        match_id = str(match_id)
        if match_id in self._match_aliases:
            return set(self._match_aliases[match_id])
        match = self._matches.get(match_id)
        if match and match.get("match_id"):
            final_id = str(match.get("match_id"))
            aliases = {match_id, final_id}
            for alias, payload in self._matches.items():
                if str(payload.get("match_id")) == final_id:
                    aliases.add(str(alias))
            self._register_match_aliases(aliases)
            return aliases
        return {match_id}

    def _touch_matches(self, aliases: set[str]) -> None:
        now = time.monotonic()
        for alias in aliases:
            self._match_updated_at[str(alias)] = now

    def _delete_match_aliases(self, match_id: str) -> None:
        aliases = self._aliases_for(str(match_id))
        for alias in aliases:
            task = self._tasks.pop(alias, None)
            if task:
                task.cancel()
            self._matches.pop(alias, None)
            self._match_updated_at.pop(alias, None)
        for alias in aliases:
            self._match_aliases.pop(alias, None)
        for alias, group in list(self._match_aliases.items()):
            if group & aliases:
                self._match_aliases.pop(alias, None)

    @staticmethod
    def _poll_interval_for_window(window_seconds: float) -> float:
        return max(0.25, min(float(QUEUE_POLL_INTERVAL), float(window_seconds) / 4.0))

    def _choose_starting_player_id(self, player_ids: list[int]) -> int:
        """Choose the first turn owner independently from p1/p2 ordering."""
        return int(random.choice(player_ids))

    async def find_match(
        self,
        user_id: int,
        trophies: int,
        user_max_level: int,
        selected_deck_id: int | None = None,
        game_mode: str = "classic",
        canonical_mode: str | None = None,
    ) -> Dict[str, Any]:
        """
        Основная точка входа: мгновенно запускает PvE если трофеи < 300,
        иначе ставит в очередь и возвращает статус 'queued'.
        """
        canonical = canonical_mode or match_modes.resolve_canonical_mode_id(game_mode)
        mode_config = match_modes.resolve_mode_config(canonical)
        if not mode_config.available:
            return {"status": "error", **match_modes.mode_unavailable_payload(canonical)}

        # Soft Start: only if bots_allowed and trophies < threshold
        if mode_config.classic.bots_allowed and trophies < MM_TROPHY_LIMIT_CLASSIC:
            self._logger.info(
                "PvE soft-start: user_id=%s trophies=%s max_level=%s deck=%s",
                user_id, trophies, user_max_level, selected_deck_id,
            )
            soft_key = (int(user_id), canonical)
            try:
                async with self._lock:
                    existing = self._find_active_match_for_user_locked(user_id, canonical)
                    if existing:
                        return existing
                    task = self._soft_start_tasks.get(soft_key)
                    if task is None:
                        self._drop_existing(user_id)
                        self._logger.debug(
                            "Calling bot_factory.create_match user_id=%s trophies=%s max_level=%s",
                            user_id,
                            trophies,
                            user_max_level,
                        )
                        task = asyncio.create_task(
                            self._create_bot_match(
                                user_id,
                                trophies,
                                user_max_level,
                                selected_deck_id,
                                game_mode=canonical,
                            )
                        )
                        self._soft_start_tasks[soft_key] = task

                match_payload = await task
                async with self._lock:
                    if self._soft_start_tasks.get(soft_key) is task:
                        self._soft_start_tasks.pop(soft_key, None)
                self._logger.debug("Bot factory returned payload: %s", match_payload)
                return match_payload
            except Exception as exc:  # noqa: BLE001
                async with self._lock:
                    task = self._soft_start_tasks.get(soft_key)
                    if task and task.done():
                        self._soft_start_tasks.pop(soft_key, None)
                # Логируем крит и стек, чтобы увидеть точку падения, и возвращаем ошибку вместо зависания.
                self._logger.error(
                    "CRITICAL ERROR in create_bot_match for user_id=%s: %s", user_id, exc, exc_info=True
                )
                return {"status": "error", "message": "bot_match_create_failed", "user_id": user_id, "is_bot": True}

        seeker = QueueEntry(
            user_id=user_id,
            trophies=trophies,
            max_level=user_max_level,
            enqueued_at=time.monotonic(),
            selected_deck_id=selected_deck_id,
            game_mode=game_mode,
            canonical_mode=canonical,
        )

        async with self._lock:
            # Если игрок уже в очереди, очищаем старую запись, чтобы не было дублей
            self._drop_existing(user_id)
            # Мгновенно пытаемся найти соперника в уже существующей очереди
            opponent = self._find_candidate(seeker, SEARCH_WINDOWS[0])
            if opponent:
                return await self._pair_players(seeker, opponent, game_mode=canonical)

            # Никого не нашли - ставим в очередь и создаем статус ожидания
            self._queue.append(seeker)
            self._matches[seeker.match_id] = {
                "status": "waiting",
                "match_id": seeker.match_id,
                "opponent_id": None,
                "is_bot": False,
                "queued_at": seeker.enqueued_at,
                "user_id": seeker.user_id,
                "game_mode": canonical,
                "mode_config": match_modes.serialize_mode_config(mode_config),
                "search_timeout_seconds": 30 if not mode_config.classic.bots_allowed else MM_BOT_TIMEOUT,
            }
            self._register_match_aliases({seeker.match_id})

            # Запускаем фоновую корутину поиска с расширяющимися окнами
            max_wait = 30 if not mode_config.classic.bots_allowed else None
            task = asyncio.create_task(self._search_loop(seeker, game_mode=canonical, max_wait_seconds=max_wait))
            self._tasks[seeker.match_id] = task

            result = {
                "status": "waiting",
                "match_id": seeker.match_id,
                "opponent_id": None,
                "is_bot": False,
                "game_mode": canonical,
                "mode_config": match_modes.serialize_mode_config(mode_config),
                "search_timeout_seconds": 30 if not mode_config.classic.bots_allowed else MM_BOT_TIMEOUT,
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
                return dict(match)

        # Если матча нет в кеше - считаем, что он не найден/просрочен.
        return {"status": "not_found", "match_id": match_id}

    async def get_active_match_for_user(self, user_id: int) -> Dict[str, Any]:
        """Return a found-but-not-necessarily-initialized match for reconnect/startup recovery."""
        try:
            user_id_int = int(user_id)
        except (TypeError, ValueError):
            return {"status": "not_found", "user_id": user_id}

        async with self._lock:
            match = self._find_active_match_for_user_locked(user_id_int)
            if match:
                return match

        return {"status": "not_found", "user_id": user_id_int}

    def _find_active_match_for_user_locked(
        self,
        user_id: int,
        canonical_mode: str | None = None,
    ) -> Dict[str, Any] | None:
        user_id_int = int(user_id)
        for match in self._matches.values():
            if match.get("status") != "found":
                continue
            if canonical_mode is not None and match.get("game_mode") != canonical_mode:
                continue

            participant_ids: set[int] = set()
            for raw_id in match.get("player_ids") or ():
                try:
                    participant_ids.add(int(raw_id))
                except (TypeError, ValueError):
                    continue
            for key in ("user_id", "opponent_id"):
                try:
                    participant_ids.add(int(match.get(key)))
                except (TypeError, ValueError):
                    continue

            if user_id_int in participant_ids:
                return dict(match)
        return None

    async def cancel_match(
        self,
        match_id: str,
        *,
        message: str = "Режим сейчас недоступен. Попробуйте позже.",
        error: str = "mode_unavailable",
        game_mode: str | None = None,
    ) -> Dict[str, Any]:
        """Mark a queued/found matchmaking entry as canceled with a stable terminal status."""
        async with self._lock:
            aliases = self._aliases_for(str(match_id))
            for alias in aliases:
                task = self._tasks.pop(alias, None)
                if task:
                    task.cancel()
            self._queue = [entry for entry in self._queue if entry.match_id not in aliases]
            existing = self._matches.get(match_id, {})
            payload = {
                "status": "canceled",
                "match_id": str(existing.get("match_id") or match_id),
                "error": error,
                "message": message,
                "game_mode": game_mode or existing.get("game_mode"),
                "terminal_at": time.monotonic(),
            }
            if existing.get("user_id") is not None:
                payload["user_id"] = existing.get("user_id")
            for alias in aliases:
                alias_payload = dict(payload)
                if alias != payload["match_id"]:
                    alias_payload["alias_match_id"] = alias
                alias_existing = self._matches.get(alias, {})
                if alias_existing.get("user_id") is not None:
                    alias_payload["user_id"] = alias_existing.get("user_id")
                self._matches[alias] = alias_payload
            self._register_match_aliases(aliases)
            return dict(self._matches.get(str(match_id), payload))

    async def mark_match_finished(
        self,
        match_id: str,
        *,
        winner_id: int | str | None = None,
    ) -> Dict[str, Any]:
        """Mark a found match as terminal so future matchmaking cannot resume it."""
        async with self._lock:
            aliases = self._aliases_for(str(match_id))
            existing = self._matches.get(str(match_id), {})
            payload = {
                "status": "finished",
                "match_id": str(existing.get("match_id") or match_id),
                "winner_id": winner_id,
                "game_mode": existing.get("game_mode"),
                "terminal_at": time.monotonic(),
            }
            for key in ("user_id", "opponent_id", "player_ids", "is_bot"):
                if existing.get(key) is not None:
                    payload[key] = existing.get(key)

            for alias in aliases:
                alias_existing = self._matches.get(alias, {})
                alias_payload = dict(payload)
                if alias != payload["match_id"]:
                    alias_payload["alias_match_id"] = alias
                if alias_existing.get("user_id") is not None:
                    alias_payload["user_id"] = alias_existing.get("user_id")
                self._matches[alias] = alias_payload
            self._register_match_aliases(aliases)
            return dict(self._matches.get(str(match_id), payload))

    async def _search_loop(
        self,
        seeker: QueueEntry,
        game_mode: str = "classic",
        max_wait_seconds: int | None = None,
    ) -> None:
        """
        Пошаговый поиск соперника:
        - Проверяем окна 50 / 200 / 500.
        - Между проверками спим по QUEUE_POLL_INTERVAL.
        - По истечении MM_BOT_TIMEOUT выдаем бота (или cancel если bots_allowed=False).
        """
        effective_max_wait = max_wait_seconds or MM_BOT_TIMEOUT
        search_deadline = time.monotonic() + effective_max_wait
        per_window_timeout = effective_max_wait / len(SEARCH_WINDOWS)

        for window in SEARCH_WINDOWS:
            window_deadline = min(time.monotonic() + per_window_timeout, search_deadline)
            while time.monotonic() < window_deadline:
                async with self._lock:
                    if seeker.matched:
                        return  # Уже подобран другим запросом

                    opponent = self._find_candidate(seeker, window)
                    if opponent:
                        await self._pair_players(seeker, opponent, game_mode=game_mode)
                        return

                sleep_for = min(
                    self._poll_interval_for_window(per_window_timeout),
                    max(0.0, window_deadline - time.monotonic()),
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        # Таймаут - обрабатываем по политике режима
        mode_config = match_modes.resolve_mode_config(game_mode)
        if not mode_config.classic.bots_allowed:
            await self._handle_cancel_timeout(seeker, game_mode=game_mode)
        else:
            await self._handle_bot_timeout(seeker, game_mode=game_mode)

    async def _handle_cancel_timeout(self, seeker: QueueEntry, game_mode: str = "classic") -> None:
        """Отмена поиска по таймауту в режимах без ботов."""
        async with self._lock:
            if seeker.matched:
                return
            self._queue = [entry for entry in self._queue if entry.match_id != seeker.match_id]
            self._tasks.pop(seeker.match_id, None)
            self._matches[seeker.match_id] = {
                "status": "canceled",
                "match_id": seeker.match_id,
                "user_id": seeker.user_id,
                "game_mode": game_mode,
                "message": "Сейчас не так много выбирающих этот режим — попробуй сыграть в классику или возвращайся позже.",
                "terminal_at": time.monotonic(),
            }
            self._register_match_aliases({seeker.match_id})

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
        try:
            bot_match = await self._create_bot_match(
                seeker.user_id,
                seeker.trophies,
                seeker.max_level,
                selected_deck_id=seeker.selected_deck_id,
                game_mode=game_mode,
            )
        except Exception as exc:
            self._logger.error(
                "_handle_bot_timeout: failed to create bot match for user=%s mode=%s: %s",
                seeker.user_id,
                game_mode,
                exc,
                exc_info=True,
            )
            async with self._lock:
                self._matches[seeker.match_id] = {
                    "status": "canceled",
                    "match_id": seeker.match_id,
                    "user_id": seeker.user_id,
                    "game_mode": game_mode,
                    "error": "bot_match_create_failed",
                    "message": "Не удалось подготовить бой. Попробуйте начать поиск заново.",
                    "terminal_at": time.monotonic(),
                }
                self._register_match_aliases({seeker.match_id})
            return
        async with self._lock:
            self._matches[seeker.match_id] = bot_match
            aliases = {seeker.match_id, str(bot_match.get("match_id", seeker.match_id))}
            self._register_match_aliases(aliases)

    def _find_candidate(self, seeker: QueueEntry, window: int) -> Optional[QueueEntry]:
        """Поиск соперника в очереди по допуску по трофеям."""
        candidates: list[QueueEntry] = []
        for entry in self._queue:
            if entry.user_id == seeker.user_id or entry.matched:
                continue
            if entry.canonical_mode != seeker.canonical_mode:
                continue
            if abs(entry.trophies - seeker.trophies) <= window:
                candidates.append(entry)
        if not candidates:
            return None
        return min(candidates, key=lambda entry: (abs(entry.trophies - seeker.trophies), entry.enqueued_at))

    async def _pair_players(self, seeker: QueueEntry, opponent: QueueEntry, game_mode: str = "classic") -> Dict[str, Any]:
        """Формируем матч между двумя игроками и фиксируем статус для обоих.
        ВАЖНО: вызывающая сторона уже держит self._lock."""
        final_match_id = str(uuid.uuid4())
        player_ids = [seeker.user_id, opponent.user_id]
        starting_player_id = self._choose_starting_player_id(player_ids)
        player_decks = {
            str(seeker.user_id): seeker.selected_deck_id,
            str(opponent.user_id): opponent.selected_deck_id,
        }

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
            "player_ids": player_ids,
            "player_decks": player_decks,
            "starting_player_id": starting_player_id,
            "game_mode": game_mode,
            "mode_config": match_modes.serialize_mode_config(match_modes.resolve_mode_config(game_mode)),
        }
        self._matches[opponent.match_id] = {
            "status": "found",
            "match_id": final_match_id,
            "opponent_id": seeker.user_id,
            "is_bot": False,
            "user_id": opponent.user_id,
            "player_ids": player_ids,
            "player_decks": player_decks,
            "starting_player_id": starting_player_id,
            "game_mode": game_mode,
            "mode_config": match_modes.serialize_mode_config(match_modes.resolve_mode_config(game_mode)),
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
            "player_ids": player_ids,
            "starting_player_id": starting_player_id,
            # Сохраняем выбранные пресеты для каждого игрока
            "player_decks": player_decks,
            # Для совместимости со старым форматом возвращаем user_id/opponent_id,
            # даже если запрос к этому ключу не привязан к конкретному игроку.
            "user_id": seeker.user_id,
            "opponent_id": opponent.user_id,
            "is_bot": False,
            "game_mode": game_mode,
            "mode_config": match_modes.serialize_mode_config(match_modes.resolve_mode_config(game_mode)),
        }
        self._register_match_aliases({seeker.match_id, opponent.match_id, final_match_id})

        # Останавливаем фоновые задачи обоих игроков, если они были
        for match_id in (seeker.match_id, opponent.match_id):
            task = self._tasks.pop(match_id, None)
            if task:
                task.cancel()

        # Пробуем зарегистрировать бой в движке, если он предоставлен
        await self._safe_call_battle_engine(final_match_id, player_ids, is_bot=False)
        return dict(self._matches[seeker.match_id])

    async def _create_bot_match(
        self,
        user_id: int,
        trophies: int,
        user_max_level: int,
        selected_deck_id: int | None = None,
        game_mode: str = "classic",
    ) -> Dict[str, Any]:
        """
        Генерация боя против бота.
        Приоритет - использовать bot_factory.create_match, иначе падаем на генератор ботов.
        
        Args:
            user_id: ID игрока
            trophies: Трофеи игрока
            user_max_level: Максимальный уровень карт выбранной колоды игрока
            selected_deck_id: ID выбранного пресета колоды (опционально)
        """
        mode_config = match_modes.resolve_mode_config(game_mode)
        if not mode_config.available:
            return {"status": "error", **match_modes.mode_unavailable_payload(game_mode)}
        game_mode = mode_config.mode_id

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
                maybe_result = factory_callable(user_id=user_id, trophies=trophies)
                result = await maybe_result if asyncio.iscoroutine(maybe_result) else maybe_result
                if not isinstance(result, dict) or not result:
                    # Если фабрика вернула пустоту, логируем и переходим на fallback
                    raise ValueError("bot_factory.create_match вернула пустой результат")
                # Гарантируем минимальный набор ключей
                match_id = str(result.get("match_id") or uuid.uuid4())
                opponent_id = result.get("opponent_id", result.get("bot_id"))
                if opponent_id is None:
                    opponent_id = -1
                player_ids = [user_id, opponent_id]
                starting_player_id = self._choose_starting_player_id(player_ids)
                payload = {
                    "status": "found",
                    "match_id": match_id,
                    "opponent_id": opponent_id,
                    "is_bot": True,
                    "user_id": user_id,
                    "game_mode": game_mode,
                    "mode_config": match_modes.serialize_mode_config(mode_config),
                    "player_ids": player_ids,
                    "starting_player_id": starting_player_id,
                    "selected_deck_id": selected_deck_id,
                    "player_decks": {
                        str(user_id): selected_deck_id,
                    },
                    "bot_info": result.get("bot_info", {}),
                }
                # Сохраняем для поллинга
                async with self._lock:
                    self._matches[match_id] = payload
                    self._register_match_aliases({match_id})
                return payload
            except Exception as exc:  # noqa: BLE001 - нам нужно логировать любые сбои фабрики
                self._logger.error("bot_factory.create_match упал: %s", exc, exc_info=True)

        # Fallback: строим бота сами через BotGenerator
        opponent_id = -1
        bot_name = None
        bot_avatar_url = None
        bot_difficulty = "lite"
        bot_deck_ids = []
        bot_cosmetics: dict[str, Any] = {}
        bot_extra_pass: str | None = None
        bot_trophies = 0
        bot_difficulty_label = bot_difficulty
        bot_strength_tier = bot_difficulty
        bot_brain_profile = None
        bot_selection = None
        bot_temperature = None
        bot_card_level_policy = None
        bot_deck_policy = None
        try:
            if isinstance(self._bot_factory, BotGenerator):
                bot_profile = await self._bot_factory.get_or_create_bot(
                    player_id=user_id,
                    player_trophies=trophies,
                )
                opponent_id = bot_profile.get("user_id", -1)
                bot_name = bot_profile.get("name")
                bot_avatar_url = bot_profile.get("avatar_url")
                bot_difficulty = bot_profile.get("difficulty", "lite")
                bot_deck_ids = bot_profile.get("deck_ids", [])
                bot_cosmetics = bot_profile.get("cosmetics", {})
                bot_extra_pass = bot_profile.get("extra_pass")
                bot_trophies = bot_profile.get("trophies", 0)
                bot_difficulty_label = bot_profile.get("difficulty_label", bot_difficulty)
                bot_strength_tier = bot_profile.get("strength_tier", bot_difficulty)
                bot_brain_profile = bot_profile.get("brain_profile")
                bot_selection = bot_profile.get("selection")
                bot_temperature = bot_profile.get("temperature")
                bot_card_level_policy = bot_profile.get("card_level_policy")
                bot_deck_policy = bot_profile.get("deck_policy")
        except Exception as exc:
            self._logger.error("BotGenerator fallback не удался: %s", exc, exc_info=True)

        # Per-card levels from difficulty table (match-scoped, на основе player_max_level)
        from ai.bot_factory import BotGenerator as _BG
        card_levels = _BG._build_bot_card_levels(
            bot_difficulty, user_max_level, len(bot_deck_ids)
        )

        match_id = str(uuid.uuid4())
        player_ids = [user_id, opponent_id]
        starting_player_id = self._choose_starting_player_id(player_ids)
        
        bot_info_payload = {
            "name": bot_name,
            "avatar_url": bot_avatar_url,
            "user_id": opponent_id,
            "difficulty": bot_difficulty,
            "difficulty_label": bot_difficulty_label,
            "strength_tier": bot_strength_tier,
            "brain_profile": bot_brain_profile,
            "selection": bot_selection,
            "temperature": bot_temperature,
            "card_level_policy": bot_card_level_policy,
            "deck_policy": bot_deck_policy,
            "deck_ids": bot_deck_ids,
            "card_levels": card_levels,
            "trophies": bot_trophies,
            "cosmetics": bot_cosmetics,
            "extra_pass": bot_extra_pass,
        }
        
        self._logger.debug(
            "_create_bot_match creating payload with bot_info=%s",
            bot_info_payload
        )
        
        payload = {
            "status": "found",
            "match_id": match_id,
            "opponent_id": opponent_id,
            "is_bot": True,
            "user_id": user_id,
            # Единый список участников для удобства server-side инициализации боя.
            "player_ids": player_ids,
            "starting_player_id": starting_player_id,
            # ВСЕГДА добавляем bot_info для передачи в server.py
            "bot_info": bot_info_payload,
            "selected_deck_id": selected_deck_id,
            "game_mode": game_mode,
            "mode_config": match_modes.serialize_mode_config(mode_config),
            "player_decks": {
                str(user_id): selected_deck_id,
            }
        }

        async with self._lock:
            self._matches[match_id] = payload
            self._register_match_aliases({match_id})

        await self._safe_call_battle_engine(match_id, player_ids, is_bot=True)
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

    async def prune_expired_matches(
        self,
        *,
        now: float | None = None,
        ttl_seconds: float = 600.0,
    ) -> int:
        """Drop terminal matchmaking records and their aliases after TTL."""
        now_value = time.monotonic() if now is None else float(now)
        removed_groups = 0
        async with self._lock:
            for match_id, payload in list(self._matches.items()):
                if payload.get("status") not in {"canceled", "error", "finished"}:
                    continue
                updated_at = self._match_updated_at.get(match_id, payload.get("terminal_at", now_value))
                if now_value - float(updated_at) < ttl_seconds:
                    continue
                if match_id not in self._matches:
                    continue
                self._delete_match_aliases(match_id)
                removed_groups += 1
        return removed_groups

    def _drop_existing(self, user_id: int) -> None:
        """Удаляет старые записи игрока из очереди/кеша статусов и отменяет фоновые задачи."""
        match_ids_to_drop = [entry.match_id for entry in self._queue if entry.user_id == user_id]
        self._queue = [entry for entry in self._queue if entry.user_id != user_id]

        for match_id in match_ids_to_drop:
            self._delete_match_aliases(match_id)

        for match_id, data in list(self._matches.items()):
            if data.get("user_id") == user_id:
                self._delete_match_aliases(match_id)
