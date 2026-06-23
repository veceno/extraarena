from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.bot_factory import BotGenerator
from infrastructure.database import Database
from infrastructure.match_config import (
    MM_BOT_TIMEOUT,
    MM_SOFT_START_BOT_DELAY_RANGE,
    QUEUE_POLL_INTERVAL,
    SEARCH_WINDOWS,
)
from infrastructure.config import DECK_SIZE, MM_TROPHY_LIMIT_CLASSIC
import infrastructure.match_modes as match_modes


@dataclass
class StreakAdjustment:
    """One-shot streak pressure for matchmaking and bot difficulty."""

    active: bool = False
    direction: str | None = None
    n: int = 0


@dataclass(frozen=True)
class TrophySearchWindow:
    """Directional trophy window relative to the seeker."""

    min_delta: int
    max_delta: int
    magnitude: int


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
    search_generation: int = 0
    streak_adjustment: StreakAdjustment = field(default_factory=StreakAdjustment)
    player_display_name: str | None = None
    active_window: TrophySearchWindow = field(
        default_factory=lambda: TrophySearchWindow(-SEARCH_WINDOWS[0], SEARCH_WINDOWS[0], SEARCH_WINDOWS[0])
    )


class Matchmaker:
    """
    Простая in-memory реализация матчмейкера с приоритетом «мгновенного старта» (Soft Start).

    Допущения:
    - bot_factory предоставляет create_match или BotGenerator для сборки бота.
    - battle_engine (если передан) умеет регистрировать бои через create_match.
    - Хранилище - оперативное: рестарт процесса очистит очередь и кеш.
    """

    def __init__(
        self,
        db: Database,
        bot_factory: Any,
        battle_engine: Any | None = None,
        soft_start_bot_delay_range: tuple[float, float] | None = None,
    ) -> None:
        self._db = db
        self._bot_factory = bot_factory
        self._battle_engine = battle_engine
        self._soft_start_bot_delay_range = (
            soft_start_bot_delay_range
            if soft_start_bot_delay_range is not None
            else MM_SOFT_START_BOT_DELAY_RANGE
        )

        self._queue: List[QueueEntry] = []
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._match_aliases: Dict[str, set[str]] = {}
        self._match_updated_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._soft_start_tasks: Dict[tuple[Any, ...], asyncio.Task] = {}
        self._user_search_generations: Dict[int, int] = {}
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

    @staticmethod
    def _streak_adjustment_for(
        trophies: int,
        kind: str | None,
        length: int,
    ) -> StreakAdjustment:
        trophies_value = max(0, int(trophies or 0))
        streak_length = max(0, int(length or 0))
        if kind not in {"win", "loss", "lose"} or streak_length <= 0:
            return StreakAdjustment()

        normalized_kind = "loss" if kind == "lose" else kind
        if trophies_value < MM_TROPHY_LIMIT_CLASSIC:
            loss_threshold = 2
            win_threshold = 5
        elif trophies_value >= 5000:
            loss_threshold = None
            win_threshold = 3
        else:
            loss_threshold = 5
            win_threshold = 3

        threshold = win_threshold if normalized_kind == "win" else loss_threshold
        if threshold is None or streak_length < threshold or streak_length % threshold != 0:
            return StreakAdjustment()

        n = streak_length // threshold
        return StreakAdjustment(
            active=True,
            direction="up" if normalized_kind == "win" else "down",
            n=n,
        )

    @staticmethod
    def _search_window_for_adjustment(
        window: int,
        adjustment: StreakAdjustment | None = None,
    ) -> TrophySearchWindow:
        magnitude = int(window)
        if adjustment and adjustment.active and adjustment.direction == "down":
            return TrophySearchWindow(-magnitude, 0, magnitude)
        if adjustment and adjustment.active and adjustment.direction == "up":
            return TrophySearchWindow(0, magnitude, magnitude)
        return TrophySearchWindow(-magnitude, magnitude, magnitude)

    @staticmethod
    def _windows_for_adjustment(adjustment: StreakAdjustment | None = None) -> tuple[int, ...]:
        if not adjustment or not adjustment.active or adjustment.n <= 1:
            return SEARCH_WINDOWS
        start_idx = min(len(SEARCH_WINDOWS) - 1, max(0, int(adjustment.n) - 1))
        return SEARCH_WINDOWS[start_idx:]

    async def _load_streak_adjustment(self, user_id: int, trophies: int) -> StreakAdjustment:
        streak_reader = getattr(self._db, "get_current_result_streak", None)
        if not callable(streak_reader):
            return StreakAdjustment()
        try:
            maybe_streak = streak_reader(user_id)
            streak = await maybe_streak if asyncio.iscoroutine(maybe_streak) else maybe_streak
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to load result streak for user_id=%s: %s", user_id, exc, exc_info=True)
            return StreakAdjustment()
        if not isinstance(streak, dict):
            return StreakAdjustment()
        return self._streak_adjustment_for(
            trophies,
            streak.get("kind") or streak.get("current_streak_result"),
            int(streak.get("length") or streak.get("current_streak_count") or 0),
        )

    def _choose_starting_player_id(self, player_ids: list[int]) -> int:
        """Choose the first turn owner independently from p1/p2 ordering."""
        return int(random.choice(player_ids))

    def _soft_start_delay_seconds(self) -> float:
        try:
            lo_raw, hi_raw = self._soft_start_bot_delay_range
            lo = max(0.0, float(lo_raw))
            hi = max(0.0, float(hi_raw))
        except (TypeError, ValueError):
            return 0.0
        if hi < lo:
            lo, hi = hi, lo
        if hi <= 0:
            return 0.0
        if hi == lo:
            return lo
        return random.uniform(lo, hi)

    def _advance_user_generation_locked(self, user_id: int) -> int:
        user_id_int = int(user_id)
        generation = int(self._user_search_generations.get(user_id_int, 0) or 0) + 1
        self._user_search_generations[user_id_int] = generation
        return generation

    def _current_user_generation_locked(self, user_id: int) -> int:
        return int(self._user_search_generations.get(int(user_id), 0) or 0)

    def _is_current_seeker_locked(self, seeker: QueueEntry) -> bool:
        if seeker.matched:
            return False
        if self._current_user_generation_locked(seeker.user_id) != int(seeker.search_generation or 0):
            return False
        return any(entry is seeker or entry.match_id == seeker.match_id for entry in self._queue)

    async def _create_soft_start_bot_match(
        self,
        user_id: int,
        trophies: int,
        user_max_level: int,
        selected_deck_id: int | None = None,
        game_mode: str = "classic",
        search_generation: int = 0,
        streak_adjustment: StreakAdjustment | None = None,
        player_display_name: str | None = None,
    ) -> Dict[str, Any]:
        delay = self._soft_start_delay_seconds()
        if delay > 0:
            self._logger.debug(
                "PvE soft-start delay %.2fs for user_id=%s mode=%s",
                delay,
                user_id,
                game_mode,
            )
            await asyncio.sleep(delay)
        async with self._lock:
            if self._current_user_generation_locked(user_id) != int(search_generation or 0):
                return {
                    "status": "canceled",
                    "user_id": user_id,
                    "selected_deck_id": selected_deck_id,
                    "game_mode": game_mode,
                    "error": "search_replaced",
                }
        payload = await self._create_bot_match(
            user_id,
            trophies,
            user_max_level,
            selected_deck_id,
            game_mode=game_mode,
            streak_adjustment=streak_adjustment,
            player_display_name=player_display_name,
        )
        async with self._lock:
            if self._current_user_generation_locked(user_id) != int(search_generation or 0):
                match_id = str(payload.get("match_id") or "")
                if match_id:
                    self._delete_match_aliases(match_id)
                return {
                    "status": "canceled",
                    "user_id": user_id,
                    "selected_deck_id": selected_deck_id,
                    "game_mode": game_mode,
                    "error": "search_replaced",
                }
        return payload

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

        streak_adjustment = await self._load_streak_adjustment(user_id, trophies)

        player_display_name: str | None = None
        try:
            reader = getattr(self._db, "get_display_name", None)
            if callable(reader):
                maybe_name = reader(user_id)
                resolved = await maybe_name if asyncio.iscoroutine(maybe_name) else maybe_name
                if isinstance(resolved, str) and resolved.strip():
                    player_display_name = resolved.strip()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to load display name for user_id=%s: %s", user_id, exc, exc_info=True)

        # Soft Start: only if bots_allowed and trophies < threshold
        if mode_config.classic.bots_allowed and trophies < MM_TROPHY_LIMIT_CLASSIC:
            self._logger.info(
                "PvE soft-start: user_id=%s trophies=%s max_level=%s deck=%s",
                user_id, trophies, user_max_level, selected_deck_id,
            )
            soft_key = (
                int(user_id),
                canonical,
                selected_deck_id,
                int(trophies or 0),
                int(user_max_level or 0),
                streak_adjustment.direction,
                int(streak_adjustment.n or 0),
            )
            try:
                async with self._lock:
                    existing = self._find_active_match_for_user_locked(user_id, canonical)
                    if existing:
                        return existing
                    task = self._soft_start_tasks.get(soft_key)
                    if task is None:
                        self._drop_existing(user_id)
                        search_generation = self._current_user_generation_locked(user_id)
                        self._logger.debug(
                            "Calling bot_factory.create_match user_id=%s trophies=%s max_level=%s",
                            user_id,
                            trophies,
                            user_max_level,
                        )
                        task = asyncio.create_task(
                            self._create_soft_start_bot_match(
                                user_id,
                                trophies,
                                user_max_level,
                                selected_deck_id,
                                game_mode=canonical,
                                search_generation=search_generation,
                                streak_adjustment=streak_adjustment,
                                player_display_name=player_display_name,
                            )
                        )
                        self._soft_start_tasks[soft_key] = task

                match_payload = await task
                async with self._lock:
                    if self._soft_start_tasks.get(soft_key) is task:
                        self._soft_start_tasks.pop(soft_key, None)
                self._logger.debug("Bot factory returned payload: %s", match_payload)
                return match_payload
            except asyncio.CancelledError:
                return {
                    "status": "canceled",
                    "user_id": user_id,
                    "selected_deck_id": selected_deck_id,
                    "game_mode": canonical,
                    "error": "search_replaced",
                }
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
            streak_adjustment=streak_adjustment,
            player_display_name=player_display_name,
        )

        async with self._lock:
            # Если игрок уже в очереди, очищаем старую запись, чтобы не было дублей
            self._drop_existing(user_id)
            seeker.search_generation = self._current_user_generation_locked(user_id)
            # Мгновенно пытаемся найти соперника в уже существующей очереди
            first_window = self._search_window_for_adjustment(
                self._windows_for_adjustment(seeker.streak_adjustment)[0],
                seeker.streak_adjustment,
            )
            seeker.active_window = first_window
            opponent = self._find_candidate(seeker, first_window)
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
                if match.get("status") == "found":
                    self._touch_matches(self._aliases_for(str(match_id)))
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
            for key in (
                "user_id",
                "opponent_id",
                "player_ids",
                "player_decks",
                "starting_player_id",
                "is_bot",
                "bot_info",
                "selected_deck_id",
                "mode_config",
            ):
                if existing.get(key) is not None:
                    payload[key] = existing.get(key)
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
        search_windows = self._windows_for_adjustment(seeker.streak_adjustment)
        per_window_timeout = effective_max_wait / len(search_windows)

        for raw_window in search_windows:
            window = self._search_window_for_adjustment(raw_window, seeker.streak_adjustment)
            async with self._lock:
                if not self._is_current_seeker_locked(seeker):
                    return
                seeker.active_window = window
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
            if not self._is_current_seeker_locked(seeker):
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
            if not self._is_current_seeker_locked(seeker):
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
                streak_adjustment=seeker.streak_adjustment,
                player_display_name=seeker.player_display_name,
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
            if self._current_user_generation_locked(seeker.user_id) != int(seeker.search_generation or 0):
                stale_match_id = str(bot_match.get("match_id") or "")
                if stale_match_id:
                    self._delete_match_aliases(stale_match_id)
                return
            self._matches[seeker.match_id] = bot_match
            aliases = {seeker.match_id, str(bot_match.get("match_id", seeker.match_id))}
            self._register_match_aliases(aliases)

    @staticmethod
    def _window_allows_pair(seeker: QueueEntry, opponent: QueueEntry, window: TrophySearchWindow) -> bool:
        delta = int(opponent.trophies) - int(seeker.trophies)
        return int(window.min_delta) <= delta <= int(window.max_delta)

    def _find_candidate(self, seeker: QueueEntry, window: int | TrophySearchWindow) -> Optional[QueueEntry]:
        """Поиск соперника в очереди по допуску по трофеям."""
        seeker_window = (
            self._search_window_for_adjustment(window, seeker.streak_adjustment)
            if isinstance(window, int)
            else window
        )
        candidates: list[QueueEntry] = []
        for entry in self._queue:
            if entry.user_id == seeker.user_id or entry.matched:
                continue
            if entry.canonical_mode != seeker.canonical_mode:
                continue
            if not self._window_allows_pair(seeker, entry, seeker_window):
                continue
            if not self._window_allows_pair(entry, seeker, entry.active_window):
                continue
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
        difficulty_override: str | None = None,
        difficulty: str | None = None,
        streak_adjustment: StreakAdjustment | None = None,
        player_display_name: str | None = None,
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
        requested_difficulty = difficulty_override if difficulty_override is not None else difficulty
        base_difficulty = BotGenerator._calc_difficulty(trophies)

        def _bot_match_create_failed(message: str = "Не удалось подготовить бой с ботом. Попробуйте начать поиск заново.") -> Dict[str, Any]:
            return {
                "status": "canceled",
                "match_id": str(uuid.uuid4()),
                "user_id": user_id,
                "is_bot": True,
                "game_mode": game_mode,
                "mode_config": match_modes.serialize_mode_config(mode_config),
                "selected_deck_id": selected_deck_id,
                "error": "bot_match_create_failed",
                "message": message,
                "terminal_at": time.monotonic(),
            }

        try:
            normalized_requested_difficulty = (
                BotGenerator._normalize_difficulty(requested_difficulty)
                if requested_difficulty is not None
                else None
            )
        except KeyError:
            return _bot_match_create_failed()

        streak_difficulty: str | None = None
        if normalized_requested_difficulty is None and streak_adjustment and streak_adjustment.active:
            shifted = BotGenerator._shift_difficulty_by_streak(
                base_difficulty,
                streak_adjustment.direction,
                streak_adjustment.n,
            )
            if shifted != base_difficulty:
                streak_difficulty = shifted
        effective_difficulty_override = normalized_requested_difficulty or streak_difficulty

        def _difficulty_for_bot() -> str:
            if normalized_requested_difficulty is not None:
                return normalized_requested_difficulty
            if streak_difficulty is not None:
                return BotGenerator._normalize_difficulty(streak_difficulty)
            return base_difficulty

        def _normalize_bot_info(bot_info: dict[str, Any]) -> dict[str, Any]:
            return BotGenerator.normalize_bot_info(
                bot_info,
                player_trophies=trophies,
                user_max_level=user_max_level,
                difficulty=requested_difficulty or _difficulty_for_bot(),
            )

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
                factory_kwargs: dict[str, Any] = {
                    "user_id": user_id,
                    "trophies": trophies,
                    "user_max_level": user_max_level,
                    "difficulty": effective_difficulty_override,
                    "difficulty_override": effective_difficulty_override,
                    "player_display_name": player_display_name,
                }
                try:
                    signature = inspect.signature(factory_callable)
                    accepts_kwargs = any(
                        param.kind == inspect.Parameter.VAR_KEYWORD
                        for param in signature.parameters.values()
                    )
                    if not accepts_kwargs:
                        factory_kwargs = {
                            key: value
                            for key, value in factory_kwargs.items()
                            if key in signature.parameters
                        }
                except (TypeError, ValueError):
                    factory_kwargs = {"user_id": user_id, "trophies": trophies}

                maybe_result = factory_callable(**factory_kwargs)
                result = await maybe_result if asyncio.iscoroutine(maybe_result) else maybe_result
                if not isinstance(result, dict) or not result:
                    # Если фабрика вернула пустоту, логируем и переходим на fallback
                    raise ValueError("bot_factory.create_match вернула пустой результат")
                opponent_id = result.get("opponent_id")
                bot_info = result.get("bot_info")
                bot_deck_ids = (bot_info or {}).get("deck_ids") if isinstance(bot_info, dict) else None
                if opponent_id is None or not isinstance(bot_info, dict) or len(bot_deck_ids or []) != DECK_SIZE:
                    raise ValueError("bot_factory.create_match вернула неполный bot payload")
                bot_info = _normalize_bot_info(bot_info)
                match_id = str(result.get("match_id") or uuid.uuid4())
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
                    "bot_info": bot_info,
                }
                # Сохраняем для поллинга
                async with self._lock:
                    self._matches[match_id] = payload
                    self._register_match_aliases({match_id})
                return payload
            except ValueError as exc:
                self._logger.warning("bot_factory.create_match returned unusable payload: %s", exc)
            except Exception as exc:  # noqa: BLE001 - нам нужно логировать любые сбои фабрики
                self._logger.error("bot_factory.create_match упал: %s", exc, exc_info=True)

        # Fallback: строим бота сами через BotGenerator
        if not isinstance(self._bot_factory, BotGenerator):
            self._logger.error("Bot match cannot be created: no BotGenerator fallback is configured")
            return _bot_match_create_failed()

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
            bot_profile = await self._bot_factory.get_or_create_bot(
                player_id=user_id,
                player_trophies=trophies,
                difficulty_override=effective_difficulty_override,
                player_display_name=player_display_name,
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
            return _bot_match_create_failed()

        if len(bot_deck_ids or []) != DECK_SIZE:
            self._logger.error(
                "BotGenerator returned invalid bot deck size for user=%s bot=%s size=%s",
                user_id,
                opponent_id,
                len(bot_deck_ids or []),
            )
            return _bot_match_create_failed()

        # Per-card levels from difficulty table (match-scoped, на основе player_max_level)
        card_levels = BotGenerator._build_bot_card_levels(
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
        """Drop terminal or stale found matchmaking records and their aliases after TTL."""
        now_value = time.monotonic() if now is None else float(now)
        removed_groups = 0
        async with self._lock:
            for match_id, payload in list(self._matches.items()):
                if payload.get("status") not in {"canceled", "error", "finished", "found"}:
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
        self._advance_user_generation_locked(user_id)
        user_id_int = int(user_id)
        for soft_key, task in list(self._soft_start_tasks.items()):
            if int(soft_key[0]) != user_id_int:
                continue
            self._soft_start_tasks.pop(soft_key, None)
            if task and not task.done():
                task.cancel()
        match_ids_to_drop = [entry.match_id for entry in self._queue if entry.user_id == user_id]
        self._queue = [entry for entry in self._queue if entry.user_id != user_id]

        for match_id in match_ids_to_drop:
            self._delete_match_aliases(match_id)

        for match_id, data in list(self._matches.items()):
            if data.get("user_id") == user_id:
                self._delete_match_aliases(match_id)
