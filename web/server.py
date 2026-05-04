from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl

from aiohttp import web
import socketio

from bot.constants import ADMIN_ID
from infrastructure.config import DECK_SIZE
from infrastructure.database import Card, Database
from ai.bot_factory import BotGenerator
from ai.bot_ai import BotAI
from ai.bot_brain import BerserkInference
from battle_engine import BattleEngine, BattleEventEmitter
from infrastructure.matchmaking import Matchmaker
from infrastructure.case_system import roll_tier_upgrade, process_case_opening
from infrastructure.payments_logic import process_successful_payment

WEBAPP_DIR = Path(__file__).resolve().parents[1] / "webapp"
DESIGN_ASSETS_DIR = Path(__file__).resolve().parents[1] / "DesignAssets"
# Единый URL-путь к изображениям карт, чтобы фронт и боевая логика не зависели
# от устаревших image_file_id.
CARD_IMAGE_URL_PREFIX = "/DesignAssets/Cards"

# Глобальный кеш активных боёв: match_id -> экземпляр движка
ACTIVE_MATCHES: dict[str, BattleEngine] = {}

# Глобальный эмиттер событий для всех боёв
BATTLE_EVENT_EMITTER = BattleEventEmitter()

# Словарь для отслеживания связей sid -> {match_id, user_id}
SID_TO_MATCH: dict[str, dict[str, Any]] = {}

# ONNX-мозг Берсерка для умных ботов (ID начинается на 810416)
# Модель: ai/models/extra-lr-v1.onnx (997 фич -> 200 действий)
# Автоматически активируется для ботов с user_id начинающимся на 810416
# Fallback на rule-based BotAI если модель не загрузилась
BERSERK_BRAIN: Optional[BerserkInference] = None

# Socket.io сервер для WebSocket подключений
sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)


def initialize_game_services(db: Database, battle_engine: BattleEngine | None = None) -> dict[str, Any]:
    """
    Единая точка инициализации игровых сервисов.

    Создает бота-генератор, матчмейкер и, при наличии, боевой движок.
    Возвращает словарь для DI в web.Application.
    """
    # Генератор ботов всегда требуется матчмейкеру
    bot_generator = BotGenerator(db)

    # Боевой движок можем принять извне или создать здесь.
    engine = battle_engine or BattleEngine(db, ACTIVE_MATCHES)

    # Матчмейкер получает ссылки на БД, генератор ботов и опциональный движок.
    matchmaker = Matchmaker(db, bot_generator, engine)

    return {
        "bot_generator": bot_generator,
        "battle_engine": engine,
        "matchmaker": matchmaker,
        "active_matches": ACTIVE_MATCHES,
        "event_emitter": BATTLE_EVENT_EMITTER,
    }


# ---------------------------------------------------------------------- #
# Обработчики Socket.io для реального времени
# ---------------------------------------------------------------------- #
@sio.event
async def connect(sid: str, environ: dict[str, Any]) -> None:
    """
    Обработчик подключения клиента к Socket.io.
    Логируем подключение и сохраняем сессию.
    """
    match_id = environ.get("HTTP_MATCH_ID") or environ.get("match_id") or "unknown"
    logging.info("[SOCKET] connect sid=%s match_id=%s", sid, match_id)


@sio.event
async def disconnect(sid: str) -> None:
    """
    Обработчик отключения клиента.
    При разрыве соединения мгновенно переводит игрока в статус AFK.
    """
    from core.state import ReplacementStatus
    
    # Проверяем, есть ли данные по этому sid
    session_data = SID_TO_MATCH.get(sid)
    
    if session_data:
        match_id = session_data.get("match_id")
        user_id = session_data.get("user_id")
        
        # Находим активный BattleEngine для этого матча
        engine = ACTIVE_MATCHES.get(match_id)
        
        if engine and user_id:
            # Переводим игрока в статус AFK через метод engine
            if hasattr(engine, 'set_player_replacement_status'):
                engine.set_player_replacement_status(user_id, ReplacementStatus.AFK)
                logging.warning(
                    "[SOCKET] Player %s disconnected. Status set to AFK in match %s",
                    user_id, match_id
                )
            
            # Проверяем, нужно ли запустить бота (если отключился игрок, чей ход сейчас)
            if hasattr(engine, 'get_current_player_id'):
                current_player_id = engine.get_current_player_id()
                if current_player_id == user_id:
                    logging.info(
                        "[SOCKET] Disconnected player %s was on turn. Starting bot replacement.",
                        user_id
                    )
                    # Запускаем бота асинхронно
                    asyncio.create_task(check_and_run_bot(match_id, ACTIVE_MATCHES))
        
        # Удаляем sid из словаря
        del SID_TO_MATCH[sid]
    else:
        logging.info("[SOCKET] disconnect sid=%s match_id=unknown", sid)


@sio.event
async def join_match(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент присоединяется к комнате матча.
    
    Параметры:
        - match_id: идентификатор матча
        - user_id: идентификатор игрока (строка по раздел 8.2 MULTIPLAYER.md)
    """
    try:
        match_id = str(data.get("match_id", ""))
        user_id = str(data.get("user_id", ""))
        
        if not match_id or not user_id:
            await sio.emit("error", {"message": "match_id и user_id обязательны"}, to=sid)
            return
        
        # Создаём комнату для матча (формат: match:<match_id>)
        # ВАЖНО: используем единое имя комнаты = match_id (без префикса),
        # потому что в других местах сервера эмиты идут в `room=match_id`.
        # Раньше здесь был `match:{match_id}`, из-за чего часть событий "терялась"
        # (клиент сидит в одной комнате, сервер эмитит в другую).
        room_name = str(match_id)
        await sio.enter_room(sid, room_name)
        
        # Сохраняем связь sid -> {match_id, user_id} для обработки disconnect
        SID_TO_MATCH[sid] = {"match_id": match_id, "user_id": int(user_id)}
        
        logging.info(
            "[SOCKET] join_match sid=%s match_id=%s user_id=%s",
            sid,
            match_id,
            user_id,
        )
        
        # Отправляем подтверждение клиенту
        await sio.emit("joined_match", {"match_id": match_id, "user_id": user_id}, to=sid)
        
    except Exception as exc:
        logging.error(f"Ошибка при присоединении к матчу: {exc}")
        await sio.emit("error", {"message": str(exc)}, to=sid)


@sio.event
async def leave_match(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент покидает комнату матча.
    """
    try:
        match_id = str(data.get("match_id", ""))
        if not match_id:
            return
        
        room_name = str(match_id)
        await sio.leave_room(sid, room_name)
        
        logging.info("[SOCKET] leave_match sid=%s match_id=%s", sid, match_id)
        
    except Exception as exc:
        logging.error(f"Ошибка при выходе из матча: {exc}")


@sio.event
async def client_ready(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент сигнализирует о том, что загрузил состояние боя и готов к игре.
    После этого сервер может запустить бота, если тот ходит первым.
    
    Параметры:
        - match_id: идентификатор матча
        - user_id: идентификатор игрока
    """
    logger = logging.getLogger(__name__)
    try:
        match_id = str(data.get("match_id", ""))
        user_id = data.get("user_id", "")
        logger.info(
            "[SOCKET] client_ready sid=%s match_id=%s user_id=%s",
            sid,
            match_id or "unknown",
            user_id,
        )
        
        if not match_id:
            logger.warning("client_ready: получен запрос без match_id от sid=%s", sid)
            await sio.emit("error", {"message": "match_id обязателен"}, to=sid)
            return
        
        # ИСПРАВЛЕНО: Используем глобальный ACTIVE_MATCHES вместо обращения к sio.server
        engine = ACTIVE_MATCHES.get(match_id)
        
        if not engine:
            logger.warning("client_ready: движок не найден для match_id=%s", match_id)
            await sio.emit("error", {"message": "Match not found"}, to=sid)
            return
        
        # Помечаем клиента как готового
        if hasattr(engine, 'mark_client_ready'):
            engine.mark_client_ready()
            logger.info("client_ready: клиент готов для match_id=%s, user_id=%s", match_id, user_id)
            print(f"!!! [SERVER] client_ready: клиент готов для match_id={match_id}, user_id={user_id}")
        else:
            logger.warning("client_ready: движок не имеет метода mark_client_ready для match_id=%s", match_id)
        
        # Проверяем и запускаем бота, если нужно
        try:
            await check_and_run_bot(match_id, ACTIVE_MATCHES)
        except Exception as exc:
            logger.error("client_ready: ошибка при запуске check_and_run_bot для match_id=%s: %s", match_id, exc, exc_info=True)
            print(f"!!! [SERVER] client_ready: ошибка check_and_run_bot: {exc}")
        
        # Отправляем подтверждение клиенту
        await sio.emit("client_ready_ack", {"match_id": match_id}, to=sid)
        
    except Exception as exc:
        logger.error("client_ready: ошибка обработки: %s", exc, exc_info=True)
        await sio.emit("error", {"message": str(exc)}, to=sid)


def calculate_trophy_delta(
    current_trophies: int,
    is_winner: bool,
    status: "ReplacementStatus"
) -> tuple[int, str, dict]:
    """
    Динамический расчёт изменения трофеев на основе тира прогрессии.
    
    Args:
        current_trophies: Текущее количество трофеев игрока
        is_winner: True если игрок победил, False если проиграл
        status: Статус замены (ACTIVE/AFK/SURRENDERED)
    
    Returns:
        Кортеж (trophy_delta, tier_name, tier_data)
    """
    from infrastructure.config import TROPHY_TIERS
    from core.state import ReplacementStatus
    import random
    
    # Определяем тир игрока
    tier_name = None
    tier_data = None
    for name, data in TROPHY_TIERS.items():
        if current_trophies in data["range"]:
            tier_name = name
            tier_data = data
            break
    
    # Fallback на последний тир если выше максимума
    if tier_data is None:
        tier_name = "master"
        tier_data = TROPHY_TIERS["master"]
    
    # Если победитель AFK/SURRENDERED -> обнуляем награды
    if is_winner and status in (ReplacementStatus.AFK, ReplacementStatus.SURRENDERED):
        return 0, tier_name, tier_data
    
    # Рассчитываем дельту трофеев
    if is_winner:
        # Победа: случайное значение из диапазона win
        win_min, win_max = tier_data["win"]
        delta = random.randint(win_min, win_max)
    else:
        # Поражение: случайное значение из диапазона loss (отрицательное)
        loss_min, loss_max = tier_data["loss"]
        
        # Если игрок SURRENDERED -> максимальный штраф
        if status == ReplacementStatus.SURRENDERED:
            delta = -loss_max
        else:
            delta = -random.randint(loss_min, loss_max)
    
    return delta, tier_name, tier_data


def calculate_coins_reward(
    tier_data: dict,
    is_winner: bool,
    status: "ReplacementStatus"
) -> int:
    """
    Динамический расчёт награды монетами на основе тира.
    
    Args:
        tier_data: Данные тира из TROPHY_TIERS
        is_winner: True если игрок победил
        status: Статус замены (ACTIVE/AFK/SURRENDERED)
    
    Returns:
        Количество монет (0 если проиграл или AFK/SURRENDERED)
    """
    from core.state import ReplacementStatus
    import random
    
    # Монеты только победителю с активным статусом
    if not is_winner:
        return 0
    
    if status in (ReplacementStatus.AFK, ReplacementStatus.SURRENDERED):
        return 0
    
    # Случайное значение из диапазона coin_range
    coin_min, coin_max = tier_data["coin_range"]
    return random.randint(coin_min, coin_max)


async def _process_battle_end(
    app: web.Application, 
    match_id: str, 
    engine: Any, 
    winner_id: Optional[int]
) -> None:
    """
    Обработка завершения боя: начисление трофеев и сохранение результата.
    
    Args:
        app: Приложение aiohttp (для доступа к DB)
        match_id: ID матча
        engine: Экземпляр BattleEngine
        winner_id: ID победителя (None при ничье)
    """
    logger = logging.getLogger(__name__)
    db = app.get("db")
    
    if not db:
        logger.error("❌ Database not available for processing battle end")
        return
    
    # КРИТИЧНО: Проверяем флаг rewards_granted - если трофеи уже начислены, выходим
    if getattr(engine, 'rewards_granted', False):
        logger.info("⚠️ Rewards already granted for match %s, skipping duplicate trophy award", match_id)
        return
    
    # Определяем игроков
    p1_id = engine.p1_state.user_id
    p2_id = engine.p2_state.user_id
    
    # Проверяем, что оба игрока - реальные пользователи (не боты)
    try:
        p1_id_int = int(p1_id)
        p2_id_int = int(p2_id)
    except (ValueError, TypeError):
        logger.info("🤖 Battle involves bot, skipping trophy calculation")
        return
    
    # Получаем статусы замены игроков
    from core.state import ReplacementStatus
    p1_status = getattr(engine.p1_state, "replacement_status", ReplacementStatus.ACTIVE)
    p2_status = getattr(engine.p2_state, "replacement_status", ReplacementStatus.ACTIVE)
    
    # Определяем победителя и проигравшего
    if winner_id is None:
        logger.info("🤝 Draw - no trophies awarded")
        loser_id = None
        winner_trophy_delta = 0
        loser_trophy_delta = 0
    else:
        try:
            winner_id_int = int(winner_id)
        except (ValueError, TypeError):
            logger.error("❌ Invalid winner_id: %s", winner_id)
            return
        
        loser_id = p2_id_int if winner_id_int == p1_id_int else p1_id_int
        
        # Определяем статусы победителя и проигравшего
        winner_status = p1_status if winner_id_int == p1_id_int else p2_status
        loser_status = p2_status if winner_id_int == p1_id_int else p1_status
        
        # Получаем состояния игроков для проверки surrender_processed
        winner_state = engine.p1_state if winner_id_int == p1_id_int else engine.p2_state
        loser_state = engine.p2_state if winner_id_int == p1_id_int else engine.p1_state
        
        # Проверяем флаги surrender_processed
        winner_surrender_processed = getattr(winner_state, "surrender_processed", False)
        loser_surrender_processed = getattr(loser_state, "surrender_processed", False)
        
        # Получаем текущие трофеи из БД для расчёта тира
        try:
            winner_info = await db.get_user_info(winner_id_int)
            loser_info = await db.get_user_info(loser_id)
            
            winner_current_trophies = winner_info.get("trophies", 0) if winner_info else 0
            loser_current_trophies = loser_info.get("trophies", 0) if loser_info else 0
        except Exception as exc:
            logger.error("❌ Failed to get user trophies: %s", exc)
            winner_current_trophies = 0
            loser_current_trophies = 0
        
        # Динамический расчёт трофеев и монет на основе тиров
        winner_trophy_delta, winner_tier, winner_tier_data = calculate_trophy_delta(
            winner_current_trophies,
            is_winner=True,
            status=winner_status
        )
        
        loser_trophy_delta, loser_tier, loser_tier_data = calculate_trophy_delta(
            loser_current_trophies,
            is_winner=False,
            status=loser_status
        )
        
        # Рассчитываем награду монетами для победителя
        winner_coins_delta = calculate_coins_reward(
            winner_tier_data,
            is_winner=True,
            status=winner_status
        )
        
        logger.info(
            "🏆 Trophy calculation (dynamic): Winner=%s (+%d, trophies=%d, tier=%s, status=%s), "
            "Loser=%s (%d, trophies=%d, tier=%s, status=%s)",
            winner_id_int, winner_trophy_delta, winner_current_trophies, winner_tier, winner_status.value,
            loser_id, loser_trophy_delta, loser_current_trophies, loser_tier, loser_status.value
        )
        
        logger.info(
            "[ECONOMY] Winner %s earned %d coins (Tier: %s)",
            winner_id_int, winner_coins_delta, winner_tier
        )
        
        # Начисляем трофеи победителю (если не обнулены и не было surrender_processed)
        try:
            if winner_surrender_processed:
                logger.info("⚠️ Winner %s already processed surrender, skipping trophy award", winner_id_int)
            elif winner_trophy_delta > 0:
                winner_result = await db.update_user_trophies(winner_id_int, winner_trophy_delta)
                logger.info(
                    "✅ Winner trophies updated: %s -> %d (max: %d)",
                    winner_id_int, winner_result.get("trophies", 0), winner_result.get("max_trophies", 0)
                )
                
                # Сохраняем изменение трофеев в состоянии движка для отправки на фронт
                if not hasattr(engine, "_trophy_changes"):
                    engine._trophy_changes = {}
                engine._trophy_changes[winner_id_int] = winner_trophy_delta
                
                # ДОБАВЛЕНО: Сохраняем текущее количество трофеев напрямую из БД
                if not hasattr(engine, "_trophy_totals"):
                    engine._trophy_totals = {}
                engine._trophy_totals[winner_id_int] = winner_result.get("trophies", 0)
            else:
                logger.info("⚠️ Winner rewards nullified (status: %s)", winner_status.value)
            
            # Начисление монет победителю (динамически на основе тира)
            if winner_coins_delta > 0:
                try:
                    # Начисляем монеты
                    coins_result = await db.update_user_coins(winner_id_int, winner_coins_delta)
                    new_coins_total = coins_result.get("coins", 0) if coins_result else 0
                    
                    logger.info(
                        "✅ Winner coins updated: %s -> %d (+%d, tier=%s)",
                        winner_id_int, new_coins_total, winner_coins_delta, winner_tier
                    )
                    
                    # Сохраняем данные о монетах в движке для фронтенда
                    if not hasattr(engine, "_coins_changes"):
                        engine._coins_changes = {}
                    engine._coins_changes[winner_id_int] = winner_coins_delta
                    
                    if not hasattr(engine, "_coins_totals"):
                        engine._coins_totals = {}
                    engine._coins_totals[winner_id_int] = new_coins_total
                    
                except Exception as coin_exc:
                    logger.error("❌ Failed to update winner coins: %s", coin_exc)
            else:
                logger.info("⚠️ Winner coins nullified (status: %s)", winner_status.value)
            
        except Exception as exc:
            logger.error("❌ Failed to update winner trophies: %s", exc)
        
        # Отнимаем трофеи у проигравшего (если не было surrender_processed)
        try:
            if loser_surrender_processed:
                logger.info("⚠️ Loser %s already processed surrender, skipping trophy deduction", loser_id)
            else:
                loser_result = await db.update_user_trophies(loser_id, loser_trophy_delta)
                logger.info(
                    "✅ Loser trophies updated: %s -> %d (max: %d)",
                    loser_id, loser_result.get("trophies", 0), loser_result.get("max_trophies", 0)
                )
                
                # Сохраняем изменение трофеев для проигравшего напрямую из БД
                if not hasattr(engine, "_trophy_changes"):
                    engine._trophy_changes = {}
                engine._trophy_changes[loser_id] = loser_trophy_delta
                
                # ДОБАВЛЕНО: Сохраняем текущее количество трофеев напрямую из БД
                if not hasattr(engine, "_trophy_totals"):
                    engine._trophy_totals = {}
                engine._trophy_totals[loser_id] = loser_result.get("trophies", 0)
            
        except Exception as exc:
            logger.error("❌ Failed to update loser trophies: %s", exc)
    
    # КРИТИЧНО: Устанавливаем флаги завершения игры
    # is_ended - останавливает все фоновые задачи для этого матча
    # rewards_granted - предотвращает повторное начисление трофеев
    engine.is_ended = True
    engine.rewards_granted = True
    logger.info("✅ Battle end flags set for match %s (is_ended=True, rewards_granted=True)", match_id)
    
    # Сохраняем результат боя в таблицу battle_results
    try:
        battle_result_id = await db.save_battle_result(
            match_id=match_id,
            winner_id=winner_id if winner_id else None,
            loser_id=loser_id if winner_id else None,
            p1_hp=engine.p1_state.hero_hp,
            p2_hp=engine.p2_state.hero_hp,
            trophy_change=abs(winner_trophy_delta) if winner_id else 0
        )
        
        if battle_result_id:
            logger.info("✅ Battle result saved: ID=%s, match=%s", battle_result_id, match_id)
        else:
            logger.warning("⚠️ Failed to save battle result to database")
            
    except Exception as exc:
        logger.error("❌ Error saving battle result: %s", exc)


@sio.event
async def surrender(sid: str, data: dict[str, Any]) -> None:
    """
    Обработчик сдачи игрока. Трофеи списываются немедленно,
    но бой продолжается под управлением бота.
    
    Параметры:
        - match_id: идентификатор матча
        - user_id: идентификатор игрока, который сдаётся
    """
    logger = logging.getLogger(__name__)
    try:
        match_id = str(data.get("match_id", ""))
        user_id = data.get("user_id")
        
        logger.info(
            "[SOCKET] surrender sid=%s match_id=%s user_id=%s",
            sid, match_id or "unknown", user_id
        )
        
        if not match_id or user_id is None:
            logger.warning("surrender: получен запрос без match_id/user_id от sid=%s", sid)
            await sio.emit("error", {"message": "match_id и user_id обязательны"}, to=sid)
            return
        
        # Получаем движок
        engine = ACTIVE_MATCHES.get(match_id)
        if not engine:
            logger.warning("surrender: движок не найден для match_id=%s", match_id)
            await sio.emit("error", {"message": "Match not found"}, to=sid)
            return
        
        # Помечаем игрока как сдавшегося
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            logger.error("surrender: invalid user_id=%s", user_id)
            await sio.emit("error", {"message": "Invalid user_id"}, to=sid)
            return
        
        # Проверяем, что игрок участвует в матче
        if str(engine.p1_state.user_id) != str(user_id_int) and str(engine.p2_state.user_id) != str(user_id_int):
            logger.error("surrender: игрок %s не участвует в матче %s", user_id_int, match_id)
            await sio.emit("error", {"message": "User not in match"}, to=sid)
            return
        
        # Помечаем игрока как SURRENDERED (бот продолжит играть)
        engine.mark_surrender(user_id_int)
        
        # Получаем состояние сдавшегося игрока
        player_state = engine.get_player_state(user_id_int)
        
        # Проверяем, что трофеи ещё не были списаны
        if player_state.surrender_processed:
            logger.warning("Surrender already processed for player %s", user_id_int)
            await sio.emit("error", {"message": "Surrender already processed"}, to=sid)
            return
        
        # Получаем app для доступа к БД
        app = getattr(sio, 'app', None)
        if not app:
            logger.error("surrender: не удалось получить app для начисления трофеев")
            await sio.emit("error", {"message": "Database unavailable"}, to=sid)
            return
        
        db = app.get("db")
        if not db:
            logger.error("Database not available")
            await sio.emit("error", {"message": "Database unavailable"}, to=sid)
            return
        
        # Получаем текущие трофеи из БД
        try:
            user_info = await db.get_user_info(user_id_int)
            current_trophies = user_info.get("trophies", 0) if user_info else 0
        except Exception as exc:
            logger.error("Failed to get user trophies: %s", exc)
            current_trophies = 0
        
        # Рассчитываем штраф (максимальный для SURRENDERED)
        from core.state import ReplacementStatus
        penalty_delta, tier_name, tier_data = calculate_trophy_delta(
            current_trophies,
            is_winner=False,
            status=ReplacementStatus.SURRENDERED
        )
        
        # Списываем трофеи немедленно
        try:
            result = await db.update_user_trophies(user_id_int, penalty_delta)
            new_trophies = result.get("trophies", 0)
            
            logger.warning(
                "[SURRENDER_IMMEDIATE] Player %s lost %d trophies instantly (%d -> %d). Bot takeover initiated.",
                user_id_int, abs(penalty_delta), current_trophies, new_trophies
            )
            
            # Устанавливаем флаг, чтобы избежать двойного списания в конце боя
            player_state.surrender_processed = True
            
        except Exception as exc:
            logger.error("Failed to update trophies for surrendered player: %s", exc)
            await sio.emit("error", {"message": "Trophy update failed"}, to=sid)
            return
        
        # Отправляем подтверждение клиенту с информацией о штрафе
        await sio.emit(
            "surrender_ack",
            {
                "match_id": match_id,
                "user_id": user_id_int,
                "trophy_penalty": penalty_delta,
                "new_trophies": new_trophies
            },
            to=sid
        )
        
        # СРАЗУ вызываем проверку окончания игры (согласно правилам)
        game_over_result = engine.check_game_over()
        
        # Явно отправляем game_over текущему сокету, если матч завершился
        if game_over_result.get("game_over"):
            logger.info("surrender: матч %s завершен после сдачи (game_over=True)", match_id)
            winner_id = game_over_result.get("winner_id")
            
            # Начисляем награды победителю и сохраняем результат
            await _process_battle_end(app, match_id, engine, winner_id)
            
            # Отправляем game_over именно сдавшемуся игроку (sid) перед тем как он уйдет
            await sio.emit(
                "game_over",
                {
                    "game_over": True,
                    "winner_id": winner_id,
                    "p1_hp": engine.p1_state.hero_hp,
                    "p2_hp": engine.p2_state.hero_hp,
                    "reason": "surrender"
                },
                to=sid
            )
            
            # Также уведомляем комнату, если есть другие участники
            await sio.emit(
                "game_over",
                {
                    "game_over": True,
                    "winner_id": winner_id,
                    "p1_hp": engine.p1_state.hero_hp,
                    "p2_hp": engine.p2_state.hero_hp,
                    "reason": "surrender"
                },
                room=match_id
            )
        
        # Запускаем бота, если сейчас ход сдавшегося игрока и игра НЕ закончена
        if not game_over_result.get("game_over") and engine.current_player_id == user_id_int:
            logger.info("surrender: запускаем бота для сдавшегося игрока %s", user_id_int)
            await check_and_run_bot(match_id, ACTIVE_MATCHES)
        
    except Exception as exc:
        logger.error("surrender: ошибка обработки: %s", exc, exc_info=True)
        await sio.emit("error", {"message": str(exc)}, to=sid)


def setup_battle_events() -> None:
    """
    Регистрирует обработчики событий движка для отправки через Socket.io.
    Вызывается один раз при старте сервера.
    """
    print("DEBUG: setup_battle_events() вызвана!")  # Временная отладка
    
    def emit_to_match(match_id: str, event_data: dict[str, Any]) -> None:
        """
        Персонализированный broadcast: отправляет каждому участнику состояние с его legal_actions.
        Для каждого клиента вызывается engine.get_full_state(viewer_id=user_id).
        """
        try:
            event_type = event_data.get("event_type", "state_changed")
            
            # Получаем движок для этого матча
            engine = ACTIVE_MATCHES.get(str(match_id))
            if not engine:
                logging.warning(f"[SOCKET_EMIT] Движок не найден для матча {match_id}")
                return
            
            # Находим всех участников матча через SID_TO_MATCH
            participants = {}  # sid -> user_id
            for sid, session_data in SID_TO_MATCH.items():
                if str(session_data.get("match_id")) == str(match_id):
                    user_id = session_data.get("user_id")
                    if user_id:
                        participants[sid] = user_id
            
            if not participants:
                logging.warning(f"[SOCKET_EMIT] Нет подключенных участников для матча {match_id}")
                return
            
            logging.info(
                f"[SOCKET_EMIT] Персональная рассылка {event_type} для матча {match_id}, "
                f"участников: {len(participants)}"
            )
            
            # Отправляем персонализированное состояние каждому участнику
            for sid, user_id in participants.items():
                try:
                    # КРИТИЧНО: Получаем состояние с legal_actions для конкретного игрока
                    personalized_state = engine.get_full_state(viewer_id=user_id)
                    
                    logging.info(
                        f"[SOCKET_EMIT] ✉️ {event_type} -> sid={sid[:8]} user_id={user_id} | "
                        f"legal_actions={len(personalized_state.get('legal_actions', []))} | "
                        f"turn={personalized_state.get('turn')} | "
                        f"current_player={personalized_state.get('current_player_id')}"
                    )
                    
                    # Отправляем персонально через to=sid
                    asyncio.create_task(
                        sio.emit(
                            event_type,
                            {
                                "match_id": match_id,
                                "state": personalized_state,
                                "data": event_data.get("data", {}),
                            },
                            to=sid,
                        )
                    )
                except Exception as exc:
                    logging.error(
                        f"[SOCKET_EMIT] Ошибка персонализации для sid={sid[:8]} user_id={user_id}: {exc}"
                    )
            
        except Exception as exc:
            logging.error(f"[SOCKET_EMIT] Ошибка при персональной рассылке: {exc}", exc_info=True)
    
    # Регистрируем обработчики для всех типов событий
    BATTLE_EVENT_EMITTER.on("turn_start", emit_to_match)
    BATTLE_EVENT_EMITTER.on("card_played", emit_to_match)
    BATTLE_EVENT_EMITTER.on("attack", emit_to_match)
    BATTLE_EVENT_EMITTER.on("turn_end", emit_to_match)
    BATTLE_EVENT_EMITTER.on("turn_switched", emit_to_match)  # КРИТИЧНО: Смена хода после бота
    BATTLE_EVENT_EMITTER.on("state_changed", emit_to_match)
    BATTLE_EVENT_EMITTER.on("potion_used", emit_to_match)  # НОВОЕ: Поддержка зелий
    
    # КРИТИЧНО: Событийный триггер бота - запускается при каждом начале хода
    # Это гарантирует, что бот "проснется" всегда: и после хода игрока, и после авто-смены по таймеру
    def bot_trigger_listener(match_id: str, event_data: dict[str, Any]) -> None:
        """
        Автоматически запускает бота при начале его хода.
        Вызывается при событии turn_start.
        """
        if event_data.get("event_type") == "turn_start":
            logging.info(f"[BOT_TRIGGER] 🤖 Событие turn_start получено для матча {match_id}")
            # Создаем задачу запуска бота (не блокируем эмиттер)
            asyncio.create_task(check_and_run_bot(match_id, ACTIVE_MATCHES))
    
    BATTLE_EVENT_EMITTER.on("turn_start", bot_trigger_listener)
    
    logging.info("Socket.io: обработчики событий боя зарегистрированы (включая bot_trigger)")
    print("DEBUG: Socket.io обработчики зарегистрированы (+ bot_trigger)!")  # Временная отладка



def _serialize_datetime(obj: Any) -> Any:
    """Рекурсивно преобразовать все datetime объекты в строки для JSON."""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_datetime(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    # Для других типов (например, asyncpg.Record) пытаемся преобразовать в dict
    elif hasattr(obj, '__dict__'):
        return _serialize_datetime(obj.__dict__)
    else:
        return obj


def _verify_init_data(init_data: str, bot_token: str) -> dict[str, str] | None:
    """Проверить подпись initData от Telegram и вернуть параметры."""
    try:
        data_dict = dict(parse_qsl(init_data))
        received_hash = data_dict.pop("hash", "")
        if not received_hash:
            return None

        data_check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(data_dict.items())
        )

        secret_key = hmac.new(
            key=b"WebAppData",
            msg=bot_token.encode(),
            digestmod=hashlib.sha256,
        ).digest()

        calculated_hash = hmac.new(
            key=secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if calculated_hash != received_hash:
            return None

        return data_dict
    except Exception:
        return None


def _extract_user_id_from_init_data(data_dict: dict[str, str]) -> int | None:
    """Извлечь user_id из проверенного initData."""
    try:
        user_str = data_dict.get("user", "")
        if not user_str:
            return None
        import json

        user_data = json.loads(user_str)
        return int(user_data.get("id"))
    except Exception:
        return None


def _create_ssl_disabled_session():
    """Создать aiohttp сессию с отключенной проверкой SSL для локальной разработки."""
    import aiohttp
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


def _create_ssl_disabled_session():
    """Создать aiohttp сессию с отключенной проверкой SSL для локальной разработки."""
    import aiohttp
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


# ============================================================================
# Модульные функции управления ботами (вынесены на уровень модуля)
# ============================================================================

async def check_and_run_bot(match_id: str, active_matches: dict[str, BattleEngine]) -> None:
    """
    Функция-триггер для проверки и запуска бота.
    Проверяет: если текущий игрок == бот, запускает run_bot_routine.
    Вызывается:
    1. После создания матча (если бот ходит первым)
    2. После передачи хода игроком (если ход перешел к боту)
    3. После получения сигнала client_ready от фронтенда
    
    Args:
        match_id: идентификатор матча
        active_matches: словарь активных матчей
    """
    logger = logging.getLogger(__name__)
    
    # Получаем движок из активных матчей
    engine = active_matches.get(match_id)
    if not engine:
        print(f"!!! [SERVER] check_and_run_bot: движок не найден для match_id={match_id}")
        logger.warning("check_and_run_bot: engine not found for match_id=%s", match_id)
        return
    
    # КРИТИЧНО: Проверяем, готов ли клиент к началу боя
    # Это предотвращает преждевременный ход бота до того, как игрок загрузит состояние
    if hasattr(engine, 'client_ready') and not engine.client_ready:
        print(f"!!! [SERVER] check_and_run_bot: клиент не готов (match_id={match_id}), бот ждёт")
        logger.info("check_and_run_bot: client not ready for match_id=%s, bot waiting", match_id)
        return
    
    # Проверяем, является ли текущий игрок ботом
    try:
        # ДОБАВЛЕНО: Проверяем, не окончена ли игра
        current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
        state = engine.get_full_state(viewer_id=current_player) if hasattr(engine, "get_full_state") else {}
        if state.get("is_ended") or state.get("game_over"):
            print(f"!!! [SERVER] check_and_run_bot: игра окончена (match_id={match_id}), бот не запускается")
            logger.info("check_and_run_bot: game already ended for match_id=%s, skipping bot", match_id)
            return
        is_bot = engine.is_bot(current_player) if hasattr(engine, "is_bot") else False
        
        # Проверяем статус замены игрока (AFK/SURRENDERED)
        from core.state import ReplacementStatus
        player_status = ReplacementStatus.ACTIVE
        if hasattr(engine, "_arena") and engine._arena:
            arena_state = engine._arena.state
            if arena_state.p1.user_id == current_player:
                player_status = getattr(arena_state.p1, "replacement_status", ReplacementStatus.ACTIVE)
            elif arena_state.p2.user_id == current_player:
                player_status = getattr(arena_state.p2, "replacement_status", ReplacementStatus.ACTIVE)
        
        # КРИТИЧНО: Проверяем ситуацию "Бот против Бота"
        # Если оба игрока - боты (изначально или заменены), немедленно завершаем матч
        p1_is_bot_or_replaced = False
        p2_is_bot_or_replaced = False
        
        if hasattr(engine, "_arena") and engine._arena:
            arena_state = engine._arena.state
            p1 = arena_state.p1
            p2 = arena_state.p2
            
            # Игрок 1 считается ботом если: изначально бот ИЛИ заменён (AFK/SURRENDERED)
            p1_is_bot_or_replaced = (
                p1.is_bot or 
                getattr(p1, "replacement_status", ReplacementStatus.ACTIVE) in [
                    ReplacementStatus.AFK, 
                    ReplacementStatus.SURRENDERED
                ]
            )
            
            # Игрок 2 считается ботом если: изначально бот ИЛИ заменён (AFK/SURRENDERED)
            p2_is_bot_or_replaced = (
                p2.is_bot or 
                getattr(p2, "replacement_status", ReplacementStatus.ACTIVE) in [
                    ReplacementStatus.AFK, 
                    ReplacementStatus.SURRENDERED
                ]
            )
            
            # Если оба - боты, немедленно завершаем матч
            if p1_is_bot_or_replaced and p2_is_bot_or_replaced:
                print(f"[MATCH_TERMINATED] Match {match_id} closed automatically: Bot vs Bot scenario detected.")
                logger.warning(
                    "[MATCH_TERMINATED] Match %s closed automatically: Bot vs Bot scenario detected "
                    "(p1_status=%s, p1_is_bot=%s, p2_status=%s, p2_is_bot=%s)",
                    match_id,
                    getattr(p1, "replacement_status", ReplacementStatus.ACTIVE).value,
                    p1.is_bot,
                    getattr(p2, "replacement_status", ReplacementStatus.ACTIVE).value,
                    p2.is_bot
                )
                
                # Помечаем игру как завершённую
                engine.is_ended = True
                
                # Удаляем матч из активных
                if match_id in active_matches:
                    del active_matches[match_id]
                
                # Рассылаем событие завершения через Socket.IO (если есть подключенные клиенты)
                try:
                    await sio.emit(
                        "match_terminated",
                        {
                            "match_id": match_id,
                            "reason": "bot_vs_bot",
                            "message": "Матч автоматически завершён: оба игрока покинули игру"
                        },
                        room=match_id
                    )
                except Exception as emit_exc:
                    logger.error("Failed to emit match_terminated event: %s", emit_exc)
                
                return
        
        # Бот играет, если: 1) игрок - бот ИЛИ 2) игрок AFK/SURRENDERED
        should_run_bot = is_bot or (player_status != ReplacementStatus.ACTIVE)
        
        print(f"!!! [SERVER] check_and_run_bot: match_id={match_id}, current_player={current_player}, is_bot={is_bot}, status={player_status.value}, should_run_bot={should_run_bot}")
        
        if should_run_bot:
            if player_status != ReplacementStatus.ACTIVE:
                print(f"!!! [SERVER] Игрок {current_player} заменён ботом (статус: {player_status.value})")
                logger.info("check_and_run_bot: player %s replaced by bot (status=%s)", current_player, player_status.value)
            else:
                print(f"!!! [SERVER] Ход бота: {match_id}, bot_id={current_player}")
                logger.info("check_and_run_bot: starting bot routine for match_id=%s, bot_id=%s", match_id, current_player)
            
            # Запускаем ход бота асинхронно
            asyncio.create_task(run_bot_routine(engine, current_player))
        else:
            print(f"!!! [SERVER] check_and_run_bot: текущий игрок активен (match_id={match_id}, current_player={current_player})")
    except Exception as exc:
        logger.error("check_and_run_bot: ошибка проверки бота для match_id=%s: %s", match_id, exc, exc_info=True)
        print(f"!!! [SERVER] check_and_run_bot ERROR: {exc}")


async def run_bot_routine(engine: BattleEngine, bot_id: int | str) -> None:
    """
    Асинхронный сценарий хода бота:
    - короткая «задержка обдумывания»,
    - принятие решения через BotAI,
    - последовательное выполнение всех действий,
    - завершение хода.
    """
    logger = logging.getLogger(__name__)
    logger.info("BOT ROUTINE STARTED for %s", bot_id)
    print(f"!!! [SERVER] Bot routine started: bot_id={bot_id}, match_id={getattr(engine, 'match_id', 'unknown')}")
    
    # КРИТИЧНО: ПЕРВАЯ ПРОВЕРКА - это вообще ход бота?
    if engine.current_player_id != bot_id:
        print(f"!!! [SERVER] run_bot_routine: НЕ ХОД БОТА! current_player={engine.current_player_id}, bot={bot_id}")
        logger.warning("run_bot_routine called but not bot's turn (current=%s, bot=%s)", engine.current_player_id, bot_id)
        return
    
    # КРИТИЧНО: Проверяем флаг is_ended СРАЗУ - если игра завершена, бот МГНОВЕННО останавливается
    if hasattr(engine, 'is_ended') and engine.is_ended:
        print(f"!!! [SERVER] run_bot_routine: engine.is_ended=True, бот НЕ начинает думать")
        logger.info("run_bot_routine: engine.is_ended=True, bot aborting immediately")
        return
    
    # ДОБАВЛЕНО: Проверяем, не окончена ли игра перед действиями бота
    try:
        state = engine.get_full_state(viewer_id=bot_id) if hasattr(engine, "get_full_state") else {}
        if state.get("is_ended") or state.get("game_over"):
            print(f"!!! [SERVER] run_bot_routine: игра окончена (через state), бот НЕ делает ход")
            logger.info("run_bot_routine: game already ended (via state), bot skipping turn")
            return
    except Exception as exc:
        print(f"!!! [SERVER] run_bot_routine: Failed to check game state: {exc}")
    
    # Логируем состояние маны бота перед принятием решений
    try:
        bot_state = engine.get_player_state(bot_id)
        bot_mana = getattr(bot_state, "mana", 0)
        bot_max_mana = getattr(bot_state, "max_mana", 0)
        print(f"!!! [SERVER] run_bot_routine: bot_id={bot_id}, mana={bot_mana}/{bot_max_mana}, turn={getattr(engine, 'turn', 'unknown')}")
        logger.info("BOT STATE: bot_id=%s mana=%s/%s turn=%s", bot_id, bot_mana, bot_max_mana, getattr(engine, 'turn', 'unknown'))
    except Exception as exc:
        print(f"!!! [SERVER] run_bot_routine: Failed to get bot state: {exc}")
        logger.error("Failed to get bot state: %s", exc, exc_info=True)

    try:
        # Получаем сложность бота для расчёта задержек
        difficulty = getattr(engine, "bot_difficulty", "medium")
        match_id = getattr(engine, "match_id", "unknown")
        
        # Единая задержка хода (Turn Delay): бот «раздумывает» перед серией действий
        if difficulty in ("hard", "max"):
            turn_delay = random.uniform(1.5, 2.5)
        else:  # lite, easy, medium, noob
            turn_delay = random.uniform(4.0, 6.0)
        
        logger.info(
            "[BOT_THINKING] Match: %s | Difficulty: %s | Turn delay: %.2fs",
            match_id, difficulty, turn_delay
        )
        print(f"!!! [SERVER] 🤔 Бот раздумывает: {turn_delay:.2f}s (difficulty={difficulty})")
        await asyncio.sleep(turn_delay)
        
        # КРИТИЧНО: Еще раз проверяем is_ended перед тем как бот начнет думать
        if hasattr(engine, 'is_ended') and engine.is_ended:
            print(f"!!! [SERVER] run_bot_routine: engine.is_ended=True перед действиями, бот НЕ думает")
            logger.info("run_bot_routine: engine.is_ended=True before actions, bot aborting")
            return
        
        # Определяем тип бота: Берсерк (810416*) или rule-based
        bot_id_str = str(bot_id)
        use_berserk = bot_id_str.startswith("810416") and BERSERK_BRAIN is not None
        
        if use_berserk:
            logger.info("[SERVER] Используется ONNX-модель Берсерк для bot_id=%s", bot_id)
            print(f"!!! [SERVER] 🧠 БЕРСЕРК АКТИВИРОВАН для bot_id={bot_id}")
        else:
            logger.info("[SERVER] Используется rule-based BotAI для bot_id=%s", bot_id)
            print(f"!!! [SERVER] 🎲 Rule-based AI для bot_id={bot_id}")
        
        # Пошаговое выполнение действий
        max_actions = 20
        action_count = 0
        total_action_delays = 0.0  # Суммарное время на технические паузы
        
        for step in range(max_actions):
            # Проверка завершения игры
            if hasattr(engine, 'is_ended') and engine.is_ended:
                print(f"!!! [SERVER] run_bot_routine: игра завершена на шаге {step}")
                logger.info("run_bot_routine: game ended at step %d", step)
                break
            
            # Проверка хода
            if engine.current_player_id != bot_id:
                print(f"!!! [SERVER] run_bot_routine: не ход бота на шаге {step}")
                logger.info("run_bot_routine: not bot's turn at step %d", step)
                break
            
            # Безопасность по времени: если осталось < 5 сек, пропускаем все паузы
            time_remaining = engine.get_turn_time_remaining() if hasattr(engine, "get_turn_time_remaining") else 25
            emergency_mode = time_remaining < 5.0
            
            # Получаем легальные действия из core/engine
            try:
                if not hasattr(engine, '_arena') or engine._arena is None:
                    logger.error("[SERVER] engine._arena не инициализирован")
                    break
                
                legal_actions_obj = engine._arena.get_legal_actions(bot_id)
                legal_actions_dict = [engine._serialize_action(a) for a in legal_actions_obj]
                
                if not legal_actions_dict:
                    logger.info("[SERVER] Нет легальных действий, завершаем ход")
                    print(f"!!! [SERVER] Нет легальных действий, принудительный end_turn")
                    if engine.current_player_id == bot_id:
                        engine.end_turn(bot_id)
                        
                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break
                
                logger.debug("[SERVER] Доступно %d действий на шаге %d", len(legal_actions_dict), step)
                
                # Выбор действия
                action_id = 0
                if use_berserk:
                    # ONNX-инференс с учетом difficulty
                    try:
                        action_id = BERSERK_BRAIN.get_action(
                            engine._arena.state,
                            bot_id,
                            legal_actions_obj,
                            difficulty=difficulty,
                        )
                        logger.info(
                            "[BERSERK] Выбрано действие ID=%d из %d (difficulty=%s)",
                            action_id, len(legal_actions_obj), difficulty
                        )
                    except Exception as exc:
                        logger.error("[BERSERK] Ошибка инференса: %s, fallback на rule-based", exc)
                        # Fallback на rule-based
                        chosen_action = BotAI.decide_action(legal_actions_dict)
                        if chosen_action:
                            action_id = legal_actions_dict.index(chosen_action)
                else:
                    # Rule-based выбор
                    chosen_action = BotAI.decide_action(legal_actions_dict)
                    if chosen_action:
                        action_id = legal_actions_dict.index(chosen_action)
                
                # Проверка валидности
                if action_id < 0 or action_id >= len(legal_actions_dict):
                    logger.warning("[SERVER] Невалидный action_id=%d, принудительный end_turn", action_id)
                    if engine.current_player_id == bot_id:
                        engine.end_turn(bot_id)
                        
                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break
                
                action_dict = legal_actions_dict[action_id]
                action_type = action_dict.get("type")
                
                print(f"!!! [SERVER] Шаг {step}: {action_type} -> {action_dict}")
                logger.info("[SERVER] Executing action: %s", action_dict)
                
                # Выполнение действия через execute_bot_action
                result = engine.execute_bot_action(action_dict)
                action_count += 1
                
                # Минимальная техническая пауза для анимаций UI (0.4 - 0.8 сек)
                # Пропускается в emergency_mode (осталось < 5 сек)
                if not emergency_mode:
                    action_gap = random.uniform(0.4, 0.8)
                    total_action_delays += action_gap
                    await asyncio.sleep(action_gap)
                else:
                    logger.warning(
                        "[BOT_EMERGENCY] Match: %s | Time remaining: %.1fs | Skipping delays",
                        match_id, time_remaining
                    )
                
                if not result.get("success", True):
                    logger.warning("[SERVER] Действие не выполнено: %s", result.get("error"))
                    # Пытаемся завершить ход
                    if engine.current_player_id == bot_id:
                        engine.end_turn(bot_id)
                        
                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break
                
                # Проверка game_over
                if result.get("game_over"):
                    logger.info("[SERVER] Игра завершена после действия")
                    break
                
                # Если это был end_turn, выходим
                if action_type == "end_turn":
                    logger.info("[SERVER] Ход бота завершен (end_turn), выполнено действий: %d", action_count)
                    print(f"!!! [SERVER] Бот завершил ход, действий: {action_count}")
                    
                    # КРИТИЧНО: Отправляем обновленное состояние после смены хода (для нового текущего игрока)
                    try:
                        new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                        full_state = engine.get_full_state(viewer_id=new_current_player)
                        event_data = {
                            "event_type": "turn_switched",
                            "match_id": match_id,
                            "state_p1": full_state,
                        }
                        BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        logger.info("[SERVER] Отправлено событие turn_switched после end_turn бота")
                    except Exception as emit_exc:
                        logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    
                    break
                
            except Exception as exc:
                logger.error("[SERVER] Ошибка на шаге %d: %s", step, exc, exc_info=True)
                # Попытка завершить ход
                try:
                    if engine.current_player_id == bot_id:
                        engine.end_turn(bot_id)
                        
                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                except:
                    pass
                break
        
        # Итоговая статистика хода бота
        total_turn_time = turn_delay + total_action_delays
        logger.info(
            "[BOT_TURN_SUMMARY] Match: %s | Difficulty: %s | Total thinking time: %.2fs "
            "(turn_delay=%.2fs + action_gaps=%.2fs) | Actions executed: %d",
            match_id, difficulty, total_turn_time, turn_delay, total_action_delays, action_count
        )
        print(
            f"!!! [SERVER] ⏱️ Итого время бота: {total_turn_time:.2f}s "
            f"(раздумья={turn_delay:.2f}s + паузы={total_action_delays:.2f}s), действий={action_count}"
        )
        
        # Финальная проверка: если бот все еще владеет ходом, завершаем принудительно
        if engine.current_player_id == bot_id and not (hasattr(engine, 'is_ended') and engine.is_ended):
            logger.warning("[SERVER] Бот не завершил ход явно, принудительный end_turn")
            print(f"!!! [SERVER] Принудительное завершение хода бота")
            try:
                engine.end_turn(bot_id)
                
                # КРИТИЧНО: Отправляем обновленное состояние после принудительного end_turn (для нового текущего игрока)
                try:
                    new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                    full_state = engine.get_full_state(viewer_id=new_current_player)
                    event_data = {
                        "event_type": "turn_switched",
                        "match_id": match_id,
                        "state_p1": full_state,
                    }
                    BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                    logger.info("[SERVER] Отправлено событие turn_switched после принудительного end_turn")
                except Exception as emit_exc:
                    logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    
            except Exception as exc:
                logger.error("[SERVER] Ошибка принудительного end_turn: %s", exc)
    
    except Exception as exc:
        logger.error("run_bot_routine fatal error: %s", exc, exc_info=True)


async def trigger_bot_move(match_id: str) -> None:
    """
    Триггер для запуска хода бота после действия игрока.
    Проверяет, перешел ли ход к боту, и если да - запускает run_bot_routine.
    
    Args:
        match_id: ID матча
    """
    logger = logging.getLogger(__name__)
    
    engine = ACTIVE_MATCHES.get(match_id)
    if not engine:
        logger.warning("[trigger_bot_move] Движок не найден для match_id=%s", match_id)
        return
    
    # Проверяем, что игра еще идет
    if hasattr(engine, 'is_ended') and engine.is_ended:
        logger.debug("[trigger_bot_move] Игра завершена, бот не запускается")
        return
    
    # Проверяем, является ли текущий игрок ботом
    current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else None
    if current_player is None:
        logger.warning("[trigger_bot_move] Не удалось определить текущего игрока")
        return
    
    is_bot = engine.is_bot(current_player) if hasattr(engine, "is_bot") else False
    
    if is_bot:
        logger.info("[trigger_bot_move] Ход перешел к боту %s, запускаем run_bot_routine", current_player)
        print(f"!!! [SERVER] trigger_bot_move: запуск бота {current_player} для match_id={match_id}")
        
        # Запускаем ход бота асинхронно
        asyncio.create_task(run_bot_routine(engine, current_player))
    else:
        logger.debug("[trigger_bot_move] Текущий игрок %s не является ботом", current_player)


def create_web_app(
    db: Database,
    bot_token: str,
    payment_service=None,
    webapp_url: str | None = None,
    stars_rate_rub: float = 1.5,
    stars_markup: float = 1.2,
    stars_test_mode: bool = False,
    battle_engine=None,
) -> web.Application:
    # Подавляем лишние логи доступа aiohttp и системные пинги сокетов
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)

    app = web.Application()
    app["db"] = db
    app["bot_token"] = bot_token
    app["payment_service"] = payment_service
    app["webapp_url"] = webapp_url or "https://t.me/your_bot"
    app["stars_rate_rub"] = stars_rate_rub
    app["stars_markup"] = stars_markup
    app["stars_test_mode"] = stars_test_mode
    app["admin_ids"] = {ADMIN_ID}
    # Инициализируем игровые сервисы и прокладываем их в контекст приложения
    services = initialize_game_services(db, battle_engine=battle_engine)
    app["bot_generator"] = services["bot_generator"]
    app["battle_engine"] = services["battle_engine"]
    app["matchmaker"] = services["matchmaker"]
    app["active_matches"] = services["active_matches"]
    app["event_emitter"] = services["event_emitter"]
    
    # Инициализация ONNX-мозга Берсерка с профилями сложности
    global BERSERK_BRAIN
    try:
        from infrastructure.config import BOT_DIFFICULTY_PROFILES
        BERSERK_BRAIN = BerserkInference(profiles=BOT_DIFFICULTY_PROFILES)
        loaded_profiles = list(BERSERK_BRAIN.sessions.keys())
        logging.getLogger(__name__).info(
            f"✅ ONNX Берсерк загружен: {len(loaded_profiles)} профилей ({', '.join(loaded_profiles)})"
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "⚠️ Не удалось загрузить ONNX Берсерк: %s (боты будут использовать rule-based AI)",
            exc,
        )
        BERSERK_BRAIN = None
    
    # Настраиваем обработчики событий для Socket.io
    setup_battle_events()
    
    # Прикрепляем Socket.io к приложению aiohttp
    sio.attach(app)
    
    # КРИТИЧНО: Сохраняем ссылку на app в sio для доступа из обработчиков Socket.io
    sio.app = app

    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        """
        CORS middleware с защитой от None response.
        Если handler возвращает None, сразу возвращаем None без обработки headers.
        """
        response = await handler(request)
        
        # КРИТИЧНО: Проверяем, что response не None перед добавлением headers
        if response is None:
            return response
        
        # Добавляем CORS headers только если response существует
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    app.middlewares.append(cors_middleware)

    async def health_check(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "extracards-webapp"})

    async def index(_: web.Request) -> web.FileResponse:
        index_path = WEBAPP_DIR / "index.html"
        if not index_path.exists():
            raise web.HTTPInternalServerError(text="index.html not found")
        return web.FileResponse(index_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    async def battle_page_handler(_: web.Request) -> web.Response:
        """
        Отдает выделенную страницу боя, чтобы фронтенд мог открывать бой по отдельному URL.
        Читаем HTML с диска и возвращаем его целиком, так как здесь нет шаблонизации.
        """
        arena_path = WEBAPP_DIR / "arena.html"
        if not arena_path.exists():
            raise web.HTTPInternalServerError(text="arena.html not found")

        return web.Response(
            text=arena_path.read_text(encoding="utf-8"),
            content_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def static_handler(request: web.Request) -> web.FileResponse:
        relative_path = request.match_info["path"]
        if ".." in relative_path:
            raise web.HTTPForbidden()
        
        # Если путь начинается с DesignAssets, ищем в DesignAssets
        if relative_path.startswith("DesignAssets/"):
            file_path = DESIGN_ASSETS_DIR / relative_path.replace("DesignAssets/", "", 1)
        else:
            # Иначе ищем в webapp
            file_path = WEBAPP_DIR / relative_path
        
        if not file_path.exists() or not file_path.is_file():
            raise web.HTTPNotFound()
        
        return web.FileResponse(file_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    async def profile_handler(request: web.Request) -> web.Response:
        init_data = request.rel_url.query.get("_auth")
        user_id = None
        photo_url = None
        first_name = None

        # Если _auth это число (user_id из initDataUnsafe), используем его напрямую
        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        # Извлекаем данные пользователя из initData
        username = None
        first_name_from_data = None
        last_name = None
        
        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)
                user_str = verified_data.get("user", "")
                if user_str:
                    import json
                    try:
                        user_data = json.loads(user_str)
                        photo_url = user_data.get("photo_url")
                        first_name = user_data.get("first_name")
                        first_name_from_data = first_name
                        username = user_data.get("username")
                        last_name = user_data.get("last_name")
                    except Exception:
                        pass

        # Если photo_url не получен из initData, пытаемся получить через Bot API
        if user_id and not photo_url:
            try:
                async with _create_ssl_disabled_session() as session:
                    url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
                    async with session.get(url, params={"user_id": user_id, "limit": 1}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                file_id = data["result"]["photos"][0][0]["file_id"]
                                # Получаем URL файла
                                file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
                                async with session.get(file_url, params={"file_id": file_id}) as file_resp:
                                    if file_resp.status == 200:
                                        file_data = await file_resp.json()
                                        if file_data.get("ok"):
                                            file_path = file_data["result"]["file_path"]
                                            photo_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
            except Exception:
                pass  # Игнорируем ошибки получения фото

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        # Проверяем, есть ли пользователь
        record = await db.get_user_profile(user_id)
        welcome_should_show = False
        
        if not record:
            # Пользователя нет - возвращаем специальный ответ для показа приветствия
            # НЕ создаем пользователя здесь - это будет сделано после завершения приветствия
            return web.json_response({
                "error": "user_not_found",
                "should_show_welcome": True,
                "need_registration": True
            }, status=404)

        settings_record = await db.get_user_settings(user_id)
        settings_data = {}
        if settings_record:
            settings_data = {
                "notif_cases": settings_record["notif_cases"],
                "notif_daily_rewards": settings_record["notif_daily_rewards"],
                "notif_game_invites": settings_record["notif_game_invites"],
                "notif_friend_requests": settings_record["notif_friend_requests"],
                "notif_events": settings_record["notif_events"],
                "notif_news": settings_record["notif_news"],
                "ads_enabled": settings_record["ads_enabled"],
                "sound_music": settings_record["sound_music"],
                "sound_sfx": settings_record["sound_sfx"],
                "social_block_friend_requests": settings_record["social_block_friend_requests"],
            }

        # Убеждаемся, что title есть (по умолчанию "Игрок")
        title = record.get("title") or "Игрок"
        
        payload: dict[str, Any] = {
            "user_id": record["user_id"],
            "username": record.get("username"),
            "first_name": first_name or record.get("first_name"),
            "photo_url": photo_url,
            "extra_pass": record.get("extra_pass", "inactive"),
            "trophies": record.get("trophies", 0),
            "max_trophies": record.get("max_trophies", 0),
            "keys": record.get("keys", 0),
            "gems": record.get("gems", 0),
            "coins": record.get("coins", 0),
            "squad_id": record.get("squad_id"),
            "status": record.get("status", "active"),
            "reg_date": record["reg_date"].isoformat() if record.get("reg_date") else None,
            "stars": record.get("stars", 0),
            "energy": record.get("energy", 5),
            "energy_cd": record.get("energy_cd").isoformat() if record.get("energy_cd") else None,
            "season": record.get("season", 0),
            "title": title,
            "img": record.get("img", ""),
            "selected_hero_id": record.get("selected_hero_id", 0),
            "custom_nickname": record.get("custom_nickname"),
            "nickname_changed": record.get("nickname_changed", False),
            "settings": settings_data,
            "should_show_welcome": welcome_should_show,  # Флаг для фронтенда
        }

        return web.json_response(payload)

    async def settings_handler(request: web.Request) -> web.Response:
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        # Если _auth это число (user_id из initDataUnsafe), используем его напрямую
        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method == "GET":
            try:
                # Проверяем, существует ли пользователь
                user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
                if not user_exists:
                    # Пользователя нет - возвращаем настройки по умолчанию
                    return web.json_response({
                        "notif_cases": False,
                        "notif_daily_rewards": False,
                        "notif_game_invites": False,
                        "notif_friend_requests": False,
                        "notif_events": False,
                        "notif_news": False,
                        "ads_enabled": False,
                        "sound_music": True,
                        "sound_sfx": True,
                        "social_block_friend_requests": False,
                    })
                
                settings_record = await db.get_user_settings(user_id)
                if not settings_record:
                    # Создаем настройки по умолчанию, если их нет
                    db_instance = request.app["db"]
                    await db_instance.execute(
                        """
                        INSERT INTO user_settings (user_id)
                        VALUES ($1)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        user_id,
                    )
                    settings_record = await db.get_user_settings(user_id)
                
                # Преобразуем в словарь и сериализуем datetime
                settings_dict = dict(settings_record)
                if "updated_at" in settings_dict and settings_dict["updated_at"]:
                    from datetime import datetime
                    if isinstance(settings_dict["updated_at"], datetime):
                        settings_dict["updated_at"] = settings_dict["updated_at"].isoformat()
                
                return web.json_response(settings_dict)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка получения настроек для user_id %s: %s", user_id, e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error"}, status=500
                )

        elif request.method == "POST":
            try:
                data = await request.json()
                import logging
                logging.getLogger(__name__).info(
                    "Сохранение настроек для user_id %s: %s", user_id, data
                )
                await db.update_user_settings(user_id, **data)
                
                # Проверяем, что настройки сохранились
                updated_settings = await db.get_user_settings(user_id)
                logging.getLogger(__name__).info(
                    "Настройки сохранены для user_id %s: %s", user_id, dict(updated_settings) if updated_settings else "не найдены"
                )
                
                # Преобразуем в словарь и сериализуем datetime
                settings_dict = {}
                if updated_settings:
                    settings_dict = dict(updated_settings)
                    if "updated_at" in settings_dict and settings_dict["updated_at"]:
                        from datetime import datetime
                        if isinstance(settings_dict["updated_at"], datetime):
                            settings_dict["updated_at"] = settings_dict["updated_at"].isoformat()
                
                return web.json_response({"status": "ok", "settings": settings_dict})
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка сохранения настроек для user_id %s: %s", user_id, e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error", "message": str(e)}, status=500
                )

    async def admin_players_handler(request: web.Request) -> web.Response:
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method == "GET":
            # Получаем список всех игроков
            try:
                players = await db.fetch(
                    """
                    SELECT u.user_id, u.username, u.first_name, u.extra_pass, 
                           u.trophies, u.status, u.reg_date
                    FROM users u
                    ORDER BY u.trophies DESC
                    LIMIT 100
                    """
                )
                total_count = await db.fetchval("SELECT COUNT(*) FROM users")
                
                # Преобразуем в словари и сериализуем datetime
                from datetime import datetime
                players_list = []
                for p in players:
                    player_dict = dict(p)
                    if "reg_date" in player_dict and player_dict["reg_date"]:
                        if isinstance(player_dict["reg_date"], datetime):
                            player_dict["reg_date"] = player_dict["reg_date"].isoformat()
                    players_list.append(player_dict)
                
                return web.json_response({
                    "players": players_list,
                    "total": total_count or len(players_list)
                })
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка получения списка игроков: %s", e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error", "message": str(e)}, status=500
                )

        elif request.method == "POST":
            data = await request.json()
            action = data.get("action")
            target_user_id = data.get("user_id")
            
            if action == "ban":
                await db.execute(
                    "UPDATE users SET status = 'banned' WHERE user_id = $1",
                    target_user_id
                )
            elif action == "unban":
                await db.execute(
                    "UPDATE users SET status = 'active' WHERE user_id = $1",
                    target_user_id
                )
            elif action == "warn":
                await db.execute(
                    "UPDATE users SET status = 'warn' WHERE user_id = $1",
                    target_user_id
                )
            
            return web.json_response({"status": "ok"})

    async def admin_stats_handler(request: web.Request) -> web.Response:
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        stats = await db.get_statistics()
        return web.json_response(stats)

    async def change_nickname_handler(request: web.Request) -> web.Response:
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            new_nickname = data.get("nickname", "").strip()
            
            if not new_nickname:
                return web.json_response({"error": "nickname_required"}, status=400)
            
            if len(new_nickname) > 20:
                return web.json_response({"error": "nickname_too_long"}, status=400)
            
            # Проверяем, первая ли это смена
            profile = await db.fetchrow(
                "SELECT nickname_changed FROM profiles WHERE user_id = $1", user_id
            )
            nickname_changed = profile["nickname_changed"] if profile else False
            cost_gems = 0 if not nickname_changed else 500
            
            result = await db.change_nickname(user_id, new_nickname, cost_gems)
            
            if not result["success"]:
                return web.json_response(result, status=400)
            
            return web.json_response({
                "success": True,
                "nickname": new_nickname,
                "cost": cost_gems,
                "is_first_change": result["is_first_change"]
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка смены никнейма для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_use_handler(request: web.Request) -> web.Response:
        """Обработчик использования промокода."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            code = data.get("code", "").strip()
            
            if not code:
                return web.json_response({"error": "code_required"}, status=400)
            
            result = await db.use_promocode(user_id, code)
            
            if not result["success"]:
                error_messages = {
                    "not_found": "Промокод не найден",
                    "expired": "Промокод истек",
                    "already_used": "Вы уже использовали этот промокод",
                    "not_eligible": "Этот промокод доступен только новым игрокам"
                }
                return web.json_response({
                    "success": False,
                    "error": result["error"],
                    "message": error_messages.get(result["error"], "Ошибка использования промокода")
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка использования промокода для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания промокода (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            code = data.get("code", "").strip().upper()
            type = data.get("type", "permanent")
            reward_gems = data.get("reward_gems", 0)
            reward_coins = data.get("reward_coins", 0)
            reward_keys = data.get("reward_keys", 0)
            reward_extrapass = data.get("reward_extrapass", False)
            expires_at = data.get("expires_at")
            
            if not code:
                return web.json_response({"error": "code_required"}, status=400)
            
            if type not in ["permanent", "personal", "welcome"]:
                return web.json_response({"error": "invalid_type"}, status=400)
            
            expires_datetime = None
            if expires_at:
                from datetime import datetime
                try:
                    expires_datetime = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                except Exception:
                    return web.json_response({"error": "invalid_expires_at"}, status=400)
            
            result = await db.create_promocode(
                code=code,
                type=type,
                reward_gems=reward_gems,
                reward_coins=reward_coins,
                reward_keys=reward_keys,
                reward_extrapass=reward_extrapass,
                created_by=user_id,
                expires_at=expires_datetime
            )
            
            if not result["success"]:
                error_messages = {
                    "code_exists": "Промокод с таким кодом уже существует"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка создания промокода")
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания промокода для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка промокодов (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        try:
            promocodes = await db.get_promocodes_list(created_by=user_id)
            # Преобразуем datetime в строки для JSON
            for p in promocodes:
                if p.get("created_at"):
                    p["created_at"] = p["created_at"].isoformat()
                if p.get("expires_at"):
                    p["expires_at"] = p["expires_at"].isoformat()
            return web.json_response({"promocodes": promocodes})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка промокодов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )


    async def admin_cards_create_handler(request: web.Request) -> web.Response:
        """
        Обработчик создания карты (только для админа).
        
        Принимает POST запрос с JSON данными:
        - name: название карты (обязательно)
        - description: описание карты (опционально)
        - rarity: редкость карты (обязательно, должна быть из списка допустимых)
        - power: сила карты (обязательно, целое число >= 0)
        
        Возвращает JSON с результатом создания карты и card_id.
        """
        # Извлекаем user_id из параметров запроса
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        # Пытаемся получить user_id из init_data (если это число)
        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        # Если не получилось, проверяем подпись init_data
        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        # Если все еще нет user_id, пытаемся получить из параметра user_id
        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        # Проверяем, что пользователь является админом
        if not user_id or user_id != ADMIN_ID:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        # Проверяем метод запроса
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            # Получаем данные из запроса
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "").strip()
            rarity = data.get("rarity", "common")
            power = int(data.get("power", 0))
            
            # Валидация названия карты
            if not name:
                return web.json_response({"error": "name_required"}, status=400)
            
            # Валидация редкости карты
            valid_rarities = ["common", "rare", "superrare", "epic", "legendary", "mythic", "divine", "limited", "start"]
            if rarity not in valid_rarities:
                return web.json_response({"error": "invalid_rarity"}, status=400)
            
            # Валидация силы карты
            if power < 0:
                return web.json_response({"error": "invalid_power"}, status=400)
            
            # Создаем карту в базе данных
            # image_file_id больше не используется - изображения берутся из DesignAssets/Cards/<card_id>.png
            result = await db.create_card(
                name=name,
                description=description,
                rarity=rarity,
                power=power,
                image_file_id=None,  # Больше не используется
                created_by=user_id
            )
            
            # Проверяем результат создания
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания карты"
                }, status=400)
            
            # Возвращаем успешный результат с ID созданной карты
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания карты для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_cards_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка карт (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        try:
            cards = await db.get_cards_list()
            # Преобразуем все datetime объекты в строки для JSON
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_items_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания предмета (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "").strip()
            rarity = data.get("rarity", "common")
            power = int(data.get("power", 0))
            image_file_id = data.get("image_file_id")
            image_file_id = image_file_id.strip() if image_file_id else None
            
            if not name:
                return web.json_response({"error": "name_required"}, status=400)
            
            if rarity not in ["common", "rare", "epic", "legendary", "mythic", "divine", "limited", "start"]:
                return web.json_response({"error": "invalid_rarity"}, status=400)
            
            result = await db.create_item(
                name=name,
                description=description,
                rarity=rarity,
                power=power,
                image_file_id=image_file_id,
                created_by=user_id
            )
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания предмета"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания предмета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_items_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка предметов (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        try:
            items = await db.get_items_list()
            # Преобразуем datetime в строки для JSON
            for item in items:
                if item.get("created_at"):
                    item["created_at"] = item["created_at"].isoformat()
            return web.json_response({"items": items})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка предметов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_presets_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка пресетов колод пользователя."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        try:
            presets = await db.get_user_deck_presets(user_id)
            # Подтягиваем данные героев одной пачкой, чтобы фронт сразу получил описание героя
            hero_ids = {p.get("hero_id", 0) for p in presets if p.get("hero_id") is not None}
            hero_map = await db.get_heroes_by_ids(list(hero_ids)) if hero_ids else {}
            for preset in presets:
                hero_id_value = preset.get("hero_id", 0)
                if not preset.get("hero") and hero_id_value in hero_map:
                    preset["hero"] = hero_map[hero_id_value]
            # Преобразуем datetime в строки для JSON
            for preset in presets:
                if preset.get("updated_at"):
                    preset["updated_at"] = preset["updated_at"].isoformat()
            return web.json_response({"presets": presets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения пресетов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_save_handler(request: web.Request) -> web.Response:
        """Обработчик сохранения пресета колоды (9 карт + герой)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number", 1))
            preset_name = data.get("preset_name", "Колода").strip()
            card_slots = data.get("card_slots", [])
            hero_id_raw = data.get("hero_id", 0)
            
            if len(card_slots) != DECK_SIZE:
                return web.json_response({"error": "invalid_slots_count"}, status=400)
            
            # Преобразуем строки в int или None
            card_slots_processed = []
            for slot in card_slots:
                if slot is None or slot == "":
                    card_slots_processed.append(None)
                else:
                    try:
                        card_slots_processed.append(int(slot))
                    except (ValueError, TypeError):
                        card_slots_processed.append(None)

            try:
                hero_id_value = int(hero_id_raw) if hero_id_raw is not None else 0
            except (TypeError, ValueError):
                return web.json_response({"error": "invalid_hero_id"}, status=400)

            hero_exists = await db.get_hero(hero_id_value)
            if hero_id_value != 0 and not hero_exists:
                return web.json_response({"error": "hero_not_found"}, status=404)
            
            result = await db.save_deck_preset(
                user_id=user_id,
                preset_number=preset_number,
                preset_name=preset_name,
                card_slots=card_slots_processed,
                hero_id=hero_id_value,
            )
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка сохранения пресета"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка сохранения пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания нового пресета колоды."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_name = data.get("preset_name", "Новая колода").strip()
            
            if not preset_name:
                preset_name = "Новая колода"
            
            result = await db.create_deck_preset(
                user_id=user_id,
                preset_name=preset_name
            )
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания пресета"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_delete_handler(request: web.Request) -> web.Response:
        """Обработчик удаления пресета колоды."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number"))
            
            result = await db.delete_deck_preset(
                user_id=user_id,
                preset_number=preset_number
            )
            
            if not result["success"]:
                error_messages = {
                    "min_presets_required": "Нельзя удалить пресет. Минимум 2 пресета должно остаться."
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": result.get("message") or error_messages.get(result.get("error"), "Ошибка удаления пресета")
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_rename_handler(request: web.Request) -> web.Response:
        """Обработчик переименования пресета колоды."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number"))
            new_name = data.get("new_name", "").strip()
            
            if not new_name:
                return web.json_response({"error": "empty_name"}, status=400)
            
            result = await db.rename_deck_preset(
                user_id=user_id,
                preset_number=preset_number,
                new_name=new_name
            )
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка переименования пресета"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка переименования пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def cards_catalog_handler(_: web.Request) -> web.Response:
        """Публичный список карт с текущими статами (уровень 1)."""
        try:
            cards = await db.get_cards_list()
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения каталога карт: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def user_cards_handler(request: web.Request) -> web.Response:
        """Обработчик получения карт пользователя."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        try:
            cards = await db.get_user_cards(user_id)
            # Преобразуем все datetime объекты в строки для JSON
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_get_all_cards_handler(request: web.Request) -> web.Response:
        """Обработчик получения всех карт в коллекцию админа (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            result = await db.add_all_cards_to_user(user_id)
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка добавления карт"
                }, status=400)
            # Преобразуем все datetime объекты в строки для JSON
            result = _serialize_datetime(result)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения всех карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_delete_all_cards_handler(request: web.Request) -> web.Response:
        """Обработчик удаления всех карт из коллекции админа (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            result = await db.delete_all_user_cards(user_id)
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка удаления карт"
                }, status=400)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления всех карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def card_upgrade_handler(request: web.Request) -> web.Response:
        """Обработчик улучшения карты."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            card_id = int(data.get("card_id"))
            
            result = await db.upgrade_card(user_id, card_id)
            
            if not result["success"]:
                error_messages = {
                    "card_not_found": "Карта не найдена",
                    "insufficient_particles": f"Недостаточно частиц. Нужно: {result.get('required', 0)}, имеется: {result.get('current', 0)}",
                    "insufficient_coins": f"Недостаточно монет. Нужно: {result.get('required', 0)}, имеется: {result.get('current', 0)}",
                    "max_level_reached": "Карта уже достигла максимального уровня (10)"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка улучшения карты"),
                    "required": result.get("required"),
                    "current": result.get("current")
                }, status=400)
            
            # Преобразуем datetime объекты в строки
            result = _serialize_datetime(result)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка улучшения карты для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def card_add_particles_handler(request: web.Request) -> web.Response:
        """Обработчик добавления частиц к карте."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            card_id = int(data.get("card_id"))
            particles = int(data.get("particles", 0))
            
            if particles <= 0:
                return web.json_response({
                    "success": False,
                    "error": "invalid_particles",
                    "message": "Количество частиц должно быть больше 0"
                }, status=400)
            
            result = await db.add_particles_to_card(user_id, card_id, particles)
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка добавления частиц"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка добавления частиц для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def community_posts_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка постов коммьюнити."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    pass

        try:
            limit = int(request.rel_url.query.get("limit", 50))
            posts = await db.get_community_posts(limit=limit, user_id=user_id)
            # Преобразуем datetime в строки для JSON
            for post in posts:
                if post.get("created_at"):
                    post["created_at"] = post["created_at"].isoformat()
                # Убеждаемся, что likes_count есть
                if "likes_count" not in post:
                    post["likes_count"] = 0
                if "is_liked" not in post:
                    post["is_liked"] = False
            
            # Получаем фото авторов через Bot API используя author_id
            # Используем семафор для ограничения параллельных запросов к Telegram API
            # Telegram имеет ограничения на количество запросов в секунду (rate limiting)
            import aiohttp
            import asyncio
            
            # Создаем семафор для ограничения параллельных запросов (максимум 3 одновременно)
            # Это помогает избежать rate limiting со стороны Telegram API
            photo_semaphore = asyncio.Semaphore(3)
            
            async def get_user_photo(user_id: int, retry_count: int = 3) -> str | None:
                """
                Получить фото пользователя из Telegram API с повторными попытками.
                
                Args:
                    user_id: ID пользователя в Telegram
                    retry_count: Количество попыток при ошибке
                
                Returns:
                    URL фото или None, если не удалось получить
                """
                for attempt in range(retry_count):
                    try:
                        # Используем семафор для ограничения параллельных запросов
                        async with photo_semaphore:
                            async with _create_ssl_disabled_session() as session:
                                # Увеличиваем таймаут до 15 секунд для медленных соединений
                                timeout = aiohttp.ClientTimeout(total=15, connect=10)
                                
                                # Получаем список фото пользователя
                                url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
                                async with session.get(
                                    url, 
                                    params={"user_id": user_id, "limit": 1}, 
                                    timeout=timeout
                                ) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                            file_id = data["result"]["photos"][0][0]["file_id"]
                                            
                                            # Получаем путь к файлу
                                            file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
                                            async with session.get(
                                                file_url, 
                                                params={"file_id": file_id}, 
                                                timeout=timeout
                                            ) as file_resp:
                                                if file_resp.status == 200:
                                                    file_data = await file_resp.json()
                                                    if file_data.get("ok"):
                                                        file_path = file_data["result"]["file_path"]
                                                        photo_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                                                        import logging
                                                        logging.getLogger(__name__).info(f"Получено фото для user_id {user_id}: {photo_url}")
                                                        return photo_url
                                    elif resp.status == 429:
                                        # Rate limit - ждем перед повторной попыткой
                                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2, 4, 6 секунд
                                        import logging
                                        logging.getLogger(__name__).warning(
                                            f"Rate limit при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}, ждем {wait_time}с"
                                        )
                                        await asyncio.sleep(wait_time)
                                        continue
                    except asyncio.TimeoutError:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Таймаут при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Ошибка получения фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}: {e}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                
                return None
            
            # Получаем уникальные ID авторов
            author_ids = list(set(post.get("author_id") for post in posts if post.get("author_id")))
            # Получаем фото параллельно с ограничением через семафор
            photo_tasks = [get_user_photo(aid) for aid in author_ids]
            photo_results = await asyncio.gather(*photo_tasks, return_exceptions=True)
            photo_map = {aid: result for aid, result in zip(author_ids, photo_results) if not isinstance(result, Exception) and result}
            
            # Присваиваем фото постам
            for post in posts:
                if post.get("author_id") and post["author_id"] in photo_map:
                    post["author_photo_url"] = photo_map[post["author_id"]]
            return web.json_response({"posts": posts})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения постов коммьюнити: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def community_post_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания поста коммьюнити (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            photo_file_id = data.get("photo_file_id")
            photo_file_id = photo_file_id.strip() if photo_file_id else None

            if not title:
                return web.json_response({"error": "title_required"}, status=400)
            if not content:
                return web.json_response({"error": "content_required"}, status=400)

            result = await db.create_community_post(
                author_id=user_id,
                title=title,
                content=content,
                photo_file_id=photo_file_id
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания поста"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания поста для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def global_chat_messages_handler(request: web.Request) -> web.Response:
        """Обработчик получения сообщений глобального чата."""
        try:
            limit = int(request.rel_url.query.get("limit", 100))
            messages = await db.get_chat_messages(limit=limit)
            # Преобразуем datetime в строки для JSON
            for msg in messages:
                if msg.get("created_at"):
                    msg["created_at"] = msg["created_at"].isoformat()
            
            # Получаем фото пользователей через Bot API используя user_id
            # Используем семафор для ограничения параллельных запросов к Telegram API
            # Telegram имеет ограничения на количество запросов в секунду (rate limiting)
            import aiohttp
            import asyncio
            
            # Создаем семафор для ограничения параллельных запросов (максимум 3 одновременно)
            # Это помогает избежать rate limiting со стороны Telegram API
            photo_semaphore = asyncio.Semaphore(3)
            
            async def get_user_photo(user_id: int, retry_count: int = 3) -> str | None:
                """
                Получить фото пользователя из Telegram API с повторными попытками.
                
                Args:
                    user_id: ID пользователя в Telegram
                    retry_count: Количество попыток при ошибке
                
                Returns:
                    URL фото или None, если не удалось получить
                """
                for attempt in range(retry_count):
                    try:
                        # Используем семафор для ограничения параллельных запросов
                        async with photo_semaphore:
                            async with _create_ssl_disabled_session() as session:
                                # Увеличиваем таймаут до 15 секунд для медленных соединений
                                timeout = aiohttp.ClientTimeout(total=15, connect=10)
                                
                                # Получаем список фото пользователя
                                url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
                                async with session.get(
                                    url, 
                                    params={"user_id": user_id, "limit": 1}, 
                                    timeout=timeout
                                ) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                            file_id = data["result"]["photos"][0][0]["file_id"]
                                            
                                            # Получаем путь к файлу
                                            file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
                                            async with session.get(
                                                file_url, 
                                                params={"file_id": file_id}, 
                                                timeout=timeout
                                            ) as file_resp:
                                                if file_resp.status == 200:
                                                    file_data = await file_resp.json()
                                                    if file_data.get("ok"):
                                                        file_path = file_data["result"]["file_path"]
                                                        photo_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                                                        import logging
                                                        logging.getLogger(__name__).info(f"Получено фото для user_id {user_id}: {photo_url}")
                                                        return photo_url
                                    elif resp.status == 429:
                                        # Rate limit - ждем перед повторной попыткой
                                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2, 4, 6 секунд
                                        import logging
                                        logging.getLogger(__name__).warning(
                                            f"Rate limit при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}, ждем {wait_time}с"
                                        )
                                        await asyncio.sleep(wait_time)
                                        continue
                    except asyncio.TimeoutError:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Таймаут при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Ошибка получения фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}: {e}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                
                return None
            
            # Получаем уникальные ID пользователей
            user_ids = list(set(msg.get("user_id") for msg in messages if msg.get("user_id")))
            # Получаем фото параллельно с ограничением через семафор
            photo_tasks = [get_user_photo(uid) for uid in user_ids]
            photo_results = await asyncio.gather(*photo_tasks, return_exceptions=True)
            photo_map = {uid: result for uid, result in zip(user_ids, photo_results) if not isinstance(result, Exception) and result}
            
            # Присваиваем фото сообщениям
            for msg in messages:
                if msg.get("user_id") and msg["user_id"] in photo_map:
                    msg["user_photo_url"] = photo_map[msg["user_id"]]
            return web.json_response({"messages": messages})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения сообщений чата: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def global_chat_send_handler(request: web.Request) -> web.Response:
        """Обработчик отправки сообщения в глобальный чат."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            message = data.get("message", "").strip()

            if not message:
                return web.json_response({"error": "message_required"}, status=400)

            # Проверяем extra_pass для определения CD
            user_profile = await db.get_user_profile(user_id)
            has_extra_pass = user_profile and user_profile.get("extra_pass") == "active"
            cooldown_seconds = 3 if has_extra_pass else 15

            # Проверяем CD на отправку сообщений
            last_message = await db.fetchrow(
                """
                SELECT created_at FROM global_chat
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id
            )

            if last_message and last_message.get("created_at"):
                from datetime import datetime, timezone
                last_time = last_message["created_at"]
                
                # Если это datetime объект из БД
                if isinstance(last_time, datetime):
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
                elif isinstance(last_time, str):
                    # Парсим ISO формат
                    try:
                        if 'T' in last_time:
                            last_time = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                        else:
                            last_time = datetime.fromisoformat(last_time)
                        if last_time.tzinfo is None:
                            last_time = last_time.replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError) as e:
                        import logging
                        logging.getLogger(__name__).error(f"Ошибка парсинга времени: {e}, last_time={last_time}")
                        # Если не удалось распарсить, пропускаем проверку CD
                        last_time = None
                
                if last_time and isinstance(last_time, datetime):
                    now = datetime.now(timezone.utc)
                    time_diff = (now - last_time).total_seconds()
                    
                    if time_diff < cooldown_seconds:
                        remaining = int(cooldown_seconds - time_diff)
                        return web.json_response({
                            "success": False,
                            "error": "cooldown",
                            "message": f"Подождите {remaining} секунд перед отправкой следующего сообщения",
                            "cooldown_remaining": remaining
                        }, status=429)

            result = await db.create_chat_message(
                user_id=user_id,
                message=message
            )

            if not result["success"]:
                error_messages = {
                    "empty_message": "Сообщение не может быть пустым",
                    "message_too_long": "Сообщение слишком длинное (максимум 500 символов)"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка отправки сообщения")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка отправки сообщения для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    # ========== Хэндлеры для работы с кейсами ==========
    
    async def user_cases_handler(request: web.Request) -> web.Response:
        """Получить список неоткрытых кейсов пользователя."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response({"error": "user_id must be integer"}, status=400)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            # Система кейсов удалена, возвращаем пустой список
            # await db.sync_user_key_cases(user_id)
            # cases = await db.get_user_cases(user_id)
            return web.json_response({"success": True, "cases": []})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения кейсов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def user_case_detail_handler(request: web.Request) -> web.Response:
        """Получить информацию о конкретном кейсе пользователя."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response({"error": "user_id must be integer"}, status=400)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        user_case_id = request.match_info.get("user_case_id")
        if not user_case_id:
            return web.json_response({"error": "user_case_id_required"}, status=400)

        try:
            await db.sync_user_key_cases(user_id)
            user_case_id = int(user_case_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)
            return web.json_response({"success": True, "case": user_case})
        except ValueError:
            return web.json_response({"error": "invalid_user_case_id"}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения кейса %s для user_id %s: %s", user_case_id, user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_tap_handler(request: web.Request) -> web.Response:
        """Обработать тап по кейсу (один из 4 тапов с проверкой апгрейда тира)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response({"error": "user_id must be integer"}, status=400)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))
            current_tier = int(data.get("current_tier", 1))
            tap_number = int(data.get("tap_number"))  # 1-4

            if not user_case_id or not (1 <= tap_number <= 4):
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Проверяем, что кейс принадлежит пользователю
            await db.sync_user_key_cases(user_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)

            # Используем актуальный тир из БД, а не переданный параметр
            actual_tier = user_case["tier"]
            
            # Проверяем апгрейд тира
            new_tier = roll_tier_upgrade(actual_tier, tap_number)
            upgraded = new_tier > actual_tier

            # Обновляем тир в БД, если произошел апгрейд
            if upgraded:
                await db.update_case_tier(user_case_id, user_id, new_tier)
                actual_tier = new_tier

            return web.json_response({
                "success": True,
                "upgraded": upgraded,
                "old_tier": user_case["tier"],
                "new_tier": actual_tier,
                "tap_number": tap_number,
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка обработки тапа для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_open_handler(request: web.Request) -> web.Response:
        """Открыть кейс и получить награды."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response({"error": "user_id must be integer"}, status=400)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))
            tap_results = data.get("tap_results", [])  # Список из 4 тиров после каждого тапа

            if not user_case_id or len(tap_results) != 4:
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Обрабатываем открытие кейса
            await db.sync_user_key_cases(user_id)
            result = await process_case_opening(db, user_id, user_case_id, tap_results)
            
            if not result.get("success"):
                return web.json_response(result, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка открытия кейса для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_skip_handler(request: web.Request) -> web.Response:
        """Пропустить анимацию и сразу открыть кейс."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response({"error": "user_id must be integer"}, status=400)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))

            if not user_case_id:
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Получаем текущий тир кейса
            await db.sync_user_key_cases(user_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)

            # Симулируем все 4 тапа с апгрейдом тира
            current_tier = user_case["tier"]
            tap_results = []
            for tap_num in range(1, 5):  # Тапы 1-4
                new_tier = roll_tier_upgrade(current_tier, tap_num)
                if new_tier > current_tier:
                    current_tier = new_tier
                    # Обновляем тир в БД после каждого апгрейда
                    await db.update_case_tier(user_case_id, user_id, new_tier)
                tap_results.append(current_tier)

            # Открываем кейс
            result = await process_case_opening(db, user_id, user_case_id, tap_results)
            
            if not result.get("success"):
                return web.json_response(result, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка пропуска анимации для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def _prepare_and_cache_engine(
        request: web.Request,
        match_id: str,
        player_ids: list[int | str],
        is_bot: bool,
        bot_info: dict[str, Any] | None = None,
        player_decks: dict[str, int | None] | None = None,
    ) -> bool:
        """
        Собирает и кеширует BattleEngine для match_id.
        Возвращает True, если движок создан и помещен в ACTIVE_MATCHES.
        """
        logger = logging.getLogger(__name__)
        db_instance: Database = request.app["db"]

        # ------------------------------------------------------------------
        # Нормализация идентификаторов игроков (совместимость int <-> str)
        # ------------------------------------------------------------------
        # В разных местах пайплайна (JS → API → матчмейкер → движок) user_id может
        # оказаться строкой ("6803...") или числом (6803...). Если не нормализовать,
        # возникают тонкие баги:
        # - кеш колод в Database пишется под ключом "6803...", а движок читает по ключу 6803
        # - сравнения current_player_id/user_id дают ложные запреты хода
        #
        # Поэтому здесь централизованно приводим player_ids к «стабильному» виду:
        # - числовые строки -> int
        # - нечисловые значения -> str (или 0 как фолбэк)
        def _normalize_player_id(raw: Any) -> int | str:
            if raw is None:
                return 0
            if isinstance(raw, int):
                return raw
            try:
                return int(raw)  # "6803..." -> 6803...
            except Exception:
                return str(raw)

        normalized_player_ids: list[int | str] = []
        for raw in (player_ids or []):
            normalized_player_ids.append(_normalize_player_id(raw))
        # Гарантируем минимум двух участников, чтобы не падать на IndexError.
        while len(normalized_player_ids) < 2:
            normalized_player_ids.append(0)

        p1_id = normalized_player_ids[0]
        p2_id = normalized_player_ids[1]

        def _as_int(raw: int | str) -> int:
            try:
                return int(raw)
            except Exception:
                return 0

        p1_id_int = _as_int(p1_id)
        p2_id_int = _as_int(p2_id)

        # Загружаем профили игроков для получения имен и аватаров
        p1_name, p1_avatar_url = "Игрок 1", None
        p2_name, p2_avatar_url = ("Бот" if is_bot else "Игрок 2"), None
        
        try:
            # Загружаем профиль первого игрока
            if p1_id_int > 0:
                p1_profile = await db_instance.get_user_profile(p1_id_int)
                print(f"🔍 DEBUG PROFILE P1 (id={p1_id_int}): {p1_profile}")
                if p1_profile:
                    # Приоритет: display_name > nickname > name > username > fallback
                    p1_name = (
                        p1_profile.get("display_name") or 
                        p1_profile.get("nickname") or 
                        p1_profile.get("name") or 
                        p1_profile.get("username") or
                        f"Игрок {p1_id_int}"
                    )
                    # Для аватара проверяем img (основное поле), затем fallback на старые поля
                    p1_avatar_url = (
                        p1_profile.get("img") or 
                        p1_profile.get("photo_url") or 
                        p1_profile.get("avatar_file_id") or 
                        p1_profile.get("avatar_url")
                    )
                    print(f"✅ P1 name resolved: {p1_name}, avatar: {p1_avatar_url}")
            
            # Загружаем профиль второго игрока
            if is_bot:
                print(f"🔍 DEBUG: Attempting to resolve bot name for ID {p2_id_int}")
                
                # Для бота ПРИОРИТЕТ: bot_info -> профиль из БД -> дефолт "Бот"
                if bot_info:
                    # Сначала проверяем bot_info (данные из bot_factory)
                    p2_name = bot_info.get("name") or p2_name
                    p2_avatar_url = bot_info.get("avatar_url") or p2_avatar_url
                    print(f"🤖 BOT INFO: name={p2_name}, avatar={p2_avatar_url}")
                    logger.info("Battle init: bot name from bot_info: %s, avatar: %s", p2_name, p2_avatar_url)
                else:
                    print(f"⚠️ WARNING: bot_info is empty or None!")
                
                # ПРИНУДИТЕЛЬНО загружаем профиль из БД, если имя еще не определено или это дефолт
                if p2_id_int > 0 and (p2_name == "Бот" or not p2_name or not bot_info or not bot_info.get("name")):
                    print(f"🔄 FORCING DB profile load for bot ID {p2_id_int}")
                    p2_profile = await db_instance.get_user_profile(p2_id_int)
                    print(f"🔍 DEBUG PROFILE P2 BOT (id={p2_id_int}): {p2_profile}")
                    if p2_profile:
                        # Берем имя из БД
                        db_name = (
                            p2_profile.get("display_name") or 
                            p2_profile.get("nickname") or 
                            p2_profile.get("name") or 
                            p2_profile.get("username")
                        )
                        if db_name:
                            p2_name = db_name
                            print(f"✅ P2 BOT name FORCED from DB: {p2_name}")
                        
                        # Берем аватар из БД только если bot_info не предоставил аватар
                        if not p2_avatar_url:
                            p2_avatar_url = (
                                p2_profile.get("img") or 
                                p2_profile.get("photo_url") or 
                                p2_profile.get("avatar_file_id") or 
                                p2_profile.get("avatar_url")
                            )
                            if p2_avatar_url:
                                print(f"✅ P2 BOT avatar from DB: {p2_avatar_url}")
                    else:
                        print(f"❌ WARNING: No profile found in DB for bot ID {p2_id_int}")
                
                print(f"📝 FINAL BOT DATA: name={p2_name}, avatar={p2_avatar_url}")
            else:
                # Для обычного игрока (не бота) загружаем профиль из БД
                if p2_id_int > 0:
                    p2_profile = await db_instance.get_user_profile(p2_id_int)
                    print(f"🔍 DEBUG PROFILE P2 (id={p2_id_int}): {p2_profile}")
                    if p2_profile:
                        p2_name = (
                            p2_profile.get("display_name") or 
                            p2_profile.get("nickname") or 
                            p2_profile.get("name") or 
                            p2_profile.get("username") or
                            f"Игрок {p2_id_int}"
                        )
                        p2_avatar_url = (
                            p2_profile.get("img") or 
                            p2_profile.get("photo_url") or 
                            p2_profile.get("avatar_file_id") or 
                            p2_profile.get("avatar_url")
                        )
                        print(f"✅ P2 name resolved: {p2_name}, avatar: {p2_avatar_url}")
                
            logger.info("Battle init: loaded player names p1=%s, p2=%s", p1_name, p2_name)
        except Exception as profile_exc:
            logger.warning("Battle init: failed to load player profiles: %s", profile_exc)
            print(f"❌ ERROR loading profiles: {profile_exc}")

        try:
            p1_deck_id = player_decks.get(str(p1_id_int)) if player_decks else None
            p2_deck_id = player_decks.get(str(p2_id_int)) if player_decks else None
            
            (p1_raw_deck, p1_hero_hp), (p2_raw_deck, p2_hero_hp) = await asyncio.wait_for(
                asyncio.gather(
                    _load_player_deck_and_hero(p1_id_int, p1_deck_id),
                    _load_player_deck_and_hero(p2_id_int, p2_deck_id),
                ),
                timeout=3.0,
            )
            logger.info(
                "Battle init: decks loaded (p1=%s cards, p2=%s cards)",
                len(p1_raw_deck or []),
                len(p2_raw_deck or []),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Battle init: deck load timed out, falling back to empty seeds (p1=%s, p2=%s)",
                p1_id,
                p2_id,
            )
            p1_raw_deck, p2_raw_deck, p1_hero_hp, p2_hero_hp = [], [], None, None
        except Exception as deck_exc:  # noqa: BLE001 - не блокируем бой при сбоях загрузки
            logger.warning(
                "Battle init: deck load failed (%s / %s): %s",
                p1_id,
                p2_id,
                deck_exc,
                exc_info=True,
            )
            p1_raw_deck, p2_raw_deck, p1_hero_hp, p2_hero_hp = [], [], None, None

        try:
            card_cache = await asyncio.wait_for(_load_card_cache(), timeout=3.0)
            logger.info("Battle init: card cache loaded, items=%s", len(card_cache))
        except asyncio.TimeoutError:
            logger.warning("Battle init: card cache load timed out, using empty cache")
            card_cache = {}
        except Exception as cache_exc:  # noqa: BLE001
            logger.warning("Battle init: card cache load failed: %s", cache_exc, exc_info=True)
            card_cache = {}

        p1_deck_ids = _normalize_deck_with_cache(p1_raw_deck, card_cache)
        p2_deck_ids = _normalize_deck_with_cache(p2_raw_deck, card_cache)

        try:
            deck_cache = getattr(db_instance, "deck_presets_cache", None)
            if not isinstance(deck_cache, dict):
                deck_cache = {}
            # Важно: кладем под int-ключи, потому что BattleEngine читает кеш по int(user_id).
            deck_cache[p1_id_int] = {"cards": list(p1_deck_ids)}
            deck_cache[p2_id_int] = {"cards": list(p2_deck_ids)}
            db_instance.deck_presets_cache = deck_cache
        except Exception as cache_exc:  # noqa: BLE001 - кеш не критичен для старта боя
            logger.debug("Не удалось записать deck_presets_cache: %s", cache_exc)

        try:
            engine = BattleEngine(
                db=db_instance,
                match_id=match_id,
                player_ids=normalized_player_ids,
                is_bot_match=is_bot,
                card_cache=card_cache,
                active_matches=request.app["active_matches"],
                event_emitter=request.app.get("event_emitter"),
            )
            
            # Вызываем create_match для инициализации core/engine.ArenaEnvironment
            # Конвертируем deck_ids в int для совместимости с converter.deck_from_card_ids
            p1_deck_int_ids = [int(str(d).split(":")[0]) for d in p1_deck_ids if d]
            p2_deck_int_ids = [int(str(d).split(":")[0]) for d in p2_deck_ids if d]
            
            # КРИТИЧНО: Извлекаем difficulty из bot_info для передачи в движок
            bot_difficulty = "lite"  # Дефолт для новичков
            if is_bot and bot_info:
                bot_difficulty = bot_info.get("difficulty", "lite")
            
            create_result = await engine.create_match(
                match_id=match_id,
                p1_data={
                    "user_id": p1_id_int,
                    "deck_ids": p1_deck_int_ids,
                    "name": p1_name,
                    "avatar_url": p1_avatar_url,
                    "is_bot": False,
                    "trophies": 0,
                },
                p2_data={
                    "user_id": p2_id_int,
                    "deck_ids": p2_deck_int_ids,
                    "name": p2_name,
                    "avatar_url": p2_avatar_url,
                    "is_bot": is_bot,
                    "trophies": 0,
                    "difficulty": bot_difficulty,  # КРИТИЧНО: Передаем сложность бота
                },
            )
            
            if not create_result.get("success"):
                logger.error("Battle init: create_match failed: %s", create_result.get("error"))
                return False
                
        except Exception as engine_exc:  # noqa: BLE001
            logger.error(
                "Battle init: engine creation failed for match_id=%s players=%s: %s",
                match_id,
                player_ids,
                engine_exc,
                exc_info=True,
            )
            return False

        request.app["active_matches"][match_id] = engine
        logger.info("Battle init: engine cached for match_id=%s", match_id)

        # УДАЛЕНО: Больше не запускаем бота сразу после создания движка
        # Теперь бот запускается только после получения сигнала 'client_ready' от фронтенда
        # Это предотвращает преждевременный ход бота до того, как игрок загрузит состояние боя
        print(f"!!! [SERVER] _prepare_and_cache_engine: движок создан для match_id={match_id}, ждём сигнала client_ready")
        logger.info("Battle init: engine ready, waiting for client_ready signal for match_id=%s", match_id)

        return True

    async def match_find_handler(request: web.Request) -> web.Response:
        """Найти матч: Soft Start <300 трофеев и очередь для остальных."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        user_id = data.get("user_id")
        trophies = data.get("trophies")
        user_avg_level = data.get("user_avg_level") or data.get("avg_level") or 1
        selected_deck_id = data.get("selected_deck_id") or data.get("deck_id")

        logger = logging.getLogger(__name__)
        try:
            user_id = int(user_id)
            trophies = int(trophies)
            user_avg_level = int(user_avg_level)
            if selected_deck_id is not None:
                selected_deck_id = int(selected_deck_id)
        except Exception:
            return web.json_response({"error": "invalid_parameters"}, status=400)

        matchmaker: Matchmaker = request.app["matchmaker"]
        try:
            logger.info(
                "match_find_handler: user_id=%s trophies=%s avg_level=%s deck_id=%s",
                user_id, trophies, user_avg_level, selected_deck_id
            )
            result = await matchmaker.find_match(user_id, trophies, user_avg_level, selected_deck_id)

            # Если матч найден сразу - прогреваем движок до ответа, чтобы арена
            # по прямому редиректу не упала с 404.
            if result.get("status") == "found":
                match_id = str(result.get("match_id"))
                # Совместимость с несколькими форматами матчмейкера:
                # - старый: user_id + opponent_id
                # - новый: player_ids=[p1, p2]
                raw_player_ids = result.get("player_ids")
                player_ids = (
                    list(raw_player_ids)
                    if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                    else [result.get("user_id"), result.get("opponent_id")]
                )
                if match_id and match_id not in request.app["active_matches"]:
                    try:
                        # Извлекаем информацию о боте (если есть)
                        bot_info = result.get("bot_info") if result.get("is_bot") else None
                        logger.info(
                            "match_find_handler: preparing engine for match_id=%s, is_bot=%s, bot_info=%s",
                            match_id,
                            result.get("is_bot"),
                            bot_info
                        )
                        print(f"🔍 [MATCH_FIND] bot_info from matchmaker: {bot_info}")
                        
                        ok = await asyncio.wait_for(
                            _prepare_and_cache_engine(
                                request,
                                match_id=match_id,
                                player_ids=player_ids,
                                is_bot=bool(result.get("is_bot")),
                                bot_info=bot_info,
                                player_decks=result.get("player_decks"),
                            ),
                            timeout=6.0,
                        )
                        logger.info("match_find_handler battle init result: %s (match_id=%s)", ok, match_id)
                        if not ok:
                            return web.json_response({"status": "error", "message": "battle_init_failed"}, status=500)
                    except asyncio.TimeoutError:
                        logging.getLogger(__name__).error(
                            "match_find_handler: battle init timeout (match_id=%s)", match_id, exc_info=True
                        )
                        return web.json_response({"status": "error", "message": "battle_init_timeout"}, status=504)
                    except Exception as exc:  # noqa: BLE001
                        logging.getLogger(__name__).error(
                            "match_find_handler: battle init failed (match_id=%s): %s", match_id, exc, exc_info=True
                        )
                        return web.json_response({"status": "error", "message": "battle_init_failed"}, status=500)

            if result.get("status") == "found":
                match_id = str(result.get("match_id"))
                if match_id:
                    current_state = None
                    engine = request.app["active_matches"].get(match_id)
                    if engine:
                        try:
                            current_state = engine.get_full_state() if hasattr(engine, "get_full_state") else engine.get_state()
                        except Exception:
                            current_state = None
                    logger.info(
                        "match_find_handler battle snapshot: match_id=%s state=%s",
                        match_id,
                        current_state,
                    )

            logger.info("match_find_handler response: %s", result)
            return web.json_response(result)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка матчмейкинга: %s", e, exc_info=True)
            return web.json_response({"error": "matchmaking_failed"}, status=500)

    async def match_vs_bot_handler(request: web.Request) -> web.Response:
        """Немедленный бой против бота с заданной сложностью."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        user_id = data.get("user_id")
        difficulty = data.get("difficulty", "medium")
        selected_deck_id = data.get("deck_id")

        valid_difficulties = ("lite", "easy", "medium", "hard", "max")
        if difficulty not in valid_difficulties:
            difficulty = "medium"

        try:
            user_id = int(user_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_user_id"}, status=400)

        matchmaker: Matchmaker = request.app["matchmaker"]
        try:
            result = await matchmaker._create_bot_match(
                user_id=user_id,
                trophies=0,
                user_avg_level=1,
                selected_deck_id=selected_deck_id,
            )
            # Override difficulty with the user's explicit choice
            if result.get("bot_info"):
                result["bot_info"]["difficulty"] = difficulty
            else:
                result["bot_info"] = {"difficulty": difficulty}

            match_id = str(result["match_id"])
            player_ids = result.get("player_ids", [user_id, result.get("opponent_id", -1)])
            bot_info = result.get("bot_info")

            ok = await asyncio.wait_for(
                _prepare_and_cache_engine(
                    request,
                    match_id=match_id,
                    player_ids=player_ids,
                    is_bot=True,
                    bot_info=bot_info,
                    player_decks=result.get("player_decks"),
                ),
                timeout=10.0,
            )
            if not ok:
                return web.json_response({"error": "battle_init_failed"}, status=500)

            return web.json_response({"status": "found", "match_id": match_id})
        except asyncio.TimeoutError:
            return web.json_response({"error": "timeout"}, status=504)
        except Exception as e:
            logging.getLogger(__name__).error("vs_bot error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    async def match_status_handler(request: web.Request) -> web.Response:
        """Статус матча по match_id для периодического поллинга фронта."""
        match_id = request.rel_url.query.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        matchmaker: Matchmaker = request.app["matchmaker"]
        try:
            result = await matchmaker.get_status(match_id)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка статуса матчмейкинга: %s", e, exc_info=True)
            return web.json_response({"error": "matchmaking_failed"}, status=500)

        if result.get("status") == "not_found":
            return web.json_response(result, status=404)

        # Если матч найден, но движок еще не создан (например, матчмейкер вернул статус
        # в поллинге), создаем и кешируем BattleEngine здесь же.
        if result.get("status") == "found":
            match_id = str(result.get("match_id"))
            if match_id and match_id not in request.app["active_matches"]:
                try:
                    # Совместимость с несколькими форматами матчмейкера:
                    # - старый: user_id + opponent_id
                    # - новый: player_ids=[p1, p2]
                    raw_player_ids = result.get("player_ids")
                    player_ids = (
                        list(raw_player_ids)
                        if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                        else [result.get("user_id"), result.get("opponent_id")]
                    )
                    logging.getLogger(__name__).info(
                        "match_status_handler: preparing engine match_id=%s players=%s", match_id, player_ids
                    )
                    # Извлекаем информацию о боте (если есть)
                    bot_info = result.get("bot_info") if result.get("is_bot") else None
                    ok = await asyncio.wait_for(
                        _prepare_and_cache_engine(
                            request,
                            match_id=match_id,
                            player_ids=player_ids,
                            is_bot=bool(result.get("is_bot")),
                            bot_info=bot_info,
                        ),
                        timeout=5.0,
                    )
                    if not ok:
                        return web.json_response({"error": "battle_init_failed"}, status=500)
                    else:
                        engine = request.app["active_matches"].get(match_id)
                        try:
                            snapshot = engine.get_full_state() if engine and hasattr(engine, "get_full_state") else None
                        except Exception:
                            snapshot = None
                        logging.getLogger(__name__).info(
                            "match_status_handler battle snapshot: match_id=%s state=%s",
                            match_id,
                            snapshot,
                        )
                except asyncio.TimeoutError:
                    logging.getLogger(__name__).error(
                        "match_status_handler: battle init timeout (match_id=%s)", match_id, exc_info=True
                    )
                    return web.json_response({"error": "battle_init_timeout"}, status=504)
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).error(
                        "Не удалось инициализировать бой при поллинге статуса (match_id=%s): %s",
                        match_id,
                        exc,
                        exc_info=True,
                    )
                    return web.json_response({"error": "battle_init_failed"}, status=500)

        return web.json_response(result)

    def _get_match_engine(match_id: str) -> BattleEngine | None:
        """
        Безопасно достаем движок боя из глобального кеша.
        Держим функцию рядом с хендлерами, чтобы не плодить дубли.
        """
        return ACTIVE_MATCHES.get(match_id)

    def _extract_engine_state(engine: BattleEngine) -> Any:
        """
        Унифицируем получение текущего состояния боя, чтобы не зависеть
        от конкретной версии BattleEngine (state attr или get_state()).
        """
        if hasattr(engine, "get_state"):
            return engine.get_state()
        if hasattr(engine, "state"):
            return getattr(engine, "state")
        # Фолбэк - минимальный срез, чтобы фронт хотя бы видел чей ход.
        return {
            "current_player": getattr(engine, "current_player_id", None),
            "turn": getattr(engine, "turn", 0),
        }

    def _decorate_card_image(card_obj: Card) -> Card:
        """
        Проставляем путь к изображению в DesignAssets/Cards/<id>.png, чтобы
        движок и фронт тянули файлы напрямую из статики, независимо от БД.
        """
        try:
            card_obj.image_url = f"{CARD_IMAGE_URL_PREFIX}/{card_obj.id}.png"
        except Exception:
            # Если по какой-то причине объект не допускает атрибут, тихо продолжаем.
            pass
        return card_obj

    async def _load_card_cache() -> dict[str, Card]:
        """
        Загружаем полный каталог карт и приводим к dict[str, Card].
        Нужен для BattleEngine, чтобы сразу возвращать названия и статы.
        """
        cache: dict[str, Card] = {}
        try:
            cards = await db.get_cards_list()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Не удалось загрузить каталог карт: %s", exc, exc_info=True)
            return cache

        for entry in cards:
            try:
                card_obj = entry if isinstance(entry, Card) else Card.from_row(entry)
                # Прописываем url картинки, чтобы дальше в бою приходила статика из DesignAssets.
                card_obj = _decorate_card_image(card_obj)
                cache[str(card_obj.id)] = card_obj
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug("Пропущена карта при построении кеша: %s", exc)
        return cache

    def _normalize_deck_with_cache(deck_ids: list[Any], card_cache: dict[str, Card]) -> list[str]:
        """
        Оставляем только валидные card_id, совпадающие с кешом карт.
        Если валидных карт нет, подбираем первые доступные из кеша.
        """
        normalized: list[str] = []
        available_ids = set(card_cache.keys())

        for raw in deck_ids or []:
            try:
                if isinstance(raw, dict):
                    cid = raw.get("id")
                    level = raw.get("level", 1)
                    candidate = f"{cid}:{level}" if cid is not None else None
                elif isinstance(raw, str) and ":" in raw:
                    candidate = raw
                    cid = raw.split(":", 1)[0]
                else:
                    cid = raw
                    candidate = str(raw) if raw is not None else None
                if cid is None or candidate is None:
                    continue
                if str(cid) in available_ids:
                    normalized.append(candidate)
            except Exception:
                continue

        if not normalized and available_ids:
            # Подстраховка: выдаем первые 9 карт из кеша, чтобы бой мог стартовать.
            normalized = list(list(available_ids)[:9])

        return normalized

    def _resolve_user_id_from_payload(payload: dict[str, Any]) -> int | None:
        """
        Унифицированное извлечение user_id из JSON тела (_auth или явный user_id).
        """
        if not payload:
            return None
        if "user_id" in payload:
            try:
                return int(payload.get("user_id"))
            except Exception:
                pass

        auth_token = payload.get("_auth") or payload.get("auth")
        if not auth_token:
            return None
        verified = _verify_init_data(str(auth_token), bot_token)
        if not verified:
            return None
        return _extract_user_id_from_init_data(verified)

    async def _load_player_deck_and_hero(user_id: int, selected_deck_id: int | None = None) -> tuple[list[str], int | None]:
        """
        Асинхронно загружаем выбранный пресет игрока (список из 9 карт, включая героя).
        
        НОВАЯ ЛОГИКА (Герой внутри колоды):
        - Загружаем все 9 слотов колоды
        - Герой находится внутри колоды как карта с card_type == 'hero'
        - BattleEngine сам найдет героя и отфильтрует его из игровой колоды
        - Возвращаем hero_hp=None, чтобы BattleEngine использовал героя из колоды
        
        Возвращает: (deck_ids: list[str], hero_hp: None)
        """
        is_bot_user = False
        try:
            # Определяем, бот ли пользователь, чтобы приоритетно брать «бот-колоду».
            bot_flag = await db.fetchval("SELECT COALESCE(is_bot, FALSE) FROM users WHERE user_id = $1", user_id)
            is_bot_user = bool(bot_flag)
        except Exception as exc:  # noqa: BLE001 - не прерываем бой, даже если проверка флагов упала
            logging.getLogger(__name__).warning(
                "Не удалось проверить флаг is_bot для %s: %s", user_id, exc, exc_info=True
            )
            # Боты генерируются в выделенном диапазоне id, поэтому используем его как эвристику.
            is_bot_user = user_id >= 900_000_000

        try:
            presets = await db.get_user_deck_presets(user_id)
        except Exception as exc:  # noqa: BLE001 - логируем любые сбои БД
            logging.getLogger(__name__).warning(
                "Не удалось получить колоду пользователя %s: %s", user_id, exc, exc_info=True
            )
            return [], None

        if not presets:
            return [], None

        # Пытаемся найти запрошенный пресет
        preset = None
        if selected_deck_id:
            preset = next((p for p in presets if p.get("preset_number") == selected_deck_id), None)
            if preset:
                logging.getLogger(__name__).info("Using selected deck preset %s for user %s", selected_deck_id, user_id)

        if not preset:
            # Для ботов отдаем приоритет пресетам, помеченным used_by_bot, чтобы не использовать случайные пользовательские данные.
            preset_candidates = presets
            if is_bot_user:
                bot_presets = [p for p in presets if p.get("used_by_bot")]
                if bot_presets:
                    preset_candidates = bot_presets

            # Берем первый непустой пресет среди кандидатов, иначе первый по порядку.
            preset = next(
                (p for p in preset_candidates if any(p.get(f"card_slot_{idx}") for idx in range(1, 10))),
                preset_candidates[0],
            )

        # Загружаем все 9 слотов колоды (включая героя)
        deck_ids = [
            str(preset.get(f"card_slot_{idx}"))
            for idx in range(1, 10)
            if preset.get(f"card_slot_{idx}") is not None
        ]

        if not deck_ids:
            logging.getLogger(__name__).warning(
                "Колода пользователя %s не найдена в БД (preset_number=%s, used_by_bot=%s)",
                user_id,
                preset.get("preset_number"),
                preset.get("used_by_bot"),
            )

        # ВАЖНО: Возвращаем hero_hp=None, чтобы BattleEngine использовал героя из колоды
        # Старое поле "hero" игнорируется - герой теперь часть колоды
        return deck_ids, None

    async def battle_state_handler(request: web.Request) -> web.Response:
        """Вернуть актуальное состояние боя из кешированного движка."""
        match_id = request.rel_url.query.get("match_id") or request.rel_url.query.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            # Попытка ленивой инициализации боя, если матч уже найден матчмейкером.
            try:
                matchmaker: Matchmaker = request.app["matchmaker"]
                status = await matchmaker.get_status(match_id)
                if status.get("status") == "found":
                    # Совместимость с несколькими форматами матчмейкера:
                    # - старый: user_id + opponent_id
                    # - новый: player_ids=[p1, p2]
                    raw_player_ids = status.get("player_ids")
                    player_ids = (
                        list(raw_player_ids)
                        if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                        else [status.get("user_id"), status.get("opponent_id")]
                    )
                    logging.getLogger(__name__).info(
                        "battle_state_handler: engine missing, preparing on-demand match_id=%s players=%s",
                        match_id,
                        player_ids,
                    )
                    # Извлекаем информацию о боте (если есть)
                    bot_info = status.get("bot_info") if status.get("is_bot") else None
                    ok = await asyncio.wait_for(
                        _prepare_and_cache_engine(
                            request,
                            match_id=match_id,
                            player_ids=player_ids,
                            is_bot=bool(status.get("is_bot")),
                            bot_info=bot_info,
                            player_decks=status.get("player_decks"),
                        ),
                        timeout=5.0,
                    )
                    if ok:
                        engine = _get_match_engine(match_id)
            except asyncio.TimeoutError:
                return web.json_response({"error": "battle_init_timeout"}, status=504)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).error(
                    "battle_state_handler: failed to init engine on-demand (match_id=%s): %s",
                    match_id,
                    exc,
                    exc_info=True,
                )

        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        # ДОБАВЛЕНО: Проверка истечения таймера хода
        if hasattr(engine, "is_turn_expired") and hasattr(engine, "current_player_id"):
            if engine.is_turn_expired() and not engine.is_ended:
                current_player = engine.current_player_id
                # Проверяем, что сейчас не ход бота (боту не нужен автозавершение по таймеру)
                is_bot_turn = engine.is_current_player_bot() if hasattr(engine, "is_current_player_bot") else False
                
                if not is_bot_turn and current_player:
                    logger = logging.getLogger(__name__)
                    logger.info(
                        "⏰ Turn timer expired for player %s in match %s, auto-ending turn",
                        current_player, match_id
                    )
                    try:
                        # Автоматически завершаем ход
                        engine.end_turn(current_player)
                        # Проверяем и запускаем бота если нужно
                        await check_and_run_bot(match_id, request.app["active_matches"])
                    except Exception as exc:
                        logger.warning("Failed to auto-end turn on timer expiry: %s", exc)

        try:
            # Получаем user_id из query параметров для правильного определения is_my_turn
            # КРИТИЧНО: Приводим к int, чтобы сравнение с current_player_id работало
            raw_user_id = request.rel_url.query.get("user_id")
            viewer_id = None
            if raw_user_id:
                try:
                    viewer_id = int(raw_user_id)
                except (ValueError, TypeError):
                    # Если не удалось конвертировать, оставляем None
                    viewer_id = None
            
            if hasattr(engine, "get_full_state"):
                # Передаем viewer_id — состояние включит legal_actions для текущего игрока
                state = engine.get_full_state(viewer_id=viewer_id)
            else:
                state = _extract_engine_state(engine)
                state["match_id"] = match_id
            
            # Загружаем ExtraPass статус игрока для премиального визуала
            if viewer_id is not None:
                try:
                    viewer_id_int = int(viewer_id) if not isinstance(viewer_id, int) else viewer_id
                    db_inst = request.app.get("db")
                    if db_inst:
                        user_data = await db_inst.get_user_info(viewer_id_int)
                        if user_data:
                            state["extra_pass"] = user_data.get("extra_pass", "inactive")
                except Exception as extra_pass_exc:
                    logging.getLogger(__name__).debug("Failed to load ExtraPass status: %s", extra_pass_exc)
            
            return web.json_response(state)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Не удалось получить состояние боя %s: %s", match_id, exc, exc_info=True
            )
            return web.json_response({"error": "battle_state_failed"}, status=500)

    async def battle_play_card_handler(request: web.Request) -> web.Response:
        """
        Розыгрыш карты игрока через core/actions.PlayCardAction.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        # Поддержка: card_id, card_id_from_hand, hand_index
        raw_card_id = payload.get("card_id") or payload.get("card_id_from_hand") or payload.get("hand_index")
        board_position = payload.get("target_position") or payload.get("board_position") or payload.get("position") or 0
        user_id_int = _resolve_user_id_from_payload(payload)
        
        # Параметры для зелий / battlecry
        target_id = payload.get("target_id")
        target_is_hero = payload.get("target_is_hero", False)

        logger = logging.getLogger(__name__)
        logger.info(
            "play_card_handler: match_id=%s card=%s pos=%s user=%s target=%s hero=%s",
            match_id, raw_card_id, board_position, user_id_int, target_id, target_is_hero
        )

        if not match_id or raw_card_id is None or user_id_int is None:
            return web.json_response({"error": "invalid_parameters"}, status=400)
        
        try:
            board_position = int(board_position)
        except (ValueError, TypeError):
            board_position = 0

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        try:
            result = engine.play_card(
                user_id_int, 
                raw_card_id, 
                board_position,
                target_id=target_id,
                target_is_hero=target_is_hero
            )
            
            # Обработка завершения игры
            if result.get("game_over"):
                logger.info("🏁 Game Over after play_card! Winner: %s", result.get("winner"))
                await _process_battle_end(request.app, match_id, engine, result.get("winner"))
                
                sio_inst = request.app.get("socketio")
                if sio_inst:
                    await sio_inst.emit("game_over", {
                        "game_over": True,
                        "winner_id": result.get("winner"),
                        "p1_hp": engine.p1_state.hero_hp,
                        "p2_hp": engine.p2_state.hero_hp,
                        "reason": "hero_death"
                    }, room=match_id)
            
            # Получаем состояние с legal_actions
            state = engine.get_full_state(viewer_id=user_id_int)
            
            # Добавляем данные о трофеях/монетах если game_over
            if result.get("game_over"):
                trophy_changes = getattr(engine, "_trophy_changes", {})
                trophy_totals = getattr(engine, "_trophy_totals", {})
                coins_changes = getattr(engine, "_coins_changes", {})
                coins_totals = getattr(engine, "_coins_totals", {})
                
                if user_id_int in trophy_changes:
                    state["trophy_delta"] = trophy_changes[user_id_int]
                    state["trophy_total"] = trophy_totals.get(user_id_int, 0)
                if user_id_int in coins_changes:
                    state["coins_delta"] = coins_changes[user_id_int]
                    state["coins_total"] = coins_totals.get(user_id_int, 0)
            
            # Рассылаем обновление всем через Socket.IO
            sio_inst = request.app.get("socketio")
            if sio_inst and not result.get("game_over"):
                await sio_inst.emit("state_changed", {
                    "match_id": match_id,
                    "action": "play_card",
                    "state": state
                }, room=match_id)
            
            # Проверяем, перешел ли ход к боту (если игра не завершена)
            if not result.get("game_over"):
                await trigger_bot_move(match_id)
            
            return web.json_response({"result": result, "state": state})
        except Exception as exc:
            logger.warning("Ошибка розыгрыша карты в матче %s: %s", match_id, exc, exc_info=True)
            return web.json_response({"error": "play_card_failed", "details": str(exc)}, status=400)

    async def battle_attack_handler(request: web.Request) -> web.Response:
        """
        Атака существом через core/actions.AttackAction.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        attacker_id = payload.get("attacker_id")
        target_id = payload.get("target_id")
        target_is_hero = bool(payload.get("target_is_hero"))
        user_id_int = _resolve_user_id_from_payload(payload)

        if not match_id or attacker_id is None or user_id_int is None:
            return web.json_response({"error": "invalid_parameters"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        logger = logging.getLogger(__name__)
        logger.info("attack_handler: match=%s attacker=%s target=%s hero=%s user=%s",
                    match_id, attacker_id, target_id, target_is_hero, user_id_int)

        try:
            result = engine.attack_target(
                user_id_int,
                attacker_id,
                target_id,
                target_is_hero=target_is_hero,
            )
            
            # Обработка завершения игры
            if result.get("game_over"):
                logger.info("🏁 Game Over after attack! Winner: %s", result.get("winner"))
                await _process_battle_end(request.app, match_id, engine, result.get("winner"))
                
                sio_inst = request.app.get("socketio")
                if sio_inst:
                    await sio_inst.emit("game_over", {
                        "game_over": True,
                        "winner_id": result.get("winner"),
                        "p1_hp": engine.p1_state.hero_hp,
                        "p2_hp": engine.p2_state.hero_hp,
                        "reason": "hero_death"
                    }, room=match_id)
            
            # Получаем состояние с legal_actions
            state = engine.get_full_state(viewer_id=user_id_int)
            
            # Добавляем данные о трофеях/монетах если game_over
            if result.get("game_over"):
                trophy_changes = getattr(engine, "_trophy_changes", {})
                trophy_totals = getattr(engine, "_trophy_totals", {})
                coins_changes = getattr(engine, "_coins_changes", {})
                coins_totals = getattr(engine, "_coins_totals", {})
                
                if user_id_int in trophy_changes:
                    state["trophy_delta"] = trophy_changes[user_id_int]
                    state["trophy_total"] = trophy_totals.get(user_id_int, 0)
                if user_id_int in coins_changes:
                    state["coins_delta"] = coins_changes[user_id_int]
                    state["coins_total"] = coins_totals.get(user_id_int, 0)
            
            # Рассылаем обновление всем через Socket.IO
            sio_inst = request.app.get("socketio")
            if sio_inst and not result.get("game_over"):
                await sio_inst.emit("state_changed", {
                    "match_id": match_id,
                    "action": "attack",
                    "state": state
                }, room=match_id)
            
            # Проверяем, перешел ли ход к боту (если игра не завершена)
            if not result.get("game_over"):
                await trigger_bot_move(match_id)
            
            return web.json_response({"result": result, "state": state})
        except Exception as exc:
            logger.warning("Ошибка атаки в матче %s: %s", match_id, exc, exc_info=True)
            return web.json_response({"error": "attack_failed", "details": str(exc)}, status=400)

    # calculate_trophy_delta и calculate_coins_reward вынесены на уровень модуля
    
    # [MOVED TO MODULE LEVEL]

    # УДАЛЕНО: check_and_run_bot и run_bot_routine перенесены на уровень модуля
    # См. определения функций перед create_web_app()

    async def battle_surrender_handler(request: web.Request) -> web.Response:
        """
        Обработчик сдачи игрока (surrender).
        Трофеи списываются немедленно, но бой продолжается под управлением бота.
        """
        logger = logging.getLogger(__name__)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        # Получаем match_id из URL-параметра
        match_id = request.match_info.get("match_id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        # Получаем user_id из payload
        user_id_int = _resolve_user_id_from_payload(payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        # Проверяем, что игрок участвует в матче
        if str(engine.p1_state.user_id) != str(user_id_int) and str(engine.p2_state.user_id) != str(user_id_int):
            return web.json_response({"error": "user_not_in_match"}, status=403)

        # Помечаем игрока как SURRENDERED (бот продолжит играть)
        engine.mark_surrender(user_id_int)
        
        # Получаем состояние сдавшегося игрока
        player_state = engine.get_player_state(user_id_int)
        
        # Проверяем, что трофеи ещё не были списаны
        if player_state.surrender_processed:
            logger.warning("Surrender already processed for player %s", user_id_int)
            return web.json_response({"error": "surrender_already_processed"}, status=400)
        
        # Получаем БД
        db = request.app.get("db")
        if not db:
            logger.error("Database not available")
            return web.json_response({"error": "database_unavailable"}, status=500)
        
        # Получаем текущие трофеи из БД
        try:
            user_info = await db.get_user_info(user_id_int)
            current_trophies = user_info.get("trophies", 0) if user_info else 0
        except Exception as exc:
            logger.error("Failed to get user trophies: %s", exc)
            current_trophies = 0
        
        # Рассчитываем штраф (максимальный для SURRENDERED)
        from core.state import ReplacementStatus
        penalty_delta, tier_name, tier_data = calculate_trophy_delta(
            current_trophies,
            is_winner=False,
            status=ReplacementStatus.SURRENDERED
        )
        
        # Списываем трофеи немедленно
        try:
            result = await db.update_user_trophies(user_id_int, penalty_delta)
            new_trophies = result.get("trophies", 0)
            
            logger.warning(
                "[SURRENDER_IMMEDIATE] Player %s lost %d trophies instantly (%d -> %d). Bot takeover initiated.",
                user_id_int, abs(penalty_delta), current_trophies, new_trophies
            )
            
            # Устанавливаем флаг, чтобы избежать двойного списания в конце боя
            player_state.surrender_processed = True
            
        except Exception as exc:
            logger.error("Failed to update trophies for surrendered player: %s", exc)
            return web.json_response({"error": "trophy_update_failed"}, status=500)
        
        # СРАЗУ вызываем проверку окончания игры (согласно правилам)
        game_over_result = engine.check_game_over()
        
        # Если матч завершился, обрабатываем финал
        if game_over_result.get("game_over"):
            winner_id = game_over_result.get("winner_id")
            await _process_battle_end(request.app, match_id, engine, winner_id)
            
            # Уведомляем через Socket.IO если доступно
            sio_inst = request.app.get("socketio")
            if sio_inst:
                await sio_inst.emit("game_over", {
                    "game_over": True,
                    "winner_id": winner_id,
                    "p1_hp": engine.p1_state.hero_hp,
                    "p2_hp": engine.p2_state.hero_hp,
                    "reason": "surrender"
                }, room=match_id)

        # Запускаем бота, если сейчас ход сдавшегося игрока и игра НЕ закончена
        if not game_over_result.get("game_over") and engine.current_player_id == user_id_int:
            await check_and_run_bot(match_id, ACTIVE_MATCHES)
        
        # Получаем текущее состояние
        current_state = engine.get_full_state(viewer_id=user_id_int)

        return web.json_response({
            "success": True,
            "message": "surrender_processed",
            "trophy_penalty": penalty_delta,
            "new_trophies": new_trophies,
            "state": current_state,
            "game_over": game_over_result.get("game_over", False)
        })

    async def battle_turn_end_handler(request: web.Request) -> web.Response:
        """
        Завершение хода игрока через core/actions.EndTurnAction.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)
        
        user_id_int = _resolve_user_id_from_payload(payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        logger = logging.getLogger(__name__)
        logger.info("turn_end_handler: match=%s user=%s", match_id, user_id_int)

        try:
            result = engine.end_turn(user_id_int)
            logger.info("turn_end_handler: success, current_player=%s", engine.get_current_player_id())
        except Exception as exc:
            logger.warning("Ошибка завершения хода для матча %s: %s", match_id, exc, exc_info=True)
            return web.json_response({"error": "turn_end_failed", "details": str(exc)}, status=400)

        # Запускаем бота, если ход перешел к нему
        try:
            await check_and_run_bot(match_id, ACTIVE_MATCHES)
        except Exception as exc:
            logger.warning("Не удалось запустить проверку бота: %s", exc, exc_info=True)

        # Получаем состояние с legal_actions для нового текущего игрока
        state = engine.get_full_state(viewer_id=user_id_int)
        
        # Рассылаем обновление всем через Socket.IO
        sio_inst = request.app.get("socketio")
        if sio_inst:
            await sio_inst.emit("turn_end", {
                "match_id": match_id,
                "action": "end_turn",
                "state": state,
                "new_turn_player": engine.get_current_player_id()
            }, room=match_id)

        return web.json_response({"match_id": match_id, "state": state})

    async def battle_preview_handler(request: web.Request) -> web.Response:
        """
        Предпросмотр урона для действия без его выполнения.
        Возвращает изменения HP объектов.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id")
        action_data = payload.get("action")
        
        if not match_id or not action_data:
            return web.json_response({"error": "match_id_and_action_required"}, status=400)
        
        user_id_int = _resolve_user_id_from_payload(payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        logger = logging.getLogger(__name__)
        logger.info("preview_handler: match=%s user=%s action=%s", match_id, user_id_int, action_data)

        # Парсим действие из payload
        try:
            from core.actions import PlayCardAction, AttackAction, EndTurnAction
            
            action_type = action_data.get("type")
            
            if action_type == "play_card":
                action = PlayCardAction(
                    hand_index=action_data.get("hand_index", 0),
                    target_id=action_data.get("target_id"),
                    position=action_data.get("position"),
                )
            elif action_type == "attack":
                action = AttackAction(
                    attacker_id=str(action_data.get("attacker_id", "")),
                    target_id=action_data.get("target_id"),
                    target_is_hero=action_data.get("target_is_hero", False),
                )
            elif action_type == "end_turn":
                action = EndTurnAction()
            else:
                return web.json_response({"error": "unknown_action_type"}, status=400)
            
        except Exception as exc:
            logger.warning("Ошибка парсинга действия: %s", exc, exc_info=True)
            return web.json_response({"error": "invalid_action_format", "details": str(exc)}, status=400)

        # Получаем предпросмотр
        try:
            preview_delta = engine.get_preview_delta(action)
            return web.json_response({
                "success": True,
                "preview_data": preview_delta
            })
        except Exception as exc:
            logger.warning("Ошибка получения предпросмотра: %s", exc, exc_info=True)
            return web.json_response({"error": "preview_failed", "details": str(exc)}, status=400)

    # ========== Регистрация роутов ==========
    
    app.router.add_get("/health", health_check)
    app.router.add_get("/", index)
    
    # === СТАТИКА: РАЗДАЕМ РЕСУРСЫ КАРТ (ДОЛЖНО БЫТЬ В НАЧАЛЕ) ===
    # Добавляем статическую раздачу ресурсов перед всеми остальными роутами
    import os
    # Путь от web/server.py -> .. -> DesignAssets
    design_assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DesignAssets"))
    if os.path.exists(design_assets_path):
        app.router.add_static("/DesignAssets/", path=design_assets_path, name="design_assets")
        logging.getLogger(__name__).info("✅ Static route added: /DesignAssets/ -> %s", design_assets_path)
        print(f"✅ [SERVER] Static route: /DesignAssets/ -> {design_assets_path}")
    else:
        logging.getLogger(__name__).error("❌ DesignAssets directory NOT FOUND at: %s", design_assets_path)
        print(f"❌ [SERVER] DesignAssets NOT FOUND at: {design_assets_path}")
    
    # Отдельная HTML-страница боя; /arena - основной путь редиректа из фронтенда
    app.router.add_get("/arena", battle_page_handler)   # основной маршрут для UI арены
    app.router.add_get("/battle", battle_page_handler)  # обратная совместимость
    app.router.add_get("/api/profile", profile_handler)
    app.router.add_post("/api/match/find", match_find_handler)
    app.router.add_post("/api/match/vs-bot", match_vs_bot_handler)
    app.router.add_get("/api/match/status", match_status_handler)
    app.router.add_get("/api/battle/state", battle_state_handler)
    # Новые роуты с дефисами (используются фронтендом)
    app.router.add_post("/api/battle/play-card", battle_play_card_handler)
    app.router.add_post("/api/battle/attack", battle_attack_handler)
    app.router.add_post("/api/battle/end-turn", battle_turn_end_handler)
    app.router.add_post("/api/battle/preview", battle_preview_handler)
    # Старые роуты для обратной совместимости
    app.router.add_post("/api/battle/action/play_card", battle_play_card_handler)
    app.router.add_post("/api/battle/action/attack_target", battle_attack_handler)
    app.router.add_post("/api/battle/turn_end", battle_turn_end_handler)
    app.router.add_post("/api/matches/{match_id}/surrender", battle_surrender_handler)
    app.router.add_get("/api/settings", settings_handler)
    app.router.add_post("/api/settings", settings_handler)
    app.router.add_post("/api/change-nickname", change_nickname_handler)
    app.router.add_post("/api/promocode/use", promocode_use_handler)
    app.router.add_post("/api/admin/promocodes/create", promocode_create_handler)
    app.router.add_get("/api/admin/promocodes/list", promocode_list_handler)
    app.router.add_post("/api/admin/cards/create", admin_cards_create_handler)
    app.router.add_get("/api/admin/cards/list", admin_cards_list_handler)
    app.router.add_post("/api/admin/items/create", admin_items_create_handler)
    app.router.add_get("/api/admin/items/list", admin_items_list_handler)
    app.router.add_get("/api/deck/presets", deck_presets_list_handler)
    app.router.add_post("/api/deck/presets/save", deck_preset_save_handler)
    app.router.add_post("/api/deck/presets/create", deck_preset_create_handler)
    app.router.add_post("/api/deck/presets/delete", deck_preset_delete_handler)
    app.router.add_post("/api/deck/presets/rename", deck_preset_rename_handler)
    app.router.add_get("/api/cards", cards_catalog_handler)
    app.router.add_get("/api/cards/user", user_cards_handler)
    app.router.add_get("/api/cases/user", user_cases_handler)
    app.router.add_get("/api/cases/{user_case_id}", user_case_detail_handler)
    app.router.add_post("/api/cases/tap", case_tap_handler)
    app.router.add_post("/api/cases/open", case_open_handler)
    app.router.add_post("/api/cases/skip", case_skip_handler)
    app.router.add_post("/api/admin/cards/get-all", admin_get_all_cards_handler)
    app.router.add_post("/api/admin/cards/delete-all", admin_delete_all_cards_handler)
    app.router.add_post("/api/cards/upgrade", card_upgrade_handler)
    app.router.add_post("/api/cards/add-particles", card_add_particles_handler)
    app.router.add_get("/api/admin/players", admin_players_handler)
    app.router.add_post("/api/admin/players", admin_players_handler)
    app.router.add_get("/api/admin/stats", admin_stats_handler)
    
    async def admin_tps_handler(request: web.Request) -> web.Response:
        """Обработчик получения TPS статистики (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        # Проверяем, что это админ
        ADMIN_ID = 6803854304
        if not user_id or user_id != ADMIN_ID:
            return web.json_response({"error": "Forbidden"}, status=403)

        try:
            from tps_monitor import get_tps_monitor
            
            monitor = get_tps_monitor()
            stats = monitor.get_statistics()
            
            return web.json_response(stats)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка получения TPS: {e}", exc_info=True)
            return web.json_response(
                {"error": f"Ошибка получения TPS: {str(e)}"}, status=500
            )
    
    app.router.add_get("/api/admin/tps", admin_tps_handler)
    
    async def admin_stars_test_mode_toggle_handler(request: web.Request) -> web.Response:
        """Обработчик переключения тестового режима Stars (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        # Проверяем, что это админ
        if not user_id or user_id not in app.get("admin_ids", set()):
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            # Переключаем тестовый режим
            current_mode = app.get("stars_test_mode", False)
            new_mode = not current_mode
            app["stars_test_mode"] = new_mode
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Тестовый режим Stars переключен администратором {user_id}: {current_mode} -> {new_mode}"
            )
            
            return web.json_response({
                "success": True,
                "stars_test_mode": new_mode,
                "message": f"Тестовый режим {'включен' if new_mode else 'выключен'}"
            })
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка переключения тестового режима Stars: {e}", exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )
    
    app.router.add_post("/api/admin/stars-test-mode/toggle", admin_stars_test_mode_toggle_handler)
    
    async def post_delete_handler(request: web.Request) -> web.Response:
        """Обработчик удаления поста (только для админа)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id or user_id != 6803854304:
            return web.json_response(
                {"error": "admin_access_required"}, status=403
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            
            if not post_id:
                return web.json_response({"error": "post_id_required"}, status=400)
            
            result = await db.delete_community_post(post_id, user_id)
            
            if not result["success"]:
                error_messages = {
                    "admin_only": "Только администратор может удалять посты",
                    "post_not_found": "Пост не найден"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка удаления поста")
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления поста для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def post_like_handler(request: web.Request) -> web.Response:
        """Обработчик лайка/дизлайка поста."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            
            if not post_id:
                return web.json_response({"error": "post_id_required"}, status=400)
            
            result = await db.toggle_post_like(post_id, user_id)
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка обработки лайка"
                }, status=400)
            
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка обработки лайка для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def yookassa_webhook_handler(request: web.Request) -> web.Response:
        """Обработчик вебхуков от YooKassa."""
        import logging
        webhook_logger = logging.getLogger(__name__)
        
        try:
            # Получаем сырые данные для логирования
            raw_data = await request.read()
            data = await request.json() if raw_data else {}
            
            webhook_logger.info(f"Получен вебхук от YooKassa: {data.get('event', 'unknown')}")
            
            payment_service = request.app.get("payment_service")
            
            if not payment_service:
                webhook_logger.error("Payment service не настроен")
                return web.json_response(
                    {
                        "error": "payment_service_not_configured",
                        "message": "Платежный сервис не настроен. Проверьте настройки YooKassa."
                    }, 
                    status=503
                )
            
            # Парсим уведомление от YooKassa (SDK автоматически проверяет подпись)
            webhook_data = payment_service.parse_webhook(data)
            
            if not webhook_data:
                webhook_logger.warning(f"Не удалось распарсить вебхук: {data}")
                return web.json_response(
                    {"error": "invalid_webhook"}, status=400
                )
            
            event = webhook_data.get("event")
            payment_id = webhook_data.get("payment_id")
            status = webhook_data.get("status")
            
            webhook_logger.info(
                f"Вебхук обработан: event={event}, payment_id={payment_id}, status={status}"
            )
            
            if not payment_id or not status:
                webhook_logger.warning(f"Отсутствуют обязательные данные в вебхуке: {webhook_data}")
                return web.json_response(
                    {"error": "missing_payment_data"}, status=400
                )
            
            # Получаем или создаем запись о платеже
            payment_record = await db.get_payment_by_id(payment_id)
            
            if not payment_record:
                # Если платежа нет в БД, возможно это первый вебхук
                # Создаем запись с данными из вебхука
                webhook_logger.warning(f"Платеж {payment_id} не найден в БД, создаем запись")
                metadata = webhook_data.get("metadata", {})
                user_id = metadata.get("user_id")
                
                if user_id:
                    await db.create_payment(
                        user_id=int(user_id),
                        payment_id=payment_id,
                        amount=float(webhook_data.get("amount", 0)),
                        currency=webhook_data.get("currency", "RUB"),
                        description=f"Платеж {payment_id}",
                        metadata=metadata
                    )
                    payment_record = await db.get_payment_by_id(payment_id)
            
            # Обновляем статус платежа в БД
            await db.update_payment_status(
                payment_id=payment_id,
                status=status
            )
            
            # Получаем обновленную запись после изменения статуса
            payment_record = await db.get_payment_by_id(payment_id)
            
            # Обрабатываем разные события
            if event == "payment.succeeded" and status == "succeeded" and webhook_data.get("paid"):
                if not payment_record:
                    webhook_logger.error("Платеж %s отсутствует в БД даже после попытки создания", payment_id)
                else:
                    # Унифицированная выдача наград и писем (общая для Stars и YooKassa)
                    processing_result = await process_successful_payment(
                        db=db,
                        payment_id=payment_id,
                        payment_record=payment_record,
                        source="yookassa_webhook",
                        logger=webhook_logger,
                    )
                    if processing_result["status"] == "already_processed":
                        webhook_logger.info("Платеж %s уже был обработан ранее, повтор не требуется", payment_id)
                    elif processing_result["status"] == "missing_payment":
                        webhook_logger.error("Платеж %s пропал во время обработки", payment_id)
                    elif processing_result.get("rewards_text"):
                        webhook_logger.info(
                            "Платеж %s обработан, награды: %s",
                            payment_id,
                            processing_result["rewards_text"],
                        )
                    else:
                        webhook_logger.info("Платеж %s обработан без дополнительных наград", payment_id)
            elif event == "payment.canceled" and status == "canceled":
                webhook_logger.info(f"❌ Платеж {payment_id} отменен")
                
            elif event == "payment.waiting_for_capture":
                webhook_logger.info(f"⏳ Платеж {payment_id} ожидает подтверждения")
            
            return web.json_response({"status": "ok"})
        except Exception as e:
            webhook_logger.error(
                f"Ошибка обработки вебхука YooKassa: {e}", exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def create_payment_handler(request: web.Request) -> web.Response:
        """Обработчик создания платежа."""
        import logging
        logger = logging.getLogger(__name__)
        
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        # Логируем для отладки (без чувствительных данных)
        logger.info(f"Создание платежа: init_data присутствует: {bool(init_data)}, длина: {len(init_data) if init_data else 0}")

        # Сначала пробуем как число (user_id напрямую)
        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
                logger.info(f"Использован user_id из цифрового параметра: {user_id}")
            except ValueError:
                pass

        # Если не получилось, пробуем проверить как initData
        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)
                logger.info(f"Использован user_id из проверенного initData: {user_id}")
            else:
                logger.warning("initData не прошел проверку подписи")

        # Fallback: пробуем user_id из отдельного параметра
        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                    logger.info(f"Использован user_id из параметра user_id: {user_id}")
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            logger.error("Не удалось определить user_id. init_data присутствует: %s", bool(init_data))
            return web.json_response(
                {"error": "authentication required", "message": "Не удалось определить пользователя. Убедитесь, что вы открыли игру через Telegram."}, status=401
            )

        payment_service = request.app.get("payment_service")
        if not payment_service:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Payment service не настроен в create_payment_handler")
            return web.json_response(
                {
                    "error": "payment_service_not_configured",
                    "message": "Платежный сервис не настроен. Проверьте настройки YooKassa в .env файле."
                }, 
                status=503
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            amount = float(data.get("amount", 0))
            description = data.get("description", "Покупка в ExtraCards")
            metadata = dict(data.get("metadata") or {})
            metadata["user_id"] = user_id
            metadata.setdefault("item_name", data.get("item_name") or metadata.get("item_type") or description)
            metadata.setdefault("amount_rub", amount)
            
            if amount <= 0:
                return web.json_response({"error": "invalid_amount"}, status=400)
            
            # Получаем URL для возврата после оплаты
            return_url = data.get("return_url", request.app.get("webapp_url", "https://t.me/your_bot"))
            
            # Создаем платеж в YooKassa
            logger.info(f"Создание платежа: amount={amount}, currency={data.get('currency', 'RUB')}, description={description}")
            payment_result = payment_service.create_payment(
                amount=amount,
                currency=data.get("currency", "RUB"),
                description=description,
                return_url=return_url,
                metadata=metadata
            )
            
            logger.info(f"Результат создания платежа: success={payment_result.get('success')}, error={payment_result.get('error')}")
            
            if not payment_result.get("success"):
                error_msg = payment_result.get("error", "unknown")
                logger.error(f"Ошибка создания платежа в YooKassa: {error_msg}")
                return web.json_response({
                    "success": False,
                    "error": error_msg,
                    "message": f"Ошибка создания платежа: {error_msg}"
                }, status=400)
            
            payment_id = payment_result.get("payment_id")
            
            # Сохраняем платеж в БД
            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=amount,
                currency=data.get("currency", "RUB"),
                description=description,
                metadata=metadata
            )
            
            if not db_result.get("success"):
                import logging
                logging.getLogger(__name__).warning(f"Не удалось сохранить платеж {payment_id} в БД: {db_result.get('error')}")
            
            return web.json_response({
                "success": True,
                "payment_id": payment_id,
                "confirmation_url": payment_result.get("confirmation_url"),
                "status": payment_result.get("status"),
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания платежа для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def get_payment_status_handler(request: web.Request) -> web.Response:
        """Обработчик получения статуса платежа."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        payment_id = request.rel_url.query.get("payment_id")
        if not payment_id:
            return web.json_response({"error": "payment_id_required"}, status=400)

        try:
            # Проверяем, что платеж принадлежит пользователю
            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                return web.json_response({"error": "payment_not_found"}, status=404)
            
            if payment_record["user_id"] != user_id:
                return web.json_response({"error": "access_denied"}, status=403)
            
            # Для платежей звездами (начинаются с "stars_") возвращаем статус из БД
            if payment_id.startswith("stars_"):
                return web.json_response({
                    "payment_id": payment_id,
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                    "paid": payment_record["status"] == "succeeded",
                })
            
            # Для платежей YooKassa используем payment_service
            payment_service = request.app.get("payment_service")
            if not payment_service:
                # Если payment_service не настроен, возвращаем статус из БД
                return web.json_response({
                    "payment_id": payment_id,
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                })
            
            # Получаем актуальный статус из YooKassa
            status_result = payment_service.get_payment_status(payment_id)
            
            if not status_result.get("success"):
                # Если не удалось получить из YooKassa, возвращаем из БД
                return web.json_response({
                    "payment_id": payment_id,
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                })
            
            # Обновляем статус в БД, если изменился
            if status_result.get("status") != payment_record["status"]:
                await db.update_payment_status(
                    payment_id=payment_id,
                    status=status_result["status"]
                )
                # Получаем обновленную запись
                payment_record = await db.get_payment_by_id(payment_id)
            
            # Добавляем информацию о rewards_processed в ответ
            status_result["rewards_processed"] = payment_record.get("rewards_processed", False)
            
            return web.json_response(status_result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения статуса платежа %s: %s", payment_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def user_mail_handler(request: web.Request) -> web.Response:
        """Обработчик получения писем пользователя."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        try:
            category = request.rel_url.query.get("category")
            unread_only = request.rel_url.query.get("unread_only", "false").lower() == "true"
            limit = int(request.rel_url.query.get("limit", 50))
            
            mail_list = await db.get_user_mail(
                user_id=user_id,
                category=category,
                limit=limit,
                unread_only=unread_only
            )
            
            # Преобразуем datetime в строки для JSON и добавляем mail_id для совместимости
            from datetime import datetime
            for mail in mail_list:
                if mail.get("created_at"):
                    if isinstance(mail["created_at"], datetime):
                        mail["created_at"] = mail["created_at"].isoformat()
                # Добавляем mail_id для совместимости с фронтендом
                if "id" in mail and "mail_id" not in mail:
                    mail["mail_id"] = mail["id"]
            
            unread_count = await db.get_unread_mail_count(user_id)
            
            return web.json_response({
                "mail": mail_list,
                "unread_count": unread_count
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения писем для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def mark_mail_read_handler(request: web.Request) -> web.Response:
        """Обработчик отметки письма как прочитанного."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            mail_id = int(data.get("mail_id"))
            
            result = await db.mark_mail_as_read(mail_id, user_id)
            
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown")
                }, status=400)
            
            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка отметки письма как прочитанного для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def get_unread_mail_count_handler(request: web.Request) -> web.Response:
        """Обработчик получения количества непрочитанных писем."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        try:
            unread_count = await db.get_unread_mail_count(user_id)
            
            return web.json_response({
                "count": unread_count
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения количества непрочитанных писем для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_get("/api/community/posts", community_posts_list_handler)
    app.router.add_post("/api/community/posts/create", community_post_create_handler)
    app.router.add_post("/api/community/posts/delete", post_delete_handler)
    app.router.add_post("/api/community/posts/like", post_like_handler)
    app.router.add_get("/api/community/chat/messages", global_chat_messages_handler)
    app.router.add_post("/api/community/chat/send", global_chat_send_handler)
    async def payment_config_handler(_: web.Request) -> web.Response:
        """Отдать конфигурацию платежей (курсы и коэффициенты)."""
        return web.json_response({
            "stars_rate_rub": app.get("stars_rate_rub", 1.5),
            "stars_markup": app.get("stars_markup", 1.2),
            "stars_test_mode": app.get("stars_test_mode", False),
        })
    
    # Обработчик создания инвойса через Telegram Stars
    async def create_stars_invoice_handler(request: web.Request) -> web.Response:
        """Обработчик создания инвойса через Telegram Stars."""
        import logging
        logger = logging.getLogger(__name__)
        
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            # Безопасная обработка amount_rub с проверкой на None
            amount_rub_value = data.get("amount_rub") or data.get("amount") or 0
            try:
                amount_rub = float(amount_rub_value)
            except (ValueError, TypeError):
                return web.json_response({"error": "invalid_amount", "message": "amount_rub must be a valid number"}, status=400)
            
            description = data.get("description", "Покупка в ExtraCards")
            metadata = dict(data.get("metadata") or {})
            metadata["user_id"] = user_id
            metadata.setdefault("amount_rub", amount_rub)
            
            # Убеждаемся, что item_type сохраняется в metadata (даже если он был None/undefined)
            item_type = metadata.get("item_type")
            if item_type:
                metadata["item_type"] = str(item_type)
            elif item_type is None or item_type == "":
                # Если item_type не передан, логируем предупреждение
                logger.warning(
                    f"Платеж Stars: item_type не указан в metadata для user_id={user_id}, "
                    f"description={description}, metadata={metadata}"
                )
            
            item_name = metadata.get("item_name") or data.get("item_name") or metadata.get("item_type") or description or "Покупка в ExtraCards"
            metadata["item_name"] = item_name
            
            # Логируем metadata для отладки
            logger.info(
                f"Создание платежа Stars: user_id={user_id}, item_type={metadata.get('item_type')}, "
                f"item_name={item_name}, description={description}"
            )
            
            if amount_rub <= 0:
                return web.json_response({"error": "invalid_amount", "message": "amount_rub must be greater than 0"}, status=400)
            
            stars_rate_rub = float(app.get("stars_rate_rub", 1.5))
            stars_markup = float(app.get("stars_markup", 1.2))
            stars_test_mode = bool(app.get("stars_test_mode", False))
            is_admin_user = user_id in app.get("admin_ids", set())
            stars_amount = max(1, math.ceil((amount_rub / stars_rate_rub) * stars_markup))
            if stars_test_mode and is_admin_user:
                logger.info(
                    "Stars test mode активен: пользователь %s (admin) платит фиксированные 1 ⭐",
                    user_id,
                )
                stars_amount = 1
            
            # Создаем уникальный invoice_payload для идентификации платежа
            import uuid
            invoice_payload = str(uuid.uuid4())
            
            # Сохраняем информацию о платеже в БД до отправки инвойса
            payment_id = f"stars_{invoice_payload}"
            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=float(stars_amount),
                currency="XTR",  # Telegram Stars
                description=description,
                metadata={
                    **metadata,
                    "amount_rub": amount_rub,
                    "stars_rate_rub": stars_rate_rub,
                    "stars_markup": stars_markup,
                    "stars_amount": stars_amount,
                    "stars_test_mode": stars_test_mode,
                    "is_admin_test_purchase": stars_test_mode and is_admin_user,
                },
                status="pending"  # Платеж ожидает подтверждения
            )
            
            if not db_result.get("success"):
                logger.warning(f"Не удалось сохранить платеж {payment_id} в БД: {db_result.get('error')}")
            
            # Отправляем инвойс через Telegram Bot API
            from aiogram import Bot
            bot = Bot(token=bot_token)
            
            try:
                # Формируем детали счета с пояснениями для пользователя
                detailed_message = (
                    "<b>🧾 Счет на оплату</b>\n"
                    f"Товар: {item_name}\n"
                    f"Сумма: {amount_rub:.2f} ₽ → {stars_amount} ⭐\n"
                    f"ID платежа: <code>{payment_id}</code>\n"
                    "После оплаты вернитесь в игру - награды выдаются автоматически."
                )
                await bot.send_message(user_id, detailed_message, parse_mode="HTML")

                # Формируем параметры для sendInvoice
                invoice_title = item_name[:32]  # Максимум 32 символа
                invoice_description = (
                    f"{item_name} • {amount_rub:.2f} ₽ • {stars_amount} ⭐\n"
                    "После оплаты вернитесь в ExtraArena."
                )[:255]
                
                # Создаем LabeledPrice для инвойса
                from aiogram.types import LabeledPrice
                prices = [LabeledPrice(label=invoice_title, amount=stars_amount)]
                
                # Отправляем инвойс
                sent_message = await bot.send_invoice(
                    chat_id=user_id,
                    title=invoice_title,
                    description=invoice_description,
                    payload=invoice_payload,
                    provider_token=None,  # Для Stars не нужен provider_token
                    currency="XTR",  # Telegram Stars
                    prices=prices,
                    start_parameter=invoice_payload[:64],  # Максимум 64 символа
                )
                
                logger.info(
                    f"Инвойс Stars отправлен пользователю {user_id}, invoice_payload={invoice_payload}, "
                    f"amount_rub={amount_rub}, stars={stars_amount}"
                )
                
                return web.json_response({
                    "success": True,
                    "payment_id": payment_id,
                    "invoice_payload": invoice_payload,
                    "message_id": sent_message.message_id,
                    "stars_amount": stars_amount,
                    "amount_rub": amount_rub,
                    "stars_rate_rub": stars_rate_rub,
                    "stars_markup": stars_markup,
                    "stars_test_mode": stars_test_mode,
                    "is_admin_test_purchase": stars_test_mode and is_admin_user,
                })
            except Exception as e:
                logger.error(f"Ошибка отправки инвойса Stars пользователю {user_id}: {e}", exc_info=True)
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "message": f"Ошибка отправки инвойса: {str(e)}"
                }, status=500)
            finally:
                await bot.session.close()
        except Exception as e:
            logger.error(
                "Ошибка создания инвойса Stars для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )
    
    app.router.add_post("/api/payments/create", create_payment_handler)
    app.router.add_get("/api/payments/status", get_payment_status_handler)
    app.router.add_get("/api/payments/config", payment_config_handler)
    app.router.add_post("/api/payments/webhook", yookassa_webhook_handler)
    app.router.add_post("/api/payments/webhook/", yookassa_webhook_handler)  # С поддержкой слэша в конце
    app.router.add_post("/api/payments/stars/create", create_stars_invoice_handler)
    app.router.add_get("/api/mail", user_mail_handler)
    app.router.add_post("/api/mail/read", mark_mail_read_handler)
    app.router.add_get("/api/mail/unread-count", get_unread_mail_count_handler)
    
    async def shop_buy_handler(request: web.Request) -> web.Response:
        """Обработчик покупки товара за гемы."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            user_id_param = request.rel_url.query.get("user_id")
            if user_id_param:
                try:
                    user_id = int(user_id_param)
                except ValueError:
                    return web.json_response(
                        {"error": "user_id must be integer"}, status=400
                    )

        if not user_id:
            return web.json_response(
                {"error": "authentication required"}, status=401
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            item_type = data.get("item_type")
            gems_amount = int(data.get("gems_amount", 0))
            item_name = data.get("item_name", "Товар")

            if not item_type:
                return web.json_response({"error": "item_type_required"}, status=400)

            is_admin = user_id == 6803854304
            admin_case_tier: int | None = None
            if item_type.startswith("admin_case_tier_"):
                try:
                    admin_case_tier = int(item_type.split("_")[-1])
                except (ValueError, AttributeError):
                    return web.json_response({"error": "invalid_admin_case_tier"}, status=400)

            # Получаем профиль пользователя
            user_profile = await db.get_user_profile(user_id)
            if not user_profile:
                return web.json_response({"error": "user_not_found"}, status=404)

            current_gems = user_profile["gems"] if user_profile.get("gems") is not None else 0

            if admin_case_tier is None:
                if gems_amount <= 0:
                    return web.json_response({"error": "invalid_gems_amount"}, status=400)
                if current_gems < gems_amount:
                    return web.json_response({
                        "success": False,
                        "error": "insufficient_gems",
                        "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                    }, status=400)

                # Списываем гемы
                await db.execute(
                    "UPDATE users SET gems = gems - $1 WHERE user_id = $2",
                    gems_amount, user_id
                )
            else:
                if not is_admin:
                    return web.json_response({"error": "admin_only"}, status=403)
                if not (1 <= admin_case_tier <= 5):
                    return web.json_response({"error": "invalid_admin_case_tier"}, status=400)

                # Для админов покупки кейсов стоят 1 гем для тестов
                gems_amount = 1
                if current_gems < gems_amount:
                    return web.json_response({
                        "success": False,
                        "error": "insufficient_gems",
                        "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                    }, status=400)
                
                # Списываем 1 гем у админа
                await db.execute(
                    "UPDATE users SET gems = gems - $1 WHERE user_id = $2",
                    gems_amount, user_id
                )

            # Выдаем товар в зависимости от типа
            if admin_case_tier is not None:
                if admin_case_tier == 1:
                    await db.increment_user_keys(user_id, 1)
                    await db.sync_user_key_cases(user_id)
                    updated_keys = await db.fetchval(
                        "SELECT COALESCE(keys, 0) FROM users WHERE user_id = $1",
                        user_id
                    )
                    return web.json_response({
                        "success": True,
                        "message": f"Добавлен тестовый кейс T1",
                        "remaining_gems": current_gems,
                        "updated_keys": updated_keys or 0,
                        "granted_case_tier": 1,
                    })
                else:
                    case_id = await db.get_admin_case_id(admin_case_tier)
                    result = await db.add_user_case(user_id, case_id, admin_case_tier)
                    if not result.get("success"):
                        return web.json_response(
                            {"error": "case_creation_failed", "details": result.get("error")},
                            status=500,
                        )
                    return web.json_response({
                        "success": True,
                        "message": f"Добавлен тестовый кейс T{admin_case_tier}",
                        "remaining_gems": current_gems,
                        "granted_case_tier": admin_case_tier,
                        "user_case_id": result.get("user_case_id"),
                    })

            if item_type == "case":
                await db.increment_user_keys(user_id, 1)
                await db.sync_user_key_cases(user_id)
                import logging
                logging.getLogger(__name__).info(
                    f"Пользователь {user_id} купил кейс за {gems_amount} гемов. Keys увеличены на 1."
                )
                updated_keys = await db.fetchval(
                    "SELECT COALESCE(keys, 0) FROM users WHERE user_id = $1",
                    user_id
                )

                return web.json_response({
                    "success": True,
                    "message": f"Успешно куплено: {item_name}",
                    "remaining_gems": current_gems - gems_amount,
                    "updated_keys": updated_keys or 0
                })

            elif item_type and item_type.startswith("case_tier_"):
                # Покупка кейса определенного тира за гемы (case_tier_1, case_tier_2, и т.д.)
                try:
                    case_tier = int(item_type.split("_")[-1])
                    if not (1 <= case_tier <= 5):
                        return web.json_response({
                            "success": False,
                            "error": "invalid_case_tier",
                            "message": f"Неверный тир кейса: {case_tier}"
                        }, status=400)
                    
                    case_id = await db.get_admin_case_id(case_tier)
                    if not case_id:
                        return web.json_response({
                            "success": False,
                            "error": "case_not_found",
                            "message": f"Кейс тира {case_tier} не найден"
                        }, status=404)
                    
                    result = await db.add_user_case(user_id, case_id, case_tier)
                    if not result.get("success"):
                        return web.json_response({
                            "success": False,
                            "error": "case_creation_failed",
                            "message": result.get("error", "Ошибка создания кейса")
                        }, status=500)
                    
                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил кейс T{case_tier} за {gems_amount} гемов"
                    )
                    
                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": current_gems - gems_amount,
                        "granted_case_tier": case_tier,
                        "user_case_id": result.get("user_case_id")
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга тира кейса из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_case_tier",
                        "message": f"Ошибка обработки покупки кейса: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("coins_"):
                # Покупка монет за гемы (coins_300, coins_1400, coins_5000, coins_20000)
                try:
                    coins_amount = int(item_type.split("_")[1])
                    await db.execute(
                        "UPDATE users SET coins = coins + $1 WHERE user_id = $2",
                        coins_amount, user_id
                    )
                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил {coins_amount} монет за {gems_amount} гемов"
                    )
                    updated_coins = await db.fetchval(
                        "SELECT COALESCE(coins, 0) FROM users WHERE user_id = $1",
                        user_id
                    )
                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": current_gems - gems_amount,
                        "updated_coins": updated_coins or 0
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга количества монет из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_coins_amount",
                        "message": f"Ошибка обработки покупки монет: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("keys_"):
                # Покупка кейсов за гемы (keys_1, keys_3, keys_10, keys_25, keys_50, keys_100)
                try:
                    keys_amount = int(item_type.split("_")[1])
                    if keys_amount <= 0:
                        return web.json_response({
                            "success": False,
                            "error": "invalid_keys_amount",
                            "message": "Количество кейсов должно быть больше 0"
                        }, status=400)
                    
                    # Увеличиваем количество кейсов у пользователя (только поле keys, без синхронизации user_cases)
                    await db.increment_user_keys(user_id, keys_amount)
                    
                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил {keys_amount} кейсов за {gems_amount} гемов"
                    )
                    
                    updated_keys = await db.fetchval(
                        "SELECT COALESCE(keys, 0) FROM users WHERE user_id = $1",
                        user_id
                    )
                    
                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": current_gems - gems_amount,
                        "updated_keys": updated_keys or 0
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга количества кейсов из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_keys_amount",
                        "message": f"Ошибка обработки покупки кейсов: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("shop_set_"):
                # Покупка набора из БД
                try:
                    set_id = int(item_type.split("_")[-1])
                    set_data = await db.get_shop_set(set_id)
                    if not set_data:
                        return web.json_response({
                            "success": False,
                            "error": "set_not_found",
                            "message": "Набор не найден"
                        }, status=404)
                    
                    if not set_data.get("is_active"):
                        return web.json_response({
                            "success": False,
                            "error": "set_inactive",
                            "message": "Набор недоступен для покупки"
                        }, status=400)
                    
                    # Проверяем валюту и списываем средства
                    set_currency = set_data.get("currency", "rubles")
                    set_price = float(set_data.get("price", 0))
                    
                    if set_currency == "gems":
                        if current_gems < set_price:
                            return web.json_response({
                                "success": False,
                                "error": "insufficient_gems",
                                "message": f"Недостаточно гемов! Нужно {set_price} 💎, у вас {current_gems} 💎"
                            }, status=400)
                        await db.execute(
                            "UPDATE users SET gems = gems - $1 WHERE user_id = $2",
                            set_price, user_id
                        )
                    elif set_currency == "coins":
                        current_coins = user_profile.get("coins", 0)
                        if current_coins < set_price:
                            return web.json_response({
                                "success": False,
                                "error": "insufficient_coins",
                                "message": f"Недостаточно монет! Нужно {set_price} 💰, у вас {current_coins} 💰"
                            }, status=400)
                        await db.execute(
                            "UPDATE users SET coins = coins - $1 WHERE user_id = $2",
                            set_price, user_id
                        )
                    else:
                        return web.json_response({
                            "success": False,
                            "error": "invalid_currency",
                            "message": "Набор можно купить только за рубли через платеж"
                        }, status=400)
                    
                    # Выдаем награды
                    result = await db.grant_shop_set_rewards(user_id, set_id)
                    if not result.get("success"):
                        return web.json_response({
                            "success": False,
                            "error": "rewards_failed",
                            "message": result.get("error", "Ошибка выдачи наград")
                        }, status=500)
                    
                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил набор {set_id} за {set_price} {set_currency}"
                    )
                    
                    updated_gems = await db.fetchval(
                        "SELECT COALESCE(gems, 0) FROM users WHERE user_id = $1",
                        user_id
                    ) if set_currency == "gems" else current_gems
                    
                    updated_coins = await db.fetchval(
                        "SELECT COALESCE(coins, 0) FROM users WHERE user_id = $1",
                        user_id
                    ) if set_currency == "coins" else user_profile.get("coins", 0)
                    
                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": updated_gems or 0,
                        "updated_coins": updated_coins or 0,
                        "granted_rewards": result.get("granted", [])
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга ID набора из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_set_id",
                        "message": f"Ошибка обработки покупки набора: {str(e)}"
                    }, status=400)

            return web.json_response({
                "success": True,
                "message": f"Успешно куплено: {item_name}",
                "remaining_gems": current_gems - gems_amount
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка покупки товара за гемы для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_post("/api/shop/buy", shop_buy_handler)
    
    async def card_image_handler(request: web.Request) -> web.Response:
        """
        Возвращает изображение карты из локальной файловой системы.
        Изображение берется из DesignAssets/Cards/<card_id>.png
        """
        card_id = request.rel_url.query.get("card_id")
        file_id = request.rel_url.query.get("file_id")  # Для обратной совместимости
        
        # Если передан file_id, пытаемся получить card_id из БД (для обратной совместимости)
        if file_id and not card_id:
            try:
                card_record = await db.fetchrow(
                    "SELECT id FROM cards WHERE image_file_id = $1 LIMIT 1",
                    file_id
                )
                if card_record:
                    card_id = str(card_record["id"])
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Не удалось найти card_id по file_id: {e}")
        
        if not card_id:
            return web.json_response({"error": "card_id required"}, status=400)
        
        try:
            card_id_int = int(card_id)
        except ValueError:
            return web.json_response({"error": "invalid_card_id"}, status=400)
        
        # Формируем путь к файлу изображения карты
        card_image_path = DESIGN_ASSETS_DIR / "Cards" / f"{card_id_int}.png"
        
        # Проверяем существование файла
        if not card_image_path.exists() or not card_image_path.is_file():
            import logging
            logging.getLogger(__name__).warning(f"Изображение карты не найдено: {card_image_path}")
            return web.json_response({"error": "card_image_not_found"}, status=404)
        
        try:
            # Читаем файл изображения
            with open(card_image_path, "rb") as f:
                image_data = f.read()
            
            # Определяем content-type по расширению файла
            content_type = "image/png"  # Все карты в формате PNG
            
            # Возвращаем изображение с правильными заголовками
            response = web.Response(
                body=image_data,
                content_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  # Кэшируем на 24 часа
                    "Access-Control-Allow-Origin": "*"  # Разрешаем CORS
                }
            )
            return response
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка чтения изображения карты: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error"}, status=500)
    
    app.router.add_get("/api/cards/image", card_image_handler)
    
    # API для управления наборами магазина (только для админа)
    async def shop_sets_list_handler(request: web.Request) -> web.Response:
        """Получить список всех наборов."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        is_admin = user_id in app.get("admin_ids", set())
        if not is_admin:
            return web.json_response({"error": "admin_only"}, status=403)

        try:
            active_only = request.rel_url.query.get("active_only", "false").lower() == "true"
            sets = await db.get_shop_sets(active_only=active_only)
            return web.json_response({"sets": sets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения наборов: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_detail_handler(request: web.Request) -> web.Response:
        """Получить детали набора по ID."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        is_admin = user_id in app.get("admin_ids", set())
        if not is_admin:
            return web.json_response({"error": "admin_only"}, status=403)

        try:
            set_id = int(request.match_info.get("set_id", 0))
            set_data = await db.get_shop_set(set_id)
            if not set_data:
                return web.json_response({"error": "set_not_found"}, status=404)
            return web.json_response({"set": set_data})
        except (ValueError, Exception) as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_create_handler(request: web.Request) -> web.Response:
        """Создать новый набор."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        is_admin = user_id in app.get("admin_ids", set())
        if not is_admin:
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "").strip() or None
            image_file_id = data.get("image_file_id", "").strip() or None
            price = float(data.get("price", 0))
            currency = data.get("currency", "rubles")
            rewards = data.get("rewards", [])

            if not name:
                return web.json_response({"error": "name_required"}, status=400)
            if price < 0:
                return web.json_response({"error": "invalid_price"}, status=400)
            if currency not in ("rubles", "gems", "coins"):
                return web.json_response({"error": "invalid_currency"}, status=400)

            result = await db.create_shop_set(
                name=name,
                description=description,
                image_file_id=image_file_id,
                price=price,
                currency=currency,
                created_by=user_id,
                rewards=rewards
            )

            if result.get("success"):
                return web.json_response({"success": True, "set_id": result.get("set_id")})
            else:
                return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка создания набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_update_handler(request: web.Request) -> web.Response:
        """Обновить набор."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        is_admin = user_id in app.get("admin_ids", set())
        if not is_admin:
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            set_id = int(data.get("set_id", 0))
            name = data.get("name")
            description = data.get("description")
            image_file_id = data.get("image_file_id")
            price = data.get("price")
            currency = data.get("currency")
            is_active = data.get("is_active")
            rewards = data.get("rewards")

            if name is not None:
                name = name.strip() or None
            if description is not None:
                description = description.strip() or None
            if image_file_id is not None:
                image_file_id = image_file_id.strip() or None

            if price is not None and price < 0:
                return web.json_response({"error": "invalid_price"}, status=400)
            if currency is not None and currency not in ("rubles", "gems", "coins"):
                return web.json_response({"error": "invalid_currency"}, status=400)

            result = await db.update_shop_set(
                set_id=set_id,
                name=name,
                description=description,
                image_file_id=image_file_id,
                price=price,
                currency=currency,
                is_active=is_active,
                rewards=rewards
            )

            if result.get("success"):
                return web.json_response({"success": True})
            else:
                return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except (ValueError, Exception) as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка обновления набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_delete_handler(request: web.Request) -> web.Response:
        """Удалить набор."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        is_admin = user_id in app.get("admin_ids", set())
        if not is_admin:
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            set_id = int(data.get("set_id", 0))

            result = await db.delete_shop_set(set_id)
            if result.get("success"):
                return web.json_response({"success": True})
            else:
                return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except (ValueError, Exception) as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка удаления набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_sets_public_handler(request: web.Request) -> web.Response:
        """Получить список активных наборов для магазина (публичный endpoint)."""
        try:
            sets = await db.get_shop_sets(active_only=True)
            return web.json_response({"sets": sets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения наборов: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    app.router.add_get("/api/shop/sets", shop_sets_public_handler)
    app.router.add_get("/api/admin/shop/sets", shop_sets_list_handler)
    app.router.add_get("/api/admin/shop/sets/{set_id}", shop_set_detail_handler)
    app.router.add_post("/api/admin/shop/sets/create", shop_set_create_handler)
    app.router.add_post("/api/admin/shop/sets/update", shop_set_update_handler)
    app.router.add_post("/api/admin/shop/sets/delete", shop_set_delete_handler)
    
    async def dice_status_handler(request: web.Request) -> web.Response:
        """Получить статус кубика (можно ли бросать, когда был последний бросок)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            status = await db.get_dice_status(user_id)
            return web.json_response(_serialize_datetime(status))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения статуса кубика: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_roll_handler(request: web.Request) -> web.Response:
        """Бросить кубик."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            result = await db.roll_dice(user_id)
            return web.json_response(_serialize_datetime(result))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка броска кубика: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_notification_prompt_status_handler(request: web.Request) -> web.Response:
        """Проверить, показывалось ли уже предложение включить уведомления."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            prompt_shown = await db.get_dice_notification_prompt_status(user_id)
            return web.json_response({"prompt_shown": prompt_shown})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка проверки статуса предложения: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_notification_prompt_mark_handler(request: web.Request) -> web.Response:
        """Отметить, что предложение включить уведомления было показано."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            await db.mark_dice_notification_prompt_shown(user_id)
            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка отметки предложения: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def welcome_status_handler(request: web.Request) -> web.Response:
        """Получить статус приветствия и данные о стартовой карте (работает даже если пользователя еще нет)."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            # Проверяем, существует ли пользователь
            user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
            
            if user_exists:
                welcome_status = await db.get_welcome_status(user_id)
                should_show = welcome_status["should_show"]
            else:
                # Пользователя нет - значит нужно показать приветствие
                should_show = True
            
            # Получаем данные о стартовой карте (ID 9)
            start_card = await db.fetchrow(
                "SELECT id, name, description, rarity, power, image_file_id FROM cards WHERE id = 9"
            )
            
            result = {
                "should_show": should_show,
                "welcome_shown": not should_show,
                "start_card": dict(start_card) if start_card else None
            }
            return web.json_response(_serialize_datetime(result))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения статуса приветствия: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def welcome_mark_shown_handler(request: web.Request) -> web.Response:
        """Отметить, что приветствие было показано."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            await db.mark_welcome_shown(user_id)
            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка отметки приветствия: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/dice/status", dice_status_handler)
    app.router.add_post("/api/dice/roll", dice_roll_handler)
    app.router.add_get("/api/dice/notification-prompt-status", dice_notification_prompt_status_handler)
    app.router.add_post("/api/dice/notification-prompt-mark", dice_notification_prompt_mark_handler)
    async def welcome_create_user_handler(request: web.Request) -> web.Response:
        """Создать пользователя после завершения приветствия."""
        init_data = request.rel_url.query.get("_auth")
        user_id = None
        username = None
        first_name_from_data = None
        last_name = None

        if init_data and init_data.isdigit():
            try:
                user_id = int(init_data)
            except ValueError:
                pass

        if not user_id and init_data:
            verified_data = _verify_init_data(init_data, bot_token)
            if verified_data:
                user_id = _extract_user_id_from_init_data(verified_data)
                user_str = verified_data.get("user", "")
                if user_str:
                    import json
                    try:
                        user_data = json.loads(user_str)
                        username = user_data.get("username")
                        first_name_from_data = user_data.get("first_name")
                        last_name = user_data.get("last_name")
                    except Exception:
                        pass

        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        # Если данных пользователя нет в initData, пытаемся получить через Bot API
        if not first_name_from_data:
            try:
                async with _create_ssl_disabled_session() as session:
                    url = f"https://api.telegram.org/bot{bot_token}/getChat"
                    async with session.get(url, params={"chat_id": user_id}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ok"):
                                user_info = data.get("result", {})
                                first_name_from_data = user_info.get("first_name")
                                username = username or user_info.get("username")
                                last_name = user_info.get("last_name")
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Не удалось получить данные пользователя через Bot API: {e}")

        # Создаем пользователя (ensure_user автоматически выдаст карту и подарок для нового пользователя)
        await db.ensure_user(
            user_id=user_id,
            username=username,
            first_name=first_name_from_data,
            last_name=last_name,
        )
        
        # Отмечаем, что приветствие было показано
        await db.mark_welcome_shown(user_id)
        
        # Загружаем профиль для ответа
        record = await db.get_user_profile(user_id)
        if not record:
            return web.json_response({"error": "failed_to_create_user"}, status=500)

        settings_record = await db.get_user_settings(user_id)
        settings_data = {}
        if settings_record:
            settings_data = {
                "notif_cases": settings_record["notif_cases"],
                "notif_daily_rewards": settings_record["notif_daily_rewards"],
                "notif_game_invites": settings_record["notif_game_invites"],
                "notif_friend_requests": settings_record["notif_friend_requests"],
                "notif_events": settings_record["notif_events"],
                "notif_news": settings_record["notif_news"],
                "ads_enabled": settings_record["ads_enabled"],
                "sound_music": settings_record["sound_music"],
                "sound_sfx": settings_record["sound_sfx"],
                "social_block_friend_requests": settings_record["social_block_friend_requests"],
            }

        title = record.get("title") or "Игрок"
        
        payload: dict[str, Any] = {
            "user_id": record["user_id"],
            "username": record.get("username"),
            "first_name": first_name_from_data or record.get("first_name"),
            "extra_pass": record.get("extra_pass", "inactive"),
            "trophies": record.get("trophies", 0),
            "max_trophies": record.get("max_trophies", 0),
            "keys": record.get("keys", 0),
            "gems": record.get("gems", 0),
            "coins": record.get("coins", 0),
            "squad_id": record.get("squad_id"),
            "status": record.get("status", "active"),
            "reg_date": record["reg_date"].isoformat() if record.get("reg_date") else None,
            "stars": record.get("stars", 0),
            "energy": record.get("energy", 5),
            "energy_cd": record.get("energy_cd").isoformat() if record.get("energy_cd") else None,
            "season": record.get("season", 0),
            "title": title,
            "img": record.get("img", ""),
            "custom_nickname": record.get("custom_nickname"),
            "nickname_changed": record.get("nickname_changed", False),
            "settings": settings_data,
            "should_show_welcome": False,
        }

        return web.json_response(payload)

    app.router.add_get("/api/welcome/status", welcome_status_handler)
    app.router.add_post("/api/welcome/mark-shown", welcome_mark_shown_handler)
    app.router.add_post("/api/welcome/create-user", welcome_create_user_handler)
    app.router.add_get("/{path:.*}", static_handler)

    # ============================================
    # ФОНОВАЯ ЗАДАЧА: АВТО-ЗАВЕРШЕНИЕ ХОДА
    # ============================================
    
    async def match_timer_checker(app: web.Application) -> None:
        """
        Фоновая задача для автоматического завершения хода по истечении времени.
        Запускается раз в секунду и проверяет все активные матчи.
        """
        logger = logging.getLogger(__name__)
        logger.info("🕒 Match timer checker started")
        
        try:
            while True:
                await asyncio.sleep(1)  # Проверка каждую секунду
                
                # Получаем копию активных матчей для безопасной итерации
                matches_to_check = list(ACTIVE_MATCHES.items())
                
                for match_id, engine in matches_to_check:
                    try:
                        # Пропускаем завершённые матчи
                        if hasattr(engine, 'is_ended') and engine.is_ended:
                            continue
                        
                        # Получаем текущего игрока для проверки таймера
                        current_player_id = engine.get_current_player_id() if hasattr(engine, 'get_current_player_id') else None
                        
                        # Получаем состояние матча для текущего игрока
                        state = engine.get_full_state(viewer_id=current_player_id) if hasattr(engine, 'get_full_state') else {}
                        
                        # Проверяем, не истекло ли время
                        time_remaining = state.get('turn_time_remaining', 99)
                        
                        if time_remaining <= 0:
                            # Время истекло - автоматически завершаем ход
                            if current_player_id is not None:
                                logger.warning(
                                    "⏰ Auto-ending turn for match %s (player %s) - time expired",
                                    match_id, current_player_id
                                )
                                
                                try:
                                    # Вызываем завершение хода
                                    result = engine.end_turn(current_player_id)
                                    
                                    # Отправляем событие через Socket.IO
                                    sio = app.get("socketio")
                                    if sio:
                                        # Получаем обновлённое состояние для нового текущего игрока
                                        new_current_player = engine.get_current_player_id() if hasattr(engine, 'get_current_player_id') else None
                                        new_state = engine.get_full_state(viewer_id=new_current_player) if hasattr(engine, 'get_full_state') else {}
                                        
                                        await sio.emit(
                                            'turn_end',
                                            {
                                                'match_id': match_id,
                                                'state': new_state,
                                                'auto_ended': True,  # Флаг автоматического завершения
                                                'reason': 'time_expired'
                                            },
                                            room=match_id
                                        )
                                        
                                        logger.info("✅ Auto-ended turn for match %s, emitted turn_end event", match_id)
                                        
                                        # КРИТИЧНО: Даем серверу прогрузить состояние перед ходом бота
                                        await asyncio.sleep(1)
                                        
                                        # КРИТИЧНО: Запускаем бота, если следующий игрок - бот
                                        await check_and_run_bot(match_id, ACTIVE_MATCHES)
                                    
                                except Exception as exc:
                                    logger.error(
                                        "❌ Failed to auto-end turn for match %s: %s",
                                        match_id, exc, exc_info=True
                                    )
                    
                    except Exception as exc:
                        logger.error(
                            "❌ Error checking timer for match %s: %s",
                            match_id, exc
                        )
        
        except asyncio.CancelledError:
            logger.info("🛑 Match timer checker stopped")
        except Exception as exc:
            logger.error("❌ Match timer checker fatal error: %s", exc, exc_info=True)
    
    async def start_background_tasks(app: web.Application) -> None:
        """Запуск фоновых задач при старте сервера."""
        app['match_timer_task'] = asyncio.create_task(match_timer_checker(app))
    
    async def cleanup_background_tasks(app: web.Application) -> None:
        """Остановка фоновых задач при остановке сервера."""
        if 'match_timer_task' in app:
            app['match_timer_task'].cancel()
            await app['match_timer_task']
    
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app
