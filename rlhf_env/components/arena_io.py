"""ArenaTransport — реплика сервервер-контракта игры для 1:1 arena.js.

Поднимает python-socketio (AsyncServer, async_mode='aiohttp') поверх aiohttp-app
и регистрирует HTTP-роуты боевых действий. Цель: верbatim-скопированный
arena.js работает БЕЗ правок — те же события, те же payload'ы, та же форма стейта
(см. RlhfBattleEngine.get_full_state, верbatim-копия BattleEngine.get_full_state).

Контракт (валидировано по web/server.py + webapp/arena.js):
  - join_match {match_id, _auth}  → joined_match {match_id, user_id}
  - client_ready {match_id}        → client_ready_ack + match_ready (state)
  - surrender {match_id, _auth, client_action_id} → surrender_ack + game_over
  - state_changed (бот-ходы), game_over, error
  - HTTP /api/battle/state → state-object напрямую
  - HTTP /api/battle/{play-card,attack,end-turn} → {result, state, sound_events} | {result, state, error}
  - HTTP /api/matches/{id}/surrender
  - HTTP /api/match/find → {status:'found', match_id, is_bot, redirect_url, player_ids, bot_info}
  - client_action_id обязателен + idempotency-кэш (только status<400)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import socketio
from aiohttp import web

from rlhf_env.components.match_runner import MatchRunner
from rlhf_env.components.policy_factory import BOT_MAX_DIFFICULTY

logger = logging.getLogger(__name__)

HUMAN_USER_ID = 1000
ACTION_CACHE_TTL_SECONDS = 300.0


# ---------------------------------------------------------------------------
# helpers (копия web/server.py:3173-3220 + _action_failure_status)
# ---------------------------------------------------------------------------

def _audio_query(spec: Optional[Dict[str, Any]]) -> str:
    """Параметры звука для redirect_url арены: &music=0/&sfx=0.

    spec.audio = {"music": bool, "sfx": bool} — выбор человека в главном меню
    среды. Арена (arena.js initArenaMusic) уже умеет читать ?music=0/?sfx=0 и
    глушить фоновую музыку / SFX соответственно. Подставляем в КАЖДЫЙ бой серии
    (первый + /api/match/find для 2..N), чтобы настройка действовала на всю серию
    без правок боевого arena.js. Всегда отдаём оба параметра явными — тогда
    arena.js applyArenaSoundSettingsFromUserSettings не перебьёт SFX серверными
    user-settings (он отступает, если в URL есть ?sfx=).
    """
    audio = (spec or {}).get("audio") if isinstance(spec, dict) else None
    if not isinstance(audio, dict):
        return ""
    parts = []
    if "music" in audio:
        parts.append(f"music={'1' if audio.get('music') else '0'}")
    if "sfx" in audio:
        parts.append(f"sfx={'1' if audio.get('sfx') else '0'}")
    return ("&" + "&".join(parts)) if parts else ""


def _client_action_id(payload: Dict[str, Any]) -> Optional[str]:
    raw = None
    for key in ("client_action_id", "nonce", "action_id"):
        if key in payload and payload.get(key) is not None:
            raw = payload.get(key)
            break
    value = str(raw).strip() if raw is not None else ""
    return value[:128] if value else None


def _action_failure_status(result: Dict[str, Any]) -> int:
    error = str((result or {}).get("error") or "")
    if error in {"not_your_turn", "turn_expired", "game_already_ended", "match_not_ready"}:
        return 409
    if error in {"not_participant", "unauthorized"}:
        return 403
    return 400


class ArenaTransport:
    """Socket.IO + HTTP транспорт арены."""

    def __init__(
        self,
        manager: Any,
        *,
        auth_token: str,
        human_user_id: int = HUMAN_USER_ID,
    ) -> None:
        self.manager = manager
        self.auth_token = auth_token
        self.human_user_id = human_user_id

        self.sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            ping_interval=25,
            ping_timeout=20,
        )
        # sid -> {"match_id": str, "user_id": int}
        self._sid_session: Dict[str, Dict[str, Any]] = {}
        # match_id -> MatchRunner
        self._runners: Dict[str, MatchRunner] = {}
        # (match_id, user_id, client_action_id) -> {"payload", "status", "created_at"}
        self._action_cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        # match_id -> asyncio.Lock для инициализации runner'а
        self._runner_locks: Dict[str, asyncio.Lock] = {}

        self._register_socket_handlers()

    # ------------------------------------------------------------------
    # attach + routes
    # ------------------------------------------------------------------
    def attach(self, app: web.Application) -> None:
        self.sio.attach(app)
        app["arena_transport"] = self
        self._register_http_routes(app)

    def _register_http_routes(self, app: web.Application) -> None:
        r = app.router
        r.add_get("/api/battle/state", self.http_battle_state)
        r.add_post("/api/battle/play-card", self.http_play_card)
        r.add_post("/api/battle/attack", self.http_attack)
        r.add_post("/api/battle/end-turn", self.http_end_turn)
        r.add_post("/api/battle/preview", self.http_preview)
        r.add_post("/api/matches/{match_id}/surrender", self.http_surrender)
        r.add_post("/api/match/find", self.http_match_find)

    # ------------------------------------------------------------------
    # auth + runner
    # ------------------------------------------------------------------
    def _check_auth(self, token: Optional[str]) -> bool:
        if not token or not self.auth_token:
            return False
        return hmac.compare_digest(str(token), str(self.auth_token))

    def _runner_lock(self, match_id: str) -> asyncio.Lock:
        lock = self._runner_locks.get(match_id)
        if lock is None:
            lock = asyncio.Lock()
            self._runner_locks[match_id] = lock
        return lock

    @staticmethod
    def _series_fields(match) -> Dict[str, Any]:
        """Поля серии боёв для клиента: group_id + battle_index/battles_planned +
        has_next_battle (есть ли следующий бой серии). Нужны пост-боевому
        модальному окну (кнопка «Следующий бой» активна только если есть бой)."""
        has_next = int(match.battle_index) + 1 < int(match.battles_planned)
        return {
            "group_id": match.group_id,
            "battle_index": int(match.battle_index),
            "battles_planned": int(match.battles_planned),
            "has_next_battle": has_next,
        }

    async def get_runner(self, match_id: str) -> Optional[MatchRunner]:
        """Лениво создаёт MatchRunner для матча и вешает broadcaster."""
        runner = self._runners.get(match_id)
        if runner is not None:
            return runner
        async with self._runner_lock(match_id):
            runner = self._runners.get(match_id)
            if runner is not None:
                return runner
            match = self.manager.get_match(match_id)
            if match is None:
                return None
            runner = MatchRunner(match)
            runner.broadcaster = self._make_broadcaster(match_id)
            self._runners[match_id] = runner
            return runner

    # ------------------------------------------------------------------
    # broadcast (вызывается из match_runner; sync — планирует async-таск)
    # ------------------------------------------------------------------
    def _make_broadcaster(self, match_id: str) -> Callable[[str, str, Dict[str, Any]], None]:
        def _bcast(_mid: str, event_type: str, data: Dict[str, Any]) -> None:
            asyncio.create_task(self._broadcast(match_id, event_type, data))
        return _bcast

    async def _broadcast(self, match_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Персонализованный state каждому sid в матче (копия _emit_personalized_match_state)."""
        match = self.manager.get_match(match_id)
        if not match:
            return
        async with match.lock:
            for sid, sess in list(self._sid_session.items()):
                if sess.get("match_id") != match_id:
                    continue
                user_id = sess.get("user_id")
                if user_id is None:
                    continue
                try:
                    state = match.engine.get_full_state(viewer_id=int(user_id))
                except Exception as exc:  # noqa: BLE001
                    logger.error("broadcast get_full_state failed: %s", exc, exc_info=True)
                    continue
                state.update(self._series_fields(match))
                payload: Dict[str, Any] = {
                    "match_id": match_id,
                    "state": state,
                    "data": (data or {}).get("data", {}),
                }
                se = (data or {}).get("sound_events")
                if se is not None:
                    payload["sound_events"] = se
                try:
                    await self.sio.emit(event_type, payload, to=sid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("broadcast emit failed sid=%s: %s", sid, exc)

    async def _broadcast_game_over(self, match_id: str, winner_id: Optional[int], reason: str) -> None:
        match = self.manager.get_match(match_id)
        if not match:
            return
        engine = match.engine
        payload = {
            "game_over": True,
            "winner_id": winner_id,
            "p1_hp": engine._arena.state.p1.hero.hp,
            "p2_hp": engine._arena.state.p2.hero.hp,
            "reason": reason,
            "players": {},
            "match_id": match_id,
        }
        # Поля серии боёв — чтобы пост-боевое модальное окно знало, есть ли
        # следующий бой, даже если последний state_changed не дошёл.
        payload.update(self._series_fields(match))
        for sid, sess in list(self._sid_session.items()):
            if sess.get("match_id") != match_id:
                continue
            try:
                await self.sio.emit("game_over", payload, to=sid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("game_over emit failed sid=%s: %s", sid, exc)

    # ------------------------------------------------------------------
    # action cache (idempotency)
    # ------------------------------------------------------------------
    def _cache_prune(self) -> None:
        now = time.time()
        for key, cached in list(self._action_cache.items()):
            if now - float(cached.get("created_at", 0) or 0) > ACTION_CACHE_TTL_SECONDS:
                self._action_cache.pop(key, None)

    def _cache_get(self, match_id: str, user_id: int, client_action_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not client_action_id:
            return None
        self._cache_prune()
        return self._action_cache.get((str(match_id), int(user_id), str(client_action_id)))

    def _cache_set(
        self, match_id: str, user_id: int, client_action_id: Optional[str],
        payload: Dict[str, Any], *, status: int = 200,
    ) -> None:
        if not client_action_id:
            return
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            status_code = 200
        if status_code >= 400:
            return  # не кэшируем ошибочные (H4: failed action cache poisoning)
        self._action_cache[(str(match_id), int(user_id), str(client_action_id))] = {
            "payload": payload,
            "status": status_code,
            "created_at": time.time(),
        }

    # ------------------------------------------------------------------
    # Socket.IO handlers
    # ------------------------------------------------------------------
    def _register_socket_handlers(self) -> None:
        sio = self.sio

        @sio.event
        async def connect(sid, environ):  # noqa: ANN001
            logger.info("[SOCKET] connect sid=%s", sid)
            return True

        @sio.event
        async def disconnect(sid):  # noqa: ANN001
            sess = self._sid_session.pop(sid, None)
            if sess:
                logger.info("[SOCKET] disconnect sid=%s match=%s", sid, sess.get("match_id"))
            return True

        @sio.event
        async def join_match(sid, data):  # noqa: ANN001
            try:
                match_id = str((data or {}).get("match_id", ""))
                if not match_id:
                    await self.sio.emit("error", {"message": "match_id required"}, to=sid)
                    return
                token = (data or {}).get("_auth") or (data or {}).get("auth")
                if not self._check_auth(token):
                    await self.sio.emit("error", {"message": "invalid_auth"}, to=sid)
                    return
                match = self.manager.get_match(match_id)
                if match is None:
                    await self.sio.emit("error", {"message": "match_not_found"}, to=sid)
                    return
                await self.sio.enter_room(sid, match_id)
                self._sid_session[sid] = {"match_id": match_id, "user_id": self.human_user_id}
                logger.info("[SOCKET] join_match sid=%s match=%s user=%s", sid, match_id, self.human_user_id)
                await self.sio.emit(
                    "joined_match",
                    {"match_id": match_id, "user_id": str(self.human_user_id)},
                    to=sid,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("join_match error: %s", exc, exc_info=True)
                await self.sio.emit("error", {"message": "internal_error"}, to=sid)

        @sio.event
        async def leave_match(sid, data):  # noqa: ANN001
            match_id = str((data or {}).get("match_id", ""))
            if match_id:
                await self.sio.leave_room(sid, match_id)
            self._sid_session.pop(sid, None)
            return True

        @sio.event
        async def client_ready(sid, data):  # noqa: ANN001
            try:
                sess = self._sid_session.get(sid)
                if not sess:
                    await self.sio.emit("error", {"message": "not_authenticated"}, to=sid)
                    return
                match_id = str(sess.get("match_id", ""))
                user_id = int(sess.get("user_id", self.human_user_id))
                match = self.manager.get_match(match_id)
                if match is None:
                    await self.sio.emit("error", {"message": "Match not found"}, to=sid)
                    return

                ready_info = match.engine.mark_client_ready(user_id)
                logger.info("[SOCKET] client_ready sid=%s match=%s ready=%s", sid, match_id, ready_info)

                # bot-матч: после готовности человека сразу стартуем.
                if ready_info.get("all_ready"):
                    await self._broadcast(match_id, "match_ready", {"readiness": ready_info})
                    await self._maybe_trigger_bot(match_id)
                else:
                    await self._broadcast(match_id, "match_waiting", {"readiness": ready_info})

                await self.sio.emit(
                    "client_ready_ack", {"match_id": match_id, **ready_info}, to=sid,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("client_ready error: %s", exc, exc_info=True)
                await self.sio.emit("error", {"message": "internal_error"}, to=sid)

        @sio.event
        async def surrender(sid, data):  # noqa: ANN001
            try:
                sess = self._sid_session.get(sid)
                if not sess:
                    await self.sio.emit("error", {"message": "not_authenticated"}, to=sid)
                    return
                match_id = str(sess.get("match_id", ""))
                user_id = int(sess.get("user_id", self.human_user_id))
                match = self.manager.get_match(match_id)
                if match is None:
                    await self.sio.emit("surrender_ack", {
                        "match_id": match_id, "user_id": user_id,
                        "already_ended": True, "already_processed": True, "trophy_penalty": 0,
                    }, to=sid)
                    return
                if match.engine.is_ended:
                    await self.sio.emit("surrender_ack", {
                        "match_id": match_id, "user_id": user_id,
                        "already_ended": True, "already_processed": True, "trophy_penalty": 0,
                    }, to=sid)
                    return
                runner = await self.get_runner(match_id)
                if runner is None:
                    await self.sio.emit("error", {"message": "Match not found"}, to=sid)
                    return
                await runner.surrender()
                winner_id = match.engine.bot_user_id
                await self.sio.emit("surrender_ack", {
                    "match_id": match_id, "user_id": user_id,
                    "trophy_penalty": 0, "new_trophies": 0,
                    "already_processed": False, "game_over": True, "reason": "surrender",
                }, to=sid)
                await self._broadcast_game_over(match_id, winner_id, reason="surrender")
            except Exception as exc:  # noqa: BLE001
                logger.error("surrender error: %s", exc, exc_info=True)
                await self.sio.emit("error", {"message": "internal_error"}, to=sid)

        @sio.event
        async def battle_talkie_settings(sid, data):  # noqa: ANN001
            # cosmetic no-op для совместимости с arena.js
            await self.sio.emit("battle_talkie_settings_ack", {"success": True}, to=sid)

    async def _maybe_trigger_bot(self, match_id: str) -> None:
        """Если сейчас ход бота — запускаем бот-рутину (как check_and_run_bot)."""
        match = self.manager.get_match(match_id)
        if match is None:
            return
        engine = match.engine
        if engine.is_ended:
            return
        if not engine.is_current_player_bot():
            return
        runner = await self.get_runner(match_id)
        if runner is None:
            return
        asyncio.create_task(runner.run_bot_turn())

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------
    def _viewer_id(self, request: web.Request, payload: Optional[Dict[str, Any]] = None) -> int:
        # Один человек — всегда human_user_id. Auth-bridge шлёт Bearer,
        # но в RLHF это фиксированный user_id.
        return self.human_user_id

    async def http_battle_state(self, request: web.Request) -> web.Response:
        match_id = request.rel_url.query.get("match_id") or request.rel_url.query.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)
        match = self.manager.get_match(match_id)
        if match is None:
            return web.json_response({"error": "match_not_found"}, status=404)
        viewer_id = self._viewer_id(request)
        try:
            state = match.engine.get_full_state(viewer_id=viewer_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("battle_state failed: %s", exc, exc_info=True)
            return web.json_response({"error": "state_failed"}, status=500)
        state.update(self._series_fields(match))
        return web.json_response(state)

    async def _http_action(
        self,
        request: web.Request,
        *,
        kind: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        if not match_id:
            return web.json_response({"error": "invalid_parameters"}, status=400)
        user_id = self._viewer_id(request, payload)
        client_action_id = _client_action_id(payload)
        if not client_action_id:
            return web.json_response({"error": "client_action_id_required"}, status=400)

        cached = self._cache_get(match_id, user_id, client_action_id)
        if cached:
            return web.json_response(cached["payload"], status=cached["status"])

        match = self.manager.get_match(match_id)
        if match is None:
            return web.json_response({"error": "match_not_found"}, status=404)
        if match.engine.is_ended:
            state = match.engine.get_full_state(viewer_id=user_id)
            state.update(self._series_fields(match))
            payload_out = {
                "result": {"success": False, "error": "game_already_ended", "game_over": True},
                "state": state, "error": "game_already_ended",
            }
            return web.json_response(payload_out, status=409)

        runner = await self.get_runner(match_id)
        if runner is None:
            return web.json_response({"error": "match_not_found"}, status=404)

        action_payload = {"type": kind}
        if extra_fields:
            action_payload.update(extra_fields)
        # Прокидываем все поля из payload (card_id_from_hand, hand_index, target_id, ...)
        for k, v in payload.items():
            if k not in {"match_id", "id", "client_action_id", "nonce", "action_id"}:
                action_payload[k] = v
        action_payload["type"] = kind  # гарантия, что не перетёрли

        try:
            resp = await runner.execute_human_action(action_payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("http %s failed: %s", kind, exc, exc_info=True)
            return web.json_response({"error": f"{kind}_failed"}, status=400)

        result = resp.get("result", {})
        accepted = bool(result.get("success", False))
        if not accepted:
            status = _action_failure_status(result)
            out = {"result": result, "state": resp.get("state"), "error": result.get("error")}
            self._cache_set(match_id, user_id, client_action_id, out, status=status)
            return web.json_response(out, status=status)

        out = {"result": result, "state": resp.get("state"), "sound_events": result.get("sound_events", [])}
        self._cache_set(match_id, user_id, client_action_id, out, status=200)
        return web.json_response(out)

    async def http_play_card(self, request: web.Request) -> web.Response:
        return await self._http_action(request, kind="play_card")

    async def http_attack(self, request: web.Request) -> web.Response:
        return await self._http_action(request, kind="attack")

    async def http_end_turn(self, request: web.Request) -> web.Response:
        return await self._http_action(request, kind="end_turn")

    async def http_preview(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        match_id = payload.get("match_id") or payload.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)
        match = self.manager.get_match(match_id)
        if match is None:
            return web.json_response({"error": "match_not_found"}, status=404)
        viewer_id = self._viewer_id(request, payload)
        try:
            # get_preview_delta(action) — пробуем собрать action из payload
            delta = match.engine.get_preview_delta(viewer_id, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview failed: %s", exc, exc_info=True)
            return web.json_response({"error": "preview_failed"}, status=400)
        return web.json_response(delta if isinstance(delta, dict) else {"delta": delta})

    async def http_surrender(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        match_id = request.match_info.get("match_id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)
        user_id = self._viewer_id(request, payload)
        client_action_id = _client_action_id(payload)
        if not client_action_id:
            return web.json_response({"error": "client_action_id_required"}, status=400)
        cached = self._cache_get(match_id, user_id, client_action_id)
        if cached:
            return web.json_response(cached["payload"], status=cached["status"])
        match = self.manager.get_match(match_id)
        if match is None:
            return web.json_response({"error": "match_not_found"}, status=404)
        if match.engine.is_ended:
            state = match.engine.get_full_state(viewer_id=user_id)
            state.update(self._series_fields(match))
            out = {"result": {"success": False, "error": "game_already_ended", "game_over": True}, "state": state}
            return web.json_response(out)
        runner = await self.get_runner(match_id)
        if runner is None:
            return web.json_response({"error": "match_not_found"}, status=404)
        await runner.surrender()
        state = match.engine.get_full_state(viewer_id=user_id)
        state.update(self._series_fields(match))
        out = {
            "result": {"success": True, "game_over": True, "winner_id": match.engine.bot_user_id},
            "state": state,
        }
        self._cache_set(match_id, user_id, client_action_id, out, status=200)
        return web.json_response(out)

    async def http_match_find(self, request: web.Request) -> web.Response:
        """POST /api/match/find — старт серии или следующий бой.

        Body: spec (поля формы) ИЛИ {"group_id": "..."} — продолжить серию.
        Возвращает {status:'found', match_id, is_bot, redirect_url, player_ids, bot_info}.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "invalid_json"}, status=400)

        group_id = body.get("group_id")
        if group_id:
            # продолжить серию → следующий бой
            match = self.manager.next_match(group_id)
            if match is None:
                return web.json_response({"error": "series_complete", "group_id": group_id}, status=404)
        else:
            try:
                match = self.manager.create_series(body)
            except Exception as exc:  # noqa: BLE001
                logger.warning("match_find create_series failed: %s", exc, exc_info=True)
                return web.json_response({"error": str(exc)}, status=400)

        match_id = match.engine.match_id
        redirect_url = f"/arena?id={match_id}&_auth={self.auth_token}&ea_platform=android_app{_audio_query(match.spec)}"
        bot_info = {
            "name": str(match.spec.get("p2_model", "random")),
            # Система сложностей удалена — модель всегда играет на максимум.
            "difficulty": BOT_MAX_DIFFICULTY,
            "is_bot": True,
        }
        return web.json_response({
            "status": "found",
            "match_id": match_id,
            "is_bot": True,
            "redirect_url": redirect_url,
            "player_ids": [self.human_user_id, match.engine.bot_user_id],
            "bot_info": bot_info,
            "group_id": match.group_id,
        })


def make_fake_jwt(seed: str = "rlhf") -> str:
    """JWT-подобная строка (3 base64url-сегмента) для looksLikeArenaJwtBearer.

    arena.js:2189 требует вид `xxx.yyy.zzz`. Содержимое не валидируется сервером —
    RLHF-среда принимает любой Bearer как фиксированный human (user_id=1000).
    """
    import base64
    def b64u(s: str) -> str:
        return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
    header = b64u('{"alg":"HS256","typ":"JWT"}')
    payload = b64u(f'{{"uid":{HUMAN_USER_ID},"src":"rlhf","seed":"{seed}"}}')
    sig = b64u("rlhf-sig-" + seed)
    return f"{header}.{payload}.{sig}"


__all__ = ["ArenaTransport", "make_fake_jwt", "HUMAN_USER_ID"]