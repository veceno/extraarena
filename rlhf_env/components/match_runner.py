"""MatchRunner — общий цикл одного боя для WS (браузер) и MCP (headless).

Единый источник правды для выполнения действий человека и бота, чтобы
арена-страница (Socket.IO) и MCP-агент вели бой идентично.

Контракт совпадает с web/server.py:
  - human action: HTTP-ответ {result, state, sound_events} | {result, state, error}
  - record_analytics_action вызывается ДО выполнения (pre-snapshot), как в проде
  - bot turn: turn_delay -> loop(max 20) -> execute -> emit state_changed ->
    end_turn -> emit turn_switched | game_over
  - game_over: finalize (battle_log.json + manifest) + emit game_over
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from core.state import GameStatus

from rlhf_env.components.log_schema import BATTLE_LOG_VERSION, new_battle_log, summarize_state

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_summary(engine) -> Dict[str, int]:
    s = engine._arena.state
    return summarize_state({
        "turn_number": s.turn_number,
        "p1_hp": s.p1.hero.hp,
        "p2_hp": s.p2.hero.hp,
        "p1_mana": s.p1.mana,
        "p2_mana": s.p2.mana,
        "p1_max_mana": s.p1.max_mana,
        "p2_max_mana": s.p2.max_mana,
        "p1_board_count": len(s.p1.board),
        "p2_board_count": len(s.p2.board),
    })


def _winner_status(engine) -> tuple[Optional[int], Optional[int], str]:
    s = engine._arena.state
    status = s.status
    if status == GameStatus.P1_WIN:
        return s.p1.user_id, s.p2.user_id, "P1_WIN"
    if status == GameStatus.P2_WIN:
        return s.p2.user_id, s.p1.user_id, "P2_WIN"
    if status == GameStatus.DRAW:
        return None, None, "DRAW"
    return None, None, "ONGOING"


def _build_action_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Форма action_json как в web/server.py (для логирования)."""
    kind = payload.get("type")
    if kind == "play_card":
        raw_card_id = payload.get("card_id_from_hand")
        if raw_card_id is None:
            raw_card_id = payload.get("hand_index")
        if raw_card_id is None:
            raw_card_id = payload.get("card_id")
        return {
            "type": "play_card",
            "card_ref": raw_card_id,
            "board_position": payload.get("target_position") or payload.get("board_position") or payload.get("position") or 0,
            "target_id": payload.get("target_id"),
            "target_is_hero": bool(payload.get("target_is_hero", False)),
        }
    if kind == "attack":
        return {
            "type": "attack",
            "attacker_id": payload.get("attacker_id"),
            "target_id": payload.get("target_id"),
            "target_is_hero": bool(payload.get("target_is_hero", False)),
        }
    return {"type": "end_turn"}


class MatchRunner:
    """Ведёт один ArenaMatch от старта до game_over."""

    def __init__(self, match: Any):
        self.match = match
        self.battle_log = new_battle_log(
            battle_id=match.battle_id,
            group_id=match.group_id,
            started_at=_utc_now_iso(),
            models=self._capture_models(),
            decks={"p1": list(match.p1_deck_ids), "p2": list(match.p2_deck_ids)},
        )
        self._start_monotonic = time.monotonic()
        self._bot_task: Optional[asyncio.Task] = None
        # sync-callback(match_id, event_type, data) — выставляется ArenaTransport'ом.
        # None → no-op (headless/MCP без WS).
        self.broadcaster: Optional[Callable[[str, str, Dict[str, Any]], None]] = None

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.broadcaster is not None:
            try:
                self.broadcaster(self.match.engine.match_id, event_type, data)
            except Exception:  # noqa: BLE001
                logger.warning("broadcaster failed: %s", exc_info=True)

    def _state_with_series(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Добавляет поля серии боёв в HTTP action-response state.

        Socket-кадры (state_changed/game_over) получают эти поля через
        ArenaTransport._broadcast → _series_fields. Но HTTP-ответ действия
        (play_card/attack/end_turn) возвращает голый engine.get_full_state — БЕЗ
        group_id/battle_index/battles_planned/has_next_battle. Клиент делает
        `currentState = result.state` (arena.js:5798/5880/5980/6041), затирая
        has_next_battle, а затем renderBattleState видит state.game_over=true
        (arena.js:3934) и САМ показывает модал результата (мимо handleGameOver,
        который сливает поля серии). Итог: has_next_battle=undefined → кнопка
        «Бои завершены» посреди серии (3 боя, после 2-го — человеческий kill).
        Сокетный game_over пришёл бы позже и поправил currentState, но
        showBattleResult глушит повторный вызов через __resultModalShown
        (arena.js:6299) → кнопка застревает «Бои завершены».

        Решение: нести поля серии в HTTP action-response тоже — единый источник
        правды для has_next_battle из любого канала (HTTP action / HTTP
        /api/battle/state / socket _broadcast).
        """
        m = self.match
        out = dict(state)
        out["group_id"] = m.group_id
        out["battle_index"] = m.battle_index
        out["battles_planned"] = m.battles_planned
        out["has_next_battle"] = int(m.battle_index) + 1 < int(m.battles_planned)
        return out

    def _capture_models(self) -> Dict[str, Any]:
        m = self.match
        human = {"name": "human", "kind": "human", "is_human": True}
        bot = {
            "name": getattr(m.bot_policy, "name", str(m.bot_policy)),
            "kind": getattr(m.bot_policy, "kind", "unknown"),
            "is_human": False,
        }
        return {"p1": human, "p2": bot}

    # ------------------------------------------------------------------
    # Аналитика-хуки (рекордер подключается в фазе 4; None → no-op)
    # ------------------------------------------------------------------
    def _rec_before(self, user_id: int, action_json: Dict[str, Any], decision_source: str,
                    action_id: Optional[int] = None) -> Any:
        # Возвращает кортеж (analytics_handle, v5_handle) — оба могут быть None/-1.
        arec = getattr(self.match, "recorder", None)
        a_handle: Any = None
        if arec is not None:
            try:
                a_handle = arec.before_action(user_id, action_json, decision_source)
            except Exception:  # noqa: BLE001
                logger.warning("recorder.before_action failed: %s", exc_info=True)
                a_handle = None
        v5 = getattr(self.match, "v5_recorder", None)
        v5_handle: int = -1
        if v5 is not None:
            try:
                v5_handle = v5.before_action(user_id, action_json, decision_source, action_id_provided=action_id)
            except Exception:  # noqa: BLE001
                logger.warning("v5.before_action failed: %s", exc_info=True)
                v5_handle = -1
        return (a_handle, v5_handle)

    def _rec_after(self, handle: Any, accepted: bool, error: Optional[str] = None) -> None:
        if handle is None or not isinstance(handle, tuple):
            return
        a_handle, v5_handle = handle
        arec = getattr(self.match, "recorder", None)
        if arec is not None and a_handle is not None:
            try:
                arec.after_action(a_handle, accepted, error=error)
            except Exception:  # noqa: BLE001
                logger.warning("recorder.after_action failed: %s", exc_info=True)
        v5 = getattr(self.match, "v5_recorder", None)
        if v5 is not None and v5_handle is not None and v5_handle >= 0:
            try:
                v5.after_action(v5_handle, accepted, error=error)
            except Exception:  # noqa: BLE001
                logger.warning("v5.after_action failed: %s", exc_info=True)

    def _append_step(self, player_id: int, action_json: Dict[str, Any], ok: bool, error: Optional[str]) -> None:
        state_after = _state_summary(self.match.engine)
        self.battle_log["actions"].append({
            "turn": self.match.engine._arena.state.turn_number,
            "actor": player_id,
            "kind": action_json.get("type", "unknown"),
            "action_dict": action_json,
            "timestamp_ms": int(time.monotonic() * 1000),
            "state_after_summary": state_after,
            "ok": bool(ok),
            "error": error,
        })

    # ------------------------------------------------------------------
    # Человеческое действие (HTTP / MCP)
    # ------------------------------------------------------------------
    async def execute_human_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        match = self.match
        engine = match.engine
        user_id = engine.human_user_id
        kind = payload.get("type", "end_turn")
        action_json = _build_action_json(payload)

        async with match.lock:
            if engine.is_ended:
                state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
                return {"result": {"success": False, "error": "game_over"}, "state": state, "error": "game_over"}

            # pre-snapshot аналитики (как в проде — ДО выполнения)
            handle = self._rec_before(user_id, action_json, "human")

            try:
                if kind == "play_card":
                    raw = action_json["card_ref"]
                    pos = int(action_json.get("board_position") or 0)
                    result = engine.play_card(
                        user_id, raw, pos,
                        target_id=action_json.get("target_id"),
                        target_is_hero=bool(action_json.get("target_is_hero", False)),
                    )
                elif kind == "attack":
                    result = engine.attack_target(
                        user_id,
                        action_json.get("attacker_id"),
                        action_json.get("target_id"),
                        target_is_hero=bool(action_json.get("target_is_hero", False)),
                    )
                else:
                    result = engine.end_turn(user_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("execute_human_action failed: %s", exc, exc_info=True)
                self._rec_after(handle, False, error="action_failed")
                state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
                return {"result": {"success": False, "error": "action_failed"}, "state": state, "error": "action_failed"}

            accepted = bool(result.get("success", False))
            self._rec_after(handle, accepted, error=result.get("error") if not accepted else None)
            self._append_step(user_id, action_json, accepted, result.get("error") if not accepted else None)

            if not accepted:
                state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
                return {"result": result, "state": state, "error": result.get("error")}

            # game_over?
            if result.get("game_over"):
                # Бродкаст финального состояния ДО _finalize — симметрично с
                # run_bot_turn (там state_changed летит перед game_over). Иначе
                # для human-kill клиентский currentState.winner_id остаётся
                # устаревшим на момент сокетного game_over (HTTP-ответ с
                # result.state гоняется с game_over-таском) → handleGameOver
                # видел null → рисовал «Ничья» при реальной победе, и
                # has_next_battle мог не успеть попасть в currentState.
                self._emit("state_changed", {
                    "actor_user_id": user_id,
                    "sound_events": result.get("sound_events", []),
                })
                await self._finalize(result.get("winner_id"), reason="hero_death")
                state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
                return {"result": result, "state": state, "sound_events": result.get("sound_events", [])}

            state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
            # если ход перешёл к боту — запускаем бот-рутин
            if engine.is_current_player_bot():
                self._schedule_bot_turn()
            return {"result": result, "state": state, "sound_events": result.get("sound_events", [])}

    # ------------------------------------------------------------------
    # Ход бота (порт run_bot_routine из web/server.py, упрощённо)
    # ------------------------------------------------------------------
    def _schedule_bot_turn(self) -> None:
        if self._bot_task and not self._bot_task.done():
            return
        self._bot_task = asyncio.create_task(self.run_bot_turn())

    async def run_bot_turn(self) -> None:
        match = self.match
        engine = match.engine
        bot_id = engine.bot_user_id
        classic = engine.mode_config.classic
        difficulty = engine.bot_difficulty

        # 1) turn delay — «обдумывание» (как run_bot_routine в web/server.py).
        async with match.lock:
            if engine.is_ended or engine.get_current_player_id() != bot_id:
                return
        delay_range = (
            classic.bot_hard_turn_delay_range
            if str(difficulty).startswith(("hard", "max"))
            else classic.bot_turn_delay_range
        )
        await asyncio.sleep(random.uniform(*delay_range))

        # 2) пошаговое выполнение (lock per-action, release между действиями —
        #    чтобы WS-бродкасты стримились клиенту во время пауз).
        max_actions = 20
        for _ in range(max_actions):
            async with match.lock:
                if engine.is_ended or engine.get_current_player_id() != bot_id:
                    return
                legal = engine.get_legal_actions_raw(bot_id)
                if not legal:
                    return
                try:
                    idx = int(match.bot_policy.select_action(engine._arena, bot_id))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bot policy failed, fallback end_turn: %s", exc)
                    idx = len(legal) - 1
                idx = max(0, min(idx, len(legal) - 1))
                chosen = legal[idx]
                action_json = chosen.to_dict()
                handle = self._rec_before(bot_id, action_json, "bot", action_id=idx)
                try:
                    result = engine.execute_action(bot_id, chosen)
                except Exception as exc:  # noqa: BLE001
                    logger.error("bot execute_action failed: %s", exc, exc_info=True)
                    self._rec_after(handle, False)
                    return
                accepted = bool(result.get("success", False))
                self._rec_after(handle, accepted, error=result.get("error") if not accepted else None)
                self._append_step(bot_id, action_json, accepted, result.get("error") if not accepted else None)

                # бродкаст состояния (WS-клиент видит ход бота).
                self._emit("state_changed", {
                    "actor_user_id": bot_id,
                    "sound_events": result.get("sound_events", []),
                })

                if result.get("game_over"):
                    await self._finalize(result.get("winner_id"), reason="bot_action")
                    return
                # после end_turn ход переходит к человеку — выходим из рутины.
                if action_json.get("type") == "end_turn" and accepted:
                    # turn_start — сброс таймера на стороне клиента.
                    self._emit("turn_start", {"actor_user_id": bot_id})
                    return
            # пауза между действиями бота (lock отпущен — бродкасты идут).
            await asyncio.sleep(random.uniform(*classic.bot_action_gap_range))

    # ------------------------------------------------------------------
    # Сдача
    # ------------------------------------------------------------------
    async def surrender(self) -> Dict[str, Any]:
        match = self.match
        engine = match.engine
        user_id = engine.human_user_id
        async with match.lock:
            # V5 terminal row: pre-snapshot ДО mark_surrender, post — после (status=P2_WIN).
            # Без этого терминальное состояние сдачи не попадает в actions.jsonl.
            v5 = getattr(match, "v5_recorder", None)
            v5_handle: int = -1
            if v5 is not None:
                try:
                    v5_handle = v5.record_terminal(user_id, "surrender", "surrender")
                except Exception:  # noqa: BLE001
                    logger.warning("v5.record_terminal failed: %s", exc_info=True)
                    v5_handle = -1
            engine.mark_surrender(user_id)
            if v5 is not None and v5_handle is not None and v5_handle >= 0:
                try:
                    v5.after_action(v5_handle, accepted=True)
                except Exception:  # noqa: BLE001
                    logger.warning("v5 surrender after_action failed: %s", exc_info=True)
            # Победитель — бот.
            winner_id = engine.bot_user_id
            await self._finalize(winner_id, reason="surrender")
            state = self._state_with_series(engine.get_full_state(viewer_id=user_id))
        return {"result": {"success": True, "game_over": True, "winner_id": winner_id}, "state": state}

    # ------------------------------------------------------------------
    # Финализация
    # ------------------------------------------------------------------
    async def _finalize(self, winner_id: Optional[int], *, reason: str) -> Dict[str, Any]:
        match = self.match
        if match.finished:
            return {}
        match.finished = True
        engine = match.engine
        engine.is_ended = True

        winner, loser, status = _winner_status(engine)
        if winner is None and winner_id is not None:
            winner = winner_id
            loser = engine.bot_user_id if winner_id == engine.human_user_id else engine.human_user_id
            status = "P1_WIN" if winner_id == engine.human_user_id else "P2_WIN"

        duration = time.monotonic() - self._start_monotonic
        self.battle_log["finished_at"] = _utc_now_iso()
        self.battle_log["duration_seconds"] = round(duration, 3)
        self.battle_log["result"] = {"status": status, "winner_user_id": winner, "loser_user_id": loser}
        self.battle_log["final_state_summary"] = _state_summary(engine)
        self.battle_log["log_version"] = BATTLE_LOG_VERSION

        # battle_log.json
        group_dir = match.manifest.group_dir
        battle_log_path = group_dir / "battles" / f"{match.battle_id}.json"
        battle_log_path.parent.mkdir(parents=True, exist_ok=True)
        battle_log_path.write_text(
            json.dumps(self.battle_log, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # манифест
        v5 = getattr(match, "v5_recorder", None)
        v5_info: Dict[str, Any] = {}
        if v5 is not None:
            try:
                v5_info = v5.finalize(winner, status.lower())
            except Exception:  # noqa: BLE001
                logger.warning("v5.finalize failed: %s", exc_info=True)

        match.manifest.append_battle_result(
            battle_id=match.battle_id,
            battle_log_path=str(battle_log_path),
            winner_user_id=winner,
            loser_user_id=loser,
            status=status,
            turns=engine._arena.state.turn_number,
            duration_seconds=round(duration, 3),
            v5_dir=v5_info.get("v5_dir"),
            v5_meta_path=v5_info.get("v5_meta_path"),
            decks_cache=v5_info.get("decks_cache"),
        )

        # аналитика (NDJSON)
        rec = getattr(match, "recorder", None)
        if rec is not None:
            try:
                rec.finalize(winner)
            except Exception:  # noqa: BLE001
                logger.warning("recorder.finalize failed: %s", exc_info=True)

        # уведомление о завершении игры через broadcaster (WS-клиенты).
        self._emit("game_over", {
            "winner_id": winner,
            "reason": reason,
            "p1_hp": engine._arena.state.p1.hero.hp,
            "p2_hp": engine._arena.state.p2.hero.hp,
        })
        return {"game_over": True, "winner_id": winner, "status": status}

    def game_over_payload(self, winner_id: Optional[int], reason: str) -> Dict[str, Any]:
        engine = self.match.engine
        return {
            "game_over": True,
            "winner_id": winner_id,
            "p1_hp": engine._arena.state.p1.hero.hp,
            "p2_hp": engine._arena.state.p2.hero.hp,
            "reason": reason,
            "players": {},
        }


__all__ = ["MatchRunner"]