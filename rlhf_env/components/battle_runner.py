"""BattleRunner — запуск одного боя с записью полного battle_log.json.

Использует core.engine.ArenaEnvironment напрямую (минимальный цикл).
Каждое действие (человека или бота) логируется с:
  - turn/actor/kind
  - action_dict (to_dict() BaseAction)
  - timestamp_ms
  - state_before_summary (HP/mana/board counts)
  - action_history_lines (последние N строк из GameState.action_history)

Поддерживает 2 режима:
  - «автоматический» (model vs model) — оба игрока = политики
  - «интерактивный» (human vs model) — главный цикл ждёт action от
    внешнего кода (см. rlhf_env.server для WS-интеграции)
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from uuid import uuid4

from rlhf_env.components.log_schema import (
    BATTLE_LOG_VERSION,
    new_battle_log,
    summarize_state,
)

logger = logging.getLogger(__name__)

# Алиасы типов
PolicyLike = Any  # ai.model_benchmark.policies.BenchmarkPolicy (утиная типизация)
ActionProvider = Callable[[int, List[Any], Any], Union[int, Awaitable[int]]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_summary(engine) -> Dict[str, int]:
    """Снимает компактное summary GameState для battle_log."""
    s = engine.state
    return summarize_state(
        {
            "turn_number": s.turn_number,
            "p1_hp": s.p1.hero.hp,
            "p2_hp": s.p2.hero.hp,
            "p1_mana": s.p1.mana,
            "p2_mana": s.p2.mana,
            "p1_max_mana": s.p1.max_mana,
            "p2_max_mana": s.p2.max_mana,
            "p1_board_count": len(s.p1.board),
            "p2_board_count": len(s.p2.board),
        }
    )


def _action_history_snapshot(engine) -> List[str]:
    """Снимает последние N строк из GameState.action_history для контекста."""
    return [text for _kind, text in list(engine.state.action_history)[-10:]]


def _winner_user_id(engine) -> tuple[Optional[int], Optional[int], str]:
    """Извлекает (winner, loser, status) из GameState.

    engine.state — это GameState с полями p1/p2 (PlayerState) и status (GameStatus enum).
    """
    from core.state import GameStatus

    s = engine.state
    status = s.status
    if status == GameStatus.P1_WIN:
        return s.p1.user_id, s.p2.user_id, "P1_WIN"
    if status == GameStatus.P2_WIN:
        return s.p2.user_id, s.p1.user_id, "P2_WIN"
    if status == GameStatus.DRAW:
        return None, None, "DRAW"
    return None, None, "ONGOING"


class BattleRunner:
    """Один матч между двумя «мозгами» (политики / человек через WS)."""

    def __init__(
        self,
        *,
        group_id: str,
        battle_id: Optional[str] = None,
        policy_a: Any,
        policy_b: Any,
        engine: Any,  # core.engine.ArenaEnvironment
        battle_log_path: Path | str,
        human_player: Optional[int] = None,  # 1 или 2; если None — оба авто
        max_turns: int = 60,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.group_id = group_id
        self.battle_id = battle_id or uuid4().hex[:12]
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.engine = engine
        self.battle_log_path = Path(battle_log_path)
        self.human_player = human_player
        self.max_turns = max_turns
        self.on_step = on_step

        self.battle_log = new_battle_log(
            battle_id=self.battle_id,
            group_id=group_id,
            started_at=_utc_now_iso(),
            models=self._capture_models(),
            decks=self._capture_decks(),
        )

    def _capture_models(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for slot, pol in (("p1", self.policy_a), ("p2", self.policy_b)):
            out[slot] = {
                "name": getattr(pol, "name", str(pol)),
                "kind": getattr(pol, "kind", "unknown") if hasattr(pol, "kind") else "unknown",
                "is_human": slot == ("p1" if self.human_player == 1 else ("p2" if self.human_player == 2 else None)),
            }
        return out

    def _capture_decks(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for slot, player in (("p1", self.engine.state.p1), ("p2", self.engine.state.p2)):
            ids = [c.card_id for c in player.deck + player.hand]
            ids.insert(0, player.hero.card_id)
            out[slot] = ids
        return out

    # ------------------------------------------------------------------
    # Главный цикл
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Запускает бой синхронно (model vs model). Возвращает battle_log."""
        return self._run_sync()

    def _pick_action_for_policy(self, policy: Any, current_pid: int, legal: List[Any]):
        """Получает от policy индекс легального действия (наши обёртки)."""
        from rlhf_env.components.policy_factory import select_action

        if not legal:
            return None
        try:
            result = int(select_action(policy, self.engine, current_pid))
        except Exception as exc:
            logger.error("[BattleRunner] policy failed, fallback to action 0: %s", exc)
            return legal[0]
        return legal[max(0, min(result, len(legal) - 1))]

    @staticmethod
    def _match_action_id_to_legal(
        engine,
        current_pid: int,
        action_id: int,
        legal: List[Any],
    ) -> Optional[int]:
        """Мапит discrete action_id (из ClassicRLEnv) в индекс в legal_actions.

        Используем точную функцию из BerserkInference._find_matching_legal_action_index,
        которая декодирует action_id через ai.train_v2.classic_actions_v1.decode_action
        и сопоставляет с реальным списком legal_actions.
        """
        try:
            from ai.train_v2.classic_actions_v1 import decode_action
            decoded = decode_action(engine.state, current_pid, action_id)
            if decoded is None:
                return None
            from ai.bot_brain import BerserkInference
            return BerserkInference._find_matching_legal_action_index(decoded, legal)
        except Exception:
            return action_id % len(legal) if legal else None

    def _run_sync(self) -> Dict[str, Any]:
        start = time.monotonic()
        self.engine.reset_to_state(self.engine.state)

        while self.engine.state.status.name == "ONGOING" and self.engine.state.turn_number <= self.max_turns:
            current_pid = self.engine.state.current_turn_owner_id
            legal = self.engine.get_legal_actions(current_pid)
            if not legal:
                break
            chosen = self._pick_action_for_policy(
                self.policy_a if current_pid == self.engine.state.p1.user_id else self.policy_b,
                current_pid, legal,
            )
            if chosen is None:
                break
            self._execute_action(current_pid, chosen)
            if self.on_step:
                try:
                    self.on_step(self._latest_step_payload)
                except Exception:
                    pass

        return self._finalize(start)

    async def arun(self, *, action_queue: Optional[asyncio.Queue] = None) -> Dict[str, Any]:
        """Асинхронный запуск.

        Если action_queue передана — для current_pid == human_player берём
        action из очереди (отправлено по WS из браузера). Иначе берём от policy.
        """
        start = time.monotonic()

        while self.engine.state.status.name == "ONGOING" and self.engine.state.turn_number <= self.max_turns:
            current_pid = self.engine.state.current_turn_owner_id
            legal = self.engine.get_legal_actions(current_pid)
            if not legal:
                break

            if self.human_player and current_pid == self.human_player:
                if action_queue is None:
                    raise RuntimeError("human_player set but no action_queue provided")
                msg = await action_queue.get()
                if msg is None:
                    break
                if not isinstance(msg, int):
                    raise TypeError(f"action must be int (index in legal_actions), got {type(msg).__name__}")
                idx = max(0, min(msg, len(legal) - 1))
                chosen = legal[idx]
            else:
                chosen = self._pick_action_for_policy(
                    self.policy_a if current_pid == self.engine.state.p1.user_id else self.policy_b,
                    current_pid, legal,
                )
                if chosen is None:
                    break

            self._execute_action(current_pid, chosen)
            if self.on_step:
                try:
                    self.on_step(self._latest_step_payload)
                except Exception:
                    pass

        return self._finalize(start)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _execute_action(self, player_id: int, action) -> None:
        """Один шаг engine.step + запись в battle_log."""
        from core.state import GameStatus

        state_before = _state_summary(self.engine)
        history_before = list(self.engine.state.action_history)
        ok, err = self.engine.step(player_id, action)
        history_after = list(self.engine.state.action_history)
        new_lines = [t for k, t in history_after[len(history_before):] if k != "system"]

        action_dict = {}
        try:
            action_dict = action.to_dict()
        except Exception:
            action_dict = {"type": "unknown", "repr": repr(action)}

        step = {
            "turn": self.engine.state.turn_number,
            "actor": player_id,
            "kind": action_dict.get("type", "unknown"),
            "action_dict": action_dict,
            "timestamp_ms": int(time.monotonic() * 1000),
            "state_before_summary": state_before,
            "state_after_summary": _state_summary(self.engine),
            "ok": bool(ok),
            "error": err if not ok else None,
            "action_history_new_lines": new_lines,
        }
        self.battle_log["actions"].append(step)
        self._latest_step_payload = step

    def _finalize(self, start_monotonic: float) -> Dict[str, Any]:
        winner, loser, status = _winner_user_id(self.engine)
        duration = time.monotonic() - start_monotonic
        self.battle_log["finished_at"] = _utc_now_iso()
        self.battle_log["duration_seconds"] = round(duration, 3)
        self.battle_log["result"] = {
            "status": status,
            "winner_user_id": winner,
            "loser_user_id": loser,
        }
        self.battle_log["final_state_summary"] = _state_summary(self.engine)
        self.battle_log["log_version"] = BATTLE_LOG_VERSION

        self.battle_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.battle_log_path.write_text(
            __import__("json").dumps(self.battle_log, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.battle_log


__all__ = ["BattleRunner"]