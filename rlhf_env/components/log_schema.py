"""Схемы battle_log.json и manifest.json для RLHF-среды.

Версии форматов:
    BATTLE_LOG_VERSION = "1.0"
    MANIFEST_VERSION = "1.0"

battle_log.json — один файл = один матч. Содержит:
  - идентификаторы (battle_id, group_id)
  - временные метки (started_at, finished_at, duration_seconds)
  - результат (winner/loser/draw, status)
  - использованные модели и колоды
  - список actions[] с подробным контекстом
  - финальное состояние

manifest.json — один файл = одна группа боёв. Содержит:
  - group_id, даты создания/завершения
  - полную спеку запуска
  - агрегированные результаты
  - список battle_ids

Все схемы — dataclass-friendly словари. Используются как в battle_runner,
так и в MCP-ответах.
"""
from __future__ import annotations

from typing import Any, Dict, List

BATTLE_LOG_VERSION = "1.0"
MANIFEST_VERSION = "1.0"

REQUIRED_BATTLE_LOG_KEYS = {
    "log_version",
    "battle_id",
    "group_id",
    "started_at",
    "finished_at",
    "duration_seconds",
    "result",
    "models",
    "decks",
    "actions",
    "final_state_summary",
}

REQUIRED_MANIFEST_KEYS = {
    "manifest_version",
    "group_id",
    "created_at",
    "spec",
    "env",
}


def new_battle_log(
    *,
    battle_id: str,
    group_id: str,
    started_at: str,
    models: Dict[str, Any],
    decks: Dict[str, List[int]],
) -> Dict[str, Any]:
    """Создаёт пустой battle_log с обязательными полями и пустым actions[]."""
    return {
        "log_version": BATTLE_LOG_VERSION,
        "battle_id": battle_id,
        "group_id": group_id,
        "started_at": started_at,
        "finished_at": "",
        "duration_seconds": 0.0,
        "result": {"status": "ONGOING", "winner_user_id": None, "loser_user_id": None},
        "models": models,
        "decks": decks,
        "actions": [],
        "final_state_summary": {},
    }


def validate_battle_log(payload: Dict[str, Any]) -> List[str]:
    """Возвращает список ошибок валидации (пустой = OK). Не падает."""
    errors: List[str] = []
    missing = REQUIRED_BATTLE_LOG_KEYS - set(payload.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if payload.get("log_version") != BATTLE_LOG_VERSION:
        errors.append(
            f"log_version mismatch: expected {BATTLE_LOG_VERSION}, "
            f"got {payload.get('log_version')!r}"
        )
    actions = payload.get("actions")
    if not isinstance(actions, list):
        errors.append("actions must be a list")
    return errors


def validate_manifest(payload: Dict[str, Any]) -> List[str]:
    """Возвращает список ошибок валидации манифеста (пустой = OK)."""
    errors: List[str] = []
    missing = REQUIRED_MANIFEST_KEYS - set(payload.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        errors.append(
            f"manifest_version mismatch: expected {MANIFEST_VERSION}, "
            f"got {payload.get('manifest_version')!r}"
        )
    return errors


def summarize_state(state_summary: Dict[str, int]) -> Dict[str, int]:
    """Нормализует state-summary для battle_log (turn_number/p1_hp/p2_hp/...)."""
    return {
        "turn_number": int(state_summary.get("turn_number", 0)),
        "p1_hp": int(state_summary.get("p1_hp", 0)),
        "p2_hp": int(state_summary.get("p2_hp", 0)),
        "p1_mana": int(state_summary.get("p1_mana", 0)),
        "p2_mana": int(state_summary.get("p2_mana", 0)),
        "p1_max_mana": int(state_summary.get("p1_max_mana", 0)),
        "p2_max_mana": int(state_summary.get("p2_max_mana", 0)),
        "p1_board_count": int(state_summary.get("p1_board_count", 0)),
        "p2_board_count": int(state_summary.get("p2_board_count", 0)),
    }


__all__ = [
    "BATTLE_LOG_VERSION",
    "MANIFEST_VERSION",
    "REQUIRED_BATTLE_LOG_KEYS",
    "REQUIRED_MANIFEST_KEYS",
    "new_battle_log",
    "validate_battle_log",
    "validate_manifest",
    "summarize_state",
]
