"""MCP-сервер RLHF-среды ExtraArena (stdio) — игра агентом headless + метаданные.

Инструменты разделены на две группы:

  Управление серией:
    start_series(spec)                — создать серию, вернуть первый match_id
    next_battle(group_id)             — следующий бой серии (или series_complete)
    list_battle_groups()              — список групп
    get_battle_group_status(gid)      — статус группы
    get_battle_group_manifest(gid)    — manifest.json
    get_dataset(group_id)             — путь к каноничному NDJSON (dataset.jsonl)
    list_battle_manifests(gid)        — список battle_log + .jsonl путей
    download_battle_logs(gid, fmt)    — zip/json папка группы

  Игра (агент играет за человека, headless, без браузера/WS):
    get_state(match_id)               — полный actor-perspective state (как /api/battle/state)
    get_legal_actions(match_id)      — список легальных действий
    submit_action(match_id, action)  — выполнить действие человека; авто-advance бота
    advance_bot(match_id)            — прокрутить один ход бота (если сейчас ход бота)
    surrender(match_id)              — сдаться → финализация + NDJSON flush

Контракт совпадает с браузерной ареной (тот же RlhfBattleEngine + MatchRunner),
поэтому данные из MCP-игры и из браузера идентичны по форме.

Запуск:
    ./rlhf_env/start_rlhf_env.sh mcp
    python -m rlhf_env.mcp_server
    python -m rlhf_env.mcp_server --models-dir /path/to/v5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.arena_match_manager import ArenaMatchManager  # noqa: E402
from rlhf_env.components.dataset_toolbox import DatasetToolbox  # noqa: E402
from rlhf_env.components.match_runner import MatchRunner  # noqa: E402
from rlhf_env.components.policy_factory import BOT_MAX_DIFFICULTY  # noqa: E402
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402
from rlhf_env.components.v5_trace_validate import (  # noqa: E402
    validate_v5_metronome_contract,
    validate_v5_timestamp_contract,
    validate_v5_trace,
)

logger = logging.getLogger(__name__)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CREDENTIAL_URL_RE = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)([^/\s@:]+):([^/\s@]+)@"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:password|passwd|secret|token|api[_-]?key|"
    r"authorization|dsn|database[_-]?url))"
    r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^,\s;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?"
    r"-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)


def _redact_sensitive_error_text(value: Any) -> str:
    """Redact common credential forms without reflecting raw exceptions."""

    text = str(value or "")
    text = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}[REDACTED]"
        ),
        text,
    )
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    return _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)


def _public_tool_error(exc: Exception) -> dict[str, str]:
    """Build a stable MCP error that cannot expose an internal exception."""

    if isinstance(exc, ValueError):
        message = _redact_sensitive_error_text(exc).strip()
        return {
            "error": message or "invalid tool request",
            "error_code": "invalid_request",
        }
    return {
        "error": "internal tool error",
        "error_code": "internal_tool_error",
    }


def _safe_component(value: Any, *, field: str) -> str:
    component = str(value or "")
    if (
        not _SAFE_COMPONENT_RE.fullmatch(component)
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
    ):
        raise ValueError(f"invalid {field}")
    return component


def _safe_session_path(root: Path, *components: tuple[str, str] | str) -> Path:
    root_resolved = root.resolve()
    current = root_resolved
    for raw in components:
        if isinstance(raw, tuple):
            value, field = raw
        else:
            value, field = raw, "path_component"
        current = current / _safe_component(value, field=field)
        if current.is_symlink():
            raise ValueError("session path crosses a symlink")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("session path escapes sessions_dir") from exc
    return resolved


def _validate_tool_arguments(
    arguments: Dict[str, Any],
    schema: Dict[str, Any],
) -> None:
    """Enforce the bounded JSON-Schema subset used by dataset tools."""
    properties = schema.get("properties") or {}
    missing = [
        name
        for name in (schema.get("required") or [])
        if name not in arguments
    ]
    if missing:
        raise ValueError(
            "missing required tool arguments: " + ", ".join(sorted(missing))
        )
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(
                "unknown tool arguments: " + ", ".join(unknown)
            )
    for name, value in arguments.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, dict):
            continue
        expected_type = field_schema.get("type")
        type_ok = (
            expected_type == "string"
            and isinstance(value, str)
            or expected_type == "boolean"
            and isinstance(value, bool)
            or expected_type == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
            or expected_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            or expected_type == "object"
            and isinstance(value, dict)
            or expected_type == "array"
            and isinstance(value, list)
            or expected_type is None
        )
        if not type_ok:
            raise ValueError(
                f"tool argument {name} must be {expected_type}"
            )
        if "enum" in field_schema and value not in field_schema["enum"]:
            raise ValueError(
                f"tool argument {name} must be one of "
                f"{field_schema['enum']}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                "minimum" in field_schema
                and value < field_schema["minimum"]
            ):
                raise ValueError(
                    f"tool argument {name} is below minimum"
                )
            if (
                "maximum" in field_schema
                and value > field_schema["maximum"]
            ):
                raise ValueError(
                    f"tool argument {name} is above maximum"
                )
            if (
                "exclusiveMinimum" in field_schema
                and value <= field_schema["exclusiveMinimum"]
            ):
                raise ValueError(
                    f"tool argument {name} is below exclusive minimum"
                )
            if (
                "exclusiveMaximum" in field_schema
                and value >= field_schema["exclusiveMaximum"]
            ):
                raise ValueError(
                    f"tool argument {name} is above exclusive maximum"
                )


def _wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denominator = 1.0 + z * z / games
    centre = p + z * z / (2.0 * games)
    margin = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * games)) / games
    )
    return max(0.0, (centre - margin) / denominator)


def _semi_synthetic_quality(
    group_dir: Path,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Behavioral gate for LLM traces; structural validity is insufficient."""

    llm_results = [
        row
        for row in results
        if str(row.get("battle_tag") or "").startswith("llm-vs-")
    ]
    games = len(llm_results)
    wins = sum(
        str(row.get("status") or "").upper() == "P1_WIN"
        for row in llm_results
    )
    decisions = rejected = end_turn = 0
    end_turn_attack = end_turn_play = mana_draw = degraded = 0
    accepted_training_rows = 0
    for result in llm_results:
        degraded += int(
            bool(result.get("degraded") or result.get("policy_warnings"))
        )
        try:
            action_path = _safe_session_path(
                group_dir,
                "battles",
                (
                    str(result.get("battle_id")),
                    "battle_id",
                ),
                "v5",
                "actions.jsonl",
            )
        except ValueError:
            degraded += 1
            continue
        if not action_path.is_file():
            continue
        with action_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("decision_source") != "llm":
                    continue
                decisions += 1
                accepted = row.get("accepted") is True
                accepted_training_rows += int(accepted)
                rejected += int(not accepted or bool(row.get("error")))
                action_type = row.get("action_type")
                mana_draw += int(action_type == "mana_draw")
                if action_type == "end_turn":
                    end_turn += 1
                    legal_types = {
                        action.get("type")
                        for action in (row.get("legal_actions") or [])
                        if isinstance(action, dict)
                    }
                    end_turn_attack += int("attack" in legal_types)
                    end_turn_play += int("play_card" in legal_types)
    win_rate = wins / games if games else 0.0
    lower_bound = _wilson_lower_bound(wins, games)
    usable = bool(
        games >= 50
        and decisions > 0
        and degraded == 0
        and rejected == 0
        and lower_bound > 0.03
    )
    return {
        "llm_battles": games,
        "llm_wins": wins,
        "p1_win_rate": win_rate,
        "p1_win_rate_wilson_lower_95": lower_bound,
        "llm_decisions": decisions,
        "accepted_training_rows": accepted_training_rows,
        "rejected_audit_rows": rejected,
        "rejection_rate": rejected / decisions if decisions else 0.0,
        "degraded_battles": degraded,
        "mana_draw_decisions": mana_draw,
        "end_turn_decisions": end_turn,
        "end_turn_with_attack_legal": end_turn_attack,
        "end_turn_with_play_legal": end_turn_play,
        "minimum_battles": 50,
        "minimum_wilson_lower_bound": 0.03,
        "semi_synthetic_usable": usable,
    }


def _aggregate_quality_reports(
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    additive = (
        "llm_battles",
        "llm_wins",
        "llm_decisions",
        "accepted_training_rows",
        "rejected_audit_rows",
        "degraded_battles",
        "mana_draw_decisions",
        "end_turn_decisions",
        "end_turn_with_attack_legal",
        "end_turn_with_play_legal",
    )
    total = {
        key: sum(int(report.get(key, 0) or 0) for report in reports)
        for key in additive
    }
    games = total["llm_battles"]
    decisions = total["llm_decisions"]
    total["p1_win_rate"] = total["llm_wins"] / games if games else 0.0
    total["p1_win_rate_wilson_lower_95"] = _wilson_lower_bound(
        total["llm_wins"],
        games,
    )
    total["rejection_rate"] = (
        total["rejected_audit_rows"] / decisions if decisions else 0.0
    )
    total["minimum_battles"] = 50
    total["minimum_wilson_lower_bound"] = 0.03
    total["semi_synthetic_usable"] = bool(
        games >= 50
        and decisions > 0
        and total["degraded_battles"] == 0
        and total["rejected_audit_rows"] == 0
        and total["p1_win_rate_wilson_lower_95"] > 0.03
    )
    return total


def _headless_auxiliary_evidence(v5_dir: Path) -> Dict[str, Any]:
    """Report timing evidence without pretending it is part of policy readiness."""

    meta_path = v5_dir / "meta.json"
    actions_path = v5_dir / "actions.jsonl"
    if (
        meta_path.is_symlink()
        or actions_path.is_symlink()
        or not meta_path.is_file()
        or not actions_path.is_file()
    ):
        return {
            "timestamp_contract_ok": False,
            "metronome_contract_ok": False,
            "metronome_observed_labels": 0,
        }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actions = [
            json.loads(raw)
            for raw in actions_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if raw.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return {
            "timestamp_contract_ok": False,
            "metronome_contract_ok": False,
            "metronome_observed_labels": 0,
        }
    return {
        "timestamp_contract_ok": not validate_v5_timestamp_contract(meta),
        "metronome_contract_ok": not validate_v5_metronome_contract(
            actions
        ),
        "metronome_observed_labels": sum(
            row.get("decision_source") == "human"
            and row.get("control_source") == "human"
            and row.get("decision_time_censored") is False
            and isinstance(
                row.get("human_decision_time_raw_ms"),
                (int, float),
            )
            and not isinstance(
                row.get("human_decision_time_raw_ms"),
                bool,
            )
            for row in actions
        ),
    }


def _compact_state(
    state: Dict[str, Any],
    *,
    history_limit: int = 8,
) -> Dict[str, Any]:
    """Decision-complete, token-bounded view for LLM player loops."""

    card_keys = (
        "instance_id",
        "card_id",
        "name",
        "description",
        "card_type",
        "mana_cost",
        "attack",
        "hp",
        "max_hp",
        "mechanics",
        "is_ready",
        "can_attack",
        "is_frozen",
        "level",
    )

    def compact_card(value: Any) -> Dict[str, Any]:
        return {
            key: value[key]
            for key in card_keys
            if isinstance(value, dict) and key in value
        }

    def compact_side(value: Any, *, include_hand: bool) -> Dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        result = {
            "user_id": raw.get("user_id"),
            "mana": raw.get("mana"),
            "max_mana": raw.get("max_mana"),
            "mana_draw_count_this_turn": raw.get(
                "mana_draw_count_this_turn",
                0,
            ),
            "hero": compact_card(raw.get("hero")),
            "board": [
                compact_card(card) for card in (raw.get("board") or [])
            ],
            "hand_count": raw.get(
                "hand_count",
                len(raw.get("hand") or []),
            ),
            "deck_count": raw.get("deck_count"),
        }
        if include_hand:
            result["hand"] = [
                compact_card(card) for card in (raw.get("hand") or [])
            ]
        return result

    top_keys = (
        "match_id",
        "turn",
        "current_player_id",
        "is_my_turn",
        "is_ended",
        "game_over",
        "winner_id",
        "ruleset",
        "sudden_death",
    )
    result = {key: state.get(key) for key in top_keys if key in state}
    result["player"] = compact_side(state.get("player"), include_hand=True)
    result["opponent"] = compact_side(
        state.get("opponent"),
        include_hand=False,
    )
    result["legal_actions"] = [
        {"legal_action_index": index, **dict(action)}
        for index, action in enumerate(state.get("legal_actions") or [])
        if isinstance(action, dict)
    ]
    history = list(state.get("action_history") or [])
    result["action_history"] = history[
        -max(0, min(int(history_limit), 20)) :
    ]
    result["compact"] = True
    return result


def _wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    centre = p + z * z / (2.0 * games)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
    return max(0.0, (centre - margin) / denom)


def _semi_synthetic_quality(gdir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Behavioral gate for LLM traces; structural validity alone is not enough."""
    llm_results = [
        r for r in results if str(r.get("battle_tag") or "").startswith("llm-vs-")
    ]
    games = len(llm_results)
    wins = sum(1 for r in llm_results if str(r.get("status", "")).upper() == "P1_WIN")
    decisions = rejected = end_turn = end_turn_attack = end_turn_play = mana_draw = 0
    degraded = 0
    for r in llm_results:
        degraded += int(bool(r.get("degraded") or r.get("policy_warnings")))
        apath = gdir / "battles" / str(r.get("battle_id")) / "v5" / "actions.jsonl"
        if not apath.exists():
            continue
        for line in apath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("decision_source") != "llm":
                continue
            decisions += 1
            rejected += int(row.get("accepted") is not True or bool(row.get("error")))
            action_type = row.get("action_type")
            mana_draw += int(action_type == "mana_draw")
            if action_type == "end_turn":
                end_turn += 1
                legal_types = {a.get("type") for a in (row.get("legal_actions") or []) if isinstance(a, dict)}
                end_turn_attack += int("attack" in legal_types)
                end_turn_play += int("play_card" in legal_types)
    win_rate = wins / games if games else 0.0
    lower = _wilson_lower_bound(wins, games)
    rejection_rate = rejected / decisions if decisions else 0.0
    # A campaign is promotable only after a bounded 50-game smoke and a
    # statistically conservative lower bound above the user's 3% floor.
    usable = bool(
        games >= 50
        and decisions > 0
        and degraded == 0
        and rejected == 0
        and lower > 0.03
    )
    return {
        "llm_battles": games,
        "llm_wins": wins,
        "p1_win_rate": win_rate,
        "p1_win_rate_wilson_lower_95": lower,
        "llm_decisions": decisions,
        "rejected_or_error_decisions": rejected,
        "rejection_rate": rejection_rate,
        "degraded_battles": degraded,
        "mana_draw_decisions": mana_draw,
        "end_turn_decisions": end_turn,
        "end_turn_with_attack_legal": end_turn_attack,
        "end_turn_with_play_legal": end_turn_play,
        "minimum_battles": 50,
        "minimum_wilson_lower_bound": 0.03,
        "semi_synthetic_usable": usable,
    }


def _aggregate_quality_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    additive = (
        "llm_battles", "llm_wins", "llm_decisions",
        "rejected_or_error_decisions", "degraded_battles",
        "mana_draw_decisions", "end_turn_decisions",
        "end_turn_with_attack_legal", "end_turn_with_play_legal",
    )
    total = {k: sum(int(r.get(k, 0) or 0) for r in reports) for k in additive}
    games = total["llm_battles"]
    decisions = total["llm_decisions"]
    total["p1_win_rate"] = total["llm_wins"] / games if games else 0.0
    total["p1_win_rate_wilson_lower_95"] = _wilson_lower_bound(total["llm_wins"], games)
    total["rejection_rate"] = (
        total["rejected_or_error_decisions"] / decisions if decisions else 0.0
    )
    total["minimum_battles"] = 50
    total["minimum_wilson_lower_bound"] = 0.03
    total["semi_synthetic_usable"] = bool(
        games >= 50
        and decisions > 0
        and total["degraded_battles"] == 0
        and total["rejected_or_error_decisions"] == 0
        and total["p1_win_rate_wilson_lower_95"] > 0.03
    )
    return total


def _compact_state(state: Dict[str, Any], *, history_limit: int = 8) -> Dict[str, Any]:
    """Decision-complete, token-bounded view for LLM player loops."""
    card_keys = (
        "instance_id", "card_id", "name", "description", "card_type",
        "mana_cost", "attack", "hp", "max_hp", "mechanics", "is_ready",
        "can_attack", "is_frozen", "level",
    )

    def card(c: Any) -> Dict[str, Any]:
        return {k: c[k] for k in card_keys if isinstance(c, dict) and k in c}

    def side(raw: Any, *, include_hand: bool) -> Dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        out = {
            "user_id": raw.get("user_id"),
            "mana": raw.get("mana"),
            "max_mana": raw.get("max_mana"),
            "mana_draw_count_this_turn": raw.get("mana_draw_count_this_turn", 0),
            "hero": card(raw.get("hero")),
            "board": [card(c) for c in (raw.get("board") or [])],
            "hand_count": raw.get("hand_count", len(raw.get("hand") or [])),
            "deck_count": raw.get("deck_count"),
        }
        if include_hand:
            out["hand"] = [card(c) for c in (raw.get("hand") or [])]
        return out

    top_keys = (
        "match_id", "turn", "current_player_id", "is_my_turn", "is_ended",
        "game_over", "winner_id", "ruleset", "sudden_death",
    )
    out = {k: state.get(k) for k in top_keys if k in state}
    out["player"] = side(state.get("player"), include_hand=True)
    out["opponent"] = side(state.get("opponent"), include_hand=False)
    # Long UUIDs are deliberately kept for inspection, but LLM players should
    # submit the stable per-state index instead of transcribing them. Small
    # models were observed mutating one or two UUID characters, turning an
    # otherwise sound choice into a rejected action.
    out["legal_actions"] = [
        {"legal_action_index": idx, **dict(action)}
        for idx, action in enumerate(state.get("legal_actions") or [])
        if isinstance(action, dict)
    ]
    history = list(state.get("action_history") or [])
    out["action_history"] = history[-max(0, min(int(history_limit), 20)):]
    out["compact"] = True
    return out


def _policy_info(policy: Any) -> Optional[Dict[str, Any]]:
    """Лёгкое provenance-описание политики: {name,kind,model_path,weights_hash,weights_version}.

    Используется в ответах start_series/register_custom_model, чтобы оркестратор
    видел resolved adapter (V2/V3/V4/V5/baseline) без полного state.
    """
    if policy is None:
        return None
    info: Dict[str, Any] = {
        "name": getattr(policy, "name", None),
        "kind": getattr(policy, "kind", None),
    }
    for attr in ("model_path", "weights_hash", "weights_version"):
        v = getattr(policy, attr, None)
        if v is not None:
            info[attr] = v
    return info


def _runner_warnings(runner: Optional[MatchRunner]) -> List[str]:
    """BUG4: проброс policy-fallback предупреждений (напр. v5-stub → end_turn)
    в ответ start_series/next_battle, чтобы MCP-клиент видел деградацию модели,
    а не молчаливый «нормальный» бой."""
    if runner is None:
        return []
    return list(getattr(runner, "policy_fallbacks", []) or [])


# ============================================================================
# HeadlessHub — реестр матчей + ленивые MatchRunner'ы (без WS/broadcaster)
# ============================================================================

class HeadlessHub:
    """Связывает ArenaMatchManager с MatchRunner'ами для headless-игры."""

    def __init__(self, *, sessions_dir: str, models_dir: str, cards_path: str,
                 registry: Optional["PolicyRegistry"] = None):
        # ЕДИНЫЙ реестр моделей на hub: MCPServer.list_models/register_custom_model
        # и ArenaMatchManager.start_series (_build_match → build_policy) должны делить
        # один объект, иначе register_custom_model добавляет модель в реестр A, а
        # start_series резолвит из реестра B → KeyError (BUG1: кастомные модели unusable).
        self.manager = ArenaMatchManager(
            sessions_dir=sessions_dir, models_dir=models_dir, cards_path=cards_path,
            registry=registry,
        )
        self._runners: Dict[str, MatchRunner] = {}

    def _match(self, match_id: str):
        return self.manager.get_match(match_id)

    async def get_runner(self, match_id: str) -> Optional[MatchRunner]:
        r = self._runners.get(match_id)
        if r is not None:
            return r
        match = self.manager.get_match(match_id)
        if match is None:
            return None
        r = MatchRunner(match)
        # broadcaster=None → WS-бродкасты не нужны (headless).
        self._runners[match_id] = r
        return r


# ============================================================================
# MCP-сервер (JSON-RPC 2.0 over stdio)
# ============================================================================

class MCPServer:
    def __init__(
        self,
        hub: HeadlessHub,
        registry: Optional[PolicyRegistry] = None,
        *,
        dataset_toolbox: Optional[DatasetToolbox] = None,
    ):
        self.hub = hub
        # BUG1 фикс: list_models/register_custom_model и start_series должны делить
        # один реестр. Авторитативный — hub.manager.registry (его использует
        # _build_match). Дополнительно вливаем спеки из переданного registry, если
        # они были созданы отдельным scan (main()/тесты), чтобы ничего не потерять.
        self.registry = hub.manager.registry
        if registry is not None and registry is not self.registry:
            for spec in registry.specs:
                if spec.name not in self.registry._name_index:
                    self.registry.add_spec(spec)
        production_enabled = str(
            os.environ.get("RLHF_ENABLE_PRODUCTION_DATASETS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.dataset_toolbox = dataset_toolbox or DatasetToolbox(
            os.environ.get(
                "RLHF_DATASETS_DIR",
                str(hub.manager.sessions_dir.parent / "datasets"),
            ),
            returnclock_salt_env=os.environ.get(
                "RLHF_RETURNCLOCK_SALT_ENV",
                "RETURNCLOCK_DATASET_SALT",
            ),
            returnclock_salt_key_id_env=os.environ.get(
                "RLHF_RETURNCLOCK_SALT_KEY_ID_ENV",
                "RETURNCLOCK_DATASET_SALT_KEY_ID",
            ),
            production_enabled=production_enabled,
            cards_path=hub.manager.cards_path,
        )
        self.tools = self._build_tools()

    # ------------------------------------------------------------------
    # tool schemas
    # ------------------------------------------------------------------
    def _build_tools(self) -> List[Dict[str, Any]]:
        tools = [
            {
                "name": "start_series",
                "description": (
                    "Создать серию боёв (человек vs модель) и вернуть первый match_id. "
                    "spec: {p2_model, battles_planned, seed?, starting_player?, "
                    "deck_strategy_p1?, deck_strategy_p2?, custom_deck_p1?, custom_deck_p2?, "
                    "p1_name?, p2_name?, ...}. Модель всегда играет на максимум (argmax); "
                    "сложность не выбирается. Агент играет за человека (p1) через submit_action."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": {
                                "p2_model": {
                                    "type": ["string", "object"],
                                    "description": "имя модели-оппонента ('random'/имя из registry) ИЛИ объект {name,path,kind} для custom model by path+adapter.",
                                },
                                "p2_model_path": {"type": "string", "description": "путь к .onnx оппонента (custom by path; под models_dir/repo root)"},
                                "p2_model_kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"], "description": "адаптер оппонента (V2/V3=legacy_onnx, V4=action_onnx, V5=v5, baseline)"},
                                "p1_model": {"type": ["string", "object"], "description": "имя/объект RL-модели для p1 (при p1_actor_type='rl', model-vs-model)"},
                                "p1_model_path": {"type": "string"},
                                "p1_model_kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"]},
                                "battles_planned": {"type": "integer", "default": 1, "minimum": 1, "maximum": 1000},
                                "seed": {"type": "integer", "default": 0},
                                "starting_player": {"type": "string", "default": "random", "enum": ["random", "p1", "p2"]},
                                "deck_strategy_p1": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom", "preset"]},
                                "deck_strategy_p2": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom", "preset"]},
                                "custom_deck_p1": {"type": "array", "items": {"type": "integer"}},
                                "custom_deck_p2": {"type": "array", "items": {"type": "integer"}},
                                "preset_number_p1": {"type": "integer", "description": "номер preset-колоды (deck_strategy_p1='preset')"},
                                "preset_number_p2": {"type": "integer"},
                                "p1_deck_source": {"type": "object", "description": "явная форма {type:'imported',preset_number:N} (движок поддерживает)"},
                                "agent_name": {"type": "string", "description": "кодовое имя суб-агента (опц.); auto-assign из пула если не задано"},
                                "p1_name": {"type": "string"}, "p2_name": {"type": "string"},
                                "p1_actor_type": {
                                    "type": "string", "enum": ["human", "llm", "rl"], "default": "llm",
                                    "description": "тип актора p1: 'llm' (MCP-модель, default), 'human' (браузер) или 'rl' (наша RL-модель auto-play, model-vs-model). Определяет battle_tag и decision_source в V5-трейсах.",
                                },
                            },
                            "required": [],
                        },
                    },
                    "required": ["spec"],
                },
            },
            {
                "name": "next_battle",
                "description": "Перейти к следующему бою серии. Возвращает match_id или series_complete.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_state",
                "description": (
                    "Actor-perspective state. compact=true keeps all decision "
                    "fields and indexed legal actions while bounding history "
                    "for efficient LLM player loops."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "match_id": {"type": "string"},
                        "compact": {"type": "boolean", "default": False},
                        "history_limit": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                            "default": 8,
                        },
                    },
                    "required": ["match_id"],
                },
            },
            {
                "name": "get_legal_actions",
                "description": (
                    "Легальные действия с stable legal_action_index, чтобы "
                    "малой модели не приходилось копировать UUID."
                ),
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "submit_action",
                "description": (
                    "Выполнить действие игрока (p1). Предпочтительно передать "
                    "legal_action_index из последнего get_state/"
                    "get_legal_actions; legacy action также поддерживается. "
                    "play_card: {type:'play_card', card_id_from_hand|hand_index, target_position?, target_id?, target_is_hero?}. "
                    "attack: {type:'attack', attacker_id, target_id, target_is_hero?}. "
                    "end_turn: {type:'end_turn'}. "
                    "mana_draw: {type:'mana_draw'} — добор карты за ману (стоимость 2*(count+1)/ход, не передаёт ход). "
                    "После play_card/attack/end_turn ход бота прокручивается автоматически; после mana_draw — нет."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "match_id": {"type": "string"},
                        "legal_action_index": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "compact_response": {
                            "type": "boolean",
                            "default": False,
                        },
                        "history_limit": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                            "default": 8,
                        },
                        "action": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["play_card", "attack", "end_turn", "mana_draw"]},
                                "card_id_from_hand": {"type": "integer"},
                                "hand_index": {"type": "integer"},
                                "target_position": {"type": "integer"},
                                "target_id": {"type": ["integer", "string"]},
                                "target_is_hero": {"type": "boolean"},
                                "attacker_id": {"type": ["integer", "string"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["match_id"],
                },
            },
            {
                "name": "advance_bot",
                "description": "Прокрутить один ход бота (если сейчас ход бота). Иначе no-op.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "surrender",
                "description": "Сдаться (человек) → финализация боя + flush NDJSON.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "list_battle_groups",
                "description": "Список всех групп боёв.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_battle_group_status",
                "description": "Статус группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_battle_group_manifest",
                "description": "Полное содержимое manifest.json группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_dataset",
                "description": "Путь к каноничному NDJSON (dataset.jsonl) и per-battle .jsonl файлам группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_battle_manifests",
                "description": "Список battle_log.json + analytics .jsonl путей по группе.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "download_battle_logs",
                "description": "Собрать логи группы в zip или вернуть список json-файлов.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["json", "zip"], "default": "json"},
                    },
                    "required": ["group_id"],
                },
            },
            {
                "name": "list_models",
                "description": "Список доступных моделей-оппонентов (из registry + зарегистрированные через register_custom_model).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "list_active_series",
                "description": (
                    "«Общая картинка» активных серий: (a) число активных игр по агенту и (b) по модели-оппоненту. "
                    "Возвращает агентов с прогрессом боёв N/M + wins/losses/draws/decks и группировку by_model."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_agent_status",
                "description": "Статус «играющего» суб-агента по кодовому имени: бои N/M, победы, поражения, ничьи, колоды, оппонент, p1_actor_type.",
                "inputSchema": {"type": "object", "properties": {"agent_name": {"type": "string"}}, "required": ["agent_name"]},
            },
            {
                "name": "get_match_status",
                "description": "Лёгкий статус боя (polling без полного state): turn, is_ended, winner, is_my_turn, current_player_id, action_count.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "get_action_history",
                "description": "История ходов боя (replay длинных боёв без re-fetch fullstate).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"match_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 200}},
                    "required": ["match_id"],
                },
            },
            {
                "name": "finish_series",
                "description": "Завершить серию досрочно: закрыть manifest (finished_at + summary) и освободить кодовое имя агента.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_preset_decks",
                "description": "Список preset-колод (ArenaENV Random / JSON-imported) для deck_strategy='preset'.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "register_custom_model",
                "description": (
                    "Зарегистрировать кастомную модель by path+adapter (V2/V3/V4/V5/baseline) in-memory: "
                    "позволяет оркестратору выбирать early-snapshot V5 и т.п. через {name,path,kind} в start_series. "
                    "path должен лежать под models_dir или корнем репо (защита от path-traversal)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string", "description": "путь к .onnx под models_dir/repo root"},
                        "kind": {"type": "string", "enum": ["auto", "action_onnx", "legacy_onnx", "v5", "random", "greedy_face", "end_turn"]},
                    },
                    "required": ["name", "path"],
                },
            },
            {
                "name": "get_v5_dataset_summary",
                "description": "V5 training orchestrator: сводка по группе — строки, v5_trace_ok, распределение battle_tag, turns/actions total.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_v5_groups",
                "description": "V5 orchestrator: список групп с v5-трейсами (фильтр по battle_tag).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"battle_tag": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 100}},
                },
            },
            {
                "name": "get_v5_trace",
                "description": "V5 orchestrator: содержимое v5/trace боя (meta|turns|actions).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "battle_id": {"type": "string"},
                        "what": {"type": "string", "enum": ["meta", "turns", "actions"]},
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 2000,
                            "default": 200,
                        },
                    },
                    "required": ["group_id", "battle_id", "what"],
                },
            },
            {
                "name": "validate_v5_traces",
                "description": (
                    "V5 orchestrator: глубокая проверка целостности v5/trace всех боёв "
                    "группы. Помимо наличия/непустоты meta/turns/actions проверяет "
                    "инварианты обучающих данных: (1) legal_action_index валиден и "
                    "action_native==legal_actions[idx]; (2) actor/decision_source "
                    "согласован с meta.actor_type и current_turn_owner_id==actor; "
                    "(3) continuity post_state[N]==pre_state[N+1] + терминал↔meta.status; "
                    "(4) battle_log actions 1:1 с trace-строками. Возвращает "
                    "{checked,ok,broken:[{battle_id,issues:[tagged ...]}]}."
                ),
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_training_data_status",
                "description": (
                    "Readiness overview for local exports and the opt-in "
                    "read-only production dataset plane. Never returns DSNs, "
                    "salts or raw telemetry."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "list_training_exports",
                "description": (
                    "List V5, Nemesis and ReturnClock artifacts under the "
                    "private datasets root."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "returnclock",
                                "returnclock_split",
                                "nemesis",
                                "nemesis_split",
                                "v5_export",
                                "v5_materialized",
                            ],
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 200,
                        },
                    },
                },
            },
            {
                "name": "inspect_training_export",
                "description": (
                    "Inspect header, checksum, size and bounded manifest "
                    "metadata for one artifact. Does not return dataset rows."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "validate_training_export",
                "description": (
                    "Deep schema/privacy/readiness validation for V5, Nemesis "
                    "or ReturnClock. Recomputes counts and split diagnostics."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "export_v5_training_dataset",
                "description": (
                    "Opt-in read-only production export of complete terminal "
                    "V5 bundles. Always pseudonymized; publishes atomically."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output": {"type": "string"},
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 365,
                            "default": 30,
                        },
                        "limit_battles": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                            "default": 1000,
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["output"],
                },
            },
            {
                "name": "materialize_v5_training_dataset",
                "description": (
                    "Convert a V5 transport export into the canonical "
                    "rlhf_v5_storage_v1 directory and publish it only after "
                    "deep validation of V5/Metronome/TimeStamp contracts. "
                    "Use a fresh versioned output_dir for crash-safe handoff."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "group_id": {"type": "string"},
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["input_path", "output_dir"],
                },
            },
            {
                "name": "export_nemesis_training_dataset",
                "description": (
                    "Extract canonical meta.nemesis_record rows from either "
                    "a V5 export or an authoritative headless group, enforce "
                    "terminal/eligibility contracts and write a pseudonymized "
                    "private dataset. Provide exactly one of input_path or "
                    "group_id."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string"},
                        "group_id": {"type": "string"},
                        "output": {"type": "string"},
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["output"],
                },
            },
            {
                "name": "export_returnclock_training_dataset",
                "description": (
                    "Opt-in read-only ReturnClock natural-return export. HMAC "
                    "salt is read only from server env; raw user IDs and salt "
                    "values never enter MCP arguments/responses."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "horizon_hours": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 744,
                            "default": 168,
                        },
                        "safety_lag_minutes": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1440,
                            "default": 10,
                        },
                        "min_analytics_version": {
                            "type": "integer",
                            "minimum": 2,
                            "default": 2,
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000000,
                            "default": 50000,
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["output"],
                },
            },
            {
                "name": "split_nemesis_training_dataset",
                "description": (
                    "Create separately audited Nemesis handoff artifacts from "
                    "a canonical pseudonymized export: Lite deck-matchup "
                    "grouped splits and, when Standard preconditions pass, a "
                    "player-disjoint split that accounts for excluded "
                    "cross-partition battles plus chronological and "
                    "deck-group evaluation splits. Lite-only sources remain "
                    "training-ready for Lite. Standard needs at least six "
                    "players, three pairwise-disjoint battles, three matchup "
                    "groups and three cutoff cohorts. Player grouping aliases "
                    "remain metadata and are forbidden as features."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "train_fraction": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                            "default": 0.70,
                        },
                        "validation_fraction": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                            "default": 0.15,
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["source", "output_dir"],
                },
            },
            {
                "name": "split_returnclock_training_dataset",
                "description": (
                    "Create deterministic organic-only train/validation/test "
                    "JSONL files from a validated ReturnClock export. Users "
                    "are disjoint between splits, user groups are ordered by "
                    "their first prediction cutoff, and treated/boundary "
                    "exclusions are recorded in the manifest."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "train_fraction": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                            "default": 0.70,
                        },
                        "validation_fraction": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                            "default": 0.15,
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["source", "output_dir"],
                },
            },
        ]
        dataset_tools = {
            "get_training_data_status",
            "list_training_exports",
            "inspect_training_export",
            "validate_training_export",
            "export_v5_training_dataset",
            "materialize_v5_training_dataset",
            "export_nemesis_training_dataset",
            "split_nemesis_training_dataset",
            "export_returnclock_training_dataset",
            "split_returnclock_training_dataset",
        }
        for tool in tools:
            if tool["name"] in dataset_tools:
                tool["inputSchema"]["additionalProperties"] = False
        return tools

    # ------------------------------------------------------------------
    # tool handlers (async)
    # ------------------------------------------------------------------
    async def _tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        hub = self.hub

        if name == "start_series":
            spec = args.get("spec", {})
            if not isinstance(spec, dict):
                raise ValueError("spec must be an object")
            spec.setdefault("battles_planned", 1)
            # MCP = LLM-актор по умолчанию (полу-синтетические бои тегируются
            # llm-vs-bot, decision_source='llm' в V5-трейсах).
            spec.setdefault("p1_actor_type", "llm")
            match = hub.manager.create_series(spec)
            # headless: прокручиваем все бот-ходы до хода human/llm ИЛИ до game_over.
            # run_auto единообразно покрывает три режима: human/llm-vs-bot (ходит
            # p2 если он стартует, стоп на p1), rl-vs-bot и rl-vs-rl (model-vs-model,
            # бой до game_over без submit_action). Прежний «if is_current_player_bot:
            # run_bot_turn()» ломал p1-as-RL (по умолчанию ходил только p2).
            runner = await hub.get_runner(match.engine.match_id)
            if runner is not None:
                await self._run_auto_bounded(runner)
            p1_actor = match.engine.p1_actor_type
            return {
                "group_id": match.group_id,
                "match_id": match.engine.match_id,
                "battle_id": match.battle_id,
                "battles_planned": match.battles_planned,
                "player_ids": [match.engine.human_user_id, match.engine.bot_user_id],
                "opponent": {"model": spec.get("p2_model"), "difficulty": BOT_MAX_DIFFICULTY},
                "p1_actor_type": p1_actor,
                "battle_tag": match.engine.battle_tag,
                "agent_name": getattr(match, "agent_name", None),
                "p1_model": _policy_info(match.p1_policy),
                "p2_model": _policy_info(match.bot_policy),
                "is_ended": match.engine.is_ended,
                "winner_id": (match.engine._get_winner_id() if match.engine.is_ended else None),
                "policy_warnings": _runner_warnings(runner),
                "degraded": bool(_runner_warnings(runner)),
            }

        if name == "next_battle":
            gid = _safe_component(args["group_id"], field="group_id")
            try:
                match = hub.manager.next_match(gid)
            except RuntimeError as exc:
                if str(exc) == "current_battle_in_progress":
                    return {
                        "error": "current_battle_in_progress",
                        "group_id": gid,
                    }
                raise
            if match is None:
                return {"status": "series_complete", "group_id": gid}
            runner = await hub.get_runner(match.engine.match_id)
            if runner is not None:
                await self._run_auto_bounded(runner)
            return {"match_id": match.engine.match_id, "battle_id": match.battle_id, "group_id": gid,
                    "agent_name": getattr(match, "agent_name", None),
                    "p1_actor_type": match.engine.p1_actor_type,
                    "battle_tag": match.engine.battle_tag,
                    "is_ended": match.engine.is_ended,
                    "winner_id": (match.engine._get_winner_id() if match.engine.is_ended else None),
                    "policy_warnings": _runner_warnings(runner)}

        if name == "get_state":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            state = match.engine.get_full_state(
                viewer_id=match.engine.human_user_id
            )
            if args.get("compact"):
                return _compact_state(
                    state,
                    history_limit=int(args.get("history_limit", 8)),
                )
            return state

        if name == "get_legal_actions":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            uid = match.engine.human_user_id
            legal_raw = (
                match.engine.get_legal_actions(uid)
                if not match.engine.is_current_player_bot()
                else []
            )
            legal = [
                {"legal_action_index": index, **dict(action)}
                for index, action in enumerate(legal_raw)
            ]
            return {"legal_actions": legal, "is_my_turn": match.engine.get_current_player_id() == uid}

        if name == "submit_action":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            # F4(audit): p1-RL (model-vs-model) не управляется через submit_action —
            # иначе внешний клиент вливает p1-action мимо match.p1_policy auto-play,
            # и он мис-тегируется decision_source='rl' в v5/actions.jsonl → портит
            # V5 training-данные. p1-RL водится только advance_bot/run_auto
            # (run_bot_turn с match.p1_policy). Симметрично surrender/get_legal_actions.
            if getattr(runner.match.engine, "p1_actor_type", "human") == "rl":
                return {"error": "submit_action_unavailable_for_rl_p1"}
            resolved_index = args.get("legal_action_index")
            if resolved_index is not None:
                engine = runner.match.engine
                uid = engine.human_user_id
                if engine.get_current_player_id() != uid:
                    return {"error": "not_your_turn"}
                legal_raw = engine.get_legal_actions_raw(uid)
                if (
                    isinstance(resolved_index, bool)
                    or not isinstance(resolved_index, int)
                ):
                    return {"error": "legal_action_index_must_be_integer"}
                if resolved_index < 0 or resolved_index >= len(legal_raw):
                    return {
                        "error": "legal_action_index_out_of_range",
                        "legal_action_count": len(legal_raw),
                    }
                action = legal_raw[resolved_index].to_dict()
            else:
                action = args.get("action")
                if not isinstance(action, dict) or not action.get("type"):
                    return {
                        "error": "action_or_legal_action_index_required"
                    }
            action = dict(action)
            resolved_action = dict(action)
            action.setdefault("client_action_id", f"mcp_{match_id}_{int(asyncio.get_event_loop().time()*1000)&0xffff}")
            resp = await runner.execute_human_action(action)
            # авто-advance бота уже выполнен в execute_human_action (create_task run_bot_turn),
            # но в headless-режоте create_task мог быть запланирован — дождёмся его.
            await self._drain_bot(runner)
            if resolved_index is not None and isinstance(resp, dict):
                resp = dict(resp)
                resp["resolved_legal_action_index"] = resolved_index
                resp["resolved_legal_action"] = resolved_action
            if args.get("compact_response") and isinstance(resp, dict):
                state = runner.match.engine.get_full_state(
                    viewer_id=runner.match.engine.human_user_id,
                )
                resp = dict(resp)
                resp["state"] = _compact_state(
                    state,
                    history_limit=int(args.get("history_limit", 8)),
                )
            return resp

        if name == "advance_bot":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            engine = runner.match.engine
            if not engine.is_current_player_bot() and getattr(runner.match, "p1_policy", None) is None:
                return {"status": "not_bot_turn"}
            cur = engine.get_current_player_id()
            if cur == engine.bot_user_id:
                await runner.run_bot_turn()
            elif cur == engine.human_user_id and getattr(runner.match, "p1_policy", None) is not None:
                # p1-as-RL (model-vs-model): один ход RL-модели.
                await runner.run_bot_turn(player_id=cur, policy=runner.match.p1_policy)
            else:
                return {"status": "not_bot_turn"}
            await self._drain_bot(runner)
            return {"status": "ok", "is_ended": engine.is_ended,
                    "winner_id": (engine._get_winner_id() if engine.is_ended else None)}

        if name == "surrender":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            resp = await runner.surrender()
            return resp

        if name == "list_battle_groups":
            return {"groups": hub.manager.list_groups()}

        if name == "get_battle_group_status":
            gid = _safe_component(args["group_id"], field="group_id")
            hub.manager.reap_completed(gid)
            m = hub.manager.list_groups()
            for g in m:
                if g["group_id"] == gid:
                    return g
            return {"error": "group not found"}

        if name == "get_battle_group_manifest":
            gid = _safe_component(args["group_id"], field="group_id")
            path = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "manifest.json",
            )
            if not path.exists():
                return {"error": "group not found"}
            return json.loads(path.read_text(encoding="utf-8"))

        if name == "get_dataset":
            gid = _safe_component(args["group_id"], field="group_id")
            gdir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
            )
            if not gdir.exists():
                return {"error": "group not found"}
            dataset = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "dataset.jsonl",
            )
            battles_dir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "battles",
            )
            per_battle = (
                sorted(
                    str(path)
                    for path in battles_dir.glob("*.jsonl")
                    if path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(
                        battles_dir.resolve()
                    )
                )
                if battles_dir.exists()
                else []
            )
            return {
                "group_id": gid,
                "dataset_jsonl": str(dataset),
                "dataset_exists": dataset.exists(),
                "dataset_rows": sum(1 for _ in dataset.open()) if dataset.exists() else 0,
                "per_battle_jsonl": per_battle,
            }

        if name == "list_battle_manifests":
            gid = _safe_component(args["group_id"], field="group_id")
            gdir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "battles",
            )
            if not gdir.exists():
                return {"error": "group not found"}
            return {
                "battles": sorted(
                    str(path)
                    for path in gdir.glob("*.json")
                    if path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(gdir.resolve())
                )
            }

        if name == "download_battle_logs":
            gid = _safe_component(args["group_id"], field="group_id")
            fmt = args.get("format", "json")
            gdir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
            )
            if not gdir.exists():
                return {"error": "group not found"}
            if fmt == "zip":
                zip_path = _safe_session_path(
                    hub.manager.sessions_dir,
                    (f"{gid}.zip", "archive_name"),
                )
                with tempfile.NamedTemporaryFile(
                    dir=hub.manager.sessions_dir,
                    prefix=f".{gid}.",
                    suffix=".zip.tmp",
                    delete=False,
                ) as temporary_handle:
                    temporary_zip = Path(temporary_handle.name)
                    os.fchmod(temporary_handle.fileno(), 0o600)
                try:
                    with zipfile.ZipFile(
                        temporary_zip,
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as archive:
                        for file_path in gdir.rglob("*"):
                            if file_path.is_symlink() or not file_path.is_file():
                                continue
                            file_path.resolve().relative_to(gdir.resolve())
                            archive.write(
                                file_path,
                                arcname=file_path.relative_to(gdir),
                            )
                    os.replace(temporary_zip, zip_path)
                    zip_path.chmod(0o600)
                except Exception:
                    temporary_zip.unlink(missing_ok=True)
                    raise
                return {"path": str(zip_path), "size": zip_path.stat().st_size, "format": "zip"}
            return {"path": str(gdir), "format": "json",
                    "files": sorted(
                        str(path.relative_to(hub.manager.sessions_dir))
                        for path in gdir.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    )}

        if name == "list_models":
            return {"models": self.registry.list_specs()}

        if name == "list_active_series":
            # Self-heal: освободить агентов завершённых серий, до которых клиент
            # не дошёл next_battle/finish_series (иначе утекают в agents_index).
            hub.manager.reap_all_completed()
            groups = hub.manager.list_groups()
            # (a) по агенту + (d) «общая картинка» серии; (b) по модели-оппоненту.
            agents: List[Dict[str, Any]] = []
            by_model: Dict[str, Dict[str, Any]] = {}
            for g in groups:
                ag = g.get("agent_name")
                agents.append({
                    "agent_name": ag,
                    "group_id": g["group_id"],
                    "status": g.get("status"),
                    "battles": f"{g.get('battles_finished',0)}/{g.get('battles_planned',0)}",
                    "wins": g.get("wins", 0),
                    "losses": g.get("losses", 0),
                    "draws": g.get("draws", 0),
                    "p1_actor_type": g.get("p1_actor_type"),
                    "opponent_model": g.get("p2_model"),
                    "current_match_id": g.get("current_match_id"),
                })
                mkey = str(g.get("p2_model") or "random")
                bm = by_model.setdefault(mkey, {"model": mkey, "groups": 0, "wins": 0, "losses": 0, "draws": 0})
                bm["groups"] += 1
                bm["wins"] += g.get("wins", 0)
                bm["losses"] += g.get("losses", 0)
                bm["draws"] += g.get("draws", 0)
            running = [g for g in groups if g.get("status") == "running"]
            return {
                "count": len(running),
                "agents": agents,
                "by_model": list(by_model.values()),
            }

        if name == "get_agent_status":
            st = hub.manager.agent_registry.status(args["agent_name"])
            # Self-heal: если серия агента доиграна, а клиент не звал next_battle/
            # finish_series — освободим имя через manager.reap_completed (in-process).
            gid = st.get("group_id")
            if gid:
                hub.manager.reap_completed(gid)
                st = hub.manager.agent_registry.status(args["agent_name"])
            return st

        if name == "get_match_status":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            # Self-heal: бой завершён и серия доиграна → освободить агента сейчас,
            # не ждать next_battle/finish_series (фикс утечки codename в agents_index).
            hub.manager.reap_completed(match.group_id)
            engine = match.engine
            uid = engine.human_user_id
            return {
                "match_id": engine.match_id,
                "group_id": match.group_id,
                "battle_id": match.battle_id,
                "agent_name": getattr(match, "agent_name", None),
                "turn": getattr(engine, "turn", None),
                "is_ended": engine.is_ended,
                "winner_id": (engine._get_winner_id() if engine.is_ended else None),
                "current_player_id": engine.get_current_player_id(),
                # Workflow-B Issue #2: на завершённой игре ничей ход — не тянем
                # current_player_id (после surrender он мог остаться human).
                "is_my_turn": (not engine.is_ended and engine.get_current_player_id() == uid),
                "p1_actor_type": engine.p1_actor_type,
                "battle_tag": engine.battle_tag,
            }

        if name == "get_action_history":
            runner = await hub.get_runner(args["match_id"])
            if runner is None:
                return {"error": "match_not_found"}
            actions = runner.battle_log.get("actions", [])
            limit = int(args.get("limit", 200) or 200)
            return {"actions": actions[-limit:], "count": len(actions)}

        if name == "finish_series":
            gid = _safe_component(args["group_id"], field="group_id")
            live = hub.manager._groups.get(gid)
            if live is None:
                return {"error": "group not found"}
            return hub.manager.finish_series(gid)

        if name == "list_preset_decks":
            # Preset-колоды (ArenaENV Random / JSON-imported) хранятся в прод-БД;
            # headless-среда без БД их не имеет. deck_strategy='random_arenaenv'
            # использует рандом-генератор движка, 'custom' — custom_deck_* из spec.
            return {"presets": [], "note": "preset decks require the prod DB; use deck_strategy='random_arenaenv' or 'custom' with custom_deck_* in headless env"}

        if name == "register_custom_model":
            from rlhf_env.components.policy_adapters import default_registry
            from rlhf_env.components.policy_registry import ModelSpec
            mname = str(args["name"]).strip()
            path = str(args["path"])
            kind = args.get("kind", "auto")
            # path-traversal защита: путь должен лежать под models_dir или корнем репо.
            safe = hub.manager._safe_model_path(path)
            reg = default_registry()
            # F7: грузим sidecar (.onnx.json) перед detect_kind — как scan_directory.
            # Иначе пользовательские V5-детекторы, читающие sidecar (obs_dim/inputs),
            # получают None и не могут определить kind для auto-registered модели.
            detected = None
            if kind == "auto":
                try:
                    from rlhf_env.components.policy_factory import _load_sidecar
                    sidecar = _load_sidecar(safe)
                    detected = reg.detect_kind(str(safe), sidecar, name=mname)
                except Exception:  # noqa: BLE001
                    detected = None
            # 'auto' (а не 'unknown') если детект не сработал — пусть build's own
            # auto-detect branch (он тоже грузит sidecar) дожмёт на start_series.
            final_kind = kind if kind != "auto" else (detected or "auto")
            self.registry.add_spec(ModelSpec(
                name=mname, path=str(safe), sidecar_path=None, kind=final_kind,
                description="custom (registered via MCP)",
            ))
            return {"registered": True, "name": mname, "path": str(safe),
                    "kind": final_kind, "detected_kind": detected}

        # --- V5 training orchestrator --------------------------------------

        if name == "get_v5_dataset_summary":
            gid = _safe_component(args["group_id"], field="group_id")
            gdir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
            )
            if not gdir.exists():
                return {"error": "group not found"}
            man = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "manifest.json",
            )
            if not man.exists():
                return {"error": "manifest not found"}
            m = json.loads(man.read_text(encoding="utf-8"))
            results = m.get("battles_results", []) or []
            tag_dist: Dict[str, int] = {}
            v5_ok = 0
            manifest_v5_ok = 0
            turns_total = 0
            actions_total = 0
            accepted_training_rows = 0
            rejected_audit_rows = 0
            timestamp_contract_battles = 0
            metronome_contract_battles = 0
            metronome_observed_labels = 0
            expected_catalog_hash = hub.manager.catalog_hash
            expected_card_count = len(hub.manager.catalog.cards)
            for r in results:
                t = r.get("battle_tag")
                if t:
                    tag_dist[t] = tag_dist.get(t, 0) + 1
                if r.get("v5_trace_ok"):
                    manifest_v5_ok += 1
                battle_id = _safe_component(
                    r["battle_id"],
                    field="battle_id",
                )
                v5dir = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (battle_id, "battle_id"),
                    "v5",
                )
                battle_log = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (f"{battle_id}.json", "battle_log"),
                )
                report = validate_v5_trace(
                    v5dir,
                    battle_log_path=battle_log,
                    expected_catalog_hash=expected_catalog_hash,
                    expected_card_count=expected_card_count,
                )
                if report["ok"]:
                    v5_ok += 1
                auxiliary = _headless_auxiliary_evidence(v5dir)
                timestamp_contract_battles += int(
                    auxiliary["timestamp_contract_ok"]
                )
                metronome_contract_battles += int(
                    auxiliary["metronome_contract_ok"]
                )
                metronome_observed_labels += int(
                    auxiliary["metronome_observed_labels"]
                )
                tpath = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (battle_id, "battle_id"),
                    "v5",
                    "turns.jsonl",
                )
                apath = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (battle_id, "battle_id"),
                    "v5",
                    "actions.jsonl",
                )
                if tpath.exists():
                    with tpath.open("r", encoding="utf-8") as handle:
                        turns_total += sum(1 for raw in handle if raw.strip())
                if apath.exists():
                    with apath.open("r", encoding="utf-8") as handle:
                        for raw in handle:
                            if not raw.strip():
                                continue
                            actions_total += 1
                            action_row = json.loads(raw)
                            if (
                                action_row.get("accepted") is True
                                and action_row.get("legal_action_index")
                                is not None
                            ):
                                accepted_training_rows += 1
                            elif action_row.get("accepted") is not True:
                                rejected_audit_rows += 1
            degraded_battles = sum(
                bool(row.get("degraded") or row.get("policy_warnings"))
                for row in results
            )
            v5_policy_training_ready = bool(
                results
                and v5_ok == len(results)
                and manifest_v5_ok == len(results)
                and degraded_battles == 0
                and accepted_training_rows > 0
            )
            return {
                "group_id": gid,
                "group_dir": str(gdir),
                "battle_ids": [
                    str(row.get("battle_id"))
                    for row in results
                    if row.get("battle_id")
                ],
                "battles": [
                    {
                        "battle_id": row.get("battle_id"),
                        "policy_warnings": list(
                            row.get("policy_warnings") or []
                        ),
                        "degraded": bool(row.get("degraded")),
                        "v5_trace_ok": bool(row.get("v5_trace_ok")),
                    }
                    for row in results
                ],
                "battles_finished": len(results),
                "battle_ids": [str(r.get("battle_id")) for r in results if r.get("battle_id")],
                "battles": [
                    {
                        "battle_id": r.get("battle_id"),
                        "policy_warnings": list(r.get("policy_warnings") or []),
                        "degraded": bool(r.get("degraded")),
                        "v5_trace_ok": bool(r.get("v5_trace_ok")),
                    }
                    for r in results
                ],
                "v5_trace_ok_count": v5_ok,
                "manifest_v5_trace_ok_count": manifest_v5_ok,
                "deep_validated": True,
                "current_catalog_hash": expected_catalog_hash,
                "current_card_count": expected_card_count,
                "battle_tag_distribution": tag_dist,
                "turns_total": turns_total,
                "actions_total": actions_total,
                "accepted_training_rows": accepted_training_rows,
                "rejected_audit_rows": rejected_audit_rows,
                "training_ready": v5_policy_training_ready,
                "training_ready_scope": "v5_policy_only",
                "v5_policy_training_ready": (
                    v5_policy_training_ready
                ),
                "metronome_training_ready": bool(
                    results
                    and metronome_contract_battles == len(results)
                    and metronome_observed_labels > 0
                ),
                # Headless runtime timestamps measure local execution/LLM
                # latency, not real player battle duration. Keep contract
                # diagnostics visible but never promote them to a TimeStamp
                # training label.
                "timestamp_training_ready": False,
                "metronome_observed_labels": (
                    metronome_observed_labels
                ),
                "metronome_contract_battles": (
                    metronome_contract_battles
                ),
                "timestamp_contract_battles": (
                    timestamp_contract_battles
                ),
                "rows": accepted_training_rows,
                "quality": _semi_synthetic_quality(gdir, results),
            }

        if name == "list_v5_groups":
            tag_filter = args.get("battle_tag")
            limit = int(args.get("limit", 100) or 100)
            out: List[Dict[str, Any]] = []
            quality_reports: List[Dict[str, Any]] = []
            sdir = hub.manager.sessions_dir
            if sdir.exists():
                for gdir in sorted(sdir.iterdir()):
                    if gdir.is_symlink() or not gdir.is_dir():
                        continue
                    try:
                        man = _safe_session_path(
                            sdir,
                            (gdir.name, "group_id"),
                            "manifest.json",
                        )
                    except ValueError:
                        continue
                    if not man.exists():
                        continue
                    try:
                        m = json.loads(man.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        continue
                    results = m.get("battles_results", []) or []
                    tags = {r.get("battle_tag") for r in results if r.get("battle_tag")}
                    if tag_filter and tag_filter not in tags:
                        continue
                    quality_reports.append(
                        _semi_synthetic_quality(gdir, results)
                    )
                    out.append({
                        "group_id": gdir.name,
                        "agent_name": m.get("agent_name"),
                        "battles_finished": len(results),
                        "battle_tags": sorted(t for t in tags if t),
                        "v5_trace_ok_count": sum(1 for r in results if r.get("v5_trace_ok")),
                        "finished_at": m.get("finished_at"),
                    })
                    if len(out) >= limit:
                        break
            return {
                "groups": out,
                "quality": _aggregate_quality_reports(quality_reports),
            }

        if name == "get_v5_trace":
            gid = _safe_component(args["group_id"], field="group_id")
            bid = _safe_component(args["battle_id"], field="battle_id")
            what = args["what"]
            v5dir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "battles",
                (bid, "battle_id"),
                "v5",
            )
            if not v5dir.exists():
                return {"error": "v5 trace not found", "path": str(v5dir)}
            fname = {"meta": "meta.json", "turns": "turns.jsonl", "actions": "actions.jsonl"}[what]
            fpath = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "battles",
                (bid, "battle_id"),
                "v5",
                fname,
            )
            if not fpath.exists():
                return {"error": f"{fname} not found", "path": str(fpath)}
            if fname.endswith(".json"):
                return {"data": json.loads(fpath.read_text(encoding="utf-8")), "rows_count": 1}
            offset = max(0, int(args.get("offset", 0) or 0))
            limit = max(1, min(int(args.get("limit", 200) or 200), 2000))
            rows: List[Dict[str, Any]] = []
            total_rows = 0
            with fpath.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    if offset <= total_rows < offset + limit:
                        rows.append(json.loads(raw))
                    total_rows += 1
            return {
                "data": rows,
                "rows_count": len(rows),
                "total_rows": total_rows,
                "offset": offset,
                "limit": limit,
                "truncated": offset + len(rows) < total_rows,
            }

        if name == "validate_v5_traces":
            gid = _safe_component(args["group_id"], field="group_id")
            gdir = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
            )
            if not gdir.exists():
                return {"error": "group not found"}
            man = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "manifest.json",
            )
            m = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {"battles_results": []}
            results = m.get("battles_results", []) or []
            checked = 0
            ok = 0
            accepted_training_rows = 0
            timestamp_contract_battles = 0
            metronome_contract_battles = 0
            metronome_observed_labels = 0
            broken: List[Dict[str, Any]] = []
            for r in results:
                checked += 1
                bid = _safe_component(r["battle_id"], field="battle_id")
                v5dir = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (bid, "battle_id"),
                    "v5",
                )
                # battle_log b_<bid>.json лежит рядом с v5/-директорией боя; путь
                # строится от того же battle_id (уже содержит префикс b_).
                battle_log = _safe_session_path(
                    hub.manager.sessions_dir,
                    (gid, "group_id"),
                    "battles",
                    (f"{bid}.json", "battle_log"),
                )
                rep = validate_v5_trace(
                    v5dir,
                    battle_log_path=battle_log,
                    expected_catalog_hash=hub.manager.catalog_hash,
                    expected_card_count=len(hub.manager.catalog.cards),
                )
                if rep["ok"]:
                    ok += 1
                else:
                    broken.append({"battle_id": bid, "issues": rep["issues"]})
                accepted_training_rows += int(
                    (rep.get("checks") or {}).get(
                        "accepted_training_rows",
                        0,
                    )
                    or 0
                )
                auxiliary = _headless_auxiliary_evidence(v5dir)
                timestamp_contract_battles += int(
                    auxiliary["timestamp_contract_ok"]
                )
                metronome_contract_battles += int(
                    auxiliary["metronome_contract_ok"]
                )
                metronome_observed_labels += int(
                    auxiliary["metronome_observed_labels"]
                )
            degraded = [
                {
                    "battle_id": row.get("battle_id"),
                    "policy_warnings": list(
                        row.get("policy_warnings") or []
                    ),
                }
                for row in results
                if row.get("degraded") or row.get("policy_warnings")
            ]
            v5_policy_training_ready = bool(
                checked > 0
                and ok == checked
                and all(
                    row.get("v5_trace_ok") is True
                    for row in results
                )
                and not degraded
                and accepted_training_rows > 0
            )
            return {
                "checked": checked,
                "ok": ok,
                "broken": broken,
                "degraded": degraded,
                "accepted_training_rows": accepted_training_rows,
                "training_ready": v5_policy_training_ready,
                "training_ready_scope": "v5_policy_only",
                "v5_policy_training_ready": (
                    v5_policy_training_ready
                ),
                "metronome_training_ready": bool(
                    checked > 0
                    and metronome_contract_battles == checked
                    and metronome_observed_labels > 0
                ),
                # Headless wall-clock data is diagnostic only.
                "timestamp_training_ready": False,
                "metronome_observed_labels": (
                    metronome_observed_labels
                ),
                "metronome_contract_battles": (
                    metronome_contract_battles
                ),
                "timestamp_contract_battles": (
                    timestamp_contract_battles
                ),
                "current_catalog_hash": hub.manager.catalog_hash,
                "current_card_count": len(hub.manager.catalog.cards),
            }

        # --- Cross-contour dataset administration --------------------------

        toolbox = self.dataset_toolbox
        if name == "get_training_data_status":
            status = toolbox.status()
            groups = hub.manager.list_groups()
            status["headless_groups"] = len(groups)
            status["headless_finished_battles"] = sum(
                int(group.get("battles_finished", 0) or 0)
                for group in groups
            )
            return status

        if name == "list_training_exports":
            return toolbox.list_artifacts(
                kind=args.get("kind"),
                limit=int(args.get("limit", 200) or 200),
            )

        if name == "inspect_training_export":
            return toolbox.inspect_artifact(args["path"])

        if name == "validate_training_export":
            return toolbox.validate_artifact(args["path"])

        if name == "export_v5_training_dataset":
            return await toolbox.export_production_v5(
                output=args["output"],
                days=int(args.get("days", 30) or 30),
                limit_battles=int(
                    args.get("limit_battles", 1000) or 1000
                ),
                overwrite=bool(args.get("overwrite", False)),
            )

        if name == "materialize_v5_training_dataset":
            return toolbox.materialize_v5(
                input_path=args["input_path"],
                output_dir=args["output_dir"],
                group_id=args.get("group_id"),
                overwrite=bool(args.get("overwrite", False)),
            )

        if name == "export_nemesis_training_dataset":
            input_path = args.get("input_path")
            group_id = args.get("group_id")
            if bool(input_path) == bool(group_id):
                raise ValueError(
                    "provide exactly one of input_path or group_id"
                )
            if input_path:
                return toolbox.export_nemesis(
                    input_path=input_path,
                    output=args["output"],
                    overwrite=bool(args.get("overwrite", False)),
                )

            gid = _safe_component(group_id, field="group_id")
            manifest_path = _safe_session_path(
                hub.manager.sessions_dir,
                (gid, "group_id"),
                "manifest.json",
            )
            if not manifest_path.is_file():
                raise ValueError("headless group manifest not found")
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not manifest.get("finished_at"):
                raise ValueError(
                    "headless group must be finalized before export"
                )
            results = manifest.get("battles_results")
            if not isinstance(results, list) or not results:
                raise ValueError(
                    "headless group has no completed battle results"
                )
            planned = int(
                (manifest.get("spec") or {}).get("battles_planned", 0)
                or 0
            )
            if planned <= 0 or len(results) != planned:
                raise ValueError(
                    "headless group is incomplete"
                )

            def group_records():
                for result in results:
                    if not isinstance(result, dict):
                        raise ValueError(
                            "headless group result must be an object"
                        )
                    battle_id = _safe_component(
                        result.get("battle_id"),
                        field="battle_id",
                    )
                    if (
                        result.get("degraded") is True
                        or result.get("policy_warnings")
                    ):
                        raise ValueError(
                            f"degraded policy result for battle {battle_id}"
                        )
                    if result.get("v5_trace_ok") is not True:
                        raise ValueError(
                            f"V5 trace is not valid for battle {battle_id}"
                        )
                    v5_dir = _safe_session_path(
                        hub.manager.sessions_dir,
                        (gid, "group_id"),
                        "battles",
                        (battle_id, "battle_id"),
                        "v5",
                    )
                    battle_log = _safe_session_path(
                        hub.manager.sessions_dir,
                        (gid, "group_id"),
                        "battles",
                        (f"{battle_id}.json", "battle_log"),
                    )
                    trace_report = validate_v5_trace(
                        v5_dir,
                        battle_log_path=battle_log,
                        expected_catalog_hash=hub.manager.catalog_hash,
                        expected_card_count=len(
                            hub.manager.catalog.cards
                        ),
                    )
                    if not trace_report.get("ok"):
                        raise ValueError(
                            "invalid V5 policy trace for battle "
                            f"{battle_id}: "
                            + "; ".join(
                                trace_report.get("issues") or []
                            )
                        )
                    if int(
                        (trace_report.get("checks") or {}).get(
                            "accepted_training_rows",
                            0,
                        )
                        or 0
                    ) <= 0:
                        raise ValueError(
                            "V5 trace has no accepted training rows for "
                            f"battle {battle_id}"
                        )
                    meta_path = _safe_session_path(
                        v5_dir,
                        "meta.json",
                    )
                    if not meta_path.is_file():
                        raise ValueError(
                            f"Nemesis meta missing for battle {battle_id}"
                        )
                    meta = json.loads(
                        meta_path.read_text(encoding="utf-8")
                    )
                    record = meta.get("nemesis_record")
                    if not isinstance(record, dict):
                        raise ValueError(
                            "canonical Nemesis record missing for battle "
                            f"{battle_id}"
                        )
                    yield record

            return toolbox.export_nemesis_records(
                records=group_records(),
                source_label=f"headless-group:{gid}",
                output=args["output"],
                overwrite=bool(args.get("overwrite", False)),
            )

        if name == "export_returnclock_training_dataset":
            return await toolbox.export_returnclock(
                output=args["output"],
                start=args.get("start"),
                end=args.get("end"),
                horizon_hours=int(
                    args.get("horizon_hours", 7 * 24) or 7 * 24
                ),
                safety_lag_minutes=int(
                    args.get("safety_lag_minutes", 10)
                    if args.get("safety_lag_minutes") is not None
                    else 10
                ),
                min_analytics_version=int(
                    args.get("min_analytics_version", 2) or 2
                ),
                limit=int(args.get("limit", 50_000) or 50_000),
                overwrite=bool(args.get("overwrite", False)),
            )

        if name == "split_nemesis_training_dataset":
            return toolbox.split_nemesis(
                source=args["source"],
                output_dir=args["output_dir"],
                train_fraction=float(
                    args.get("train_fraction", 0.70)
                ),
                validation_fraction=float(
                    args.get("validation_fraction", 0.15)
                ),
                overwrite=bool(args.get("overwrite", False)),
            )

        if name == "split_returnclock_training_dataset":
            return toolbox.split_returnclock(
                source=args["source"],
                output_dir=args["output_dir"],
                train_fraction=float(
                    args.get("train_fraction", 0.70)
                ),
                validation_fraction=float(
                    args.get("validation_fraction", 0.15)
                ),
                overwrite=bool(args.get("overwrite", False)),
            )

        raise ValueError(f"unknown tool: {name}")

    async def _run_auto_bounded(self, runner: MatchRunner) -> bool:
        """run_auto с защитой от зависания (F4): кроме turn-cap внутри run_auto,
        оборачиваем в asyncio.wait_for — если auto-play не укладывается в таймаут
        (напр. обе политики сломаны, длинный rl-vs-rl с реальными delay'ами),
        stdio-вызов не виснет. Возвращает True если auto-play завершился штатно,
        False по таймауту (бой остался незавершённым — оркестратор может
        finish_series)."""
        try:
            await asyncio.wait_for(asyncio.shield(runner.run_auto()), timeout=120.0)
            return True
        except asyncio.TimeoutError:
            logger.warning("run_auto timed out match=%s — leaving battle unfinished", runner.match.engine.match_id)
            runner.policy_fallbacks.append("run_auto timed out (120s) — battle left unfinished; use finish_series to close")
            return False
        except Exception:  # noqa: BLE001
            logger.warning("run_auto crashed: %s", exc_info=True)
            return False

    async def _drain_bot(self, runner: MatchRunner) -> None:
        """Дождаться завершения запланированной бот-рутины (если была)."""
        task = runner._bot_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("bot task timed out while draining")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------
    async def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rlhf-env", "version": __version__},
                "capabilities": {"tools": {}},
            }
        if method == "tools/list":
            return {"tools": self.tools}
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            try:
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be an object")
                tool_spec = next(
                    (
                        tool
                        for tool in self.tools
                        if tool.get("name") == name
                    ),
                    None,
                )
                if tool_spec is not None:
                    schema = tool_spec.get("inputSchema") or {}
                    if schema.get("additionalProperties") is False:
                        _validate_tool_arguments(args, schema)
                result = await self._tool(name, args)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    "structuredContent": result,
                    "isError": False,
                }
            except Exception as exc:  # noqa: BLE001
                error = _public_tool_error(exc)
                tool_label = (
                    str(name)
                    if isinstance(name, str)
                    and _SAFE_COMPONENT_RE.fullmatch(name)
                    else "<unknown>"
                )
                if isinstance(exc, ValueError):
                    logger.warning(
                        "[mcp] tool %s rejected request (%s)",
                        tool_label,
                        type(exc).__name__,
                    )
                else:
                    logger.error(
                        "[mcp] tool %s failed (%s)",
                        tool_label,
                        type(exc).__name__,
                    )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                error,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    "structuredContent": error,
                    "isError": True,
                }
        return {"error": {"code": -32601, "message": f"unknown method: {method}"}}


# ============================================================================
# Async stdio loop (единый event-loop — MatchRunner lock/tasks живы между вызовами)
# ============================================================================

async def _amain(server: MCPServer) -> None:
    """Stdio JSON-RPC loop. Один запрос — один ответ (строго 1:1, без unsolicited-
    banner'ов, чтобы не ломать MCP-клиенты и синхронные stdio-харнессы)."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"parse error: {exc}",
                        },
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
            continue
        if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id") if isinstance(msg, dict) else None,
                        "error": {
                            "code": -32600,
                            "message": "invalid request",
                        },
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
            continue
        is_notification = "id" not in msg
        try:
            result = await server.dispatch(
                msg["method"],
                msg.get("params", {}) or {},
            )
            if is_notification:
                continue
            req_id = msg.get("id")
            if (
                isinstance(result, dict)
                and set(result) == {"error"}
                and isinstance(result["error"], dict)
                and "code" in result["error"]
            ):
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": result["error"],
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result,
                }
            sys.stdout.write(json.dumps(response) + "\n")
        except Exception as exc:  # noqa: BLE001
            if is_notification:
                logger.error(
                    "[mcp] notification failed (%s)",
                    type(exc).__name__,
                )
                continue
            logger.error(
                "[mcp] request failed (%s)",
                type(exc).__name__,
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "error": {
                            "code": -32603,
                            "message": "internal server error",
                        },
                    }
                )
                + "\n"
            )
        sys.stdout.flush()


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-сервер RLHF-среды (stdio)")
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", "ai/models"))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", "rlhf_env/sessions"))
    p.add_argument(
        "--datasets-dir",
        default=os.environ.get("RLHF_DATASETS_DIR", "datasets"),
    )
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", "ai/cards.json"))
    p.add_argument(
        "--enable-production-datasets",
        action="store_true",
        default=str(
            os.environ.get("RLHF_ENABLE_PRODUCTION_DATASETS", "")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Enable read-only PostgreSQL exports for V5/ReturnClock. "
            "Disabled by default; local inspect/validate tools still work."
        ),
    )
    p.add_argument(
        "--returnclock-salt-env",
        default=os.environ.get(
            "RLHF_RETURNCLOCK_SALT_ENV",
            "RETURNCLOCK_DATASET_SALT",
        ),
    )
    p.add_argument(
        "--returnclock-salt-key-id-env",
        default=os.environ.get(
            "RLHF_RETURNCLOCK_SALT_KEY_ID_ENV",
            "RETURNCLOCK_DATASET_SALT_KEY_ID",
        ),
    )
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "WARNING"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    registry = PolicyRegistry.scan(args.models_dir)
    hub = HeadlessHub(
        sessions_dir=args.sessions_dir,
        models_dir=args.models_dir,
        cards_path=args.cards_path,
        registry=registry,
    )
    toolbox = DatasetToolbox(
        args.datasets_dir,
        returnclock_salt_env=args.returnclock_salt_env,
        returnclock_salt_key_id_env=args.returnclock_salt_key_id_env,
        production_enabled=args.enable_production_datasets,
        cards_path=args.cards_path,
    )
    server = MCPServer(hub, registry, dataset_toolbox=toolbox)
    logger.info("MCP server starting (stdio). tools=%d, models=%d", len(server.tools), len(registry.specs))
    try:
        asyncio.run(_amain(server))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
