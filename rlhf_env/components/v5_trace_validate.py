"""Глубокая валидация V5-trace инвариантов обучающих данных.

Поверхностная проверка (наличие + непустота ``meta.json``/``turns.jsonl``/
``actions.jsonl``) говорит только «файлы есть». Для offline-обучения этого
недостаточно: нужен строгий набор инвариантов, нарушение которых портит метку
или разрывает реконструкцию состояния. Этот модуль проверяет четыре класса:

  1. **legal_action_index** — индекс выбранного действия валиден в
     ``legal_actions`` (``0..N-1``); ``action_native == legal_actions[idx]``
     (метка указывает ровно на то действие, что сыграно); ``None`` на
     ``accepted``-действии = потеря обучающей метки.
  2. **actor / decision_source** — ``actor_player`` ⇔ ``actor_user_id`` ⇔
     ``meta.{p1,p2}_user_id``; ``decision_source`` согласован с
     ``meta.{p1,p2}_actor_type``; ``pre_state.current_turn_owner_id ==
     actor_user_id`` (действие только в свой ход).
  3. **continuity** — ``post_state`` строки N == ``pre_state`` строки N+1
     (состояние передаётся по цепочке без потерь/вставок); терминальный
     ``post_state`` согласован с ``meta.status``/``meta.winner_user_id``;
     ``turns.jsonl`` ↦ ``actions.jsonl`` на границе хода.
  4. **correspondence** — ``battle_log`` ``b_<bid>.json`` ``actions`` 1:1 (по
     порядку) совпадают с trace-строками (``actor``/``type``/``action_json``/
     ``accepted``/``turn``), кроме терминальных ``surrender``-строк, у которых
     нет battle_log-записи (сдача логируется только в v5-trace).

Универсально относительно версии модели: проверяется storage-surface
(``actions.jsonl`` schema + state-continuity + ``battle_log`` correspondence),
БЕЗ зависимости от ``classic_*`` кодеков / tcode / ``encode_observation_v5``.
Инварианты справедливы для любого актора (``human``/``llm``/``bot``/``rl``) и
любого ``kind`` модели — поэтому валидатор не хардкодит V5-специфику.

Используется MCP-инструментом ``validate_v5_traces`` (см. ``mcp_server.py``),
но сам по себе — чистая функция над путями на диске (тестируется in-process).
"""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Терминальные action_type (из v5_trace.record_terminal): surrender (и зарезервированные
# draw/stalemate). Дискриминатор терминальной строки — action_type (авторитетный маркер),
# НЕ форма legal_actions: обычное действие с очищенным legal-набором (потерянная метка)
# не должно классифицироваться как терминал (иначе lost-label-чек и correspondence-фильтр
# его пропустят).
_TERMINAL_TYPES = {"surrender", "draw", "stalemate"}

# Терминальные значения state.status / meta.status (core-enum'ы в lowercase).
_TERMINAL_STATUSES = {"p1_win", "p2_win", "draw", "stalemate"}

# Metronome V1 training window. Raw values outside it are still valuable audit
# data, but the row must say explicitly that the label is censored.
_METRONOME_MIN_LABEL_MS = 100
_METRONOME_MAX_LABEL_MS = 25_000

# The values serialized by ``core.state.ReplacementStatus``.  Control-plane
# transitions are the only supported exception to byte-for-byte state
# continuity, so accepting an unknown value here would silently widen that
# exception.
_REPLACEMENT_STATUSES = {"active", "afk", "surrendered"}

# Поля meta.json, обязательные для глубоких cross-meta инвариантов (actor/source,
# terminal↔meta). Реальный V5TraceRecorder._write_meta всегда их пишет; отсутствие =
# порча meta, и source-consistency-чек не должен молча пропускаться.
_REQUIRED_META_FIELDS = (
    "p1_user_id",
    "p2_user_id",
    "p1_actor_type",
    "p2_actor_type",
    "p1_is_bot",
    "p2_is_bot",
    "status",
)

# Поля action-строки, обязательные для любой (в т.ч. терминальной) строки.
_REQUIRED_ACTION_FIELDS = (
    "seq",
    "turn_number",
    "actor_user_id",
    "actor_player",
    "decision_source",
    "control_source",
    "human_decision_time_ms",
    "decision_time_censored",
    "decision_censor_reason",
    "legal_action_index",
    "action_type",
    "action_json",
    "legal_actions",
    "legal_action_count",
    "pre_state",
    "post_state",
    "accepted",
    "error",
    "timestamp_ms",
)

_VALID_DECISION_SOURCES = {"human", "llm", "bot", "rl"}

# Top-level ключи state-snapshot, сравниваемые при continuity-проверке (на случай
# если понадобится локализовать расхождение; само сравнение — полное ==).
_STATE_KEYS = (
    "turn_number",
    "current_turn_owner_id",
    "status",
    "p1",
    "p2",
    "action_history",
    "history",
    "v5_history_events",
    "pending_card_feedback_events",
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Читает .jsonl в список строк. Возвращает (rows, error): при отсутствии
    файла — (None, None); при пустом — ([], None); при ломаной строке —
    (None, 'invalid jsonl: ...')."""
    if not path.exists():
        return None, None
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for ln, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            return None, f"invalid jsonl line {ln}: {exc}"
    return rows, None


def _is_terminal(row: Dict[str, Any]) -> bool:
    """Терминальная строка (record_terminal): action_type ∈ {surrender,draw,stalemate}.
    Авторитетный маркер — action_type, а не форма legal_actions: обычное действие
    (play_card/attack/end_turn/mana_draw) с очищенным legal-набором = потерянная
    метка, НЕ терминал, и должно пройти lost-label-чек + correspondence-фильтр."""
    return row.get("action_type") in _TERMINAL_TYPES


def _state_diff_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Какие top-level ключи state-snapshot различаются (для читаемого issue)."""
    return [k for k in _STATE_KEYS if a.get(k) != b.get(k)]


def _index_control_events(
    meta: Dict[str, Any],
    *,
    max_action_seq: int,
    issues: List[str],
) -> Dict[int, List[Dict[str, Any]]]:
    """Validate and group audited control changes by action boundary.

    ``after_action_seq=N`` means that the event happened after action N and
    before action N+1.  Invalid events are never returned to the continuity
    reconciler: malformed audit metadata must not authorize a state mutation.
    """

    raw_events = meta.get("control_events")
    if raw_events is None:
        return {}
    if not isinstance(raw_events, list):
        issues.append(
            "[continuity] meta.control_events must be a list when present"
        )
        return {}

    p1_uid = meta.get("p1_user_id")
    p2_uid = meta.get("p2_user_id")
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for position, event in enumerate(raw_events, 1):
        if not isinstance(event, dict):
            issues.append(
                f"[continuity] control_event row {position}: expected object, "
                f"got {type(event).__name__}"
            )
            continue

        event_label = event.get("seq", position)
        valid = True
        if (
            not isinstance(event.get("seq"), int)
            or isinstance(event.get("seq"), bool)
            or event.get("seq") != position
        ):
            issues.append(
                f"[continuity] control_event row {position}: seq="
                f"{event.get('seq')!r} is not contiguous (expected {position})"
            )
            valid = False
        event_battle_id = event.get("battle_id")
        meta_battle_id = meta.get("battle_id")
        if (
            event_battle_id is not None
            and str(event_battle_id) != str(meta_battle_id)
        ):
            issues.append(
                f"[continuity] control_event seq={event_label}: battle_id="
                f"{event_battle_id!r} != meta.battle_id={meta_battle_id!r}"
            )
            valid = False
        missing = [
            field
            for field in (
                "after_action_seq",
                "user_id",
                "previous_status",
                "new_status",
            )
            if field not in event
        ]
        if missing:
            issues.append(
                f"[continuity] control_event seq={event_label}: "
                f"missing fields {missing}"
            )
            continue

        after_seq = event.get("after_action_seq")
        if (
            not isinstance(after_seq, int)
            or isinstance(after_seq, bool)
            or after_seq < 0
            or after_seq > max_action_seq
        ):
            issues.append(
                f"[continuity] control_event seq={event_label}: "
                f"after_action_seq={after_seq!r} outside [0,{max_action_seq}]"
            )
            valid = False

        user_id = event.get("user_id")
        if (
            not isinstance(user_id, int)
            or isinstance(user_id, bool)
            or user_id not in (p1_uid, p2_uid)
        ):
            issues.append(
                f"[continuity] control_event seq={event_label}: "
                f"user_id={user_id!r} is not meta p1/p2 user"
            )
            valid = False

        previous = event.get("previous_status")
        new = event.get("new_status")
        if previous not in _REPLACEMENT_STATUSES:
            issues.append(
                f"[continuity] control_event seq={event_label}: "
                f"previous_status={previous!r} is invalid"
            )
            valid = False
        if new not in _REPLACEMENT_STATUSES:
            issues.append(
                f"[continuity] control_event seq={event_label}: "
                f"new_status={new!r} is invalid"
            )
            valid = False
        if previous == new:
            issues.append(
                f"[continuity] control_event seq={event_label}: no-op "
                f"replacement_status transition {previous!r}→{new!r}"
            )
            valid = False

        if valid:
            grouped.setdefault(after_seq, []).append(event)
    return grouped


def _replay_control_events(
    state: Dict[str, Any],
    events: List[Dict[str, Any]],
    *,
    meta: Dict[str, Any],
    after_action_seq: int,
    issues: List[str],
) -> Dict[str, Any]:
    """Replay only audited ``replacement_status`` mutations onto a copy.

    Every event's user and previous value are checked against the state
    produced by the preceding action/event.  The caller then compares the
    complete replayed snapshot with the next pre-state, so no unrelated field
    difference can be hidden by this exception.
    """

    expected = deepcopy(state)
    p1_uid = meta.get("p1_user_id")
    p2_uid = meta.get("p2_user_id")
    for event in events:
        event_label = event.get("seq", "?")
        user_id = event["user_id"]
        side = "p1" if user_id == p1_uid else "p2" if user_id == p2_uid else None
        player = expected.get(side) if side is not None else None
        if not isinstance(player, dict):
            issues.append(
                f"[continuity] control_event seq={event_label} after action "
                f"{after_action_seq}: state has no player for user_id={user_id}"
            )
            continue
        if player.get("user_id") != user_id:
            issues.append(
                f"[continuity] control_event seq={event_label} after action "
                f"{after_action_seq}: state {side}.user_id={player.get('user_id')!r} "
                f"!= event user_id={user_id}"
            )
            continue

        actual_previous = player.get("replacement_status")
        declared_previous = event["previous_status"]
        if actual_previous != declared_previous:
            issues.append(
                f"[continuity] control_event seq={event_label} after action "
                f"{after_action_seq}: {side}.replacement_status="
                f"{actual_previous!r} != previous_status={declared_previous!r}"
            )
            continue
        player["replacement_status"] = event["new_status"]
    return expected


# ---------------------------------------------------------------------------
# Per-invariant checks
# ---------------------------------------------------------------------------

def _check_legal_action_index(rows: List[Dict[str, Any]], issues: List[str]) -> None:
    """(1) legal_action_index валиден; action_native == legal_actions[idx];
    legal_action_count == len(legal_actions)."""
    for r in rows:
        seq = r.get("seq", "?")
        idx = r.get("legal_action_index")
        legal = r.get("legal_actions") or []
        count = r.get("legal_action_count")
        if count is None or count != len(legal):
            issues.append(
                f"[legal_index] seq={seq}: legal_action_count={count} != "
                f"len(legal_actions)={len(legal)}"
            )
        if _is_terminal(r):
            # терминал: idx должен быть None, legal пуст
            if idx is not None:
                issues.append(
                    f"[legal_index] seq={seq}: terminal row has legal_action_index={idx} "
                    f"(expected None)"
                )
            continue
        accepted = r.get("accepted")
        # отклонённое действие: idx может быть None (не зарезолвили) — допустимо,
        # но если индекс всё же задан — он должен быть в диапазоне.
        if idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool):
                issues.append(f"[legal_index] seq={seq}: legal_action_index not int ({idx!r})")
                continue
            if idx < 0 or idx >= len(legal):
                issues.append(
                    f"[legal_index] seq={seq}: legal_action_index={idx} out of range "
                    f"[0,{len(legal)})"
                )
                continue
            # action_native должен совпадать с legal_actions[idx] — метка указывает
            # ровно на то действие, что сыграно.
            native = r.get("action_native")
            if native is None:
                issues.append(
                    f"[legal_index] seq={seq}: legal_action_index={idx} set but "
                    f"action_native is None (native action not captured)"
                )
            elif legal and native != legal[idx]:
                issues.append(
                    f"[legal_index] seq={seq}: action_native != legal_actions[{idx}] "
                    f"(label points at wrong action)"
                )
        elif accepted:
            # accepted (truthy), но индекс не зарезолвился → потеря обучающей метки.
            # truthy, а не `is True`: не-bool accepted (порча) тоже должна ловиться
            # здесь как lost-label (дополнительно к schema bool-check).
            issues.append(
                f"[legal_index] seq={seq}: accepted action has legal_action_index=None "
                f"(training label lost — action unresolvable in legal set)"
            )


def _check_actor_source(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], issues: List[str]
) -> None:
    """(2) actor_player ⇔ actor_user_id ⇔ meta user_id; decision_source согласован
    с actor_type; accepted action belongs to the current turn owner.

    A human seat may be controlled temporarily by an automated replacement or
    timeout action. Such rows remain ``decision_source='bot'`` and declare
    ``control_source``; they must never masquerade as human labels.
    """
    p1_uid = meta.get("p1_user_id")
    p2_uid = meta.get("p2_user_id")
    p1_type = meta.get("p1_actor_type")
    p2_type = meta.get("p2_actor_type")
    for player, actor_type in ((1, p1_type), (2, p2_type)):
        if actor_type not in _VALID_DECISION_SOURCES:
            issues.append(
                f"[actor] meta.p{player}_actor_type={actor_type!r} not in "
                f"{sorted(_VALID_DECISION_SOURCES)}"
            )
        expected_is_bot = actor_type in {"bot", "rl", "llm"}
        actual_is_bot = meta.get(f"p{player}_is_bot")
        if not isinstance(actual_is_bot, bool) or actual_is_bot != expected_is_bot:
            issues.append(
                f"[actor] meta.p{player}_is_bot={actual_is_bot!r} inconsistent "
                f"with actor_type={actor_type!r} (expected {expected_is_bot})"
            )
    for r in rows:
        seq = r.get("seq", "?")
        actor_uid = r.get("actor_user_id")
        actor_player = r.get("actor_player")
        src = r.get("decision_source")
        control_source = r.get("control_source")
        if actor_player not in (1, 2):
            issues.append(f"[actor] seq={seq}: actor_player={actor_player!r} not in (1,2)")
        else:
            expected_uid = p1_uid if actor_player == 1 else p2_uid
            if expected_uid is not None and actor_uid != expected_uid:
                issues.append(
                    f"[actor] seq={seq}: actor_player={actor_player} but "
                    f"actor_user_id={actor_uid} != meta.p{actor_player}_user_id={expected_uid}"
                )
        if src not in _VALID_DECISION_SOURCES:
            issues.append(
                f"[actor] seq={seq}: decision_source={src!r} not in "
                f"{sorted(_VALID_DECISION_SOURCES)}"
            )
        elif actor_player in (1, 2):
            expected_type = p1_type if actor_player == 1 else p2_type
            pre = r.get("pre_state") or {}
            player_snapshot = pre.get(f"p{actor_player}") or {}
            replacement_status = str(
                player_snapshot.get("replacement_status", "active") or "active"
            ).lower()
            automated_human_seat = (
                expected_type == "human"
                and src == "bot"
                and (
                    control_source in {"replacement_bot", "timeout"}
                    or replacement_status != "active"
                )
            )
            if expected_type is not None and src != expected_type and not automated_human_seat:
                issues.append(
                    f"[actor] seq={seq}: p{actor_player} decision_source={src!r} != "
                    f"meta.p{actor_player}_actor_type={expected_type!r}"
                )
            if control_source in {"replacement_bot", "timeout"} and src != "bot":
                issues.append(
                    f"[actor] seq={seq}: control_source={control_source!r} "
                    f"requires decision_source='bot'"
                )
            if control_source == "timeout" and r.get("action_type") != "end_turn":
                issues.append(
                    f"[actor] seq={seq}: control_source='timeout' requires "
                    f"action_type='end_turn'"
                )

        # Исполненное действие — только в свой ход. Отклонённая клиентская
        # попытка может легитимно прийти уже во время хода другого игрока; она
        # остаётся в audit trail, но не является обучающей меткой.
        pre = r.get("pre_state") or {}
        owner = pre.get("current_turn_owner_id")
        if r.get("accepted") is True and actor_uid is not None and owner != actor_uid:
            issues.append(
                f"[actor] seq={seq}: pre_state.current_turn_owner_id={owner!r} != "
                f"actor_user_id={actor_uid} (acted out of turn / owner missing)"
            )


def _check_timing_and_outcome(
    rows: List[Dict[str, Any]],
    issues: List[str],
) -> None:
    """Validate the human timing label and accepted/rejected row semantics."""

    previous_timestamp: Optional[int] = None
    for r in rows:
        seq = r.get("seq", "?")
        source = r.get("decision_source")
        latency = r.get("human_decision_time_ms")
        censored = r.get("decision_time_censored")
        reason = r.get("decision_censor_reason")
        timestamp = r.get("timestamp_ms")

        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            issues.append(
                f"[timing] seq={seq}: timestamp_ms={timestamp!r} must be a "
                f"non-negative int"
            )
        elif previous_timestamp is not None and timestamp < previous_timestamp:
            issues.append(
                f"[timing] seq={seq}: timestamp_ms={timestamp} is earlier than "
                f"previous={previous_timestamp}"
            )
        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            previous_timestamp = timestamp

        if not isinstance(censored, bool):
            issues.append(
                f"[timing] seq={seq}: decision_time_censored={censored!r} not bool"
            )
            # Continue checking independent fields, but do not use truthiness as
            # an authoritative censor decision.
            is_censored: Optional[bool] = None
        else:
            is_censored = censored

        if latency is not None and (
            not isinstance(latency, int)
            or isinstance(latency, bool)
            or latency < 0
        ):
            issues.append(
                f"[timing] seq={seq}: human_decision_time_ms={latency!r} must "
                f"be null or a non-negative int"
            )
            valid_latency: Optional[int] = None
        else:
            valid_latency = latency

        if reason is not None and (
            not isinstance(reason, str) or not reason.strip()
        ):
            issues.append(
                f"[timing] seq={seq}: decision_censor_reason={reason!r} must "
                f"be null or a non-empty string"
            )

        if source != "human":
            if latency is not None or censored is not False or reason is not None:
                issues.append(
                    f"[timing] seq={seq}: non-human decision_source={source!r} "
                    f"must use timing null/false/null"
                )
        else:
            if is_censored is True:
                if not isinstance(reason, str) or not reason.strip():
                    issues.append(
                        f"[timing] seq={seq}: censored human timing requires "
                        f"decision_censor_reason"
                    )
            elif is_censored is False:
                if reason is not None:
                    issues.append(
                        f"[timing] seq={seq}: uncensored human timing must not "
                        f"have decision_censor_reason"
                    )
                if valid_latency is None:
                    issues.append(
                        f"[timing] seq={seq}: uncensored human action requires "
                        f"human_decision_time_ms"
                    )

            if (
                valid_latency is not None
                and not (
                    _METRONOME_MIN_LABEL_MS
                    <= valid_latency
                    <= _METRONOME_MAX_LABEL_MS
                )
                and is_censored is not True
            ):
                issues.append(
                    f"[timing] seq={seq}: human_decision_time_ms={valid_latency} "
                    f"outside [{_METRONOME_MIN_LABEL_MS},"
                    f"{_METRONOME_MAX_LABEL_MS}] but row is not censored"
                )

        accepted = r.get("accepted")
        error = r.get("error")
        if accepted is True and error is not None:
            issues.append(
                f"[outcome] seq={seq}: accepted row has error={error!r}"
            )
        elif accepted is False:
            if not isinstance(error, str) or not error.strip():
                issues.append(
                    f"[outcome] seq={seq}: rejected row requires a non-empty error"
                )
            pre = r.get("pre_state")
            post = r.get("post_state")
            if pre is not None and post is not None and pre != post:
                issues.append(
                    f"[outcome] seq={seq}: rejected row mutated state "
                    f"(pre_state != post_state)"
                )

        if _is_terminal(r):
            legal = r.get("legal_actions")
            if accepted is not True:
                issues.append(
                    f"[outcome] seq={seq}: terminal row must be accepted=True"
                )
            if legal != [] or r.get("legal_action_count") != 0:
                issues.append(
                    f"[outcome] seq={seq}: terminal row requires empty legal_actions "
                    f"and legal_action_count=0"
                )
            if r.get("action_native") is not None:
                issues.append(
                    f"[outcome] seq={seq}: terminal row action_native must be None"
                )
            if r.get("source_card") is not None or r.get("target_card") is not None:
                issues.append(
                    f"[outcome] seq={seq}: terminal row source/target cards must be None"
                )


def _check_v5_history_schema(
    rows: List[Dict[str, Any]],
    turns: Optional[List[Dict[str, Any]]],
    issues: List[str],
) -> None:
    """Require the authoritative structured V5 history tape in every snapshot."""

    snapshots: List[Tuple[str, Any]] = []
    for r in rows:
        seq = r.get("seq", "?")
        snapshots.extend(
            (
                (f"seq={seq} pre_state", r.get("pre_state")),
                (f"seq={seq} post_state", r.get("post_state")),
            )
        )
    for index, turn in enumerate(turns or [], 1):
        snapshots.append((f"turns.jsonl row {index}", turn))

    for label, snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if "v5_history_events" not in snapshot:
            issues.append(
                f"[history] {label}: missing v5_history_events"
            )
        elif not isinstance(snapshot.get("v5_history_events"), list):
            issues.append(
                f"[history] {label}: v5_history_events must be a list"
            )


def _check_continuity(
    rows: List[Dict[str, Any]],
    turns: Optional[List[Dict[str, Any]]],
    meta: Dict[str, Any],
    issues: List[str],
) -> None:
    """(3) post_state[N] == pre_state[N+1]; терминал ↔ meta.status/winner;
    turns.jsonl ↦ actions.jsonl на границе хода."""
    if not rows:
        return
    # seq монотонный и континуальный 1..N
    seqs = [r.get("seq") for r in rows]
    for i, s in enumerate(seqs, 1):
        if s != i:
            issues.append(
                f"[continuity] seq not contiguous: position {i} has seq={s!r} "
                f"(expected {i})"
            )
            break
    control_events_by_boundary = _index_control_events(
        meta,
        max_action_seq=len(rows),
        issues=issues,
    )
    # post_state И pre_state каждой строки должны быть заполнены.
    for r in rows:
        if r.get("post_state") is None:
            issues.append(
                f"[continuity] seq={r.get('seq','?')}: post_state is None "
                f"(after_action not recorded — state gap)"
            )
        if r.get("pre_state") is None:
            issues.append(
                f"[continuity] seq={r.get('seq','?')}: pre_state is None "
                f"(before_action not recorded — state gap)"
            )
    # цепочка post→pre
    for k in range(len(rows) - 1):
        post = rows[k].get("post_state")
        nxt_pre = rows[k + 1].get("pre_state")
        if post is None or nxt_pre is None:
            continue
        after_seq = rows[k].get("seq")
        expected_pre = post
        if isinstance(after_seq, int) and not isinstance(after_seq, bool):
            boundary_events = control_events_by_boundary.get(after_seq, [])
            if boundary_events:
                expected_pre = _replay_control_events(
                    post,
                    boundary_events,
                    meta=meta,
                    after_action_seq=after_seq,
                    issues=issues,
                )
        if expected_pre != nxt_pre:
            diff = _state_diff_keys(expected_pre, nxt_pre)
            issues.append(
                f"[continuity] seq={rows[k].get('seq')}→{rows[k+1].get('seq')}: "
                f"post_state + audited control events != next pre_state "
                f"(diff keys: {diff})"
            )
    # терминальный post_state ↔ meta.status/winner. Различаем естественный конец
    # (последняя строка не терминальная — state.status мутируется движком в
    # p1_win/p2_win) и терминальную строку (surrender/draw/stalemate): mark_surrender
    # НЕ мутирует state.status (остаётся 'ongoing'), авторитет — meta.status.
    last = rows[-1]
    last_is_term = _is_terminal(last)
    last_post = last.get("post_state") or {}
    last_status = last_post.get("status")
    meta_status = meta.get("status")
    if not last_is_term:
        if last_status is None or meta_status is None:
            issues.append(
                f"[continuity] missing terminal status (post={last_status!r}, "
                f"meta={meta_status!r})"
            )
        elif last_status != meta_status:
            issues.append(
                f"[continuity] final post_state.status={last_status!r} != "
                f"meta.status={meta_status!r}"
            )
    else:
        if meta_status is None:
            issues.append("[continuity] terminal row but meta.status missing")
        elif meta_status not in _TERMINAL_STATUSES:
            issues.append(
                f"[continuity] terminal row but meta.status={meta_status!r} "
                f"not a terminal status"
            )
    # winner ↔ status согласованность. Для терминальной строки (surrender/draw/
    # stalemate) state.status не отражает исход (mark_surrender/mark_draw не мути-
    # руют его — остаётся 'ongoing'), авторитет — meta.status; для естественного
    # конца — post_state.status. Иначе terminal draw с meta.winner_user_id!=None
    # прошёл бы незамеченным (last_status='ongoing' не попадает ни в какую ветку).
    p1_uid_local = meta.get("p1_user_id")
    p2_uid_local = meta.get("p2_user_id")
    meta_winner = meta.get("winner_user_id")
    status_for_winner = meta_status if last_is_term else last_status
    if status_for_winner in ("p1_win", "p2_win"):
        expected_winner = p1_uid_local if status_for_winner == "p1_win" else p2_uid_local
        if meta_winner is not None and expected_winner is not None and meta_winner != expected_winner:
            issues.append(
                f"[continuity] final status={status_for_winner!r} implies winner="
                f"{expected_winner} but meta.winner_user_id={meta_winner}"
            )
    elif status_for_winner in ("draw", "stalemate"):
        # draw/stalemate не имеют победителя — non-None winner противоречив.
        if meta_winner is not None:
            issues.append(
                f"[continuity] final status={status_for_winner!r} but meta.winner_user_id="
                f"{meta_winner} (draw/stalemate has no winner)"
            )
    # turns.jsonl: turn_number строго возрастают, континуальны от 1, валидные снапшоты.
    if turns is not None:
        prev_tn = 0
        for i, t in enumerate(turns, 1):
            tn = t.get("turn_number")
            if tn is None:
                issues.append(f"[continuity] turns.jsonl row {i}: missing turn_number")
                continue
            if tn <= prev_tn:
                issues.append(
                    f"[continuity] turns.jsonl row {i}: turn_number={tn} not strictly "
                    f"increasing (prev={prev_tn})"
                )
            if tn != prev_tn + 1:
                issues.append(
                    f"[continuity] turns.jsonl row {i}: turn_number={tn} not contiguous "
                    f"(expected {prev_tn + 1})"
                )
            prev_tn = tn
            if "p1" not in t or "p2" not in t:
                issues.append(f"[continuity] turns.jsonl row {i}: not a state snapshot")
        # На границе хода pre_state первой action-строки этого хода == turns-снапшоту.
        if turns:
            tn_to_turn = {t.get("turn_number"): t for t in turns}
            seen_turns: set = set()
            for r in rows:
                tn = r.get("turn_number")
                if tn is None or tn in seen_turns:
                    continue
                seen_turns.add(tn)  # первая action-строка этого хода
                t_snap = tn_to_turn.get(tn)
                if t_snap is None:
                    # ход без turn-снапшота — бывает только если recorder пропустил
                    # _append_turn_row; для первого хода это реальная потеря.
                    if tn == (turns[0].get("turn_number") if turns else None):
                        pass
                    issues.append(
                        f"[continuity] seq={r.get('seq')}: turn_number={tn} has no "
                        f"turns.jsonl snapshot"
                    )
                    continue
                pre = r.get("pre_state")
                if pre is not None and pre != t_snap:
                    diff = _state_diff_keys(pre, t_snap)
                    issues.append(
                        f"[continuity] seq={r.get('seq')}: pre_state != turns.jsonl "
                        f"snapshot for turn {tn} (diff keys: {diff})"
                    )


def _check_correspondence(
    rows: List[Dict[str, Any]], battle_log_path: Path, issues: List[str]
) -> None:
    """(4) battle_log actions 1:1 (по порядку) с не-терминальными trace-строками."""
    if not battle_log_path.exists():
        issues.append(
            f"[correspondence] battle_log missing: {battle_log_path.name} — "
            f"cannot verify action correspondence"
        )
        return
    try:
        bl = json.loads(battle_log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"[correspondence] battle_log invalid json: {exc}")
        return
    bl_actions = bl.get("actions") or []
    normal = [r for r in rows if not _is_terminal(r)]
    if len(normal) != len(bl_actions):
        issues.append(
            f"[correspondence] trace normal rows={len(normal)} != battle_log "
            f"actions={len(bl_actions)} (training surface out of sync with battle_log)"
        )
    # поэлементное сравнение до min длины — ловим расхождение даже при разной длине.
    for i in range(min(len(normal), len(bl_actions))):
        tr = normal[i]
        bl = bl_actions[i]
        mism: List[str] = []
        if tr.get("actor_user_id") != bl.get("actor"):
            mism.append(f"actor {tr.get('actor_user_id')}!={bl.get('actor')}")
        if tr.get("action_type") != bl.get("kind"):
            mism.append(f"type {tr.get('action_type')!r}!={bl.get('kind')!r}")
        if tr.get("action_json") != bl.get("action_dict"):
            mism.append("action_json != action_dict")
        if tr.get("accepted") != bl.get("ok"):
            mism.append(f"accepted {tr.get('accepted')}!=ok {bl.get('ok')}")
        # trace хранит turn_number PRE-execute, battle_log `turn` — POST-execute.
        # Сравниваем battle_log.turn с trace post_state.turn_number (post-execute на
        # обеих сторонах) — универсально, без special-case'а end_turn.
        tr_post_turn = (tr.get("post_state") or {}).get("turn_number")
        if tr_post_turn is not None and tr_post_turn != bl.get("turn"):
            mism.append(f"turn(post) {tr_post_turn}!={bl.get('turn')}")
        if mism:
            issues.append(
                f"[correspondence] row {i}: trace vs battle_log mismatch ({'; '.join(mism)})"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_v5_trace(
    v5_dir: Path,
    battle_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Валидирует один V5-trace (директория ``battles/<bid>/v5``).

    Возвращает ``{ok: bool, issues: List[str], checks: {rows, turns, terminal_rows}}``.
    ``ok`` = ``issues`` пуст. ``issues`` — тегированные строки
    ``[legal_index|actor|continuity|correspondence] ...``.

    Структурные проверки (наличие/непустота meta/turns/actions) включены сюда,
    чтобы MCP-хендлер мог делегировать полностью.
    """
    issues: List[str] = []

    # --- структурные: наличие + непустота ---
    meta_path = v5_dir / "meta.json"
    turns_path = v5_dir / "turns.jsonl"
    actions_path = v5_dir / "actions.jsonl"
    for need, p in (("meta.json", meta_path), ("turns.jsonl", turns_path), ("actions.jsonl", actions_path)):
        if not p.exists():
            issues.append(f"missing {need}")
        elif need.endswith(".jsonl") and p.stat().st_size == 0:
            issues.append(f"empty {need}")

    # meta
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"[meta] invalid json: {exc}")

    # actions
    rows, err = _read_jsonl(actions_path)
    if err is not None:
        issues.append(f"[actions] {err}")
    if actions_path.exists() and actions_path.stat().st_size > 0 and rows == []:
        # пустой список после чтения нетривиального файла уже покрыт empty-чеком,
        # но _read_jsonl мог вернуть ([], None) — это валидный «только blank-строки».
        pass
    rows = rows or []

    # turns
    turns, terr = _read_jsonl(turns_path)
    if terr is not None:
        issues.append(f"[turns] {terr}")
    turns = turns or []

    if issues:
        # структурные проблемы (missing/empty/invalid files) → глубокие проверки
        # бессмысленны/небезопасны.
        return {"ok": False, "issues": issues, "checks": {"rows": len(rows), "turns": len(turns)}}

    # --- meta required fields (cross-meta инварианты actor/source, terminal↔meta) ---
    missing_meta = [f for f in _REQUIRED_META_FIELDS if f not in meta]
    if missing_meta:
        issues.append(f"[meta] missing required fields {missing_meta}")

    # --- требуемые поля каждой action-строки + bool accepted ---
    for r in rows:
        missing = [f for f in _REQUIRED_ACTION_FIELDS if f not in r]
        if missing:
            issues.append(f"[schema] seq={r.get('seq','?')}: missing fields {missing}")
        if "accepted" in r and not isinstance(r["accepted"], bool):
            issues.append(f"[schema] seq={r.get('seq','?')}: accepted={r['accepted']!r} not bool")
        control_source = r.get("control_source")
        if control_source is not None and (
            not isinstance(control_source, str) or not control_source.strip()
        ):
            issues.append(
                f"[schema] seq={r.get('seq','?')}: control_source="
                f"{control_source!r} must be null or a non-empty string"
            )

    # Глубокие инварианты запускаем при наличии строк (schema-проблемы не глушат их —
    # больше сигнала; проверки None-guarded и не падают на порченных строках).
    if rows:
        _check_legal_action_index(rows, issues)
        _check_actor_source(rows, meta, issues)
        _check_timing_and_outcome(rows, issues)
        _check_v5_history_schema(rows, turns, issues)
        _check_continuity(rows, turns, meta, issues)
        if battle_log_path is not None:
            _check_correspondence(rows, battle_log_path, issues)

    terminal_rows = sum(1 for r in rows if _is_terminal(r))
    return {
        "ok": not issues,
        "issues": issues,
        "checks": {
            "rows": len(rows),
            "turns": len(turns),
            "terminal_rows": terminal_rows,
        },
    }


__all__ = ["validate_v5_trace"]
