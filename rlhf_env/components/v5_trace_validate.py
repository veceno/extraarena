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

# Поля meta.json, обязательные для глубоких cross-meta инвариантов (actor/source,
# terminal↔meta). Реальный V5TraceRecorder._write_meta всегда их пишет; отсутствие =
# порча meta, и source-consistency-чек не должен молча пропускаться.
_REQUIRED_META_FIELDS = ("p1_user_id", "p2_user_id", "p1_actor_type", "p2_actor_type", "status")

# Поля action-строки, обязательные для любой (в т.ч. терминальной) строки.
_REQUIRED_ACTION_FIELDS = (
    "seq",
    "turn_number",
    "actor_user_id",
    "actor_player",
    "decision_source",
    "legal_action_index",
    "action_type",
    "action_json",
    "legal_actions",
    "legal_action_count",
    "pre_state",
    "post_state",
    "accepted",
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
    с actor_type; current_turn_owner_id == actor_user_id."""
    p1_uid = meta.get("p1_user_id")
    p2_uid = meta.get("p2_user_id")
    p1_type = meta.get("p1_actor_type")
    p2_type = meta.get("p2_actor_type")
    for r in rows:
        seq = r.get("seq", "?")
        actor_uid = r.get("actor_user_id")
        actor_player = r.get("actor_player")
        src = r.get("decision_source")
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
        else:
            # Согласованность source ⇔ actor_type из meta.
            if actor_player == 1 and p1_type is not None and src != p1_type:
                issues.append(
                    f"[actor] seq={seq}: p1 decision_source={src!r} != "
                    f"meta.p1_actor_type={p1_type!r}"
                )
            if actor_player == 2:
                # p2 — всегда бот (baseline/onnx), source=='bot', type=='bot'.
                if src != "bot":
                    issues.append(
                        f"[actor] seq={seq}: p2 decision_source={src!r} != 'bot'"
                    )
                if p2_type is not None and p2_type != "bot":
                    issues.append(
                        f"[actor] seq={seq}: meta.p2_actor_type={p2_type!r} != 'bot'"
                    )
        # Ход только свой: current_turn_owner_id в pre_state == actor_user_id.
        # owner=None — тоже нарушение (поле отсутствует/порчено): реальный recorder
        # всегда выставляет current_turn_owner_id; None != actor_uid → флаг.
        pre = r.get("pre_state") or {}
        owner = pre.get("current_turn_owner_id")
        if actor_uid is not None and owner != actor_uid:
            issues.append(
                f"[actor] seq={seq}: pre_state.current_turn_owner_id={owner!r} != "
                f"actor_user_id={actor_uid} (acted out of turn / owner missing)"
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
        if post != nxt_pre:
            diff = _state_diff_keys(post, nxt_pre)
            issues.append(
                f"[continuity] seq={rows[k].get('seq')}→{rows[k+1].get('seq')}: "
                f"post_state != next pre_state (diff keys: {diff})"
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

    # Глубокие инварианты запускаем при наличии строк (schema-проблемы не глушат их —
    # больше сигнала; проверки None-guarded и не падают на порченных строках).
    if rows:
        _check_legal_action_index(rows, issues)
        _check_actor_source(rows, meta, issues)
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