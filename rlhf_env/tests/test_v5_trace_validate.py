"""Глубокие инварианты V5-trace (rlhf_env/components/v5_trace_validate.py).

Реальный rl-vs-bot бой прогоняется до game_over и проходит ВСЕ глубокие
инварианты (legal_action_index / actor-source / continuity / battle_log
correspondence). Затем тот же trace мутируется по одному полю за раз —
валидатор должен поймать каждый класс нарушений с правильным тегом.

In-process (без HTTP/stdio): ArenaMatchManager + MatchRunner.run_auto гоняет
обе стороны (rl p1 + random p2) до game_over; v5/trace + battle_log читаются
прямо из sessions-директории.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rlhf_env.components.v5_trace_validate import validate_v5_trace
from rlhf_env.tests._v5_helpers import create_match, make_manager, read_jsonl, v5_dir_for


def _drive_completed(tmp_path: Path):
    """Гоняет rl-vs-random бой до game_over. Возвращает (v5_dir, battle_log_path)."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", p2_model="random",
        seed=7, starting_player="p1",
    )

    async def go():
        await runner.run_auto()
        return engine.is_ended

    ended = asyncio.run(go())
    assert ended, "rl-vs-random бой не завершился за run_auto — test fixture сломан"
    v5 = v5_dir_for(match, tmp_path)
    gdir = Path(tmp_path) / "sessions" / match.group_id
    blog = gdir / "battles" / f"{match.battle_id}.json"
    return v5, blog


def _copy_trace(src_v5: Path, src_blog: Path, dst_v5: Path, dst_blog: Path) -> None:
    dst_v5.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "turns.jsonl", "actions.jsonl"):
        shutil.copy(src_v5 / f, dst_v5 / f)
    shutil.copy(src_blog, dst_blog)


def _load_actions(v5: Path) -> List[Dict[str, Any]]:
    return read_jsonl(v5 / "actions.jsonl")


def _save_actions(v5: Path, rows: List[Dict[str, Any]]) -> None:
    (v5 / "actions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _first_normal_accepted(rows: List[Dict[str, Any]]) -> int:
    """Индекс первой не-терминальной accepted-строки (для мутаций legal_index)."""
    for i, r in enumerate(rows):
        if not (r.get("legal_action_count", -1) == 0 and r.get("legal_action_index") is None):
            if r.get("accepted") is True:
                return i
    raise AssertionError("нет accepted не-терминальной строки для мутации")


def _has_issue(rep: Dict[str, Any], tag: str) -> bool:
    return any(tag in s for s in rep["issues"])


# ---------------------------------------------------------------------------
# Реальный бой проходит все инварианты
# ---------------------------------------------------------------------------

def test_real_trace_passes_all_invariants(tmp_path):
    v5, blog = _drive_completed(tmp_path)
    rep = validate_v5_trace(v5, battle_log_path=blog)
    assert rep["ok"], f"реальный trace должен проходить, но issues={rep['issues']}"
    assert rep["checks"]["rows"] > 0
    assert rep["checks"]["turns"] > 0


def test_real_trace_via_mcp_tool(tmp_path):
    """MCP validate_v5_traces делегирует в validate_v5_trace и возвращает каноничную форму."""
    from rlhf_env.mcp_server import HeadlessHub, MCPServer
    from rlhf_env.components.policy_registry import PolicyRegistry

    repo = Path(__file__).resolve().parents[2]
    hub = HeadlessHub(
        sessions_dir=str(tmp_path / "sessions"),
        models_dir=str(repo / "ai" / "models"),
        cards_path=str(repo / "ai" / "cards.json"),
    )
    srv = MCPServer(hub, PolicyRegistry.scan(str(repo / "ai" / "models")))
    r = asyncio.run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 1, "starting_player": "p1",
    }}))
    gid = r["group_id"]
    vt = asyncio.run(srv._tool("validate_v5_traces", {"group_id": gid}))
    assert vt["checked"] == 1 and vt["ok"] == 1 and vt["broken"] == [], \
        f"deep validator flagged a clean trace: {vt}"


# ---------------------------------------------------------------------------
# Мутации — каждый класс нарушений флагается
# ---------------------------------------------------------------------------

def test_mutation_legal_index_out_of_range(tmp_path):
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    i = _first_normal_accepted(rows)
    rows[i]["legal_action_index"] = 99999
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[legal_index]"), rep["issues"]


def test_mutation_accepted_action_lost_index(tmp_path):
    """accepted действие с legal_action_index=None = потеря обучающей метки."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    i = _first_normal_accepted(rows)
    rows[i]["legal_action_index"] = None
    rows[i]["action_native"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "training label lost"), rep["issues"]


def test_mutation_action_native_label_mismatch(tmp_path):
    """action_native != legal_actions[idx] — метка указывает на чужое действие."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    i = _first_normal_accepted(rows)
    rows[i]["action_native"] = {"__mutated__": True}
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "label points at wrong action"), rep["issues"]


def test_mutation_actor_source_mismatch(tmp_path):
    """decision_source p1 не совпадает с meta.p1_actor_type."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    # p1-rl строка: actor_player==1 → source должен быть 'rl'; ломаем на 'human'.
    p1_row = next(r for r in rows if r.get("actor_player") == 1)
    p1_row["decision_source"] = "human"
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[actor]"), rep["issues"]


def test_mutation_acted_out_of_turn(tmp_path):
    """pre_state.current_turn_owner_id != actor_user_id — действие не в свой ход."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    r = rows[0]
    r["pre_state"] = dict(r["pre_state"])
    r["pre_state"]["current_turn_owner_id"] = -999
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "acted out of turn"), rep["issues"]


def test_mutation_continuity_post_ne_next_pre(tmp_path):
    """post_state[N] != pre_state[N+1] — разрыв цепочки состояния."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    if len(rows) < 2:
        pytest.skip("бой слишком короткий для continuity-мутации")
    # ломаем post_state первой строки (вкладываем чужой turn_number)
    rows[0]["post_state"] = dict(rows[0]["post_state"])
    rows[0]["post_state"]["turn_number"] = rows[0]["post_state"]["turn_number"] + 100
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[continuity]"), rep["issues"]


def test_mutation_terminal_status_disagrees_with_meta(tmp_path):
    """final post_state.status != meta.status — терминальное состояние рассинхронено."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    # ломаем meta.status в сторону, не совпадающую с финальным post_state
    meta = json.loads((dst_v5 / "meta.json").read_text(encoding="utf-8"))
    meta["status"] = "draw" if meta.get("status") != "draw" else "p1_win"
    (dst_v5 / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "final post_state.status"), rep["issues"]


def test_mutation_battle_log_count_mismatch(tmp_path):
    """battle_log actions count != trace normal rows — training surface out of sync."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    bl = json.loads(dst_blog.read_text(encoding="utf-8"))
    if len(bl["actions"]) <= 1:
        pytest.skip("бой слишком короткий для correspondence-мутации")
    bl["actions"] = bl["actions"][:-1]  # выкидываем последнее действие
    dst_blog.write_text(json.dumps(bl, ensure_ascii=False), encoding="utf-8")
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[correspondence]"), rep["issues"]


def test_mutation_battle_log_field_mismatch(tmp_path):
    """battle_log action_dict != trace action_json при совпадающей длине."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    bl = json.loads(dst_blog.read_text(encoding="utf-8"))
    bl["actions"][0]["kind"] = "__mutated__"
    dst_blog.write_text(json.dumps(bl, ensure_ascii=False), encoding="utf-8")
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[correspondence]"), rep["issues"]


def test_missing_battle_log_flagged(tmp_path):
    src_v5, _ = _drive_completed(tmp_path)
    rep = validate_v5_trace(src_v5, battle_log_path=tmp_path / "does_not_exist.json")
    assert not rep["ok"] and _has_issue(rep, "[correspondence] battle_log missing"), rep["issues"]


def test_structural_missing_files(tmp_path):
    """Отсутствие meta/turns/actions флагается (поверхностные проверки сохранены)."""
    empty = tmp_path / "empty_v5"
    empty.mkdir()
    rep = validate_v5_trace(empty, battle_log_path=tmp_path / "x.json")
    assert not rep["ok"]
    assert any("missing meta.json" in s for s in rep["issues"])
    assert any("missing turns.jsonl" in s for s in rep["issues"])
    assert any("missing actions.jsonl" in s for s in rep["issues"])


# ---------------------------------------------------------------------------
# Дыры, найденные adversarial-верификацией (workflow verify-v5-trace-validator)
# ---------------------------------------------------------------------------

def _drive_surrender(tmp_path: Path, seed: int = 11):
    """Гоняет llm-vs-random бой несколько ходов, затем p1 сдаётся. Возвращает
    (v5_dir, battle_log_path). Терминальная строка surrender не имеет battle_log-
    записи; mark_surrender не мутирует state.status (остаётся 'ongoing')."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="llm", p2_model="random", seed=seed, starting_player="p1",
    )

    async def go():
        for _ in range(6):
            if engine.is_ended:
                break
            if engine.get_current_player_id() == engine.human_user_id:
                await runner.execute_human_action({"type": "end_turn"})
            else:
                await runner.run_bot_turn()
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    v5 = v5_dir_for(match, tmp_path)
    gdir = Path(tmp_path) / "sessions" / match.group_id
    return v5, gdir / "battles" / f"{match.battle_id}.json"


def test_real_surrender_trace_passes(tmp_path):
    """FP-регресс: реальный surrender-бой валиден (mark_surrender не мутирует
    state.status — терминальная строка не сравнивается с meta.status по status)."""
    v5, blog = _drive_surrender(tmp_path)
    rep = validate_v5_trace(v5, battle_log_path=blog)
    assert rep["ok"], f"surrender trace должен проходить, но issues={rep['issues']}"
    assert rep["checks"]["terminal_rows"] == 1


def test_mutation_lost_label_emptied_legal_set(tmp_path):
    """FIX: play_card с очищенным legal_actions = потерянная метка, НЕ терминал.
    _is_terminal дискриминирует по action_type, а не по форме legal-набора."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    tgt = next(r for r in rows if r.get("action_type") == "play_card" and r.get("accepted") is True)
    tgt["legal_actions"] = []
    tgt["legal_action_count"] = 0
    tgt["legal_action_index"] = None
    tgt["action_native"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "training label lost"), rep["issues"]


def test_mutation_middle_pre_state_none(tmp_path):
    """FIX: pre_state=None в средней строке — разрыв цепочки, не невидимый."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    if len(rows) < 3:
        pytest.skip("бой слишком короткий")
    rows[len(rows) // 2]["pre_state"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "pre_state is None"), rep["issues"]


def test_mutation_missing_meta_actor_type(tmp_path):
    """FIX: отсутствие meta.p1_actor_type + неверный source = мис-тег, не молчание."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    m = json.loads((dst_v5 / "meta.json").read_text(encoding="utf-8"))
    m.pop("p1_actor_type", None)
    (dst_v5 / "meta.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "[meta] missing required fields"), rep["issues"]


def test_mutation_owner_none_flagged(tmp_path):
    """FIX: current_turn_owner_id=None — действие не доказано в свой ход."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    r = rows[0]
    r["pre_state"] = dict(r["pre_state"])
    r["pre_state"]["current_turn_owner_id"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "owner missing"), rep["issues"]


def test_mutation_non_bool_accepted(tmp_path):
    """FIX: accepted=1 (truthy non-bool) — schema-level catch + lost-label всё равно ловится."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    i = _first_normal_accepted(rows)
    rows[i]["accepted"] = 1  # truthy, не bool
    rows[i]["legal_action_index"] = None
    rows[i]["action_native"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"]
    assert _has_issue(rep, "not bool") or _has_issue(rep, "training label lost"), rep["issues"]


def test_mutation_terminal_status_none_flagged(tmp_path):
    """FIX: final post_state.status=None при непустом meta.status — нарушение."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    last = rows[-1]
    last["post_state"] = dict(last["post_state"])
    last["post_state"]["status"] = None
    _save_actions(dst_v5, rows)
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "missing terminal status"), rep["issues"]


def test_mutation_draw_with_winner_flagged(tmp_path):
    """FIX: draw/stalemate с non-None winner противоречив (draw не имеет победителя)."""
    src_v5, src_blog = _drive_completed(tmp_path)
    dst_v5 = tmp_path / "mut" / "v5"
    dst_blog = tmp_path / "mut" / "b.json"
    _copy_trace(src_v5, src_blog, dst_v5, dst_blog)
    rows = _load_actions(dst_v5)
    last = rows[-1]
    last["post_state"] = dict(last["post_state"])
    last["post_state"]["status"] = "draw"
    _save_actions(dst_v5, rows)
    m = json.loads((dst_v5 / "meta.json").read_text(encoding="utf-8"))
    m["status"] = "draw"
    m["winner_user_id"] = 99999
    (dst_v5 / "meta.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    rep = validate_v5_trace(dst_v5, battle_log_path=dst_blog)
    assert not rep["ok"] and _has_issue(rep, "draw/stalemate has no winner"), rep["issues"]