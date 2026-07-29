"""V5 mana_draw-логирование: добор карты человеком/LLM пишется в actions.jsonl.

Покрывает WS1 (план joyful-churning-bird): ManaDrawAction проходит через
MatchRunner.execute_human_action → engine.mana_draw → v5_trace.before_action
с корректным action_native / legal_action_index / mana_draw_count_this_turn в
снапшоте. Non-V5 bot guard сохраняет совместимость, а V5 parallel mana head
доходит до реальной механики и логируется как bot action.
"""
from __future__ import annotations

import asyncio

from core.actions import EndTurnAction, ManaDrawAction
from rlhf_env.tests._v5_helpers import (
    create_match,
    drive_until_mana_draw_then,
    make_manager,
    read_jsonl,
    v5_dir_for,
)


def _drew_match(tmp_path, *, p1_actor_type="llm", p2_model="random", seed=42):
    """Гоняет бой до первого mana_draw p1, затем сдаётся. Возвращает (match, drew)."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type=p1_actor_type, p2_model=p2_model, seed=seed
    )

    async def go():
        drew, _ = await drive_until_mana_draw_then(runner, engine)
        if not engine.is_ended:
            await runner.surrender()
        return drew

    return match, asyncio.run(go())


def test_mana_draw_row_logged_with_action_native_and_index(tmp_path):
    """actions.jsonl содержит строку mana_draw с action_native + legal_index."""
    match, drew = _drew_match(tmp_path)
    assert drew, "mana_draw ни разу не стал легальным за бой — тест не имеет смысла"

    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    md = [r for r in rows if r.get("action_type") == "mana_draw"]
    assert md, "нет ни одной mana_draw-строки в actions.jsonl"

    r = md[0]
    assert r["action_native"] == {"type": "mana_draw"}, r["action_native"]
    assert r["legal_action_index"] is not None, "legal_action_index не разрешён"
    assert isinstance(r["legal_action_index"], int) and r["legal_action_index"] >= 0


def test_mana_draw_increments_count_and_spends_mana(tmp_path):
    """pre/post state: mana_draw_count_this_turn растёт, мана тратится (2), рука растёт."""
    match, drew = _drew_match(tmp_path)
    assert drew
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    r = next(r for r in rows if r.get("action_type") == "mana_draw")

    pre_p1 = r["pre_state"]["p1"]
    post_p1 = r.get("post_state", {}).get("p1", {})
    assert pre_p1.get("mana_draw_count_this_turn", 0) == 0, "счётчик должен стартовать с 0"
    assert post_p1.get("mana_draw_count_this_turn", 0) >= 1, "счётчик не инкрементирован"
    # стоимость 2*(0+1)=2 на первом доборе
    assert pre_p1.get("mana", 0) - post_p1.get("mana", 0) == 2, "мана не потрачена (2)"
    assert len(post_p1.get("hand", [])) == len(pre_p1.get("hand", [])) + 1, "рука не выросла на 1"


def test_mana_draw_count_in_every_p1_snapshot(tmp_path):
    """mana_draw_count_this_turn присутствует в каждом p1-снапшоте (рендер плитки «+»)."""
    match, _ = _drew_match(tmp_path)
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows, "actions.jsonl пуст"
    for r in rows:
        assert "mana_draw_count_this_turn" in r["pre_state"]["p1"], (
            "mana_draw_count_this_turn отсутствует в p1-снапшоте"
        )


class _PreferManaPolicy:
    name = "prefer-mana"

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def select_action(self, arena, player_id: int) -> int:
        legal = arena.get_legal_actions(player_id)
        for index, action in enumerate(legal):
            if isinstance(action, ManaDrawAction):
                return index
        return next(
            index
            for index, action in enumerate(legal)
            if isinstance(action, EndTurnAction)
        )


async def _no_sleep(_delay: float) -> None:
    return None


def _run_bot_with_mana_surface(
    tmp_path,
    monkeypatch,
    *,
    policy_kind: str,
):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr,
        p2_model="random",
        starting_player="p2",
    )
    match.bot_policy = _PreferManaPolicy(policy_kind)
    state = engine._arena.state
    state.p2.mana = 2
    state.p2.max_mana = max(2, state.p2.max_mana)
    assert state.p2.deck
    monkeypatch.setattr(
        "rlhf_env.components.match_runner.asyncio.sleep",
        _no_sleep,
    )

    asyncio.run(runner.run_bot_turn())
    return match, engine


def test_non_v5_bot_mana_draw_is_replaced_by_end_turn(
    tmp_path,
    monkeypatch,
):
    """The compatibility guard remains active for V4/non-V5 policies."""

    match, engine = _run_bot_with_mana_surface(
        tmp_path,
        monkeypatch,
        policy_kind="random",
    )
    assert engine._arena.state.p2.mana_draw_count_this_turn == 0
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert not [
        row
        for row in rows
        if row.get("decision_source") == "bot"
        and row.get("action_type") == "mana_draw"
    ]


def test_v5_bot_parallel_head_executes_and_logs_mana_draw(
    tmp_path,
    monkeypatch,
):
    """A V5 policy must reach the latest Arena mana-draw mechanic."""

    match, engine = _run_bot_with_mana_surface(
        tmp_path,
        monkeypatch,
        policy_kind="v5",
    )
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    mana_rows = [
        row
        for row in rows
        if row.get("decision_source") == "bot"
        and row.get("action_type") == "mana_draw"
    ]
    assert mana_rows
    assert engine._arena.state.p2.mana_draw_count_this_turn == 1
