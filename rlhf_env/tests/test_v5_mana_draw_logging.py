"""V5 mana_draw-логирование: добор карты человеком/LLM пишется в actions.jsonl.

Покрывает WS1 (план joyful-churning-bird): ManaDrawAction проходит через
MatchRunner.execute_human_action → engine.mana_draw → v5_trace.before_action
с корректным action_native / legal_action_index / mana_draw_count_this_turn в
снапшоте, а бот НИКОГДА не выполняет mana_draw (post-pick guard в run_bot_turn).
"""
from __future__ import annotations

from rlhf_env.tests._v5_helpers import (
    create_match,
    drive_until_mana_draw_then,
    make_manager,
    read_jsonl,
    v5_dir_for,
)


def _drew_match(tmp_path, *, p1_actor_type="llm", p2_model="random", seed=42):
    """Гоняет бой до первого mana_draw p1, затем сдаётся. Возвращает (match, drew)."""
    import asyncio

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


def test_bot_never_performs_mana_draw(tmp_path):
    """Бот не выполняет mana_draw: post-pick guard в run_bot_turn срабатывает.

    Guard должен заменять ManaDrawAction на EndTurnAction для любого бота
    (V4 слеп к mana_draw по 601-кандидатной маске, guard = belt-and-suspenders).
    """
    match, _ = _drew_match(tmp_path, p2_model="random")
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    bot_rows = [r for r in rows if r.get("decision_source") == "bot"]
    assert bot_rows, "бот не сделал ни одного хода — тест не имеет смысла"
    bot_md = [r for r in bot_rows if r.get("action_type") == "mana_draw"]
    assert not bot_md, f"бот выполнил mana_draw: {bot_md}"