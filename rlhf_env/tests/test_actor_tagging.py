"""Теггинг актора llm-vs-bot (WS3 плана joyful-churning-bird).

Проверяет, что decision_source расширен до {human|llm|bot}, battle_tag и
p1_actor_type/p2_actor_type прошиваются через всю цепочку:
spec → _build_match → engine → match_runner → v5_trace(meta) → manifest.
MCP-LLM бой = llm-vs-bot; браузерный бой = human-vs-bot.
"""
from __future__ import annotations

import asyncio

from rlhf_env.tests._v5_helpers import (
    create_match,
    drive_until_mana_draw_then,
    make_manager,
    read_json,
    read_jsonl,
    v5_dir_for,
)


def _play(tmp_path, *, p1_actor_type, p2_model="random", seed=42):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type=p1_actor_type, p2_model=p2_model, seed=seed
    )

    async def go():
        # несколько ходов, чтобы набрать action-строки обоих акторов
        for _ in range(6):
            if engine.is_ended:
                break
            if engine.get_current_player_id() == engine.human_user_id:
                await runner.execute_human_action(
                    {"type": "end_turn", "client_action_id": f"c_et_{_}"}
                )
            else:
                await runner.run_bot_turn()
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    return mgr, match, engine


def test_engine_exposes_actor_fields(tmp_path):
    """engine.p1_actor_type / p2_actor_type / battle_tag выставлены."""
    _, match, engine = _play(tmp_path, p1_actor_type="llm")
    assert engine.p1_actor_type == "llm"
    assert engine.p2_actor_type == "bot"
    assert engine.battle_tag == "llm-vs-bot"


def test_meta_json_carries_battle_tag_and_actors(tmp_path):
    """v5/meta.json пишет battle_tag, p1_actor_type, p2_actor_type, v5_trace_present."""
    _, match, _ = _play(tmp_path, p1_actor_type="llm")
    meta = read_json(v5_dir_for(match, tmp_path) / "meta.json")
    assert meta["battle_tag"] == "llm-vs-bot"
    assert meta["p1_actor_type"] == "llm"
    assert meta["p2_actor_type"] == "bot"
    assert meta["v5_trace_present"] is True
    # back-compat поля сохранены
    assert "p1_is_bot" in meta and "p2_is_bot" in meta


def test_decision_source_per_row_llm_vs_bot(tmp_path):
    """actions.jsonl: p1-rows → decision_source 'llm', p2-rows → 'bot'."""
    _, match, engine = _play(tmp_path, p1_actor_type="llm")
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows
    sources = {r.get("decision_source") for r in rows}
    assert sources <= {"llm", "bot"}, f"посторонние decision_source: {sources}"
    assert "llm" in sources, "нет ни одной llm-строки (p1)"
    assert "bot" in sources, "нет ни одной bot-строки (p2)"


def test_human_vs_bot_tagging(tmp_path):
    """Браузерный путь: p1_actor_type='human' → battle_tag 'human-vs-bot', rows 'human'."""
    _, match, engine = _play(tmp_path, p1_actor_type="human")
    assert engine.battle_tag == "human-vs-bot"
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows
    sources = {r.get("decision_source") for r in rows}
    assert "human" in sources, "нет human-строк в human-vs-bot бое"
    assert "bot" in sources
    meta = read_json(v5_dir_for(match, tmp_path) / "meta.json")
    assert meta["battle_tag"] == "human-vs-bot"
    assert meta["p1_actor_type"] == "human"


def test_manifest_carries_battle_tag_and_p1_actor(tmp_path):
    """manifest.json battles_results[] пишет battle_tag, p1_actor_type, v5_trace_ok."""
    mgr, match, _ = _play(tmp_path, p1_actor_type="llm")
    manifest_path = mgr.sessions_dir / match.group_id / "manifest.json"
    manifest = read_json(manifest_path)
    results = manifest.get("battles_results", [])
    assert results, "manifest не записал результат боя"
    r = results[-1]
    assert r["battle_tag"] == "llm-vs-bot"
    assert r["p1_actor_type"] == "llm"
    assert r["v5_trace_ok"] is True


def test_invalid_p1_actor_defaults_to_human(tmp_path):
    """Невалидный p1_actor_type падает обратно к 'human' (default)."""
    _, match, engine = _play(tmp_path, p1_actor_type="robot-overlord-9000")
    # _build_match валидирует в {human,llm}; невалидный → human
    assert engine.p1_actor_type == "human"
    assert engine.battle_tag == "human-vs-bot"