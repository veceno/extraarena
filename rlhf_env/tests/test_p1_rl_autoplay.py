"""Block C — p1-as-RL auto-play (model-vs-model) tests.

Покрывает:
  - p1_actor_type='rl', p1_model='random' → run_auto доигрывает до game_over БЕЗ
    submit_action; manifest p1_actor_type=='rl', battle_tag=='rl-vs-bot',
    v5/meta.json p1_is_bot==True, actions.jsonl содержит decision_source=='rl'.
  - battle_tag 'rl-vs-rl' когда p2 — onnx-kind (v5 stub), без запуска игры.
  - regression: human/llm путь — match.p1_policy is None, p1_is_bot False.
  - surrender недоступен для p1-RL.
"""
from __future__ import annotations

import asyncio
import json

from rlhf_env.tests._v5_helpers import (
    create_match,
    make_manager,
    read_json,
    read_jsonl,
    v5_dir_for,
)


def test_p1_rl_autoplays_to_game_over(tmp_path):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", starting_player="p1",
    )
    assert match.p1_policy is not None
    asyncio.run(runner.run_auto())
    assert engine.is_ended is True

    man = json.loads((tmp_path / "sessions" / match.group_id / "manifest.json").read_text("utf-8"))
    res = man["battles_results"][0]
    assert res["p1_actor_type"] == "rl"
    assert res["battle_tag"] == "rl-vs-bot"

    meta = read_json(v5_dir_for(match, tmp_path) / "meta.json")
    assert meta["p1_is_bot"] is True

    actions = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    sources = {a.get("decision_source") for a in actions}
    assert "rl" in sources, f"decision_source 'rl' missing: {sources}"


def test_battle_tag_rl_vs_rl_when_p2_is_onnx_kind(tmp_path, monkeypatch):
    """p2 = v5 stub (onnx-kind) → battle_tag 'rl-vs-rl'. Без run_auto (stub raises).

    C1: the registry 'v5' slot now builds the real V5RlhfAdapter, which loads a
    real ONNX. This test needs the stub's raise-on-select behaviour, so it
    temporarily swaps the 'v5' slot back to the retained stub factory
    (``_factory_v5`` → ``V5StubAdapter``) for the duration of the build — the
    stub stays importable as a test double. The path is intentionally nonexistent
    (the stub never loads it)."""
    from rlhf_env.components.policy_adapters import default_registry, _factory_v5
    reg = default_registry()
    monkeypatch.setitem(reg._factories, "v5", _factory_v5)
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random",
        p2_model="random", p2_model_path="ai/models/fake_v5.onnx", p2_model_kind="v5",
        starting_player="p1",
    )
    assert match.bot_policy.kind == "v5"
    assert engine.battle_tag == "rl-vs-rl"


def test_human_path_p1_policy_none(tmp_path):
    """regression: human/llm — p1_policy is None, p1 не бот."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="human", starting_player="p1",
    )
    assert match.p1_policy is None
    assert engine.p1_actor_type == "human"
    assert engine.battle_tag == "human-vs-bot"
    state = engine._arena.state
    # p1 — человек (не бот)
    p1 = state.p1 if state.p1.user_id == engine.human_user_id else state.p2
    assert p1.is_bot is False


def test_llm_path_p1_policy_none(tmp_path):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="llm", starting_player="p1",
    )
    assert match.p1_policy is None
    assert engine.p1_actor_type == "llm"
    assert engine.battle_tag == "llm-vs-bot"


def test_surrender_unavailable_for_rl_p1(tmp_path):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", starting_player="p1",
    )
    resp = asyncio.run(runner.surrender())
    assert resp.get("error") == "surrender_unavailable_for_rl_p1"