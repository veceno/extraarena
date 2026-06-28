"""Block D — AgentRegistry + codenames unit tests (без полных игр — fast).

Покрывает:
  - claim/release/is_busy.
  - claim_auto из фиксированного пула; pool-exhaustion → random-fallback.
  - pin_group + persist + reload (agents_index.json).
  - status(name) aggregate из манифеста (после create_series).
  - кодовые имена включают фиксированный список + имена карт.
  - manifest top-level agent_name + per-battle agent_name + v5/meta.json agent_name.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlhf_env.components.agent_registry import (
    AgentRegistry,
    _FIXED_CODENAMES,
    _build_codename_pool,
)
from rlhf_env.tests._v5_helpers import make_manager, create_match, v5_dir_for, read_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CARDS_PATH = str(_REPO_ROOT / "ai" / "cards.json")


def test_claim_release_busy(tmp_path):
    # F11(audit): tmp_path вместо хардкода /tmp/_ar_busy.json — общий /tmp копил
    # leaked-имена между запусками (collision risk под parallel/cross-worktree).
    reg = AgentRegistry(index_path=tmp_path / "_ar_busy.json", cards_path=_CARDS_PATH)
    reg._busy.clear()
    assert reg.claim("Veceno") is True
    assert reg.is_busy("Veceno") is True
    assert reg.claim("Veceno") is False  # already busy
    reg.release("Veceno")
    assert reg.is_busy("Veceno") is False


def test_claim_auto_from_fixed_pool(tmp_path):
    # tmp_path — изоляция: общий /tmp/_ar_*.json копил leaked-имена между запусками
    # (claim без release персистит; _busy.clear() не чистит диск, _load перечитывает)
    # → фиксированный пул исчерпывался ~за 11 запусков, claim_auto возвращал card-имя.
    reg = AgentRegistry(index_path=tmp_path / "_ar_auto.json", cards_path=_CARDS_PATH)
    reg._busy.clear()
    name = reg.claim_auto()
    assert name in _FIXED_CODENAMES or name.startswith("Agent-")
    assert reg.is_busy(name) is True


def test_claim_auto_pool_exhaustion_random_fallback(tmp_path):
    reg = AgentRegistry(index_path=tmp_path / "_ar_exh.json", cards_path=_CARDS_PATH)
    reg._busy.clear()
    # исчерпать весь пул
    pool = _build_codename_pool(_CARDS_PATH)
    for n in pool:
        reg.claim(n)
    name = reg.claim_auto()
    assert name.startswith("Agent-")
    assert reg.is_busy(name) is True


def test_pin_group_persist_reload(tmp_path):
    idx = tmp_path / "agents_index.json"
    reg = AgentRegistry(index_path=idx, sessions_dir=tmp_path, cards_path=_CARDS_PATH)
    reg.claim("Mentalist")
    reg.pin_group("Mentalist", "grp123")
    assert reg.group_of("Mentalist") == "grp123"
    # reload from disk
    reg2 = AgentRegistry(index_path=idx, sessions_dir=tmp_path, cards_path=_CARDS_PATH)
    assert reg2.is_busy("Mentalist") is True
    assert reg2.group_of("Mentalist") == "grp123"


def test_release_group(tmp_path):
    reg = AgentRegistry(index_path=tmp_path / "idx.json", cards_path=_CARDS_PATH)
    reg._busy.clear()
    reg.claim("Sinaf")
    reg.pin_group("Sinaf", "g1")
    reg.release_group("g1")
    assert reg.is_busy("Sinaf") is False


def test_codename_pool_has_fixed_and_cards():
    pool = _build_codename_pool(_CARDS_PATH)
    for n in _FIXED_CODENAMES:
        assert n in pool
    # pool dedup case-insensitive
    lower = [n.lower() for n in pool]
    assert len(lower) == len(set(lower))


def test_status_aggregate_from_manifest(tmp_path):
    """create_series → manifest хранит agent_name (top-level + per-battle) + v5/meta.json."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", agent_name="Pvwell", starting_player="p1",
    )
    import asyncio
    asyncio.run(runner.run_auto())

    man_path = tmp_path / "sessions" / match.group_id / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["agent_name"] == "Pvwell"
    assert man["battles_results"][0]["agent_name"] == "Pvwell"
    assert man["battles_results"][0]["p1_actor_type"] == "rl"
    assert man["battles_results"][0]["battle_tag"] == "rl-vs-bot"

    # v5/meta.json agent_name + p1_is_bot
    meta = read_json(v5_dir_for(match, tmp_path) / "meta.json")
    assert meta["agent_name"] == "Pvwell"
    assert meta["p1_is_bot"] is True

    # status aggregate. Серия из 1 боя доиграна → manifest auto-finalized →
    # agent-leak фикс (L2 self-heal): агент освобождён, busy=False, status=completed.
    st = mgr.agent_registry.status("Pvwell")
    assert st["agent_name"] == "Pvwell"
    assert st["busy"] is False
    assert st["status"] == "completed"
    assert st["battles_finished"] == 1
    assert st["p1_actor_type"] == "rl"
    assert st["opponent_model"] == "random"


def test_finish_series_releases_agent(tmp_path):
    # multi-battle серия: после боя 0 (1/3) manifest НЕ finalized → агент ещё
    # genuinely busy (self-heal не срабатывает). finish_series досрочно закрывает
    # серию и освобождает имя.
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", agent_name="Movi",
        starting_player="p1", battles_planned=3,
    )
    import asyncio
    asyncio.run(runner.run_auto())  # доигрывает бой 0 (1/3) — серия не завершена
    assert mgr.agent_registry.is_busy("Movi") is True
    mgr.finish_series(match.group_id)
    assert mgr.agent_registry.is_busy("Movi") is False


def test_auto_assigned_codename_when_unspecified(tmp_path):
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random", starting_player="p1",
    )
    assert match.agent_name is not None
    assert mgr.agent_registry.is_busy(match.agent_name) is True