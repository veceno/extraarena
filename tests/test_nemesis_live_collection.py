from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from battle_engine import BattleEngine
from core.state import GameStatus


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "ai" / "cards.json"
CATALOG_ROWS = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
CARD_CACHE = {int(card["id"]): card for card in CATALOG_ROWS}
HERO_ID = next(
    int(card["id"])
    for card in CATALOG_ROWS
    if card.get("card_type") == "hero"
)
UNIT_IDS = [
    int(card["id"])
    for card in CATALOG_ROWS
    if card.get("card_type") != "hero"
]
SIMPLIFIED_ID = next(
    int(card["id"])
    for card in CATALOG_ROWS
    if card.get("simplified_levelup")
)


def _deck(offset: int = 0, *, include_simplified: bool = False) -> list[int]:
    selected = [
        UNIT_IDS[(offset + index) % len(UNIT_IDS)]
        for index in range(8)
    ]
    if include_simplified and SIMPLIFIED_ID not in selected:
        selected[-1] = SIMPLIFIED_ID
    return [HERO_ID, *selected]


def _snapshot(wins: int = 7, losses: int = 3) -> dict:
    return {
        "captured_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "profile": {
            "wins": wins,
            "losses": losses,
            "trophies": 1234,
        },
        "summary": {
            "history_total": wins + losses,
            "total": wins + losses,
            "wins": wins,
            "losses": losses,
            "draws": 0,
            "win_rate": 70.0,
            "trophy_delta": 80,
            "avg_turns": 9.0,
            "avg_duration_seconds": 90.0,
        },
        "recent": [],
    }


class _DB:
    def __init__(self, *, fail_user_id: int | None = None) -> None:
        self.fail_user_id = fail_user_id
        self.snapshot_calls: list[int] = []

    async def get_user_cards(self, _user_id: int) -> list[dict]:
        return [
            {"id": int(card_id), "level": 6}
            for card_id in CARD_CACHE
        ]

    async def get_nemesis_profile_snapshot(
        self,
        user_id: int,
        *,
        history_limit: int,
    ) -> dict:
        assert history_limit == 32
        self.snapshot_calls.append(int(user_id))
        if int(user_id) == self.fail_user_id:
            raise RuntimeError("snapshot unavailable")
        return _snapshot()


class _Brain:
    def get_profile_provenance(
        self,
        difficulty: str,
        *,
        model_id: str | None = None,
    ) -> dict:
        return {
            "model_id": model_id,
            "model_family": "extra-lr-v5",
            "model_version": "h299",
            "checkpoint_id": "h299",
            "weights_hash": "a" * 64,
            "adapter_kind": f"v5:{difficulty}",
        }


class _Aux:
    def dataset_provenance(
        self,
        *,
        include_policy_assists: bool,
        include_metronome: bool,
    ) -> dict:
        assert include_metronome is True
        result = {
            "metronome": {
                "model_id": "extra-lr-metronome-v1",
                "model_family": "extra-lr-aux-v1",
                "model_version": "metronome",
                "checkpoint_id": "metronome",
                "weights_hash": "b" * 64,
                "adapter_kind": "onnx_aux_v1",
            }
        }
        if include_policy_assists:
            result["assembler"] = {
                "model_id": "extra-lr-assembler-v1",
                "model_family": "extra-lr-aux-v1",
                "model_version": "assembler",
                "checkpoint_id": "assembler",
                "weights_hash": "c" * 64,
                "adapter_kind": "onnx_aux_v1",
            }
        return result


async def _create(
    *,
    db: _DB,
    p2_bot: bool = False,
    trace_generation: int = 1,
) -> BattleEngine:
    engine = BattleEngine(
        db=db,
        match_id="nemesis-live",
        player_ids=[101, -202 if p2_bot else 202],
        active_matches={},
        card_cache=dict(CARD_CACHE),
    )
    engine.v5_dataset_generation = trace_generation
    engine.v5_dataset_generation_reason = (
        "initial" if trace_generation == 1 else "friendly_rehydrate"
    )
    engine.berserk_brain = _Brain()
    engine.extra_lr_aux_runtime = _Aux()
    p1_deck = _deck(0, include_simplified=True)
    p2_deck = _deck(12)
    result = await engine.create_match(
        "nemesis-live",
        {
            "user_id": 101,
            "deck_ids": p1_deck,
            "is_bot": False,
            "trophies": 1234,
        },
        {
            "user_id": -202 if p2_bot else 202,
            "deck_ids": p2_deck,
            "is_bot": p2_bot,
            "trophies": 1400,
            "difficulty": "tier_hard_4500",
            "brain_profile": "extra-lr-v5-ultra",
            "card_levels": [10] * len(p2_deck),
        },
        starting_player_id=101,
    )
    assert result["success"] is True
    return engine


def test_human_human_record_is_single_causal_standard_and_lite_base() -> None:
    db = _DB()
    engine = asyncio.run(_create(db=db))
    meta = engine.get_v5_dataset_snapshot()["meta"]
    record = meta["nemesis_record"]

    assert db.snapshot_calls == [101, 202]
    assert record["features"]["base"]["domain"] == "human-human"
    assert record["features"]["base"]["starting_player"] == "p1"
    assert record["features"]["extended"]["p1"]["profile"]["wins"] == 7
    assert record["features"]["extended"]["p2"]["profile"]["losses"] == 3
    assert record["quality"]["eligible_lite"] is True
    assert record["quality"]["eligible_standard"] is True
    assert record["provenance"]["split_group"].startswith("deck_pair:")
    assert record["features"]["base"]["catalog_hash"] == hashlib.sha256(
        CATALOG_PATH.read_bytes()
    ).hexdigest()
    simplified = next(
        card
        for card in record["features"]["base"]["seats"]["p1"]["initial_deck"]
        if card["card_id"] == SIMPLIFIED_ID
    )
    assert simplified["level"] == 2
    assert meta["nemesis_record"] == record

    engine._arena.state.status = GameStatus.P1_WIN
    terminal = engine.finalize_v5_dataset(
        winner_user_id=101,
        status="p1_win",
    )
    assert terminal["meta"]["nemesis_record"]["label"]["status"] == "p1_win"
    assert terminal["meta"]["nemesis_record"]["label"]["winner_seat"] == "p1"


def test_human_bot_is_lite_primary_with_human_extension_and_bot_provenance() -> None:
    db = _DB()
    engine = asyncio.run(_create(db=db, p2_bot=True))
    record = engine.get_v5_dataset_snapshot()["meta"]["nemesis_record"]

    assert db.snapshot_calls == [101]
    assert record["features"]["base"]["domain"] == "human-bot"
    assert record["features"]["extended"]["p1"] is not None
    assert record["features"]["extended"]["p2"] is None
    assert record["quality"]["eligible_lite"] is True
    assert record["quality"]["eligible_standard"] is False
    assert "human_bot_standard_auxiliary_only" in record["quality"][
        "exclusion_reasons"
    ]
    p2 = record["features"]["base"]["seats"]["p2"]
    assert p2["model_provenance"]["checkpoint_id"] == "h299"
    assert set(p2["aux_model_provenance"]) == {
        "assembler",
        "metronome",
    }


def test_human_bot_snapshot_failure_preserves_lite_record() -> None:
    engine = asyncio.run(_create(db=_DB(fail_user_id=101), p2_bot=True))
    meta = engine.get_v5_dataset_snapshot()["meta"]

    assert "nemesis_record" in meta
    record = meta["nemesis_record"]
    assert record["features"]["base"]["domain"] == "human-bot"
    assert record["features"]["extended"]["p1"] is None
    assert record["quality"]["eligible_lite"] is True
    assert record["quality"]["eligible_standard"] is False
    assert record["quality"]["exclusion_reasons"] == [
        "human_bot_standard_auxiliary_only"
    ]


def test_snapshot_failure_and_rehydrate_fail_closed_without_breaking_gameplay() -> None:
    failed = asyncio.run(_create(db=_DB(fail_user_id=202)))
    failed_record = failed.get_v5_dataset_snapshot()["meta"]["nemesis_record"]
    assert failed_record["quality"]["eligible_lite"] is True
    assert failed_record["quality"]["eligible_standard"] is False
    assert "p2_snapshot_unavailable" in failed_record["quality"][
        "exclusion_reasons"
    ]

    rehydrated = asyncio.run(_create(db=_DB(), trace_generation=2))
    rehydrated_record = rehydrated.get_v5_dataset_snapshot()["meta"][
        "nemesis_record"
    ]
    assert rehydrated_record["quality"]["eligible_lite"] is False
    assert rehydrated_record["quality"]["sample_weight"] == 0.0
    assert "rehydrated_trace_generation" in rehydrated_record["quality"][
        "exclusion_reasons"
    ]

    aborted = failed.abort_v5_dataset("server_reload")
    assert aborted["meta"]["status"] == "aborted"
    assert aborted["meta"]["nemesis_record"]["label"] is None
