from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import pytest

from battle_engine import BattleEngine
from core.engine import ArenaEnvironment
from core.state import (
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
    ReplacementStatus,
)
from core.v5_dataset import V5DatasetRecorder
from infrastructure.database import _build_v5_export_bundle
from rlhf_env.components.v5_trace_validate import validate_v5_trace
from scripts.materialize_v5_dataset_export import materialize_export


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _card(
    card_id: int,
    *,
    card_type: CardType = CardType.WARRIOR,
    mana_cost: int = 1,
    attack: int = 2,
    hp: int = 3,
    ready: bool = False,
    level: int = 1,
) -> CardInstance:
    return CardInstance(
        card_id=card_id,
        name=f"Card {card_id}",
        card_type=card_type,
        mana_cost=mana_cost,
        attack=attack,
        hp=hp,
        max_hp=hp,
        mechanics=[],
        is_ready=ready,
        level=level,
    )


def _hero(card_id: int) -> CardInstance:
    return _card(
        card_id,
        card_type=CardType.HERO,
        mana_cost=0,
        attack=0,
        hp=30,
        ready=True,
    )


def _engine(
    *,
    p1_is_bot: bool = False,
    clock: _Clock | None = None,
    match_id: str = "prod-v5-test",
) -> BattleEngine:
    state = GameState(
        p1=PlayerState(
            user_id=101,
            is_bot=p1_is_bot,
            hero=_hero(1),
            mana=5,
            max_mana=5,
            hand=[_card(11, mana_cost=1, level=5)],
            deck=[_card(12, mana_cost=2, level=2)],
        ),
        p2=PlayerState(
            user_id=202,
            hero=_hero(2),
            mana=5,
            max_mana=5,
            hand=[_card(21, mana_cost=1)],
            deck=[_card(22, mana_cost=2)],
        ),
        current_turn_owner_id=101,
        turn_number=1,
        status=GameStatus.ONGOING,
    )
    engine = BattleEngine(match_id=match_id, player_ids=[101, 202])
    engine._arena = ArenaEnvironment(state, apply_start_effects=False)
    engine.current_player_id = 101
    engine.turn = 1
    if clock is not None:
        engine.v5_dataset_recorder = V5DatasetRecorder(
            monotonic_clock=clock,
            wall_clock=lambda: 1_800_000_000.0 + clock.value,
        )
    return engine


def test_direct_accepted_action_records_canonical_full_transition_and_label() -> None:
    engine = _engine()

    result = engine.play_card(101, 0, 0)

    assert result["success"] is True
    payload = engine.get_v5_dataset_snapshot()
    assert payload["schema_version"] == "rlhf_v5_storage_v1"
    assert payload["visibility"] == "omniscient_offline_only"
    assert payload["counts"] == {
        "turns": 1,
        "actions": 1,
        "accepted_actions": 1,
        "rejected_actions": 0,
        "training_labels": 1,
        "control_events": 0,
        "pending_actions": 0,
    }

    row = payload["actions"][0]
    assert row["seq"] == 1
    assert row["accepted"] is True
    assert row["error"] is None
    assert row["is_training_label"] is True
    assert row["legal_action_index"] is not None
    assert row["action_native"]["type"] == "play_card"
    assert row["training_action_native"] == row["action_native"]
    assert row["acting_user_id"] == row["actor_user_id"] == 101
    assert row["acting_player"] == row["actor_player"] == 1
    assert row["state_json"] == row["pre_state"]
    assert len(row["pre_state"]["p1"]["hand"]) == 1
    assert len(row["post_state"]["p1"]["hand"]) == 0
    assert row["pre_state"]["p2"]["hand"]
    assert row["pre_state"]["p2"]["deck"]
    assert row["human_decision_time_ms"] is None
    assert row["decision_time_censored"] is True
    assert row["decision_censor_reason"] == "not_observed"


def test_rejected_preflight_and_invalid_hand_are_retained_but_never_labels() -> None:
    engine = _engine()

    attack = engine.attack_target(
        101,
        attacker_id="missing-attacker",
        target_id=engine._arena.state.p2.hero.instance_id,
        target_is_hero=True,
    )
    bad_card = engine.play_card(101, "not-in-hand", 0)
    accepted = engine.end_turn(101)

    assert attack == {"success": False, "error": "attacker_not_found", "action": "attack"}
    assert bad_card["success"] is False
    assert bad_card["error"] == "card_not_found_in_hand"
    assert accepted["success"] is True

    rows = engine.get_v5_dataset_snapshot()["actions"]
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["accepted"] for row in rows] == [False, False, True]
    assert [row["is_training_label"] for row in rows] == [False, False, True]
    assert rows[0]["error"] == "attacker_not_found"
    assert rows[1]["error"] == "card_not_found_in_hand"
    assert rows[0]["pre_state"] == rows[0]["post_state"]
    assert rows[1]["pre_state"] == rows[1]["post_state"]
    assert rows[0]["training_action_native"] is None
    assert rows[1]["training_action_native"] is None


def test_invalid_hand_consumes_queued_context_before_the_next_valid_action() -> None:
    engine = _engine()
    engine.arm_human_decision_clock(101, now_monotonic=100.0)
    engine.record_analytics_action(
        101,
        {"type": "play_card", "card_ref": "missing"},
        request_monotonic=100.5,
        client_action_id="rejected-nonce",
    )
    assert engine.play_card(101, "missing", 0)["success"] is False

    engine.record_analytics_action(
        101,
        {"type": "end_turn"},
        request_monotonic=100.9,
        client_action_id="accepted-nonce",
    )
    assert engine.end_turn(101)["success"] is True

    rows = engine.get_v5_dataset_snapshot()["actions"]
    assert [row["client_action_id"] for row in rows] == [
        "rejected-nonce",
        "accepted-nonce",
    ]
    assert [row["action_type"] for row in rows] == ["play_card", "end_turn"]
    assert [row["accepted"] for row in rows] == [False, True]
    assert engine._analytics_actions[0]["context_json"]["accepted"] is False
    assert engine._analytics_actions[1]["context_json"]["accepted"] is True


def test_legacy_next_action_context_captures_human_timing_and_metronome_fields() -> None:
    clock = _Clock(100.0)
    engine = _engine(clock=clock)
    engine.arm_human_decision_clock(101, now_monotonic=100.0)

    captured = engine.record_analytics_action(
        101,
        {"type": "end_turn"},
        request_monotonic=100.8,
        client_action_id="nonce-1",
        metronome_prediction_ms=850.0,
        metronome_applied_ms=800.0,
    )
    result = engine.end_turn(101)

    assert result["success"] is True
    assert captured is not None
    row = engine.get_v5_dataset_snapshot()["actions"][0]
    assert row["decision_source"] == "human"
    assert row["control_source"] == "human"
    assert row["actor_type"] == "human"
    assert row["human_decision_time_ms"] == pytest.approx(800.0)
    assert row["human_decision_time_raw_ms"] == pytest.approx(800.0)
    assert row["decision_time_censored"] is False
    assert row["decision_censor_reason"] is None
    assert row["client_action_id"] == "nonce-1"
    assert row["metronome_prediction_ms"] == 850.0
    assert row["metronome_applied_ms"] == 800.0
    assert engine._analytics_actions[0]["context_json"]["accepted"] is True
    assert (
        engine._analytics_actions[0]["context_json"]["v5_action_context"][
            "human_decision_time_ms"
        ]
        == pytest.approx(800.0)
    )


def test_replacement_and_timeout_use_bot_decision_source_with_detailed_control_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = _engine()
    replacement.set_player_replacement_status(
        101, ReplacementStatus.AFK, reason="disconnect"
    )
    replacement.record_analytics_action(
        101,
        {"type": "end_turn"},
        metronome_prediction_ms=700.0,
        metronome_applied_ms=650.0,
    )
    assert replacement.end_turn(101)["success"] is True
    replacement_row = replacement.get_v5_dataset_snapshot()["actions"][0]
    assert replacement_row["decision_source"] == "bot"
    assert replacement_row["control_source"] == "replacement_bot"
    assert replacement_row["actor_type"] == "bot"
    assert replacement_row["human_decision_time_ms"] is None
    assert replacement_row["decision_time_censored"] is False
    assert replacement_row["decision_censor_reason"] is None
    assert replacement.get_v5_dataset_snapshot()["control_events"][0][
        "new_status"
    ] == "afk"

    timeout = _engine()
    monkeypatch.setattr(timeout, "is_turn_expired", lambda: True)
    timeout.record_analytics_action(101, {"type": "end_turn"})
    assert timeout.end_turn(101)["success"] is True
    timeout_row = timeout.get_v5_dataset_snapshot()["actions"][0]
    assert timeout_row["decision_source"] == "bot"
    assert timeout_row["control_source"] == "timeout"
    assert timeout_row["actor_type"] == "bot"
    assert timeout_row["action_type"] == "end_turn"


def test_timing_outside_training_window_is_censored_but_raw_value_is_retained() -> None:
    engine = _engine()
    engine.arm_human_decision_clock(101, now_monotonic=10.0)
    engine.record_analytics_action(
        101,
        {"type": "end_turn"},
        request_monotonic=40.0,
    )

    assert engine.end_turn(101)["success"] is True
    row = engine.get_v5_dataset_snapshot()["actions"][0]
    assert row["human_decision_time_ms"] is None
    assert row["human_decision_time_raw_ms"] == 30_000.0
    assert row["decision_time_censored"] is True
    assert row["decision_censor_reason"] == "outside_training_window"


def test_checkpoint_finalize_abort_and_snapshot_are_detached_and_shared_safe() -> None:
    clock = _Clock(100.0)
    engine = _engine(clock=clock)
    engine.set_v5_dataset_metadata(
        {
            "p1_deck": [
                {"card_id": 1, "level": 1, "instance_id": None, "slot": 0},
                {"card_id": 11, "level": 5, "instance_id": None, "slot": 1},
            ],
            "model_provenance": {"p2": {"model_id": "extra-lr-v5-ultra"}},
            "aux_model_provenance": {
                "assembler": {"model_id": "extra-lr-assembler-v1"},
                "cardoptimum": {"model_id": "extra-lr-cardoptimum-v1"},
            },
        }
    )
    engine.mark_v5_battle_started(
        reason="client_ready",
        now_monotonic=101.0,
        now_wall=1_800_000_101.0,
    )
    engine.mark_v5_battle_started(
        reason="duplicate_client_ready",
        now_monotonic=104.0,
        now_wall=1_800_000_104.0,
    )

    checkpoints: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        checkpoints.extend(
            pool.map(
                lambda _: engine.checkpoint_v5_dataset(reason="periodic"),
                range(12),
            )
        )
    assert all(item["counts"]["pending_actions"] == 0 for item in checkpoints)

    clock.value = 106.25
    engine._arena.state.status = GameStatus.P1_WIN
    finalized = engine.finalize_v5_dataset(
        winner_user_id=101,
        status="p1_win",
        reason="hero_death",
    )
    assert finalized["finalized"] is True
    assert finalized["meta"]["duration_seconds"] == pytest.approx(5.25)
    assert finalized["meta"]["match_start_unix_ms"] == 1_800_000_101_000
    assert (
        finalized["meta"]["start_metadata"]["duplicate_start_anchor_count"]
        == 1
    )
    assert finalized["meta"]["winner_user_id"] == 101
    assert finalized["meta"]["timestamp_features"]["turns"] == 1
    assert finalized["meta"]["p1_deck"][1]["level"] == 5
    assert finalized["meta"]["aux_model_provenance"]["assembler"]["model_id"].endswith(
        "assembler-v1"
    )

    finalized["meta"]["status"] = "corrupted-by-consumer"
    assert engine.get_v5_dataset_snapshot()["meta"]["status"] == "p1_win"
    # Finalization wins over a late abort and remains idempotent.
    late_abort = engine.abort_v5_dataset("server_reload")
    assert late_abort["meta"]["status"] == "p1_win"
    assert late_abort["aborted"] is False

    unfinished = _engine()
    aborted = unfinished.abort_v5_dataset("server_reload")
    assert aborted["finalized"] is True
    assert aborted["aborted"] is True
    assert aborted["meta"]["status"] == "aborted"
    assert aborted["meta"]["abort_reason"] == "server_reload"


def test_v5_policy_degradation_is_monotonic_and_secret_free() -> None:
    engine = _engine()

    warning = engine.mark_v5_policy_degraded("decode_failed")
    engine.set_v5_dataset_metadata(
        {
            "degraded": False,
            "policy_warnings": [],
        }
    )
    snapshot = engine.get_v5_dataset_snapshot()

    assert warning == "v5_policy_failure:decode_failed"
    assert snapshot["meta"]["degraded"] is True
    assert snapshot["meta"]["policy_warnings"] == [warning]


def test_production_snapshot_passes_the_shared_v5_trace_validator(
    tmp_path: Path,
) -> None:
    engine = _engine()
    engine.arm_human_decision_clock(101, now_monotonic=100.0)
    engine.record_analytics_action(
        101,
        {"type": "play_card", "card_ref": 0},
        request_monotonic=100.6,
    )
    assert engine.play_card(101, 0, 0)["success"] is True
    payload = engine.get_v5_dataset_snapshot()

    v5_dir = tmp_path / "v5"
    v5_dir.mkdir()
    (v5_dir / "meta.json").write_text(
        json.dumps(payload["meta"], ensure_ascii=False),
        encoding="utf-8",
    )
    (v5_dir / "turns.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in payload["turns"]
        ),
        encoding="utf-8",
    )
    (v5_dir / "actions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in payload["actions"]
        ),
        encoding="utf-8",
    )

    report = validate_v5_trace(v5_dir)
    assert report["ok"] is True, report["issues"]


def test_terminal_human_vs_bot_snapshot_materializes_after_prod_pseudonymization(
    tmp_path: Path,
) -> None:
    engine = _engine(match_id="tutorial-987654321")
    state = engine._arena.state
    state.p2.is_bot = True
    attacker = _card(13, attack=5, hp=4, ready=True)
    state.p1.board.append(attacker)
    state.p2.hero.hp = 1
    state.p2.hero.max_hp = 30

    readiness = engine.mark_client_ready(101)
    assert readiness["all_ready"] is True
    engine.record_analytics_action(
        101,
        {
            "type": "attack",
            "attacker_id": str(attacker.instance_id),
            "target_id": None,
            "target_is_hero": True,
        },
        decision_source="human",
        client_action_id="terminal-human-attack",
    )
    result = engine.attack_target(
        101,
        attacker.instance_id,
        None,
        target_is_hero=True,
    )
    assert result["game_over"] is True
    assert result["winner_id"] == 101

    payload = engine.get_v5_dataset_snapshot()
    assert payload["meta"]["status"] == "p1_win"
    assert payload["meta"]["battle_tag"] == "human-vs-bot"
    source = tmp_path / "production-export.jsonl"
    header = {
        "record_type": "header",
        "format": "extraarena_v5_dataset_export_v1",
        "format_version": 1,
        "storage_schema": "rlhf_v5_storage_v1",
        "created_at": payload["meta"]["finished_at"],
        "privacy": "side_pseudonyms_p1_1_p2_2",
        "include_players": False,
        "record_id_scheme": "random_per_export_record_ids_v1",
        "days": 1,
        "limit_battles": 1,
        "battle_count": 1,
        "skipped_invalid": 0,
    }
    bundle = _build_v5_export_bundle(
        {
        "battle_id": engine.match_id,
        "storage_schema": "rlhf_v5_storage_v1",
        "status": payload["meta"]["status"],
        "finished_at": payload["meta"]["finished_at"],
        "winner_user_id": payload["meta"]["winner_user_id"],
        "p1_user_id": payload["meta"]["p1_user_id"],
        "p2_user_id": payload["meta"]["p2_user_id"],
        "p1_actor_type": payload["meta"]["p1_actor_type"],
        "p2_actor_type": payload["meta"]["p2_actor_type"],
        "action_count": len(payload["actions"]),
        "meta_json": payload["meta"],
        "turns_json": payload["turns"],
        "actions_json": payload["actions"],
        },
        include_players=False,
    )
    assert bundle is not None
    exported_battle_id = bundle["battle_id"]
    assert re.fullmatch(r"record_[0-9a-f]{32}", exported_battle_id)
    assert bundle["meta"]["battle_id"] == exported_battle_id
    assert bundle["meta"]["match_id"] == exported_battle_id
    assert "987654321" not in json.dumps(bundle, sort_keys=True)
    bundle = {"record_type": "battle", **bundle}
    source.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (header, bundle)
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "materialized"
    manifest = materialize_export(
        source,
        output,
        group_id="production-hvb-e2e",
    )

    assert manifest["battle_ids"] == [exported_battle_id]
    assert manifest["battles_results"][0]["collection_class"] == "human-vs-bot"
    report = validate_v5_trace(
        output / "battles" / exported_battle_id / "v5"
    )
    assert report["ok"] is True, report["issues"]
