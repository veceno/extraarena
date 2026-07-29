from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlhf_env.components.v5_trace_validate import validate_v5_trace
from scripts.materialize_v5_dataset_export import (
    MaterializationError,
    main,
    materialize_export,
)


def _state(*, owner: int = 1) -> dict:
    return {
        "turn_number": 1,
        "current_turn_owner_id": owner,
        "status": "ongoing",
        "p1": {
            "user_id": 1,
            "is_bot": False,
            "replacement_status": "active",
        },
        "p2": {
            "user_id": 2,
            "is_bot": True,
            "replacement_status": "active",
        },
        "action_history": [],
        "history": [],
        "v5_history_events": [],
        "pending_card_feedback_events": [],
    }


def _bundle(*, battle_id: str = "battle-1", status: str = "p2_win") -> dict:
    state = _state()
    meta = {
        "schema_version": "rlhf_v5_storage_v1",
        "visibility": "omniscient_offline_only",
        "battle_id": battle_id,
        "match_id": battle_id,
        "created_at": "2026-07-28T10:00:00Z",
        "started_at": "2026-07-28T10:00:05Z",
        "finished_at": "2026-07-28T10:01:05Z",
        "status": status,
        "winner_user_id": 2 if status == "p2_win" else None,
        "duration_seconds": 60.0,
        "turns": 1,
        "p1_user_id": 1,
        "p2_user_id": 2,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "p1_is_bot": False,
        "p2_is_bot": True,
        "battle_tag": "human-vs-bot",
        "p1_deck": [{"card_id": 1, "level": 1, "instance_id": None}],
        "p2_deck": [{"card_id": 2, "level": 1, "instance_id": None}],
        "start_metadata": {
            "client_ready_anchored": True,
            "anchor_reason": "client_ready",
        },
        "timestamp_features": {
            "p1_deck_size": 1,
            "p2_deck_size": 1,
            "starting_player": "p1",
            "duration_seconds": 60.0,
            "turns": 1,
        },
    }
    action = {
        "seq": 1,
        "battle_id": battle_id,
        "turn_number": 1,
        "actor_user_id": 1,
        "actor_player": 1,
        "decision_source": "human",
        "control_source": "human",
        "human_decision_time_ms": 1200,
        "human_decision_time_raw_ms": 1200,
        "decision_time_censored": False,
        "decision_censor_reason": None,
        "metronome_prediction_ms": None,
        "metronome_applied_ms": None,
        "metronome_fallback_used": None,
        "legal_action_index": None,
        "action_type": "surrender",
        "action_json": {"type": "surrender", "reason": "player_surrender"},
        "action_native": None,
        "source_card": None,
        "target_card": None,
        "legal_actions": [],
        "legal_action_count": 0,
        "pre_state": state,
        "post_state": state,
        "deltas": None,
        "accepted": True,
        "error": None,
        "timestamp_ms": 1_000,
    }
    return {
        "record_type": "battle",
        "battle_id": battle_id,
        "storage_schema": "rlhf_v5_storage_v1",
        "status": status,
        "finished_at": "2026-07-28T10:01:05Z",
        "meta": meta,
        "turns": [state],
        "actions": [action],
    }


def _write_export(path: Path, battles: list[dict], *, count: int | None = None) -> None:
    header = {
        "record_type": "header",
        "format": "extraarena_v5_dataset_export_v1",
        "format_version": 1,
        "storage_schema": "rlhf_v5_storage_v1",
        "created_at": "2026-07-28T12:00:00Z",
        "privacy": "side_pseudonyms_p1_1_p2_2",
        "include_players": False,
        "days": 30,
        "limit_battles": 1000,
        "battle_count": len(battles) if count is None else count,
        "skipped_invalid": 0,
    }
    path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (header, *battles)
        )
        + "\n",
        encoding="utf-8",
    )


def test_materializes_canonical_layout_and_deep_validates(tmp_path: Path) -> None:
    source = tmp_path / "export.jsonl"
    output = tmp_path / "production-group"
    _write_export(source, [_bundle()])

    manifest = materialize_export(
        source,
        output,
        group_id="production-human-20260728",
    )

    v5_dir = output / "battles" / "battle-1" / "v5"
    assert (v5_dir / "meta.json").is_file()
    assert (v5_dir / "turns.jsonl").is_file()
    assert (v5_dir / "actions.jsonl").is_file()
    assert validate_v5_trace(v5_dir)["ok"] is True

    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["schema_version"] == "rlhf_v5_storage_v1"
    assert manifest["battle_ids"] == ["battle-1"]
    assert manifest["results"]["battles_finished"] == 1
    assert manifest["battles_results"][0] == {
        "battle_id": "battle-1",
        "winner_user_id": 2,
        "loser_user_id": 1,
        "status": "P2_WIN",
        "turns": 1,
        "duration_seconds": 60.0,
        "battle_tag": "human-vs-bot",
        "collection_class": "human-vs-bot",
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "v5_dir": "battles/battle-1/v5",
        "v5_meta_path": "battles/battle-1/v5/meta.json",
        "v5_trace_ok": True,
        "validation_scope": "v5_trace_without_legacy_battle_log",
        "finished_at": "2026-07-28T10:01:05Z",
    }


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda battle: battle.update(status="ongoing"),
            "non-terminal",
        ),
        (
            lambda battle: battle.update(battle_id="../escape"),
            "safe path component",
        ),
        (
            lambda battle: battle.update(actions=[]),
            "actions must be a non-empty",
        ),
        (
            lambda battle: battle["meta"]["start_metadata"].update(
                client_ready_anchored=False
            ),
            "anchored to client_ready",
        ),
        (
            lambda battle: battle["actions"][0].pop(
                "metronome_prediction_ms"
            ),
            "timing fields missing",
        ),
        (
            lambda battle: battle["meta"].update(started_at="not-a-date"),
            "valid timezone-aware",
        ),
        (
            lambda battle: battle["meta"].update(
                started_at="2026-07-28T10:02:05Z"
            ),
            "precedes",
        ),
        (
            lambda battle: battle["meta"]["timestamp_features"].update(
                duration_seconds=-123
            ),
            "must match meta.duration_seconds",
        ),
        (
            lambda battle: (
                battle["meta"].update(duration_seconds=999_999_999),
                battle["meta"]["timestamp_features"].update(
                    duration_seconds=999_999_999
                ),
            ),
            "inconsistent with started_at/finished_at",
        ),
        (
            lambda battle: battle["meta"]["timestamp_features"].update(
                p1_deck_size=50
            ),
            "p1_deck_size must match",
        ),
    ],
)
def test_rejects_unsafe_or_incomplete_bundles_without_publish(
    tmp_path: Path,
    mutate,
    needle: str,
) -> None:
    source = tmp_path / "bad.jsonl"
    output = tmp_path / "must-not-exist"
    battle = _bundle()
    mutate(battle)
    _write_export(source, [battle])

    with pytest.raises(MaterializationError, match=needle):
        materialize_export(source, output, group_id="safe-group")

    assert not output.exists()
    assert list(tmp_path.glob(".must-not-exist.tmp-*")) == []


def test_header_count_mismatch_does_not_replace_existing_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad-count.jsonl"
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    _write_export(source, [_bundle()], count=2)

    with pytest.raises(MaterializationError, match="battle_count=2"):
        materialize_export(
            source,
            output,
            group_id="safe-group",
            overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (output / "manifest.json").exists()


def test_cli_overwrite_replaces_only_after_success(tmp_path: Path) -> None:
    source = tmp_path / "export.jsonl"
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    _write_export(source, [_bundle()])

    assert (
        main(
            [
                str(source),
                str(output),
                "--group-id",
                "cli-group",
                "--overwrite",
            ]
        )
        == 0
    )
    assert not (output / "old.txt").exists()
    assert (output / "manifest.json").is_file()


def test_duplicate_battle_id_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    output = tmp_path / "dataset"
    _write_export(source, [_bundle(), _bundle()], count=2)

    with pytest.raises(MaterializationError, match="duplicate battle_id"):
        materialize_export(source, output, group_id="safe-group")

    assert not output.exists()
