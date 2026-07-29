from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from core.nemesis_dataset import (
    NEMESIS_EXPORT_FORMAT,
    NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME,
    NEMESIS_RAW_RECORD_ID_SCHEME,
    NemesisBattleCollector,
    NemesisContractError,
    export_nemesis_ndjson,
    validate_nemesis_record,
    write_nemesis_export,
)


def _meta(
    *,
    p1_actor: str = "human",
    p2_actor: str = "bot",
) -> dict:
    return {
        "battle_id": "v5g2-0123456789abcdef0123456789abcdef",
        "match_id": "gameplay-match",
        "started_at": "2026-07-28T12:00:01Z",
        "game_mode": "classic",
        "ruleset": "classic",
        "catalog_hash": "catalog-sha256",
        "card_params_schema": "train_v3_card_params_v1",
        "deck_params_schema": "train_v3_deck_params_v1",
        "starting_player": "p1",
        "p1_user_id": 101,
        "p2_user_id": -5001,
        "p1_actor_type": p1_actor,
        "p2_actor_type": p2_actor,
        "p1_deck": [
            {"slot": 0, "card_id": 1, "level": 5},
            {"slot": 1, "card_id": 11, "level": 3},
        ],
        "p2_deck": [
            {"slot": 0, "card_id": 2, "level": 5},
            {"slot": 1, "card_id": 12, "level": 4},
        ],
        "model_provenance": {
            "p1": None,
            "p2": {
                "model_id": "extra-lr-v5-ultra",
                "checkpoint_id": "h299",
                "weights_hash": "weights-sha256",
                "adapter_kind": "v5",
            },
        },
        "aux_model_provenance": {
            "p2": {
                "assembler": {
                    "model_id": "extra-lr-assembler-v1",
                    "checkpoint_id": "assembler-final",
                },
                "cardoptimum": {
                    "model_id": "extra-lr-cardoptimum-v1",
                    "checkpoint_id": "cardoptimum-final",
                },
            }
        },
        # These terminal fields must never be copied into features.
        "status": "p2_win",
        "winner_user_id": -5001,
        "duration_seconds": 99,
        "turns": 17,
    }


def _extended() -> dict:
    return {
        "captured_at": "2026-07-28T12:00:00Z",
        "profile": {"wins": 10, "losses": 4, "trophies": 1200},
        "summary": {
            "history_total": 16,
            "total": 14,
            "wins": 10,
            "losses": 4,
            "draws": 0,
            "win_rate": 71.4,
            "avg_turns": 9.2,
            "avg_duration_seconds": 83,
            "favorite_mode": "classic",
            "current_streak_result": "win",
            "current_streak_count": 2,
            "current_win_streak": 2,
            "max_win_streak": 5,
        },
        "recent": [
            {
                "result": "win",
                "opponent_actor_type": "bot",
                "game_mode": "classic",
                "completed_at": "2026-07-28T11:59:00Z",
                "duration_seconds": 80,
                "turns_count": 9,
                "trophy_change": 20,
                "started_first": True,
            }
        ],
    }


def test_collector_freezes_lite_and_nullable_extended_without_outcome_leakage() -> None:
    collector = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
        extended_by_seat={"p1": _extended(), "p2": None},
    )
    open_record = collector.snapshot()

    assert open_record["features"]["base"]["domain"] == "human-bot"
    assert open_record["features"]["base"]["seats"]["p1"]["initial_deck"][1] == {
        "slot": 1,
        "card_id": 11,
        "level": 3,
    }
    provenance = open_record["features"]["base"]["seats"]["p2"]["model_provenance"]
    assert provenance["model_id"] == "extra-lr-v5-ultra"
    assert provenance["checkpoint_id"] == "h299"
    assert (
        open_record["features"]["base"]["seats"]["p2"]["aux_model_provenance"][
            "assembler"
        ]["checkpoint_id"]
        == "assembler-final"
    )
    assert open_record["features"]["extended"]["p2"] is None
    assert open_record["label"] is None
    serialized_features = json.dumps(open_record["features"])
    assert "winner_user_id" not in serialized_features
    assert "duration_seconds\": 99" not in serialized_features

    terminal = collector.finalize(status="p1_win")
    assert terminal["label"] == {
        "status": "p1_win",
        "winner_seat": "p1",
        "duration_seconds": None,
        "turns_count": None,
    }
    assert collector.finalize(status="p1_win") == terminal
    with pytest.raises(NemesisContractError, match="conflict"):
        collector.finalize(status="p2_win")


@pytest.mark.parametrize(
    ("actors", "domain"),
    [
        (("human", "human"), "human-human"),
        (("human", "rl"), "human-bot"),
        (("llm", "rl"), "model-model"),
    ],
)
def test_explicit_domain_is_derived_from_actor_provenance(
    actors: tuple[str, str],
    domain: str,
) -> None:
    collector = NemesisBattleCollector.from_v5_meta(
        _meta(p1_actor=actors[0], p2_actor=actors[1]),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    )
    assert collector.snapshot()["features"]["base"]["domain"] == domain


def test_extended_snapshot_must_be_time_causal() -> None:
    extended = _extended()
    extended["captured_at"] = "2026-07-28T12:00:00.500Z"
    with pytest.raises(NemesisContractError, match="after feature_cutoff"):
        NemesisBattleCollector.from_v5_meta(
            _meta(),
            feature_cutoff_at="2026-07-28T12:00:00Z",
            extended_by_seat={"p1": extended},
        )

    extended = _extended()
    extended["recent"][0]["completed_at"] = "2026-07-28T12:00:00.100Z"
    with pytest.raises(NemesisContractError, match="not time-causal"):
        NemesisBattleCollector.from_v5_meta(
            _meta(),
            feature_cutoff_at="2026-07-28T12:00:00Z",
            extended_by_seat={"p1": extended},
        )

    with pytest.raises(NemesisContractError, match="must not follow"):
        NemesisBattleCollector.from_v5_meta(
            _meta(),
            feature_cutoff_at="2026-07-28T12:00:02Z",
        )


def test_extended_block_is_nullable_and_hidden_terminal_fields_are_rejected() -> None:
    collector = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    )
    record = collector.snapshot()
    assert record["features"]["extended"] is None

    record["winner_user_id"] = 101
    with pytest.raises(NemesisContractError, match="unexpected top-level"):
        export_nemesis_ndjson(
            [
                {
                    **record,
                    "label": {
                        "status": "p1_win",
                        "winner_seat": "p1",
                        "duration_seconds": None,
                        "turns_count": None,
                    },
                }
            ]
        )


def test_export_is_whole_battle_terminal_and_pseudonymized_by_default() -> None:
    collector = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    )
    with pytest.raises(NemesisContractError, match="terminal label"):
        export_nemesis_ndjson([collector.snapshot()])

    payload = export_nemesis_ndjson([collector.finalize(status="p2_win")])
    lines = [json.loads(line) for line in payload.splitlines()]
    assert lines[0]["format"] == NEMESIS_EXPORT_FORMAT
    assert lines[0]["battle_count"] == 1
    assert (
        lines[0]["record_id_scheme"]
        == NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
    )
    battle = lines[1]
    assert battle["record_type"] == "battle"
    assert battle["privacy"]["identity_scheme"] == "side_pseudonyms_p1_1_p2_2"
    assert battle["features"]["base"]["seats"]["p1"]["participant_id"] == 1
    assert battle["features"]["base"]["seats"]["p2"]["participant_id"] == 2
    assert re.fullmatch(r"record_[0-9a-f]{32}", battle["battle_id"])
    assert re.fullmatch(r"record_[0-9a-f]{32}", battle["match_id"])
    assert battle["privacy"]["record_id_scheme"] == (
        NEMESIS_PSEUDONYMIZED_RECORD_ID_SCHEME
    )
    aliases = battle["privacy"]["player_group_aliases"]
    assert re.fullmatch(r"player_[0-9a-f]{32}", aliases["p1"])
    assert re.fullmatch(r"player_[0-9a-f]{32}", aliases["p2"])
    assert aliases["p1"] != aliases["p2"]
    assert battle["label"]["status"] == "p2_win"
    assert battle["label"]["winner_seat"] == "p2"


def test_catalog_unavailable_is_explicit_lite_only_quality_policy() -> None:
    meta = _meta()
    meta["catalog_hash"] = None
    record = NemesisBattleCollector.from_v5_meta(
        meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="draw", duration_seconds=90, turns_count=8)
    assert record["features"]["base"]["catalog_available"] is False
    assert record["quality"] == {
        "eligible_lite": True,
        "eligible_standard": False,
        "sample_weight": 0.5,
        "exclusion_reasons": [
            "catalog_unavailable",
            "human_bot_standard_auxiliary_only",
        ],
    }
    assert record["provenance"]["split_group"].startswith("deck_pair:")
    assert len(record["provenance"]["split_fingerprint"]) == 64
    assert record["label"]["duration_seconds"] == 90


def test_standard_eligibility_requires_two_human_pre_match_snapshots() -> None:
    human_human = _meta(p1_actor="human", p2_actor="human")
    complete = NemesisBattleCollector.from_v5_meta(
        human_human,
        feature_cutoff_at="2026-07-28T12:00:00Z",
        extended_by_seat={"p1": _extended(), "p2": _extended()},
    ).snapshot()
    assert complete["quality"]["eligible_standard"] is True

    missing = NemesisBattleCollector.from_v5_meta(
        human_human,
        feature_cutoff_at="2026-07-28T12:00:00Z",
        extended_by_seat={"p1": _extended(), "p2": None},
    ).snapshot()
    assert missing["quality"]["eligible_lite"] is True
    assert missing["quality"]["eligible_standard"] is False
    assert missing["quality"]["exclusion_reasons"] == [
        "p2_snapshot_unavailable"
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("profile", "telegram_id"), 987654321),
        (("recent", 0, "opponent_user_id"), 123456789),
    ],
)
def test_standard_extension_rejects_unexpected_identity_fields(
    path,
    value,
) -> None:
    human_human = _meta(p1_actor="human", p2_actor="human")
    record = NemesisBattleCollector.from_v5_meta(
        human_human,
        feature_cutoff_at="2026-07-28T12:00:00Z",
        extended_by_seat={"p1": _extended(), "p2": _extended()},
    ).snapshot()
    target = record["features"]["extended"]["p1"]
    if path[0] == "profile":
        target["profile"][path[1]] = value
    else:
        target["recent"][path[1]][path[2]] = value

    with pytest.raises(
        NemesisContractError,
        match="non-canonical or unexpected",
    ):
        validate_nemesis_record(record)


def test_rehydrated_generation_is_retained_but_ineligible() -> None:
    meta = _meta(p1_actor="rl", p2_actor="rl")
    meta["dataset_generation"] = 2
    meta["dataset_generation_reason"] = "friendly_rehydrate"
    record = NemesisBattleCollector.from_v5_meta(
        meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).snapshot()
    assert record["quality"]["eligible_lite"] is False
    assert record["quality"]["eligible_standard"] is False
    assert record["quality"]["sample_weight"] == 0.0
    assert "rehydrated_trace_generation" in record["quality"][
        "exclusion_reasons"
    ]


def test_authorized_export_retains_ids_and_atomic_writer(tmp_path: Path) -> None:
    record = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="draw")
    destination = tmp_path / "nested" / "nemesis.jsonl"

    result = write_nemesis_export(
        [record],
        destination,
        include_players=True,
    )

    assert result == destination
    lines = [json.loads(line) for line in destination.read_text().splitlines()]
    assert lines[0]["identity_scheme"] == "raw_player_ids"
    assert lines[0]["record_id_scheme"] == NEMESIS_RAW_RECORD_ID_SCHEME
    assert lines[1]["features"]["base"]["seats"]["p1"]["participant_id"] == 101
    assert lines[1]["features"]["base"]["seats"]["p2"]["participant_id"] == -5001
    assert lines[1]["battle_id"] == record["battle_id"]
    assert lines[1]["match_id"] == record["match_id"]
    assert lines[1]["privacy"]["player_group_aliases"] == {
        "p1": "101",
        "p2": "-5001",
    }


def test_player_group_aliases_are_export_local_and_stable() -> None:
    first = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="p1_win")
    second_meta = _meta()
    second_meta["battle_id"] = "second-battle"
    second_meta["match_id"] = "second-match"
    second = NemesisBattleCollector.from_v5_meta(
        second_meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="p2_win")
    swapped_meta = _meta()
    swapped_meta["battle_id"] = "swapped-battle"
    swapped_meta["match_id"] = "swapped-match"
    swapped_meta["p1_user_id"], swapped_meta["p2_user_id"] = (
        swapped_meta["p2_user_id"],
        swapped_meta["p1_user_id"],
    )
    swapped = NemesisBattleCollector.from_v5_meta(
        swapped_meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="draw")

    first_export = [
        json.loads(line)
        for line in export_nemesis_ndjson(
            [first, second, swapped]
        ).splitlines()
    ][1:]
    second_export = [
        json.loads(line)
        for line in export_nemesis_ndjson([first]).splitlines()
    ][1]

    assert (
        first_export[0]["privacy"]["player_group_aliases"]
        == first_export[1]["privacy"]["player_group_aliases"]
    )
    assert (
        first_export[0]["privacy"]["player_group_aliases"]["p1"]
        == first_export[2]["privacy"]["player_group_aliases"]["p2"]
    )
    assert (
        first_export[0]["privacy"]["player_group_aliases"]["p2"]
        == first_export[2]["privacy"]["player_group_aliases"]["p1"]
    )
    assert (
        first_export[0]["privacy"]["player_group_aliases"]
        != second_export["privacy"]["player_group_aliases"]
    )
    reexported = [
        json.loads(line)
        for line in export_nemesis_ndjson(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "record_type"
                }
                for row in first_export
            ]
        ).splitlines()
    ][1:]
    assert (
        reexported[0]["privacy"]["player_group_aliases"]
        == reexported[1]["privacy"]["player_group_aliases"]
    )
    assert (
        reexported[0]["privacy"]["player_group_aliases"]["p1"]
        == reexported[2]["privacy"]["player_group_aliases"]["p2"]
    )


def test_pseudonymized_export_removes_user_id_bearing_record_ids() -> None:
    meta = _meta()
    meta["battle_id"] = "tutorial-987654321"
    meta["match_id"] = "tutorial-987654321"
    record = NemesisBattleCollector.from_v5_meta(
        meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="p1_win")

    payload = export_nemesis_ndjson([record])
    lines = [json.loads(line) for line in payload.splitlines()]
    battle = lines[1]

    assert battle["battle_id"] == battle["match_id"]
    assert re.fullmatch(r"record_[0-9a-f]{32}", battle["battle_id"])
    assert "987654321" not in payload


def test_pseudonymized_privacy_record_ids_fail_closed() -> None:
    record = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="p1_win")
    exported = json.loads(
        export_nemesis_ndjson([record]).splitlines()[1]
    )
    exported.pop("record_type")
    exported["match_id"] = "tutorial-987654321"

    with pytest.raises(
        NemesisContractError,
        match="random opaque battle_id and match_id",
    ):
        validate_nemesis_record(exported, require_terminal=True)


def test_export_rejects_duplicate_battle_ids() -> None:
    record = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="stalemate")
    with pytest.raises(NemesisContractError, match="duplicate battle_id"):
        export_nemesis_ndjson([record, record])


def test_cli_validates_and_exports_headerless_records(tmp_path: Path) -> None:
    record = NemesisBattleCollector.from_v5_meta(
        _meta(),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status="p1_win")
    source = tmp_path / "records.jsonl"
    destination = tmp_path / "export.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_nemesis_dataset.py",
            str(source),
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [json.loads(line) for line in destination.read_text().splitlines()]
    assert lines[0]["format"] == NEMESIS_EXPORT_FORMAT
    assert lines[1]["privacy"]["include_players"] is False


def test_cli_extracts_exact_nested_record_from_v5_bundle(tmp_path: Path) -> None:
    record = NemesisBattleCollector.from_v5_meta(
        _meta(), feature_cutoff_at="2026-07-28T12:00:00Z"
    ).finalize(status="p1_win")
    source = tmp_path / "v5.jsonl"
    output = tmp_path / "nemesis.jsonl"
    source.write_text(
        json.dumps({"record_type": "battle", "meta": {"nemesis_record": record}})
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/export_nemesis_dataset.py", str(source), str(output)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    exported = json.loads(output.read_text().splitlines()[1])
    assert re.fullmatch(r"record_[0-9a-f]{32}", exported["battle_id"])
    assert exported["battle_id"] != record["battle_id"]


def test_cli_fails_closed_for_v5_bundle_without_nemesis_record(tmp_path: Path) -> None:
    source = tmp_path / "v5.jsonl"
    output = tmp_path / "nemesis.jsonl"
    source.write_text(json.dumps({"record_type": "battle", "meta": {}}) + "\n")
    completed = subprocess.run(
        [sys.executable, "scripts/export_nemesis_dataset.py", str(source), str(output)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not output.exists()
