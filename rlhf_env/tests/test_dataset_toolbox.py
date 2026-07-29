"""Focused safety and readiness tests for the cross-contour dataset toolbox."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.nemesis_dataset import (
    NemesisBattleCollector,
    _deck_pair_split_fingerprint,
    write_nemesis_export,
)
from infrastructure.database import RETURNCLOCK_DATASET_SCHEMA
from infrastructure.returnclock_dataset import (
    materialize_returnclock_dataset,
    write_returnclock_jsonl,
)
from rlhf_env.components.dataset_toolbox import (
    DatasetToolbox,
    DatasetToolboxError,
)
from rlhf_env.tests.test_v5_trace_validate import _drive_completed
from scripts.materialize_v5_dataset_export import MATERIALIZED_FORMAT


UTC = timezone.utc
PRIVACY_SALT = "dataset-toolbox-test-salt-with-at-least-32-bytes"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _replace_participant_ids(
    value: object,
    *,
    p1_user_id: int,
    p2_user_id: int,
) -> object:
    if isinstance(value, dict):
        return {
            key: _replace_participant_ids(
                nested,
                p1_user_id=p1_user_id,
                p2_user_id=p2_user_id,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_participant_ids(
                nested,
                p1_user_id=p1_user_id,
                p2_user_id=p2_user_id,
            )
            for nested in value
        ]
    if value == p1_user_id:
        return 1
    if value == p2_user_id:
        return 2
    return value


def _session(
    session_id: str,
    *,
    user_id: int,
    started_at: datetime,
) -> dict:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "source": "android_app",
        "started_at": started_at,
        "ended_at": started_at + timedelta(minutes=5),
        "duration_seconds": 300,
        "screens_visited": [
            {"screen": "menu"},
            {"screen": "collection"},
        ],
        "battles_played": 0,
        "cases_opened": 0,
        "analytics_version": 2,
        "timezone": "Europe/Moscow",
        "utc_offset_minutes": 180,
        "entrypoint": "direct",
    }


@pytest.fixture
def returnclock_artifact(tmp_path: Path) -> tuple[DatasetToolbox, Path]:
    root = tmp_path / "datasets"
    toolbox = DatasetToolbox(root)
    start = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session(
                f"user-{user_id}-session",
                user_id=user_id,
                started_at=start + timedelta(days=offset),
            )
            for offset, user_id in enumerate(range(101, 107))
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=PRIVACY_SALT,
        horizon_hours=24,
    )
    artifact = write_returnclock_jsonl(
        dataset,
        root / "returnclock" / "sample.jsonl",
    )
    return toolbox, artifact


@pytest.fixture(scope="module")
def valid_v5_materialized(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[DatasetToolbox, Path, str]:
    """Build one real terminal trace and wrap it as a training artifact."""

    tmp_path = tmp_path_factory.mktemp("dataset-toolbox-v5")
    with patch(
        "rlhf_env.components.match_runner.random.uniform",
        return_value=0.0,
    ):
        source_v5, _ = _drive_completed(tmp_path)
    source_group = source_v5.parents[2]
    battle_id = source_v5.parent.name
    toolbox = DatasetToolbox(tmp_path / "datasets")
    artifact = toolbox.root / "v5" / "valid-materialized"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_group, artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("agent_name", None)
    manifest.pop("v5_storage", None)
    manifest["materialized_format"] = MATERIALIZED_FORMAT
    manifest["schema_version"] = "rlhf_v5_storage_v1"
    manifest["storage_schema"] = "rlhf_v5_storage_v1"
    manifest["spec"] = {
        "source_format": "extraarena_v5_dataset_export_v1",
        "source_file": "fixture.jsonl",
        "privacy": "side_pseudonyms_p1_1_p2_2",
        "include_players": False,
        "record_id_scheme": "native_opaque_record_ids_v1",
        "days": 30,
        "limit_battles": 1000,
        "source_skipped_invalid": 0,
        "collection_classes": ["rl-vs-bot"],
    }
    manifest["env"] = {
        "materializer": "scripts.materialize_v5_dataset_export.py",
        "deep_validator": (
            "rlhf_env.components.v5_trace_validate.validate_v5_trace"
        ),
        "validation_scope": "v5_trace_without_legacy_battle_log",
    }
    meta_path = artifact / "battles" / battle_id / "v5" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p1_user_id = int(meta["p1_user_id"])
    p2_user_id = int(meta["p2_user_id"])
    meta = _replace_participant_ids(
        meta,
        p1_user_id=p1_user_id,
        p2_user_id=p2_user_id,
    )
    finished_at = datetime.fromisoformat(
        str(meta["finished_at"]).replace("Z", "+00:00")
    )
    duration = float(meta["duration_seconds"])
    meta["started_at"] = (
        finished_at - timedelta(seconds=duration)
    ).isoformat()
    meta["starting_player"] = "p1"
    meta["start_metadata"] = {"client_ready_anchored": True}
    meta["timestamp_features"] = {
        "p1_deck_size": len(meta["p1_deck"]),
        "p2_deck_size": len(meta["p2_deck"]),
        "starting_player": "p1",
        "duration_seconds": duration,
        "turns": int(meta["turns"]),
    }
    for policy_key in ("p1_policy", "bot_policy"):
        policy = meta.get(policy_key)
        assert isinstance(policy, dict)
        policy["weights_hash"] = "0" * 64
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    original_result = manifest["battles_results"][0]
    manifest["battles_results"] = [
        {
            "battle_id": battle_id,
            "winner_user_id": _replace_participant_ids(
                original_result.get("winner_user_id"),
                p1_user_id=p1_user_id,
                p2_user_id=p2_user_id,
            ),
            "loser_user_id": _replace_participant_ids(
                original_result.get("loser_user_id"),
                p1_user_id=p1_user_id,
                p2_user_id=p2_user_id,
            ),
            "status": original_result["status"],
            "turns": int(meta["turns"]),
            "duration_seconds": float(meta["duration_seconds"]),
            "battle_tag": str(meta["battle_tag"]),
            "collection_class": "human-vs-bot",
            "p1_actor_type": meta["p1_actor_type"],
            "p2_actor_type": meta["p2_actor_type"],
            "v5_dir": f"battles/{battle_id}/v5",
            "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
            "v5_trace_ok": True,
            "validation_scope": "v5_trace_without_legacy_battle_log",
            "finished_at": meta["finished_at"],
        }
    ]
    result_status = manifest["battles_results"][0]["status"]
    manifest["results"] = {
        "battles_planned": 1,
        "battles_finished": 1,
        "p1_wins": int(result_status == "P1_WIN"),
        "p2_wins": int(result_status == "P2_WIN"),
        "draws": int(result_status in {"DRAW", "STALEMATE"}),
        "winrate_p1": float(result_status == "P1_WIN"),
        "winrate_p2": float(result_status == "P2_WIN"),
        "avg_turns": float(meta["turns"]),
        "avg_duration_seconds": round(float(meta["duration_seconds"]), 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    actions_path = (
        artifact / "battles" / battle_id / "v5" / "actions.jsonl"
    )
    actions = _replace_participant_ids(
        _read_jsonl(actions_path),
        p1_user_id=p1_user_id,
        p2_user_id=p2_user_id,
    )
    assert isinstance(actions, list)
    for action in actions:
        action["human_decision_time_raw_ms"] = None
        action["metronome_prediction_ms"] = None
        action["metronome_applied_ms"] = 0.0
        action["metronome_fallback_used"] = True
    _write_jsonl(actions_path, actions)
    turns_path = artifact / "battles" / battle_id / "v5" / "turns.jsonl"
    turns = _replace_participant_ids(
        _read_jsonl(turns_path),
        p1_user_id=p1_user_id,
        p2_user_id=p2_user_id,
    )
    assert isinstance(turns, list)
    _write_jsonl(turns_path, turns)
    toolbox._attach_current_catalog(artifact)
    validation = toolbox.validate_artifact(artifact)
    assert validation["training_ready"] is True, validation
    return toolbox, artifact, battle_id


def _copy_v5_artifact(
    source: tuple[DatasetToolbox, Path, str],
    name: str,
) -> tuple[DatasetToolbox, Path, str]:
    toolbox, artifact, battle_id = source
    destination = toolbox.root / "v5" / name
    shutil.copytree(artifact, destination)
    return toolbox, destination, battle_id


def test_v5_materialized_reports_contour_specific_readiness(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, _ = valid_v5_materialized

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is True
    assert validation["training_ready_scope"] == "v5_policy_only"
    assert validation["v5_policy_training_ready"] is True
    # This fixture is rl-vs-bot: its auxiliary shape is valid, but it has no
    # observed player timing/duration labels.
    assert validation["metronome_training_ready"] is False
    assert validation["timestamp_training_ready"] is False


def _nemesis_meta(
    *,
    toolbox: DatasetToolbox,
    index: int,
    catalog_hash: str | None = None,
    rehydrated: bool = False,
) -> dict:
    card_ids = sorted(
        int(card_id)
        for card_id in toolbox._catalog_payload["cards"]
    )
    value = {
        "battle_id": f"v5g2-{index:032x}",
        "match_id": f"nemesis-toolbox-{index}",
        "started_at": "2026-07-28T12:00:01Z",
        "game_mode": "classic",
        "ruleset": "classic",
        "catalog_hash": (
            toolbox.current_catalog_hash
            if catalog_hash is None
            else catalog_hash
        ),
        "card_params_schema": "train_v3_card_params_v1",
        "deck_params_schema": "train_v3_deck_params_v1",
        "starting_player": "p1" if index % 2 else "p2",
        "p1_user_id": -1000 - index,
        "p2_user_id": -2000 - index,
        "p1_actor_type": "rl",
        "p2_actor_type": "rl",
        "p1_deck": [
            {
                "slot": 0,
                "card_id": card_ids[index % len(card_ids)],
                "level": 5,
            },
            {
                "slot": 1,
                "card_id": card_ids[(index + 10) % len(card_ids)],
                "level": 3,
            },
        ],
        "p2_deck": [
            {
                "slot": 0,
                "card_id": card_ids[(index + 20) % len(card_ids)],
                "level": 5,
            },
            {
                "slot": 1,
                "card_id": card_ids[(index + 30) % len(card_ids)],
                "level": 4,
            },
        ],
        "model_provenance": {
            "p1": {"model_id": "extra-lr-v5-lite"},
            "p2": {"model_id": "extra-lr-v5"},
        },
    }
    if rehydrated:
        value["dataset_generation"] = 2
        value["dataset_generation_reason"] = "regression_rehydrate"
    return value


def _nemesis_record(
    *,
    toolbox: DatasetToolbox,
    index: int,
    catalog_hash: str | None = None,
    rehydrated: bool = False,
) -> dict:
    return NemesisBattleCollector.from_v5_meta(
        _nemesis_meta(
            toolbox=toolbox,
            index=index,
            catalog_hash=catalog_hash,
            rehydrated=rehydrated,
        ),
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(
        status="p1_win" if index % 2 else "p2_win",
        duration_seconds=90 + index,
        turns_count=8 + index,
    )


def _standard_nemesis_records(
    *,
    toolbox: DatasetToolbox,
    count: int = 6,
    connected: bool = False,
    star: bool = False,
) -> list[dict]:
    records: list[dict] = []
    for index in range(1, count + 1):
        cutoff = datetime(2026, 7, 20 + index, 12, tzinfo=UTC)
        meta = _nemesis_meta(toolbox=toolbox, index=index)
        meta["started_at"] = (
            cutoff + timedelta(seconds=1)
        ).isoformat()
        meta["p1_actor_type"] = "human"
        meta["p2_actor_type"] = "human"
        if star:
            meta["p1_user_id"] = 9999
            meta["p2_user_id"] = 20000 + index
        elif connected:
            meta["p1_user_id"] = 10000 + index - 1
            meta["p2_user_id"] = 10000 + index
        snapshot = {
            "captured_at": cutoff.isoformat(),
            "profile": {
                "wins": 20 + index,
                "losses": 10,
                "trophies": 1000 + index,
            },
            "summary": None,
            "recent": [],
        }
        records.append(
            NemesisBattleCollector.from_v5_meta(
                meta,
                feature_cutoff_at=cutoff.isoformat(),
                extended_by_seat={
                    "p1": snapshot,
                    "p2": snapshot,
                },
            ).finalize(
                status="p1_win" if index % 2 else "p2_win",
                duration_seconds=90 + index,
                turns_count=8 + index,
            )
        )
    return records


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.jsonl",
        "../../outside.jsonl",
    ],
)
def test_resolve_rejects_lexical_traversal(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")

    with pytest.raises(DatasetToolboxError, match="inside datasets_dir"):
        toolbox.resolve(unsafe_path)


def test_resolve_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")

    with pytest.raises(DatasetToolboxError, match="inside datasets_dir"):
        toolbox.resolve(tmp_path / "outside.jsonl")


def test_resolve_accepts_canonical_alias_for_datasets_root(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical" / "datasets"
    toolbox = DatasetToolbox(canonical_root)
    alias_root = tmp_path / "datasets-alias"
    alias_root.symlink_to(canonical_root, target_is_directory=True)

    resolved = toolbox.resolve(alias_root / "nemesis" / "sample.jsonl")

    assert resolved == canonical_root / "nemesis" / "sample.jsonl"


def test_resolve_rejects_symlink_component_that_escapes_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    outside = tmp_path / "outside"
    outside.mkdir()
    toolbox = DatasetToolbox(root)
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DatasetToolboxError, match="symlink"):
        toolbox.resolve("escape/export.jsonl")


def test_constructor_rejects_symlink_as_datasets_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-datasets"
    real_root.mkdir()
    linked_root = tmp_path / "linked-datasets"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(DatasetToolboxError, match="symlink"):
        DatasetToolbox(linked_root)


def test_list_inspect_and_validate_returnclock_artifact(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact

    inventory = toolbox.list_artifacts(kind="returnclock")
    assert inventory["count"] == 1
    assert inventory["artifacts"][0]["relative_path"] == (
        "returnclock/sample.jsonl"
    )
    assert inventory["artifacts"][0]["kind"] == "returnclock"

    inspection = toolbox.inspect_artifact("returnclock/sample.jsonl")
    assert inspection["kind"] == "returnclock"
    assert inspection["data_record_count"] == 6
    assert inspection["mode"] == "0o600"
    assert inspection["sha256"]
    assert inspection["header"]["summary"]["example_count"] == 6
    assert inspection["header"]["ingested_before"]

    validation = toolbox.validate_artifact(artifact)
    assert validation["ok"] is True
    assert validation["training_ready"] is True
    assert validation["summary"]["distinct_users"] == 6
    assert validation["summary"]["distinct_cutoffs"] == 6
    assert validation["summary"]["grouped_user_split_possible"] is True
    assert validation["summary"]["temporal_split_possible"] is True


@pytest.mark.parametrize(
    ("feature", "value", "issue"),
    [
        ("timezone", 3, "feature_timezone"),
        ("timezone_known", 1, "feature_timezone_known"),
        ("local_weekday", 7, "feature_local_weekday"),
        ("local_hour", 24, "feature_local_hour"),
        ("local_hour_sin", 2.0, "feature_local_hour_sin"),
        ("sessions_1d", -1, "feature_sessions_1d"),
        (
            "hours_since_previous_session",
            -0.1,
            "feature_hours_since_previous_session",
        ),
        (
            "recent_local_start_hours",
            [1, 24],
            "feature_recent_local_start_hours",
        ),
        ("notifications_24h", -1, "feature_notifications_24h"),
        ("last_session_end_inferred", "false", "feature_last_session_end_inferred"),
    ],
)
def test_returnclock_validator_rejects_invalid_feature_types_and_ranges(
    returnclock_artifact: tuple[DatasetToolbox, Path],
    feature: str,
    value: object,
    issue: str,
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    rows[1]["features"][feature] = value
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(issue in item for item in validation["issues"])


def test_returnclock_validator_rejects_censor_time_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    toolbox = DatasetToolbox(root)
    start = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("s-1", user_id=1, started_at=start),
            _session(
                "s-2",
                user_id=2,
                started_at=start + timedelta(hours=1),
            ),
        ],
        dataset_end=start + timedelta(hours=12),
        privacy_salt=PRIVACY_SALT,
        horizon_hours=24,
    )
    artifact = write_returnclock_jsonl(
        dataset,
        root / "returnclock" / "censored.jsonl",
    )
    rows = _read_jsonl(artifact)
    assert rows[1]["label"]["right_censored"] is True
    rows[1]["label"]["observation_window_minutes"] += 5
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "censor_time_mismatch" in issue for issue in validation["issues"]
    )


def test_returnclock_validator_rejects_assignment_arm_inconsistency(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    post_cutoff = rows[1]["post_cutoff"]
    post_cutoff["assignments"] = [
        {
            "experiment_id": "returnclock-v1",
            "treatment_arm": "returnclock",
            "assignment_probability": 0.5,
            "decision": "send",
            "decision_source": "policy",
            "policy_version": "v1",
            "model_version": None,
        }
    ]
    post_cutoff["treatment_arms"] = ["forged-arm"]
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "assignment_count_mismatch" in issue
        for issue in validation["issues"]
    )
    assert any(
        "treatment_arms_mismatch" in issue
        for issue in validation["issues"]
    )
    assert any(
        "treatment_assigned_mismatch" in issue
        for issue in validation["issues"]
    )


def test_returnclock_validator_rejects_unknown_post_cutoff_identity_field(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    rows[1]["post_cutoff"]["telegram_user_ids"] = [123456789]
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "post_cutoff_contract" in issue for issue in validation["issues"]
    )
    assert any("raw_user_id" in issue for issue in validation["issues"])


def test_returnclock_validator_rejects_row_before_header_cutoff_start(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    first_cutoff = datetime.fromisoformat(
        rows[1]["prediction_cutoff_at"].replace("Z", "+00:00")
    )
    rows[0]["cutoff_start"] = (
        first_cutoff + timedelta(minutes=1)
    ).isoformat().replace("+00:00", "Z")
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "cutoff_before_cutoff_start" in issue
        for issue in validation["issues"]
    )


def test_returnclock_validator_requires_canonical_z_header_timestamps(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    rows[0]["generated_at"] = rows[0]["generated_at"].replace(
        "Z", "+00:00"
    )
    rows[0]["dataset_end"] = rows[0]["dataset_end"].replace(
        "Z", "+00:00"
    )
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert "generated_at_not_canonical_utc" in validation["issues"]
    assert "dataset_end_not_canonical_utc" in validation["issues"]


def test_returnclock_validator_requires_canonical_z_prediction_cutoff(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    rows[1]["prediction_cutoff_at"] = rows[1][
        "prediction_cutoff_at"
    ].replace("Z", "+00:00")
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "prediction_cutoff_not_canonical_utc" in issue
        for issue in validation["issues"]
    )


def test_returnclock_all_treated_artifact_is_valid_but_not_natural_ready(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    for row in rows[1:]:
        row["post_cutoff"] = {
            "notification_decision_count": 1,
            "provider_accepted_count": 0,
            "notification_sent_count": 0,
            "notification_opened_count": 0,
            "notification_channels": [],
            "treatment_arms": ["observational"],
            "assignments": [
                {
                    "experiment_id": None,
                    "treatment_arm": "observational",
                    "assignment_probability": 1.0,
                    "decision": "send",
                    "decision_source": "notification_system",
                    "policy_version": "observational-v1",
                    "model_version": None,
                }
            ],
            "notification_attributed": False,
            "treatment_assigned": True,
            "organic_candidate": False,
        }
    rows[0]["summary"]["treated_intervals"] = len(rows) - 1
    rows[0]["summary"]["organic_candidates"] = 0
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["ok"] is True
    assert validation["training_ready"] is False
    assert validation["training_ready_scope"] == (
        "natural_return_observational"
    )
    assert validation["natural_return_training_ready"] is False
    assert validation["causal_notification_training_ready"] is False


def test_returnclock_split_is_user_grouped_temporal_and_private(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact

    result = toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "returnclock" / "split-v1"
    manifest = json.loads(
        (split_root / "manifest.json").read_text(encoding="utf-8")
    )
    users_by_split: dict[str, set[str]] = {}
    total_examples = 0
    for split_name in ("train", "validation", "test"):
        split_path = split_root / f"{split_name}.jsonl"
        rows = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
        ]
        examples = rows[1:]
        users_by_split[split_name] = {
            row["user_id_hash"]
            for row in examples
        }
        total_examples += len(examples)
        sort_keys = [
            (row["prediction_cutoff_at"], row["user_id_hash"])
            for row in examples
        ]
        assert sort_keys == sorted(sort_keys)
        assert stat.S_IMODE(split_path.stat().st_mode) == 0o600
        assert rows[0]["split_name"] == split_name
        assert rows[0]["pseudonymization_key_id"] == (
            manifest["pseudonymization_key_id"]
        )

    assert users_by_split["train"].isdisjoint(
        users_by_split["validation"]
    )
    assert users_by_split["train"].isdisjoint(users_by_split["test"])
    assert users_by_split["validation"].isdisjoint(users_by_split["test"])
    assert total_examples == 6
    assert result["training_ready"] is True
    assert result["natural_return_training_ready"] is True
    assert result["causal_notification_training_ready"] is False
    assert result["source_sha256"] == manifest["source_sha256"]
    assert manifest["assignment_basis"] == "user_first_prediction_cutoff"
    assert stat.S_IMODE((split_root / "manifest.json").stat().st_mode) == 0o600

    validation = toolbox.validate_artifact("returnclock/split-v1")
    inspection = toolbox.inspect_artifact("returnclock/split-v1")
    assert validation["ok"] is True
    assert validation["training_ready"] is True
    assert validation["natural_return_training_ready"] is True
    assert validation["causal_notification_training_ready"] is False
    assert validation["summary"]["user_leakage"] is False
    assert validation["sha256"] == inspection["sha256"]
    assert inspection["mode"] == "0o700"
    assert inspection["manifest_mode"] == "0o600"
    assert inspection["manifest"]["training_filter"] == {
        "field": "post_cutoff.organic_candidate",
        "equals": True,
    }
    assert inspection["manifest"]["excluded_treated_count"] == 0
    assert (
        inspection["manifest"]["excluded_temporal_boundary_count"]
        == 0
    )
    assert (
        inspection["manifest"]["post_cutoff_excluded_from_features"]
        is True
    )

    with pytest.raises(DatasetToolboxError, match="already exists"):
        toolbox.split_returnclock(
            source=artifact,
            output_dir="returnclock/split-v1",
        )


def test_returnclock_split_validator_detects_cross_partition_user_leakage(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "returnclock" / "split-v1"
    train_rows = [
        json.loads(line)
        for line in (split_root / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    with (split_root / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                train_rows[1],
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    validation = toolbox.validate_artifact("returnclock/split-v1")

    assert validation["ok"] is False
    assert validation["training_ready"] is False
    assert any(
        issue == "test:user_leakage"
        for issue in validation["issues"]
    )
    assert "split_rows_not_in_organic_source" in validation["issues"]


def test_returnclock_split_validator_detects_dropped_source_row(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "returnclock" / "split-v1"
    train_path = split_root / "train.jsonl"
    rows = _read_jsonl(train_path)
    assert len(rows) > 2
    _write_jsonl(train_path, rows[:-1])

    validation = toolbox.validate_artifact("returnclock/split-v1")

    assert validation["training_ready"] is False
    assert "train:example_count_mismatch" in validation["issues"]


def test_returnclock_split_files_contain_only_organic_rows(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    rows = _read_jsonl(artifact)
    treated = rows[1]["post_cutoff"]
    treated.update(
        {
            "notification_decision_count": 1,
            "treatment_arms": ["observational"],
            "assignments": [
                {
                    "experiment_id": None,
                    "treatment_arm": "observational",
                    "assignment_probability": 1.0,
                    "decision": "send",
                    "decision_source": "notification_system",
                    "policy_version": "observational-v1",
                    "model_version": None,
                }
            ],
            "treatment_assigned": True,
            "organic_candidate": False,
        }
    )
    rows[0]["summary"]["organic_candidates"] -= 1
    rows[0]["summary"]["treated_intervals"] += 1
    _write_jsonl(artifact, rows)

    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/organic-split",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "returnclock" / "organic-split"
    manifest = json.loads(
        (split_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_example_count"] == 6
    assert manifest["example_count"] == 5
    assert manifest["excluded_treated_count"] == 1
    assert manifest["training_filter"] == {
        "field": "post_cutoff.organic_candidate",
        "equals": True,
    }
    for split_name in ("train", "validation", "test"):
        for row in _read_jsonl(split_root / f"{split_name}.jsonl")[1:]:
            assert row["post_cutoff"]["organic_candidate"] is True


def test_returnclock_split_trims_late_rows_from_earlier_user_cohort(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    toolbox = DatasetToolbox(root)
    start = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    sessions = [
        _session(
            f"user-{user_id}-first",
            user_id=user_id,
            started_at=start + timedelta(days=offset),
        )
        for offset, user_id in enumerate(range(101, 107))
    ]
    sessions.append(
        _session(
            "user-101-late",
            user_id=101,
            started_at=start + timedelta(days=9),
        )
    )
    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        dataset_end=start + timedelta(days=20),
        privacy_salt=PRIVACY_SALT,
        horizon_hours=24,
    )
    source = write_returnclock_jsonl(
        dataset,
        root / "returnclock" / "late-user.jsonl",
    )

    toolbox.split_returnclock(
        source=source,
        output_dir="returnclock/late-user-split",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = root / "returnclock" / "late-user-split"
    manifest = json.loads(
        (split_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["excluded_temporal_boundary_count"] == 1
    train_rows = _read_jsonl(split_root / "train.jsonl")[1:]
    validation_rows = _read_jsonl(
        split_root / "validation.jsonl"
    )[1:]
    test_rows = _read_jsonl(split_root / "test.jsonl")[1:]
    assert max(
        row["prediction_cutoff_at"] for row in train_rows
    ) < min(row["prediction_cutoff_at"] for row in validation_rows)
    assert max(
        row["prediction_cutoff_at"] for row in validation_rows
    ) < min(row["prediction_cutoff_at"] for row in test_rows)
    assert toolbox.validate_artifact(split_root)["training_ready"] is True


def test_returnclock_split_manifest_rejects_extra_identity_field(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    manifest_path = (
        toolbox.root / "returnclock" / "split-v1" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_user_id"] = 123456789
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact("returnclock/split-v1")
    assert validation["training_ready"] is False
    assert "split_manifest_allowlist_mismatch" in validation["issues"]
    assert "split_manifest_raw_user_id" in validation["issues"]


def test_returnclock_split_validator_recomputes_manifest_bounds(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    manifest_path = (
        toolbox.root / "returnclock" / "split-v1" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["first_cutoff_min"] = (
        "2099-01-01T00:00:00+00:00"
    )
    manifest["splits"]["train"]["row_cutoff_max"] = (
        "2099-01-01T00:00:00+00:00"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact("returnclock/split-v1")

    assert validation["training_ready"] is False
    assert "train:first_cutoff_min_mismatch" in validation["issues"]
    assert "train:row_cutoff_max_mismatch" in validation["issues"]


def test_returnclock_split_rejects_manifest_path_escape(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    manifest_path = (
        toolbox.root / "returnclock" / "split-v1" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["test"]["file"] = "../sample.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact("returnclock/split-v1")

    assert validation["ok"] is False
    assert validation["training_ready"] is False
    assert "test:unsafe_file_name" in validation["issues"]


def test_inventory_and_inspection_do_not_follow_manifest_symlink(
    returnclock_artifact: tuple[DatasetToolbox, Path],
) -> None:
    toolbox, artifact = returnclock_artifact
    toolbox.split_returnclock(
        source=artifact,
        output_dir="returnclock/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "returnclock" / "split-v1"
    manifest_path = split_root / "manifest.json"
    external_manifest = toolbox.root / "external-manifest.json"
    manifest_path.replace(external_manifest)
    manifest_path.symlink_to(external_manifest)

    inventory = toolbox.list_artifacts(kind="returnclock_split")

    assert inventory["count"] == 0
    with pytest.raises(DatasetToolboxError, match="symlink"):
        toolbox.inspect_artifact("returnclock/split-v1")
    validation = toolbox.validate_artifact("returnclock/split-v1")
    assert validation["ok"] is False
    assert validation["training_ready"] is False


@pytest.mark.asyncio
async def test_production_exports_are_disabled_by_default(tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeDatabase:
        def __init__(self, settings):
            calls.append(f"constructed:{settings}")

        async def connect(self):
            calls.append("connected")

        async def close(self):
            calls.append("closed")

        async def fetch_returnclock_dataset_rows(self, **kwargs):
            calls.append("fetched")
            return {
                "schema": RETURNCLOCK_DATASET_SCHEMA,
                "sessions": [],
                "decisions": [],
                "delivery_events": [],
            }

    toolbox = DatasetToolbox(
        tmp_path / "datasets",
        production_enabled=False,
        database_factory=FakeDatabase,
        settings_factory=lambda: SimpleNamespace(database="private-dsn"),
    )

    with pytest.raises(DatasetToolboxError, match="production_data_disabled"):
        await toolbox.export_production_v5(output="v5/export.jsonl")
    with pytest.raises(DatasetToolboxError, match="production_data_disabled"):
        await toolbox.export_returnclock(output="returnclock/export.jsonl")

    assert calls == []
    assert not (tmp_path / "datasets" / "v5" / "export.jsonl").exists()
    assert not (
        tmp_path / "datasets" / "returnclock" / "export.jsonl"
    ).exists()


@pytest.mark.asyncio
async def test_returnclock_export_never_leaks_salt_or_raw_user_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "returnclock-secret-salt-never-return-this-0123456789"
    salt_env = "TEST_RETURNCLOCK_DATASET_SALT"
    salt_key_id_env = "TEST_RETURNCLOCK_DATASET_SALT_KEY_ID"
    monkeypatch.setenv(salt_env, secret)
    monkeypatch.setenv(salt_key_id_env, "returnclock-test-key-2026-07")
    start = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    class FakeDatabase:
        def __init__(self, settings):
            assert settings == "opaque-test-settings"

        async def connect(self):
            return None

        async def close(self):
            return None

        async def fetch_returnclock_dataset_rows(self, **kwargs):
            return {
                "schema": RETURNCLOCK_DATASET_SCHEMA,
                "sessions": [
                    _session("s-987654321", user_id=987654321, started_at=start),
                    _session(
                        "s-123456789",
                        user_id=123456789,
                        started_at=start + timedelta(days=1),
                    ),
                ],
                "decisions": [],
                "delivery_events": [],
            }

    toolbox = DatasetToolbox(
        tmp_path / "datasets",
        returnclock_salt_env=salt_env,
        returnclock_salt_key_id_env=salt_key_id_env,
        production_enabled=True,
        database_factory=FakeDatabase,
        settings_factory=lambda: SimpleNamespace(
            database="opaque-test-settings",
        ),
    )
    result = await toolbox.export_returnclock(
        output="returnclock/export.jsonl",
        start=start.isoformat(),
        end=(start + timedelta(days=10)).isoformat(),
        horizon_hours=24,
    )
    inspection = toolbox.inspect_artifact("returnclock/export.jsonl")
    validation = toolbox.validate_artifact("returnclock/export.jsonl")
    artifact_text = (
        tmp_path / "datasets" / "returnclock" / "export.jsonl"
    ).read_text(encoding="utf-8")
    public_surface = json.dumps(
        {
            "result": result,
            "inspection": inspection,
            "validation": validation,
            "status": toolbox.status(),
        },
        ensure_ascii=False,
        default=str,
    )

    assert result["ok"] is True
    assert validation["ok"] is True
    assert validation["training_ready"] is False
    assert validation["summary"]["organic_distinct_users"] == 2
    assert (
        validation["summary"]["organic_distinct_first_cutoff_cohorts"]
        == 2
    )
    assert secret not in public_surface
    assert secret not in artifact_text
    assert "987654321" not in artifact_text
    assert "123456789" not in artifact_text
    assert os.environ[salt_env] == secret


@pytest.mark.asyncio
async def test_failed_overwrite_preserves_previous_returnclock_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    salt_env = "TEST_RETURNCLOCK_OVERWRITE_SALT"
    key_id_env = "TEST_RETURNCLOCK_OVERWRITE_KEY_ID"
    monkeypatch.setenv(
        salt_env,
        "overwrite-regression-salt-with-at-least-32-secret-bytes",
    )
    monkeypatch.setenv(key_id_env, "returnclock-overwrite-test-v1")
    start = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    class FakeDatabase:
        def __init__(self, settings):
            pass

        async def connect(self):
            return None

        async def close(self):
            return None

        async def fetch_returnclock_dataset_rows(self, **kwargs):
            return {
                "schema": RETURNCLOCK_DATASET_SCHEMA,
                "sessions": [
                    _session("s-1", user_id=1, started_at=start),
                    _session(
                        "s-2",
                        user_id=2,
                        started_at=start + timedelta(days=1),
                    ),
                ],
                "decisions": [],
                "delivery_events": [],
            }

    toolbox = DatasetToolbox(
        tmp_path / "datasets",
        returnclock_salt_env=salt_env,
        returnclock_salt_key_id_env=key_id_env,
        production_enabled=True,
        database_factory=FakeDatabase,
        settings_factory=lambda: SimpleNamespace(database="test"),
    )
    destination = toolbox.root / "returnclock" / "stable.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text("previous-valid-artifact\n", encoding="utf-8")
    monkeypatch.setattr(
        toolbox,
        "validate_artifact",
        lambda path: {
            "ok": False,
            "training_ready": False,
            "issues": ["forced_validation_failure"],
        },
    )

    with pytest.raises(
        DatasetToolboxError,
        match="failed validation",
    ):
        await toolbox.export_returnclock(
            output="returnclock/stable.jsonl",
            start=start.isoformat(),
            end=(start + timedelta(days=10)).isoformat(),
            horizon_hours=24,
            overwrite=True,
        )

    assert destination.read_text(encoding="utf-8") == (
        "previous-valid-artifact\n"
    )


def test_nemesis_zero_eligible_weight_is_not_training_ready(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    records = [
        _nemesis_record(
            toolbox=toolbox,
            index=index,
            rehydrated=True,
        )
        for index in (1, 2)
    ]
    artifact = write_nemesis_export(
        records,
        toolbox.root / "nemesis" / "zero-weight.jsonl",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["ok"] is True
    assert validation["training_ready"] is False
    assert validation["training_ready_lite"] is False
    assert validation["training_ready_standard"] is False
    assert validation["summary"]["positive_weight_records"] == 0


def test_nemesis_standard_readiness_requires_player_disjoint_split(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    snapshot = {
        "captured_at": "2026-07-28T12:00:00Z",
        "profile": {"wins": 10, "losses": 5, "trophies": 1200},
        "summary": None,
        "recent": [],
    }
    records = []
    for index in (1, 2, 3):
        meta = _nemesis_meta(toolbox=toolbox, index=index)
        meta["p1_actor_type"] = "human"
        meta["p2_actor_type"] = "human"
        records.append(
            NemesisBattleCollector.from_v5_meta(
                meta,
                feature_cutoff_at="2026-07-28T12:00:00Z",
                extended_by_seat={
                    "p1": snapshot,
                    "p2": snapshot,
                },
            ).finalize(
                status="p1_win" if index % 2 else "p2_win",
                duration_seconds=90,
                turns_count=8,
            )
        )
    artifact = write_nemesis_export(
        records,
        toolbox.root / "nemesis" / "standard-unsplit.jsonl",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["ok"] is True
    assert validation["training_ready_lite"] is True
    assert validation["training_ready_standard"] is False
    assert validation["standard_readiness_blockers"] == [
        "player_disjoint_split_not_materialized"
    ]


@pytest.mark.parametrize(
    ("record_count", "expected_ready"),
    [(2, False), (3, True)],
)
def test_nemesis_lite_requires_three_split_groups(
    tmp_path: Path,
    record_count: int,
    expected_ready: bool,
) -> None:
    toolbox = DatasetToolbox(tmp_path / f"datasets-{record_count}")
    artifact = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=index)
            for index in range(1, record_count + 1)
        ],
        toolbox.root / "nemesis" / "lite-threshold.jsonl",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["ok"] is True
    assert validation["summary"]["lite_distinct_split_groups"] == record_count
    assert validation["training_ready_lite"] is expected_ready


def test_nemesis_split_materializes_lite_only_headless_export(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=index)
            for index in range(1, 7)
        ],
        toolbox.root / "nemesis" / "headless-lite.jsonl",
    )

    result = toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/headless-lite-split",
    )
    split_root = toolbox.root / "nemesis" / "headless-lite-split"
    manifest = json.loads(
        (split_root / "manifest.json").read_text(encoding="utf-8")
    )
    validation = toolbox.validate_artifact(split_root)

    assert result["training_ready"] is True
    assert result["training_ready_lite"] is True
    assert result["training_ready_standard"] is False
    assert result["standard_readiness_blockers"] == [
        "no_eligible_standard_records"
    ]
    assert set(manifest["artifacts"]) == {"lite_deck_grouped"}
    assert validation["ok"] is True
    assert validation["training_ready"] is True
    assert validation["training_ready_lite"] is True
    assert validation["training_ready_standard"] is False
    for split_name in ("train", "validation", "test"):
        assert (
            split_root
            / "lite_deck_grouped"
            / f"{split_name}.jsonl"
        ).is_file()


def test_nemesis_public_exclusion_summary_is_bounded(tmp_path: Path) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    fingerprints = [f"{index:064x}" for index in range(25)]

    summary = toolbox._bounded_nemesis_exclusions(
        {
            "standard_player_disjoint": {
                "source_eligible_count": 50,
                "included_count": 25,
                "excluded_cross_partition_count": 25,
                "excluded_cross_partition_fingerprints": fingerprints,
                "assigned_player_count": 10,
                "assigned_players_by_split": {
                    "train": 6,
                    "validation": 2,
                    "test": 2,
                },
            }
        }
    )

    player = summary["standard_player_disjoint"]
    assert len(
        player["excluded_cross_partition_fingerprints_sample"]
    ) == 20
    assert (
        player["excluded_cross_partition_fingerprints_truncated"]
        is True
    )
    assert "excluded_cross_partition_fingerprints" not in player


def test_nemesis_split_materializes_distinct_audited_assignments(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        _standard_nemesis_records(toolbox=toolbox),
        toolbox.root / "nemesis" / "source-v1.jsonl",
    )

    result = toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    split_root = toolbox.root / "nemesis" / "split-v1"
    validation = toolbox.validate_artifact(split_root)
    manifest = json.loads(
        (split_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert result["training_ready"] is True
    assert validation["ok"] is True
    assert validation["training_ready_lite"] is True
    assert validation["training_ready_standard"] is True
    assert manifest["training_readiness"] == {
        "training_ready_lite": True,
        "training_ready_standard": True,
        "standard_readiness_blockers": [],
        "standard_primary_assignment": "standard_player_disjoint",
        "standard_evaluation_assignments": [
            "standard_chronological",
            "standard_deck_grouped",
        ],
        "one_split_satisfies_all_constraints": False,
    }
    assert manifest["feature_contract"][
        "player_group_aliases_are_features"
    ] is False
    assert set(manifest["artifacts"]) == {
        "lite_deck_grouped",
        "standard_player_disjoint",
        "standard_chronological",
        "standard_deck_grouped",
    }

    player_aliases_by_split: dict[str, set[str]] = {}
    artifact_paths: set[str] = set()
    for regime, entries in manifest["artifacts"].items():
        for split_name, entry in entries.items():
            artifact_paths.add(entry["file"])
            split_path = split_root / entry["file"]
            assert stat.S_IMODE(split_path.stat().st_mode) == 0o600
            rows = _read_jsonl(split_path)[1:]
            assert rows
            assert all(
                "player_" not in json.dumps(
                    row["features"],
                    sort_keys=True,
                )
                for row in rows
            )
            if regime == "standard_player_disjoint":
                player_aliases_by_split[split_name] = {
                    alias
                    for row in rows
                    for alias in row["privacy"][
                        "player_group_aliases"
                    ].values()
                }
    assert len(artifact_paths) == 12
    assert player_aliases_by_split["train"].isdisjoint(
        player_aliases_by_split["validation"]
    )
    assert player_aliases_by_split["train"].isdisjoint(
        player_aliases_by_split["test"]
    )
    assert player_aliases_by_split["validation"].isdisjoint(
        player_aliases_by_split["test"]
    )

    inventory = toolbox.list_artifacts(kind="nemesis_split")
    assert inventory["count"] == 1
    inspection = toolbox.inspect_artifact("nemesis/split-v1")
    assert inspection["kind"] == "nemesis_split"
    assert inspection["sha256"] == validation["sha256"]
    assert inspection["mode"] == "0o700"
    assert inspection["manifest_mode"] == "0o600"
    assert inspection["manifest"]["training_readiness"][
        "training_ready_standard"
    ] is True


def test_nemesis_split_supports_connected_player_graph_by_excluding_edges(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        _standard_nemesis_records(
            toolbox=toolbox,
            connected=True,
        ),
        toolbox.root / "nemesis" / "connected.jsonl",
    )

    result = toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/connected-split-v1",
    )
    manifest = json.loads(
        (
            toolbox.root
            / "nemesis"
            / "connected-split-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    player_filter = manifest["exclusions"]["standard_player_disjoint"]

    assert result["training_ready_standard"] is True
    assert player_filter["excluded_cross_partition_count"] > 0
    assert player_filter["included_count"] > 0
    assert (
        player_filter["included_count"]
        + player_filter["excluded_cross_partition_count"]
        == player_filter["source_eligible_count"]
    )


def test_nemesis_split_degrades_to_lite_without_three_disjoint_battles(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        _standard_nemesis_records(
            toolbox=toolbox,
            star=True,
        ),
        toolbox.root / "nemesis" / "star.jsonl",
    )

    result = toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/star-split-v1",
    )
    manifest = json.loads(
        (
            toolbox.root
            / "nemesis"
            / "star-split-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert result["training_ready"] is True
    assert result["training_ready_lite"] is True
    assert result["training_ready_standard"] is False
    assert result["standard_readiness_blockers"] == [
        "player_disjoint_preconditions_not_met"
    ]
    assert set(manifest["artifacts"]) == {"lite_deck_grouped"}


def test_nemesis_split_validator_detects_player_partition_leakage(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        _standard_nemesis_records(toolbox=toolbox),
        toolbox.root / "nemesis" / "source.jsonl",
    )
    toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/split-v1",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    split_root = toolbox.root / "nemesis" / "split-v1"
    manifest_path = split_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = (
        split_root
        / manifest["artifacts"]["standard_player_disjoint"]["train"]["file"]
    )
    validation_path = (
        split_root
        / manifest["artifacts"]["standard_player_disjoint"][
            "validation"
        ]["file"]
    )
    train_rows = _read_jsonl(train_path)
    validation_rows = _read_jsonl(validation_path)
    validation_rows.append(train_rows[1])
    validation_rows[0]["battle_count"] = len(validation_rows) - 1
    _write_jsonl(validation_path, validation_rows)
    entry = manifest["artifacts"]["standard_player_disjoint"][
        "validation"
    ]
    entry["sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    entry["example_count"] = len(validation_rows) - 1
    entry["player_group_count"] = len(
        {
            alias
            for row in validation_rows[1:]
            for alias in row["privacy"]["player_group_aliases"].values()
        }
    )
    entry["group_count"] += 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(split_root)

    assert validation["ok"] is False
    assert validation["training_ready_standard"] is False
    assert any(
        "standard_player_disjoint:validation:player_leakage"
        in issue
        for issue in validation["issues"]
    )


def test_nemesis_split_validator_rejects_unlisted_bundle_file(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    source = write_nemesis_export(
        _standard_nemesis_records(toolbox=toolbox),
        toolbox.root / "nemesis" / "source.jsonl",
    )
    toolbox.split_nemesis(
        source=source,
        output_dir="nemesis/split-v1",
    )
    unexpected = toolbox.root / "nemesis" / "split-v1" / "raw-ids.json"
    unexpected.write_text('{"user_id": 123}\\n', encoding="utf-8")
    unexpected.chmod(0o600)

    validation = toolbox.validate_artifact("nemesis/split-v1")

    assert validation["ok"] is False
    assert validation["training_ready"] is False
    assert "bundle_file_allowlist_mismatch" in validation["issues"]


def test_nemesis_stale_catalog_is_not_training_ready(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    records = [
        _nemesis_record(
            toolbox=toolbox,
            index=index,
            catalog_hash="a" * 64,
        )
        for index in (1, 2)
    ]
    artifact = write_nemesis_export(
        records,
        toolbox.root / "nemesis" / "stale-catalog.jsonl",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "catalog_hash_not_current" in issue
        for issue in validation["issues"]
    )


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    [
        ("telegram_user_id", 987654321, "raw_user_id"),
        ("api_token", "nemesis-secret-token", "sensitive_field_forbidden"),
    ],
)
def test_nemesis_row_privacy_findings_mark_report_unsafe(
    tmp_path: Path,
    field: str,
    value: object,
    issue: str,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    artifact = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=1),
            _nemesis_record(toolbox=toolbox, index=2),
        ],
        toolbox.root / "nemesis" / f"unsafe-{field}.jsonl",
    )
    rows = _read_jsonl(artifact)
    rows[1]["provenance"][field] = value
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)
    serialized = json.dumps(validation, ensure_ascii=False)

    assert validation["ok"] is False
    assert validation["training_ready"] is False
    assert validation["privacy_safe"] is False
    assert any(issue in finding for finding in validation["issues"])
    assert str(value) not in serialized


def test_nemesis_user_bearing_record_id_marks_privacy_unsafe(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    artifact = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=1),
            _nemesis_record(toolbox=toolbox, index=2),
        ],
        toolbox.root / "nemesis" / "unsafe-record-id.jsonl",
    )
    rows = _read_jsonl(artifact)
    rows[1]["battle_id"] = "tutorial-987654321"
    rows[1]["match_id"] = "tutorial-987654321"
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["ok"] is False
    assert validation["training_ready"] is False
    assert validation["privacy_safe"] is False
    assert any(
        "record_id_not_opaque" in issue
        for issue in validation["issues"]
    )


def test_nemesis_forged_split_fingerprint_is_rejected(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    artifact = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=1),
            _nemesis_record(toolbox=toolbox, index=2),
        ],
        toolbox.root / "nemesis" / "forged-split.jsonl",
    )
    rows = _read_jsonl(artifact)
    rows[1]["provenance"]["split_fingerprint"] = "b" * 64
    rows[1]["provenance"]["split_group"] = f"deck_pair:{'b' * 64}"
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "split_fingerprint does not match" in issue
        for issue in validation["issues"]
    )


def test_nemesis_rejects_card_outside_current_catalog(
    tmp_path: Path,
) -> None:
    toolbox = DatasetToolbox(tmp_path / "datasets")
    artifact = write_nemesis_export(
        [
            _nemesis_record(toolbox=toolbox, index=1),
            _nemesis_record(toolbox=toolbox, index=2),
        ],
        toolbox.root / "nemesis" / "unknown-card.jsonl",
    )
    rows = _read_jsonl(artifact)
    rows[1]["features"]["base"]["seats"]["p1"]["initial_deck"][0][
        "card_id"
    ] = 999999
    base = rows[1]["features"]["base"]
    fingerprint = _deck_pair_split_fingerprint(
        seats=base["seats"],
        ruleset=base["ruleset"],
        catalog_hash=base["catalog_hash"],
    )
    rows[1]["provenance"]["split_fingerprint"] = fingerprint
    rows[1]["provenance"]["split_group"] = f"deck_pair:{fingerprint}"
    _write_jsonl(artifact, rows)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "card_id_not_in_current_catalog" in issue
        for issue in validation["issues"]
    )


def test_v5_materialized_rejects_battle_id_path_escape(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, _ = _copy_v5_artifact(
        valid_v5_materialized,
        "battle-id-path-escape",
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["battle_ids"] = ["../../outside"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "unsafe battle_id" in issue or "path component" in issue
        for issue in validation["issues"]
    )


def test_v5_materialized_rejects_symlinked_battle_directory(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
    tmp_path: Path,
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        "battle-symlink-escape",
    )
    battle_path = artifact / "battles" / battle_id
    outside = tmp_path / "outside-battle"
    shutil.copytree(battle_path, outside)
    shutil.rmtree(battle_path)
    battle_path.symlink_to(outside, target_is_directory=True)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any("symlink" in issue for issue in validation["issues"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("telegram_user_id", 987654321),
        ("telegramId", 987654321),
        ("chatId", 987654321),
        ("player_ids", [1, 987654321]),
        ("userIds", [1, 987654321]),
    ],
)
def test_v5_materialized_rejects_raw_identity_variants_and_lists(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
    field: str,
    value: object,
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        f"raw-identity-{field}",
    )
    meta_path = artifact / "battles" / battle_id / "v5" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[field] = value
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "outside_pseudonymous_seats" in issue
        or "pii_field_forbidden" in issue
        for issue in validation["issues"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "database_dsn",
            "postgresql://alice:dataset-secret@db/prod",
        ),
        ("api_token", "dataset-secret-token"),
        (
            "connection",
            "redis://alice:dataset-secret@cache/0",
        ),
    ],
)
def test_v5_materialized_rejects_nested_sensitive_data(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
    field: str,
    value: str,
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        f"nested-sensitive-{field}",
    )
    meta_path = artifact / "battles" / battle_id / "v5" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["debug_context"] = {field: value}
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)
    serialized = json.dumps(validation, ensure_ascii=False)

    assert validation["training_ready"] is False
    assert validation["privacy_safe"] is False
    assert any(
        "sensitive_field_forbidden" in issue
        or "credential_uri_forbidden" in issue
        for issue in validation["issues"]
    )
    assert value not in serialized
    assert "dataset-secret" not in serialized


def test_v5_materialized_rejects_manifest_battle_result_id_mismatch(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, _ = _copy_v5_artifact(
        valid_v5_materialized,
        "manifest-result-id-mismatch",
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["battles_results"][0]["battle_id"] = "different-battle"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert "manifest_battle_result_ids_mismatch" in validation["issues"]


def test_v5_materialized_rejects_manifest_battle_result_status_mismatch(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        "manifest-result-status-mismatch",
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_status = manifest["battles_results"][0]["status"]
    manifest["battles_results"][0]["status"] = (
        "P2_WIN" if original_status == "P1_WIN" else "P1_WIN"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert (
        f"{battle_id}:manifest_battle_result_status_mismatch"
        in validation["issues"]
    )


def test_v5_materialized_rejects_manifest_aggregate_mismatch(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, _ = _copy_v5_artifact(
        valid_v5_materialized,
        "manifest-aggregate-mismatch",
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["battles_finished"] = 0
    manifest["results"]["avg_duration_seconds"] = 0.0
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert "manifest_results_battles_finished_mismatch" in (
        validation["issues"]
    )
    assert "manifest_results_avg_duration_seconds_mismatch" in (
        validation["issues"]
    )


def test_v5_materialized_rejects_stale_catalog(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        "stale-catalog",
    )
    meta_path = artifact / "battles" / battle_id / "v5" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["catalog_hash"] = "a" * 64
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any("[ruleset]" in issue for issue in validation["issues"])


def test_v5_materialized_rejects_card_outside_current_catalog(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        "unknown-card-id",
    )
    meta_path = artifact / "battles" / battle_id / "v5" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["p1_deck"][0]["card_id"] = 999999
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is False
    assert any(
        "not in the attached current catalog" in issue
        for issue in validation["issues"]
    )


def test_v5_materialized_rejects_missing_auxiliary_timing_field(
    valid_v5_materialized: tuple[DatasetToolbox, Path, str],
) -> None:
    toolbox, artifact, battle_id = _copy_v5_artifact(
        valid_v5_materialized,
        "missing-aux-timing",
    )
    actions_path = (
        artifact / "battles" / battle_id / "v5" / "actions.jsonl"
    )
    actions = _read_jsonl(actions_path)
    actions[0].pop("human_decision_time_raw_ms", None)
    _write_jsonl(actions_path, actions)

    validation = toolbox.validate_artifact(artifact)

    assert validation["training_ready"] is True
    assert validation["v5_policy_training_ready"] is True
    assert validation["metronome_training_ready"] is False
    assert validation["timestamp_training_ready"] is False
    assert any(
        "human_decision_time_raw_ms" in issue
        for issue in validation["issues"]
    )


def test_status_exposes_only_salt_configuration_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "status-secret-never-return-this-value-0123456789"
    salt_env = "TEST_RETURNCLOCK_DATASET_SALT"
    monkeypatch.setenv(salt_env, secret)
    toolbox = DatasetToolbox(
        tmp_path / "datasets",
        returnclock_salt_env=salt_env,
    )

    status = toolbox.status()
    serialized = json.dumps(status, ensure_ascii=False)

    assert status["returnclock_salt_configured"] is True
    assert secret not in serialized
    assert salt_env not in serialized
