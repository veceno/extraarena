from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = ROOT / "TrainV3.5" / "scripts" / "build_nemesis_lite_preview_dataset.py"
SPEC = importlib.util.spec_from_file_location(
    "build_nemesis_lite_preview_dataset",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _campaign(campaign_id: str, checkpoint_hash: str) -> object:
    profile = builder.CHECKPOINT_PROFILES[checkpoint_hash]
    return builder.Campaign(
        campaign_id=campaign_id,
        directory=Path("/source"),
        manifest_path=Path("/source/dataset_manifest.json"),
        manifest={
            "created_at": "2026-07-27T19:21:16Z",
            "ruleset": "python TrainV3ClassicEnv/current production mechanics",
        },
        manifest_sha256="manifest-hash",
        checkpoint_path=Path("/checkpoint.npz"),
        checkpoint_sha256=checkpoint_hash,
        profile=dict(profile),
        campaign_started_at="2026-07-27T18:58:20Z",
        rows=[],
        clean_rows=0,
        eligible_quartets={},
        shard_files=[],
    )


def _row(*, candidate_seat: int, starting_player: int = 2) -> dict:
    candidate = [1, 8, 11, 14, 15, 16, 17, 18, 19]
    opponent = [3, 10, 12, 20, 21, 22, 23, 24, 25]
    return {
        "schema": builder.SOURCE_SCHEMA,
        "battle_id": "syn-00000000",
        "global_index": 0,
        "cell_id": "cell-000000",
        "repeat": 0,
        "seed": 12345,
        "candidate_seat": candidate_seat,
        "starting_player_id": starting_player,
        "candidate_deck_ids": candidate,
        "candidate_levels": {str(card_id): (7 if card_id == 11 else 3) for card_id in candidate},
        "opponent_deck_ids": opponent,
        "opponent_levels": {str(card_id): (9 if card_id == 12 else 4) for card_id in opponent},
        "checkpoint_sha256": (
            "6128da46bc7e79d5dfdae5307e834e1eea6cf30315ef536486aa8e20516be01c"
        ),
        "completed": True,
        "truncated": False,
        "error": None,
        "invalid_actions": 0,
        "status": "p2_win",
        "winner_id": 2,
        "turns": 12,
    }


def test_candidate_seat_maps_exact_decks_levels_and_starter() -> None:
    catalog = builder.load_catalog(ROOT / "ai" / "cards.json")
    campaign = _campaign(
        "u29250-test",
        "6128da46bc7e79d5dfdae5307e834e1eea6cf30315ef536486aa8e20516be01c",
    )

    p1_record, p1_meta, normalized = builder.convert_source_row(
        _row(candidate_seat=1),
        campaign=campaign,
        catalog=catalog,
        ruleset=campaign.manifest["ruleset"],
    )
    p1_base = p1_record["features"]["base"]
    assert [card["card_id"] for card in p1_base["seats"]["p1"]["initial_deck"]] == [
        1, 8, 11, 14, 15, 16, 17, 18, 19
    ]
    assert [card["card_id"] for card in p1_base["seats"]["p2"]["initial_deck"]] == [
        3, 10, 12, 20, 21, 22, 23, 24, 25
    ]
    assert p1_base["starting_player"] == "p2"
    assert p1_record["label"] == {
        "status": "p2_win",
        "winner_seat": "p2",
        "duration_seconds": None,
        "turns_count": 12,
    }
    assert p1_base["seats"]["p1"]["initial_deck"][2]["level"] == 2
    assert p1_base["seats"]["p2"]["initial_deck"][2]["level"] == 2
    assert normalized == 2
    assert p1_meta["candidate_seat"] == 1

    p2_record, p2_meta, _ = builder.convert_source_row(
        _row(candidate_seat=2),
        campaign=campaign,
        catalog=catalog,
        ruleset=campaign.manifest["ruleset"],
    )
    p2_base = p2_record["features"]["base"]
    assert [card["card_id"] for card in p2_base["seats"]["p1"]["initial_deck"]] == [
        3, 10, 12, 20, 21, 22, 23, 24, 25
    ]
    assert [card["card_id"] for card in p2_base["seats"]["p2"]["initial_deck"]] == [
        1, 8, 11, 14, 15, 16, 17, 18, 19
    ]
    assert p1_meta["matchup_group_id"] == p2_meta["matchup_group_id"]


def test_namespaced_ids_remove_legacy_cross_campaign_collision() -> None:
    source_id = "syn-00000000"
    first = builder.namespaced_battle_id(
        "u29250-selfplay",
        "6128da46bc7e79d5dfdae5307e834e1eea6cf30315ef536486aa8e20516be01c",
        source_id,
    )
    second = builder.namespaced_battle_id(
        "h299-selfplay",
        "1c4e95eda44ac663218a35242a81f965fde56922dc20f26f0eea1172dd63251e",
        source_id,
    )

    assert first != second
    assert first.endswith(source_id)
    assert second.endswith(source_id)


def test_build_rejects_duplicate_campaign_ids_before_output(tmp_path: Path) -> None:
    catalog = builder.load_catalog(ROOT / "ai" / "cards.json")
    checkpoint = (
        "6128da46bc7e79d5dfdae5307e834e1eea6cf30315ef536486aa8e20516be01c"
    )
    campaign_a = _campaign("duplicate", checkpoint)
    campaign_b = _campaign("duplicate", checkpoint)

    with pytest.raises(ValueError, match="campaign IDs must be unique"):
        builder.build_preview(
            [campaign_a, campaign_b],
            catalog=catalog,
            output_dir=tmp_path,
            selection_seed=1,
        )


def test_training_metadata_header_is_json_serializable() -> None:
    header = {
        "record_type": "header",
        "format": builder.TRAINING_METADATA_SCHEMA,
        "battle_count": 1,
    }
    assert json.loads(json.dumps(header))["format"] == builder.TRAINING_METADATA_SCHEMA
