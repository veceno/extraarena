from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
TRAIN_PYTHON = ROOT / "TrainV3.5" / "python"
for value in (str(ROOT), str(TRAIN_PYTHON)):
    if value not in sys.path:
        sys.path.insert(0, value)

from ai.nemesis_lite_preview import NemesisLitePreview
from core.nemesis_dataset import (
    NemesisBattleCollector,
    export_nemesis_ndjson,
)
from train_v3.export_nemesis_lite_preview_onnx import (
    export_nemesis_lite_preview,
)
from train_v3.nemesis_lite_preview import (
    CLASS_NAMES,
    ModelConfig,
    NemesisLitePreviewModel,
    TrainingConfig,
    class_prevalence_report,
    classification_metrics,
    evaluate_test_baselines,
    grouped_matchup_split,
    load_catalog_contract,
    load_unified_jsonl,
)
from train_v3.train_nemesis_lite_preview import train_from_paths


CATALOG_PATH = ROOT / "ai" / "cards.json"
RULESET = "classic-v5-nemesis-preview"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_parts() -> tuple[list[int], list[int], dict[int, int]]:
    cards = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    heroes = [
        int(card["id"])
        for card in cards
        if card.get("card_type") == "hero"
    ]
    units = [
        int(card["id"])
        for card in cards
        if card.get("card_type") != "hero"
    ]
    maxima = {
        int(card["id"]): 2 if card.get("simplified_levelup") else 10
        for card in cards
    }
    return heroes, units, maxima


def _deck(seed: int) -> list[dict[str, int]]:
    heroes, units, maxima = _catalog_parts()
    hero = heroes[seed % len(heroes)]
    offset = (seed * 3) % len(units)
    selected = [
        units[(offset + index) % len(units)]
        for index in range(8)
    ]
    card_ids = [hero, *selected]
    return [
        {
            "slot": slot,
            "card_id": card_id,
            "level": 1 + ((seed + slot) % maxima[card_id]),
        }
        for slot, card_id in enumerate(card_ids)
    ]


def _record(
    index: int,
    *,
    p1_seed: int | None = None,
    p2_seed: int | None = None,
    status: str | None = None,
    starting_player: str | None = None,
) -> dict:
    terminal = status or CLASS_NAMES[index % len(CLASS_NAMES)]
    if terminal == "draw":
        terminal = "stalemate" if index % 2 else "draw"
    meta = {
        "battle_id": f"nemesis-{index:04d}",
        "match_id": f"match-{index:04d}",
        "started_at": "2026-07-28T12:00:01Z",
        "game_mode": "classic",
        "ruleset": RULESET,
        "catalog_hash": _sha256(CATALOG_PATH),
        "card_params_schema": "train_v3_card_params_v1",
        "deck_params_schema": "train_v3_deck_params_v1",
        "starting_player": starting_player or ("p1" if index % 2 else "p2"),
        "p1_user_id": -(index * 2 + 1),
        "p2_user_id": -(index * 2 + 2),
        "p1_actor_type": "rl",
        "p2_actor_type": "rl",
        "p1_deck": _deck(index * 2 if p1_seed is None else p1_seed),
        "p2_deck": _deck(index * 2 + 1 if p2_seed is None else p2_seed),
        "model_provenance": {},
        "aux_model_provenance": {},
    }
    return NemesisBattleCollector.from_v5_meta(
        meta,
        feature_cutoff_at="2026-07-28T12:00:00Z",
    ).finalize(status=terminal)


def _write_export(path: Path, records: list[dict]) -> Path:
    path.write_text(
        export_nemesis_ndjson(records),
        encoding="utf-8",
    )
    return path


def test_canonical_nested_export_loads_and_group_holdout_blocks_swap_leakage(
    tmp_path: Path,
) -> None:
    # Rows 0 and 1 are the same exact matchup with seats exchanged.
    records = [
        _record(0, p1_seed=20, p2_seed=21, status="p1_win"),
        _record(
            1,
            p1_seed=21,
            p2_seed=20,
            status="p2_win",
            starting_player="p2",
        ),
        *[_record(index) for index in range(2, 14)],
    ]
    dataset = _write_export(tmp_path / "nemesis.ndjson", records)
    catalog = load_catalog_contract(CATALOG_PATH)

    rows, sources = load_unified_jsonl(
        [dataset],
        catalog=catalog,
        ruleset=RULESET,
    )
    split = grouped_matchup_split(
        rows,
        seed=7,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    assert len(rows) == len(records)
    assert sources == [
        {
            "path": "nemesis.ndjson",
            "sha256": _sha256(dataset),
            "rows": len(records),
            "records": len(records),
            "excluded_rows": 0,
            "catalog_unverified_rows": 0,
        }
    ]
    assert (
        rows[0].source_matchup_group_id
        == records[0]["provenance"]["split_group"]
    )
    assert rows[0].sample_weight == 1.0
    assert rows[0].catalog_verified is True
    assert rows[0].matchup_key == rows[1].matchup_key
    memberships = [
        name
        for name, group_ids in (
            ("train", split.train_groups),
            ("validation", split.validation_groups),
            ("test", split.test_groups),
        )
        if rows[0].matchup_key in group_ids
    ]
    assert len(memberships) == 1
    assert split.manifest()["leakage_free"] is True


def test_unified_quality_gate_weight_and_catalog_unavailable_lite_policy(
    tmp_path: Path,
) -> None:
    excluded = _record(90)
    excluded["quality"] = {
        "eligible_lite": False,
        "eligible_standard": False,
        "sample_weight": 0.0,
        "exclusion_reasons": ["invalid_terminal_trace"],
    }
    unverified = _record(91)
    unverified["features"]["base"]["catalog_hash"] = None
    unverified["features"]["base"]["catalog_available"] = False
    unverified["quality"] = {
        "eligible_lite": True,
        "eligible_standard": False,
        "sample_weight": 0.5,
        "exclusion_reasons": ["catalog_unavailable"],
    }
    dataset = _write_export(
        tmp_path / "quality.ndjson",
        [excluded, unverified],
    )

    rows, sources = load_unified_jsonl(
        [dataset],
        catalog=load_catalog_contract(CATALOG_PATH),
        ruleset=RULESET,
    )

    assert len(rows) == 1
    assert rows[0].battle_id == unverified["battle_id"]
    assert rows[0].sample_weight == 0.5
    assert rows[0].catalog_verified is False
    assert sources[0]["records"] == 2
    assert sources[0]["rows"] == 1
    assert sources[0]["excluded_rows"] == 1
    assert sources[0]["catalog_unverified_rows"] == 1


def test_architecture_is_permutation_invariant_and_exactly_swap_consistent() -> None:
    torch.manual_seed(11)
    model = NemesisLitePreviewModel(
        ModelConfig(
            max_card_id=50,
            deck_size=9,
            embedding_dim=8,
            deck_hidden_dim=12,
            deck_output_dim=10,
            outcome_hidden_dim=14,
        )
    ).eval()
    p1_ids = torch.tensor([[1, 8, 10, 11, 12, 13, 14, 15, 16]])
    p1_levels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10]])
    p2_ids = torch.tensor([[3, 17, 18, 19, 20, 21, 22, 23, 24]])
    p2_levels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    starter = torch.tensor([[1]])

    with torch.no_grad():
        logits = model(p1_ids, p1_levels, p2_ids, p2_levels, starter)
        swapped = model(
            p2_ids,
            p2_levels,
            p1_ids,
            p1_levels,
            1 - starter,
        )
        permuted = model(
            p1_ids.flip(1),
            p1_levels.flip(1),
            p2_ids,
            p2_levels,
            starter,
        )

    torch.testing.assert_close(logits, swapped[:, [2, 1, 0]], atol=1e-7, rtol=0)
    torch.testing.assert_close(logits, permuted, atol=1e-7, rtol=0)


def test_required_classification_metrics_are_well_defined() -> None:
    targets = np.asarray([0, 1, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)

    metrics = classification_metrics(targets, probabilities, ece_bins=3)

    assert metrics["accuracy"] == 1.0
    assert metrics["logloss"] == pytest.approx(0.0)
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)
    assert list(metrics).count("logloss") == 1


def test_baselines_fit_train_only_and_draw_prevalence_is_descriptive(
    tmp_path: Path,
) -> None:
    dataset = _write_export(
        tmp_path / "baseline.ndjson",
        [_record(index) for index in range(6)],
    )
    rows, _ = load_unified_jsonl(
        [dataset],
        catalog=load_catalog_contract(CATALOG_PATH),
        ruleset=RULESET,
    )
    train = [
        replace(rows[0], target=0, starting_side=1),
        replace(rows[1], target=0, starting_side=1),
        replace(rows[2], target=2, starting_side=0),
        replace(rows[3], target=1, starting_side=0),
    ]
    test = [
        replace(rows[4], target=0, starting_side=1),
        replace(rows[5], target=1, starting_side=0),
    ]

    baselines = evaluate_test_baselines(train, test)
    prevalence = class_prevalence_report(test)

    assert baselines["fit_scope"] == "grouped_train_only"
    assert baselines["evaluation_scope"] == "grouped_test_only"
    assert baselines["majority_class"]["predicted_class"] == "p1_win"
    assert baselines["majority_class"]["train_probabilities"] == {
        "p1_win": 0.5,
        "draw": 0.25,
        "p2_win": 0.25,
    }
    assert baselines["starter_only_empirical"]["train_probabilities"] == {
        "p2_starts": {"p1_win": 0.0, "draw": 0.5, "p2_win": 0.5},
        "p1_starts": {"p1_win": 1.0, "draw": 0.0, "p2_win": 0.0},
    }
    assert prevalence["class_counts"] == {
        "p1_win": 1,
        "draw": 1,
        "p2_win": 0,
    }
    assert prevalence["draw_observation"]["row_prevalence"] == 0.5
    assert "descriptive" in prevalence["draw_observation"]["note"]


def test_tiny_train_export_and_runtime_contract(
    tmp_path: Path,
) -> None:
    source_a = [_record(index) for index in range(18)]
    source_b = [
        _record(
            100 + index,
            p1_seed=index * 2,
            p2_seed=index * 2 + 1,
        )
        for index in range(18)
    ]
    for record in source_a:
        record["provenance"]["checkpoint_mix"] = ["h299"]
    for record in source_b:
        record["provenance"]["checkpoint_mix"] = ["u18500"]
    dataset_a = _write_export(
        tmp_path / "nemesis-a.ndjson",
        source_a,
    )
    dataset_b = _write_export(
        tmp_path / "nemesis-b.ndjson",
        source_b,
    )
    output_dir = tmp_path / "trained"
    training = train_from_paths(
        [dataset_a, dataset_b],
        output_dir=output_dir,
        ruleset=RULESET,
        catalog_path=CATALOG_PATH,
        config=TrainingConfig(
            seed=3,
            epochs=3,
            batch_size=8,
            patience=3,
            validation_fraction=0.2,
            test_fraction=0.2,
            embedding_dim=8,
            deck_hidden_dim=12,
            deck_output_dim=10,
            outcome_hidden_dim=14,
        ),
    )
    artifact = Path(training["artifact_path"])
    onnx_path = tmp_path / "extra_lr_nemesis_lite_preview.onnx"
    export_result = export_nemesis_lite_preview(artifact, onnx_path)

    assert artifact.is_file()
    assert onnx_path.is_file()
    assert Path(str(onnx_path) + ".json").is_file()
    assert training["split"]["leakage_free"] is True
    for split_name in ("train_unaugmented", "validation", "test"):
        metrics = training["training"]["metrics"][split_name]
        assert {
            "accuracy",
            "logloss",
            "brier",
            "ece",
            "swap_consistency",
        }.issubset(metrics)
        assert metrics["swap_consistency"]["max_abs"] < 1.0e-6
        assert "class_counts" in metrics["observed_labels"]
        assert "draw_observation" in metrics["observed_labels"]
    baselines = training["training"]["metrics"]["test_baselines"]
    assert baselines["fit_scope"] == "grouped_train_only"
    assert {
        "accuracy",
        "logloss",
        "brier",
        "ece",
    }.issubset(baselines["majority_class"]["metrics"])
    breakdowns = training["training"]["metrics"]["test_breakdowns"]
    assert breakdowns["scope"] == "grouped_test_only"
    assert {
        item["source"] for item in breakdowns["by_source"]
    } == {"nemesis-a.ndjson", "nemesis-b.ndjson"}
    assert {
        tuple(item["checkpoint_mix"])
        for item in breakdowns["by_checkpoint_mix"]
    } == {("h299",), ("u18500",)}
    assert len(breakdowns["by_source_checkpoint"]) == 2
    prevalence = training["training"]["metrics"]["observed_class_prevalence"]
    assert set(prevalence) == {
        "all",
        "train",
        "validation",
        "test",
        "interpretation",
    }
    assert export_result["max_abs_swap_consistency_error"] < 2.0e-5

    runtime = NemesisLitePreview(
        onnx_path,
        ruleset=RULESET,
        catalog_path=CATALOG_PATH,
    )
    p1_deck = [
        {"card_id": row["card_id"], "level": row["level"]}
        for row in _deck(40)
    ]
    p2_deck = [
        {"card_id": row["card_id"], "level": row["level"]}
        for row in _deck(41)
    ]
    try:
        original = runtime.predict(
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            starting_player="p1",
        )
        swapped = runtime.predict(
            p1_deck=p2_deck,
            p2_deck=p1_deck,
            starting_player="p2",
        )
    finally:
        runtime.close()

    assert original["probabilities"]["p1_win"] == pytest.approx(
        swapped["probabilities"]["p2_win"],
        abs=2.0e-5,
    )
    assert original["probabilities"]["draw"] == pytest.approx(
        swapped["probabilities"]["draw"],
        abs=2.0e-5,
    )
    assert original["probabilities"]["p2_win"] == pytest.approx(
        swapped["probabilities"]["p1_win"],
        abs=2.0e-5,
    )
