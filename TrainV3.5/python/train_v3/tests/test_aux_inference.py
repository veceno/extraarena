from __future__ import annotations

import importlib.util
import json
import random
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_v3.aux_inference import (
    AssemblerV1,
    CardOptimumV1,
    ForcedDrawRandom,
    MetronomeV1,
    TimeStampDuoV1,
    TimeStampMonoV1,
    cardoptimum_features,
    deck_vector,
    metronome_features_from_trace,
    trace_visible_state,
)


ROOT = Path(__file__).resolve().parents[4]
AUX_DATA = (
    ROOT / "TrainV3.5/runs/aux_synthetic_v1_u29250_10000_20260723_2247"
)
AUX_MODELS = (
    ROOT / "TrainV3.5/runs/phase_c_aux_v1_u29250_h299_20260727/models"
)


def _training_module():
    path = ROOT / "TrainV3.5/scripts/train_aux_models_v1.py"
    spec = importlib.util.spec_from_file_location("train_aux_models_v1_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


def test_deck_vector_matches_training_contract() -> None:
    training = _training_module()
    row = _first_jsonl(AUX_DATA / "assembler_matchups.jsonl")
    np.testing.assert_allclose(
        deck_vector(row["candidate_deck_ids"], row["candidate_levels"]),
        training._deck_vector(row["candidate_deck_ids"], row["candidate_levels"]),
    )


def test_assembler_runtime_prediction_matches_training_contract() -> None:
    training = _training_module()
    row = _first_jsonl(AUX_DATA / "assembler_matchups.jsonl")
    runtime = AssemblerV1(AUX_MODELS / "extra_lr_assembler_v1.npz")
    runtime_score = runtime.score(
        candidate_deck_ids=row["candidate_deck_ids"],
        opponent_deck_ids=row["opponent_deck_ids"],
        allowed_pool_ids=row["allowed_pool_ids"],
        candidate_levels=row["candidate_levels"],
        opponent_levels=row["opponent_levels"],
    )
    with np.load(
        AUX_MODELS / "extra_lr_assembler_v1.npz",
        allow_pickle=False,
    ) as artifact:
        training_score = float(
            np.clip(training._predict(artifact, training._assembler_features(row)), 0, 1)
        )
    assert runtime_score == training_score


def test_cardoptimum_features_match_training_contract() -> None:
    training = _training_module()
    row = _first_jsonl(AUX_DATA / "cardoptimum_counterfactual.jsonl")
    score = row["candidate_scores"][0]
    np.testing.assert_allclose(
        cardoptimum_features(row["state"], int(score["card_id"])),
        training._cardopt_features(row["state"], score),
    )
    runtime = CardOptimumV1(AUX_MODELS / "extra_lr_cardoptimum_v1.npz")
    assert runtime.model["coef"].shape == (82,)


def test_forced_draw_rng_selects_target_and_advances_base_stream() -> None:
    def card(card_id: int, mana_cost: int, skip_count: int = 0):
        return SimpleNamespace(
            card_id=card_id,
            mana_cost=mana_cost,
            skip_count=skip_count,
        )

    base = random.Random(1234)
    control = random.Random(1234)
    wrapper = ForcedDrawRandom(base)
    player = SimpleNamespace(
        hand=[card(90, 1)],
        deck=[card(10, 1), card(20, 3), card(30, 5)],
    )
    assert wrapper.arm(player, 20)
    forced_value = wrapper.random()
    control.random()
    assert base.random() == control.random()

    weights = [1.8, 1.5, 1.8]
    target = forced_value * sum(weights)
    assert weights[0] < target < weights[0] + weights[1]
    cloned = copy.deepcopy(wrapper)
    assert isinstance(cloned, ForcedDrawRandom)
    assert cloned.base is not wrapper.base


def test_trace_metronome_projection_is_human_visible_and_stateful() -> None:
    row = {
        "actor_player": 1,
        "action_type": "attack",
        "legal_action_count": 7,
        "pre_state": {
            "turn_number": 5,
            "p1": {
                "hero": {"card_id": 1, "hp": 30, "max_hp": 40},
                "mana": 4,
                "max_mana": 6,
                "mana_draw_count_this_turn": 1,
                "hand": [{"card_id": 8, "is_ready": False}],
                "deck": [{"card_id": 9}, {"card_id": 10}],
                "board": [
                    {
                        "card_id": 20,
                        "attack": 6,
                        "hp": 4,
                        "max_hp": 8,
                        "is_ready": True,
                    }
                ],
                "graveyard": [],
            },
            "p2": {
                "hero": {"card_id": 2, "hp": 20, "max_hp": 50},
                "mana": 3,
                "max_mana": 5,
                "hand": [{"card_id": 30}, {"card_id": 31}],
                "deck": [{"card_id": 32}],
                "board": [],
            },
        },
    }
    visible = trace_visible_state(row["pre_state"], 1)
    assert visible["actor"]["hand"][0]["card_id"] == 8
    assert "hand" not in visible["opponent"]
    assert visible["opponent"]["hand_count"] == 2
    features = metronome_features_from_trace(row)
    assert features.shape == (26,)
    assert np.count_nonzero(features[:20]) >= 10
    assert features[1] == 0.75
    assert features[2] == 0.4
    assert features[14] == 0.2
    assert features[-1] == 1.0


def test_metronome_trace_prediction_and_timestamp_mono_duo_contracts() -> None:
    training = _training_module()
    trace_path = next(
        (
            ROOT
            / "TrainV3.5/runs/phase_c_human_freeze_u29250_299_20260727/sessions"
        ).glob("*/battles/*/v5/actions.jsonl")
    )
    trace_row = next(
        row
        for row in map(json.loads, trace_path.read_text(encoding="utf-8").splitlines())
        if row.get("decision_source") == "human" and row.get("accepted") is True
    )
    np.testing.assert_allclose(
        training._human_action_features(trace_row),
        metronome_features_from_trace(trace_row),
    )
    metronome = MetronomeV1(AUX_MODELS / "extra_lr_metronome_v1.npz")
    prediction = metronome.predict_trace(trace_row)
    assert 100.0 <= prediction["point"] <= 25_000.0
    assert prediction["p90"] >= prediction["p50"]

    battle = _first_jsonl(
        ROOT
        / "TrainV3.5/runs/phase_c_aux_v1_u29250_h299_20260727/datasets"
        / "timestamp_human.jsonl"
    )
    arguments = {
        "actor_deck_ids": battle["mono_deck_ids"],
        "opponent_deck_ids": battle["opponent_deck_ids"],
        "actor_levels": {
            int(card_id): int(level)
            for card_id, level in battle["mono_levels"].items()
        },
        "opponent_levels": {
            int(card_id): int(level)
            for card_id, level in battle["opponent_levels"].items()
        },
        "actor_starts": battle["starting_player_relative"] == "first",
    }
    mono = TimeStampMonoV1(AUX_MODELS / "extra_lr_timestamp_v1_mono.npz")
    duo = TimeStampDuoV1(AUX_MODELS / "extra_lr_timestamp_v1_duo.npz")
    results = {
        "mono": mono.predict(
            actor_deck_ids=arguments["actor_deck_ids"],
            actor_levels=arguments["actor_levels"],
            actor_starts=arguments["actor_starts"],
        ),
        "duo": duo.predict(**arguments),
    }
    for result in results.values():
        assert result["turns"] >= 1.0
        assert result["duration_seconds"] >= 0.0
        assert result["duration_p90_seconds"] >= result["duration_p50_seconds"]
    for label, is_duo in (("mono", False), ("duo", True)):
        with np.load(
            AUX_MODELS / f"extra_lr_timestamp_v1_{label}.npz",
            allow_pickle=False,
        ) as artifact:
            artifact = {key: np.asarray(artifact[key]) for key in artifact.files}
        turn_model = {
            key.removeprefix("turn_"): value
            for key, value in artifact.items()
            if key.startswith("turn_")
        }
        duration_model = {
            key.removeprefix("duration_"): value
            for key, value in artifact.items()
            if key.startswith("duration_")
            and key != "duration_residual_log_quantiles"
        }
        expected_log_turns = float(
            training._predict(
                turn_model,
                training._timestamp_features(battle, duo=is_duo),
            )
        )
        expected_log_duration = float(
            training._predict(
                duration_model,
                training._duration_calibration_features(
                    battle,
                    expected_log_turns,
                    duo=is_duo,
                    include_deck_summary=True,
                ),
            )
        )
        assert results[label]["turns"] == np.expm1(expected_log_turns)
        assert results[label]["duration_seconds"] == np.expm1(
            expected_log_duration
        )
