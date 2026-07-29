from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai.aux_models import (
    CARD_CATALOG,
    CARD_IDS,
    CARD_INDEX,
    HERO_CARD_IDS,
    ExtraLRAuxRuntime,
    cardoptimum_features,
    deck_summary,
    deck_vector,
    effective_card_level,
    metronome_features,
    pool_vector,
    state_scalars,
    visible_state,
)
from core.engine import draw_one_from_deck


def _card(card_id: int, *, level: int = 4, skip_count: int = 0) -> SimpleNamespace:
    row = CARD_CATALOG[int(card_id)]
    hp = int(row.get("base_hp", 0) or 0)
    return SimpleNamespace(
        card_id=int(card_id),
        level=level,
        attack=int(row.get("base_attack", 0) or 0),
        hp=hp,
        max_hp=hp,
        mana_cost=int(row.get("mana_cost", 0) or 0),
        is_ready=True,
        is_frozen=False,
        skip_count=skip_count,
        mechanics=list(row.get("mechanics") or []),
    )


def _state() -> SimpleNamespace:
    heroes = sorted(HERO_CARD_IDS)
    nonheroes = [card_id for card_id in CARD_IDS if card_id not in HERO_CARD_IDS]
    p1 = SimpleNamespace(
        user_id=101,
        hero=_card(heroes[0]),
        mana=5,
        max_mana=7,
        mana_draw_count_this_turn=1,
        hand=[_card(nonheroes[0])],
        deck=[
            _card(nonheroes[1], skip_count=0),
            _card(nonheroes[2], skip_count=1),
            _card(nonheroes[3], skip_count=2),
        ],
        board=[_card(nonheroes[4])],
        graveyard=[],
    )
    p2 = SimpleNamespace(
        user_id=202,
        hero=_card(heroes[1]),
        mana=3,
        max_mana=6,
        mana_draw_count_this_turn=0,
        hand=[_card(nonheroes[5]), _card(nonheroes[6])],
        deck=[_card(nonheroes[7]), _card(nonheroes[8])],
        board=[],
        graveyard=[_card(nonheroes[9])],
    )
    return SimpleNamespace(p1=p1, p2=p2, turn_number=8)


def _legal_deck(hero_index: int = 0, unit_offset: int = 0) -> list[int]:
    heroes = sorted(HERO_CARD_IDS)
    nonheroes = [card_id for card_id in CARD_IDS if card_id not in HERO_CARD_IDS]
    return [heroes[hero_index], *nonheroes[unit_offset : unit_offset + 8]]


def test_exact_50_card_feature_contract_dimensions() -> None:
    state = _state()
    snapshot = visible_state(state, state.p1.user_id)
    candidate = int(state.p1.deck[0].card_id)

    assert len(CARD_IDS) == 50
    assert len(CARD_IDS) == len(set(CARD_IDS))
    assert deck_vector(_legal_deck()).shape == (100,)
    assert pool_vector(CARD_IDS).shape == (50,)
    assert state_scalars(snapshot).shape == (20,)
    assert cardoptimum_features(snapshot, candidate).shape == (82,)
    assert metronome_features(
        snapshot,
        action_type="attack",
        legal_action_count=17,
        actor_is_p1=True,
    ).shape == (26,)
    assert deck_summary(_legal_deck(), {}).shape == (9,)
    assert "hand" not in snapshot["opponent"]
    assert snapshot["opponent"]["hand_count"] == 2


def test_real_onnx_bundle_and_deterministic_legal_assembler_decks() -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    try:
        opponent = _legal_deck()
        disabled = {max(CARD_IDS), min(HERO_CARD_IDS)}
        first = runtime.assemble_deck(
            opponent,
            seed=20260728,
            disabled_card_ids=disabled,
            candidate_count=64,
        )
        second = runtime.assemble_deck(
            opponent,
            seed=20260728,
            disabled_card_ids=disabled,
            candidate_count=64,
        )
        permuted = runtime.assemble_deck(
            list(reversed(opponent)),
            seed=20260728,
            disabled_card_ids=disabled,
            candidate_count=64,
        )

        assert first == second
        assert first == permuted
        deck = first["deck_ids"]
        assert len(deck) == 9
        assert len(set(deck)) == 9
        assert sum(card_id in HERO_CARD_IDS for card_id in deck) == 1
        assert not set(deck).intersection(disabled)
        assert first["telemetry"]["candidates_scored"] == 64
        assert math.isfinite(first["telemetry"]["raw_score"])

        allowed, candidates = runtime.assembler.generate_candidates(
            seed=7,
            candidate_count=8,
        )
        scores = runtime.assembler.score_candidates(
            candidates=candidates,
            opponent_deck_ids=opponent,
            allowed_pool_ids=allowed,
        )
        assert scores.shape == (8,)
        assert np.isfinite(scores).all()
    finally:
        runtime.close()


def test_assembler_ranking_is_genuinely_conditioned_on_the_opponent() -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    try:
        allowed, candidates = runtime.assembler.generate_candidates(
            seed=5,
            candidate_count=256,
        )
        heroes = sorted(HERO_CARD_IDS)
        nonheroes = [
            card_id for card_id in CARD_IDS if card_id not in HERO_CARD_IDS
        ]
        opponent_a = [heroes[0], *nonheroes[:8]]
        opponent_b = [heroes[0], *nonheroes[32:40]]
        score_a = runtime.assembler.score_candidates(
            candidates=candidates,
            opponent_deck_ids=opponent_a,
            allowed_pool_ids=allowed,
        )
        score_b = runtime.assembler.score_candidates(
            candidates=candidates,
            opponent_deck_ids=opponent_b,
            allowed_pool_ids=allowed,
        )

        # A mere concatenated ridge shifts every candidate by one constant.
        # The bilinear counter-card contract must change score gaps and the
        # actual selected deck while keeping deck-order invariance.
        assert np.ptp(score_a - score_b) > 0.01
        assert int(np.argmax(score_a)) != int(np.argmax(score_b))
        permuted = runtime.assembler.score_candidates(
            candidates=candidates,
            opponent_deck_ids=list(reversed(opponent_a)),
            allowed_pool_ids=allowed,
        )
        np.testing.assert_allclose(score_a, permuted, rtol=0.0, atol=1.0e-7)
    finally:
        runtime.close()


def test_auxiliary_components_load_independently(tmp_path) -> None:
    """An experimental TimeStamp failure must not disable live assistants."""

    source_dir = Path("ai/models")
    for stem in (
        "extra_lr_assembler_v1.onnx",
        "extra_lr_cardoptimum_v1.onnx",
        "extra_lr_metronome_v1.onnx",
    ):
        shutil.copy2(source_dir / stem, tmp_path / stem)
        shutil.copy2(
            Path(str(source_dir / stem) + ".json"),
            Path(str(tmp_path / stem) + ".json"),
        )

    runtime = ExtraLRAuxRuntime.from_model_dir(tmp_path)
    try:
        assert runtime.availability == {
            "assembler": True,
            "cardoptimum": True,
            "metronome": True,
            "timestamp_mono": False,
            "timestamp_duo": False,
        }
    finally:
        runtime.close()


def test_nonfinite_onnx_output_is_rejected(monkeypatch) -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()

    class NonFiniteSession:
        def run(self, *_args, **_kwargs):
            return [np.asarray([[np.nan]], dtype=np.float32)]

    try:
        monkeypatch.setattr(runtime.metronome, "_session", NonFiniteSession())
        with pytest.raises(RuntimeError, match="non-finite ONNX output"):
            runtime.metronome.predict_ms(
                _state(),
                101,
                action_type="end_turn",
                legal_action_count=1,
            )
    finally:
        runtime.close()


def test_assembler_scores_the_levels_that_the_live_deck_will_receive(
    monkeypatch,
) -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    candidate = _legal_deck()
    opponent = _legal_deck(hero_index=1, unit_offset=8)
    candidate_slot_levels = tuple(range(2, 11))
    opponent_levels = {
        card_id: 10 - index
        for index, card_id in enumerate(opponent)
    }
    captured = {}

    def fake_run(**inputs):
        captured["features"] = inputs["features"].copy()
        return (np.zeros((len(inputs["features"]), 1), dtype=np.float32),)

    monkeypatch.setattr(runtime.assembler, "run", fake_run)
    try:
        runtime.assembler.score_candidates(
            candidates=[candidate],
            opponent_deck_ids=opponent,
            allowed_pool_ids=CARD_IDS,
            candidate_slot_levels=candidate_slot_levels,
            opponent_levels=opponent_levels,
        )
    finally:
        runtime.close()

    features = captured["features"][0]
    assert features.shape == (2753,)
    for card_id, level in zip(candidate, candidate_slot_levels, strict=True):
        assert features[50 + CARD_INDEX[card_id]] == pytest.approx(
            effective_card_level(card_id, level) / 10.0
        )
    for card_id, level in opponent_levels.items():
        assert features[100 + 50 + CARD_INDEX[card_id]] == pytest.approx(
            effective_card_level(card_id, level) / 10.0
        )


def test_cardoptimum_ranking_and_draw_rng_alignment_and_isolation() -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    try:
        state = _state()
        ranking = runtime.cardoptimum.rank(state, state.p1.user_id)
        assert {row["card_id"] for row in ranking} == {
            card.card_id for card in state.p1.deck
        }
        assert all(math.isfinite(float(row["score"])) for row in ranking)

        base = random.Random(1234)
        control = random.Random(1234)
        wrapper = runtime.wrap_draw_rng(
            base,
            state=state,
            assisted_player_id=state.p1.user_id,
        )
        isolated = runtime.wrap_draw_rng(
            random.Random(1234),
            state=state,
            assisted_player_id=state.p1.user_id,
        )
        assert wrapper.prepare_draw(state.p1)
        assert wrapper.last_decision is not None
        selected = int(wrapper.last_decision["selected_card_id"])

        forced_value = wrapper.random()
        control.random()  # The wrapper consumed the matching natural draw.
        assert base.random() == control.random()

        weights = []
        cheap = sum(card.mana_cost <= 2 for card in state.p1.hand)
        expensive = sum(card.mana_cost >= 4 for card in state.p1.hand)
        for card in state.p1.deck:
            bias = 0.0
            if card.mana_cost <= 2:
                bias = max(0, 1 - cheap) * 0.3
            elif card.mana_cost >= 4:
                bias = max(0, 1 - expensive) * 0.3
            weights.append(1.0 + (card.skip_count + 1) * 0.5 + bias)
        selected_index = next(
            index
            for index, card in enumerate(state.p1.deck)
            if card.card_id == selected
        )
        weighted_target = forced_value * sum(weights)
        assert sum(weights[:selected_index]) < weighted_target
        assert weighted_target < sum(weights[: selected_index + 1])

        # Pending choices are per-wrapper.  An untouched match remains natural.
        isolated_control = random.Random(1234)
        assert isolated.random() == isolated_control.random()

        # Wrong-player preparation fails open and still preserves RNG parity.
        wrong_base = random.Random(99)
        wrong_control = random.Random(99)
        wrong = runtime.wrap_draw_rng(
            wrong_base,
            state=state,
            assisted_player_id=state.p1.user_id,
        )
        assert not wrong.prepare_draw(state.p2)
        assert wrong.random() == wrong_control.random()

        # The real weighted-draw hook lands on the card selected by the ONNX
        # model, including the engine's pre-draw skip_count increment.
        live_state = _state()
        live_wrapper = runtime.wrap_draw_rng(
            random.Random(5),
            state=live_state,
            assisted_player_id=live_state.p1.user_id,
        )
        expected = runtime.cardoptimum.choose(
            live_state,
            live_state.p1.user_id,
        )["selected_card_id"]
        assert draw_one_from_deck(
            live_state.p1,
            overdraw_to_discard=False,
            source="unit-test",
            rng=live_wrapper,
        )
        assert live_state.p1.hand[-1].card_id == expected
    finally:
        runtime.close()


def test_metronome_sampling_and_timestamp_real_onnx_bounds() -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    try:
        state = _state()
        prediction = runtime.metronome.predict_ms(
            state,
            state.p1.user_id,
            action_type="play_card",
            legal_action_count=23,
        )
        assert set(prediction) == {"point", "p50", "p90"}
        assert all(100.0 <= value <= 25_000.0 for value in prediction.values())
        assert prediction["p90"] >= prediction["p50"]

        sample_a = runtime.metronome.sample_ms(
            state,
            state.p1.user_id,
            action_type="play_card",
            legal_action_count=23,
            rng=random.Random(51),
        )
        sample_b = runtime.metronome.sample_ms(
            state,
            state.p1.user_id,
            action_type="play_card",
            legal_action_count=23,
            rng=random.Random(51),
        )
        assert sample_a == sample_b
        assert 100.0 <= sample_a <= 25_000.0

        actor_deck = _legal_deck()
        opponent_deck = _legal_deck(hero_index=1, unit_offset=8)
        mono = runtime.timestamp_mono.predict(
            actor_deck_ids=actor_deck,
            actor_starts=True,
        )
        duo = runtime.timestamp_duo.predict(
            actor_deck_ids=actor_deck,
            opponent_deck_ids=opponent_deck,
            actor_starts=False,
        )
        for estimate in (mono, duo):
            assert estimate["turns"] >= 1.0
            assert estimate["duration_seconds"] >= 0.0
            assert estimate["duration_p90_seconds"] >= 0.0
            assert all(math.isfinite(value) for value in estimate.values())
    finally:
        runtime.close()


def test_closed_runtime_rejects_inference() -> None:
    runtime = ExtraLRAuxRuntime.from_model_dir()
    runtime.close()
    with pytest.raises(RuntimeError, match="session is closed"):
        runtime.assembler.score(
            candidate_deck_ids=_legal_deck(),
            opponent_deck_ids=_legal_deck(hero_index=1, unit_offset=8),
            allowed_pool_ids=CARD_IDS,
        )
