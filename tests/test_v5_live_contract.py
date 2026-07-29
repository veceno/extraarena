"""Narrow regressions for the production V5 observation/action bridge."""
from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ai.bot_brain import BerserkInference
from ai.aux_models import ExtraLRAuxRuntime
from ai.train_v2.v5_contracts import (
    ENEMY_DECK_SLOTS,
    ENEMY_HAND_SLOTS,
    HISTORY_EVENT_DIM,
    HISTORY_EVENT_SOURCE_OFFSET,
    HISTORY_EVENTS,
    OBS_V1_DIM,
    OWN_DECK_SLOTS,
    OWN_HAND_SLOTS,
    PRIVATE_CARD_SLOT_DIM,
    PRIVATE_INFO_DIM,
    V5_GLOBAL_DIM,
)
from battle_engine import BattleEngine
from core.actions import AttackAction, EndTurnAction, PlayCardAction
from core.engine import ArenaEnvironment, MANA_DRAW_BASE
from core.state import (
    V5_HISTORY_EVENTS,
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
)


def _card(card_id: int, name: str, *, card_type: CardType = CardType.WARRIOR) -> CardInstance:
    return CardInstance(
        instance_id=uuid4(),
        card_id=card_id,
        name=name,
        card_type=card_type,
        mana_cost=1,
        attack=2,
        hp=3,
        max_hp=3,
        mechanics=[],
        is_ready=True,
    )


def _hero(card_id: int, name: str) -> CardInstance:
    return _card(card_id, name, card_type=CardType.HERO)


def _private_slot_offset(slot: int) -> int:
    return OBS_V1_DIM + V5_GLOBAL_DIM + slot * PRIVATE_CARD_SLOT_DIM


def _latest_history_vector(state: GameState, player_id: int = 1) -> np.ndarray:
    from ai.train_v2.obs_v5 import encode_observation_v5

    obs = encode_observation_v5(
        state,
        player_id,
        history_events=list(state.v5_history_events),
    )
    history_base = OBS_V1_DIM + V5_GLOBAL_DIM + PRIVATE_INFO_DIM
    last_event = history_base + (HISTORY_EVENTS - 1) * HISTORY_EVENT_DIM
    return obs[last_event : last_event + HISTORY_EVENT_DIM]


def test_v5_live_path_uses_full_private_info_and_phase_c_history(monkeypatch):
    """The deployed encoder must not fall back to hidden-enemy defaults."""
    own_hand = _card(101, "own-hand")
    own_deck = _card(102, "own-deck")
    enemy_hand = _card(201, "enemy-hand")
    enemy_deck_first = _card(202, "enemy-deck-first")
    enemy_deck_second = _card(203, "enemy-deck-second")
    state = GameState(
        p1=PlayerState(
            user_id=1,
            is_bot=True,
            hero=_hero(1, "bot-hero"),
            mana=0,
            max_mana=0,
            hand=[own_hand],
            deck=[own_deck],
        ),
        p2=PlayerState(
            user_id=2,
            hero=_hero(2, "human-hero"),
            mana=0,
            max_mana=0,
            hand=[enemy_hand],
            deck=[enemy_deck_first, enemy_deck_second],
        ),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
        history=[{"type": "play_card", "hand_index": 0}],
        v5_history_events=deque(
            [
                {
                    "actor_id": 1,
                    "action_id": 1,
                    "action_type": "play_card",
                    "enemy_hero_hp_delta": 0,
                    "own_hero_hp_delta": 0,
                    "my_board_count_delta": 1,
                    "enemy_board_count_delta": 0,
                    "board_power_delta": 6.0,
                    "turn_number": 1,
                    "source_card": None,
                    "target_card": None,
                }
            ],
            maxlen=V5_HISTORY_EVENTS,
        ),
    )
    legal = [EndTurnAction()]

    import ai.train_v2.classic_actions_v1 as action_codec
    import ai.train_v2.obs_v5 as obs_codec

    captured: dict[str, object] = {}
    real_encode = obs_codec.encode_observation_v5

    def spy_encode(game_state, player_id, **kwargs):
        captured["game_state"] = game_state
        captured["player_id"] = player_id
        captured.update(kwargs)
        encoded = real_encode(game_state, player_id, **kwargs)
        captured["encoded"] = encoded.copy()
        return encoded

    mask = np.zeros(601, dtype=np.float32)
    mask[0] = 1.0
    monkeypatch.setattr(obs_codec, "encode_observation_v5", spy_encode)
    monkeypatch.setattr(action_codec, "build_action_mask", lambda *_a, **_k: mask.copy())
    monkeypatch.setattr(
        action_codec,
        "encode_action_features",
        lambda *_a, **_k: np.zeros((601, 171), dtype=np.float32),
    )
    monkeypatch.setattr(action_codec, "decode_action", lambda *_a, **_k: legal[0])

    class FakeSession:
        def run(self, output_names, input_feed):
            captured["output_names"] = output_names
            captured["onnx_observation"] = input_feed["observation"].copy()
            logits = np.full((1, 601), -1.0, dtype=np.float32)
            logits[0, 0] = 1.0
            return [
                logits,
                np.zeros((1, 1), dtype=np.float32),
                np.asarray([[-10.0]], dtype=np.float32),
            ]

    profile = {
        "session": FakeSession(),
        "obs_dim": 7128,
        "max_candidate_actions": 601,
        "action_feature_dim": 171,
        "placement_mode": "append_only",
        "verify_mask": False,
        "input_names": ["observation", "action_features"],
        "output_names": ["logits", "value", "mana_draw_logit"],
    }

    brain = BerserkInference.__new__(BerserkInference)
    assert brain._get_action_v5(state, 1, legal, "v5-test", profile) == 0

    info_mode = captured["info_mode"]
    assert info_mode.own_hand_identity_known is True
    assert info_mode.own_deck_known is True
    assert info_mode.enemy_hand_known is True
    assert info_mode.enemy_deck_known is True
    assert info_mode.enemy_deck_order_known is True
    assert info_mode.draw_assist_enabled is False

    assist_mode = captured["assist_mode"]
    assert assist_mode.assembler_enabled is False
    assert assist_mode.desirerer_enabled is False
    assert assist_mode.teacher_hint_available is False
    assert captured["history_events"] == list(state.v5_history_events)
    assert captured["history_events"] != state.history

    obs = captured["onnx_observation"][0]
    # Slot groups are own hand, own deck, enemy hand, enemy deck.  Assert
    # identities from every private zone reached the actual ONNX input.
    own_hand_slot = 0
    own_deck_slot = OWN_HAND_SLOTS
    enemy_hand_slot = OWN_HAND_SLOTS + OWN_DECK_SLOTS
    enemy_deck_slot = enemy_hand_slot + ENEMY_HAND_SLOTS
    for slot, card_id in (
        (own_hand_slot, 101),
        (own_deck_slot, 102),
        (enemy_hand_slot, 201),
        (enemy_deck_slot, 202),
        (enemy_deck_slot + 1, 203),
    ):
        base = _private_slot_offset(slot)
        assert obs[base] == 1.0
        assert obs[base + 1] == pytest.approx(card_id / 1000.0)

    # The first/second enemy-deck slots preserve the live order.
    assert enemy_deck_slot + 1 < (
        OWN_HAND_SLOTS + OWN_DECK_SLOTS + ENEMY_HAND_SLOTS + ENEMY_DECK_SLOTS
    )
    assert obs[OBS_V1_DIM + 6] == pytest.approx(1.0 / 20.0)


def test_execute_bot_action_runs_live_mana_draw():
    """The V5 parallel mana head's action dict reaches the real core engine."""
    drawn = _card(301, "drawn-card")
    state = GameState(
        p1=PlayerState(
            user_id=1,
            is_bot=True,
            hero=_hero(1, "bot-hero"),
            mana=4,
            max_mana=10,
            hand=[],
            deck=[drawn],
        ),
        p2=PlayerState(
            user_id=2,
            hero=_hero(2, "human-hero"),
            mana=0,
            max_mana=0,
        ),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
    )
    engine = BattleEngine(match_id="v5-live-mana", player_ids=[1, 2], is_bot_match=True)
    engine._arena = ArenaEnvironment(state)
    engine.current_player_id = 1

    result = engine.execute_bot_action({"type": "mana_draw"})

    assert result["success"] is True
    assert state.p1.mana == 4 - MANA_DRAW_BASE
    assert state.p1.mana_draw_count_this_turn == 1
    assert state.p1.hand == [drawn]
    assert state.p1.deck == []
    assert state.history[-1] == {"type": "mana_draw"}
    event = state.v5_history_events[-1]
    assert event == {
        "actor_id": 1,
        "action_id": 0,
        "action_type": "mana_draw",
        "enemy_hero_hp_delta": 0,
        "own_hero_hp_delta": 0,
        "my_board_count_delta": 0,
        "enemy_board_count_delta": 0,
        "board_power_delta": 0.0,
        "turn_number": 1,
        "source_card": None,
        "target_card": None,
    }
    # The V5 tape stays trace/JSON-safe and mana_draw gets its dedicated
    # metadata bit despite being a parallel (not 601-way) action.
    json.dumps(event)
    encoded = _latest_history_vector(state)
    assert encoded[0] == 1.0
    assert encoded[1] == 1.0
    assert encoded[2] == 0.0
    assert encoded[3:6].tolist() == [0.0, 0.0, 0.0]
    assert encoded[11] == pytest.approx(1.0 / 50.0)
    assert encoded[13] == 1.0


def test_play_card_records_rich_v5_history_and_encoded_outcome():
    played = _card(311, "played-card")
    state = GameState(
        p1=PlayerState(
            user_id=1,
            hero=_hero(1, "p1-hero"),
            mana=5,
            max_mana=5,
            hand=[played],
        ),
        p2=PlayerState(user_id=2, hero=_hero(2, "p2-hero")),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
    )
    env = ArenaEnvironment(state)

    assert env.step(1, PlayCardAction(hand_index=0, position=0)) == (True, "")

    assert state.history[-1] == {
        "type": "play_card",
        "hand_index": 0,
        "target_id": None,
        "position": 0,
    }
    event = state.v5_history_events[-1]
    assert event["action_type"] == "play_card"
    assert event["actor_id"] == 1
    assert event["action_id"] == 1
    assert event["my_board_count_delta"] == 1
    assert event["enemy_board_count_delta"] == 0
    assert event["board_power_delta"] == 6.0
    assert event["source_card"]["card_id"] == 311
    assert event["source_card"]["card_type"] == "warrior"
    assert event["target_card"] is None
    json.dumps(event)

    encoded = _latest_history_vector(state)
    assert encoded[0:6].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert encoded[6] == pytest.approx(1.0 / 600.0)
    assert encoded[9] == pytest.approx(1.0 / 7.0)
    assert encoded[12] == pytest.approx(6.0 / 200.0)
    assert encoded[HISTORY_EVENT_SOURCE_OFFSET + 1] == 1.0


def test_attack_records_rich_v5_history_and_encoded_damage():
    attacker = _card(321, "attacker")
    target_hero = _hero(2, "target-hero")
    target_hero.hp = target_hero.max_hp = 30
    state = GameState(
        p1=PlayerState(
            user_id=1,
            hero=_hero(1, "p1-hero"),
            board=[attacker],
        ),
        p2=PlayerState(user_id=2, hero=target_hero),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
    )
    env = ArenaEnvironment(state)

    action = AttackAction(
        attacker_id=str(attacker.instance_id),
        target_is_hero=True,
    )
    assert env.step(1, action) == (True, "")

    assert state.history[-1] == action.to_dict()
    event = state.v5_history_events[-1]
    assert event["action_type"] == "attack"
    assert event["actor_id"] == 1
    assert event["action_id"] == 552
    assert event["enemy_hero_hp_delta"] == 2
    assert event["own_hero_hp_delta"] == 0
    assert event["source_card"]["card_id"] == 321
    assert event["source_card"]["is_ready"] is True
    assert state.p1.board[0].is_ready is False
    assert event["target_card"]["card_type"] == "hero"
    assert event["target_card"]["hp"] == 30
    assert state.p2.hero.hp == 28
    json.dumps(event)

    encoded = _latest_history_vector(state)
    assert encoded[0:6].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert encoded[6] == pytest.approx(552.0 / 600.0)
    assert encoded[7] == pytest.approx(2.0 / 50.0)
    target_offset = HISTORY_EVENT_SOURCE_OFFSET + 73
    assert encoded[target_offset] == 1.0


def test_end_turn_records_post_transition_turn_and_enemy_actor_view():
    state = GameState(
        p1=PlayerState(user_id=1, hero=_hero(1, "p1-hero")),
        p2=PlayerState(user_id=2, hero=_hero(2, "p2-hero")),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
    )
    env = ArenaEnvironment(state)

    assert env.step(1, EndTurnAction()) == (True, "")

    assert state.history[-1] == {"type": "end_turn"}
    event = state.v5_history_events[-1]
    assert event["action_type"] == "end_turn"
    assert event["actor_id"] == 1
    assert event["action_id"] == 0
    assert event["turn_number"] == 2
    assert event["enemy_hero_hp_delta"] == 0
    assert event["own_hero_hp_delta"] == 0
    json.dumps(event)

    encoded_for_actor = _latest_history_vector(state, player_id=1)
    assert encoded_for_actor[1] == 1.0
    assert encoded_for_actor[2] == 0.0
    assert encoded_for_actor[3] == 1.0
    assert encoded_for_actor[11] == pytest.approx(2.0 / 50.0)

    encoded_for_opponent = _latest_history_vector(state, player_id=2)
    assert encoded_for_opponent[1] == 0.0
    assert encoded_for_opponent[2] == 1.0
    assert encoded_for_opponent[3] == 1.0


def test_v5_history_is_a_bounded_ring_separate_from_native_history():
    state = GameState(
        p1=PlayerState(user_id=1, hero=_hero(1, "p1-hero")),
        p2=PlayerState(user_id=2, hero=_hero(2, "p2-hero")),
        current_turn_owner_id=1,
    )
    state.history.append({"type": "end_turn"})
    for index in range(V5_HISTORY_EVENTS + 5):
        state.v5_history_events.append(
            {
                "actor_id": 1,
                "action_id": index,
                "action_type": "end_turn",
                "enemy_hero_hp_delta": 0,
                "own_hero_hp_delta": 0,
                "my_board_count_delta": 0,
                "enemy_board_count_delta": 0,
                "board_power_delta": 0.0,
                "turn_number": index + 1,
                "source_card": None,
                "target_card": None,
            }
        )

    assert state.v5_history_events.maxlen == V5_HISTORY_EVENTS
    assert len(state.v5_history_events) == V5_HISTORY_EVENTS
    assert state.v5_history_events[0]["action_id"] == 5
    assert state.v5_history_events[-1]["action_id"] == 24
    assert state.history == [{"type": "end_turn"}]


def test_trace_offline_and_prod_train_encoders_share_exact_v5_tape():
    train_python = Path(__file__).resolve().parents[1] / "TrainV3.5" / "python"
    if str(train_python) not in sys.path:
        sys.path.insert(0, str(train_python))

    from ai.train_v2.offline_dataset_loader import reconstruct_gamestate
    from ai.train_v2.obs_v5 import encode_observation_v5 as encode_prod
    from ai.train_v2.v5_contracts import (
        AssistModeV5 as ProdAssistMode,
        InfoModeV5 as ProdInfoMode,
    )
    from rlhf_env.components.v5_trace import V5TraceRecorder
    from train_v3.contracts import (
        AssistModeV5 as TrainAssistMode,
        InfoModeV5 as TrainInfoMode,
    )
    from train_v3.obs_v5 import encode_observation_v5 as encode_train

    attacker = _card(401, "snapshot-attacker")
    state = GameState(
        p1=PlayerState(
            user_id=1,
            hero=_hero(1, "p1-hero"),
            hand=[_card(402, "own-hand")],
            board=[attacker],
            deck=[_card(403, "own-deck")],
        ),
        p2=PlayerState(
            user_id=2,
            hero=_hero(2, "p2-hero"),
            hand=[_card(404, "enemy-hand")],
            deck=[_card(405, "enemy-deck")],
        ),
        current_turn_owner_id=1,
    )
    env = ArenaEnvironment(state)
    assert env.step(
        1,
        AttackAction(attacker_id=str(attacker.instance_id), target_is_hero=True),
    ) == (True, "")

    recorder = V5TraceRecorder.__new__(V5TraceRecorder)
    recorder.engine = SimpleNamespace(
        _arena=SimpleNamespace(state=state),
        _snapshot_card=lambda card: BattleEngine._snapshot_card(card),
    )
    snapshot = recorder._snapshot_state()

    assert snapshot["history"] == state.history
    assert snapshot["v5_history_events"] == list(state.v5_history_events)
    json.dumps(snapshot["v5_history_events"])

    reconstructed = reconstruct_gamestate(snapshot)
    assert reconstructed.history == state.history
    assert list(reconstructed.v5_history_events) == list(state.v5_history_events)
    assert reconstructed.v5_history_events.maxlen == V5_HISTORY_EVENTS

    prod_info = ProdInfoMode(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )
    train_info = TrainInfoMode(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )
    prod_obs = encode_prod(
        state,
        1,
        info_mode=prod_info,
        assist_mode=ProdAssistMode(),
        history_events=list(state.v5_history_events),
    )
    train_obs = encode_train(
        reconstructed,
        1,
        info_mode=train_info,
        assist_mode=TrainAssistMode(),
        history_events=snapshot["v5_history_events"],
    )
    assert np.array_equal(prod_obs, train_obs)
    assert prod_obs.tobytes() == train_obs.tobytes()


def test_train_env_history_cards_are_frozen_before_step(monkeypatch):
    train_python = Path(__file__).resolve().parents[1] / "TrainV3.5" / "python"
    if str(train_python) not in sys.path:
        sys.path.insert(0, str(train_python))

    import train_v3.env_v5 as env_v5

    attacker = _card(411, "train-attacker")
    target = _hero(2, "train-target")
    target.hp = target.max_hp = 30
    state = GameState(
        p1=PlayerState(user_id=1, hero=_hero(1, "p1-hero"), board=[attacker]),
        p2=PlayerState(user_id=2, hero=target),
        current_turn_owner_id=1,
    )
    action = AttackAction(
        attacker_id=str(attacker.instance_id),
        target_is_hero=True,
    )
    monkeypatch.setattr(env_v5, "decode_action", lambda *_args: action)
    wrapper = env_v5.TrainV3ClassicEnv.__new__(env_v5.TrainV3ClassicEnv)
    wrapper.env = SimpleNamespace(_env=SimpleNamespace(state=state))

    event = wrapper._build_event(1, 552)
    attacker.is_ready = False
    target.hp = 28

    assert event["source_card"]["is_ready"] is True
    assert event["target_card"]["hp"] == 30


def test_failed_action_restores_cardoptimum_state_and_rng(monkeypatch):
    """Transactional rollback must keep the match-scoped draw assistant live."""

    state = GameState(
        p1=PlayerState(
            user_id=1,
            is_bot=True,
            hero=_hero(1, "bot-hero"),
            mana=0,
            max_mana=0,
            deck=[_card(301, "candidate")],
        ),
        p2=PlayerState(
            user_id=2,
            hero=_hero(2, "human-hero"),
            mana=0,
            max_mana=0,
        ),
        current_turn_owner_id=1,
        status=GameStatus.ONGOING,
    )
    engine = BattleEngine(
        match_id="v5-cardoptimum-rollback",
        player_ids=[1, 2],
        is_bot_match=True,
    )
    base_rng = random.Random(20260728)
    engine._arena = ArenaEnvironment(state, rng=base_rng)
    engine.current_player_id = 1
    engine.turn = state.turn_number

    runtime = ExtraLRAuxRuntime.from_model_dir()
    try:
        wrapper = runtime.wrap_draw_rng(
            base_rng,
            state=state,
            assisted_player_id=1,
        )
        engine._arena._rng = wrapper
        before_rng = base_rng.getstate()

        def reject_after_consuming_rng(_user_id, _action):
            wrapper.random()
            return False, "injected_failure"

        monkeypatch.setattr(engine._arena, "step", reject_after_consuming_rng)
        result = engine.execute_action(1, EndTurnAction())

        assert result == {"success": False, "error": "injected_failure"}
        assert base_rng.getstate() == before_rng
        assert wrapper._state is engine._arena.state
        assert engine._arena.state.arena_engine is engine._arena
        assert engine._arena.state.classic_params is engine._arena.classic_params
        assert wrapper.last_decision is None
    finally:
        runtime.close()
