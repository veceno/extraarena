from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
for path in (ROOT, TRAINV3_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState  # noqa: E402
from core.actions import ManaDrawAction  # noqa: E402
from core.engine import ArenaEnvironment  # noqa: E402
from train_v3.contracts import (  # noqa: E402
    HISTORY_EVENT_DIM,
    HISTORY_EVENTS,
    OBS_V1_DIM,
    PRIVATE_INFO_DIM,
    V5_GLOBAL_DIM,
    InfoModeV5,
)
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.obs_v5 import encode_observation_v5  # noqa: E402


def _hero(owner: int) -> CardInstance:
    return CardInstance(
        instance_id=uuid4(),
        card_id=owner,
        name=f"hero-{owner}",
        card_type=CardType.HERO,
        rarity="common",
        mana_cost=0,
        attack=0,
        hp=30,
        max_hp=30,
    )


def _player(user_id: int) -> PlayerState:
    return PlayerState(user_id=user_id, hero=_hero(user_id), hand=[], deck=[], board=[])


def test_global_16_marks_first_starter_not_current_turn_owner() -> None:
    state = GameState(
        p1=_player(1),
        p2=_player(2),
        current_turn_owner_id=2,
        turn_number=2,
        status=GameStatus.ONGOING,
    )

    p1_obs = encode_observation_v5(state, 1)
    p2_obs = encode_observation_v5(state, 2)

    assert p1_obs[OBS_V1_DIM + 16] == pytest.approx(1.0)
    assert p2_obs[OBS_V1_DIM + 16] == pytest.approx(0.0)


def test_default_v5_observation_contains_enemy_hand_and_deck() -> None:
    """Enemy hand/deck are base V5 input, independent of AssistModeV5."""
    p1 = _player(1)
    p2 = _player(2)
    p2.hand.append(CardInstance(card_id=37, name="enemy-hand", card_type=CardType.WARRIOR, mana_cost=2, attack=3, hp=4, max_hp=4))
    p2.deck.append(CardInstance(card_id=38, name="enemy-deck", card_type=CardType.POTION, mana_cost=3))
    state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, status=GameStatus.ONGOING)

    default_obs = encode_observation_v5(state, 1)
    hidden_obs = encode_observation_v5(
        state,
        1,
        info_mode=InfoModeV5(enemy_hand_known=False, enemy_deck_known=False),
    )
    assert default_obs[OBS_V1_DIM + 3] == pytest.approx(1.0)
    assert default_obs[OBS_V1_DIM + 4] == pytest.approx(1.0)
    assert default_obs[OBS_V1_DIM + 5] == pytest.approx(1.0)
    assert not np.array_equal(default_obs, hidden_obs)


def test_train_v3_env_threads_120_max_turns_to_python_eval_env() -> None:
    env = TrainV3ClassicEnv(TrainV3EnvConfig(seed=7))

    assert env.env._max_turns == 120


def test_last_twenty_actions_are_capped_and_mana_draw_is_distinguishable() -> None:
    state = GameState(
        p1=_player(1), p2=_player(2), current_turn_owner_id=1, status=GameStatus.ONGOING
    )
    for turn in range(25):
        state.v5_history_events.append(
            {
                "actor_id": 1,
                "action_type": "mana_draw" if turn == 24 else "end_turn",
                "action_id": 601 if turn == 24 else 0,
                "turn_number": turn,
            }
        )
    assert len(state.v5_history_events) == HISTORY_EVENTS
    obs = encode_observation_v5(state, 1, history_events=list(state.v5_history_events))
    history_base = OBS_V1_DIM + V5_GLOBAL_DIM + PRIVATE_INFO_DIM
    # The oldest retained event is turn 5, proving the ring buffer supplies a
    # true last-20 window rather than unbounded/UI-only history.
    first = history_base
    assert obs[first] == pytest.approx(1.0)
    assert obs[first + 11] == pytest.approx(5.0 / 50.0)
    last = history_base + (HISTORY_EVENTS - 1) * HISTORY_EVENT_DIM
    assert obs[last + 13] == pytest.approx(1.0)


def test_production_engine_populates_structured_v5_history_for_mana_draw() -> None:
    p1 = _player(1)
    p1.mana = 2
    p1.max_mana = 2
    p1.deck.append(
        CardInstance(card_id=37, name="draw-target", card_type=CardType.WARRIOR, mana_cost=1)
    )
    state = GameState(p1=p1, p2=_player(2), current_turn_owner_id=1, status=GameStatus.ONGOING)
    env = ArenaEnvironment(state, apply_start_effects=False)
    ok, error = env.step(1, ManaDrawAction())
    assert ok, error
    assert len(state.v5_history_events) == 1
    event = state.v5_history_events[-1]
    assert event["action_type"] == "mana_draw"
    obs = encode_observation_v5(state, 1, history_events=list(state.v5_history_events))
    history_base = OBS_V1_DIM + V5_GLOBAL_DIM + PRIVATE_INFO_DIM
    last = history_base + (HISTORY_EVENTS - 1) * HISTORY_EVENT_DIM
    assert obs[last + 13] == pytest.approx(1.0)
