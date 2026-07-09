from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[4]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
for path in (ROOT, TRAINV3_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState  # noqa: E402
from train_v3.contracts import OBS_V1_DIM  # noqa: E402
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


def test_train_v3_env_threads_120_max_turns_to_python_eval_env() -> None:
    env = TrainV3ClassicEnv(TrainV3EnvConfig(seed=7))

    assert env.env._max_turns == 120
