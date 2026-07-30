"""Regression coverage for browser play-card target canonicalization."""
from __future__ import annotations

from types import SimpleNamespace

from core.actions import EndTurnAction, PlayCardAction
from rlhf_env.components.v5_trace import V5TraceRecorder


def _card(instance_id: str) -> SimpleNamespace:
    return SimpleNamespace(instance_id=instance_id)


def _recorder() -> tuple[V5TraceRecorder, SimpleNamespace]:
    own_hero = _card("own-hero")
    enemy_hero = _card("enemy-hero")
    explicit_target = _card("explicit-target")
    source = _card("source-card")
    p1 = SimpleNamespace(
        user_id=1000,
        hero=own_hero,
        hand=[source],
        board=[],
    )
    p2 = SimpleNamespace(
        user_id=2000,
        hero=enemy_hero,
        hand=[],
        board=[explicit_target],
    )
    state = SimpleNamespace(p1=p1, p2=p2)

    def resolve_hand_index(hand, card_ref):
        return int(card_ref) if 0 <= int(card_ref) < len(hand) else -1

    engine = SimpleNamespace(
        _arena=SimpleNamespace(state=state),
        _resolve_hand_index=resolve_hand_index,
        _snapshot_card=lambda card: {"instance_id": str(card.instance_id)},
    )
    recorder = object.__new__(V5TraceRecorder)
    recorder.engine = engine
    return recorder, explicit_target


def test_explicit_target_id_wins_over_target_is_hero_hint():
    """Match the action the engine actually executes when the UI sends both."""
    recorder, explicit_target = _recorder()
    action_json = {
        "type": "play_card",
        "card_ref": 0,
        "board_position": 0,
        "target_id": explicit_target.instance_id,
        "target_is_hero": True,
    }
    legal = [
        PlayCardAction(
            hand_index=0,
            target_id=explicit_target.instance_id,
            position=3,
        ),
        EndTurnAction(),
    ]

    legal_index = recorder._resolve_action_id(1000, action_json, None, legal)
    _, target_card = recorder._resolve_source_target(1000, action_json)

    assert legal_index == 0
    assert target_card == {"instance_id": explicit_target.instance_id}


def test_target_is_hero_is_fallback_when_target_id_is_absent():
    """Keep compatibility with clients that send only the hero-target flag."""
    recorder, _ = _recorder()
    action_json = {
        "type": "play_card",
        "card_ref": 0,
        "board_position": 0,
        "target_id": None,
        "target_is_hero": True,
    }
    legal = [
        PlayCardAction(hand_index=0, target_id="enemy-hero", position=1),
        EndTurnAction(),
    ]

    legal_index = recorder._resolve_action_id(1000, action_json, None, legal)
    _, target_card = recorder._resolve_source_target(1000, action_json)

    assert legal_index == 0
    assert target_card == {"instance_id": "enemy-hero"}


def test_spurious_target_falls_back_only_for_unique_same_hand_action():
    """Preserve the label when an untargeted card carries a stale target id."""
    recorder, _ = _recorder()
    action_json = {
        "type": "play_card",
        "card_ref": 0,
        "target_id": "stale-board-instance",
        "target_is_hero": False,
    }
    legal = [
        PlayCardAction(hand_index=0, target_id=None, position=None),
        EndTurnAction(),
    ]

    assert recorder._resolve_action_id(1000, action_json, None, legal) == 0


def test_spurious_target_stays_unresolved_when_same_hand_has_multiple_targets():
    """Never guess between genuinely distinct targeted-card actions."""
    recorder, _ = _recorder()
    action_json = {
        "type": "play_card",
        "card_ref": 0,
        "target_id": "unknown-target",
        "target_is_hero": False,
    }
    legal = [
        PlayCardAction(hand_index=0, target_id="enemy-hero", position=None),
        PlayCardAction(hand_index=0, target_id="explicit-target", position=None),
        EndTurnAction(),
    ]

    assert recorder._resolve_action_id(1000, action_json, None, legal) is None
