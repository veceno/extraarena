from core.actions import AttackAction, EndTurnAction, PlayCardAction
from ai.bot_ai import BotAI


def test_decide_action_accepts_action_objects_and_keeps_object_identity(monkeypatch):
    play_card = PlayCardAction(hand_index=0)
    board_attack = AttackAction(attacker_id="unit", target_id="enemy-unit", target_is_hero=False)
    hero_attack = AttackAction(attacker_id="unit", target_id=None, target_is_hero=True)

    monkeypatch.setattr("ai.bot_ai.random.choice", lambda actions: actions[0])

    chosen = BotAI.decide_action([play_card, board_attack, hero_attack, EndTurnAction()])

    assert chosen is hero_attack


def test_decide_turn_plans_single_action_without_mutating_engine():
    class PlanningEngine:
        is_ended = False

        def __init__(self):
            self.execute_calls = []

        def get_legal_actions(self, bot_id):
            assert bot_id == 1
            return [{"type": "play_card", "hand_index": 0}, {"type": "end_turn"}]

        def execute_bot_action(self, action):
            self.execute_calls.append(action)
            raise AssertionError("decide_turn must not mutate the live engine")

    engine = PlanningEngine()

    assert BotAI.decide_turn(engine, 1) == [{"type": "play_card", "hand_index": 0}]
    assert engine.execute_calls == []
