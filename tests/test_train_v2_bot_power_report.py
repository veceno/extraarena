from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.bot_power_report import BotSpec, BrainCorePolicy


def test_brain_core_policy_uses_live_env_legal_actions(monkeypatch):
    class FakeBrain:
        seen_legal_len = None

        def __init__(self, profiles):
            pass

        def get_action(self, state, player_id, legal_actions, difficulty):
            self.seen_legal_len = len(legal_actions)
            return len(legal_actions) - 1

    monkeypatch.setattr("ai.train_v2.bot_power_report.BerserkInference", FakeBrain)

    env = ClassicRLEnv(seed=5, verify_mask=False, placement_mode="append_only")
    env.reset(seed=5)
    cp = env.current_player_id()
    live_legal = env._env.get_legal_actions(cp)

    spec = BotSpec(
        key="test",
        label="test",
        model_name="fake",
        profile={"model_path": "unused.onnx"},
        difficulty="test",
        trophy_range="test",
        player_max_level=1,
    )
    policy = BrainCorePolicy(spec)
    action = policy.select_core_action(env, cp)

    assert policy.brain.seen_legal_len == len(live_legal)
    assert action.to_dict() == live_legal[-1].to_dict()
