"""
Tests for ClassicRLEnv, policies, and evaluate harness.
"""
import random as rand_mod

import numpy as np
import pytest

from core.state import CardType
from core.actions import EndTurnAction

from ai.train_v2.classic_rl_env import ClassicRLEnv, _normalize_card_catalog
from ai.train_v2.policies import (
    RandomLegalPolicy,
    EndTurnPolicy,
    GreedyFacePolicy,
)
from ai.train_v2.evaluate import play_episode, evaluate_matchup
from ai.train_v2.classic_actions_v1 import MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM
from ai.train_v2.classic_obs_v1 import OBS_DIM


# ============================================================================
# MECHANICS STRING NORMALIZATION
# ============================================================================

class TestMechanicsNormalization:
    def test_string_mechanics_converted_to_list(self):
        raw = [{"id": 1, "card_type": "warrior", "mechanics": '["taunt", "shield"]', "name": "Test"}]
        catalog = _normalize_card_catalog(raw)
        assert isinstance(catalog[1]["mechanics"], list)
        assert catalog[1]["mechanics"] == ["taunt", "shield"]

    def test_already_list_mechanics_unchanged(self):
        raw = [{"id": 1, "card_type": "warrior", "mechanics": ["taunt"], "name": "Test"}]
        catalog = _normalize_card_catalog(raw)
        assert catalog[1]["mechanics"] == ["taunt"]

    def test_invalid_json_falls_back_to_empty(self):
        raw = [{"id": 1, "card_type": "warrior", "mechanics": "not-json", "name": "Test"}]
        catalog = _normalize_card_catalog(raw)
        assert catalog[1]["mechanics"] == []

    def test_mechanics_visible_via_codec_after_normalization(self):
        from ai.train_v2.classic_card_shape_v1 import encode_card_shape
        from core.converter import deck_from_card_ids

        cards_raw = [
            {"id": 100, "card_type": "hero", "mechanics": "[]", "name": "Hero", "mana_cost": 0, "base_attack": 0, "base_hp": 30, "rarity": "common"},
            {"id": 200, "card_type": "warrior", "mechanics": '["taunt", "shield"]', "name": "TauntShield", "mana_cost": 3, "base_attack": 4, "base_hp": 5, "rarity": "common"},
        ]
        catalog = _normalize_card_catalog(cards_raw)
        deck = deck_from_card_ids([100, 200], catalog)
        shape = encode_card_shape(deck[1])
        assert shape[14] == 1.0, "taunt flag should be 1"
        assert shape[15] == 1.0, "shield flag should be 1"


# ============================================================================
# ClassicRLEnv TESTS
# ============================================================================

class TestClassicRLEnv:
    def test_reset_shapes(self):
        env = ClassicRLEnv(seed=42)
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert "current_player_id" in info
        mask = env.action_mask()
        assert mask.shape == (MAX_CANDIDATE_ACTIONS,)
        assert mask.dtype == np.float32
        features = env.action_features()
        assert features.shape == (MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)
        assert features.dtype == np.float32

    def test_step_end_turn(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        mask = env.action_mask()
        assert mask[0] == 1.0, "end turn must be legal"
        prev_player = env.current_player_id()
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["success"]
        assert not terminated
        assert not truncated
        assert env.current_player_id() != prev_player, "player must switch after end turn"
        assert obs.shape == (OBS_DIM,)

    def test_illegal_action_no_state_mutation(self):
        env = ClassicRLEnv(seed=42)
        obs0, info0 = env.reset()
        p1_hp_before = info0["p1_hp"]
        prev_player = env.current_player_id()

        obs, reward, terminated, truncated, info = env.step(999)
        assert reward < 0, f"illegal action should give negative reward, got {reward}"
        assert info["invalid_action"]
        assert not terminated
        assert not truncated
        assert env.current_player_id() == prev_player, "state must not advance"

        info2 = env._make_info(action_id=-1, success=True, error="")
        assert info2["p1_hp"] == p1_hp_before, "hero HP must be unchanged"

    def test_perspective_switch(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        obs_p1 = env.observe(1)
        obs_p2 = env.observe(2)

        assert not np.array_equal(obs_p1, obs_p2), "P1 and P2 observations must differ"

        env.step(0)

        assert env.current_player_id() == 2
        obs_current = env.observe()
        assert np.array_equal(obs_current, env.observe(2)), "default observe must target current player"

    def test_deterministic_reset_same_seed(self):
        env1 = ClassicRLEnv(seed=42)
        obs1, _ = env1.reset(seed=100)

        env2 = ClassicRLEnv(seed=42)
        obs2, _ = env2.reset(seed=100)

        st1 = env1.clone_state()
        st2 = env2.clone_state()

        assert st1.p1.mana == st2.p1.mana
        assert st1.p2.mana == st2.p2.mana
        assert [c.name for c in st1.p1.hand] == [c.name for c in st2.p1.hand]
        assert [c.name for c in st1.p1.deck] == [c.name for c in st2.p1.deck]

    def test_deterministic_reset_different_seed_different_decks(self):
        env1 = ClassicRLEnv(seed=42)
        env1.reset(seed=100)
        st1 = env1.clone_state()

        env2 = ClassicRLEnv(seed=42)
        env2.reset(seed=200)
        st2 = env2.clone_state()

        hands_same = [c.name for c in st1.p1.hand] == [c.name for c in st2.p1.hand]
        decks_same = [c.name for c in st1.p1.deck] == [c.name for c in st2.p1.deck]
        assert not (hands_same and decks_same), "different seeds should likely produce different decks"

    def test_default_deck_has_hero_and_min_3_warriors(self):
        env = ClassicRLEnv(seed=999)
        env.reset()
        st = env.clone_state()

        assert st.p1.hero.card_type == CardType.HERO
        warrior_count = sum(1 for c in st.p1.hand if c.card_type == CardType.WARRIOR) + \
                        sum(1 for c in st.p1.deck if c.card_type == CardType.WARRIOR)
        assert warrior_count >= 3, f"default deck must have >= 3 warriors, got {warrior_count}"
        assert len(st.p1.hand) >= 1

    def test_legal_action_ids(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        legal = env.legal_action_ids()
        assert 0 in legal
        assert len(legal) >= 1
        for aid in legal:
            assert 0 <= aid < MAX_CANDIDATE_ACTIONS

    def test_winner_id(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        assert env.winner_id() is None

    def test_clone_state(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        st = env.clone_state()
        assert st.p1.user_id == 1
        assert st.p2.user_id == 2
        assert st.current_turn_owner_id == 1

    def test_reset_can_start_p2(self):
        env = ClassicRLEnv(seed=42)
        obs, info = env.reset(seed=123, starting_player_id=2)
        st = env.clone_state()

        assert obs.shape == (OBS_DIM,)
        assert info["current_player_id"] == 2
        assert env.current_player_id() == 2
        assert st.current_turn_owner_id == 2
        assert st.p1.mana == 0
        assert st.p1.max_mana == 0
        assert st.p2.mana == 1
        assert st.p2.max_mana == 1

    def test_reset_default_still_starts_p1(self):
        env = ClassicRLEnv(seed=42)
        _, info = env.reset(seed=123)
        st = env.clone_state()

        assert info["current_player_id"] == 1
        assert st.current_turn_owner_id == 1
        assert st.p1.mana == 1
        assert st.p2.mana == 0


# ============================================================================
# POLICIES
# ============================================================================

class TestPolicies:
    def test_random_legal_returns_valid_id(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        p = RandomLegalPolicy(seed=1)
        aid = p.select_action(env, 1)
        assert aid in env.legal_action_ids(1)

    def test_end_turn_policy_returns_zero(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        p = EndTurnPolicy()
        aid = p.select_action(env, 1)
        assert aid == 0

    def test_greedy_face_no_invalid(self):
        env = ClassicRLEnv(seed=42)
        p = GreedyFacePolicy()

        for seed in range(10):
            env.reset(seed=seed + 1000)
            for _ in range(50):
                cp = env.current_player_id()
                aid = p.select_action(env, cp)
                obs, reward, terminated, truncated, info = env.step(aid)
                assert not info.get("invalid_action"), f"GreedyFace produced invalid action at seed={seed}"
                if terminated or truncated:
                    break


# ============================================================================
# FULL EPISODE / EVALUATION
# ============================================================================

class TestFullEpisode:
    def test_random_vs_random_completes(self):
        env = ClassicRLEnv(seed=42)
        p = RandomLegalPolicy(seed=1)
        result = play_episode(p, p, seed=123, env=env, max_steps=500)
        assert result["winner_id"] is not None or result["truncated"]
        assert result["steps"] > 0
        assert result["turns"] > 0
        assert result["p1_policy"] == "random"

    def test_greedy_vs_end_turn_completes(self):
        env = ClassicRLEnv(seed=42)
        p1 = GreedyFacePolicy()
        p2 = EndTurnPolicy()
        result = play_episode(p1, p2, seed=456, env=env)
        assert result["invalid_actions"] == 0
        assert result["winner_id"] is not None
        assert result["winner_id"] == 1

    def test_evaluate_matchup(self):
        env = ClassicRLEnv(seed=42)
        p = RandomLegalPolicy(seed=5)
        result = evaluate_matchup(p, p, seeds=[1, 2, 3])
        assert result["games"] == 3
        assert result["p1_wins"] + result["p2_wins"] + result["draws"] == 3
        assert 0.0 <= result["p1_winrate"] <= 1.0
        assert result["avg_turns"] > 0
        assert result["avg_steps"] > 0


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

class TestLegacyCompat:
    def test_all_imports_succeed(self):
        from ai.arena_env import ArenaEnv
        from ai.train_v2.classic_rl_env import ClassicRLEnv as CREnv
        assert True


# ============================================================================
# ISSUE #1: MASK REJECTION BEFORE MUTATION
# ============================================================================

class TestMaskRejectionBeforeMutation:
    def test_masked_out_action_rejected_no_state_mutation(self):
        from core.state import CardInstance, CardType, PlayerState, GameState, GameStatus

        cards_manual = {
            1: {"id": 1, "card_type": "hero", "mechanics": [], "name": "Hero", "base_attack": 0, "base_hp": 30, "mana_cost": 0, "rarity": "common"},
            2: {"id": 2, "card_type": "warrior", "mechanics": ["battlecry_buff_2_3"], "name": "Buffer", "base_attack": 3, "base_hp": 4, "mana_cost": 2, "rarity": "common"},
        }
        env = ClassicRLEnv(cards_data=cards_manual, seed=1, mana_per_turn=1)
        env.reset(p1_deck_ids=[1, 2], p2_deck_ids=[1, 2], seed=100)

        masked_aid = None
        mask = env.action_mask(1)
        for aid in range(1, 545):
            if mask[aid] != 1.0:
                _pos_idx, tcode = divmod(aid - 1, 17)
                hand_idx, _pos_idx = divmod(_pos_idx, 8)
                if hand_idx == 0 and tcode == 16:
                    masked_aid = aid
                    break

        assert masked_aid is not None, "should have a masked-out buff-own-hero action candidate"
        assert mask[masked_aid] != 1.0, "this action should be masked out"

        state_before = env.clone_state()
        obs, reward, terminated, truncated, info = env.step(masked_aid)

        assert reward < 0, f"masked action should give negative reward, got {reward}"
        assert info["invalid_action"] is True
        assert info["error"] == "illegal_action"
        assert terminated is False
        assert truncated is False

        state_after = env.clone_state()
        assert state_after.p1.hero.hp == state_before.p1.hero.hp
        assert state_after.p2.hero.hp == state_before.p2.hero.hp
        assert len(state_after.p1.board) == len(state_before.p1.board)
        assert len(state_after.p2.board) == len(state_before.p2.board)
        assert len(state_after.p1.hand) == len(state_before.p1.hand)
        assert state_after.current_turn_owner_id == state_before.current_turn_owner_id


# ============================================================================
# ISSUE #2: ACTING / CURRENT PLAYER IN INFO
# ============================================================================

class TestActingCurrentPlayerInfo:
    def test_end_turn_acting_vs_current(self):
        env = ClassicRLEnv(seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["acting_player_id"] == 1, f"acting should be P1, got {info['acting_player_id']}"
        assert info["current_player_id"] == 2, f"current should be P2, got {info['current_player_id']}"
        assert info["action"]["type"] == "end_turn"


# ============================================================================
# ISSUE #3: DEFAULT DECK SYMMETRY
# ============================================================================

class TestDefaultDeckSymmetry:
    def test_both_none_produces_same_decks(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=999)
        st = env.clone_state()

        p1_ids = sorted(c.card_id for c in [st.p1.hero] + st.p1.hand + st.p1.deck)
        p2_ids = sorted(c.card_id for c in [st.p2.hero] + st.p2.hand + st.p2.deck)
        assert p1_ids == p2_ids, f"default decks should be symmetric: P1={p1_ids} vs P2={p2_ids}"

    def test_one_side_provided_other_generated(self):
        env0 = ClassicRLEnv(seed=42)
        env0.reset(seed=1)
        st0 = env0.clone_state()

        p2_ids = [c.card_id for c in [st0.p2.hero] + st0.p2.hand + st0.p2.deck]

        env = ClassicRLEnv(seed=42)
        env.reset(p1_deck_ids=p2_ids, seed=1)
        st = env.clone_state()
        assert st.p1.hero is not None
        assert len(st.p1.hand) + len(st.p1.deck) >= 1
        assert st.p2.hero is not None
        assert len(st.p2.hand) + len(st.p2.deck) >= 1


# ============================================================================
# ISSUE #4: NORMALIZE PROVIDED CARDS_DATA
# ============================================================================

class TestNormalizeProvidedCardsData:
    def test_provided_dict_with_string_mechanics_normalized(self):
        from core.converter import deck_from_card_ids
        from ai.train_v2.classic_card_shape_v1 import encode_card_shape

        cards_dict = {
            1: {"id": 1, "card_type": "hero", "mechanics": "[]", "name": "Hero", "base_attack": 0, "base_hp": 30, "mana_cost": 0, "rarity": "common"},
            2: {"id": 2, "card_type": "warrior", "mechanics": '["taunt", "shield"]', "name": "TauntShield", "base_attack": 4, "base_hp": 5, "mana_cost": 3, "rarity": "common"},
        }

        env = ClassicRLEnv(cards_data=cards_dict, seed=1)
        env.reset(p1_deck_ids=[1, 2], p2_deck_ids=[1, 2], seed=200)
        st = env.clone_state()

        card = st.p1.hand[0] if st.p1.hand else st.p1.deck[0]
        enc = encode_card_shape(card)
        if "taunt" in card.mechanics or "shield" in card.mechanics:
            assert enc[14] == 1.0, "taunt mech flag must be 1 after normalization"
            assert enc[15] == 1.0, "shield mech flag must be 1 after normalization"


# ============================================================================
# ISSUE #5: DETERMINISTIC EVALUATION
# ============================================================================

class TestDeterministicEvaluation:
    def test_play_episode_random_vs_random_deterministic(self):
        result1 = play_episode(RandomLegalPolicy(seed=1), RandomLegalPolicy(seed=2), seed=123)
        result2 = play_episode(RandomLegalPolicy(seed=1), RandomLegalPolicy(seed=2), seed=123)

        assert result1["winner_id"] == result2["winner_id"]
        assert result1["steps"] == result2["steps"]
        assert result1["turns"] == result2["turns"]
        assert result1["p1_reward"] == result2["p1_reward"]
        assert result1["p2_reward"] == result2["p2_reward"]

    def test_play_episode_different_seeds_diverge(self):
        result1 = play_episode(RandomLegalPolicy(seed=1), RandomLegalPolicy(seed=2), seed=100)
        result2 = play_episode(RandomLegalPolicy(seed=1), RandomLegalPolicy(seed=2), seed=200)

        fields_same = 0
        for k in ["winner_id", "steps", "turns", "p1_reward", "p2_reward"]:
            if result1[k] == result2[k]:
                fields_same += 1
        assert fields_same < 5, "different seeds should produce at least some different results"

    def test_evaluate_matchup_deterministic(self):
        seeds = [42, 43, 44]
        p1a = RandomLegalPolicy(seed=10)
        p2a = RandomLegalPolicy(seed=20)

        r1 = evaluate_matchup(p1a, p2a, seeds=seeds)

        p1b = RandomLegalPolicy(seed=10)
        p2b = RandomLegalPolicy(seed=20)

        r2 = evaluate_matchup(p1b, p2b, seeds=seeds)

        for key in ["games", "p1_wins", "p2_wins", "draws", "p1_winrate", "avg_turns", "avg_steps", "invalid_actions"]:
            assert r1[key] == r2[key], f"field {key} differs: {r1[key]} vs {r2[key]}"
