"""
Tests for fast action cache, placement mode, and verify_mask integration.
"""
import numpy as np
import pytest

from core.state import CardInstance, CardType, PlayerState, GameState, GameStatus
from uuid import uuid4

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.fast_action_cache import ActionCache
from ai.train_v2.classic_actions_v1 import (
    build_action_mask,
    encode_action_features,
    MAX_CANDIDATE_ACTIONS,
    ACTION_FEATURE_DIM,
)


# ============================================================================
# ActionCache tests
# ============================================================================

class TestActionCache:
    def test_cache_shape_and_dtype(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        m = cache.mask()
        assert m.shape == (MAX_CANDIDATE_ACTIONS,)
        assert m.dtype == np.float32
        f = cache.features(include_preview=False)
        assert f.shape == (MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)
        assert f.dtype == np.float32

    def test_cache_invalidation(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        m1 = cache.mask()
        cache.invalidate()
        m2 = cache.mask()
        assert np.array_equal(m1, m2)
        assert cache._legal_ids is None
        assert cache._features_valid is False

    def test_cache_set_state(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        m1 = cache.mask()
        p1 = cache._player_id
        env.step(0)  # end turn
        new_st = env.clone_state()
        new_cp = env.current_player_id()
        cache.set_state(new_st, new_cp)
        m2 = cache.mask()
        assert cache._player_id != p1, "player_id should change after end turn"
        assert m2 is not m1, "mask should be a new object after set_state"
        assert cache._mask_valid is True
        assert cache._features_valid is False

    def test_cache_legal_ids(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        legal = cache.legal_ids()
        mask = cache.mask()
        expected = [i for i, v in enumerate(mask) if v == 1.0]
        assert legal == expected
        assert 0 in legal

    def test_cache_features_fast_vs_full(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        fast = cache.features(include_preview=False)
        full = cache.features(include_preview=True)
        assert np.allclose(fast[:, :142], full[:, :142], atol=1e-7)
        assert np.all(fast[:, 142:] == 0.0)

    def test_cache_reuses_features_when_preview_flag_unchanged(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()
        cache = ActionCache(st, cp, verify_mask=False, placement_mode="full")
        f1 = cache.features(include_preview=False)
        f2 = cache.features(include_preview=False)
        assert f1 is f2


# ============================================================================
# placement_mode tests
# ============================================================================

class TestPlacementMode:
    def test_append_only_zeroes_warrior_positions(self):
        # state with one warrior in hand
        h1 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        h2 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        w = CardInstance(instance_id=uuid4(), card_id=100, card_type=CardType.WARRIOR,
                         mana_cost=3, attack=4, hp=5, max_hp=5, mechanics=[], is_ready=False)
        p1 = PlayerState(user_id=1, hero=h1, mana=10, max_mana=10, hand=[w], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        full_mask = build_action_mask(gs, 1, verify_mask=False, placement_mode="full")
        append_mask = build_action_mask(gs, 1, verify_mask=False, placement_mode="append_only")

        # end turn must be legal in both
        assert full_mask[0] == 1.0
        assert append_mask[0] == 1.0

        # For warrior in hand (hand_idx=0), board is empty so only position 0 is valid in append_only
        # full allows positions 0..7 (up to NUM_PLAY_POS), append_only only position 0
        play_base = 1 + 0 * (8 * 17)
        full_positions = []
        append_positions = []
        for pos in range(8):
            base = play_base + pos * 17
            if full_mask[base] == 1.0:
                full_positions.append(pos)
            if append_mask[base] == 1.0:
                append_positions.append(pos)

        assert len(full_positions) > 0
        assert append_positions == [0]
        # append_only mask should be subset of full
        assert np.all((append_mask <= full_mask) | (full_mask == 0))

    def test_append_only_with_existing_board(self):
        h1 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        h2 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        w1 = CardInstance(instance_id=uuid4(), card_id=100, card_type=CardType.WARRIOR,
                          mana_cost=3, attack=4, hp=5, max_hp=5, mechanics=[], is_ready=False)
        w2 = CardInstance(instance_id=uuid4(), card_id=101, card_type=CardType.WARRIOR,
                          mana_cost=2, attack=3, hp=4, max_hp=4, mechanics=[], is_ready=False)
        p1 = PlayerState(user_id=1, hero=h1, mana=10, max_mana=10, hand=[w2], board=[w1], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        append_mask = build_action_mask(gs, 1, verify_mask=False, placement_mode="append_only")
        play_base = 1 + 0 * (8 * 17)
        # board has 1 unit, so valid position is 1 (append at end)
        valid_pos = [pos for pos in range(8) if append_mask[play_base + pos * 17] == 1.0]
        assert valid_pos == [1]

    def test_invalid_placement_mode_raises(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        st = env.clone_state()
        with pytest.raises(ValueError):
            build_action_mask(st, 1, verify_mask=False, placement_mode="invalid")


# ============================================================================
# verify_mask tests
# ============================================================================

class TestVerifyMask:
    def test_verify_mask_true_runs(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        st = env.clone_state()
        m = build_action_mask(st, 1, verify_mask=True, placement_mode="append_only")
        assert m[0] == 1.0

    def test_verify_mask_false_faster(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        st = env.clone_state()
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            build_action_mask(st, 1, verify_mask=True, placement_mode="append_only")
        t_true = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(10):
            build_action_mask(st, 1, verify_mask=False, placement_mode="append_only")
        t_false = time.perf_counter() - t0

        assert t_false < t_true * 1.5, f"verify_mask=False should not be slower: {t_false:.4f} vs {t_true:.4f}"


# ============================================================================
# ClassicRLEnv with new args
# ============================================================================

class TestClassicRLEnvNewArgs:
    def test_env_with_verify_mask_false(self):
        env = ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only")
        env.reset(seed=100)
        mask = env.action_mask()
        assert mask[0] == 1.0

    def test_env_with_placement_mode_append_only(self):
        env = ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only")
        env.reset(seed=100)
        mask = env.action_mask()
        legal = env.legal_action_ids()
        assert 0 in legal
        # No play-card positions other than append should be legal for warriors
        for aid in legal:
            if 1 <= aid <= 544:
                flat = aid - 1
                pos_idx, _ = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx < len(env._env.state.p1.hand):
                    card = env._env.state.p1.hand[hand_idx]
                    if card.card_type == CardType.WARRIOR:
                        assert pos_idx == len(env._env.state.p1.board), (
                            f"append_only mode allowed position {pos_idx} but board size is {len(env._env.state.p1.board)}"
                        )

    def test_env_cache_used(self):
        env = ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only")
        env.reset(seed=100)
        # first call populates cache
        m1 = env.action_mask()
        f1 = env.action_features(include_preview=False)
        # second call from cache
        m2 = env.action_mask()
        f2 = env.action_features(include_preview=False)
        assert np.array_equal(m1, m2)
        assert np.array_equal(f1, f2)
        # after step cache is refreshed
        env.step(0)
        m3 = env.action_mask()
        f3 = env.action_features(include_preview=False)
        # masks may differ because turn changed, but cache should be valid
        assert env._cache is not None
        assert env._cache._mask_valid is True
        assert env._cache._features_valid is True

    def test_env_training_info_can_skip_legal_action_recount(self, monkeypatch):
        env = ClassicRLEnv(
            seed=42,
            verify_mask=True,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
        )
        env.reset(seed=100)
        aid = env.legal_action_ids()[0]

        def fail_legal_action_ids(*args, **kwargs):
            raise AssertionError("legal_action_ids should not be recomputed for training-fast info")

        monkeypatch.setattr(env, "legal_action_ids", fail_legal_action_ids)
        _, _, _, _, info = env.step(aid)

        assert info["legal_actions"] is None
        assert info["success"] is True

    def test_legal_action_ids_uses_action_cache(self, monkeypatch):
        env = ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only")
        env.reset(seed=100)
        expected = env._cache.legal_ids()

        def fail_action_mask(*args, **kwargs):
            raise AssertionError("legal_action_ids should use ActionCache.legal_ids when available")

        monkeypatch.setattr(env, "action_mask", fail_action_mask)
        assert env.legal_action_ids() == expected

    def test_env_step_rejects_illegal_in_append_only(self):
        env = ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only")
        env.reset(seed=100)
        # Try to find a warrior play action at position != len(board)
        mask = env.action_mask()
        for aid in range(1, 545):
            if mask[aid] != 1.0:
                flat = aid - 1
                pos_idx, _ = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx < len(env._env.state.p1.hand):
                    card = env._env.state.p1.hand[hand_idx]
                    if card.card_type == CardType.WARRIOR and pos_idx != len(env._env.state.p1.board):
                        # step should reject
                        _, reward, _, _, info = env.step(aid)
                        assert info["invalid_action"], f"expected illegal for aid={aid} pos={pos_idx}"
                        assert reward < 0
                        return
        pytest.skip("no masked-out warrior position found in this seed")


# ============================================================================
# float16 smoke
# ============================================================================

class TestFloat16:
    @pytest.mark.skipif(not __import__("importlib").util.find_spec("mlx"), reason="mlx not installed")
    def test_train_smoke_float16(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        import shutil, tempfile

        tmp = tempfile.mkdtemp()
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=5,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=f"{tmp}/ckpts",
                action_features_dtype="float16",
                verify_mask=False,
                placement_mode="append_only",
            )
            result = train(config)
            assert result["updates"] == 1
            assert result["episodes"] >= 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
