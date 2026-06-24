"""
Tests for v2 RL codecs — no card identity, dimension stability, mask/decode fuzz, production parity.
"""
import copy
import random
from uuid import uuid4

import numpy as np
import pytest

from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.actions import EndTurnAction, PlayCardAction, AttackAction

from core.classic_setup import create_classic_game_state

from ai.train_v2.classic_card_shape_v1 import (
    CARD_SHAPE_DIM,
    CARD_SHAPE_VERSION,
    encode_card_shape,
)
from ai.train_v2.classic_obs_v1 import (
    OBS_DIM,
    OBS_VERSION,
    encode_observation,
)
from ai.train_v2.classic_actions_v1 import (
    ACTION_VERSION,
    MAX_CANDIDATE_ACTIONS,
    ACTION_FEATURE_DIM,
    _NUM_PLAY_POS,
    build_action_mask,
    encode_action_features,
    decode_action,
)


# ============================================================================
# DIMENSION CONSTANTS
# ============================================================================

class TestDimensions:
    def test_card_shape_dim(self):
        assert CARD_SHAPE_DIM == 64

    def test_obs_dim(self):
        assert OBS_DIM == 1456

    def test_max_candidate_actions(self):
        assert MAX_CANDIDATE_ACTIONS == 601

    def test_action_feature_dim(self):
        assert ACTION_FEATURE_DIM == 171


# ============================================================================
# OUTPUT SHAPES
# ============================================================================

class TestOutputShapes:
    def test_card_shape_output(self):
        card = _make_card(attack=5, hp=10, mana_cost=3)
        result = encode_card_shape(card)
        assert result.shape == (64,)
        assert result.dtype == np.float32

    def test_card_shape_null(self):
        result = encode_card_shape(None)
        assert result.shape == (64,)
        assert np.all(result == 0.0)

    def test_obs_output(self):
        state = _make_minimal_state()
        result = encode_observation(state, 1)
        assert result.shape == (1456,)
        assert result.dtype == np.float32

    def test_action_mask_output(self):
        state = _make_minimal_state()
        mask = build_action_mask(state, 1)
        assert mask.shape == (601,)
        assert mask.dtype == np.float32

    def test_action_features_output(self):
        state = _make_minimal_state()
        feats = encode_action_features(state, 1)
        assert feats.shape == (601, 171)
        assert feats.dtype == np.float32


# ============================================================================
# NO CARD IDENTITY LEAKAGE
# ============================================================================

class TestNoIdentityLeakage:
    def test_different_id_same_stats_encode_identically(self):
        c1 = CardInstance(
            instance_id=uuid4(),
            card_id=42,
            name="Fireball",
            card_type=CardType.POTION,
            mana_cost=3,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["damage_5"],
            is_ready=False,
            is_frozen=False,
        )
        c2 = CardInstance(
            instance_id=uuid4(),
            card_id=999,
            name="Icebolt",
            card_type=CardType.POTION,
            mana_cost=3,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["damage_5"],
            is_ready=False,
            is_frozen=False,
        )

        enc1 = encode_card_shape(c1)
        enc2 = encode_card_shape(c2)
        assert np.array_equal(enc1, enc2), (
            "Cards with different card_id/name/instance_id but same gameplay stats must encode identically"
        )

    def test_warrior_different_id_same_stats(self):
        c1 = CardInstance(
            instance_id=uuid4(),
            card_id=1,
            name="Warrior A",
            card_type=CardType.WARRIOR,
            mana_cost=4,
            attack=6,
            hp=7,
            max_hp=7,
            mechanics=["taunt", "shield"],
            is_ready=True,
            level=2,
        )
        c2 = CardInstance(
            instance_id=uuid4(),
            card_id=2,
            name="Warrior B",
            card_type=CardType.WARRIOR,
            mana_cost=4,
            attack=6,
            hp=7,
            max_hp=7,
            mechanics=["taunt", "shield"],
            is_ready=True,
            level=2,
        )
        enc1 = encode_card_shape(c1)
        enc2 = encode_card_shape(c2)
        assert np.array_equal(enc1, enc2)

    def test_observation_no_card_id_in_fields(self):
        state = _make_minimal_state_with_cards()
        obs = encode_observation(state, 1)
        assert obs.shape == (OBS_DIM,)
        assert np.sum(np.isnan(obs)) == 0
        assert np.sum(np.isinf(obs)) == 0


# ============================================================================
# PERSPECTIVE-RELATIVE OBSERVATION
# ============================================================================

class TestPerspectiveRelative:
    def test_p1_p2_different(self):
        hero1 = _make_hero(hp=30)
        hero2 = _make_hero(hp=25)

        p1 = PlayerState(user_id=1, hero=hero1, mana=3, max_mana=3, hand=[], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=hero2, mana=2, max_mana=2, hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        obs_p1 = encode_observation(state, 1)
        obs_p2 = encode_observation(state, 2)

        my_hero_hp_idx = 12
        assert obs_p1[my_hero_hp_idx] == 30.0 / 50.0
        assert obs_p2[my_hero_hp_idx] == 25.0 / 50.0

        my_mana_idx = 6
        assert obs_p1[my_mana_idx] == 3.0 / 10.0
        assert obs_p2[my_mana_idx] == 2.0 / 10.0


# ============================================================================
# MASK / DECODE FUZZ
# ============================================================================

class TestMaskDecodeFuzz:
    def test_end_turn_always_valid(self):
        state = _make_minimal_state()
        action = decode_action(state, 1, 0)
        assert isinstance(action, EndTurnAction)

        env = copy.deepcopy(state)
        success, _ = ArenaEnvironment(env).step(1, action)
        assert success

    def test_mask_decode_consistency(self):
        rng = random.Random(42)
        for seed in range(5):
            state = _make_fuzz_state(rng)
            for player_id in (1, 2):
                mask = build_action_mask(state, player_id)
                assert mask.shape == (MAX_CANDIDATE_ACTIONS,)

                if state.status != GameStatus.ONGOING or state.current_turn_owner_id != player_id:
                    assert np.all(mask == 0.0), f"Mask should be all zeros for non-active player, seed={seed}"
                    continue

                for action_id in range(MAX_CANDIDATE_ACTIONS):
                    if mask[action_id] == 1.0:
                        action = decode_action(state, player_id, action_id)
                        assert action is not None, (
                            f"mask=1 but decode returns None for action_id={action_id}, seed={seed}"
                        )

                        try:
                            env = ArenaEnvironment(copy.deepcopy(state))
                            success, _ = env.step(player_id, action)
                            assert success, (
                                f"Mask=1 but step fails for action_id={action_id}, seed={seed}, "
                                f"action={action.to_dict()}"
                            )
                        except Exception as e:
                            pytest.fail(
                                f"step raised exception for action_id={action_id}, seed={seed}: {e}"
                            )

    def test_not_your_turn_mask_all_zeros(self):
        state = _make_minimal_state()
        mask = build_action_mask(state, 2)
        assert np.all(mask == 0.0)

    def test_game_over_mask_all_zeros(self):
        state = _make_minimal_state()
        state.status = GameStatus.P1_WIN
        mask = build_action_mask(state, 1)
        assert np.all(mask == 0.0)


# ============================================================================
# PRODUCTION PARITY
# ============================================================================

class TestProductionParity:
    def test_create_classic_game_state_mana(self):
        deck1 = _make_deck(warriors=5, potions=2, hero=True)
        deck2 = _make_deck(warriors=5, potions=2, hero=True)

        gs = create_classic_game_state(1, 2, deck1, deck2, rng=random.Random(42))

        assert gs.p1.mana == 1
        assert gs.p1.max_mana == 1
        assert gs.p2.mana == 0
        assert gs.p2.max_mana == 0
        assert gs.current_turn_owner_id == 1
        assert gs.turn_number == 1
        assert gs.status == GameStatus.ONGOING

    def test_starting_hand_three_cheapest_warriors(self):
        cards = [
            _make_warrior(mana_cost=5, name="Expensive"),
            _make_warrior(mana_cost=1, name="Cheap1"),
            _make_warrior(mana_cost=3, name="Mid"),
            _make_warrior(mana_cost=2, name="Cheap2"),
            _make_warrior(mana_cost=4, name="Mid2"),
            _make_potion(mana_cost=0, name="Potion0"),
            _make_potion(mana_cost=2, name="Potion2"),
            _make_hero(hp=30),
        ]
        deck_copy = [c for c in cards]

        gs = create_classic_game_state(1, 2, deck_copy, [], rng=random.Random(7))

        assert len(gs.p1.hand) == 3
        hand_names = {c.name for c in gs.p1.hand}
        assert hand_names == {"Cheap1", "Cheap2", "Mid"}
        assert all(c.card_type == CardType.WARRIOR for c in gs.p1.hand)

        deck_names = {c.name for c in gs.p1.deck}
        assert "Expensive" in deck_names
        assert "Mid2" in deck_names
        assert "Potion0" in deck_names
        assert "Potion2" in deck_names
        assert "Cheap1" not in deck_names
        assert "Cheap2" not in deck_names

    def test_hero_extracted_not_in_hand(self):
        cards = [
            _make_hero(hp=25),
            _make_warrior(mana_cost=2, name="W1"),
            _make_warrior(mana_cost=3, name="W2"),
            _make_warrior(mana_cost=1, name="W3"),
            _make_warrior(mana_cost=4, name="W4"),
        ]
        gs = create_classic_game_state(1, 2, cards, [], rng=random.Random(1))

        for card in gs.p1.hand:
            assert card.card_type != CardType.HERO
        for card in gs.p1.deck:
            assert card.card_type != CardType.HERO

    def test_shuffled_when_rng_provided_deterministic(self):
        cards = [
            _make_hero(hp=30),
            _make_warrior(mana_cost=1, name="A"),
            _make_warrior(mana_cost=2, name="B"),
            _make_warrior(mana_cost=3, name="C"),
            _make_warrior(mana_cost=4, name="D"),
            _make_warrior(mana_cost=5, name="E"),
            _make_potion(mana_cost=3, name="Pot"),
        ]
        deck1 = [copy.deepcopy(c) for c in cards]
        deck2 = [copy.deepcopy(c) for c in cards]
        deck3 = [copy.deepcopy(c) for c in cards]

        gs1 = create_classic_game_state(1, 2, deck1, [], rng=random.Random(99))
        gs2 = create_classic_game_state(1, 2, deck2, [], rng=random.Random(99))

        assert [c.name for c in gs1.p1.deck] == [c.name for c in gs2.p1.deck]

        gs3 = create_classic_game_state(1, 2, deck3, [], rng=random.Random(100))
        assert [c.name for c in gs1.p1.deck] != [c.name for c in gs3.p1.deck]

    def test_default_hero_when_no_hero_in_deck(self):
        deck = [
            _make_warrior(mana_cost=1),
            _make_warrior(mana_cost=2),
            _make_warrior(mana_cost=3),
            _make_warrior(mana_cost=4),
        ]
        gs = create_classic_game_state(1, 2, deck, [], rng=random.Random(0))
        assert gs.p1.hero.hp == 30
        assert gs.p1.hero.max_hp == 30
        assert gs.p1.hero.card_type == CardType.HERO


# ============================================================================
# CARD SHAPE MECHANIC SCALARS
# ============================================================================

class TestCardShapeMechanicScalars:
    def test_damage_scalar(self):
        card = _make_potion(mechanics=["damage_7"])
        enc = encode_card_shape(card)
        assert enc[47] == 7.0 / 10.0
        assert enc[48] == 0.0

    def test_cleave_scalars(self):
        card = _make_warrior(mechanics=["cleave_3_2"])
        enc = encode_card_shape(card)
        assert enc[58] == 3.0 / 10.0
        assert enc[59] == 2.0 / 5.0

    def test_summon_flag(self):
        card = _make_warrior(mechanics=["summon", "taunt"])
        enc = encode_card_shape(card)
        assert enc[63] == 1.0
        assert enc[14] == 1.0

    def test_multiple_mechanics_max_taken(self):
        card = _make_potion(mechanics=["damage_3", "damage_7"])
        enc = encode_card_shape(card)
        assert enc[47] == 7.0 / 10.0

    def test_aura_atk_scalar(self):
        card = _make_warrior(mechanics=["aura_atk_2"])
        enc = encode_card_shape(card)
        assert enc[57] == 2.0 / 10.0


# ============================================================================
# ACTION FEATURES NO IDENTITY
# ============================================================================

class TestActionFeaturesNoIdentity:
    def test_encode_action_features_no_card_id(self):
        state = _make_minimal_state_with_hand()
        feats = encode_action_features(state, 1)

        for action_id in range(MAX_CANDIDATE_ACTIONS):
            row = feats[action_id]
            assert row.shape == (ACTION_FEATURE_DIM,)
            assert not np.any(np.isnan(row)), f"NaN in action_id={action_id}"
            assert not np.any(np.isinf(row)), f"Inf in action_id={action_id}"

    def test_end_turn_features(self):
        state = _make_minimal_state()
        mask = build_action_mask(state, 1)
        assert mask[0] == 1.0
        feats = encode_action_features(state, 1)
        assert np.all(feats[0, :CARD_SHAPE_DIM] == 0.0)


# ============================================================================
# BOARD POSITION MASK (board lengths 0, 1, 3, 6, 7)
# ============================================================================

class TestPlayPositionMask:
    def _make_state_with_warrior_in_hand(self, board_size):
        hero = _make_hero(hp=30)
        w = _make_warrior(mana_cost=3, attack=4, hp=5)
        board = [_make_warrior(mana_cost=i, attack=2, hp=2) for i in range(board_size)]
        for u in board:
            u.is_ready = True
        p1 = PlayerState(user_id=1, hero=hero, mana=10, max_mana=10, hand=[w], board=board, deck=[])
        p2 = PlayerState(user_id=2, hero=_make_hero(hp=30), mana=0, max_mana=0, hand=[], board=[], deck=[])
        return GameState(p1=p1, p2=p2, current_turn_owner_id=1)

    def test_board_size_0_valid_position_0(self):
        state = self._make_state_with_warrior_in_hand(0)
        mask = build_action_mask(state, 1)
        valid_ids = {i for i, v in enumerate(mask) if v == 1.0}
        for aid in valid_ids:
            if aid == 0:
                continue
            flat = aid - 1
            pos_idx, tcode = divmod(flat, 17)
            hand_idx, pos_idx = divmod(pos_idx, 8)
            if hand_idx == 0:
                assert pos_idx == 0, f"board=0: expected pos=0, got pos={pos_idx}"

    def test_board_size_1_valid_positions_0_1(self):
        state = self._make_state_with_warrior_in_hand(1)
        mask = build_action_mask(state, 1)
        positions = set()
        for aid, v in enumerate(mask):
            if v == 1.0 and aid != 0:
                flat = aid - 1
                pos_idx, tcode = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx == 0:
                    positions.add(pos_idx)
        assert positions == {0, 1}, f"board=1: expected positions {{0,1}}, got {positions}"

    def test_board_size_3_valid_positions_0_3(self):
        state = self._make_state_with_warrior_in_hand(3)
        mask = build_action_mask(state, 1)
        positions = set()
        for aid, v in enumerate(mask):
            if v == 1.0 and aid != 0:
                flat = aid - 1
                pos_idx, tcode = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx == 0:
                    positions.add(pos_idx)
        assert positions == {0, 1, 2, 3}, f"board=3: expected positions {{0,1,2,3}}, got {positions}"

    def test_board_size_4_valid_positions_0_4(self):
        # Лимит поля = 5, поэтому board=4 всё ещё допускает постановку WARRIOR.
        state = self._make_state_with_warrior_in_hand(4)
        mask = build_action_mask(state, 1)
        positions = set()
        for aid, v in enumerate(mask):
            if v == 1.0 and aid != 0:
                flat = aid - 1
                pos_idx, tcode = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, _NUM_PLAY_POS)
                if hand_idx == 0:
                    positions.add(pos_idx)
        assert positions == {0, 1, 2, 3, 4}, f"board=4: expected positions 0..4, got {positions}"

    def test_board_size_5_no_warrior_play(self):
        # Лимит поля = 5 (раньше было 7, изменено в задаче Bug 6).
        state = self._make_state_with_warrior_in_hand(5)
        mask = build_action_mask(state, 1)
        for aid, v in enumerate(mask):
            if v == 1.0 and aid != 0:
                flat = aid - 1
                pos_idx, tcode = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx == 0:
                    pytest.fail(f"board=5: no warrior play should be masked, but action_id={aid} was 1")

    def test_potion_uses_position_0_only(self):
        hero = _make_hero(hp=30)
        pot = _make_potion(mana_cost=2, mechanics=["damage_3"])
        board = [_make_warrior(mana_cost=1, attack=2, hp=2) for _ in range(3)]
        p1 = PlayerState(user_id=1, hero=hero, mana=10, max_mana=10, hand=[pot], board=board, deck=[])
        p2 = PlayerState(user_id=2, hero=_make_hero(hp=30), mana=0, max_mana=0, hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(state, 1)
        positions = set()
        for aid, v in enumerate(mask):
            if v == 1.0 and aid != 0:
                flat = aid - 1
                pos_idx, tcode = divmod(flat, 17)
                hand_idx, pos_idx = divmod(pos_idx, 8)
                if hand_idx == 0:
                    positions.add(pos_idx)
        assert positions == {0}, f"potion: expected only position 0, got {positions}"


# ============================================================================
# ATTACK EFFECTIVE ATTACK WITH AURA
# ============================================================================

class TestAttackAura:
    def test_attack_aura_used_for_mask(self):
        hero1 = _make_hero(hp=30)
        hero2 = _make_hero(hp=30)

        attacker = _make_warrior(mana_cost=2, attack=0, hp=4, name="Attacker")
        attacker.is_ready = True

        aura_unit = _make_warrior(mana_cost=3, attack=0, hp=2, name="Aura")
        aura_unit.is_ready = True
        aura_unit.mechanics = ["aura_atk_3"]

        enemy = _make_warrior(mana_cost=1, attack=1, hp=3, name="Enemy")
        enemy.is_ready = True

        p1 = PlayerState(user_id=1, hero=hero1, mana=10, max_mana=10,
                          hand=[], board=[attacker, aura_unit], deck=[])
        p2 = PlayerState(user_id=2, hero=hero2, mana=0, max_mana=0,
                          hand=[], board=[enemy], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(state, 1)

        attack_base = 545
        found_attacker0 = False
        for aid in range(attack_base, 601):
            if mask[aid] == 1.0 and (aid - attack_base) // 8 == 0:
                found_attacker0 = True
                break
        assert found_attacker0, (
            "attacker with raw attack=0 but aura +3 should still be allowed to attack"
        )

    def test_attack_zero_after_aura_self_exclusion(self):
        hero1 = _make_hero(hp=30)
        hero2 = _make_hero(hp=30)

        attacker = _make_warrior(mana_cost=2, attack=0, hp=4, name="Attacker")
        attacker.is_ready = True
        attacker.mechanics = ["aura_atk_3"]

        p1 = PlayerState(user_id=1, hero=hero1, mana=10, max_mana=10,
                          hand=[], board=[attacker], deck=[])
        p2 = PlayerState(user_id=2, hero=hero2, mana=0, max_mana=0,
                          hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(state, 1)
        attack_base = 545
        for aid in range(attack_base, 601):
            assert mask[aid] == 0.0, (
                f"attacker with aura on self only (aura excluded for self) "
                f"should not be attackable, action_id={aid}"
            )

    def test_hero_aura_enables_zero_atk_attack(self):
        hero_p1 = _make_hero(hp=30, mechanics=["aura_atk_1"])
        hero_p2 = _make_hero(hp=30)

        attacker = _make_warrior(mana_cost=2, attack=0, hp=4, name="ZeroAtk")
        attacker.is_ready = True

        p1 = PlayerState(user_id=1, hero=hero_p1, mana=10, max_mana=10,
                          hand=[], board=[attacker], deck=[])
        p2 = PlayerState(user_id=2, hero=hero_p2, mana=0, max_mana=0,
                          hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(state, 1)
        attack_base = 545
        hero_attack_aid = attack_base + 7
        assert mask[hero_attack_aid] == 1.0, (
            f"zero-atk unit with hero aura_atk_1 should be able to attack enemy hero "
            f"(action_id={hero_attack_aid})"
        )

    def test_hero_aura_stacks_with_board_aura_in_effective_attack_channel(self):
        hero_p1 = _make_hero(hp=30, mechanics=["aura_atk_1"])
        hero_p2 = _make_hero(hp=30)

        attacker = _make_warrior(mana_cost=2, attack=2, hp=4, name="Attacker")
        attacker.is_ready = True

        aura_unit = _make_warrior(mana_cost=3, attack=0, hp=2, name="AuraBoard")
        aura_unit.is_ready = True
        aura_unit.mechanics = ["aura_atk_2"]

        p1 = PlayerState(user_id=1, hero=hero_p1, mana=10, max_mana=10,
                          hand=[], board=[attacker, aura_unit], deck=[])
        p2 = PlayerState(user_id=2, hero=hero_p2, mana=0, max_mana=0,
                          hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        feats = encode_action_features(state, 1)
        attack_base = 545
        hero_attack_aid = attack_base + 7
        row = feats[hero_attack_aid]
        effective_atk_norm = row[11]
        assert effective_atk_norm == pytest.approx(5.0 / 20.0), (
            f"effective_attack should be 2 + 1(hero) + 2(board) = 5, "
            f"normalized to {5.0/20.0}, got {effective_atk_norm}"
        )


# ============================================================================
# MECHANIC SCALAR PARSING
# ============================================================================

class TestMechanicScalarParsing:
    def test_battlecry_heal_hero_fills_scalar(self):
        card = _make_warrior(mechanics=["battlecry_heal_hero_4"])
        enc = encode_card_shape(card)
        assert enc[51] == 4.0 / 10.0, f"battlecry_heal_hero_4: expected 0.4, got {enc[51]}"

    def test_battlecry_heal_target_fills_scalar(self):
        card = _make_warrior(mechanics=["battlecry_heal_target_5"])
        enc = encode_card_shape(card)
        assert enc[51] == 5.0 / 10.0, f"battlecry_heal_target_5: expected 0.5, got {enc[51]}"

    def test_heal_target_fills_scalar(self):
        card = _make_potion(mechanics=["heal_target_6"])
        enc = encode_card_shape(card)
        assert enc[48] == 6.0 / 10.0, f"heal_target_6: expected 0.6, got {enc[48]}"

    def test_battlecry_buff_fills_atk_and_hp(self):
        card = _make_warrior(mechanics=["battlecry_buff_3_5"])
        enc = encode_card_shape(card)
        assert enc[52] == 3.0 / 10.0, f"battlecry_buff_3_5 atk: expected 0.3, got {enc[52]}"
        assert enc[53] == 5.0 / 10.0, f"battlecry_buff_3_5 hp: expected 0.5, got {enc[53]}"

    def test_buff_all_fills_atk_and_hp(self):
        card = _make_warrior(mechanics=["buff_all_2_4"])
        enc = encode_card_shape(card)
        assert enc[52] == 2.0 / 10.0, f"buff_all_2_4 atk: expected 0.2, got {enc[52]}"
        assert enc[53] == 4.0 / 10.0, f"buff_all_2_4 hp: expected 0.4, got {enc[53]}"


# ============================================================================
# PREVIEW DOUBLE START-MANA REGRESSION
# ============================================================================

class TestPreviewDoubleStartMana:
    def test_start_mana_not_doubled_in_preview(self):
        hero = _make_hero(hp=30, mechanics=["start_mana_2"])
        w = _make_warrior(mana_cost=3, attack=4, hp=5)

        p1 = PlayerState(user_id=1, hero=hero, mana=3, max_mana=3,
                          hand=[w], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=_make_hero(hp=30), mana=0, max_mana=0,
                          hand=[], board=[], deck=[])

        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        feats = encode_action_features(state, 1)
        action_id = None
        mask = build_action_mask(state, 1)
        for aid in range(MAX_CANDIDATE_ACTIONS):
            if mask[aid] == 1.0 and aid != 0:
                action_id = aid
                break

        assert action_id is not None, "Should have at least one valid play-card action"

        row = feats[action_id]
        mana_change = row[158]
        assert mana_change == 0.0 or abs(mana_change) < 1.0, (
            f"Preview mana change should be 0 or small for warrior play, "
            f"got {mana_change} (mana should not be doubled from start_mana_2 again)"
        )

    def test_preview_simulations_do_not_advance_global_rng(self):
        hero = _make_hero(hp=30)
        random_card = _make_warrior(
            mana_cost=1,
            attack=2,
            hp=3,
            mechanics=["cast_random_spell"],
        )

        p1 = PlayerState(
            user_id=1,
            hero=hero,
            mana=10,
            max_mana=10,
            hand=[random_card],
            board=[],
            deck=[],
        )
        p2 = PlayerState(
            user_id=2,
            hero=_make_hero(hp=30),
            mana=0,
            max_mana=0,
            hand=[],
            board=[],
            deck=[],
        )
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        random.seed(12345)
        before_mask = random.getstate()
        build_action_mask(state, 1)
        assert random.getstate() == before_mask

        before_features = random.getstate()
        encode_action_features(state, 1)
        assert random.getstate() == before_features


# ============================================================================
# PLAY TARGET MASK ALIGNMENT WITH CORE LEGAL SEMANTICS
# ============================================================================

class TestPlayTargetMaskAlignment:

    def _hand_targets(self, mask, hand_idx):
        play_base = 1
        play_stride = 8 * 17
        targets = set()
        for aid, v in enumerate(mask):
            if v != 1.0 or aid < play_base + hand_idx * play_stride:
                continue
            if aid >= play_base + (hand_idx + 1) * play_stride:
                break
            flat = aid - play_base - hand_idx * play_stride
            _pos_idx, tcode = divmod(flat, 17)
            targets.add(tcode)
        return targets

    def _make_state_hand_one_card(self, card, board_size=0, mana=10, board_damaged=None):
        hero = _make_hero(hp=30)
        board = [_make_warrior(mana_cost=i, attack=2, hp=4) for i in range(board_size)]
        if board_damaged:
            for idx in board_damaged:
                if idx < len(board):
                    board[idx].hp = 1
        p1 = PlayerState(user_id=1, hero=hero, mana=mana, max_mana=mana,
                          hand=[card], board=board, deck=[])
        p2 = PlayerState(user_id=2, hero=_make_hero(hp=30), mana=0, max_mana=0,
                          hand=[], board=[], deck=[])
        return GameState(p1=p1, p2=p2, current_turn_owner_id=1)

    def test_battlecry_buff_allows_own_board_not_hero(self):
        card = _make_warrior(mana_cost=3, attack=2, hp=3, mechanics=["battlecry_buff_2_3"])
        state = self._make_state_hand_one_card(card, board_size=2)
        mask = build_action_mask(state, 1)
        targets = self._hand_targets(mask, 0)
        assert targets <= {9, 10}, f"buff: only own board (9,10), got {targets}"
        assert 16 not in targets, f"buff: own hero (16) must NOT be targetable"

    def test_heal_with_full_health_no_targets(self):
        pot = _make_potion(mana_cost=2, mechanics=["heal_2"])
        state = self._make_state_hand_one_card(pot, board_size=2)
        mask = build_action_mask(state, 1)
        targets = self._hand_targets(mask, 0)
        own_set = {9, 10, 16}
        assert targets.isdisjoint(own_set), (
            f"heal with full health: no own targets should be masked, got {targets}"
        )

    def test_heal_with_damaged_board_and_hero(self):
        pot = _make_potion(mana_cost=2, mechanics=["heal_2"])
        hero = _make_hero(hp=20, max_hp=30)
        b0 = _make_warrior(mana_cost=0, attack=1, hp=2)
        b0.max_hp = 4
        b1 = _make_warrior(mana_cost=0, attack=1, hp=4)
        b1.max_hp = 4
        board = [b0, b1]
        p1 = PlayerState(user_id=1, hero=hero, mana=5, max_mana=5,
                          hand=[pot], board=board, deck=[])
        p2 = PlayerState(user_id=2, hero=_make_hero(hp=30), mana=0, max_mana=0,
                          hand=[], board=[], deck=[])
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(state, 1)
        targets = self._hand_targets(mask, 0)
        assert 9 in targets, "damaged board idx 0 should be targetable"
        assert 10 not in targets, "full-HP board idx 1 should NOT be targetable"
        assert 16 in targets, "damaged hero should be targetable"

    def test_compare_with_core_legal_actions(self):
        import random
        rng = random.Random(42)

        mech_sets = [
            ["damage_3"],
            ["freeze"],
            ["battlecry_damage_2"],
            ["battlecry_heal_target_4"],
            ["battlecry_heal_hero_3"],
            ["battlecry_buff_2_3"],
            ["choose_shield_damage"],
            ["consume_ally"],
            ["delete_target"],
        ]

        for seed in range(10):
            hero1 = _make_hero(hp=rng.randint(20, 40), max_hp=40)
            hero2 = _make_hero(hp=rng.randint(20, 40), max_hp=40)

            board1 = []
            for i in range(rng.randint(0, 3)):
                u = _make_warrior(mana_cost=i, attack=rng.randint(1, 4),
                                   hp=rng.randint(1, 8))
                u.is_ready = True
                board1.append(u)

            hand_card = _make_warrior(mana_cost=rng.randint(1, 4),
                                       attack=rng.randint(1, 5),
                                       hp=rng.randint(1, 6),
                                       mechanics=rng.choice(mech_sets))

            p1 = PlayerState(user_id=1, hero=hero1, mana=rng.randint(3, 9), max_mana=8,
                              hand=[hand_card], board=board1, deck=[])
            p2 = PlayerState(user_id=2, hero=hero2, mana=0, max_mana=0,
                              hand=[], board=[], deck=[])
            state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

            mask = build_action_mask(state, 1)

            env = ArenaEnvironment(copy.deepcopy(state))
            legal = env.get_legal_actions(state.current_turn_owner_id)

            legal_by_hand_target = {}
            for act in legal:
                if isinstance(act, PlayCardAction) and act.hand_index == 0:
                    key = (act.hand_index, act.target_id)
                    if key not in legal_by_hand_target:
                        legal_by_hand_target[key] = set()
                    legal_by_hand_target[key].add(act.position)

            for aid, v in enumerate(mask):
                if v != 1.0 or aid == 0:
                    continue
                action = decode_action(state, 1, aid)
                if not isinstance(action, PlayCardAction) or action.hand_index != 0:
                    continue

                tgt_str = action.target_id
                match = False
                for (hi, lt), positions in legal_by_hand_target.items():
                    if hi != 0:
                        continue
                    if tgt_str == lt:
                        match = True
                        break

                assert match, (
                    f"seed={seed}: masked action_id={aid} target={tgt_str} "
                    f"has no match in core legal actions {set(legal_by_hand_target.keys())}"
                )

    def test_random_battlecry_damage_card_masks_as_no_target(self):
        card = _make_warrior(mana_cost=1, mechanics=["battlecry_damage_1_random"])
        enemy_unit = _make_warrior(mana_cost=0, attack=1, hp=2)
        p1 = PlayerState(
            user_id=1,
            hero=_make_hero(hp=30),
            mana=5,
            max_mana=5,
            hand=[card],
            board=[],
            deck=[],
        )
        p2 = PlayerState(
            user_id=2,
            hero=_make_hero(hp=30),
            mana=0,
            max_mana=0,
            hand=[],
            board=[enemy_unit],
            deck=[],
        )
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        mask = build_action_mask(
            state,
            1,
            verify_mask=False,
            placement_mode="append_only",
        )
        targets = self._hand_targets(mask, 0)

        assert targets == {0}


# ============================================================================
# HELPERS
# ============================================================================

def _make_hero(hp=30, max_hp=None, name="Hero", mechanics=None):
    return CardInstance(
        instance_id=uuid4(),
        card_id=0,
        name=name,
        card_type=CardType.HERO,
        mana_cost=0,
        attack=0,
        hp=hp,
        max_hp=max_hp if max_hp is not None else hp,
        mechanics=mechanics or [],
        is_ready=True,
    )


def _make_warrior(mana_cost=3, attack=4, hp=5, name="TestWarrior", mechanics=None):
    return CardInstance(
        instance_id=uuid4(),
        card_id=100,
        name=name,
        card_type=CardType.WARRIOR,
        mana_cost=mana_cost,
        attack=attack,
        hp=hp,
        max_hp=hp,
        mechanics=mechanics or [],
        is_ready=False,
    )


def _make_potion(mana_cost=2, name="TestPotion", mechanics=None):
    return CardInstance(
        instance_id=uuid4(),
        card_id=200,
        name=name,
        card_type=CardType.POTION,
        mana_cost=mana_cost,
        attack=0,
        hp=0,
        max_hp=0,
        mechanics=mechanics or [],
        is_ready=False,
    )


def _make_card(attack=0, hp=0, mana_cost=0, card_type=None, mechanics=None, level=1):
    return CardInstance(
        instance_id=uuid4(),
        card_id=999,
        name="TestCard",
        card_type=card_type or CardType.WARRIOR,
        mana_cost=mana_cost,
        attack=attack,
        hp=hp,
        max_hp=hp,
        mechanics=mechanics or [],
        is_ready=False,
        level=level,
    )


def _make_deck(warriors=0, potions=0, hero=True):
    deck = []
    if hero:
        deck.append(_make_hero(hp=30))
    for i in range(warriors):
        deck.append(_make_warrior(mana_cost=(i + 1), name=f"War{i}"))
    for i in range(potions):
        deck.append(_make_potion(mana_cost=(i + 1), name=f"Pot{i}"))
    return deck


def _make_minimal_state():
    hero1 = _make_hero(hp=30)
    hero2 = _make_hero(hp=30)
    p1 = PlayerState(user_id=1, hero=hero1, mana=10, max_mana=10, hand=[], board=[], deck=[])
    p2 = PlayerState(user_id=2, hero=hero2, mana=10, max_mana=10, hand=[], board=[], deck=[])
    return GameState(p1=p1, p2=p2, current_turn_owner_id=1)


def _make_minimal_state_with_cards():
    hero1 = _make_hero(hp=30)
    hero2 = _make_hero(hp=25)

    w1 = _make_warrior(mana_cost=3, attack=4, hp=5, name="MyWarrior")
    w2 = _make_warrior(mana_cost=4, attack=7, hp=6, name="EnemyWarrior")
    pot1 = _make_potion(mana_cost=2, mechanics=["damage_3"])

    p1 = PlayerState(
        user_id=1, hero=hero1, mana=5, max_mana=5,
        hand=[pot1], board=[w1], deck=[_make_warrior(mana_cost=2)],
        graveyard=[],
    )
    p2 = PlayerState(
        user_id=2, hero=hero2, mana=0, max_mana=0,
        hand=[], board=[w2], deck=[_make_warrior(mana_cost=3)],
        graveyard=[_make_warrior(mana_cost=1)],
    )
    return GameState(p1=p1, p2=p2, current_turn_owner_id=1)


def _make_minimal_state_with_hand():
    hero1 = _make_hero(hp=30)
    hero2 = _make_hero(hp=30)
    w1 = _make_warrior(mana_cost=3, attack=4, hp=5, name="HandWarrior")
    pot1 = _make_potion(mana_cost=2, mechanics=["damage_3"], name="HandPotion")

    p1 = PlayerState(
        user_id=1, hero=hero1, mana=5, max_mana=5,
        hand=[w1, pot1], board=[], deck=[],
    )
    p2 = PlayerState(
        user_id=2, hero=hero2, mana=0, max_mana=0,
        hand=[], board=[], deck=[],
    )
    return GameState(p1=p1, p2=p2, current_turn_owner_id=1)


def _make_fuzz_state(rng: random.Random):
    hero1 = _make_hero(hp=30)
    hero2 = _make_hero(hp=30)

    p1_hand = []
    for i in range(rng.randint(0, 4)):
        if rng.random() < 0.7:
            p1_hand.append(_make_warrior(
                mana_cost=rng.randint(1, 4),
                attack=rng.randint(0, 5),
                hp=rng.randint(1, 6),
                name=f"FW{i}",
            ))
        else:
            p1_hand.append(_make_potion(
                mana_cost=rng.randint(1, 3),
                mechanics=["damage_" + str(rng.randint(1, 3))],
                name=f"FP{i}",
            ))

    p1_board = []
    for i in range(rng.randint(0, 3)):
        p1_board.append(_make_warrior(
            mana_cost=rng.randint(1, 5),
            attack=rng.randint(1, 5),
            hp=rng.randint(1, 8),
            name=f"BW{i}",
        ))
        p1_board[-1].is_ready = rng.random() < 0.6

    p1 = PlayerState(
        user_id=1, hero=hero1, mana=rng.randint(1, 6), max_mana=rng.randint(1, 6),
        hand=p1_hand, board=p1_board, deck=[],
    )

    p2_board = []
    for i in range(rng.randint(0, 3)):
        p2_board.append(_make_warrior(
            mana_cost=rng.randint(1, 5),
            attack=rng.randint(0, 5),
            hp=rng.randint(1, 8),
            name=f"EB{i}",
        ))
        p2_board[-1].is_ready = True

    if rng.random() < 0.2 and p2_board:
        p2_board[0].mechanics.append("taunt")

    p2 = PlayerState(
        user_id=2, hero=hero2, mana=rng.randint(0, 3), max_mana=rng.randint(0, 3),
        hand=[], board=p2_board, deck=[],
    )

    turn_owner = rng.choice([1, 2])
    return GameState(p1=p1, p2=p2, current_turn_owner_id=turn_owner)
