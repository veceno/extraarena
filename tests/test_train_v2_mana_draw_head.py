"""Tests for the V5 mana_draw parallel binary head (Block 0 component 3 — spec
§0.89 decision γ + §6.186).

SOURCE-VS-SOURCE ORACLE (Block -1 lesson): the mask is asserted against the
REAL-GAME oracle, NOT a self-referential fixture. The oracle is the exact
``golden_trace.py:523`` predicate

    mana_draw_legal = any(isinstance(a, ManaDrawAction)
                           for a in env.get_legal_actions(player_id))

which inherits ``core/engine.py`` ``get_legal_actions``'s full mana_draw
emission path (game-over :1206-1207, wrong-turn :1210-1211, unknown-player
:1214-1216, hand_full :1344/781-782, insufficient_mana :1345-1346/785-786,
emit :1347). We build REAL ``GameState``s via ``ArenaEnvironment`` and compare
``mana_draw_legal_mask(state, pid)`` to that oracle across many states.

FROZEN-CLASSIC GUARD: ``classic_actions_v1.py`` is never touched
(``MAX_CANDIDATE_ACTIONS=601`` stays frozen at :46; ``ManaDrawAction`` is
``core/actions.py:76``, outside the 601 space). mana_draw is a PARALLEL BINARY
HEAD, not a 602nd candidate.

MLX note: the mask module (``mana_draw_head_v5``) is PURE PYTHON — no mlx
import — so the mask/selection tests run green WITHOUT mlx. The single
MLX-head test (``TestV5PolicyManaDrawHead``) is skip-gated via
``pytest.importorskip("mlx")`` so it runs when mlx is present and skips
cleanly when absent (per the component-4 warm-start skip-gate pattern).
"""
from __future__ import annotations

import pytest

from core.actions import ManaDrawAction
from core.engine import HAND_CAP, MANA_DRAW_BASE, ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState

from train_v3.mana_draw_head_v5 import (
    mana_draw_cost,
    mana_draw_legal_mask,
    select_includes_mana_draw,
)


# ---------------------------------------------------------------------------
# GameState builders (mirror test_train_v3_obs_v5.py conventions).
# ---------------------------------------------------------------------------
def _hero(user_id: int) -> CardInstance:
    return CardInstance(
        card_id=0,
        name=f"hero_{user_id}",
        card_type=CardType.HERO,
        mana_cost=0,
        attack=0,
        hp=30,
        max_hp=30,
        level=1,
    )


def _card(card_id: int) -> CardInstance:
    return CardInstance(
        card_id=card_id,
        name=f"card_{card_id}",
        card_type=CardType.WARRIOR,
        mana_cost=1,
        attack=1,
        hp=1,
        max_hp=1,
        level=1,
    )


def _player(user_id: int, *, hand_size: int = 0, mana: int = 10,
            mana_draw_count_this_turn: int = 0) -> PlayerState:
    return PlayerState(
        user_id=user_id,
        hero=_hero(user_id),
        mana=mana,
        max_mana=max(mana, 10),
        mana_draw_count_this_turn=mana_draw_count_this_turn,
        hand=[_card(100 + i) for i in range(hand_size)],
    )


def _state(*, me_hand_size=2, me_mana=10, me_count=0, me_user_id=1,
            enemy_user_id=2, turn_owner=None, status=GameStatus.ONGOING) -> GameState:
    p1 = _player(me_user_id, hand_size=me_hand_size, mana=me_mana,
                 mana_draw_count_this_turn=me_count)
    p2 = _player(enemy_user_id)
    st = GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=turn_owner if turn_owner is not None else me_user_id,
        turn_number=1,
    )
    st.status = status
    return st


def _oracle(state: GameState, player_id: int) -> bool:
    """The golden_trace.py:523 predicate on the REAL engine legal-action list.

    ``mana_draw_legal = any(isinstance(a, ManaDrawAction)
                             for a in env.get_legal_actions(player_id))``

    ``ArenaEnvironment(state, apply_start_effects=False)`` preserves the
    hand-crafted mana/hand/count (skips start_mana / turn-mode effects); only
    ``_ensure_base_snapshots`` runs (harmless card base-snapshotting).
    """
    env = ArenaEnvironment(state, apply_start_effects=False)
    legal = env.get_legal_actions(player_id)
    return any(isinstance(a, ManaDrawAction) for a in legal)


# ============================================================================
# mana_draw_cost helper (engine.py:784, :1345)
# ============================================================================
class TestManaDrawCost:
    def test_cost_is_2_times_count_plus_1(self):
        # MANA_DRAW_BASE=2 (engine.py:59); cost = 2*(count+1) → 2, 4, 6, 8, ...
        assert MANA_DRAW_BASE == 2
        assert mana_draw_cost(0) == 2
        assert mana_draw_cost(1) == 4
        assert mana_draw_cost(2) == 6
        assert mana_draw_cost(3) == 8
        assert mana_draw_cost(5) == 12

    def test_cost_coerces_int(self):
        # Defensive: float/str-ish inputs coerce to int (mirrors int() cast).
        assert mana_draw_cost(0.0) == 2
        assert mana_draw_cost(1) == 4


# ============================================================================
# Hand-full guard (engine.py:781-782, :1344)
# ============================================================================
class TestManaDrawLegalMaskHandFull:
    def test_mana_draw_legal_mask_hand_full_unset(self):
        """hand >= HAND_CAP=4 → mask False, parity engine.py:781-782 (hand_full)
        + :1344 (``if len(player.hand) < HAND_CAP``). Plenty of mana, so the ONLY
        failing condition is the full hand."""
        assert HAND_CAP == 4
        for hand_size in (4, 5, 6):
            state = _state(me_hand_size=hand_size, me_mana=10, me_count=0)
            assert mana_draw_legal_mask(state, 1) is False, (
                f"hand_size={hand_size} (>= HAND_CAP=4) must be illegal even "
                f"with ample mana (parity engine.py:781-782)"
            )

    def test_hand_full_beats_sufficient_mana(self):
        """hand_full is checked before insufficient_mana in _handle_mana_draw
        (engine.py:781 before :785); the mask returns False regardless of mana
        when the hand is full."""
        state = _state(me_hand_size=4, me_mana=12, me_count=0)
        assert mana_draw_legal_mask(state, 1) is False


# ============================================================================
# Insufficient-mana guard (engine.py:785-786, :1345-1346)
# ============================================================================
class TestManaDrawLegalMaskInsufficientMana:
    def test_mana_draw_legal_mask_insufficient_mana_unset(self):
        """mana < MANA_DRAW_BASE*(count+1) → mask False, parity engine.py:785-786
        (insufficient_mana) + :1345-1346. Room in hand (so the ONLY failing
        condition is mana)."""
        # count=0 → cost 2; mana 0 and 1 both < 2 → illegal.
        for mana in (0, 1):
            state = _state(me_hand_size=0, me_mana=mana, me_count=0)
            assert mana_draw_legal_mask(state, 1) is False, (
                f"mana={mana} < cost=2 (count=0) must be illegal "
                f"(parity engine.py:785-786)"
            )

    def test_insufficient_mana_with_count_increases_cost(self):
        """count=2 → cost = 2*(2+1) = 6; mana=5 < 6 → illegal; mana=6 → legal."""
        state = _state(me_hand_size=1, me_mana=5, me_count=2)
        assert mana_draw_legal_mask(state, 1) is False, "mana=5 < cost=6"
        state_ok = _state(me_hand_size=1, me_mana=6, me_count=2)
        assert mana_draw_legal_mask(state_ok, 1) is True, "mana=6 == cost=6 → legal"

    def test_mana_equal_to_cost_is_legal(self):
        """mana == cost is legal (the gate is ``player.mana >= mana_draw_cost``,
        engine.py:1346, NOT strict-greater)."""
        # count=0 → cost 2; mana exactly 2 → legal.
        state = _state(me_hand_size=0, me_mana=2, me_count=0)
        assert mana_draw_legal_mask(state, 1) is True


# ============================================================================
# Legal when room and mana (engine.py:1347)
# ============================================================================
class TestManaDrawLegalMaskLegal:
    def test_mana_draw_legal_mask_legal_when_room_and_mana(self):
        """hand < HAND_CAP=4 AND mana >= cost → True (engine.py:1347 emits)."""
        # hand=2, mana=10, count=0 → cost 2; 10 >= 2 and 2 < 4 → legal.
        state = _state(me_hand_size=2, me_mana=10, me_count=0)
        assert mana_draw_legal_mask(state, 1) is True

    def test_legal_across_hand_sizes_below_cap(self):
        for hand_size in (0, 1, 2, 3):
            state = _state(me_hand_size=hand_size, me_mana=10, me_count=0)
            assert mana_draw_legal_mask(state, 1) is True, (
                f"hand_size={hand_size} (< 4) with mana=10 must be legal"
            )

    def test_legal_across_increasing_counts_with_mana(self):
        """Each successive draw costs +2; mana keeps pace → legal."""
        for count, cost in ((0, 2), (1, 4), (2, 6), (3, 8)):
            state = _state(me_hand_size=0, me_mana=cost, me_count=count)
            assert mana_draw_legal_mask(state, 1) is True, (
                f"count={count} cost={cost} mana={cost} → legal"
            )


# ============================================================================
# Game-status / turn-owner / unknown-player guards (engine.py:1206-1216)
# ============================================================================
class TestManaDrawLegalMaskGuards:
    def test_game_over_unset(self):
        """status != ONGOING → get_legal_actions returns [] → mask False
        (engine.py:1206-1207; golden_trace.py:523 oracle inherits this)."""
        for status in (GameStatus.P1_WIN, GameStatus.P2_WIN, GameStatus.DRAW):
            state = _state(me_hand_size=2, me_mana=10, me_count=0, status=status)
            assert mana_draw_legal_mask(state, 1) is False, (
                f"status={status} must be illegal (game over)"
            )

    def test_wrong_turn_unset(self):
        """current_turn_owner_id != player_id → [] → False (engine.py:1210-1211).
        The hand has room and mana, so the ONLY failing condition is the turn."""
        state = _state(me_hand_size=2, me_mana=10, me_count=0,
                       turn_owner=2)  # enemy's turn
        assert mana_draw_legal_mask(state, 1) is False, (
            "asking for player 1's mana_draw on player 2's turn must be illegal"
        )
        # The actual turn owner (player 2) IS legal.
        assert mana_draw_legal_mask(state, 2) is True

    def test_unknown_player_unset(self):
        """player_id matching neither p1 nor p2 → [] → False
        (engine.py:1214-1216)."""
        state = _state(me_hand_size=2, me_mana=10, me_count=0)
        assert mana_draw_legal_mask(state, 999) is False, (
            "unknown player_id must be illegal"
        )


# ============================================================================
# BYTE-PARITY with golden_trace.py:523 across real GameStates
# ============================================================================
class TestGoldenTraceByteParity:
    """Source-vs-source oracle: for many real GameStates, the mask must equal
    ``any(isinstance(a, ManaDrawAction) for a in env.get_legal_actions(pid))``
    — the EXACT golden_trace.py:523 predicate. This is the real-game oracle,
    not a re-derived approximation."""

    @pytest.mark.parametrize(
        "hand_size,mana,count",
        [
            # hand_room × mana × count matrix covering legal + both guards.
            (0, 0, 0),    # hand empty, mana 0 < cost 2 → insufficient_mana
            (0, 1, 0),    # mana 1 < cost 2 → insufficient_mana
            (0, 2, 0),    # mana == cost 2 → legal (boundary)
            (0, 10, 0),   # ample mana, empty hand → legal
            (1, 2, 0),    # legal
            (2, 10, 0),   # legal
            (3, 10, 0),   # legal (hand one below cap)
            (4, 10, 0),   # hand_full
            (5, 10, 0),   # hand_full (synthetic >cap)
            (0, 3, 1),    # count=1 → cost 4; mana 3 < 4 → insufficient
            (0, 4, 1),    # count=1 → cost 4; mana == 4 → legal (boundary)
            (1, 5, 2),    # count=2 → cost 6; mana 5 < 6 → insufficient
            (1, 6, 2),    # count=2 → cost 6; mana == 6 → legal
            (0, 7, 3),    # count=3 → cost 8; mana 7 < 8 → insufficient
            (0, 8, 3),    # count=3 → cost 8; mana == 8 → legal
            (0, 12, 5),   # count=5 → cost 12; mana == 12 → legal
            (0, 11, 5),   # count=5 → cost 12; mana 11 < 12 → insufficient
        ],
    )
    def test_mask_matches_oracle_hand_mana_count(self, hand_size, mana, count):
        state = _state(me_hand_size=hand_size, me_mana=mana, me_count=count)
        assert mana_draw_legal_mask(state, 1) == _oracle(state, 1), (
            f"mask/oracle divergence for hand={hand_size} mana={mana} "
            f"count={count}: mask={mana_draw_legal_mask(state, 1)} "
            f"oracle={_oracle(state, 1)}"
        )

    @pytest.mark.parametrize("status", [GameStatus.P1_WIN, GameStatus.P2_WIN, GameStatus.DRAW])
    def test_mask_matches_oracle_game_over(self, status):
        """game over → oracle [] → False; mask must agree."""
        state = _state(me_hand_size=2, me_mana=10, me_count=0, status=status)
        assert mana_draw_legal_mask(state, 1) == _oracle(state, 1) is False

    def test_mask_matches_oracle_wrong_turn(self):
        """wrong turn → oracle [] for the non-owner; mask must agree for BOTH
        the non-owner (False) and the owner (True)."""
        state = _state(me_hand_size=2, me_mana=10, me_count=0, turn_owner=2)
        assert mana_draw_legal_mask(state, 1) == _oracle(state, 1)  # False == False
        assert mana_draw_legal_mask(state, 2) == _oracle(state, 2)  # True == True

    def test_mask_matches_oracle_unknown_player(self):
        state = _state(me_hand_size=2, me_mana=10, me_count=0)
        assert mana_draw_legal_mask(state, 999) == _oracle(state, 999)  # False == False

    def test_mask_matches_oracle_deck_empty_still_legal(self):
        """mana_draw is LEGAL even with an empty deck: get_legal_actions
        (engine.py:1344-1347) only checks hand<cap and mana>=cost — the deck
        check lives on the APPLY path (_handle_mana_draw returns
        'no_cards_to_draw', engine.py:797-801), NOT the legal-actions path.
        So the oracle emits ManaDrawAction and the mask must agree True.
        (Player built with no deck → empty deck; ArenaEnvironment constructed
        without one.)"""
        p1 = PlayerState(
            user_id=1, hero=_hero(1), mana=10, max_mana=10,
            mana_draw_count_this_turn=0, hand=[_card(11)], deck=[],
        )
        p2 = _player(2)
        state = GameState(p1=p1, p2=p2, current_turn_owner_id=1, turn_number=1)
        assert mana_draw_legal_mask(state, 1) == _oracle(state, 1) is True


# ============================================================================
# Selection helper (spec §6.186)
# ============================================================================
class TestSelectIncludesManaDraw:
    def test_illegal_never_selected_even_if_logit_higher(self):
        """The legal mask dominates the head: illegal → False regardless of
        logit (the engine never emits the action, so it can never be taken)."""
        assert select_includes_mana_draw(9.0, 1.0, False) is False

    def test_legal_and_higher_logit_selected(self):
        assert select_includes_mana_draw(2.0, 1.0, True) is True

    def test_legal_and_lower_logit_not_selected(self):
        assert select_includes_mana_draw(1.0, 2.0, True) is False

    def test_tie_favors_candidate(self):
        """mana_draw logit == best candidate logit → NOT selected (the 601 path
        is the default action space; ties deterministically favor the
        candidate)."""
        assert select_includes_mana_draw(1.0, 1.0, True) is False

    def test_legal_mask_dominates_all_logit_combos(self):
        """For any logit pair, illegal → False (mask wins)."""
        for md, best in [(5.0, 0.0), (0.0, 5.0), (5.0, 5.0), (-1.0, -2.0)]:
            assert select_includes_mana_draw(md, best, False) is False


# ============================================================================
# MLX-head wiring (skip-gated: runs iff mlx importable, else skips cleanly)
# ============================================================================
class TestV5PolicyManaDrawHead:
    """Skip-gated MLX test (component-4 warm-start skip-gate pattern). When mlx
    is importable this asserts the V5 head wiring; when mlx is absent it skips
    without failing. The pure-python mask tests above run regardless."""

    def test_policy_returns_3tuple_with_finite_mana_draw_logit(self):
        mlx = pytest.importorskip("mlx")  # skip if mlx absent
        import mlx.core as mx  # noqa: F401
        import numpy as np

        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM
        from train_v3.v5_policy import V5ActionConditionedPolicy

        pol = V5ActionConditionedPolicy()
        # mana_draw_head must be a (hidden_dim, 1) Linear alongside value_head.
        # MLX Linear exposes dims via weight.shape == (out, in) = (1, hidden_dim).
        assert hasattr(pol, "mana_draw_head"), "mana_draw_head must exist"
        assert pol.mana_draw_head.weight.shape == (1, pol.hidden_dim), (
            "mana_draw_head must be Linear(hidden_dim, 1) "
            f"(got weight.shape={pol.mana_draw_head.weight.shape})"
        )
        assert pol.mana_draw_head.bias.shape == (1,), "mana_draw_head bias (1,)"
        # value_head still present and untouched (frozen 601 path unaffected).
        assert hasattr(pol, "value_head")
        assert pol.value_head.weight.shape == (1, pol.hidden_dim)

        batch = 3
        obs = mx.zeros((batch, OBS_V5_DIM))
        action_features = mx.zeros((batch, 601, ACTION_FEATURE_DIM))

        # Default (no mask): raw head logit — must be finite for finite input.
        out = pol(obs, action_features)
        assert len(out) == 3, "must return 3-tuple (logits, value, mana_draw_logit)"
        logits, value, mana_draw_logit = out
        assert logits.shape == (batch, 601), "601 candidate path must stay 601"
        assert value.shape == (batch,)
        assert mana_draw_logit.shape == (batch,)
        mx.eval(logits, value, mana_draw_logit)
        assert bool(mx.all(mx.isfinite(mana_draw_logit))), \
            "raw mana_draw_logit must be finite for finite input"

        # Gating: legal rows keep the finite logit, illegal rows → -inf.
        out2 = pol(obs, action_features, mana_draw_legal=mx.array([True, False, True]))
        md2 = out2[2]
        mx.eval(md2)
        md2_np = np.asarray(md2)
        assert np.isfinite(md2_np[0]) and np.isfinite(md2_np[2]), \
            "legal rows must keep finite logit"
        assert np.isinf(md2_np[1]) and md2_np[1] < 0, \
            "illegal row must be -inf (masked)"

        # 601 candidate path is frozen — logits unaffected by the mask arg.
        out3 = pol(obs, action_features)  # no mask
        logits3 = np.asarray(out3[0])
        mx.eval(out3[0])
        logits2 = np.asarray(out2[0])
        mx.eval(out2[0])
        np.testing.assert_array_equal(logits2, logits3, err_msg=(
            "candidate logits must not change when mana_draw_legal is passed "
            "(601 path frozen, no 602nd candidate)"
        ))

    def test_policy_accepts_python_bool_mask(self):
        """A scalar Python bool mask must also gate (broadcast over the batch)."""
        pytest.importorskip("mlx")
        import mlx.core as mx
        import numpy as np

        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM
        from train_v3.v5_policy import V5ActionConditionedPolicy

        pol = V5ActionConditionedPolicy()
        obs = mx.zeros((2, OBS_V5_DIM))
        af = mx.zeros((2, 601, ACTION_FEATURE_DIM))
        # all-legal (True) → finite; all-illegal (False) → all -inf.
        out_t = pol(obs, af, mana_draw_legal=True)
        out_f = pol(obs, af, mana_draw_legal=False)
        mx.eval(out_t[2], out_f[2])
        assert bool(mx.all(mx.isfinite(out_t[2]))), "all-legal → finite"
        assert bool(mx.all(mx.isinf(out_f[2]))), "all-illegal → -inf"