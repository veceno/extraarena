"""V5 mana_draw parallel binary head — legal mask + selection (pure Python).

Block 0 component 3 (spec §0.89 decision γ + §6.186). ``mana_draw`` is a
PARALLEL BINARY HEAD — NOT a 602nd candidate. ``MAX_CANDIDATE_ACTIONS=601``
stays frozen at ``classic_actions_v1.py:46``. ``ManaDrawAction`` is
``core/actions.py:76``, a core action OUTSIDE the 601 candidate space.

This module is the SOURCE OF TRUTH for the mana_draw legal predicate. The MLX
``Linear(hidden_dim, 1)`` head in ``v5_policy.py`` consumes the mask to gate its
logit (illegal → -inf). The Rust kernel mirrors the same predicate (Block -1
Phase 2 mana_draw legality — see ``kernel.rs``).

The mask mirrors the real-game oracle BYTE-FOR-BEHAVIOR:

  golden_trace.py:523
      mana_draw_legal = any(isinstance(a, ManaDrawAction)
                            for a in env.get_legal_actions(player_id))

That predicate inherits ``core/engine.py`` ``get_legal_actions``'s FULL mana_draw
emission path:

  - game not ONGOING            → []            → False  (engine.py:1206-1207)
  - not this player's turn      → []            → False  (engine.py:1210-1211)
  - unknown player_id           → []            → False  (engine.py:1214-1216)
  - hand >= HAND_CAP (=4)        → no ManaDrawAction       (engine.py:1344;
                                                              :781-782 hand_full)
  - mana < MANA_DRAW_BASE*(cnt+1) → no ManaDrawAction     (engine.py:1345-1346;
                                                              :785-786 insufficient_mana)
  - otherwise                   → ManaDrawAction emitted  → True  (engine.py:1347)

``_handle_mana_draw`` (engine.py:781-786) repeats the hand_full +
insufficient_mana guards as defense-in-depth on the apply path; the
legal-actions gate at :1344-1347 already enforces them, so the mask only needs
the legal-actions emission path (which subsumes the apply guards).

Pure Python: NO mlx import. This is the testable-here counterpart to the MLX
head (which is skip-gated — MLX is not importable in this worktree).
"""
from __future__ import annotations

from core.engine import HAND_CAP, MANA_DRAW_BASE
from core.state import GameState, GameStatus

# Re-exported so consumers can import the ruleset constants from one place.
__all__ = [
    "HAND_CAP",
    "MANA_DRAW_BASE",
    "mana_draw_cost",
    "mana_draw_legal_mask",
    "select_includes_mana_draw",
]


def _resolve_player(state: GameState, player_id: int):
    """Mirror ``ArenaEnvironment._resolve_player_pair`` (core/engine.py:293-298).

    p1 is checked first (byte-parity with the engine ordering).
    """
    if state.p1.user_id == player_id:
        return state.p1, state.p2
    if state.p2.user_id == player_id:
        return state.p2, state.p1
    return None, None


def mana_draw_cost(mana_draw_count_this_turn: int) -> int:
    """Cost of the next player-initiated mana draw this turn.

    ``MANA_DRAW_BASE * (count + 1)`` = ``2 * (count + 1)`` → 2, 4, 6, ...
    (core/engine.py:784, :1345). The counter resets to 0 at the start of each
    owner turn (engine.py:699 / state.py:144).
    """
    return MANA_DRAW_BASE * (int(mana_draw_count_this_turn) + 1)


def mana_draw_legal_mask(state: GameState, player_id: int) -> bool:
    """Legal predicate for the parallel mana_draw binary head.

    Returns ``True`` iff the engine would emit a ``ManaDrawAction`` in
    ``get_legal_actions(player_id)`` — i.e. exactly the
    ``golden_trace.py:523`` oracle
    ``any(isinstance(a, ManaDrawAction) for a in env.get_legal_actions(pid))``.

    Byte-parity with ``core/engine.py``:

      * ``state.status != GameStatus.ONGOING`` → False (engine.py:1206-1207).
      * ``state.current_turn_owner_id != player_id`` → False
        (engine.py:1210-1211).
      * unknown ``player_id`` → False (engine.py:1214-1216).
      * ``len(player.hand) >= HAND_CAP`` (=4) → False
        (engine.py:1344 + hand_full guard :781-782).
      * ``player.mana < MANA_DRAW_BASE * (mana_draw_count_this_turn + 1)``
        → False (engine.py:1345-1346 + insufficient_mana guard :785-786).
      * deck or graveyard must contain a drawable card; otherwise the engine
        does not emit a non-executable ManaDrawAction.
    """
    # engine.py:1206-1207 — game over → get_legal_actions returns [].
    if state.status != GameStatus.ONGOING:
        return False
    # engine.py:1210-1211 — not this player's turn → get_legal_actions returns [].
    if state.current_turn_owner_id != player_id:
        return False
    player, _ = _resolve_player(state, player_id)
    # engine.py:1214-1216 — unknown player → get_legal_actions returns [].
    if player is None:
        return False
    # engine.py:1344 + _handle_mana_draw hand_full guard (engine.py:781-782).
    if len(player.hand) >= HAND_CAP:
        return False
    # engine.py:1345-1346 + _handle_mana_draw insufficient_mana guard (:785-786).
    if player.mana < mana_draw_cost(player.mana_draw_count_this_turn):
        return False
    if not player.deck and not player.graveyard:
        return False
    # engine.py:1347 — ManaDrawAction emitted.
    return True


def select_includes_mana_draw(
    mana_draw_logit: float,
    best_candidate_logit: float,
    mana_draw_legal: bool,
) -> bool:
    """Selection helper: whether the policy takes the mana_draw action this step.

    The parallel head is a Bernoulli gate, trained as ``P(draw)=sigmoid(logit)``.
    It must not be numerically compared to an individual candidate logit.
    Deterministic inference chooses draw exactly when its probability exceeds
    one half (raw logit > 0). ``best_candidate_logit`` remains in the public
    signature for existing ONNX callers, but is intentionally not used by the
    factorized decision rule. When ``mana_draw_legal`` is False the legal mask
    dominates the head output.
    """
    if not mana_draw_legal:
        return False
    del best_candidate_logit
    return float(mana_draw_logit) > 0.0
