"""Block A component A1 — BC dataset builder + V5 601-tcode resolver + human filter.

Consumes recorded pilot v5 traces via the ``offline_dataset_loader`` PUBLIC API
(``ai/train_v2/offline_dataset_loader.py`` — Block 0 component 5, DONE, NOT
frozen) and emits BC (behavior-cloning) training tuples. RESOLVES the V5
601-tcode that Block 0 component 5 explicitly deferred
(``offline_dataset_loader.py`` OfflineTransition.action_tcode_or_index stores
the RECORDED ``legal_action_index`` — index into ``get_legal_actions_raw``
0..N-1 — NOT the V5 601-tcode).

601-tcode RESOLUTION (source-vs-source; engine = oracle, decode_action = UUT):
  reconstruct ``pre_state`` via ``reconstruct_gamestate(snapshot)``; build the
  legal 601-candidate mask via
  ``build_action_mask(pre_state, actor, verify_mask=False,
  placement_mode='append_only')`` (``classic_actions_v1.py:188``, FROZEN
  read-only); enumerate ``legal_action_ids = np.flatnonzero(mask)``; decode
  each candidate via ``decode_action(pre_state, actor, candidate_id)``
  (``classic_actions_v1.py:70``, FROZEN read-only); value-equality match
  ``decode_action(candidate).to_dict()`` against the ENGINE-sourced
  ``action_native`` (``v5_trace.py:481`` ``action_native =
  legal[legal_index].to_dict()`` where ``legal = engine.get_legal_actions_raw``
  — the engine's own ``BaseAction``, INDEPENDENT of ``decode_action``).

  The engine (``core/engine.py:1260-1266``) emits warrior ``PlayCardAction`` ONLY
  at ``position=len(player.board)`` — i.e. append_only placement — so the
  ``append_only`` mask candidate set corresponds EXACTLY to the engine's emitted
  legal actions. A correct ``decode_action`` at an append_only-masked candidate
  reproduces the engine ``BaseAction``'s ``to_dict()``, so a matching candidate
  proves codec-vs-engine PARITY and a non-match proves a REGRESSION. This makes
  the round-trip assertion
  ``decode_action(pre_state, actor, resolved_tcode).to_dict() == action_native``
  a TRUE source-vs-source check (codec vs engine), NOT a self-referential
  codec-vs-codec check (Block -1/0 lesson — see ``test_train_v2_offline_bridge.py``
  which sources ``action_native`` from ``decode_action`` itself, making its
  round-trip decode_action-vs-decode_action and unable to detect a
  codec-vs-engine regression; ``tests/test_bc_dataset.py`` forks a NEW helper
  that sources ``action_native = legal_raw[legal_index].to_dict()`` — the
  engine's ``BaseAction``, same source as ``v5_trace.py:481``).

decision_source filter (verifier finding 4b): the pilot deploys a placeholder
  BOT against humans, so recorded v5 traces contain BOTH human actions
  (``decision_source='human'``) AND placeholder-bot actions
  (``decision_source in {'bot','rl','llm'}`` — ``match_runner.py:360``
  ``decision_source = "rl" if is_p1_rl else "bot"``, carried at
  ``v5_trace.py:496``). The BC target is HUMAN actions only (D3).
  ``offline_dataset_loader``'s ``OfflineTransition.meta`` is EXTENDED
  ADDITIVELY to include ``decision_source`` (the loader is NOT frozen — the
  additive meta extension does NOT touch ``classic_*`` or ``v5_trace.py``;
  the 6 offline-bridge tests still pass). ``build_bc_dataset`` requires both
  ``decision_source=='human'`` and ``accepted is True`` before emitting
  ``BCTransition``; rejected attempts remain audit rows only.

No import of ``v5_trace`` recorder code (data-contract READ-ONLY — avoids
prod-rlhf coupling); frozen-classic guard held (``classic_actions_v1`` /
``classic_rl_env`` / ``v5_trace.py`` NOT modified); ``core/state.py`` NOT
modified; no TrainV3.5 import into prod.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from ai.train_v2.classic_actions_v1 import (
    build_action_mask,
    decode_action,
    encode_action_features,
)
from ai.train_v2.offline_dataset_loader import (
    iter_offline_transitions,
    reconstruct_gamestate,
)
from core.state import GameState

from train_v3.contracts import AssistModeV5, InfoModeV5

logger = logging.getLogger(__name__)

# Terminal action_type set — mirrors offline_dataset_loader._TERMINAL_TYPES
# (surrender/draw/stalemate synthetic rows). Declared locally so bc_dataset does
# not reach into the loader's private symbol, but stays byte-aligned with it.
# These rows carry no 601 target (no real engine action): action_native is None
# (v5_trace.record_terminal writes no BaseAction).
_TERMINAL_ACTION_TYPES = frozenset({"surrender", "draw", "stalemate"})

# placement_mode='append_only' mirrors the engine's legal-action emission:
# core/engine.py:1260-1266 emits warrior PlayCardAction ONLY at
# position=len(player.board). The V5 policy scores the 601-candidate space with
# this mask; the resolver AND the BC tuple's legal_mask/action_features use the
# SAME mask so the decoded candidate reproduces the engine BaseAction.to_dict()
# (source-vs-source parity) and the BC tuple is internally consistent
# (legal_mask == action_features nonzero rows == engine legal actions).
_PLACEMENT_MODE = "append_only"

# mana_draw action_type — ManaDrawAction is OUTSIDE the 601 candidate space
# (core/actions.py:76; mana_draw_head_v5.py:4-6). BC targets the parallel
# mana_draw binary head for these rows, not a 601 slot.
_MANA_DRAW_ACTION_TYPE = "mana_draw"


class TcodeResolutionError(ValueError):
    """Raised (in strict mode) when a non-mana_draw, non-terminal row's
    ENGINE-sourced ``action_native`` cannot be matched to any legal
    601-candidate.

    A well-formed trace never hits this. It signals EITHER a
    decode_action-vs-engine regression (the FROZEN codec diverged from the
    engine's ``BaseAction`` emission) OR a snapshot/reconstruction drift (the
    rebuilt ``pre_state`` does not match the state the engine emitted
    ``action_native`` from). When ``strict=False`` (default) the row is skipped
    with a logged warning (production-robust); when ``strict=True`` it
    propagates so the source-vs-source test gate fails fast.
    """


@dataclass
class BCTransition:
    """One behavior-cloning training tuple emitted by ``build_bc_dataset``.

    Fields:
        obs: ``encode_observation_v5`` of the reconstructed ``pre_state`` from
            the actor's perspective (shape ``(7128,)``). Reused from the
            loader's ``OfflineTransition.obs`` (same pre_state, same
            info/assist mode — the loader encodes it once with the
            info_mode/assist_mode passed to ``build_bc_dataset``).
        action_features: ``encode_action_features`` of the reconstructed
            ``pre_state`` with ``placement_mode='append_only'`` and
            ``include_preview=False`` (shape ``(601, 171)``, preview channels
            = 0). REBUILT by BC (NOT reused from the loader) so the feature
            mask matches the engine's append_only legal actions AND this
            transition's ``legal_mask`` (BC loss consistency). The loader's own
            ``action_features`` uses ``placement_mode='full'`` (Block 0 default)
            which emits warrior candidates the engine does not offer; BC
            requires ``'append_only'`` to match the engine.
        target_tcode: the resolved V5 601-candidate id (0..600) for a normal OR
            natural-lethal row (a real engine action); ``None`` for mana_draw
            rows (BC targets the parallel mana_draw head, not a 601 slot) and
            for terminal synthetic rows (surrender/draw/stalemate — no real 601
            action).
        is_mana_draw: ``True`` iff ``action_type=='mana_draw'`` (the human took
            ``ManaDrawAction``, outside the 601 space —
            ``mana_draw_head_v5.py:4-6``).
        mana_draw_legal: ``mana_draw_head_v5.mana_draw_legal_mask`` of the
            reconstructed ``pre_state`` for the actor (reused from the loader —
            parity with the parallel binary head's legal predicate).
        legal_mask: ``build_action_mask(pre_state, actor, verify_mask=False,
            placement_mode='append_only')`` — the ``(601,)`` mask the BC policy
            gates its 601 logits with. Matches ``action_features`` nonzero rows
            (encode_action_features fills exactly the masked rows).
        reward: ``compute_offline_reward`` mirror of
            ``classic_rl_env._compute_reward`` (reused from the loader).
        terminal: ``True`` iff the loader classified the row terminal
            (surrender synthetic OR natural-lethal ``post_state.status``).
        meta: provenance dict (carries ``decision_source``, ``action_type``,
            ``status``, ``battle_id``, ``seq``, ``actor_user_id``,
            ``turn_number``).
    """

    obs: np.ndarray
    action_features: np.ndarray
    target_tcode: Optional[int]
    is_mana_draw: bool
    mana_draw_legal: bool
    legal_mask: np.ndarray
    reward: float
    terminal: bool
    meta: Dict[str, Any] = field(default_factory=dict)


def resolve_v5_tcode(
    pre_state: GameState,
    actor: int,
    action_native: Optional[Dict[str, Any]],
    *,
    mask: Optional[np.ndarray] = None,
    strict: bool = False,
) -> Optional[int]:
    """Resolve the V5 601-tcode for an ENGINE-sourced ``action_native``.

    Source-vs-source (Block -1/0 lesson): ``action_native`` MUST be the engine's
    own ``BaseAction.to_dict()`` (``v5_trace.py:481``
    ``legal[legal_index].to_dict()`` where
    ``legal = engine.get_legal_actions_raw(...)``), NOT a
    ``decode_action(...).to_dict()`` output. The caller (``build_bc_dataset``)
    guarantees engine sourcing by reading ``OfflineTransition.action_native``
    (populated by the loader from the recorded row's ENGINE-sourced field);
    tests guarantee it by forking a helper that sources
    ``action_native = legal_raw[legal_index].to_dict()`` (the engine's
    ``BaseAction``, same source as ``v5_trace.py:481``) — NOT the legacy
    ``test_train_v2_offline_bridge.py:_write_real_trace`` which sources it from
    ``decode_action`` (self-referential).

    This function decodes each legal 601-candidate via the FROZEN
    ``decode_action`` (``classic_actions_v1.py:70``) and value-equality matches
    ``.to_dict()`` against ``action_native``. The mask is built with
    ``placement_mode='append_only'`` (mirroring ``core/engine.py:1260``
    ``position=len(player.board)``), so the candidate set corresponds EXACTLY to
    the engine's emitted legal actions: a correct ``decode_action`` reproduces
    the engine ``BaseAction``'s ``to_dict()``, so a matching candidate proves
    codec-vs-engine PARITY and a non-match proves a REGRESSION.

    Args:
        pre_state: the reconstructed ``GameState`` (from the v5_trace snapshot
            via ``reconstruct_gamestate``).
        actor: the acting player's ``user_id``.
        action_native: the ENGINE-sourced action dict (engine ``BaseAction``
            ``to_dict()``). ``None`` for terminal synthetic rows (caller does
            not call this for those — handled by the mana_draw/terminal
            branches in ``build_bc_dataset``); returns ``None`` defensively.
        mask: optional pre-built ``append_only`` legal mask (avoids a rebuild
            when the caller already built it for ``BCTransition.legal_mask``).
        strict: ``False`` (default) -> log a warning + return ``None`` on no-match
            (production-robust: ``build_bc_dataset`` skips the row);
            ``True`` -> raise ``TcodeResolutionError`` (test fail-fast on a
            codec-vs-engine regression).

    Returns:
        The matching candidate_id (0..600), or ``None`` if no candidate matches
        (shouldn't happen for well-formed traces).
    """
    if action_native is None:
        return None
    if mask is None:
        mask = build_action_mask(
            pre_state, actor, verify_mask=False, placement_mode=_PLACEMENT_MODE,
        )
    legal_ids = np.flatnonzero(mask == 1.0)
    for cid in legal_ids:
        decoded = decode_action(pre_state, actor, int(cid))
        if decoded is None:
            # decode_action returns None for out-of-range hand/board indices;
            # the append_only mask never sets those bits, so this is defensive.
            continue
        if decoded.to_dict() == action_native:
            return int(cid)
    if strict:
        raise TcodeResolutionError(
            f"no 601-candidate decodes to action_native={action_native!r} "
            f"(actor={actor}, legal_ids_count={int(legal_ids.size)}); "
            f"this signals a decode_action-vs-engine regression or "
            f"snapshot/reconstruction drift"
        )
    logger.warning(
        "resolve_v5_tcode: no 601-candidate match for action_native=%r "
        "(actor=%s, legal_ids_count=%d) — skipping row",
        action_native, actor, int(legal_ids.size),
    )
    return None


def build_bc_dataset(
    group_dir: "str | Path",
    *,
    info_mode: Optional[InfoModeV5] = None,
    assist_mode: Optional[AssistModeV5] = None,
    max_battles: Optional[int] = None,
    strict: bool = False,
) -> Iterator[BCTransition]:
    """Yield ``BCTransition`` tuples for HUMAN actions across recorded v5 traces.

    Consumes ``offline_dataset_loader.iter_offline_transitions`` (PUBLIC API)
    and filters to ``decision_source=='human'`` (verifier finding 4b — the
    binding BC gate; bot/rl/llm rows are EXCLUDED). For each human row:

      * mana_draw row (``action_type=='mana_draw'``) -> ``target_tcode=None``,
        ``is_mana_draw=True`` (BC targets the parallel mana_draw head, not a 601
        slot; ``ManaDrawAction`` is outside the 601 space,
        ``mana_draw_head_v5.py:4-6``).
      * terminal synthetic row (``action_type in {surrender,draw,stalemate}``)
        -> ``target_tcode=None``, ``is_mana_draw=False`` (no real 601 action;
        ``action_native`` is ``None`` for these rows).
      * normal OR natural-lethal row (a real engine action; the loader marks
        natural-lethal ``terminal=True`` via ``post_state.status``) ->
        ``target_tcode = resolve_v5_tcode(pre_state, actor, action_native)``
        (the ENGINE-sourced action dict carried by the loader).

    ``legal_mask`` + ``action_features`` are REBUILT with
    ``placement_mode='append_only'`` (matching the engine's legal-action
    emission, ``core/engine.py:1260``) so the BC tuple is internally consistent
    (``legal_mask`` == ``action_features`` nonzero rows == engine legal
    actions). ``obs`` / ``reward`` / ``terminal`` / ``mana_draw_legal`` are
    reused from the loader (same pre_state, same encoding).

    Args:
        group_dir: recorded-traces root containing ``manifest.json`` (the
            loader iterates its ``battles_results``).
        info_mode / assist_mode: V5 visibility/assist flags forwarded to the
            loader (drives ``encode_observation_v5``); default
            ``InfoModeV5()`` / ``AssistModeV5()``.
        max_battles: forwarded to the loader (caps the number of battles).
        strict: ``False`` (default) skips a row whose ``action_native`` fails to
            resolve (logged warning) — production-robust; ``True`` raises
            ``TcodeResolutionError`` so the source-vs-source test gate fails
            fast on a codec-vs-engine regression.

    Yields:
        ``BCTransition`` for each HUMAN row (mana_draw / normal / natural-lethal
        / terminal-synthetic). Unresolvable rows are skipped (strict=False) or
        raise (strict=True).
    """
    info_mode = info_mode or InfoModeV5()
    assist_mode = assist_mode or AssistModeV5()
    for t in iter_offline_transitions(
        group_dir,
        info_mode=info_mode,
        assist_mode=assist_mode,
        max_battles=max_battles,
    ):
        # Human-only BC gate. Rejected UI/MCP attempts remain in the canonical
        # trace for auditability but did not change the environment and must
        # never become policy targets. Missing/non-bool accepted is rejected
        # too: provenance has to prove successful execution.
        if t.meta.get("decision_source") != "human":
            continue
        if t.meta.get("accepted") is not True:
            continue
        # The loader carries the raw pre_state snapshot (additive Block-A
        # field) so BC can reconstruct the GameState for the append_only mask +
        # tcode resolution + rebuilt action_features. Defensive skip if missing
        # (a well-formed loader yield always carries it).
        if t.pre_state_snapshot is None:
            continue
        pre_state = reconstruct_gamestate(t.pre_state_snapshot)
        actor = t.meta.get("actor_user_id")
        action_type = t.meta.get("action_type")
        is_mana_draw = action_type == _MANA_DRAW_ACTION_TYPE

        # Build the append_only legal mask ONCE; reuse for tcode resolution AND
        # action_features encoding (single mask build — DRY + consistent).
        legal_mask = build_action_mask(
            pre_state, actor, verify_mask=False, placement_mode=_PLACEMENT_MODE,
        )

        if is_mana_draw:
            target_tcode: Optional[int] = None
        elif action_type in _TERMINAL_ACTION_TYPES:
            # Surrender/draw/stalemate synthetic row: no real 601 action
            # (action_native is None). terminal flag comes from the loader.
            target_tcode = None
        else:
            # Normal OR natural-lethal real engine action: resolve the V5
            # 601-tcode against the ENGINE-sourced action_native. The mask is
            # passed so resolve_v5_tcode does not rebuild it.
            target_tcode = resolve_v5_tcode(
                pre_state, actor, t.action_native, mask=legal_mask, strict=strict,
            )
            if target_tcode is None:
                # Non-strict: resolve_v5_tcode already logged + returned None;
                # skip the row (production-robust). Strict: it already raised.
                continue

        # Rebuild action_features with the SAME append_only mask so the BC
        # tuple is internally consistent (legal_mask == action_features nonzero
        # rows). include_preview=False skips the deep-copy preview simulation
        # (preview channels = 0) — parity with the loader's action_features but
        # with the engine-matching append_only placement.
        action_features = encode_action_features(
            pre_state, actor, mask=legal_mask, include_preview=False,
        )

        yield BCTransition(
            obs=t.obs,
            action_features=action_features,
            target_tcode=target_tcode,
            is_mana_draw=is_mana_draw,
            mana_draw_legal=t.mana_draw_legal,
            legal_mask=legal_mask,
            reward=t.reward,
            terminal=t.terminal,
            meta=t.meta,
        )


def load_bc_dataset(
    group_dir: "str | Path",
    *,
    info_mode: Optional[InfoModeV5] = None,
    assist_mode: Optional[AssistModeV5] = None,
    max_battles: Optional[int] = None,
    strict: bool = False,
) -> list[BCTransition]:
    """Materialize the full BC dataset as a list (eager).

    For streaming/lazy consumption use ``build_bc_dataset`` directly.
    """
    return list(
        build_bc_dataset(
            group_dir,
            info_mode=info_mode,
            assist_mode=assist_mode,
            max_battles=max_battles,
            strict=strict,
        )
    )


__all__ = [
    "BCTransition",
    "TcodeResolutionError",
    "resolve_v5_tcode",
    "build_bc_dataset",
    "load_bc_dataset",
]
