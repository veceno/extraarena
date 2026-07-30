"""Offline-bridge: recorded v5/ traces -> AWAC/CRR offline-PPO replay tuples.

Block 0 component 5 (spec §6.185 / BLOCK0_FOUNDATION_PLAN.md component 5).
This is the D1/D9 offline-bridge that the V5 audit (memory
rlhf-training-data-v5-audit) identified as missing: V5 (``ai/train_v2``)
is online-PPO-only with NO offline jsonl consumer; this loader feeds
recorded ``rlhf_v5_storage_v1`` traces into offline-PPO training as
``(obs, action_features, action_id, reward, next_obs, terminal,
mana_draw_legal)`` replay tuples.

DATA-CONTRACT DEPENDENCY (READ JSON, NOT CODE IMPORT).
The trace schema is authored by ``rlhf_env/components/v5_trace.py``
(``V5TraceRecorder``) + ``rlhf_env/components/arena_engine.py``
(``RlhfBattleEngine._snapshot_card``). This loader READS the emitted
``actions.jsonl`` / ``meta.json`` / ``manifest.json`` files; it does NOT
import ``v5_trace`` recorder code into the loader path (avoids
prod-rlhf coupling — frozen-classic guard). The deserializer below is
the NEW snapshot->GameState builder (no such helper existed in the repo;
grep-confirmed).

Three pieces:

  (a) ``reconstruct_gamestate(snapshot)`` — snapshot->GameState
      DESERIALIZER. Rebuilds ``core.state`` dataclasses from the v5_trace
      snapshot dict (``_snapshot_state``/``_snapshot_player``/
      ``arena_engine._snapshot_card`` schema). ``core.state`` dataclasses
      are plain ``@dataclass`` with NO ``to_dict``/``from_dict``/``__getstate__``
      (core/state.py:78-184) — so the deserializer lives here, in the bridge.

  (b) ``compute_offline_reward(...)`` — REWARD MIRROR of
      ``classic_rl_env._compute_reward`` (classic_rl_env.py:383-421),
      byte-for-byte. ``classic_rl_env.py`` is NOT modified (frozen-classic
      guard): the bridge MIRRORS the formula, does not edit it.

  (c) ``load_offline_dataset(group_dir, ...)`` — iterate ``manifest.json``
      battle index; for each battle with ``v5_trace_ok`` True AND
      ``meta.status`` terminal (skip orphans = ``ongoing``/missing); for
      each ``actions.jsonl`` row reconstruct ``pre_state`` -> obs +
      action_features + reward + ``post_state`` -> next_obs, and emit an
      ``OfflineTransition``.

GAPS HANDLED (per explorer map):
  - No deserialize helper -> bridge writes one (here).
  - Snapshot OMITS ``pending_mana_drain_by_player`` /
    ``sudden_death_turns_by_player`` /
    ``sudden_death_last_applied_turn_by_player``
    (v5_trace._snapshot_state vs golden_trace._state_payload) — left at
    default ``{}``; HARMLESS for ``encode_observation_v5`` which reads only
    p1/p2/turn_number/current_turn_owner_id/history (obs_v5.py:43-67).
  - ``CardInstance.base_*`` missing in snapshot (arena_engine._snapshot_card
    DROPS base_*) — defaulted to ``None``; ``encode_observation_v5`` /
    ``encode_card_shape_v5`` read only CURRENT ``attack``/``hp``/
    ``max_hp``/``mana_cost``/``mechanics`` (v5_card_shape_v1.py:165-178),
    NOT ``base_*``, so this is harmless. ``ensure_base_snapshot`` is NOT
    called (no engine wrap) so ``base_*`` stay ``None`` — documented.
  - ``meta.p1_deck/p2_deck`` are PRE-SHUFFLE — unused here (each
    ``pre_state`` is self-contained per the continuity invariant
    v5_trace_validate.py:244-386; the loader never needs init decks).
  - ``status`` is enum-name lowercased (``ongoing`` at init -> overwritten
    by finalize with ``p1_win``/``p2_win``/``draw``/``stalemate``); the
    loader filters on ``meta.status`` (authoritative for surrender rows
    where ``state.status`` stays ``ongoing`` — mark_surrender does not
    mutate it, v5_trace.py:283-285).
  - Terminal rows discriminated by ``action_type`` in
    ``_TERMINAL_TYPES`` (v5_trace_validate.py:48); natural-lethal rows
    (``action_type=attack/play_card`` with terminal ``post_state.status``)
    are ALSO terminal (RL ``done`` semantic, mirroring classic_rl_env
    ``terminated = st.status != ONGOING``).

ACTION REPRESENTATION CHOICE (documented, NOT a Block-0 binding gate):
  ``OfflineTransition.action_tcode_or_index`` stores the RECORDED
  ``legal_action_index`` (index into ``get_legal_actions_raw`` 0..N-1,
  v5_trace.py:471-475) — directly present in the action row, requires NO
  engine re-run and NO fragile tcode-layout reconstruction. The V5
  ``tcode`` (0..600, classic_actions_v1 codec) computation from
  ``action_native`` + ``pre_state`` is deferred to a later block (the
  v5_trace docstring v5_trace.py:23-28 describes it as a one-shot bridge
  computation; Block 0 explicitly does NOT bind action-identity
  correctness — binding gates are obs round-trip + reward mirror +
  orphan-skip + surrender-terminal). For terminal rows
  ``legal_action_index`` is ``None``. ``action_features`` (601,171) is
  produced via ``encode_action_features(reconstructed, actor,
  verify_mask=False, include_preview=False)`` — ``build_action_mask``
  mirrors ``get_legal_actions`` from state fields directly
  (classic_actions_v1.py:188-217, no engine call) and
  ``include_preview=False`` skips the deep-copy preview simulation
  (preview channels = 0); this needs NO ``ArenaEnvironment`` wrap.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID, uuid4

import numpy as np

from core.state import (
    ACTION_HISTORY_MAXLEN,
    V5_HISTORY_EVENTS,
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
    ReplacementStatus,
)

from ai.train_v2.classic_actions_v1 import encode_action_features

from train_v3.contracts import AssistModeV5, InfoModeV5
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask
from train_v3.obs_v5 import encode_observation_v5

logger = logging.getLogger(__name__)

# Terminal action_type set (v5_trace_validate.py:48) — surrender/draw/stalemate
# synthetic rows. Discriminator is action_type (authoritative marker), NOT
# empty legal_actions (a lost-label row has empty legal_actions but is NOT
# terminal).
_TERMINAL_TYPES = {"surrender", "draw", "stalemate"}

# Terminal meta.status / state.status values (lowercased core enums +
# match-runner-only 'stalemate' which has no GameStatus member).
_TERMINAL_STATUSES = {"p1_win", "p2_win", "draw", "stalemate"}


def _history_events_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the authoritative V5 tape, with a guarded legacy-trace fallback."""

    if "v5_history_events" in snapshot:
        raw = snapshot.get("v5_history_events") or []
    else:
        # Pre-contract fixtures/traces sometimes embedded rich events in the
        # native history field. Never feed ordinary native action payloads to
        # the temporal encoder.
        raw = [
            event
            for event in (snapshot.get("history") or [])
            if isinstance(event, dict) and "action_type" in event
        ]
    return [event for event in raw if isinstance(event, dict)]


# ---------------------------------------------------------------------------
# OfflineTransition
# ---------------------------------------------------------------------------


@dataclass
class OfflineTransition:
    """One (s, a, r, s', done) replay tuple from a recorded v5 action row.

    Fields:
        obs: ``encode_observation_v5`` of the reconstructed ``pre_state`` from
            the actor's perspective (shape ``(7128,)``).
        action_features: ``encode_action_features`` of the reconstructed
            ``pre_state`` (shape ``(601, 171)``, preview channels = 0).
        action_tcode_or_index: the RECORDED ``legal_action_index`` (index into
            ``get_legal_actions_raw`` 0..N-1); ``None`` for terminal rows
            (surrender/draw/stalemate). The V5 601-candidate ``tcode`` is
            deferred (see ACTION REPRESENTATION CHOICE above).
        reward: ``compute_offline_reward`` mirror of
            ``classic_rl_env._compute_reward``.
        next_obs: ``encode_observation_v5`` of the reconstructed ``post_state``
            from the actor's perspective.
        terminal: ``True`` iff the resolved row status is terminal (covers
            both surrender rows and natural-lethal rows whose
            ``post_state.status`` is ``p1_win``/``p2_win``/``draw``).
        mana_draw_legal: ``mana_draw_head_v5.mana_draw_legal_mask`` of the
            reconstructed ``pre_state`` for the actor (parity with the
            parallel binary head's legal predicate).
        meta: provenance dict ``{battle_id, seq, action_type, actor_user_id,
            actor_player, turn_number, status, decision_source, accepted, ...}``.
            ``accepted`` is carried verbatim so every policy consumer can use
            the strict ``is True`` gate; rejected attempts remain available to
            audit/timing consumers without becoming action targets.
        action_native: the ENGINE-sourced action dict (``legal[legal_index]
            .to_dict()`` from ``v5_trace.py:481`` — the engine's own
            ``BaseAction``, INDEPENDENT of ``decode_action``). ``None`` for
            terminal synthetic rows (surrender/draw/stalemate) and any
            unfinalized row. Additive Block-A field: downstream BC
            (``train_v3.bc_dataset``) resolves the V5 601-tcode by
            value-matching ``decode_action(candidate).to_dict()`` against
            this ENGINE oracle — a true source-vs-source check (codec vs
            engine), NOT a self-referential codec-vs-codec check. Carrying
            it here keeps a single source of truth (the loader reads the
            row once; BC does not re-read ``actions.jsonl``).
        pre_state_snapshot: the raw ``pre_state`` snapshot dict (v5_trace
            ``_snapshot_state`` schema). Additive Block-A field: downstream
            BC reconstructs the ``GameState`` via ``reconstruct_gamestate``
            to build the append_only legal mask + resolve the 601-tcode +
            (Block-C C0) the loader's own ``action_features`` now uses
            ``placement_mode='append_only'`` to match the engine's
            legal-action emission (``core/engine.py:1260``
            ``position=len(player.board)``), so BC's append_only rebuild
            and the loader's field agree on one engine-faithful source. Carried
            as the JSON-friendly snapshot (not the mutable ``GameState``) so
            the transition stays serializable-ish and BC reconstruction is
            spec-literal (``reconstruct_gamestate(snapshot)``).
        post_state_snapshot: the matching raw ``post_state`` snapshot.  The
            Phase-C replay bridge uses it to construct a human-perspective
            macro transition across intervening bot actions, including a
            terminal loss that happens on the bot's turn.
    """

    obs: np.ndarray
    action_features: np.ndarray
    action_tcode_or_index: Optional[int]
    reward: float
    next_obs: np.ndarray
    terminal: bool
    mana_draw_legal: bool
    meta: Dict[str, Any] = field(default_factory=dict)
    # Additive Block-A fields (defaulted -> existing 6 offline-bridge tests
    # untouched; they only read the legacy fields above).
    action_native: Optional[Dict[str, Any]] = None
    pre_state_snapshot: Optional[Dict[str, Any]] = None
    post_state_snapshot: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# (a) snapshot -> GameState DESERIALIZER
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _parse_card_type(value: Any) -> CardType:
    """Map the snapshot ``type``/``card_type`` str -> ``CardType`` enum.

    ``arena_engine._snapshot_card`` writes ``getattr(card.card_type, "value",
    str(card.card_type))`` (arena_engine.py:922-923) — the enum value str
    ("hero"/"warrior"/"potion"). Reconstruct via ``CardType(value)``.
    """
    if value is None:
        return CardType.WARRIOR
    if isinstance(value, CardType):
        return value
    try:
        return CardType(str(value))
    except (ValueError, KeyError):
        # Defensive: unknown type string -> WARRIOR (never HERO; HERO is only
        # for the player hero slot which is handled explicitly).
        return CardType.WARRIOR


def _parse_replacement_status(value: Any) -> ReplacementStatus:
    """Map snapshot ``replacement_status`` str -> ``ReplacementStatus``.

    ``v5_trace._snapshot_player`` writes the enum suffix lowercased
    (v5_trace.py:285): ``active``/``afk``/``surrendered``.
    """
    if value is None:
        return ReplacementStatus.ACTIVE
    if isinstance(value, ReplacementStatus):
        return value
    name = str(value).strip().lower()
    try:
        return ReplacementStatus(name)
    except (ValueError, KeyError):
        return ReplacementStatus.ACTIVE


def _parse_game_status(value: Any) -> GameStatus:
    """Map snapshot ``status`` str -> ``GameStatus``.

    ``v5_trace._snapshot_state`` writes ``st.status.name.lower()``
    (v5_trace.py:320) — ``ongoing``/``p1_win``/``p2_win``/``draw``. ``st.status``
    is a ``GameStatus`` enum member (core/state.py:64-69) so ``stalemate`` is
    NEVER written into a STATE snapshot (only into ``meta.status`` by
    match_runner); reconstruct_gamestate therefore only ever sees the four
    enum values. ``stalemate`` falls back to ``ONGOING`` defensively.
    """
    if value is None:
        return GameStatus.ONGOING
    if isinstance(value, GameStatus):
        return value
    name = str(value).strip().lower()
    try:
        return GameStatus(name)
    except (ValueError, KeyError):
        # 'stalemate' has no GameStatus member (only in meta.status). State
        # snapshots never carry it, but fall back to ONGOING defensively.
        return GameStatus.ONGOING


def _parse_instance_id(value: Any) -> UUID:
    if value is None:
        return uuid4()
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return uuid4()


def reconstruct_card(snap: Optional[Dict[str, Any]]) -> Optional[CardInstance]:
    """Rebuild a ``CardInstance`` (core/state.py:78-130) from a v5 card snapshot.

    Mirrors ``arena_engine._snapshot_card`` (arena_engine.py:914-936) field
    set. The snapshot carries both current runtime stats (``atk``/``hp``/
    ``max_hp``/``is_ready``/``is_frozen``) and the ``card_params`` block; the
    deserializer reads the top-level fields (authoritative) with ``card_params``
    fallbacks. ``base_*`` are NOT in the snapshot (DROPPED by
    ``arena_engine._snapshot_card``) -> defaulted to ``None``; the obs encoder
    reads only CURRENT values (v5_card_shape_v1.py:165-178), so this is
    harmless. ``skip_count``/``instant_kill_used``/``simplified_levelup``
    (also dropped) default to their dataclass defaults.
    """
    if snap is None:
        return None
    card_id = _coerce_int(snap.get("card_id", snap.get("id")))
    # arena_engine writes both 'atk' (current attack) and 'attack'
    # (card_params['attack']); they are equal (= current attack). Prefer
    # 'atk' (the runtime value), fall back to 'attack' then 'card_params'.
    attack = snap.get("atk")
    if attack is None:
        attack = snap.get("attack")
        if attack is None and isinstance(snap.get("card_params"), dict):
            attack = snap["card_params"].get("attack")
    # mana_cost: 'mana' (getattr card.mana_cost) == 'mana_cost'
    # (card_params['mana_cost']); prefer 'mana'.
    mana_cost = snap.get("mana")
    if mana_cost is None:
        mana_cost = snap.get("mana_cost")
        if mana_cost is None and isinstance(snap.get("card_params"), dict):
            mana_cost = snap["card_params"].get("mana_cost")
    mechanics = list(snap.get("mechanics") or [])
    return CardInstance(
        instance_id=_parse_instance_id(snap.get("instance_id")),
        card_id=card_id,
        name=_coerce_str(snap.get("name")),
        card_type=_parse_card_type(snap.get("card_type", snap.get("type"))),
        rarity=_coerce_str(snap.get("rarity"), "common"),
        mana_cost=_coerce_int(mana_cost),
        attack=_coerce_int(attack),
        hp=_coerce_int(snap.get("hp")),
        max_hp=_coerce_int(snap.get("max_hp")),
        mechanics=mechanics,
        is_ready=_coerce_bool(snap.get("is_ready")),
        is_frozen=_coerce_bool(snap.get("is_frozen")),
        level=_coerce_int(snap.get("level"), 1),
        # DROPPED by arena_engine._snapshot_card -> defaults (obs_v5 does not
        # read base_*; v5_card_shape_v1 reads only current stats).
        base_attack=None,
        base_hp=None,
        base_max_hp=None,
        base_mana_cost=None,
        base_mechanics=None,
        instant_kill_used=False,
        skip_count=0,
        simplified_levelup=False,
    )


def reconstruct_player(snap: Dict[str, Any]) -> PlayerState:
    """Rebuild a ``PlayerState`` (core/state.py:133-157) from a v5 player
    snapshot.

    Mirrors ``v5_trace._snapshot_player`` (v5_trace.py:274-296) field set.
    """
    hero = reconstruct_card(snap.get("hero"))
    if hero is None:
        # Defensive: a player always has a hero; synthesize a placeholder so
        # downstream encoders (which read hero.hp/attack) do not crash on a
        # corrupted snapshot. The real recorder never writes a null hero.
        hero = CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name="Hero",
            card_type=CardType.HERO,
            hp=0,
            max_hp=0,
        )
    return PlayerState(
        user_id=_coerce_int(snap.get("user_id")),
        is_bot=_coerce_bool(snap.get("is_bot")),
        replacement_status=_parse_replacement_status(snap.get("replacement_status")),
        hero=hero,
        mana=_coerce_int(snap.get("mana")),
        max_mana=_coerce_int(snap.get("max_mana")),
        mana_draw_count_this_turn=_coerce_int(snap.get("mana_draw_count_this_turn")),
        hand=[reconstruct_card(c) for c in (snap.get("hand") or [])],
        board=[reconstruct_card(c) for c in (snap.get("board") or [])],
        deck=[reconstruct_card(c) for c in (snap.get("deck") or [])],
        graveyard=[reconstruct_card(c) for c in (snap.get("graveyard") or [])],
        trophies=_coerce_int(snap.get("trophies")),
        # surrender_processed is NOT in the snapshot; default False.
        surrender_processed=False,
    )


def reconstruct_gamestate(snapshot: Dict[str, Any]) -> GameState:
    """Rebuild a ``GameState`` (core/state.py:159-184) from a v5 state snapshot.

    Mirrors ``v5_trace._snapshot_state`` (v5_trace.py:298-327) field set.

    ``action_history`` is a ``deque(maxlen=100)`` of ``tuple[str, str]``
    (core/state.py:177-179); the snapshot serializes it as
    ``list[list[str, str]]`` (v5_trace.py:306) -> rebuilt as tuples.

    ``pending_mana_drain_by_player`` /
    ``sudden_death_turns_by_player`` /
    ``sudden_death_last_applied_turn_by_player`` are OMITTED by
    ``v5_trace._snapshot_state`` (compare golden_trace._state_payload which
    includes all three) -> left at default ``{}``. HARMLESS for
    ``encode_observation_v5`` (reads only p1/p2/turn_number/
    current_turn_owner_id/v5_history_events, obs_v5.py:43-67); flagged here so any
    future re-stepping path is aware.

    ``classic_params`` / ``arena_engine`` are ``None`` on the rebuilt state:
    ``encode_observation_v5`` does not read them, and
    ``encode_action_features(verify_mask=False, include_preview=False)``
    builds the mask by mirroring ``get_legal_actions`` from state fields
    (classic_actions_v1.py:188-217) without an engine.
    """
    p1 = reconstruct_player(snapshot["p1"])
    p2 = reconstruct_player(snapshot["p2"])
    # action_history: list[list[str,str]] -> deque(maxlen=N) of tuple[str,str]
    raw_ah = snapshot.get("action_history") or []
    action_history = deque(
        (tuple(t) for t in raw_ah),
        maxlen=ACTION_HISTORY_MAXLEN,
    )
    v5_history_events = deque(
        _history_events_from_snapshot(snapshot),
        maxlen=V5_HISTORY_EVENTS,
    )
    return GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=_coerce_int(snapshot.get("current_turn_owner_id")),
        turn_number=_coerce_int(snapshot.get("turn_number"), 1),
        history=list(snapshot.get("history") or []),
        action_history=action_history,
        v5_history_events=v5_history_events,
        status=_parse_game_status(snapshot.get("status")),
        pending_card_feedback_events=list(snapshot.get("pending_card_feedback_events") or []),
        # OMITTED by v5_trace._snapshot_state -> defaults (safe for obs_v5).
        pending_mana_drain_by_player={},
        sudden_death_turns_by_player={},
        sudden_death_last_applied_turn_by_player={},
        # classic_params / arena_engine -> None (no engine wrap; obs_v5 +
        # action_mask(verify_mask=False) do not need them).
        classic_params=None,
        arena_engine=None,
    )


# ---------------------------------------------------------------------------
# (b) REWARD MIRROR of classic_rl_env._compute_reward
# ---------------------------------------------------------------------------


def reward_view_from_snapshot(
    state_snapshot: Dict[str, Any], actor_player: int
) -> Dict[str, Any]:
    """Build a classic_rl_env._snapshot-shape reward dict from a v5 state
    snapshot.

    Mirrors ``classic_rl_env._snapshot`` (classic_rl_env.py:370-381) field
    set EXACTLY:

        my_hero_hp      = me.hero.hp
        enemy_hero_hp   = enemy.hero.hp
        my_board_hp     = [u.hp for u in me.board]
        enemy_board_hp  = [u.hp for u in enemy.board]
        my_mana         = me.mana
        enemy_mana      = enemy.mana
        opponent_id     = enemy.user_id

    plus ``p1_user_id``/``p2_user_id`` (from the snapshot) so
    ``compute_offline_reward`` can mirror the terminal ``actor_id ==
    st.p1.user_id`` check (classic_rl_env.py:388-391) without a live state.

    ``actor_player`` (1|2) selects me/enemy: ``1`` -> p1 is me, ``2`` -> p2 is
    me (v5_trace.py:489 ``actor_player = 1 if user_id == st.p1.user_id else 2``).
    """
    p1 = state_snapshot.get("p1") or {}
    p2 = state_snapshot.get("p2") or {}
    if actor_player == 1:
        me, enemy = p1, p2
    else:
        me, enemy = p2, p1
    me_hero = me.get("hero") or {}
    enemy_hero = enemy.get("hero") or {}
    return {
        "my_hero_hp": _coerce_int(me_hero.get("hp")),
        "enemy_hero_hp": _coerce_int(enemy_hero.get("hp")),
        "my_board_hp": [_coerce_int(c.get("hp")) for c in (me.get("board") or [])],
        "enemy_board_hp": [_coerce_int(c.get("hp")) for c in (enemy.get("board") or [])],
        "my_mana": _coerce_int(me.get("mana")),
        "enemy_mana": _coerce_int(enemy.get("mana")),
        "opponent_id": _coerce_int(enemy.get("user_id")),
        # p1/p2 user ids from the snapshot (immutable per battle) so the
        # terminal actor-win check mirrors classic_rl_env exactly.
        "p1_user_id": _coerce_int(p1.get("user_id")),
        "p2_user_id": _coerce_int(p2.get("user_id")),
    }


def compute_offline_reward(
    actor_id: int,
    pre: Dict[str, Any],
    post: Dict[str, Any],
    accepted: Optional[bool],
    status: str,
    *,
    is_mana_draw: bool = False,
) -> float:
    """Mirror ``classic_rl_env._compute_reward`` (classic_rl_env.py:383-421)
    EXACTLY.

    Args:
        actor_id: the acting player's ``user_id`` (``actor_user_id`` from the
            action row).
        pre / post: classic_rl_env._snapshot-shape reward dicts (see
            ``reward_view_from_snapshot``), carrying ``my_hero_hp``/
            ``enemy_hero_hp``/``my_board_hp``/``enemy_board_hp``/``my_mana``/
            ``enemy_mana``/``p1_user_id``/``p2_user_id``.
        accepted: ``True`` for a successfully applied action, ``False`` for an
            invalid/rejected action, ``None`` for an unfinalized row
            (treated as not-accepted -> -0.05, matching ``if not success``).
        status: the resolved row status string (``ongoing``/``p1_win``/
            ``p2_win``/``draw``/``stalemate``). The loader resolves this per
            row: ``post_state.status`` if terminal, else ``meta.status`` for
            ``_TERMINAL_TYPES`` rows, else ``ongoing``.

    Formula (classic_rl_env.py:383-421):
        not success                  -> -0.05
        status == P1_WIN             -> +1.0 if actor_id == p1.user_id else -1.0
        status == P2_WIN             -> +1.0 if actor_id == p2.user_id else -1.0
        status == DRAW               -> 0.0
        else (shaped):
          +0.02 * enemy_hp_delta      if enemy_hp_delta > 0
          -0.01 * own_hp_delta        if own_hp_delta > 0
          +0.03 * enemy_killed        if enemy_killed > 0
          -0.02 * own_killed          if own_killed > 0
          +min(0.02, 0.005*mana_spent) if mana_spent > 0

    ``stalemate`` (no GameStatus member; only in meta.status) is mapped to
    0.0 (no-winner terminal, draw-equivalent) — documented; classic_rl_env
    itself never observes stalemate (no such enum), so this is the faithful
    no-winner terminal reward.
    """
    # classic_rl_env.py:385-386 — invalid action -> -0.05 (BEFORE status
    # checks; mirrors the env's invalid-action early return at :257-262).
    if not accepted:
        return -0.05

    # classic_rl_env.py:388-393 — terminal win/loss/draw. ``actor_id ==
    # p1_user_id`` mirrors ``actor_id == st.p1.user_id`` (the snapshot's
    # p1_user_id == the live st.p1.user_id).
    if status == "p1_win":
        return 1.0 if actor_id == _coerce_int(pre.get("p1_user_id")) else -1.0
    if status == "p2_win":
        return 1.0 if actor_id == _coerce_int(pre.get("p2_user_id")) else -1.0
    if status == "draw":
        return 0.0
    if status == "stalemate":
        # No GameStatus member; no-winner terminal -> 0.0 (documented).
        return 0.0

    # classic_rl_env.py:395-421 — shaped reward.
    reward = 0.0

    enemy_hp_delta = _coerce_int(pre.get("enemy_hero_hp")) - _coerce_int(post.get("enemy_hero_hp"))
    if enemy_hp_delta > 0:
        reward += 0.02 * enemy_hp_delta

    own_hp_delta = _coerce_int(pre.get("my_hero_hp")) - _coerce_int(post.get("my_hero_hp"))
    if own_hp_delta > 0:
        reward -= 0.01 * own_hp_delta

    pre_enemy_count = len(pre.get("enemy_board_hp") or [])
    post_enemy_count = len(post.get("enemy_board_hp") or [])
    enemy_killed = pre_enemy_count - post_enemy_count
    if enemy_killed > 0:
        reward += 0.03 * enemy_killed

    pre_own_count = len(pre.get("my_board_hp") or [])
    post_own_count = len(post.get("my_board_hp") or [])
    own_killed = pre_own_count - post_own_count
    if own_killed > 0:
        reward -= 0.02 * own_killed

    mana_spent = _coerce_int(pre.get("my_mana")) - _coerce_int(post.get("my_mana"))
    # Mana draw spends mana but does not deserve the generic card-play spend
    # bonus. Keep C replay reward byte-aligned with the fixed Rust online path.
    if mana_spent > 0 and not is_mana_draw:
        reward += min(0.02, 0.005 * mana_spent)

    return reward


# ---------------------------------------------------------------------------
# (c) OFFLINE LOADER
# ---------------------------------------------------------------------------


def _resolve_row_status(row: Dict[str, Any], meta_status: Optional[str]) -> str:
    """Resolve the per-row status for reward + terminal classification.

    Mirrors classic_rl_env reading ``st.status`` (post-step):
      - a NATURAL lethal: the last normal action's ``post_state.status`` is
        ``p1_win``/``p2_win``/``draw`` (the engine mutates status on lethal)
        -> use ``post_state.status``.
      - a SURRENDER / draw / stalemate synthetic row: ``post_state.status``
        stays ``ongoing`` (mark_surrender does NOT mutate state.status,
        v5_trace.py:283-285) -> use ``meta.status`` (authoritative terminal).
      - a normal ongoing action -> ``ongoing`` -> shaped reward.
    """
    post = row.get("post_state") or {}
    post_status = post.get("status")
    if post_status in _TERMINAL_STATUSES:
        return str(post_status)
    if row.get("action_type") in _TERMINAL_TYPES:
        return str(meta_status) if meta_status in _TERMINAL_STATUSES else "ongoing"
    return "ongoing"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_battle_actions(group_dir: Path, battle_result: Dict[str, Any]) -> Optional[Path]:
    """Resolve the actions.jsonl path for a battle record. Returns the path
    if it exists, else ``None``."""
    battle_id = battle_result.get("battle_id")
    v5_dir_rel = battle_result.get("v5_dir")
    if v5_dir_rel:
        v5_dir = (group_dir / v5_dir_rel).resolve()
    elif battle_id:
        v5_dir = (group_dir / "battles" / str(battle_id) / "v5").resolve()
    else:
        return None
    actions_path = v5_dir / "actions.jsonl"
    return actions_path if actions_path.exists() else None


def _load_battle_meta(group_dir: Path, battle_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    battle_id = battle_result.get("battle_id")
    v5_meta_rel = battle_result.get("v5_meta_path")
    if v5_meta_rel:
        meta_path = (group_dir / v5_meta_rel).resolve()
    elif battle_result.get("v5_dir"):
        meta_path = (group_dir / battle_result["v5_dir"] / "meta.json").resolve()
    elif battle_id:
        meta_path = (group_dir / "battles" / str(battle_id) / "v5" / "meta.json").resolve()
    else:
        return None
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def iter_offline_transitions(
    group_dir: Path,
    *,
    info_mode: Optional[InfoModeV5] = None,
    assist_mode: Optional[AssistModeV5] = None,
    max_battles: Optional[int] = None,
) -> Iterator[OfflineTransition]:
    """Yield ``OfflineTransition`` replay tuples for every actionable row in
    every terminal, v5-traced battle under ``group_dir``.

    Iterates ``manifest.json`` ``battles_results`` (manifest.py:141-168). For
    each battle with ``v5_trace_ok`` True AND ``meta.status`` in
    ``_TERMINAL_STATUSES`` (orphans = ``ongoing``/missing are SKIPPED), reads
    ``actions.jsonl`` and emits one ``OfflineTransition`` per row whose
    ``pre_state`` and ``post_state`` are both present.

    ``info_mode`` / ``assist_mode`` default to ``InfoModeV5()`` /
    ``AssistModeV5()`` (train_v3.contracts). The trace is omniscient; the
    caller chooses the visibility flags at load time (e.g. an omniscient
    InfoModeV5(enemy_hand_known=True, enemy_deck_known=True) for an oracle
    baseline, or the default self-visible mode for the training policy).
    """
    group_dir = Path(group_dir)
    manifest_path = group_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    info_mode = info_mode or InfoModeV5()
    assist_mode = assist_mode or AssistModeV5()

    battles = manifest.get("battles_results") or []
    emitted_battles = 0
    for battle_result in battles:
        if max_battles is not None and emitted_battles >= max_battles:
            return
        # G4: skip battles whose v5 trace failed/silent.
        if not battle_result.get("v5_trace_ok"):
            continue
        meta = _load_battle_meta(group_dir, battle_result)
        if meta is None:
            continue
        meta_status = meta.get("status")
        # Orphan skip: only terminal battles feed offline training.
        if meta_status not in _TERMINAL_STATUSES:
            continue
        actions_path = _load_battle_actions(group_dir, battle_result)
        if actions_path is None:
            continue
        battle_id = battle_result.get("battle_id")
        try:
            rows = _read_jsonl(actions_path)
        except OSError:
            continue
        emitted_battles += 1
        for row in rows:
            pre_snap = row.get("pre_state")
            post_snap = row.get("post_state")
            # Skip rows where pre_state/post_state missing (unfinalized).
            if pre_snap is None or post_snap is None:
                continue
            actor_user_id = _coerce_int(row.get("actor_user_id"))
            actor_player = _coerce_int(row.get("actor_player"), 1)
            if actor_player not in (1, 2):
                # Defensive: malformed row -> infer from pre_state owner.
                actor_player = 1 if pre_snap.get("p1", {}).get("user_id") == actor_user_id else 2

            resolved_status = _resolve_row_status(row, meta_status)
            terminal = resolved_status in _TERMINAL_STATUSES

            pre_view = reward_view_from_snapshot(pre_snap, actor_player)
            post_view = reward_view_from_snapshot(post_snap, actor_player)
            reward = compute_offline_reward(
                actor_user_id, pre_view, post_view,
                row.get("accepted"), resolved_status,
                is_mana_draw=str(row.get("action_type", "")) == "mana_draw",
            )

            # Reconstruct pre_state -> obs + action_features + mana_draw mask.
            pre_gs = reconstruct_gamestate(pre_snap)
            obs = encode_observation_v5(
                pre_gs, actor_user_id,
                info_mode=info_mode, assist_mode=assist_mode,
                history_events=_history_events_from_snapshot(pre_snap),
            )
            # action_features: (601,171). verify_mask=False builds the mask by
            # mirroring get_legal_actions from state fields (no engine);
            # include_preview=False skips the deep-copy preview simulation
            # (preview channels = 0). No ArenaEnvironment wrap needed.
            # placement_mode='append_only': only emit warrior PlayCard
            # candidates at the engine's sole legal position
            # ``position=len(player.board)`` (core/engine.py:1260). The 'full'
            # default would over-include warriors at non-append positions the
            # engine does NOT offer, breaking the consistency invariant
            # (action_features nonzero rows vs get_legal_actions_raw count).
            # Block-C C0 fix (D-C5): one engine-faithful source for ALL
            # consumers (BC already rebuilds with 'append_only'; the new C3
            # offline-replay path consumes the loader field directly).
            action_features = encode_action_features(
                pre_gs, actor_user_id,
                verify_mask=False, include_preview=False,
                placement_mode='append_only',
            )
            mana_draw_legal = mana_draw_legal_mask(pre_gs, actor_user_id)

            # Reconstruct post_state -> next_obs.
            post_gs = reconstruct_gamestate(post_snap)
            next_obs = encode_observation_v5(
                post_gs, actor_user_id,
                info_mode=info_mode, assist_mode=assist_mode,
                history_events=_history_events_from_snapshot(post_snap),
            )

            yield OfflineTransition(
                obs=obs,
                action_features=action_features,
                action_tcode_or_index=row.get("legal_action_index"),
                reward=reward,
                next_obs=next_obs,
                terminal=terminal,
                mana_draw_legal=mana_draw_legal,
                meta={
                    "battle_id": battle_id,
                    "seq": row.get("seq"),
                    "action_type": row.get("action_type"),
                    "actor_user_id": actor_user_id,
                    "turn_number": row.get("turn_number"),
                    "status": resolved_status,
                    # Additive Block-A: decision_source carried in meta so BC
                    # (train_v3.bc_dataset) can filter to decision_source=='human'
                    # (verifier finding 4b) from the single loader source of
                    # truth, without re-reading actions.jsonl. The raw row
                    # carries it at v5_trace.py:496.
                    "decision_source": row.get("decision_source"),
                    "actor_player": actor_player,
                    "accepted": row.get("accepted"),
                    "human_decision_time_ms": row.get("human_decision_time_ms"),
                    "decision_time_censored": row.get("decision_time_censored"),
                    "decision_censor_reason": row.get("decision_censor_reason"),
                    "control_source": row.get("control_source"),
                },
                # Additive Block-A: ENGINE-sourced action dict (v5_trace.py:481
                # ``legal[legal_index].to_dict()``) for 601-tcode resolution, and
                # the raw pre_state snapshot for BC to reconstruct the GameState
                # (append_only mask + tcode resolution). Both None-safe for
                # terminal synthetic rows (surrender/draw/stalemate).
                action_native=row.get("action_native"),
                pre_state_snapshot=pre_snap,
                post_state_snapshot=post_snap,
            )


def load_offline_dataset(
    group_dir: Path,
    *,
    info_mode: Optional[InfoModeV5] = None,
    assist_mode: Optional[AssistModeV5] = None,
    max_battles: Optional[int] = None,
) -> List[OfflineTransition]:
    """Materialize the full offline dataset as a list (eager).

    For streaming/lazy consumption use ``iter_offline_transitions`` directly.
    """
    return list(
        iter_offline_transitions(
            group_dir,
            info_mode=info_mode,
            assist_mode=assist_mode,
            max_battles=max_battles,
        )
    )


__all__ = [
    "OfflineTransition",
    "reconstruct_card",
    "reconstruct_player",
    "reconstruct_gamestate",
    "reward_view_from_snapshot",
    "compute_offline_reward",
    "iter_offline_transitions",
    "load_offline_dataset",
]
