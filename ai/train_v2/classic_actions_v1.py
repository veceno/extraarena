"""
Fixed candidate-action codec for RL training — no card identity in features.

Action ID layout (601 candidates):
  0:        end turn
  1..544:   play-card  (4 hand_slots × 8 board_positions × 17 target_codes)
  545..600: attack     (7 attacker_slots × 8 target_codes)

Target codes (play-card):
  0 = no target   8 = enemy hero
  1..7 = enemy board slots
  9..15 = own board slots   16 = own hero

Target codes (attack):
  0..6 = enemy board slots   7 = enemy hero

Mask: built by mirroring get_legal_actions, then verified by deep-copy step.

Action features (171):
  [0:64]    source card shape
  [64:128]  target card shape (zeros if no target)
  [128:131] action type one-hot (end_turn, play_card, attack)
  [131:139] relation flags (8)
  [139:142] positional scalars (0.0 = N/A)
  [142:171] preview deltas (29, only for masked actions)
"""
from __future__ import annotations

import copy
import random as rand_mod
import numpy as np

from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from core.engine import ArenaEnvironment
from core.actions import BaseAction, EndTurnAction, PlayCardAction, AttackAction
from core.effects import (
    get_taunt_targets,
    has_taunt,
    is_random_battlecry_damage_card,
    requires_target,
)

from ai.train_v2.classic_card_shape_v1 import CARD_SHAPE_DIM, encode_card_shape

ACTION_VERSION = "classic_actions_v1"
MAX_CANDIDATE_ACTIONS = 601
ACTION_FEATURE_DIM = 171

_NUM_HAND = 4
_NUM_BOARD = 5
_NUM_PLAY_POS = 8
_NUM_PLAY_TARGETS = 17
_NUM_ATTACK_TARGETS = 8

_PLAY_STRIDE = _NUM_PLAY_POS * _NUM_PLAY_TARGETS
_PLAY_BASE = 1
_ATTACK_BASE = _PLAY_BASE + _NUM_HAND * _PLAY_STRIDE


def _get_me_enemy(state: GameState, player_id: int):
    me = state.p1 if state.p1.user_id == player_id else state.p2
    enemy = state.p2 if state.p1.user_id == player_id else state.p1
    return me, enemy


# ============================================================================
# ACTION DECODE
# ============================================================================

def decode_action(state: GameState, player_id: int, action_id: int) -> BaseAction | None:
    me, enemy = _get_me_enemy(state, player_id)

    if action_id == 0:
        return EndTurnAction()

    if 1 <= action_id <= 544:
        flat = action_id - _PLAY_BASE
        pos_idx, tcode = divmod(flat, _NUM_PLAY_TARGETS)
        hand_idx, pos_idx = divmod(pos_idx, _NUM_PLAY_POS)
        card = me.hand[hand_idx] if hand_idx < len(me.hand) else None
        if card is None:
            return None

        target_id = _resolve_target(me, enemy, tcode, is_attack=False)
        return PlayCardAction(
            hand_index=hand_idx,
            target_id=target_id,
            position=pos_idx if card.card_type == CardType.WARRIOR else None,
        )

    if 545 <= action_id <= 600:
        flat = action_id - _ATTACK_BASE
        attacker_idx, tcode = divmod(flat, _NUM_ATTACK_TARGETS)
        if attacker_idx >= len(me.board):
            return None
        target_id = _resolve_target(me, enemy, tcode, is_attack=True)
        return AttackAction(
            attacker_id=str(me.board[attacker_idx].instance_id),
            target_id=target_id,
            target_is_hero=(tcode == 7),
        )

    return None


def _resolve_target(me, enemy, tcode, is_attack):
    if is_attack:
        if 0 <= tcode <= 6 and tcode < len(enemy.board):
            return str(enemy.board[tcode].instance_id)
        if tcode == 7:
            return None
        return None

    if tcode == 0:
        return None
    if tcode == 8:
        return str(enemy.hero.instance_id)
    if tcode == 16:
        return str(me.hero.instance_id)
    if 1 <= tcode <= 7 and tcode - 1 < len(enemy.board):
        return str(enemy.board[tcode - 1].instance_id)
    if 9 <= tcode <= 15 and tcode - 9 < len(me.board):
        return str(me.board[tcode - 9].instance_id)
    return None


def _parse_aura_atk(mechanic: str) -> int | None:
    if not mechanic.startswith("aura_atk_"):
        return None
    value_part = mechanic[9:]
    if "_" in value_part:
        return None
    try:
        return int(value_part)
    except ValueError:
        return None


def _compute_effective_attack(unit: CardInstance, board, hero=None) -> int:
    bonus = 0
    for aura_unit in board:
        if aura_unit.instance_id == unit.instance_id:
            continue
        for mechanic in aura_unit.mechanics:
            val = _parse_aura_atk(mechanic)
            if val is not None:
                bonus += val
    if hero is not None:
        for mechanic in hero.mechanics:
            val = _parse_aura_atk(mechanic)
            if val is not None:
                bonus += val
    return unit.attack + bonus


# ============================================================================
# PLACEMENT MODE
# ============================================================================

_PLACEMENT_FULL = "full"
_PLACEMENT_APPEND_ONLY = "append_only"
_VALID_PLACEMENT_MODES = frozenset({_PLACEMENT_FULL, _PLACEMENT_APPEND_ONLY})


def _apply_placement_mode(mask: np.ndarray, me, placement_mode: str) -> None:
    if placement_mode == _PLACEMENT_FULL:
        return
    if placement_mode == _PLACEMENT_APPEND_ONLY:
        # zero-out all warrior play positions except position == len(me.board)
        for hand_idx in range(min(len(me.hand), _NUM_HAND)):
            card = me.hand[hand_idx]
            if card.card_type != CardType.WARRIOR:
                continue
            expected_pos = len(me.board)
            for pos_idx in range(_NUM_PLAY_POS):
                if pos_idx == expected_pos:
                    continue
                base = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos_idx * _NUM_PLAY_TARGETS
                for tcode in range(_NUM_PLAY_TARGETS):
                    mask[base + tcode] = 0.0
        return


# ============================================================================
# ACTION MASK
# ============================================================================

def build_action_mask(
    state: GameState,
    player_id: int,
    *,
    verify_mask: bool = True,
    placement_mode: str = "full",
) -> np.ndarray:
    if placement_mode not in _VALID_PLACEMENT_MODES:
        raise ValueError(f"placement_mode must be one of {_VALID_PLACEMENT_MODES}, got {placement_mode}")

    mask = np.zeros(MAX_CANDIDATE_ACTIONS, dtype=np.float32)

    if state.status != GameStatus.ONGOING:
        return mask
    if state.current_turn_owner_id != player_id:
        return mask

    me, enemy = _get_me_enemy(state, player_id)

    mask[0] = 1.0

    _mask_play_actions(mask, me, enemy)
    _mask_attack_actions(mask, me, enemy)

    _apply_placement_mode(mask, me, placement_mode)

    if verify_mask:
        _verify_mask(state, player_id, mask)

    return mask


def _mask_play_actions(mask, me, enemy):
    for hand_idx in range(min(len(me.hand), _NUM_HAND)):
        card = me.hand[hand_idx]
        if me.mana < card.mana_cost:
            continue

        is_warrior = card.card_type == CardType.WARRIOR

        if is_warrior and len(me.board) >= _NUM_BOARD:
            continue

        mechanics = card.mechanics
        needs_tgt = requires_target(mechanics) and not is_random_battlecry_damage_card(card)
        has_csd = "choose_shield_damage" in mechanics

        if is_warrior:
            num_positions = min(len(me.board) + 1, _NUM_PLAY_POS)
            num_positions = max(num_positions, 1)
        else:
            num_positions = 1

        for pos_idx in range(num_positions):
            base = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos_idx * _NUM_PLAY_TARGETS

            if not needs_tgt and not has_csd:
                mask[base + 0] = 1.0
                continue

            _mask_targets_for_card(mask, base, mechanics, me, enemy)


def _mask_attack_actions(mask, me, enemy):
    taunt_units = get_taunt_targets(enemy.board)
    has_taunts = len(taunt_units) > 0
    taunt_indices = {i for i, u in enumerate(enemy.board) if u in taunt_units}

    for att_idx in range(min(len(me.board), _NUM_BOARD)):
        attacker = me.board[att_idx]
        if not attacker.is_ready or attacker.is_frozen:
            continue

        eff_atk = _compute_effective_attack(attacker, me.board, me.hero)
        if eff_atk <= 0:
            continue

        can_bypass = "bypass_taunt" in attacker.mechanics
        base = _ATTACK_BASE + att_idx * _NUM_ATTACK_TARGETS

        for t_idx in range(len(enemy.board)):
            if has_taunts and not can_bypass:
                if t_idx in taunt_indices:
                    mask[base + t_idx] = 1.0
            else:
                mask[base + t_idx] = 1.0

        if not has_taunts or can_bypass:
            mask[base + 7] = 1.0


def _mask_targets_for_card(mask, base, mechanics, me, enemy):
    is_consume = any("consume_ally" in m for m in mechanics)
    is_damage = any("damage" in m for m in mechanics)
    is_heal_target = any("heal_target" in m for m in mechanics)
    is_heal = any("heal" in m for m in mechanics)
    is_buff = any("buff" in m for m in mechanics)
    is_delete = any("delete_target" in m for m in mechanics)
    is_freeze = any("freeze" in m or "battlecry_freeze" in m for m in mechanics)
    is_csd = any("choose_shield_damage" in m for m in mechanics)
    # TAMHP (card 52): target_ally_max_hp_plus[_universal]_N. Frozen-classic
    # ADDITIVE exception (user-authorized): a new branch using EXISTING
    # target-slot space (friendly-minion codes 9..15 + own-hero code 16 —
    # both already part of the frozen 17-target layout). Strictly additive:
    # only cards whose mechanics contain `target_ally_max_hp_plus` reach this
    # branch; for all other cards is_max_hp_plus_* is False and every existing
    # branch below/above is byte-identical. The universal variant allows the
    # own-hero target (code 16); the bare variant forbids hero. Mirrors
    # core/engine.py `get_valid_targets` is_max_hp_plus_universal / is_max_hp_plus.
    is_max_hp_plus_universal = any(
        m.startswith("target_ally_max_hp_plus_universal") for m in mechanics
    )
    is_max_hp_plus = any(
        m.startswith("target_ally_max_hp_plus_")
        and not m.startswith("target_ally_max_hp_plus_universal")
        for m in mechanics
    )

    if is_consume:
        for t_idx in range(len(me.board)):
            mask[base + 9 + t_idx] = 1.0
        return

    if is_csd:
        for t_idx in range(len(enemy.board)):
            mask[base + 1 + t_idx] = 1.0
        mask[base + 8] = 1.0
        mask[base + 0] = 1.0
        return

    if is_freeze and not is_damage:
        for t_idx in range(len(enemy.board)):
            mask[base + 1 + t_idx] = 1.0
        return

    if is_damage or is_freeze:
        for t_idx in range(len(enemy.board)):
            mask[base + 1 + t_idx] = 1.0
        mask[base + 8] = 1.0
        return

    if is_delete:
        for t_idx in range(len(enemy.board)):
            mask[base + 1 + t_idx] = 1.0
        return

    if is_heal_target:
        for t_idx in range(len(me.board)):
            mask[base + 9 + t_idx] = 1.0
        mask[base + 16] = 1.0
        return

    # TAMHP universal (card 52): friendly board minions (9..15) + own hero
    # (16). Additive — only card 52 reaches here.
    if is_max_hp_plus_universal:
        for t_idx in range(len(me.board)):
            mask[base + 9 + t_idx] = 1.0
        mask[base + 16] = 1.0
        return

    # TAMHP non-universal: friendly board minions only (no hero).
    if is_max_hp_plus:
        for t_idx in range(len(me.board)):
            mask[base + 9 + t_idx] = 1.0
        return

    if is_heal:
        for t_idx, unit in enumerate(me.board):
            if unit.hp < unit.max_hp:
                mask[base + 9 + t_idx] = 1.0
        if me.hero.hp < me.hero.max_hp:
            mask[base + 16] = 1.0
        return

    if is_buff:
        for t_idx in range(len(me.board)):
            mask[base + 9 + t_idx] = 1.0
        return


def _verify_mask(state, player_id, mask):
    me, enemy = _get_me_enemy(state, player_id)

    rng_state = rand_mod.getstate()
    try:
        for action_id in range(MAX_CANDIDATE_ACTIONS):
            if mask[action_id] != 1.0:
                continue

            action = decode_action(state, player_id, action_id)
            if action is None:
                mask[action_id] = 0.0
                continue

            if isinstance(action, PlayCardAction):
                card = me.hand[action.hand_index] if action.hand_index < len(me.hand) else None
                if card and card.card_type == CardType.WARRIOR:
                    if action.position is not None and (action.position < 0 or action.position > len(me.board)):
                        mask[action_id] = 0.0
                        continue

            try:
                env = _make_preview_env(state)
                success, _ = env.step(player_id, action)
                if not success:
                    mask[action_id] = 0.0
            except Exception:
                mask[action_id] = 0.0
    finally:
        rand_mod.setstate(rng_state)


def _make_preview_env(state: GameState) -> ArenaEnvironment:
    env = ArenaEnvironment.__new__(ArenaEnvironment)
    env.state = _clone_state_for_preview(state)
    env.mana_per_turn = 1
    try:
        from infrastructure.match_modes import ClassicParams
        env.classic_params = ClassicParams(mana_per_turn=env.mana_per_turn)
    except Exception:
        pass
    env.sudden_death_turns_by_player = {}
    # Preview-env обходит __init__, поэтому устанавливаем _rng вручную.
    # Используется core.engine._handle_end_turn (и эффектами в core.effects)
    # при взвешенном No-FIFO доборе карт. Без инициализации _rng любой
    # step, который заканчивает ход, падает с AttributeError и затирает
    # легальное действие end_turn в action_mask.
    #
    # Детерминизм preview-env критичен для _verify_mask: два evaluate_matchup
    # вызова с одинаковыми seeds должны давать идентичный результат. Если
    # родительский state.arena_engine уже есть (нормальный flow через
    # ClassicRLEnv) — клонируем state его _rng через getstate/setstate
    # (deepcopy Random через pickle дорог, а getstate — O(1)).
    parent_engine = getattr(state, "arena_engine", None)
    if parent_engine is not None and hasattr(parent_engine, "_rng"):
        cloned = rand_mod.Random()
        cloned.setstate(parent_engine._rng.getstate())
        env._rng = cloned
    else:
        # Cold path: derive deterministic seed from state fingerprint so
        # the same state always produces the same preview RNG.
        seed_val = (
            state.turn_number * 1_000_003
            ^ state.current_turn_owner_id * 31
            ^ len(state.p1.hand) * 7919
            ^ len(state.p2.hand) * 17
            ^ len(state.p1.deck) * 113
            ^ len(state.p2.deck) * 257
        )
        env._rng = rand_mod.Random(seed_val & 0xFFFFFFFF)

    def _preview_apply_aura_bonuses(self, attacker, player):
        return _compute_effective_attack(attacker, player.board, player.hero)

    env._apply_aura_bonuses = _preview_apply_aura_bonuses.__get__(env, ArenaEnvironment)
    return env


def _clone_card_for_preview(card: CardInstance) -> CardInstance:
    return CardInstance(
        instance_id=card.instance_id,
        card_id=card.card_id,
        name=card.name,
        card_type=card.card_type,
        rarity=card.rarity,
        mana_cost=card.mana_cost,
        attack=card.attack,
        hp=card.hp,
        max_hp=card.max_hp,
        mechanics=list(card.mechanics),
        is_ready=card.is_ready,
        is_frozen=card.is_frozen,
        description=card.description,
        mechanics_desc=card.mechanics_desc,
        level=card.level,
        simplified_levelup=card.simplified_levelup,
        base_attack=card.base_attack,
        base_hp=card.base_hp,
        base_max_hp=card.base_max_hp,
        base_mana_cost=card.base_mana_cost,
        base_mechanics=list(card.base_mechanics) if card.base_mechanics is not None else None,
        instant_kill_used=card.instant_kill_used,
    )


def _clone_player_for_preview(player: PlayerState) -> PlayerState:
    return PlayerState(
        user_id=player.user_id,
        is_bot=player.is_bot,
        replacement_status=player.replacement_status,
        hero=_clone_card_for_preview(player.hero),
        mana=player.mana,
        max_mana=player.max_mana,
        hand=[_clone_card_for_preview(c) for c in player.hand],
        board=[_clone_card_for_preview(c) for c in player.board],
        deck=[_clone_card_for_preview(c) for c in player.deck],
        graveyard=[_clone_card_for_preview(c) for c in player.graveyard],
        trophies=player.trophies,
        surrender_processed=player.surrender_processed,
    )


def _clone_state_for_preview(state: GameState) -> GameState:
    return GameState(
        p1=_clone_player_for_preview(state.p1),
        p2=_clone_player_for_preview(state.p2),
        current_turn_owner_id=state.current_turn_owner_id,
        turn_number=state.turn_number,
        history=[dict(item) for item in state.history],
        action_history=list(state.action_history),
        v5_history_events=list(getattr(state, "v5_history_events", ())),
        status=state.status,
        sudden_death_turns_by_player=dict(state.sudden_death_turns_by_player),
        sudden_death_last_applied_turn_by_player=dict(state.sudden_death_last_applied_turn_by_player),
        pending_mana_drain_by_player=dict(state.pending_mana_drain_by_player),
    )


# ============================================================================
# ACTION FEATURES
# ============================================================================

def encode_action_features(
    state: GameState,
    player_id: int,
    *,
    include_preview: bool = True,
    verify_mask: bool = True,
    placement_mode: str = "full",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns shape (601, 171) — feature vector for each candidate action.
    Only masked actions have preview deltas filled in.
    Set include_preview=False to skip deep-copy preview simulation (faster, preview channels = 0).
    """
    if mask is None:
        mask = build_action_mask(state, player_id, verify_mask=verify_mask, placement_mode=placement_mode)
    feats = np.zeros((MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32)

    me, enemy = _get_me_enemy(state, player_id)

    for action_id in range(MAX_CANDIDATE_ACTIONS):
        if mask[action_id] != 1.0:
            continue

        _encode_one_action(feats[action_id], state, me, enemy, action_id, include_preview=include_preview)

    return feats


def _encode_one_action(out, state, me, enemy, action_id, *, include_preview: bool = True):

    src_card, tgt_card = None, None
    board_pos, hand_pos, att_pos = -1, -1, -1
    rel = np.zeros(8, dtype=np.float32)

    if action_id == 0:
        out[128] = 1.0
        _fill_preview_zero(out)
        return

    if 1 <= action_id <= 544:
        out[129] = 1.0
        flat = action_id - _PLAY_BASE
        pos_idx, tcode = divmod(flat, _NUM_PLAY_TARGETS)
        hand_idx, pos_idx = divmod(pos_idx, _NUM_PLAY_POS)

        src_card = me.hand[hand_idx] if hand_idx < len(me.hand) else None
        hand_pos = hand_idx
        if src_card and src_card.card_type == CardType.WARRIOR:
            board_pos = pos_idx
        elif src_card and src_card.card_type == CardType.POTION:
            board_pos = -1

        tgt_card, rel = _resolve_target_card(me, enemy, tcode, is_attack=False)

    elif 545 <= action_id <= 600:
        out[130] = 1.0
        flat = action_id - _ATTACK_BASE
        att_idx, tcode = divmod(flat, _NUM_ATTACK_TARGETS)

        if att_idx < len(me.board):
            src_card = me.board[att_idx]
            att_pos = att_idx
            tgt_card, rel = _resolve_target_card(me, enemy, tcode, is_attack=True)

        board_pos = -1
        hand_pos = -1

    eff_atk = None
    if src_card and src_card.card_type == CardType.WARRIOR:
        eff_atk = _compute_effective_attack(src_card, me.board, me.hero)

    out[:CARD_SHAPE_DIM] = encode_card_shape(src_card, board_pos=board_pos, hand_pos=hand_pos,
                                              effective_attack=eff_atk)
    out[CARD_SHAPE_DIM : 2 * CARD_SHAPE_DIM] = encode_card_shape(tgt_card)

    out[131:139] = rel

    out[139] = (board_pos + 1) / (_NUM_PLAY_POS + 1) if board_pos >= 0 else 0.0
    out[140] = (hand_pos + 1) / (_NUM_HAND + 1) if hand_pos >= 0 else 0.0
    out[141] = (att_pos + 1) / (_NUM_BOARD + 1) if att_pos >= 0 else 0.0

    if include_preview:
        _fill_preview_delta(out, state, action_id)
    else:
        _fill_preview_zero(out)


def _resolve_target_card(me, enemy, tcode, is_attack):
    rel = np.zeros(8, dtype=np.float32)

    if is_attack:
        if 0 <= tcode <= 6 and tcode < len(enemy.board):
            rel[1] = 1.0
            return enemy.board[tcode], rel
        if tcode == 7:
            rel[0] = 1.0
            return enemy.hero, rel
        return None, rel

    if tcode == 0:
        rel[7] = 1.0
        return None, rel
    if tcode == 8:
        rel[0] = 1.0
        return enemy.hero, rel
    if tcode == 16:
        rel[2] = 1.0
        return me.hero, rel
    if 1 <= tcode <= 7 and tcode - 1 < len(enemy.board):
        rel[1] = 1.0
        return enemy.board[tcode - 1], rel
    if 9 <= tcode <= 15 and tcode - 9 < len(me.board):
        rel[3] = 1.0
        return me.board[tcode - 9], rel
    return None, rel


def _fill_preview_delta(out, state, action_id):
    player_id = state.current_turn_owner_id
    action = decode_action(state, player_id, action_id)
    if action is None:
        _fill_preview_zero(out)
        return

    rng_state = rand_mod.getstate()
    try:
        temp_env = _make_preview_env(state)
        hp_before = _snapshot_hp(temp_env.state)
        success, _ = temp_env.step(player_id, action)

        if not success:
            _fill_preview_zero(out)
            return

        hp_after = _snapshot_hp(temp_env.state)
        me = temp_env.state.p1 if temp_env.state.p1.user_id == player_id else temp_env.state.p2
        enemy = temp_env.state.p2 if temp_env.state.p1.user_id == player_id else temp_env.state.p1

        out[142] = np.clip((hp_after["my_hero"] - hp_before["my_hero"]) / 20.0, -1.0, 1.0)
        out[143] = np.clip((hp_after["enemy_hero"] - hp_before["enemy_hero"]) / 20.0, -1.0, 1.0)

        for i in range(5):
            out[144 + i] = np.clip(
                (hp_after.get(f"my_board_{i}", 0) - hp_before.get(f"my_board_{i}", 0)) / 20.0,
                -1.0, 1.0,
            )
        for i in range(5):
            out[151 + i] = np.clip(
                (hp_after.get(f"enemy_board_{i}", 0) - hp_before.get(f"enemy_board_{i}", 0)) / 20.0,
                -1.0, 1.0,
            )

        out[158] = (me.mana - (hp_before.get("my_mana") or 0)) / 10.0
        out[159] = (enemy.mana - (hp_before.get("enemy_mana") or 0)) / 10.0

        out[160] = np.clip((len(me.board) - hp_before.get("my_board_count", 0)) / 3.0, -1.0, 1.0)
        out[161] = np.clip((len(enemy.board) - hp_before.get("enemy_board_count", 0)) / 3.0, -1.0, 1.0)

        out[162] = np.clip((len(me.hand) - hp_before.get("my_hand_count", 0)) / 3.0, -1.0, 1.0)

        source_card = _get_source_card(state, action_id, me)
        target_card = _get_target_card(state, action_id, me, enemy)
        out[163] = 1.0 if source_card and source_card.hp <= 0 else 0.0
        out[164] = 1.0 if target_card and target_card.hp <= 0 else 0.0

        if isinstance(action, AttackAction):
            attacker = _find_unit(me.board, action.attacker_id)
            out[165] = 1.0 if attacker and "lifesteal" in attacker.mechanics and attacker.attack > 0 else 0.0
            out[166] = (hp_before["enemy_hero"] - hp_after["enemy_hero"]) / 20.0 if action.target_is_hero else 0.0

        out[167] = 1.0

        out[168] = 0.0
        out[169] = 0.0
        out[170] = 0.0

    except Exception:
        _fill_preview_zero(out)
    finally:
        rand_mod.setstate(rng_state)


def _fill_preview_zero(out):
    out[142:171] = 0.0


def _snapshot_hp(state):
    snap = {
        "my_hero": state.p1.hero.hp if state.p1.user_id == state.current_turn_owner_id else state.p2.hero.hp,
        "enemy_hero": state.p2.hero.hp if state.p1.user_id == state.current_turn_owner_id else state.p1.hero.hp,
    }
    me = state.p1 if state.p1.user_id == state.current_turn_owner_id else state.p2
    enemy = state.p2 if state.p1.user_id == state.current_turn_owner_id else state.p1
    for i, u in enumerate(me.board):
        snap[f"my_board_{i}"] = u.hp
    for i, u in enumerate(enemy.board):
        snap[f"enemy_board_{i}"] = u.hp
    snap["my_mana"] = me.mana
    snap["enemy_mana"] = enemy.mana
    snap["my_board_count"] = len(me.board)
    snap["enemy_board_count"] = len(enemy.board)
    snap["my_hand_count"] = len(me.hand)
    return snap


def _get_source_card(state, action_id, me):
    if action_id == 0:
        return None
    if 1 <= action_id <= 544:
        flat = action_id - _PLAY_BASE
        pos_idx, tcode = divmod(flat, _NUM_PLAY_TARGETS)
        hand_idx, pos_idx = divmod(pos_idx, _NUM_PLAY_POS)
        return me.hand[hand_idx] if hand_idx < len(me.hand) else None
    if 545 <= action_id <= 600:
        flat = action_id - _ATTACK_BASE
        att_idx, tcode = divmod(flat, _NUM_ATTACK_TARGETS)
        return me.board[att_idx] if att_idx < len(me.board) else None
    return None


def _get_target_card(state, action_id, me, enemy):
    if action_id == 0:
        return None
    if 1 <= action_id <= 544:
        flat = action_id - _PLAY_BASE
        pos_idx, tcode = divmod(flat, _NUM_PLAY_TARGETS)
        card, _ = _resolve_target_card(me, enemy, tcode, is_attack=False)
        return card
    if 545 <= action_id <= 600:
        flat = action_id - _ATTACK_BASE
        att_idx, tcode = divmod(flat, _NUM_ATTACK_TARGETS)
        card, _ = _resolve_target_card(me, enemy, tcode, is_attack=True)
        return card
    return None


def _find_unit(board, unit_id):
    for u in board:
        if str(u.instance_id) == unit_id:
            return u
    return None
