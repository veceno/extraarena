"""
TitanV4Inference — Production inference adapter for TITAN V4 ONNX models.
==========================================================================

Bridges the gap between the RL training environment (ArenaEnv, 636-dim obs,
109-action space) and the production game engine (GameState + legal_actions
from ArenaEnvironment.get_legal_actions()).

Key responsibilities:
  1. Extract a 636-dim observation from GameState using the SAME encoding
     as ArenaEnv._get_obs() — ensuring training/inference consistency.
  2. Run the ONNX model to get 109 action logits.
  3. For each BaseAction from engine.get_legal_actions(), compute its
     ArenaEnv action index and read the corresponding logit.
  4. Apply temperature-scaled softmax and sample (or take greedy argmax).
  5. Return the index into legal_actions (compatible with BerserkInference
     usage pattern).

Action space mapping (must match arena_env.py exactly):
    0                           : EndTurnAction
    1 + hand_idx*17 + t_code    : PlayCardAction  (t_code 0-16)
    1 + 4*17 + slot*8 + t_code  : AttackAction    (t_code 0-7)

  t_code for PlayCardAction:
    0              : no target (warrior without battlecry / potion w/o target)
    1..7           : opponent board slots 0-6
    8              : opponent hero
    9..15          : allied board slots 0-6
    16             : own hero

  t_code for AttackAction:
    0..6           : opponent board slots 0-6
    7              : opponent hero

Integration example:
    from ai.bot_inference_v4 import TitanV4Inference

    bot = TitanV4Inference("ai/models/titan_v4_hard.onnx")

    # Inside battle handler, same signature as BerserkInference.get_action():
    action_idx = bot.get_action(game_state, player_id, legal_actions)
    chosen_action = legal_actions[action_idx]
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, List, Optional
from uuid import UUID

import numpy as np
import onnxruntime as ort

from core.actions import BaseAction, PlayCardAction, AttackAction, EndTurnAction
from core.state import GameState, PlayerState, CardInstance, MECHANICS_LIST

logger = logging.getLogger(__name__)

# Must match arena_env.py constants exactly
_MAX_HAND  = 4
_MAX_BOARD = 5
_FEAT_GLOBAL = 6
_FEAT_HERO   = 2 + len(MECHANICS_LIST)           # 35
_FEAT_CARD   = 7 + len(MECHANICS_LIST)           # 40
_OBS_SIZE    = (
    _FEAT_GLOBAL
    + 2 * _FEAT_HERO
    + 2 * _MAX_BOARD * _FEAT_CARD
    + _MAX_HAND * _FEAT_CARD
)  # 636
_TOTAL_ACTIONS = 1 + _MAX_HAND * 17 + _MAX_BOARD * 8  # 109


class TitanV4Inference:
    """
    ONNX inference for TITAN V4 models (636-dim obs → 109 action logits).

    Compatible drop-in for BerserkInference where get_action() is called with
    (game_state, player_id, legal_actions) and returns an index into legal_actions.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.5,
        greedy: bool = False,
    ):
        """
        Args:
            model_path:  Path to the .onnx file exported by export_onnx_v4.py.
            temperature: Softmax temperature for sampling (ignored if greedy=True).
            greedy:      If True, always pick argmax (deterministic).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.temperature = temperature
        self.greedy = greedy

        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        logger.info(
            "[TitanV4] Loaded %s  in=%s out=%s  T=%.2f greedy=%s",
            model_path.name, in_shape, out_shape, temperature, greedy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
    ) -> int:
        """
        Returns index into legal_actions.

        Args:
            game_state:    Current game state.
            player_id:     ID of the bot whose turn it is.
            legal_actions: List of BaseAction from engine.get_legal_actions().
        """
        if not legal_actions:
            logger.warning("[TitanV4] No legal actions — returning 0")
            return 0

        obs = self._extract_obs(game_state, player_id)
        logits = self.session.run(
            None, {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        )[0].flatten()  # shape (109,)

        # Score each legal action by its ArenaEnv action index
        me  = game_state.p1 if game_state.p1.user_id == player_id else game_state.p2
        opp = game_state.p2 if game_state.p1.user_id == player_id else game_state.p1

        scores = np.full(len(legal_actions), -1e9, dtype=np.float32)
        for i, action in enumerate(legal_actions):
            idx = self._action_to_idx(action, me, opp)
            if 0 <= idx < len(logits):
                scores[i] = logits[idx]

        if self.greedy:
            return int(np.argmax(scores))

        # Temperature-scaled softmax sampling
        scores = scores / max(self.temperature, 1e-6)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum() + 1e-8
        return int(np.random.choice(len(legal_actions), p=probs))

    # ------------------------------------------------------------------
    # Observation extraction  (mirrors ArenaEnv._get_obs exactly)
    # ------------------------------------------------------------------

    def _extract_obs(self, state: GameState, agent_id: int) -> np.ndarray:
        me  = state.p1 if state.p1.user_id == agent_id else state.p2
        opp = state.p2 if state.p1.user_id == agent_id else state.p1

        obs: List[float] = [
            state.turn_number / 50.0,
            1.0 if state.current_turn_owner_id == agent_id else 0.0,
            me.mana  / 10.0,
            me.max_mana / 10.0,
            opp.mana / 10.0,
            opp.max_mana / 10.0,
        ]

        obs.extend(self._vec_hero(me.hero))
        obs.extend(self._vec_hero(opp.hero))

        for i in range(_MAX_BOARD):
            obs.extend(self._vec_card(me.board[i])  if i < len(me.board)  else [0.0] * _FEAT_CARD)
        for i in range(_MAX_BOARD):
            obs.extend(self._vec_card(opp.board[i]) if i < len(opp.board) else [0.0] * _FEAT_CARD)
        for i in range(_MAX_HAND):
            obs.extend(self._vec_card(me.hand[i])   if i < len(me.hand)   else [0.0] * _FEAT_CARD)

        return np.array(obs, dtype=np.float32)

    def _vec_hero(self, h: CardInstance) -> List[float]:
        v = [h.hp / 50.0, h.max_hp / 50.0] + [0.0] * len(MECHANICS_LIST)
        for m in h.mechanics:
            base = m.split("_")[0]
            if base in MECHANICS_LIST:
                v[2 + MECHANICS_LIST.index(base)] = 1.0
        return v

    def _vec_card(self, c: CardInstance) -> List[float]:
        v = [
            c.mana_cost / 10.0,
            c.attack    / 20.0,
            c.hp        / 20.0,
            c.max_hp    / 20.0,
            float(c.is_ready),
            float(c.is_frozen),
            c.level     / 10.0,
        ] + [0.0] * len(MECHANICS_LIST)
        for m in c.mechanics:
            for known in MECHANICS_LIST:
                if m == known or m.startswith(known + "_"):
                    v[7 + MECHANICS_LIST.index(known)] = 1.0
                    break
        return v

    # ------------------------------------------------------------------
    # Action → ArenaEnv index mapping
    # ------------------------------------------------------------------

    def _action_to_idx(
        self,
        action: BaseAction,
        me: PlayerState,
        opp: PlayerState,
    ) -> int:
        """Convert a production BaseAction to its ArenaEnv action index (0-108)."""

        if isinstance(action, EndTurnAction):
            return 0

        if isinstance(action, PlayCardAction):
            hand_idx = action.hand_index
            target_id: Optional[str] = action.target_id

            t_code = 0  # default: no target
            if target_id is not None:
                # Check opponent board
                for slot, unit in enumerate(opp.board):
                    if str(unit.instance_id) == target_id:
                        t_code = 1 + slot
                        break
                else:
                    # Check opponent hero
                    if str(opp.hero.instance_id) == target_id:
                        t_code = 8
                    else:
                        # Check allied board
                        for slot, unit in enumerate(me.board):
                            if str(unit.instance_id) == target_id:
                                t_code = 9 + slot
                                break
                        else:
                            # Allied hero
                            if str(me.hero.instance_id) == target_id:
                                t_code = 16

            if hand_idx < 0 or hand_idx >= _MAX_HAND:
                return 0
            return 1 + hand_idx * 17 + t_code

        if isinstance(action, AttackAction):
            attacker_id: str = action.attacker_id
            target_id    = action.target_id
            target_is_hero: bool = action.target_is_hero

            # Find attacker slot in my board
            attacker_slot = None
            for slot, unit in enumerate(me.board):
                if str(unit.instance_id) == attacker_id:
                    attacker_slot = slot
                    break
            if attacker_slot is None:
                return 0

            # Find target slot
            if target_is_hero:
                t_code = 7
            else:
                t_code = 0
                if target_id is not None:
                    for slot, unit in enumerate(opp.board):
                        if str(unit.instance_id) == target_id:
                            t_code = slot
                            break

            base = 1 + _MAX_HAND * 17  # 69
            return base + attacker_slot * 8 + t_code

        logger.warning("[TitanV4] Unknown action type: %s", type(action))
        return 0


# ---------------------------------------------------------------------------
# Convenience factory  (mirrors create_berserk_bot pattern)
# ---------------------------------------------------------------------------

def create_titan_v4_bot(
    model_path: str,
    temperature: float = 0.5,
    greedy: bool = False,
) -> TitanV4Inference:
    """
    Factory function.  Usage:

        bot = create_titan_v4_bot("ai/models/titan_v4_hard.onnx", temperature=0.3)
        idx = bot.get_action(game_state, player_id, legal_actions)
    """
    return TitanV4Inference(model_path, temperature=temperature, greedy=greedy)
