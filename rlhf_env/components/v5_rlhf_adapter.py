"""V5 RLHF adapter — deploys a trained V5 policy as the bot opponent in rlhf_env.

Block C component C1 (D-C0: build in the worktree's rlhf_env now). Mirrors the
V4 action-conditioned ONNX pattern (``ai.train_v2.onnx_policy.OnnxActionPolicy``)
but targets the V5 model contract: the V5 net is action-conditioned and emits a
3-tuple ``(logits, value, mana_draw_logit)`` (the dense evaluator form,
``rust_ppo.py:755-759``).

The adapter is the rlhf_env-side counterpart of ``_BerserkPolicyAdapter``
(``policy_factory.py:172-196``) + ``_LiveArenaShim`` (``policy_factory.py:198-271``,
``placement_mode='append_only'``) + ``OnnxActionPolicy``
(``ai/train_v2/onnx_policy.py:16-85``). It satisfies the rlhf_env adapter contract:
``select_action(engine, player_id) -> int`` returning an index into
``engine.get_legal_actions(player_id)`` (a ``List[BaseAction]``), plus the
provenance attrs ``name/kind/model_path/weights_hash/weights_version`` read by
``arena_match_manager`` v5 bot_policy_info and ``match_runner._capture_models``.

D11 OMNISCIENT (design.md:46/125): the server-side bot sees the FULL GameState,
so the observation is built with ``InfoModeV5(enemy_hand_known=True,
enemy_deck_known=True)`` — explicitly, NOT the ``InfoModeV5()`` default
(``contracts.py:46-47`` defaults both to False, which would violate D11).

ONNX I/O contract (documented for the future V5 export, design.md:158 — the real
export is an operational step, so the real-load path is exercised only with a
fake ``InferenceSession`` in tests):
  inputs:
    "observation"     : float32[1, OBS_V5_DIM]            (7128)
    "action_features" : float32[1, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM]
                                                          (1, 601, 171)
  outputs:
    "logits"          : float32[1, MAX_CANDIDATE_ACTIONS]  (1, 601)
    "value"           : float32[1, 1]
    "mana_draw_logit" : float32[1, 1]
The ``inference`` callable accepted by ``__init__`` is the un-batched front end:
``inference(obs, action_features) -> (logits, value, mana_draw_logit)`` where
``obs`` is 1-D ``[OBS_V5_DIM]`` and ``action_features`` is 2-D
``[601, 171]``; the real ONNX path batches internally and slices ``[0]``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from rlhf_env.components.policy_factory import _onnx_sha256

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# sys.path bootstrap for train_v3 (V5 encoders live ONLY in TrainV3.5/python).
# rlhf_env entrypoints add only REPO_ROOT to sys.path (mcp_server.py:44), not
# TrainV3.5/python — so this module idempotently adds it. Computed from the
# repo root the same way rlhf_env does (Path(__file__).resolve().parents[2]).
# ----------------------------------------------------------------------------

_HERE = Path(__file__).resolve()              # .../rlhf_env/components/v5_rlhf_adapter.py
_REPO_ROOT = _HERE.parents[2]                  # repo root
_TRAIN_V3_PY = _REPO_ROOT / "TrainV3.5" / "python"
if str(_TRAIN_V3_PY) not in sys.path:
    sys.path.insert(0, str(_TRAIN_V3_PY))


class V5RlhfAdapter:
    """rlhf_env adapter for a deployed V5 ONNX policy (kind='v5').

    Built by ``_factory_v5_real(spec, registry)`` (module-level 2-arg factory,
    mirroring ``_factory_berserk`` / ``_factory_v5`` in ``policy_adapters.py``).
    """

    kind = "v5"

    def __init__(self, spec: Dict[str, Any], *, inference: Optional[Callable] = None):
        self._spec = spec or {}
        self.name = self._spec.get("name", "v5-deploy")
        self.model_path = self._spec.get("path")
        # provenance for V5-meta bot_policy (mirror _BerserkPolicyAdapter:177-181).
        self.weights_hash = _onnx_sha256(self.model_path)
        self.weights_version = self._spec.get("weights_version")

        # inference: explicit kwarg first, then spec['inference'] (so both
        # V5RlhfAdapter(spec, inference=fn) and _factory_v5_real reading
        # spec.get('inference') / direct V5RlhfAdapter({'inference': fn}) work).
        if inference is None:
            inference = self._spec.get("inference")

        if inference is not None:
            # TEST injection / custom inference front end.
            self._inference = inference
        elif self.model_path:
            # Real ONNX load (mirrors OnnxActionPolicy:33). The V5 ONNX export is
            # a FUTURE operational step (design.md:158); exercised only with a
            # fake InferenceSession in tests.
            self._inference = self._build_onnx_inference(self.model_path)
        else:
            raise ValueError(
                "V5RlhfAdapter requires either an `inference` callable or a 'path' "
                "to a V5 ONNX model in spec."
            )

    @staticmethod
    def _build_onnx_inference(model_path: str) -> Callable:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "V5RlhfAdapter real-load requires onnxruntime, which is not installed."
            ) from exc
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        def _infer(obs: np.ndarray, action_features: np.ndarray):
            obs_batch = obs[np.newaxis, :].astype(np.float32)
            af_batch = action_features[np.newaxis, :, :].astype(np.float32)
            outputs = session.run(
                ["logits", "value", "mana_draw_logit"],
                {"observation": obs_batch, "action_features": af_batch},
            )
            # Un-batch: logits -> [601], value -> scalar, mana_draw_logit -> scalar.
            return outputs[0][0], float(outputs[1][0]), float(outputs[2][0])

        return _infer

    def select_action(self, engine: Any, player_id: int) -> int:
        # 1. legal = engine.get_legal_actions(player_id); empty -> 0
        #    (mirror _BerserkPolicyAdapter:184-186 / _LegacyOnnxBotPolicy:267-269).
        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0

        # 2. Live GameState (server-side bot has FULL state -> omniscient).
        state = engine.state

        # Lazy imports: train_v3 encoders live only in TrainV3.5/python (added to
        # sys.path at module import). Import failure is a clear runtime error only
        # when a V5 bot actually plays, not at rlhf_env boot.
        try:
            from train_v3.contracts import AssistModeV5, InfoModeV5
            from train_v3.obs_v5 import encode_observation_v5
        except ImportError as exc:
            raise RuntimeError(
                f"V5RlhfAdapter needs train_v3 (TrainV3.5/python on sys.path) to "
                f"encode the V5 observation, but import failed: {exc}. Ensure "
                f"TrainV3.5/python is present at {_TRAIN_V3_PY}."
            ) from exc
        from ai.train_v2.classic_actions_v1 import (
            build_action_mask,
            encode_action_features,
        )

        # 3. D11 OMNISCIENT observation: enemy_hand_known=True + enemy_deck_known=True
        #    EXPLICIT (InfoModeV5() default is SELF-VISIBLE, contracts.py:46-47 —
        #    would VIOLATE D11). AssistModeV5() default = no assist channels.
        info_mode = InfoModeV5(enemy_hand_known=True, enemy_deck_known=True)
        assist_mode = AssistModeV5()

        # history_events: encode_observation_v5 expects list[dict[str, Any]] and
        # fills the 20x144 history channel from them (_encode_history, obs_v5.py
        # :120). The CORRECT live field is ``state.history`` (List[Dict],
        # core/state.py:165), which the engine populates via
        # ``state.history.append(action.to_dict())`` (core/engine.py:405) during
        # play. v5_trace captures ``history = list(state.history)`` into the
        # snapshot (v5_trace.py:310) and the offline loader feeds
        # ``pre_snap["history"]`` to encode_observation_v5 (offline_dataset_loader
        # .py:736) — so reading ``state.history`` here gives TRAIN/DEPLOY PARITY
        # (deploy obs history == offline training obs history). Do NOT read
        # ``state.action_history`` (core/state.py:177) — that is a Deque of
        # (log_type, text) UI-log tuples, a different field; filtering it to dicts
        # yields [] and silently drops the history channel (train/deploy mismatch).
        raw_history = getattr(state, "history", None) or []
        history_events = [e for e in raw_history if isinstance(e, dict)]

        obs = encode_observation_v5(
            state,
            player_id,
            info_mode=info_mode,
            assist_mode=assist_mode,
            history_events=history_events,
        )

        # 4. action_features (C0 fix: placement_mode='append_only'; mirror
        #    _LiveArenaShim:223). include_preview=False — live inference does not
        #    need preview deltas and this skips the deep-copy simulation.
        action_features = encode_action_features(
            state,
            player_id,
            verify_mask=False,
            placement_mode="append_only",
            include_preview=False,
        )

        # 5. legal_mask — the legal candidate set (frozen classic_actions_v1:188).
        legal_mask = build_action_mask(
            state, player_id, verify_mask=False, placement_mode="append_only",
        )

        # 6. V5 3-tuple forward (logits, value, mana_draw_logit).
        logits, _value, _mana_draw_logit = self._inference(obs, action_features)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)

        # 7. argmax over legal-masked candidates -> candidate id -> legal index.
        masked = np.where(legal_mask.astype(bool), logits, -np.inf).astype(np.float32)
        if not np.any(np.isfinite(masked)):
            # No masked-in candidate (shouldn't happen — legal was non-empty).
            return 0
        chosen_candidate = int(np.argmax(masked))
        return self._candidate_to_legal_index(
            state, player_id, chosen_candidate, legal
        )

    @staticmethod
    def _candidate_to_legal_index(state, player_id: int, candidate_id: int, legal) -> int:
        """Map a V5/train_v2 candidate action_id (0..600) to the index of the
        matching BaseAction in ``legal`` (``engine.get_legal_actions``).

        Mirrors ``_LegacyOnnxBotPolicy`` idx-matching (policy_factory.py:267+):
        exact @dataclass eq first, then loosen-match by type+fields (warrior
        position may drift), then type-fallback, then last legal.
        """
        from ai.train_v2.classic_actions_v1 import decode_action
        from core.actions import AttackAction, EndTurnAction, PlayCardAction

        base = decode_action(state, player_id, candidate_id)

        if base is not None:
            # 1) Exact value-equality (@dataclass auto-eq on fields).
            for i, a in enumerate(legal):
                if a == base:
                    return i
            # 2) Loosen-match: warrior position may have drifted.
            if isinstance(base, PlayCardAction):
                for i, a in enumerate(legal):
                    if (isinstance(a, PlayCardAction)
                            and a.hand_index == base.hand_index
                            and a.target_id == base.target_id):
                        return i
            elif isinstance(base, AttackAction):
                for i, a in enumerate(legal):
                    if (isinstance(a, AttackAction)
                            and a.attacker_id == base.attacker_id
                            and a.target_id == base.target_id
                            and a.target_is_hero == base.target_is_hero):
                        return i

        # 3) Type fallback (never return an invalid idx).
        if candidate_id == 0 or base is None or isinstance(base, EndTurnAction):
            for i, a in enumerate(legal):
                if isinstance(a, EndTurnAction):
                    return i
        if isinstance(base, PlayCardAction):
            for i, a in enumerate(legal):
                if isinstance(a, PlayCardAction):
                    return i
        if isinstance(base, AttackAction):
            for i, a in enumerate(legal):
                if isinstance(a, AttackAction):
                    return i
        # 4) Last legal.
        return len(legal) - 1


# ----------------------------------------------------------------------------
# Module-level 2-arg factory (mirrors _factory_berserk / _factory_v5).
# AdapterRegistry.resolve(kind) returns the factory, then it is CALLED as
# (spec, registry) -> adapter. registry may be None.
# ----------------------------------------------------------------------------

def _factory_v5_real(spec: Dict[str, Any], registry: Optional[Any]) -> "V5RlhfAdapter":
    """Build a V5RlhfAdapter from a spec.

    spec keys: name, path (V5 ONNX), weights_version (optional), inference
    (optional callable for TEST injection — not set by the registry path).
    """
    inference = spec.get("inference") if isinstance(spec, dict) else None
    return V5RlhfAdapter(spec, inference=inference)


__all__ = ["V5RlhfAdapter", "_factory_v5_real"]