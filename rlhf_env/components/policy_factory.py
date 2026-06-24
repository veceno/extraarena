"""Тонкая обёртка над ai.bot_brain.BerserkInference для RLHF-среды.

Использует:
- V4 action-conditioned → BerserkInference (принимает GameState + legal_actions, возвращает idx)
- V3 legacy (V2/V3 one-input) → при наличии sidecar с train_v2_classic_v1 — BerserkInference,
  иначе встроенный LegacyBerserkAdapter (onnxruntime напрямую)
- Baselines (random/greedy_face/end_turn) → собственные RLHF*-классы, работают
  с ArenaEnvironment напрямую (без gym-обёртки)

Все политики приводятся к единому интерфейсу:
    select_action(engine, player_id) -> int  (idx в engine.get_legal_actions(player_id))
"""
from __future__ import annotations

import json
import logging
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rlhf_env.components.inference_params import default_inference_params

logger = logging.getLogger(__name__)

# Кеш загруженных BerserkInference: (model_path, difficulty) -> instance
_BERSERK_CACHE: Dict[Tuple[str, str], Any] = {}
_CACHE_LOCK = threading.Lock()

# Lazy default registry — создаётся при первом обращении к ONNX без явного registry
_DEFAULT_REGISTRY: Optional[Any] = None


def _load_sidecar(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.exists():
        return {}
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[policy_factory] bad sidecar %s: %s", sidecar_path, exc)
        return {}


# ============================================================================
# Baselines: работают напрямую с ArenaEnvironment
# ============================================================================

class _RLHFRandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.kind = "random"

    def select_action(self, engine, player_id: int) -> int:
        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0
        return self._rng.randrange(len(legal))


class _RLHFEndTurnPolicy:
    name = "end_turn"

    def __init__(self):
        self.kind = "end_turn"

    def select_action(self, engine, player_id: int) -> int:
        from core.actions import EndTurnAction
        legal = engine.get_legal_actions(player_id)
        for i, a in enumerate(legal):
            if isinstance(a, EndTurnAction):
                return i
        return len(legal) - 1 if legal else 0


class _RLHFGreedyFacePolicy:
    name = "greedy_face"

    def __init__(self):
        self.kind = "greedy_face"

    def select_action(self, engine, player_id: int) -> int:
        from core.actions import AttackAction, EndTurnAction, PlayCardAction

        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0

        # 1. Атака героя
        for i, a in enumerate(legal):
            if isinstance(a, AttackAction) and getattr(a, "target_is_hero", False):
                return i
        # 2. Атака любого юнита
        for i, a in enumerate(legal):
            if isinstance(a, AttackAction):
                return i
        # 3. Розыгрыш карты без цели
        for i, a in enumerate(legal):
            if isinstance(a, PlayCardAction) and getattr(a, "target_id", None) is None:
                return i
        # 4. Розыгрыш любой карты
        for i, a in enumerate(legal):
            if isinstance(a, PlayCardAction):
                return i
        # 5. Завершить ход
        for i, a in enumerate(legal):
            if isinstance(a, EndTurnAction):
                return i
        return 0


# ============================================================================
# ONNX-политики (V3 legacy + V4 action-conditioned) через BerserkInference
# ============================================================================

class _BerserkPolicyAdapter:
    def __init__(self, brain: Any, difficulty: str, name: str, kind: str):
        self._brain = brain
        self._difficulty = difficulty
        self.name = name
        self.kind = kind

    def select_action(self, engine, player_id: int) -> int:
        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0
        return int(self._brain.get_action(engine.state, player_id, legal, self._difficulty))


def _load_berserk(
    model_path: str, *, difficulty: str, sidecar: Optional[Dict[str, Any]] = None
) -> Any:
    """Загружает ONNX через BerserkInference (lazy, кешируется)."""
    from ai.bot_brain import BerserkInference

    key = (model_path, difficulty)
    with _CACHE_LOCK:
        if key in _BERSERK_CACHE:
            return _BERSERK_CACHE[key]

    sidecar = sidecar if sidecar is not None else _load_sidecar(model_path)
    profile: Dict[str, Any] = {
        "model_path": model_path,
        "temperature_range": [0.1, 1.8],
        "selection": "argmax",
        "format": "train_v2_classic_v1",
        "verify_mask": False,
    }
    if sidecar:
        if "obs_dim" in sidecar:
            profile["obs_dim"] = int(sidecar["obs_dim"])
        if "action_feature_dim" in sidecar:
            profile["action_feature_dim"] = int(sidecar["action_feature_dim"])
        if "max_candidate_actions" in sidecar:
            profile["max_candidate_actions"] = int(sidecar["max_candidate_actions"])
        if "placement_mode" in sidecar:
            profile["placement_mode"] = sidecar["placement_mode"]
        if "format" in sidecar:
            profile["format"] = sidecar["format"]

    brain = BerserkInference({difficulty: profile})
    with _CACHE_LOCK:
        _BERSERK_CACHE[key] = brain
    return brain


# ============================================================================
# Главная фабрика
# ============================================================================

def build_policy(spec: Dict[str, Any], *, registry=None) -> Any:
    """Создаёт политику по спеке.

    spec: {"name": str, "kind": str (optional), "path": str (optional),
           "difficulty": str (default "default"),
           "temperature": float (optional), "mode": str (optional), "seed": int (optional), ...}

    registry: PolicyRegistry (опционально). Если не передан и нужно резолвить ONNX —
              создаётся через PolicyRegistry.scan("ai/models") (лениво).
    """
    name = spec.get("name")
    if not name:
        raise ValueError("policy spec requires 'name'")

    kind = spec.get("kind")
    path = spec.get("path")
    difficulty = str(spec.get("difficulty", "default"))
    seed = int(spec.get("seed", 0))

    # Baselines
    name_lower = name.lower()
    if name_lower in {"random", "random_legal"}:
        return _RLHFRandomPolicy(seed=seed)
    if name_lower in {"greedy_face", "greedy"}:
        return _RLHFGreedyFacePolicy()
    if name_lower in {"end_turn", "end"}:
        return _RLHFEndTurnPolicy()

    # Lazy registry
    global _DEFAULT_REGISTRY
    if registry is None:
        from rlhf_env.components.policy_registry import PolicyRegistry
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = PolicyRegistry.scan("ai/models")
        registry = _DEFAULT_REGISTRY

    # Из registry (если path не указан)
    if path is None and registry is not None:
        resolved = registry.resolve_spec(name, override_kind=kind)
        path = resolved.get("path")
        if not kind:
            kind = resolved.get("kind")

    if path is None:
        raise ValueError(f"policy spec requires 'path' or registry resolution for {name!r}")

    # Auto-detect kind
    if not kind or kind == "auto":
        try:
            from ai.model_benchmark.policies import inspect_model
            kind = inspect_model(path).kind
        except Exception as exc:
            raise ValueError(f"could not auto-detect kind for {path}: {exc}") from exc

    if kind == "unknown":
        raise ValueError(f"model kind unknown for {path}")

    sidecar = _load_sidecar(path)
    params = default_inference_params(sidecar, kind)
    for k in ("temperature", "mode", "seed"):
        if k in spec:
            params[k] = spec[k]

    brain = _load_berserk(path, difficulty=difficulty, sidecar=sidecar)
    logger.info(
        "[policy_factory] built BerserkPolicy name=%s kind=%s path=%s difficulty=%s",
        name, kind, path, difficulty,
    )
    return _BerserkPolicyAdapter(brain, difficulty=difficulty, name=name, kind=kind)


def select_action(policy: Any, engine: Any, player_id: int) -> int:
    """Единый вызов для всех типов политик в RLHF-среде."""
    return int(policy.select_action(engine, player_id))


def clear_cache() -> None:
    """Сбрасывает кеш. Полезно при переключении моделей на лету."""
    with _CACHE_LOCK:
        _BERSERK_CACHE.clear()


__all__ = [
    "build_policy",
    "select_action",
    "clear_cache",
    "_RLHFRandomPolicy",
    "_RLHFEndTurnPolicy",
    "_RLHFGreedyFacePolicy",
]
