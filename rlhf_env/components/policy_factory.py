"""Тонкая обёртка над ai.bot_brain.BerserkInference для RLHF-среды.

Использует:
- V4 action-conditioned (и V3-legacy при наличии sidecar .onnx.json с obs_dim) →
  BerserkInference (принимает GameState + legal_actions, возвращает idx)
- Baselines (random/greedy_face/end_turn) → собственные RLHF*-классы, работают
  с ArenaEnvironment напрямую (без gym-обёртки)

ВНИМАНИЕ: onnx-модель БЕЗ sidecar (.onnx.json с obs_dim) НЕ сможет загрузиться через
BerserkInference (obs_dim=0 → контракт-валидация падает). build_policy в таком случае
РЕШИТЕЛЬНО поднимает исключение, а не падает молча в rule-based fallback — иначе
battle_log/manifest соврали бы, что играла onnx-модель (см. аудит SYN-1).
Если нужна v3-модель без sidecar — сначала сгенерируйте sidecar (obs_dim/action_feature_dim/
inputs/outputs) либо используйте baseline (random/greedy_face/end_turn).

Все политики приводятся к единому интерфейсу:
    select_action(engine, player_id) -> int  (idx в engine.get_legal_actions(player_id))
"""
from __future__ import annotations

import copy
import json
import logging
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rlhf_env.components.inference_params import default_inference_params

logger = logging.getLogger(__name__)

# «Система сложностей» удалена: модели ВСЕГДА играют на максимум своих
# возможностей. В RLHF-среде это = детерминированный argmax по логитам политики
# (selection="argmax"), без температурного сэмплирования/исследований. Значение
# BOT_MAX_DIFFICULTY — фиксированный ключ профиля BerserkInference (не выбор
# пользователя): ai/bot_brain.BerserkInference требует хотя бы один profile-key,
# поэтому передаём константу "max". Baselines (random/greedy_face/end_turn)
# сложность игнорируют в принципе.
BOT_MAX_DIFFICULTY: str = "max"

# Кеш загруженных BerserkInference: (model_path, difficulty) -> instance
_BERSERK_CACHE: Dict[Tuple[str, str], Any] = {}
# Кеш LegacyOnnxPolicy (onnx-сессия дорогая): model_path -> instance
_LEGACY_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

# Lazy default registry — создаётся при первом обращении к ONNX без явного registry
_DEFAULT_REGISTRY: Optional[Any] = None

# Кеш sha256(.onnx) по пути — provenance для V5-meta bot_policy.weights_hash.
_ONNX_HASH_CACHE: Dict[str, str] = {}


def _onnx_sha256(path: Optional[str]) -> Optional[str]:
    """16-символов sha256 от байтов .onnx — детектор изменения весов между записью
    трека и V5-обучением. None для heuristic-политик (path=None)."""
    if not path:
        return None
    try:
        p = str(path)
        cached = _ONNX_HASH_CACHE.get(p)
        if cached is not None:
            return cached
        import hashlib
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
        _ONNX_HASH_CACHE[p] = h
        return h
    except Exception:  # noqa: BLE001
        return None


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
        # heuristic-политика: поведение пинится engine_version.core_engine_commit,
        # weights_* сознательно None (нет onnx-весов).
        self.model_path = None
        self.weights_hash = None
        self.weights_version = None

    def select_action(self, engine, player_id: int) -> int:
        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0
        return self._rng.randrange(len(legal))


class _RLHFEndTurnPolicy:
    name = "end_turn"

    def __init__(self):
        self.kind = "end_turn"
        self.model_path = None
        self.weights_hash = None
        self.weights_version = None

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
        self.model_path = None
        self.weights_hash = None
        self.weights_version = None

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
    def __init__(self, brain: Any, difficulty: str, name: str, kind: str, path: Optional[str] = None):
        self._brain = brain
        self._difficulty = difficulty
        self.name = name
        self.kind = kind
        # provenance для V5-meta: путь к onnx + хэш весов (детектор изменения).
        self.model_path = path
        self.weights_hash = _onnx_sha256(path)
        self.weights_version = None

    def select_action(self, engine, player_id: int) -> int:
        legal = engine.get_legal_actions(player_id)
        if not legal:
            return 0
        return int(self._brain.get_action(engine.state, player_id, legal, self._difficulty))


# ============================================================================
# Legacy V2/V3 ONNX (один input-observation) — БЕЗ BerserkInference.
# Переиспользует ai.model_benchmark.policies.LegacyOnnxPolicy (кодек НЕ
# реимплементируется). Тонкий read-only shim над живой ArenaEnvironment даёт
# интерфейс ClassicRLEnv (clone_state/action_mask/legal_action_ids), нужный
# LegacyOnnxPolicy.select_action; __init__/reset/step НЕ вызываются.
# ============================================================================

class _LiveArenaShim:
    """Read-only мост engine._arena (ArenaEnvironment) -> ClassicRLEnv-интерфейс.

    Нужен только для LegacyOnnxPolicy.select_action, который использует:
      clone_state(), action_mask(player_id), legal_action_ids(player_id).
    Все три метода зависят лишь от живого self._arena.state и классических
    кодеков; reset()/step() инжектить НЕ надо (они нужны только для новой игры
    и дефолт-колоды). Shim НЕ мутирует арену: copy.deepcopy клонирует state.
    """

    def __init__(self, arena: Any):
        self._arena = arena

    def clone_state(self):
        return copy.deepcopy(self._arena.state)

    def action_mask(self, player_id: int) -> np.ndarray:
        from ai.train_v2.classic_actions_v1 import build_action_mask
        # placement_mode='append_only' КРИТИЧНО: engine.get_legal_actions всегда
        # кладёт warrior position=len(board), поэтому маска должна разрешать
        # только этот position — иначе decoded PlayCardAction.position не
        # совпадёт с легальным. verify_mask=False безопасно (живая арена уже
        # отфильтровала легальность) и сильно быстрее.
        return build_action_mask(
            self._arena.state, player_id,
            verify_mask=False, placement_mode="append_only",
        )

    def legal_action_ids(self, player_id: int) -> List[int]:
        return [int(i) for i in np.flatnonzero(self.action_mask(player_id) == 1.0)]


class _LegacyOnnxBotPolicy:
    """Adapter RLHF-контракту: select_action(engine_arena, player_id) -> int idx.

    idx — индекс в engine_arena.get_legal_actions(player_id) (List[BaseAction]),
    именно так match_runner.run_bot_turn его использует. Внутри:
      1) обернуть живую арену в read-only _LiveArenaShim;
      2) LegacyOnnxPolicy.select_action(shim, player_id) -> TrainV2 id;
      3) decode_action(state, player_id, train_v2_id) -> BaseAction;
      4) найти idx value-equality match в get_legal_actions(player_id);
         при отсутствии точного match — fallback по типу действия.
    """

    def __init__(self, model_path: str, *, name: str):
        self._inner = self._load_inner(model_path)
        self.name = name
        self.kind = "legacy_onnx"
        # provenance для V5-meta bot_policy.
        self.model_path = model_path
        self.weights_hash = _onnx_sha256(model_path)
        self.weights_version = None

    @staticmethod
    def _load_inner(model_path: str):
        from ai.model_benchmark.policies import LegacyOnnxPolicy
        with _CACHE_LOCK:
            cached = _LEGACY_CACHE.get(model_path)
            if cached is not None:
                return cached
        inner = LegacyOnnxPolicy(model_path)
        with _CACHE_LOCK:
            _LEGACY_CACHE[model_path] = inner
        return inner

    def select_action(self, engine_arena: Any, player_id: int) -> int:
        from ai.train_v2.classic_actions_v1 import decode_action
        from core.actions import EndTurnAction, PlayCardAction, AttackAction

        legal = engine_arena.get_legal_actions(player_id)
        if not legal:
            return 0

        shim = _LiveArenaShim(engine_arena)
        train_v2_id = int(self._inner.select_action(shim, player_id))

        state = shim.clone_state()
        base = decode_action(state, player_id, train_v2_id)

        if base is not None:
            # 1) Точное value-equality совпадение (@dataclass авто-eq по полям).
            for i, a in enumerate(legal):
                if a == base:
                    return i
            # 2) Loosen-match: position для warrior мог рассогласоваться.
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

        # 3) Fallback по типу действия (НЕ возвращаем невалидный idx).
        if train_v2_id == 0 or base is None or isinstance(base, EndTurnAction):
            for i, a in enumerate(legal):
                if isinstance(a, EndTurnAction):
                    logger.warning(
                        "[policy_factory] legacy_onnx train_v2_id=%s decoded=%s "
                        "not in legal -> fallback end_turn",
                        train_v2_id, base,
                    )
                    return i
        if isinstance(base, PlayCardAction):
            for i, a in enumerate(legal):
                if isinstance(a, PlayCardAction):
                    logger.warning(
                        "[policy_factory] legacy_onnx train_v2_id=%s decoded=%s "
                        "no exact match -> fallback first legal PlayCard",
                        train_v2_id, base,
                    )
                    return i
        if isinstance(base, AttackAction):
            for i, a in enumerate(legal):
                if isinstance(a, AttackAction):
                    logger.warning(
                        "[policy_factory] legacy_onnx train_v2_id=%s decoded=%s "
                        "no exact match -> fallback first legal Attack",
                        train_v2_id, base,
                    )
                    return i
        # 4) Последний легальный (включая случай, когда base None и нет
        #    типового совпадения).
        logger.warning(
            "[policy_factory] legacy_onnx train_v2_id=%s decoded=%s -> fallback last legal",
            train_v2_id, base,
        )
        return len(legal) - 1


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
    # Профиль «макс. способность»: selection="argmax" → всегда жадный выбор
    # лучшего по логитам действия (без softmax/temperature-сэмплирования).
    # temperature_range требуется валидатором BerserkInference, но при argmax
    # не используется. difficulty здесь — фиксированный ключ "max".
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
           "difficulty": str (IGNORED — всегда BOT_MAX_DIFFICULTY="max"; система
                             сложностей удалена, модель играет на максимум),
           "temperature": float (optional), "mode": str (optional), "seed": int (optional), ...}

    registry: PolicyRegistry (опционально). Если не передан и нужно резолвить ONNX —
              создаётся через PolicyRegistry.scan("ai/models") (лениво).
    """
    name = spec.get("name")
    if not name:
        raise ValueError("policy spec requires 'name'")

    kind = spec.get("kind")
    path = spec.get("path")
    # Система сложностей удалена — модель всегда играет на максимум (argmax).
    # Любой spec.difficulty игнорируется; принудительно фиксируем "max".
    difficulty = BOT_MAX_DIFFICULTY
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

    # Legacy V2/V3 one-input ONNX — отдельный путь, БЕЗ BerserkInference
    # (obs_dim=1456 контракт валиден только для V4 action-conditioned).
    # Переиспользуем ai.model_benchmark.policies.LegacyOnnxPolicy + классический
    # декодер decode_action; shim над живой ареной даёт ClassicRLEnv-интерфейс.
    if kind == "legacy_onnx":
        adapter = _LegacyOnnxBotPolicy(path, name=name)
        logger.info(
            "[policy_factory] built LegacyOnnxBot name=%s path=%s",
            name, path,
        )
        return adapter

    sidecar = _load_sidecar(path)
    params = default_inference_params(sidecar, kind)
    for k in ("temperature", "mode", "seed"):
        if k in spec:
            params[k] = spec[k]

    brain = _load_berserk(path, difficulty=difficulty, sidecar=sidecar)
    # Защита целостности логов (аудит SYN-1): если onnx-модель молча свалилась в
    # rule-based fallback (нет sidecar / obs_dim=0 / контракт не прошёл) — поднимаем
    # явную ошибку, чтобы battle_log/manifest НЕ соврали, что играла onnx-модель.
    if not brain.has_profile(difficulty):
        raise RuntimeError(
            f"onnx-модель {name!r} ({path}) не загрузилась через BerserkInference "
            f"(difficulty={difficulty}) и молча ушла в rule-based fallback. "
            f"Скорее всего отсутствует sidecar-файл '{path}.json' с obs_dim/inputs/outputs. "
            f"Сгенерируйте sidecar либо используйте baseline (random/greedy_face/end_turn). "
            f"Молчаливый fallback запрещён — это нарушило бы целостность обучающих логов."
        )
    logger.info(
        "[policy_factory] built BerserkPolicy name=%s kind=%s path=%s difficulty=%s",
        name, kind, path, difficulty,
    )
    return _BerserkPolicyAdapter(brain, difficulty=difficulty, name=name, kind=kind, path=path)


def select_action(policy: Any, engine: Any, player_id: int) -> int:
    """Единый вызов для всех типов политик в RLHF-среде."""
    return int(policy.select_action(engine, player_id))


def clear_cache() -> None:
    """Сбрасывает кеш. Полезно при переключении моделей на лету."""
    with _CACHE_LOCK:
        _BERSERK_CACHE.clear()
        _LEGACY_CACHE.clear()


__all__ = [
    "build_policy",
    "select_action",
    "clear_cache",
    "BOT_MAX_DIFFICULTY",
    "_RLHFRandomPolicy",
    "_RLHFEndTurnPolicy",
    "_RLHFGreedyFacePolicy",
]
