"""Модульный реестр адаптеров политик (tracked layer B).

Замена жёсткому if/elif kind-dispatch в ``policy_factory.build_policy``.
Одна точка расширения для новых видов моделей (V5 и далее):

    from rlhf_env.components.policy_adapters import default_registry
    default_registry().register("v5", MyV5AdapterFactory)
    default_registry().register_detector(my_v5_kind_detector)

— без правок if/elif в фабрике. Существующие V3-legacy/V4/baselines
продолжают работать через реестр; onnx-импорт из gitignored
``ai.model_benchmark`` (layer A) изолирован try/except — в worktree без
layer A baselines работают, onnx-модели дают явную ``ValueError`` с
понятным сообщением (а не ``ModuleNotFoundError``); в prod layer A
доступен → V3/V4 работают.

Контракт адаптера идентичен историческому (см. ``policy_factory``
docstring): ``select_action(engine, player_id) -> int`` (idx в
``engine.get_legal_actions(player_id)``) + attrs ``name/kind/model_path/
weights_hash/weights_version`` (читаются ``arena_match_manager`` v5
bot_policy_info и ``match_runner._capture_models``).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Контракт адаптера
# ----------------------------------------------------------------------------

# Фабрика: (spec, registry) -> PolicyAdapter. registry опционален (PolicyRegistry
# или None). spec содержит name/kind/path/seed/difficulty/...
AdapterFactory = Callable[[Dict[str, Any], Optional[Any]], Any]

# Детектор kind по (path, sidecar, name). Возвращает kind-строку или None
# (ни один детектор не сработал). Вызываются LIFO — позднее зарегистрированный
# детектор имеет приоритет (пользовательский V5-детектор перекрывает fallback).
KindDetector = Callable[[Optional[str], Dict[str, Any], Optional[str]], Optional[str]]

# Baseline-имена, не требующие файла (built-in). kind по имени.
_BASELINE_NAMES: Dict[str, str] = {
    "random": "random",
    "random_legal": "random",
    "greedy_face": "greedy_face",
    "greedy": "greedy_face",
    "end_turn": "end_turn",
    "end": "end_turn",
}


def _baseline_kind(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return _BASELINE_NAMES.get(name.lower())


class PolicyAdapter:  # noqa: D101 - структурный Protocol-образец (runtime-класс)
    """Документация контракта (структурная типизация — наследовать не обязательно).

    Атрибуты:
        name: имя модели/политики (для логов/манифеста).
        kind: ``random``|``greedy_face``|``end_turn``|``legacy_onnx``|
              ``action_onnx``|``v4``|``v5``|...
        model_path: путь к onnx или None (heuristic-политики).
        weights_hash: 16-символов sha256(.onnx) или None.
        weights_version: версия весов или None.
    """

    name: str
    kind: str
    model_path: Optional[str]
    weights_hash: Optional[str]
    weights_version: Optional[str]

    def select_action(self, engine: Any, player_id: int) -> int:  # pragma: no cover
        raise NotImplementedError


# ----------------------------------------------------------------------------
# V5 stub — зарезервированный слот (plumbing + hook). Block 0 (зона пользователя)
# реализует encode_observation_v5 и регистрирует свою фабрику через
# default_registry().register("v5", MyV5AdapterFactory), перезаписывая stub.
# ----------------------------------------------------------------------------

class V5StubAdapter:
    """Шаблон V5-адаптера. ``NotImplementedError`` направляет пользователя к
    ``encode_observation_v5`` (Block 0). Не падает молча — даёт явное сообщение,
    что V5-слот зарезервирован, но impl ещё не зарегистрирован."""

    kind = "v5"
    name = "v5-stub"
    model_path = None
    weights_hash = None
    weights_version = None

    def __init__(self, spec: Dict[str, Any]) -> None:
        self._spec = spec or {}
        self.name = self._spec.get("name", "v5-stub")
        self.model_path = self._spec.get("path")
        # weights_hash/weights_version оставляем None — у stub'а нет весов.

    def select_action(self, engine: Any, player_id: int) -> int:
        raise NotImplementedError(
            "V5 adapter is a reserved slot. Implement encode_observation_v5 + register your "
            "adapter via default_registry().register('v5', MyV5AdapterFactory) (Block 0, user "
            "zone). See policy_adapters.V5StubAdapter."
        )


# ----------------------------------------------------------------------------
# Built-in фабрики (defer к policy_factory-классам и gitignored layer A)
# ----------------------------------------------------------------------------

def _factory_random(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    from rlhf_env.components.policy_factory import _RLHFRandomPolicy
    return _RLHFRandomPolicy(seed=int(spec.get("seed", 0)))


def _factory_greedy_face(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    from rlhf_env.components.policy_factory import _RLHFGreedyFacePolicy
    return _RLHFGreedyFacePolicy()


def _factory_end_turn(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    from rlhf_env.components.policy_factory import _RLHFEndTurnPolicy
    return _RLHFEndTurnPolicy()


def _factory_legacy_onnx(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    from rlhf_env.components.policy_factory import _LegacyOnnxBotPolicy
    path = spec.get("path")
    if not path:
        raise ValueError("legacy_onnx adapter requires 'path' in spec")
    name = spec.get("name", "legacy_onnx")
    try:
        return _LegacyOnnxBotPolicy(path, name=name)
    except ModuleNotFoundError as exc:
        if "ai.model_benchmark" in str(exc):
            raise ValueError(
                f"legacy_onnx adapter for {name!r} requires ai.model_benchmark (gitignored "
                f"layer A) which is absent in this worktree. In prod it is available."
            ) from exc
        raise


def _factory_berserk(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    """V4 action-conditioned (и любой action_onnx) через BerserkInference."""
    from rlhf_env.components.policy_factory import (
        BOT_MAX_DIFFICULTY,
        _BerserkPolicyAdapter,
        _load_berserk,
        _load_sidecar,
    )
    from rlhf_env.components.inference_params import default_inference_params

    name = spec.get("name", "action_onnx")
    path = spec.get("path")
    if not path:
        raise ValueError("action_onnx/v4 adapter requires 'path' in spec")
    kind = spec.get("kind") or "action_onnx"
    difficulty = BOT_MAX_DIFFICULTY

    sidecar = _load_sidecar(path)
    params = default_inference_params(sidecar, kind)
    for k in ("temperature", "mode", "seed"):
        if k in spec:
            params[k] = spec[k]

    brain = _load_berserk(path, difficulty=difficulty, sidecar=sidecar)
    # Защита целостности логов (аудит SYN-1): молчаливый rule-based fallback
    # запрещён — иначе battle_log/manifest соврали бы, что играла onnx-модель.
    if not brain.has_profile(difficulty):
        raise RuntimeError(
            f"onnx-модель {name!r} ({path}) не загрузилась через BerserkInference "
            f"(difficulty={difficulty}) и молча ушла в rule-based fallback. "
            f"Скорее всего отсутствует sidecar-файл '{path}.json' с obs_dim/inputs/outputs. "
            f"Сгенерируйте sidecar либо используйте baseline (random/greedy_face/end_turn). "
            f"Молчаливый fallback запрещён — это нарушило бы целостность обучающих логов."
        )
    logger.info(
        "[policy_adapters] built BerserkPolicy name=%s kind=%s path=%s difficulty=%s",
        name, kind, path, difficulty,
    )
    return _BerserkPolicyAdapter(brain, difficulty=difficulty, name=name, kind=kind, path=path)


def _factory_v5(spec: Dict[str, Any], registry: Optional[Any]) -> Any:
    return V5StubAdapter(spec)


# ----------------------------------------------------------------------------
# Built-in детекторы kind
# ----------------------------------------------------------------------------

def _layer_a_fallback_detector(
    path: Optional[str], sidecar: Dict[str, Any], name: Optional[str]
) -> Optional[str]:
    """Fallback-детектор: defer к gitignored ``ai.model_benchmark.inspect_model``.
    В worktree без layer A → None (build поднимет явную ValueError). В prod →
    ``action_onnx``|``legacy_onnx``|``unknown``."""
    if not path:
        return None
    try:
        from ai.model_benchmark.policies import inspect_model
        return inspect_model(path).kind
    except Exception:  # noqa: BLE001 - layer A absent / inspect error → None
        return None


def _sidecar_kind_detector(
    path: Optional[str], sidecar: Dict[str, Any], name: Optional[str]
) -> Optional[str]:
    """Sidecar-based kind detector — работает БЕЗ gitignored layer A.

    V4-экспорты несут inference-контракт в sidecar (``.onnx.json``):
    ``model_version`` / ``inputs`` / ``outputs`` / ``action_feature_dim``
    достаточно, чтобы промаршрутизировать модель в ``action_onnx`` или
    ``legacy_onnx`` без inspect-проба. Если sidecar не указывает kind —
    возвращаем None, и цепочка детекторов идёт дальше (→ layer-A fallback).

    Ported from the earlier ``_derive_kind_from_sidecar`` iteration so prod
    RLHF deploys can scan V4 ONNX files without the benchmark-only inspector.
    """
    if not path or not sidecar:
        return None
    model_version = str(sidecar.get("model_version") or "")
    inputs = {str(n) for n in sidecar.get("inputs", [])}
    outputs = {str(n) for n in sidecar.get("outputs", [])}
    if (
        model_version == "classic_action_conditioned_onnx_v1"
        or {"observation", "action_features"}.issubset(inputs)
        or sidecar.get("action_feature_dim") is not None
    ):
        return "action_onnx"
    if model_version.startswith("classic_") and "action_conditioned" not in model_version:
        return "legacy_onnx"
    if inputs and "observation" in inputs and "logits" in outputs:
        return "legacy_onnx"
    return None


# ----------------------------------------------------------------------------
# Реестр
# ----------------------------------------------------------------------------

class AdapterRegistry:
    """Реестр фабрик адаптеров + детекторов kind.

    Единственная точка расширения: ``register(kind, factory)`` добавляет новый
    вид модели, ``register_detector(detector)`` — новый способ определения kind
    (V5 и др.), без правок if/elif в фабрике.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, AdapterFactory] = {}
        self._detectors: List[KindDetector] = []

    # -- фабрики -----------------------------------------------------------

    def register(self, kind: str, factory: AdapterFactory) -> None:
        """Регистрирует/перезаписывает фабрику для kind. V5-слот изначально
        указывает на ``V5StubAdapter``; пользователь перезаписывает его через
        ``register('v5', MyV5AdapterFactory)``."""
        self._factories[kind] = factory

    def resolve(self, kind: str) -> AdapterFactory:
        try:
            return self._factories[kind]
        except KeyError as exc:
            raise KeyError(f"unknown adapter kind: {kind!r}") from exc

    def has(self, kind: str) -> bool:
        return kind in self._factories

    def kinds(self) -> List[str]:
        return sorted(self._factories.keys())

    # -- детекторы ---------------------------------------------------------

    def register_detector(self, detector: KindDetector) -> None:
        """Добавляет детектор в начало списка (LIFO — приоритет над ранее
        зарегистрированными, вкл. layer-A fallback)."""
        self._detectors.insert(0, detector)

    def detect_kind(
        self, path: Optional[str], sidecar: Dict[str, Any], *, name: Optional[str] = None
    ) -> Optional[str]:
        for detector in self._detectors:
            try:
                kind = detector(path, sidecar or {}, name)
            except Exception:  # noqa: BLE001 - детектор не должен валить build
                logger.warning("[policy_adapters] detector %r raised", detector, exc_info=True)
                continue
            if kind:
                return kind
        return None

    # -- сборка ------------------------------------------------------------

    def build(self, spec: Dict[str, Any], *, registry: Optional[Any] = None) -> Any:
        """Создаёт политику по спеке через реестр (замена if/elif из build_policy).

        spec: {"name": str, "kind": str (optional), "path": str (optional),
               "seed": int (optional), ...}
        registry: PolicyRegistry (опционален). Если не передан и нужно резолвить
                  ONNX по имени — лениво создаётся PolicyRegistry.scan("ai/models").
        """
        name = spec.get("name")
        if not name:
            raise ValueError("policy spec requires 'name'")

        kind = spec.get("kind")
        path = spec.get("path")

        # 1. Baseline по имени — высший приоритет (как в исторической фабрике):
        #    random/greedy_face/end_turn не требуют файла даже с registry.
        #    F3: имя-baseline побеждает ВСЕГДА (когда kind не задан или совпадает),
        #    даже если в spec есть path — иначе name="random"+path уходил в
        #    auto-detect и строил ONNX вместо запрошенного baseline (регрессия).
        base_kind = _baseline_kind(name)
        if base_kind is not None and (not kind or kind == "auto" or kind == base_kind):
            return self.resolve(base_kind)(spec, registry)

        # 2. Ленивый registry, если нужно резолвить ONNX по имени.
        if registry is None and path is None and base_kind is None:
            from rlhf_env.components.policy_registry import PolicyRegistry
            registry = PolicyRegistry.scan("ai/models")

        # 3. Резолв имя → path+kind через registry (если path не задан явно).
        if path is None and registry is not None:
            resolved = registry.resolve_spec(name, override_kind=(kind if kind and kind != "auto" else None))
            path = resolved.get("path")
            if not kind or kind == "auto":
                cached = resolved.get("kind")
                # F1(audit): не clobber'им kind устаревшим 'unknown' из registry-кэша
                # (scan-time layer-A мог быть отсутствующим). Берём кэш только при
                # конкретном kind — иначе оставляем None/'auto', чтобы шаг 5
                # переопределил свежезарегистрированными детекторами (fresh-detect).
                if cached and cached != "unknown":
                    kind = cached

        # 4. Baseline через registry (resolve_spec отдал baseline без path).
        if path is None:
            if base_kind is not None:
                if not kind or kind == "auto":
                    kind = base_kind
                if kind == base_kind:
                    return self.resolve(base_kind)(spec, registry)
            raise ValueError(
                f"policy spec requires 'path' or registry resolution for {name!r}"
            )

        # 5. Auto-detect kind через детекторы (V5/пользовательские + layer-A fallback).
        if not kind or kind == "auto":
            from rlhf_env.components.policy_factory import _load_sidecar
            sidecar = _load_sidecar(path)
            detected = self.detect_kind(path, sidecar, name=name)
            if not detected:
                raise ValueError(
                    f"could not auto-detect kind for {path} (no detector matched; "
                    f"ai.model_benchmark layer A likely absent in this worktree)."
                )
            kind = detected

        if kind == "unknown":
            raise ValueError(f"model kind unknown for {path}")

        # 6. Фабрика по kind. Workflow-B find: пишем резолвнутые path/kind
        # обратно в spec — иначе _factory_berserk / _factory_legacy_onnx, читающие
        # spec.get('path'), получают None («action_onnx/v4 adapter requires 'path'»),
        # и зарегистрированная по имени onnx-модель не использовалась как p2_model.
        spec["path"] = path
        spec["kind"] = kind
        return self.resolve(kind)(spec, registry)


# ----------------------------------------------------------------------------
# Module-level singleton + built-ins
# ----------------------------------------------------------------------------

_DEFAULT_ADAPTERS: Optional[AdapterRegistry] = None


def default_registry() -> AdapterRegistry:
    """Глобальный реестр с зарегистрированными built-in адаптерами/детекторами.
    Пользователь расширяет его через ``register``/``register_detector``."""
    global _DEFAULT_ADAPTERS
    if _DEFAULT_ADAPTERS is None:
        reg = AdapterRegistry()
        _register_builtins(reg)
        _DEFAULT_ADAPTERS = reg
    return _DEFAULT_ADAPTERS


def _register_builtins(reg: AdapterRegistry) -> None:
    reg.register("random", _factory_random)
    reg.register("greedy_face", _factory_greedy_face)
    reg.register("end_turn", _factory_end_turn)
    reg.register("legacy_onnx", _factory_legacy_onnx)
    reg.register("action_onnx", _factory_berserk)
    reg.register("v4", _factory_berserk)
    reg.register("v5", _factory_v5)
    # layer-A fallback детектор — низший приоритет (добавлен первым, LIFO → последний).
    reg.register_detector(_layer_a_fallback_detector)
    # sidecar-детектор — выше layer-A fallback (LIFO): определяет V4/V3 по sidecar
    # без gitignored inspect_model. Добавлен последним → в голове списка.
    reg.register_detector(_sidecar_kind_detector)


__all__ = [
    "PolicyAdapter",
    "AdapterRegistry",
    "AdapterFactory",
    "KindDetector",
    "V5StubAdapter",
    "default_registry",
    "_baseline_kind",
]