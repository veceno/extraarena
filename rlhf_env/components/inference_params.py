"""Дефолты параметров инференса из sidecar (V4) или hardcoded (V3, baselines).

Используется в:
- policy_factory.build_policy(... overrides ...): применяет эти параметры
- manifest: фиксируется, с какими параметрами стартовали бои
- web UI: подставляются в форму как «пиковая мощность» модели
"""
from __future__ import annotations

from typing import Any, Dict


# Базовые дефолты для разных семейств политик
BASELINE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "random": {"mode": "argmax", "temperature": 1.0, "seed": 0},
    "random_legal": {"mode": "sample", "temperature": 1.0, "seed": 0},
    "greedy_face": {"mode": "argmax", "temperature": 1.0},
    "end_turn": {"mode": "argmax", "temperature": 1.0},
}

V3_LEGACY_DEFAULTS: Dict[str, Any] = {
    "mode": "argmax",
    "temperature": 1.0,
    "verify_mask": True,
}

V4_ACTION_DEFAULTS: Dict[str, Any] = {
    "mode": "argmax",
    "temperature": 1.0,
    "verify_mask": False,
    "placement_mode": "append_only",
    "include_preview_features": False,
}


def default_inference_params(sidecar: Dict[str, Any] | None, kind: str) -> Dict[str, Any]:
    """Возвращает дефолты параметров инференса на основе sidecar + kind.

    Args:
        sidecar: словарь из .onnx.json (или пустой {} если файла нет)
        kind:    "action_onnx" | "legacy_onnx" | "random" | "greedy_face" | "end_turn"
    """
    sidecar = sidecar or {}
    config = sidecar.get("config", {}) if isinstance(sidecar.get("config"), dict) else {}

    if kind in BASELINE_DEFAULTS:
        merged = {**BASELINE_DEFAULTS[kind], **sidecar, **{"kind": kind}}
        return merged

    if kind == "legacy_onnx":
        # V2/V3 нет нормальных дефолтов в sidecar — берём hardcoded.
        merged = {**V3_LEGACY_DEFAULTS, "kind": kind}
        merged["obs_dim"] = sidecar.get("obs_dim", 621)
        return merged

    if kind == "action_onnx":
        merged = {**V4_ACTION_DEFAULTS, "kind": kind}
        for k, v in sidecar.items():
            if k in V4_ACTION_DEFAULTS:
                merged[k] = v
        for k, v in config.items():
            if k in V4_ACTION_DEFAULTS:
                merged[k] = v
        merged.setdefault("obs_dim", int(sidecar.get("obs_dim", 1456)))
        merged.setdefault("action_feature_dim", int(sidecar.get("action_feature_dim", 171)))
        merged.setdefault("max_candidate_actions", int(sidecar.get("max_candidate_actions", 601)))
        return merged

    # Неизвестный kind — возвращаем минимальный набор
    return {"kind": kind, "mode": "argmax", "temperature": 1.0}


def describe_inference(params: Dict[str, Any]) -> str:
    """Короткое человеко-читаемое описание для UI и манифеста."""
    kind = params.get("kind", "?")
    mode = params.get("mode", "argmax")
    temp = params.get("temperature", 1.0)
    extras = []
    if "verify_mask" in params:
        extras.append(f"verify_mask={params['verify_mask']}")
    if "placement_mode" in params:
        extras.append(f"placement={params['placement_mode']}")
    extra_str = ", ".join(extras)
    return f"{kind} (mode={mode}, T={temp}" + (f", {extra_str}" if extra_str else "") + ")"


__all__ = [
    "BASELINE_DEFAULTS",
    "V3_LEGACY_DEFAULTS",
    "V4_ACTION_DEFAULTS",
    "default_inference_params",
    "describe_inference",
]