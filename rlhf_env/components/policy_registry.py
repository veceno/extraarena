"""Реестр доступных ONNX-моделей для RLHF-среды.

Сканирует указанную директорию на *.onnx + *.onnx.json sidecar-файлы,
определяет kind через inspect_model (V4 action-conditioned / V2/V3 legacy / random / etc.),
формирует спеки для policy_factory.

Использование:
    registry = PolicyRegistry.scan(models_dir=Path("ai/models"))
    for spec in registry.list_specs():
        print(spec["name"], spec["kind"])
    registry.save_index(Path("rlhf_env/state/registry_index.json"))
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Спека одной модели для UI / манифеста / MCP."""

    name: str  # e.g. "extra-lr-v4-max"
    path: str  # абсолютный путь к .onnx
    sidecar_path: Optional[str]
    kind: str  # action_onnx | legacy_onnx | random | greedy_face | end_turn | unknown
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_builtin_baseline(name: str) -> Optional[Dict[str, Any]]:
    """Эти «модели» не требуют файла — это built-in baselines из policy_factory."""
    name_lower = name.lower()
    if name_lower in {"random", "random_legal"}:
        return {"kind": "random", "description": "Случайные легальные действия"}
    if name_lower in {"greedy_face", "greedy"}:
        return {"kind": "greedy_face", "description": "Атака по лицу (heuristic)"}
    if name_lower in {"end_turn", "end"}:
        return {"kind": "end_turn", "description": "Всегда завершает ход"}
    return None


def scan_directory(models_dir: Path | str) -> List[ModelSpec]:
    """Сканирует директорию на .onnx + sidecar и возвращает список спеков."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.warning("[PolicyRegistry] models dir does not exist: %s", models_dir)
        return []

    specs: List[ModelSpec] = []
    seen: set[str] = set()

    for onnx_path in sorted(models_dir.glob("*.onnx")):
        name = onnx_path.stem
        sidecar_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
        sidecar: Dict[str, Any] = {}
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[PolicyRegistry] bad sidecar %s: %s", sidecar_path, exc)

        # Kind определяется через модульный реестр детекторов (policy_adapters):
        # pluggable V5/пользовательские детекторы + layer-A fallback (defer к
        # gitignored ai.model_benchmark.inspect_model). В worktree без layer A
        # fallback вернёт None → kind="unknown" (как раньше, без падения).
        from rlhf_env.components.policy_adapters import default_registry

        kind = default_registry().detect_kind(str(onnx_path), sidecar, name=name)
        if not kind:
            logger.warning(
                "[PolicyRegistry] kind not detected for %s (layer A absent?)", onnx_path
            )
            kind = "unknown"

        description = sidecar.get("model_version") or sidecar.get("source_checkpoint") or ""
        extra = {
            "obs_dim": sidecar.get("obs_dim"),
            "action_feature_dim": sidecar.get("action_feature_dim"),
            "max_candidate_actions": sidecar.get("max_candidate_actions"),
            "input_names": list(sidecar.get("inputs", [])),
            "output_names": list(sidecar.get("outputs", [])),
        }

        spec = ModelSpec(
            name=name,
            path=str(onnx_path.resolve()),
            sidecar_path=str(sidecar_path.resolve()) if sidecar_path.exists() else None,
            kind=kind,
            description=description,
            extra=extra,
        )
        specs.append(spec)
        seen.add(name)

    return specs


class PolicyRegistry:
    """Реестр моделей + baseline-спеки."""

    def __init__(self, specs: List[ModelSpec]):
        self.specs = list(specs)
        self._name_index = {s.name: s for s in specs}

    @classmethod
    def scan(cls, models_dir: Path | str) -> "PolicyRegistry":
        """Фабрика: сканирует директорию и возвращает готовый реестр."""
        return cls(scan_directory(models_dir))

    def list_specs(self) -> List[Dict[str, Any]]:
        """Возвращает спеки + baseline-спеки для UI."""
        out: List[Dict[str, Any]] = []
        for s in self.specs:
            out.append(s.to_dict())
        # Добавим встроенные baselines (без файла)
        for baseline_name, info in [
            ("random", {"kind": "random", "description": "Случайные легальные действия"}),
            ("greedy_face", {"kind": "greedy_face", "description": "Атака по лицу"}),
            ("end_turn", {"kind": "end_turn", "description": "Всегда завершает ход"}),
        ]:
            out.append({
                "name": baseline_name,
                "path": None,
                "sidecar_path": None,
                "kind": info["kind"],
                "description": info["description"],
                "extra": {},
            })
        return out

    def get(self, name: str) -> Optional[ModelSpec]:
        """Поиск модели по имени. Возвращает None для baselines (kind != path-based)."""
        return self._name_index.get(name)

    def resolve_spec(
        self,
        name: str,
        *,
        override_kind: Optional[str] = None,
        override_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Резолвит имя → спеку для policy_factory.

        Args:
            name: имя модели (e.g. "extra-lr-v4-max" / "random" / "greedy_face")
            override_kind: форсировать kind (для unit-тестов / специальных прогонов)
            override_path: путь к onnx для custom-модели by-path. Если имя не
                найдено в реестре и override_path задан — возвращается спека с
                этим путём без KeyError (custom model by path+adapter).

        Raises:
            KeyError: если имя не найдено, это не baseline и override_path не задан
        """
        # 1) Baseline
        baseline = _is_builtin_baseline(name)
        if baseline is not None:
            spec: Dict[str, Any] = {"name": name, "kind": baseline["kind"]}
            if override_kind:
                spec["kind"] = override_kind
            return spec

        # 2) ONNX из реестра
        spec_obj = self.get(name)
        if spec_obj is None:
            if override_path:
                # Custom model by path+adapter — не падаем, отдаём спеку с путём.
                return {
                    "name": name,
                    "kind": override_kind or "auto",
                    "path": override_path,
                }
            raise KeyError(f"model not found in registry: {name!r}")

        spec = {"name": name, "kind": override_kind or spec_obj.kind, "path": spec_obj.path}
        return spec

    def add_spec(self, spec: ModelSpec) -> None:
        """In-memory добавление спеки (для MCP ``register_custom_model``).
        Не персистит в index.json — кастомные модели живут до рестарта процесса."""
        self.specs.append(spec)
        self._name_index[spec.name] = spec

    def save_index(self, dest_path: Path | str) -> None:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.list_specs(), indent=2, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        # F3(audit): считаем по факту из list_specs (specs + built-in baselines),
        # а не magic +3 — чтобы计数 не разъехался с реальным набором baselines.
        return len(self.list_specs())


__all__ = ["ModelSpec", "PolicyRegistry", "scan_directory"]