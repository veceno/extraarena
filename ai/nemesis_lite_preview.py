"""Standalone ONNX runtime for ExtraLR Nemesis Lite Preview.

The runtime intentionally does not import TrainV3.5 or gameplay state.  It
accepts only the two frozen initial decks and starting seat, while enforcing
the catalog/ruleset compatibility recorded by the ONNX sidecar.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort


ONNX_MANIFEST_SCHEMA = "extra_lr_nemesis_lite_preview_onnx_v1"
CLASS_NAMES = ("p1_win", "draw", "p2_win")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "ai" / "cards.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NemesisLitePreview:
    """Validated owner for one Nemesis Lite Preview ONNX session."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        ruleset: str,
        catalog_path: str | Path = DEFAULT_CATALOG,
    ) -> None:
        self.artifact = Path(artifact).expanduser().resolve()
        sidecar_path = Path(str(self.artifact) + ".json")
        if not self.artifact.is_file():
            raise FileNotFoundError(self.artifact)
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        self.sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if self.sidecar.get("schema") != ONNX_MANIFEST_SCHEMA:
            raise ValueError(f"{sidecar_path}: unsupported Nemesis ONNX schema")
        if self.sidecar.get("artifact_sha256") != _sha256(self.artifact):
            raise ValueError(f"{self.artifact}: sha256 mismatch")
        if self.sidecar.get("class_order") != list(CLASS_NAMES):
            raise ValueError(f"{sidecar_path}: class order mismatch")
        catalog = self.sidecar.get("catalog") or {}
        live_catalog = Path(catalog_path).expanduser().resolve()
        if catalog.get("sha256") != _sha256(live_catalog):
            raise ValueError("Nemesis catalog mismatch")
        declared_ruleset = (self.sidecar.get("ruleset") or {}).get("id")
        if str(ruleset) != declared_ruleset:
            raise ValueError(
                f"Nemesis ruleset mismatch: expected {declared_ruleset!r}"
            )
        self.ruleset = str(ruleset)
        self.deck_size = int(
            self.sidecar["inputs"]["p1_card_ids"][-1]
        )
        self._card_ids = {
            int(card_id)
            for card_id in catalog.get("ordered_card_ids") or []
        }
        self._hero_ids = {
            int(card_id)
            for card_id in catalog.get("hero_card_ids") or []
        }
        self._max_levels = {
            int(card_id): int(maximum)
            for card_id, maximum in (catalog.get("level_policy") or {}).items()
        }
        if self._card_ids != set(self._max_levels):
            raise ValueError("Nemesis sidecar level policy is incomplete")
        self._session: ort.InferenceSession | None = ort.InferenceSession(
            str(self.artifact),
            providers=["CPUExecutionProvider"],
        )
        graph_inputs = {
            item.name: item
            for item in self._session.get_inputs()
        }
        expected_inputs = {
            "p1_card_ids",
            "p1_levels",
            "p2_card_ids",
            "p2_levels",
            "starting_side",
        }
        if set(graph_inputs) != expected_inputs:
            raise ValueError("Nemesis ONNX input names do not match the contract")
        graph_outputs = {
            item.name
            for item in self._session.get_outputs()
        }
        if graph_outputs != {
            "outcome_logits",
            "outcome_probabilities",
        }:
            raise ValueError("Nemesis ONNX output names do not match the contract")

    def _deck_arrays(
        self,
        deck: Sequence[Mapping[str, Any]],
        *,
        field: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(deck) != self.deck_size:
            raise ValueError(
                f"{field} must contain exactly {self.deck_size} cards"
            )
        ids: list[int] = []
        levels: list[int] = []
        for index, card in enumerate(deck):
            if not isinstance(card, Mapping):
                raise ValueError(f"{field}[{index}] must be an object")
            card_id = card.get("card_id")
            level = card.get("level")
            if (
                not isinstance(card_id, int)
                or isinstance(card_id, bool)
                or card_id not in self._card_ids
            ):
                raise ValueError(f"{field}[{index}].card_id is invalid")
            maximum = self._max_levels[card_id]
            if (
                not isinstance(level, int)
                or isinstance(level, bool)
                or not 1 <= level <= maximum
            ):
                raise ValueError(
                    f"{field}[{index}].level must be in [1, {maximum}]"
                )
            ids.append(card_id)
            levels.append(level)
        if len(set(ids)) != len(ids):
            raise ValueError(f"{field} contains duplicate card ids")
        if sum(card_id in self._hero_ids for card_id in ids) != 1:
            raise ValueError(f"{field} must contain exactly one hero")
        return (
            np.asarray([ids], dtype=np.int64),
            np.asarray([levels], dtype=np.int64),
        )

    def predict(
        self,
        *,
        p1_deck: Sequence[Mapping[str, Any]],
        p2_deck: Sequence[Mapping[str, Any]],
        starting_player: str,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Nemesis Lite Preview session is closed")
        if starting_player not in {"p1", "p2"}:
            raise ValueError("starting_player must be p1 or p2")
        p1_ids, p1_levels = self._deck_arrays(p1_deck, field="p1_deck")
        p2_ids, p2_levels = self._deck_arrays(p2_deck, field="p2_deck")
        logits, probabilities = self._session.run(
            ["outcome_logits", "outcome_probabilities"],
            {
                "p1_card_ids": p1_ids,
                "p1_levels": p1_levels,
                "p2_card_ids": p2_ids,
                "p2_levels": p2_levels,
                "starting_side": np.asarray(
                    [[1 if starting_player == "p1" else 0]],
                    dtype=np.int64,
                ),
            },
        )
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        ).reshape(-1)
        if (
            logits.shape != (3,)
            or probabilities.shape != (3,)
            or not np.isfinite(logits).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not math.isclose(
                float(probabilities.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
        ):
            raise RuntimeError("Nemesis ONNX returned an invalid distribution")
        return {
            "model_id": self.sidecar["model_id"],
            "status": self.sidecar["status"],
            "ruleset": self.ruleset,
            "class_order": list(CLASS_NAMES),
            "logits": {
                name: float(logits[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "predicted_outcome": CLASS_NAMES[int(np.argmax(probabilities))],
        }

    def close(self) -> None:
        self._session = None


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_CATALOG",
    "NemesisLitePreview",
    "ONNX_MANIFEST_SCHEMA",
]
