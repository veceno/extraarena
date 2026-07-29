#!/usr/bin/env python3
"""Build and validate the production ExtraLR model manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ai" / "models"
CATALOG_PATH = ROOT / "ai" / "cards.json"
MANIFEST_PATH = MODEL_DIR / "manifest.json"


MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "extra-lr-v4-micro",
        "artifact": "extra-lr-v4-micro.onnx",
        "adapter": "v4",
        "status": "stable",
        "role": "newcomer_default",
    },
    {
        "id": "extra-lr-v4-lite",
        "artifact": "extra-lr-v4-lite.onnx",
        "adapter": "v4",
        "status": "stable",
        "role": "benchmark_reference",
    },
    {
        "id": "extra-lr-v4-opti",
        "artifact": "extra-lr-v4-opti.onnx",
        "adapter": "v4",
        "status": "stable",
        "role": "benchmark_reference",
    },
    {
        "id": "extra-lr-v4-max",
        "artifact": "extra-lr-v4-max.onnx",
        "adapter": "v4",
        "status": "stable",
        "role": "benchmark_reference",
    },
    {
        "id": "extra-lr-v5-lite",
        "artifact": "extra-lr-v5-lite.onnx",
        "adapter": "v5",
        "status": "stable",
        "role": "live_policy",
        "checkpoint_alias": "u18500",
    },
    {
        "id": "extra-lr-v5",
        "artifact": "extra-lr-v5.onnx",
        "adapter": "v5",
        "status": "stable",
        "role": "live_policy",
        "checkpoint_alias": "h299",
    },
    {
        "id": "extra-lr-assembler-v1",
        "artifact": "extra_lr_assembler_v1.onnx",
        "adapter": "onnx_aux_v1",
        "status": "stable",
        "role": "ultra_deck_assembler",
        "training_readiness": "bootstrap_ready",
    },
    {
        "id": "extra-lr-cardoptimum-v1",
        "artifact": "extra_lr_cardoptimum_v1.onnx",
        "adapter": "onnx_aux_v1",
        "status": "beta",
        "role": "ultra_draw_assistant",
        "training_readiness": "candidate_ready",
    },
    {
        "id": "extra-lr-metronome-v1",
        "artifact": "extra_lr_metronome_v1.onnx",
        "adapter": "onnx_aux_v1",
        "status": "beta",
        "role": "all_live_policy_decision_timing",
        "training_readiness": "candidate_ready",
    },
    {
        "id": "extra-lr-timestamp-v1-mono",
        "artifact": "extra_lr_timestamp_v1_mono.onnx",
        "adapter": "onnx_aux_v1",
        "status": "experimental",
        "role": "duration_estimator",
        "training_readiness": "experimental",
    },
    {
        "id": "extra-lr-timestamp-v1-duo",
        "artifact": "extra_lr_timestamp_v1_duo.onnx",
        "adapter": "onnx_aux_v1",
        "status": "experimental",
        "role": "duration_estimator",
        "training_readiness": "experimental",
    },
    {
        "id": "extra-lr-nemesis-lite-preview",
        "artifact": "extra_lr_nemesis_lite_preview.onnx",
        "adapter": "nemesis_lite_v1",
        "status": "preview",
        "role": "deck_match_outcome_estimator",
        "training_readiness": "preview_binary_strong_draw_sparse",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_schema(value_info: Any) -> list[Any]:
    dims: list[Any] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def _model_entry(spec: dict[str, Any]) -> dict[str, Any]:
    path = MODEL_DIR / str(spec["artifact"])
    if not path.is_file():
        raise FileNotFoundError(path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph
    sidecar_path = Path(str(path) + ".json")
    sidecar = (
        json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar_path.is_file()
        else {}
    )
    source_checkpoint = sidecar.get("source_checkpoint")
    if source_checkpoint:
        # Old V4 sidecars predate the portable provenance contract and contain
        # absolute developer-machine paths.  A production manifest must remain
        # relocatable, just like the V5/aux sidecars exported below.
        source_checkpoint = Path(str(source_checkpoint)).name
    return {
        **spec,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "opset": int(model.opset_import[0].version),
        "inputs": {
            value.name: _tensor_schema(value)
            for value in graph.input
        },
        "outputs": {
            value.name: _tensor_schema(value)
            for value in graph.output
        },
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": sidecar.get("source_checkpoint_sha256"),
    }


def build_manifest() -> dict[str, Any]:
    cards = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    card_ids = sorted(int(card["id"]) for card in cards)
    if len(card_ids) != 50 or len(set(card_ids)) != 50:
        raise ValueError(
            f"production catalog contract requires 50 unique cards, got {len(card_ids)}"
        )
    return {
        "schema": "extra_lr_production_model_manifest_v1",
        "generated_at": "2026-07-28",
        "runtime_adapters": [
            "v4",
            "v5",
            "onnx_aux_v1",
            "nemesis_lite_v1",
        ],
        "game_policy_adapters": ["v4", "v5"],
        "card_catalog": {
            "path": "ai/cards.json",
            "sha256": _sha256(CATALOG_PATH),
            "card_count": len(card_ids),
            "ordered_card_ids": card_ids,
        },
        "live_progression": [
            {"trophies": "0-299", "profile": "extra-lr-v4-micro"},
            {"trophies": "300-1199", "profile": "extra-lr-v5-lite"},
            {"trophies": "1200-4499", "profile": "extra-lr-v5"},
            {"trophies": "4500+", "profile": "extra-lr-v5-ultra"},
        ],
        "composites": [
            {
                "id": "extra-lr-v5-ultra",
                "status": "beta_assisted",
                "policy": "extra-lr-v5",
                "components": [
                    "extra-lr-assembler-v1",
                    "extra-lr-cardoptimum-v1",
                    "extra-lr-metronome-v1",
                ],
                "notes": (
                    "V5 Ultra shares the h299 policy ONNX with ExtraLR V5; "
                    "its match-scoped Assembler and CardOptimum assistants "
                    "define the Ultra runtime profile."
                ),
            }
        ],
        "models": [_model_entry(spec) for spec in MODEL_SPECS],
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
