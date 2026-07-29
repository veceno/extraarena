"""Export Nemesis Lite Preview NPZ artifacts to a validated ONNX contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .nemesis_lite_preview import (
    CLASS_NAMES,
    MODEL_ID,
    NemesisLitePreviewModel,
    load_model_artifact,
    sha256_file,
)


ONNX_MANIFEST_SCHEMA = "extra_lr_nemesis_lite_preview_onnx_v1"


class _OnnxContract(nn.Module):
    def __init__(self, model: NemesisLitePreviewModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        p1_card_ids: torch.Tensor,
        p1_levels: torch.Tensor,
        p2_card_ids: torch.Tensor,
        p2_levels: torch.Tensor,
        starting_side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(
            p1_card_ids,
            p1_levels,
            p2_card_ids,
            p2_levels,
            starting_side,
        )
        return logits, torch.softmax(logits, dim=-1)


def _parity_batch(
    metadata: dict[str, Any],
    *,
    rows: int = 32,
) -> dict[str, np.ndarray]:
    catalog = metadata.get("catalog") or {}
    card_ids = np.asarray(catalog.get("ordered_card_ids") or [], dtype=np.int64)
    level_policy = catalog.get("level_policy") or {}
    deck_size = int(metadata["architecture"]["config"]["deck_size"])
    if card_ids.size < deck_size:
        raise ValueError("training manifest catalog cannot fill one deck")
    generator = np.random.default_rng(20260728)

    def decks() -> tuple[np.ndarray, np.ndarray]:
        ids = np.stack(
            [
                generator.choice(card_ids, size=deck_size, replace=False)
                for _ in range(rows)
            ]
        ).astype(np.int64)
        levels = np.empty_like(ids)
        for row_index in range(rows):
            for column, card_id in enumerate(ids[row_index]):
                maximum = int(level_policy.get(str(int(card_id)), 10))
                levels[row_index, column] = generator.integers(
                    1,
                    maximum + 1,
                )
        return ids, levels

    p1_card_ids, p1_levels = decks()
    p2_card_ids, p2_levels = decks()
    return {
        "p1_card_ids": p1_card_ids,
        "p1_levels": p1_levels,
        "p2_card_ids": p2_card_ids,
        "p2_levels": p2_levels,
        "starting_side": generator.integers(
            0,
            2,
            size=(rows, 1),
            dtype=np.int64,
        ),
    }


def _assert_parity_and_swap(
    output_path: Path,
    contract: _OnnxContract,
    metadata: dict[str, Any],
) -> dict[str, float]:
    feeds = _parity_batch(metadata)
    torch_inputs = tuple(
        torch.from_numpy(feeds[name])
        for name in (
            "p1_card_ids",
            "p1_levels",
            "p2_card_ids",
            "p2_levels",
            "starting_side",
        )
    )
    with torch.no_grad():
        expected_logits, expected_probabilities = contract(*torch_inputs)
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    actual_logits, actual_probabilities = session.run(
        ["outcome_logits", "outcome_probabilities"],
        feeds,
    )
    logits_error = float(
        np.max(
            np.abs(
                expected_logits.numpy().astype(np.float32)
                - actual_logits
            )
        )
    )
    probabilities_error = float(
        np.max(
            np.abs(
                expected_probabilities.numpy().astype(np.float32)
                - actual_probabilities
            )
        )
    )
    swapped_feeds = {
        "p1_card_ids": feeds["p2_card_ids"],
        "p1_levels": feeds["p2_levels"],
        "p2_card_ids": feeds["p1_card_ids"],
        "p2_levels": feeds["p1_levels"],
        "starting_side": 1 - feeds["starting_side"],
    }
    swapped_probabilities = session.run(
        ["outcome_probabilities"],
        swapped_feeds,
    )[0]
    restored = swapped_probabilities[:, [2, 1, 0]]
    swap_error = float(np.max(np.abs(actual_probabilities - restored)))
    if max(logits_error, probabilities_error) >= 2.0e-5:
        raise AssertionError(
            f"{output_path.name}: ONNX parity drift "
            f"{max(logits_error, probabilities_error):.3e}"
        )
    if swap_error >= 2.0e-5:
        raise AssertionError(
            f"{output_path.name}: swap consistency drift {swap_error:.3e}"
        )
    return {
        "max_abs_logits_parity_error": logits_error,
        "max_abs_probabilities_parity_error": probabilities_error,
        "max_abs_swap_consistency_error": swap_error,
    }


def export_nemesis_lite_preview(
    source_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model, training_metadata = load_model_artifact(source)
    model.eval()
    contract = _OnnxContract(model).eval()
    deck_size = model.config.deck_size
    example = (
        torch.ones(1, deck_size, dtype=torch.int64),
        torch.ones(1, deck_size, dtype=torch.int64),
        torch.ones(1, deck_size, dtype=torch.int64),
        torch.ones(1, deck_size, dtype=torch.int64),
        torch.ones(1, 1, dtype=torch.int64),
    )
    input_names = [
        "p1_card_ids",
        "p1_levels",
        "p2_card_ids",
        "p2_levels",
        "starting_side",
    ]
    torch.onnx.export(
        contract,
        example,
        str(output),
        input_names=input_names,
        output_names=["outcome_logits", "outcome_probabilities"],
        dynamic_axes={
            **{name: {0: "batch"} for name in input_names},
            "outcome_logits": {0: "batch"},
            "outcome_probabilities": {0: "batch"},
        },
        opset_version=int(opset),
        dynamo=False,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    parity = _assert_parity_and_swap(output, contract, training_metadata)
    sidecar = {
        "schema": ONNX_MANIFEST_SCHEMA,
        "model_id": MODEL_ID,
        "status": "preview",
        "format": "onnx",
        "opset": int(opset),
        "artifact": output.name,
        "artifact_sha256": sha256_file(output),
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": sha256_file(source),
        "catalog": training_metadata["catalog"],
        "ruleset": training_metadata["ruleset"],
        "inputs": {
            "p1_card_ids": ["batch", deck_size],
            "p1_levels": ["batch", deck_size],
            "p2_card_ids": ["batch", deck_size],
            "p2_levels": ["batch", deck_size],
            "starting_side": ["batch", 1],
        },
        "input_dtypes": {
            name: "int64"
            for name in input_names
        },
        "outputs": {
            "outcome_logits": ["batch", len(CLASS_NAMES)],
            "outcome_probabilities": ["batch", len(CLASS_NAMES)],
        },
        "class_order": list(CLASS_NAMES),
        "starting_side_values": {"p2": 0, "p1": 1},
        "swap_contract": (
            "swap p1/p2 deck tensors, flip starting_side, exchange "
            "p1_win/p2_win output columns"
        ),
        "parity": parity,
        "training": {
            "metrics": training_metadata["training"]["metrics"],
            "split": training_metadata["split"],
            "dataset_contract": training_metadata["dataset_contract"],
        },
    }
    sidecar_path = Path(str(output) + ".json")
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "sidecar": str(sidecar_path),
        "sha256": sidecar["artifact_sha256"],
        **parity,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    result = export_nemesis_lite_preview(
        args.source,
        args.output,
        opset=args.opset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "ONNX_MANIFEST_SCHEMA",
    "export_nemesis_lite_preview",
]
