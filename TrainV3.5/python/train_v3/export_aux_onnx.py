"""Export the ExtraLR V1 auxiliary ridge models to production ONNX.

The Phase-C auxiliary trainers deliberately emit framework-neutral ``.npz``
artifacts.  This module is the single audited bridge from those artifacts to
the ONNX contracts consumed by the game runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch


_RIDGE_SPECS = {
    "assembler": {
        "stem": "extra_lr_assembler_v1",
        "output": "matchup_score",
    },
    "cardoptimum": {
        "stem": "extra_lr_cardoptimum_v1",
        "output": "card_score",
    },
    "metronome": {
        "stem": "extra_lr_metronome_v1",
        "output": "predicted_log_ms",
    },
}

_TIMESTAMP_SPECS = {
    "timestamp_mono": "extra_lr_timestamp_v1_mono",
    "timestamp_duo": "extra_lr_timestamp_v1_duo",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


class _Ridge(torch.nn.Module):
    def __init__(self, arrays: dict[str, np.ndarray]):
        super().__init__()
        for name in ("feature_mean", "feature_scale", "coef", "intercept"):
            self.register_buffer(
                name,
                torch.from_numpy(np.asarray(arrays[name], dtype=np.float32)),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return (normalized @ self.coef + self.intercept[0]).unsqueeze(-1)


class _TimeStamp(torch.nn.Module):
    """Two-stage TimeStamp model.

    ``duration_context`` is the duration feature vector without its leading
    predicted-log-turns field.  Keeping the first stage inside ONNX prevents a
    serving implementation from accidentally using a differently rounded turn
    prediction in the duration calibrator.
    """

    def __init__(self, arrays: dict[str, np.ndarray]):
        super().__init__()
        for prefix in ("turn_", "duration_"):
            for suffix in ("feature_mean", "feature_scale", "coef", "intercept"):
                name = f"{prefix}{suffix}"
                self.register_buffer(
                    name,
                    torch.from_numpy(np.asarray(arrays[name], dtype=np.float32)),
                )

    def forward(
        self,
        turn_features: torch.Tensor,
        duration_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        turn_normalized = (
            turn_features - self.turn_feature_mean
        ) / self.turn_feature_scale
        predicted_log_turns = (
            turn_normalized @ self.turn_coef + self.turn_intercept[0]
        ).unsqueeze(-1)
        duration_features = torch.cat(
            [predicted_log_turns, duration_context],
            dim=-1,
        )
        duration_normalized = (
            duration_features - self.duration_feature_mean
        ) / self.duration_feature_scale
        predicted_log_duration = (
            duration_normalized @ self.duration_coef
            + self.duration_intercept[0]
        ).unsqueeze(-1)
        return predicted_log_turns, predicted_log_duration


def _source_metadata(npz_path: Path) -> dict[str, Any]:
    json_path = npz_path.with_suffix(".json")
    if not json_path.is_file():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    payload.pop("artifact", None)
    return payload


def _write_sidecar(
    output_path: Path,
    *,
    kind: str,
    source_path: Path,
    contract: dict[str, Any],
) -> None:
    payload = {
        "schema": "extra_lr_aux_onnx_v1",
        "kind": kind,
        "format": "onnx",
        "opset": 17,
        "source_checkpoint": source_path.name,
        "source_checkpoint_sha256": _sha256(source_path),
        **contract,
        "training": _source_metadata(source_path),
    }
    Path(str(output_path) + ".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_ridge_parity(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    output_name: str,
) -> float:
    rng = np.random.default_rng(20260728)
    features = rng.normal(
        size=(32, int(arrays["feature_mean"].shape[0]))
    ).astype(np.float32)
    expected = (
        (features - arrays["feature_mean"]) / arrays["feature_scale"]
    ) @ arrays["coef"] + float(arrays["intercept"][0])
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run([output_name], {"features": features})[0].reshape(-1)
    max_abs = float(np.max(np.abs(expected.astype(np.float32) - actual)))
    if max_abs >= 2.0e-5:
        raise AssertionError(f"{output_path.name}: ONNX parity drift {max_abs:.3e}")
    return max_abs


def _assert_timestamp_parity(
    output_path: Path,
    arrays: dict[str, np.ndarray],
) -> float:
    rng = np.random.default_rng(20260728)
    turn_features = rng.normal(
        size=(32, int(arrays["turn_feature_mean"].shape[0]))
    ).astype(np.float32)
    context_dim = int(arrays["duration_feature_mean"].shape[0]) - 1
    duration_context = rng.normal(size=(32, context_dim)).astype(np.float32)
    expected_turn = (
        (turn_features - arrays["turn_feature_mean"])
        / arrays["turn_feature_scale"]
    ) @ arrays["turn_coef"] + float(arrays["turn_intercept"][0])
    duration_features = np.concatenate(
        [expected_turn[:, None], duration_context],
        axis=1,
    )
    expected_duration = (
        (duration_features - arrays["duration_feature_mean"])
        / arrays["duration_feature_scale"]
    ) @ arrays["duration_coef"] + float(arrays["duration_intercept"][0])
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    actual_turn, actual_duration = session.run(
        ["predicted_log_turns", "predicted_log_duration"],
        {
            "turn_features": turn_features,
            "duration_context": duration_context,
        },
    )
    max_abs = max(
        float(np.max(np.abs(expected_turn.astype(np.float32) - actual_turn.reshape(-1)))),
        float(
            np.max(
                np.abs(
                    expected_duration.astype(np.float32)
                    - actual_duration.reshape(-1)
                )
            )
        ),
    )
    if max_abs >= 2.0e-5:
        raise AssertionError(f"{output_path.name}: ONNX parity drift {max_abs:.3e}")
    return max_abs


def export_aux_model(
    source_path: str | Path,
    output_path: str | Path,
    *,
    kind: str,
    opset: int = 17,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _load_npz(source)

    if kind in _RIDGE_SPECS:
        output_name = str(_RIDGE_SPECS[kind]["output"])
        model = _Ridge(arrays).eval()
        feature_dim = int(arrays["feature_mean"].shape[0])
        torch.onnx.export(
            model,
            (torch.zeros(1, feature_dim),),
            str(output),
            input_names=["features"],
            output_names=[output_name],
            dynamic_axes={
                "features": {0: "batch"},
                output_name: {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
        max_abs = _assert_ridge_parity(
            output,
            arrays,
            output_name=output_name,
        )
        contract = {
            "inputs": {"features": [None, feature_dim]},
            "outputs": {output_name: [None, 1]},
            "postprocess": (
                "expm1_clamp_100_25000_ms"
                if kind == "metronome"
                else "identity"
            ),
        }
        if "residual_log_quantiles" in arrays:
            contract["residual_log_quantiles"] = (
                arrays["residual_log_quantiles"].astype(float).tolist()
            )
    elif kind in _TIMESTAMP_SPECS:
        model = _TimeStamp(arrays).eval()
        turn_dim = int(arrays["turn_feature_mean"].shape[0])
        context_dim = int(arrays["duration_feature_mean"].shape[0]) - 1
        torch.onnx.export(
            model,
            (
                torch.zeros(1, turn_dim),
                torch.zeros(1, context_dim),
            ),
            str(output),
            input_names=["turn_features", "duration_context"],
            output_names=["predicted_log_turns", "predicted_log_duration"],
            dynamic_axes={
                "turn_features": {0: "batch"},
                "duration_context": {0: "batch"},
                "predicted_log_turns": {0: "batch"},
                "predicted_log_duration": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
        max_abs = _assert_timestamp_parity(output, arrays)
        contract = {
            "inputs": {
                "turn_features": [None, turn_dim],
                "duration_context": [None, context_dim],
            },
            "outputs": {
                "predicted_log_turns": [None, 1],
                "predicted_log_duration": [None, 1],
            },
            "postprocess": "expm1",
            "duration_residual_log_quantiles": arrays[
                "duration_residual_log_quantiles"
            ]
            .astype(float)
            .tolist(),
        }
    else:
        raise ValueError(f"unsupported auxiliary model kind: {kind}")

    _write_sidecar(
        output,
        kind=kind,
        source_path=source,
        contract=contract,
    )
    return {
        "kind": kind,
        "output": str(output),
        "sha256": _sha256(output),
        "max_abs_parity_error": max_abs,
    }


def export_aux_bundle(
    source_dir: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    results: list[dict[str, Any]] = []
    for kind, spec in _RIDGE_SPECS.items():
        stem = str(spec["stem"])
        results.append(
            export_aux_model(
                source / f"{stem}.npz",
                output / f"{stem}.onnx",
                kind=kind,
            )
        )
    for kind, stem in _TIMESTAMP_SPECS.items():
        results.append(
            export_aux_model(
                source / f"{stem}.npz",
                output / f"{stem}.onnx",
                kind=kind,
            )
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_aux_bundle(args.source_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()


__all__ = ["export_aux_bundle", "export_aux_model"]
