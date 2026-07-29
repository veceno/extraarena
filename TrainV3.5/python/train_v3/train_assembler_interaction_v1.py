"""Train the opponent-conditioned ExtraLR Assembler V1 artifact.

The original V1 ridge concatenated candidate and opponent deck vectors.  A
linear model over that representation can only add an opponent-specific
constant to every candidate score, so it cannot change a candidate ranking.

This trainer preserves the audited base representation and adds a 50x50
bilinear counter-card matrix.  The interaction is fitted to the residual of
the base model and validation-selects both regularization and shrinkage.  The
result is still a single portable linear artifact consumable by the existing
ONNX exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CARD_CATALOG = {
    int(card["id"]): card
    for card in json.loads(
        (ROOT / "ai" / "cards.json").read_text(encoding="utf-8")
    )
}
CARD_IDS = tuple(sorted(CARD_CATALOG))
CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
SIMPLIFIED_LEVEL_CARD_IDS = frozenset(
    card_id
    for card_id, card in CARD_CATALOG.items()
    if bool(card.get("simplified_levelup", False))
)
BASE_FEATURE_DIM = len(CARD_IDS) * 5 + 3
INTERACTION_FEATURE_DIM = len(CARD_IDS) ** 2
FEATURE_DIM = BASE_FEATURE_DIM + INTERACTION_FEATURE_DIM

DEFAULT_ALPHAS = (
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    18.0,
    30.0,
    60.0,
    100.0,
    180.0,
    300.0,
    600.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
)
DEFAULT_SHRINKAGES = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_level(card_id: int, raw_level: Any) -> int:
    maximum = 2 if card_id in SIMPLIFIED_LEVEL_CARD_IDS else 10
    try:
        level = int(raw_level or 1)
    except (TypeError, ValueError):
        level = 1
    return max(1, min(maximum, level))


def _deck_vector(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None,
) -> np.ndarray:
    output = np.zeros(len(CARD_IDS) * 2, dtype=np.float64)
    levels = levels or {}
    for raw_card_id in deck_ids:
        card_id = int(raw_card_id)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        output[index] += 1.0
        level = _effective_level(
            card_id,
            levels.get(str(card_id), levels.get(card_id, 1)),
        )
        output[len(CARD_IDS) + index] = max(
            output[len(CARD_IDS) + index],
            level / 10.0,
        )
    return output


def _deck_strength(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None,
) -> np.ndarray:
    """Return card-presence strength for the bilinear counter matrix."""

    output = np.zeros(len(CARD_IDS), dtype=np.float64)
    levels = levels or {}
    for raw_card_id in deck_ids:
        card_id = int(raw_card_id)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        maximum = 2 if card_id in SIMPLIFIED_LEVEL_CARD_IDS else 10
        level = _effective_level(
            card_id,
            levels.get(str(card_id), levels.get(card_id, 1)),
        )
        output[index] += 0.5 + 0.5 * (level / maximum)
    return output


def _pool_vector(card_ids: Iterable[int]) -> np.ndarray:
    output = np.zeros(len(CARD_IDS), dtype=np.float64)
    for raw_card_id in card_ids:
        index = CARD_INDEX.get(int(raw_card_id))
        if index is not None:
            output[index] = 1.0
    return output


def assembler_features(row: dict[str, Any]) -> np.ndarray:
    candidate_ids = tuple(int(card_id) for card_id in row["candidate_deck_ids"])
    opponent_ids = tuple(int(card_id) for card_id in row["opponent_deck_ids"])
    candidate_levels = row.get("candidate_levels")
    opponent_levels = row.get("opponent_levels")
    candidate_strength = _deck_strength(candidate_ids, candidate_levels)
    opponent_strength = _deck_strength(opponent_ids, opponent_levels)
    features = np.concatenate(
        [
            _deck_vector(candidate_ids, candidate_levels),
            _deck_vector(opponent_ids, opponent_levels),
            _pool_vector(row["allowed_pool_ids"]),
            np.asarray(
                [
                    len(set(candidate_ids)) / 9.0,
                    len(set(opponent_ids)) / 9.0,
                    # Serving does not know how many simulator repetitions
                    # produced a training label. Label quality belongs in the
                    # sample weight, not in the inference feature contract.
                    1.0,
                ],
                dtype=np.float64,
            ),
            np.outer(candidate_strength, opponent_strength).reshape(-1),
        ]
    )
    if features.shape != (FEATURE_DIM,):
        raise AssertionError(
            f"assembler feature shape {features.shape} != {(FEATURE_DIM,)}"
        )
    return features


def _read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") not in {"train", "validation", "test"}:
                    raise ValueError(
                        f"{path}:{line_number}: missing provided split"
                    )
                row["_source_path"] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise ValueError("no assembler rows loaded")
    return rows


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    clipped = np.clip(np.asarray(prediction), 0.0, 1.0)
    error = clipped - np.asarray(target)
    return {
        "rows": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "median_ae": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


@dataclass(frozen=True)
class _RidgePreparation:
    mean: np.ndarray
    scale: np.ndarray
    normalized: np.ndarray
    weighted_design: np.ndarray
    feature_center: np.ndarray
    target_center: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_target: np.ndarray


def _prepare_ridge_family(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    sample_weight: np.ndarray,
) -> _RidgePreparation:
    train_features = features[train_mask]
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    normalized = (features - mean) / scale

    weights = sample_weight[train_mask]
    square_root_weight = np.sqrt(weights)
    feature_center = np.average(
        normalized[train_mask],
        axis=0,
        weights=weights,
    )
    target_center = float(np.average(target[train_mask], weights=weights))
    weighted_design = (
        normalized[train_mask] - feature_center
    ) * square_root_weight[:, None]
    weighted_target = (
        target[train_mask] - target_center
    ) * square_root_weight

    kernel = weighted_design @ weighted_design.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    projected_target = eigenvectors.T @ weighted_target
    return _RidgePreparation(
        mean=mean,
        scale=scale,
        normalized=normalized,
        weighted_design=weighted_design,
        feature_center=feature_center,
        target_center=target_center,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_target=projected_target,
    )


def _materialize_ridge(
    prepared: _RidgePreparation,
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    dual = prepared.eigenvectors @ (
        prepared.projected_target / (prepared.eigenvalues + float(alpha))
    )
    coefficient = prepared.weighted_design.T @ dual
    intercept = float(
        prepared.target_center - prepared.feature_center @ coefficient
    )
    prediction = prepared.normalized @ coefficient + intercept
    return coefficient, intercept, prediction


def _select_base(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    sample_weight: np.ndarray,
    alphas: Sequence[float],
) -> tuple[_RidgePreparation, float, np.ndarray, float, np.ndarray]:
    prepared = _prepare_ridge_family(
        features,
        target,
        train_mask,
        sample_weight,
    )
    candidates = []
    for alpha in alphas:
        coefficient, intercept, prediction = _materialize_ridge(prepared, alpha)
        validation = _metrics(
            target[validation_mask],
            prediction[validation_mask],
        )
        candidates.append(
            (
                float(validation["mae"]),
                float(validation["rmse"]),
                float(alpha),
                coefficient,
                intercept,
                prediction,
            )
        )
    _mae, _rmse, alpha, coefficient, intercept, prediction = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return prepared, alpha, coefficient, intercept, prediction


def train(
    rows: Sequence[dict[str, Any]],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    shrinkages: Sequence[float] = DEFAULT_SHRINKAGES,
    base_model: dict[str, np.ndarray] | None = None,
    base_source: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features = np.stack([assembler_features(row) for row in rows])
    base_features = features[:, :BASE_FEATURE_DIM]
    interaction_features = features[:, BASE_FEATURE_DIM:]
    target = np.asarray(
        [float(row["expected_matchup_score"]) for row in rows],
        dtype=np.float64,
    )
    splits = np.asarray([str(row["split"]) for row in rows])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) <= 0:
        raise ValueError("train, validation and test splits must all be non-empty")
    sample_weight = np.asarray(
        [
            max(0.1, float(row.get("usable_battles", 1)) / 20.0)
            for row in rows
        ],
        dtype=np.float64,
    )

    if base_model is None:
        (
            base_prepared,
            base_alpha,
            base_coefficient,
            base_intercept,
            base_prediction,
        ) = _select_base(
            base_features,
            target,
            train_mask,
            validation_mask,
            sample_weight,
            alphas,
        )
        base_mean = base_prepared.mean
        base_scale = base_prepared.scale
        base_selection: dict[str, Any] = {
            "mode": "validation_selected_ridge",
            "alpha": base_alpha,
        }
    else:
        required = {"feature_mean", "feature_scale", "coef", "intercept"}
        missing = required - set(base_model)
        if missing:
            raise ValueError(f"base model is missing arrays: {sorted(missing)}")
        base_mean = np.asarray(base_model["feature_mean"], dtype=np.float64)
        base_scale = np.asarray(base_model["feature_scale"], dtype=np.float64)
        base_coefficient = np.asarray(base_model["coef"], dtype=np.float64)
        base_intercept = float(
            np.asarray(base_model["intercept"]).reshape(-1)[0]
        )
        if (
            base_mean.shape != (BASE_FEATURE_DIM,)
            or base_scale.shape != (BASE_FEATURE_DIM,)
            or base_coefficient.shape != (BASE_FEATURE_DIM,)
        ):
            raise ValueError(
                "base Assembler artifact does not implement the 253-feature "
                "audited contract"
            )
        base_prediction = (
            (base_features - base_mean) / base_scale
        ) @ base_coefficient + base_intercept
        base_selection = {
            "mode": "curated_frozen_base",
            "artifact": base_source.name if base_source else None,
            "sha256": _sha256(base_source) if base_source else None,
        }

    residual_target = target - base_prediction
    interaction_prepared = _prepare_ridge_family(
        interaction_features,
        residual_target,
        train_mask,
        sample_weight,
    )
    interaction_candidates = []
    for alpha in alphas:
        coefficient, intercept, prediction = _materialize_ridge(
            interaction_prepared,
            alpha,
        )
        for shrinkage in shrinkages:
            combined_prediction = (
                base_prediction + float(shrinkage) * prediction
            )
            validation = _metrics(
                target[validation_mask],
                combined_prediction[validation_mask],
            )
            interaction_candidates.append(
                (
                    float(validation["mae"]),
                    float(validation["rmse"]),
                    float(alpha),
                    float(shrinkage),
                    coefficient,
                    intercept,
                    combined_prediction,
                )
            )
    (
        _validation_mae,
        _validation_rmse,
        interaction_alpha,
        interaction_shrinkage,
        interaction_coefficient,
        interaction_intercept,
        combined_prediction,
    ) = min(
        interaction_candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    if interaction_shrinkage <= 0.0:
        raise RuntimeError(
            "validation selected zero interaction shrinkage; refusing to emit "
            "an opponent-invariant Assembler artifact"
        )

    model = {
        "feature_mean": np.concatenate(
            [base_mean, interaction_prepared.mean]
        ).astype(np.float32),
        "feature_scale": np.concatenate(
            [base_scale, interaction_prepared.scale]
        ).astype(np.float32),
        "coef": np.concatenate(
            [
                base_coefficient,
                interaction_shrinkage * interaction_coefficient,
            ]
        ).astype(np.float32),
        "intercept": np.asarray(
            [
                base_intercept
                + interaction_shrinkage * interaction_intercept
            ],
            dtype=np.float32,
        ),
    }
    emitted_prediction = (
        (features - model["feature_mean"]) / model["feature_scale"]
    ) @ model["coef"] + float(model["intercept"][0])
    max_composition_drift = float(
        np.max(np.abs(emitted_prediction - combined_prediction))
    )
    if max_composition_drift >= 2.0e-5:
        raise AssertionError(
            f"composed artifact parity drift {max_composition_drift:.3e}"
        )

    base_test = _metrics(target[test_mask], base_prediction[test_mask])
    combined_test = _metrics(target[test_mask], emitted_prediction[test_mask])
    base_validation = _metrics(
        target[validation_mask],
        base_prediction[validation_mask],
    )
    combined_validation = _metrics(
        target[validation_mask],
        emitted_prediction[validation_mask],
    )
    readiness = (
        "bootstrap_ready"
        if (
            float(combined_validation["mae"])
            < float(base_validation["mae"])
            and float(combined_test["mae"])
            <= float(base_test["mae"]) + 0.01
            and float(combined_test["rmse"])
            <= float(base_test["rmse"]) + 0.005
        )
        else "research_only"
    )
    source_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row["_source_path"])
        source_counts.setdefault(
            source,
            {"rows": 0, "usable_battles": 0},
        )
        source_counts[source]["rows"] += 1
        source_counts[source]["usable_battles"] += int(
            row.get("usable_battles", 0) or 0
        )
    manifest = {
        "schema": "extra_lr_aux_model_v1",
        "model": "ExtraLR Assembler V1",
        "target": "paired_seed_expected_matchup_score",
        "feature_schema": "assembler_bilinear_counter_v1",
        "feature_dim": FEATURE_DIM,
        "base_feature_dim": BASE_FEATURE_DIM,
        "interaction_feature_dim": INTERACTION_FEATURE_DIM,
        "interaction_shape": [len(CARD_IDS), len(CARD_IDS)],
        "effective_level_policy": {
            "simplified_levelup_max": 2,
            "standard_max": 10,
        },
        "split": "provided compositional pool split",
        "selection": {
            "base": base_selection,
            "interaction_alpha": interaction_alpha,
            "interaction_shrinkage": interaction_shrinkage,
            "criterion": "validation_mae_then_rmse",
        },
        "metrics": {
            "validation": combined_validation,
            "base_validation": base_validation,
            "test": combined_test,
            "base_test": base_test,
            "constant_test": _metrics(
                target[test_mask],
                np.full(
                    int(test_mask.sum()),
                    np.average(
                        target[train_mask],
                        weights=sample_weight[train_mask],
                    ),
                ),
            ),
            "max_composition_drift": max_composition_drift,
            "interaction_coefficient_l2": float(
                np.linalg.norm(
                    interaction_shrinkage * interaction_coefficient
                )
            ),
        },
        "training_sources": [
            {
                "path": str(
                    Path(Path(path).parent.name) / Path(path).name
                ),
                "sha256": _sha256(Path(path)),
                **counts,
            }
            for path, counts in sorted(source_counts.items())
        ],
        "readiness": readiness,
    }
    return model, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=Path,
        help="Authoritative assembler_matchups.jsonl; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=None,
        help=(
            "Optional curated 253-feature Assembler NPZ. Its base ranking is "
            "kept frozen while the opponent interaction is fitted."
        ),
    )
    args = parser.parse_args()

    datasets = [path.expanduser().resolve() for path in args.dataset]
    rows = _read_rows(datasets)
    base_path = (
        args.base_artifact.expanduser().resolve()
        if args.base_artifact is not None
        else None
    )
    base_model = None
    if base_path is not None:
        with np.load(base_path, allow_pickle=False) as loaded:
            base_model = {
                name: np.asarray(loaded[name])
                for name in loaded.files
            }
    model, manifest = train(
        rows,
        base_model=base_model,
        base_source=base_path,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "extra_lr_assembler_v1.npz"
    np.savez_compressed(artifact, **model)
    payload = {
        **manifest,
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
    }
    metadata = artifact.with_suffix(".json")
    metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
