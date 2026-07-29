"""Train ExtraLR Nemesis Lite Preview from unified Nemesis NDJSON/JSONL.

This command is intentionally not wired into any automatic pipeline.  A real
run must be started explicitly with one or more audited dataset paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nemesis_lite_preview import (
    TrainingConfig,
    build_training_manifest,
    load_catalog_contract,
    load_unified_jsonl,
    save_model_artifact,
    train_model,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "ai" / "cards.json"


def train_from_paths(
    dataset_paths: list[str | Path],
    *,
    output_dir: str | Path,
    ruleset: str,
    catalog_path: str | Path = DEFAULT_CATALOG,
    deck_size: int = 9,
    config: TrainingConfig = TrainingConfig(),
) -> dict:
    catalog = load_catalog_contract(catalog_path)
    rows, sources = load_unified_jsonl(
        dataset_paths,
        catalog=catalog,
        ruleset=ruleset,
        deck_size=deck_size,
    )
    model, split, training_report = train_model(
        rows,
        catalog=catalog,
        deck_size=deck_size,
        config=config,
    )
    manifest = build_training_manifest(
        catalog=catalog,
        ruleset=ruleset,
        deck_size=deck_size,
        dataset_sources=sources,
        rows=rows,
        split=split,
        training_config=config,
        training_report=training_report,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact, sidecar = save_model_artifact(
        model,
        destination / "extra_lr_nemesis_lite_preview.npz",
        manifest,
    )
    result = json.loads(sidecar.read_text(encoding="utf-8"))
    result["artifact_path"] = str(artifact)
    result["manifest_path"] = str(sidecar)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=Path,
        help=(
            "Unified extraarena_nemesis_battle_v1 NDJSON/JSONL; may be repeated."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ruleset",
        required=True,
        help="Exact features.base.ruleset value pinned into the model manifest.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument("--deck-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--deck-hidden-dim", type=int, default=48)
    parser.add_argument("--deck-output-dim", type=int, default=32)
    parser.add_argument("--outcome-hidden-dim", type=int, default=48)
    args = parser.parse_args()

    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        device=args.device,
        embedding_dim=args.embedding_dim,
        deck_hidden_dim=args.deck_hidden_dim,
        deck_output_dim=args.deck_output_dim,
        outcome_hidden_dim=args.outcome_hidden_dim,
    )
    result = train_from_paths(
        [path.expanduser().resolve() for path in args.dataset],
        output_dir=args.output_dir,
        ruleset=str(args.ruleset),
        catalog_path=args.catalog,
        deck_size=args.deck_size,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = ["train_from_paths"]
