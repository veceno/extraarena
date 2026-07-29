from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_ai_model_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ai" / "models"
MANIFEST_PATH = MODEL_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_production_model_manifest_is_current_and_portable() -> None:
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert committed == build_manifest()
    assert committed["game_policy_adapters"] == ["v4", "v5"]
    assert committed["card_catalog"]["card_count"] == 50
    assert committed["live_progression"] == [
        {"trophies": "0-299", "profile": "extra-lr-v4-micro"},
        {"trophies": "300-1199", "profile": "extra-lr-v5-lite"},
        {"trophies": "1200-4499", "profile": "extra-lr-v5"},
        {"trophies": "4500+", "profile": "extra-lr-v5-ultra"},
    ]

    models = {entry["id"]: entry for entry in committed["models"]}
    for entry in models.values():
        artifact = MODEL_DIR / entry["artifact"]
        assert artifact.is_file()
        assert entry["sha256"] == _sha256(artifact)
        assert entry["bytes"] == artifact.stat().st_size
        source = entry.get("source_checkpoint")
        assert not source or Path(source).name == source

    assert models["extra-lr-cardoptimum-v1"]["status"] == "beta"
    assert models["extra-lr-metronome-v1"]["status"] == "beta"
    assert models["extra-lr-timestamp-v1-mono"]["status"] == "experimental"
    assert models["extra-lr-timestamp-v1-duo"]["status"] == "experimental"
    nemesis = models["extra-lr-nemesis-lite-preview"]
    assert nemesis["status"] == "preview"
    assert nemesis["adapter"] == "nemesis_lite_v1"
    assert nemesis["inputs"] == {
        "p1_card_ids": ["batch", 9],
        "p1_levels": ["batch", 9],
        "p2_card_ids": ["batch", 9],
        "p2_levels": ["batch", 9],
        "starting_side": ["batch", 1],
    }
    assert nemesis["outputs"] == {
        "outcome_logits": ["batch", 3],
        "outcome_probabilities": ["batch", 3],
    }
    assert models["extra-lr-assembler-v1"]["inputs"]["features"] == [
        "batch",
        2753,
    ]
    assembler_sidecar = json.loads(
        Path(
            str(MODEL_DIR / models["extra-lr-assembler-v1"]["artifact"])
            + ".json"
        ).read_text(encoding="utf-8")
    )
    assert (
        assembler_sidecar["training"]["feature_schema"]
        == "assembler_bilinear_counter_v1"
    )
    assert assembler_sidecar["training"]["readiness"] == "bootstrap_ready"


def test_ultra_manifest_declares_the_complete_assist_stack() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    composites = {entry["id"]: entry for entry in manifest["composites"]}
    ultra = composites["extra-lr-v5-ultra"]

    assert ultra["policy"] == "extra-lr-v5"
    assert ultra["components"] == [
        "extra-lr-assembler-v1",
        "extra-lr-cardoptimum-v1",
        "extra-lr-metronome-v1",
    ]
